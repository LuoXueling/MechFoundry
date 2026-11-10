"""Small UI adapter around the durable LangGraph FEM workflow."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.agents_wrapper import AgentsWrapper, require_compute_environment
from src.agents_wrapper.fem_workflow.specialists import (
    REASONING_HEADROOM,
    provider_factory,
)
from src.agents_wrapper.fem_workflow import WorkflowState, build_graph, resume_command
from src.agents_wrapper.fem_workflow.graph import (
    failed_step, open_session_graph, reopen_card, session_thread,
)
from src.agents_wrapper.fem_workflow.models import write_text_atomically
from src.agents_wrapper.fem_workflow.monitor import CAD_GENERATORS, worker_paths
from src.agents_wrapper.fem_workflow.runtime_files import (
    SOLVE_RUNNING_MESSAGE, SOLVE_START_WAIT, SolveAlreadyRunning, clear_pending_stop,
    clear_solve_stop, pending_stop, solve_running, solve_stop_requested, stop_solve,
)
from src.agents_wrapper.fem_workflow.workers import (
    DEFAULT_COMPUTE_ENV,
    WORKER_RELAUNCH_BACKOFF,
    _REQUESTED_TIER,
    ensure_cad_worker,
    ensure_hunyuan3d_worker,
    ensure_geometry_worker,
    ensure_partfield_worker,
    ensure_solver_worker,
    ensure_tool_worker,
    ensure_vomp_worker,
    hunyuan3d_for_card,
    request_warm,
    worker_is_serving,
)

__all__ = [
    "DEFAULT_COMPUTE_ENV", "SOLVE_RUNNING_MESSAGE", "WORKER_RELAUNCH_BACKOFF", "WorkflowController",
    "WorkflowEvent", "day_heading", "interrupt_marker", "request_warm",
    "restored_interrupt", "worker_is_serving",
]


def day_heading(timestamp: float) -> str:
    """The divider a conversation of this date belongs under.

    Relative wording for the two days anyone thinks of by name, and the date
    itself beyond that -- with the year only when it is not this one, which is
    the only time it tells the reader anything.
    """
    when = datetime.fromtimestamp(timestamp).date()
    today = date.today()
    if when == today:
        return "Today"
    if when == today - timedelta(days=1):
        return "Yesterday"
    stamp = f"{when:%b} {when.day}"
    return stamp if when.year == today.year else f"{stamp}, {when.year}"


@dataclass(frozen=True)
class _Answer:
    """An answer on its way to the step waiting for it."""

    value: Any


@dataclass(frozen=True)
class WorkflowEvent:
    session_id: str
    interrupt: dict[str, Any] | None
    workflow: dict[str, Any]
    checkpoint_id: str | None = None

    @property
    def complete(self) -> bool:
        return self.workflow.get("stage") == "complete"


def segmentation_method_choices(
        partfield_ready: bool, current: str | None,
        cad_faces: bool = False) -> tuple[list[str], str]:
    """Return UI choices and a valid selection.

    A CAD part's own faces come first when the surface has them, then a ready
    PartField, then shape diameter.
    """
    methods = (["cad_faces"] if cad_faces else []) + (
        ["partfield", "shape_diameter"] if partfield_ready else ["shape_diameter"]
    )
    return methods, current if current in methods else methods[0]


def faces_slider_options(current: Any) -> tuple[list[int], int]:
    """Return the face-target slider's options and a valid selection.

    The slider offers 1,000-50,000 and -1 (unlimited). A resumed session can carry
    another target: the operator skill caps a depth-8 or -9 TripoSG extraction at
    300,000 faces, and Streamlit's select_slider raises ValueError on a value that
    is not an option, so the stored value is offered as it is.
    """
    options = [*range(1000, 50001, 1000), 300000]
    try:
        value = int(current)
    except (TypeError, ValueError):
        value = -1
    if value != -1 and value not in options and value > 0:
        options = sorted([*options, value])
    elif value <= 0:
        value = -1
    return [*options, -1], value


def moved_here(value: Any, session_dir: Path) -> Any:
    """``value`` with paths recorded under another copy of this session moved into this one.

    The graph rebases its own state when a session is opened from a folder
    other than the one it was recorded in, but what the page keeps -- the
    transcript's card payloads, a rewound step's payload, the snapshot read
    for the sidebar -- was written where the session used to be, and drew its
    images from there.  A path belongs to the session when it runs through a
    folder named for it (the id is random, so nothing else does).
    """
    here = str(session_dir)
    marker = os.sep + session_dir.name

    def move(text: str) -> str:
        if not text.startswith(os.sep) or text == here or text.startswith(here + os.sep):
            return text
        index = text.find(marker + os.sep)
        if index < 0:
            index = len(text) - len(marker) if text.endswith(marker) else -1
        return text if index < 0 else here + text[index + len(marker):]

    if isinstance(value, str):
        return move(value)
    if isinstance(value, dict):
        return {key: moved_here(item, session_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [moved_here(item, session_dir) for item in value]
    return value


MATERIAL_FIELD_PREFIX = "Material field · "
# The material field the viewer opens on.  The three fields are written
# together, Young's modulus first and density last (vomp_wrapper.artifact
# CHANNELS), so "the newest artifact" alone opened every assigned material on
# its density -- the field that matters least to a stiffness solve.
PREFERRED_MATERIAL_FIELD = MATERIAL_FIELD_PREFIX + "Young's modulus"


def material_field_root(workflow: dict[str, Any], session_dir: Path) -> Path | None:
    """The material-field artifact the viewer lists, or None when it lists none.

    The fields describe the case's material, so an approved material decides:
    uniform lists nothing, VoMP its field, per-part the parts field.  Choosing
    uniform after a field was assigned keeps that field in the state and on
    disk -- versions/ and a rewind to the field's branch rely on both -- and
    the viewer, which looked only at what was there, went on offering Young's
    modulus, Poisson ratio and density for a field the case no longer used.

    Before a material is approved (the material card is up, or a rewind went
    back past it) the newest field is the one under review: the newer of the
    state's VoMP and per-part artifacts, or else the newest on disk -- an
    earlier-card rewind clears the state, and like the geometry, the reviewed
    artifact is still the most useful thing to display.
    """

    def built(path: Path) -> bool:
        return (path / "manifest.json").is_file()

    def newest(paths) -> Path | None:
        return max((path for path in paths if path.is_dir() and built(path)),
                   key=lambda path: (path / "manifest.json").stat().st_mtime, default=None)

    material = workflow.get("material")
    if material:
        mode = material.get("mode", "uniform")
        value = {"vomp": workflow.get("material_artifact"),
                 "parts": workflow.get("parts_artifact")}.get(mode) or (
            material.get("artifact") if mode != "uniform" else None)
        return Path(value) if value and built(Path(value)) else None
    fields = newest(Path(value) for value in (
        workflow.get("material_artifact"), workflow.get("parts_artifact")) if value)
    if fields is not None:
        return fields
    materials = session_dir / "materials"
    return newest([*materials.glob("vomp*"), materials / "parts" / "field"])


def preferred_visualization(entries: list[tuple[str, float]]) -> str | None:
    """Which visualization the viewer selects when its artifacts change.

    ``entries`` are (label, modification time).  The newest artifact is what
    just happened and is shown -- unless it is a material field, in which case
    the material is opened on Young's modulus when that field is present.  The
    caller applies this only when the set of artifacts changes, so a field the
    user picks afterwards stays picked.
    """
    if not entries:
        return None
    newest, _ = max(entries, key=lambda item: item[1])
    labels = {label for label, _ in entries}
    if newest.startswith(MATERIAL_FIELD_PREFIX) and PREFERRED_MATERIAL_FIELD in labels:
        return PREFERRED_MATERIAL_FIELD
    return newest


def visualization_key(session_id: str, label: str, component: str | None = None) -> str:
    """What the 3-D viewer counts as one visualization for its auto-spin.

    The viewer spins whenever this changes and stops for good once the user
    drags, wheels or touches it.  It is the session, the selected artifact and
    (for a field) the component -- deliberately not the asset's path or
    modification time: a live solve rewrites the asset under the same
    selection, and a rerun rebuilds the frame, and neither is the user
    switching to something else, so neither may restart a spin they stopped.
    """
    return json.dumps([session_id, label, component])


def interrupt_marker(event: WorkflowEvent) -> str:
    """Identify one waiting step, so a rerun does not record it twice.

    The payload alone is not an identity.  A step rebuilt from workflow state --
    physical scaling, say -- comes back byte-identical after the step before it
    is re-decided, and matching on the payload would read that as "already
    recorded" and skip it, leaving the earlier card to draw the new step.  The
    checkpoint is what makes each arrival distinct.
    """
    return json.dumps(
        {"checkpoint": event.checkpoint_id, "interrupt": event.interrupt},
        sort_keys=True, default=str,
    )


def restored_interrupt(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild a recorded step's payload, folding in the answer it was sent.

    A payload is captured as the step stood when the user acted on it, so the
    values they typed into a form live in the answer rather than the payload.
    Folding them back in is what makes an earlier card redraw with its own
    numbers instead of the ones it was born with.
    """
    interrupt = entry.get("interrupt")
    if not isinstance(interrupt, dict):
        return None
    answer = entry.get("answer")
    if not isinstance(answer, dict):
        return interrupt
    restored = {**interrupt,
                **{key: value for key, value in answer.items() if key != "action"}}
    defaults = interrupt.get("defaults")
    if isinstance(defaults, dict):
        restored["defaults"] = {
            **defaults,
            **{key: value for key, value in answer.items() if key in defaults},
        }
    # Boundary-condition answers are keyed by region name; the card reads the
    # ordered region lists the payload carries.
    for field, submitted in (("dirichlet", "dirichlet_values"),
                             ("neumann", "neumann_values")):
        regions, values = interrupt.get(field), answer.get(submitted)
        if not isinstance(regions, list) or not isinstance(values, dict):
            continue
        merged = []
        for region in regions:
            value = values.get(region.get("name")) if isinstance(region, dict) else None
            if isinstance(value, dict):
                merged.append({**region, **value})
            elif isinstance(value, list):
                merged.append({**region, "traction": value})
            else:
                merged.append(region)
        restored[field] = merged
    return restored


class WorkflowController:
    """Create, resume, and inspect workflow sessions without UI dependencies."""

    def __init__(self, output_root: Path):
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    @lru_cache(maxsize=32)
    def probe(provider: str, model: str | None) -> str:
        api = provider_factory(provider)()
        agent = AgentsWrapper(api, model=model, max_request_attempts=1)
        # A reasoning model spends output tokens thinking before it answers, so a
        # small probe comes back empty even when the model is healthy: 8 tokens
        # was not enough, nor was 64 for deepseek-flash, which thinks about even
        # "reply OK".  Budget for the thinking rather than keep raising it.
        agent.generate("Reply exactly: OK", system_prompt="Return only OK.",
                       max_output_tokens=REASONING_HEADROOM, temperature=0.0)
        return agent.model

    def create(self, provider: str, model: str | None,
               compute_env: str, vision_model: str | None = None,
               vision_provider: str | None = None) -> WorkflowEvent:
        require_compute_environment(compute_env)
        # The image role may sit on another service, so it is probed against its
        # own provider rather than the text one's.
        vision_provider = vision_provider or provider
        resolved_model = self.probe(provider, model)
        resolved_vision = (
            self.probe(vision_provider, vision_model) if vision_model else None
        )
        session_id = uuid.uuid4().hex[:12]
        session_dir = self.output_root / session_id
        session_dir.mkdir(parents=True)
        initial = {"workflow": WorkflowState(
            session_id=session_id,
            session_dir=str(session_dir),
            provider=provider,
            vision_provider=vision_provider,
            model=resolved_model,
            vision_model=resolved_vision,
            compute_env=compute_env,
        ).model_dump(mode="json")}
        return self._invoke(session_id, initial)

    def resume(self, session_id: str, answer: Any | None = None,
               checkpoint_id: str | None = None) -> WorkflowEvent:
        session_dir = self.session_dir(session_id)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"unknown session: {session_id}")
        # Before the rewind an answer may ask for, which puts back the very files
        # a running solve's results belong to.
        if answer is not None and solve_running(session_dir):
            raise SolveAlreadyRunning()
        # Wrapped once the graph is open, so the answer can name the step it is for.
        payload = None if answer is None else _Answer(answer)
        return self._invoke(session_id, payload, checkpoint_id)

    def solve_running(self, session_id: str) -> bool:
        """Whether a solve of this session is running, from any tab or front end."""
        return solve_running(self.session_dir(session_id))

    def stop_solve(self, session_id: str, *, asked: bool = False) -> bool:
        """End the session's solve; False if there was none this could reach.

        ``asked``: the caller has a solve on its way that may not have claimed yet.
        """
        return stop_solve(self.session_dir(session_id), start_wait=SOLVE_START_WAIT,
                          asked=asked)

    def solve_asked_for(self, session_id: str) -> None:
        """A new solve was asked for: no earlier stop, or record of one, is about it."""
        session_dir = self.session_dir(session_id)
        clear_pending_stop(session_dir)
        clear_solve_stop(session_dir)

    def drop_pending_stop(self, session_id: str) -> None:
        """The job that asked for a solve has ended, so a stop for it has nothing to stop."""
        clear_pending_stop(self.session_dir(session_id))

    def stop_under_way(self, session_id: str) -> bool:
        """Whether a stop has been asked for and the solve has not yet let go."""
        session_dir = self.session_dir(session_id)
        return pending_stop(session_dir) or (
            solve_running(session_dir) and solve_stop_requested(session_dir))

    def solve_stopped(self, session_id: str) -> bool:
        """Whether the session's last solve was stopped on request."""
        return solve_stop_requested(self.session_dir(session_id))

    def open(self, session_id: str, *, step_running: bool = False) -> WorkflowEvent:
        """Show a session as it stands, running none of its steps.

        Opening used to resume the graph, and a session stopped ahead of a step
        ran that step: on 2026-11-06 each opening of a session waiting at
        ``solve`` started the solve again (eight in four minutes, on the page's
        own thread, past the one-at-a-time pool).  A session with no card waiting
        gets its newest card back instead, and running anything is left to an
        answer.  While a step is running, in this tab (``step_running``) or as a
        solve anywhere, the thread is not the opener's to fork, so the session is
        only read.
        """
        session_dir = self.session_dir(session_id)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"unknown session: {session_id}")
        with open_session_graph(session_dir, build_graph) as graph:
            thread = session_thread(session_id, session_dir)
            snapshot = graph.get_state(thread)
            # A rejected answer stands against its card and would be replayed.
            parked = snapshot.interrupts and not failed_step(graph, thread)
            if (snapshot.next and not parked and not step_running
                    and not self.solve_running(session_id)):
                reopen_card(graph, thread)
                snapshot = graph.get_state(thread)
        interrupts = snapshot.interrupts
        current = moved_here(interrupts[0].value if interrupts else None, session_dir)
        workflow = moved_here(
            snapshot.values.get("workflow") or self.read_workflow(session_id), session_dir)
        self._preload_for_interrupt(session_id, current, workflow)
        return WorkflowEvent(session_id, current, workflow,
                             snapshot.config.get("configurable", {}).get("checkpoint_id"))

    def peek(self, session_id: str) -> WorkflowEvent:
        """Show a session another process is driving, writing nothing.

        For an agent session (``agent_session.py``): ``drive.py`` owns its graph, so the page
        must not do what ``open`` may -- re-offer a card (a fork of the thread under a running
        step) or start workers for the card it sees.  Only the checkpoint is read.
        """
        session_dir = self.session_dir(session_id)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"unknown session: {session_id}")
        with open_session_graph(session_dir, build_graph) as graph:
            snapshot = graph.get_state(session_thread(session_id, session_dir))
        interrupts = snapshot.interrupts
        current = moved_here(interrupts[0].value if interrupts else None, session_dir)
        workflow = moved_here(
            snapshot.values.get("workflow") or self.read_workflow(session_id), session_dir)
        return WorkflowEvent(session_id, current, workflow,
                             snapshot.config.get("configurable", {}).get("checkpoint_id"))

    def _invoke(self, session_id: str, payload: Any,
                checkpoint_id: str | None = None) -> WorkflowEvent:
        session_dir = self.session_dir(session_id)
        with open_session_graph(session_dir, build_graph) as graph:
            thread = session_thread(session_id, session_dir)
            if checkpoint_id:
                self._rewind_thread(graph, thread, checkpoint_id)
            elif failed_step(graph, thread):
                # Opened while its last answer stands rejected: that answer
                # would be replayed on the way in, so the card is put back.
                reopen_card(graph, thread)
            if isinstance(payload, _Answer):
                payload = resume_command(graph, thread, payload.value)
            result = graph.invoke(payload, config=thread)
            snapshot = graph.get_state(thread)
            current_checkpoint = snapshot.config.get("configurable", {}).get("checkpoint_id")
        interrupts = result.get("__interrupt__", ())
        current = moved_here(interrupts[0].value if interrupts else None, session_dir)
        workflow = moved_here(
            result.get("workflow") or self.read_workflow(session_id), session_dir)
        self._preload_for_interrupt(session_id, current, workflow)
        return WorkflowEvent(session_id, current, workflow, current_checkpoint)

    def _preload_for_interrupt(
        self, session_id: str, current: Any, workflow: dict[str, Any]
    ) -> None:
        """Ensure the model belonging to the card being entered is loading."""
        if not isinstance(current, dict):
            return
        # A card that paints starts the paint worker; any other card stops it.
        with contextlib.suppress(OSError):
            hunyuan3d_for_card(
                self.output_root, current, workflow.get("compute_env", DEFAULT_COMPUTE_ENV))
        if current.get("kind") == "geometry_input":
            # Start before returning the event, so TripoSG is already loading by
            # the time Streamlit can display the geometry card.
            self.preload_geometry_model(
                session_id, compute_env=workflow.get("compute_env")
            )
        elif current.get("kind") == "geometry_physics":
            # PartField is the default installed backend.  Use the scale-review
            # card to hide its checkpoint load before segmentation is reachable.
            # A CAD part is segmented by its own faces first, so loading
            # PartField would only evict the generator for nothing.
            from src.partfield_wrapper.availability import partfield_status
            if partfield_status().ready and not current.get("cad_faces"):
                self.preload_partfield_worker(
                    workflow.get("compute_env", DEFAULT_COMPUTE_ENV)
                )
        elif (current.get("kind") == "bc_input"
              and (current.get("partfield") or {}).get("ready")
              and (current.get("segmentation") or {}).get("method") == "partfield"):
            # Launch before returning the event.  The UI can draw immediately,
            # while the 1.2 GB checkpoint finishes loading in the background.
            self.preload_partfield_worker(
                workflow.get("compute_env", DEFAULT_COMPUTE_ENV)
            )

    @staticmethod
    def _rewind_thread(graph: Any, thread: dict[str, Any], checkpoint_id: str) -> None:
        """Make an earlier step the thread's head again, waiting on its answer.

        Pinning the checkpoint on the invocation does not do this: the answer is
        delivered to whichever task is pending now, or dropped, so re-deciding a
        step silently answered a later one instead.  Forking that checkpoint and
        replaying it discards everything the superseded answer produced and
        re-raises the step's own interrupt, which the new answer then lands on.

        The head is forked too when its step rejected the last answer: LangGraph
        keeps that answer with the error and would replay it in place of the
        corrected one, so the card could never be answered again.
        """
        head = graph.get_state(thread).config.get("configurable", {}).get("checkpoint_id")
        if head == checkpoint_id and not failed_step(graph, thread):
            return
        reopen_card(graph, thread, checkpoint_id)

    def rewind(self, session_id: str, checkpoint_id: str) -> WorkflowEvent:
        """Restore an interrupt checkpoint as the active UI branch."""
        with open_session_graph(self.session_dir(session_id), build_graph) as graph:
            snapshot = graph.get_state({"configurable": {
                "thread_id": session_id, "checkpoint_id": checkpoint_id,
            }})
        if not snapshot.values:
            raise ValueError("checkpoint is not available")
        interrupts = [item.value for task in snapshot.tasks for item in task.interrupts]
        if not interrupts:
            raise ValueError("checkpoint is not waiting for user input")
        session_dir = self.session_dir(session_id)
        workflow = moved_here(snapshot.values.get("workflow") or {}, session_dir)
        current = moved_here(interrupts[0], session_dir)
        self._preload_for_interrupt(session_id, current, workflow)
        return WorkflowEvent(session_id, current, workflow, checkpoint_id)

    def session_dir(self, session_id: str) -> Path:
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("invalid session id")
        return self.output_root / session_id

    def read_workflow(self, session_id: str) -> dict[str, Any]:
        """The session's last snapshot, or nothing if it cannot be read.

        The sidebar reads this for every session on every rerun, so one file
        that cannot be parsed must cost that session its title, not take the
        page down.  Writes are atomic, so reaching here means the file is
        genuinely damaged rather than half-written.
        """
        path = self.session_dir(session_id) / "session.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        folder = self.session_dir(session_id)
        # Only a session opened from somewhere new needs its paths moved.
        return value if value.get("session_dir") == str(folder) else moved_here(value, folder)

    def sessions(self) -> list[str]:
        return [session for session, _ in self.dated_sessions()]

    def dated_sessions(self) -> list[tuple[str, float]]:
        """Every session with when a step last changed it, newest first.

        The sidebar groups by day, and grouping has to agree with the order it
        is drawn in, so both come from the one timestamp.

        That timestamp is the state snapshot's, not the directory's or the
        checkpoint's.  Simply opening a conversation re-invokes the graph, which
        rewrites the checkpoint and the transcript and touches the directory --
        so those three record when a session was last *looked at*, and reading
        an old conversation sent it to the top of the list.  ``session.json`` is
        written only when a node changes the state, which is the operation the
        user means.  A session with no snapshot has not got past its first card,
        and falls back to the directory.
        """
        found = []
        for path in self.output_root.iterdir():
            if not path.is_dir() or not (path / "checkpoints.sqlite").exists():
                continue
            snapshot = path / "session.json"
            marker = snapshot if snapshot.is_file() else path
            found.append((path.name, marker.stat().st_mtime))
        return sorted(found, key=lambda item: item[1], reverse=True)

    def delete_session(self, session_id: str) -> None:
        """Remove a session and everything it produced. There is no undo.

        Every checkpointer connection is opened and closed around its own call,
        so nothing here holds the SQLite file open.
        """
        target = self.session_dir(session_id)
        if target.is_dir():
            shutil.rmtree(target)

    def save_upload(self, session_id: str, name: str, data: bytes) -> Path:
        suffix = Path(name).suffix.lower()
        target = self.session_dir(session_id) / "ui_uploads" / f"{uuid.uuid4().hex[:12]}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def worker_paths(self, kind: str, compute_env: str) -> tuple[Path, Path]:
        """Where a resident worker of this kind announces itself.

        The one definition the monitor also dispatches by, so the process that
        starts a worker and the process that calls it cannot disagree.
        """
        socket_path, pid_path = worker_paths(self.output_root, kind, compute_env)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        return socket_path, pid_path

    def preload_geometry_model(self, session_id: str,
                               compute_env: str | None = None,
                               generator: str | None = None) -> bool:
        """Start the chosen generator's persistent worker without blocking the UI.

        Without a choice, the one the session last generated with.
        """
        session_dir = self.session_dir(session_id)
        if not compute_env or not generator:
            workflow = self.read_workflow(session_id)
            compute_env = compute_env or workflow.get("compute_env", DEFAULT_COMPUTE_ENV)
            generator = generator or (workflow.get("generation") or {}).get("generator")
        if generator in CAD_GENERATORS:
            return ensure_cad_worker(self.output_root, generator, compute_env)
        if generator == "hunyuan3d":
            return ensure_hunyuan3d_worker(self.output_root, compute_env)
        return ensure_geometry_worker(self.output_root, session_dir, compute_env)

    def preload_tool_worker(self, compute_env: str | None = None,
                            tier: str = "light") -> bool:
        """Keep a warm tool worker up, at least as warm as this step will need."""
        return ensure_tool_worker(self.output_root, compute_env, tier)

    def preload_partfield_worker(self, compute_env: str = DEFAULT_COMPUTE_ENV) -> bool:
        """Start the optional resident PartField model server."""
        return ensure_partfield_worker(self.output_root, compute_env)

    def preload_solver_worker(self, compute_env: str | None = None) -> bool:
        """Start the process the next forward solve will run in."""
        return ensure_solver_worker(self.output_root, compute_env)

    def preload_vomp_worker(self, compute_env: str = DEFAULT_COMPUTE_ENV) -> bool:
        """Start VoMP only in response to an explicit adaptive-material action."""
        return ensure_vomp_worker(self.output_root, compute_env)

    def history(self, session_id: str) -> list[dict[str, Any]]:
        path = self.session_dir(session_id) / "ui_history.json"
        if not path.exists():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        # Card payloads recorded before a session moved still name its old folder.
        return moved_here(value, self.session_dir(session_id)) if isinstance(value, list) else []

    def append_history(self, session_id: str, role: str, content: str,
                       checkpoint_id: str | None = None,
                       interrupt: dict[str, Any] | None = None,
                       replace_last: bool = False) -> None:
        """Record a conversation step.

        ``interrupt`` is the payload the step is waiting on; storing it lets the
        UI redraw that step's card later without reopening the checkpointer.
        ``replace_last`` overwrites the final entry instead of appending, which
        is how repeated solves stay on one card.
        """
        history = self.history(session_id)
        entry = {"role": role, "content": content}
        if checkpoint_id:
            entry["checkpoint_id"] = checkpoint_id
        if interrupt is not None:
            entry["interrupt"] = json.loads(json.dumps(interrupt, default=str))
        if replace_last and history:
            history[-1] = entry
        else:
            history.append(entry)
        self._write_history(session_id, history)

    def update_history_entry(self, session_id: str, index: int, **fields: Any) -> None:
        """Merge fields into one recorded step, leaving the rest untouched."""
        history = self.history(session_id)
        if not -len(history) <= index < len(history):
            return
        entry = dict(history[index])
        for name, value in fields.items():
            entry[name] = (None if value is None
                           else json.loads(json.dumps(value, default=str)))
        history[index] = entry
        self._write_history(session_id, history)

    def truncate_history(self, session_id: str, length: int) -> None:
        history = self.history(session_id)
        if length < 0 or length > len(history):
            raise ValueError("invalid history length")
        self._write_history(session_id, history[:length])

    def _write_history(self, session_id: str,
                       history: list[dict[str, Any]]) -> None:
        path = self.session_dir(session_id) / "ui_history.json"
        write_text_atomically(path, json.dumps(history, indent=2))
