#!/usr/bin/env python
"""Start or resume the interactive FEM workflow."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import uuid
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "fem_workflow"
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.agents_wrapper import (
    AgentsWrapper, AgentsWrapperError, ComputeEnvironmentError,
    require_compute_environment,
)
from src.agents_wrapper.fem_workflow import (
    DEFAULT_TEXT_MODELS, DEFAULT_VISION_MODELS, WorkflowState, build_graph, resume_command,
)
from src.agents_wrapper.fem_workflow.graph import (
    CAD_GENERATORS as CAD_LABELS, failed_step, needs_cad_revision, reopen_card,
)
from src.agents_wrapper.fem_workflow.specialists import (
    PROVIDERS,
    REASONING_HEADROOM,
    provider_factory,
)
from src.agents_wrapper.fem_workflow.assets import (
    prefetch_solution_assets, solution_prefetch_requests,
)
from src.agents_wrapper.fem_workflow.monitor import CAD_GENERATORS, DEFAULT_COMPUTE_ENV
from src.agents_wrapper.fem_workflow.runtime_files import current_environment
from src.agents_wrapper.fem_workflow.workers import (
    ensure_cad_worker, ensure_geometry_worker, ensure_hunyuan3d_worker, ensure_partfield_worker,
    ensure_solver_worker, ensure_tool_worker, ensure_vomp_worker, hunyuan3d_for_card,
    stop_hunyuan3d_worker,
)
from src.partfield_wrapper.availability import partfield_status


def read_compute_env(session_dir: Path) -> str:
    """Which environment this session's tools run in, for a resumed run.

    A session made before the MechFoundry rename stores an ``implicit-*`` name; it resumes in
    the renamed environment once the old one is uninstalled (``current_environment``).
    """
    snapshot = session_dir / "session.json"
    try:
        stored = json.loads(snapshot.read_text(encoding="utf-8")).get(
            "compute_env", DEFAULT_COMPUTE_ENV
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return DEFAULT_COMPUTE_ENV
    return current_environment(stored) if isinstance(stored, str) else DEFAULT_COMPUTE_ENV


def unrevised_cad_warning(interrupt, answer) -> str | None:
    """A reminder when an unrevised prompt is about to go to a CAD generator."""
    if not (isinstance(interrupt, dict) and interrupt.get("kind") == "geometry_input"
            and isinstance(answer, dict)) or answer.get("revise_prompt") \
            or answer.get("action") == "draw_image":
        return None
    generator = answer.get("generator") or (answer.get("generation_settings") or {}).get("generator")
    if not needs_cad_revision(generator, str(answer.get("description", "")),
                              interrupt.get("cad_revision")):
        return None
    return ("Warning: this prompt was not revised for " + CAD_LABELS[generator] + ". "
            'Send it again with "revise_prompt": true first -- it does not know object '
            "names, and an unrevised prompt usually comes back a box.")


def parse_answer(raw: str):
    raw = raw.strip()
    if raw.lower() in ("yes", "y"):
        return True
    if raw.lower() in ("no", "n"):
        return False
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parser():
    root = argparse.ArgumentParser(description="Economic API-assisted FEM workflow")
    sub = root.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    start.add_argument("--provider", choices=tuple(PROVIDERS), default="deepseek")
    start.add_argument("--model", help="text-only calls")
    start.add_argument(
        "--vision-provider", choices=tuple(PROVIDERS),
        help="service for calls that send an image; defaults to --provider",
    )
    start.add_argument("--vision-model", help="calls that send an image")
    start.add_argument("--compute-env", default=DEFAULT_COMPUTE_ENV)
    start.add_argument("--octree-depth", type=int, default=7)
    start.add_argument("--session-id")
    resume = sub.add_parser("resume")
    resume.add_argument("session_id")
    resume.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return root


def probe_api(provider: str, model: str | None) -> str:
    """Make one tiny request and return the resolved, known-working model name."""

    agent = AgentsWrapper(
        provider_factory(provider)(), model=model, max_request_attempts=1
    )
    # A reasoning model spends output tokens thinking before it answers, so a
    # small probe comes back empty even when the model is healthy: deepseek-flash
    # thinks about even "reply OK".  Budget for the thinking.
    agent.generate(
        "Reply exactly: OK",
        system_prompt="Return only OK.",
        max_output_tokens=REASONING_HEADROOM,
        temperature=0.0,
    )
    return agent.model


def select_available_api(provider: str, model: str | None) -> tuple[str, str]:
    """Probe selections until one works; never switch providers silently."""

    while True:
        try:
            return provider, probe_api(provider, model)
        except (AgentsWrapperError, ValueError) as exc:
            print(json.dumps({
                "kind": "api_unavailable",
                "provider": provider,
                "model": model,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}",
                "message": "Enter another provider/model.",
                "fields": {"provider": "|".join(PROVIDERS),
                           "model": "model name or null"},
            }, indent=2))
            answer = parse_answer(input("> "))
            if not isinstance(answer, dict):
                raise ValueError("replacement API selection must be a JSON object")
            replacement = answer.get("provider")
            if replacement not in PROVIDERS:
                raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
            provider = replacement
            candidate = answer.get("model")
            model = str(candidate).strip() if candidate else None


def preload_for(output_root: Path, session_dir: Path, compute_env: str,
                interrupt, paint: bool | None = None, solver_spare: bool = True,
                keep_paint: bool = False, hunyuan_shape_only: bool = False) -> None:
    """Start what the step after this one will want, while the user is reading.

    The same schedule the app keeps, for the same reasons: a tool worker warm
    enough for the next tool, TripoSG loading while the geometry is described,
    and the deeper imports bought during generation rather than in front of the
    first call that needs them.  The monitor uses a worker whenever one is
    listening, so this is the only thing standing between a command-line run and
    the warm path -- and none of it is required for the run to work.

    A caller that knows its next answer narrows it: ``paint=False`` when this card
    will not be painted (no 17 GB paint worker loaded for a uniform material), and
    ``solver_spare=False`` when no solve follows (a finished one).  ``keep_paint=True`` when a
    paint is known to come on a later card (a plan that paints): the paint worker is started or
    kept instead of stopped, so its load is paid while the regions are read.
    ``hunyuan_shape_only=True`` when a Hunyuan3D generation will not paint (the operator runner
    generates shape only): the worker loads the shape stage alone.
    """
    if not isinstance(interrupt, dict):
        return
    kind = interrupt.get("kind")
    try:
        # The paint worker lives exactly as long as a card that paints is open.
        if paint is False:
            if not (kind == "geometry_input" and interrupt.get("generator") == "hunyuan3d"):
                stop_hunyuan3d_worker(output_root)
        elif keep_paint:
            ensure_hunyuan3d_worker(output_root, compute_env, paint=True)
        else:
            hunyuan3d_for_card(output_root, interrupt, compute_env)
        if kind == "geometry_input":
            if interrupt.get("generator") in CAD_GENERATORS:
                ensure_cad_worker(output_root, interrupt["generator"], compute_env)
            elif interrupt.get("generator") == "hunyuan3d":
                ensure_hunyuan3d_worker(output_root, compute_env,
                                        **({"shape_only": True} if hunyuan_shape_only else {}))
            else:
                ensure_geometry_worker(output_root, session_dir, compute_env)
            ensure_tool_worker(output_root, compute_env, "light")
        else:
            # Everything past the geometry card reaches a GPU tool: segmentation
            # first, then a resolve for every set of regions confirmed.
            ensure_tool_worker(output_root, compute_env, "cuda")
            # PartField is the default backend when installed.  Begin loading
            # its 1.2 GB checkpoint while physical scaling is being reviewed,
            # one user interaction before segmentation can be requested.
            if (kind == "geometry_physics" and partfield_status().ready
                    and not interrupt.get("cad_faces")):
                ensure_partfield_worker(output_root, compute_env)
            elif (kind == "bc_input" and not keep_paint   # (PartField would stop the paint worker)
                  and interrupt.get("partfield", {}).get("ready")
                  and interrupt.get("segmentation", {}).get("method") == "partfield"):
                ensure_partfield_worker(output_root, compute_env)
            elif kind == "shape_opt_settings" and solver_spare:
                # The next click may be the solve; its process starts now.
                ensure_solver_worker(output_root, compute_env)
    except OSError:
        pass


def after_answer(output_root: Path, compute_env: str, interrupt, answer=None) -> None:
    """Buy the GPU tools' imports while generation has the GPU to itself."""
    if isinstance(interrupt, dict) and interrupt.get("kind") == "geometry_input":
        try:
            ensure_tool_worker(output_root, compute_env, "cuda")
        except OSError:
            pass
    if (isinstance(interrupt, dict) and interrupt.get("kind") == "material_input"
            and isinstance(answer, dict)
            and answer.get("action") in ("infer_vomp",)):
        try:
            ensure_vomp_worker(output_root, compute_env)
        except OSError:
            pass


def start_prefetch(session_dir: Path, compute_env: str, interrupt,
                   started: set) -> threading.Thread | None:
    """Colour a finished solve in the background, as the app does.

    A command-line run has no viewer of its own, but the session it leaves
    behind is opened in one, and this is the moment the next several things
    anyone looks at are known.  Runs on its own thread so the prompt comes back
    immediately; a run that exits before it finishes has only left the assets to
    be built on demand, which is where they came from anyway.
    """
    if not isinstance(interrupt, dict) or not interrupt.get("solve_complete"):
        return None
    requests = solution_prefetch_requests(session_dir)
    key = tuple(sorted(request["output"] for request in requests))
    if not requests or key in started:
        return None
    started.add(key)
    thread = threading.Thread(
        target=prefetch_solution_assets,
        args=(str(session_dir), compute_env, requests),
        name="fem-prefetch", daemon=True,
    )
    thread.start()
    return thread


def finish_hint(interrupt) -> str | None:
    """How to end a session whose solve has finished, which the card itself does not say."""
    if isinstance(interrupt, dict) and interrupt.get("solve_complete"):
        return ('The solve finished. Send {"action":"finish"} to end the session; '
                "any other answer runs again ({} repeats this run).")
    return None


def main():
    args = parser().parse_args()
    # Absolute, so the workers this run starts are the ones the tools look for:
    # both key them on the output root's path.
    output_root = args.output_root.expanduser().resolve()
    if args.command == "start":
        try:
            environment = require_compute_environment(args.compute_env)
        except ComputeEnvironmentError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps({"kind": "compute_environment_ready", "name": environment.name,
                          "details": environment.summary}, separators=(",", ":")))
        # The text role's own default, as the README and the app have it; the
        # provider class's default is not always that (Groq's is its vision model).
        provider, model = select_available_api(
            args.provider, args.model or DEFAULT_TEXT_MODELS.get(args.provider))
        # The image role may run on another service, so it is probed against its
        # own provider rather than the text one's.
        vision_provider = args.vision_provider or provider
        vision_model = args.vision_model or DEFAULT_VISION_MODELS.get(vision_provider)
        vision_provider, vision_model = select_available_api(
            vision_provider, vision_model
        )
        session_id = args.session_id or uuid.uuid4().hex[:12]
        session_dir = output_root / session_id
        if session_dir.exists():
            raise SystemExit(f"session already exists: {session_id}")
        session_dir.mkdir(parents=True)
        initial = {"workflow": WorkflowState(
            session_id=session_id, session_dir=str(session_dir.resolve()),
            provider=provider, model=model, vision_model=vision_model,
            vision_provider=vision_provider,
            compute_env=args.compute_env,
            octree_depth=args.octree_depth,
        ).model_dump(mode="json")}
    else:
        session_id = args.session_id
        session_dir = output_root / session_id
        if not session_dir.is_dir():
            raise SystemExit(f"session does not exist: {session_id}")
        initial = None

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise SystemExit("Install the API-backed agent packages listed in README.md") from exc

    connection = sqlite3.connect(session_dir / "checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    graph = build_graph().compile(checkpointer=checkpointer)
    # The folder is named so a session copied or moved elsewhere works in its
    # own folder rather than the one it was recorded in.
    config = {"configurable": {"thread_id": session_id, "session_dir": str(session_dir)}}
    compute_env = (initial or {}).get("workflow", {}).get("compute_env") if initial else None
    if compute_env is None:
        compute_env = read_compute_env(session_dir)
    # The tools that run before the first question is even asked -- staging,
    # inspecting and rendering an uploaded surface -- are worth a warm worker.
    try:
        ensure_tool_worker(output_root, compute_env, "light")
    except OSError:
        pass
    prefetched: set = set()
    prefetching = None
    # A run that ended on a rejected answer left it stored against its card,
    # where it would be replayed; that card is put back as it was instead.
    reopened = reopen_card(graph, config) if initial is None and failed_step(graph, config) else None
    result = reopened if reopened is not None else graph.invoke(initial, config=config)
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        preload_for(output_root, session_dir, compute_env, interrupt_value)
        prefetching = start_prefetch(
            session_dir, compute_env, interrupt_value, prefetched
        ) or prefetching
        print(json.dumps(interrupt_value, indent=2) if isinstance(interrupt_value, dict) else interrupt_value)
        hint = finish_hint(interrupt_value)
        if hint:
            print(hint, file=sys.stderr)
        answer = parse_answer(input("> "))
        warning = unrevised_cad_warning(interrupt_value, answer)
        if warning:
            print(warning, file=sys.stderr)
        after_answer(output_root, compute_env, interrupt_value, answer)
        answered = graph.get_state(config).config["configurable"]["checkpoint_id"]
        try:
            result = graph.invoke(resume_command(graph, config, answer), config=config)
        except Exception as exc:  # any step's rejection; the card is restored below
            print(json.dumps({
                "kind": "answer_rejected",
                "error": f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:400]}",
                "message": "The answer was not applied; the same card follows. Correct it and send again.",
            }, indent=2))
            result = reopen_card(graph, config, answered)
    if prefetching is not None and prefetching.is_alive():
        # Nothing depends on it, so it gets the time it has and no more.
        prefetching.join(timeout=30.0)
    workflow = result.get("workflow", {})
    api_calls = sum(not receipt.get("cached", False) for receipt in workflow.get("api_receipts", []))
    print(json.dumps({"session_id": session_id, "stage": workflow.get("stage"),
                      "api_calls": api_calls}, separators=(",", ":")))
    connection.close()


if __name__ == "__main__":
    main()
