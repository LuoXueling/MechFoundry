"""Machine load and solver progress, read for the UI's tool window.

Sampling runs on its own thread because the interesting moments are exactly the
ones when the Streamlit script is blocked: a forward solve holds the main thread
inside ``subprocess.run`` for minutes.  The sampler keeps filling its buffer
throughout, so when the page comes back it can draw what the machine was doing
while it was away.
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import altair as alt
import pandas as pd

# Where a solve's outputs are found and named.  Shared with the command line,
# which runs the same prefetch, so it cannot live in a Streamlit-only module.
from src.agents_wrapper.fem_workflow.assets import load_step_surfaces  # noqa: F401
# The newest solve log and the failed solve's record, read as the workflow and the runner do.
from src.agents_wrapper.fem_workflow.runtime_files import (  # noqa: F401
    SOLVE_FAILURE, newest_solve_log, solve_failure,
)

SAMPLE_INTERVAL = 1.0
# Ten minutes at one sample a second: enough to cover a solve without letting a
# session that stays open all day grow without bound.
HISTORY = 600


@dataclass(frozen=True)
class Sample:
    """One instant of machine load, every figure a percentage or None.

    ``gpus`` holds one (utilisation, memory) pair per card.  A machine with
    several of them runs a solve on one at a time, so pooling them into a single
    figure would average the busy card away.
    """

    timestamp: float
    cpu: float | None
    memory: float | None
    gpus: tuple[tuple[float | None, float | None], ...] = ()


def read_memory(meminfo: str) -> float | None:
    """Used system memory as a percentage, from /proc/meminfo text."""
    values = {}
    for line in meminfo.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0])
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return 100.0 * (total - available) / total


def read_cpu_totals(stat: str) -> tuple[int, int] | None:
    """(busy, total) jiffies from the aggregate line of /proc/stat.

    Utilisation is a rate, so a single reading says nothing; the caller keeps the
    previous pair and divides the differences.
    """
    for line in stat.splitlines():
        if line.startswith("cpu "):
            fields = [int(value) for value in line.split()[1:] if value.isdigit()]
            if len(fields) < 4:
                return None
            total = sum(fields)
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            return total - idle, total
    return None


def read_gpu(output: str) -> tuple[tuple[float | None, float | None], ...]:
    """One (utilisation, memory use) pair per card, as percentages.

    nvidia-smi prints a line per card; each is reported on its own so a solve
    pinning one card is visible rather than averaged into the idle ones.
    """
    cards = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            utilisation, used, total = (float(part) for part in parts[:3])
        except ValueError:
            continue
        cards.append((utilisation, 100.0 * used / total if total else None))
    return tuple(cards)


class LoadSampler:
    """Samples machine load on a daemon thread until the process exits."""

    def __init__(self, interval: float = SAMPLE_INTERVAL, history: int = HISTORY):
        self.interval = interval
        self._samples: deque[Sample] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._previous_cpu: tuple[int, int] | None = None
        self._gpu_available = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="fem-ui-telemetry")
        self._thread.start()

    def history(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)

    def latest(self) -> Sample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def sample(self) -> Sample:
        cpu = None
        totals = _read_file("/proc/stat", read_cpu_totals)
        if totals and self._previous_cpu:
            busy = totals[0] - self._previous_cpu[0]
            elapsed = totals[1] - self._previous_cpu[1]
            if elapsed > 0:
                cpu = max(0.0, min(100.0, 100.0 * busy / elapsed))
        if totals:
            self._previous_cpu = totals
        return Sample(time.time(), cpu, _read_file("/proc/meminfo", read_memory),
                      self._read_gpu())

    def _read_gpu(self) -> tuple[tuple[float | None, float | None], ...]:
        if not self._gpu_available:
            return ()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # No driver, or nvidia-smi is missing: stop paying for the attempt.
            self._gpu_available = False
            return ()
        return read_gpu(result.stdout) if result.returncode == 0 else ()

    def _run(self) -> None:
        while True:
            try:
                sample = self.sample()
            except Exception:
                # Telemetry must never take the page down with it.
                time.sleep(self.interval)
                continue
            with self._lock:
                self._samples.append(sample)
            time.sleep(self.interval)


def _read_file(path: str, parse):
    try:
        return parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


LOAD_STEP = re.compile(r"=+\s*Load step (\d+)/(\d+)")
ITERATION = re.compile(
    r"scale=([0-9.eE+-]+)\s+Iter (\d+):\s*\|\|R\|\|\s*=\s*([0-9.eE+-]+),"
    r"\s*rel\s*=\s*([0-9.eE+-]+)"
)
# An optimization runs a whole Newton solve per iteration, so what the load-step
# banner is to a ramp this banner is to a shape optimization: the boundary at
# which the residual restarts from a shape that has just changed.
OPT_ITERATION = re.compile(r"=+\s*Optimization Iteration (\d+)")
# What the optimizer is actually minimising, logged once the iteration's
# gradient step has been taken.
OBJECTIVE = re.compile(r"Opt iter (\d+):\s*obj=([0-9.eE+-]+)")
# The solve's own record of how many iterations it was asked for, and how often
# it saves a shape, written by run_forward before the first of them -- without
# them the progress bar has a numerator and no denominator, and the viewer's
# iterations are numbered by the order they were written rather than by which
# iteration wrote them.
OPT_TOTAL = re.compile(r"max_optimize_iters=(\d+)")
OPT_SAVE_EVERY = re.compile(r"output_per_optimize_step=(\d+)")
# The learning rate gd_lr_auto chose at each update (core abstract_field,
# _gd_lr_auto_choose), and the line it writes instead when the gradient was zero.
_NUM = r"([0-9.eE+-]+)"
GD_LR_AUTO = re.compile(
    rf"gd_lr_auto: iteration (\d+): gd_lr {_NUM} = {_NUM} \* (\w+) {_NUM} / "
    rf"max\|dJ/ds\| {_NUM}")
GD_LR_AUTO_KEPT = re.compile(r"gd_lr_auto: iteration (\d+): .*keeping gd_lr " + _NUM)
# Every line the solver writes is stamped, so how long a run has taken is in
# the log rather than something the panel has to time for itself -- which also
# means a log read after the fact reports the same elapsed times as one watched
# live.
STAMP = re.compile(r"^\[(\d\d-\d\d-\d{4},\d\d:\d\d:\d\d\.\d+)\]")
STAMP_FORMAT = "%m-%d-%Y,%H:%M:%S.%f"


def _stamped(line: str) -> datetime | None:
    """When the solver wrote this line, or None if it carries no stamp."""
    match = STAMP.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), STAMP_FORMAT)
    except ValueError:
        return None


@dataclass(frozen=True)
class SolverProgress:
    """What a solve log says about how far along it is.

    A forward solve walks load steps; a shape optimization walks iterations,
    each of which is a whole Newton solve of its own on a shape that has just
    moved.  Both are read here, because both are the outer loop the panel has
    to report against -- a bar reading "load step 1/1" says nothing at all
    about an optimization a hundred iterations long.
    """

    load_step: int
    load_steps: int
    iterations: list[dict[str, float]]
    finished: bool
    # Where the optimization has got to, and what it was asked for.  Zero
    # iterations means this is a forward solve, which has none.
    opt_iteration: int = 0
    opt_iterations: int = 0
    # One reading of the objective per optimization iteration, each carrying
    # the second it was logged at so it plots against the same clock as the
    # residuals beside it.
    objective: list[dict[str, float]] = field(default_factory=list)
    # What gd_lr_auto chose at each update, in order, when the run had it on
    # (gd_lr_auto_note words it); empty otherwise.
    gd_lr_auto: list[dict[str, Any]] = field(default_factory=list)

    @property
    def optimizing(self) -> bool:
        """Whether this log is an optimization rather than a forward solve."""
        return bool(self.opt_iterations or self.objective or self.opt_iteration)

    @property
    def fraction(self) -> float:
        if self.optimizing and self.opt_iterations:
            # Iterations are logged from zero, so the first one finished is one
            # of however many were asked for.
            return min(1.0, (self.opt_iteration + 1) / self.opt_iterations)
        return min(1.0, self.load_step / self.load_steps) if self.load_steps else 0.0


FINISHED = ("Simulation finalized", "Optimization finished")


def read_solver_progress(log: str) -> SolverProgress:
    """Parse load-step and residual lines out of a solve log.

    A single-step solve logs no load-step banner at all, so the absence of one
    means one step rather than none.
    """
    load_step, load_steps, iterations = 0, 0, []
    opt_iteration, opt_iterations, objective = 0, 0, []
    optimizing = False
    started = None
    gd_lr_auto = []
    for line in log.splitlines():
        stamp = _stamped(line)
        if stamp is not None and started is None:
            # From the solver's first line, so the clock covers the setup an
            # iteration count says nothing about -- meshing and BC resolution
            # are most of a short solve.
            started = stamp
        total = OPT_TOTAL.search(line)
        if total:
            opt_iterations = int(total.group(1))
        opt = OPT_ITERATION.search(line)
        if opt:
            opt_iteration, optimizing = int(opt.group(1)), True
            # Each iteration solves a shape of its own, so the load scale and
            # the Newton count both start again here.
            load_step = 0
            continue
        if "gd_lr_auto: " in line:
            chosen = GD_LR_AUTO.search(line)
            if chosen:
                gd_lr_auto.append({
                    "iteration": int(chosen.group(1)), "applied": True,
                    "gd_lr": float(chosen.group(2)), "fraction": float(chosen.group(3)),
                    "length_name": chosen.group(4), "length": float(chosen.group(5)),
                    "max_abs_grad": float(chosen.group(6)),
                })
                continue
            kept = GD_LR_AUTO_KEPT.search(line)
            if kept:
                gd_lr_auto.append({"iteration": int(kept.group(1)), "applied": False,
                                   "gd_lr": float(kept.group(2))})
                continue
        value = OBJECTIVE.search(line)
        if value:
            reading = {
                "opt_iteration": float(value.group(1)),
                "value": float(value.group(2)),
            }
            if stamp is not None and started is not None:
                reading["elapsed"] = (stamp - started).total_seconds()
            objective.append(reading)
            continue
        step = LOAD_STEP.search(line)
        if step:
            load_step, load_steps = int(step.group(1)), int(step.group(2))
            continue
        iteration = ITERATION.search(line)
        if iteration:
            row = {
                # Which load step this iteration belongs to, counted from the
                # banners rather than the logged scale: the scale on an iteration
                # line is the solving object's own, which a mesh rebuild resets.
                "load_step": float(max(load_step, 1)),
                # And which optimization iteration, for the same reason: the
                # residual restarts at every one of them.
                "opt_iteration": float(opt_iteration),
                "load_scale": float(iteration.group(1)),
                "iteration": float(iteration.group(2)),
                "residual": float(iteration.group(3)),
                "relative": float(iteration.group(4)),
            }
            if stamp is not None and started is not None:
                row["elapsed"] = (stamp - started).total_seconds()
            iterations.append(row)
    finished = any(marker in log for marker in FINISHED)
    if not load_steps:
        load_steps = 1
        load_step = 1 if (iterations or finished) else 0
    if finished:
        load_step = load_steps
        if optimizing and opt_iterations:
            # A run that converged early stopped where it stopped; the bar
            # should read full rather than wherever the tolerance caught it.
            opt_iteration = opt_iterations - 1
    return SolverProgress(load_step, load_steps, iterations, finished,
                          opt_iteration, opt_iterations if optimizing else 0,
                          objective, gd_lr_auto)


def gd_lr_auto_note(progress: SolverProgress) -> str | None:
    """The Solver panel's line on the learning rates gd_lr_auto chose, or None.

    The rate is sized again at every update, so the line gives the latest and,
    once there are several, the range so far.  None for a run without the
    option (nothing in its log), so the panel of every other solve is unchanged.
    """
    chosen = progress.gd_lr_auto
    if not chosen:
        return None
    latest = chosen[-1]
    sized = [entry for entry in chosen if entry.get("applied")]
    if not sized:
        return (f"Auto learning rate: the gradient was zero, so the GD learning rate "
                f"{latest['gd_lr']:.4g} was kept.")
    rates = [entry["gd_lr"] for entry in chosen]
    first = sized[0]
    unit = "ε" if first.get("length_name") == "epsilon" else "L"
    spread = (f" (range {min(rates):.3g}–{max(rates):.3g} over {len(rates)} steps)"
              if len(rates) > 1 else "")
    kept = "" if latest.get("applied") else ", kept: the gradient was zero"
    return (f"Auto learning rate: gd_lr {latest['gd_lr']:.4g} at iteration "
            f"{latest['iteration']}{kept}{spread}; each step moves the surface by up to "
            f"{first['fraction']:g} {unit} ({unit} = {first['length']:.3g}).")


# Both the config line and the first iteration banner land within the first few
# hundred lines of a log, so the head is enough to say what kind of solve wrote
# it -- and cheap enough for the viewer to ask on every redraw.
LOG_HEAD = 32_768


def optimization_saves(log_path: Path | None) -> int | None:
    """How often the run that wrote this log saved a shape, or None if it was
    a forward solve.

    Read from the solve's own log rather than from the settings now in the
    workflow, because the two can disagree: re-deciding an earlier card rewinds
    the state past the run that produced the surfaces still sitting on disk.
    What wrote them is what the viewer has to walk.

    A log from before run_forward recorded its save interval still says plainly
    that it was an optimization, and every such run saved every iteration.
    """
    if log_path is None:
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(LOG_HEAD)
    except OSError:
        return None
    if "run_mode=shape_optimization" not in head and not OPT_ITERATION.search(head):
        return None
    every = OPT_SAVE_EVERY.search(head)
    return max(int(every.group(1)), 1) if every else 1


def last_solve_failure(session_dir: Path) -> dict[str, Any] | None:
    """The record a failed solve left (``assets.SOLVE_FAILURE``), or None.

    ``started`` is when that solve was asked for, ``error`` why it stopped,
    ``log`` its own log if it got far enough to write one, and ``written``
    when it failed.  A solve that succeeds removes the record.
    """
    record = solve_failure(session_dir)
    if record is None:
        return None
    try:
        record["written"] = (session_dir / "solve" / SOLVE_FAILURE).stat().st_mtime
    except OSError:
        return None
    return record


def optimization_shown(status: dict[str, Any] | None) -> str | None:
    """One line on what a failed shape optimization leaves in the viewer.

    ``status`` is run_forward's ``optimization_status.json`` as the failure
    record carries it; None for a forward solve or a run from before it.
    """
    if not isinstance(status, dict) or status.get("status") != "failed":
        return None
    failed, shown = status.get("failed_iteration"), status.get("shown_iteration")
    if shown is None:
        return (f"The optimization failed at iteration {failed}, before any iteration "
                "solved, so the viewer shows no result of it.")
    return (f"The optimization failed at iteration {failed}; the viewer shows iteration "
            f"{shown}, the last that solved. The failed iteration's outputs are under "
            "failed_iteration/ in its run folder.")



def solver_failure_notes(failure: dict[str, Any] | None) -> list[str]:
    """The Solver panel's lines for a failed solve, in order.

    Plain notes, never the error itself: the panel used to open with a red box
    quoting the error's first line, which repeated the settings card's report
    (where the full error and the advice that goes with it are) in a narrow
    window that cut it off.  What is left says the run failed, what a failed
    optimization leaves in the viewer, and, for a solve that stopped before it
    wrote a log, why there are no residuals to show.
    """
    if failure is None:
        return []
    notes = ["The last solve failed; the settings card shows the error."]
    shown = optimization_shown(failure.get("optimization"))
    if shown:
        notes.append(shown)
    log = failure.get("log")
    if not log or not Path(log).is_file():
        notes.append("It stopped before the solver wrote a log.")
    return notes


def awaiting_solve(log_path: Path | None, requested: float | None) -> bool:
    """Whether a solve that was asked for has yet to write anything.

    Between the click and the solver's first line there is nothing new to read,
    and the monitor was filling that gap with the previous solve -- its progress
    bar complete, its residuals converged.  A log older than the request belongs
    to that previous run, so the panel waits instead of reporting it.
    """
    if not requested:
        return False
    if log_path is None:
        return True
    try:
        return log_path.stat().st_mtime < requested
    except OSError:
        return True


def read_backend_events(session_dir: Path, limit: int = 40) -> list[dict[str, Any]]:
    """Recent API calls and tool runs, oldest first, as display records.

    The three sources are written independently -- a JSONL transcript, one JSON
    receipt per tool run, and the solver's own log -- so they are merged on the
    timestamps they carry.
    """
    events = list(_api_events(session_dir)) + list(_tool_events(session_dir))
    events.sort(key=lambda event: event["timestamp"])
    return events[-limit:]


def _api_events(session_dir: Path) -> Iterator[dict[str, Any]]:
    transcript = session_dir / "api_log.jsonl"
    if not transcript.is_file():
        return
    try:
        lines = transcript.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        # A failed call is logged with its error and whatever it returned.
        failure = entry.get("error")
        yield {
            "timestamp": float(entry.get("timestamp", 0.0)),
            "kind": "api",
            "title": f"{entry.get('purpose', 'api')} · {entry.get('model', '')}"
                     + (" · cached" if entry.get("cached") else "")
                     + (" · failed" if failure else ""),
            "body": (f"request\n{entry.get('prompt', '')}\n\n"
                     f"response\n{entry.get('response', '')}"
                     + (f"\n\nerror\n{failure}" if failure else "")),
            "detail": _token_detail(entry),
        }


def _token_detail(entry: dict[str, Any]) -> str:
    tokens = entry.get("total_tokens")
    return f"{tokens} tokens" if tokens else ""


def _tool_events(session_dir: Path) -> Iterator[dict[str, Any]]:
    receipts = session_dir / "tool_receipts"
    if not receipts.is_dir():
        return
    for path in receipts.glob("*.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        streams = []
        for name in ("stdout_path", "stderr_path"):
            stream = receipt.get(name)
            text = _tail(Path(stream)) if stream else ""
            if text:
                streams.append(f"{name.removesuffix('_path')}\n{text}")
        yield {
            "timestamp": path.stat().st_mtime,
            "kind": "tool",
            "title": f"{receipt.get('tool', path.stem)} · exit {receipt.get('returncode')}",
            "body": "\n\n".join(streams) or "(no output)",
            "detail": ", ".join(Path(a).name for a in receipt.get("artifacts", [])),
        }


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text if len(text) <= limit else "…\n" + text[-limit:]


# However fast a solve is, the residual plot opens on five seconds, so the
# first iterations do not arrive spread across an axis that reads in
# hundredths and then compress as the run goes on.
MINIMUM_SPAN = 5.0


def residual_chart(
    iterations: list[dict[str, float]],
    objective: list[dict[str, float]] | None = None,
) -> alt.Chart:
    """Residual history against the wallclock, broken between load steps.

    Every iteration is drawn at the second it was reached, so the horizontal
    distance between two points is the time that passed between them -- which
    is what says whether a run is converging fast or merely converging.  The
    gaps between one load step's line and the next are part of the reading: a
    step boundary costs a mesh rebuild that no iteration count shows.

    Residuals fall by orders of magnitude, so the vertical scale is log: on a
    linear axis the first drop uses the whole height and the rest lies flat on
    zero.  Each load step is its own line because the residual restarts when the
    load steps up -- joining across that boundary would draw a climb that never
    happened -- and every iteration gets a marker, since there are only a
    handful and where they sit is the reading.

    The axis starts at zero and spans at least a second, so a solve is watched
    on a fixed scale that grows rather than one that rescales under every point
    that lands.  A log with no timestamps to read is plotted against the
    iteration ordinal instead.

    A shape optimization breaks the same way at every one of its iterations:
    each is a whole Newton solve on a shape that has just moved, so its
    residual restarts exactly as a load step's does, and joining across the
    boundary would draw a climb that never happened.

    ``objective`` is what the optimizer is minimising, one reading per
    iteration.  It shares the clock but nothing else -- it is not a residual
    and is not measured in orders of magnitude -- so it is layered on its own
    scale against the right-hand axis.
    """
    timed = bool(iterations) and all("elapsed" in row for row in iterations)
    field = "seconds" if timed else "step"
    # Only against the clock: the objective is logged once per iteration and the
    # residuals many times within one, so there is no shared ordinal to plot it
    # against when the log carries no timestamps.
    readings = [row for row in (objective or []) if "elapsed" in row] if timed else []
    frame = pd.DataFrame(
        [
            {
                field: row["elapsed"] if timed else index,
                "series": series,
                "load step": int(row["load_step"]),
                "opt iteration": int(row.get("opt_iteration", 0)),
                "log residual": math.log10(max(row[source], 1e-300)),
            }
            for index, row in enumerate(iterations)
            for series, source in (("absolute", "residual"), ("relative", "relative"))
        ]
    )
    if timed:
        # Over both series: the objective is logged after the iteration's
        # residuals, so a span read off the residuals alone would leave the last
        # reading of the thing being minimised outside the axis.
        span = max(MINIMUM_SPAN,
                   *(row["elapsed"] for row in iterations),
                   *(row["elapsed"] for row in readings))
        axis = alt.X(
            "seconds:Q",
            title=None,
            scale=alt.Scale(domain=[0, span], nice=False, clamp=True),
            axis=alt.Axis(labelExpr="datum.value + 's'"),
        )
    else:
        axis = alt.X("step:Q", title=None, axis=alt.Axis(tickMinStep=1))
    residuals = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=axis,
            y=alt.Y("log residual:Q", title=None),
            color=alt.Color("series:N", title=None),
            # Two fields rather than one: a forward solve breaks only between
            # load steps, an optimization only between iterations, and a run
            # that did both would have to break at either.
            detail=[alt.Detail("load step:N"), alt.Detail("opt iteration:N")],
        )
    )
    if not readings:
        return residuals
    values = (
        alt.Chart(
            pd.DataFrame(
                [
                    {
                        "seconds": row["elapsed"],
                        "series": "objective",
                        "objective": row["value"],
                    }
                    for row in readings
                ]
            )
        )
        .mark_line(point=True)
        .encode(
            x=axis,
            y=alt.Y(
                "objective:Q",
                title=None,
                # The reading is how far the objective has come down, not how
                # far it is from zero, so the axis frames the range the
                # readings occupy rather than stretching from an origin
                # nothing approaches.
                scale=alt.Scale(zero=False),
                axis=alt.Axis(orient="right"),
            ),
            color=alt.Color("series:N", title=None),
        )
    )
    return alt.layer(residuals, values).resolve_scale(y="independent")


def iteration_numbers(count: int, save_every: int | None) -> list[int]:
    """Which optimization iteration each saved surface came from.

    The writer numbers surfaces in the order it wrote them, so the fourth file
    is the fourth save rather than the fourth iteration.  A run saving every
    tenth iteration has to be labelled with what it actually saved, or the
    slider claims a shape belongs to an iteration that never wrote one.
    """
    every = max(int(save_every or 1), 1)
    return [index * every for index in range(max(count, 0))]


# How many shapes the viewer carries at once.  Each is a field asset of roughly
# a megabyte, decoded into a mesh of its own, so this is a memory budget rather
# than a preference: a hundred-iteration run would otherwise hand the page a
# hundred megabytes and a hundred meshes to hold at once.
VIEWER_FRAMES = 16


def chosen_iterations(count: int, cap: int = VIEWER_FRAMES) -> list[int]:
    """Which of a finished run's surfaces to carry into the viewer, in order.

    A short run is carried whole.  A long one is thinned evenly, keeping the
    first and the last, so the slider spans the whole optimization at a coarser
    step rather than covering its first dozen iterations and stopping.  Every
    position is labelled with the iteration it came from, so the step between
    them is read off the slider rather than assumed.

    A run still writing is filled differently -- see the live bridge, which
    favours the most recent shapes over an even spread, because what an
    optimization is doing now is what someone watching it is there for.
    """
    if count <= 0:
        return []
    if count <= cap:
        return list(range(count))
    return sorted({round(index * (count - 1) / (cap - 1)) for index in range(cap)})


def next_live_frame(seen: int, count: int, held: int,
                    cap: int = VIEWER_FRAMES) -> tuple[int, int | None] | None:
    """The next shape to hand a running viewer, and which of its own makes way.

    ``seen`` is the highest surface already handed over -- a high-water mark,
    not a record of what the viewer still holds.  The two differ once the viewer
    is full and starts discarding, and reading the second as the first is what
    made a run jump backwards: a shape dropped to make room read as one that had
    never been sent, so it was sent again, and the picture fell ten iterations
    behind before the next one pulled it forward.

    Only the newest is offered.  Building one takes longer than an iteration
    does, so a bridge working through a backlog would fall further behind the
    run it is meant to be showing; the slider is labelled with the iteration
    each shape came from, so a skipped one is visible rather than filled in.

    The frame that makes way is the second, never the first: the shape the run
    started from is what the newest one is worth comparing against.
    """
    newest = count - 1
    if newest <= seen:
        return None
    return newest, (1 if held >= cap else None)


TOKEN_BUDGET_START = 10_000


def session_token_usage(workflow: dict[str, Any]) -> tuple[int, int]:
    """Input and output tokens billed across a whole conversation.

    A cached call cost nothing the second time, but its receipt still carries
    the counts of the call that filled the cache, so cached receipts are left
    out rather than charged twice.
    """
    total_in = total_out = 0
    for receipt in workflow.get("api_receipts") or []:
        if not isinstance(receipt, dict) or receipt.get("cached"):
            continue
        total_in += int(receipt.get("input_tokens") or 0)
        total_out += int(receipt.get("output_tokens") or 0)
    return total_in, total_out


def token_budget(total: int, start: int = TOKEN_BUDGET_START) -> int:
    """The bar's scale: start, doubling each time the usage fills it.

    The bar keeps its length and the scale behind it doubles, so a long
    conversation stays readable instead of pinning at full.
    """
    budget = max(start, 1)
    while total > budget:
        budget *= 2
    return budget


def token_bar_markup(workflow: dict[str, Any]) -> str:
    """A fixed-length bar showing what the conversation has spent.

    The scale behind it doubles each time usage fills it, so the bar keeps its
    length and a long conversation stays readable instead of pinned at full.
    """
    used_in, used_out = session_token_usage(workflow)
    budget = token_budget(used_in + used_out)

    def share(value: int) -> float:
        return 100.0 * value / budget

    def compact(value: int) -> str:
        return f"{value / 1000:.1f}k" if value >= 1000 else str(value)

    title = (
        f"{used_in + used_out:,} of {budget:,} tokens - "
        f"{used_in:,} in, {used_out:,} out, uncached calls only"
    )
    return (
        f'<div class="token-meter" title="{html.escape(title)}">'
        f'<div class="token-bar">'
        f'<div class="token-fill token-in" style="width:{share(used_in):.4g}%"></div>'
        f'<div class="token-fill token-out" style="width:{share(used_out):.4g}%"></div>'
        f"</div>"
        f'<span class="token-label">{compact(used_in + used_out)}'
        f"/{compact(budget)}</span></div>"
    )


# How far the deformation slider goes: enough to move the surface by this much
# of the object, and no further.  Real deflections are a fraction of a percent
# of the part, so a true-scale picture is indistinguishable from the undeformed
# one; three tenths is visibly bent without being a caricature.
DEFORMATION_REACH = 0.3
DEFORMATION_STEPS = 10


def contour_shows(contour: Path, surface: Path | None) -> bool:
    """Whether the solve-time displacement asset was coloured from ``surface``.

    The exporter records its source surface; an optimization rewrites the asset
    while it runs, and saves land asynchronously, so a file's age alone can
    claim the newest iteration for an asset built from the one before.  Assets
    from before the source was recorded fall back to that age test.
    """
    if surface is None or not contour.is_file():
        return False
    try:
        meta = json.loads(contour.with_suffix(contour.suffix + ".json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    source = meta.get("source") if isinstance(meta, dict) else None
    try:
        if source:
            return Path(source).resolve() == Path(surface).resolve()
        return contour.stat().st_mtime >= Path(surface).stat().st_mtime
    except OSError:
        return False


def _largest_displacement(solve_root: Path) -> float | None:
    """The largest displacement the solve's assets record, newest first.

    The solve-time asset when there is one; otherwise any asset exported from
    the solve's surfaces, each of which records ``max_displacement`` -- a
    session whose solve-time asset was removed (a failed run from before
    run_forward kept the last good iteration's) still gets its slider.
    """
    try:
        return float(json.loads((solve_root / "displacement_magnitude.glb.json").read_text(
            encoding="utf-8"))["maximum"])
    except (OSError, KeyError, TypeError, ValueError):
        pass
    for sidecar in sorted(solve_root.glob("*.glb.json"),
                          key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            value = float(json.loads(sidecar.read_text(encoding="utf-8"))["max_displacement"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if value > 0.0:
            return value
    return None


def deformation_scales(root: Path, workflow: dict) -> list[float]:
    """The slider's values: zero, then up to the reach, in equal steps.

    Sized from the largest displacement the solve produced, so the top of the
    slider means the same thing -- a visibly deformed shape -- whatever the part
    and however hard it was pushed.
    """
    largest = _largest_displacement(root / "solve")
    if largest is None:
        return []
    extent = workflow.get("physical_extent")
    try:
        extent = float(extent)
    except (TypeError, ValueError):
        return []
    if largest <= 0.0 or extent <= 0.0:
        return []
    top = DEFORMATION_REACH * extent / largest
    return [top * index / (DEFORMATION_STEPS - 1) for index in range(DEFORMATION_STEPS)]


def surface_deformation(asset: Path, scales: list[float]) -> dict[str, Any] | None:
    """The displacements beside an asset, ready to hand to the viewer.

    Written by the exporter in the asset's own vertex order.  Absent for a solve
    that predates them, or for an asset with nothing to move, in which case the
    viewer simply shows no slider.
    """
    if not scales:
        return None
    payload = asset.with_suffix(asset.suffix + ".disp.bin")
    try:
        raw = payload.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "max_scale": scales[-1],
        "stops": len(scales),
    }
