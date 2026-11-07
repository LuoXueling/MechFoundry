"""Where the workflow's runtime files live, and how to read them, in the standard library alone.

One copy for every front end: the workflow itself (monitor, workers, assets), the app's
telemetry, and the operator runner's ``sessionlib``.  The runner loads this file by its path
rather than through the package, whose ``__init__`` builds the graph and so needs LangGraph,
which the environments its helpers run in may not have.  Keep it that way: standard library
only, no imports from the package.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# ------------------------------------------------------------ interpreters --

# The project was renamed MechFoundry (2026-10), and its conda environments with it:
# ``implicit-*`` became ``mechfoundry-*``, and PartField's ``partfield`` (which had no prefix)
# followed as ``mechfoundry-partfield`` (2026-10-23).  Every session, checkpoint and batch record written
# before then stores the old name (``compute_env: "implicit-warp117"`` in session.json and in
# the checkpointed state), and those records are never rewritten.  So every place that turns a
# stored name into an interpreter or a ``conda run -n`` asks ``current_environment`` first: the
# stored name while that environment is still installed (a machine not yet migrated runs as
# before), its new name once the old one is gone.  The one table, so no front end can disagree
# with another about which name a session's workers are keyed and launched under.
RENAMED_ENVIRONMENTS = {
    "implicit-warp117": "mechfoundry-warp117",
    "implicit-agents": "mechfoundry-agents",
    "implicit-text2cad": "mechfoundry-text2cad",
    "implicit-cadcoder": "mechfoundry-cadcoder",
    "implicit-hunyuan3d": "mechfoundry-hunyuan3d",
    "implicit-vomp": "mechfoundry-vomp",
    "partfield": "mechfoundry-partfield",
}


def current_environment(environment: str) -> str:
    """``environment``, or its MechFoundry name when it is a pre-rename name no longer installed.

    The old name wins while its interpreter is found, so a session keeps the environment it was
    made in; when neither interpreter is found (conda's root not discoverable from here) the new
    name is returned, because ``conda run -n`` with it is what will work after the rename.
    """
    renamed = RENAMED_ENVIRONMENTS.get(environment)
    if renamed is None or _find_interpreter(environment) is not None:
        return environment
    return renamed


def direct_interpreter(environment: str) -> Path | None:
    """A conda environment's own interpreter, or None when it cannot be found (then: conda run).

    Under conda's root as the shell's conda names it (CONDA_EXE, CONDA_ROOT, CONDA_PREFIX_1), else
    beside the interpreter running this (its own ``envs/`` folder, or the base it belongs to).
    Calling it directly skips conda run, which costs 1.7 s a call (2026-10-09) to start conda's
    own Python before the one wanted.  A pre-rename name finds its renamed environment once the
    old one is gone (``current_environment``).
    """
    return _find_interpreter(current_environment(environment))


def _find_interpreter(environment: str) -> Path | None:
    """``direct_interpreter`` for exactly this name, with no rename applied."""
    if not environment or Path(environment).name != environment:
        # Never build a filesystem path out of a name carrying separators.
        return None
    roots = []
    executable = os.environ.get("CONDA_EXE")
    if executable:
        roots.append(Path(executable).resolve().parents[1])
    for variable in ("CONDA_ROOT", "CONDA_PREFIX_1"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).resolve())
    running = Path(sys.executable)
    roots += [parent.parent for parent in running.parents if parent.name == "envs"][:1]
    if len(running.parents) > 1:
        roots.append(running.parents[1])
    for root in roots:
        candidate = root / "envs" / environment / "bin" / "python"
        if candidate.is_file():
            return candidate
    return None


HANDED_OVER = "FEM_OPERATOR_HANDED_OVER"


def run_in_environment(environment: str, script: str, argv: list[str]) -> None:
    """Make this process ``script`` under ``environment``'s own interpreter, unless it already is.

    The operator runner's commands are typed as ``python3 -u .../drive.py ...``: whichever Python
    3 answers parses the script (standard library only up to here) and hands it over with
    ``os.execv``, with no conda in between; conda run only when the environment's interpreter
    cannot be found.  ``HANDED_OVER`` keeps an environment found under another name from handing
    over in a loop.
    """
    environment = current_environment(environment)
    if Path(sys.prefix).name == environment or os.environ.get(HANDED_OVER) == environment:
        return
    os.environ[HANDED_OVER] = environment
    interpreter = direct_interpreter(environment)
    command = ([os.fspath(interpreter), "-u", script, *argv] if interpreter is not None else
               ["conda", "run", "--no-capture-output", "-n", environment, "python", "-u", script, *argv])
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        # Said as the runner's own result, so the operator skill needs no separate probe of
        # conda (its `conda --version` call cost each run one API call of the whole context).
        print(json.dumps({"ok": False, "error": f"neither the {environment} environment's interpreter nor "
                                                f"conda was found ({type(exc).__name__}: {exc}): this "
                                                "machine cannot run the workflow"}), flush=True)
        raise SystemExit(2) from None


# ---------------------------------------------------------------- workers --


def worker_key(output_root: Path, compute_env: str) -> str:
    """The key a resident worker's socket and pid file are named by.

    The root is resolved before it is hashed: keyed on the spelling, a worker
    started from a relative ``--output-root`` was one the monitor, which spells
    it absolutely, never called and a stop by the absolute path never found.
    The environment is keyed by its current name, so a resumed pre-rename
    session and a new one find the same workers (``current_environment``).
    """
    output_root = Path(output_root).expanduser().resolve()
    compute_env = current_environment(compute_env)
    return hashlib.sha256(f"{output_root}\n{compute_env}".encode()).hexdigest()[:12]


def worker_socket(key: str, kind: str) -> Path:
    """The socket of a resident worker of this kind under ``key``.

    The socket does not live beside the sessions, because a Unix socket path is
    limited to about 108 bytes: measured from this repository the natural place
    for it comes to 83, and an installation nested a little deeper would fail at
    bind time -- silently, since the worker is detached, and endlessly, since
    nothing would then stop the next page draw starting another.  It gets a
    short directory of its own instead.
    """
    return Path(tempfile.gettempdir()) / f"implicit-fem-{key}" / f"{kind}.sock"


def worker_paths(output_root: Path, kind: str, compute_env: str) -> tuple[Path, Path]:
    """Where a resident worker of this kind announces itself: (socket, pid file).

    Keyed by output root and compute environment, so two installations or two
    environments never answer each other's calls, and every session sharing one
    warms a single worker rather than one apiece.  The pid stays with the
    session data, where it can be found and killed.
    """
    key = worker_key(output_root, compute_env)
    output_root = Path(output_root).expanduser().resolve()
    return worker_socket(key, kind), output_root / ".workers" / f"{kind}-{key}.pid"


def process_state(pid: int) -> str | None:
    """The state letter ``/proc`` gives this pid (R, S, Z ...), or None if there is none.

    The command sits in parentheses and may contain spaces or parentheses of its
    own, so the state is read from after the last one.
    """
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return line[line.rindex(")") + 1:].split()[0]
    except (OSError, ValueError, IndexError):
        return None


def exited(pid: int) -> bool:
    """Whether this pid has run to completion and is only awaiting collection.

    A worker is started with ``start_new_session``, which gives it its own
    session but leaves it a child of whoever started it.  When it exits -- on
    its idle timeout, most often -- it stays in the process table as a zombie
    until that parent collects it, and a zombie answers signal 0 exactly as a
    running process does.  The state in ``/proc`` is what separates them, and it
    is readable whether or not this is the process that launched it.
    """
    return process_state(pid) == "Z"


def stop_worker_at(socket_path: Path, pid_path: Path, kind: str) -> None:
    """Best-effort stop of a resident worker, including one still warming."""
    try:
        pid = int(pid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pid = None
    if socket_path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(3.0)
                client.connect(os.fspath(socket_path))
                # Ask first, then read one line.  Every other model server
                # greets a connection before reading it, but TripoSG's reads
                # the request first, so waiting for a greeting sat out the
                # whole timeout each time TripoSG made way -- 3 s between
                # skipping the texture and the scaling card, where PartField
                # takes its place.  A greeting server buffers its greeting and
                # reads the request already waiting; TripoSG answers it.  Either
                # way a line comes back, and staying for it keeps the
                # connection open until the request has been read.
                client.sendall(b'{"action":"shutdown"}\n')
                client.makefile("r", encoding="utf-8").readline()
        except OSError:
            pass
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    pid_path.unlink(missing_ok=True)
    socket_path.unlink(missing_ok=True)
    # Stopped on purpose, so its launch did not fail: switching back to this
    # model straight away must not wait out the relaunch backoff.
    (pid_path.parent / f"{kind}.attempted").unlink(missing_ok=True)


# ----------------------------------------------------------------- solves --

# One solve per session.  On 2026-11-06 the sidebar re-ran a session's pending solve each time
# the conversation was opened, and nothing said one was running: eight solves of one case began
# in four minutes on GPU 0, and the out-of-memory errors they caused one another came back as
# that case's failure, while three copies went on for an hour behind the failure card.
#
# Two kernel locks (flock), because no one process is there for a whole solve.  The monitor holds
# the claim from before it touches anything until the solve returns, so a refused duplicate has
# unloaded no model and written no receipt.  The solve holds the run lock itself, in whichever
# process it lands: a one-shot launch, the warm solver worker, or the operator's backend run.
# That one is what outlives its parent, as a solve does when Streamlit is restarted beneath it.
# The kernel drops a lock with its holder, so a killed process leaves nothing stale to clear, and
# each open has its own file description, so threads of one process conflict as processes do.
SOLVE_CLAIM_LOCK = ".solve_claim.lock"
SOLVE_RUN_LOCK = ".solve_run.lock"
# Written before a requested stop signals anything, and left until the next solve is claimed:
# it is what tells "stopped on request" from a kill nobody asked for, for the graph, and for
# the solver panel, whose log of a stopped run never says it finished.
SOLVE_STOP_REQUEST = ".solve_stop_requested"
# A stop asked for before there is a process to signal.  Stop FEA is on offer from the click on
# Run FEA, and until the solve holds its run lock (a job in the queue, the case being built,
# models unloading, a spare warming, torch importing: a minute, in the worst case) no wait is
# long enough, and on 2026-11-06 a click there said "Nothing was stopped" and the solve then ran.
# So the request outlives the click: the monitor reads it after its claim and again just before
# it launches, and ``solve_process`` reads it once it holds the lock.  The stop writes this and
# then looks for the lock, the solve takes the lock and then reads this, so whichever is first,
# one of them sees the other, on every route.  Read, it becomes SOLVE_STOP_REQUEST.
#
# It holds the identity of the process that asked, because it lives on disk and must never stop
# a later run: left by a server that has since gone, it is void.  From one still up, a new Run
# FEA clears it, as does the end of the solve it was for, and of the job that asked.
SOLVE_STOP_PENDING = ".solve_stop_pending"
# How long a stopped solve is given to go on SIGTERM before SIGKILL.
SOLVE_STOP_GRACE = 5.0
# How long a stop keeps the page waiting to see a claimed solve end or reach its run lock, for
# the page's sake: the request stands however long the solve takes to get there.
SOLVE_START_WAIT = 30.0
SOLVE_RUNNING_MESSAGE = (
    "A solve is already running for this session, and it was left undisturbed. "
    "Wait for it to finish before running again."
)
# ``solve_running`` takes a lock for an instant to read it, which a claim must not mistake for a
# solve: a few short tries see past a probe, and a real holder is there for minutes.
LOCK_ATTEMPTS = 4
LOCK_RETRY_SECONDS = 0.05


class SolveAlreadyRunning(RuntimeError):
    """This session has a solve running.  A refusal, not a failed solve: nothing was attempted,
    so the failure path (dropping the viewer's assets, the failure card) must not run on it."""

    def __init__(self, message: str = SOLVE_RUNNING_MESSAGE):
        super().__init__(message)


class SolveStopped(RuntimeError):
    """A stop reached this solve before it began.  With the request on record, the graph takes
    it down the stopped path, not the failure path, as it does a solve signalled while running."""

    def __init__(self, message: str = "the solve was stopped before it began"):
        super().__init__(message)


def _hold_lock(path: Path) -> int | None:
    """The descriptor holding ``path``'s lock, or None while someone else holds it."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    for attempt in range(LOCK_ATTEMPTS):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except BlockingIOError:
            if attempt + 1 < LOCK_ATTEMPTS:
                time.sleep(LOCK_RETRY_SECONDS)
    os.close(descriptor)
    return None


def _lock_held(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:           # never locked: reading must not leave a file in an old session
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)


def solve_running(session_dir: Path | str) -> bool:
    """Whether a solve of this session is claimed or running, in any process."""
    root = Path(session_dir)
    return _lock_held(root / SOLVE_CLAIM_LOCK) or _lock_held(root / SOLVE_RUN_LOCK)


@contextlib.contextmanager
def solve_claim(session_dir: Path | str):
    """Hold the session's one solve for this block, or raise ``SolveAlreadyRunning``."""
    root = Path(session_dir)
    descriptor = _hold_lock(root / SOLVE_CLAIM_LOCK)
    if descriptor is None:
        raise SolveAlreadyRunning()
    try:
        if _lock_held(root / SOLVE_RUN_LOCK):    # a solve whose monitor has gone
            raise SolveAlreadyRunning()
        clear_solve_stop(root)      # that of a run nobody was left to report, or a lost race
        yield
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def solve_process(session_dir: Path | str):
    """Mark the solve itself as running, for as long as it runs."""
    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = _hold_lock(root / SOLVE_RUN_LOCK)
    if descriptor is None:
        raise SolveAlreadyRunning()
    try:
        # Who holds it, for stop_solve.  The lock says someone does; this says who.
        os.pwrite(descriptor, json.dumps(_process_identity(os.getpid())).encode(), 0)
        if consume_pending_stop(root):
            raise SolveStopped()
        yield
    finally:
        os.ftruncate(descriptor, 0)
        os.close(descriptor)


def _process_identity(pid: int) -> dict[str, Any] | None:
    """What makes this pid this process: a pid is reused, its command and start time are not.

    The start time is what separates a solve from a later solve of the same session, whose
    command line is identical.
    """
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        started = line[line.rindex(")") + 1:].split()[19]
    except (OSError, ValueError, IndexError):
        return None
    return {"pid": pid, "command": command.hex(), "started": started}


def _descendants(pid: int) -> list[int]:
    """Every process below this one.  By the parent links and never by process group: a one-shot
    solve shares its group with whoever launched it, Streamlit included."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = (entry / "stat").read_text(encoding="utf-8")
            children.setdefault(int(line[line.rindex(")") + 1:].split()[1]), []).append(
                int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    found: list[int] = []
    waiting = [pid]
    while waiting:
        below = children.get(waiting.pop(), [])
        found += below
        waiting += below
    return found


def _still_running(identity: dict[str, Any]) -> bool:
    return (_process_identity(identity["pid"]) == identity
            and process_state(identity["pid"]) not in (None, "Z", "X"))


def _signal_all(family: list[dict[str, Any]], number: int) -> None:
    for member in family:
        if _still_running(member):       # read again at the signal: the pid may be another's now
            with contextlib.suppress(OSError):
                os.kill(member["pid"], number)


def stop_solve(session_dir: Path | str, grace: float = SOLVE_STOP_GRACE,
               start_wait: float = 0.0, asked: bool = False) -> bool:
    """End the session's solve on request; False if there was none to reach.

    SIGTERM, then SIGKILL after ``grace``, to the process holding the run lock and those below
    it, whichever route launched it, and only while each is still the process recorded.  With
    no such process yet, for a solve claimed or, with ``asked``, one this caller has just asked
    for and that has not reached its claim, the request is left for it to find
    (SOLVE_STOP_PENDING) and True means it will not run; ``start_wait`` is how long to stay and
    see that through.  A solve already stopped is not a failure to stop one, so a second click
    is True too.  False for a solve that ended first, and for one started before the holder
    recorded itself, which has no record to verify.
    """
    root = Path(session_dir)
    requested = False
    if not _lock_held(root / SOLVE_RUN_LOCK):
        claimed = _lock_held(root / SOLVE_CLAIM_LOCK)
        if not claimed and not asked:
            return False
        if claimed and solve_stop_requested(root):
            return True
        request_solve_stop(root)
        requested = True
        deadline = time.monotonic() + start_wait
        while (time.monotonic() < deadline and _lock_held(root / SOLVE_CLAIM_LOCK)
               and pending_stop(root) and not _lock_held(root / SOLVE_RUN_LOCK)):
            time.sleep(0.05)
        if not _lock_held(root / SOLVE_RUN_LOCK):
            # Gone with nothing stopped: the solve finished ahead of the request.
            return pending_stop(root) or solve_stop_requested(root)
    try:
        holder = json.loads((root / SOLVE_RUN_LOCK).read_text(encoding="utf-8"))
        family = [holder]
        running = _still_running(holder)
    except (OSError, ValueError, KeyError, TypeError):
        return requested       # just taking the lock: it reads the request next
    if not running:
        return requested
    (root / SOLVE_STOP_REQUEST).write_text(str(time.time()), encoding="utf-8")
    for pid in _descendants(holder["pid"]):
        identity = _process_identity(pid)
        if identity is not None:
            family.append(identity)
    _signal_all(family, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while any(_still_running(member) for member in family) and time.monotonic() < deadline:
        time.sleep(0.05)
    _signal_all(family, signal.SIGKILL)
    deadline = time.monotonic() + SOLVE_STOP_GRACE
    while _lock_held(root / SOLVE_RUN_LOCK) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _lock_held(root / SOLVE_RUN_LOCK)


def solve_stop_requested(session_dir: Path | str) -> bool:
    return (Path(session_dir) / SOLVE_STOP_REQUEST).exists()


def clear_solve_stop(session_dir: Path | str) -> None:
    (Path(session_dir) / SOLVE_STOP_REQUEST).unlink(missing_ok=True)


def request_solve_stop(session_dir: Path | str) -> None:
    """Ask that the solve on its way not run; ``consume_pending_stop`` is the other half."""
    root = Path(session_dir)
    written = root / f"{SOLVE_STOP_PENDING}.{os.getpid()}.{threading.get_ident()}"
    written.write_text(json.dumps(_process_identity(os.getpid())), encoding="utf-8")
    os.replace(written, root / SOLVE_STOP_PENDING)   # whole, for a reader in another process


def clear_pending_stop(session_dir: Path | str) -> None:
    (Path(session_dir) / SOLVE_STOP_PENDING).unlink(missing_ok=True)


def pending_stop(session_dir: Path | str) -> bool:
    """Whether a stop is waiting for its solve, and whoever asked is still there to mean it."""
    path = Path(session_dir) / SOLVE_STOP_PENDING
    try:
        return _still_running(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def consume_pending_stop(session_dir: Path | str) -> bool:
    """Turn a live pending stop into the record of a requested stop; True if there was one.

    One atomic rename, so of two readers only one finds it.  A void one is removed, unread.
    """
    root = Path(session_dir)
    if not pending_stop(root):
        clear_pending_stop(root)
        return False
    try:
        os.replace(root / SOLVE_STOP_PENDING, root / SOLVE_STOP_REQUEST)
    except FileNotFoundError:
        return False
    return True


# What a failed solve leaves in ``solve/``: when it was asked for (``started``), why it stopped
# (``error``) and its own log if it got far enough to write one.  A solve that succeeds removes it.
SOLVE_FAILURE = "failed.json"
# Written by run_forward beside a shape optimization's log: which iteration the
# viewer is shown and, for a failed run, which iteration failed.
OPTIMIZATION_STATUS = "optimization_status.json"


def newest_solve_log(session_dir: Path) -> Path | None:
    """The log of the most recent solve, which is the one worth watching."""
    logs = [path for path in (Path(session_dir) / "solve").glob("*/log.txt") if path.is_file()]
    return max(logs, key=lambda path: path.stat().st_mtime, default=None)


def solve_failure_in(solve_root: Path) -> dict[str, Any] | None:
    """The record the last failed solve left in this ``solve/`` folder, or None."""
    try:
        record = json.loads((Path(solve_root) / SOLVE_FAILURE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def solve_failure(session_dir: Path) -> dict[str, Any] | None:
    """The record the session's last failed solve left (``solve/failed.json``), or None."""
    return solve_failure_in(Path(session_dir) / "solve")
