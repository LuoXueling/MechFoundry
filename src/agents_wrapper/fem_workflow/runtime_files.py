"""Where the workflow's runtime files live, and how to read them, in the standard library alone.

One copy for every front end: the workflow itself (monitor, workers, assets), the app's
telemetry, and the operator runner's ``sessionlib``.  The runner loads this file by its path
rather than through the package, whose ``__init__`` builds the graph and so needs LangGraph,
which the environments its helpers run in may not have.  Keep it that way: standard library
only, no imports from the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import sys
import tempfile
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
