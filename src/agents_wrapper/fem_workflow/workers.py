"""Starting and keeping the resident workers, for whichever front end is driving.

The monitor already *uses* a worker whenever one is listening, so the app and the
command line both get the warm path for free -- but only something has started
one.  That used to live in the Streamlit controller, which meant a command-line
run always paid the cold start, and got the fast path only by accident when an
app session happened to have left a worker behind.  Both call in here instead.

Nothing about correctness depends on any of this: every worker is an
optimisation the monitor falls back from, so a front end that never calls these
functions behaves exactly as it did before they existed.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .monitor import CAD_GENERATORS, DEFAULT_COMPUTE_ENV, SOLVER_WARM_WAIT, _direct_interpreter
from .runtime_files import current_environment, exited, stop_worker_at, worker_paths
from src.partfield_wrapper.availability import partfield_status, settings as partfield_settings
from src.hunyuan3d_wrapper.availability import hunyuan3d_status, settings as hunyuan3d_settings

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"

# How long to leave a worker alone after a launch that did not survive.  The
# preloads are called repeatedly -- on every rerun in the app -- so without this
# one that cannot start (a broken environment, a socket path the system will not
# take) would be started again several times a second.
WORKER_RELAUNCH_BACKOFF = 60.0

# A torch extension build lock older than this is left over, whoever's cache
# it is in: no build takes this long.  See _clear_stale_build_locks.
STALE_BUILD_LOCK_AGE = 1800.0

# A warm request is worth almost nothing if it has to wait: the tool that needs
# those imports will make them itself a second or two later.  Kept short because
# the app calls this while drawing the page, and a worker busy with a
# segmentation would otherwise hold the page still for the whole of it.
WARM_REQUEST_TIMEOUT = 0.5

# Which tier has been asked of which worker, identified by the pid serving it,
# so a restarted worker is asked again and a running one is not asked twice.
_REQUESTED_TIER: dict[str, tuple[str, str]] = {}


# Whether a pid is a zombie awaiting collection (runtime_files.exited explains why it matters).
_exited = exited


def worker_is_serving(socket_path: Path, pid_path: Path) -> bool:
    """Whether a resident worker is serving, or on its way to serving.

    The pid says so, not the socket.  A worker binds its socket only once it is
    warm, so between launch and readiness there is a live process and no socket
    -- and a worker killed outright leaves a socket behind with no process,
    which taken as proof of life wedges preloading for the rest of a session
    while every call quietly goes back to a cold start.  Whoever finds the
    remains clears them, so the next attempt starts a worker rather than
    trusting one.

    A worker that idled out is the same wedge wearing a disguise: it has exited,
    but until its parent collects it the pid still answers, so the app that
    started it would go on believing in it and never start another.  It is
    collected here when this is the process that started it, and recognised as
    gone either way.
    """
    try:
        pid = int(pid_path.read_text(encoding="utf-8"))
        with contextlib.suppress(ChildProcessError):
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                raise ProcessLookupError(pid)
        if _exited(pid):
            raise ProcessLookupError(pid)
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        pid_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
        return False


def ensure_solver_worker(output_root: Path,
                         compute_env: str = DEFAULT_COMPUTE_ENV) -> bool:
    """Keep one forward-solve process started ahead of the solve.

    A solve runs in a process of its own, so that it has the GPU to itself and
    gives all of it back -- and so every solve used to pay ~5s of interpreter,
    torch, warp and solver imports, a CUDA context and AMGX's initialisation
    before its first line of work.  This pays all of that while the settings
    card is being read.  The process serves exactly one solve and exits
    (``tool_server --one-shot``), so the isolation is the same as before; the
    card that follows a solve asks for the next one.
    """
    output_root = Path(output_root)
    compute_env = compute_env or DEFAULT_COMPUTE_ENV
    socket_path, pid_path = worker_paths(output_root, "solver", compute_env)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        try:
            warming_for = time.time() - pid_path.stat().st_mtime
        except OSError:
            warming_for = 0.0
        if socket_path.exists() or warming_for < SOLVER_WARM_WAIT:
            return True
        # Alive, long past a normal warm-up and still not serving: replace it
        # rather than trust it, or every later solve would wait on it in turn.
        stop_worker(output_root, "solver", compute_env)
    interpreter = _direct_interpreter(compute_env)
    launcher = ([str(interpreter)] if interpreter is not None
                else ["conda", "run", "--no-capture-output", "-n", current_environment(compute_env),
                      "python"])
    cache = output_root / ".runtime_cache"
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    return _launch(
        output_root, "solver", compute_env,
        [*launcher, "-u", str(SCRIPT_DIR / "tool_server.py"),
         "--serve", str(socket_path), "--warm", "solver", "--one-shot"],
        cwd=output_root,
        # The environment ToolMonitor gives a solve it launches itself.
        cache=cache,
        env={"MPLCONFIGDIR": str(cache / "matplotlib"),
             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


def stop_solver_worker(output_root: Path,
                       compute_env: str = DEFAULT_COMPUTE_ENV) -> None:
    stop_worker(output_root, "solver", compute_env or DEFAULT_COMPUTE_ENV)


def request_warm(socket_path: Path, tier: str, serving: str) -> None:
    """Ask a running worker to pay a deeper tier's imports now, once.

    Best effort, and deliberately not waited on: the greeting says the worker
    heard us, and the imports themselves take seconds that belong to the
    background rather than to the caller.  A worker still warming has no socket
    yet, a busy one does not greet in time, and one that has gone away is the
    next call's problem -- in every case the tool that needs those imports still
    gets them, it just pays for them.
    """
    key = os.fspath(socket_path)
    if _REQUESTED_TIER.get(key) == (serving, tier) or not socket_path.exists():
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(WARM_REQUEST_TIMEOUT)
            client.connect(os.fspath(socket_path))
            client.makefile("r", encoding="utf-8").readline()
            client.sendall((json.dumps({"action": "warm", "tier": tier}) + "\n").encode())
        _REQUESTED_TIER[key] = (serving, tier)
    except OSError:
        pass


def stop_worker(output_root: Path, kind: str, environment: str) -> None:
    """Best-effort stop of an owned model worker, including one still warming."""
    stop_worker_at(*worker_paths(Path(output_root), kind, environment), kind)


def _clear_stale_build_locks(cache: Path, owned: bool, log) -> None:
    """Remove torch extension build locks that nobody will ever release.

    torch compiles an extension (nvdiffrast, for VoMP's renderer) under a lock
    file that the building process deletes when it is done.  A worker killed
    mid-build leaves the lock, and every later worker waits on it forever: the
    card sits at "starting" with the log ending in "Lock file exists in build
    directory".  A cache ``owned`` by this worker kind is used by nobody else,
    and a launch happens only when no worker of the kind is alive, so any lock
    in it is left over.  A shared cache may hold a one-shot tool's build in
    progress, so there only a lock older than any build is.
    """
    for lock in cache.glob("torch_extensions/*/*/lock"):
        with contextlib.suppress(OSError):
            if owned or time.time() - lock.stat().st_mtime > STALE_BUILD_LOCK_AGE:
                lock.unlink()
                log.write(f"Removed a stale build lock left by a killed build: {lock}\n")


def kernel_cache_env() -> dict[str, str]:
    """Warp's kernel cache for a process whose XDG cache is redirected to an output root.

    Warp keeps its compiled kernels under the XDG cache unless WARP_CACHE_PATH says otherwise,
    so each output root compiled every kernel again -- 53 modules, about 40 s of the first
    solve in each new root (an operator batch slot, a fresh UI root; 2026-10-09).  One user-level
    cache serves them all: a module is reused only for identical source (its name carries the
    hash), and Warp writes it to a temporary directory and renames it, so concurrent processes
    are safe.  The other caches (matplotlib, torch extensions, Hugging Face) stay per root.
    """
    if os.environ.get("WARP_CACHE_PATH"):
        return {"WARP_CACHE_PATH": os.environ["WARP_CACHE_PATH"]}
    home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return {"WARP_CACHE_PATH": os.path.join(home, "warp")}


def _launch(output_root: Path, kind: str, compute_env: str, command: list[str],
            cwd: Path, cache: Path, env: dict[str, str] | None = None,
            owns_cache: bool = False) -> bool:
    """Start a detached worker and record its pid, without blocking.

    Declines to start one that has just failed to start.  Getting here means no
    process is alive, so a launch a moment ago that has already left nothing
    behind did not work, and repeating it several times a second helps nobody.
    The reason is in the worker's own stderr log.
    """
    socket_path, pid_path = worker_paths(output_root, kind, compute_env)
    worker_dir = pid_path.parent
    worker_dir.mkdir(parents=True, exist_ok=True)
    attempted = worker_dir / f"{kind}.attempted"
    try:
        if time.time() - attempted.stat().st_mtime < WORKER_RELAUNCH_BACKOFF:
            return False
    except OSError:
        pass
    attempted.touch()
    cache.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **kernel_cache_env(), "XDG_CACHE_HOME": str(cache), **(env or {})}
    stdout = (worker_dir / f"{kind}.stdout.log").open("a", encoding="utf-8")
    stderr = (worker_dir / f"{kind}.stderr.log").open("a", encoding="utf-8")
    try:
        _clear_stale_build_locks(cache, owns_cache, stderr)
        stderr.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        pid_path.write_text(str(process.pid), encoding="utf-8")
    finally:
        stdout.close()
        stderr.close()
    return True


def ensure_geometry_worker(output_root: Path, session_dir: Path,
                           compute_env: str = DEFAULT_COMPUTE_ENV,
                           keep_hunyuan3d: bool = False) -> bool:
    """Start the shared persistent TripoSG worker without blocking the caller.

    ``keep_hunyuan3d`` leaves a Hunyuan3D worker warmed for a likely next route (the operator
    runner's drawing route, then Hunyuan3D) beside it, as ``ensure_hunyuan3d_worker``'s
    ``keep_triposg`` does the other way round; the caller checks the card has room.
    """
    output_root = Path(output_root)
    partfield_environment, _, _ = partfield_settings()
    stop_worker(output_root, "partfield", partfield_environment)
    stop_cad_workers(output_root)
    if not keep_hunyuan3d:
        stop_hunyuan3d_worker(output_root)
    stop_vomp_worker(output_root)
    socket_path, pid_path = worker_paths(output_root, "triposg", compute_env)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        return True
    return _launch(
        output_root, "triposg", compute_env,
        ["conda", "run", "--no-capture-output", "-n", current_environment(compute_env), "python", "-u",
         str(SCRIPT_DIR / "generate_surface.py"),
         "--serve", str(socket_path), "--preload", "scribble"],
        cwd=Path(session_dir),
        # Beside the pid, where it has always been -- the socket's own directory
        # is a short-lived one under the system temp root, and the model weights
        # cached here are worth keeping.
        cache=pid_path.parent / ".runtime_cache",
    )


def ensure_tool_worker(output_root: Path,
                       compute_env: str = DEFAULT_COMPUTE_ENV,
                       tier: str = "light") -> bool:
    """Keep a warm tool worker up, at least as warm as the next step will need.

    Cheap enough to call repeatedly, which is the point: a worker that idled out
    or was killed is simply back at the next opportunity, so the only cost of
    losing one is the cold start of whichever call finds it gone.  No session is
    needed -- the tools take absolute paths, and one worker serves every
    conversation in this environment.
    """
    output_root = Path(output_root)
    compute_env = compute_env or DEFAULT_COMPUTE_ENV
    socket_path, pid_path = worker_paths(output_root, "tools", compute_env)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        if tier != "light":
            request_warm(socket_path, tier, pid_path.read_text(encoding="utf-8"))
        return True
    interpreter = _direct_interpreter(compute_env)
    launcher = ([str(interpreter)] if interpreter is not None
                else ["conda", "run", "--no-capture-output", "-n", current_environment(compute_env),
                      "python"])
    return _launch(
        output_root, "tools", compute_env,
        [*launcher, "-u", str(SCRIPT_DIR / "tool_server.py"),
         "--serve", str(socket_path), "--warm", tier],
        cwd=output_root,
        # The kernel cache the one-shot tools and the forward solve use.  A
        # worker with its own would compile the warp modules a second time and
        # share the result with nobody.
        cache=output_root / ".runtime_cache",
    )


def ensure_partfield_worker(
        output_root: Path, compute_env: str = DEFAULT_COMPUTE_ENV,
        keep: tuple[str, ...] = ()) -> bool:
    """Start the optional PartField model server through its interpreter directly.

    ``keep`` names resident workers ("hunyuan3d", "triposg") left loaded beside it: PartField's
    worker is under 1 GB and a segmentation adds about 5 GB (measured), so the operator runner
    keeps the paint worker it warmed for the material card, and at a geometry stop the TripoSG
    worker a regeneration would use.  The caller checks the card has room.
    """
    status = partfield_status()
    if not status.ready or not status.interpreter:
        return False
    output_root = Path(output_root)
    stop_vomp_worker(output_root)
    if "triposg" not in keep:
        stop_worker(output_root, "triposg", compute_env)
    stop_cad_workers(output_root)
    if "hunyuan3d" not in keep:
        stop_hunyuan3d_worker(output_root)
    socket_path, pid_path = worker_paths(
        output_root, "partfield", status.environment
    )
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        return True
    server = Path(__file__).resolve().parents[2] / "partfield_wrapper" / "server.py"
    return _launch(
        output_root, "partfield", status.environment,
        [status.interpreter, "-u", str(server), "--serve", str(socket_path)],
        cwd=output_root, cache=output_root / ".runtime_cache" / "partfield", owns_cache=True,
    )


def stop_cad_workers(output_root: Path, keep: str | None = None) -> None:
    """Stop every text-to-CAD model server but ``keep``."""
    for kind, (_, settings_of, _) in CAD_GENERATORS.items():
        if kind != keep:
            stop_worker(output_root, kind, settings_of()[0])


def ensure_cad_worker(output_root: Path, kind: str,
                      compute_env: str = DEFAULT_COMPUTE_ENV) -> bool:
    """Start a text-to-CAD model server, unloading every other model."""
    status_of, _, _ = CAD_GENERATORS[kind]
    status = status_of()
    if not status.ready or not status.interpreter:
        return False
    output_root = Path(output_root)
    stop_worker(output_root, "triposg", compute_env)
    stop_worker(output_root, "partfield", partfield_settings()[0])
    stop_cad_workers(output_root, keep=kind)
    stop_hunyuan3d_worker(output_root)
    stop_vomp_worker(output_root)
    socket_path, pid_path = worker_paths(output_root, kind, status.environment)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        return True
    server = Path(__file__).resolve().parents[2] / "cad_wrapper" / f"{kind}_wrapper" / "server.py"
    return _launch(
        output_root, kind, status.environment,
        [status.interpreter, "-u", str(server), "--serve", str(socket_path)],
        cwd=output_root, cache=output_root / ".runtime_cache" / kind, owns_cache=True,
    )


def stop_hunyuan3d_worker(output_root: Path) -> None:
    stop_worker(output_root, "hunyuan3d", hunyuan3d_settings()[0])


def ensure_hunyuan3d_worker(output_root: Path,
                            compute_env: str = DEFAULT_COMPUTE_ENV,
                            paint: bool = False, shape_only: bool = False,
                            keep_triposg: bool = False, keep_partfield: bool = False) -> bool:
    """Start the Hunyuan3D-2.1 model server, unloading every other model.

    ``paint`` starts it for a card that paints an existing surface: it loads
    the paint stage and SDXL's reference models and not the shape stage, which
    cost 31 of a cold paint's 80 s.  ``shape_only`` starts it for a generation
    without colour (``texture`` off): the shape stage alone, about a third of the
    memory, the paint stage loading on a later paint's first use.  A worker
    already serving is kept whatever it loaded, since either kind can paint and
    generate (loading what it lacks on first use).

    ``keep_triposg`` leaves a resident TripoSG worker loaded beside it: warming
    Hunyuan3D ahead of a likely next generation (the operator runner, at a
    TripoSG body's geometry stop) must not cost the TripoSG worker that a retry
    of its own would use.  The two fit on one card (about 7.5 GB beside the
    shape stage's).  ``keep_partfield`` leaves PartField's worker beside a paint worker warmed
    ahead of the material card, so the segmentation between them stays warm (see
    ``ensure_partfield_worker``).
    """
    status = hunyuan3d_status()
    if not status.ready or not status.interpreter:
        return False
    output_root = Path(output_root)
    if not keep_triposg:
        stop_worker(output_root, "triposg", compute_env)
    if not keep_partfield:
        stop_worker(output_root, "partfield", partfield_settings()[0])
    stop_cad_workers(output_root)
    if not paint:
        # VoMP shares the material card with painting and fits beside the
        # paint stage (42.5 GB at VoMP's peak, measured), so only a full
        # generator makes it go.
        stop_vomp_worker(output_root)
    socket_path, pid_path = worker_paths(output_root, "hunyuan3d", status.environment)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        return True
    server = Path(__file__).resolve().parents[2] / "hunyuan3d_wrapper" / "server.py"
    return _launch(
        output_root, "hunyuan3d", status.environment,
        [status.interpreter, "-u", str(server), "--serve", str(socket_path),
         *(["--paint"] if paint else ["--no-texture"] if shape_only else [])],
        cwd=output_root, cache=output_root / ".runtime_cache" / "hunyuan3d", owns_cache=True,
    )


# The cards that offer to paint the surface with Hunyuan3D-2.1's paint stage.
PAINT_CARDS = ("geometry_texture", "material_input")


def hunyuan3d_for_card(output_root: Path, interrupt: dict,
                       compute_env: str = DEFAULT_COMPUTE_ENV) -> None:
    """Keep the paint worker exactly while a card that paints is open.

    Entering one starts it, so its load (about 20 s) is paid while the card is
    read rather than after the button; re-entering it after a paint finds it
    serving, so repaints are warm.  Any other card stops it and frees its
    17 GB -- except the geometry card set to Hunyuan3D, whose preload wants it.
    """
    kind = interrupt.get("kind")
    if kind in PAINT_CARDS:
        ensure_hunyuan3d_worker(output_root, compute_env, paint=True)
    elif not (kind == "geometry_input" and interrupt.get("generator") == "hunyuan3d"):
        stop_hunyuan3d_worker(output_root)


def stop_vomp_worker(output_root: Path) -> None:
    try:
        from src.vomp_wrapper.availability import settings as vomp_settings
    except ImportError:
        return
    stop_worker(output_root, "vomp", vomp_settings()[0])


def ensure_vomp_worker(output_root: Path, compute_env: str = DEFAULT_COMPUTE_ENV,
                       *, environment: str | None = None,
                       python: str | None = None) -> bool:
    """Start VoMP only after adaptive material is explicitly selected."""
    from src.vomp_wrapper.availability import vomp_status

    status = vomp_status(environment, python)
    if not status.ready or not status.interpreter:
        return False
    output_root = Path(output_root)
    stop_worker(output_root, "triposg", compute_env)
    stop_worker(output_root, "partfield", partfield_settings()[0])
    stop_cad_workers(output_root)
    # The paint worker stays: VoMP is on the card that paints, and fits beside it.
    socket_path, pid_path = worker_paths(output_root, "vomp", status.environment)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if worker_is_serving(socket_path, pid_path):
        return True
    server = Path(__file__).resolve().parents[2] / "vomp_wrapper" / "server.py"
    return _launch(
        output_root, "vomp", status.environment,
        [status.interpreter, "-u", str(server), "--serve", str(socket_path)],
        cwd=output_root, cache=output_root / ".runtime_cache" / "vomp", owns_cache=True,
    )
