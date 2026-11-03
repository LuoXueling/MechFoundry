"""Allowlisted subprocess monitor for predefined FEM workflow tools."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import ToolReceipt
from src.cad_wrapper.cadcoder_wrapper.availability import cadcoder_status, settings as cadcoder_settings
from src.cad_wrapper.text2cad_wrapper.availability import text2cad_status, settings as text2cad_settings
from src.hunyuan3d_wrapper.availability import (
    hunyuan3d_status, sdxl_status, settings as hunyuan3d_settings, texture_status,
)
from src.partfield_wrapper.availability import partfield_status, settings as partfield_settings


# The text-to-CAD generators, each in its own environment with its own worker:
# kind -> (readiness check, settings, name shown to people).
CAD_GENERATORS = {
    "text2cad": (text2cad_status, text2cad_settings, "Text2CAD"),
    "cadcoder": (cadcoder_status, cadcoder_settings, "CAD-Coder"),
}


# The image-to-3D generator with colour, in its own environment with its own worker.
HUNYUAN3D_TOOL = "generate_hunyuan3d_image_surface"
# Its paint stage alone, colouring a surface that already exists (for VoMP).
HUNYUAN3D_PAINT_TOOL = "paint_hunyuan3d_surface"
# SDXL drawing an object from text, as an image input for the geometry card.  It
# runs in the Hunyuan3D environment, whose cache holds SDXL for the painter, as a
# one-shot process: it needs none of Hunyuan3D's models, so the resident worker
# (which loads them all) is neither needed nor started.
SDXL_DRAW_TOOL = "draw_sdxl_image"


def cad_tool_kind(name: str) -> str | None:
    """The CAD generator a tool name runs, or None for any other tool."""
    kind = name.removeprefix("generate_").removesuffix("_text_surface")
    return kind if kind in CAD_GENERATORS and name == f"generate_{kind}_text_surface" else None


class ToolRejected(ValueError):
    pass


class ToolFailed(RuntimeError):
    """A tool that ran and exited non-zero; ``receipt`` is its record (return code, logs)."""

    def __init__(self, message: str, receipt: ToolReceipt):
        super().__init__(message)
        self.receipt = receipt


# The signals of a process that crashed on its own: a fault in native code.  Its inputs name
# nothing to change, and the same inputs crash again (a held-out run, 2026-10-10: one paint
# segfaulted four times, seeds 0 and 1).  SIGKILL (the out-of-memory killer, or a user), SIGTERM
# and SIGINT (a stop, Ctrl-C) are not: they say nothing about the inputs.
CRASH_SIGNALS = frozenset({signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGILL, signal.SIGFPE})
# faulthandler's first line for each of them, as a worker's fault report has it (fault_report),
# "Fatal Python error: <name>": CPython's own names, checked by raising each signal under
# faulthandler (SIGFPE is "Floating point exception", without a hyphen).
FAULT_NAMES = {"Segmentation fault": signal.SIGSEGV, "Bus error": signal.SIGBUS,
               "Aborted": signal.SIGABRT, "Illegal instruction": signal.SIGILL,
               "Floating point exception": signal.SIGFPE}
# The signals a shell or launcher reports as 128 + n for a child they took down.  Any other code
# over 128 (255 from sys.exit(-1) or a launcher's own failure) is an exit status, not a signal.
EXIT_SIGNALS = CRASH_SIGNALS | frozenset({signal.SIGKILL, signal.SIGTERM, signal.SIGINT})


def signal_of(returncode: int) -> int | None:
    """The signal a return code says the process died of: negative (subprocess's own report of a
    death by signal), or a shell's 128 + n for one of EXIT_SIGNALS."""
    if returncode < 0:
        return -returncode
    if returncode > 128 and returncode - 128 in EXIT_SIGNALS:
        return returncode - 128
    return None


def crashed(receipt: ToolReceipt | None) -> bool:
    """Whether a tool crashed on its own (CRASH_SIGNALS), so that the same inputs crash again."""
    return receipt is not None and signal_of(receipt.returncode) in CRASH_SIGNALS


def stopped(receipt: ToolReceipt | None) -> bool:
    """Whether a tool was stopped from outside (SIGKILL, SIGTERM, SIGINT): its inputs are not
    at fault, and the same call may run once there is room."""
    return receipt is not None and signal_of(receipt.returncode) in EXIT_SIGNALS - CRASH_SIGNALS


def fault_report(socket_path: Path) -> Path:
    """Where a resident worker writes faulthandler's report of a fatal signal: beside its socket.
    The Hunyuan3D worker (hunyuan3d_wrapper/server.py) names it the same way."""
    return Path(socket_path).with_name(Path(socket_path).name + ".fault")


# Workers that write a fault report: one of them gone mid-request without one was killed
# (SIGKILL), not crashed.  For the others a death mid-request is run again here, as before.
FAULT_REPORTING_WORKERS = frozenset({"hunyuan3d"})
# How long a worker that closed the connection mid-request is given to be gone.
WORKER_DEATH_WAIT = 5.0


@dataclass(frozen=True)
class ToolSpec:
    name: str
    expensive: bool
    timeout: int
    command: Callable[[dict[str, Any]], list[str]]
    expected: Callable[[dict[str, Any]], list[Path]]


# `conda run` re-bootstraps conda on every invocation, which measured ~1.8s here
# -- far more than a lightweight tool's own work (scale_surface itself runs in
# ~0.4s), and pure latency before a GPU tool reaches its first kernel.  These
# tools are launched through the compute environment's interpreter directly.
# Torch and Warp both work that way: Warp JIT-compiles through NVRTC rather than
# the conda toolchain, verified against a cold kernel cache with no activation.
# The forward solve now does too.  The doubt was the assembled backend, which is
# linked against JAX-AMG and AMGX and was the one thing not checked without the
# activation hooks; both import and initialise with LD_LIBRARY_PATH and
# CONDA_PREFIX unset entirely.  The same case was solved twice each way: the
# final displacement fields differ by 2.0e-09 across launchers against 1.7e-09
# between two runs of the same launcher, so the difference is the solver's own
# GPU non-determinism and not the launcher.  19.1s became 17.3s.
# The generators keep `conda run`: TripoSG has not been checked this way.
# Tools that must run on every call.  The segmentation and BC resolvers keep
# state the caller needs rebuilt; the forward solve is the user pressing "Run
# FEA", which asks for a fresh solve whether or not anything changed.
NEVER_CACHED = frozenset({"segment_surface", "resolve_bcs", "run_forward"})

DIRECT_LAUNCH = frozenset({
    "stage_asset", "inspect_surface", "scale_surface",
    "render_surface", "render_segmentation", "build_case", "export_surface_field",
    "render_part_montage", "build_part_field", "plain_colour_surface",
    "segment_surface", "resolve_bcs", "run_forward",
})

# Tools a resident worker may serve, skipping the imports each would otherwise
# pay -- 0.45s for the mesh and picture tools, 3.4s for the two that add torch
# and the warp kernel modules.  `run_forward` is deliberately absent: it runs
# for minutes, streams a log the UI follows, and should have the GPU to itself.
# It gets a spare of its own instead, which serves one solve and exits.
WORKER_TOOLS = DIRECT_LAUNCH - frozenset({"run_forward"})

# How long a solve waits for its spare process to finish warming (~5s) before
# launching a process of its own instead.
SOLVER_WARM_WAIT = 60.0

# How long to wait for a free worker to say hello.  It greets the instant it
# accepts, which measures under a millisecond, so this only has to survive a
# loaded machine -- and every bit of it is added to the cold start of a call
# that finds the worker busy and runs the tool itself.
WORKER_ACK_TIMEOUT = 0.25

# What a VoMP inference adds on the GPU above its idle worker: 24.6 GB measured
# on a 1.1 GB-occupied card beside the 17 GB paint worker (peak 42.5 GB), plus
# a margin.  The paint worker stays for it when this much is free.
VOMP_RUN_MIB = 26_000

# Room the GPU needs for two models to stay side by side instead of one at a time (measured on a
# 48 GB card, 2026-10-09).  A PartField segmentation adds 5.2 GB above its worker; the paint
# stage's run 13.1 GB above its 17.8 GB resident worker (30.9 GB peak), and 31 GB with the load
# when no paint-stage worker is up; an SDXL drawing 8-13 GB.  With the room, the warm worker that
# is not running stays: the paint worker warmed for the material card through a segmentation, the
# PartField worker through a paint, a Hunyuan3D worker warmed for the next route through a drawing.
PARTFIELD_RUN_MIB = 6_000
PAINT_RUN_MIB = 14_000
PAINT_COLD_MIB = 31_000
SDXL_RUN_MIB = 14_000


# How long a solve waits for the model workers it stopped to end (and so free their memory)
# before killing them.  A graceful exit with a 17 GB model takes a few seconds.
MODEL_EXIT_WAIT = 20.0


def _cmdline(pid: int) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None


def _wait_for_exit(workers: list[tuple[int, bytes | None]], timeout: float) -> None:
    """Wait until every (pid, command line) has ended (a zombie counts as ended), killing any still
    alive after ``timeout`` -- only while the pid still runs the command it was recorded with (a
    reused pid is left alone), by its process group when it leads one, else by itself."""
    from .runtime_files import exited

    pids = [pid for pid, _ in workers]
    recorded = dict(workers)

    def alive(pid: int) -> bool:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return False
        except ChildProcessError:
            pass
        except OSError:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return not exited(pid)

    deadline = time.monotonic() + timeout
    waiting = [pid for pid in pids if alive(pid)]
    while waiting and time.monotonic() < deadline:
        time.sleep(0.1)
        waiting = [pid for pid in waiting if alive(pid)]
    for pid in list(waiting):
        if recorded.get(pid) is not None and _cmdline(pid) != recorded[pid]:
            waiting.remove(pid)                  # the pid is another process's now
            continue
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.monotonic() + 5.0
    while waiting and time.monotonic() < deadline:
        time.sleep(0.1)
        waiting = [pid for pid in waiting if alive(pid)]


def gpu_free_mib() -> int | None:
    """Free memory on the GPU this process's children will use, or None if unknown.

    The first device of ``CUDA_VISIBLE_DEVICES``, else device 0.  nvidia-smi
    numbers devices in PCI bus order, which is CUDA's only under
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; under any other order the least free
    memory of all the cards is reported, which can only err towards caution.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"
    pci_order = os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    if not visible.isdigit():  # a UUID or MIG name
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        free = {int(index): int(value) for index, value in
                (line.split(",") for line in result.stdout.strip().splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if pci_order or len(free) == 1:
        return free.get(int(visible))
    return min(free.values()) if free else None


# Where a resident worker announces itself (socket, pid file): one definition for the workflow,
# the app and the operator runner, which reads it without importing this package.
from .runtime_files import worker_paths  # noqa: E402,F401


# Locate the compute environment's own interpreter, or None to use conda run: one lookup for the
# workflow and the operator runner (runtime_files.direct_interpreter).
from .runtime_files import direct_interpreter as _direct_interpreter  # noqa: E402
# A session made before the MechFoundry rename names an ``implicit-*`` environment; it runs under
# the renamed one once the old is gone (runtime_files.current_environment).
from .runtime_files import current_environment  # noqa: E402

DEFAULT_COMPUTE_ENV = "mechfoundry-warp117"


_DIGEST_CACHE_LIMIT = 256
_digest_cache: OrderedDict[
    tuple[str, int, int, int, int, int], str
] = OrderedDict()
_digest_cache_lock = threading.Lock()


def _digest_key(path: Path, stat: os.stat_result) -> tuple[str, int, int, int, int, int]:
    """Identity of file contents without opening the file itself."""
    return (
        os.fspath(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


def _digest(path: Path) -> str:
    """sha256 of a file, streamed so a large mesh or npz costs no memory.

    A directory artifact -- the solve output root -- has no single digest and
    records as an empty string.
    """
    try:
        before = path.stat()
    except OSError:
        return ""
    if not path.is_file():
        return ""
    key = _digest_key(path, before)
    with _digest_cache_lock:
        cached = _digest_cache.get(key)
        if cached is not None:
            _digest_cache.move_to_end(key)
            return cached
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    try:
        after = path.stat()
    except OSError:
        return digest
    # Do not cache a digest if the file changed while it was being read.
    if _digest_key(path, after) != key:
        return _digest(path)
    with _digest_cache_lock:
        _digest_cache[key] = digest
        _digest_cache.move_to_end(key)
        while len(_digest_cache) > _DIGEST_CACHE_LIMIT:
            _digest_cache.popitem(last=False)
    return digest


def _field_files(directory: Path) -> list[Path]:
    """Every file a material-field tool leaves in its artifact directory.

    The field is a directory -- manifest.json, material_grid.npy, and one GLB per
    channel with its sidecars -- written at the same paths by every run.  Its
    receipt used to name the manifest alone, so the cache's "are these still my
    files" test (_artifacts_intact, _restore_artifacts) looked at the manifest
    and nothing else: after a rewind put an earlier field back, re-assigning the
    values of a later one hit that later receipt, had its manifest copied back,
    and left the earlier field's GLBs and grid in place -- the viewer, and the
    solve, kept the old values (2026-10-20).  Every file is named instead, so a
    cache hit is only taken when all of them are this receipt's own, or can all
    be put back from the session's versions.
    """
    directory = Path(directory)
    manifest = directory / "manifest.json"
    files = sorted(path for path in directory.glob("*") if path.is_file()) if directory.is_dir() else []
    return [manifest, *(path for path in files if path != manifest)]


def _artifacts_intact(receipt: "ToolReceipt") -> bool:
    """Whether a cached receipt's artifacts are still the files it produced.

    Every run of a tool writes to the same path, so a later run with different
    settings replaces them.  A receipt whose fingerprint we match again is only
    reusable when what sits at those paths is still its own output; otherwise
    the tool has to run again and rewrite them.  Receipts written before these
    digests existed cannot be checked, so they too earn a re-run.
    """
    for path in receipt.artifacts:
        target = Path(path)
        if not target.exists():
            return False
        if target.is_dir():
            continue
        if _digest(target) != receipt.artifact_digests.get(path):
            return False
    return True


# The session's store of every version of its branch files, by content (graph.FILE_VERSIONS):
# ``versions/<sha256><suffix>``, the digest being _digest's.
FILE_VERSIONS = "versions"


def _restore_artifacts(receipt: "ToolReceipt", session_dir: Path) -> bool:
    """Put a cached receipt's own artifacts back from the session's versions, when all are kept.

    A rewind puts back the files its checkpoint recorded, so a painting made after that
    checkpoint is replaced at its path by the earlier one, and the same call (same surface,
    words and seed) found its receipt "not intact" and ran again: three repaints of unchanged
    words, about 30 s each, in one acceptance-batch session (2026-10-10).  Every artifact the
    graph saw at a checkpoint is in ``versions/`` by content, so the receipt's are copied back
    from there instead -- only inside this session, never through a symbolic link, and only
    when every one is found (a directory artifact, or a version never kept, means the tool runs
    again).  The file each one displaces is kept among the versions first, as the graph's own
    rewind keeps what it replaces (graph._restore_branch_files).
    """
    import shutil
    import tempfile

    root = session_dir.resolve()
    wanted = []
    for path in receipt.artifacts:
        target = Path(path)
        digest = receipt.artifact_digests.get(path)
        if target.is_dir():
            continue
        if not digest or target.is_symlink() or root not in target.resolve().parents:
            return False
        if _digest(target) == digest:
            continue
        kept = root / FILE_VERSIONS / f"{digest}{target.suffix}"
        if not kept.is_file():
            return False
        wanted.append((kept, target))

    def copy(source: Path, target: Path) -> None:
        """``source`` over ``target`` in one step, keeping its modification time."""
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        os.close(handle)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    for kept, target in wanted:
        found = _digest(target)
        displaced = root / FILE_VERSIONS / f"{found}{target.suffix}"
        if found and not displaced.is_file():
            copy(target, displaced)
        copy(kept, target)
    return _artifacts_intact(receipt)


def _required(args, *names):
    missing = [name for name in names if name not in args]
    if missing:
        raise ToolRejected(f"missing arguments: {', '.join(missing)}")


class ToolMonitor:
    """Execute only registered commands with validated, non-shell arguments."""

    def __init__(self, session_dir: Path, compute_env: str = DEFAULT_COMPUTE_ENV):
        self.session_dir = session_dir.resolve()
        self.compute_env = current_environment(compute_env)
        self.interpreter = _direct_interpreter(compute_env)
        self.receipt_dir = self.session_dir / "tool_receipts"
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        scripts = Path(__file__).resolve().parent / "scripts"
        workflow = scripts / "workflow_tool.py"
        generate = scripts / "generate_surface.py"
        partfield_segment = Path(__file__).resolve().parents[2] / "partfield_wrapper" / "segment.py"
        vomp_infer = Path(__file__).resolve().parents[2] / "vomp_wrapper" / "inference.py"

        def wf(subcommand, fields, optional=()):
            """Build a tool's argv.  `optional` fields travel only when given.

            An argument the tool defaults sensibly should not become a thing
            every caller must remember -- and because the cache fingerprint is
            taken over the arguments, one that is absent stays absent rather
            than being spelled out as a default nobody asked for.
            """
            def build(a):
                _required(a, *fields)
                result = [str(workflow), subcommand]
                for field in (*fields, *(f for f in optional if f in a)):
                    value = a[field]
                    flag = "--" + field.replace("_", "-")
                    result.extend([flag, str(value)])
                return result
            return build

        def outputs(*fields):
            return lambda a: [Path(a[field]) for field in fields]

        def segment_command(a):
            _required(a, "surface", "output_npz", "output_json", "output_glb", "clusters")
            if a.get("method", "shape_diameter") == "partfield":
                _required(a, "feature_cache")
                result = [
                    str(partfield_segment),
                    "--surface", str(a["surface"]),
                    "--output-npz", str(a["output_npz"]),
                    "--output-json", str(a["output_json"]),
                    "--output-glb", str(a["output_glb"]),
                    "--feature-cache", str(a["feature_cache"]),
                    "--clusters", str(a["clusters"]),
                ]
                if self.interpreter is not None:
                    result.extend(["--cluster-interpreter", str(self.interpreter)])
                else:
                    result.extend(["--cluster-environment", self.compute_env])
                return result
            return wf(
                "segment-surface",
                ("surface", "output_npz", "output_json", "output_glb", "clusters",
                 "rays", "smoothing", "cone_angle_deg"),
                optional=("method", "cad_faces"),
            )(a)

        def generator(mode):
            def build(a):
                _required(a, "output")
                result = [str(generate), "--mode", mode, "--output", str(a["output"])]
                for key in (
                    "prompt", "input", "seed", "steps", "octree_depth",
                    "init_octree_depth", "faces", "guidance_scale",
                    "prompt_confidence", "scribble_confidence", "erosion_radius",
                ):
                    if a.get(key) is not None:
                        result.extend(["--" + key.replace("_", "-"), str(a[key])])
                return result
            return build

        self.registry = {
            f"generate_{mode}_surface": ToolSpec(f"generate_{mode}_surface", True, 3600, generator(mode), outputs("output"))
            for mode in ("text", "image", "scribble", "text_scribble")
        }
        def cad_command(kind):
            script = (Path(__file__).resolve().parents[2] / "cad_wrapper"
                      / f"{kind}_wrapper" / "generate.py")

            def build(a):
                _required(a, "output", "prompt")
                result = [str(script), "--output", str(a["output"]), "--prompt", str(a["prompt"])]
                for key in ("seed", "faces"):
                    if a.get(key) is not None:
                        result.extend(["--" + key, str(a[key])])
                return result
            return build

        for kind in CAD_GENERATORS:
            name = f"generate_{kind}_text_surface"
            self.registry[name] = ToolSpec(name, True, 1800, cad_command(kind), outputs("output"))

        def hunyuan3d_command(a):
            _required(a, "output", "input")
            script = Path(__file__).resolve().parents[2] / "hunyuan3d_wrapper" / "generate.py"
            result = [str(script), "--output", str(a["output"]), "--input", str(a["input"])]
            for key in ("seed", "faces", "steps", "octree_resolution", "texture_size"):
                if a.get(key) is not None:
                    result.extend(["--" + key.replace("_", "-"), str(a[key])])
            if a.get("texture") is False:
                result.append("--no-texture")
            if a.get("flashvdm") is False:
                result.append("--no-flashvdm")
            if a.get("upscale"):
                result.append("--upscale")
            return result

        self.registry[HUNYUAN3D_TOOL] = ToolSpec(
            HUNYUAN3D_TOOL, True, 3600, hunyuan3d_command, outputs("output"))

        def hunyuan3d_paint_command(a):
            _required(a, "surface", "output", "output_reference")
            script = Path(__file__).resolve().parents[2] / "hunyuan3d_wrapper" / "generate.py"
            result = [str(script), "--surface", str(a["surface"]), "--output", str(a["output"]),
                      "--reference", str(a["output_reference"]), "--prompt", str(a.get("prompt") or "")]
            if a.get("input"):
                result.extend(["--input", str(a["input"])])
            for key in ("seed", "texture_size"):
                if a.get(key) is not None:
                    result.extend(["--" + key.replace("_", "-"), str(a[key])])
            if a.get("upscale"):
                result.append("--upscale")
            return result

        self.registry[HUNYUAN3D_PAINT_TOOL] = ToolSpec(
            HUNYUAN3D_PAINT_TOOL, True, 900, hunyuan3d_paint_command, outputs("output", "output_reference"))

        def sdxl_draw_command(a):
            _required(a, "prompt", "output", "output_raw")
            script = Path(__file__).resolve().parents[2] / "hunyuan3d_wrapper" / "draw_image.py"
            result = [str(script), "--prompt", str(a["prompt"]), "--output", str(a["output"]),
                      "--output-raw", str(a["output_raw"])]
            for key in ("seed", "steps"):
                if a.get(key) is not None:
                    result.extend(["--" + key, str(a[key])])
            return result

        self.registry[SDXL_DRAW_TOOL] = ToolSpec(
            SDXL_DRAW_TOOL, True, 900, sdxl_draw_command, outputs("output", "output_raw"))
        self.registry.update({
            "stage_asset": ToolSpec("stage_asset", False, 120, wf("stage-asset", ("source", "output")), outputs("output")),
            "inspect_surface": ToolSpec("inspect_surface", False, 120, wf("inspect-surface", ("surface", "output")), outputs("output")),
            "scale_surface": ToolSpec("scale_surface", False, 120, wf("scale-surface", ("surface", "output", "axis", "extent", "units", "octree_depth")), outputs("output")),
            "render_surface": ToolSpec("render_surface", False, 180, wf("render-surface", ("surface", "output")), outputs("output")),
            "plain_colour_surface": ToolSpec("plain_colour_surface", False, 120, wf("plain-colour-surface", ("surface", "rgb", "output"), optional=("segments_npz", "segments_json", "part_rgb")), outputs("output")),
            "segment_surface": ToolSpec("segment_surface", True, 1800, segment_command, outputs("output_npz", "output_json", "output_glb")),
            "render_segmentation": ToolSpec("render_segmentation", False, 180, wf("render-segmentation", ("surface", "segments_npz", "segments_json", "output")), outputs("output")),
            "render_part_montage": ToolSpec("render_part_montage", False, 180, wf("render-part-montage", ("surface", "segments_npz", "segments_json", "output")), outputs("output")),
            "build_part_field": ToolSpec("build_part_field", False, 300, wf("build-part-field", ("surface", "segments_npz", "materials_json", "unit_system", "output")), lambda a: _field_files(Path(a["output"]))),
            "resolve_bcs": ToolSpec("resolve_bcs", False, 900, wf("resolve-bcs", ("surface", "bc_json", "segments_npz", "output_npz", "output_json", "output_glb")), outputs("output_npz", "output_json", "output_glb")),
            "export_surface_field": ToolSpec(
                "export_surface_field", False, 180,
                wf("export-surface-field",
                   ("vtu", "output", "field", "component", "deform_scale"),
                   optional=("value_range_from", "yield_strength", "hover_version")),
                outputs("output"),
            ),
            "build_case": ToolSpec("build_case", False, 120, wf("build-case", ("surface", "bc_json", "segments_npz", "material_json", "settings_json", "transform_json", "output", "solve_output", "octree_depth"), optional=("loads_json",)), outputs("output")),
            "infer_material_field": ToolSpec(
                "infer_material_field", True, 7200,
                lambda a: [str(vomp_infer), "--surface", str(a["surface"]),
                           "--artifact", str(a["artifact"]), "--cache", str(a["cache"]),
                           "--unit-system", str(a["unit_system"])],
                lambda a: _field_files(Path(a["artifact"])),
            ),
            "run_forward": ToolSpec(
                "run_forward", True, 7200, wf("run-forward", ("case",)),
                lambda a: [Path(json.loads(Path(a["case"]).read_text(encoding="utf-8"))["output"])],
            ),
        })

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolReceipt:
        if name not in self.registry:
            raise ToolRejected(f"tool is not registered: {name}")
        spec = self.registry[name]
        command_tail = spec.command(arguments)
        self._validate_arguments(arguments)
        is_partfield = name == "segment_surface" and arguments.get("method") == "partfield"
        cad_kind = cad_tool_kind(name)
        is_hunyuan3d = name in (HUNYUAN3D_TOOL, HUNYUAN3D_PAINT_TOOL)
        is_sdxl = name == SDXL_DRAW_TOOL
        is_vomp = name == "infer_material_field"
        if is_sdxl:
            status = sdxl_status()
            if not status.ready or not status.interpreter:
                raise ToolRejected(f"SDXL is not available: {status.summary}")
            launcher = [status.interpreter]
        elif is_vomp:
            from src.vomp_wrapper.availability import vomp_status
            status = vomp_status()
            if not status.ready or not status.interpreter:
                raise ToolRejected(f"VoMP is not ready: {status.summary}")
            launcher = [status.interpreter]
        elif is_partfield:
            status = partfield_status()
            if not status.ready or not status.interpreter:
                raise ToolRejected(f"PartField is not ready: {status.summary}")
            launcher = [status.interpreter]
        elif is_hunyuan3d:
            # Painting from words also needs the reference-image models.
            status = (texture_status() if name == HUNYUAN3D_PAINT_TOOL and not arguments.get("input")
                      else hunyuan3d_status())
            if not status.ready or not status.interpreter:
                raise ToolRejected(f"Hunyuan3D-2.1 is not ready: {status.summary}")
            launcher = [status.interpreter]
        elif cad_kind:
            status_of, _, label = CAD_GENERATORS[cad_kind]
            status = status_of()
            if not status.ready or not status.interpreter:
                raise ToolRejected(f"{label} is not ready: {status.summary}")
            launcher = [status.interpreter]
        else:
            launcher = (
                [str(self.interpreter)]
                if self.interpreter is not None and name in DIRECT_LAUNCH
                else ["conda", "run", "-n", self.compute_env, "python"]
            )
        command = [*launcher, *command_tail]
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps([name, arguments], sort_keys=True, default=str).encode())
        script = Path(command_tail[0])
        if script.is_file():
            fingerprint.update(bytes.fromhex(_digest(script)))
        # Paths remain stable throughout a session, but their contents (notably
        # shape_opt.json and case.json) change when the user edits and retries.
        # Include every existing input file so cached tools cannot reuse stale settings.
        for argument_name, value in sorted(arguments.items()):
            if argument_name.startswith("output") or argument_name == "solve_output":
                continue
            if isinstance(value, (str, Path)):
                input_path = Path(value)
                try:
                    is_file = input_path.is_file()
                except OSError:
                    # Text such as a long prompt, which no file system takes as
                    # a name; it is already in the fingerprint as an argument.
                    is_file = False
                if is_file:
                    fingerprint.update(argument_name.encode())
                    fingerprint.update(bytes.fromhex(_digest(input_path)))
        key = fingerprint.hexdigest()[:20]
        receipt_path = self.receipt_dir / f"{key}.json"
        if receipt_path.exists() and name not in NEVER_CACHED:
            previous = ToolReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
            if previous.returncode == 0 and (_artifacts_intact(previous)
                                             or _restore_artifacts(previous, self.session_dir)):
                return previous
        # CUDA/Warp compilation dominates the first solve. Keep the XDG/JIT
        # cache at the workflow-output level rather than inside one session so
        # later conversations can reuse compiled kernels just like retries do.
        runtime_cache = self.session_dir.parent / ".runtime_cache"
        runtime_cache.mkdir(exist_ok=True)
        matplotlib_cache = runtime_cache / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        from .workers import kernel_cache_env
        environment = {
            **os.environ,
            # Warp's kernels in one user-level cache, not this root's (workers.kernel_cache_env).
            **kernel_cache_env(),
            "MPLCONFIGDIR": str(matplotlib_cache),
            "XDG_CACHE_HOME": str(runtime_cache),
        }
        if name == "run_forward":
            # Prevent JAX/XLA from reserving most GPU memory before the FEM
            # solve has allocated its own Warp and linear-solver buffers.
            environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            # The model stays resident across regenerations -- reloading it for
            # every attempt is the slowest part of the card people iterate on
            # most -- but a solve is what it has to make room for.  Here rather
            # than behind the button, so every route into a solve unloads it:
            # a resumed graph, or a second solve on the same card.
            self.release_models_for_solve()
        elif is_vomp:
            self._shutdown_geometry_worker()
            self._shutdown_partfield_worker()
            self._shutdown_cad_workers()
            # The paint worker belongs to this card and is kept for a repaint,
            # unless the GPU has no room for the inference beside it.
            free = gpu_free_mib()
            if free is None or free < VOMP_RUN_MIB:
                self._shutdown_hunyuan3d_worker()
        elif is_partfield:
            self._shutdown_geometry_worker()
            self._shutdown_cad_workers()
            # A paint worker warmed for the material card stays when the segmentation fits beside it.
            if not self._worker_up("hunyuan3d", hunyuan3d_settings()[0]) \
                    or (gpu_free_mib() or 0) < PARTFIELD_RUN_MIB:
                self._shutdown_hunyuan3d_worker()
            self._shutdown_vomp_worker()
        elif cad_kind:
            # One generator resident at a time, as for TripoSG and PartField.
            self._shutdown_geometry_worker()
            self._shutdown_partfield_worker()
            self._shutdown_cad_workers(keep=cad_kind)
            self._shutdown_hunyuan3d_worker()
            self._shutdown_vomp_worker()
        elif is_hunyuan3d:
            self._shutdown_geometry_worker()
            if name != HUNYUAN3D_PAINT_TOOL:
                self._shutdown_partfield_worker()
            else:
                # A paint keeps PartField's small worker (a re-segmentation) when it fits beside.
                needed = PAINT_RUN_MIB if self._hunyuan3d_paint_loaded() else PAINT_COLD_MIB
                if not self._worker_up("partfield", partfield_settings()[0]) \
                        or (gpu_free_mib() or 0) < needed:
                    self._shutdown_partfield_worker()
            self._shutdown_cad_workers()
            if name != HUNYUAN3D_PAINT_TOOL:
                # Painting shares the material card with VoMP, which fits beside it.
                self._shutdown_vomp_worker()
        elif is_sdxl:
            # The drawing goes to TripoSG or Hunyuan3D next, so either may stay
            # resident beside SDXL's ~12 GB; nothing else is needed at this step.
            self._shutdown_partfield_worker()
            self._shutdown_cad_workers()
            self._shutdown_vomp_worker()
            if self._worker_up("hunyuan3d", hunyuan3d_settings()[0]) \
                    and (gpu_free_mib() or SDXL_RUN_MIB) < SDXL_RUN_MIB:
                # no room for the drawing beside a resident Hunyuan3D worker
                self._shutdown_hunyuan3d_worker()
        elif name.startswith("generate_"):
            self._shutdown_partfield_worker()
            self._shutdown_cad_workers()
            self._shutdown_hunyuan3d_worker()
            self._shutdown_vomp_worker()
        elif name == "segment_surface":
            self._shutdown_partfield_worker()
        result = None
        if cad_kind:
            # Started here rather than by a preload alone, so the model loads
            # once and stays for the regenerations that follow.
            from .workers import ensure_cad_worker
            ensure_cad_worker(self.session_dir.parent, cad_kind)
            result = self._run_worker(
                cad_kind, status.environment, command_tail, spec.timeout,
                wait_for_start=True,
            )
        elif is_hunyuan3d:
            from .workers import ensure_hunyuan3d_worker
            # A generation without colour needs only the shape stage loaded.
            shape_only = name == HUNYUAN3D_TOOL and arguments.get("texture") is False
            ensure_hunyuan3d_worker(self.session_dir.parent,
                                    paint=name == HUNYUAN3D_PAINT_TOOL,
                                    **({"shape_only": True} if shape_only else {}))
            result = self._run_worker(
                "hunyuan3d", status.environment, command_tail, spec.timeout,
                wait_for_start=True,
            )
        elif name.startswith("generate_"):
            result = self._run_geometry_worker(name, arguments, spec.timeout)
        elif is_partfield:
            result = self._run_worker(
                "partfield", partfield_status().environment, command_tail, spec.timeout,
                wait_for_start=True,
            )
        elif is_vomp:
            from src.vomp_wrapper.availability import vomp_status
            status = vomp_status()
            result = self._run_worker(
                "vomp", status.environment, command_tail, spec.timeout,
                wait_for_start=True,
            )
        elif name in WORKER_TOOLS:
            result = self._run_tool_worker(command_tail, spec.timeout)
        elif name == "run_forward":
            result = self._run_solver_worker(command_tail, spec.timeout)
        if result is None:
            result = subprocess.run(command, cwd=self.session_dir, capture_output=True, text=True,
                                    timeout=spec.timeout, check=False, env=environment)
        stdout_path = self.receipt_dir / f"{key}.stdout.log"
        stderr_path = self.receipt_dir / f"{key}.stderr.log"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        artifacts = [str(path.resolve()) for path in spec.expected(arguments) if path.exists()]
        receipt = ToolReceipt(tool=name, arguments=arguments, returncode=result.returncode,
                              stdout_path=str(stdout_path), stderr_path=str(stderr_path),
                              artifacts=artifacts,
                              artifact_digests={path: _digest(Path(path)) for path in artifacts})
        receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        if result.returncode != 0:
            lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
            diagnostic = next(
                (line for line in reversed(lines)
                 if line.startswith(("ValueError:", "RuntimeError:"))),
                lines[-1] if lines else "no diagnostic was emitted",
            )
            number = signal_of(result.returncode)
            if number is not None:
                # The last line of a process killed by a signal is whatever it printed before (a
                # progress bar, in the held-out run of 2026-10-10), which reads as the cause.
                try:
                    signame = signal.Signals(number).name
                except ValueError:
                    signame = f"signal {number}"
                if crashed(receipt):
                    diagnostic = (f"crashed ({signame}, return code {result.returncode}) without a Python "
                                  "error; the same inputs crash again")
                elif number == signal.SIGKILL:
                    diagnostic = (f"killed by {signame} (return code {result.returncode}): out of memory, or "
                                  "stopped by a user; nothing about its inputs (run it again on a card with room)")
                else:          # a stop (SIGTERM, SIGINT): what it printed is kept, never read as memory
                    diagnostic = (f"stopped by {signame} (return code {result.returncode}); nothing about its "
                                  "inputs" + (f"; its last output: {diagnostic}" if lines else ""))
            raise ToolFailed(f"{name} failed: {diagnostic}; see {stderr_path}", receipt)
        missing = [str(path) for path in spec.expected(arguments) if not path.exists()]
        if missing:
            raise RuntimeError(f"{name} did not create expected artifacts: {missing}")
        return receipt

    def _worker_paths(self, kind: str, environment: str | None = None) -> tuple[Path, Path]:
        return worker_paths(
            self.session_dir.parent, kind, environment or self.compute_env
        )

    def _geometry_worker_paths(self) -> tuple[Path, Path]:
        return self._worker_paths("triposg")

    def _run_tool_worker(self, command_tail: list[str], timeout: int):
        """Hand a tool to the resident worker, or return None to run it here.

        Never waits for a worker to warm up or to finish something else: warming
        costs what the cold start it saves costs, so an absent or busy worker is
        simply not used.  A socket left behind by a worker that died would send
        every later call down this path and back again, so a failed attempt
        clears it -- which is also what lets the next preload start a new one.
        """
        return self._run_worker("tools", self.compute_env, command_tail, timeout)

    def _run_solver_worker(self, command_tail: list[str], timeout: int):
        """Hand the solve to the spare started for it, or return None to launch one.

        The spare (``workers.ensure_solver_worker``) has already paid the
        imports, CUDA context and AMGX start-up a new process would, and it
        serves this one solve and exits.  One still warming is waited for: its
        start-up is under way and would only be paid again by launching another.
        """
        socket_path, pid_path = self._worker_paths("solver")
        deadline = time.monotonic() + SOLVER_WARM_WAIT
        while not socket_path.exists():
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8")), 0)
            except (OSError, ValueError):
                return None
            if time.monotonic() >= deadline:
                # Alive and still not ready: stuck, not slow.  Out of the way,
                # so the process launched instead has the GPU to itself.
                from .workers import stop_solver_worker
                stop_solver_worker(self.session_dir.parent, self.compute_env)
                return None
            time.sleep(0.1)
        result = self._run_worker("solver", self.compute_env, command_tail, timeout)
        if result is not None:
            # It is exiting now, on purpose.  Forget it at once, so the card
            # that follows the solve starts the next spare instead of taking
            # this one's last moments for a live worker, or waiting out the
            # backoff meant for spares that fail to start.
            pid_path.unlink(missing_ok=True)
            (pid_path.parent / "solver.attempted").unlink(missing_ok=True)
        return result

    def _run_worker(
        self, kind: str, environment: str, command_tail: list[str], timeout: int,
        *, wait_for_start: bool = False,
    ):
        """Hand one argv request to a compatible resident worker."""
        socket_path, pid_path = self._worker_paths(kind, environment)
        if not socket_path.exists():
            if not wait_for_start or not pid_path.exists():
                return None
            # PartField binds its socket only after the model is resident.  If
            # the UI has already started that preload, wait for it instead of
            # launching a second cold model load into the same GPU.
            deadline = time.monotonic() + timeout
            while not socket_path.exists():
                try:
                    pid = int(pid_path.read_text(encoding="utf-8"))
                    os.kill(pid, 0)
                except (OSError, ValueError):
                    pid_path.unlink(missing_ok=True)
                    return None
                if time.monotonic() >= deadline:
                    return subprocess.CompletedProcess(
                        [], 1, "", f"Timed out waiting for the {kind} worker to start."
                    )
                time.sleep(0.1)
        # The same working directory the subprocess would have been given, so a
        # relative path in the arguments resolves to the same file either way.
        request = json.dumps({"argv": [str(item) for item in command_tail[1:]],
                              "cwd": str(self.session_dir)})
        sent = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(WORKER_ACK_TIMEOUT)
                client.connect(os.fspath(socket_path))
                stream = client.makefile("r", encoding="utf-8")
                # The worker greets before it will read anything.  Waiting for
                # that first, and only then sending, is what makes giving up
                # safe: a worker too busy to greet has not been told to do
                # anything, so nothing it might do can collide with doing it
                # here instead.
                if not json.loads(stream.readline()).get("ready"):
                    return None
                client.sendall((request + "\n").encode())
                sent = True
                client.settimeout(timeout)
                response = json.loads(stream.readline())
            if response.get("stale"):
                # The worker is holding a tool module older than the one on
                # disk, and is shutting down.  Run this call ourselves; the next
                # preload starts a worker with the current code.
                return None
            return subprocess.CompletedProcess(
                [], 0 if response.get("ok") else 1,
                response.get("stdout", ""), response.get("stderr", ""),
            )
        except (socket.timeout, TimeoutError):
            # Busy, not gone.  Run the tool here and leave the worker alone --
            # a timeout is the one failure that must not clear its socket.
            return None
        except (OSError, ValueError, json.JSONDecodeError):
            died = self._worker_death(kind, socket_path, pid_path) if sent else None
            socket_path.unlink(missing_ok=True)
            pid_path.unlink(missing_ok=True)
            return died

    def _worker_death(self, kind: str, socket_path: Path, pid_path: Path):
        """The result of a request whose worker went away while running it, or None to run it here.

        A worker that crashed on a request crashes the same way on it run here, from a cold
        load (a held-out run, 2026-10-10: each seed of a paint crashed twice, the worker and then
        the one-shot run): its fault report becomes the result, with the signal's return code.
        A fault-reporting worker gone without a report was killed (SIGKILL: out of memory, or a
        user), which running it again here at once would only repeat.  Any other worker's request
        is run here, as before.
        """
        if kind not in FAULT_REPORTING_WORKERS:
            return None
        try:
            pid = int(pid_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pid = None
        deadline = time.monotonic() + WORKER_DEATH_WAIT
        from .runtime_files import process_state
        while pid is not None and process_state(pid) not in (None, "Z", "X") \
                and time.monotonic() < deadline:
            time.sleep(0.1)
        if pid is not None and process_state(pid) not in (None, "Z", "X"):
            return None             # alive: it closed the connection, it did not die
        try:
            report = fault_report(socket_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            report = ""
        number = next((int(number) for name, number in FAULT_NAMES.items()
                       if f"Fatal Python error: {name}" in report), None)
        if number is not None:
            return subprocess.CompletedProcess(
                [], -number, "", f"the {kind} worker crashed during this request:\n{report}")
        return subprocess.CompletedProcess(
            [], -int(signal.SIGKILL), "",
            f"the {kind} worker died during this request without a fault report: killed "
            "(out of memory, or stopped by a user)")

    def _shutdown_geometry_worker(self) -> None:
        """Release the resident TripoSG weights, so a solve has the GPU.

        Not after each generation: the model stays loaded across the attempts
        people make on the geometry card, which is the card most iterated on.
        """
        socket_path, pid_path = self._geometry_worker_paths()
        pid = None
        try:
            pid = int(pid_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        was_serving = socket_path.exists()
        if was_serving:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(5.0)
                    client.connect(os.fspath(socket_path))
                    client.sendall(b'{"action":"shutdown"}\n')
                    client.makefile("r", encoding="utf-8").readline()
            except OSError:
                pass
            deadline = time.monotonic() + 5.0
            while socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        # A cached result may request shutdown before model loading has created
        # the socket.  The worker was launched in its own process group, so stop
        # that owned group if graceful shutdown did not complete.
        graceful_exit = was_serving and not socket_path.exists()
        if pid is not None and not graceful_exit:
            try:
                os.kill(pid, 0)
            except OSError:
                pid = None
        if pid is not None and not graceful_exit:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
        pid_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
        # Evicted, not failed: the next preload may start it again at once.
        (pid_path.parent / "triposg.attempted").unlink(missing_ok=True)

    def _model_worker_pids(self) -> list[tuple[int, bytes | None]]:
        """Live pids, with their command lines, of the model workers a solve does not use (every
        kind but the tools worker and the solver's spare)."""
        kinds = [("triposg", self.compute_env), ("partfield", partfield_settings()[0]),
                 ("hunyuan3d", hunyuan3d_settings()[0]),
                 *((kind, settings_of()[0]) for kind, (_, settings_of, _) in CAD_GENERATORS.items())]
        try:
            from src.vomp_wrapper.availability import settings as vomp_settings
            kinds.append(("vomp", vomp_settings()[0]))
        except ImportError:
            pass
        pids = []
        for kind, environment in kinds:
            _, pid_path = self._worker_paths(kind, environment)
            try:
                pid = int(pid_path.read_text(encoding="utf-8"))
                os.kill(pid, 0)
            except (OSError, ValueError):
                continue
            pids.append((pid, _cmdline(pid)))
        return pids

    def release_models_for_solve(self) -> None:
        """Stop every model worker a solve does not use and wait until their memory is free.

        Asking a worker to stop returns before its process has let go of the card: a 17 GB
        paint worker tearing down its CUDA context still holds it while the solve allocates.
        So the stop is followed by waiting for each process to end (``MODEL_EXIT_WAIT``), and a
        process still there after that is killed.  Every solve, forward or optimization, at any
        depth, and whatever route started it (the graph's card, a backend re-run), comes here.
        """
        pids = self._model_worker_pids()
        self._shutdown_geometry_worker()
        self._shutdown_partfield_worker()
        self._shutdown_cad_workers()
        self._shutdown_hunyuan3d_worker()
        self._shutdown_vomp_worker()
        _wait_for_exit(pids, MODEL_EXIT_WAIT)

    def _shutdown_partfield_worker(self) -> None:
        environment, _, _ = partfield_settings()
        self._shutdown_simple_worker("partfield", environment)

    def _shutdown_cad_workers(self, keep: str | None = None) -> None:
        """Release every text-to-CAD model but ``keep``."""
        for kind, (_, settings_of, _) in CAD_GENERATORS.items():
            if kind != keep:
                self._shutdown_simple_worker(kind, settings_of()[0])

    def _shutdown_hunyuan3d_worker(self) -> None:
        self._shutdown_simple_worker("hunyuan3d", hunyuan3d_settings()[0])

    def _worker_up(self, kind: str, environment: str) -> bool:
        """Whether a resident worker of this kind has a live process (serving or loading)."""
        _, pid_path = self._worker_paths(kind, environment)
        try:
            os.kill(int(pid_path.read_text(encoding="utf-8")), 0)
        except (OSError, ValueError):
            return False
        return True

    def _hunyuan3d_paint_loaded(self) -> bool:
        """Whether a live Hunyuan3D worker holds the paint stage (one started for a shape-only
        generation, --no-texture, loads it on its first paint)."""
        _, pid_path = self._worker_paths("hunyuan3d", hunyuan3d_settings()[0])
        try:
            pid = int(pid_path.read_text(encoding="utf-8"))
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            return False
        return b"--no-texture" not in command

    def _shutdown_vomp_worker(self) -> None:
        from src.vomp_wrapper.availability import settings as vomp_settings
        environment, _, _, _ = vomp_settings()
        self._shutdown_simple_worker("vomp", environment)

    def _shutdown_simple_worker(self, kind: str, environment: str) -> None:
        socket_path, pid_path = self._worker_paths(kind, environment)
        try:
            pid = int(pid_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pid = None
        graceful_exit = False
        if socket_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(3.0)
                    client.connect(os.fspath(socket_path))
                    client.makefile("r", encoding="utf-8").readline()
                    client.sendall(b'{"action":"shutdown"}\n')
            except OSError:
                pass
            deadline = time.monotonic() + 5.0
            while socket_path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            graceful_exit = not socket_path.exists()
        if pid is not None and not graceful_exit:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        pid_path.unlink(missing_ok=True)
        socket_path.unlink(missing_ok=True)
        (pid_path.parent / f"{kind}.attempted").unlink(missing_ok=True)

    def _run_geometry_worker(self, name: str, arguments: dict[str, Any], timeout: int):
        """Use the shared preloaded worker, or return None for one-shot fallback."""
        socket_path, pid_path = self._geometry_worker_paths()
        if not socket_path.exists() and not pid_path.exists():
            return None
        deadline = time.monotonic() + timeout
        while not socket_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8"))
                os.kill(pid, 0)
            except (OSError, ValueError):
                return None
            if time.monotonic() >= deadline:
                return subprocess.CompletedProcess([], 1, "", "TripoSG worker readiness timed out")
            time.sleep(0.25)
        mode = name.removeprefix("generate_").removesuffix("_surface")
        request = {
            "mode": mode,
            "prompt": arguments.get("prompt"),
            "input": arguments.get("input"),
            "output": arguments["output"],
            "device": "cuda",
            "seed": int(arguments.get("seed", 42)),
            "steps": int(arguments.get("steps", 16)),
            "init_octree_depth": int(arguments.get("init_octree_depth", 4)),
            "octree_depth": int(arguments.get("octree_depth", 7)),
            "faces": int(arguments.get("faces", -1)),
            # Passed through as given, absent included: the generator takes
            # a guidance scale on the image mode alone, and supplying one
            # here for the others would put back the argument the tool is
            # now explicit about refusing.
            "guidance_scale": (None if arguments.get("guidance_scale") is None
                               else float(arguments["guidance_scale"])),
            "prompt_confidence": float(arguments.get("prompt_confidence", 1.0)),
            "scribble_confidence": float(arguments.get("scribble_confidence", 0.4)),
            "erosion_radius": int(arguments.get("erosion_radius", 0)),
            "weights_dir": None,
        }
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(max(1.0, deadline - time.monotonic()))
                client.connect(os.fspath(socket_path))
                client.sendall((json.dumps(request) + "\n").encode())
                response = json.loads(client.makefile("r", encoding="utf-8").readline())
            if response.get("ok"):
                return subprocess.CompletedProcess(
                    [], 0, json.dumps(response["result"], separators=(",", ":")) + "\n", ""
                )
            if response.get("stale"):
                # Loaded before its code changed on disk; it has exited, so the
                # job runs once here and the next preload starts a current one.
                return None
            return subprocess.CompletedProcess([], 1, "", response.get("error", "worker failed"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _validate_arguments(self, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolRejected("tool arguments must be an object")
        for key, value in arguments.items():
            if key == "cache":
                path = Path(value).resolve()
                try:
                    path.relative_to(self.session_dir.parent)
                except ValueError as exc:
                    raise ToolRejected("cache must be inside the workflow output directory") from exc
            elif key.startswith("output") or key in {"bc_json", "segments_npz", "segments_json", "material_json", "materials_json", "loads_json", "settings_json", "transform_json", "case", "surface", "input", "solve_output", "feature_cache", "cad_faces", "artifact"}:
                path = Path(value).resolve()
                try:
                    path.relative_to(self.session_dir)
                except ValueError as exc:
                    raise ToolRejected(f"{key} must be inside the session directory") from exc
