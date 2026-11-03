"""Reading a workflow session from the outside: paths, logs, receipts, GPUs.

Everything here is plain Python plus the standard library, so the driver, the
inspector and the tests can all use it in any environment -- the agents
environment has no matplotlib, the compute environment no LangGraph, and a
status check should need neither.  Nothing here writes into a session except
under its ``operator/`` folder, which belongs to these scripts alone.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = SKILL_DIR.parent                     # src/agents_wrapper/fem_workflow
REPO_ROOT = PACKAGE_DIR.parents[2]
UI_OUTPUT_ROOT = REPO_ROOT / "output" / "fem_workflow"
# Resident workers are shared by every front end that uses the same output root
# (their sockets are keyed on it), and the Streamlit UI starts them without
# pinning a GPU.  A UI open on the shared root could therefore serve an agent's
# step on a busy card.  The driver's own root keeps its workers its own.
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "fem_workflow_operator"
DEFAULT_COMPUTE_ENV = "mechfoundry-warp117"
# A batch running several agents at once gives each its own output root (and so its own
# workers, which the workflow keys on the root) and its own GPUs.  Run together on one root,
# two runs shared workers and the GPU guard: one run's preload stopped the other's paint
# worker, a paint ran out of memory beside the sibling's, and the sibling's work caused 7
# waits for an idle GPU.  The agent still sees every session at
# DEFAULT_OUTPUT_ROOT/<id> (``start`` links it there), the path the skill names.
OUTPUT_ROOT_ENV, GPUS_ENV = "FEM_OPERATOR_OUTPUT_ROOT", "FEM_OPERATOR_GPUS"


def _runtime_files():
    """The workflow's ``runtime_files`` (worker sockets and pid files, the solve-failure record),
    loaded from its file: importing it through the package would run the package ``__init__``,
    which needs LangGraph, and this module must work in every environment."""
    import importlib.util
    import sys

    name = "src.agents_wrapper.fem_workflow.runtime_files"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location("fem_workflow_runtime_files", PACKAGE_DIR / "runtime_files.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_files = _runtime_files()


def output_root_default() -> Path:
    """The runner's output root: ``FEM_OPERATOR_OUTPUT_ROOT`` (relative to the repository) or the default."""
    value = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    return (REPO_ROOT / value).resolve() if value else DEFAULT_OUTPUT_ROOT


def allowed_gpus() -> set[int] | None:
    """The GPU indices ``FEM_OPERATOR_GPUS`` (comma-separated) confines the runner to, or None for all."""
    value = os.environ.get(GPUS_ENV, "").strip()
    return {int(v) for v in value.split(",") if v.strip()} if value else None


def relative_paths(value: Any) -> Any:
    """Every path under the repository in ``value`` written relative to its root.

    What the runner prints, saves for the agent to read (``last_run.json``) or writes into the
    report is part of the skill: the user's rule is no absolute paths, and an agent copies a
    printed path into its plan.  Commands run from the repository root, where these resolve.
    """
    prefix = str(REPO_ROOT) + os.sep
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        root = output_root_default()
        if root != DEFAULT_OUTPUT_ROOT:             # a batch slot's root: the linked path the skill names
            value = value.replace(str(root) + os.sep, str(DEFAULT_OUTPUT_ROOT) + os.sep)
        return value.replace(prefix, "")
    if isinstance(value, dict):
        return {k: relative_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [relative_paths(v) for v in value]
    return value


# ---------------------------------------------------------------- sessions --

def session_dir(session_id: str, output_root: Path | None = None) -> Path:
    if not session_id or Path(session_id).name != session_id:
        raise ValueError(f"invalid session id: {session_id!r}")
    return Path(output_root or DEFAULT_OUTPUT_ROOT).resolve() / session_id


def operator_dir(session: Path) -> Path:
    path = session / "operator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    """Atomic, so a status check never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=str) + "\n")


def tail(path: Path, lines: int = 20, max_bytes: int = 200_000) -> list[str]:
    try:
        with Path(path).open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(size - max_bytes, 0))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def solve_runs(session: Path) -> list[Path]:
    """Every forward/optimization run folder, oldest first."""
    root = session / "solve"
    runs = [p for p in root.glob("*") if p.is_dir() and (p / "log.txt").exists()]
    return sorted(runs, key=lambda p: (p / "log.txt").stat().st_mtime)


def newest_run(session: Path) -> Path | None:
    log = runtime_files.newest_solve_log(session)        # the app's telemetry reads the same one
    return log.parent if log is not None else None


def newest_receipt(session: Path) -> dict[str, Any] | None:
    receipts = sorted((session / "tool_receipts").glob("*.json"),
                      key=lambda p: p.stat().st_mtime)
    if not receipts:
        return None
    value = read_json(receipts[-1], {}) or {}
    value["_path"] = str(receipts[-1])
    value["_mtime"] = receipts[-1].stat().st_mtime
    return value


# ------------------------------------------------------------------- logs --

_STAMP = re.compile(r"^\[(\d\d)-(\d\d)-(\d{4}),(\d\d):(\d\d):(\d\d)\.(\d+)\]\s?")
_NEWTON = re.compile(r"scale=([\d.eE+-]+) Iter (\d+): \|\|R\|\| = ([\d.eE+-]+|nan|inf), rel = ([\d.eE+-]+|nan|inf)")
_LOAD_STEP = re.compile(r"Load step (\d+)/(\d+)")
_OPT_HEADER = re.compile(r"Optimization Iteration (\d+)")
_OPT_ITER = re.compile(r"Opt iter (\d+): obj=([\d.eE+-]+|nan|inf)")
_SE = re.compile(
    r"SE=([\d.eE+-]+) \(raw=([\d.eE+-]+)\), Vol=([\d.eE+-]+) \(raw=([\d.eE+-]+), "
    r"target=([\d.eE+-]+), norm=([\d.eE+-]+), penalty=([\d.eE+-]+)\), Total=([\d.eE+-]+)"
)
_NODES = re.compile(r"outer (\d+) nodes, inner (\d+) nodes")
_BETA = re.compile(r"Beta: \[([\d.eE+-]+), ([\d.eE+-]+)\]")
_EFFECTIVE = re.compile(r"Workflow effective config: (.*)$")
_DIRICHLET = re.compile(r"Added flexible Dirichlet BC: (\d+) nodes")
_FAILURE = re.compile(r"(did not converge|NaN residual|Linear solver|Traceback|Error|ERROR|failed)", re.I)


def _seconds(line: str) -> float | None:
    match = _STAMP.match(line)
    if not match:
        return None
    month, day, year, hour, minute, second, fraction = match.groups()
    stamp = time.mktime((int(year), int(month), int(day), int(hour), int(minute),
                         int(second), 0, 0, -1))
    return stamp + float("0." + fraction)


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_solve_log(path: Path) -> dict[str, Any]:
    """What a solver log says, as numbers: Newton history, optimization, markers."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"exists": False}
    newton: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    objective: dict[int, float] = {}
    failures: list[str] = []
    info: dict[str, Any] = {"exists": True, "path": str(path)}
    current_opt = None
    load_step = None
    first = last = None
    for line in lines:
        stamp = _seconds(line)
        if stamp is not None:
            first = stamp if first is None else first
            last = stamp
        body = _STAMP.sub("", line)
        if match := _NEWTON.search(body):
            newton.append({"opt_iter": current_opt, "load_step": load_step,
                           "scale": _float(match[1]), "iter": int(match[2]),
                           "residual": _float(match[3]), "rel": _float(match[4])})
        elif match := _LOAD_STEP.search(body):
            load_step = int(match[1])
            info["load_steps"] = int(match[2])
        elif match := _OPT_HEADER.search(body):
            current_opt = int(match[1])
        elif match := _SE.search(body):
            trajectory.append({
                "opt_iter": current_opt or 0, "se": _float(match[1]), "se_raw": _float(match[2]),
                "vol": _float(match[3]), "vol_raw": _float(match[4]), "target": _float(match[5]),
                "norm": _float(match[6]), "penalty": _float(match[7]), "total": _float(match[8]),
            })
        elif match := _OPT_ITER.search(body):
            objective[int(match[1])] = _float(match[2])
        elif match := _NODES.search(body):
            info.setdefault("nodes", []).append({"outer": int(match[1]), "inner": int(match[2])})
        elif match := _BETA.search(body):
            info["beta_range"] = [_float(match[1]), _float(match[2])]
        elif match := _EFFECTIVE.search(body):
            info["effective_config"] = dict(
                item.split("=", 1) for item in match[1].split(", ") if "=" in item
            )
        elif match := _DIRICHLET.search(body):
            info.setdefault("dirichlet_nodes", []).append(int(match[1]))
        if "Simulation finalized" in body:
            info["finalized"] = True
        if _FAILURE.search(body) and "Queued" not in body and "Saved" not in body:
            failures.append(body.strip()[:300])
    info["newton"] = newton
    info["trajectory"] = trajectory
    info["objective"] = [objective[k] for k in sorted(objective)]
    info["failures"] = failures[-10:]
    info["duration_s"] = (last - first) if first is not None and last is not None else None
    info.setdefault("finalized", False)
    info["last_line"] = _STAMP.sub("", lines[-1]).strip()[:300] if lines else ""
    return info


_LOAD_STEP_SCALE = re.compile(r"Load step (\d+)/(\d+): scale=([\d.eE+-]+)")


def load_step_progress(path: Path) -> dict[str, Any] | None:
    """How far a failed forward solve's load stepping got, from its log: each "Load step k/N:
    scale=s" line starts step k, so every step before the last one started converged.
    {"steps": N, "converged": k - 1, "last_converged_scale": s(k-1), "failed_at_scale": s(k)}, or
    None without load steps or before a step converged.  An optimization ramps only its first
    solve (solver_settings.md §1), so its log says nothing about where a later solve failed: None."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    started: list[tuple[int, int, float]] = []
    for line in lines:
        body = _STAMP.sub("", line)
        if _OPT_HEADER.search(body):
            return None
        match = _LOAD_STEP_SCALE.search(body)
        if match:
            if int(match[1]) == 1:
                started = []          # a new ramp: only the last one is the failed solve's
            started.append((int(match[1]), int(match[2]), _float(match[3])))
    if len(started) < 2 or started[-1][1] < 2:
        return None
    last, done = started[-1], started[-2]
    if not (0.0 < done[2] < last[2] <= 1.0):
        return None
    return {"steps": last[1], "converged": done[0], "last_converged_scale": done[2],
            "failed_at_scale": last[2]}


def newton_summary(newton: list[dict[str, Any]]) -> dict[str, Any]:
    """The last solve's convergence, and the signature of divergence (rel > 1)."""
    if not newton:
        return {}
    # Each Newton solve restarts at iteration 0: one per load step, or one per
    # optimization iteration.  The last is what the run ended on.
    solves: list[list[dict[str, Any]]] = []
    for item in newton:
        if item["iter"] == 0 or not solves:
            solves.append([])
        solves[-1].append(item)
    final = solves[-1]
    return {
        "solves": len(solves),
        "final_iterations": final[-1]["iter"],
        "final_residual": final[-1]["residual"],
        "final_rel": final[-1]["rel"],
        "max_iterations": max(solve[-1]["iter"] for solve in solves),
        # The solve ended farther from equilibrium than it started: a divergence,
        # or a float32-floor stall whose plateau can lie above 1 (0.8 to 1.3
        # seen).  An inexact-Newton solve may pass through rel > 1
        # on its way down, so only the end counts.
        "rel_above_one": final[-1]["rel"] > 1.0 and final[-1]["iter"] > 0,
    }


def progress_line(info: dict[str, Any]) -> str:
    """One line for a status check while a solve or optimization is running."""
    if not info.get("exists"):
        return "no solver log yet"
    parts = []
    if info.get("trajectory"):
        last = info["trajectory"][-1]
        parts.append(f"opt iter {last['opt_iter']}: SE={last['se']:.4f} Vol={last['vol']:.4f} "
                     f"(target {last['target']:.4f}) total={last['total']:.4e}")
    if info.get("newton"):
        item = info["newton"][-1]
        parts.append(f"newton iter {item['iter']} |R|={item['residual']:.3e} rel={item['rel']:.3e}")
    if info.get("finalized"):
        parts.append("finalized")
    if info.get("failures"):
        parts.append("last problem: " + info["failures"][-1])
    return "; ".join(parts) or info.get("last_line", "")


# ------------------------------------------------------------------- VTUs --

def read_vtu(path: Path, names: Iterable[str] | None = None) -> dict[str, Any]:
    """Arrays from one of the solver's VTU files, as numpy arrays.

    The writer stores every DataArray inline as base64 of an 8-byte (UInt64)
    length followed by the raw little-endian values, uncompressed, so this
    needs numpy alone -- no meshio or vtk, neither of which the environments
    carry.  ``names`` limits the arrays decoded; ``Points`` is always included.
    """
    import base64
    import numpy as np

    dtypes = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
              "UInt8": "u1", "UInt32": "<u4", "UInt64": "<u8"}
    text = Path(path).read_text(encoding="latin-1")
    if 'compressor="' in text[:2000]:
        raise ValueError(f"{path}: compressed VTU is not supported")
    wanted = None if names is None else set(names) | {"Points"}
    arrays: dict[str, Any] = {}
    for match in re.finditer(r'<DataArray([^>]*)>\s*([^<]*)</DataArray>', text):
        attributes, payload = match.group(1), match.group(2).strip()
        name = re.search(r'Name="([^"]+)"', attributes)
        kind = re.search(r'type="([^"]+)"', attributes)
        if not name or not kind or (wanted is not None and name[1] not in wanted):
            continue
        components = re.search(r'NumberOfComponents="(\d+)"', attributes)
        raw = base64.b64decode(payload)
        size = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        values = np.frombuffer(raw[8:8 + size], dtype=dtypes[kind[1]])
        if components and int(components[1]) > 1:
            values = values.reshape(-1, int(components[1]))
        arrays[name[1]] = values
    return arrays


def von_mises(voigt) -> Any:
    """Von Mises of stresses stored as Voigt 6-vectors (xx, yy, zz, xy, yz, xz)."""
    import numpy as np

    s = np.asarray(voigt, dtype=np.float64)
    xx, yy, zz, xy, yz, xz = (s[:, i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + yz ** 2 + xz ** 2))


# ------------------------------------------------------------- body pieces --

def face_pieces(mesh):
    """Each face's surface component (the connected surface it lies on) and their number."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(mesh.faces)
    pairs = np.asarray(mesh.face_adjacency).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    return labels, int(count)


def solid_pieces(mesh):
    """Each face's solid piece and the number of solid pieces: the surface components grouped by the
    solid they bound.  A sealed hollow body has two surface components (its outer skin and the
    inner one of its void), one solid: counted by surface, the void's inner skin read as a piece no
    support holds (42 % of the surface of a sealed shell, a review of 2026-10-10).  A component of
    negative signed volume (an inward-facing skin, a void's) belongs to the smallest outward one
    whose box holds its box; an outward component inside a void stays a piece of its own.
    Shared by the runner and the session summary (its loose-pieces warning)."""
    import numpy as np
    labels, count = face_pieces(mesh)
    if count < 2:
        return labels, count
    triangles = np.asarray(mesh.triangles)
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])) / 6.0
    volume = np.bincount(labels, weights=signed, minlength=count)
    lo = np.full((count, 3), np.inf)
    hi = np.full((count, 3), -np.inf)
    corners = triangles.reshape(-1, 3)
    owner = np.repeat(labels, 3)
    np.minimum.at(lo, owner, corners)
    np.maximum.at(hi, owner, corners)
    solid = np.arange(count)
    outward = [k for k in range(count) if volume[k] > 0]
    tol = 1e-9 * float(np.ptp(corners, axis=0).max() or 1.0)
    for k in range(count):
        if volume[k] >= 0:
            continue
        hosts = [j for j in outward if (lo[j] <= lo[k] + tol).all() and (hi[k] <= hi[j] + tol).all()]
        if hosts:
            solid[k] = min(hosts, key=lambda j: float(np.prod(hi[j] - lo[j])))
    _, solid = np.unique(solid, return_inverse=True)
    solid = solid[labels]
    return solid, int(solid.max()) + 1


# ------------------------------------------------------------------- GPUs --

@dataclass
class GPU:
    index: int
    uuid: str
    name: str
    used_mib: float
    total_mib: float
    util: float


@dataclass
class GPUVerdict:
    index: int
    name: str
    total_mib: float
    used_mib: float
    ours_mib: float
    foreign_mib: float
    util: float
    foreign_pids: list[int] = field(default_factory=list)
    our_workers: list[str] = field(default_factory=list)
    idle: bool = False
    reason: str = ""

    @property
    def free_mib(self) -> float:
        return self.total_mib - self.used_mib

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "name": self.name, "idle": self.idle, "reason": self.reason,
            "used_mib": round(self.used_mib), "total_mib": round(self.total_mib),
            "free_mib": round(self.free_mib), "ours_mib": round(self.ours_mib),
            "foreign_mib": round(self.foreign_mib), "util_pct": round(self.util, 1),
            "foreign_pids": self.foreign_pids, "our_workers": self.our_workers,
        }


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _smi(query: str, kind: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()[:200]}")
    return [[cell.strip() for cell in row.split(",")]
            for row in result.stdout.strip().splitlines() if row.strip()]


def query_gpus() -> list[GPU]:
    rows = _smi("index,uuid,name,memory.used,memory.total,utilization.gpu", "gpu")
    return [GPU(int(r[0]), r[1], r[2], _float(r[3]), _float(r[4]), _float(r[5])) for r in rows]


def query_apps() -> list[dict[str, Any]]:
    try:
        rows = _smi("pid,gpu_uuid,used_memory", "compute-apps")
    except RuntimeError:
        return []
    return [{"pid": int(r[0]), "uuid": r[1], "used_mib": _float(r[2])}
            for r in rows if r and r[0].isdigit()]


def _alive(pid: int) -> bool:
    state = runtime_files.process_state(pid)
    return state is not None and state not in ("Z", "X")


def worker_files(output_root: Path) -> list[tuple[int, str, Path]]:
    """(pid, kind, pid file) for every live resident worker of this output root.

    Found by scanning the pid files rather than recomputing their names: the
    name carries a hash of the output root and the worker's environment
    (runtime_files.worker_key), and the runner does not know which environment
    each model worker runs in.
    """
    found = []
    for pid_file in (Path(output_root) / ".workers").glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if _alive(pid):
            found.append((pid, pid_file.stem.rsplit("-", 1)[0], pid_file))
    return found


def our_workers(output_root: Path) -> dict[int, str]:
    """Process groups of this output root's resident workers: pgid -> kind.

    Workers are started with ``start_new_session``, so the pid in each pid file
    is also the process group every child of that worker belongs to.
    """
    return {pid: kind for pid, kind, _ in worker_files(output_root)}


def stop_worker_file(pid: int, kind: str, pid_file: Path) -> None:
    """Stop one resident worker the way the workflow does, found by its pid file.

    The workflow's own stop (runtime_files.stop_worker_at: ask over its socket,
    signal its process group, remove its files), with the socket named by the
    same key as the pid file.
    """
    key = pid_file.stem.rsplit("-", 1)[-1]
    runtime_files.stop_worker_at(runtime_files.worker_socket(key, kind), pid_file, kind)


def pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def process_env(pid: int, name: str) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for item in raw.split(b"\0"):
        if item.startswith(name.encode() + b"="):
            return item.split(b"=", 1)[1].decode(errors="replace")
    return None


WORKFLOW_MARKERS = ("fem_workflow/scripts/", "partfield_wrapper/", "hunyuan3d_wrapper/",
                    "vomp_wrapper/", "cad_wrapper/", "triposg_wrapper/")


def workflow_process(pid: int) -> str | None:
    """A label when ``pid`` is one of this user's workflow processes, else None.

    Covers a worker whose pid file is already gone -- a solve spare deletes its
    own the moment it answers, while it still holds its memory -- and workers
    another front end of the same user started.  Another user's copy of the
    workflow is foreign: its files are not under this repository.
    """
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return None
        # A worker just stopped is a zombie for a moment, its command line already empty and
        # its pid file gone, while nvidia-smi still lists its memory (the paint worker's 17 GB,
        # stopped as the settings card opened, held the solve's GPU check 14.6 s; 2026-10-09).
        # One of this user's processes on its way out is nobody else's job.
        if runtime_files.process_state(pid) in ("Z", "X"):
            return "exiting"
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace")
    except (OSError, IndexError):
        return None
    if str(REPO_ROOT) not in cmdline:
        return None
    for marker in WORKFLOW_MARKERS:
        if marker in cmdline:
            script = cmdline.split(marker, 1)[1].split()[0]
            return f"{marker.split('/')[0]}:{script}"
    return None


def app_owners(apps: list[dict[str, Any]], output_root: Path,
               sticky: dict[int, str] | None = None) -> dict[int, str | None]:
    """pid -> label for our processes, None for foreign ones."""
    workers = our_workers(output_root)
    own_group = os.getpgid(0)
    owners: dict[int, str | None] = {}
    for app in apps:
        pid = app["pid"]
        group = pgid(pid)
        label = (workers.get(group) if group in workers
                 else "driver" if group == own_group
                 else (sticky or {}).get(pid) or workflow_process(pid))
        owners[pid] = label
        if label and sticky is not None:
            sticky[pid] = label
    return owners


# Memory not attributed to any process of ours that still lets utilization be called ours: the
# driver's CUDA context and rounding between nvidia-smi's two queries stay under it.
FOREIGN_SLACK_MIB = 256.0


def classify(samples: list[list[GPU]], apps: list[dict[str, Any]],
             owners: dict[int, str | None],
             max_foreign_mib: float = 1024.0, max_util: float = 10.0,
             sample_apps: list[list[dict[str, Any]]] | None = None) -> list[GPUVerdict]:
    """Decide which GPUs are idle from repeated samples and the compute apps.

    Memory is attributed per process (``owners``: a label for ours, None for
    foreign) -- another user's model that merely holds a context here is
    foreign.  A GPU is idle when foreign memory stays under ``max_foreign_mib``
    on every sample, and its utilisation stays under ``max_util`` percent on
    every sample in which someone else is on it (a foreign process, or more
    memory than ours by ``FOREIGN_SLACK_MIB``) or none of our processes is.  Utilisation cannot be
    attributed per process, so a GPU running only our own work -- the previous
    step winding down, a worker the runner itself just started loading -- used
    to read as busy, and the runner waited for itself: 135 s of 10 s rounds over
    five single-GPU runs.  Another user's job still makes it busy.

    ``sample_apps``, one compute-app list per sample, attributes memory sample by
    sample, so a worker that exits between samples is not counted as foreign;
    without it the one ``apps`` snapshot serves every sample.

    Memory no listed process accounts for counts as foreign, and makes the utilisation count,
    only as much of it as is there in every sample.  nvidia-smi's total and its process list
    are read apart, and a process starting or exiting is in one and not the other for a moment:
    a new CUDA context shows in the total before its process is listed (228 MiB unattributed
    in the first sample after the runner started a solver spare, listed as ours two seconds
    later), and a stopped worker's memory stays in the total after it leaves the list (the
    paint worker's 17 GB, stopped as the settings card opened, read as foreign memory in the
    solve's check).  Each held a solve 14.6 s (2026-10-09).  Another user's job that nvidia-smi
    cannot list is in every sample, so it still counts; a listed foreign process always does.
    A single-sample check cannot tell, so it counts it all, and ``select_gpu`` re-checks with
    three samples at once instead of waiting.
    """
    per_sample = sample_apps if sample_apps is not None and len(sample_apps) == len(samples) \
        else [apps] * len(samples)
    verdicts = []
    for gpu in samples[-1]:
        used = util = listed_foreign = 0.0
        ours_mib, foreign_pids, workers = 0.0, set(), {}
        unlisted, unlisted_utils = [], []   # per sample: memory nobody listed; utilisation beside it
        for sample, sample_apps_ in zip(samples, per_sample):
            g = next((x for x in sample if x.uuid == gpu.uuid), None)
            if g is None:
                continue
            ours_here, foreign_here, foreign_listed_here = 0.0, [], 0.0
            for app in sample_apps_:
                if app["uuid"] != gpu.uuid:
                    continue
                label = owners.get(app["pid"])
                if label:
                    ours_here += app["used_mib"]
                    workers[app["pid"]] = f"{label}(pid {app['pid']}, {round(app['used_mib'])} MiB)"
                else:
                    foreign_here.append(app["pid"])
                    foreign_listed_here += app["used_mib"]
            unlisted_here = max(g.used_mib - ours_here - foreign_listed_here, 0.0)
            used, ours_mib = max(used, g.used_mib), max(ours_mib, ours_here)
            listed_foreign = max(listed_foreign, foreign_listed_here)
            foreign_pids.update(foreign_here)
            unlisted.append(unlisted_here)
            if foreign_here:
                # someone else is on it: the load is not ours to discount
                util = max(util, g.util)
            elif unlisted_here >= FOREIGN_SLACK_MIB or ours_here <= 0:
                # memory nobody listed accounts for, or nobody we know on it: someone else's
                # only if it lasts (see the docstring)
                unlisted_utils.append(g.util)
        lasting = min(unlisted) if len(unlisted) >= 2 else (unlisted[0] if unlisted else 0.0)
        foreign = listed_foreign + lasting
        if unlisted_utils and (len(unlisted_utils) == len(unlisted) or len(unlisted) < 2):
            util = max(util, *unlisted_utils)
        reasons = []
        if foreign >= max_foreign_mib:
            reasons.append(f"foreign memory {foreign:.0f} MiB >= {max_foreign_mib:.0f}")
        if util >= max_util:
            reasons.append(f"utilization {util:.0f}% >= {max_util:.0f}% with another user on it")
        verdicts.append(GPUVerdict(
            gpu.index, gpu.name, gpu.total_mib, used, ours_mib, foreign,
            max(x.util for sample in samples for x in sample if x.uuid == gpu.uuid),
            sorted(foreign_pids), list(workers.values()), idle=not reasons,
            reason="; ".join(reasons) or "idle",
        ))
    return verdicts


def sample_gpus(count: int = 3, interval: float = 1.0) -> list[list[GPU]]:
    samples = []
    for i in range(count):
        samples.append(query_gpus())
        if i + 1 < count:
            time.sleep(interval)
    return samples


def gpu_table(output_root: Path, count: int = 3, interval: float = 1.0,
              max_foreign_mib: float = 1024.0, max_util: float = 10.0) -> list[GPUVerdict]:
    samples, sample_apps, owners = [], [], {}
    for i in range(count):
        samples.append(query_gpus())
        sample_apps.append(query_apps())             # memory attributed per sample
        # Owners resolved as each sample is taken: a worker that is exiting is identifiable
        # (workflow_process) only while its /proc entry lasts, which is not until the last sample.
        owners.update({pid: label for pid, label in app_owners(sample_apps[-1], output_root).items()
                       if label or pid not in owners})
        if i + 1 < count:
            time.sleep(interval)
    verdicts = classify(samples, sample_apps[-1], owners,
                        max_foreign_mib=max_foreign_mib, max_util=max_util, sample_apps=sample_apps)
    allowed = allowed_gpus()
    return [v for v in verdicts if allowed is None or v.index in allowed]


# A GPU found idle this recently (by any command of this output root) is checked with one
# sample instead of three: a fresh process re-sampled for 2.2 s before every GPU step, even
# seconds after the previous command picked the same GPU.  Five minutes, not one: an agent
# often reads a stop for longer than a minute, and one run paid three full
# checks (7 s) for it.  The one sample is still a check -- another user's process or memory on
# it is busy, and a busy single sample is re-checked with three at once (select_gpu).
VERDICT_FRESH_S = 300.0


def recent_idle(output_root: Path, index: int | None = None) -> bool:
    """Whether a GPU (any, or ``index``) was found idle with nobody else on it within
    VERDICT_FRESH_S, by the verdict cache ``select_gpu`` keeps."""
    record = read_json(Path(output_root) / ".runtime_cache" / "gpu_verdict.json", {}) or {}
    if time.time() - float(record.get("time") or 0) > VERDICT_FRESH_S:
        return False
    return any(v.get("idle") and not v.get("foreign_pids") and (index is None or v.get("index") == index)
               for v in record.get("gpus") or [])


def note_verdicts(output_root: Path, verdicts: list[GPUVerdict]) -> None:
    try:
        path = Path(output_root) / ".runtime_cache" / "gpu_verdict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"time": time.time(), "gpus": [v.as_dict() for v in verdicts]})
    except OSError:
        pass


def choose_gpu(verdicts: list[GPUVerdict], warm: set[int] | frozenset[int] = frozenset()) -> GPUVerdict | None:
    """The idle GPU to run on: one holding a worker the step needs (``warm``, by index) first,
    then the most free memory.  Free memory alone counts our own warm worker as used, so it
    picked the other GPU and the worker was stopped and reloaded there: three paints of one run
    alternated GPU 0/1/0, all cold at 85-87 s against 29-42 s warm."""
    idle = [v for v in verdicts if v.idle]
    return max(idle, key=lambda v: (v.index in warm, v.free_mib)) if idle else None


def worker_gpu(pid: int, apps: list[dict[str, Any]], gpus: list[GPU]) -> list[int]:
    """Which physical GPUs a worker's process group uses, or will use.

    A worker holding memory shows up among the compute apps.  One still loading
    shows nothing yet, so its ``CUDA_VISIBLE_DEVICES`` says where it will land;
    unset means every device is visible and CUDA picks device 0.
    """
    uuid_to_index = {g.uuid: g.index for g in gpus}
    used = sorted({uuid_to_index[a["uuid"]] for a in apps
                   if pgid(a["pid"]) == pid and a["uuid"] in uuid_to_index})
    if used:
        return used
    visible = process_env(pid, "CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return [0]
    try:
        return [int(visible.split(",")[0])]
    except ValueError:
        return [0]


class GPUSampler(threading.Thread):
    """Samples every GPU while a step runs, to report its peak VRAM.

    Also notices another job arriving on the chosen GPU mid-step: contention on
    a shared card has corrupted runs here before, and a result produced under it
    deserves a second look.
    """

    def __init__(self, output_root: Path, chosen: int | None, interval: float = 2.0):
        super().__init__(name="gpu-sampler", daemon=True)
        self.output_root = Path(output_root)
        self.chosen = chosen
        self.interval = interval
        self.stop_event = threading.Event()
        self.baseline: dict[int, float] = {}
        self.peak: dict[int, float] = {}
        self.peak_util: dict[int, float] = {}
        self.foreign_baseline: float | None = None
        self.foreign_peak = 0.0
        self.foreign_new_pids: set[int] = set()
        self.samples = 0
        self.started = time.time()
        self._initial_foreign_pids: set[int] | None = None
        self._sticky: dict[int, str] = {}

    def _sample(self) -> None:
        try:
            gpus = query_gpus()
            apps = query_apps()
        except Exception:
            return
        self.samples += 1
        for gpu in gpus:
            self.baseline.setdefault(gpu.index, gpu.used_mib)
            self.peak[gpu.index] = max(self.peak.get(gpu.index, 0.0), gpu.used_mib)
            self.peak_util[gpu.index] = max(self.peak_util.get(gpu.index, 0.0), gpu.util)
        if self.chosen is None:
            return
        verdict = next((v for v in classify(
            [gpus], apps, app_owners(apps, self.output_root, self._sticky),
        ) if v.index == self.chosen), None)
        if verdict is None:
            return
        pids = set(verdict.foreign_pids)
        if self._initial_foreign_pids is None:
            self._initial_foreign_pids = pids
            self.foreign_baseline = verdict.foreign_mib
        self.foreign_new_pids |= pids - self._initial_foreign_pids
        self.foreign_peak = max(self.foreign_peak, verdict.foreign_mib)
        # What the workflow itself held on the card, resident models included --
        # a preloaded model is in the baseline, so the delta alone understates it.
        self.ours_peak = max(getattr(self, "ours_peak", 0.0), verdict.ours_mib)

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        self.join(timeout=10)
        self._sample()
        report: dict[str, Any] = {"duration_s": round(time.time() - self.started, 1),
                                  "samples": self.samples}
        if self.chosen is not None and self.chosen in self.peak:
            base = self.baseline[self.chosen]
            report.update({
                "index": self.chosen,
                "baseline_mib": round(base), "peak_mib": round(self.peak[self.chosen]),
                "peak_delta_mib": round(self.peak[self.chosen] - base),
                "workflow_peak_mib": round(getattr(self, "ours_peak", 0.0)),
                "peak_util_pct": round(self.peak_util.get(self.chosen, 0.0)),
                "foreign_appeared": bool(self.foreign_new_pids) or (
                    self.foreign_baseline is not None
                    and self.foreign_peak - self.foreign_baseline > 512.0
                ),
            })
        report["all_gpus_peak_mib"] = {str(k): round(v) for k, v in self.peak.items()}
        # A GPU step that left the chosen card untouched while another one grew
        # by gigabytes most likely ran elsewhere -- a resident worker started by
        # another front end, unpinned.  It cannot be proven from here (another
        # user's job looks the same), so it is reported, not asserted.
        if self.chosen is not None and self.chosen in self.peak:
            others = {k: self.peak[k] - self.baseline.get(k, self.peak[k])
                      for k in self.peak if k != self.chosen}
            grew = {k: round(v) for k, v in others.items() if v > 1024}
            if grew and report.get("peak_delta_mib", 0) < 256:
                report["suspected_elsewhere"] = grew
        return report
