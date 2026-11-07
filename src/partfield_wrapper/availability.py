"""Configuration and cheap readiness checks for optional PartField support."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "src" / "dependency" / "PartField"


def settings() -> tuple[str, Path, Path]:
    root = Path(os.environ.get("PARTFIELD_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    checkpoint = Path(os.environ.get(
        "PARTFIELD_CHECKPOINT", root / "model" / "model_objaverse.ckpt"
    )).expanduser().resolve()
    return os.environ.get("PARTFIELD_CONDA_ENV", "mechfoundry-partfield"), root, checkpoint


def _current_environment(name: str) -> str:
    """``runtime_files.current_environment``: the one table of the MechFoundry env rename.

    An override such as ``HUNYUAN3D_CONDA_ENV=implicit-hunyuan3d`` written before the rename
    (in a shell profile or the agents environment's conda variables) otherwise names an
    environment that no longer exists once the old ones are removed, and every optional model
    reads as not installed although its ``mechfoundry-*`` environment is (2026-10-23 reinstall).
    """
    try:  # imported here: the package builds the graph on import and needs LangGraph
        from src.agents_wrapper.fem_workflow.runtime_files import current_environment
    except (ImportError, TypeError):
        # An interpreter that cannot load the workflow takes the name as given: one without its
        # dependencies (ImportError), or mechfoundry-text2cad, which upstream pins to Python 3.9,
        # where the package's runtime `X | Y` unions (wrapper.py's aliases) raise TypeError.
        return name
    return current_environment(name)


def conda_interpreter(name: str) -> Path | None:
    """Resolve a named environment once; callers launch it without ``conda run``.

    A pre-rename ``implicit-*`` name resolves to its ``mechfoundry-*`` environment once the old
    one is gone, as stored compute environments do.
    """
    name = _current_environment(name)
    # The normal installation shares a conda root with the running UI.  Taking
    # that direct path avoids launching conda merely to rediscover it on every
    # fresh workflow process.
    roots = []
    executable = os.environ.get("CONDA_EXE")
    if executable:
        roots.append(Path(executable).resolve().parents[1])
    for variable in ("CONDA_ROOT", "CONDA_PREFIX_1"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).resolve())
    for root in roots:
        candidate = root / "envs" / name / "bin" / "python"
        if candidate.is_file():
            return candidate
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"], capture_output=True, text=True,
            timeout=15, check=False,
        )
        paths = json.loads(result.stdout).get("envs", []) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        paths = []
    for raw in paths:
        path = Path(raw)
        candidate_name = "base" if path.parent.name not in ("envs", "env") else path.name
        if candidate_name == name:
            interpreter = path / "bin" / "python"
            return interpreter if interpreter.is_file() else None
    return None


@dataclass(frozen=True)
class PartFieldStatus:
    ready: bool
    environment: str
    root: str
    checkpoint: str
    summary: str
    interpreter: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None

    def model_dump(self) -> dict:
        return self.__dict__.copy()


_PROBE = """
import importlib.metadata
import importlib.util
import json
import sys
sys.path.insert(0, sys.argv[1])
required = ('torch', 'lightning', 'torch_scatter', 'sklearn', 'trimesh', 'pytz', 'partfield')
missing = [name for name in required if importlib.util.find_spec(name) is None]
try:
    torch_version = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    torch_version = None
result = dict(ok=not missing, torch=torch_version, cuda=None,
              error=('missing packages: ' + ', '.join(missing)) if missing else None)
print(json.dumps(result, separators=(',', ':')))
"""


@lru_cache(maxsize=8)
def partfield_status() -> PartFieldStatus:
    environment, root, checkpoint = settings()
    interpreter = conda_interpreter(environment)
    base = dict(environment=environment, root=str(root), checkpoint=str(checkpoint))
    if interpreter is None:
        return PartFieldStatus(False, **base, summary=f"Conda environment {environment!r} is not installed")
    if not (root / "partfield_inference.py").is_file() or not (root / "partfield").is_dir():
        return PartFieldStatus(False, **base, interpreter=str(interpreter),
                               summary=f"PartField source is missing at {root}")
    if not checkpoint.is_file():
        return PartFieldStatus(False, **base, interpreter=str(interpreter),
                               summary=f"PartField checkpoint is missing at {checkpoint}")
    try:
        result = subprocess.run(
            [str(interpreter), "-c", _PROBE, str(root)], capture_output=True, text=True,
            timeout=45, check=False,
        )
        payload = json.loads(result.stdout.splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as exc:
        return PartFieldStatus(False, **base, interpreter=str(interpreter),
                               summary=f"PartField probe failed: {type(exc).__name__}: {exc}")
    if result.returncode != 0 or not payload.get("ok"):
        detail = payload.get("error") or result.stderr.strip() or "probe failed"
        return PartFieldStatus(False, **base, interpreter=str(interpreter), summary=detail,
                               torch_version=payload.get("torch"), cuda_version=payload.get("cuda"))
    return PartFieldStatus(True, **base, interpreter=str(interpreter),
                           summary=f"Torch {payload.get('torch')} (CUDA checked at launch)",
                           torch_version=payload.get("torch"), cuda_version=payload.get("cuda"))
