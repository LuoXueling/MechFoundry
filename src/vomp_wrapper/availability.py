"""Cheap discovery and compatibility checks for the optional VoMP runtime."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.partfield_wrapper.availability import conda_interpreter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "src" / "dependency" / "VoMP"
PINNED_SOURCE_REVISION = "ac6826f25ca5cfa122eb7bacb74be91b5a9eadf4"
PINNED_MODEL_REVISION = "f9639381b6b8012a8a975b5b9503aa2ea04f5709"


def settings() -> tuple[str, Path, Path, str | None]:
    root = Path(os.environ.get("VOMP_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    weights = Path(os.environ.get("VOMP_WEIGHTS", root / "weights")).expanduser().resolve()
    # The conda environment ships BLENDER_BIN as "./blender-3.0.1-linux-x64/
    # blender", which resolves only for a process whose working directory is
    # the VoMP root.  Inference runs from the session directory, so a relative
    # setting is resolved against the root it is relative to, and a setting
    # that names nothing on disk gives way to the bundled build rather than
    # travelling on to fail inside VoMP, where the path is reported without
    # saying which of the two supplied it.
    blender = os.environ.get("VOMP_BLENDER") or os.environ.get("BLENDER_BIN")
    if blender and not Path(blender).expanduser().is_absolute():
        blender = str((root / blender).resolve())
    if not blender or not Path(blender).expanduser().is_file():
        bundled = root / "blender-3.0.1-linux-x64" / "blender"
        blender = str(bundled) if bundled.is_file() else None
    return os.environ.get("VOMP_CONDA_ENV", "mechfoundry-vomp"), root, weights, blender


@dataclass(frozen=True)
class VompStatus:
    ready: bool
    environment: str
    root: str
    weights: str
    summary: str
    interpreter: str | None = None
    blender: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None

    def model_dump(self) -> dict:
        return self.__dict__.copy()


_PROBE = r"""
import importlib.metadata, importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
required = ('torch', 'torchvision', 'trimesh', 'scipy', 'safetensors', 'vomp')
missing = [name for name in required if importlib.util.find_spec(name) is None]
torch_version = cuda_version = None
cuda_ok = False
import_error = None
if not missing:
    try:
        import torch
        from vomp.inference import Vomp
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_ok = torch.cuda.is_available()
    except Exception as exc:
        import_error = type(exc).__name__ + ': ' + str(exc)
error = ('missing packages: ' + ', '.join(missing)) if missing else (import_error or (
    None if cuda_ok else 'torch cannot access a CUDA device'))
print(json.dumps(dict(ok=not missing and cuda_ok, torch=torch_version,
                      cuda=cuda_version, error=error), separators=(',', ':')))
"""


def _source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=8)
def vomp_status(environment: str | None = None, python: str | None = None) -> VompStatus:
    configured_env, root, weights, blender = settings()
    environment = environment or configured_env
    interpreter = Path(python).expanduser().resolve() if python else conda_interpreter(environment)
    base = dict(
        environment=environment, root=str(root), weights=str(weights), blender=blender,
        interpreter=str(interpreter) if interpreter else None,
    )
    if interpreter is None or not interpreter.is_file():
        return VompStatus(False, **base, summary=f"Conda environment {environment!r} is not installed")
    if not (root / "vomp" / "inference" / "vomp.py").is_file():
        return VompStatus(False, **base, summary=f"VoMP source is missing at {root}")
    revision = _source_revision(root)
    if revision and revision != PINNED_SOURCE_REVISION:
        return VompStatus(False, **base, summary=f"VoMP source revision is {revision}, expected {PINNED_SOURCE_REVISION}")
    required_weights = ("inference.json", "geometry_transformer.pt", "matvae.safetensors", "normalization_params.json")
    missing_weights = [name for name in required_weights if not (weights / name).is_file()]
    if missing_weights:
        return VompStatus(False, **base, summary="missing VoMP weights: " + ", ".join(missing_weights))
    if not blender or not Path(blender).expanduser().is_file():
        return VompStatus(False, **base, summary="mesh inference requires VOMP_BLENDER or VoMP's bundled Blender")
    try:
        result = subprocess.run(
            [str(interpreter), "-c", _PROBE, str(root)], capture_output=True,
            text=True, timeout=60, check=False,
        )
        payload = json.loads(result.stdout.splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as exc:
        return VompStatus(False, **base, summary=f"VoMP probe failed: {type(exc).__name__}: {exc}")
    if result.returncode or not payload.get("ok"):
        detail = payload.get("error") or result.stderr.strip() or "probe failed"
        return VompStatus(False, **base, summary=detail,
                          torch_version=payload.get("torch"), cuda_version=payload.get("cuda"))
    return VompStatus(
        True, **base,
        summary=f"Torch {payload.get('torch')} / CUDA {payload.get('cuda')}",
        torch_version=payload.get("torch"), cuda_version=payload.get("cuda"),
    )
