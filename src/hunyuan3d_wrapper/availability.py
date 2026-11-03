"""Configuration and cheap readiness checks for the optional Hunyuan3D-2.1 generator."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.partfield_wrapper.availability import conda_interpreter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "src" / "dependency" / "Hunyuan3D-2.1"
SOURCE = Path("hy3dshape") / "hy3dshape" / "pipelines.py"
REPO_ID = "tencent/Hunyuan3D-2.1"
# The upstream folders the two stages load: shape (DiT + VAE) and paint.
SUBFOLDERS = ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1", "hunyuan3d-paintpbr-v2-1")
REALESRGAN = Path("hy3dpaint") / "ckpt" / "RealESRGAN_x4plus.pth"
# rembg's background-removal model, found through U2NET_HOME.
U2NET = Path("pretrained_weights") / "u2net" / "u2net.onnx"
# Painting a bare surface from text: SDXL with a depth ControlNet draws the
# reference image the paint stage follows (OpenRAIL++, ungated), with the
# half-precision-safe VAE.
REFERENCE_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_REPO_ID = "diffusers/controlnet-depth-sdxl-1.0"
REFERENCE_VAE_REPO_ID = "madebyollin/sdxl-vae-fp16-fix"
REFERENCE_REPOS = (REFERENCE_REPO_ID, CONTROLNET_REPO_ID, REFERENCE_VAE_REPO_ID)


def settings() -> tuple[str, Path, Path]:
    """The environment, the upstream checkout, and the Hugging Face cache holding its weights."""
    root = Path(os.environ.get("HUNYUAN3D_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    weights = Path(os.environ.get(
        "HUNYUAN3D_WEIGHTS", root / "pretrained_weights" / "hf_cache"
    )).expanduser().resolve()
    return os.environ.get("HUNYUAN3D_CONDA_ENV", "mechfoundry-hunyuan3d"), root, weights


def snapshot_dir(weights: Path, repo_id: str = REPO_ID) -> Path | None:
    """The downloaded snapshot of ``repo_id`` in the cache ``weights``, if any."""
    snapshots = weights / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    found = sorted(snapshots.glob("*"), key=lambda path: path.stat().st_mtime) if snapshots.is_dir() else []
    return found[-1] if found else None


@dataclass(frozen=True)
class Hunyuan3DStatus:
    ready: bool
    environment: str
    root: str
    weights: str
    summary: str
    interpreter: str | None = None

    def model_dump(self) -> dict:
        return dict(self.__dict__)


_PROBE = """
import importlib.util, json
missing = [name for name in ("torch", "diffusers", "custom_rasterizer", "bpy", "trimesh", "scipy")
           if importlib.util.find_spec(name) is None]
print(json.dumps({"missing": missing}))
"""


@lru_cache(maxsize=8)
def hunyuan3d_status() -> Hunyuan3DStatus:
    environment, root, weights = settings()

    def status(ready: bool, summary: str, interpreter: Path | None = None):
        return Hunyuan3DStatus(
            ready, environment, str(root), str(weights), summary,
            str(interpreter) if interpreter else None,
        )

    interpreter = conda_interpreter(environment)
    if interpreter is None:
        return status(False, f"conda environment {environment!r} was not found")
    if not (root / SOURCE).is_file():
        return status(False, f"Hunyuan3D-2.1 source is missing under {root}", interpreter)
    snapshot = snapshot_dir(weights)
    missing = [name for name in SUBFOLDERS if snapshot is None or not (snapshot / name).is_dir()]
    if snapshot_dir(weights, "facebook/dinov2-giant") is None:
        missing.append("facebook/dinov2-giant")
    if missing:
        return status(False, f"Hunyuan3D-2.1 weights are missing in {weights}: "
                             + ", ".join(missing), interpreter)
    for weight in (REALESRGAN, U2NET):
        if not (root / weight).is_file():
            return status(False, f"{weight.name} is missing at {root / weight}", interpreter)
    try:
        probe = subprocess.run(
            [str(interpreter), "-c", _PROBE], capture_output=True, text=True,
            timeout=60, check=False,
        )
        missing = json.loads(probe.stdout.strip().splitlines()[-1])["missing"]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError, KeyError):
        return status(False, f"could not probe {environment!r}", interpreter)
    if missing:
        return status(False, f"{environment!r} lacks {', '.join(missing)}", interpreter)
    return status(True, "Hunyuan3D-2.1 ready", interpreter)


_SDXL_PROBE = """
import importlib.util, json
missing = [name for name in ("torch", "diffusers", "PIL", "scipy")
           if importlib.util.find_spec(name) is None]
print(json.dumps({"missing": missing}))
"""


@lru_cache(maxsize=8)
def sdxl_status() -> Hunyuan3DStatus:
    """Whether SDXL can draw an object from text, for the geometry card's image input.

    It needs only the SDXL base model and its half-precision-safe VAE -- the
    painter's reference models, in the same cache -- and an environment with
    diffusers, which the Hunyuan3D environment is.  Hunyuan3D-2.1 itself need not
    be installed: the drawing feeds TripoSG as well.
    """
    environment, root, weights = settings()

    def status(ready: bool, summary: str, interpreter: Path | None = None):
        return Hunyuan3DStatus(ready, environment, str(root), str(weights), summary,
                               str(interpreter) if interpreter else None)

    interpreter = conda_interpreter(environment)
    if interpreter is None:
        return status(False, f"conda environment {environment!r} was not found")
    missing = [repo for repo in (REFERENCE_REPO_ID, REFERENCE_VAE_REPO_ID)
               if snapshot_dir(weights, repo) is None]
    if missing:
        return status(False, f"SDXL weights are missing in {weights}: {', '.join(missing)}",
                      interpreter)
    try:
        probe = subprocess.run(
            [str(interpreter), "-c", _SDXL_PROBE], capture_output=True, text=True,
            timeout=60, check=False,
        )
        missing = json.loads(probe.stdout.strip().splitlines()[-1])["missing"]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError, KeyError):
        return status(False, f"could not probe {environment!r}", interpreter)
    if missing:
        return status(False, f"{environment!r} lacks {', '.join(missing)}", interpreter)
    return status(True, "SDXL ready", interpreter)


@lru_cache(maxsize=8)
def texture_status() -> Hunyuan3DStatus:
    """Whether a bare surface can be painted: Hunyuan3D-2.1 ready, and the reference models present.

    An uploaded reference image needs only the former; this reports on the
    text path, which needs both.
    """
    base = hunyuan3d_status()
    if not base.ready:
        return base
    weights = Path(base.weights)
    missing = [repo for repo in REFERENCE_REPOS if snapshot_dir(weights, repo) is None]
    if missing:
        return Hunyuan3DStatus(False, base.environment, base.root, base.weights,
                               f"text-guided texturing needs {', '.join(missing)} in {weights}",
                               base.interpreter)
    return Hunyuan3DStatus(True, base.environment, base.root, base.weights,
                           "Hunyuan3D-2.1 paint with SDXL reference ready", base.interpreter)
