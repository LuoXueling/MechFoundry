"""Configuration and cheap readiness checks for the optional CAD-Coder generator."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.partfield_wrapper.availability import conda_interpreter


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "src" / "dependency" / "CAD-Coder"
SOURCE = Path("inference.py")


def settings() -> tuple[str, Path, Path]:
    """The environment, the upstream checkout, and the model weights."""
    root = Path(os.environ.get("CADCODER_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    weights = Path(os.environ.get(
        "CADCODER_WEIGHTS", root / "pretrained_weights" / "CAD-Coder"
    )).expanduser().resolve()
    return os.environ.get("CADCODER_CONDA_ENV", "mechfoundry-cadcoder"), root, weights


@dataclass(frozen=True)
class CADCoderStatus:
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
missing = [name for name in ("torch", "transformers", "cadquery", "trimesh")
           if importlib.util.find_spec(name) is None]
print(json.dumps({"missing": missing}))
"""


@lru_cache(maxsize=8)
def cadcoder_status() -> CADCoderStatus:
    environment, root, weights = settings()

    def status(ready: bool, summary: str, interpreter: Path | None = None):
        return CADCoderStatus(
            ready, environment, str(root), str(weights), summary,
            str(interpreter) if interpreter else None,
        )

    interpreter = conda_interpreter(environment)
    if interpreter is None:
        return status(False, f"conda environment {environment!r} was not found")
    if not (root / SOURCE).is_file():
        return status(False, f"CAD-Coder source is missing under {root}", interpreter)
    if not (weights / "config.json").is_file() or not any(weights.glob("*.safetensors")):
        return status(False, f"CAD-Coder weights are missing at {weights}", interpreter)
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
    return status(True, "CAD-Coder ready", interpreter)
