"""Configuration and cheap readiness checks for the optional Text2CAD generator."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.cad_wrapper.cad_common.mesh import face_labels_path  # noqa: F401  (re-exported)
from src.partfield_wrapper.availability import conda_interpreter


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "src" / "dependency" / "Text2CAD"
SOURCES = (
    Path("Cad_VLM") / "models" / "text2cad.py",
    Path("CadSeqProc") / "cad_sequence.py",
)
# The text encoder the decoder was trained against, fetched from the Hub.
TEXT_ENCODER = "google-bert/bert-large-uncased"


def settings() -> tuple[str, Path, Path]:
    """The environment, the upstream checkout, and the model checkpoint."""
    root = Path(os.environ.get("TEXT2CAD_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    checkpoint = Path(os.environ.get(
        "TEXT2CAD_CHECKPOINT",
        root / "pretrained_weights" / "text2cad_v1.0" / "Text2CAD_1.0.pth",
    )).expanduser().resolve()
    return os.environ.get("TEXT2CAD_CONDA_ENV", "mechfoundry-text2cad"), root, checkpoint


def text_encoder_cache(root: Path) -> Path:
    """Where the BERT encoder is cached, beside the checkpoint."""
    return Path(root) / "pretrained_weights" / "hf_cache"


@dataclass(frozen=True)
class Text2CADStatus:
    ready: bool
    environment: str
    root: str
    checkpoint: str
    summary: str
    interpreter: str | None = None

    def model_dump(self) -> dict:
        return dict(self.__dict__)


_PROBE = """
import importlib.util, json
missing = [name for name in ("torch", "transformers", "OCC.Core", "trimesh")
           if importlib.util.find_spec(name) is None]
print(json.dumps({"missing": missing}))
"""


@lru_cache(maxsize=8)
def text2cad_status() -> Text2CADStatus:
    environment, root, checkpoint = settings()

    def status(ready: bool, summary: str, interpreter: Path | None = None):
        return Text2CADStatus(
            ready, environment, str(root), str(checkpoint), summary,
            str(interpreter) if interpreter else None,
        )

    interpreter = conda_interpreter(environment)
    if interpreter is None:
        return status(False, f"conda environment {environment!r} was not found")
    if not all((root / source).is_file() for source in SOURCES):
        return status(False, f"Text2CAD source is missing under {root}", interpreter)
    if not checkpoint.is_file():
        return status(False, f"Text2CAD checkpoint is missing at {checkpoint}", interpreter)
    encoder = text_encoder_cache(root) / ("models--" + TEXT_ENCODER.replace("/", "--"))
    if not encoder.is_dir():
        return status(False, f"BERT text encoder is not downloaded to {encoder.parent}", interpreter)
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
    return status(True, f"Text2CAD ready ({checkpoint.name})", interpreter)
