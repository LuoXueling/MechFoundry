"""Exact surface samples carried to the viewer, independent of display colors."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


def surface_hover(path: Path) -> dict | None:
    """Read values in GLB vertex order, or segment IDs in triangle order.

    Old assets without samples still support coordinates. Never invert a color
    ramp: quantization and display clipping discard the underlying values.
    """
    values = path.with_suffix(path.suffix + ".values.bin")
    if values.is_file():
        try:
            metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
            raw = values.read_bytes()
            if len(raw) % 8:
                return None
            return {
                "kind": "scalar", "data": base64.b64encode(raw).decode("ascii"),
                "field": metadata.get("field"), "component": metadata.get("component"),
                "yield_strength": metadata.get("yield_strength"),
            }
        except (OSError, ValueError):
            return None
    if path.name == "segments.glb":
        try:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                labels = np.asarray(saved["segment_ids"], dtype="<i4")
            return {"kind": "segment", "data": base64.b64encode(labels.tobytes()).decode("ascii")}
        except (OSError, ValueError, KeyError):
            return None
    return None
