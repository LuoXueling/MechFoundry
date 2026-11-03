"""Canonical, memory-mappable VoMP material-grid artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


# 2: VoMP's voxels sit at the cell centres where their features were sampled
# (1 kept VoMP's reported positions, half a cell toward +x, +y, +z of that).
SCHEMA_VERSION = 2
READABLE_SCHEMAS = (1, 2)
GRID_SIZE = 64
CHANNELS = ("youngs_modulus", "poisson_ratio", "density")


@dataclass(frozen=True)
class MaterialArtifact:
    root: Path
    values: np.ndarray
    valid: np.ndarray
    manifest: dict

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.manifest["bounds"], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_scales(unit_system: str) -> tuple[float, float]:
    """Return SI-to-workflow multipliers for modulus and density."""
    if unit_system == "SI (m-kg-s)":
        return 1.0, 1.0
    if unit_system == "SI-mm (mm-tonne-s)":
        return 1.0e-6, 1.0e-12
    if unit_system == "US-inch (in-lbf-s)":
        return 1.0 / 6894.757293168, 9.357254e-8
    raise ValueError(f"unsupported unit system: {unit_system}")


def dense_grid_from_sparse(
    coords: np.ndarray,
    properties: np.ndarray,
    *,
    center: np.ndarray,
    scale: float,
    unit_system: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter VoMP voxels and nearest-fill the fixed 64^3 grid once."""
    # SciPy belongs to the optional VoMP inference environment.  Keep it off
    # the artifact read path so the UI/agent environment can inspect a field
    # without duplicating VoMP's heavy numerical dependency stack.
    from scipy import ndimage

    coords = np.asarray(coords, dtype=np.float64)
    properties = np.asarray(properties, dtype=np.float32)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    scale = float(scale)
    if coords.ndim != 2 or coords.shape[1] != 3 or properties.shape != (len(coords), 3):
        raise ValueError("VoMP coordinates and properties must have shapes (N,3)")
    if not len(coords) or not np.isfinite(coords).all() or not np.isfinite(properties).all():
        raise ValueError("VoMP returned empty or non-finite material data")
    if scale <= 0:
        raise ValueError("VoMP transform scale must be positive")
    normalized = (coords - center[None, :]) / scale
    # VoMP samples its voxels on the lattice k / 64 - 1/2 of its unit box --
    # the centres trimesh's voxeliser returns -- and ``inference.restore_frame``
    # hands them over at those points.  They are the cell centres here, so the
    # grid starts half a cell below the box.  Only the far face of the longest
    # axis (k = 64) falls outside, into the last cell.
    indices = np.rint((normalized + 0.5) * GRID_SIZE).astype(np.int64)
    indices = np.clip(indices, 0, GRID_SIZE - 1)
    e_scale, rho_scale = unit_scales(unit_system)
    converted = properties.copy()
    converted[:, 0] *= e_scale
    converted[:, 2] *= rho_scale
    if np.any(converted[:, 0] <= 0) or np.any(converted[:, 2] <= 0):
        raise ValueError("VoMP modulus and density must be positive")
    if np.any(converted[:, 1] <= -1) or np.any(converted[:, 1] >= 0.5):
        raise ValueError("VoMP Poisson ratios must lie strictly between -1 and 0.5")

    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    counts = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.int32)
    # Duplicate voxel coordinates are reduced deterministically by their mean.
    for channel in range(3):
        np.add.at(grid[..., channel], tuple(indices.T), converted[:, channel])
    np.add.at(counts, tuple(indices.T), 1)
    valid = counts > 0
    grid[valid] /= counts[valid, None]
    if not valid.all():
        nearest = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        grid[~valid] = grid[tuple(axis[~valid] for axis in nearest)]
    half_cell = 0.5 / GRID_SIZE
    bounds = np.stack((center - (0.5 + half_cell) * scale,
                       center + (0.5 - half_cell) * scale))
    return np.ascontiguousarray(grid), valid, bounds


def write_artifact(
    root: str | Path,
    results: Mapping[str, object],
    *,
    unit_system: str,
    provenance: Mapping[str, object] | None = None,
) -> MaterialArtifact:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(results["voxel_coords_world"])
    properties = np.column_stack((
        results["youngs_modulus"], results["poisson_ratio"], results["density"],
    ))
    center = np.asarray(results.get("transform_center", (coords.min(0) + coords.max(0)) * 0.5))
    scale = float(results.get("transform_scale", np.ptp(coords, axis=0).max()))
    values, valid, bounds = dense_grid_from_sparse(
        coords, properties, center=center, scale=scale, unit_system=unit_system,
    )
    values_path = root / "material_grid.npy"
    valid_path = root / "valid_mask.npy"
    np.save(values_path, values, allow_pickle=False)
    np.save(valid_path, valid, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shape": list(values.shape),
        "channels": list(CHANNELS),
        "axis_order": "xyz_channels_last",
        "sampling": "nearest",
        "unit_system": unit_system,
        "bounds": bounds.tolist(),
        "occupied_voxels": int(valid.sum()),
        "source_voxels": int(len(coords)),
        "ranges": {
            channel: [float(values[..., index].min()), float(values[..., index].max())]
            for index, channel in enumerate(CHANNELS)
        },
        "provenance": dict(provenance or {}),
    }
    manifest["values_sha256"] = _sha256(values_path)
    manifest["valid_sha256"] = _sha256(valid_path)
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, root / "manifest.json")
    return MaterialArtifact(root, values, valid, manifest)


def load_artifact(root: str | Path, *, mmap: bool = True, verify: bool = False) -> MaterialArtifact:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in READABLE_SCHEMAS:
        raise ValueError(f"unsupported material artifact schema: {manifest.get('schema_version')}")
    values_path, valid_path = root / "material_grid.npy", root / "valid_mask.npy"
    if verify and (_sha256(values_path) != manifest.get("values_sha256")
                   or _sha256(valid_path) != manifest.get("valid_sha256")):
        raise ValueError("material artifact checksum mismatch")
    values = np.load(values_path, mmap_mode="r" if mmap else None, allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if values.shape != (GRID_SIZE, GRID_SIZE, GRID_SIZE, 3) or values.dtype != np.float32:
        raise ValueError(f"invalid material grid shape or dtype: {values.shape} {values.dtype}")
    if valid.shape != values.shape[:3] or valid.dtype != np.bool_:
        raise ValueError("invalid material valid-mask shape or dtype")
    return MaterialArtifact(root, values, valid, manifest)
