"""Turn a per-face part labelling and one material per part into a material field.

The field is written as the same 64³ artifact VoMP produces
(``src.vomp_wrapper.artifact``), so everything that reads a VoMP field -- the
solver, the case builder, the UI preview -- reads this one unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

# Surface samples per cell edge: dense enough that every cell the surface
# crosses holds a sample, so thin parts are not lost to the fill.
SAMPLES_PER_CELL = 3
MAX_SAMPLES = 4_000_000


def _surface_samples(mesh, cell: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over the surface, each with the face it lies on."""
    corners = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    ) / 2
    wanted = int(area.sum() / (cell / SAMPLES_PER_CELL) ** 2)
    count = min(max(wanted, len(corners)), MAX_SAMPLES)
    # Every face's centroid, so no face goes unrepresented, plus area-weighted points.
    face = np.concatenate([
        np.arange(len(corners)),
        rng.choice(len(corners), max(count - len(corners), 0), p=area / area.sum()),
    ])
    r = rng.random((len(face), 2))
    r = np.where(r.sum(1, keepdims=True) > 1, 1 - r, r)
    weights = np.column_stack([1 - r.sum(1), r])
    weights[: len(corners)] = 1 / 3
    return np.einsum("ij,ijk->ik", weights, corners[face]), face


def part_grid(mesh, face_labels: np.ndarray, grid_size: int = 64, seed: int = 0):
    """Occupied lattice points and the part each belongs to.

    The lattice is VoMP's: ``centre + (k / n - 1/2) * scale`` over the bounding
    box's longest side.  A cell is occupied when the surface crosses it or it is
    enclosed by such cells; each takes the part of the nearest surface point, so
    a part's interior carries that part's material.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    face_labels = np.asarray(face_labels)
    if face_labels.shape != (len(mesh.faces),):
        raise ValueError(
            f"{len(face_labels)} part labels for a surface of {len(mesh.faces)} faces"
        )
    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    center, scale = (lo + hi) / 2, float((hi - lo).max())
    cell = scale / grid_size
    points, face = _surface_samples(mesh, cell, np.random.default_rng(seed))
    index = np.clip(
        np.rint(((points - center) / scale + 0.5) * grid_size).astype(np.int64),
        0, grid_size - 1,
    )
    occupied = np.zeros((grid_size,) * 3, dtype=bool)
    occupied[tuple(index.T)] = True
    occupied = ndimage.binary_fill_holes(occupied)
    cells = np.argwhere(occupied)
    coords = center + (cells / grid_size - 0.5) * scale
    _, nearest = cKDTree(points).query(coords, workers=-1)
    return coords, face_labels[face[nearest]], center, scale


def field_from_parts(
    mesh,
    face_labels: np.ndarray,
    part_values: Mapping[int, tuple[float, float, float]],
    output: str | Path,
    *,
    unit_system: str = "SI (m-kg-s)",
    provenance: Mapping[str, object] | None = None,
):
    """Write the artifact for ``part_values``: part id -> (E Pa, ν, ρ kg/m³).

    Values are SI; the artifact converts them to ``unit_system`` as it does
    VoMP's.  Every part present on the surface needs a value.
    """
    from src.vomp_wrapper.artifact import write_artifact

    coords, labels, center, scale = part_grid(mesh, face_labels)
    missing = sorted(set(np.unique(labels).tolist()) - {int(k) for k in part_values})
    if missing:
        raise ValueError(f"no material for parts {missing}")
    table = {int(k): v for k, v in part_values.items()}
    values = np.asarray([table[int(label)] for label in labels], dtype=np.float64)
    return write_artifact(
        output,
        {
            "voxel_coords_world": coords,
            "youngs_modulus": values[:, 0],
            "poisson_ratio": values[:, 1],
            "density": values[:, 2],
            "transform_center": center,
            "transform_scale": scale,
        },
        unit_system=unit_system,
        provenance={"method": "parts", **dict(provenance or {})},
    )
