"""Backend-neutral persistence for reviewed surface segmentations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TAB20 = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
)


def _geometry_arrays(mesh):
    """Cache immutable per-face geometry on a resident worker's mesh."""
    cached = getattr(mesh, "_fem_segmentation_arrays", None)
    if cached is None:
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        cached = (
            triangles, np.asarray(mesh.area_faces),
            np.asarray(mesh.face_normals), np.asarray(mesh.bounds),
        )
        mesh._fem_segmentation_arrays = cached
    return cached


def _colored_glb(mesh, labels: np.ndarray, colors: dict[int, str], output: Path,
                 triangles: np.ndarray | None = None):
    import trimesh

    palette = np.asarray([
        tuple(int(colors[index][offset:offset + 2], 16)
              for offset in (1, 3, 5)) + (255,)
        for index in range(len(colors))
    ], dtype=np.uint8)
    rgba = palette[labels]
    if triangles is None:
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    colored = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(len(triangles) * 3).reshape(-1, 3),
        process=False,
    )
    colored.visual.vertex_colors = np.repeat(rgba, 3, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    colored.export(output)


def write_segmentation(mesh, labels, *, method: str, num_clusters: int,
                       output_npz: Path, output_json: Path, output_glb: Path,
                       method_settings: dict | None = None,
                       shape_diameter=None,
                       shape_diameter_range: tuple[float, float] | None = None,
                       segment_fields: dict[int, dict] | None = None) -> list[str]:
    """Write the common segmentation artifacts consumed by the FEM workflow.

    ``segment_fields`` adds entries to a segment's metadata, keyed by the
    label as given (before labels are renumbered from zero).

    ``shape_diameter`` is per face and normalized to [0, 1] across the part;
    ``shape_diameter_range`` is the smoothed thickness its 0 and 1 stand for.
    With it, a segment's ``shape_diameter`` is a thickness in the surface's
    length units and the normalized figures move to ``shape_diameter_relative``;
    without it, only the normalized figures are known, and are written under
    the latter name alone, so neither can be mistaken for the other.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != len(mesh.faces):
        raise ValueError(
            f"segmentation has {labels.shape[0]} labels for {len(mesh.faces)} faces"
        )
    if labels.size == 0 or labels.min() < 0:
        raise ValueError("segmentation labels must be nonempty and nonnegative")
    given, labels = np.unique(labels, return_inverse=True)
    diameter = None if shape_diameter is None else np.asarray(
        shape_diameter, dtype=np.float32
    ).reshape(-1)
    if diameter is not None and diameter.shape != labels.shape:
        raise ValueError("shape diameter must contain one value per face")

    for output in (output_npz, output_json, output_glb):
        output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"segment_ids": labels}
    if diameter is not None:
        arrays["shape_diameter"] = diameter
    np.savez_compressed(output_npz, **arrays)

    segment_count = int(labels.max()) + 1
    colors = {index: TAB20[index % len(TAB20)] for index in range(segment_count)}
    triangles, face_areas, normals, mesh_bounds = _geometry_arrays(mesh)
    total_area = float(face_areas.sum())
    counts = np.bincount(labels, minlength=segment_count)
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ordered_areas = face_areas[order]
    area_sums = np.add.reduceat(ordered_areas, starts)
    face_centroids = triangles.mean(axis=1)
    centroids = np.add.reduceat(face_centroids[order], starts, axis=0) / counts[:, None]
    face_minimum = triangles.min(axis=1)
    face_maximum = triangles.max(axis=1)
    bounds_minimum = np.minimum.reduceat(face_minimum[order], starts, axis=0)
    bounds_maximum = np.maximum.reduceat(face_maximum[order], starts, axis=0)
    ordered_normals = normals[order]
    facing_by_axis = {}
    threshold = np.sqrt(0.5)
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for sign, condition in (
            ("+", ordered_normals[:, axis_index] >= threshold),
            ("-", ordered_normals[:, axis_index] <= -threshold),
        ):
            facing_by_axis[f"{sign}{axis_name}"] = (
                np.add.reduceat(condition.astype(np.int32), starts) / counts
            )
    if diameter is not None:
        ordered_diameter = diameter[order]
        diameter_minimum = np.minimum.reduceat(ordered_diameter, starts)
        diameter_maximum = np.maximum.reduceat(ordered_diameter, starts)
        diameter_mean = np.add.reduceat(ordered_diameter, starts) / counts

    segments = []
    for segment_id in range(segment_count):
        facing = {
            name: float(values[segment_id])
            for name, values in facing_by_axis.items()
        }
        entry = {
            "id": segment_id,
            "color": colors[segment_id],
            "faces": int(counts[segment_id]),
            "area_fraction": float(area_sums[segment_id] / max(total_area, 1e-12)),
            "bounds": [bounds_minimum[segment_id].tolist(),
                       bounds_maximum[segment_id].tolist()],
            "centroid": centroids[segment_id].tolist(),
            "facing_fraction_45deg": facing,
        }
        entry.update((segment_fields or {}).get(int(given[segment_id]), {}))
        if diameter is not None:
            relative = {
                "min": float(diameter_minimum[segment_id]),
                "mean": float(diameter_mean[segment_id]),
                "max": float(diameter_maximum[segment_id]),
            }
            entry["shape_diameter_relative"] = relative
            if shape_diameter_range is not None:
                low, high = (float(value) for value in shape_diameter_range)
                entry["shape_diameter"] = {
                    key: low + value * (high - low) for key, value in relative.items()
                }
        segments.append(entry)
    payload = {
        "method": method,
        "num_clusters": int(num_clusters),
        "segments": segments,
        "bounds": mesh_bounds.tolist(),
        **(method_settings or {}),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _colored_glb(mesh, labels, colors, output_glb, triangles)
    return [str(path.resolve()) for path in (output_npz, output_json, output_glb)]
