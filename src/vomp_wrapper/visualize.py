"""Project canonical VoMP grids onto an input surface for UI review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifact import CHANNELS, GRID_SIZE, MaterialArtifact


# The colour range: a few voxels predicted far outside the rest (88 GPa against
# a 0.2-5 GPa object) otherwise take most of the palette, and the parts VoMP
# does tell apart come out in one or two colours.  Values beyond it are clipped.
COLOUR_PERCENTILES = (2.0, 98.0)

PALETTE = np.asarray([
    [35, 65, 180], [30, 175, 210], [55, 190, 95],
    [245, 205, 55], [205, 35, 45],
], dtype=np.float64)


def surface_visualizations_current(root: str | Path) -> bool:
    """Whether all review assets were drawn from this grid, to the current colour contract.

    The grid is named by the manifest's ``values_sha256``, which each sidecar
    records.  A per-part field is rewritten in place when its values are
    edited, and checking the colour settings alone passed the previews of the
    values it replaced as current, so an edit never reached the viewer.
    """
    root = Path(root)
    try:
        grid = json.loads((root / "manifest.json").read_text(encoding="utf-8")).get(
            "values_sha256")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    for channel in CHANNELS:
        asset = root / f"{channel}.glb"
        sidecar = asset.with_suffix(asset.suffix + ".json")
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = "logarithmic" if channel in ("youngs_modulus", "density") else "linear"
        if (not asset.is_file() or metadata.get("scale") != expected
                or metadata.get("percentiles") != list(COLOUR_PERCENTILES)
                or not grid or metadata.get("values_sha256") != grid):
            return False
    return True


def sample_vertices(artifact: MaterialArtifact, vertices: np.ndarray) -> np.ndarray:
    """Nearest-sample all surface vertices in one vectorized host operation."""
    vertices = np.asarray(vertices, dtype=np.float64)
    bounds = artifact.bounds
    extent = bounds[1] - bounds[0]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or np.any(extent <= 0):
        raise ValueError("invalid surface vertices or material bounds")
    indices = np.floor((vertices - bounds[0]) / extent * GRID_SIZE).astype(np.int64)
    np.clip(indices, 0, GRID_SIZE - 1, out=indices)
    return np.asarray(artifact.values[tuple(indices.T)], dtype=np.float32)


def colour_range(values: np.ndarray) -> tuple[float, float]:
    """The values the palette spans: the percentiles, or all of them if flat."""
    low, high = np.percentile(values, COLOUR_PERCENTILES)
    if high <= low:
        low, high = values.min(), values.max()
    return float(low), float(high)


def _colors(values: np.ndarray, *, logarithmic: bool = False) -> np.ndarray:
    mapped = np.log10(values) if logarithmic else values
    minimum, maximum = colour_range(values)
    mapped_min, mapped_max = (
        (np.log10(minimum), np.log10(maximum)) if logarithmic else (minimum, maximum)
    )
    span = max(mapped_max - mapped_min, np.finfo(np.float32).eps)
    positions = np.clip((mapped - mapped_min) / span, 0.0, 1.0) * (len(PALETTE) - 1)
    lower = np.minimum(positions.astype(np.int64), len(PALETTE) - 2)
    blend = (positions - lower)[:, None]
    rgb = PALETTE[lower] * (1.0 - blend) + PALETTE[lower + 1] * blend
    return np.column_stack((rgb.astype(np.uint8), np.full(len(rgb), 255, dtype=np.uint8)))


def write_surface_visualizations(
    artifact: MaterialArtifact, surface: str | Path,
) -> list[Path]:
    """Write one colored GLB and metadata sidecar per material channel."""
    import trimesh

    loaded = trimesh.load(Path(surface), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices):
        raise ValueError("surface contains no triangle mesh")
    sampled = sample_vertices(artifact, mesh.vertices)
    outputs = []
    for index, channel in enumerate(CHANNELS):
        values = sampled[:, index]
        logarithmic = channel in ("youngs_modulus", "density")
        output = artifact.root / f"{channel}.glb"
        contour = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            vertex_colors=_colors(values, logarithmic=logarithmic), process=False,
        )
        contour.export(output)
        output.with_suffix(output.suffix + ".values.bin").write_bytes(
            np.ascontiguousarray(values, dtype="<f8").tobytes()
        )
        minimum, maximum = colour_range(values)
        output.with_suffix(output.suffix + ".json").write_text(json.dumps({
            "field": channel,
            # What the colours span, for the colour bar; the data go beyond it.
            "minimum": minimum,
            "maximum": maximum,
            "percentiles": list(COLOUR_PERCENTILES),
            "data_minimum": float(values.min()),
            "data_maximum": float(values.max()),
            "sampling": "nearest",
            "scale": "logarithmic" if logarithmic else "linear",
            # The grid this was drawn from (see surface_visualizations_current).
            "values_sha256": artifact.manifest.get("values_sha256"),
        }, indent=2, sort_keys=True), encoding="utf-8")
        outputs.append(output)
    return outputs
