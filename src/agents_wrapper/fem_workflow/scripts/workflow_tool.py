#!/usr/bin/env python
"""Deterministic geometry, selector, and case tools used by the monitor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

# Octree root cube side as a multiple of the geometry's largest extent, and the
# clip-bbox pad as a fraction of that extent.  Both come from the production
# examples (optimization/shape_opt_capture, optimization/hidden_phase_field),
# which put a 2.0-unit part in a 3.84-unit cube and clip at bounds +/- 0.3.
OCTREE_CUBE_PAD = 3.84 / 2.0
CLIP_BBOX_PAD = 0.3 / 2.0


def _octree_cube_pad(args) -> float:
    return float(getattr(args, "cube_pad", None) or OCTREE_CUBE_PAD)


_MESH_CACHE_LIMIT = 4
_mesh_cache: OrderedDict[tuple[str, int, int, int, int, int], object] = OrderedDict()
_BC_GEOMETRY_CACHE_LIMIT = 2
_bc_geometry_cache: OrderedDict[
    tuple[tuple[str, int, int, int, int, int], str], tuple
] = OrderedDict()


def _file_key(path: Path) -> tuple[str, int, int, int, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return (
        os.fspath(resolved), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


def _mesh(path: Path):
    import trimesh

    key = _file_key(path)
    loaded = _mesh_cache.get(key)
    if loaded is not None:
        _mesh_cache.move_to_end(key)
        return loaded
    loaded = trimesh.load(key[0], force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"not a nonempty triangle surface: {path}")
    _mesh_cache[key] = loaded
    while len(_mesh_cache) > _MESH_CACHE_LIMIT:
        _mesh_cache.popitem(last=False)
    return loaded


def _bc_geometry(path: Path, device: str):
    """Reusable selector inputs and derived geometry for resident previews."""
    import torch

    file_key = _file_key(path)
    key = (file_key, device)
    cached = _bc_geometry_cache.get(key)
    if cached is not None:
        _bc_geometry_cache.move_to_end(key)
        return cached
    mesh = _mesh(path)
    result = (
        mesh,
        torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device),
        torch.as_tensor(mesh.faces, dtype=torch.int64, device=device),
        np.asarray(mesh.area_faces),
        np.asarray(mesh.triangles),
        np.asarray(mesh.triangles_center),
        np.asarray(mesh.face_normals),
    )
    _bc_geometry_cache[key] = result
    while len(_bc_geometry_cache) > _BC_GEOMETRY_CACHE_LIMIT:
        _bc_geometry_cache.popitem(last=False)
    return result


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def stage_asset(args) -> list[str]:
    source, target = args.source.resolve(), args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".stl" and target.suffix.lower() != ".stl":
        # STL stores every triangle's own three corners and no connectivity,
        # and surfaces are read with process=False, so a closed part came in
        # as a soup of open triangles and failed the watertight check.  Staged
        # into a format that keeps shared vertices (the graph asks for GLB, the
        # one the viewer draws), with coincident corners welded; the triangles
        # and their order are unchanged.
        import trimesh

        mesh = trimesh.load(source, force="mesh", process=False)
        mesh.merge_vertices()
        mesh.export(target)
    else:
        shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = target.with_suffix(target.suffix + ".json")
    _write_json(metadata, {"source_name": source.name, "path": str(target), "sha256": digest})
    return [str(target), str(metadata)]


def inspect_surface(args) -> list[str]:
    mesh = _mesh(args.surface)
    components = mesh.split(only_watertight=False)
    value = {
        "surface": str(args.surface.resolve()),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": int(len(components)),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "extents": np.asarray(mesh.extents).tolist(),
        "centroid": np.asarray(mesh.centroid).tolist(),
        "area": float(mesh.area),
    }
    _write_json(args.output, value)
    if not value["watertight"]:
        raise ValueError("surface is not watertight; revise geometry before FEM")
    return [str(args.output.resolve())]


def scale_surface(args) -> list[str]:
    mesh = _mesh(args.surface).copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    geo_min = vertices.min(axis=0)
    geo_max = vertices.max(axis=0)
    if np.any(geo_max - geo_min < 1.0e-6):
        raise RuntimeError("degenerate geometry cannot be scaled")
    # Octree root cube side, in the source mesh's own units.  The leaf size is
    # cube / 2**depth, so sizing the cube from the geometry's largest extent is
    # what makes a given depth mean the same resolution here as it does in
    # optimization/shape_opt_capture and optimization/hidden_phase_field, which
    # wrap a 2.0-unit part in a 3.84-unit cube.  Sizing it from the generator's
    # fixed [-1, 1] domain instead left this path roughly twice as fine.
    cube_norm = _octree_cube_pad(args) * float((geo_max - geo_min).max())
    h_norm = cube_norm / (2 ** args.octree_depth)
    geo_min_np = np.asarray(
        [round(float(value) / h_norm) * h_norm for value in geo_min], dtype=float
    )
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis.lower()]
    geo_max_ref = round(float(geo_max[axis_idx]) / h_norm) * h_norm
    scale = float(args.extent) / (geo_max_ref - geo_min_np[axis_idx])
    # Pre-snap cube centre.  Centre it on the geometry rather than on the source
    # frame's origin, on the same lattice as geo_min_np, so the cube covers the
    # part whatever its placement.  run_forward's clip bbox re-snaps this to the
    # clipped lattice; this value is the fallback when no clip bbox is given.
    center_norm = geo_min_np + np.round(
        (0.5 * (geo_min + geo_max) - geo_min_np) / h_norm
    ) * h_norm
    mesh.vertices = (np.asarray(mesh.vertices) - geo_min_np) * scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    metadata = args.output.with_suffix(args.output.suffix + ".transform.json")
    _write_json(metadata, {
        "axis": args.axis, "physical_extent": args.extent, "units": args.units,
        "octree_depth": args.octree_depth, "h_norm": h_norm,
        "octree_cube_pad": _octree_cube_pad(args),
        "octree_cube_side": cube_norm * scale,
        "uniform_scale": scale, "geo_min": geo_min_np.tolist(),
        "mesh_center": ((center_norm - geo_min_np) * scale).tolist(),
        "h_target": h_norm * scale,
    })
    return [str(args.output.resolve()), str(metadata.resolve())]


def _prepare_rendering() -> None:
    import matplotlib
    matplotlib.use("Agg")


_VIEW_ELEVATION = np.radians(20.0)
_VIEW_COS = float(np.cos(_VIEW_ELEVATION))
_VIEW_SIN = float(np.sin(_VIEW_ELEVATION))

# (eye, screen-up, label), in the geometry's y-up world frame.  The four side
# cameras lean 20 degrees toward +y to retain depth cues; their up vectors are
# the projection of world +y onto the image plane.  Looking along +/-y makes
# world y parallel to the sightline, so those two axial views use +z as up.
_VIEWS = (
    ((_VIEW_COS, _VIEW_SIN, 0.0), (-_VIEW_SIN, _VIEW_COS, 0.0), "+x"),
    ((-_VIEW_COS, _VIEW_SIN, 0.0), (_VIEW_SIN, _VIEW_COS, 0.0), "-x"),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "+y"),
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), "-y"),
    ((0.0, _VIEW_SIN, _VIEW_COS), (0.0, _VIEW_COS, -_VIEW_SIN), "+z"),
    ((0.0, _VIEW_SIN, -_VIEW_COS), (0.0, _VIEW_COS, _VIEW_SIN), "-z"),
)


def _barycentric_lattice(rows: int) -> np.ndarray:
    """Barycentric weights on a regular triangular lattice, shape ``(P, 3)``."""
    i, j = np.meshgrid(np.arange(rows), np.arange(rows), indexing="ij")
    keep = (i + j) < rows
    a = (i[keep] + 1.0 / 3.0) / rows
    b = (j[keep] + 1.0 / 3.0) / rows
    return np.stack([a, b, 1.0 - a - b], axis=1)


# Lattice sizes in ~sqrt(2) steps, so rounding a facet up to the next size
# never more than doubles its sample count (a power-of-two ladder quadruples it).
_LATTICE_ROWS = np.array(
    [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024],
    dtype=np.int64,
)


def _raster_view(corners, scale, shaded, eye, right, up, width, height, visible):
    """Z-buffer one view into an ``(height, width, 3)`` float image.

    Matplotlib's ``Poly3DCollection`` emits one Agg path per facet per view and
    depth-sorts them in Python, which costs seconds for a six-view montage of a
    150k-facet surface.  Here each facet is sampled on a barycentric lattice
    sized to its own screen area, so the work tracks painted pixels rather than
    facet count, and the nearest sample wins each pixel.  Facets are bucketed by
    lattice size so the surface rasterizes in a handful of vectorized passes.
    """
    screen_x = (corners @ right) * scale + width / 2.0
    screen_y = height / 2.0 - (corners @ up) * scale
    depth_at_corner = corners @ eye

    area = 0.5 * np.abs(
        (screen_x[:, 1] - screen_x[:, 0]) * (screen_y[:, 2] - screen_y[:, 0])
        - (screen_x[:, 2] - screen_x[:, 0]) * (screen_y[:, 1] - screen_y[:, 0])
    )
    # A lattice of ``rows`` rows carries ``rows*(rows+1)/2`` samples; aim at ~2
    # per covered pixel, because at one per pixel a regular lattice slips
    # between pixel centres and freckles the surface with background.
    rows = np.ceil(np.sqrt(4.0 * np.maximum(area, 0.25))).astype(np.int64)
    buckets = _LATTICE_ROWS[
        np.clip(np.searchsorted(_LATTICE_ROWS, rows), 0, len(_LATTICE_ROWS) - 1)
    ]

    image = np.ones((height * width, 3), dtype=np.float32)
    pixels, depths, facets = [], [], []
    for size in np.unique(buckets[visible]):
        selected = visible[buckets[visible] == size]
        lattice = _barycentric_lattice(int(size))
        sample_ix = np.floor(screen_x[selected] @ lattice.T).astype(np.int64)
        sample_iy = np.floor(screen_y[selected] @ lattice.T).astype(np.int64)
        inside = (
            (sample_ix >= 0) & (sample_ix < width)
            & (sample_iy >= 0) & (sample_iy < height)
        ).ravel()
        if not inside.any():
            continue
        pixels.append((sample_iy * width + sample_ix).ravel()[inside])
        depths.append((depth_at_corner[selected] @ lattice.T).ravel()[inside])
        facets.append(np.repeat(selected, lattice.shape[0])[inside])
    if not pixels:
        return image.reshape(height, width, 3)

    pixel = np.concatenate(pixels)
    depth = np.concatenate(depths)
    facet = np.concatenate(facets)

    # Pack (pixel, depth toward the camera, facet) into one int64 so a single
    # np.sort resolves the z-buffer and carries the winning facet with it --
    # cheaper than lexsort over separate keys plus an argsort gather.
    facet_bits = max(1, int(len(corners) - 1).bit_length())
    depth_bits, depth_max = 20, (1 << 20) - 1
    low, high = depth.min(), depth.max()
    quantized = np.rint(
        (depth - low) / max(float(high - low), 1e-12) * depth_max
    ).astype(np.int64)
    key = (
        (pixel << (depth_bits + facet_bits))
        | ((depth_max - quantized) << facet_bits)
        | facet
    )
    key.sort()
    ordered_pixel = key >> (depth_bits + facet_bits)
    winner = np.empty(len(key), dtype=bool)
    winner[0] = True
    np.not_equal(ordered_pixel[1:], ordered_pixel[:-1], out=winner[1:])
    image[ordered_pixel[winner]] = shaded[key[winner] & ((1 << facet_bits) - 1)]
    return image.reshape(height, width, 3)


def _render(mesh, output: Path, colors=None, title="Surface", legend=None) -> None:
    import matplotlib.pyplot as plt

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    radius = float(np.linalg.norm(vertices - center, axis=1).max()) or 1.0
    width, height = 560, 520
    scale = min(width, height) * 0.46 / radius

    corners = (vertices - center)[faces]
    face_rgb = (
        np.asarray(colors, dtype=np.float64)[:, :3] if colors is not None
        else np.broadcast_to([0.68, 0.75, 0.84], (len(faces), 3)).copy()
    )
    edges = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = edges / np.maximum(np.linalg.norm(edges, axis=1, keepdims=True), 1e-12)
    # Back faces of a consistently wound surface are always overdrawn by the
    # z-buffer, so skipping them halves the samples.  Inconsistent winding would
    # turn that into holes, so fall back to rasterizing every facet.
    try:
        cull = bool(mesh.is_winding_consistent and mesh.is_watertight)
    except Exception:
        cull = False

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    for index, (eye_values, up_values, label) in enumerate(_VIEWS, 1):
        eye = np.asarray(eye_values, dtype=np.float64)
        up = np.asarray(up_values, dtype=np.float64)
        # _raster_view projects onto (right, up); this cross-product gives the
        # right-handed image basis for a vector pointing from object to camera.
        right = np.cross(up, eye)
        facing = normals @ eye
        visible = (
            np.flatnonzero(facing > 0.0) if cull
            else np.arange(len(faces), dtype=np.int64)
        )
        # Mild diffuse term so silhouettes and creases read, without shifting
        # the segment hues the API has to match against the legend.
        shaded = face_rgb * (0.72 + 0.28 * np.abs(facing))[:, None]
        ax = fig.add_subplot(2, 3, index)
        ax.imshow(
            _raster_view(corners, scale, shaded, eye, right, up, width, height, visible),
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(title)
    if legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=color, label=label) for label, color in legend]
        fig.legend(handles=handles, loc="lower center", ncol=min(10, len(handles)), fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the output and renamed over it: a render made while the workflow records
    # the session's files (graph's branch record skips dot-named files) is never seen half done.
    output = Path(output)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    try:
        fig.savefig(temporary, dpi=140)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    plt.close(fig)


def _export_colored_glb(mesh, face_colors, output: Path) -> None:
    """Export an interactive copy with the supplied per-face RGBA colors."""
    import trimesh

    rgba = np.asarray(face_colors)
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = np.clip(rgba * 255.0, 0, 255)
    rgba = rgba.astype(np.uint8)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    colored = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(len(triangles) * 3).reshape(-1, 3),
        process=False,
    )
    colored.visual.vertex_colors = np.repeat(rgba, 3, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    colored.export(output)


# Where a face's texture is read: its corners, edge midpoints and centroid.
_FACE_SAMPLES = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [1 / 3, 1 / 3, 1 / 3],
])


def _texture_face_colors(mesh):
    """Per-face RGBA in [0, 1] read from a UV texture, or None without one.

    Each face averages the texture at a few points inside it rather than at its
    corners alone: a coarse mesh under a detailed texture (a panel of wood
    grain on two triangles) would otherwise show only what its vertices land on.
    """
    from src.hunyuan3d_wrapper.surface import sample_texture, texture_of

    try:
        image = texture_of(mesh)
    except (AttributeError, RuntimeError):
        return None
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return None
    corners = np.asarray(uv, dtype=np.float64)[np.asarray(mesh.faces)]
    points = np.einsum("wk,fkd->fwd", _FACE_SAMPLES, corners).reshape(-1, 2)
    rgba = sample_texture(image, points).reshape(len(corners), len(_FACE_SAMPLES), 4)
    rgba = rgba.mean(axis=1) / 255.0
    factor = getattr(mesh.visual.material, "baseColorFactor", None)
    if factor is not None:
        factor = np.asarray(factor, dtype=np.float64)[:4]
        rgba[:, : len(factor)] *= factor / 255.0 if factor.max() > 1.0 else factor
    return np.clip(rgba, 0.0, 1.0)


def _vertex_face_colors(mesh):
    """Per-face RGBA in [0, 1] from the mesh's own colour, or None.

    A generator that keeps the object's colour (Hunyuan3D-2.1), or the paint
    step, stores it per vertex; an uploaded model usually carries a UV texture
    instead, which is read directly.  A mesh without either -- or with
    trimesh's uniform default -- renders in the usual neutral tone.
    """
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "kind", None) == "texture":
        return _texture_face_colors(mesh)
    if visual is None or getattr(visual, "kind", None) != "vertex":
        return None
    rgba = np.asarray(visual.vertex_colors, dtype=np.float64) / 255.0
    if len(rgba) != len(mesh.vertices) or np.ptp(rgba[:, :3], axis=0).max() == 0:
        return None
    return rgba[np.asarray(mesh.faces)].mean(axis=1)


def render_surface(args) -> list[str]:
    # Import Matplotlib before trimesh/scipy selects the system C++ runtime.
    _prepare_rendering()
    mesh = _mesh(args.surface)
    _render(mesh, args.output, colors=_vertex_face_colors(mesh))
    return [str(args.output.resolve())]


def plain_colour_surface(args) -> list[str]:
    """The surface with one flat vertex colour, unchanged otherwise: the stand-in the workflow
    writes where the painter crashed on the surface (graph._paint_surface), so a run that needs a
    coloured surface still reaches the materials and the solve, and says it was not painted."""
    import trimesh

    rgb = [int(value) for value in str(args.rgb).split(",")]
    if len(rgb) != 3 or not all(0 <= value <= 255 for value in rgb):
        raise ValueError(f"--rgb takes three integers 0-255, got {args.rgb!r}")
    loaded = trimesh.load(args.surface, force="mesh", process=False)
    mesh = trimesh.Trimesh(loaded.vertices, loaded.faces, process=False)
    if getattr(args, "segments_npz", None) is not None:
        colours = _part_vertex_colours(mesh, args, rgb)
    else:
        colours = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    return [str(args.output.resolve())]


def _part_vertex_colours(mesh, args, rgb) -> np.ndarray:
    """One flat colour per part of a segmentation (``--segments-npz``, one label per face): the
    part's colour in ``--part-rgb`` ({"<id>": [r, g, b]}), else its colour in ``--segments-json``
    (the colour the part montage shows it in), else ``--rgb``.  A vertex takes the colour of the
    part with the most face area around it, so the surface keeps its vertices and faces.

    The painter draws one reference and spreads it over every face, so it places no colour by part:
    in a held-out run (2026-10-10) seven paints never put the asked colours on the asked parts."""
    labels = np.load(args.segments_npz)["segment_ids"].astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(f"part labels cover {len(labels)} faces but the surface has {len(mesh.faces)}; "
                         "they belong to a different surface")
    metadata = json.loads(args.segments_json.read_text(encoding="utf-8")) if getattr(args, "segments_json", None) else {}
    palette = {}
    for segment in metadata.get("segments") or []:
        colour = str(segment.get("color") or "")
        if colour.startswith("#") and len(colour) == 7:
            palette[int(segment["id"])] = [int(colour[k:k + 2], 16) for k in (1, 3, 5)]
    for key, value in json.loads(getattr(args, "part_rgb", None) or "{}").items():
        palette[int(key)] = [int(c) for c in value]
    parts = np.unique(labels)
    column = {int(p): k for k, p in enumerate(parts)}
    weights = np.zeros((len(mesh.vertices), len(parts)))
    np.add.at(weights, (np.asarray(mesh.faces).ravel(), np.repeat([column[int(p)] for p in labels], 3)),
              np.repeat(np.asarray(mesh.area_faces), 3))
    vertex_part = parts[np.argmax(weights, axis=1)]
    colours = np.array([[*palette.get(int(p), rgb), 255] for p in vertex_part], dtype=np.uint8)
    if np.ptp(colours[:, :3], axis=0).max() == 0 and np.ptp(colours[0, :3]) == 0:
        colours[:, 2] = colours[:, 2] + 1 if colours[0, 2] < 255 else colours[:, 2] - 1   # not read as bare
    return colours


def _cad_face_segmentation(args) -> list[str]:
    """The CAD model's own faces, one segment per B-rep face."""
    from src.agents_wrapper.fem_workflow.segmentation_artifacts import write_segmentation

    if args.cad_faces is None:
        raise ValueError("CAD-face segmentation needs --cad-faces")
    mesh = _mesh(args.surface)
    with np.load(args.cad_faces) as saved:
        labels = saved["face_ids"].astype(np.int64)
        surface_types = [str(kind) for kind in saved["surface_types"]]
    if labels.shape[0] != len(mesh.faces):
        raise ValueError(
            f"CAD face labels cover {labels.shape[0]} triangles but the surface has "
            f"{len(mesh.faces)}; they belong to a different surface"
        )
    present = np.unique(labels)
    return write_segmentation(
        mesh, labels, method="cad_faces", num_clusters=len(present),
        output_npz=args.output_npz, output_json=args.output_json,
        output_glb=args.output_glb,
        segment_fields={int(face): {"cad_surface": surface_types[face]} for face in present},
    )


def segment_surface(args) -> list[str]:
    if getattr(args, "method", "shape_diameter") == "cad_faces":
        return _cad_face_segmentation(args)
    import torch
    from core.bc_surface_selectors import (
        _build_adj_pairs, _compute_face_attributes,
    )
    from core.bc_surface_segmentation import segment_surface as compute

    if not torch.cuda.is_available():
        raise RuntimeError("shape-diameter segmentation requires CUDA")
    if getattr(args, "method", "shape_diameter") != "shape_diameter":
        raise ValueError("workflow_tool handles only shape-diameter segmentation")
    mesh = _mesh(args.surface)
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device="cuda")
    centroids, normals, _ = _compute_face_attributes(vertices, faces)
    adjacency = _build_adj_pairs(faces)
    result = compute(
        vertices, faces, centroids, normals, adjacency,
        num_clusters=args.clusters, smoothing=args.smoothing,
        num_rays=args.rays, cone_angle_deg=args.cone_angle_deg,
    )
    segment_ids = result.segment_ids.detach().cpu().numpy()
    shape_diameter = result.shape_diameter.detach().cpu().numpy()
    from src.agents_wrapper.fem_workflow.segmentation_artifacts import write_segmentation
    return write_segmentation(
        mesh, segment_ids, method="shape_diameter", num_clusters=args.clusters,
        output_npz=args.output_npz, output_json=args.output_json,
        output_glb=args.output_glb, shape_diameter=shape_diameter,
        shape_diameter_range=result.shape_diameter_range,
        method_settings={"num_rays": args.rays, "smoothing": args.smoothing,
                         "cone_angle_deg": args.cone_angle_deg},
    )


def render_segmentation(args) -> list[str]:
    """Create the API-only multiview montage lazily from saved segment IDs."""
    _prepare_rendering()
    mesh = _mesh(args.surface)
    segment_ids = np.load(args.segments_npz)["segment_ids"]
    metadata = json.loads(args.segments_json.read_text(encoding="utf-8"))
    from matplotlib.colors import to_rgba

    colors = {int(item["id"]): to_rgba(item["color"]) for item in metadata["segments"]}
    palette = np.asarray([colors[int(segment_id)] for segment_id in segment_ids])
    legend = [(f"segment {item['id']}", item["color"]) for item in metadata["segments"]]
    title = {"partfield": "PartField Segments", "cad_faces": "CAD Face Segments"}.get(
        metadata.get("method"), "Shape Diameter Segments")
    _render(mesh, args.output, colors=palette, title=title, legend=legend)
    return [str(args.output.resolve())]


def render_part_montage(args) -> list[str]:
    """The surface's own colour above its numbered parts, from the same six cameras.

    A material is read off the colour and texture, and a part off the flat fill
    that hides them, so one picture carries both: a panel in the top half and
    the one below it show the same view.
    """
    import tempfile

    from PIL import Image

    _prepare_rendering()
    mesh = _mesh(args.surface)
    colors = _vertex_face_colors(mesh)
    if colors is None:
        raise ValueError("the surface has neither vertex colours nor a UV texture to read materials from")
    segment_ids = np.load(args.segments_npz)["segment_ids"]
    if len(segment_ids) != len(mesh.faces):
        raise ValueError(
            f"part labels cover {len(segment_ids)} faces but the surface has "
            f"{len(mesh.faces)}; they belong to a different surface"
        )
    metadata = json.loads(args.segments_json.read_text(encoding="utf-8"))
    from matplotlib.colors import to_rgba

    fill = {int(item["id"]): to_rgba(item["color"]) for item in metadata["segments"]}
    palette = np.asarray([fill[int(segment_id)] for segment_id in segment_ids])
    legend = [(f"part {item['id']}", item["color"]) for item in metadata["segments"]]
    with tempfile.TemporaryDirectory() as scratch:
        top, bottom = Path(scratch) / "colour.png", Path(scratch) / "parts.png"
        _render(mesh, top, colors=colors, title="Surface colour")
        _render(mesh, bottom, colors=palette, title="Parts (same views)", legend=legend)
        upper, lower = Image.open(top).convert("RGB"), Image.open(bottom).convert("RGB")
        montage = Image.new("RGB", (max(upper.width, lower.width), upper.height + lower.height), "white")
        montage.paste(upper, (0, 0))
        montage.paste(lower, (0, upper.height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    return [str(args.output.resolve())]


def build_part_field(args) -> list[str]:
    """Write the material field for one set of isotropic parameters per part.

    ``--materials-json`` holds the part stage's answer, ``{"parts": [{"id": 0,
    "youngs_modulus": ..., "poisson_ratio": ..., "density": ...}, ...]}``, in
    ``--unit-system``'s units, like a uniform material.
    """
    from src.material_parts import field_from_parts
    from src.vomp_wrapper.artifact import load_artifact, unit_scales
    from src.vomp_wrapper.visualize import write_surface_visualizations

    mesh = _mesh(args.surface)
    segment_ids = np.load(args.segments_npz)["segment_ids"]
    parts = json.loads(args.materials_json.read_text(encoding="utf-8"))["parts"]
    e_scale, rho_scale = unit_scales(args.unit_system)
    field_from_parts(
        mesh, segment_ids,
        {int(part["id"]): (float(part["youngs_modulus"]) / e_scale,
                           float(part["poisson_ratio"]),
                           float(part["density"]) / rho_scale) for part in parts},
        args.output, unit_system=args.unit_system,
        provenance={"parts": sorted(parts, key=lambda part: int(part["id"]))},
    )
    # The field's review previews are drawn here, beside it, as VoMP's are: the
    # field is rewritten in place when its values are edited, and previews left
    # to the page were those of the values it replaced.
    write_surface_visualizations(load_artifact(args.output), args.surface)
    return [str(args.output.resolve())]


def _selector(spec, preview_segment_ids=None):
    from core import bc_surface_selectors as selectors

    names = {
        "bounding_box": "BoundingBoxSelector", "sphere": "SphereSelector",
        "cylinder": "CylinderSelector", "half_space": "HalfSpaceSelector",
        "slab": "SlabSelector", "lowest": "LowestSelector", "highest": "HighestSelector",
        "percentile": "PercentileSelector", "furthest_from_center": "FurthestFromCenterSelector",
        "normal": "NormalSelector", "aligned_normal": "AlignedNormalSelector",
        "curvature": "CurvatureSelector", "flat_region": "FlatRegionSelector",
        "sharp_feature": "SharpFeatureSelector", "dihedral_angle": "DihedralAngleSelector",
    }
    kind, params = spec["type"], dict(spec.get("params", {}))
    children = [_selector(child, preview_segment_ids) for child in spec.get("children", [])]
    if kind in names:
        return getattr(selectors, names[kind])(**params)
    if kind == "and": return children[0] & children[1]
    if kind == "or": return children[0] | children[1]
    if kind == "subtract": return children[0] - children[1]
    if kind == "not": return ~children[0]
    if kind == "surface_segment":
        return selectors.SurfaceSegmentSelector(
            segmentation=preview_segment_ids, **params
        )
    if kind in ("shape_diameter_segment", "preview_segments"):
        # One selector, two ways in.  Naming segment ids reuses the segmentation
        # the user previewed instead of solving an identical one again, and it
        # selects exactly the partition they reviewed.
        return selectors.ShapeDiameterSegmentSelector(
            children[0] if children else None,
            segmentation=preview_segment_ids, **params,
        )
    if kind == "flood_fill": return selectors.FloodFillSelector(children[0], **params)
    if kind == "grow_selection": return selectors.GrowSelectionSelector(children[0], **params)
    if kind == "boundary_edge": return selectors.BoundaryEdgeSelector(children[0])
    if kind == "connected_component":
        return selectors.ConnectedComponentSelector(children[0] if children else None, **params)
    raise ValueError(f"unsupported selector: {kind}")


def _uses_saved_segments(spec) -> bool:
    """True when the selector names segments from the saved segmentation.

    Those ids index labels computed on the source surface, so such a selector has
    to be resolved against that surface.  The solver rebuilds a coarser marching-
    cubes surface with a different facet count, where the labels do not apply.
    ``shape_diameter_segment`` carrying ``segment_ids`` is the merged spelling of
    the former ``preview_segments``; both count.
    """
    if spec.get("type") in ("preview_segments", "surface_segment"):
        return True
    if (spec.get("type") == "shape_diameter_segment"
            and "segment_ids" in (spec.get("params") or {})):
        return True
    return any(_uses_saved_segments(child) for child in spec.get("children", []))


# Held regions are cool and loaded regions warm, so the two roles read apart at
# a glance while each region within a role stays its own colour.  Ordered for
# contrast against their neighbours rather than as a ramp: adjacent feet want to
# be told apart, not shaded.
BC_COLORS = {
    "dirichlet": (
        "#2659f2", "#00b8d4", "#6a3fd6", "#00897b", "#4fc3f7", "#3949ab",
    ),
    "neumann": (
        "#f00000", "#ff8f00", "#c2185b", "#ff5252", "#e65100", "#ad1457",
    ),
}
BC_ROLE = {"dirichlet": "held", "neumann": "load"}


def _child_face_counts(spec, preview_segment_ids, vertices, faces, limit: int = 12) -> str:
    """For a selector that selected nothing: each node of its tree under the root, resolved
    alone, with its face count ("and > slab(axis y, 0.02..0.05): 0 faces"), so the part that
    empties the intersection is named.  Empty for a leaf selector (nothing to list)."""
    rows: list[str] = []

    def label(node):
        params = node.get("params") or {}
        short = ", ".join(f"{k} {v}" for k, v in list(params.items())[:3])
        return f"{node.get('type')}({short})" if short else str(node.get("type"))

    def walk(node, path):
        for child in node.get("children") or []:
            if len(rows) >= limit:
                return
            name = f"{path} > {label(child)}" if path else label(child)
            try:
                selector = _selector(child, preview_segment_ids)
                selector.resolve(vertices, faces)
                count = int(selector._selected_face_mask.detach().cpu().numpy().sum())
                rows.append(f"{name}: {count} faces")
            except Exception as exc:  # a diagnostic, never the error itself
                rows.append(f"{name}: not resolved ({type(exc).__name__}: {str(exc)[:80]})")
            walk(child, name)

    walk(spec, "")
    return "; ".join(rows)


def resolve_bcs(args) -> list[str]:
    import torch
    from src.agents_wrapper.fem_workflow.models import BCProposal

    proposal = BCProposal.model_validate_json(args.bc_json.read_text(encoding="utf-8"))
    segments_path = getattr(args, "segments_npz", None)
    preview_segment_ids = (
        np.load(segments_path)["segment_ids"] if segments_path is not None else None
    )
    for output in (args.output_npz, args.output_json, args.output_glb):
        output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (mesh, vertices, faces, face_areas, triangles,
     face_centroids, face_normals) = _bc_geometry(args.surface, device)
    records, masks = [], []
    total_area = float(face_areas.sum())
    for category, entries in (("dirichlet", proposal.dirichlet), ("neumann", proposal.neumann)):
        for entry in entries:
            selector = _selector(entry.selector.model_dump(), preview_segment_ids)
            selector.resolve(vertices, faces)
            mask = selector._selected_face_mask.detach().cpu().numpy()
            if not mask.any():
                counts = _child_face_counts(entry.selector.model_dump(), preview_segment_ids,
                                            vertices, faces)
                raise ValueError(f"BC {entry.name!r} selected no faces"
                                 + (f"; each part of its selector alone selects: {counts}" if counts else ""))
            key = f"{category}_{len(records)}"
            masks.append((key, mask))
            selected_area = face_areas[mask]
            area = float(selected_area.sum())
            points = triangles[mask].reshape(-1, 3)
            # The area a traction on this region actually loads: the solver
            # weights each point by max(0, n . n_avg) against the region's one
            # average normal, so a force spread over it must be divided by this.
            normals = face_normals[mask]
            average = (normals * selected_area[:, None]).sum(0)
            average = average / max(float(np.linalg.norm(average)), 1.0e-12)
            records.append({
                "key": key, "category": category, "name": entry.name,
                "faces": int(mask.sum()), "area": area,
                "loaded_area": float((selected_area * np.clip(normals @ average, 0.0, None)).sum()),
                "area_fraction": area / max(total_area, 1.0e-12),
                "bounds": [points.min(0).tolist(), points.max(0).tolist()],
                "centroid": np.average(
                    face_centroids[mask], axis=0, weights=selected_area
                ).tolist(),
                "mean_normal": np.average(
                    face_normals[mask], axis=0, weights=selected_area
                ).tolist(),
                "selector": entry.selector.model_dump(mode="json"),
            })
    np.savez_compressed(args.output_npz, **dict(masks))
    from matplotlib.colors import to_rgba

    # One colour per region, not one per category: with every support the same
    # blue there was no way to see which of four feet a selector had actually
    # caught, or that two regions had landed on the same face.  The families
    # stay apart -- held is cool, loaded is warm -- so the distinction that
    # matters at a glance survives being able to tell the regions apart.
    legend = []
    used = {"dirichlet": 0, "neumann": 0}
    painted = np.zeros(len(mesh.faces), dtype=bool)
    overlapped = np.zeros(len(mesh.faces), dtype=bool)
    face_colors = np.tile(np.array([0.72, 0.75, 0.78, 1.0]), (len(mesh.faces), 1))
    for record, (_, mask) in zip(records, masks):
        family = record["category"]
        swatch = BC_COLORS[family][used[family] % len(BC_COLORS[family])]
        used[family] += 1
        record["color"] = swatch
        legend.append({
            "label": f"{record['name']} · {BC_ROLE[family]}",
            "color": swatch,
        })
        overlapped |= mask & painted
        face_colors[mask] = to_rgba(swatch)
        painted |= mask
    if overlapped.any():
        # Last, so it is not itself overwritten by a region drawn after it.
        face_colors[overlapped] = np.array([0.6, 0.1, 0.7, 1.0])
        legend.append({"label": "Overlapping regions", "color": "#991ab3"})
    _write_json(args.output_json, {"regions": records, "legend": legend})
    _export_colored_glb(mesh, face_colors, args.output_glb)
    return [str(p.resolve()) for p in (
        args.output_npz, args.output_json, args.output_glb
    )]


def build_case(args) -> list[str]:
    from src.agents_wrapper.fem_workflow.models import (
        BCProposal, MaterialParameters, ShapeOptSettings,
    )

    proposal = BCProposal.model_validate_json(args.bc_json.read_text(encoding="utf-8"))
    material = MaterialParameters.model_validate_json(args.material_json.read_text(encoding="utf-8"))
    settings = ShapeOptSettings.model_validate_json(args.settings_json.read_text(encoding="utf-8"))
    transform = json.loads(args.transform_json.read_text(encoding="utf-8"))
    if int(transform["octree_depth"]) != args.octree_depth:
        raise ValueError("case octree depth must match the geometry transform")
    loads = {
        "body_force": list(material.body_force),
        "acceleration": list(getattr(args, "acceleration", (0.0, 0.0, 0.0))),
    }
    # The graph writes load state into material.json for compatibility with old
    # tools; an optional sidecar carries the new acceleration vector.
    loads_json = getattr(args, "loads_json", None)
    if loads_json is not None and Path(loads_json).is_file():
        loads.update(json.loads(Path(loads_json).read_text(encoding="utf-8")))
    case = {"version": 2, "surface": str(args.surface.resolve()),
            "output": str(args.solve_output.resolve()),
            "h_target": float(transform["h_target"]),
            "mesh_center": transform["mesh_center"],
            "geometry_transform": transform,
            "octree_depth": args.octree_depth, "material": material.model_dump(mode="json"),
            "loads": loads,
            "shape_opt": settings.model_dump(mode="json"),
            "boundary_conditions": proposal.model_dump(mode="json")}
    segments_path = getattr(args, "segments_npz", None)
    if segments_path is not None:
        case["segments_npz"] = str(segments_path.resolve())
    _write_json(args.output, case)
    return [str(args.output.resolve())]


# Voigt order as compute_strain_tensor/compute_stress_tensor emit it.
TENSOR_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
# Component names by field width: a 3-wide field is a vector, a 6-wide one is a
# symmetric tensor in Voigt order.  Naming them by width rather than assuming
# Voigt keeps a vector's x from being reported as xx.
COMPONENT_NAMES = {3: ("x", "y", "z"), 6: TENSOR_COMPONENTS}
VON_MISES = "von_mises"


def _von_mises(voigt: "np.ndarray", field: str) -> "np.ndarray":
    """Equivalent scalar of a symmetric tensor held in Voigt order.

    Stress takes sqrt(3/2 s:s), strain the energy-conjugate sqrt(2/3 e:e), both
    over the deviatoric part.  The Voigt shear slots hold tensor components --
    half the engineering shear, as compute_strain_tensor writes them -- so the
    off-diagonals enter the double contraction doubled, not sextupled.
    """
    normal, shear = voigt[:, :3], voigt[:, 3:]
    deviatoric = normal - normal.mean(axis=1, keepdims=True)
    contraction = (deviatoric ** 2).sum(axis=1) + 2.0 * (shear ** 2).sum(axis=1)
    return np.sqrt((1.5 if field == "stress" else 2.0 / 3.0) * contraction)


def _read_surface(vtu_path: Path, field: str, component: str | None):
    """One surface VTU as points, triangles, displacement, and one scalar.

    ``component`` is a Voigt name for a tensor field, or None for a vector's
    magnitude.  Shared with the range reference below, so the scalar a step is
    coloured by and the scalar a range is taken from are the same thing.
    """
    import base64
    import xml.etree.ElementTree as ET

    root = ET.parse(vtu_path).getroot()
    dtype_names = {
        "Float32": "<f4", "Float64": "<f8", "Int32": "<i4",
        "Int64": "<i8", "UInt8": "u1", "UInt32": "<u4", "UInt64": "<u8",
    }

    def values(element):
        raw = base64.b64decode((element.text or "").strip())
        size = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        return np.frombuffer(raw[8:8 + size], dtype=dtype_names[element.attrib["type"]])

    points_element = root.find(".//Points/DataArray")
    field_element = root.find(f".//PointData/DataArray[@Name='{field}']")
    connectivity_element = root.find(".//Cells/DataArray[@Name='connectivity']")
    offsets_element = root.find(".//Cells/DataArray[@Name='offsets']")
    if any(item is None for item in (
        points_element, field_element, connectivity_element, offsets_element
    )):
        raise ValueError(f"surface VTU lacks points, triangles, or {field!r} data")
    points = values(points_element).reshape(-1, 3)
    displacement_element = root.find(".//PointData/DataArray[@Name='displacement']")
    displacement = (
        values(displacement_element).reshape(-1, 3)
        if displacement_element is not None else np.zeros_like(points)
    )
    width = int(field_element.attrib.get("NumberOfComponents", 1))
    sampled = values(field_element).reshape(-1, width)
    offsets = values(offsets_element)
    if len(offsets) == 0 or not np.all(np.diff(np.r_[0, offsets]) == 3):
        raise ValueError("surface VTU contains non-triangular cells")
    faces = values(connectivity_element).reshape(-1, 3)
    names = COMPONENT_NAMES.get(width, ())
    if component is None:
        magnitude = np.linalg.norm(sampled, axis=1)
    elif component == VON_MISES:
        if width != 6:
            raise ValueError(f"{field!r} is not a tensor; it has no von Mises scalar")
        magnitude = _von_mises(sampled, field)
    else:
        if component not in names:
            raise ValueError(
                f"{field!r} has no component {component!r}; "
                f"available: {', '.join(names) or 'none'}"
            )
        magnitude = sampled[:, names.index(component)]
    return points, faces, displacement, magnitude


def _export_surface_field(
    vtu_path: Path, output: Path, field: str = "displacement",
    component: str | None = None, deform_scale: float = 0.0,
    value_min: float | None = None, value_max: float | None = None,
    yield_strength: float | None = None,
) -> None:
    """Colour one surface field, or one component of it, into a viewport asset.

    ``component`` is a Voigt name for a tensor field, or None for a vector's
    magnitude.  Each combination is a separate small asset, built when it is
    first asked for rather than up front: a solve has fourteen of them and a
    session normally looks at one or two.

    ``deform_scale`` moves each vertex along its own displacement before the
    colours are applied, so the shape shown is the deformed one.  Real
    deflections are usually far too small to see at true scale, and the colour
    map alone says how much a surface moved but not which way.  The colours are
    unaffected: they are the field's values, not the positions'.

    ``value_min`` and ``value_max`` fix the range the colours are stretched
    over, instead of taking it from this surface alone.  A load ramp needs that:
    left to itself every step paints its own largest value the same red, so a
    ramp that triples its load looks identical at every step while one legend
    claims to speak for all of them.  Given the range of the fully loaded step,
    the earlier ones come out proportionally cooler, which is what the step
    slider is for.
    """
    import trimesh

    safety = component == "safety_factor"
    if safety and (field != "stress" or yield_strength is None or not np.isfinite(yield_strength) or yield_strength <= 0):
        raise ValueError("Safety factor requires stress and positive finite yield strength")
    points, faces, displacement, magnitude = _read_surface(vtu_path, field, VON_MISES if safety else component)
    hover_values = magnitude.copy()
    if safety:
        if not np.all(np.isfinite(magnitude)):
            raise ValueError("Non-finite stress cannot produce a safety factor")
        magnitude = np.divide(yield_strength, magnitude, out=np.full_like(magnitude, np.inf), where=magnitude > 0)
        value_min, value_max = 0.0, 5.0
    largest_displacement = float(np.linalg.norm(displacement, axis=1).max()) if len(
        displacement
    ) else 0.0
    if deform_scale:
        points = points + displacement * float(deform_scale)
    observed_min, observed_max = float(magnitude.min()), float(magnitude.max())
    value_min = observed_min if value_min is None else float(value_min)
    value_max = observed_max if value_max is None else float(value_max)
    span = max(value_max - value_min, np.finfo(np.float64).eps)
    scaled = np.clip((magnitude - value_min) / span, 0.0, 1.0)
    anchors = np.asarray([
        [35, 65, 180], [30, 175, 210], [55, 190, 95],
        [245, 205, 55], [205, 35, 45],
    ], dtype=np.float64)
    if safety:
        anchors = np.asarray([[205, 35, 45], [205, 35, 45], [245, 205, 55], [55, 190, 95], [55, 190, 95]], dtype=np.float64)
        positions = np.interp(np.clip(magnitude, 0, 5), [0, 0.95, 1, 2, 5], [0, 1, 2, 3, 4])
    else:
        positions = scaled * (len(anchors) - 1)
    lower = np.minimum(positions.astype(int), len(anchors) - 2)
    blend = (positions - lower)[:, None]
    rgb = anchors[lower] * (1.0 - blend) + anchors[lower + 1] * blend
    rgba = np.column_stack((rgb.astype(np.uint8), np.full(len(rgb), 255, dtype=np.uint8)))
    contour = trimesh.Trimesh(
        vertices=points, faces=faces, vertex_colors=rgba, process=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    contour.export(output)
    output.with_suffix(output.suffix + ".values.bin").write_bytes(
        np.ascontiguousarray(hover_values, dtype="<f8").tobytes()
    )
    # The same displacements, in the vertex order the asset carries, so the
    # viewer can deform the surface itself instead of asking for a new asset
    # every time the scale changes.  trimesh keeps the order it is handed when
    # `process=False`, which is what makes the two line up.
    if largest_displacement > 0.0:
        output.with_suffix(output.suffix + ".disp.bin").write_bytes(
            np.ascontiguousarray(displacement, dtype="<f4").tobytes()
        )
    _write_json(output.with_suffix(output.suffix + ".json"), {
        "field": field if component else f"{field}_magnitude",
        "component": component,
        "minimum": value_min, "maximum": value_max,
        # What this surface actually holds, which a shared range hides.  A step
        # of a ramp is coloured against the whole ramp, so its own extremes are
        # worth keeping even though the legend speaks for the range above.
        "observed_minimum": observed_min if np.isfinite(observed_min) else None,
        "observed_maximum": observed_max if np.isfinite(observed_max) else None,
        **({"yield_strength": yield_strength, "display_cap": 5.0,
            "surface_sample_fraction_below_one": float(np.mean(magnitude < 1))} if safety else {}),
        # What the viewer sizes its deformation slider from, recorded beside
        # every asset so it does not depend on which one was looked at first.
        "max_displacement": largest_displacement,
        "deform_scale": float(deform_scale),
        # The surface this was coloured from.  An optimization rewrites its
        # displacement asset while it runs, and a file's age cannot say which
        # iteration it holds once an export and the next save race each other.
        "source": str(Path(vtu_path).resolve()),
    })


def _export_displacement_contour(vtu_path: Path, output: Path) -> None:
    """The one asset built eagerly, because every solve is looked at."""
    _export_surface_field(vtu_path, output, "displacement", None)


# runtime_files.OPTIMIZATION_STATUS, spelled here rather than imported: every
# workflow import is deferred in this script, since the package's __init__
# builds the graph, and the tools start on every call.
OPTIMIZATION_STATUS = "optimization_status.json"
# Where a failed optimization's own last outputs are put, inside its run
# directory: kept for inspection, out of the folders the viewer reads.
FAILED_ITERATION_DIR = "failed_iteration"
_STAMPED_ASSET = re.compile(r"_s(\d+)(?:_d[0-9p.eE+-]+)?\.glb(?:\.json|\.disp\.bin)?$")


class _OptimizationView:
    """Keeps what the viewer shows of a shape optimization on an iteration that solved.

    Two things a forward solve gets for free and an optimization did not.  The
    displacement asset (``displacement_magnitude.glb`` with its ``.json`` range
    and ``.disp.bin`` deformation sidecars) was built only once the whole loop
    had returned, so while it ran the viewer had no displacement entry and no
    deformation slider, and a run that failed never got one.  And the loop's
    failure path saves the state it failed in (``AbstractField.optimize``) as
    the newest surface, so the viewer, which shows the newest surface, moved
    on to an iteration that never solved -- a NaN or unconverged displacement
    on a shape the next step had already changed.

    So the displacement asset is refreshed after every save, from the newest
    surface the writer has finished (saves are asynchronous, so it can trail by
    one), and a failure sets aside every output from the failed iteration on,
    rebuilds the asset from the last iteration that solved, and records which
    iteration is shown and which one failed in ``optimization_status.json``
    beside the run's log.
    """

    def __init__(self, output: Path, run: Path, save_every: int) -> None:
        self.output = output
        # The solver's own run directory (AbstractField.path, where its log is).
        self.run = Path(run)
        self.save_every = int(save_every)
        # The viewer assets already in solve/ as this run started, by modification
        # time: one this run wrote or rewrote is told from them exactly, which a
        # comparison with the clock is not (file times trail time.time() by a tick).
        self.before = self._assets()
        self.exported: Path | None = None

    def _assets(self) -> dict[Path, int]:
        found = {}
        for path in self.output.glob("*.glb*"):
            try:
                found[path] = path.stat().st_mtime_ns
            except OSError:
                pass
        return found

    def _written_by_this_run(self, path: Path) -> bool:
        try:
            return self.before.get(path) != path.stat().st_mtime_ns
        except OSError:
            return False

    def run_dir(self) -> Path | None:
        return self.run if self.run.is_dir() else None

    def surfaces(self) -> list[Path]:
        # surface_tNNNN is the writer's save counter, not the iteration: save i
        # is iteration i * save_every, and the failure save is the last one.
        return sorted(self.run.glob("vtu_surface/surface_t*.vtu"))

    def iteration(self, index: int) -> int:
        return index * max(self.save_every, 1)

    def _export(self, surface: Path) -> None:
        _export_displacement_contour(surface, self.output / "displacement_magnitude.glb")
        self.exported = surface

    def refresh(self) -> None:
        """Bring the displacement asset up to the newest finished surface."""
        from src.agents_wrapper.fem_workflow.assets import fully_written

        finished = [path for path in self.surfaces() if fully_written(path)]
        if finished and finished[-1] != self.exported:
            self._export(finished[-1])

    def _status(self, record: dict) -> dict:
        run = self.run_dir()
        if run is not None:
            _write_json(run / OPTIMIZATION_STATUS, record)
        return record

    def complete(self, extra: dict | None = None) -> dict:
        """Show the last iteration; ``extra`` (the gd_lr_auto choices) is recorded too."""
        surfaces = self.surfaces()
        if surfaces:
            self._export(surfaces[-1])
        return self._status({
            "status": "complete",
            "shown_iteration": self.iteration(len(surfaces) - 1) if surfaces else None,
            "shown_surface": surfaces[-1].name if surfaces else None,
            **(extra or {}),
        })

    def fail(self, problem, error: BaseException, extra: dict | None = None) -> dict | None:
        """Set the failed iteration's outputs aside and show the last one that solved.

        None when the loop never started (a failed load ramp before it), which
        leaves nothing of an optimization to show.  ``extra`` is recorded with
        the status (run_forward's step sizes against epsilon).
        """
        failed = getattr(problem, "_optimize_iteration", None)
        if failed is None:
            return None
        try:
            # Flush the writers: the failure save is queued asynchronously, and
            # a failure outside the solve skips the loop's own finalize.
            # Finalizing twice only rewrites the PVD collections.
            problem.finalize()
        except Exception:
            pass
        run = self.run_dir()
        surfaces = self.surfaces()
        # Iterations before the failed one that were saved: 0, N, 2N, ... < failed.
        good = -(-int(failed) // self.save_every) if self.save_every > 0 else 0
        good = min(good, len(surfaces))
        while good > 0 and not _finite_displacement(surfaces[good - 1]):
            # Solved without raising but holding non-finite values: no better.
            good -= 1
        set_aside = [path.name for path in surfaces[good:]]
        if run is not None:
            for index in range(good, len(surfaces)):
                _set_aside_save(run, index)
            _drop_missing_from_collections(run)
        # Assets the viewer built from those surfaces while the run was live.
        for asset in self.output.glob("*.glb*"):
            match = _STAMPED_ASSET.search(asset.name)
            if match and int(match.group(1)) >= good and self._written_by_this_run(asset):
                asset.unlink(missing_ok=True)
        contour = self.output / "displacement_magnitude.glb"
        if good:
            self._export(surfaces[good - 1])
        else:
            for path in self.output.glob("displacement_magnitude.glb*"):
                if self._written_by_this_run(path):
                    path.unlink(missing_ok=True)
        return self._status({
            "status": "failed",
            "failed_iteration": int(failed),
            # Whether the failed iteration's own solve raised, or something
            # after it (the design update, a remesh, the BC stitch) did.
            "failed_in_solve": getattr(problem, "_optimize_failed_iteration", None) is not None,
            "shown_iteration": self.iteration(good - 1) if good else None,
            "shown_surface": surfaces[good - 1].name if good else None,
            "set_aside": set_aside,
            "displacement_asset": str(contour) if good else None,
            "error": f"{type(error).__name__}: {error}"[:2000],
            **(extra or {}),
        })


def _finite_displacement(surface: Path) -> bool:
    try:
        _, _, displacement, magnitude = _read_surface(surface, "displacement", None)
    except Exception:
        return False
    return bool(np.isfinite(displacement).all() and np.isfinite(magnitude).all())


def _set_aside_save(run: Path, index: int) -> None:
    """Move save ``index``'s files (surface, GLB, volume, SDF) under FAILED_ITERATION_DIR."""
    for path in list(run.glob(f"*/*_t{index:04d}.*")):
        if path.parent.name == FAILED_ITERATION_DIR or not path.is_file():
            continue
        target = run / FAILED_ITERATION_DIR / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


def _drop_missing_from_collections(run: Path) -> None:
    """Rewrite the run's PVD collections without the saves that were set aside."""
    import xml.etree.ElementTree as ET

    for pvd in run.glob("*.pvd"):
        try:
            tree = ET.parse(pvd)
        except (ET.ParseError, OSError):
            continue
        collection = tree.getroot().find("Collection")
        if collection is None:
            continue
        missing = [item for item in collection.findall("DataSet")
                   if not (run / item.get("file", "")).is_file()]
        if not missing:
            continue
        for item in missing:
            collection.remove(item)
        tree.write(pvd, encoding="unicode", xml_declaration=True)


def export_surface_field(args) -> list[str]:
    """Colour one surface, optionally against another surface's range.

    `--value-range-from` names the surface whose extremes the colours are
    stretched over, which is how every step of a load ramp is drawn to one
    scale.  Naming this surface itself, or nothing at all, means the usual
    thing: take the range from what is being coloured.
    """
    component = args.component or None
    reference = getattr(args, "value_range_from", None) or None
    span = (None, None)
    if component != "safety_factor" and reference is not None and Path(reference).resolve() != args.vtu.resolve():
        _, _, _, magnitude = _read_surface(Path(reference), args.field, component)
        span = (float(magnitude.min()), float(magnitude.max()))
    _export_surface_field(
        args.vtu, args.output, args.field, component,
        float(getattr(args, "deform_scale", 0.0) or 0.0), *span,
        yield_strength=getattr(args, "yield_strength", None),
    )
    return [str(args.output.resolve())]


def _bc_preservation(settings) -> dict:
    """BC region preservation, asked for only by a run that updates the design.

    Preservation carries the boundary regions across a design update: the
    frozen background mask is read by the reinit after a remesh, the zip
    patch by the stitch beside it.  A forward run performs no design
    update and so reaches neither, while still paying to build both -- and
    "zip", the alternative to the card's default "mask", is the costly one:
    it cuts the exact BC patch from the initial surface and validates its
    boundary loops, raising outright on a geometry whose loops it cannot
    resolve.  A forward solve has no business failing there.
    """
    if settings.run_mode == "shape_optimization":
        return {"sdf_bc_fix_method": settings.sdf_bc_fix_method}
    return {"preserve_bc_regions": False}


_LINEAR_RESIDUAL = "||b - Kx||/||b||="
SINGULAR_NOTE = (
    "; ||b - Kx||/||b|| of 1 or more on the unscaled system: the linear solve found nothing "
    "better than zero displacement, the mark of a singular or nearly singular system -- a "
    "rigid-body mode the holds leave (a hold of some components only, a piece no hold "
    "reaches, a bulky part hanging from the holds by members a few leaves thick) or a part "
    "far softer or nearly incompressible against its neighbours; a scaling or a penalty cap "
    "does not change that: check the supports, the components and the parts first"
)
SCALED_NOTE = (
    "; this is the residual of the Jacobi-scaled system (scale_system_to_unit_diagonal=True), "
    "which can exceed 1 where the unscaled system's is far below it: remove the scaling "
    "first, and read a residual of 1 or more as singular only without it"
)


def _singular_message(text: str, scaled: bool = False) -> str | None:
    """A linear-solver failure at ||b - Kx||/||b|| >= 1, reworded for what it is.

    The solver's own message names scale_system_to_unit_diagonal and beta_max_over_e for
    any stall.  Unscaled, a residual of 1 or more marks a singular or nearly singular
    system, which neither changes: that advice is replaced by SINGULAR_NOTE.  Scaled, the
    residual is the scaled system's, which in logged sessions read 2 to 6e3 where the same
    case unscaled read 1e-4 to 5e-2: SCALED_NOTE says to remove the scaling.  The solver's
    "it stalled ..." clause is kept either way: the operator's runner takes a stall's
    fallback on it (a fallback has converged at 1.05)."""
    if ("Linear solver failed" not in text or _LINEAR_RESIDUAL not in text
            or " in float32" in text):
        # a float32 solve below the strain floor stalls above 1 too (graph.py,
        # FLOAT32_RESIDUAL_FACTOR): its own message names use_mixed_precision
        return None
    try:
        value = float(text.split(_LINEAR_RESIDUAL, 1)[1].split(")", 1)[0])
    except ValueError:
        return None
    if value < 1.0:
        return None
    return text.split(" -- try", 1)[0] + (SCALED_NOTE if scaled else SINGULAR_NOTE)


# Appended to a JAX-AMG linear failure (not a singular one: no backend solves that).  It
# names the settings card's other choice and avoids the phrases the operator's runner
# reads ("-- try ", "stalled", "ran out of iterations"), so its triage is unchanged.
PYAMGX_NOTE = ('. Also worth a retry: the pyamgx linear solver (settings: linear_solver '
               '"pyamgx"), which keeps its last iterate and lets Newton\'s residual decide')


def _pyamgx_advice(text: str, linear_solver: str) -> str | None:
    """``text`` with PYAMGX_NOTE, for a JAX-AMG failure of a jaxamg solve; else None."""
    if (linear_solver != "jaxamg" or "JAX-AMG solve failed" not in text
            or SINGULAR_NOTE in text or PYAMGX_NOTE in text):
        return None
    return text + PYAMGX_NOTE


def _gd_lr_auto_record(problem, settings) -> dict | None:
    """The learning rates gd_lr_auto chose (core ``_gd_lr_auto``), for the status file.

    One entry per update under ``steps`` (iteration, max|dJ/ds|, gd_lr, whether
    it was sized or kept), the latest at the top level (``gd_lr_effective``).
    None when the option is off, or on but the loop failed before its first
    update; a JSON-ready copy otherwise.
    """
    if not settings.gd_lr_auto:
        return None
    chosen = getattr(problem, "_gd_lr_auto", None)
    if not isinstance(chosen, dict):
        return None
    return json.loads(json.dumps(chosen, default=float))


def _solve_with_diagnosis(solve, scaled: bool = False, linear_solver: str = "jaxamg") -> None:
    """Run ``solve``; a linear-solver failure at a residual of 1 or more is re-raised with
    _singular_message's wording (the monitor and the operator read solve_error from it),
    and a JAX-AMG one of a jaxamg solve also recommends pyamgx (_pyamgx_advice)."""
    try:
        solve()
    except RuntimeError as exc:
        message = _singular_message(str(exc), scaled)
        advised = _pyamgx_advice(message or str(exc), linear_solver)
        if advised is not None:
            message = advised
        if message is None:
            raise
        raise RuntimeError(message) from exc


def assembled_backend(linear_solver: str) -> str:
    """The solver's ``cg_backend_assembled`` for the settings card's choice.

    "jaxamg" maps to "amgx", the backend this workflow always pinned: JAX-AMG,
    or pyamgx only when JAX-AMG cannot be imported at all (core's
    ``resolve_amgx_backend``; a JAX-AMG solve that fails is not retried with
    pyamgx).  "pyamgx" selects pyamgx directly.  JAX-AMG and pyamgx cannot
    reliably initialise AMGX one after the other in one process (the solver's
    tests run them in two), so a pyamgx solve must start in a process where
    JAX-AMG's AMGX was never loaded.  Every solve gets one: run_forward is
    never served by the resident tools worker, and the spare solver process
    serves exactly one solve and imports neither library while warming
    (tool_server.warm).  ``_require_fresh_amgx`` checks it rather than trusts it.
    """
    return {"jaxamg": "amgx", "pyamgx": "pyamgx"}[linear_solver]


# The modules JAX-AMG's native AMGX extension loads as (``from jaxamg._ext
# import _amgx`` registers ``jaxamg._ext`` and ``jaxamg._amgx``; checked in
# mechfoundry-warp117 after a JAX-AMG solve).
_JAXAMG_NATIVE = ("jaxamg._ext", "jaxamg._amgx", "jaxamg._ext._amgx")


def _require_fresh_amgx(backend: str) -> None:
    """Refuse a pyamgx solve in a process whose AMGX JAX-AMG already started."""
    if backend == "pyamgx" and any(name in sys.modules for name in _JAXAMG_NATIVE):
        raise RuntimeError(
            "pyamgx was selected, but this process already loaded JAX-AMG's AMGX, "
            "and the two cannot share a process; run the solve in a fresh process"
        )


def run_forward(args) -> list[str]:
    import torch

    from core.nitsche_field import bc_match_tolerance
    from core.shape_opt import ShapeOptTrimmed
    from src.agents_wrapper.fem_workflow.models import (
        BCProposal, FIELD_MODES, MaterialParameters, ShapeOptSettings,
        optimization_failure_advice, optimization_step_record,
    )

    case = json.loads(args.case.read_text(encoding="utf-8"))
    mesh = _mesh(Path(case["surface"]))
    material = MaterialParameters.model_validate(case["material"])
    settings = ShapeOptSettings.model_validate(case["shape_opt"])
    if material.mode in FIELD_MODES and settings.run_mode != "forward":
        raise ValueError("spatial material fields currently support forward solves only")
    loads = case.get("loads") or {
        "body_force": material.body_force, "acceleration": (0.0, 0.0, 0.0)
    }
    body_force = tuple(float(value) for value in loads.get("body_force", (0.0, 0.0, 0.0)))
    acceleration = tuple(float(value) for value in loads.get("acceleration", (0.0, 0.0, 0.0)))
    if material.mode == "uniform":
        if any(acceleration) and material.density is None:
            raise ValueError("uniform density is required when acceleration is nonzero")
        density = float(material.density or 0.0)
        effective_body_force = tuple(
            body_force[i] + density * acceleration[i] for i in range(3)
        )
    else:
        effective_body_force = body_force
        artifact_grid = np.load(
            Path(material.artifact) / "material_grid.npy", mmap_mode="r",
            allow_pickle=False,
        )
        representative_E = float(np.median(artifact_grid[..., 0]))
        representative_nu = float(np.median(artifact_grid[..., 1]))
    analysis_h_target = float(case["h_target"]) * 2.0 ** (
        int(case["octree_depth"]) - settings.remesh_octree_depth
    )
    phase_field_epsilon = settings.phase_field_epsilon_factor * analysis_h_target
    # Shed the cube padding the way the production examples do: clip to the
    # geometry bounds plus a proportional pad, offset by half a cell so flat
    # faces fall mid-cell instead of tangent to a grid plane (which degenerates
    # marching cubes into slivers).  The cube itself stays whole and cubic.
    mesh_lo, mesh_hi = (np.asarray(bound, dtype=float) for bound in mesh.bounds)
    clip_pad = CLIP_BBOX_PAD * float((mesh_hi - mesh_lo).max()) + 0.5 * analysis_h_target
    bcs = BCProposal.model_validate(case["boundary_conditions"])
    preview_segment_ids = (
        np.load(case["segments_npz"])["segment_ids"] if case.get("segments_npz") else None
    )
    direct_specs = [
        item.selector.model_dump() for item in (*bcs.dirichlet, *bcs.neumann)
        if _uses_saved_segments(item.selector.model_dump())
    ]
    source_vertices = source_faces = None
    if direct_specs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        source_vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
        source_faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=device)

    def case_selector(spec):
        saved = _uses_saved_segments(spec)
        # Hand the saved labels only to a selector resolved here, against the
        # surface they were computed on.  A seeded selector is resolved later by
        # the solver, on its own surface, and must segment that one afresh.
        # Its slack is the solver's own BC tolerance, so a saved-segment region
        # claims the same band as every selector the solver resolves (it was h,
        # the solver's 0.75 h at the default eps = h/2).  analysis_h_target is
        # the solver's finest leaf (checked equal on the agentic sessions).
        selector = _selector(spec, preview_segment_ids if saved else None)
        if saved:
            selector.resolve(
                source_vertices, source_faces,
                eps_expansion=bc_match_tolerance(phase_field_epsilon, analysis_h_target),
            )
        return selector

    dirichlet = {}
    for item in bcs.dirichlet:
        value = item.value
        if item.components is None:
            encoded = value
        elif isinstance(value, tuple):
            if sorted(item.components) == [0, 1, 2]:
                encoded = value
            else:
                selected_values = [value[index] for index in item.components]
                if len(set(selected_values)) != 1:
                    raise ValueError(
                        "partial Dirichlet components require one shared scalar value"
                    )
                encoded = (item.components, selected_values[0])
        else:
            encoded = (item.components, value)
        dirichlet[case_selector(item.selector.model_dump())] = encoded
    if any(item.traction is None for item in bcs.neumann):
        raise ValueError("all traction values must be supplied before the forward solve")
    neumann = {
        case_selector(item.selector.model_dump()): item.traction
        for item in bcs.neumann
    }
    output = Path(case["output"])
    params = {
        "project_name": "api_fem_forward", "output_root": str(output), "async_io": True,
        "dim": 3, "elem_order": 1,
        "device": "cuda", "output_per_step": 0,
        "relative_tol": settings.relative_tol, "absolute_tol": settings.absolute_tol,
        "max_iter": settings.max_newton_iters,
        "line_search_enabled": settings.line_search_enabled, "update_damping_limit": 0.0,
        "load_steps": settings.load_steps,
        "linear_solver": "cg", "ew_enabled": False,
        "cg_backend_assembled": assembled_backend(settings.linear_solver),
        "E": material.youngs_modulus if material.mode == "uniform" else representative_E,
        "nu": material.poisson_ratio if material.mode == "uniform" else representative_nu,
        "B": effective_body_force,
        **({
            "material_artifact": material.artifact,
            "material_acceleration": acceleration,
        } if material.mode in FIELD_MODES else {}),
        "dirichlet_bcs_dict": dirichlet, "neumann_bcs_dict": neumann,
        "phase_field_epsilon": phase_field_epsilon,
        "update_beta_per_newton": False, "sharp_volumetric": False,
        "tangent_f64": False,
        "use_mixed_precision": settings.use_mixed_precision,
        # Gradient descent is the only optimizer this workflow drives; the
        # solver default is MMA, so pin it.
        "optimizer": "gd", "gd_lr": settings.gd_lr,
        # Only when asked for, so a run without it passes exactly what it did.
        # Its reference length is the solver's own epsilon, phase_field_epsilon
        # above (factor x analysis leaf), the one the failure advice uses.
        **({"gd_lr_auto": True, "gd_lr_auto_fraction": settings.gd_lr_auto_fraction}
           if settings.gd_lr_auto else {}),
        "max_optimize_iters": settings.max_optimize_iters,
        "output_per_optimize_step": settings.output_per_optimize_step,
        "optimize_relative_tol": settings.optimize_relative_tol,
        "target_volume_fraction": settings.target_volume_fraction,
        "volume_as_is": settings.volume_as_is,
        "shape_opt_w_vol": settings.shape_opt_w_vol,
        # No reinit_n_iters: it is the Sussman reinit's iteration count, and
        # the marching-cubes reinit selected below never reads it.
        "reinit_sdf": settings.reinit_sdf, "reinit_every": settings.reinit_every,
        "output_surface_vtu": True,
        "output_surface_glb": True, "output_surface_glb_deformed": False,
        "remeshing": True, "remeshing_method": "marchingcubes",
        "reinit_sdf_method": "marchingcubes",
        **_bc_preservation(settings),
        "design_variable": "sdf", "phi_reproject_band_k": 3.0,
        "remesh_octree_depth": settings.remesh_octree_depth,
        "remesh_init_octree_depth": settings.remesh_init_octree_depth,
        "scale_system_to_unit_diagonal": settings.scale_system_to_unit_diagonal,
        **({"beta_max": settings.beta_max_over_e * material.youngs_modulus}
           if settings.beta_max_over_e and material.mode == "uniform" else {}),
        "remeshing_trimesh": {"mesh": mesh, "mesh_center": case["mesh_center"],
                              "h_target": analysis_h_target,
                              "clip_bbox": (tuple(mesh_lo - clip_pad),
                                            tuple(mesh_hi + clip_pad))},
    }
    _require_fresh_amgx(params["cg_backend_assembled"])
    problem = ShapeOptTrimmed(
        "src.agents_wrapper.fem_workflow.config", override_params=params
    )
    problem.log(
        "Workflow effective config: "
        f"use_mixed_precision={bool(problem.params.use_mixed_precision)}, "
        f"cg_backend_assembled={problem.params.cg_backend_assembled}, "
        f"relative_tol={float(problem.params.relative_tol):.3e}, "
        f"absolute_tol={float(problem.params.absolute_tol):.3e}, "
        f"max_iter={int(problem.params.max_iter)}, "
        f"load_steps={int(problem.params.load_steps)}, "
        f"remesh_depth={int(problem.params.remesh_octree_depth)}, "
        f"remesh_init_depth={int(problem.params.remesh_init_octree_depth)}, "
        # The outer loop's own numbers, so the monitor reading this log can
        # report an optimization against what it was asked for rather than
        # against a load-step count that means nothing here.
        f"run_mode={settings.run_mode}, "
        f"max_optimize_iters={int(problem.params.max_optimize_iters)}, "
        f"output_per_optimize_step={int(problem.params.output_per_optimize_step)}, "
        f"volume_as_is={bool(problem.params.volume_as_is)}"
    )
    scaled = bool(settings.scale_system_to_unit_diagonal)
    if settings.run_mode == "shape_optimization":
        view = _OptimizationView(output, Path(problem.path), settings.output_per_optimize_step)
        original_save = problem.save

        def save_and_show(*save_args, **save_kwargs):
            original_save(*save_args, **save_kwargs)
            # The loop's failure path saves the state it failed in; that one
            # is never shown (see _OptimizationView).
            if getattr(problem, "_optimize_failed_iteration", None) is not None:
                return
            try:
                view.refresh()
            except Exception as exc:  # the viewer's asset must never fail the run
                problem.log(f"Workflow: displacement asset not refreshed: {exc}")

        problem.save = save_and_show
        try:
            if int(settings.load_steps) > 1:
                # optimize() solves every iteration at full load, so a ramp asked
                # for here used to be dropped.  It is spent on the first shape: the
                # ramp leaves the displacement the first iteration warm-starts from,
                # and every later one starts from its predecessor's.
                _solve_with_diagnosis(problem.forward, scaled, settings.linear_solver)
            _solve_with_diagnosis(problem.optimize, scaled, settings.linear_solver)
        except Exception as exc:
            # How large the steps were against the diffuse width, for the
            # learning rate the error recommends (and optimization_status.json).
            # With gd_lr_auto the steps were taken at the rates it chose, not
            # the card's gd_lr; the record and the advice use the last of them
            # (each step's own rate is in _optimize_steps).
            auto = _gd_lr_auto_record(problem, settings)
            rate = auto["gd_lr_effective"] if auto else settings.gd_lr
            step_record = optimization_step_record(
                rate, getattr(problem, "_optimize_steps", None),
                phase_field_epsilon, analysis_h_target)
            if auto:
                step_record["gd_lr_auto"] = auto
            try:
                status = view.fail(problem, exc, {"step_size": step_record,
                                                  **({"gd_lr_auto": auto} if auto else {})})
            except Exception as view_exc:  # never mask the solver's own error
                status = None
                problem.log(f"Workflow: could not restore the last good iteration: {view_exc}")
            advice = optimization_failure_advice(rate, step_record)
            if status is None:
                problem.log(f"Workflow: shape optimization failed. {advice}")
                raise
            shown = status["shown_iteration"]
            note = (f"; shape optimization failed at iteration {status['failed_iteration']}; "
                    + (f"the results shown are iteration {shown}, the last that solved"
                       if shown is not None else "no iteration solved, so none is shown")
                    + f". {advice}")
            problem.log(f"Workflow: {note[2:]}")
            try:
                replacement = type(exc)(f"{exc}{note}")
            except Exception:
                raise exc
            raise replacement from exc
        auto = _gd_lr_auto_record(problem, settings)
        status = view.complete({"gd_lr_auto": auto} if auto else None)
        problem.log(f"Workflow: shape optimization complete; the results shown are "
                    f"iteration {status['shown_iteration']}")
        return [str(output.resolve())]
    else:
        # forward() is what reads load_steps and ramps current_load_scale over
        # them; step() is one Newton solve at whatever scale is already set, so
        # calling it directly silently ran every case in a single step.
        ramped = max(1, int(settings.load_steps)) > 1
        _solve_with_diagnosis(lambda: problem.forward(
            # Saving inside the ramp is what leaves a surface per step for the
            # viewer to walk through; a single-step run saves once as before.
            on_load_step=(lambda index, total: problem.save(
                time=float(index + 1) / float(total)
            )) if ramped else None
        ), scaled, settings.linear_solver)
        if not ramped:
            problem.save(time=0)
        problem.finalize()
    surface_vtus = sorted(
        output.glob("*/vtu_surface/surface_t*.vtu"),
        key=lambda path: path.stat().st_mtime,
    )
    if surface_vtus:
        _export_displacement_contour(
            surface_vtus[-1], output / "displacement_magnitude.glb"
        )
    return [str(output.resolve())]


def parser():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage-asset"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=stage_asset)
    p = sub.add_parser("inspect-surface"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=inspect_surface)
    p = sub.add_parser("scale-surface"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--axis", choices="xyz", required=True); p.add_argument("--extent", type=float, required=True); p.add_argument("--units", required=True); p.add_argument("--octree-depth", type=int, default=7); p.add_argument("--cube-pad", type=float, default=OCTREE_CUBE_PAD); p.set_defaults(func=scale_surface)
    p = sub.add_parser("render-surface"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=render_surface)
    p = sub.add_parser("plain-colour-surface"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--rgb", required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--segments-npz", type=Path); p.add_argument("--segments-json", type=Path); p.add_argument("--part-rgb"); p.set_defaults(func=plain_colour_surface)
    p = sub.add_parser("segment-surface"); p.add_argument("--method", choices=("shape_diameter", "cad_faces"), default="shape_diameter"); p.add_argument("--cad-faces", type=Path); p.add_argument("--surface", type=Path, required=True); p.add_argument("--output-npz", type=Path, required=True); p.add_argument("--output-json", type=Path, required=True); p.add_argument("--output-glb", type=Path, required=True); p.add_argument("--clusters", type=int, default=5); p.add_argument("--rays", type=int, default=25); p.add_argument("--smoothing", type=float, default=0.25); p.add_argument("--cone-angle-deg", type=float, default=120.0); p.set_defaults(func=segment_surface)
    p = sub.add_parser("render-segmentation"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--segments-npz", type=Path, required=True); p.add_argument("--segments-json", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=render_segmentation)
    p = sub.add_parser("render-part-montage"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--segments-npz", type=Path, required=True); p.add_argument("--segments-json", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=render_part_montage)
    p = sub.add_parser("build-part-field"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--segments-npz", type=Path, required=True); p.add_argument("--materials-json", type=Path, required=True); p.add_argument("--unit-system", required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=build_part_field)
    p = sub.add_parser("resolve-bcs"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--bc-json", type=Path, required=True); p.add_argument("--segments-npz", type=Path); p.add_argument("--output-npz", type=Path, required=True); p.add_argument("--output-json", type=Path, required=True); p.add_argument("--output-glb", type=Path, required=True); p.set_defaults(func=resolve_bcs)
    p = sub.add_parser("build-case"); p.add_argument("--surface", type=Path, required=True); p.add_argument("--bc-json", type=Path, required=True); p.add_argument("--segments-npz", type=Path); p.add_argument("--material-json", type=Path, required=True); p.add_argument("--loads-json", type=Path); p.add_argument("--settings-json", type=Path, required=True); p.add_argument("--transform-json", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--solve-output", type=Path, required=True); p.add_argument("--octree-depth", type=int, default=7); p.set_defaults(func=build_case)
    p = sub.add_parser("export-surface-field"); p.add_argument("--vtu", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--field", required=True); p.add_argument("--component", default=""); p.add_argument("--deform-scale", type=float, default=0.0); p.add_argument("--value-range-from", default=""); p.add_argument("--yield-strength", type=float); p.add_argument("--hover-version", type=int, default=1); p.set_defaults(func=export_surface_field)
    p = sub.add_parser("run-forward"); p.add_argument("--case", type=Path, required=True); p.set_defaults(func=run_forward)
    return root


def main():
    args = parser().parse_args()
    artifacts = args.func(args)
    print(json.dumps({"artifacts": artifacts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
