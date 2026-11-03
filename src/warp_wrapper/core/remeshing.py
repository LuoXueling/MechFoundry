"""Adaptive octree remeshing helpers for ``ShapeOptTrimmed``.

Pure-function helpers — no class state. The flow is:

  1. Maintain a background ``(2^d+1)^3`` cube grid SDF in normalized
     ``[-1, 1]^3`` space (the Sussman target).
  2. Each iteration, build a new outer octree from the cleaned background
     grid via ``hierarchical_extraction`` + ``sdf2octree``.
  3. Convert the resulting octree (which lives in the cube ``[-1, 1]^3``)
     into physical space via per-axis ``axis_scale`` / ``geo_min``
     mirroring ``LatentShapeOpt._normalized_to_physical``.

The ``geometric_func`` consumed by ``hierarchical_extraction`` queries
points in the SAME normalized cube space, so we never have to coordinate
transform inside the geometric callback.
"""

import os
import sys

import torch

# Make ``src.triposg_wrapper`` importable regardless of the launching CWD.
# Example runners add ``src/warp_wrapper`` explicitly so the repo
# root isn't on ``sys.path`` by default; the latent pipeline-driven entry
# points add it manually.  Doing it here keeps the helpers self-contained.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TRIPOSG_DEP = os.path.join(_REPO_ROOT, "src", "dependency", "TripoSG")
for _p in (_REPO_ROOT, _TRIPOSG_DEP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.triposg_wrapper.hierarchical_extraction import (
    hierarchical_extraction,
)  # noqa: E402
from src.triposg_wrapper.utils import sdf2octree  # noqa: E402


# ---------------------------------------------------------------------------
# Trilinear interpolation in normalized cube space
# ---------------------------------------------------------------------------


def _trilinear_sample(grid_sdf, pts_norm, cube_lo=-1.0, cube_hi=1.0):
    """Trilinear sample of a structured cube SDF at arbitrary points.

    Args:
        grid_sdf: ``(N, N, N)`` float tensor in normalized cube space.
        pts_norm: ``(M, 3)`` query points in the same space.
        cube_lo, cube_hi: extents of the cube grid (default ``[-1, 1]``).

    Returns:
        ``(M,)`` float32 tensor of sampled SDF values.
    """
    N = grid_sdf.shape[0]
    spacing = (cube_hi - cube_lo) / (N - 1)
    p = pts_norm.to(grid_sdf.dtype).to(grid_sdf.device)
    u = ((p - cube_lo) / spacing).clamp(0.0, float(N - 1) - 1e-6)
    i0 = u.floor().long()
    f = u - i0.to(u.dtype)
    i1 = (i0 + 1).clamp(max=N - 1)

    c000 = grid_sdf[i0[:, 0], i0[:, 1], i0[:, 2]]
    c100 = grid_sdf[i1[:, 0], i0[:, 1], i0[:, 2]]
    c010 = grid_sdf[i0[:, 0], i1[:, 1], i0[:, 2]]
    c110 = grid_sdf[i1[:, 0], i1[:, 1], i0[:, 2]]
    c001 = grid_sdf[i0[:, 0], i0[:, 1], i1[:, 2]]
    c101 = grid_sdf[i1[:, 0], i0[:, 1], i1[:, 2]]
    c011 = grid_sdf[i0[:, 0], i1[:, 1], i1[:, 2]]
    c111 = grid_sdf[i1[:, 0], i1[:, 1], i1[:, 2]]

    fx = f[:, 0]
    fy = f[:, 1]
    fz = f[:, 2]
    one_fx = 1.0 - fx
    one_fy = 1.0 - fy
    one_fz = 1.0 - fz

    out = (
        one_fx * one_fy * one_fz * c000
        + fx * one_fy * one_fz * c100
        + one_fx * fy * one_fz * c010
        + fx * fy * one_fz * c110
        + one_fx * one_fy * fz * c001
        + fx * one_fy * fz * c101
        + one_fx * fy * fz * c011
        + fx * fy * fz * c111
    )
    return out.float()


def make_trilinear_geo_func(grid_sdf, cube_lo=-1.0, cube_hi=1.0):
    """Return a ``geometric_func`` callable for ``hierarchical_extraction``.

    The callable matches the signature consumed by
    ``hierarchical_extraction`` / ``nitsche_field._build_adaptive_quadrature``:
    ``(1, M, 3) float16 → (1, M, 1) float32``.
    """

    def geo_func(pts_norm):
        # pts_norm: (1, M, 3) in normalized cube space (typically float16)
        pts = pts_norm.squeeze(0).float()
        sdf = _trilinear_sample(grid_sdf, pts, cube_lo, cube_hi)
        return sdf.unsqueeze(0).unsqueeze(-1)

    return geo_func


# ---------------------------------------------------------------------------
# Octree construction wrappers
# ---------------------------------------------------------------------------


def _surface_vertices_to_must_query_positions(
    vertices_norm,
    init_octree_depth,
    octree_depth,
    cube_lo=-1.0,
    cube_hi=1.0,
):
    """Deduplicate surface vertices into final-depth parent groups.

    ``hierarchical_extraction`` consumes lattice *nodes*, while the desired
    guarantee is cell based: every finest cell containing a known surface
    vertex must be present at ``octree_depth``.  This helper finds each
    containing finest cell's parent and emits that parent's all-odd centre
    node.  The eight finest cells touching the centre are exactly the parent's
    eight children.  The extractor already carries every must-query point
    through all preceding transitions, so emitting coarser centres here would
    only duplicate ancestry construction and GPU sorting.

    Args:
        vertices_norm: ``(V, 3)`` vertices in normalized cube coordinates.
        init_octree_depth: depth of the extraction's initial dense grid.
        octree_depth: requested final octree depth.
        cube_lo, cube_hi: normalized cube bounds.

    Returns:
        ``(positions, counts)`` where ``positions`` is a deduplicated
        ``(K, 3)`` float64 tensor containing one centre per final-depth parent
        and ``counts`` is ``[(octree_depth, K)]``.  An empty tensor is returned
        when no adaptive refinement is required.
    """
    vertices = torch.as_tensor(vertices_norm)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(
            f"vertices_norm must have shape (V, 3), got {tuple(vertices.shape)}"
        )

    device = vertices.device
    vertices = vertices.to(dtype=torch.float64)
    if octree_depth <= init_octree_depth or vertices.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float64, device=device), []

    finite = torch.isfinite(vertices).all(dim=-1)
    vertices = vertices[finite]
    if vertices.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.float64, device=device), []

    lo = float(cube_lo)
    hi = float(cube_hi)
    span = hi - lo
    if span <= 0.0:
        raise ValueError(f"cube_hi must exceed cube_lo, got {cube_lo}, {cube_hi}")

    # Clamping before floor also handles a vertex on the upper cube boundary:
    # its containing cell is the last valid cell, not the out-of-range index N.
    unit = ((vertices.clamp(lo, hi) - lo) / span).clamp(0.0, 1.0)
    depth = int(octree_depth)
    n_cells = 2**depth
    parent_n = n_cells // 2
    cell = torch.floor(unit * n_cells).long().clamp(0, n_cells - 1)
    parents = torch.div(cell, 2, rounding_mode="floor")

    # A 1-D integer unique is substantially cheaper than unique(dim=0) on the
    # (V, 3) parent tensor and gives an exact, collision-free lattice key.
    parent_key = (
        (parents[:, 0] * parent_n + parents[:, 1]) * parent_n + parents[:, 2]
    )
    parent_key = torch.unique(parent_key)
    parent_x = torch.div(parent_key, parent_n * parent_n, rounding_mode="floor")
    parent_yz = parent_key.remainder(parent_n * parent_n)
    parent_y = torch.div(parent_yz, parent_n, rounding_mode="floor")
    parent_z = parent_yz.remainder(parent_n)
    unique_parents = torch.stack((parent_x, parent_y, parent_z), dim=-1)

    # 2*parent+1 is an all-odd depth-d lattice node, so all cells touching it
    # belong to this one parent.  The extractor propagates it from init to d.
    centre_idx = unique_parents * 2 + 1
    positions = lo + centre_idx.to(torch.float64) * (span / n_cells)
    return positions, [(depth, int(parent_key.shape[0]))]


def remesh_to_octree(
    geometric_func,
    init_octree_depth,
    octree_depth,
    domain_length=2.0,
    device="cuda",
    must_query_positions=None,
    sdf_truncation=0.01,
):
    """Run ``hierarchical_extraction`` + ``sdf2octree`` end to end.

    Returns a normalized-space octree mesh dict with keys
    ``nodes, elements, sdf, grid_sdf, existing_anchors`` and optional
    ``edge_hanging`` / ``face_hanging`` entries when present.

    Coordinate space: nodes are in the centered cube
    ``[-domain_length/2, domain_length/2]^3``. With ``domain_length=2.0``
    this is the canonical normalized cube ``[-1, 1]^3``.

    ``must_query_positions``: optional ``(K, 3)`` float tensor in
    normalized cube coords.  Forwarded to ``hierarchical_extraction`` —
    see its docstring.
    """
    grid_sdf, anchors = hierarchical_extraction(
        geometric_func=geometric_func,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_octree_depth,
        octree_depth=octree_depth,
        device=device,
        must_query_positions=must_query_positions,
        sdf_truncation=sdf_truncation,
    )
    nodes, node_sdf, elements, edge_h, face_h = sdf2octree(
        cell_tensor=anchors,
        sdf_tensor=grid_sdf,
        domain_length=domain_length,
    )
    octree = {
        "nodes": nodes,
        "elements": elements,
        "sdf": node_sdf,
        "grid_sdf": grid_sdf,
        "existing_anchors": anchors,
    }
    if edge_h is not None and len(edge_h) > 0:
        octree["edge_hanging"] = edge_h
    if face_h is not None and len(face_h) > 0:
        octree["face_hanging"] = face_h
    return octree


# ---------------------------------------------------------------------------
# Normalized ↔ physical coordinate conversion
# ---------------------------------------------------------------------------


def scale_octree_nodes_to_physical(octree_norm, axis_scale_t, geo_min_t):
    """Convert a normalized-cube octree dict's NODES (only) to physical space.

    Mirrors ``LatentShapeOpt._normalized_to_physical``:
        physical = (normalized - geo_min) * axis_scale

    The SDF VALUES are left untouched — callers in this module always
    work with the bg grid in physical SDF units (the value space matches
    the input mesh's SDF values, with no normalization), so the octree's
    nodal SDF (sampled from that grid) is already physical.

    Hanging-node entries pass through unchanged (they store node indices,
    not coordinates).
    """
    out = dict(octree_norm)
    nodes_norm = octree_norm["nodes"].to(torch.float64)
    out["nodes"] = (nodes_norm - geo_min_t) * axis_scale_t
    out["sdf"] = octree_norm["sdf"].float()
    return out


# ---------------------------------------------------------------------------
# Background-grid bookkeeping
# ---------------------------------------------------------------------------


def cube_grid_points_normalized(n, device="cuda", dtype=torch.float64):
    """Return ``(n^3, 3)`` cube-grid points in ``[-1, 1]^3``.

    Indexing matches ``hierarchical_extraction`` / ``sdf2octree`` —
    ``ij`` ordering with ``x`` slowest, ``z`` fastest.
    """
    coords = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)


# ---------------------------------------------------------------------------
# Trilinear interpolation over axis-aligned octree hex mesh
# ---------------------------------------------------------------------------


def octree_hex_interpolate(nodes, elements, node_field, query_points):
    """Trilinear interpolation of a nodal field over an axis-aligned octree hex mesh.

    For each query point, finds the containing hex element (axis-aligned
    bbox check), computes local coordinates ``(ξ,η,ζ) ∈ [0,1]³``, and
    evaluates the standard 8-node trilinear basis using the octree node
    ordering (0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1)
    6:(1,1,1) 7:(0,1,1)).  Query points outside every element are snapped
    to the nearest element center.

    Args:
        nodes: ``(N, 3)`` tensor of node positions.
        elements: ``(E, 8)`` tensor of element-to-node indices.
        node_field: ``(N,)`` or ``(N, C)`` tensor of nodal values.
        query_points: ``(M, 3)`` tensor of query positions.

    Returns:
        ``(M,)`` or ``(M, C)`` tensor of interpolated values, dtype
        matching ``node_field``.
    """
    device = query_points.device
    nodes_d = nodes.to(dtype=torch.float64, device=device)
    elements_l = elements.to(dtype=torch.long, device=device)
    field_d = node_field.to(dtype=torch.float64, device=device)
    query_d = query_points.to(dtype=torch.float64, device=device)

    squeeze_out = False
    if field_d.dim() == 1:
        field_d = field_d.unsqueeze(-1)
        squeeze_out = True

    elem_nodes = nodes_d[elements_l]  # (E, 8, 3)
    elem_lo = elem_nodes.amin(dim=1)  # (E, 3)
    elem_hi = elem_nodes.amax(dim=1)  # (E, 3)

    M = query_d.shape[0]
    elem_extent = (elem_hi - elem_lo).clamp_min(1e-30)
    elem_tol = elem_extent.amin(dim=1, keepdim=True) * 1e-6  # (E, 1)

    elem_idx = torch.full((M,), -1, dtype=torch.long, device=device)
    chunk = 512
    for i in range(0, M, chunk):
        q = query_d[i : i + chunk]  # (c, 3)
        in_lo = (q.unsqueeze(1) >= (elem_lo.unsqueeze(0) - elem_tol.unsqueeze(0))).all(
            dim=-1
        )
        in_hi = (q.unsqueeze(1) <= (elem_hi.unsqueeze(0) + elem_tol.unsqueeze(0))).all(
            dim=-1
        )
        contained = in_lo & in_hi
        any_hit = contained.any(dim=1)
        first_hit = contained.int().argmax(dim=1)
        first_hit[~any_hit] = -1
        elem_idx[i : i + chunk] = first_hit

    missing = elem_idx == -1
    if bool(missing.any()):
        centers = 0.5 * (elem_lo + elem_hi)  # (E, 3)
        q_miss = query_d[missing]
        d2 = ((q_miss.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        elem_idx[missing] = d2.argmin(dim=1)

    sel_lo = elem_lo[elem_idx]
    sel_extent = elem_extent[elem_idx]
    local = ((query_d - sel_lo) / sel_extent).clamp(0.0, 1.0)
    xi, eta, zeta = local[:, 0], local[:, 1], local[:, 2]
    one_xi, one_eta, one_zeta = 1.0 - xi, 1.0 - eta, 1.0 - zeta

    N_basis = torch.stack(
        [
            one_xi * one_eta * one_zeta,
            xi * one_eta * one_zeta,
            xi * eta * one_zeta,
            one_xi * eta * one_zeta,
            one_xi * one_eta * zeta,
            xi * one_eta * zeta,
            xi * eta * zeta,
            one_xi * eta * zeta,
        ],
        dim=-1,
    )  # (M, 8)

    sel_elem_nodes = elements_l[elem_idx]  # (M, 8)
    sel_values = field_d[sel_elem_nodes]  # (M, 8, C)
    result = (N_basis.unsqueeze(-1) * sel_values).sum(dim=1)  # (M, C)

    if squeeze_out:
        result = result.squeeze(-1)
    return result.to(dtype=node_field.dtype)


# ---------------------------------------------------------------------------
# Fast scatter-based octree → bg cube grid (no point-in-element search)
# ---------------------------------------------------------------------------


def octree_node_bg_indices(
    octree_nodes_phys,
    n,
    axis_scale,
    geo_min,
    device="cuda",
):
    """Compute the (i, j, k) bg-grid integer indices for each octree node.

    All octree nodes built by ``hierarchical_extraction`` (depth ``d``) lie at
    integer multiples of ``2/(n-1)`` in the normalized cube ``[-1, 1]^3`` with
    ``n = 2^d + 1``, so this is an exact lookup, not a snap-to-nearest.

    Returns a ``(V, 3)`` int64 tensor.
    """
    axis_scale_t = torch.as_tensor(axis_scale, device=device, dtype=torch.float64)
    geo_min_t = torch.as_tensor(geo_min, device=device, dtype=torch.float64)
    nodes_phys = octree_nodes_phys.to(device=device, dtype=torch.float64)
    # physical = (normalized - geo_min) * axis_scale  ⇒  normalized = phys/scale + geo_min
    nodes_norm = nodes_phys / axis_scale_t + geo_min_t
    spacing = 2.0 / (n - 1)
    idx = ((nodes_norm + 1.0) / spacing).round().long()
    return idx.clamp_(0, n - 1)


def scatter_octree_to_cube_grid(
    octree_nodes_phys,
    octree_sdf_phys,
    base_grid,
    axis_scale,
    geo_min,
    device="cuda",
):
    """Scatter octree-node SDF values onto the bg cube grid in O(V).

    Each octree node has a unique ``(i, j, k)`` index on the bg lattice
    (``hierarchical_extraction`` builds octree corners at integer multiples
    of ``2/(2^d)``).  Bg-grid points NOT covered by any octree node are
    interior to a coarse leaf — we keep their previous values from
    ``base_grid`` (typically the previous post-Sussman grid).  The
    subsequent Sussman pass propagates the freshly-scattered values to fix
    any drift.

    Args:
        octree_nodes_phys: ``(V, 3)`` float octree node positions (physical).
        octree_sdf_phys: ``(V,)`` float octree nodal SDF (physical units).
        base_grid: ``(n, n, n)`` float32 baseline (previous bg grid).
        axis_scale: per-axis physical extent / 2 (length-3, ``[-1,1]→phys``).
        geo_min: per-axis ``-mesh_center / axis_scale`` (length-3).

    Returns:
        ``(n, n, n)`` float32 tensor in PHYSICAL SDF units.
    """
    n = base_grid.shape[0]
    out = base_grid.to(device=device, dtype=torch.float32).clone()
    idx = octree_node_bg_indices(
        octree_nodes_phys, n, axis_scale, geo_min, device=device
    )
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = octree_sdf_phys.to(
        device=device, dtype=torch.float32
    )
    return out


# ---------------------------------------------------------------------------
# Uniform hex mesh aligned to the bg-grid lattice
# ---------------------------------------------------------------------------


def build_bg_aligned_hex_mesh(clean_sdf, axis_scale, geo_min, clip_bbox=None):
    """Build a uniform hex mesh from bg-grid lattice points.

    Nodes are placed at exact integer positions on the ``(n,n,n)`` bg-grid
    lattice so that ``octree_node_bg_indices`` gives an exact lookup —
    no interpolation needed for per-iter SDF updates.  When ``clip_bbox`` is
    supplied the mesh is snapped to the nearest bg-grid nodes that contain the
    box; no octree machinery or post-hoc trimming is required.

    Args:
        clean_sdf:  ``(n, n, n)`` float32 bg-grid SDF (physical units).
        axis_scale: length-3 sequence — physical half-extent per axis.
        geo_min:    length-3 sequence — ``-center / axis_scale`` per axis.
        clip_bbox:  optional ``((lo_x,lo_y,lo_z),(hi_x,hi_y,hi_z))`` in
                    physical coords.  Snapped outward to the nearest bg-grid
                    node so all clip-box corners are included.

    Returns:
        dict with ``nodes`` (V,3) float64, ``elements`` (E,8) int32,
        ``sdf`` (V,) float32 — ready to pass as ``octree_mesh``.
    """
    n = int(clean_sdf.shape[0])
    device = clean_sdf.device

    axis_scale_t = torch.as_tensor(axis_scale, dtype=torch.float64, device=device)
    geo_min_t = torch.as_tensor(geo_min, dtype=torch.float64, device=device)

    # Physical position of bg-grid node at index i along axis a:
    #   norm_i = -1 + i * 2/(n-1)
    #   phys_i = (norm_i - geo_min[a]) * axis_scale[a]
    idx = torch.arange(n, dtype=torch.float64, device=device)
    norm = -1.0 + idx * (2.0 / (n - 1))
    # Per-axis physical coordinates: shape (3, n)
    phys_axes = [(norm - geo_min_t[a]) * axis_scale_t[a] for a in range(3)]

    if clip_bbox is not None:
        lo, hi = clip_bbox
        # Snap outward: first index >= lo, last index <= hi
        eps = 1e-6
        ranges = []
        for a in range(3):
            mask_lo = (phys_axes[a] >= lo[a] - eps).nonzero(as_tuple=True)[0]
            mask_hi = (phys_axes[a] <= hi[a] + eps).nonzero(as_tuple=True)[0]
            i_lo = int(mask_lo[0].item()) if len(mask_lo) else 0
            i_hi = int(mask_hi[-1].item()) if len(mask_hi) else n - 1
            ranges.append((i_lo, i_hi))
    else:
        ranges = [(0, n - 1)] * 3

    (ix_lo, ix_hi), (iy_lo, iy_hi), (iz_lo, iz_hi) = ranges
    nx = ix_hi - ix_lo  # cells in x
    ny = iy_hi - iy_lo
    nz = iz_hi - iz_lo

    # Build node positions via meshgrid (all on device, no CPU roundtrip).
    xx, yy, zz = torch.meshgrid(
        phys_axes[0][ix_lo : ix_hi + 1],
        phys_axes[1][iy_lo : iy_hi + 1],
        phys_axes[2][iz_lo : iz_hi + 1],
        indexing="ij",
    )
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)

    # Hex elements in VTK ordering — fully vectorized, no Python loops.
    # Node index: i*(ny+1)*(nz+1) + j*(nz+1) + k.
    Ny1, Nz1 = ny + 1, nz + 1
    ii, jj, kk = torch.meshgrid(
        torch.arange(nx, device=device),
        torch.arange(ny, device=device),
        torch.arange(nz, device=device),
        indexing="ij",
    )
    ii, jj, kk = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)

    def nid(i, j, k):
        return i * Ny1 * Nz1 + j * Nz1 + k

    elements = torch.stack(
        [
            nid(ii, jj, kk),
            nid(ii + 1, jj, kk),
            nid(ii + 1, jj + 1, kk),
            nid(ii, jj + 1, kk),
            nid(ii, jj, kk + 1),
            nid(ii + 1, jj, kk + 1),
            nid(ii + 1, jj + 1, kk + 1),
            nid(ii, jj + 1, kk + 1),
        ],
        dim=1,
    ).to(torch.int32)

    # SDF: direct slice of the bg-grid (no interpolation).
    sdf_vals = clean_sdf[
        ix_lo : ix_hi + 1, iy_lo : iy_hi + 1, iz_lo : iz_hi + 1
    ].reshape(-1)

    return {
        "nodes": nodes,
        "elements": elements,
        "sdf": sdf_vals.float(),
        # Store the bg-grid index ranges for O(1) per-iter SDF updates.
        "_bg_ix": (ix_lo, ix_hi),
        "_bg_iy": (iy_lo, iy_hi),
        "_bg_iz": (iz_lo, iz_hi),
    }


# ---------------------------------------------------------------------------
# Bounding-box clipping of an octree mesh
# ---------------------------------------------------------------------------


def clip_octree_to_bbox(octree, bbox_lo, bbox_hi, tol=1e-6):
    """Drop octree elements whose physical centroid falls outside the bbox.

    Keeps the "shape" of the octree around the geometry and discards the cube
    padding introduced by the isotropic cube embedding.  Elements survive iff
    every component of their 8-corner mean lies within ``[bbox_lo, bbox_hi]``
    (with a small ``tol`` slack so elements right at the face are kept).

    Nodes referenced only by dropped elements are deleted and the connectivity
    is renumbered.  Hanging-node entries that reference a deleted node are
    dropped; the rest are remapped.  ``grid_sdf`` / ``existing_anchors`` pass
    through unchanged — those describe the full cube bg grid, not the octree
    mesh, and remain valid for the trilinear geometric_func.

    Args:
        octree: dict with keys ``nodes``, ``elements``, ``sdf`` (and optional
            ``edge_hanging`` / ``face_hanging`` / ``grid_sdf`` /
            ``existing_anchors``).  Physical space.
        bbox_lo, bbox_hi: length-3 physical bounds.
        tol: numerical slack in physical units.

    Returns:
        A new octree dict with the same keys, clipped.
    """
    nodes = octree["nodes"]
    elements = octree["elements"]
    device = nodes.device

    bbox_lo_t = torch.as_tensor(bbox_lo, device=device, dtype=nodes.dtype)
    bbox_hi_t = torch.as_tensor(bbox_hi, device=device, dtype=nodes.dtype)

    # Element centroid = mean of its 8 corners.
    elem_corners = nodes[elements.to(torch.long)]  # (M, 8, 3)
    centroids = elem_corners.mean(dim=1)  # (M, 3)

    inside = ((centroids >= bbox_lo_t - tol) & (centroids <= bbox_hi_t + tol)).all(
        dim=-1
    )

    if bool(inside.all().item()):
        return octree

    new_elements_raw = elements[inside]  # (M', 8) — old indices
    unique_old, inverse = torch.unique(
        new_elements_raw.reshape(-1).to(torch.long), return_inverse=True
    )
    new_elements = inverse.reshape(-1, 8).to(elements.dtype)
    new_nodes = nodes[unique_old]
    new_sdf = octree["sdf"][unique_old]

    V_old = nodes.shape[0]
    old_to_new = torch.full((V_old,), -1, dtype=torch.long, device=device)
    old_to_new[unique_old] = torch.arange(
        unique_old.shape[0], device=device, dtype=torch.long
    )

    out = dict(octree)
    out["nodes"] = new_nodes
    out["elements"] = new_elements
    out["sdf"] = new_sdf

    # Remap hanging-node entries; drop rows that reference a deleted node.
    for key in ("edge_hanging", "face_hanging"):
        if key not in octree:
            continue
        raw = octree[key]
        if raw.numel() == 0:
            out.pop(key, None)
            continue
        remapped = old_to_new[raw.to(torch.long)]
        keep_row = (remapped >= 0).all(dim=-1)
        if bool(keep_row.any().item()):
            out[key] = remapped[keep_row].to(raw.dtype)
        else:
            out.pop(key, None)

    return out


def _snap_mesh_center_to_bbox_lattice(
    mesh_lo, mesh_hi, bbox_lo, bbox_hi, h_target, octree_depth
):
    """Choose mesh_center so the isotropic cube lattice aligns with bbox_lo.

    The cube lattice is ``{bbox_lo + (n - k)*h : n = 0, ..., 2^d}`` for some
    non-negative integer offset ``k`` per axis.  Aligning to ``bbox_lo``
    guarantees that every baseline uniform node at ``bbox_lo + n*h`` sits
    exactly on a cube-grid corner, which in turn keeps the clipped cube
    octree lattice byte-identical to the baseline uniform hex lattice in
    the region where the two overlap.  Without this snap, the cube is
    centered on the geometry and the lattice is offset by up to ``h/2``.

    Pure-Python scalar arithmetic — each axis is independent and the cost
    is negligible.

    Returns:
        ``(mesh_center, k)`` — ``mesh_center`` is a length-3 tuple of
        floats; ``k`` is a length-3 tuple of ints for logging.
    """
    import math

    mesh_lo = tuple(float(x) for x in mesh_lo)
    mesh_hi = tuple(float(x) for x in mesh_hi)
    bbox_lo = tuple(float(x) for x in bbox_lo)
    bbox_hi = tuple(float(x) for x in bbox_hi)

    L_cube = h_target * (2**octree_depth)
    half_L = L_cube / 2.0

    mesh_center = [0.0, 0.0, 0.0]
    k_out = [0, 0, 0]
    for i in range(3):
        # half_lo_gap = L_cube/2 - (bbox_hi - bbox_lo)/2 = half the slack.
        half_lo_gap = half_L - 0.5 * (bbox_hi[i] - bbox_lo[i])
        k_i = max(0, int(math.floor(half_lo_gap / h_target)))
        cube_lo_i = bbox_lo[i] - k_i * h_target
        cube_hi_i = cube_lo_i + L_cube
        if cube_lo_i > mesh_lo[i] + 1e-9 or cube_hi_i < mesh_hi[i] - 1e-9:
            raise ValueError(
                f"Snapped cube axis {i} [{cube_lo_i:.4f} → {cube_hi_i:.4f}] "
                f"does not cover geometry [{mesh_lo[i]:.4f} → {mesh_hi[i]:.4f}] "
                f"at h={h_target}, depth={octree_depth}"
            )
        mesh_center[i] = cube_lo_i + half_L
        k_out[i] = k_i

    return tuple(mesh_center), tuple(k_out)


# ---------------------------------------------------------------------------
# Initial-octree bootstrap from a trimesh surface (caller-side)
# ---------------------------------------------------------------------------


def resolve_explicit_surface_sdf_truncation(h_target, sdf_truncation=None):
    """Return the physical SDF refinement width for an explicit surface.

    Explicit-surface SDF samples are physical distances, so the default must
    scale with the mesh rather than use a dimensionless-looking constant.
    ``hierarchical_extraction`` applies its existing 0.9 multiplier to this
    value. A caller-supplied value remains a physical-length override.
    """
    value = h_target if sdf_truncation is None else sdf_truncation
    value = float(value)
    if value <= 0.0:
        raise ValueError("sdf_truncation must be a positive physical length")
    return value


def build_initial_octree_from_trimesh(
    mesh,
    mesh_center,
    h_target,
    octree_depth,
    init_octree_depth,
    device="cuda",
    clip_bbox=None,
    build_octree=True,
    sdf_truncation=None,
):
    """Build the FIRST outer octree for the remeshing path from a surface mesh.

    Accepts either a ``trimesh.Trimesh`` or a ``warp.Mesh`` as ``mesh``.

    * **trimesh.Trimesh**: converted to a warp BVH at the start of this
      function using :func:`~core.analytical_sdf.build_warp_mesh`
      (``mesh_query_point_sign_winding_number``).
    * **warp.Mesh**: used directly, skipping the conversion step.  Sign
      convention is auto-detected by querying the bounding-box centroid.

    The warp BVH wraps the surface so ``hierarchical_extraction`` queries it
    directly via :func:`~core.analytical_sdf.compute_nodal_sdf`.  Subsequent
    remesh iterations switch to ``make_trilinear_geo_func`` over the maintained
    background grid.  Surface vertices also seed one deduplicated final-parent
    ``must_query_positions`` centre per occupied parent group.  The extractor
    propagates those centres through the coarser levels, so every finest cell
    containing a supplied vertex is refined to ``octree_depth`` without
    introducing partial parent splits.

    The cube embedding is **isotropic**: cube side ``L = h_target * 2^depth``
    so each cube leaf has side ``h_target`` after the cube→physical map.
    The cube is centered on ``mesh_center`` and may extend beyond the
    geometry's bbox along the thin axes — that's intentional, so the bg grid
    spacing matches the baseline uniform-mesh element size.

    The mesh is supplied in PHYSICAL coordinates; the wrapper converts cube
    query points to physical before calling the BVH so the returned SDF is in
    physical units, matching the convention used by the rest of the remeshing
    path (bg grid, FEM Nitsche field, Sussman PDE all in physical units).

    Args:
        mesh: ``trimesh.Trimesh`` or ``warp.Mesh`` in physical coordinates.
        mesh_center: length-3 physical center for the cube embedding.
        h_target: physical leaf spacing for the bg grid (e.g. baseline
            uniform-mesh element size).
        octree_depth: target octree depth (bg grid is ``(2^d + 1)^3``).
        init_octree_depth: coarse depth for ``hierarchical_extraction``.
        device: target device.
        clip_bbox: optional ``(bbox_lo, bbox_hi)`` pair in physical coords.
            When provided, the cube lattice is snapped so its corners land
            exactly on the ``bbox_lo + n*h`` lattice (matching a baseline
            uniform hex mesh that starts at ``bbox_lo``), and the returned
            octree is clipped to elements whose centroid falls inside
            ``[bbox_lo, bbox_hi]``.  This sheds the isotropic-cube padding
            while keeping the per-iter bg grid / Sussman machinery cubic.
        sdf_truncation: physical SDF width passed to hierarchical extraction.
            Defaults to ``h_target``, so its 0.9 refinement band is 0.9 finest
            leaf widths. An explicit value is interpreted in physical units.

    Returns:
        A dict with keys:
            ``octree_mesh``  — physical-space octree dict (nodes/elements/sdf
                /grid_sdf/existing_anchors and optional hanging-node entries).
                Drop-in for ``override_params["octree_mesh"]``.
            ``bg_grid_sdf``  — ``(N, N, N)`` float32 cube-indexed bg grid in
                physical SDF units (the value the trilinear geo_func will
                sample on subsequent iters).
            ``axis_scale``   — list of 3 floats; per-axis cube→physical scale
                (isotropic — all three equal to ``L/2``).
            ``geo_min``      — list of 3 floats; per-axis cube origin offset.
            ``grid_shape``   — ``(N, N, N)`` background-grid shape.
            ``grid_spacing`` — per-axis physical spacing of the bg grid
                (isotropic — all three equal to ``h_target``).
    """
    import math
    import numpy as np
    import warp as wp

    from .analytical_sdf import build_warp_mesh, compute_nodal_sdf

    mesh_center = tuple(float(x) for x in mesh_center)
    if len(mesh_center) != 3:
        raise ValueError(f"mesh_center must have length 3, got {len(mesh_center)}")

    # ------------------------------------------------------------------
    # Convert trimesh → warp mesh, or accept warp.Mesh directly.
    # Both paths produce: wp_mesh, negate, bbox_diag, mesh_lo, mesh_hi.
    # ------------------------------------------------------------------
    if isinstance(mesh, wp.Mesh):
        # Direct warp.Mesh — extract bounds from the point array.
        pts_t = wp.to_torch(mesh.points)  # (V, 3) float32
        surface_vertices_phys = pts_t.to(device=device, dtype=torch.float64)
        lo_t = pts_t.min(0).values
        hi_t = pts_t.max(0).values
        mesh_lo = tuple(lo_t.tolist())
        mesh_hi = tuple(hi_t.tolist())
        bbox_diag = float(torch.linalg.norm(hi_t - lo_t))
        # Detect sign convention by probing the bounding-box centroid.
        centroid_t = ((lo_t + hi_t) / 2.0).unsqueeze(0).to(device=device)
        probe_sdf = compute_nodal_sdf(
            mesh,
            centroid_t,
            negate=False,
            max_dist=10.0 * bbox_diag,
            device=device,
        )
        negate = float(probe_sdf[0]) > 0.0
        wp_mesh = mesh
        _keep_alive = None
    else:
        # trimesh.Trimesh — build warp BVH via build_warp_mesh.
        mesh_lo = tuple(float(x) for x in mesh.bounds[0])
        mesh_hi = tuple(float(x) for x in mesh.bounds[1])
        bbox_diag = math.sqrt(sum((mesh_hi[i] - mesh_lo[i]) ** 2 for i in range(3)))
        verts = torch.from_numpy(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        ).to(device)
        surface_vertices_phys = verts.to(dtype=torch.float64)
        faces = torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int32)).to(
            device
        )
        # Probe: centroid nudged toward vertex[0] to avoid sign-ambiguity at
        # perfectly symmetric interior points.
        probe = torch.as_tensor(
            mesh.centroid + 0.01 * (mesh.vertices[0] - mesh.centroid),
            dtype=torch.float32,
            device=device,
        )
        wp_mesh, negate, _keep_alive = build_warp_mesh(
            verts, faces, probe, bbox_diag, device=device
        )

    # ------------------------------------------------------------------
    # Sanity-check: depth must be deep enough to fit the mesh in the cube.
    # ------------------------------------------------------------------
    half_extent_needed = max(
        max(abs(mesh_lo[i] - mesh_center[i]) for i in range(3)),
        max(abs(mesh_hi[i] - mesh_center[i]) for i in range(3)),
    )
    d_min = max(1, int(math.ceil(math.log2(2.0 * half_extent_needed / h_target))))
    if octree_depth < d_min:
        print(
            f"  Remesh: octree_depth={octree_depth} too small for "
            f"h_target={h_target} and mesh half-extent={half_extent_needed:.4f}; "
            f"raising to {d_min}"
        )
        octree_depth = d_min

    # If ``clip_bbox`` is provided, snap the mesh center so the cube lattice
    # aligns with ``bbox_lo`` — this makes the surviving cube nodes coincide
    # with the baseline uniform hex nodes where they overlap.
    clip_bbox_tuple = None
    if clip_bbox is not None:
        bbox_lo, bbox_hi = clip_bbox
        clip_bbox_tuple = (
            tuple(float(x) for x in bbox_lo),
            tuple(float(x) for x in bbox_hi),
        )
        mesh_center, k_offsets = _snap_mesh_center_to_bbox_lattice(
            mesh_lo,
            mesh_hi,
            clip_bbox_tuple[0],
            clip_bbox_tuple[1],
            h_target,
            octree_depth,
        )
        print(
            f"  Remesh: clip_bbox={clip_bbox_tuple[0]} → {clip_bbox_tuple[1]} "
            f"→ snapped mesh_center={mesh_center}, k={k_offsets}"
        )

    # ISOTROPIC cube: L = h_target * 2^d so each cube leaf has physical side h_target.
    L_cube = h_target * (2**octree_depth)
    half_L = L_cube / 2.0
    axis_scale = (half_L, half_L, half_L)
    # geo_min[i] = -mesh_center[i] / axis_scale[i] s.t. cube=0 ↔ phys=mesh_center.
    geo_min = tuple(-mesh_center[i] / axis_scale[i] for i in range(3))

    axis_scale_t = torch.tensor(axis_scale, device=device, dtype=torch.float64)
    geo_min_t = torch.tensor(geo_min, device=device, dtype=torch.float64)

    cube_phys_lo = tuple(mesh_center[i] - half_L for i in range(3))
    cube_phys_hi = tuple(mesh_center[i] + half_L for i in range(3))
    if any(mesh_lo[i] < cube_phys_lo[i] - 1e-9 for i in range(3)) or any(
        mesh_hi[i] > cube_phys_hi[i] + 1e-9 for i in range(3)
    ):
        raise ValueError(
            f"Mesh bounds {mesh_lo} → {mesh_hi} fall outside cube "
            f"{cube_phys_lo} → {cube_phys_hi} (depth={octree_depth}, h={h_target})"
        )
    sdf_truncation = resolve_explicit_surface_sdf_truncation(
        h_target, sdf_truncation
    )
    print(
        f"  Remesh: isotropic cube L={L_cube:.4f}, depth={octree_depth}, "
        f"leaf={h_target:.4f}, mesh fits inside "
        f"{cube_phys_lo} → {cube_phys_hi}"
    )
    print(
        f"  Remesh: physical SDF truncation={sdf_truncation:.6g}, "
        f"refinement band={0.9 * sdf_truncation:.6g}"
    )

    # Warp BVH geo_func: converts cube-space pts → physical → compute_nodal_sdf.
    # _keep_alive is captured in the closure to prevent garbage collection.
    def cube_geo_func(pts_cube):
        # pts_cube: (1, M, 3) float16 cube-space points
        pts_phys = (
            (pts_cube.squeeze(0).to(dtype=torch.float64, device=device) - geo_min_t)
            * axis_scale_t
        ).float()
        return (
            compute_nodal_sdf(
                wp_mesh, pts_phys, negate, max_dist=10.0 * bbox_diag, device=device
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )

    cube_geo_func._keep_alive = _keep_alive  # prevent warp array GC

    # A direct surface supplies exact feature locations that the coarse SDF
    # band detector may miss.  Convert its vertices to normalized cube space
    # and deduplicate them into final-parent must-query centres.  The extractor
    # supplies their ancestry; the uniform path needs no help because every
    # cell is already at final depth.
    surface_vertices_norm = surface_vertices_phys / axis_scale_t + geo_min_t
    surface_mq, surface_mq_counts = _surface_vertices_to_must_query_positions(
        surface_vertices_norm,
        init_octree_depth=init_octree_depth,
        octree_depth=octree_depth,
    )
    if surface_mq_counts:
        counts_text = ", ".join(f"d{d}:{n}" for d, n in surface_mq_counts)
        print(
            f"  Remesh: forcing {surface_vertices_phys.shape[0]} surface vertices "
            f"through {surface_mq.shape[0]} parent groups ({counts_text})"
        )

    # Build cube-space octree.  grid_sdf is the dense (2^d+1)^3 cube grid;
    # node_sdf is the per-node value at the active octree corners.  Both are
    # in physical SDF units (since the geo_func returns physical distances).
    grid_sdf, anchors = hierarchical_extraction(
        geometric_func=cube_geo_func,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_octree_depth,
        octree_depth=octree_depth,
        device=device,
        must_query_positions=surface_mq,
        sdf_truncation=sdf_truncation,
    )
    bg_n = 2**octree_depth + 1
    bg_grid_phys = grid_sdf.to(device=device, dtype=torch.float32)

    # Isotropic embedding: physical spacing = h_target uniformly.
    grid_spacing = (float(h_target), float(h_target), float(h_target))

    result = {
        "bg_grid_sdf": bg_grid_phys,
        "init_grid_h16": grid_sdf,   # float16 (bg_n,bg_n,bg_n) for sdf2octree reuse
        "init_anchors": anchors,      # cell refinement levels for sdf2octree reuse
        "axis_scale": list(axis_scale),
        "geo_min": list(geo_min),
        "grid_shape": (bg_n, bg_n, bg_n),
        "grid_spacing": grid_spacing,
    }
    if clip_bbox is not None:
        result["clip_bbox"] = clip_bbox_tuple

    if not build_octree:
        return result

    nodes_cube, node_sdf, elements, edge_h, face_h = sdf2octree(
        cell_tensor=anchors,
        sdf_tensor=grid_sdf,
        domain_length=2.0,
    )

    # Scale octree nodes from cube to physical (anisotropic).
    nodes_phys = (nodes_cube.to(torch.float64) - geo_min_t) * axis_scale_t

    octree_mesh = {
        "nodes": nodes_phys,
        "elements": elements,
        "sdf": node_sdf.float(),
        "grid_sdf": grid_sdf,
        "existing_anchors": anchors,
    }
    if edge_h is not None and len(edge_h) > 0:
        octree_mesh["edge_hanging"] = edge_h
    if face_h is not None and len(face_h) > 0:
        octree_mesh["face_hanging"] = face_h

    # Clip the outer octree to the caller-supplied physical bbox so the FEM
    # mesh only covers the region baseline covers.  The bg grid is left full-
    # cube: Sussman / hierarchical_extraction keep operating on it each iter.
    if clip_bbox_tuple is not None:
        n_elems_before = int(octree_mesh["elements"].shape[0])
        n_nodes_before = int(octree_mesh["nodes"].shape[0])
        octree_mesh = clip_octree_to_bbox(
            octree_mesh, clip_bbox_tuple[0], clip_bbox_tuple[1]
        )
        n_elems_after = int(octree_mesh["elements"].shape[0])
        n_nodes_after = int(octree_mesh["nodes"].shape[0])
        print(
            f"  Remesh clip: elements {n_elems_before} → {n_elems_after}, "
            f"nodes {n_nodes_before} → {n_nodes_after}"
        )

    result["octree_mesh"] = octree_mesh
    return result


# ---------------------------------------------------------------------------
# BC graft helpers (used by both LatentShapeOpt and ShapeOptTrimmed)
# ---------------------------------------------------------------------------


def extract_boundary_loops(faces_t):
    """Extract ordered boundary loops from a triangle mesh.

    GPU ops identify boundary directed edges (O(F log F) sort + searchsorted).
    The sequential loop walk runs on CPU using a torch ``next_v`` scatter — no
    numpy, no dict.  The boundary edge set is small (O(loop length)), so the
    single GPU→CPU transfer is cheap.

    Args:
        faces_t: ``(F, 3)`` int tensor on any device.

    Returns:
        List of loops; each loop is a list of vertex indices where consecutive
        pairs form a boundary directed edge matching the source face winding.
    """
    if faces_t.shape[0] == 0:
        return []

    f = faces_t.to(torch.int64)
    directed = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    V = int(f.max().item()) + 1
    fwd_keys = directed[:, 0] * V + directed[:, 1]
    rev_keys = directed[:, 1] * V + directed[:, 0]
    fwd_sorted, _ = fwd_keys.sort()
    idx = torch.searchsorted(fwd_sorted, rev_keys)
    idx_clamped = idx.clamp(max=max(fwd_sorted.shape[0] - 1, 0))
    found = fwd_sorted[idx_clamped] == rev_keys
    boundary = directed[~found]  # (E, 2) — small, O(loop length)

    if boundary.shape[0] == 0:
        return []

    # Single transfer: boundary edges only (small, O(loop length)).
    bnd_list = boundary.cpu().tolist()  # list of [src, dst] pairs

    # Build adjacency using a dict of lists to correctly handle non-manifold
    # boundary vertices (corners of the BC cut may have 2 outgoing edges).
    out_edges: dict = {}
    for a, b in bnd_list:
        out_edges.setdefault(a, []).append(b)

    loops = []
    while out_edges:
        start_v = next(iter(out_edges))
        loop = [start_v]
        cur = start_v
        while True:
            nxts = out_edges.get(cur)
            if not nxts:
                break
            nxt = nxts.pop()
            if not nxts:
                del out_edges[cur]
            if nxt == start_v:
                break
            loop.append(nxt)
            cur = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def extract_boundary_loops_strict(faces_t):
    """Extract only closed boundary loops and report topology validity.

    Boundary edges are directed according to face winding.  A valid closed
    manifold-with-boundary loop set must satisfy per-boundary-vertex
    in-degree==1 and out-degree==1.  Any branch, dangling chain, or repeated
    vertex in a walk marks the topology invalid.

    Args:
        faces_t: ``(F, 3)`` integer tensor.

    Returns:
        tuple ``(loops, is_valid, reason)`` where:
          - loops: list of closed loops (each is list[int])
          - is_valid: bool
          - reason: short diagnostic string when invalid
    """
    if faces_t.shape[0] == 0:
        return [], True, "empty"

    f = faces_t.to(torch.int64)
    directed = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    V = int(f.max().item()) + 1
    fwd_keys = directed[:, 0] * V + directed[:, 1]
    rev_keys = directed[:, 1] * V + directed[:, 0]
    fwd_sorted, _ = fwd_keys.sort()
    idx = torch.searchsorted(fwd_sorted, rev_keys)
    idx_clamped = idx.clamp(max=max(fwd_sorted.shape[0] - 1, 0))
    found = fwd_sorted[idx_clamped] == rev_keys
    boundary = directed[~found]

    if boundary.shape[0] == 0:
        return [], True, "no-boundary"

    bnd = boundary.cpu().tolist()

    out_next = {}
    in_deg = {}
    out_deg = {}
    for a, b in bnd:
        out_deg[a] = out_deg.get(a, 0) + 1
        in_deg[b] = in_deg.get(b, 0) + 1
        if a in out_next:
            return [], False, "branching-out-degree"
        out_next[a] = b

    boundary_vertices = set(list(out_deg.keys()) + list(in_deg.keys()))
    for v in boundary_vertices:
        if in_deg.get(v, 0) != 1 or out_deg.get(v, 0) != 1:
            return [], False, "non-manifold-boundary-degree"

    loops = []
    visited = set()
    for start in list(out_next.keys()):
        if start in visited:
            continue
        loop = []
        cur = start
        local_seen = set()
        while True:
            if cur in local_seen:
                return [], False, "self-intersection-or-repeat"
            local_seen.add(cur)
            visited.add(cur)
            loop.append(cur)

            nxt = out_next.get(cur)
            if nxt is None:
                return [], False, "open-chain"
            if nxt == start:
                break
            cur = nxt

        if len(loop) < 3:
            return [], False, "degenerate-loop"
        loops.append(loop)

    return loops, True, "ok"


def _connected_components_tri_faces(faces_t):
    """Return connected components of a triangle-face set by face adjacency.

    Faces are adjacent if they share at least one vertex.
    """
    if faces_t.shape[0] == 0:
        return []

    faces_list = faces_t.to(torch.int64).cpu().tolist()
    n_faces = len(faces_list)
    v2f = {}
    for fi, tri in enumerate(faces_list):
        for v in tri:
            v2f.setdefault(v, []).append(fi)

    visited = [False] * n_faces
    components = []
    for seed in range(n_faces):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            tri = faces_list[cur]
            for v in tri:
                for nb in v2f.get(v, []):
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
        components.append(comp)

    return components


def _loop_perimeter(verts, loop):
    """Return the perimeter length of an ordered closed loop."""
    if len(loop) < 2:
        return 0.0
    device = verts.device
    idx = torch.tensor(loop, dtype=torch.int64, device=device)
    pts = verts[idx].float()
    return float((pts - torch.roll(pts, -1, 0)).norm(dim=1).sum().item())


def _cap_loop_fan(verts, loop):
    """Cap a single ordered loop with a best-fit-plane centroid fan.

    Returns:
        ``(verts_new, cap_tris_t)`` where ``cap_tris_t`` is ``(N, 3)``.
    """
    device = verts.device
    loop_t = torch.tensor(loop, dtype=torch.int64, device=device)
    pts = verts[loop_t].float()
    N = pts.shape[0]

    centroid3d = pts.mean(0)
    centred = pts - centroid3d.unsqueeze(0)
    _, _, Vh = torch.linalg.svd(centred, full_matrices=False)
    u_axis = Vh[0]
    v_axis = Vh[1]

    coords2d = torch.stack([
        (centred * u_axis).sum(-1),
        (centred * v_axis).sum(-1),
    ], dim=1)
    angles = torch.atan2(coords2d[:, 1], coords2d[:, 0])
    order = torch.argsort(angles)

    sorted_loop = loop_t[order]
    c_idx = int(verts.shape[0])
    verts = torch.cat([verts, centroid3d.to(verts.dtype).unsqueeze(0)], dim=0)

    c_col = torch.full((N,), c_idx, dtype=torch.int64, device=device)
    cap_t = torch.stack([c_col, torch.roll(sorted_loop, -1, 0), sorted_loop], dim=1)
    return verts, cap_t


def _classify_inner_hole_loops(verts, hole_loops):
    """Return a boolean mask marking loops that are nested inside others.

    Nesting uses the same AABB containment rule as ``_cap_inner_hole_loops``.
    """
    if len(hole_loops) < 2:
        return torch.zeros(len(hole_loops), dtype=torch.bool, device=verts.device)

    device = verts.device
    idx_t = [torch.tensor(lp, dtype=torch.int64, device=device) for lp in hole_loops]

    lo = torch.stack([verts[idx].min(0).values for idx in idx_t])  # (L, 3)
    hi = torch.stack([verts[idx].max(0).values for idx in idx_t])  # (L, 3)
    nested_lo = lo[:, None, :] >= lo[None, :, :]   # (L, L, 3)
    nested_hi = hi[:, None, :] <= hi[None, :, :]   # (L, L, 3)
    is_nested = (nested_lo & nested_hi).all(dim=2)  # (L, L)
    is_nested.fill_diagonal_(False)
    return is_nested.any(dim=1)


def count_non_manifold_edges(faces_t):
    """Return the number of edges not shared by exactly 2 faces.

    Zero means the mesh is closed and manifold.
    """
    if faces_t.shape[0] == 0:
        return 0
    f = faces_t.to(torch.int64)
    directed = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    undirected, _ = directed.sort(dim=1)
    V = int(f.max().item()) + 1
    keys = undirected[:, 0] * V + undirected[:, 1]
    _, counts = torch.unique(keys, return_counts=True)
    return int((counts != 2).sum().item())


def filter_non_watertight_components(faces_t, keep_largest=True):
    """Drop connected components whose edges aren't all shared by exactly
    two faces.  The largest CC (typically the body) is kept unconditionally
    when ``keep_largest=True`` to avoid catastrophic loss when MC produces
    a near-watertight body surface with a few sliver-edge defects.

    Returns a face-index tensor (LongTensor) selecting the surviving faces.
    """
    if faces_t.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=faces_t.device)
    ccs = _connected_components_tri_faces(faces_t)
    if not ccs:
        return torch.zeros(0, dtype=torch.long, device=faces_t.device)
    biggest = max(ccs, key=len) if keep_largest else None
    keep_ids = []
    for cc in ccs:
        if cc is biggest:
            keep_ids.extend(cc)
            continue
        sub = faces_t[torch.tensor(cc, dtype=torch.long, device=faces_t.device)]
        if count_non_manifold_edges(sub) == 0:
            keep_ids.extend(cc)
    return torch.tensor(sorted(keep_ids), dtype=torch.long, device=faces_t.device)


def _check_no_nested_loops(verts, loops):
    """Raise if any boundary loop's AABB is strictly contained in another's.

    A loop nested inside another indicates a topological hole in the BC patch
    (genus > 0), which the downstream zipper cannot handle.

    Uses axis-aligned bounding-box containment as a fast proxy.  This is
    sufficient for BC patches on roughly-planar surfaces (e.g. beam faces).
    All pair-wise comparisons are batched into a single ``(L, L, 3)`` tensor op.
    """
    device = verts.device
    idx_t = [torch.tensor(lp, dtype=torch.int64, device=device) for lp in loops]
    lo = torch.stack([verts[idx].min(0).values for idx in idx_t])  # (L, 3)
    hi = torch.stack([verts[idx].max(0).values for idx in idx_t])  # (L, 3)

    # is_nested[i, j] = True iff AABB(i) ⊆ AABB(j) for all 3 axes
    nested_lo = lo[:, None, :] >= lo[None, :, :]   # (L, L, 3)
    nested_hi = hi[:, None, :] <= hi[None, :, :]   # (L, L, 3)
    is_nested = (nested_lo & nested_hi).all(dim=2)  # (L, L)
    is_nested.fill_diagonal_(False)

    if is_nested.any():
        pair = is_nested.nonzero()[0]
        i, j = int(pair[0]), int(pair[1])
        raise RuntimeError(
            f"BC zip init failed: boundary loop {i} is nested inside "
            f"loop {j} — check physical BC settings or mesh resolution."
        )


def compute_bc_patch(verts_phys, faces, frozen_node_positions_phys, radius,
                     grow_rings=2):
    """Extract the BC-region patch from an iter-0 surface mesh.

    Selects surface vertices within ``radius`` of any frozen BC node and the
    faces whose three corners are all BC verts.  Renumbers patch-local indices
    to ``0..K-1`` and extracts boundary loops for later zipper stitching.

    The raw vertex-proximity selection can leave bowtie / pinch vertices on
    small patches (two patch sheets meeting at a single vertex), whose boundary
    is non-manifold and fails ``extract_boundary_loops_strict`` with
    ``branching-out-degree``.  ``grow_rings`` dilates the selected patch face set
    by that many edge-adjacency rings before boundary extraction: growing fills
    the fan around each pinch vertex so it becomes interior, yielding a clean
    manifold disk with a simple boundary loop (and a larger, more robust removal
    target for the later stitch).  Note that morphological *closing*
    (dilate+erode) does not help — erosion re-opens the pinch; pure dilation
    does.  Only the restored patch grows, not the physical BC selector region.
    Set ``grow_rings=0`` for the exact legacy behaviour.

    Args:
        verts_phys: ``(V, 3)`` float64 tensor — iter-0 surface vertices in
            physical space.
        faces: ``(F, 3)`` int tensor.
        frozen_node_positions_phys: ``(N, 3)`` float64 — physical positions of
            BC-frozen nodes.
        radius: float — search radius in physical units.
        grow_rings: int — edge-adjacency dilation rings applied to the patch
            face set (default 2).  0 reproduces the legacy vertex-proximity
            patch exactly.

    Returns:
        dict with keys ``verts_phys`` (K, 3), ``faces`` (F', 3), ``n_verts``
        (K), ``radius``, ``boundary_loops_local`` (list of loops in patch-local
        indices).  Returns ``None`` if no surface verts fall within ``radius``.
    """
    device = verts_phys.device
    V = verts_phys.shape[0]
    faces_i64 = faces.to(torch.int64)

    dists = torch.cdist(verts_phys.float(), frozen_node_positions_phys.float())
    bc_mask = dists.min(dim=1).values < radius
    K = int(bc_mask.sum())
    if K == 0:
        return None

    face_mask = (
        bc_mask[faces_i64[:, 0]] & bc_mask[faces_i64[:, 1]] & bc_mask[faces_i64[:, 2]]
    )
    # Dilate the patch face set by ``grow_rings`` edge-adjacency rings so small
    # patches don't leave bowtie/pinch vertices on their boundary (see docstring).
    # The patch vertex set is then all vertices touched by the grown face set.
    if grow_rings > 0 and int(face_mask.sum()) > 0:
        adj = _build_face_adj_sparse(faces_i64)
        for _ in range(grow_rings):
            reached = torch.sparse.mm(
                adj, face_mask.float().unsqueeze(1)
            ).squeeze(1) > 0
            face_mask = face_mask | reached
        bc_mask = torch.zeros(V, dtype=torch.bool, device=device)
        bc_mask[torch.unique(faces_i64[face_mask].reshape(-1))] = True
        K = int(bc_mask.sum())

    bc_faces_old = faces_i64[face_mask]
    new_idx = torch.full((V,), -1, dtype=torch.int64, device=device)
    new_idx[bc_mask] = torch.arange(K, device=device)
    bc_faces_new = new_idx[bc_faces_old].to(torch.int32).contiguous()

    patch_verts = verts_phys[bc_mask].clone()
    boundary_loops, ok, reason = extract_boundary_loops_strict(bc_faces_new)
    if not ok:
        raise RuntimeError(f"BC zip init failed: invalid patch boundary ({reason})")
    if len(boundary_loops) >= 2:
        _check_no_nested_loops(patch_verts, boundary_loops)

    frozen_f = frozen_node_positions_phys.float()

    # Split BC patch into connected components. Keep only components that
    # expose boundary loops (closed components do not participate in zippering).
    components = []
    face_components = _connected_components_tri_faces(bc_faces_new)
    for cid, face_ids in enumerate(face_components):
        comp_faces_global = bc_faces_new[
            torch.tensor(face_ids, dtype=torch.int64, device=device)
        ]
        comp_vert_ids = torch.unique(comp_faces_global.reshape(-1))

        comp_new_idx = torch.full((K,), -1, dtype=torch.int64, device=device)
        comp_new_idx[comp_vert_ids] = torch.arange(
            comp_vert_ids.shape[0], device=device
        )
        comp_faces_local = comp_new_idx[comp_faces_global].to(torch.int32).contiguous()
        comp_verts = patch_verts[comp_vert_ids]

        comp_loops_local, comp_ok, comp_reason = extract_boundary_loops_strict(
            comp_faces_local
        )
        if not comp_ok:
            raise RuntimeError(
                f"BC zip init failed: invalid component boundary (cid={cid}, {comp_reason})"
            )
        if len(comp_loops_local) == 0:
            continue
        if len(comp_loops_local) >= 2:
            _check_no_nested_loops(comp_verts, comp_loops_local)

        # Component-local frozen subset for per-component excision.
        d_comp = torch.cdist(comp_verts.float(), frozen_f)
        near_any = d_comp.min(dim=0).values < radius
        comp_frozen = frozen_node_positions_phys[near_any]
        if comp_frozen.shape[0] == 0:
            # Fallback: keep all frozen nodes for robustness.
            comp_frozen = frozen_node_positions_phys

        comp_frozen_f = comp_frozen.float()
        components.append(
            {
                "cid": int(cid),
                "verts_phys": comp_verts.clone(),
                "faces": comp_faces_local,
                "n_verts": int(comp_verts.shape[0]),
                "boundary_loops_local": comp_loops_local,
                "frozen_phys": comp_frozen,
                "frozen_aabb_min": comp_frozen_f.min(0).values,
                "frozen_aabb_max": comp_frozen_f.max(0).values,
                "centroid": comp_verts.float().mean(0),
            }
        )

    return {
        "verts_phys": patch_verts,
        "faces": bc_faces_new,
        "n_verts": K,
        "radius": float(radius),
        "boundary_loops_local": boundary_loops,
        "components": components,
        "frozen_aabb_min": frozen_f.min(0).values,
        "frozen_aabb_max": frozen_f.max(0).values,
    }


def _build_face_adj_sparse(faces_t: torch.Tensor) -> torch.Tensor:
    """Build a sparse face-face adjacency matrix for BFS ring expansion.

    Two faces are adjacent if they share an edge.  The returned matrix has
    shape ``(F, F)`` with float32 values (1.0 per adjacent pair); it is
    symmetric and suitable for ``torch.sparse.mm``.

    Args:
        faces_t: ``(F, 3)`` int64 tensor.

    Returns:
        ``(F, F)`` sparse COO float32 tensor (coalesced).
    """
    F = faces_t.shape[0]
    device = faces_t.device
    if F == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (0, 0),
            device=device,
        ).coalesce()

    V = int(faces_t.max().item()) + 1

    # 3 directed edges per face
    v0 = faces_t[:, [0, 1, 2]]  # (F, 3)
    v1 = faces_t[:, [1, 2, 0]]  # (F, 3)
    fi = torch.arange(F, device=device).unsqueeze(1).expand(-1, 3)

    starts = v0.reshape(-1)
    ends = v1.reshape(-1)
    fi_flat = fi.reshape(-1)

    # Undirected edge keys: (min, max) encoded
    ea = torch.minimum(starts, ends)
    eb = torch.maximum(starts, ends)
    ukeys = ea * V + eb  # (3F,)

    sorted_keys, sort_perm = ukeys.sort()
    sorted_fi = fi_flat[sort_perm]

    # Consecutive identical keys → shared edge → face pair
    same = sorted_keys[1:] == sorted_keys[:-1]  # (3F-1,)
    pair_pos = same.nonzero(as_tuple=True)[0]
    fi_a = sorted_fi[pair_pos]
    fi_b = sorted_fi[pair_pos + 1]

    rows = torch.cat([fi_a, fi_b])
    cols = torch.cat([fi_b, fi_a])
    vals = torch.ones(rows.shape[0], dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (F, F), device=device
    ).coalesce()


def _close_open_loops(verts_t, faces_t):
    """Close every pre-existing open boundary loop with a centroid fan.

    For each open loop in ``faces_t``, insert a centroid vertex and fan-
    triangulate to the loop vertices.  Winding is chosen so cap normals
    match the outward orientation of the surrounding mesh.

    Args:
        verts_t: ``(V, 3)`` float tensor.
        faces_t: ``(F, 3)`` int tensor.

    Returns:
        ``(verts_new, faces_new)`` — closed mesh.  When ``faces_t`` has no
        open loops, returns the inputs unchanged.
    """
    loops = extract_boundary_loops(faces_t)
    if not loops:
        return verts_t, faces_t

    device = verts_t.device
    new_v = [verts_t]
    new_f = [faces_t]
    V = verts_t.shape[0]

    for loop in loops:
        lp = torch.tensor(loop, dtype=torch.int64, device=device)
        cen = verts_t[lp].mean(0, keepdim=True)  # (1, 3)
        new_v.append(cen)

        N = lp.shape[0]
        cid = V
        V += 1
        lp_next = torch.roll(lp, -1, 0)

        # Directed boundary edges run lp[i] → lp_next[i]; cap fan winds
        # [centroid, lp_next, lp] so its normal matches the outward
        # orientation of adjacent faces.
        fan = torch.stack(
            [
                torch.full((N,), cid, dtype=torch.int32, device=device),
                lp_next.to(torch.int32),
                lp.to(torch.int32),
            ],
            dim=1,
        )
        new_f.append(fan)

    return torch.cat(new_v, dim=0), torch.cat(new_f, dim=0)


def _zipper_stitch(h_loop, p_loop_nat, h_pos_t, p_pos_nat_t):
    """Zipper-stitch a hole loop to a patch loop via Fuchs-Kedem-Uselton DP.

    Finds the minimum-cost cylindrical triangulation, producing a watertight,
    non-self-intersecting bridge regardless of loop-size mismatch.  Cost per
    triangle: sum of squared edge lengths.

    Args:
        h_loop: ordered list of hole-loop vertex indices (global).
        p_loop_nat: ordered list of patch-loop vertex indices (global, natural
            face-winding order — reversed internally).
        h_pos_t: ``(H, 3)`` float tensor — positions of h_loop verts.
        p_pos_nat_t: ``(P, 3)`` float tensor — positions of p_loop_nat verts.

    Returns:
        List of ``(a, b, c)`` int tuples with consistent manifold winding.
    """
    H = len(h_loop)
    P = len(p_loop_nat)
    if H < 3 or P < 3:
        return []

    p_loop = list(reversed(p_loop_nat))
    p_t = torch.flip(p_pos_nat_t.float(), [0])
    h_t = h_pos_t.float()

    j_start = int(((p_t - h_t[0:1]) ** 2).sum(1).argmin().item())
    if j_start:
        p_loop = p_loop[j_start:] + p_loop[:j_start]
        p_t = torch.roll(p_t, -j_start, 0)

    h_ext = torch.cat([h_t, h_t[:1]], 0)
    p_ext = torch.cat([p_t, p_t[:1]], 0)
    D = ((h_ext[:, None, :] - p_ext[None, :, :]) ** 2).sum(-1)  # (H+1, P+1)

    sq_hh = ((h_ext[1:] - h_ext[:-1]) ** 2).sum(-1)
    sq_pp = ((p_ext[1:] - p_ext[:-1]) ** 2).sum(-1)
    cost_h = sq_hh[:, None] + D[:H, :] + D[1:, :]
    cost_p = D[:, :P] + sq_pp[None, :] + D[:, 1:]

    INF = float("inf")
    dp = torch.full((H + 1, P + 1), INF)
    par = torch.zeros((H + 1, P + 1), dtype=torch.bool)
    dp[0, 0] = 0.0

    for d in range(1, H + P + 1):
        i_lo = max(0, d - P)
        i_hi = min(H, d)
        i_arr = torch.arange(i_lo, i_hi + 1)
        j_arr = d - i_arr
        fh = torch.full((len(i_arr),), INF)
        fp = torch.full((len(i_arr),), INF)
        mh = i_arr >= 1
        if mh.any():
            ih, jh = i_arr[mh] - 1, j_arr[mh]
            fh[mh] = dp[ih, jh] + cost_h[ih, jh]
        mp = j_arr >= 1
        if mp.any():
            ip, jp = i_arr[mp], j_arr[mp] - 1
            fp[mp] = dp[ip, jp] + cost_p[ip, jp]
        use_p = fp < fh
        dp[i_arr, j_arr] = torch.minimum(fh, fp)
        par[i_arr, j_arr] = use_p

    par_np = par.numpy()
    tris = []
    i, j = H, P
    while i > 0 or j > 0:
        if i > 0 and (j == 0 or not par_np[i, j]):
            tris.append((h_loop[i % H], h_loop[i - 1], p_loop[j % P]))
            i -= 1
        else:
            tris.append((h_loop[i % H], p_loop[j - 1], p_loop[j % P]))
            j -= 1
    return tris


def _subdivide_bridge_faces(
    faces_t: torch.Tensor,
    verts_t: torch.Tensor,
    bridge_mask: torch.Tensor,
    threshold: float,
    max_iter: int = 10,
) -> tuple:
    """Crack-free red-green midpoint subdivision for bridge faces.

    Each iteration:

    1. Classify every undirected edge of the mesh by the number of bridge
       faces it touches.  *Internal* bridge edges are shared by exactly two
       bridge faces; *external* bridge edges are shared with a non-bridge
       neighbour (or lie on the mesh boundary).
    2. Mark internal bridge edges whose length exceeds ``threshold``.
    3. Split each marked edge at its midpoint (deduplicated globally).
    4. Re-triangulate every bridge face according to how many of its three
       edges are marked — a red-green template guaranteeing no T-junctions:

       * 0 marks → keep unchanged
       * 1 mark  → 2 sub-triangles (longest-edge bisection)
       * 2 marks → 3 sub-triangles (corner + quad split)
       * 3 marks → 4 sub-triangles (regular loop-style)

    External bridge edges are never marked, so neighbouring non-bridge
    (base / patch) triangles never get a hanging vertex on their edges.

    Args:
        faces_t: ``(F, 3)`` int32 — all active faces.
        verts_t: ``(V, 3)`` float64 — vertex positions.
        bridge_mask: ``(F,)`` bool — which faces are bridge.
        threshold: float — edge-length threshold.
        max_iter: safety cap on outer iterations.
    """
    device = faces_t.device

    for _ in range(max_iter):
        F = faces_t.shape[0]
        V = verts_t.shape[0]
        f64 = faces_t.to(torch.int64)

        # Build the face-edge tensor in a fixed (F, 3) layout so we can later
        # look up "midpoint for edge k of face i" by indexing `face_mids[i, k]`.
        e0 = torch.stack([f64[:, 0], f64[:, 1]], dim=1)  # edge 0: v0-v1
        e1 = torch.stack([f64[:, 1], f64[:, 2]], dim=1)  # edge 1: v1-v2
        e2 = torch.stack([f64[:, 2], f64[:, 0]], dim=1)  # edge 2: v2-v0
        e_all = torch.cat([e0, e1, e2], dim=0)  # (3F, 2) — blocks by edge slot
        e_und = torch.stack(
            [torch.minimum(e_all[:, 0], e_all[:, 1]),
             torch.maximum(e_all[:, 0], e_all[:, 1])],
            dim=1,
        )

        V_big = max(V + 1, 2)
        e_keys = e_und[:, 0] * V_big + e_und[:, 1]
        uniq_keys, inv = torch.unique(e_keys, return_inverse=True)
        n_edges = uniq_keys.shape[0]

        face_id = torch.arange(F, device=device).repeat(3)  # (3F,)
        is_bridge_edge_entry = bridge_mask[face_id]

        bridge_count = torch.zeros(n_edges, dtype=torch.int64, device=device)
        bridge_count.scatter_add_(0, inv, is_bridge_edge_entry.to(torch.int64))
        total_count = torch.zeros(n_edges, dtype=torch.int64, device=device)
        total_count.scatter_add_(0, inv, torch.ones_like(inv))
        # Internal to bridge iff both incident faces are bridge.
        is_internal = (bridge_count == 2) & (total_count == 2)

        # Edge length at the first occurrence of each unique edge.
        first_occ = torch.scatter_reduce(
            torch.full((n_edges,), e_keys.shape[0], dtype=torch.int64, device=device),
            0,
            inv,
            torch.arange(e_keys.shape[0], device=device),
            reduce="amin",
            include_self=True,
        )
        edge_lens = (
            verts_t[e_und[first_occ, 0]] - verts_t[e_und[first_occ, 1]]
        ).norm(dim=1)

        to_split = is_internal & (edge_lens > threshold)
        if not to_split.any():
            return faces_t, verts_t, bridge_mask

        split_ids = to_split.nonzero(as_tuple=True)[0]
        n_new = split_ids.shape[0]
        mid_pos = 0.5 * (
            verts_t[e_und[first_occ[split_ids], 0]]
            + verts_t[e_und[first_occ[split_ids], 1]]
        )
        verts_t = torch.cat([verts_t, mid_pos], dim=0)

        edge_to_mid = torch.full((n_edges,), -1, dtype=torch.int64, device=device)
        edge_to_mid[split_ids] = V + torch.arange(n_new, device=device)

        m_e0 = edge_to_mid[inv[0:F]]
        m_e1 = edge_to_mid[inv[F : 2 * F]]
        m_e2 = edge_to_mid[inv[2 * F : 3 * F]]
        face_mids = torch.stack([m_e0, m_e1, m_e2], dim=1)  # (F, 3)
        split_cnt = (face_mids >= 0).sum(dim=1)

        new_faces_parts = []
        new_bridge_parts = []

        # Case 0: unchanged.
        m0 = split_cnt == 0
        if m0.any():
            new_faces_parts.append(f64[m0])
            new_bridge_parts.append(bridge_mask[m0])

        # Case 1: one edge split → 2 triangles.  Rotate so split edge is e0.
        m1 = split_cnt == 1
        if m1.any():
            f_ = f64[m1]
            m_ = face_mids[m1]
            split_idx = (m_ >= 0).to(torch.int64).argmax(dim=1)  # (N,)
            rot = (torch.arange(3, device=device).unsqueeze(0)
                   + split_idx.unsqueeze(1)) % 3
            fv = torch.gather(f_, 1, rot)
            fm = torch.gather(m_, 1, rot)
            v0 = fv[:, 0]; v1 = fv[:, 1]; v2 = fv[:, 2]
            mA = fm[:, 0]
            t1 = torch.stack([v0, mA, v2], dim=1)
            t2 = torch.stack([mA, v1, v2], dim=1)
            new_faces_parts.extend([t1, t2])
            br = bridge_mask[m1]
            new_bridge_parts.extend([br, br])

        # Case 2: two edges split → 3 triangles.  Rotate so the non-split edge
        # is e2 (v2→v0); then e0 and e1 are the two split edges.
        #
        # With ``fv[i] = f_[(i + rot_amount) % 3]`` the new edge j equals old
        # edge ``(j + rot_amount) % 3``.  We need new e2 == old e_not_split, so
        # ``rot_amount = (not_split_idx + 1) % 3``.  A prior version used
        # ``(2 - not_split_idx) % 3`` which is correct only when
        # ``not_split_idx == 2``; the other two cases mis-rotated so that the
        # not-split edge landed at new position 1 and ``mB`` was taken from the
        # -1 sentinel slot, producing triangles with an out-of-range vertex
        # and visible diamond-shaped holes / non-manifold edges.
        m2 = split_cnt == 2
        if m2.any():
            f_ = f64[m2]
            m_ = face_mids[m2]
            not_split_idx = (m_ < 0).to(torch.int64).argmax(dim=1)
            rot_amount = (not_split_idx + 1) % 3
            rot = (torch.arange(3, device=device).unsqueeze(0)
                   + rot_amount.unsqueeze(1)) % 3
            fv = torch.gather(f_, 1, rot)
            fm = torch.gather(m_, 1, rot)
            v0 = fv[:, 0]; v1 = fv[:, 1]; v2 = fv[:, 2]
            mA = fm[:, 0]  # midpoint on v0-v1
            mB = fm[:, 1]  # midpoint on v1-v2
            # Sub-triangles (preserve CCW winding of [v0,v1,v2]):
            t1 = torch.stack([v0, mA, mB], dim=1)
            t2 = torch.stack([v0, mB, v2], dim=1)
            t3 = torch.stack([mA, v1, mB], dim=1)
            new_faces_parts.extend([t1, t2, t3])
            br = bridge_mask[m2]
            new_bridge_parts.extend([br, br, br])

        # Case 3: all edges split → 4 triangles (regular loop split).
        m3 = split_cnt == 3
        if m3.any():
            f_ = f64[m3]
            m_ = face_mids[m3]
            v0 = f_[:, 0]; v1 = f_[:, 1]; v2 = f_[:, 2]
            mA = m_[:, 0]  # on v0-v1
            mB = m_[:, 1]  # on v1-v2
            mC = m_[:, 2]  # on v2-v0
            t1 = torch.stack([v0, mA, mC], dim=1)
            t2 = torch.stack([mA, v1, mB], dim=1)
            t3 = torch.stack([mC, mB, v2], dim=1)
            t4 = torch.stack([mA, mB, mC], dim=1)
            new_faces_parts.extend([t1, t2, t3, t4])
            br = bridge_mask[m3]
            new_bridge_parts.extend([br, br, br, br])

        faces_t = torch.cat(new_faces_parts, dim=0).to(faces_t.dtype)
        bridge_mask = torch.cat(new_bridge_parts, dim=0)

    return faces_t, verts_t, bridge_mask


def _cap_inner_hole_loops(verts, hole_loops):
    """Cap inner (nested) hole loops with a projected Delaunay-like fan and return outer loops.

    After excising a candidate region that enclosed an island (non-candidate
    faces surrounded by the candidate), the candidate boundary contains both an
    outer loop and inner loops around the island.  Inner loops must not be
    stitched to the patch — they are sealed by a cap so the island's boundary
    is closed.

    **Nesting detection** (vectorised): builds ``(L, L, 3)`` AABB containment
    tensors in a single broadcast; loop ``i`` is inner iff its AABB is
    strictly contained in another loop's AABB.

    **Cap triangulation**: each inner loop is closed by a minimum-area fan
    from the loop's best-fit-plane centroid, computed fully without Python
    iteration over triangles:

    1. Project loop vertices onto the loop's best-fit plane (SVD of the
       centred positions) to get 2-D coordinates.
    2. Sort them by polar angle around the projected centroid so the fan is
       non-self-intersecting even for non-convex (but simple) holes.
    3. Create ``(N, 3)`` triangle index tensor in one ``torch.stack`` call.
    4. The centroid is added as a new global vertex.

    The cap triangles are returned with label 2 and go through the same
    Taubin smoothing pass as stitch triangles, which relaxes any initial
    metric distortion.

    Args:
        verts: ``(V, 3)`` float64 — global vertex positions.  One new centroid
            vertex is appended per capped inner loop.
        hole_loops: list of directed hole-loop vertex-index lists.

    Returns:
        ``(verts, outer_loops, cap_tris_t)`` where ``outer_loops`` are the
        non-nested loops for stitching, and ``cap_tris_t`` is an ``(M, 3)``
        int64 tensor of cap triangle indices (empty if no inner loops found).
    """
    if len(hole_loops) < 2:
        empty = torch.zeros((0, 3), dtype=torch.int64, device=verts.device)
        return verts, hole_loops, empty

    device = verts.device
    is_inner = _classify_inner_hole_loops(verts, hole_loops)

    if not is_inner.any():
        empty = torch.zeros((0, 3), dtype=torch.int64, device=verts.device)
        return verts, hole_loops, empty

    all_cap_tris = []
    outer_loops = []

    for i, (loop, inner) in enumerate(zip(hole_loops, is_inner.tolist())):
        if not inner:
            outer_loops.append(loop)
            continue

        verts, cap_t = _cap_loop_fan(verts, loop)
        all_cap_tris.append(cap_t)

    if all_cap_tris:
        cap_tris_t = torch.cat(all_cap_tris, dim=0)        # (M, 3) int64
    else:
        cap_tris_t = torch.zeros((0, 3), dtype=torch.int64, device=device)

    return verts, outer_loops, cap_tris_t


def _bridge_two_hole_loops(verts, loop_a, loop_b, target_edge_length=None):
    """Merge two hole loops by inserting a triangle-strip bridge at the closest edge pair.

    **Edge-pair selection**: a fully vectorised ``(N×M)`` cost matrix
    ``||a_i - b_j||² + ||a_{i+1} - b_{j+1}||²`` finds the best pair.

    **Strip construction**: the bridge spans a quad whose "cross" edges are
    ``a_i→b_{j+1}`` and ``a_{i+1}→b_j``.  If ``target_edge_length`` is given
    and either cross edge exceeds it, both are subdivided into ``n`` segments
    with ``n-1`` evenly-spaced midpoint vertices each, producing a ``2n``
    triangle strip whose boundary edges match the target size.  With
    ``target_edge_length=None`` or when the cross edges are already short
    enough, the bridge collapses to the minimal 2-triangle quad.

    **Topology** (per strip row ``k = 0 … n-1``):

    - ``T_a = (r2[k], r1[k], r1[k+1])``
    - ``T_b = (r2[k], r1[k+1], r2[k+1])``
    - ``r1`` rail: ``a_i → … → b_{j+1}`` (boundary of merged loop)
    - ``r2`` rail: ``a_{i+1} → … → b_j`` (boundary, reversed direction)
    - Row 0 pairs consumed ``a_i→a_{i+1}``; row ``n-1`` pairs consumed
      ``b_j→b_{j+1}``; adjacent rows share interior edges.

    Args:
        verts: ``(V, 3)`` float64 tensor — global vertex positions.  May be
            extended with new midpoint vertices (returned as first element).
        loop_a, loop_b: lists of global vertex indices (directed hole-loop,
            CCW around hole).
        target_edge_length: float or ``None``.  When set, bridge boundary
            edges are subdivided to stay within this length.

    Returns:
        ``(verts, merged_loop, bridge_tris)`` where ``verts`` may have grown,
        ``merged_loop`` is a single directed loop (list of ints, length
        ``N + M + 2*(n-1)``), and ``bridge_tris`` is a list of ``2n``
        ``(int, int, int)`` tuples with consistent manifold winding.
    """
    device = verts.device
    N, M = len(loop_a), len(loop_b)
    a_pos = verts[torch.tensor(loop_a, dtype=torch.int64, device=device)].float()  # (N, 3)
    b_pos = verts[torch.tensor(loop_b, dtype=torch.int64, device=device)].float()  # (M, 3)

    # "next" positions via cyclic roll — fully vectorised.
    a_next = torch.roll(a_pos, -1, 0)  # (N, 3)  positions of a_{i+1}
    b_next = torch.roll(b_pos, -1, 0)  # (M, 3)  positions of b_{j+1}

    # (N, M) cost matrix: ||a_i - b_j||² + ||a_{i+1} - b_{j+1}||²
    d_ij  = ((a_pos[:, None, :] - b_pos[None, :, :]) ** 2).sum(-1)    # (N, M)
    d_ij1 = ((a_next[:, None, :] - b_next[None, :, :]) ** 2).sum(-1)  # (N, M)
    cost  = d_ij + d_ij1  # (N, M)

    flat_idx = int(cost.argmin().item())
    i_star, j_star = flat_idx // M, flat_idx % M

    ai  = loop_a[i_star]
    ai1 = loop_a[(i_star + 1) % N]
    bj  = loop_b[j_star]
    bj1 = loop_b[(j_star + 1) % M]

    # Positions of the four quad corners (float64 for vertex creation).
    pos_ai  = verts[ai]
    pos_ai1 = verts[ai1]
    pos_bj  = verts[bj]
    pos_bj1 = verts[bj1]

    # Determine strip subdivision count n.
    L1 = float((pos_ai.float() - pos_bj1.float()).norm().item())   # cross edge a_i → b_{j+1}
    L2 = float((pos_ai1.float() - pos_bj.float()).norm().item())   # cross edge a_{i+1} → b_j
    if target_edge_length is not None and target_edge_length > 0:
        n = max(1, round(max(L1, L2) / target_edge_length))
    else:
        n = 1

    # Build rail vertex indices.  r1: a_i → b_{j+1};  r2: a_{i+1} → b_j.
    # Endpoints are existing vertices; interior points (k=1…n-1) are new.
    if n == 1:
        r1_ids = [ai, bj1]
        r2_ids = [ai1, bj]
    else:
        # Vectorised interpolation of 2*(n-1) new midpoints.
        t = torch.linspace(0, 1, n + 1, device=device, dtype=verts.dtype)[1:-1]  # (n-1,)
        new_r1 = pos_ai.unsqueeze(0) + t.unsqueeze(1) * (pos_bj1 - pos_ai).unsqueeze(0)  # (n-1, 3)
        new_r2 = pos_ai1.unsqueeze(0) + t.unsqueeze(1) * (pos_bj - pos_ai1).unsqueeze(0)  # (n-1, 3)
        new_pts = torch.cat([new_r1, new_r2], dim=0)  # (2*(n-1), 3)
        V0 = verts.shape[0]
        verts = torch.cat([verts, new_pts], dim=0)

        r1_mid = list(range(V0, V0 + n - 1))
        r2_mid = list(range(V0 + n - 1, V0 + 2 * (n - 1)))
        r1_ids = [ai] + r1_mid + [bj1]   # length n+1
        r2_ids = [ai1] + r2_mid + [bj]    # length n+1

    # Build strip triangles (2n triangles, consistent winding).
    # Row k: T_a = (r2[k], r1[k], r1[k+1]),  T_b = (r2[k], r1[k+1], r2[k+1])
    bridge_tris = []
    for k in range(n):
        bridge_tris.append((r2_ids[k], r1_ids[k], r1_ids[k + 1]))
        bridge_tris.append((r2_ids[k], r1_ids[k + 1], r2_ids[k + 1]))

    # Merged loop: A rotated from a_{i+1}, then r1 midpoints (a_i → b_{j+1}),
    # then B rotated from b_{j+1}, then r2 midpoints reversed (b_j → a_{i+1}).
    ip1 = (i_star + 1) % N
    jp1 = (j_star + 1) % M
    a_rot = loop_a[ip1:] + loop_a[:ip1]  # ends with a_i
    b_rot = loop_b[jp1:] + loop_b[:jp1]  # ends with b_j

    # Rail 1 boundary (a_i → … → b_{j+1}): midpoints only (endpoints are in a_rot/b_rot).
    # Rail 2 boundary (b_j → … → a_{i+1}): midpoints reversed.
    r1_mid_ids = r1_ids[1:-1]                   # interior midpoints of rail 1
    r2_mid_ids = list(reversed(r2_ids[1:-1]))    # interior midpoints of rail 2, reversed

    merged_loop = a_rot + r1_mid_ids + b_rot + r2_mid_ids

    return verts, merged_loop, bridge_tris


def _cluster_and_bridge_hole_loops(
    verts, hole_loops, n_target, patch_centroids, target_edge_length=None,
):
    """Cluster hole loops into ``n_target`` groups and bridge within each group.

    **Assignment**: stack all hole-loop centroids into ``(N, 3)`` and all patch
    centroids into ``(M, 3)``, compute a single ``torch.cdist`` call for the
    ``(N, M)`` distance matrix, then ``argmin(dim=1)`` assigns each hole loop
    to the nearest patch loop.  Empty groups raise ``RuntimeError``.

    **Merging within a group**: centroids of the ``K`` current loops are stacked
    into ``(K, 3)`` and a ``(K, K)`` upper-triangle pairwise-distance matrix
    finds the closest pair in one vectorised ``argmin`` — no Python loop over
    pairs.  Loops are merged iteratively with ``_bridge_two_hole_loops`` until
    each group contains at most two loops; if two remain, the smaller one (by
    perimeter) is capped and removed from the stitching set.

    Args:
        verts: ``(V, 3)`` float64 — global vertex positions.  May grow if
            ``target_edge_length`` triggers midpoint insertion in bridges.
        hole_loops: list of N directed hole-loop vertex-index lists, N > n_target.
        n_target: int — desired output loop count (= number of patch loops).
        patch_centroids: list of n_target ``(3,)`` float32 tensors, one per
            patch loop, used for group assignment.
        target_edge_length: float or ``None``.  Forwarded to
            ``_bridge_two_hole_loops`` to keep bridge boundary edges within
            this length.

    Returns:
        ``(verts, merged_loops, all_bridge_tris, cap_tris)`` where ``verts``
        may have grown, ``merged_loops`` has ``n_target`` entries,
        ``all_bridge_tris`` is a flat list of bridge triangles, and
        ``cap_tris`` is a flat list of cap triangles generated when a cluster
        ends with exactly two loops.
    """
    device = verts.device

    # Vectorised hole-loop centroids: (N, 3).
    hole_cents = torch.stack([
        verts[torch.tensor(lp, dtype=torch.int64, device=device)].float().mean(0)
        for lp in hole_loops
    ])

    # Patch centroids: (M, 3).
    patch_cents_t = torch.stack(patch_centroids).to(device=device, dtype=torch.float32)

    # Single cdist → (N, M); argmin(dim=1) → per-hole assignment.
    assignments = torch.cdist(hole_cents, patch_cents_t).argmin(dim=1).tolist()

    groups = [[] for _ in range(n_target)]
    for k, g in enumerate(assignments):
        groups[g].append(k)

    for g, grp in enumerate(groups):
        if len(grp) == 0:
            raise RuntimeError(
                f"BC zip bridge-merge: patch loop {g} has no assigned hole "
                f"loops — degenerate mesh topology."
            )

    merged_loops = []
    all_bridge_tris = []
    all_cap_tris = []

    for grp in groups:
        cur_loops = [hole_loops[k] for k in grp]

        while len(cur_loops) > 2:
            K = len(cur_loops)
            # Vectorised pairwise distances: (K, K) upper-triangle masked.
            cents = torch.stack([
                verts[torch.tensor(lp, dtype=torch.int64, device=device)].float().mean(0)
                for lp in cur_loops
            ])
            pdist = torch.cdist(cents, cents)  # (K, K)
            triu_mask = torch.ones(K, K, dtype=torch.bool, device=device).triu(diagonal=1)
            pdist = torch.where(triu_mask, pdist, torch.full_like(pdist, float("inf")))
            flat = int(pdist.argmin().item())
            ii, jj = flat // K, flat % K

            verts, merged, bridge_tris = _bridge_two_hole_loops(
                verts, cur_loops[ii], cur_loops[jj],
                target_edge_length=target_edge_length,
            )
            all_bridge_tris.extend(bridge_tris)
            cur_loops = [cur_loops[k] for k in range(K) if k not in (ii, jj)]
            cur_loops.append(merged)

        if len(cur_loops) == 2:
            perims = [
                _loop_perimeter(verts, cur_loops[0]),
                _loop_perimeter(verts, cur_loops[1]),
            ]
            cap_idx = 0 if perims[0] <= perims[1] else 1
            keep_idx = 1 - cap_idx
            verts, cap_t = _cap_loop_fan(verts, cur_loops[cap_idx])
            all_cap_tris.extend(cap_t.tolist())
            merged_loops.append(cur_loops[keep_idx])
        else:
            merged_loops.append(cur_loops[0])

    return verts, merged_loops, all_bridge_tris, all_cap_tris


def _fair_bridge_faces(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_labels: torch.Tensor,
    order: int,
    base_rings: int,
    cg_iters: int,
    cg_tol: float,
    direct_max_unknowns: int,
    log=None,
) -> torch.Tensor:
    """Higher-order fairing of the bridge strip against clamped boundary rings.

    Minimises ``∫|Δ^(order−1) ∇x|²`` over the strip's free vertices, discretised
    as ``L (M⁻¹ L)^(order−1)`` with cotangent weights ``L`` and barycentric
    masses ``M`` — the operator is symmetric for every order.  ``order=2`` is
    the thin-plate
    (bi-Laplacian) problem and leaves the strip **tangent-continuous (G1)** with
    the surfaces it joins; ``order=3`` adds curvature continuity (G2).

    Free vertices are exactly the strip's interior ones — any bridge vertex
    shared with a non-bridge face stays pinned, so the exact BC patch and the
    base surface are never moved.  The face set is grown ``order`` rings past
    the strip so the stencil of a free vertex beside a seam reaches real
    geometry; those rings are the clamped data that fixes the tangent (and, at
    ``order=3``, the curvature).  This is what the plain Taubin path cannot do:
    it stops the graph at the seam, so the boundary carries position only.

    ``L`` is symmetric PSD, hence its principal submatrix on the free set is SPD
    and plain CG applies with no normal equations.  The solve is matrix-free
    apart from one coalesced sparse operator, batched over all three coordinates
    in a single ``mm`` per matvec, warm-started from the current positions, and
    free of any per-vertex Python loop; convergence is polled every 25 iterations
    so the device syncs 25× less often than the iteration count.

    Args:
        verts: ``(V, 3)`` float64 vertex positions.
        faces: ``(F, 3)`` int32 faces of the assembled mesh.
        face_labels: ``(F,)`` int32 — 0 base, 1 patch, 2 zip/bridge.
        order: 2 for G1 (thin-plate), 3 for G2.
        base_rings: how many rings of base surface beyond the strip may also
            move.  0 pins both seam rings; the patch side is pinned regardless.
        cg_iters: iteration cap.
        cg_tol: relative residual tolerance.

    Returns:
        ``(V, 3)`` vertices with the free strip interior faired.
    """
    device = verts.device
    f64 = faces.to(torch.int64)
    n_v = verts.shape[0]

    bridge_mask = face_labels == 2
    bridge_v = torch.unique(f64[bridge_mask].reshape(-1))

    # The exact BC patch may never move: anything a patch face touches — which
    # includes the whole patch-side seam ring — is pinned for good.
    patch_locked = torch.zeros(n_v, dtype=torch.bool, device=device)
    patch_locked[torch.unique(f64[face_labels == 1].reshape(-1))] = True

    # Free = the strip's interior.  Fairing that alone pins *both* seam rings,
    # which fixes the surface normal on both sides and leaves the transition as
    # a corner however high the order goes — the strip can only bend between two
    # clamped ends.  `base_rings` releases that on the base side only: the base
    # surface is the optimiser's own evolving geometry, so letting the fillet run
    # a few rings up into it is what actually makes the junction smooth.
    free_mask = torch.zeros(n_v, dtype=torch.bool, device=device)
    free_mask[bridge_v] = True
    free_mask[torch.unique(f64[~bridge_mask].reshape(-1))] = False
    # Seed the growth from the strip's *seam rings*, not from its interior: no
    # base face touches an interior vertex, so seeding there never fires.
    base_mask = face_labels == 0
    seed = torch.zeros(n_v, dtype=torch.bool, device=device)
    seed[bridge_v] = True
    for _ in range(max(base_rings, 0)):
        grow = base_mask & seed[f64].any(dim=1)
        new_v = f64[grow].reshape(-1)
        free_mask[new_v] = True
        seed[new_v] = True
    free_mask &= ~patch_locked
    free_v = free_mask.nonzero(as_tuple=True)[0]
    if free_v.numel() == 0:
        return verts

    # Grow the face set `order` rings past the free set: a free vertex's L^order
    # stencil reaches `order` hops, so the clamped data must extend that far.
    vmask = free_mask | torch.isin(torch.arange(n_v, device=device), bridge_v)
    fsel = bridge_mask.clone()
    for _ in range(order + max(base_rings, 0)):
        fsel |= vmask[f64].any(dim=1)
        vmask[f64[fsel].reshape(-1)] = True

    sub = f64[fsel]

    # Cotangent weights and barycentric vertex areas.  Uniform D − A weights
    # look equivalent and are not: they ignore how long the strip's edges
    # actually are, and on the slivers the zipper leaves behind they drag
    # vertices straight through their neighbours (measured: 178° dihedrals,
    # i.e. flipped faces).  Block k of `ea` is the edge opposite corner k, so
    # the cotangent at that corner is exactly the weight it contributes.
    ea = torch.cat([sub[:, [1, 2]], sub[:, [2, 0]], sub[:, [0, 1]]], dim=0)
    ea, _ = ea.sort(dim=1)
    e, inv = torch.unique(ea, dim=0, return_inverse=True)

    p = verts[sub]                                        # (Fs, 3, 3)
    d0, d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]
    dbl_a = torch.cross(d0, -d2, dim=1).norm(dim=1).clamp(min=1e-30)
    cot = torch.cat([(d0 * -d2).sum(1), (d1 * -d0).sum(1), (d2 * -d1).sum(1)])
    cot = 0.5 * cot / dbl_a.repeat(3)

    touched = torch.unique(torch.cat([e.reshape(-1), free_v]))
    n = touched.shape[0]
    g2l = torch.full((n_v,), -1, dtype=torch.int64, device=device)
    g2l[touched] = torch.arange(n, device=device)
    u, v = g2l[e[:, 0]], g2l[e[:, 1]]

    w = torch.zeros(e.shape[0], dtype=verts.dtype, device=device)
    w.scatter_add_(0, inv, cot)
    # Obtuse triangles give negative cotangents, which would cost L its positive
    # semi-definiteness and with it the right to use CG.  Clamp, then lift by a
    # scale-relative floor so no edge of the graph goes fully slack.
    w = w.clamp(min=0.0)
    w = w + 1e-6 * w.mean()

    deg_w = torch.zeros(n, dtype=verts.dtype, device=device)
    deg_w.scatter_add_(0, u, w)
    deg_w.scatter_add_(0, v, w)
    diag = torch.arange(n, device=device)
    L = torch.sparse_coo_tensor(
        torch.stack([torch.cat([u, v, diag]), torch.cat([v, u, diag])]),
        torch.cat([-w, -w, deg_w]),
        (n, n),
        device=device,
    ).coalesce()

    # Barycentric mass: a third of each incident face's area.
    mass = torch.zeros(n, dtype=verts.dtype, device=device)
    mass.scatter_add_(0, g2l[sub].reshape(-1),
                      (dbl_a / 6.0).repeat_interleave(3))
    m_inv = 1.0 / mass.clamp(min=1e-30).unsqueeze(1)

    fl = g2l[free_v]

    # Assemble ``A = L (M⁻¹ L)^(order−1)`` once instead of replaying `order`
    # mat-vecs every iteration: it costs one sparse-sparse product per extra
    # order up front, makes each iteration a single spmm, and is the only way to
    # get ``diag(A)`` for the preconditioner.  ``M`` is diagonal, so ``A`` is
    # symmetric and its free-set principal submatrix is SPD.
    A = L
    if order > 1:
        ml = torch.sparse_coo_tensor(
            torch.stack([diag, diag]), m_inv.squeeze(1), (n, n), device=device
        ).coalesce()
        ml = torch.sparse.mm(ml, L).coalesce()          # M⁻¹ L
        for _ in range(order - 1):
            A = torch.sparse.mm(A, ml).coalesce()

    x = verts[touched]
    z0 = x.clone()
    z0[fl] = 0.0
    b = -torch.sparse.mm(A, z0)[fl]       # contribution of the clamped rings

    n_f = fl.shape[0]
    loc = torch.full((n,), -1, dtype=torch.int64, device=device)
    loc[fl] = torch.arange(n_f, device=device)

    ai, av = A._indices(), A._values()
    keep = (loc[ai[0]] >= 0) & (loc[ai[1]] >= 0)
    A_ff = torch.sparse_coo_tensor(
        torch.stack([loc[ai[0][keep]], loc[ai[1][keep]]]), av[keep],
        (n_f, n_f), device=device,
    ).coalesce()

    # Jacobi preconditioning.  The biharmonic conditions like h⁻⁴, so undamped
    # CG stalls far short of the clamped solution — measured at order 2, the
    # strip was still sitting on the zipper's original cone after 20 000
    # iterations, which reads as "the solver did nothing" rather than as a
    # failure.  Scaling by the diagonal is what makes it actually arrive.
    d = torch.zeros(n_f, dtype=verts.dtype, device=device)
    ii, vv = A_ff._indices(), A_ff._values()
    on_diag = ii[0] == ii[1]
    d.scatter_add_(0, ii[0][on_diag], vv[on_diag])
    m_pc = torch.where(d > 0, 1.0 / d, torch.ones_like(d)).unsqueeze(1)

    # The strip is local — this system is a few tens of thousands of unknowns
    # even on a large mesh (13 214 free of a 60 k-vertex mesh on the figure).  At
    # that size the biharmonic is a *direct*-solve problem: sparse LU factors it
    # in ~150 ms to machine precision, where Jacobi-PCG needs O(h⁻²) iterations —
    # 20 000 of them, ~10 s — to reach a worse answer.  CG remains the fallback
    # for when SciPy is absent, or the strip is big enough that fill-in would
    # not fit.
    y = None
    if n_f <= direct_max_unknowns:
        try:
            import numpy as _np
            import scipy.sparse as _sp
            import scipy.sparse.linalg as _spl

            idx = A_ff._indices().cpu().numpy()
            lu = _spl.splu(_sp.coo_matrix(
                (A_ff._values().cpu().numpy(), (idx[0], idx[1])),
                shape=(n_f, n_f)).tocsc())
            b_np = b.cpu().numpy()
            y = torch.as_tensor(
                _np.stack([lu.solve(b_np[:, k]) for k in range(3)], axis=1),
                dtype=verts.dtype, device=device)
        except Exception as exc:  # noqa: BLE001 - any SciPy/memory failure
            if log is not None:
                log(f"  BC zip: direct fairing solve unavailable ({exc}); using CG")
            y = None

    if y is not None:
        verts = verts.clone()
        verts[free_v] = y
        return verts

    y = x[fl].clone()                     # warm start: the zipper's own strip
    r = b - torch.sparse.mm(A_ff, y)
    z = m_pc * r
    p = z.clone()
    rz = (r * z).sum(dim=0)
    tol = (cg_tol * cg_tol) * (b * b).sum(dim=0).clamp(min=1e-300)
    for it in range(cg_iters):
        Ap = torch.sparse.mm(A_ff, p)
        den = (p * Ap).sum(dim=0)
        alpha = rz / torch.where(den == 0, torch.ones_like(den), den)
        y = y + alpha * p
        r = r - alpha * Ap
        if it % 25 == 24 and bool(((r * r).sum(dim=0) <= tol).all()):
            break
        z = m_pc * r
        rz_new = (r * z).sum(dim=0)
        p = z + (rz_new / torch.where(rz == 0, torch.ones_like(rz), rz)) * p
        rz = rz_new

    verts = verts.clone()
    verts[free_v] = y
    return verts


def stitch_bc_patches(
    base_verts_phys: torch.Tensor,
    base_faces: torch.Tensor,
    bc_patches: list,
    aabb_scale: float = 1.1,
    subdivision_size_ratio: float = 3.0,
    smooth_iters: int = 10,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    smooth_order: int = 1,
    smooth_base_rings: int = 0,
    smooth_cg_iters: int = 20000,
    smooth_direct_max: int = 400000,
    smooth_cg_tol: float = 1e-8,
    log=None,
) -> tuple:
    """Stitch stored BC surface patches into the current base mesh.

    Workflow per patch:

    1. Scale the patch vertex AABB by ``aabb_scale`` and select active base
       faces with at least one vertex in the scaled box.
    2. Validate: the candidate region must be a single connected component
       with the patch's number of boundary loops, and removing it must leave
       the base mesh manifold.  If not, expand the candidate by one BFS ring
       and re-check — repeat until valid.
    3. Remove the candidate faces.  The hole loops in the kept mesh are the
       candidate's own boundary loops, reversed.
    4. Append all patch vertices as new global vertices, remap patch faces
       (label 1), and stitch each hole loop to its centroid-nearest patch
       loop via the FKU DP (``_zipper_stitch``).  Stitch triangles get
       label 2 (bridge).
    5. Subdivide bridge triangles whose max edge exceeds
       ``subdivision_size_ratio × avg_base_edge``.
    6. Apply Taubin λ/μ smoothing to bridge vertices (fixing all others).
       The alternating positive λ + negative μ steps smooth kinks without
       the shrinkage that a plain Laplacian smoother would cause.

    Args:
        base_verts_phys: ``(V, 3)`` float64 in physical space.  The caller
            is expected to have already closed any pre-existing open loops.
        base_faces: ``(F, 3)`` int32.
        bc_patches: list of component dicts with keys ``verts_phys``,
            ``faces``, ``boundary_loops_local``.
        aabb_scale: multiplier on the patch AABB half-extents when selecting
            the base faces to remove.  Sets how far past the patch boundary the
            removal reaches — the axial gap between the hole loop and the patch
            loop is ``(aabb_scale - 1) / 2`` of the patch's own extent — and so
            how gradually the zip strip can span a change in cross-section.
        subdivision_size_ratio: edge-length multiplier for bridge refinement.
        smooth_iters: number of Taubin iterations (each applies ``L`` twice,
            once with λ and once with μ).
        taubin_lambda: positive smoothing step (typical 0.3–0.6).
        taubin_mu: negative un-shrinking step with ``|μ| > λ`` (typical −0.53).
        smooth_order: continuity of the zip strip against the surfaces it joins.
            ``1`` (default) is the Taubin λ/μ pass — a membrane energy pinned on
            one ring, so the strip is smooth inside but only **G0** at the two
            seams.  ``2`` solves the thin-plate problem ``Δ²x = 0`` with two
            clamped rings per side for **G1**; ``3`` solves ``Δ³x = 0`` with
            three for **G2**.  Orders above 1 ignore ``smooth_iters`` and the
            Taubin coefficients.  Note the seam *angle* is set by the geometry
            (see ``aabb_scale``), not by this — raising the order rounds the
            corner into a fillet, it does not shorten the run.
        smooth_base_rings: with ``smooth_order > 1``, how many rings of *base*
            surface beyond the strip are allowed to move so the fillet can run
            into it.  ``0`` (default) pins both seam rings, which holds the
            normal fixed on both sides and leaves the junction a corner no
            matter how high the order.  The exact BC patch is never released.
        smooth_cg_iters: iteration cap for the ``smooth_order > 1`` CG solve.
            A *cap*, not a target — the solve exits on ``smooth_cg_tol`` and
            usually long before this.  It has to be generous: the biharmonic
            conditions like h⁻⁴, and an under-converged solve does not look like
            a failure, it looks like the fairing having no effect.
        smooth_cg_tol: relative residual tolerance for that solve.
        smooth_direct_max: free-vertex count up to which the fairing system is
            solved by sparse LU instead of CG.  The strip is local, so this is
            the normal path; CG is the fallback when SciPy is missing or the
            strip is larger than this.
        log: optional callable ``log(msg)`` used for diagnostic messages
            (skipped patches, non-manifold warnings).  Pass the calling
            object's ``log`` method so messages route through its logger.

    Returns:
        ``(out_verts, out_faces, face_labels)`` where ``face_labels`` is
        int32 with values 0 (base), 1 (bc), 2 (bridge).
    """
    if not bc_patches:
        labels = torch.zeros(
            base_faces.shape[0], dtype=torch.int32, device=base_verts_phys.device
        )
        return base_verts_phys, base_faces, labels

    device = base_verts_phys.device

    # Caller is expected to have already filtered out non-watertight
    # non-largest CCs (typically via ``filter_non_watertight_components``
    # before ``_close_open_loops``).  Watertight non-largest CCs
    # (e.g. legitimate interior voids carved by the phase-field design
    # path) are passed through unchanged: candidate-region selection
    # (closest CC to the patch centroid) and the per-patch validator
    # below naturally exclude them from the BC excision, and the
    # ``_cap_inner_hole_loops`` / ``_cluster_and_bridge_hole_loops``
    # paths handle any extra hole loops the excision happens to produce.

    verts = base_verts_phys.clone()
    base_all = base_faces
    base_active = torch.ones(base_all.shape[0], dtype=torch.bool, device=device)

    # Appended BC/bridge geometry (accumulated, cat at end).
    extra_faces: list = []
    extra_labels: list = []

    for patch in bc_patches:
        patch_verts = patch["verts_phys"].to(device=device, dtype=torch.float64)
        patch_loops_local = patch["boundary_loops_local"]
        patch_faces_local = patch["faces"].to(torch.int64).to(device)
        n_loops = len(patch_loops_local)
        if n_loops == 0:
            continue

        # B.1: scaled AABB of patch vertices.
        pv_min = patch_verts.min(0).values
        pv_max = patch_verts.max(0).values
        half = (pv_max - pv_min) * 0.5 * aabb_scale
        c = (pv_min + pv_max) * 0.5
        aabb_lo, aabb_hi = c - half, c + half

        # B.2: initial candidate = active base faces with >=1 vert in AABB.
        cur_faces = base_all[base_active]  # (A, 3) int32
        cur_f64 = cur_faces.to(torch.int64)
        v_in = ((verts >= aabb_lo) & (verts <= aabb_hi)).all(-1)  # (V,)
        cand = (
            v_in[cur_f64[:, 0]] | v_in[cur_f64[:, 1]] | v_in[cur_f64[:, 2]]
        )  # (A,)

        # Restrict initial candidate to the connected component whose centroid is
        # closest to the patch centroid (MC mesh may have multiple disconnected
        # pieces, and the AABB can span more than one).
        if cand.any():
            cand_idx = cand.nonzero(as_tuple=True)[0]      # (Nc,) indices into cur_faces
            ccs = _connected_components_tri_faces(cur_faces[cand_idx])
            if len(ccs) > 1:
                patch_center = (pv_min + pv_max) * 0.5
                best_cc, best_d = None, float("inf")
                for cc in ccs:
                    cc_t = torch.tensor(cc, dtype=torch.int64, device=device)
                    cc_global = cand_idx[cc_t]
                    cc_center = verts[cur_f64[cc_global]].mean(dim=1).mean(0)
                    d = float(torch.linalg.norm(cc_center - patch_center).item())
                    if d < best_d:
                        best_d, best_cc = d, cc
                new_cand = torch.zeros_like(cand)
                cc_t = torch.tensor(best_cc, dtype=torch.int64, device=device)
                new_cand[cand_idx[cc_t]] = True
                cand = new_cand

        A_ff = _build_face_adj_sparse(cur_f64)  # (A, A) sparse

        def _validate(mask):
            """``(ok, cand_loops_local, cand_uv, reason)`` — candidate loops
            are in compacted indices; ``cand_uv`` maps compacted → global
            vertex indices.

            Passes only when the candidate is one connected component with
            enough strict boundary loops that, after capping inner loops,
            at least ``n_loops`` loops remain for patch pairing.  Surplus
            loops are reduced by bridge-merging after validation.  The
            remainder's loop count is not required to equal ``n_loops`` —
            prior patches may have left base-mesh holes that are logically
            closed by their appended stitch faces but still show as open
            loops on the base alone.
            """
            cand_f = cur_faces[mask]
            rem_f = cur_faces[~mask]
            if cand_f.shape[0] == 0 or rem_f.shape[0] == 0:
                return False, None, None, "empty"
            if len(_connected_components_tri_faces(cand_f)) != 1:
                return False, None, None, "cand-not-single-cc"
            uv, ui = torch.unique(
                cand_f.to(torch.int64).reshape(-1), return_inverse=True
            )
            c_local = ui.reshape(-1, 3).to(torch.int32)
            c_loops, c_ok, c_reason = extract_boundary_loops_strict(c_local)
            if not c_ok:
                return False, None, None, f"cand-strict-fail:{c_reason}"
            n_inner = int(_classify_inner_hole_loops(verts[uv], c_loops).sum().item())
            if len(c_loops) - n_inner < n_loops:
                return False, None, None, (
                    f"cand-loops={len(c_loops)} inner={n_inner} "
                    f"< required={n_loops} after capping"
                )
            _, ri = torch.unique(rem_f.to(torch.int64).reshape(-1), return_inverse=True)
            r_local = ri.reshape(-1, 3).to(torch.int32)
            _, r_ok, r_reason = extract_boundary_loops_strict(r_local)
            if not r_ok:
                return False, None, None, f"rem-strict-fail:{r_reason}"
            return True, c_loops, uv, "ok"

        ok, cand_loops_local, cand_uv, reason = _validate(cand)

        # B.3: expand by BFS rings until validation passes (or give up).
        MAX_RINGS = 80
        rings = 0
        while not ok and rings < MAX_RINGS:
            reach = torch.sparse.mm(
                A_ff, cand.to(torch.float32).unsqueeze(1)
            ).squeeze(1)
            new = (reach > 0) & ~cand
            if not new.any():
                if not ok:
                    raise RuntimeError(
                        "BC region is being isolated — check physical settings "
                        "or decrease lr"
                    )
                break
            cand = cand | new
            rings += 1
            ok, cand_loops_local, cand_uv, reason = _validate(cand)

        if not ok:
            if log is not None:
                log(
                    f"  BC zip: could not validate removal region "
                    f"(rings={rings}, last_reason={reason}); skipping patch"
                )
            continue

        # B.4: commit removal.
        active_idx = base_active.nonzero(as_tuple=True)[0]  # (A,)
        base_active[active_idx[cand]] = False

        # B.5: hole loops = candidate's boundary loops, reversed, in global indices.
        hole_loops_global = []
        for lp in cand_loops_local:
            g = cand_uv[torch.tensor(lp, dtype=torch.int64, device=device)].tolist()
            hole_loops_global.append(list(reversed(g)))

        # B.5a: cap any inner (nested) hole loops — they border islands, not the patch.
        cap_tris_list = []
        if len(hole_loops_global) >= 2:
            verts, hole_loops_global, cap_tris_t = _cap_inner_hole_loops(verts, hole_loops_global)
            if cap_tris_t.shape[0] > 0:
                cap_tris_list = cap_tris_t.tolist()

        if len(hole_loops_global) < n_loops:
            raise RuntimeError(
                f"BC zip failed: capping reduced hole loops to "
                f"{len(hole_loops_global)} < required {n_loops}"
            )

        # B.5b: bridge-merge excess hole loops to match patch loop count.
        merge_tris = []
        cluster_cap_tris = []
        if len(hole_loops_global) > n_loops:
            # Compute avg base-mesh edge length for bridge strip sizing.
            rem_f_b5 = base_all[base_active].to(torch.int64)
            if rem_f_b5.shape[0] > 0:
                bridge_target_h = float(
                    torch.cat([
                        (verts[rem_f_b5[:, 1]] - verts[rem_f_b5[:, 0]]).norm(dim=1),
                        (verts[rem_f_b5[:, 2]] - verts[rem_f_b5[:, 1]]).norm(dim=1),
                        (verts[rem_f_b5[:, 0]] - verts[rem_f_b5[:, 2]]).norm(dim=1),
                    ]).mean().item()
                )
            else:
                bridge_target_h = None
            patch_cents_bridge = [
                patch_verts[
                    torch.tensor(patch_loops_local[pi], dtype=torch.int64, device=device)
                ].float().mean(0)
                for pi in range(n_loops)
            ]
            verts, hole_loops_global, merge_tris, cluster_cap_tris = _cluster_and_bridge_hole_loops(
                verts, hole_loops_global, n_loops, patch_cents_bridge,
                target_edge_length=bridge_target_h,
            )
            if log is not None:
                log(
                    f"  BC zip: bridged {len(cand_loops_local)} hole loops "
                    f"→ {n_loops} (added {len(merge_tris)} bridge tris, "
                    f"{len(cluster_cap_tris)} cap tris)"
                )

        # B.6: append ALL used patch vertices as new global verts; remap faces (label 1).
        used = torch.unique(patch_faces_local.reshape(-1))  # (U,) int64
        patch_to_global = torch.full(
            (patch_verts.shape[0],), -1, dtype=torch.int64, device=device
        )
        V0 = verts.shape[0]
        patch_to_global[used] = torch.arange(V0, V0 + used.shape[0], device=device)
        verts = torch.cat([verts, patch_verts[used]], dim=0)

        remapped_pf = patch_to_global[patch_faces_local].to(torch.int32)
        extra_faces.append(remapped_pf)
        extra_labels.append(
            torch.ones(remapped_pf.shape[0], dtype=torch.int32, device=device)
        )

        # B.7: 1:1 hole ↔ patch loop matching by centroid; FKU stitch each pair.
        hole_cents = torch.stack(
            [
                verts[torch.tensor(lp, dtype=torch.int64, device=device)].mean(0)
                for lp in hole_loops_global
            ]
        )
        patch_cents = torch.stack(
            [
                patch_verts[torch.tensor(lp, dtype=torch.int64, device=device)].mean(0)
                for lp in patch_loops_local
            ]
        )
        assign = torch.cdist(hole_cents.float(), patch_cents.float()).argmin(dim=1)

        stitch_tris = []
        for hi, pi in enumerate(assign.tolist()):
            h_loop = hole_loops_global[hi]
            p_loop_nat = patch_to_global[
                torch.tensor(patch_loops_local[pi], dtype=torch.int64, device=device)
            ].tolist()
            # _zipper_stitch uses CPU-resident DP tensors; pass positions on CPU.
            h_pos = verts[torch.tensor(h_loop, dtype=torch.int64, device=device)].float().cpu()
            p_pos = verts[torch.tensor(p_loop_nat, dtype=torch.int64, device=device)].float().cpu()
            stitch_tris.extend(_zipper_stitch(h_loop, p_loop_nat, h_pos, p_pos))

        # Include bridge-merge and cap triangles so they go through subdivision+smoothing.
        if merge_tris:
            stitch_tris.extend(merge_tris)
        if cluster_cap_tris:
            cap_tris_list.extend(cluster_cap_tris)
        if cap_tris_list:
            stitch_tris.extend(cap_tris_list)

        if not stitch_tris:
            continue
        st_t = torch.tensor(stitch_tris, dtype=torch.int32, device=device)

        # B.8: subdivide stitch triangles whose max edge exceeds threshold.
        rem_f = base_all[base_active].to(torch.int64)
        if rem_f.shape[0] > 0:
            avg_edge = float(
                torch.cat(
                    [
                        (verts[rem_f[:, 1]] - verts[rem_f[:, 0]]).norm(dim=1),
                        (verts[rem_f[:, 2]] - verts[rem_f[:, 1]]).norm(dim=1),
                        (verts[rem_f[:, 0]] - verts[rem_f[:, 2]]).norm(dim=1),
                    ]
                )
                .mean()
                .item()
            )
            threshold = subdivision_size_ratio * avg_edge
            bm = torch.ones(st_t.shape[0], dtype=torch.bool, device=device)
            st_t, verts, bm = _subdivide_bridge_faces(st_t, verts, bm, threshold)

        extra_faces.append(st_t)
        extra_labels.append(
            torch.full((st_t.shape[0],), 2, dtype=torch.int32, device=device)
        )

    # Assemble full mesh.
    base_part = base_all[base_active]
    base_lbl = torch.zeros(base_part.shape[0], dtype=torch.int32, device=device)
    if extra_faces:
        extra_f = torch.cat(extra_faces, dim=0)
        extra_l = torch.cat(extra_labels, dim=0)
    else:
        extra_f = torch.zeros((0, 3), dtype=torch.int32, device=device)
        extra_l = torch.zeros(0, dtype=torch.int32, device=device)

    out_faces = torch.cat([base_part, extra_f], dim=0)
    face_labels = torch.cat([base_lbl, extra_l], dim=0)

    # Taubin λ/μ smoothing on bridge vertices (fixed everything else).  The
    # alternating signs prevent the inward shrinkage of a plain Laplacian.
    bridge_mask = face_labels == 2
    if smooth_order not in (1, 2, 3):
        raise ValueError(f"smooth_order must be 1, 2 or 3, got {smooth_order}")
    if bridge_mask.any() and smooth_order > 1:
        verts = _fair_bridge_faces(
            verts, out_faces, face_labels, smooth_order, smooth_base_rings,
            smooth_cg_iters, smooth_cg_tol, smooth_direct_max, log,
        )
    elif bridge_mask.any() and smooth_iters > 0:
        bridge_faces_t = out_faces[bridge_mask].to(torch.int64)
        # Free vertices = bridge-triangle vertices.
        free_v_global = torch.unique(bridge_faces_t.reshape(-1))

        # Unique undirected edges among bridge faces for the smoothing graph.
        e01 = torch.stack([bridge_faces_t[:, 0], bridge_faces_t[:, 1]], dim=1)
        e12 = torch.stack([bridge_faces_t[:, 1], bridge_faces_t[:, 2]], dim=1)
        e20 = torch.stack([bridge_faces_t[:, 2], bridge_faces_t[:, 0]], dim=1)
        edges = torch.cat([e01, e12, e20], dim=0)  # (3B, 2)
        edges_und, _ = edges.sort(dim=1)
        edges_und = torch.unique(edges_und, dim=0)  # (E, 2)

        # Compact index space over all vertices touched by bridge edges.
        touched = torch.unique(torch.cat([edges_und.reshape(-1), free_v_global]))
        n_t = touched.shape[0]
        g2l = torch.full(
            (int(touched.max().item()) + 1,), -1, dtype=torch.int64, device=device
        )
        g2l[touched] = torch.arange(n_t, device=device)

        # Free/fixed masks in local (touched) index space.
        is_free_local = torch.zeros(n_t, dtype=torch.bool, device=device)
        is_free_local[g2l[free_v_global]] = True
        # A bridge vertex is "fixed" if it is on the boundary of the bridge
        # region (i.e. shared with a non-bridge face in out_faces).  Detect
        # this by building the full-mesh vertex set and checking adjacency.
        out_f64 = out_faces.to(torch.int64)
        non_bridge_v = torch.unique(out_f64[~bridge_mask].reshape(-1))
        on_boundary_mask_global = torch.zeros(
            out_f64.max().item() + 1, dtype=torch.bool, device=device
        )
        on_boundary_mask_global[non_bridge_v] = True
        is_fixed_local = torch.zeros(n_t, dtype=torch.bool, device=device)
        is_fixed_local[g2l[touched[on_boundary_mask_global[touched]]]] = True
        is_free_local = is_free_local & ~is_fixed_local

        # Graph Laplacian in local space: L x = x − avg_neighbors(x).
        e_loc = g2l[edges_und]  # (E, 2) local
        u = e_loc[:, 0]
        v = e_loc[:, 1]
        rows = torch.cat([u, v])
        cols = torch.cat([v, u])
        ones = torch.ones(rows.shape[0], dtype=verts.dtype, device=device)
        deg = torch.zeros(n_t, dtype=verts.dtype, device=device)
        deg.scatter_add_(0, rows, ones)
        deg_safe = deg.clamp(min=1.0)
        off_vals = -1.0 / deg_safe[rows]
        diag_idx = torch.arange(n_t, device=device)
        L = torch.sparse_coo_tensor(
            torch.stack([torch.cat([rows, diag_idx]), torch.cat([cols, diag_idx])]),
            torch.cat([off_vals, torch.ones(n_t, dtype=verts.dtype, device=device)]),
            (n_t, n_t),
            device=device,
        ).coalesce()

        x = verts[touched].clone()
        x_fixed = x.clone()
        free_idx = is_free_local.nonzero(as_tuple=True)[0]
        for _ in range(smooth_iters):
            Lx = torch.sparse.mm(L, x)
            x[free_idx] = x[free_idx] - taubin_lambda * Lx[free_idx]
            Lx = torch.sparse.mm(L, x)
            x[free_idx] = x[free_idx] - taubin_mu * Lx[free_idx]
        # Re-pin any fixed verts (should be unchanged, but guard against drift).
        x[~is_free_local] = x_fixed[~is_free_local]

        verts = verts.clone()
        verts[touched] = x

    nm = count_non_manifold_edges(out_faces)
    if nm != 0 and log is not None:
        log(f"  BC zip: {nm} non-manifold edges in output mesh")

    # Compact verts: drop any unreferenced ones so downstream (BVH, octree)
    # gets a clean, minimal mesh.
    used_v = torch.unique(out_faces.to(torch.int64).reshape(-1))
    remap = torch.full((verts.shape[0],), -1, dtype=torch.int64, device=device)
    remap[used_v] = torch.arange(used_v.shape[0], device=device)
    out_faces = remap[out_faces.to(torch.int64)].to(torch.int32)
    verts = verts[used_v]

    return verts, out_faces, face_labels
