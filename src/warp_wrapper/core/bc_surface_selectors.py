"""Surface-mesh-based BC region selectors for diffused Nitsche FEM.

These objects act as keys in ``dirichlet_bcs_dict`` / ``neumann_bcs_dict``
in place of manual lambda predicates.  When ``NitscheNeoHookean._build_mesh_state``
runs it calls ``_resolve_surface_selector_bcs()``, which:

1. Builds (or reuses) a marching-cubes surface mesh.
2. Calls ``selector.resolve(verts_phys, faces, eps_expansion)`` for each
   unresolved selector.
3. ``resolve`` computes per-face attributes (centroids, normals, areas, curvatures)
   and a face-adjacency graph (``_build_adj_pairs``), then calls ``select()``
   to obtain a boolean face mask.
4. A Warp BVH is built from the selected triangles, plus one axis-aligned box
   around the whole selected patch, expanded by ``eps_expansion``.
5. ``__call__(x, y, z)`` drops the points outside that box (a far-point cull)
   and accepts the rest whose BVH unsigned distance to the selected triangles
   is ≤ ``eps_expansion``.

Selectors are resolved **once** on the initial MC surface and then frozen;
``_resolve_surface_selector_bcs`` skips already-resolved selectors on
subsequent iterations, keeping the BC region fixed in space during optimization.

Usage
-----
::

    from core.bc_surface_selectors import NormalSelector, LowestSelector

    dirichlet_bcs_dict = {
        NormalSelector((0, -1, 0), 60) & LowestSelector('y', 0.1): 0.0
    }
"""

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_T = torch.Tensor


# ===========================================================================
# Internal helpers
# ===========================================================================

def _to_tensor(v, device="cpu", dtype=torch.float32) -> _T:
    """Convert a sequence or tensor to a float32 tensor on the given device.

    Parameters
    ----------
    v : tensor or sequence
        Input data.
    device : str or torch.device
        Target device.
    dtype : torch.dtype
        Target dtype.

    Returns
    -------
    torch.Tensor
        1-D tensor of the input values.
    """
    if isinstance(v, _T):
        return v.to(device=device, dtype=dtype)
    return torch.tensor(list(v), dtype=dtype, device=device)


def _loaded_area(areas: _T, normals: _T) -> float:
    """``sum(area * max(0, n . n_avg))``, n_avg the area-weighted mean normal."""
    if areas.numel() == 0:
        return 0.0
    mean = (normals * areas[:, None]).sum(dim=0)
    norm = float(mean.norm())
    if norm <= 0.0:
        return 0.0
    return float((areas * (normals @ (mean / norm)).clamp_min(0.0)).sum())


def _compute_face_attributes(
    verts: _T, faces: _T
) -> Tuple[_T, _T, _T]:
    """Compute per-face centroids, unit outward normals, and triangle areas.

    Parameters
    ----------
    verts : Tensor, shape (V, 3), float
        Mesh vertex positions.
    faces : Tensor, shape (F, 3), int
        Triangle face indices.

    Returns
    -------
    centroids : Tensor, shape (F, 3), float
        Face barycentres.
    normals : Tensor, shape (F, 3), float
        Unit face normals (cross product direction).
    areas : Tensor, shape (F,), float
        Triangle areas.
    """
    fi = faces.long()
    v0, v1, v2 = verts[fi[:, 0]], verts[fi[:, 1]], verts[fi[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    area = cross.norm(dim=1) * 0.5
    normals = cross / cross.norm(dim=1, keepdim=True).clamp(min=1e-10)
    centroids = (v0 + v1 + v2) / 3.0
    return centroids, normals, area


def _compute_face_curvatures(verts: _T, faces: _T) -> Optional[_T]:
    """Per-face mean curvature magnitude via the cotangent Laplace-Beltrami formula.

    Parameters
    ----------
    verts : Tensor, shape (V, 3), float
        Mesh vertex positions.
    faces : Tensor, shape (F, 3), int
        Triangle face indices.

    Returns
    -------
    H_face : Tensor, shape (F,), float, or None
        Per-face mean curvature magnitude averaged from vertices.
        Returns ``None`` if computation fails (degenerate mesh, etc.).
    """
    try:
        V = int(verts.shape[0])
        device = verts.device
        fi = faces.long()
        v0, v1, v2 = verts[fi[:, 0]], verts[fi[:, 1]], verts[fi[:, 2]]

        def _cot(a, b, c):
            e1, e2 = b - a, c - a
            cos_t = (e1 * e2).sum(1)
            sin_t = torch.linalg.cross(e1, e2).norm(dim=1).clamp(min=1e-10)
            return cos_t / sin_t

        cot0 = _cot(v0, v1, v2)
        cot1 = _cot(v1, v2, v0)
        cot2 = _cot(v2, v0, v1)

        raw_cross = torch.linalg.cross(v1 - v0, v2 - v0)
        tri_area = raw_cross.norm(dim=1) * 0.5

        A_v = torch.zeros(V, device=device)
        A_v.index_add_(0, fi[:, 0], tri_area / 3)
        A_v.index_add_(0, fi[:, 1], tri_area / 3)
        A_v.index_add_(0, fi[:, 2], tri_area / 3)
        A_v.clamp_(min=1e-10)

        H_vec = torch.zeros(V, 3, device=device)
        for (i, j, cij) in [(0, 1, cot2), (1, 2, cot0), (2, 0, cot1)]:
            vi_idx, vj_idx = fi[:, i], fi[:, j]
            diff_ij = verts[vi_idx] - verts[vj_idx]
            diff_ji = -diff_ij
            H_vec.index_add_(0, vi_idx, cij.unsqueeze(1) * diff_ij)
            H_vec.index_add_(0, vj_idx, cij.unsqueeze(1) * diff_ji)

        H_mag = (H_vec / (2.0 * A_v.unsqueeze(1))).norm(dim=1)
        H_face = (H_mag[fi[:, 0]] + H_mag[fi[:, 1]] + H_mag[fi[:, 2]]) / 3.0
        return H_face
    except Exception:
        return None


def _compute_gaussian_curvature(verts: _T, faces: _T) -> Optional[_T]:
    """Per-face Gaussian curvature via the angle-defect formula, averaged to faces.

    Parameters
    ----------
    verts : Tensor, shape (V, 3), float
        Mesh vertex positions.
    faces : Tensor, shape (F, 3), int
        Triangle face indices.

    Returns
    -------
    K_face : Tensor, shape (F,), float, or None
        Per-face Gaussian curvature.
        Returns ``None`` if computation fails.
    """
    try:
        V = int(verts.shape[0])
        device = verts.device
        fi = faces.long()
        v0, v1, v2 = verts[fi[:, 0]], verts[fi[:, 1]], verts[fi[:, 2]]

        def _angle_at(a, b, c):
            e1 = (b - a)
            e2 = (c - a)
            e1 = e1 / e1.norm(dim=1, keepdim=True).clamp(min=1e-10)
            e2 = e2 / e2.norm(dim=1, keepdim=True).clamp(min=1e-10)
            return torch.acos((e1 * e2).sum(1).clamp(-1.0, 1.0))

        ang0 = _angle_at(v0, v1, v2)
        ang1 = _angle_at(v1, v2, v0)
        ang2 = _angle_at(v2, v0, v1)

        raw_cross = torch.linalg.cross(v1 - v0, v2 - v0)
        tri_area = raw_cross.norm(dim=1) * 0.5

        angle_sum = torch.full((V,), 2.0 * math.pi, device=device)
        A_v = torch.zeros(V, device=device)
        for k, ang in [(0, ang0), (1, ang1), (2, ang2)]:
            idx = fi[:, k]
            angle_sum.index_add_(0, idx, -ang)
            A_v.index_add_(0, idx, tri_area / 3)

        A_v.clamp_(min=1e-10)
        K_v = angle_sum / A_v
        K_face = (K_v[fi[:, 0]] + K_v[fi[:, 1]] + K_v[fi[:, 2]]) / 3.0
        return K_face
    except Exception:
        return None


def _build_adj_pairs(faces: _T) -> _T:
    """Build symmetric directed face-adjacency pairs from a triangle mesh.

    For every shared edge between two faces i and j, both directed pairs
    (i, j) and (j, i) are returned so that downstream code can treat the
    result as a directed graph.

    Parameters
    ----------
    faces : Tensor, shape (F, 3), int
        Triangle face indices.

    Returns
    -------
    adj_pairs : Tensor, shape (2*E, 2), int64
        Each row ``[i, j]`` means face i and face j share an edge.
        Both orientations are present: if ``[i, j]`` is a row, so is ``[j, i]``.
    """
    fi = faces.long()
    F = int(fi.shape[0])
    V = int(fi.max().item()) + 1
    device = fi.device

    # 3 directed edges per face as sorted vertex pairs
    v_lo = torch.minimum(fi[:, [0, 1, 2]], fi[:, [1, 2, 0]])   # (F, 3)
    v_hi = torch.maximum(fi[:, [0, 1, 2]], fi[:, [1, 2, 0]])   # (F, 3)
    v_lo_f = v_lo.reshape(-1)                                    # (3F,)
    v_hi_f = v_hi.reshape(-1)                                    # (3F,)
    fids = torch.arange(F, device=device).unsqueeze(1).expand(-1, 3).reshape(-1)  # (3F,)

    keys = v_lo_f * V + v_hi_f                                   # (3F,) unique edge key
    order = keys.argsort(stable=True)
    skeys = keys[order]
    sfids = fids[order]

    # Adjacent pairs: consecutive entries with the same edge key
    dup = (skeys[:-1] == skeys[1:]).nonzero(as_tuple=True)[0]   # indices of first in pair
    fa = sfids[dup]
    fb = sfids[dup + 1]

    # Return symmetric pairs
    adj_pairs = torch.cat([
        torch.stack([fa, fb], dim=1),
        torch.stack([fb, fa], dim=1),
    ], dim=0)   # (2*E, 2)
    return adj_pairs


def _cc_by_label_prop(face_mask: _T, adj_pairs: _T) -> List[_T]:
    """Find connected components of selected faces via iterative label propagation.

    Each selected face starts with its own index as label.  In each iteration,
    for every shared edge between two selected faces the minimum label is
    propagated to both endpoints.  Iteration stops when labels no longer change
    or after ``min(200, num_selected)`` rounds.

    Parameters
    ----------
    face_mask : Tensor, shape (F,), bool
        True for faces that belong to the set of interest.
    adj_pairs : Tensor, shape (2*E, 2), int64
        Symmetric directed adjacency pairs from ``_build_adj_pairs``.

    Returns
    -------
    components : list of Tensor (int64)
        Each element is a 1-D tensor of face indices belonging to one
        connected component, ordered by face index.
    """
    sel = face_mask.nonzero(as_tuple=True)[0]   # (S,) selected face indices
    if sel.numel() == 0:
        return []

    F = int(face_mask.shape[0])
    device = face_mask.device

    # Initialise each face's label to its own index
    labels = torch.arange(F, dtype=torch.long, device=device)

    pi = adj_pairs[:, 0]
    pj = adj_pairs[:, 1]

    # Only propagate between selected faces
    both_sel = face_mask[pi] & face_mask[pj]
    pi_s = pi[both_sel]
    pj_s = pj[both_sel]

    max_iter = min(200, int(sel.numel()))
    import warnings as _w
    for _ in range(max_iter):
        old = labels.clone()
        # For each edge (i, j) among selected faces: labels[i] = min(labels[i], labels[j])
        min_ij = torch.minimum(labels[pi_s], labels[pj_s])
        with _w.catch_warnings():
            _w.filterwarnings("ignore", "index_reduce")
            labels.index_reduce_(0, pi_s, min_ij, 'amin', include_self=True)
            labels.index_reduce_(0, pj_s, min_ij, 'amin', include_self=True)
        if (labels == old).all():
            break

    # Group selected faces by their final label
    sel_labels = labels[sel]
    unique_labels = sel_labels.unique()
    components: List[_T] = []
    for lbl in unique_labels:
        comp = sel[sel_labels == lbl]
        components.append(comp)
    return components


def _dihedral_per_face(normals: _T, adj_pairs: _T) -> _T:
    """Compute per-face mean dihedral angle (degrees) with adjacent faces.

    Uses vectorized scatter-accumulate; faces with no adjacent selected
    neighbors receive angle 0.

    Parameters
    ----------
    normals : Tensor, shape (F, 3), float
        Unit face normals.
    adj_pairs : Tensor, shape (2*E, 2), int64
        Symmetric directed adjacency pairs from ``_build_adj_pairs``.

    Returns
    -------
    angles : Tensor, shape (F,), float32
        Mean dihedral angle per face in degrees.
    """
    F = int(normals.shape[0])
    device = normals.device

    pi = adj_pairs[:, 0]
    pj = adj_pairs[:, 1]

    # Process each undirected edge once
    keep = pi < pj
    pi_u = pi[keep]
    pj_u = pj[keep]

    ni = normals[pi_u].float()
    nj = normals[pj_u].float()
    cos_t = (ni * nj).sum(dim=1).clamp(-1.0, 1.0)
    angles_deg = torch.acos(cos_t) * (180.0 / math.pi)

    angle_sum = torch.zeros(F, dtype=torch.float32, device=device)
    count = torch.zeros(F, dtype=torch.float32, device=device)
    ones = torch.ones(pi_u.shape[0], dtype=torch.float32, device=device)

    angle_sum.index_add_(0, pi_u, angles_deg)
    angle_sum.index_add_(0, pj_u, angles_deg)
    count.index_add_(0, pi_u, ones)
    count.index_add_(0, pj_u, ones)

    valid = count > 0
    result = torch.zeros(F, dtype=torch.float32, device=device)
    result[valid] = angle_sum[valid] / count[valid]
    return result


def _bfs_expand(mask: _T, adj_pairs: _T, num_rings: int) -> _T:
    """Expand a face selection outward by ``num_rings`` of face-adjacent neighbors.

    Vectorized ring expansion: in each ring, any face adjacent to a currently
    selected face becomes selected.

    Parameters
    ----------
    mask : Tensor, shape (F,), bool
        Initial selection mask.
    adj_pairs : Tensor, shape (2*E, 2), int64
        Symmetric directed adjacency pairs from ``_build_adj_pairs``.
    num_rings : int
        Number of expansion steps.

    Returns
    -------
    result : Tensor, shape (F,), bool
        Expanded selection after ``num_rings`` steps.
    """
    pi = adj_pairs[:, 0]
    pj = adj_pairs[:, 1]
    result = mask.clone()
    for _ in range(num_rings):
        src_sel = result[pi]           # edges whose source is selected
        new = result.clone()
        new[pj[src_sel]] = True        # activate destination faces
        result = new
    return result


def _boundary_faces(mask: _T, adj_pairs: _T) -> _T:
    """Select faces on the perimeter of a region.

    A face is a boundary face if it is selected and has at least one
    adjacent face that is not selected.

    Parameters
    ----------
    mask : Tensor, shape (F,), bool
        Selection mask.
    adj_pairs : Tensor, shape (2*E, 2), int64
        Symmetric directed adjacency pairs from ``_build_adj_pairs``.

    Returns
    -------
    boundary : Tensor, shape (F,), bool
        True for selected faces that have at least one non-selected neighbour.
    """
    F = int(mask.shape[0])
    device = mask.device
    pi = adj_pairs[:, 0]
    pj = adj_pairs[:, 1]

    # Edges where source is selected and destination is not
    boundary_edge = mask[pi] & ~mask[pj]
    boundary = torch.zeros(F, dtype=torch.bool, device=device)
    boundary[pi[boundary_edge]] = True
    return boundary & mask


# ===========================================================================
# Base class
# ===========================================================================

class SurfaceSelector:
    """Base class for surface-mesh-based BC region selectors.

    Subclass and override :meth:`select`.  Use the instance as a key in
    ``dirichlet_bcs_dict`` or ``neumann_bcs_dict``; the system calls
    :meth:`resolve` automatically before the first BC mask build.

    The resolution pipeline is:

    1. ``resolve(verts_phys, faces)`` computes per-face attributes and calls
       ``select()`` to get a boolean face mask.
    2. A Warp BVH is built from the selected triangles, with one box around
       the whole selected patch (expanded by ``eps_expansion``).
    3. ``__call__(x, y, z)`` culls query points outside that box and accepts
       the rest within ``eps_expansion`` of the selected triangles (BVH).

    Boolean operators ``&``, ``|``, ``-``, ``~`` return composite selectors::

        sel = NormalSelector((0, -1, 0), 60) & LowestSelector('y', 0.1)

    Parameters
    ----------
    None — subclasses add their own ``__init__`` parameters.

    Attributes
    ----------
    _patch_lo, _patch_hi : Tensor (3,) or None
        Set by ``resolve()``: the selected patch's bounding box, expanded by
        ``eps_expansion``.  ``None`` for an empty selection.
    """

    def __init__(self):
        self._resolved = False
        self._patch_lo: Optional[_T] = None
        self._patch_hi: Optional[_T] = None
        self._wp_mesh = None
        self._wp_keep_alive = None
        self._eps_expansion: float = 0.0
        self._bvh_device: str = "cpu"
        self._sdf_grid_context = None
        self._selected_face_mask: Optional[_T] = None
        self._patch_areas: Optional[_T] = None
        self._patch_normals: Optional[_T] = None
        self._surface_segmentation = None

    @property
    def is_resolved(self) -> bool:
        """bool: True after ``resolve()`` has been called successfully."""
        return self._resolved

    def loaded_area(self) -> float:
        """Surface area a traction on this patch loads.

        ``sum(area * max(0, n . n_avg))`` over the selected faces, where
        ``n_avg`` is their area-weighted mean normal.  This is the
        ``loaded_area`` that resolve-bcs reports and that a force is divided by
        to give a traction.  It is also what the diffuse Neumann term's weight
        ``max(0, -grad(phi) . n_face)`` integrates to over an unbounded band,
        so NitscheNeoHookean rescales each Neumann region to it.  Sign
        conventions cancel: the weight is taken against the patch's own mean
        normal.  Returns 0.0 for an unresolved or empty selection.
        """
        if self._patch_areas is None:
            return 0.0
        return _loaded_area(self._patch_areas, self._patch_normals)

    def loaded_area_on(self, verts: _T, faces: _T) -> float:
        """``loaded_area`` of this selector's region on another surface.

        Runs ``select`` on ``(verts, faces)`` (e.g. the source surface of a
        solver that resolved this selector on its own MC surface) and weighs
        the selected faces as ``loaded_area`` does.  Nothing ``__call__``
        reads is touched: the mask, BVH and patch box stay those of
        ``resolve``.
        """
        centroids, normals, areas = _compute_face_attributes(verts, faces)
        curvatures = _compute_face_curvatures(verts, faces)
        adj_pairs = _build_adj_pairs(faces)
        mask = self.select(
            verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs
        )
        return _loaded_area(areas[mask].double(), normals[mask].double())

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def select(
        self,
        verts: _T,
        faces: _T,
        centroids: _T,
        normals: _T,
        areas: _T,
        curvatures: Optional[_T] = None,
        adj_pairs: Optional[_T] = None,
    ) -> _T:
        """Return a boolean face mask for this selector's criterion.

        Parameters
        ----------
        verts : Tensor, shape (V, 3), float
            Physical-space vertex positions.
        faces : Tensor, shape (F, 3), int
            Triangle face indices.
        centroids : Tensor, shape (F, 3), float
            Pre-computed face barycentres.
        normals : Tensor, shape (F, 3), float
            Pre-computed unit face normals.
        areas : Tensor, shape (F,), float
            Pre-computed triangle areas.
        curvatures : Tensor, shape (F,), float or None
            Pre-computed mean curvature magnitudes.
        adj_pairs : Tensor, shape (2*E, 2), int64 or None
            Pre-built face-adjacency pairs from ``_build_adj_pairs``.
            Selectors that need adjacency use this to avoid rebuilding it.

        Returns
        -------
        mask : Tensor, shape (F,), bool
            True for every face that this selector considers "selected".
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        verts_phys: _T,
        faces: _T,
        eps_expansion: float = 0.0,
        sdf_grid=None,
    ) -> "SurfaceSelector":
        """Resolve this selector against a physical-space MC surface mesh.

        Computes per-face attributes, calls :meth:`select`, and builds a Warp
        BVH from the selected triangles for the distance queries in
        :meth:`__call__`, plus the patch's expanded bounding box that culls
        far points before them.

        Parameters
        ----------
        verts_phys : Tensor, shape (V, 3), float
            Physical-space vertex positions of the marching-cubes surface.
        faces : Tensor, shape (F, 3), int
            Triangle face indices.
        eps_expansion : float
            Distance threshold of the BVH test in :meth:`__call__`; also the
            expansion of the far-point culling box.

        Returns
        -------
        self : SurfaceSelector
            Returns self for chaining.
        """
        self._sdf_grid_context = sdf_grid
        centroids, normals, areas = _compute_face_attributes(verts_phys, faces)
        curvatures = _compute_face_curvatures(verts_phys, faces)
        # Build adj_pairs once and share it with select().
        adj_pairs = _build_adj_pairs(faces)
        face_mask = self.select(
            verts_phys, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs
        )
        self._selected_face_mask = face_mask.detach().clone()
        # The patch's own face areas and normals: what ``loaded_area`` needs to
        # tell a load on this region how much surface it covers.
        self._patch_areas = areas[face_mask].detach().double()
        self._patch_normals = normals[face_mask].detach().double()

        self._eps_expansion = float(eps_expansion)
        self._bvh_device = str(verts_phys.device)
        if int(face_mask.sum().item()) == 0:
            warnings.warn(
                f"{self.__class__.__name__} selected 0 faces from the "
                f"MC surface mesh — BC region will be empty.",
                stacklevel=3,
            )
            self._patch_lo = self._patch_hi = None
            self._wp_mesh = None
            self._resolved = True
            return self

        sel_faces = faces[face_mask]
        sel_verts = verts_phys[sel_faces.reshape(-1).long()].float()
        self._patch_lo = (sel_verts.amin(0) - self._eps_expansion).contiguous()
        self._patch_hi = (sel_verts.amax(0) + self._eps_expansion).contiguous()
        # No fallback: without the BVH the only region left would be the
        # whole-patch box, far larger than the BC region.
        from .analytical_sdf import build_warp_mesh_patch
        self._wp_mesh, self._wp_keep_alive = build_warp_mesh_patch(
            verts_phys, sel_faces, device=self._bvh_device
        )
        self._resolved = True
        return self

    # ------------------------------------------------------------------
    # Callable interface (used by identify_boundary_nodes)
    # ------------------------------------------------------------------

    def __call__(self, x: _T, y: _T, z: _T) -> _T:
        """Test query points against the resolved BC region.

        A point is in the region iff its unsigned distance to the selected
        triangles is ≤ ``eps_expansion``.  Points outside the selected patch's
        expanded bounding box cannot be, so they are culled before the BVH
        query.

        This replaced a cover of small per-component boxes as the pre-filter.
        The BVH query, bounded by ``max_dist = 2 eps``, already rejects far
        points at its top levels, so the box cover only cost time: on the
        bunny example at depth 7 (1.4 M Gauss points, 321-1248 boxes) it took
        30-110 ms per call against 1-2.4 ms for the cull + BVH, and 17-48 ms
        per ``resolve`` to build.  The accepted sets are identical.

        Parameters
        ----------
        x, y, z : Tensor, shape (N,), float
            Query point coordinates.

        Returns
        -------
        inside : Tensor, shape (N,), bool
            True for every point within ``eps_expansion`` of the selected surface.

        Raises
        ------
        RuntimeError
            If called before :meth:`resolve`.
        """
        if not self.is_resolved:
            raise RuntimeError(
                f"{self.__class__.__name__} is not resolved. "
                "Use it as a key in dirichlet_bcs_dict / neumann_bcs_dict "
                "and let NitscheNeoHookean._resolve_surface_selector_bcs() "
                "handle resolution, or call resolve() manually."
            )
        result = torch.zeros_like(x, dtype=torch.bool)
        if self._wp_mesh is None:
            return result

        device = x.device
        lo = self._patch_lo.to(device)
        hi = self._patch_hi.to(device)
        pts = torch.stack([x, y, z], dim=1).to(dtype=lo.dtype)       # (N, 3)
        near = ((pts >= lo) & (pts <= hi)).all(dim=1)
        cand = near.nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            return result

        from .analytical_sdf import compute_unsigned_dist
        dists = compute_unsigned_dist(
            self._wp_mesh,
            pts[cand],
            max_dist=max(self._eps_expansion * 2.0, 1e-8),
            device=self._bvh_device,
        )
        result[cand] = dists.to(device) <= self._eps_expansion
        return result

    # ------------------------------------------------------------------
    # Boolean operators
    # ------------------------------------------------------------------

    def __and__(self, other: "SurfaceSelector") -> "IntersectSelector":
        """Return ``self & other`` — intersection of two selectors."""
        return IntersectSelector(self, other)

    def __or__(self, other: "SurfaceSelector") -> "UnionSelector":
        """Return ``self | other`` — union of two selectors."""
        return UnionSelector(self, other)

    def __sub__(self, other: "SurfaceSelector") -> "SubtractSelector":
        """Return ``self - other`` — faces in self but not in other."""
        return SubtractSelector(self, other)

    def __invert__(self) -> "ComplementSelector":
        """Return ``~self`` — faces NOT selected by self."""
        return ComplementSelector(self)

    def __repr__(self) -> str:
        status = "resolved" if self.is_resolved else "unresolved"
        n = (int(self._selected_face_mask.sum()) if self.is_resolved
             and self._selected_face_mask is not None else "?")
        return f"{self.__class__.__name__}({status}, {n} faces)"


# ===========================================================================
# Geometric Selectors
# ===========================================================================

class BoundingBoxSelector(SurfaceSelector):
    """Select faces whose centroid is inside an axis-aligned bounding box.

    Parameters
    ----------
    min_pt : sequence of float, length 3
        Lower corner ``(x_min, y_min, z_min)`` of the selection box.
    max_pt : sequence of float, length 3
        Upper corner ``(x_max, y_max, z_max)`` of the selection box.
    """

    def __init__(
        self,
        min_pt: Sequence[float],
        max_pt: Sequence[float],
    ):
        super().__init__()
        self._min_pt = list(min_pt)
        self._max_pt = list(max_pt)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose centroid lies within the stored bounding box.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = centroids.device
        lo = torch.tensor(self._min_pt, dtype=torch.float32, device=device)
        hi = torch.tensor(self._max_pt, dtype=torch.float32, device=device)
        return ((centroids >= lo.unsqueeze(0)) & (centroids <= hi.unsqueeze(0))).all(-1)


class SphereSelector(SurfaceSelector):
    """Select faces whose centroid is within ``radius`` of ``center``.

    Parameters
    ----------
    center : sequence of float, length 3
        Centre of the selection sphere.
    radius : float
        Radius of the selection sphere.
    """

    def __init__(self, center: Sequence[float], radius: float):
        super().__init__()
        self._center = list(center)
        self._radius = float(radius)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose centroid distance to center is ≤ radius.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = centroids.device
        c = torch.tensor(self._center, dtype=torch.float32, device=device)
        return (centroids - c.unsqueeze(0)).norm(dim=1) <= self._radius


class CylinderSelector(SurfaceSelector):
    """Select faces within ``radius`` of the segment from ``p1`` to ``p2``.

    A face is kept when its centroid's distance to the closest point of the
    segment (the projection parameter clamped to [0, 1]) is at most
    ``radius``.  The region is therefore a capsule: a finite cylinder with a
    hemispherical cap of the same radius past each endpoint, not an infinite
    or open-ended one.

    Parameters
    ----------
    p1 : sequence of float, length 3
        First endpoint of the cylinder axis.
    p2 : sequence of float, length 3
        Second endpoint of the cylinder axis.
    radius : float
        Cylinder radius.
    """

    def __init__(self, p1: Sequence[float], p2: Sequence[float], radius: float):
        super().__init__()
        self._p1 = list(p1)
        self._p2 = list(p2)
        self._radius = float(radius)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose centroid is within radius of the cylinder axis.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = centroids.device
        p1 = torch.tensor(self._p1, dtype=torch.float32, device=device)
        p2 = torch.tensor(self._p2, dtype=torch.float32, device=device)
        axis = p2 - p1
        axis_len_sq = (axis * axis).sum().clamp(min=1e-10)
        # Project centroid onto axis: t = (c - p1) · axis / |axis|²
        diff = centroids - p1.unsqueeze(0)    # (F, 3)
        t = (diff * axis.unsqueeze(0)).sum(-1) / axis_len_sq   # (F,)
        t = t.clamp(0.0, 1.0)
        closest = p1.unsqueeze(0) + t.unsqueeze(1) * axis.unsqueeze(0)
        dist = (centroids - closest).norm(dim=1)
        return dist <= self._radius


class HalfSpaceSelector(SurfaceSelector):
    """Select faces where ``(centroid - point) · normal ≥ 0``.

    Parameters
    ----------
    point : sequence of float, length 3
        A point on the dividing plane.
    normal : sequence of float, length 3
        Outward normal of the half-space (need not be unit length).
    """

    def __init__(self, point: Sequence[float], normal: Sequence[float]):
        super().__init__()
        self._point = list(point)
        n = list(normal)
        mag = math.sqrt(sum(v * v for v in n))
        # Normalize once at init — avoids redundant norm on every select() call.
        self._normal = [v / max(mag, 1e-10) for v in n]

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces on the positive side of the dividing plane.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = centroids.device
        pt = torch.tensor(self._point, dtype=torch.float32, device=device)
        n = torch.tensor(self._normal, dtype=torch.float32, device=device)
        return ((centroids - pt.unsqueeze(0)) * n.unsqueeze(0)).sum(-1) >= 0.0


class SlabSelector(SurfaceSelector):
    """Select faces whose centroid coordinate along ``axis`` ∈ [min_val, max_val].

    Parameters
    ----------
    axis : str or int
        ``'x'`` / 0, ``'y'`` / 1, or ``'z'`` / 2.
    min_val : float
        Lower bound (inclusive).
    max_val : float
        Upper bound (inclusive).
    """

    _AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, axis: Union[str, int], min_val: float, max_val: float):
        super().__init__()
        self._axis = self._AXIS_MAP[axis]
        self._min_val = float(min_val)
        self._max_val = float(max_val)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose centroid[axis] is within [min_val, max_val].

        A band of no thickness is widened to one face, centred on it.  The test
        is on centroids, so ``min_val == max_val`` -- naming a plane rather than
        a slab, which is what a request for "the face at y = -0.00089" looks
        like -- would otherwise match nothing at all, since no centroid lands
        exactly on it.  A band that has any thickness is left alone: the caller
        asked for a specific interval and is entitled to it, however thin.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; ``centroids`` and ``areas`` are used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        c = centroids[:, self._axis]
        low, high = self._min_val, self._max_val
        if high <= low and areas is not None and areas.numel() > 0:
            face_size = float(torch.sqrt(torch.median(areas)))
            middle = 0.5 * (low + high)
            low, high = middle - 0.5 * face_size, middle + 0.5 * face_size
        return (c >= low) & (c <= high)


class LowestSelector(SurfaceSelector):
    """Select faces in the bottom ``fraction`` of the bbox extent along ``axis``.

    Parameters
    ----------
    axis : str or int
        ``'x'`` / 0, ``'y'`` / 1, or ``'z'`` / 2.
    fraction : float
        Fraction of the axis extent to include (e.g. 0.1 = bottom 10 %).
    """

    _AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, axis: Union[str, int] = 'y', fraction: float = 0.1):
        super().__init__()
        self._axis = self._AXIS_MAP[axis]
        self._fraction = float(fraction)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces in the lowest fraction of the centroid range along axis.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        c = centroids[:, self._axis]
        # Keep threshold as a GPU scalar — no .item() round-trips needed.
        threshold = c.min() + self._fraction * (c.max() - c.min())
        return c <= threshold


class HighestSelector(SurfaceSelector):
    """Select faces in the top ``fraction`` of the bbox extent along ``axis``.

    Parameters
    ----------
    axis : str or int
        ``'x'`` / 0, ``'y'`` / 1, or ``'z'`` / 2.
    fraction : float
        Fraction of the axis extent to include (e.g. 0.1 = top 10 %).
    """

    _AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(self, axis: Union[str, int] = 'y', fraction: float = 0.1):
        super().__init__()
        self._axis = self._AXIS_MAP[axis]
        self._fraction = float(fraction)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces in the highest fraction of the centroid range along axis.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        c = centroids[:, self._axis]
        # Keep threshold as a GPU scalar — no .item() round-trips needed.
        threshold = c.max() - self._fraction * (c.max() - c.min())
        return c >= threshold


class PercentileSelector(SurfaceSelector):
    """Select faces in the [lo_pct, hi_pct] percentile of centroid[axis].

    Parameters
    ----------
    axis : str or int
        ``'x'`` / 0, ``'y'`` / 1, or ``'z'`` / 2.
    lo_pct : float
        Lower percentile bound (0–100).
    hi_pct : float
        Upper percentile bound (0–100).
    """

    _AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}

    def __init__(
        self,
        axis: Union[str, int],
        lo_pct: float = 0.0,
        hi_pct: float = 10.0,
    ):
        super().__init__()
        self._axis = self._AXIS_MAP[axis]
        self._lo_pct = float(lo_pct)
        self._hi_pct = float(hi_pct)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose centroid[axis] falls in the given percentile range.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        c = centroids[:, self._axis]
        # Keep quantile thresholds as 0-d tensors — no .item() round-trips needed.
        cf = c.float()
        lo = torch.quantile(cf, self._lo_pct / 100.0)
        hi = torch.quantile(cf, self._hi_pct / 100.0)
        return (c >= lo) & (c <= hi)


class FurthestFromCenterSelector(SurfaceSelector):
    """Select the ``fraction`` of faces radially farthest from the mesh centroid.

    Parameters
    ----------
    fraction : float
        Fraction of all faces to select (e.g. 0.1 = top 10 % by radial distance).
    """

    def __init__(self, fraction: float = 0.1):
        super().__init__()
        self._fraction = float(fraction)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces with the largest radial distance from the mesh centroid.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``centroids`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        center = centroids.mean(0, keepdim=True)
        dists = (centroids - center).norm(dim=1)
        k = max(1, int(round(self._fraction * float(dists.shape[0]))))
        # Keep threshold as a 0-d tensor — no .item() round-trip needed.
        threshold = dists.topk(k).values.min()
        return dists >= threshold


# ===========================================================================
# Normal and Curvature Selectors
# ===========================================================================

class NormalSelector(SurfaceSelector):
    """Select faces where the angle between face normal and ``target_vector`` ≤ ``max_angle_deg``.

    The dot product is unsigned: both ``+n`` and ``-n`` match a given target.
    Use :class:`AlignedNormalSelector` for orientation-sensitive matching.

    Parameters
    ----------
    target_vector : sequence of float, length 3
        Reference direction (need not be unit length).
    max_angle_deg : float
        Maximum angle in degrees between face normal and target_vector.
    """

    def __init__(self, target_vector: Sequence[float], max_angle_deg: float):
        super().__init__()
        tv = list(target_vector)
        mag = math.sqrt(sum(v * v for v in tv))
        self._target = [v / max(mag, 1e-10) for v in tv]
        self._cos_thresh = math.cos(math.radians(max_angle_deg))

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose normal is within max_angle_deg of target_vector (unsigned).

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``normals`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = normals.device
        t = torch.tensor(self._target, dtype=torch.float32, device=device)
        cos_ang = (normals * t.unsqueeze(0)).sum(-1).abs()
        return cos_ang >= self._cos_thresh


class AlignedNormalSelector(SurfaceSelector):
    """Like :class:`NormalSelector` but respects orientation.

    With ``oriented=True`` (default), the dot product must be ≥ cos(angle)
    (not abs), so a target of (0, 0, 1) selects upward-facing faces only.

    Parameters
    ----------
    target_vector : sequence of float, length 3
        Reference direction (need not be unit length).
    max_angle_deg : float
        Maximum angle in degrees between face normal and target_vector.
    oriented : bool
        If True (default), only faces with dot product ≥ cos(max_angle_deg) are
        selected.  If False, behaves identically to :class:`NormalSelector`.
    """

    def __init__(
        self,
        target_vector: Sequence[float],
        max_angle_deg: float,
        oriented: bool = True,
    ):
        super().__init__()
        tv = list(target_vector)
        mag = math.sqrt(sum(v * v for v in tv))
        self._target = [v / max(mag, 1e-10) for v in tv]
        self._cos_thresh = math.cos(math.radians(max_angle_deg))
        self._oriented = bool(oriented)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose normal aligns with target_vector within max_angle_deg.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; only ``normals`` is used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        device = normals.device
        t = torch.tensor(self._target, dtype=torch.float32, device=device)
        dot = (normals * t.unsqueeze(0)).sum(-1)
        if self._oriented:
            return dot >= self._cos_thresh
        return dot.abs() >= self._cos_thresh


class CurvatureSelector(SurfaceSelector):
    """Select faces where curvature ∈ [min_kappa, max_kappa].

    Parameters
    ----------
    min_kappa : float
        Inclusive lower curvature bound.
    max_kappa : float
        Inclusive upper curvature bound.  Use ``math.inf`` for an open upper bound.
    curvature_type : str
        ``'mean'`` (default) uses the cotangent mean curvature.
        ``'gaussian'`` uses the angle-defect Gaussian curvature.
    """

    def __init__(
        self,
        min_kappa: float,
        max_kappa: float,
        curvature_type: str = "mean",
    ):
        super().__init__()
        self._min_kappa = float(min_kappa)
        self._max_kappa = float(max_kappa)
        self._curv_type = curvature_type

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose curvature falls in [min_kappa, max_kappa].

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature.  ``curvatures`` is used for mean
            curvature; Gaussian curvature is computed on demand.

        Returns
        -------
        mask : Tensor, shape (F,), bool

        Raises
        ------
        ValueError
            If curvature computation fails (degenerate mesh).
        """
        if self._curv_type == "mean":
            kappa = curvatures
        else:
            kappa = _compute_gaussian_curvature(verts, faces)
        if kappa is None:
            raise ValueError(
                f"CurvatureSelector(type='{self._curv_type}'): "
                "curvature computation failed on this mesh."
            )
        return (kappa >= self._min_kappa) & (kappa <= self._max_kappa)


class FlatRegionSelector(CurvatureSelector):
    """Select faces with mean curvature ≤ ``max_curvature`` (flat regions).

    Parameters
    ----------
    max_curvature : float
        Upper mean-curvature bound; faces with κ ≤ this value are selected.
    """

    def __init__(self, max_curvature: float = 0.1):
        super().__init__(0.0, float(max_curvature), curvature_type="mean")


class SharpFeatureSelector(CurvatureSelector):
    """Select faces with mean curvature ≥ ``min_curvature`` (sharp features).

    Parameters
    ----------
    min_curvature : float
        Lower mean-curvature bound; faces with κ ≥ this value are selected.
    """

    def __init__(self, min_curvature: float = 1.0):
        super().__init__(float(min_curvature), math.inf, curvature_type="mean")


class DihedralAngleSelector(SurfaceSelector):
    """Select faces whose mean dihedral angle with neighbors ∈ [min_deg, max_deg].

    Faces with no adjacent neighbors are excluded (angle = 0).  Uses
    the fully vectorized ``_dihedral_per_face`` helper.

    Parameters
    ----------
    min_angle_deg : float
        Inclusive lower bound on mean dihedral angle in degrees.
    max_angle_deg : float
        Inclusive upper bound on mean dihedral angle in degrees.
    """

    def __init__(self, min_angle_deg: float = 0.0, max_angle_deg: float = 180.0):
        super().__init__()
        self._min = float(min_angle_deg)
        self._max = float(max_angle_deg)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces whose mean dihedral angle with neighbours is in [min, max].

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature; ``normals`` and ``faces`` are used.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        angles = _dihedral_per_face(normals, adj_pairs)
        return (angles >= self._min) & (angles <= self._max)


class SurfaceSegmentSelector(SurfaceSelector):
    """Select whole parts from a previously reviewed per-face segmentation."""

    def __init__(self, segment_ids, segmentation):
        super().__init__()
        self._segment_ids = sorted({int(value) for value in segment_ids})
        if not self._segment_ids:
            raise ValueError("segment_ids must name at least one segment")
        self._segmentation = segmentation

    def select(self, verts, faces, centroids, normals, areas, curvatures=None,
               adj_pairs=None):
        from .bc_surface_segmentation import SurfaceSegmentation

        segmentation = self._segmentation
        if not isinstance(segmentation, SurfaceSegmentation):
            segmentation = SurfaceSegmentation.from_labels(
                segmentation, device=faces.device
            )
        if segmentation.segment_ids.numel() != faces.shape[0]:
            raise ValueError(
                "supplied segmentation has "
                f"{segmentation.segment_ids.numel()} labels for "
                f"{faces.shape[0]} faces"
            )
        labels = segmentation.segment_ids.to(faces.device)
        available = torch.unique(labels)
        requested = torch.tensor(
            self._segment_ids, dtype=torch.long, device=faces.device
        )
        unknown = requested[~torch.isin(requested, available)]
        if unknown.numel():
            raise ValueError(
                f"unknown segment ids: {sorted(int(v) for v in unknown)}; "
                f"available {sorted(int(v) for v in available)}"
            )
        return torch.isin(labels, requested)


class ShapeDiameterSegmentSelector(SurfaceSelector):
    """Expand seed faces to coherent thickness/concavity surface segments.

    This is the GPU-native counterpart of the CGAL shape-diameter workflow.
    A seed identifies *which* automatically discovered part carries the BC; it
    may be a point, a sequence of points, or another ``SurfaceSelector``.  Each
    disconnected seed component independently claims its best-overlap segment.
    ``adjacent_segments`` can then add that many spatially nearest neighboring
    segments per component.  ``adjacent_direction`` optionally prefers
    candidates in a physical direction, for example ``(0, 0, -1)`` when a
    highest-point seed should grow toward the base of a part.
    """

    _CACHE = {}
    _CACHE_LIMIT = 8

    def __init__(
        self,
        seed=None,
        num_clusters: int = 5,
        smoothing: float = 0.25,
        num_rays: int = 25,
        cone_angle_deg: float = 120.0,
        preserve_seed: bool = True,
        max_segment_area_fraction: float = 1.0,
        adjacent_segments: int = 0,
        adjacent_direction=None,
        *,
        segment_ids=None,
        segmentation=None,
    ):
        super().__init__()
        if num_clusters < 2:
            raise ValueError("num_clusters must be at least 2")
        if num_rays < 1:
            raise ValueError("num_rays must be positive")
        if (seed is None) == (segment_ids is None):
            raise ValueError(
                "supply either a seed to expand or segment_ids to take, not both"
            )
        self._seed = seed
        self._segment_ids = (
            None if segment_ids is None
            else sorted({int(value) for value in segment_ids})
        )
        if self._segment_ids is not None and not self._segment_ids:
            raise ValueError("segment_ids must name at least one segment")
        # A caller holding an earlier segmentation of this same surface passes it
        # here; the expensive solve is then skipped and the selection is taken
        # against exactly the partition the user already reviewed.
        self._segmentation = segmentation
        self._num_clusters = int(num_clusters)
        self._smoothing = float(smoothing)
        self._num_rays = int(num_rays)
        self._cone_angle_deg = float(cone_angle_deg)
        self._preserve_seed = bool(preserve_seed)
        if not 0.0 < max_segment_area_fraction <= 1.0:
            raise ValueError("max_segment_area_fraction must be in (0, 1]")
        self._max_segment_area_fraction = float(max_segment_area_fraction)
        if adjacent_segments < 0:
            raise ValueError("adjacent_segments must be nonnegative")
        self._adjacent_segments = int(adjacent_segments)
        if adjacent_direction is None:
            self._adjacent_direction = None
        else:
            direction = torch.as_tensor(adjacent_direction, dtype=torch.float32)
            if direction.numel() != 3 or float(torch.linalg.vector_norm(direction)) == 0.0:
                raise ValueError("adjacent_direction must be a nonzero 3-vector")
            self._adjacent_direction = tuple((direction / torch.linalg.vector_norm(direction)).tolist())

    def _seed_mask(self, verts, faces, centroids, normals, areas, curvatures, adj_pairs):
        seed = self._seed
        if isinstance(seed, SurfaceSelector):
            return seed.select(
                verts, faces, centroids, normals, areas, curvatures,
                adj_pairs=adj_pairs,
            )
        points = torch.as_tensor(seed, dtype=torch.float32, device=verts.device)
        if points.ndim == 1:
            if points.numel() != 3:
                raise ValueError("point seed must contain three coordinates")
            points = points.unsqueeze(0)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("seed must be a point, points, or SurfaceSelector")
        # All point-to-centroid distances are parallel; P is deliberately tiny.
        nearest = torch.cdist(points, centroids.float()).argmin(dim=1)
        mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=verts.device)
        mask[nearest] = True
        return mask

    def _precomputed(self, faces):
        """Adapt a caller-supplied segmentation, or None to solve for one."""
        if self._segmentation is None:
            return None
        from .bc_surface_segmentation import SurfaceSegmentation

        segmentation = self._segmentation
        if not isinstance(segmentation, SurfaceSegmentation):
            segmentation = SurfaceSegmentation.from_labels(
                segmentation, device=faces.device
            )
        if segmentation.segment_ids.numel() != faces.shape[0]:
            raise ValueError(
                "supplied segmentation has "
                f"{segmentation.segment_ids.numel()} labels for "
                f"{faces.shape[0]} faces"
            )
        if segmentation.segment_ids.device != faces.device:
            segmentation = SurfaceSegmentation.from_labels(
                segmentation.segment_ids, segmentation.shape_diameter,
                device=faces.device,
            )
        return segmentation

    def _take_segments(self, segmentation, device):
        """Select whole segments by id, the reviewed-preview path."""
        available = torch.unique(segmentation.segment_ids)
        requested = torch.tensor(self._segment_ids, dtype=torch.long, device=device)
        unknown = requested[~torch.isin(requested, available)]
        if unknown.numel():
            raise ValueError(
                f"unknown segment ids: {sorted(int(v) for v in unknown)}; "
                f"available {sorted(int(v) for v in available)}"
            )
        return torch.isin(segmentation.segment_ids, requested)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        if verts.device.type != "cuda":
            raise RuntimeError("ShapeDiameterSegmentSelector requires a CUDA surface")
        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        from .bc_surface_segmentation import segment_surface

        grid_ptr = 0
        if self._sdf_grid_context is not None:
            grid_ptr = self._sdf_grid_context.values.data_ptr()
        key = (
            verts.data_ptr(), faces.data_ptr(), tuple(verts.shape), tuple(faces.shape),
            grid_ptr, self._num_clusters, self._smoothing, self._num_rays,
            self._cone_angle_deg,
            self._max_segment_area_fraction,
        )
        segmentation = self._precomputed(faces) or self._CACHE.get(key)
        if segmentation is None:
            segmentation = segment_surface(
                verts, faces, centroids, normals, adj_pairs,
                num_clusters=self._num_clusters, smoothing=self._smoothing,
                num_rays=self._num_rays, cone_angle_deg=self._cone_angle_deg,
                sdf_grid=self._sdf_grid_context,
            )
            if len(self._CACHE) >= self._CACHE_LIMIT:
                self._CACHE.pop(next(iter(self._CACHE)))
            self._CACHE[key] = segmentation
        self._surface_segmentation = segmentation

        if self._segment_ids is not None:
            return self._take_segments(segmentation, faces.device)

        seed_mask = self._seed_mask(
            verts, faces, centroids, normals, areas, curvatures, adj_pairs
        )
        if not bool(seed_mask.any()):
            raise RuntimeError("shape-diameter segment seed selected no surface faces")
        result = torch.zeros_like(seed_mask)
        segments = segmentation.segment_ids
        segment_area = torch.zeros(
            segmentation.num_segments, dtype=areas.dtype, device=areas.device
        )
        segment_area.index_add_(0, segments, areas)
        area_limit = areas.sum() * self._max_segment_area_fraction
        eligible = segment_area <= area_limit
        segment_centroid = torch.zeros(
            (segmentation.num_segments, 3), dtype=centroids.dtype, device=areas.device
        )
        segment_centroid.index_add_(0, segments, centroids * areas[:, None])
        segment_centroid /= segment_area.clamp_min(1.0e-20)[:, None]
        if self._adjacent_segments:
            pair_labels = torch.sort(segments[adj_pairs], dim=1).values
            segment_edges = torch.unique(
                pair_labels[pair_labels[:, 0] != pair_labels[:, 1]], dim=0
            )
        direction = None
        if self._adjacent_direction is not None:
            direction = torch.tensor(
                self._adjacent_direction, dtype=centroids.dtype, device=centroids.device
            )
        for component in _cc_by_label_prop(seed_mask, adj_pairs):
            ids = segments[component]
            overlap = torch.zeros(
                segmentation.num_segments, dtype=areas.dtype, device=areas.device
            )
            overlap.index_add_(0, ids, areas[component])
            overlap = torch.where(eligible, overlap, torch.full_like(overlap, -1.0))
            if bool((overlap >= 0.0).any()):
                active = torch.zeros(
                    segmentation.num_segments, dtype=torch.bool, device=areas.device
                )
                active[overlap.argmax()] = True
                anchor = centroids[component].mean(0)
                for _ in range(self._adjacent_segments):
                    touched = torch.zeros_like(active, dtype=torch.int32)
                    u, v = segment_edges[:, 0], segment_edges[:, 1]
                    touched.index_add_(0, v, active[u].to(torch.int32))
                    touched.index_add_(0, u, active[v].to(torch.int32))
                    candidates = (touched > 0) & ~active & eligible
                    if direction is not None:
                        preferred = candidates & (
                            ((segment_centroid - anchor[None, :]) * direction[None, :]).sum(1) > 0
                        )
                        candidates = torch.where(preferred.any(), preferred, candidates)
                    if not bool(candidates.any()):
                        break
                    distance = torch.linalg.vector_norm(
                        segment_centroid - anchor[None, :], dim=1
                    )
                    distance = torch.where(
                        candidates, distance, torch.full_like(distance, float("inf"))
                    )
                    active[distance.argmin()] = True
                result |= active[segments]
            else:
                raise RuntimeError("no segment satisfies max_segment_area_fraction")
        if self._preserve_seed:
            result |= seed_mask
        return result


# ===========================================================================
# Topological Selectors
# ===========================================================================

class FloodFillSelector(SurfaceSelector):
    """BFS flood-fill from a seed face, stopping at sharp edges.

    Each candidate neighbor is accepted when the dihedral angle at the edge it
    shares with the face being expanded -- the angle between the two faces'
    normals -- is within ``max_feature_angle_deg``.  The test is pairwise, not
    against the seed's normal, so the fill follows any surface that turns
    gradually: on a smooth MC mesh whose adjacent normals differ by < 1°
    everywhere it can reach the whole connected surface.  It stops only where
    the surface creases.

    Parameters
    ----------
    seed : int, tuple of float, or SurfaceSelector
        - ``int``: face index used directly as the BFS seed.
        - ``(x, y, z)`` 3-tuple: the face with centroid nearest to this point.
        - :class:`SurfaceSelector`: the face with the largest area among those
          selected by this sub-selector.
    max_feature_angle_deg : float
        BFS stops crossing an edge when the dihedral angle between the two
        adjacent face normals exceeds this threshold.  On smooth surface regions
        (within a cone, along a flat face) adjacent normals differ by < 1° so
        the fill propagates freely.  At a concave valley between two spines, or
        where a cone base meets the body, the normals diverge sharply and the
        BFS stops.  Typical values: 20–45° for well-separated features.

    Notes
    -----
    The BFS is inherently sequential; adjacency is built via ``_build_adj_pairs``
    and converted to a compact CSR structure for O(1) neighbour lookup without a
    Python dict.
    """

    def __init__(
        self,
        seed: Union[int, Sequence[float], "SurfaceSelector"],
        max_feature_angle_deg: float = 30.0,
    ):
        super().__init__()
        self._seed = seed
        self._max_angle_deg = float(max_feature_angle_deg)

    def _resolve_seed(self, centroids: _T, areas: _T, verts, faces) -> int:
        """Determine the integer face index for the BFS seed.

        Parameters
        ----------
        centroids : Tensor, shape (F, 3)
        areas : Tensor, shape (F,)
        verts, faces : tensors
            Needed if seed is a SurfaceSelector.

        Returns
        -------
        seed_idx : int
        """
        seed = self._seed
        if isinstance(seed, int):
            return seed
        if isinstance(seed, SurfaceSelector):
            _, normals, _a = _compute_face_attributes(verts, faces)
            mask = seed.select(verts, faces, centroids, normals, _a)
            if not mask.any():
                return 0
            masked_areas = areas.clone()
            masked_areas[~mask] = -1.0
            return int(masked_areas.argmax().item())
        # Assume (x, y, z) point: find nearest centroid
        pt = torch.tensor(list(seed), dtype=torch.float32, device=centroids.device)
        return int((centroids - pt.unsqueeze(0)).norm(dim=1).argmin().item())

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Run BFS flood-fill from the seed face, gated on angle from seed normal.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature.

        Returns
        -------
        mask : Tensor, shape (F,), bool
            True for every seed face and every face reachable from them through
            edges whose dihedral angle is within ``max_feature_angle_deg``.
        """
        F = int(faces.shape[0])
        device = verts.device
        seed_idx = self._resolve_seed(centroids, areas, verts, faces)
        seed_idx = max(0, min(seed_idx, F - 1))
        cos_thresh = math.cos(math.radians(self._max_angle_deg))

        # Build adjacency and convert to CSR for fast neighbour lookup
        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        adj_np = adj_pairs.cpu().numpy()
        order = adj_np[:, 0].argsort()
        sorted_src = adj_np[order, 0]
        sorted_dst = adj_np[order, 1]
        offsets = np.searchsorted(sorted_src, np.arange(F + 1))

        normals_np = normals.cpu().numpy()

        selected = np.zeros(F, dtype=bool)
        visited = np.zeros(F, dtype=bool)

        if isinstance(self._seed, SurfaceSelector):
            # Multi-source BFS: pre-select all seed faces and expand from all of them.
            # Using only one seed face as the BFS origin causes the fill to stop after
            # a single ring when the threshold is tight — the full seed region must be
            # the starting frontier so the BFS grows outward from its entire boundary.
            seed_mask = self._seed.select(verts, faces, centroids, normals, areas)
            if seed_mask.any():
                seed_indices = seed_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                # Pre-seed all selected faces as visited so the BFS grows outward
                selected[seed_indices] = True
                visited[seed_indices] = True
                queue = seed_indices.tolist()
            else:
                queue = [seed_idx]
        else:
            queue = [seed_idx]

        while queue:
            fi = queue.pop()
            # Accept neighbour only if the dihedral angle at the shared edge is
            # below the threshold, i.e. the surface does not turn sharply there.
            for nb in sorted_dst[offsets[fi]:offsets[fi + 1]]:
                if not visited[nb]:
                    dot = float(np.clip(
                        np.dot(normals_np[fi], normals_np[nb]), -1.0, 1.0
                    ))
                    if dot >= cos_thresh:
                        visited[nb] = True
                        selected[nb] = True
                        queue.append(nb)
        return torch.tensor(selected, dtype=torch.bool, device=device)


class GrowSelectionSelector(SurfaceSelector):
    """Expand a base selector's result by ``num_rings`` of face-adjacent neighbors.

    Uses the vectorized ``_bfs_expand`` helper — no Python loops over faces.

    Parameters
    ----------
    base_selector : SurfaceSelector
        Initial selection to expand from.
    num_rings : int
        Number of adjacency rings to grow by.
    """

    def __init__(self, base_selector: SurfaceSelector, num_rings: int = 1):
        super().__init__()
        self._base = base_selector
        self._num_rings = int(num_rings)

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Expand the base selection outward by num_rings adjacency rings.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature.

        Returns
        -------
        mask : Tensor, shape (F,), bool
            Expanded boolean selection mask.
        """
        base_mask = self._base.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        return _bfs_expand(base_mask, adj_pairs, self._num_rings)


class BoundaryEdgeSelector(SurfaceSelector):
    """Select faces on the perimeter of the base selector's region.

    A perimeter face has at least one edge shared with a non-selected face.
    Uses the vectorized ``_boundary_faces`` helper — no Python loops over faces.

    Parameters
    ----------
    base_selector : SurfaceSelector
        The region whose boundary is to be selected.
    """

    def __init__(self, base_selector: SurfaceSelector):
        super().__init__()
        self._base = base_selector

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select boundary faces of the base selector's region.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature.

        Returns
        -------
        mask : Tensor, shape (F,), bool
            True for selected faces that border non-selected faces.
        """
        base_mask = self._base.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        return _boundary_faces(base_mask, adj_pairs)


class ConnectedComponentSelector(SurfaceSelector):
    """Pick one connected component from a base selector (or all faces).

    Uses the vectorized ``_cc_by_label_prop`` helper for component finding.

    Parameters
    ----------
    base_selector : SurfaceSelector or None
        Input set of faces.  ``None`` uses all faces.
    component_index : int
        Which CC to keep after sorting by ``sort_by`` (0 = largest/first).
    sort_by : str
        Sorting key for components.  One of:
        ``'size'`` (face count), ``'area'``, ``'centroid_x'``,
        ``'centroid_y'``, ``'centroid_z'``.
    """

    _SORT_KEYS = ('size', 'area', 'centroid_x', 'centroid_y', 'centroid_z')

    def __init__(
        self,
        base_selector: Optional[SurfaceSelector] = None,
        component_index: int = 0,
        sort_by: str = "size",
    ):
        super().__init__()
        self._base = base_selector
        self._idx = int(component_index)
        self._sort_by = sort_by

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Select faces belonging to the chosen connected component.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature.

        Returns
        -------
        mask : Tensor, shape (F,), bool
            True for faces in the chosen connected component.
        """
        if self._base is None:
            base_mask = torch.ones(faces.shape[0], dtype=torch.bool, device=verts.device)
        else:
            base_mask = self._base.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)

        if adj_pairs is None:
            adj_pairs = _build_adj_pairs(faces)
        components = _cc_by_label_prop(base_mask, adj_pairs)
        if not components:
            return torch.zeros(faces.shape[0], dtype=torch.bool, device=verts.device)

        def _key(comp_tensor: _T):
            if self._sort_by == "size":
                return int(comp_tensor.numel())
            if self._sort_by == "area":
                return float(areas[comp_tensor.to(areas.device)].sum().item())
            ax = {'centroid_x': 0, 'centroid_y': 1, 'centroid_z': 2}[self._sort_by]
            return float(centroids[comp_tensor.to(centroids.device), ax].mean().item())

        components.sort(key=_key, reverse=True)
        chosen_idx = min(self._idx, len(components) - 1)
        chosen = components[chosen_idx]

        result = torch.zeros(faces.shape[0], dtype=torch.bool, device=verts.device)
        result[chosen.to(verts.device)] = True
        return result


# ===========================================================================
# Boolean Composite Selectors
# ===========================================================================

class IntersectSelector(SurfaceSelector):
    """``A & B`` — faces selected by both A and B.

    Parameters
    ----------
    left : SurfaceSelector
        Left operand.
    right : SurfaceSelector
        Right operand.
    """

    def __init__(self, left: SurfaceSelector, right: SurfaceSelector):
        super().__init__()
        self._left = left
        self._right = right

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Return ``left.select() & right.select()``.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature forwarded to both operands.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        return (
            self._left.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
            & self._right.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
        )


class UnionSelector(SurfaceSelector):
    """``A | B`` — faces selected by either A or B.

    Parameters
    ----------
    left : SurfaceSelector
        Left operand.
    right : SurfaceSelector
        Right operand.
    """

    def __init__(self, left: SurfaceSelector, right: SurfaceSelector):
        super().__init__()
        self._left = left
        self._right = right

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Return ``left.select() | right.select()``.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature forwarded to both operands.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        return (
            self._left.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
            | self._right.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
        )


class SubtractSelector(SurfaceSelector):
    """``A - B`` — faces in A but not in B.

    Parameters
    ----------
    left : SurfaceSelector
        Left operand (set to subtract from).
    right : SurfaceSelector
        Right operand (set to subtract).
    """

    def __init__(self, left: SurfaceSelector, right: SurfaceSelector):
        super().__init__()
        self._left = left
        self._right = right

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Return ``left.select() & ~right.select()``.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature forwarded to both operands.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        return (
            self._left.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
            & ~self._right.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
        )


class ComplementSelector(SurfaceSelector):
    """``~A`` — faces NOT selected by A.

    Parameters
    ----------
    base : SurfaceSelector
        The selector to complement.
    """

    def __init__(self, base: SurfaceSelector):
        super().__init__()
        self._base = base

    def select(self, verts, faces, centroids, normals, areas, curvatures=None, adj_pairs=None):
        """Return ``~base.select()``.

        Parameters
        ----------
        verts, faces, centroids, normals, areas, curvatures
            Standard ``select`` signature forwarded to the base selector.

        Returns
        -------
        mask : Tensor, shape (F,), bool
        """
        return ~self._base.select(verts, faces, centroids, normals, areas, curvatures, adj_pairs=adj_pairs)
