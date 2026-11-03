"""Analytical SDF at octree nodes via Warp marching cubes + BVH.

Pipeline (used by :class:`LatentShapeOpt`):

    1. ``extract_surface(grid_sdf, domain_length)``
       Run ``wp.MarchingCubes`` on the raw decoder SDF grid.  Returns the
       triangle surface in normalized coordinates and a known-interior
       probe point (the deepest voxel of ``grid_sdf``).
    2. *(optional)* Caller transforms vertices and probe to whichever
       coordinate system the queries will be in (e.g. physical space
       with per-axis scaling, or vertex pinning in the BC region).
    3. ``build_warp_mesh(verts, faces, probe_pt, bbox_diag)``
       Construct a ``wp.Mesh`` (BVH) from the triangle soup.  Auto-detects
       sign convention by querying the probe point.
    4. ``compute_nodal_sdf(wp_mesh, query_pts, negate, bbox_diag)``
       Launch a kernel that computes signed distance from each query point
       to the mesh via ``wp.mesh_query_point_sign_normal``.

Patterned on ``src/triposg_wrapper/trimesh_sdf.py``.
"""

from typing import Tuple

import torch
import warp as wp


# ---------------------------------------------------------------------------
# Warp kernel: GPU-native signed distance via mesh queries
# ---------------------------------------------------------------------------


@wp.kernel
def _sdf_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=wp.float32),
    max_dist: float,
):
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist)
    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        sdf_out[tid] = wp.length(cp - p) * query.sign
    else:
        sdf_out[tid] = max_dist


@wp.kernel
def _unsigned_dist_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    dist_out: wp.array(dtype=wp.float32),
    max_dist: float,
):
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist_out[tid] = wp.length(cp - p)
    else:
        dist_out[tid] = max_dist


@wp.kernel
def _closest_point_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    closest_out: wp.array(dtype=wp.vec3),
    max_dist: float,
):
    """Write the closest point on the mesh for each query point."""
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_winding_number(mesh_id, p, max_dist)
    if query.result:
        closest_out[tid] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    else:
        closest_out[tid] = p  # fallback: leave unchanged


# ---------------------------------------------------------------------------
# Surface extraction: wp.MarchingCubes wrapper
# ---------------------------------------------------------------------------


def extract_surface(
    grid_sdf: torch.Tensor,
    domain_length: float,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the zero level set of ``grid_sdf`` via Warp marching cubes.

    Parameters
    ----------
    grid_sdf : torch.Tensor
        Shape ``(Nx, Ny, Nz)``, float, on GPU.  Standard convention:
        negative inside, positive outside.
    domain_length : float
        Physical side length of the cubic grid; vertices are mapped from
        index space ``[0, N-1]`` to normalized space ``[-L/2, L/2]``.
    device : str
        CUDA device string.

    Returns
    -------
    verts_norm : torch.Tensor
        ``(V, 3)`` float32 vertex positions in normalized coordinates.
    faces : torch.Tensor
        ``(F, 3)`` int32 triangle indices.
    probe_norm : torch.Tensor
        ``(3,)`` float32 position of the deepest-interior voxel in
        normalized coordinates — pass to :func:`build_warp_mesh` for
        sign-convention detection.
    """
    grid = grid_sdf.detach().to(device=device, dtype=torch.float32).contiguous()
    Nx, Ny, Nz = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])

    # Heuristic capacity: surface area scales as O(N^2), give it generous slack.
    max_verts = max(8 * Nx * Ny, 1 << 16)
    max_tris = max(16 * Nx * Ny, 1 << 16)

    field_wp = wp.from_torch(grid, dtype=wp.float32)
    mc = wp.MarchingCubes(
        nx=Nx,
        ny=Ny,
        nz=Nz,
        max_verts=max_verts,
        max_tris=max_tris,
        device=device,
    )
    mc.surface(field=field_wp, threshold=0.0)

    verts_idx = wp.to_torch(mc.verts).clone()  # (V, 3) float32, in [0, N-1]
    faces = wp.to_torch(mc.indices).clone().view(-1, 3).to(torch.int32)

    if verts_idx.numel() == 0 or faces.numel() == 0:
        raise RuntimeError(
            "wp.MarchingCubes produced an empty surface — "
            "grid_sdf has no zero crossing"
        )

    # Index space [0, N-1] → normalized space [-L/2, L/2].
    scale = torch.tensor(
        [domain_length / (Nx - 1), domain_length / (Ny - 1), domain_length / (Nz - 1)],
        dtype=torch.float32,
        device=device,
    )
    half = 0.5 * domain_length
    verts_norm = verts_idx * scale - half

    # Deepest-interior probe: argmin over the full grid.
    flat = grid.reshape(-1)
    min_idx = int(torch.argmin(flat).item())
    i = min_idx // (Ny * Nz)
    j = (min_idx // Nz) % Ny
    k = min_idx % Nz
    probe_norm = torch.tensor(
        [
            i * (domain_length / (Nx - 1)) - half,
            j * (domain_length / (Ny - 1)) - half,
            k * (domain_length / (Nz - 1)) - half,
        ],
        dtype=torch.float32,
        device=device,
    )

    return verts_norm, faces, probe_norm


# ---------------------------------------------------------------------------
# Warp mesh + sign convention
# ---------------------------------------------------------------------------


def _closed_surface_orientation(verts: torch.Tensor, faces: torch.Tensor) -> float | None:
    """Signed volume of a triangle soup, or None when it settles nothing.

    Positive means outward-facing normals, which is the "negative inside"
    convention the rest of the pipeline expects.  The divergence-theorem sum is
    exact for a closed, consistently wound surface and meaningless otherwise, so
    a result too small to trust relative to the mesh's own scale returns None and
    the caller keeps its probe.
    """
    triangles = verts[faces.view(-1, 3).long()]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    signed_volume = float(torch.einsum("ij,ij->i", a, torch.cross(b, c, dim=1)).sum()) / 6.0
    extent = float((verts.amax(dim=0) - verts.amin(dim=0)).max())
    if extent <= 0.0:
        return None
    # A closed surface's volume scales with extent cubed; anything far below that
    # is an open or self-cancelling surface whose sign carries no information.
    return signed_volume if abs(signed_volume) > 1.0e-9 * extent ** 3 else None


def build_warp_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    probe_pt: torch.Tensor,
    bbox_diag: float,
    device: str = "cuda",
):
    """Build a ``wp.Mesh`` and detect sign convention.

    Parameters
    ----------
    verts : torch.Tensor
        ``(V, 3)`` float vertex positions (any coordinate system the
        caller chooses; the kernel uses Euclidean distances in this same
        space).
    faces : torch.Tensor
        ``(F, 3)`` int32 triangle indices.
    probe_pt : torch.Tensor
        ``(3,)`` point in the same space as ``verts``, used to detect the sign
        convention only when the surface's signed volume cannot.  It must lie in
        material to be meaningful, which a bounding-box centroid does not for a
        hollow part, so it is a fallback rather than the primary test.
    bbox_diag : float
        Bounding-box diagonal in the same space; used to set ``max_dist``.
    device : str
        CUDA device.

    Returns
    -------
    wp_mesh : wp.Mesh
    negate : bool
        ``True`` if the probe returned positive — caller should flip the
        sign of subsequent SDF queries to obtain "negative inside".
    keep_alive : tuple
        References that must outlive ``wp_mesh`` to keep its arrays valid.
    """
    verts_c = verts.detach().to(device=device, dtype=torch.float32).contiguous()
    faces_c = faces.detach().to(device=device, dtype=torch.int32).contiguous().view(-1)

    wp_points = wp.from_torch(verts_c, dtype=wp.vec3)
    wp_indices = wp.from_torch(faces_c, dtype=wp.int32)
    # support_winding_number=True is REQUIRED for
    # mesh_query_point_sign_winding_number to return correct results on
    # multi-component meshes (e.g. body + interior voids).
    wp_mesh = wp.Mesh(
        points=wp_points, indices=wp_indices, support_winding_number=True
    )

    probe = (
        probe_pt.detach()
        .to(device=device, dtype=torch.float32)
        .reshape(1, 3)
        .contiguous()
    )
    probe_wp = wp.from_torch(probe, dtype=wp.vec3)
    sdf_out = wp.zeros(1, dtype=wp.float32, device=device)
    max_dist = float(bbox_diag) * 10.0
    wp.launch(
        _sdf_kernel,
        dim=1,
        inputs=[wp_mesh.id, probe_wp, sdf_out, max_dist],
        device=device,
    )
    test_val = float(wp.to_torch(sdf_out)[0].item())
    # Standard convention: interior probe should be negative.
    negate = test_val > 0.0

    # The probe is only meaningful when it actually lands in material.  Callers
    # derive it from the bounding-box centroid, which sits in a void for any
    # hollow or thin-walled part -- a bracket, a frame, a shell -- and the test
    # then inverts the whole field: the solver trims nothing as exterior and the
    # reinitialised SDF annihilates the solid.  The mesh's own signed volume
    # settles the orientation without assuming where the material is, so prefer
    # it whenever the surface is closed enough for it to mean anything.
    orientation = _closed_surface_orientation(verts_c, faces_c)
    if orientation is not None:
        negate = orientation < 0.0

    keep_alive = (verts_c, faces_c, wp_points, wp_indices)
    return wp_mesh, negate, keep_alive


# ---------------------------------------------------------------------------
# Nodal SDF query
# ---------------------------------------------------------------------------


def compute_nodal_sdf(
    wp_mesh,
    query_pts: torch.Tensor,
    negate: bool,
    max_dist: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Signed distance from ``query_pts`` to ``wp_mesh``.

    Parameters
    ----------
    wp_mesh : wp.Mesh
    query_pts : torch.Tensor
        ``(M, 3)`` float points in the same coordinate system as the
        mesh vertices.
    negate : bool
        Sign flag from :func:`build_warp_mesh`.
    max_dist : float
        Maximum distance for the BVH closest-point query.  The kernel
        prunes any branch whose minimum distance exceeds this value, so
        a tight ``max_dist`` makes far-away queries cheap.  Queries that
        find no triangle within ``max_dist`` return ``max_dist`` itself
        as a sentinel — callers should detect and handle this case.
    device : str
        CUDA device.

    Returns
    -------
    torch.Tensor
        ``(M,)`` float32 signed distances.  Convention: negative inside.
        Entries equal to ``±max_dist`` are no-hit sentinels.
    """
    pts = query_pts.detach().to(device=device, dtype=torch.float32).contiguous()
    M = int(pts.shape[0])
    if M == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    pts_wp = wp.from_torch(pts, dtype=wp.vec3)
    sdf_wp = wp.zeros(M, dtype=wp.float32, device=device)
    wp.launch(
        _sdf_kernel,
        dim=M,
        inputs=[wp_mesh.id, pts_wp, sdf_wp, float(max_dist)],
        device=device,
    )
    sdf_t = wp.to_torch(sdf_wp).clone()
    if negate:
        sdf_t = -sdf_t
    return sdf_t


def build_warp_mesh_patch(
    verts: torch.Tensor,
    faces: torch.Tensor,
    device: str = "cuda",
):
    """Build a ``wp.Mesh`` BVH from an open surface patch (unsigned distance only).

    Unlike :func:`build_warp_mesh`, no probe-point sign calibration is performed.
    Use this when only unsigned distance queries are needed (e.g. BC region detection).

    Parameters
    ----------
    verts : torch.Tensor
        ``(V, 3)`` float vertex positions.
    faces : torch.Tensor
        ``(F, 3)`` int32 triangle indices.
    device : str

    Returns
    -------
    wp_mesh : wp.Mesh
    keep_alive : tuple
        Must outlive ``wp_mesh`` to keep its arrays valid.
    """
    verts_c = verts.detach().to(device=device, dtype=torch.float32).contiguous()
    faces_c = faces.detach().to(device=device, dtype=torch.int32).contiguous().view(-1)
    wp_points = wp.from_torch(verts_c, dtype=wp.vec3)
    wp_indices = wp.from_torch(faces_c, dtype=wp.int32)
    wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices, support_winding_number=False)
    keep_alive = (verts_c, faces_c, wp_points, wp_indices)
    return wp_mesh, keep_alive


def compute_unsigned_dist(
    wp_mesh,
    query_pts: torch.Tensor,
    max_dist: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Unsigned distance from ``query_pts`` to ``wp_mesh`` via BVH.

    Parameters
    ----------
    wp_mesh : wp.Mesh
    query_pts : torch.Tensor
        ``(M, 3)`` float points in the same coordinate system as the mesh.
    max_dist : float
        BVH search radius; queries with no triangle within this distance
        return ``max_dist`` as a sentinel.
    device : str

    Returns
    -------
    torch.Tensor
        ``(M,)`` float32 unsigned distances.
    """
    pts = query_pts.detach().to(device=device, dtype=torch.float32).contiguous()
    M = int(pts.shape[0])
    if M == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    pts_wp = wp.from_torch(pts, dtype=wp.vec3)
    dist_wp = wp.zeros(M, dtype=wp.float32, device=device)
    wp.launch(
        _unsigned_dist_kernel,
        dim=M,
        inputs=[wp_mesh.id, pts_wp, dist_wp, float(max_dist)],
        device=device,
    )
    return wp.to_torch(dist_wp).clone()


def compute_closest_points(
    wp_mesh,
    query_pts: torch.Tensor,
    max_dist: float,
    device: str = "cuda",
) -> torch.Tensor:
    """Closest point on ``wp_mesh`` for each query point.

    Parameters
    ----------
    wp_mesh : wp.Mesh
    query_pts : torch.Tensor
        ``(M, 3)`` float points in the same coordinate system as the mesh.
    max_dist : float
        BVH search radius; points with no triangle within this distance
        are left unchanged (returned at their original position).
    device : str

    Returns
    -------
    torch.Tensor
        ``(M, 3)`` float32 closest-surface positions.
    """
    pts = query_pts.detach().to(device=device, dtype=torch.float32).contiguous()
    M = int(pts.shape[0])
    if M == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)

    pts_wp = wp.from_torch(pts, dtype=wp.vec3)
    closest_wp = wp.zeros(M, dtype=wp.vec3, device=device)
    wp.launch(
        _closest_point_kernel,
        dim=M,
        inputs=[wp_mesh.id, pts_wp, closest_wp, float(max_dist)],
        device=device,
    )
    return wp.to_torch(closest_wp).clone()
