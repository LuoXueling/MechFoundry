"""GPU-native shape-diameter segmentation for surface boundary conditions.

The terminology follows CGAL's Surface Mesh Segmentation package: its ``SDF``
is the *shape diameter function*, not the signed-distance field used elsewhere
in this project.  When the signed-distance grid that produced the marching-
cubes surface is available, it is reused as the primary ray-intersection
backend.  A Warp mesh BVH is the fallback for surface-only callers.
"""

from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

import torch
import warp as wp


@dataclass
class SignedDistanceGrid:
    """Physical-space description of an existing dense signed-distance grid."""

    values: torch.Tensor
    origin: torch.Tensor
    spacing: torch.Tensor


@dataclass
class SurfaceSegmentation:
    """All GPU-resident products of one surface segmentation."""

    shape_diameter: torch.Tensor
    probabilities: torch.Tensor
    cluster_ids: torch.Tensor
    segment_ids: torch.Tensor
    num_segments: int
    # ``shape_diameter`` is min-max normalized to [0, 1], as CGAL orders
    # segments by it; these are the smoothed diameters its 0 and 1 stand for,
    # in the mesh's length units, so lo + d * (hi - lo) is a thickness.
    shape_diameter_range: Optional[Tuple[float, float]] = None

    @classmethod
    def from_labels(cls, segment_ids, shape_diameter=None, device=None):
        """Wrap an already-computed per-face labelling, skipping the solve.

        Callers that persisted a segmentation -- the workflow saves both arrays
        in ``segments.npz`` -- can reuse it instead of paying for an identical
        re-segmentation.  ``probabilities`` and ``cluster_ids`` are intermediate
        products of the GMM fit and are not reconstructed; they stay ``None``.
        """
        labels = torch.as_tensor(segment_ids, dtype=torch.long, device=device)
        if labels.ndim != 1:
            raise ValueError("segment_ids must be one label per face")
        if labels.numel() and int(labels.amin()) < 0:
            raise ValueError("segment_ids must be nonnegative")
        diameter = (
            torch.as_tensor(shape_diameter, dtype=torch.float32, device=labels.device)
            if shape_diameter is not None else None
        )
        return cls(diameter, None, None, labels,
                   int(labels.amax()) + 1 if labels.numel() else 0)


@wp.kernel
def _push_relabel_fused_kernel(
    offsets: wp.array(dtype=wp.int64), dst: wp.array(dtype=wp.int64),
    reverse: wp.array(dtype=wp.int64), residual: wp.array(dtype=wp.float64),
    height: wp.array(dtype=wp.int32), excess: wp.array(dtype=wp.float64),
    node_color: wp.array(dtype=wp.int32), active_color: int,
    unreachable_height: int, eps: float, local_steps: int,
):
    """Asynchronous multi-operation discharge, one owner thread per vertex."""
    u = wp.tid()
    if node_color[u] != active_color:
        return
    step = int(0)
    while step < local_steps:
        remaining = excess[u]
        if remaining <= eps:
            return
        minimum = unreachable_height
        e = offsets[u]
        end = offsets[u + 1]
        while e < end:
            available = residual[e]
            if available > eps:
                v = dst[e]
                minimum = wp.min(minimum, height[v])
                if height[u] == height[v] + 1 and remaining > eps:
                    delta = wp.min(remaining, available)
                    wp.atomic_sub(residual, e, delta)
                    wp.atomic_add(residual, reverse[e], delta)
                    wp.atomic_sub(excess, u, delta)
                    wp.atomic_add(excess, v, delta)
                    remaining -= delta
            e += wp.int64(1)
        if remaining > eps and minimum < unreachable_height:
            height[u] = wp.max(height[u], minimum + 1)
        step += 1
@wp.func
def _grid_sample_trilinear(
    grid: wp.array3d[float], p: wp.vec3, origin: wp.vec3, spacing: wp.vec3,
    nx: int, ny: int, nz: int,
):
    d = p - origin
    q = wp.vec3(d[0] / spacing[0], d[1] / spacing[1], d[2] / spacing[2])
    i = int(wp.floor(q[0]))
    j = int(wp.floor(q[1]))
    k = int(wp.floor(q[2]))
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return float(1.0e30)
    x = q[0] - float(i)
    y = q[1] - float(j)
    z = q[2] - float(k)
    c00 = grid[i, j, k] * (1.0 - x) + grid[i + 1, j, k] * x
    c10 = grid[i, j + 1, k] * (1.0 - x) + grid[i + 1, j + 1, k] * x
    c01 = grid[i, j, k + 1] * (1.0 - x) + grid[i + 1, j, k + 1] * x
    c11 = grid[i, j + 1, k + 1] * (1.0 - x) + grid[i + 1, j + 1, k + 1] * x
    c0 = c00 * (1.0 - y) + c10 * y
    c1 = c01 * (1.0 - y) + c11 * y
    return c0 * (1.0 - z) + c1 * z


@wp.func
def _cone_direction(inward: wp.vec3, ray: int, num_rays: int, cone_half_angle: float):
    # CGAL's default deterministic Vogel disk sampler: R=i/n, golden angle,
    # then map the disk into the cone using cos/sin of its half-angle.  Match
    # Cartesian_kernel::Construct_base_vector_3 exactly: discard the smallest
    # normal coordinate, then base2 = cross(normal, base1).  Vogel sampling is
    # deterministic but not rotationally invariant at finite ray counts.
    ax = wp.abs(inward[0])
    ay = wp.abs(inward[1])
    az = wp.abs(inward[2])
    tangent = wp.vec3(0.0, -inward[2], inward[1])
    if ay <= ax and ay <= az:
        tangent = wp.vec3(-inward[2], 0.0, inward[0])
    elif az < ax and az < ay:
        tangent = wp.vec3(-inward[1], inward[0], 0.0)
    tangent = wp.normalize(tangent)
    bitangent = wp.normalize(wp.cross(inward, tangent))
    radius = float(ray) / float(num_rays)
    phi = 2.399963229728653 * float(ray)
    radial = tangent * wp.cos(phi) + bitangent * wp.sin(phi)
    return wp.normalize(inward * wp.cos(cone_half_angle)
                        + radial * (wp.sin(cone_half_angle) * radius))


@wp.kernel
def _grid_shape_diameter_kernel(
    grid: wp.array3d[float], centroids: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3), origin: wp.vec3, spacing: wp.vec3,
    nx: int, ny: int, nz: int, num_rays: int, cone_half_angle: float,
    step_size: float, max_steps: int, hit_out: wp.array(dtype=float),
):
    tid = wp.tid()
    face = tid // num_rays
    ray = tid - face * num_rays
    inward = -wp.normalize(normals[face])
    direction = _cone_direction(inward, ray, num_rays, cone_half_angle)
    start_offset = step_size * 0.5
    p0 = centroids[face] + inward * start_offset
    prev_t = float(0.0)
    prev = _grid_sample_trilinear(grid, p0, origin, spacing, nx, ny, nz)
    hit = float(-1.0)
    s = int(1)
    while s <= max_steps and hit < 0.0:
        t = float(s) * step_size
        value = _grid_sample_trilinear(grid, p0 + direction * t, origin, spacing, nx, ny, nz)
        if value > 1.0e20:
            s = max_steps + 1
        elif prev <= 0.0 and value >= 0.0:
            denom = value - prev
            alpha = float(0.5)
            if wp.abs(denom) > 1.0e-12:
                alpha = -prev / denom
            hit = prev_t + wp.clamp(alpha, 0.0, 1.0) * step_size + start_offset
        else:
            prev = value
            prev_t = t
            s += 1
    hit_out[tid] = hit


@wp.kernel
def _bvh_shape_diameter_kernel(
    mesh_id: wp.uint64, centroids: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3), num_rays: int,
    cone_half_angle: float, offset: float, max_dist: float,
    hit_out: wp.array(dtype=float),
):
    tid = wp.tid()
    face_id = tid // num_rays
    ray = tid - face_id * num_rays
    inward = -wp.normalize(normals[face_id])
    direction = _cone_direction(inward, ray, num_rays, cone_half_angle)
    # CGAL starts at the exact centroid and excludes the source primitive from
    # its AABB traversal.  Warp has no skip predicate, so query once at the
    # centroid and, only if numerical self-intersection returns the source
    # face, step past it along this ray and query again.
    query = wp.mesh_query_ray(mesh_id, centroids[face_id], direction, max_dist)
    traveled = float(0.0)
    if query.result and query.face == face_id:
        traveled = offset
        query = wp.mesh_query_ray(
            mesh_id, centroids[face_id] + direction * offset,
            direction, max_dist - offset
        )
    if query.result and wp.dot(direction, query.normal) > 0.0:
        hit_out[tid] = query.t + traveled
    else:
        hit_out[tid] = -1.0


def _robust_ray_mean(hits: torch.Tensor) -> torch.Tensor:
    """CGAL's uniform-ray MAD rejection, batched over all faces."""
    valid = hits > 0.0
    count = valid.sum(1)

    # torch.nanmedian returns the lower middle value for an even population;
    # CGAL averages both middle values.  Sorting 25 rays is inexpensive and
    # reproduces that convention without leaving the GPU.
    values = torch.where(valid, hits, torch.full_like(hits, float("inf"))).sort(1).values
    lo = ((count - 1).clamp_min(0) // 2)[:, None]
    hi = (count.clamp_min(1) // 2)[:, None]
    median = 0.5 * (values.gather(1, lo) + values.gather(1, hi)).squeeze(1)
    median = torch.where(count > 0, median, torch.full_like(median, -1.0))
    dev = torch.where(valid, torch.abs(hits - median[:, None]),
                      torch.full_like(hits, float("inf")))
    sorted_dev = dev.sort(1).values
    mad = 0.5 * (sorted_dev.gather(1, lo) + sorted_dev.gather(1, hi)).squeeze(1)
    keep = valid & (dev <= 1.5 * mad[:, None])
    kept_count = keep.sum(1)
    result = (torch.where(keep, hits, torch.zeros_like(hits)).sum(1)
              / kept_count.clamp_min(1))
    # CGAL's facet wrapper stores -1 when no ray survives intersection/acute
    # testing; postprocessing recognizes and repairs that sentinel.  When rays
    # exist but MAD rejects all of them, CGAL returns their median instead.
    result = torch.where(count == 0, torch.full_like(result, -1.0),
                         torch.where(kept_count > 0, result, median))
    return result


def compute_shape_diameter(
    verts: torch.Tensor,
    faces: torch.Tensor,
    centroids: torch.Tensor,
    normals: torch.Tensor,
    *,
    num_rays: int = 25,
    cone_angle_deg: float = 120.0,
    sdf_grid: Optional[SignedDistanceGrid] = None,
    wp_mesh=None,
) -> torch.Tensor:
    """Compute one robust local-thickness value per face on the GPU."""
    device = str(verts.device)
    if verts.device.type != "cuda":
        raise RuntimeError("shape-diameter segmentation requires CUDA tensors")
    F = int(faces.shape[0])
    hits_wp = wp.empty(F * num_rays, dtype=wp.float32, device=device)
    cent_wp = wp.from_torch(centroids.float().contiguous(), dtype=wp.vec3)
    norm_wp = wp.from_torch(normals.float().contiguous(), dtype=wp.vec3)
    half_angle = math.radians(cone_angle_deg * 0.5)

    if sdf_grid is not None:
        grid = sdf_grid.values.detach().to(device=device, dtype=torch.float32).contiguous()
        origin = sdf_grid.origin.detach().to(device=device, dtype=torch.float32)
        spacing = sdf_grid.spacing.detach().to(device=device, dtype=torch.float32)
        grid_wp = wp.from_torch(grid, dtype=wp.float32)
        min_h = float(spacing.min().item())
        max_dist = float(torch.linalg.norm(spacing * (torch.tensor(grid.shape, device=verts.device) - 1)))
        wp.launch(
            _grid_shape_diameter_kernel,
            dim=F * num_rays,
            inputs=[grid_wp, cent_wp, norm_wp, wp.vec3(*origin.tolist()),
                    wp.vec3(*spacing.tolist()), *map(int, grid.shape), int(num_rays),
                    float(half_angle), 0.5 * min_h,
                    int(math.ceil(max_dist / (0.5 * min_h))) + 2, hits_wp],
            device=device,
        )
    else:
        if wp_mesh is None:
            from .analytical_sdf import build_warp_mesh_patch
            wp_mesh, keep_alive = build_warp_mesh_patch(verts, faces, device=device)
            # Keep references alive through the synchronous launch and clone.
            _ = keep_alive
        bbox_diag = float(torch.linalg.norm(verts.amax(0) - verts.amin(0)))
        offset = max(1.0e-6 * bbox_diag, 1.0e-7)
        wp.launch(
            _bvh_shape_diameter_kernel,
            dim=F * num_rays,
            inputs=[wp_mesh.id, cent_wp, norm_wp, int(num_rays), float(half_angle),
                    offset, 1.01 * bbox_diag, hits_wp],
            device=device,
        )

    hits = wp.to_torch(hits_wp).clone().reshape(F, num_rays)
    return _robust_ray_mean(hits)


def _fill_missing(values: torch.Tensor, adj_pairs: torch.Tensor):
    """Apply CGAL's single neighbor-average pass and minimum fallback.

    The batched pass is intentionally synchronous, making the result
    independent of face descriptor iteration order.
    """
    out = values.clone()
    missing = ~torch.isfinite(out) | (out < 0.0)
    src, dst = adj_pairs[:, 0], adj_pairs[:, 1]
    good_edge = ~missing[src]
    sums = torch.zeros_like(out)
    counts = torch.zeros_like(out)
    sums.index_add_(0, dst[good_edge], out[src[good_edge]])
    counts.index_add_(0, dst[good_edge], torch.ones_like(out[src[good_edge]]))
    can_fill = missing & (counts > 0)
    out[can_fill] = sums[can_fill] / counts[can_fill]
    unresolved = missing & ~can_fill
    if bool(unresolved.any()):
        valid = out[~missing]
        fallback = valid.amin() if valid.numel() else out.new_tensor(0.0)
        out[unresolved] = fallback
    return out


def _bilateral_smooth(values: torch.Tensor, adj_pairs: torch.Tensor, window: int):
    """One CGAL-equivalent bilateral pass over an edge-geodesic window."""
    num_faces = values.numel()
    roots = torch.arange(num_faces, device=values.device, dtype=torch.long)
    nodes = roots
    seen = roots * num_faces + roots
    pair_roots = [roots]
    pair_nodes = [nodes]
    pair_dist = [torch.zeros_like(roots)]
    src, dst = adj_pairs[:, 0], adj_pairs[:, 1]
    order = src.argsort(stable=True)
    src, dst = src[order], dst[order]
    degree = torch.bincount(src, minlength=num_faces)
    offsets = torch.cumsum(degree, 0) - degree
    for distance in range(1, window + 1):
        reps = degree[nodes]
        total = int(reps.sum().item())
        if total == 0:
            break
        expanded_roots = torch.repeat_interleave(roots, reps)
        starts = torch.repeat_interleave(torch.cumsum(reps, 0) - reps, reps)
        edge_ids = (torch.repeat_interleave(offsets[nodes], reps)
                    + torch.arange(total, device=values.device) - starts)
        expanded_nodes = dst[edge_ids]
        keys = torch.unique(expanded_roots * num_faces + expanded_nodes, sorted=True)
        pos = torch.searchsorted(seen, keys)
        safe = pos.clamp_max(seen.numel() - 1)
        fresh = (pos == seen.numel()) | (seen[safe] != keys)
        keys = keys[fresh]
        if keys.numel() == 0:
            break
        roots, nodes = keys // num_faces, keys % num_faces
        pair_roots.append(roots); pair_nodes.append(nodes)
        pair_dist.append(torch.full_like(roots, distance))
        seen = torch.cat((seen, keys)).sort().values

    roots = torch.cat(pair_roots)
    nodes = torch.cat(pair_nodes)
    distance = torch.cat(pair_dist).to(values.dtype)
    delta = values[nodes] - values[roots]
    count = torch.bincount(roots, minlength=num_faces).to(values.dtype)
    deviation2 = torch.zeros_like(values)
    deviation2.index_add_(0, roots, delta.square())
    sigma_r = 1.5 * torch.sqrt(deviation2 / count.clamp_min(1.0))
    sigma_s = float(window) / 2.0
    stable = sigma_r > 0.0
    denom_r = sigma_r[roots].clamp_min(torch.finfo(values.dtype).eps)
    weight = torch.exp(-0.5 * (distance / sigma_s) ** 2
                       -0.5 * (delta / denom_r) ** 2)
    weighted = torch.zeros_like(values)
    norm = torch.zeros_like(values)
    weighted.index_add_(0, roots, weight * values[nodes])
    norm.index_add_(0, roots, weight)
    smoothed = weighted / norm.clamp_min(torch.finfo(values.dtype).eps)
    return torch.where(stable, smoothed, values)


def gpu_kmeans(features: torch.Tensor, k: int, iterations: int = 10):
    """Deterministic GPU K-means for scalar or vector-valued face features.

    This is the clustering initializer used by shape-diameter segmentation,
    generalized from one scalar per face to ``(faces, channels)`` features so
    other surface descriptors can reuse it without inheriting graph-cut
    regularization or connected-component relabeling.
    """
    values = features[:, None] if features.ndim == 1 else features
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be a nonempty (faces, channels) tensor")
    n = values.shape[0]
    if not 2 <= int(k) <= n:
        raise ValueError("cluster count must be between 2 and the number of faces")
    generator = torch.Generator(device=values.device)
    generator.manual_seed(1340818006)

    def squared_distance(centers):
        # The expanded identity avoids an (N,K,C) temporary for PartField's
        # 448-channel descriptors.
        result = (values.square().sum(1, keepdim=True)
                  + centers.square().sum(1)[None, :]
                  - 2.0 * values @ centers.T)
        return result.clamp_min_(0.0)

    centers = [values[torch.randint(n, (), generator=generator, device=values.device)]]
    closest2 = values.new_full((n,), float("inf"))
    for _ in range(1, int(k)):
        delta2 = (values - centers[-1]).square().sum(1)
        closest2 = torch.minimum(closest2, delta2)
        total = closest2.sum()
        if float(total) == 0.0:
            center_matrix = torch.stack(centers)
            index = squared_distance(center_matrix).amin(1).argmax()
        else:
            target = torch.rand((), generator=generator, device=values.device) * total
            index = torch.searchsorted(torch.cumsum(closest2, 0), target, right=True)
            index = index.clamp_max(n - 1)
        centers.append(values[index])
    centers = torch.stack(centers)

    labels = torch.full((n,), -1, dtype=torch.long, device=values.device)
    for _ in range(int(iterations)):
        distances2 = squared_distance(centers)
        new_labels = distances2.argmin(1)
        changed = bool((new_labels != labels).any())
        labels = new_labels
        mass = torch.bincount(labels, minlength=int(k)).to(values.dtype)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, values)
        centers = torch.where(
            (mass > 0)[:, None], sums / mass.clamp_min(1)[:, None], centers
        )
        if not changed:
            break
    return labels, centers, squared_distance(centers)


def _gmm_1d(values: torch.Tensor, k: int, iterations: int = 10):
    """CGAL-style one-run K-means initialization followed by 1-D GMM EM.

    CGAL seeds its private RNG with a fixed value.  Torch's generator is not
    bit-compatible with CGAL::Random, so center sampling is the one explicitly
    deterministic substitution; the K-means and EM update rules match CGAL.
    """
    n = values.numel()
    labels, means, _ = gpu_kmeans(values, k, iterations)
    means = means[:, 0]

    mass = torch.bincount(labels, minlength=k).to(values.dtype)
    variance = torch.zeros(k, dtype=values.dtype, device=values.device)
    variance.index_add_(0, labels, (values - means[labels]).square())
    deviation = torch.sqrt(variance / mass.clamp_min(1))
    mix = torch.full_like(means, 1.0 / k)

    previous = values.new_tensor(-float("inf"))
    resp = torch.empty((n, k), dtype=values.dtype, device=values.device)
    eps = torch.finfo(values.dtype).tiny
    for _ in range(iterations):
        # CGAL's density deliberately omits the common sqrt(2*pi) factor and
        # defines a zero-deviation Gaussian as an equality indicator.
        delta = values[:, None] - means[None, :]
        nonzero = deviation > 0
        density = torch.where(
            nonzero[None, :],
            torch.exp(-0.5 * (delta / deviation.clamp_min(eps)[None, :]).square())
            / deviation.clamp_min(eps)[None, :],
            (delta == 0).to(values.dtype),
        ) * mix[None, :]
        total = density.sum(1).clamp_min(eps)
        resp = density / total[:, None]

        membership = resp.sum(0)
        means = (resp * values[:, None]).sum(0) / membership.clamp_min(eps)
        variance = (resp * (values[:, None] - means[None, :]).square()).sum(0)
        deviation = torch.sqrt(variance / membership.clamp_min(eps))
        mix = membership / n

        density = torch.where(
            (deviation > 0)[None, :],
            torch.exp(-0.5 * ((values[:, None] - means[None, :])
                              / deviation.clamp_min(eps)[None, :]).square())
            / deviation.clamp_min(eps)[None, :],
            (values[:, None] == means[None, :]).to(values.dtype),
        ) * mix[None, :]
        likelihood = torch.log(density.sum(1).clamp_min(eps)).sum()
        resp = density / density.sum(1).clamp_min(eps)[:, None]
        if bool((likelihood - previous) < 1.0e-3 * likelihood.abs()):
            break
        previous = likelihood

    order = means.argsort()
    return resp[:, order]


def _cgal_dihedral_costs(
    verts: torch.Tensor,
    faces: torch.Tensor,
    adj_pairs: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """CGAL's oriented four-point dihedral graph weight for every face pair."""
    f0, f1 = adj_pairs[:, 0], adj_pairs[:, 1]
    tri0, tri1 = faces[f0].long(), faces[f1].long()
    edge_a = tri0
    edge_b = tri0[:, [1, 2, 0]]
    a_in_other = (edge_a[:, :, None] == tri1[:, None, :]).any(2)
    b_in_other = (edge_b[:, :, None] == tri1[:, None, :]).any(2)
    edge_slot = (a_in_other & b_in_other).to(torch.int64).argmax(1)
    row = torch.arange(adj_pairs.shape[0], device=faces.device)
    a_id = edge_a[row, edge_slot]
    b_id = edge_b[row, edge_slot]
    c_id = tri0[row, (edge_slot + 2) % 3]
    shared_in_other = ((tri1 == a_id[:, None]) | (tri1 == b_id[:, None]))
    d_slot = (~shared_in_other).to(torch.int64).argmax(1)
    d_id = tri1[row, d_slot]

    a, b, c, d = verts[a_id], verts[b_id], verts[c_id], verts[d_id]
    ab, ac, ad = b - a, c - a, d - a
    ab_ac = torch.linalg.cross(ab, ac)
    ab_ad = torch.linalg.cross(ab, ad)
    y = torch.linalg.vector_norm(ab, dim=1) * (ab_ac * ab_ad).sum(1)
    z = (torch.linalg.cross(ab, ab_ac) * ab_ad).sum(1)
    normalized_angle = torch.atan2(z, y) / math.pi
    concave = normalized_angle > 0.0
    angle = 1.0 + torch.where(concave, -normalized_angle, normalized_angle)
    angle = torch.where(concave, angle, 0.08 * angle)
    return -torch.log(angle.clamp_min(5.0e-6)) * (100.0 * float(smoothing))


def _alpha_move_terms(
    unary_cost: torch.Tensor,
    labels: torch.Tensor,
    alpha: int,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_weight: torch.Tensor,
):
    """Construct an exact binary Potts alpha-expansion energy on the GPU.

    Binary state zero keeps the current label and state one takes ``alpha``.
    The returned nonnegative terminal costs and n-links differ from the
    original move energy only by the returned scalar constant.
    """
    row = torch.arange(labels.numel(), device=labels.device)
    d0 = unary_cost[row, labels].clone()
    d1 = unary_cost[:, int(alpha)].clone()
    lu, lv = labels[edge_u], labels[edge_v]
    a = labels.new_full((), int(alpha))
    e00 = edge_weight * (lu != lv)
    e01 = edge_weight * (lu != a)
    e10 = edge_weight * (a != lv)
    # For a metric Potts model this is nonnegative (submodularity).
    nlink = 0.5 * (e01 + e10 - e00)
    add_u = e10 - e00 - nlink
    add_v = e01 - e00 - nlink
    d1.index_add_(0, edge_u, add_u)
    d1.index_add_(0, edge_v, add_v)
    constant = e00.sum()

    # Terminal capacities must be nonnegative.  Per-node shifts are constants
    # and therefore preserve the minimizing cut exactly.
    shift = torch.minimum(d0, d1)
    d0 -= shift
    d1 -= shift
    constant += shift.sum()
    return d0, d1, edge_u, edge_v, nlink.clamp_min(0.0), constant


def _parallel_graph_coloring(num_nodes, edge_u, edge_v):
    """Deterministically color a sparse graph using parallel Torch rounds."""
    device = edge_u.device
    color = torch.full((num_nodes,), -1, dtype=torch.int32, device=device)
    degree = torch.bincount(torch.cat((edge_u, edge_v)), minlength=num_nodes)
    max_colors = int(degree.max()) + 1 if num_nodes else 1
    node_id = torch.arange(num_nodes, device=device)
    while bool((color < 0).any()):
        forbidden = torch.zeros((num_nodes, max_colors), dtype=torch.bool, device=device)
        for c in range(max_colors):
            marked = torch.zeros(num_nodes, dtype=torch.int32, device=device)
            marked.index_add_(0, edge_v, (color[edge_u] == c).to(torch.int32))
            marked.index_add_(0, edge_u, (color[edge_v] == c).to(torch.int32))
            forbidden[:, c] = marked > 0
        tentative = (~forbidden).to(torch.int32).argmax(1).to(torch.int32)
        uncolored = color < 0
        conflict = uncolored[edge_u] & uncolored[edge_v]
        conflict &= tentative[edge_u] == tentative[edge_v]
        loser = torch.maximum(edge_u[conflict], edge_v[conflict])
        blocked = torch.zeros(num_nodes, dtype=torch.int32, device=device)
        blocked[loser] = 1
        winner = uncolored & (blocked == 0)
        color[winner] = tentative[winner]
    if bool((color[edge_u] == color[edge_v]).any()):
        raise RuntimeError("parallel graph coloring produced an edge conflict")
    return color


# Each frontier pass of the traversals below is a couple of cheap GPU ops, but
# reading convergence back to the host costs a device synchronisation that
# dominates them.  The passes are idempotent once the frontier stops moving --
# a converged state relaxes to itself -- so a stride only spends a few extra
# no-op passes while removing all but 1/stride of the synchronisations.
_FRONTIER_CHECK_STRIDE = 8


def _relabel_converged(updated, previous, iteration: int) -> bool:
    """Report convergence only on stride boundaries, to bound host syncs."""
    if (iteration % _FRONTIER_CHECK_STRIDE) != _FRONTIER_CHECK_STRIDE - 1:
        return False
    return bool((updated == previous).all())


def _warp_binary_min_cut(
    d0: torch.Tensor,
    d1: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    node_color: Optional[torch.Tensor] = None,
    max_rounds: int = 4096,
):
    """Minimize a binary Potts energy with GPU Warp push--relabel.

    Returns a boolean tensor where ``False`` chooses ``d0`` and ``True``
    chooses ``d1``.  All graph storage and solver state remain on the GPU.
    """
    if d0.device.type != "cuda":
        raise RuntimeError("Warp binary min-cut requires CUDA tensors")
    n = d0.numel()
    source, sink, total_nodes = n, n + 1, n + 2
    device = d0.device
    nodes = torch.arange(n, device=device, dtype=torch.long)
    if node_color is None:
        node_color = _parallel_graph_coloring(n, edge_u, edge_v)
    num_colors = int(node_color.max()) + 1
    m = edge_u.numel()
    src = torch.cat((edge_u, edge_v,
                     torch.full_like(nodes, source), nodes,
                     nodes, torch.full_like(nodes, sink)))
    dst = torch.cat((edge_v, edge_u,
                     nodes, torch.full_like(nodes, source),
                     torch.full_like(nodes, sink), nodes))
    residual = torch.cat((edge_weight, edge_weight,
                          d1, torch.zeros_like(d1),
                          d0, torch.zeros_like(d0))).double().contiguous()
    reverse = torch.cat((
        torch.arange(m, 2 * m, device=device),
        torch.arange(0, m, device=device),
        torch.arange(2 * m + n, 2 * m + 2 * n, device=device),
        torch.arange(2 * m, 2 * m + n, device=device),
        torch.arange(2 * m + 3 * n, 2 * m + 4 * n, device=device),
        torch.arange(2 * m + 2 * n, 2 * m + 3 * n, device=device),
    ))
    order = src.argsort(stable=True)
    inverse_order = torch.empty_like(order)
    inverse_order[order] = torch.arange(order.numel(), device=device)
    src, dst, residual = src[order], dst[order], residual[order]
    reverse = inverse_order[reverse[order]]
    original_capacity = residual.clone()
    degree = torch.bincount(src, minlength=total_nodes)
    offsets = torch.cat((degree.new_zeros(1), degree.cumsum(0))).contiguous()

    excess = torch.zeros(total_nodes, dtype=torch.float64, device=device)
    source_arcs = torch.nonzero(src == source, as_tuple=True)[0]
    initial = residual[source_arcs].clone()
    residual[source_arcs] = 0.0
    residual[reverse[source_arcs]] += initial
    excess.index_add_(0, dst[source_arcs], initial)
    excess[source] = -initial.sum()

    capacity_scale = max(
        1.0, float(d0.amax()), float(d1.amax()),
        float(edge_weight.amax()) if edge_weight.numel() else 0.0,
    )
    eps = max(1.0e-12, 32.0 * torch.finfo(torch.float64).eps * capacity_scale)

    def global_relabel():
        # Exact shortest residual distance to the sink.  Relaxing reversed
        # residual arcs is a GPU-parallel breadth-first traversal.
        unreachable = total_nodes + 1
        h = torch.full((total_nodes,), unreachable, dtype=torch.int32, device=device)
        h[sink] = 0
        # ``residual`` is fixed for the whole relabel, so the residual-arc masks
        # and the gathered endpoint indices are loop invariants.  Recomputing
        # them per frontier pass costs two boolean-mask gathers over every arc,
        # which dominates the pass itself.
        sink_arcs = (residual > eps) & (src != source)
        sink_src, sink_dst = src[sink_arcs], dst[sink_arcs]
        source_arcs = (residual > eps) & (dst != sink)
        source_src, source_dst = src[source_arcs], dst[source_arcs]
        for iteration in range(total_nodes):
            candidate = torch.full_like(h, unreachable)
            candidate.scatter_reduce_(0, sink_src, h[sink_dst] + 1,
                                      reduce="amin", include_self=True)
            updated = torch.minimum(h, candidate)
            done = _relabel_converged(updated, h, iteration)
            h = updated
            if done:
                break
        sink_unreachable = h == unreachable
        to_source = torch.full_like(h, unreachable)
        to_source[source] = 0
        for iteration in range(total_nodes):
            candidate = torch.full_like(to_source, unreachable)
            candidate.scatter_reduce_(0, source_src, to_source[source_dst] + 1,
                                      reduce="amin", include_self=True)
            updated = torch.minimum(to_source, candidate)
            done = _relabel_converged(updated, to_source, iteration)
            to_source = updated
            if done:
                break
        return_height = torch.where(
            to_source < unreachable, total_nodes + to_source,
            torch.full_like(to_source, 2 * total_nodes + 1),
        )
        h = torch.where(sink_unreachable, return_height, h)
        h[source] = total_nodes
        return h

    height = global_relabel()

    converged = False
    offsets_wp = wp.from_torch(offsets, dtype=wp.int64)
    dst_wp = wp.from_torch(dst.contiguous(), dtype=wp.int64)
    reverse_wp = wp.from_torch(reverse.contiguous(), dtype=wp.int64)
    residual_wp = wp.from_torch(residual, dtype=wp.float64)
    height_wp = wp.from_torch(height, dtype=wp.int32)
    excess_wp = wp.from_torch(excess, dtype=wp.float64)
    wp.load_module(device=str(device))
    color_wp = wp.from_torch(node_color.contiguous(), dtype=wp.int32)
    wp.capture_begin(device=str(device), force_module_load=False)
    for active_color in range(num_colors):
        wp.launch(_push_relabel_fused_kernel, dim=n,
                  inputs=[offsets_wp, dst_wp, reverse_wp, residual_wp,
                          height_wp, excess_wp, color_wp, active_color,
                          2 * total_nodes + 1, eps, 64], device=str(device))
    discharge_graph = wp.capture_end(device=str(device))
    for round_index in range(max_rounds):
        wp.capture_launch(discharge_graph)
        # A global relabel scans the complete residual graph and is much more
        # expensive than a captured discharge round.  The local kernel already
        # performs 64 push/relabel steps per launch, so refresh heights only
        # after 256 launches instead of every eight.
        if round_index % 256 == 255:
            if not bool((excess[:n] > eps).any()):
                converged = True
                break
            height.copy_(global_relabel())
    if not bool((excess[:n] > eps).any()):
        converged = True
    if not converged:
        active = excess[:n] > eps
        probe = int(torch.nonzero(active, as_tuple=True)[0][0])
        pe = slice(int(offsets[probe]), int(offsets[probe + 1]))
        residual_neighbors = [
            (int(v), float(c), int(height[v]))
            for v, c in zip(dst[pe][residual[pe] > eps][:8],
                            residual[pe][residual[pe] > eps][:8])
        ]
        raise RuntimeError(
            "Warp push-relabel did not converge "
            f"({int(active.sum())} active, max excess={float(excess[:n].max()):.6g}, "
            f"max height={int(height[:n].max())}, probe={probe}, "
            f"probe_height={int(height[probe])}, arcs={residual_neighbors})"
        )

    # The sink side is state one.  Determine the source residual component
    # with parallel frontier propagation.
    reachable = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    reachable[source] = 1
    # Flow is final here, so the residual arcs are the same on every pass.
    open_arcs = residual > eps
    open_src, open_dst = src[open_arcs], dst[open_arcs]
    for iteration in range(total_nodes):
        updated = reachable.clone()
        updated.scatter_reduce_(0, open_dst, reachable[open_src],
                                reduce="amax", include_self=True)
        done = _relabel_converged(updated, reachable, iteration)
        reachable = updated
        if done:
            break
    source_side = reachable.bool()
    cut_capacity = original_capacity[source_side[src] & ~source_side[dst]].sum()
    flow = excess[sink]
    dual_tol = max(1.0e-8, 1.0e-9 * float(cut_capacity.abs().clamp_min(1.0)))
    if abs(float(flow - cut_capacity)) > dual_tol:
        raise RuntimeError(
            "Warp push-relabel failed max-flow/min-cut duality "
            f"(flow={float(flow):.9g}, cut={float(cut_capacity):.9g})"
        )
    return ~source_side[:n]


def _potts_energy(unary_cost, labels, edge_u, edge_v, edge_weight):
    row = torch.arange(labels.numel(), device=labels.device)
    return (unary_cost[row, labels].sum()
            + (edge_weight * (labels[edge_u] != labels[edge_v])).sum())


def _alpha_expansion_labels(unary_cost, labels, edge_u, edge_v, edge_weight):
    """CGAL-order alpha-expansion sweeps using Warp binary graph cuts."""
    labels = labels.clone()
    best_energy = _potts_energy(unary_cost, labels, edge_u, edge_v, edge_weight)
    debug = os.environ.get("BC_SEGMENT_DEBUG", "0") == "1"
    if debug:
        q = torch.quantile(edge_weight, edge_weight.new_tensor([0., .01, .5, .99, 1.]))
        print(f"alpha-expansion initial_energy={float(best_energy):.9g} "
              f"labels={torch.bincount(labels).cpu().tolist()} "
              f"edge_weight_q={q.cpu().tolist()}")
    tolerance = 1.0e-10
    node_color = _parallel_graph_coloring(labels.numel(), edge_u, edge_v)
    while True:
        success = False
        for alpha in range(unary_cost.shape[1]):
            d0, d1, eu, ev, ew, _ = _alpha_move_terms(
                unary_cost, labels, alpha, edge_u, edge_v, edge_weight
            )
            take_alpha = _warp_binary_min_cut(
                d0, d1, eu, ev, ew, node_color=node_color
            )
            candidate = torch.where(
                take_alpha, torch.full_like(labels, alpha), labels
            )
            energy = _potts_energy(unary_cost, candidate, edge_u, edge_v, edge_weight)
            if bool((best_energy - energy) > energy.abs() * tolerance):
                if debug:
                    changed = int((candidate != labels).sum())
                    print(f"alpha={alpha} energy={float(best_energy):.9g}->{float(energy):.9g} "
                          f"changed={changed}")
                labels = candidate
                best_energy = energy
                success = True
        if not success:
            return labels


def _regularize_labels(
    probabilities, verts, faces, adj_pairs, smoothing, iterations=20
):
    if smoothing <= 0.0:
        return probabilities.argmax(1)
    unary_cost = -torch.log(probabilities.clamp_min(5.0e-6))
    unary_cost.clamp_(min=torch.finfo(unary_cost.dtype).eps)
    labels = unary_cost.argmin(1)
    undirected = adj_pairs[:, 0] < adj_pairs[:, 1]
    pairs = adj_pairs[undirected]
    edge_weight = _cgal_dihedral_costs(verts, faces, pairs, smoothing)
    return _alpha_expansion_labels(
        unary_cost, labels, pairs[:, 0], pairs[:, 1], edge_weight
    )


def _connected_segment_ids(
    cluster_ids: torch.Tensor,
    adj_pairs: torch.Tensor,
    shape_diameter: Optional[torch.Tensor] = None,
):
    F = cluster_ids.numel()
    src, dst = adj_pairs[:, 0], adj_pairs[:, 1]
    same = cluster_ids[src] == cluster_ids[dst]
    src, dst = src[same], dst[same]
    labels = torch.arange(F, dtype=torch.long, device=cluster_ids.device)
    # Pointer jumping makes this much faster than a plain breadth-first walk,
    # while the convergence condition (rather than a fixed cap) is exact.
    while True:
        candidates = torch.minimum(labels[src], labels[dst])
        new = labels.clone()
        new.scatter_reduce_(0, src, candidates, reduce="amin", include_self=True)
        new.scatter_reduce_(0, dst, candidates, reduce="amin", include_self=True)
        new = new[new]
        if bool((new == labels).all()):
            labels = new
            break
        labels = new
    _, compact = torch.unique(labels, sorted=True, return_inverse=True)
    if shape_diameter is not None:
        count = torch.bincount(compact).to(shape_diameter.dtype)
        mean = torch.zeros_like(count)
        mean.index_add_(0, compact, shape_diameter)
        mean /= count
        order = mean.argsort(stable=True)
        remap = torch.empty_like(order)
        remap[order] = torch.arange(order.numel(), device=order.device)
        compact = remap[compact]
    return compact


def segment_surface(
    verts: torch.Tensor, faces: torch.Tensor, centroids: torch.Tensor,
    normals: torch.Tensor, adj_pairs: torch.Tensor, *, num_clusters: int = 5,
    smoothing: float = 0.25, num_rays: int = 25,
    cone_angle_deg: float = 120.0, sdf_grid: Optional[SignedDistanceGrid] = None,
    wp_mesh=None,
) -> SurfaceSegmentation:
    # Shape-diameter rays assume outward-oriented faces and cast opposite the
    # face normal.  STL stores no orientation metadata, and otherwise valid
    # closed meshes (including the paper bunny) may arrive consistently inward
    # wound.  Reverse only the local triangle winding; face IDs and all masks
    # remain unchanged.  The signed-volume reduction is fully GPU resident.
    tri = verts[faces.long()]
    signed_volume6 = (tri[:, 0] * torch.linalg.cross(tri[:, 1], tri[:, 2])).sum()
    if bool(signed_volume6 < 0.0):
        oriented_faces = faces[:, [0, 2, 1]].contiguous()
        oriented_normals = -normals
        # A caller-provided BVH has the original normals; rebuild it from the
        # locally oriented triangles so the acute-hit predicate agrees.
        wp_mesh = None
    else:
        oriented_faces = faces
        oriented_normals = normals
    raw = compute_shape_diameter(
        verts, oriented_faces, centroids, oriented_normals, num_rays=num_rays,
        cone_angle_deg=cone_angle_deg, sdf_grid=sdf_grid, wp_mesh=wp_mesh,
    )
    raw = _fill_missing(raw, adj_pairs)
    window = int(math.floor(math.sqrt(max(int(faces.shape[0]), 1) / 2000.0)) + 1)
    smooth = _bilateral_smooth(raw, adj_pairs, window)
    linear = (smooth - smooth.amin()) / (smooth.amax() - smooth.amin()).clamp_min(1.0e-8)
    normalized = torch.log(5.0 * linear + 1.0) / math.log(6.0)
    probabilities = _gmm_1d(normalized, int(num_clusters))
    cluster_ids = _regularize_labels(
        probabilities, verts, oriented_faces, adj_pairs, smoothing=float(smoothing)
    )
    # CGAL orders connected segments by the mean linearly normalized SDF,
    # while the logarithmic transform is used only for GMM fitting.
    segment_ids = _connected_segment_ids(cluster_ids, adj_pairs, linear)
    return SurfaceSegmentation(
        linear, probabilities, cluster_ids, segment_ids,
        int(segment_ids.amax().item()) + 1,
        (float(smooth.amin().item()), float(smooth.amax().item())),
    )
