"""Trimesh surface → SDF grid → octree mesh pipeline.

Drop-in replacement for ``pipe.latent2sdf`` that uses
``trimesh.proximity.signed_distance`` instead of a neural-network decoder.
The resulting ``(grid_sdf, anchors)`` tensors feed directly into ``sdf2octree``.
"""

import types
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
import trimesh.proximity
import warp as wp

from .hierarchical_extraction import hierarchical_extraction


# ---------------------------------------------------------------------------
# Warp kernel for GPU-native signed distance computation
# ---------------------------------------------------------------------------

@wp.kernel
def _sdf_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=wp.float32),
    max_dist: float,
):
    """Compute signed distance from each point to the mesh entirely on GPU."""
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist)
    if query.result:
        cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        sdf_out[tid] = wp.length(cp - p) * query.sign
    else:
        sdf_out[tid] = max_dist


# ---------------------------------------------------------------------------
# Lazy import for sdf2octree (avoids triggering TripoSG dependency chain
# in src/utils.py's top-level ``from .dependency.TripoSG...`` imports).
# ---------------------------------------------------------------------------

_sdf2octree_fn = None


def _get_sdf2octree():
    global _sdf2octree_fn
    if _sdf2octree_fn is not None:
        return _sdf2octree_fn

    utils_path = Path(__file__).parent / "utils.py"
    source = utils_path.read_text()
    lines = source.split("\n")
    clean = []
    for line in lines:
        if line.strip().startswith("from ."):
            clean.append("pass  # skipped relative import")
        else:
            clean.append(line)
    mod = types.ModuleType("_proj_utils")
    mod.__file__ = str(utils_path)
    exec(compile("\n".join(clean), str(utils_path), "exec"), mod.__dict__)
    _sdf2octree_fn = mod.sdf2octree
    return _sdf2octree_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_trimesh_geometric_func(
    mesh: trimesh.Trimesh,
    device: str = "cuda",
    chunk_size: int = 100_000,
    use_warp: bool = True,
) -> Callable:
    """Create a ``geometric_func`` compatible with :func:`hierarchical_extraction`.

    By default uses warp's GPU-native mesh queries (no CPU roundtrip).
    Falls back to CPU-based ``trimesh.proximity.signed_distance`` when
    ``use_warp=False`` or when warp mesh construction fails.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        A (preferably watertight) surface mesh.
    device : str
        CUDA device string, e.g. ``"cuda"`` or ``"cuda:0"``.
    chunk_size : int
        Batch size for the CPU ``signed_distance`` calls (only used in
        fallback mode).
    use_warp : bool
        If True (default), use warp GPU-native mesh queries.

    Returns
    -------
    callable
        ``geometric_func(x)`` where *x* is a ``(1, N, 3)`` float16 GPU
        tensor and the return value is a ``(1, N, 1)`` float16 GPU tensor
        of SDF values.
    """
    if use_warp:
        try:
            return _make_warp_geometric_func(mesh, device)
        except Exception as e:
            print(f"[trimesh_sdf] Warp mesh construction failed ({e}), "
                  f"falling back to CPU trimesh")

    return _make_cpu_geometric_func(mesh, device, chunk_size)


def _make_warp_geometric_func(
    mesh: trimesh.Trimesh,
    device: str = "cuda",
) -> Callable:
    """GPU-native SDF via warp mesh queries — no CPU roundtrip."""
    # Build warp mesh on GPU (one-time cost)
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces.flatten(), dtype=np.int32)
    wp_points = wp.array(vertices, dtype=wp.vec3, device=device)
    wp_indices = wp.array(faces, dtype=wp.int32, device=device)
    wp_mesh = wp.Mesh(points=wp_points, indices=wp_indices)
    mesh_id = wp_mesh.id

    # Compute bounding box diagonal for max_dist
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_dist = bbox_diag * 10.0

    # Auto-detect sign convention by querying a point known to be inside.
    # Avoid the exact centroid — warp's sign_normal query can be ambiguous
    # at highly symmetric points (e.g. center of a rotated cube).
    test_pt = mesh.centroid + 0.01 * (mesh.vertices[0] - mesh.centroid)
    test_pt_wp = wp.array(
        test_pt.reshape(1, 3).astype(np.float32),
        dtype=wp.vec3, device=device,
    )
    sdf_test = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(_sdf_kernel, dim=1,
              inputs=[mesh_id, test_pt_wp, sdf_test, max_dist], device=device)
    test_val = sdf_test.numpy()[0]
    # Standard convention: interior point should be negative
    negate = float(test_val) > 0

    def geometric_func(x: torch.Tensor) -> torch.Tensor:
        # x: (1, N, 3) float16 on GPU
        pts = x.squeeze(0).float().contiguous()  # (N, 3) float32 GPU
        N = pts.shape[0]
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        sdf_wp = wp.zeros(N, dtype=wp.float32, device=device)
        wp.launch(_sdf_kernel, dim=N,
                  inputs=[mesh_id, pts_wp, sdf_wp, max_dist], device=device)
        sdf_t = wp.to_torch(sdf_wp)  # (N,) float32 GPU, zero-copy
        if negate:
            sdf_t = -sdf_t
        return sdf_t.half().unsqueeze(0).unsqueeze(-1)  # (1, N, 1) float16

    # Keep references alive so warp mesh isn't garbage collected
    geometric_func._wp_mesh = wp_mesh
    geometric_func._wp_points = wp_points
    geometric_func._wp_indices = wp_indices

    return geometric_func


def _make_cpu_geometric_func(
    mesh: trimesh.Trimesh,
    device: str = "cuda",
    chunk_size: int = 100_000,
) -> Callable:
    """CPU fallback: trimesh.proximity.signed_distance with GPU↔CPU transfers."""
    centroid = mesh.centroid.reshape(1, 3)
    test_sdf = trimesh.proximity.signed_distance(mesh, centroid)
    negate = float(test_sdf[0]) > 0

    def geometric_func(x: torch.Tensor) -> torch.Tensor:
        points_np = x.squeeze(0).float().cpu().numpy()
        N = points_np.shape[0]
        sdf_all = np.empty(N, dtype=np.float64)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sdf_all[start:end] = trimesh.proximity.signed_distance(
                mesh, points_np[start:end]
            )
        if negate:
            sdf_all = -sdf_all
        sdf_t = torch.tensor(sdf_all, dtype=torch.float16, device=device)
        return sdf_t.unsqueeze(0).unsqueeze(-1)

    return geometric_func


def trimesh_to_sdf(
    mesh: trimesh.Trimesh,
    bounds: Union[Tuple[float, ...], List[float], float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    ),
    init_octree_depth: int = 4,
    octree_depth: int = 5,
    device: str = "cuda",
    chunk_size: int = 100_000,
    sdf_truncation: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute an SDF grid and octree anchors from a trimesh surface.

    This is a drop-in replacement for ``pipe.latent2sdf`` that uses
    ``trimesh.proximity.signed_distance`` instead of the TripoSG decoder.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh (watertight recommended).
    bounds : tuple or float
        Bounding box for the SDF grid — ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
    init_octree_depth : int
        Coarse initial depth for hierarchical extraction.
    octree_depth : int
        Target fine depth.
    device : str
        CUDA device.
    chunk_size : int
        Batch size for the CPU SDF evaluation.

    Returns
    -------
    grid_sdf : torch.Tensor
        Shape ``(2**octree_depth + 1, …)`` — nodal SDF values.
    existing_anchors : torch.Tensor
        Shape ``(2**octree_depth, …)`` — per-cell refinement levels.
    """
    geometric_func = make_trimesh_geometric_func(
        mesh, device=device, chunk_size=chunk_size
    )
    return hierarchical_extraction(
        geometric_func=geometric_func,
        bounds=bounds,
        init_octree_depth=init_octree_depth,
        octree_depth=octree_depth,
        device=device,
        sdf_truncation=sdf_truncation,
    )


def trimesh_to_octree(
    mesh: trimesh.Trimesh,
    bounds: Union[Tuple[float, ...], List[float], float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    ),
    init_octree_depth: int = 4,
    octree_depth: int = 5,
    domain_length: float = 2.0,
    device: str = "cuda",
    chunk_size: int = 100_000,
    sdf_truncation: float = 0.01,
) -> Dict:
    """Full pipeline: trimesh surface → SDF → octree hex-mesh dict.

    The returned dictionary is compatible with ``OctreeNeoHookean``'s
    ``octree_mesh`` constructor parameter.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh.
    bounds : tuple or float
        Bounding box for SDF sampling.
    init_octree_depth, octree_depth : int
        Coarse / fine depths for hierarchical extraction.
    domain_length : float
        Physical side-length of the cubic domain passed to ``sdf2octree``.
        With the default value of 2.0 the mesh spans ``[-1, 1]^3``.
    device : str
        CUDA device.
    chunk_size : int
        Batch size for the CPU SDF evaluation.

    Returns
    -------
    dict
        Keys: ``"nodes"``, ``"elements"``, and optionally
        ``"edge_hanging"``, ``"face_hanging"``, ``"node_sdf"``.
    """
    sdf2octree = _get_sdf2octree()

    grid_sdf, anchors = trimesh_to_sdf(
        mesh,
        bounds,
        init_octree_depth,
        octree_depth,
        device,
        chunk_size,
        sdf_truncation=sdf_truncation,
    )

    nodes, node_sdf, elements, edge_h, face_h = sdf2octree(
        anchors,
        grid_sdf,
        domain_length,
    )

    result: Dict = {
        "nodes": nodes,
        "elements": elements,
        "node_sdf": node_sdf,
        # Octree state for further refinement (e.g. adaptive quad mesh)
        "grid_sdf": grid_sdf,
        "existing_anchors": anchors,
        "octree_depth": octree_depth,
        "bounds": bounds,
        "domain_length": domain_length,
    }
    if edge_h is not None and len(edge_h) > 0:
        result["edge_hanging"] = edge_h
    if face_h is not None and len(face_h) > 0:
        result["face_hanging"] = face_h

    return result
