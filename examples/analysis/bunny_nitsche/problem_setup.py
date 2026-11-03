"""Numerical setup shared by the bunny Nitsche runner.

This module deliberately has no import-time work. ``build_problem`` recreates
the reference bunny octree and the exact ear/base selector recipes formerly
embedded in the illustration capture script.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.bc_surface_selectors import (  # noqa: E402
    FloodFillSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    HighestSelector,
    LowestSelector,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)
from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree  # noqa: E402


def _ear_selector(vertices, faces):
    """Return the reference two-ear selector on the unmirrored z-up bunny."""
    centroids, normals, areas = _compute_face_attributes(vertices, faces)
    adjacency = _build_adj_pairs(faces)
    seed = HighestSelector("z", fraction=0.05)
    seed_mask = seed.select(
        vertices, faces, centroids, normals, areas, None, adj_pairs=adjacency
    )
    components = sorted(_cc_by_label_prop(seed_mask, adjacency), key=len, reverse=True)
    if len(components) != 2:
        raise RuntimeError(f"expected two ear-tip components, found {len(components)}")

    centers = [centroids[component].mean(0).cpu().numpy() for component in components]
    axis = centers[0] - centers[1]
    axis /= np.linalg.norm(axis)
    midpoint = 0.5 * (centers[0] + centers[1])
    view_direction = np.array([1.5, -3.0, 1.25])
    first_is_front = float(centers[0] @ view_direction) > float(
        centers[1] @ view_direction
    )
    front = HalfSpaceSelector(midpoint, axis if first_is_front else -axis)
    back = HalfSpaceSelector(midpoint, -axis if first_is_front else axis)

    def one_ear(half, angle):
        return GrowSelectionSelector(
            FloodFillSelector(seed & half, angle) & half, num_rings=14
        )

    # These values are tied to the unmirrored smoothed bunny tessellation.
    return one_ear(front, 3.4) | one_ear(back, 4.6)


def build_problem():
    """Build the depth-6 reference octree and resolved surface selectors."""
    device = "cuda"
    init_depth, depth = 3, 6
    half_side = float((HERE / "bunny_sdf.meta").read_text().splitlines()[0])
    bunny = trimesh.load(HERE / "bunny.stl", force="mesh")
    domain_length = 2.0 * half_side
    h_min = domain_length / (2**depth)
    epsilon = 0.5 * h_min

    generated = trimesh_to_octree(
        bunny,
        bounds=(-half_side, -half_side, -half_side,
                half_side, half_side, half_side),
        init_octree_depth=init_depth,
        octree_depth=depth,
        domain_length=domain_length,
        sdf_truncation=0.01,
    )
    octree = {
        "nodes": generated["nodes"].to(torch.float32).to(device),
        "elements": generated["elements"].long().to(device),
        "sdf": generated["node_sdf"].to(torch.float32).to(device),
    }
    expected = (45082, 35442)
    actual = (len(octree["nodes"]), len(octree["elements"]))
    if actual != expected:
        raise RuntimeError(
            f"reference bunny octree changed: expected {expected}, got {actual}"
        )

    vertices = torch.as_tensor(
        np.ascontiguousarray(bunny.vertices), dtype=torch.float32, device=device
    )
    faces = torch.as_tensor(
        np.ascontiguousarray(bunny.faces), dtype=torch.long, device=device
    )
    ears = _ear_selector(vertices, faces)
    base = LowestSelector("z", fraction=0.05)
    # The solver only resolves selectors that arrive unresolved, so this sets the
    # BC band; keep it equal to NitscheNeoHookean._bc_match_tolerance().
    resolve_tolerance = max(1.5 * epsilon, 0.75 * h_min)
    for selector in (ears, base):
        selector.resolve(vertices, faces, eps_expansion=resolve_tolerance)

    return {
        "bunny": bunny,
        "octree": octree,
        "domain_length": domain_length,
        "epsilon": epsilon,
        "ear_selector": ears,
        "base_selector": base,
    }
