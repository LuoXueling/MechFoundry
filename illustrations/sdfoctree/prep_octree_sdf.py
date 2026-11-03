"""Prep step (conda `mechfoundry-warp117`, uses Warp GPU): build a REAL octree hex mesh
with nodal SDF for the bunny, in the SAME frame as prep_bunny_sdf's cube, and
write it as a legacy VTK UnstructuredGrid for ParaView.

Runs the actual pipeline: bunny surface -> Warp GPU signed-distance query on the
hierarchical (2:1-balanced) octree grid -> sdf2octree hex mesh.  Reuses
bunny.stl + bunny_sdf.meta written by prep_bunny_sdf.py (same normalized frame).

Writes octree.vtk (hex cells + nodal scalar 'sdf').
"""

import os
import sys
import time

import numpy as np
import torch
import trimesh

# Resolve the repo from THIS file, not the cwd: the figures moved into
# per-figure subfolders (AGENTS.md "LAYOUT"), which left ".." pointing at
# illustrations/, where there is no src package.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree  # noqa: E402

INIT_DEPTH = 3          # == octree_step1's l_init
OCTREE_DEPTH = 6        # == octree_step1's l_final

H = float(open("bunny_sdf.meta").readline())          # cube half-side (== prep_bunny_sdf)
mesh = trimesh.load("bunny.stl")


def _gen(depth):
    return trimesh_to_octree(
        mesh,
        bounds=(-H, -H, -H, H, H, H),
        init_octree_depth=INIT_DEPTH,
        octree_depth=depth,
        domain_length=2 * H,
        sdf_truncation=0.01,
    )


def _timed_gen(depth):
    """Pure generation time (ms), GPU-synced, kernels already warm."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    oc = _gen(depth)
    torch.cuda.synchronize()
    return oc, (time.perf_counter() - t0) * 1e3


_gen(OCTREE_DEPTH)                          # warm up (one-time Warp kernel JIT)
oc, gen_ms = _timed_gen(OCTREE_DEPTH)                       # the mesh we render
oc2, gen_ms2 = _timed_gen(OCTREE_DEPTH + 1)                 # one level denser
nodes = oc["nodes"].detach().cpu().numpy().astype(np.float64)
elems = oc["elements"].detach().cpu().numpy().astype(np.int64)
sdf = oc["node_sdf"].detach().cpu().numpy().astype(np.float64)
N, M = len(nodes), len(elems)
N2, M2 = len(oc2["nodes"]), len(oc2["elements"])

with open("octree.vtk", "w") as f:
    f.write("# vtk DataFile Version 3.0\n"
            "bunny octree hex mesh with nodal SDF\nASCII\n"
            "DATASET UNSTRUCTURED_GRID\n")
    f.write(f"POINTS {N} float\n")
    np.savetxt(f, nodes, fmt="%.6f")
    f.write(f"\nCELLS {M} {M * 9}\n")
    np.savetxt(f, np.hstack([np.full((M, 1), 8, np.int64), elems]), fmt="%d")
    f.write(f"\nCELL_TYPES {M}\n")
    np.savetxt(f, np.full(M, 12, np.int64), fmt="%d")
    f.write(f"\nPOINT_DATA {N}\nSCALARS sdf float 1\nLOOKUP_TABLE default\n")
    np.savetxt(f, sdf, fmt="%.6f")

gpu = torch.cuda.get_device_name(0)
with open("octree_stats.txt", "w") as f:                # consumed by compose step
    # depth OCTREE_DEPTH: N, M, ms, gpu ; then one level denser: N2, M2, ms2
    f.write(f"{N}\n{M}\n{gen_ms:.1f}\n{gpu}\n{N2}\n{M2}\n{gen_ms2:.1f}\n")

print(f"wrote octree.vtk  depth {INIT_DEPTH}->{OCTREE_DEPTH}  "
      f"sdf [{sdf.min():.3f}, {sdf.max():.3f}]  on {gpu}")
print(f"  depth {OCTREE_DEPTH}: {N} nodes, {M} hexes, {gen_ms:.1f} ms")
print(f"  depth {OCTREE_DEPTH + 1} (one denser): {N2} nodes, {M2} hexes, {gen_ms2:.1f} ms")
