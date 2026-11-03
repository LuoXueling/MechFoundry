"""Convert the prepared bunny surface to one adaptive octree VTK mesh."""

import os
import sys
import argparse

import numpy as np
import trimesh
from pathlib import Path

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
os.chdir(LEAF_DIR)
sys.path.insert(0, os.fspath(REPO_ROOT))
from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-depth", type=int, default=3)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()
    half_side = float((LEAF_DIR / "bunny_sdf.meta").read_text().splitlines()[0])
    mesh = trimesh.load(LEAF_DIR / "bunny.stl")
    octree = trimesh_to_octree(
        mesh,
        bounds=(-half_side,) * 3 + (half_side,) * 3,
        init_octree_depth=args.init_depth,
        octree_depth=args.depth,
        domain_length=2 * half_side,
        sdf_truncation=0.01,
    )
    nodes = octree["nodes"].detach().cpu().numpy().astype(np.float64)
    elems = octree["elements"].detach().cpu().numpy().astype(np.int64)
    sdf = octree["node_sdf"].detach().cpu().numpy().astype(np.float64)
    n_nodes, n_elems = len(nodes), len(elems)
    with (LEAF_DIR / "octree.vtk").open("w") as output:
        output.write("# vtk DataFile Version 3.0\nbunny octree hex mesh with nodal SDF\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        output.write(f"POINTS {n_nodes} float\n")
        np.savetxt(output, nodes, fmt="%.6f")
        output.write(f"\nCELLS {n_elems} {n_elems * 9}\n")
        np.savetxt(output, np.hstack([np.full((n_elems, 1), 8, np.int64), elems]), fmt="%d")
        output.write(f"\nCELL_TYPES {n_elems}\n")
        np.savetxt(output, np.full(n_elems, 12, np.int64), fmt="%d")
        output.write(f"\nPOINT_DATA {n_nodes}\nSCALARS sdf float 1\nLOOKUP_TABLE default\n")
        np.savetxt(output, sdf, fmt="%.6f")
    print(f"wrote octree.vtk: depth {args.init_depth}->{args.depth}, {n_nodes} nodes, {n_elems} hexes")


if __name__ == "__main__":
    main()
