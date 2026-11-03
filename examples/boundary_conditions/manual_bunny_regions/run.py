"""Select both bunny ears with composable surface-selection primitives."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.bc_surface_selectors import (
    FloodFillSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    HighestSelector,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--output", type=Path, default=HERE / "selected_ears.glb")
    args = parser.parse_args()

    half_side = float((HERE / "bunny_sdf.meta").read_text().splitlines()[0])
    bunny = trimesh.load(HERE / "bunny.stl", force="mesh")
    mirror = np.array([1.0, -1.0, 1.0])
    verts = torch.tensor(
        np.ascontiguousarray(bunny.vertices * mirror), dtype=torch.float32, device="cuda"
    )
    faces = torch.tensor(
        np.ascontiguousarray(bunny.faces[:, [0, 2, 1]]),
        dtype=torch.long,
        device="cuda",
    )
    centroids, normals, areas = _compute_face_attributes(verts, faces)
    adjacency = _build_adj_pairs(faces)

    seed = HighestSelector("z", fraction=0.05)
    seed_mask = seed.select(
        verts, faces, centroids, normals, areas, None, adj_pairs=adjacency
    )
    components = sorted(_cc_by_label_prop(seed_mask, adjacency), key=len, reverse=True)
    if len(components) != 2:
        raise RuntimeError(f"expected two ear-tip components, found {len(components)}")

    tip_centers = [centroids[component].mean(0) for component in components]
    axis = tip_centers[0] - tip_centers[1]
    axis = axis / axis.norm()
    midpoint = 0.5 * (tip_centers[0] + tip_centers[1])
    view = torch.tensor([1.5, -3.0, 1.25], device="cuda")
    first_is_front = float(tip_centers[0] @ view) > float(tip_centers[1] @ view)
    front_half = HalfSpaceSelector(midpoint, axis if first_is_front else -axis)
    back_half = HalfSpaceSelector(midpoint, -axis if first_is_front else axis)

    def ear(half, angle):
        return GrowSelectionSelector(
            FloodFillSelector(seed & half, angle) & half, num_rings=14
        )

    selector = ear(front_half, 4.6) | ear(back_half, 3.4)
    mask = selector.select(
        verts, faces, centroids, normals, areas, None, adj_pairs=adjacency
    )
    h_min = 2.0 * half_side / (2**args.depth)
    selector.resolve(
        verts,
        faces,
        eps_expansion=0.75 * h_min,
    )

    result = trimesh.Trimesh(
        vertices=verts.cpu().numpy(), faces=faces[mask].cpu().numpy(), process=False
    )
    result.export(args.output)
    print(
        f"selected {int(mask.sum())}/{len(faces)} faces; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
