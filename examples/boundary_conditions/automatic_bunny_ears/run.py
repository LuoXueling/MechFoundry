"""Identify bunny ears with GPU shape-diameter segmentation."""

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
    HighestSelector,
    ShapeDiameterSegmentSelector,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)


def structured_grid_side(path):
    """Read the physical side length from the prepared legacy VTK header."""
    dimensions = spacing = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("DIMENSIONS"):
                dimensions = tuple(map(int, line.split()[1:]))
            elif line.startswith("SPACING"):
                spacing = tuple(map(float, line.split()[1:]))
            elif line == "LOOKUP_TABLE default":
                break
    if dimensions is None or spacing is None:
        raise RuntimeError(f"invalid structured-points file: {path}")
    return spacing[0] * (dimensions[0] - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--rays", type=int, default=25)
    parser.add_argument("--smoothing", type=float, default=0.26)
    parser.add_argument("--output", type=Path, default=HERE / "selected_ears.glb")
    args = parser.parse_args()

    side = structured_grid_side(HERE / "bunny_sdf.vtk")
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
    if len(_cc_by_label_prop(seed_mask, adjacency)) != 2:
        raise RuntimeError("expected two ear-tip seed components")

    selector = ShapeDiameterSegmentSelector(
        seed,
        num_clusters=args.clusters,
        smoothing=args.smoothing,
        num_rays=args.rays,
        max_segment_area_fraction=0.12,
        adjacent_segments=1,
        adjacent_direction=(0.0, 0.0, -1.0),
    )
    h_min = side / (2**args.depth)
    selector.resolve(
        verts,
        faces,
        eps_expansion=0.75 * h_min,
    )
    mask = selector._selected_face_mask
    if int(mask.sum()) <= int(seed_mask.sum()) or int(mask.sum()) >= len(faces) // 4:
        raise RuntimeError("shape-diameter selection did not isolate the ears")

    result = trimesh.Trimesh(
        vertices=verts.cpu().numpy(), faces=faces[mask].cpu().numpy(), process=False
    )
    result.export(args.output)
    print(
        f"selected {int(mask.sum())}/{len(faces)} faces across "
        f"{selector._surface_segmentation.num_segments} segments; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
