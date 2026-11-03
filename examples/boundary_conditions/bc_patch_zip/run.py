"""Restore fixed surface patches after a geometry update with the BC zipper."""

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

from core import remeshing
from core.analytical_sdf import build_warp_mesh, compute_nodal_sdf, extract_surface


def bracket_fields(n, length, device):
    lin = torch.linspace(-length / 2, length / 2, n, device=device)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing="ij")

    def capsule(ax, ay, bx, by, radius):
        px, py = x - ax, y - ay
        dx, dy = bx - ax, by - ay
        t = ((px * dx + py * dy) / (dx * dx + dy * dy)).clamp(0, 1)
        return torch.hypot(px - t * dx, py - t * dy) - radius

    def extrude(d2, half_height=0.09):
        pair = torch.stack([d2, z.abs() - half_height], -1)
        return torch.clamp(pair.amax(-1), max=0) + torch.clamp(pair, min=0).norm(dim=-1)

    hub = torch.hypot(x, y) - 0.15
    arms = []
    for degrees in (90, 210, 330):
        angle = np.deg2rad(degrees)
        arms.append(capsule(0, 0, 0.62 * np.cos(angle), 0.62 * np.sin(angle), 0.075))
    initial = extrude(torch.stack([hub, *arms]).amin(0))
    web = capsule(-0.18, -0.12, 0.18, -0.12, 0.12)
    toes = [capsule(x0, -0.12, x0, -0.62, 0.045) for x0 in (-0.14, 0.0, 0.14)]
    updated = extrude(torch.stack([hub, arms[0], web, *toes]).amin(0))
    spacing = length / (n - 1)
    grid = torch.arange(-1.0, 1.0, spacing, device=device)
    xy = grid[grid.abs() <= 0.14]
    zz = grid[grid.abs() <= 0.09 + spacing]
    fx, fy, fz = torch.meshgrid(xy, xy, zz, indexing="ij")
    frozen = torch.stack([fx.reshape(-1), fy.reshape(-1), fz.reshape(-1)], -1)
    return initial, updated, frozen.double(), spacing


def table_fields(n, length, device):
    lin = torch.linspace(-length / 2, length / 2, n, device=device)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing="ij")
    points = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], -1)
    mesh = trimesh.load(
        HERE / "cad/WoodenTable_02/WoodenTable_02_1k.gltf", force="mesh", process=True
    )
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.vertices -= mesh.bounds.mean(0)
    mesh.vertices /= mesh.extents.max()
    wp_mesh, negate, keep = build_warp_mesh(
        torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device),
        torch.as_tensor(mesh.faces, dtype=torch.int32, device=device),
        torch.tensor([0.0, 0.0, 0.45], device=device),
        float(np.linalg.norm(mesh.extents)),
        device,
    )
    initial = compute_nodal_sdf(wp_mesh, points, negate, max_dist=length).reshape(n, n, n)
    cut, twist_top, flat_to = -0.40, 0.23, -0.28
    ramp = ((twist_top - z) / (twist_top - flat_to)).clamp(0, 1)
    ramp = (ramp**3 * (ramp * (ramp * 6 - 15) + 10)).reshape(-1)
    solid = points[(initial.reshape(-1) < 0) & (points[:, 2] <= cut)]
    radius = solid[:, :2].norm(dim=1).mean()
    angles = torch.arange(4, device=device, dtype=points.dtype) * (np.pi / 2)
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], -1)
    for _ in range(40):
        labels = torch.cdist(solid[:, :2], centers).argmin(1)
        for index in range(4):
            selected = solid[labels == index, :2]
            if selected.numel():
                centers[index] = selected.mean(0)
    center = centers[torch.cdist(points[:, :2], centers).argmin(1)]
    theta = np.deg2rad(45.0) * ramp
    rel = points[:, :2] - center
    transformed = torch.stack(
        [center[:, 0] + torch.cos(theta) * rel[:, 0] + torch.sin(theta) * rel[:, 1],
         center[:, 1] - torch.sin(theta) * rel[:, 0] + torch.cos(theta) * rel[:, 1],
         points[:, 2]], -1,
    )
    updated = compute_nodal_sdf(wp_mesh, transformed, negate, max_dist=length).reshape(n, n, n)
    frozen = points[(initial.reshape(-1) < 0) & (points[:, 2] <= cut)].double()
    return initial, updated, frozen, length / (n - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("table", "bracket"), default="table")
    parser.add_argument("--output", type=Path, default=HERE / "stitched.glb")
    args = parser.parse_args()
    device = "cuda"
    if args.case == "table":
        initial, updated, frozen, h = table_fields(161, 1.2, device)
        kwargs = {"aabb_scale": 3.0, "smooth_iters": 600, "taubin_mu": -0.51}
        length = 1.2
    else:
        initial, updated, frozen, h = bracket_fields(113, 2.0, device)
        kwargs, length = {}, 2.0

    vertices0, faces0, _ = extract_surface(initial.contiguous(), length, device)
    vertices1, faces1, _ = extract_surface(updated.contiguous(), length, device)
    patch = remeshing.compute_bc_patch(vertices0.double(), faces0, frozen, 2.0 * h)
    if patch is None:
        raise RuntimeError("no surface patch was found around the fixed nodes")
    base_v, base_f = remeshing._close_open_loops(vertices1.double(), faces1)
    out_v, out_f, labels = remeshing.stitch_bc_patches(
        base_v, base_f, patch["components"], log=print, **kwargs
    )
    vertices = out_v.cpu().numpy()
    faces = out_f.cpu().numpy()
    face_labels = labels.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    colors = np.array([[180, 180, 180, 255], [50, 120, 220, 255], [235, 140, 40, 255]])
    mesh.visual.face_colors = colors[face_labels]
    mesh.export(args.output)
    labels_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        labels_path, vertices=vertices, faces=faces, face_labels=face_labels
    )
    print(
        f"stitched {len(patch['components'])} patches; wrote {args.output} "
        f"and {labels_path}"
    )


if __name__ == "__main__":
    main()
