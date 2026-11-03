"""Run the nested-loop and surplus-loop branches of the BC zipper."""

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

from core import remeshing  # noqa: E402

DEVICE = "cuda"
DOMAIN_LENGTH = 2.0
COLORS = np.array(
    [[180, 180, 180, 255], [50, 120, 220, 255], [235, 140, 40, 255]]
)


def background_points(n):
    """Return the reference background lattice and its 2h patch radius."""
    line = torch.linspace(
        -DOMAIN_LENGTH / 2, DOMAIN_LENGTH / 2, n, device=DEVICE
    )
    x, y, z = torch.meshgrid(line, line, line, indexing="ij")
    points = torch.stack([x, y, z], -1).reshape(-1, 3)
    return points, 2.0 * DOMAIN_LENGTH / (n - 1)


def tube(
    radius,
    x0,
    x1,
    center_y=0.0,
    center_z=0.0,
    *,
    rounded_tip=False,
    sections=20,
    ring_spacing=0.03,
    end_radius=None,
    flare=0.07,
):
    """Build the uniformly tessellated reference rod used by both cases."""
    theta = 2.0 * np.pi * np.arange(sections) / sections
    circle = np.stack([np.cos(theta), np.sin(theta)], 1)
    xs = np.linspace(
        x0, x1, max(2, int(round((x1 - x0) / ring_spacing)) + 1)
    )

    def local_radius(x):
        if end_radius is None:
            return radius
        t = float(np.clip((x - (x1 - flare)) / flare, 0.0, 1.0))
        return radius + (end_radius - radius) * t * t * (3.0 - 2.0 * t)

    rings = [
        np.stack(
            [
                np.full(sections, x),
                center_y + local_radius(x) * circle[:, 0],
                center_z + local_radius(x) * circle[:, 1],
            ],
            1,
        )
        for x in xs
    ]
    tip_radius = local_radius(x1)
    if rounded_tip:
        phi = np.linspace(
            0,
            np.pi / 2,
            max(3, int(round(0.5 * np.pi * tip_radius / ring_spacing)) + 1),
        )[1:-1]
        rings.extend(
            np.stack(
                [
                    np.full(sections, x1 + tip_radius * np.sin(angle)),
                    center_y + tip_radius * np.cos(angle) * circle[:, 0],
                    center_z + tip_radius * np.cos(angle) * circle[:, 1],
                ],
                1,
            )
            for angle in phi
        )

    vertices = np.vstack(rings)
    n_rings = len(vertices) // sections
    faces = []
    for ring in range(n_rings - 1):
        a, b = ring * sections, (ring + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces.extend([[a + i, b + i, b + j], [a + i, b + j, a + j]])

    def flat_cap(base, x, outer_radius, flip):
        nonlocal vertices
        n_radial = max(1, int(round(outer_radius / ring_spacing)))
        previous = base
        for ring in range(1, n_radial):
            r = outer_radius * (1.0 - ring / n_radial)
            current = len(vertices)
            vertices = np.vstack(
                [
                    vertices,
                    np.stack(
                        [
                            np.full(sections, x),
                            center_y + r * circle[:, 0],
                            center_z + r * circle[:, 1],
                        ],
                        1,
                    ),
                ]
            )
            for i in range(sections):
                j = (i + 1) % sections
                pair = [
                    [previous + i, current + i, current + j],
                    [previous + i, current + j, previous + j],
                ]
                faces.extend([face[::-1] for face in pair] if flip else pair)
            previous = current
        center = len(vertices)
        vertices = np.vstack([vertices, [[x, center_y, center_z]]])
        for i in range(sections):
            face = [center, (i + 1) % sections + previous, i + previous]
            faces.append(face if flip else face[::-1])

    last = (n_rings - 1) * sections
    if rounded_tip:
        tip = len(vertices)
        vertices = np.vstack([vertices, [[x1 + tip_radius, center_y, center_z]]])
        faces.extend(
            [tip, last + i, last + (i + 1) % sections] for i in range(sections)
        )
    else:
        flat_cap(last, x1, tip_radius, flip=False)
    flat_cap(0, x0, local_radius(x0), flip=True)

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_volume:
        raise RuntimeError("fabricated rod is not a watertight volume")
    return mesh


def tensors(mesh):
    if not mesh.is_watertight:
        raise RuntimeError("fabricated surface is not watertight")
    return (
        torch.as_tensor(mesh.vertices, dtype=torch.float64, device=DEVICE),
        torch.as_tensor(mesh.faces, dtype=torch.int32, device=DEVICE),
    )


def stitch(initial, updated, frozen, radius):
    initial_vertices, initial_faces = tensors(initial)
    updated_vertices, updated_faces = tensors(updated)
    patch = remeshing.compute_bc_patch(
        initial_vertices, initial_faces, frozen.double(), radius
    )
    if patch is None:
        raise RuntimeError("no fixed patch was found")
    base_vertices, base_faces = remeshing._close_open_loops(
        updated_vertices, updated_faces
    )
    vertices, faces, labels = remeshing.stitch_bc_patches(
        base_vertices,
        base_faces,
        patch["components"],
        log=lambda message: print(message.strip()),
    )
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2).sort(1).values
    counts = torch.unique(edges, dim=0, return_counts=True)[1]
    if bool((counts != 2).any()):
        raise RuntimeError("stitched result is not watertight")
    return vertices, faces, labels


def nested_loop_case():
    points, radius = background_points(129)
    tilt = np.deg2rad(18.0)
    axis = torch.tensor(
        [np.cos(tilt), 0.0, np.sin(tilt)], dtype=points.dtype, device=DEVICE
    )
    bar_radius, half_length = 0.20, 0.19
    bar = tube(
        bar_radius, -half_length, half_length, sections=96, ring_spacing=0.0035
    )
    bore = tube(
        0.085,
        -half_length - 0.05,
        half_length + 0.05,
        sections=96,
        ring_spacing=0.0035,
    )
    updated = trimesh.boolean.difference([bar, bore])
    transform = trimesh.transformations.rotation_matrix(-tilt, [0, 1, 0])
    initial = bar.copy()
    initial.apply_transform(transform)
    updated.apply_transform(transform)
    axial = points @ axis
    radial = (points - axial.unsqueeze(1) * axis).norm(dim=1)
    frozen = points[
        (radial < bar_radius)
        & (axial > -half_length)
        & (axial < -half_length + 0.05)
    ]
    return stitch(initial, updated, frozen, radius)


def surplus_loop_case():
    points, radius = background_points(73)
    trunk_radius, trunk_x0, split_x = 0.21, -0.55, -0.52
    strut_x1, head_x0, head_x1 = -0.26, -0.30, -0.18
    struts = (
        (90.0, 0.12, 0.07, 0.085),
        (210.0, 0.12, 0.07, 0.085),
        (330.0, 0.15, 0.045, 0.055),
    )
    branches = [
        tube(
            r,
            split_x,
            strut_x1,
            d * np.cos(np.deg2rad(angle)),
            d * np.sin(np.deg2rad(angle)),
            sections=32,
            end_radius=end_r,
            flare=0.06,
        )
        for angle, d, r, end_r in struts
    ]
    head = tube(
        trunk_radius,
        head_x0,
        head_x1,
        rounded_tip=True,
        sections=72,
        ring_spacing=0.015,
    )
    initial = trimesh.boolean.union(
        [
            tube(
                trunk_radius,
                trunk_x0,
                head_x0,
                sections=72,
                ring_spacing=0.015,
            ),
            head,
        ]
    )
    updated = trimesh.boolean.union(
        [tube(trunk_radius, trunk_x0, split_x, sections=32), *branches, head]
    )
    frozen = points[
        (points[:, 0] > trunk_x0)
        & (points[:, 0] < trunk_x0 + 0.05)
        & (points[:, 1:].norm(dim=1) < trunk_radius)
    ]
    return stitch(initial, updated, frozen, radius)


def save_result(name, result, output_dir):
    vertices, faces, labels = (item.detach().cpu().numpy() for item in result)
    np.savez_compressed(
        output_dir / f"{name}_stitched.npz",
        vertices=vertices,
        faces=faces,
        face_labels=labels,
    )
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.face_colors = COLORS[labels]
    mesh.export(output_dir / f"{name}_stitched.glb")
    print(
        f"{name}: {len(vertices)} vertices, {len(faces)} faces, "
        f"labels={np.bincount(labels, minlength=3)[:3].tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("nested", "surplus", "both"), default="both"
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case in {"nested", "both"}:
        save_result("nested", nested_loop_case(), args.output_dir)
    if args.case in {"surplus", "both"}:
        save_result("surplus", surplus_loop_case(), args.output_dir)


if __name__ == "__main__":
    main()
