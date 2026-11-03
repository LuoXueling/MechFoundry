"""Optimize a loaded cylindrical plate supported by four fixed feet."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.bc_surface_selectors import CylinderSelector, HighestSelector, LowestSelector
from core.shape_opt import ShapeOptTrimmed


def build_geometry(radius, height, foot_circle, foot_radius, foot_height, overlap):
    """Build the watertight plate-and-four-feet design domain."""
    body = trimesh.creation.cylinder(radius=radius, height=height, sections=96)
    feet = []
    for angle in np.deg2rad([0.0, 90.0, 180.0, 270.0]):
        foot = trimesh.creation.cylinder(
            radius=foot_radius, height=foot_height + overlap, sections=48
        )
        foot.apply_translation(
            [
                foot_circle * np.cos(angle),
                foot_circle * np.sin(angle),
                -height / 2.0 - foot_height + (foot_height + overlap) / 2.0,
            ]
        )
        feet.append(foot)
    mesh = trimesh.boolean.union([body, *feet])
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise RuntimeError("four-support design domain is not watertight")
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--init-depth", type=int, default=4)
    parser.add_argument("--n-iters", type=int, default=101)
    parser.add_argument(
        "--absolute-tol",
        type=float,
        default=5.0e-7,
        help="Newton absolute residual tolerance for the remeshed solve",
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--foot-circle", type=float, default=0.6)
    parser.add_argument("--foot-radius", type=float, default=0.24)
    parser.add_argument("--foot-height", type=float, default=0.22)
    parser.add_argument("--foot-overlap", type=float, default=0.06)
    parser.add_argument("--frozen-height", type=float, default=0.02)
    parser.add_argument("--top-mode", choices=("full", "disk"), default="full")
    parser.add_argument("--top-radius", type=float, default=0.72)
    parser.add_argument("--top-fraction", type=float, default=0.06)
    parser.add_argument("--traction-x", type=float, default=0.0)
    parser.add_argument("--traction-y", type=float, default=0.0)
    parser.add_argument("--traction-z", type=float, default=-0.02)
    parser.add_argument("--gd-lr", type=float, default=8.0)
    parser.add_argument("--target-volume", type=float, default=0.5)
    parser.add_argument("--volume-weight", type=float, default=5.0)
    parser.add_argument("--bc-fix", choices=("zip", "mask", "graft"), default="zip")
    parser.add_argument("--patch-grow", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--result", type=Path, default=HERE / "result.stl")
    args = parser.parse_args()

    mesh = build_geometry(
        args.radius, args.height, args.foot_circle, args.foot_radius,
        args.foot_height, args.foot_overlap,
    )
    z_top = args.height / 2.0
    z_bottom = -args.height / 2.0 - args.foot_height
    foot_fraction = float(f"{args.frozen_height / (z_top - z_bottom):.4f}")
    fixed = LowestSelector("z", fraction=foot_fraction)
    if args.top_mode == "disk":
        loaded = CylinderSelector(
            (0.0, 0.0, z_top - 0.12),
            (0.0, 0.0, z_top + 0.12),
            args.top_radius,
        )
    else:
        loaded = HighestSelector("z", fraction=args.top_fraction)

    h_target = 3.84 / (2**args.depth)
    padding = 0.3 + 0.5 * h_target
    bounds = np.asarray(mesh.bounds)
    overrides = {
        "project_name": "four_support_shape_opt",
        "output_root": os.fspath(args.output),
        "phase_field_epsilon": 0.5 * h_target,
        "remeshing": True,
        "remeshing_method": "marchingcubes",
        "reinit_sdf_method": "marchingcubes",
        "sdf_bc_fix_method": args.bc_fix,
        "remesh_octree_depth": args.depth,
        "remesh_init_octree_depth": args.init_depth,
        "design_variable": "sdf",
        "phi_reproject_band_k": 3.0,
        "bc_zip_patch_grow": args.patch_grow,
        "output_per_optimize_step": args.save_every,
        "max_optimize_iters": args.n_iters,
        "absolute_tol": args.absolute_tol,
        "gd_lr": args.gd_lr,
        "target_volume_fraction": args.target_volume,
        "shape_opt_w_vol": args.volume_weight,
        "optimize_relative_tol": 1e-9,
        "remeshing_trimesh": {
            "mesh": mesh,
            "mesh_center": [0.0, 0.0, 0.0],
            "h_target": h_target,
            "clip_bbox": (tuple(bounds[0] - padding), tuple(bounds[1] + padding)),
        },
        "dirichlet_bcs_dict": {fixed: 0.0},
        "neumann_bcs_dict": {
            loaded: (args.traction_x, args.traction_y, args.traction_z)
        },
    }
    problem = ShapeOptTrimmed("config", override_params=overrides)
    problem.optimize()
    vertices, faces = problem._get_or_build_mc_surface_physical()
    result = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.result)
    print(f"wrote final post-remesh surface to {args.result}")


if __name__ == "__main__":
    main()
