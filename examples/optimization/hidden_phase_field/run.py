"""Run four-support shape optimization with delayed interior void creation."""

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

from core.bc_surface_selectors import HighestSelector, LowestSelector
from core.shape_opt import ShapeOptTrimmed


def build_geometry(radius, height, foot_circle, foot_radius, foot_height, overlap):
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
        raise RuntimeError("phase-field design domain is not watertight")
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--init-depth", type=int, default=4)
    parser.add_argument("--n-iters", type=int, default=91)
    parser.add_argument("--phi-start-iter", type=int, default=35)
    parser.add_argument("--phi-lr", type=float, default=12.0)
    parser.add_argument("--gd-lr", type=float, default=8.0)
    parser.add_argument("--target-volume", type=float, default=0.5)
    parser.add_argument("--volume-weight", type=float, default=5.0)
    parser.add_argument(
        "--absolute-tol",
        type=float,
        default=5.0e-7,
        help=(
            "Newton absolute residual tolerance; remeshed depth-7 solves "
            "can plateau around 3e-7"
        ),
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--foot-circle", type=float, default=0.6)
    parser.add_argument("--foot-radius", type=float, default=0.24)
    parser.add_argument("--foot-height", type=float, default=0.22)
    parser.add_argument("--foot-overlap", type=float, default=0.06)
    parser.add_argument("--frozen-height", type=float, default=0.02)
    parser.add_argument("--top-fraction", type=float, default=0.06)
    parser.add_argument("--traction-z", type=float, default=-0.02)
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
    z_foot_bottom = -args.height / 2.0 - args.foot_height
    foot_fraction = float(f"{args.frozen_height / (z_top - z_foot_bottom):.4f}")
    fixed = LowestSelector("z", fraction=foot_fraction)
    loaded = HighestSelector("z", fraction=args.top_fraction)
    h_target = 3.84 / (2**args.depth)
    padding = 0.3 + 0.5 * h_target
    bounds = np.asarray(mesh.bounds)

    overrides = {
        "project_name": "hidden_phase_field",
        "output_root": os.fspath(args.output),
        "phase_field_epsilon": 0.5 * h_target,
        "remeshing": True,
        "remeshing_method": "marchingcubes",
        "reinit_sdf_method": "marchingcubes",
        "sdf_bc_fix_method": "zip",
        "remesh_octree_depth": args.depth,
        "remesh_init_octree_depth": args.init_depth,
        "remeshing_trimesh": {
            "mesh": mesh,
            "mesh_center": [0.0, 0.0, 0.0],
            "h_target": h_target,
            "clip_bbox": (tuple(bounds[0] - padding), tuple(bounds[1] + padding)),
        },
        "bc_zip_patch_grow": args.patch_grow,
        "design_variable": "phase_field",
        "phi_reproject_band_k": 3.0,
        "pf_phi_lr": args.phi_lr,
        "pf_phi_start_iter": args.phi_start_iter,
        "optimizer": "gd",
        "gd_lr": args.gd_lr,
        "max_optimize_iters": args.n_iters,
        "optimize_relative_tol": 1e-8,
        "absolute_tol": args.absolute_tol,
        "output_per_optimize_step": args.save_every,
        "output_surface_glb": True,
        "target_volume_fraction": args.target_volume,
        "shape_opt_w_vol": args.volume_weight,
        "dirichlet_bcs_dict": {fixed: 0.0},
        "neumann_bcs_dict": {loaded: (0.0, 0.0, args.traction_z)},
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
