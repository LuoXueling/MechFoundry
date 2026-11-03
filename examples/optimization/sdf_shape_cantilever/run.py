"""Shape optimization of a 3D cantilever beam via diffused Nitsche + GD.

Initializes from a box trimesh (MC-reinitialized SDF) and optimizes the SDF
to minimize strain energy subject to a soft volume penalty.

Usage:
    conda activate mechfoundry-warp117
    cd src/warp_wrapper
    python run.py --remesh_octree_depth 6 --nx 60 --ny 23 --nz 23
"""

from argparse import ArgumentParser, BooleanOptionalAction
import os
import sys
from pathlib import Path

import trimesh

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(LEAF_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.shape_opt import ShapeOptTrimmed


def main():
    parser = ArgumentParser(description="Shape optimization of a 3D cantilever beam")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=120,
        help="Elements in x (100 for box + 10 padding each side)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=45,
        help="Elements in y (25 for box + 10 padding each side)",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=45,
        help="Elements in z (25 for box + 10 padding each side)",
    )
    parser.add_argument(
        "--pad", type=float, default=0.4, help="Mesh padding around beam"
    )
    parser.add_argument("--width", type=float, default=4.0, help="Beam width")
    parser.add_argument("--height", type=float, default=1.0, help="Beam height")
    parser.add_argument("--depth", type=float, default=1.0, help="Beam depth")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Phase-field half-width (default: auto from mesh)",
    )
    parser.add_argument(
        "--n-iters", "--n_iters", dest="n_iters", type=int, default=None,
        help="Override max optimization iterations",
    )
    parser.add_argument(
        "--w_vol", type=float, default=None, help="Override volume penalty weight"
    )
    parser.add_argument(
        "--vol_frac", type=float, default=None, help="Override target volume fraction"
    )
    parser.add_argument(
        "--volume-as-is",
        action="store_true",
        help="Penalize volume above initial: 0 penalty at V<=V_0, penalty=1 at V=2*V_0 "
        "(ignores --vol_frac)",
    )
    parser.add_argument(
        "--output", type=str, default=str(LEAF_DIR / "output"),
        help="Output directory",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["gd", "mma"],
        help="Optimizer type (default: from config)",
    )
    parser.add_argument(
        "--optimize_relative_tol",
        type=float,
        default=None,
        help="Override optimization relative tolerance",
    )
    parser.add_argument(
        "--gd_lr", type=float, default=None, help="Gradient descent learning rate"
    )
    parser.add_argument(
        "--pf_phi_lr",
        type=float,
        default=None,
        help="Phase-field mode: learning rate for hidden_phi interior update "
        "(defaults to gd_lr)",
    )
    parser.add_argument(
        "--remeshing",
        action=BooleanOptionalAction,
        default=True,
        help="Use adaptive octree remeshing (rebuild outer mesh each iter) — "
        "default on; pass --no-remeshing for the baseline fixed-mesh path",
    )
    parser.add_argument(
        "--remeshing_method",
        choices=["interpolate", "marchingcubes"],
        default="marchingcubes",
        help="geometric_func for adaptive octree (remeshing path only; ignored with --no-remeshing)",
    )
    parser.add_argument(
        "--reinit_sdf_method",
        choices=["sussman", "marchingcubes"],
        default="marchingcubes",
        help="Background-grid SDF reinit method (remeshing path only; ignored with --no-remeshing)",
    )
    parser.add_argument(
        "--sdf_bc_fix_method",
        choices=["mask", "graft", "zip"],
        default="zip",
        help="How to preserve BC region during remesh (remeshing path only; ignored with --no-remeshing)",
    )
    parser.add_argument(
        "--remesh_octree_depth",
        type=int,
        default=7,
        help="Depth of the adaptive octree (remeshing path only; ignored with --no-remeshing)",
    )
    parser.add_argument(
        "--remesh_init_octree_depth",
        type=int,
        default=4,
        help="Initial depth of the adaptive octree (remeshing path only; ignored with --no-remeshing)",
    )
    parser.add_argument(
        "--design-variable",
        choices=["sdf", "phase_field"],
        default="sdf",
        help=(
            "Design variable: 'sdf' (default) drives boundary evolution only. "
            "'phase_field' adds a persistent hidden-φ field that receives the "
            "∂J/∂φ gradient at deep interior nodes (where ∂J/∂SDF saturates) "
            "so the optimizer can carve interior voids; when hidden φ dips "
            "below 0.5, MC + winding-number reveals a new void surface in "
            "the SDF at the next post-opt."
        ),
    )
    parser.add_argument(
        "--phi-reproject-band-k",
        type=float,
        default=3.0,
        help=(
            "Width (in units of ε) of the SDF band around the interface where "
            "hidden φ is boundary-tracked to tanh(SDF/ε).  Outside this band "
            "(deep interior / exterior) φ is left to the gradient update, so "
            "dips and voids accumulate.  Only used with "
            "--design-variable=phase_field."
        ),
    )
    args = parser.parse_args()

    # Validate remeshing option combinations.
    if (
        args.reinit_sdf_method == "marchingcubes"
        and args.remeshing_method != "marchingcubes"
    ):
        parser.error(
            "--reinit_sdf_method=marchingcubes requires --remeshing_method=marchingcubes"
        )
    if args.sdf_bc_fix_method in ("graft", "zip") and (
        args.remeshing_method != "marchingcubes"
        or args.reinit_sdf_method != "marchingcubes"
    ):
        parser.error(
            f"--sdf_bc_fix_method={args.sdf_bc_fix_method} requires "
            "--remeshing_method=marchingcubes and --reinit_sdf_method=marchingcubes"
        )

    # Compute element size for auto-epsilon
    hx = (args.width + 2 * args.pad) / args.nx
    hy = (args.height + 2 * args.pad) / args.ny
    hz = (args.depth + 2 * args.pad) / args.nz
    h_min = min(hx, hy, hz)
    epsilon = args.epsilon if args.epsilon is not None else 0.5 * h_min

    print(f"Element size: hx={hx:.4f}, hy={hy:.4f}, hz={hz:.4f}")
    print(f"Phase-field epsilon: {epsilon:.4f}")

    # All paths (baseline and remeshing) use the same trimesh-based MC init.
    # The baseline differs only in that reinit_sdf=False keeps the initial
    # octree fixed throughout optimization (initial octree path = final).
    mesh_center_phys = [args.width / 2.0, args.height / 2.0, args.depth / 2.0]
    box = trimesh.creation.box(extents=[args.width, args.height, args.depth])
    box.apply_translation(mesh_center_phys)
    # Subdivide twice so triangle size roughly matches the octree leaf size.
    box = box.subdivide().subdivide()

    remesh_clip_bbox = (
        (-args.pad, -args.pad, -args.pad),
        (args.width + args.pad, args.height + args.pad, args.depth + args.pad),
    )

    # Build overrides from CLI args
    if args.remeshing:
        overrides = {
            "phase_field_epsilon": epsilon,
            "remeshing": True,
            "remeshing_method": args.remeshing_method,
            "reinit_sdf_method": args.reinit_sdf_method,
            "sdf_bc_fix_method": args.sdf_bc_fix_method,
            "remesh_octree_depth": args.remesh_octree_depth,
            "remesh_init_octree_depth": args.remesh_init_octree_depth,
            "remeshing_trimesh": {
                "mesh": box,
                "mesh_center": mesh_center_phys,
                "h_target": h_min,
                "clip_bbox": remesh_clip_bbox,
            },
        }
    else:
        # Baseline: MC init gives a clean initial mesh; per-iter Sussman reinit
        # keeps |∇φ|≈1 but the mesh structure never changes.
        # init_octree_depth == octree_depth triggers the uniform-hex-mesh path:
        # build_bg_aligned_hex_mesh builds a bg-grid-aligned hex mesh directly
        # from the clip bbox (no octree machinery), and per-iter the SDF is
        # updated via a direct bg-grid slice — no remesh_to_octree ever called.
        overrides = {
            "phase_field_epsilon": epsilon,
            "remeshing": True,
            "remesh_octree_depth": args.remesh_octree_depth,
            "remesh_init_octree_depth": args.remesh_octree_depth,  # same → hex mesh
            "remeshing_trimesh": {
                "mesh": box,
                "mesh_center": mesh_center_phys,
                "h_target": h_min,
                "clip_bbox": remesh_clip_bbox,
            },
        }
    if args.n_iters is not None:
        overrides["max_optimize_iters"] = args.n_iters
    if args.w_vol is not None:
        overrides["shape_opt_w_vol"] = args.w_vol
    if args.vol_frac is not None:
        overrides["target_volume_fraction"] = args.vol_frac
    if args.volume_as_is:
        overrides["volume_as_is"] = True
    if args.output is not None:
        overrides["output_root"] = args.output
    if args.optimizer is not None:
        overrides["optimizer"] = args.optimizer
    if args.gd_lr is not None:
        overrides["gd_lr"] = args.gd_lr
    if args.pf_phi_lr is not None:
        overrides["pf_phi_lr"] = args.pf_phi_lr
    if args.optimize_relative_tol is not None:
        overrides["optimize_relative_tol"] = args.optimize_relative_tol
    overrides["design_variable"] = args.design_variable
    overrides["phi_reproject_band_k"] = args.phi_reproject_band_k
    problem = ShapeOptTrimmed(args.config, override_params=overrides)
    problem.optimize()


if __name__ == "__main__":
    main()
