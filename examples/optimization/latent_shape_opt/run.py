"""End-to-end latent-space shape optimization of a 3D cantilever beam.

Encodes a box mesh into TripoSG latent space, then optimizes the latent
vector to minimize strain energy subject to a volume penalty.

The optimization uses ShapeOptTrimmed infrastructure via the remeshing Path B
pipeline: decoder geo_func → filter → MC reinit → adaptive octree → zip BC fix.
Per-iteration, the updated latents are decoded at octree node positions to refresh
e.dof_values before the standard scatter/reinit/rebuild path.

Usage:
    conda activate mechfoundry-warp117
    cd /path/to/implicit_optimize
    python run.py --init-octree-depth 4 --octree-depth 6 --latent-lr 5e-2 --latent-optimizer adam --disable-pullback
"""

import sys
import os
from argparse import ArgumentParser

import torch
import trimesh
from pathlib import Path

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(LEAF_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "dependency" / "TripoSG"))
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "warp_wrapper"))

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from core.latent_shape_opt import LatentShapeOpt


def main():
    parser = ArgumentParser(description="Latent-space shape optimization")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config",
    )
    parser.add_argument("--octree-depth", type=int, default=7)
    parser.add_argument("--init-octree-depth", type=int, default=4)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--latent-lr", type=float, default=1e-2)
    parser.add_argument("--latent-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--latent-optimizer",
        type=str,
        default="sgd",
        choices=["adam", "rmsprop", "sgd"],
        help="Latent optimizer: 'adam' (AdamW), 'rmsprop' (RMSProp, no weight decay), "
        "or 'sgd' (SGD+momentum, L2-normalized grad). "
        "'rmsprop' ignores --latent-weight-decay",
    )
    parser.add_argument("--decoder-chunk-size", type=int, default=10000)
    parser.add_argument(
        "--n-grad-points",
        type=int,
        default=1000,
        help="Number of boundary points sampled per iteration for latent gradient (0 = all)",
    )
    parser.add_argument("--optimize-relative-tol", type=float, default=2e-3)
    parser.add_argument("--remesh-every", type=int, default=1)
    parser.add_argument(
        "--physical-width",
        type=float,
        default=4.0,
        help="Physical length along the reference axis",
    )
    parser.add_argument(
        "--physical-width-axis",
        type=str,
        default="x",
        choices=["x", "y", "z"],
        help="Axis for proportional scaling (default: x)",
    )
    parser.add_argument(
        "--sdf-penalty-weight",
        type=float,
        default=0.0,
        help="Weight for BC-freeze SDF penalty (0 = disabled)",
    )
    parser.add_argument(
        "--disable-pullback",
        action="store_true",
        help="Disable the pull-back-metric CG correction and use the raw VJP "
        "latent gradient (L2-rescaled to sqrt(N)); the --precond-cg-* knobs are "
        "then unused",
    )
    parser.add_argument(
        "--precond-damping",
        type=float,
        default=1e-3,
        help="Tikhonov damping ratio for the pull-back-metric CG solve: x Rayleigh "
        "quotient (relative to the local curvature scale)",
    )
    parser.add_argument(
        "--precond-cg-iters",
        type=int,
        default=50,
        help="Matrix-free CG iterations for the pull-back-metric "
        "(J^T J + lambda I)^-1 solve",
    )
    parser.add_argument(
        "--precond-cg-rtol",
        type=float,
        default=1e-2,
        help="Relative residual tolerance for the pull-back-metric CG solve "
        "(truncated/early-stopped solve, not meant to converge tightly)",
    )
    parser.add_argument(
        "--precond-cg-no-warm-start",
        action="store_true",
        help="Disable warm-starting the pull-back-metric CG from the previous "
        "iteration's solution",
    )
    parser.add_argument(
        "--precond-cg-matvec",
        choices=["jvp", "reverse"],
        default="jvp",
        help="J^T J matvec for the pull-back-metric CG: 'jvp' forward-JVP+VJP "
        "(default); 'reverse' double-backward (correct but ~46GB, OOMs)",
    )
    parser.add_argument(
        "--target-volume-fraction",
        type=float,
        default=0.5,
        help="Target volume fraction for the volume penalty",
    )
    parser.add_argument(
        "--volume-penalty-weight",
        type=float,
        default=5.0,
        help="Weight of the volume penalty term in the objective (shape_opt_w_vol)",
    )
    parser.add_argument(
        "--volume-as-is",
        action="store_true",
        help="Penalize volume above initial: 0 penalty at V<=V_0, penalty=1 at V=2*V_0 "
        "(ignores --target-volume-fraction)",
    )

    parser.add_argument("--output", type=str, default=str(LEAF_DIR / "output"))
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/TripoSG"),
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # ------------------------------------------------------------------
    # 1. Load TripoSG pipeline
    # ------------------------------------------------------------------
    weights = os.path.abspath(args.weights_dir)
    print(f"Loading TripoSG from {weights}")
    pipe = TripoSGPipelineWrapper.from_pretrained(weights).to(device, dtype)

    # ------------------------------------------------------------------
    # 2. Encode a box mesh → latents
    # ------------------------------------------------------------------
    # Uniform area-weighted surface sampling for encoding (via mesh2latent): the
    # TripoSG encoder needs a well-distributed point cloud.  Subdivided box
    # vertices cluster at the corners/edges and starve the faces, which fragments
    # the decoded shape.  The same encode is reused as the revert path in
    # LatentShapeOpt (re-encoding the previous surface mesh).
    mesh = trimesh.creation.box(extents=[1.75, 0.4375, 0.4375])
    latents = pipe.mesh2latent(mesh, n_points=100000, seed=42, sample_seed=42)
    print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")

    # ------------------------------------------------------------------
    # 3. Build overrides
    # ------------------------------------------------------------------
    overrides = {
        "project_name": "latent_shape_opt_cantilever",
        "output_root": args.output,
        "max_optimize_iters": args.n_iters,
        "output_per_optimize_step": 1,
        "use_mixed_precision": True,
        "async_io": True,
        "target_volume_fraction": args.target_volume_fraction,
        "shape_opt_w_vol": args.volume_penalty_weight,
        "volume_as_is": args.volume_as_is,
        "optimize_relative_tol": args.optimize_relative_tol,
    }

    # ------------------------------------------------------------------
    # 4. Create LatentShapeOpt and run
    # ------------------------------------------------------------------
    problem = LatentShapeOpt(
        args.config,
        override_params=overrides,
        pipe=pipe,
        latents=latents,
        octree_depth=args.octree_depth,
        init_octree_depth=args.init_octree_depth,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        physical_extents=(args.physical_width, args.physical_width_axis),
        latent_lr=args.latent_lr,
        latent_weight_decay=args.latent_weight_decay,
        latent_optimizer=args.latent_optimizer,
        remesh_every=args.remesh_every,
        n_grad_points=args.n_grad_points,
        decoder_chunk_size=args.decoder_chunk_size,
        sdf_penalty_weight=args.sdf_penalty_weight,
        epsilon_to_size_ratio=0.5,
        precond={
            "enabled": not args.disable_pullback,
            "damping": args.precond_damping,
            "cg_iters": args.precond_cg_iters,
            "cg_rtol": args.precond_cg_rtol,
            "cg_warm_start": not args.precond_cg_no_warm_start,
            "cg_matvec": args.precond_cg_matvec,
        },
    )

    problem.optimize()


if __name__ == "__main__":
    main()
