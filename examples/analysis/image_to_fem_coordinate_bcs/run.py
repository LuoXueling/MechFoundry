"""Single forward FEM solve on a TripoSG-generated object from an image.

Generates a 3D object from an input image using TripoSG, scales it so that
the height (z-axis) is 3.0 physical units (x and y proportional), then runs
one Nitsche forward solve:
  - Dirichlet BC: fully fixed at the bottom face (z ≤ 0.15)
  - Neumann BC:   downward traction at the top region (z ≥ 2.7, height 0.3)

Usage:
    conda activate mechfoundry-warp117
    python run.py --image path/to/image.png
    python run.py --image path/to/image.png --octree-depth 5
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(LEAF_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/dependency/TripoSG"))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from src.dependency.TripoSG.scripts.briarmbg import BriaRMBG
from src.dependency.TripoSG.scripts.image_process import prepare_image
from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from core.shape_opt import ShapeOptTrimmed
from src.triposg_wrapper.utils import (
    filter_largest_interior_component,
    sdf2surfacemesh,
)


def main():
    parser = argparse.ArgumentParser(description="Image → single forward FEM solve")
    parser.add_argument(
        "--image",
        type=str,
        default=str(REPO_ROOT / "src/dependency/TripoSG/assets/example_data/hjswed.png"),
    )
    parser.add_argument("--octree-depth", type=int, default=7)
    parser.add_argument("--init-octree-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--output", type=str, default=str(LEAF_DIR / "output"))
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/TripoSG"),
    )
    parser.add_argument(
        "--rmbg-weights-dir",
        type=str,
        default=str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/RMBG-1.4"),
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # ------------------------------------------------------------------
    # 1. Load models
    # ------------------------------------------------------------------
    weights = os.path.abspath(args.weights_dir)
    rmbg_weights = os.path.abspath(args.rmbg_weights_dir)

    print(f"Loading RMBG from {rmbg_weights}")
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights).to(device)
    rmbg_net.eval()

    print(f"Loading TripoSG from {weights}")
    pipe = TripoSGPipelineWrapper.from_pretrained(weights).to(device, dtype)

    # ------------------------------------------------------------------
    # 2. Image → latents
    # ------------------------------------------------------------------
    print(f"Processing image: {args.image}")
    img_pil = prepare_image(
        args.image,
        bg_color=np.array([1.0, 1.0, 1.0]),
        rmbg_net=rmbg_net,
    )

    latents = pipe.img2latent(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")

    # Match LatentShapeOpt's geometry preprocessing: it promoted both the VAE
    # and latents to float32 before constructing its initial remeshing surface.
    pipe.vae = pipe.vae.float().eval()
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)
    latents = latents.to(device=device, dtype=torch.float32)

    # Decode once, turn the generated geometry into an explicit physical
    # surface, and hand that surface to ShapeOptTrimmed's mesh path.
    bounds = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    sdf, _ = pipe.latent2sdf(
        latents, bounds=bounds,
        init_octree_depth=args.init_octree_depth,
        octree_depth=args.octree_depth,
    )
    sdf = filter_largest_interior_component(sdf, domain_length=2.0)
    surface = sdf2surfacemesh(sdf, octree_depth=args.octree_depth, bounds=bounds)
    if surface.is_empty:
        raise RuntimeError("TripoSG produced an empty surface")

    physical_height = 3.0
    h_norm = 2.0 / (2 ** args.octree_depth)
    verts = torch.as_tensor(surface.vertices, dtype=torch.float64)
    axis_scale, geo_min = ShapeOptTrimmed._axis_scale_from_verts_and_extents(
        verts, (physical_height, "y"), snap_to_grid=h_norm
    )
    scale = float(axis_scale[0])
    geo_min_np = np.asarray(geo_min, dtype=np.float64)
    surface.vertices = (np.asarray(surface.vertices) - geo_min_np) * scale
    mesh_center = (-geo_min_np * scale).tolist()
    h_target = h_norm * scale

    # ------------------------------------------------------------------
    # 3. FEM parameters
    #
    # Physical scale: y-axis height = 3.0; x and z are proportional.
    # The object spans approximately y ∈ [0, 3].
    #
    # BCs:
    #   Dirichlet: fully fixed at the bottom (y ≤ 0.15)
    #   Neumann:   downward traction at the top (y ≥ 2.7, i.e. top 0.3)
    # ------------------------------------------------------------------
    y_bottom = 0.15  # support thickness at base
    y_top = physical_height - 0.05  # load region starts at y=2.7

    params = {
        # ---- project ----
        "project_name": "img2fem",
        "output_root": args.output,
        "async_io": True,
        # ---- FEM ----
        "dim": 3,
        "q_deg": 2,
        "elem_order": 1,
        # ---- solver ----
        "device": device,
        "output_per_step": 0,
        "relative_tol": 1e-6,
        "absolute_tol": 1e-8,
        "max_iter": 50,
        "line_search_enabled": False,
        "update_damping_limit": 0.0,
        "load_steps": 1,
        "linear_solver": "cg",
        "ew_enabled": False,
        # The depth-7 explicit-surface system can drive JAX-AMG below a useful
        # absolute residual while missing its relative-status gate. The pyamgx
        # adapter uses the established dDDI configuration for this f64 solve.
        "cg_backend_assembled": "pyamgx",
        # ---- material ----
        "E": 10.0,
        "nu": 0.3,
        "B": (0.0, 0.0, 0.0),
        # ---- BCs ----
        "dirichlet_bcs_dict": {
            lambda x, y, z: y <= y_bottom: 0.0,
        },
        "neumann_bcs_dict": {
            lambda x, y, z: y >= y_top: (0.0, -0.005, 0.0),
        },
        # ---- Nitsche ----
        "phase_field_epsilon": 0.5 * h_target,
        "update_beta_per_newton": False,
        "sharp_volumetric": False,
        # ---- precision ----
        "use_mixed_precision": True,
        # ---- surface output ----
        "output_surface_vtu": True,
        "output_surface_glb": True,
        "output_surface_glb_deformed": True,
        "remeshing": True,
        "remeshing_method": "marchingcubes",
        "reinit_sdf_method": "marchingcubes",
        "sdf_bc_fix_method": "mask",
        "remesh_octree_depth": args.octree_depth,
        "remesh_init_octree_depth": args.init_octree_depth,
        "remeshing_trimesh": {
            "mesh": surface,
            "mesh_center": mesh_center,
            "h_target": h_target,
        },
    }

    # ------------------------------------------------------------------
    # 4. Build solver and run a single forward solve
    # ------------------------------------------------------------------
    problem = ShapeOptTrimmed("config", override_params=params)

    print("Running forward FEM solve ...")
    problem.step()
    problem.save(time=0)
    problem.finalize()
    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()
