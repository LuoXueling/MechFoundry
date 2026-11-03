"""Single forward FEM solve on a TripoSG-generated object using automatic BC identification.

Same analysis as the coordinate-BC example, but replaces manual lambda
predicates with SurfaceSelector
objects that automatically identify BC regions by inspecting the marching-cubes
surface mesh at solve time.

BC strategy
-----------
Dirichlet (clamped):
    ``AlignedNormalSelector((0, -1, 0), max_angle_deg=45, oriented=True)``
    Strictly downward-facing faces only; catches feet/base regardless of height.

Neumann (applied traction):
    ``FloodFillSelector(seed=AlignedNormalSelector((0,1,0), 30, oriented=True), 35)``
    Seeds at strictly upward-facing faces (within 30° of +y), then flood-fills
    via pairwise dihedral checking.  The concave valley at the cone base stops
    the fill, isolating the tallest spine from the rest of the surface.

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
from core.bc_surface_selectors import (
    AlignedNormalSelector,
    LowestSelector,
    HighestSelector,
    FloodFillSelector,
)


def main():
    parser = argparse.ArgumentParser(
        description="Image → single forward FEM solve (automatic BC identification)"
    )
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

    # Match LatentShapeOpt's geometry preprocessing before extracting the
    # explicit surface used by this forward-only example.
    pipe.vae = pipe.vae.float().eval()
    for parameter in pipe.vae.parameters():
        parameter.requires_grad_(False)
    latents = latents.to(device=device, dtype=torch.float32)

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
    # 3. FEM parameters with automatic BC identification
    #
    # Physical scale: y-axis height = 3.0; x and z are proportional.
    # The object spans approximately y ∈ [0, 3].
    #
    # BCs (auto-detected from MC surface):
    #   Dirichlet: downward-facing faces in the bottom 5% of height (feet/base)
    #   Neumann:   the single topmost cone/spine, flood-filled from the tip
    # ------------------------------------------------------------------
    # Dirichlet (clamped): downward-facing faces restricted to the bottom 10% of height.
    # The height constraint prevents picking up belly/underside faces higher up.
    dirichlet_sel = AlignedNormalSelector(
        (0.0, -1.0, 0.0), max_angle_deg=45.0, oriented=True
    ) & LowestSelector("y", fraction=0.05)

    # Neumann (traction): flood-fill seeded at upward-facing faces in the top 10%.
    # The height constraint restricts the seed to the cone tips only, avoiding
    # any upward-facing faces elsewhere on the body.
    neumann_sel = FloodFillSelector(
        seed=(
            AlignedNormalSelector((0.0, 1.0, 0.0), max_angle_deg=30.0, oriented=True)
            & HighestSelector("y", fraction=0.01)
        ),
        max_feature_angle_deg=15.0,
    )

    params = {
        # ---- project ----
        "project_name": "img2fem_autoid",
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
        # See the coordinate-BC example: use the established f64 pyamgx path
        # for the large depth-7 explicit-surface system.
        "cg_backend_assembled": "pyamgx",
        # ---- material ----
        "E": 10.0,
        "nu": 0.3,
        "B": (0.0, 0.0, 0.0),
        # ---- BCs (SurfaceSelector keys, auto-resolved at solve time) ----
        "dirichlet_bcs_dict": {dirichlet_sel: 0.0},
        "neumann_bcs_dict": {neumann_sel: (0.0, -0.005, 0.0)},
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

    print("Running forward FEM solve with auto-identified BCs ...")
    problem.step()
    problem.save(time=0)
    problem.finalize()
    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()
