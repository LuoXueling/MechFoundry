import argparse
import torch
import sys
import os
import numpy as np
from pathlib import Path

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "dependency" / "TripoSG"))

from huggingface_hub import snapshot_download
from src.dependency.TripoSG.scripts.briarmbg import BriaRMBG
from src.dependency.TripoSG.scripts.image_process import prepare_image
from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper

from src.triposg_wrapper.utils import sdf2octree, octree2vtu, sdf2surfacemesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-input",
        type=str,
        default=str(REPO_ROOT / "src/dependency/TripoSG/assets/example_data/hjswed.png"),
    )
    parser.add_argument("--output-path", type=str, default=str(LEAF_DIR / "output.glb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    parser.add_argument("--init-octree-depth", type=int, default=4)
    parser.add_argument("--octree-depth", type=int, default=5)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # download pretrained weights
    triposg_weights_dir = str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/TripoSG")
    rmbg_weights_dir = str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/RMBG-1.4")
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()

    # init tripoSG pipeline
    pipe: TripoSGPipelineWrapper = TripoSGPipelineWrapper.from_pretrained(
        triposg_weights_dir
    ).to(device, dtype)

    img_pil = prepare_image(
        args.image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net
    )

    latents = pipe.img2latent(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(args.seed),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    sdf, anchors = pipe.latent2sdf(
        latents=latents,
        bounds=(-1.00, -1.00, -1.00, 1.00, 1.00, 1.00),
        init_octree_depth=args.init_octree_depth,
        octree_depth=args.octree_depth,
    )

    nodes, nodal_sdf, elements, edge_hanging, face_hanging = sdf2octree(
        cell_tensor=anchors,
        sdf_tensor=sdf,
        domain_length=1.00 * 2,
    )

    print(f"Number of edge hanging nodes: {edge_hanging.shape[0]}")
    print(edge_hanging[:10, :])

    print(f"Number of face hanging nodes: {face_hanging.shape[0]}")
    print(face_hanging[:10, :])


    octree2vtu(nodes, nodal_sdf, elements, output_dir=str(LEAF_DIR / "output"))

    mesh = sdf2surfacemesh(
        sdf=sdf,
        n_faces=args.faces,
        octree_depth=args.octree_depth,
        bounds=(-1.00, -1.00, -1.00, 1.00, 1.00, 1.00),
    )

    mesh.export(args.output_path)

    print(f"Mesh saved to {args.output_path}")


if __name__ == "__main__":
    main()
