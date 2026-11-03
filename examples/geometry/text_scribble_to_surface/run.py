"""Generate surface-mesh examples from TripoSG text/scribble latents."""

import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image


LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
TRIPOSG_ROOT = REPO_ROOT / "src" / "dependency" / "TripoSG"
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(TRIPOSG_ROOT))

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from src.triposg_wrapper.utils import sdf2surfacemesh


BOUNDS = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text-only, scribble-only, and combined TripoSG meshes."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=TRIPOSG_ROOT / "pretrained_weights" / "TripoSG-scribble",
    )
    parser.add_argument(
        "--scribble-input",
        type=Path,
        default=TRIPOSG_ROOT
        / "assets"
        / "example_scribble_data"
        / "cat_with_wings.png",
    )
    parser.add_argument("--output-dir", type=Path, default=LEAF_DIR / "output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--init-octree-depth", type=int, default=8)
    parser.add_argument("--octree-depth", type=int, default=8)
    parser.add_argument(
        "--faces",
        type=int,
        default=-1,
        help="Target face count; -1 preserves the original marching-cubes surface.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("text", "scribble", "combined"),
        default=("text", "scribble", "combined"),
        help="Examples to generate (default: all three).",
    )
    return parser.parse_args()


def latent_to_surface(pipe, latents, args, output_path):
    sdf, _ = pipe.latent2sdf(
        latents,
        bounds=BOUNDS,
        init_octree_depth=args.init_octree_depth,
        octree_depth=args.octree_depth,
    )
    mesh = sdf2surfacemesh(
        sdf=sdf,
        n_faces=args.faces,
        octree_depth=args.octree_depth,
        bounds=BOUNDS,
    )
    if mesh.is_empty:
        raise RuntimeError(f"TripoSG produced an empty surface for {output_path.name}")
    mesh.export(output_path)
    print(
        f"Saved {output_path}: {len(mesh.vertices)} vertices, "
        f"{len(mesh.faces)} faces"
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. TripoSG inference and the existing Warp "
            "surface extraction path require a working CUDA device."
        )

    snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    # Passing dtype while loading casts checkpoint parameters, while this explicit
    # conversion also casts the VAE's non-persistent frequency-embedding buffer.
    pipe = TripoSGPipelineWrapper.from_pretrained(args.weights, dtype=dtype).to(
        args.device, dtype
    )
    scribble = Image.open(args.scribble_input).convert("RGB")

    cases = {
        "text": (
            "text_chair",
            lambda generator: pipe.text2latent(
                "a modern cantilever chair",
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            ),
        ),
        "scribble": (
            "scribble_cat",
            lambda generator: pipe.scribble2latent(
                scribble,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            ),
        ),
        "combined": (
            "scribble_prompt_cat",
            lambda generator: pipe.scribble2latent(
                scribble,
                prompt="a cat with wings",
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            ),
        ),
    }

    for case in args.cases:
        name, generate_latents = cases[case]
        print(f"Generating {name} ...")
        # Use the same seed for every input, so differences come from conditioning.
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        latents = generate_latents(generator)
        latent_to_surface(pipe, latents, args, args.output_dir / f"{name}.glb")


if __name__ == "__main__":
    main()
