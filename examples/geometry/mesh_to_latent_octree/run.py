import argparse
import torch
import sys
import os
import numpy as np
import trimesh
from pathlib import Path

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "dependency" / "TripoSG"))

from huggingface_hub import snapshot_download
from src.dependency.TripoSG.scripts.briarmbg import BriaRMBG
from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper

from src.triposg_wrapper.utils import sdf2octree, octree2vtu, sdf2surfacemesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default=str(LEAF_DIR / "output.glb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    parser.add_argument("--init-octree-depth", type=int, default=4)
    parser.add_argument("--octree-depth", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # download pretrained weights
    triposg_weights_dir = str(REPO_ROOT / "src/dependency/TripoSG/pretrained_weights/TripoSG")
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)

    # init tripoSG pipeline
    pipe: TripoSGPipelineWrapper = TripoSGPipelineWrapper.from_pretrained(
        triposg_weights_dir
    ).to(device, dtype)

    mesh = trimesh.creation.box(extents=[1.5, 1, 1])
    # Uniform area-weighted surface sampling for encoding: the TripoSG encoder needs
    # a well-distributed point cloud.  Subdivided box vertices cluster at the corners
    # and edges and starve the faces, which fragments the decoded shape.
    points, face_idx = trimesh.sample.sample_surface(mesh, 100000, seed=42)
    vertices = points
    vertex_normals = mesh.face_normals[face_idx]

    print(f"Sampled surface points: {len(vertices)}")
    print(f"Sampled surface normals: {vertex_normals.shape}")

    surface = torch.FloatTensor(vertices.copy())
    normal = torch.FloatTensor(vertex_normals.copy())
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).to(device, dtype=dtype)

    latents = pipe.vae.encode(surface).latent_dist.sample()

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
