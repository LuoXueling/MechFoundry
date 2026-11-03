"""Prep step 1 (conda `mechfoundry-warp117`, GPU): production geometry for the BC-region figure.

Runs the REAL pipeline — no stand-ins:

    bunny.obj  --(pymeshlab close-holes, y-up, normalize)-->  watertight trimesh
               --pipe.mesh2latent-->                           TripoSG latents
               --pipe.latent2octree-->                         grid SDF + octree hex mesh
               --core.analytical_sdf.extract_surface-->        marching-cubes surface

``latent2octree`` is exactly what ``LatentShapeOpt`` calls, and ``extract_surface``
on the resulting ``grid_sdf`` is exactly what ``NitscheNeoHookean.
_get_or_build_mc_surface_physical`` calls to feed the ``SurfaceSelector``s.  With
``bounds=(-1,-1,-1,1,1,1)`` and ``domain_length=2`` the normalized and physical
frames coincide, so the MC vertices need no further mapping.

Writes ``bc_bunny_geom.npz`` (MC surface, octree nodes/elements/SDF, min element size).
"""

import os
import sys

import numpy as np
import torch
import trimesh
import pymeshlab

sys.path.append(os.path.abspath("../src/dependency/TripoSG"))
sys.path.append(os.path.abspath("../src/warp_wrapper"))
sys.path.append(os.path.abspath(".."))

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from core.analytical_sdf import extract_surface

BUNNY = ("/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib/python3.10/"
         "site-packages/pymeshlab/tests/sample_meshes/bunny10k_textured.obj")
WEIGHTS = "../src/dependency/TripoSG/pretrained_weights/TripoSG"

BOUNDS = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
DOMAIN_LENGTH = 2.0
INIT_OCTREE_DEPTH = 4
OCTREE_DEPTH = 8
FIT = 1.5          # max extent of the bunny inside the [-1, 1]^3 decoder box

device = "cuda"

# ---------------------------------------------------------------------------
# 1. Watertight, y-up, normalized bunny (the encoder needs a closed surface)
# ---------------------------------------------------------------------------
ms = pymeshlab.MeshSet()
ms.load_new_mesh(BUNNY)
ms.meshing_remove_duplicate_vertices()
try:
    ms.meshing_repair_non_manifold_edges()
except Exception:
    pass
ms.meshing_close_holes(maxholesize=100000, selfintersection=False)
pm = ms.current_mesh()
m = trimesh.Trimesh(vertices=pm.vertex_matrix(), faces=pm.face_matrix(), process=False)
trimesh.repair.fix_normals(m)

v = m.vertices.copy()                      # bunny10k is already y-up
v -= 0.5 * (v.min(0) + v.max(0))
v *= FIT / np.max(v.max(0) - v.min(0))
src = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
print(f"source bunny: {len(src.vertices)}V {len(src.faces)}F  "
      f"watertight={src.is_watertight}  extents={src.extents}")

# ---------------------------------------------------------------------------
# 2. Production encode → latents
# ---------------------------------------------------------------------------
print(f"loading TripoSG from {WEIGHTS}")
pipe = TripoSGPipelineWrapper.from_pretrained(os.path.abspath(WEIGHTS))
pipe = pipe.to(device, torch.float16)
pipe.vae.to(torch.float32)                 # LatentShapeOpt upcasts the VAE too

latents = pipe.mesh2latent(src, n_points=200_000, seed=42, sample_seed=42)
latents = latents.to(torch.float32)
print(f"latents: {tuple(latents.shape)} {latents.dtype}")

# ---------------------------------------------------------------------------
# 3. Production decode → filtered SDF grid + octree hex mesh
# ---------------------------------------------------------------------------
octree = pipe.latent2octree(
    latents=latents,
    bounds=BOUNDS,
    init_octree_depth=INIT_OCTREE_DEPTH,
    octree_depth=OCTREE_DEPTH,
    domain_length=DOMAIN_LENGTH,
)
grid_sdf = octree["grid_sdf"]
nodes = octree["nodes"]
elements = octree["elements"]
node_sdf = octree["sdf"]
print(f"grid_sdf: {tuple(grid_sdf.shape)}   octree: "
      f"{nodes.shape[0]} nodes, {elements.shape[0]} hex elements")

# ---------------------------------------------------------------------------
# 4. The exact MC surface the SurfaceSelectors see
# ---------------------------------------------------------------------------
verts, faces, _probe = extract_surface(
    grid_sdf, domain_length=DOMAIN_LENGTH, device=device
)
print(f"MC surface: {verts.shape[0]}V {faces.shape[0]}F  "
      f"lo={verts.amin(0).cpu().numpy()}  hi={verts.amax(0).cpu().numpy()}")

# min element size — production sizes the BC eps tolerance from it
en = nodes[elements.long()]                                   # (E, 8, 3)
edge = (en.amax(1) - en.amin(1)).amin(1)
min_element_size = float(edge.min())
print(f"min element size: {min_element_size:.5f}  "
      f"(uniform depth-{OCTREE_DEPTH} cell = {DOMAIN_LENGTH / 2**OCTREE_DEPTH:.5f})")

np.savez_compressed(
    "../_shared/bc_bunny_geom.npz",   # cache for bc_bunny_select.py (see build.sh)
    verts=verts.cpu().numpy().astype(np.float32),
    faces=faces.cpu().numpy().astype(np.int32),
    nodes=nodes.cpu().numpy().astype(np.float32),
    elements=elements.cpu().numpy().astype(np.int32),
    node_sdf=node_sdf.cpu().numpy().astype(np.float32),
    min_element_size=np.float32(min_element_size),
    octree_depth=np.int32(OCTREE_DEPTH),
    domain_length=np.float32(DOMAIN_LENGTH),
)
print("wrote bc_bunny_geom.npz")
