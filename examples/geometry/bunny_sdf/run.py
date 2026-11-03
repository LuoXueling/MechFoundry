"""Prep step (conda `mechfoundry-warp117`): write the SDF volume + bunny surface for ParaView.

ParaView cannot be imported into the conda env (segfaults), so we compute the
SDF here and hand ParaView plain files:
  * bunny_sdf.vtk  — legacy BINARY STRUCTURED_POINTS, scalar 'sdf' over a cube
  * bunny.ply      — watertight bunny surface (same centered/scaled frame)
  * bunny_sdf.meta — H (cube half-side) and Vin (inside |d| max) for the render
"""

import struct
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
from scipy.interpolate import RegularGridInterpolator
import pymeshlab

LEAF_DIR = Path(__file__).resolve().parent
os.chdir(LEAF_DIR)
BUNNY = LEAF_DIR / "assets/bunny10k_textured.obj"


def main():
    # --- load + close holes (watertight) + remap up=z + center + unit scale ------
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.fspath(BUNNY))
    ms.meshing_remove_duplicate_vertices()
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass
    ms.meshing_close_holes(maxholesize=100000, selfintersection=False)
    pm = ms.current_mesh()
    m = trimesh.Trimesh(vertices=pm.vertex_matrix(), faces=pm.face_matrix(),
                        process=False)
    trimesh.repair.fix_normals(m)

    v = m.vertices.copy()[:, [0, 2, 1]]            # orig (x,z,y) -> up = z
    v -= 0.5 * (v.min(0) + v.max(0))
    v *= 1.0 / np.max(v.max(0) - v.min(0))
    m2 = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
    # publication-quality surface: Loop-subdivide for density, Taubin-smooth to kill
    # faceting without the shrinkage plain Laplacian causes.
    try:
        m2 = m2.subdivide_loop(iterations=2)
    except Exception:
        m2 = m2.subdivide().subdivide()
    trimesh.smoothing.filter_taubin(m2, lamb=0.5, nu=-0.53, iterations=12)
    m2.export("bunny.stl")                            # STL: ParaView reads faces robustly
    print(f"wrote bunny.stl ({len(m2.faces)} faces)")

    # --- robust SDF: voxel-fill the bunny, PAD into a cube grid, distance-transform
    #     the padded cube directly (no extrapolation artifacts) --------------------
    lo, hi = v.min(0), v.max(0)
    pitch = float(np.max(hi - lo)) / 150.0
    vox = m2.voxelized(pitch).fill()
    solid = vox.matrix                              # bunny-bbox occupancy
    T = vox.transform
    org0 = np.array([T[i, 3] for i in range(3)])    # world coord of solid[0,0,0]

    H = 0.5 * 1.16 * float(np.max(hi - lo))         # cube half-side
    NC = int(np.ceil(2 * H / pitch)) + 1
    cube = np.zeros((NC, NC, NC), dtype=bool)
    off = np.round((org0 - (-H)) / pitch).astype(int)   # where the bunny box sits
    s = solid.shape
    cube[off[0]:off[0] + s[0], off[1]:off[1] + s[1], off[2]:off[2] + s[2]] = solid

    sdf_cube = ((edt(~cube) - edt(cube)) * pitch).astype(np.float32)
    Vin = float(-sdf_cube.min())
    sp = pitch
    print(f"cube {NC}^3  d range [{sdf_cube.min():.3f}, {sdf_cube.max():.3f}]  "
          f"H={H:.3f}  solid frac {cube.mean():.3f}")

    # --- write legacy BINARY VTK STRUCTURED_POINTS (x fastest -> order='F') -------
    with open("bunny_sdf.vtk", "wb") as f:
        hdr = ("# vtk DataFile Version 3.0\n"
               "bunny signed distance field\n"
               "BINARY\n"
               "DATASET STRUCTURED_POINTS\n"
               f"DIMENSIONS {NC} {NC} {NC}\n"
               f"ORIGIN {-H} {-H} {-H}\n"
               f"SPACING {sp} {sp} {sp}\n"
               f"POINT_DATA {NC**3}\n"
               "SCALARS sdf float 1\n"
               "LOOKUP_TABLE default\n")
        f.write(hdr.encode("ascii"))
        f.write(sdf_cube.ravel(order="F").astype(">f4").tobytes())
    print("wrote bunny_sdf.vtk")

    with open("bunny_sdf.meta", "w") as f:
        f.write(f"{H}\n{Vin}\n")
    print(f"wrote bunny_sdf.meta  (H={H:.4f}, Vin={Vin:.4f})")


if __name__ == "__main__":
    main()
