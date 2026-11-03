"""Build a real 2-D bunny signed-distance field (a profile slice of the
Stanford bunny) to drive the octree-workflow figure.  This is the actual
geometry the pipeline would see, reduced to a 2-D profile for a paper figure.

Produces, cached in bunny_slice.npz:
  sdf   : (N, N) float32 signed distance on the normalized [-1, 1]^2 slice
  X, Y  : (N, N) node coordinates
  N     : grid resolution (nodes per side)
The field is on a fine, uniform background grid -- exactly the object the
pipeline maintains ("fine-resolution uniform background grid").
"""

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
import pymeshlab

BUNNY = ("/home/xl2254/anaconda3/envs/mechfoundry-warp117/lib/python3.10/"
         "site-packages/pymeshlab/tests/sample_meshes/bunny10k_textured.obj")

# Fine background grid: a power-of-two-plus-one lattice so the quadtree's
# final depth lands exactly on it.  l_final = 7  ->  128 cells, 129 nodes.
L_FINAL = 7
N = 2 ** L_FINAL + 1               # 129 nodes per side


def build():
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(BUNNY)
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

    v = m.vertices.copy()[:, [0, 2, 1]]        # up = z (matches prep_bunny_sdf)
    v -= 0.5 * (v.min(0) + v.max(0))
    v *= 1.0 / np.max(v.max(0) - v.min(0))
    m = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)

    # Voxelize to a 3-D occupancy at the target lattice pitch, then take the
    # mid-width slice: the x-z plane of the bunny is the recognizable side
    # profile (ears + body).
    lo, hi = v.min(0), v.max(0)
    H = 0.5 * 1.18 * float(np.max(hi - lo))    # square half-side
    NC = N
    sp = 2 * H / (NC - 1)
    vox = m.voxelized(sp).fill()
    solid = vox.matrix
    T = vox.transform
    org = np.array([T[i, 3] for i in range(3)])

    # slice through the middle of the width (y axis, index 1); OR a few central
    # slices so thin ears survive
    jy = solid.shape[1] // 2
    sl = solid[:, max(0, jy - 2):jy + 3, :].any(axis=1)   # (nx, nz) occupancy

    # Pad the profile into a normalized square [-1,1]^2 and distance-transform.
    cube = np.zeros((NC, NC), dtype=bool)
    off = np.round((org[[0, 2]] - (-H)) / sp).astype(int)
    s = sl.shape
    x0, y0 = max(off[0], 0), max(off[1], 0)
    cube[x0:min(off[0] + s[0], NC), y0:min(off[1] + s[1], NC)] = \
        sl[x0 - off[0]:NC - off[0], y0 - off[1]:NC - off[1]]

    sdf = ((edt(~cube) - edt(cube)) * sp).astype(np.float32)

    xs = np.linspace(-H, H, NC)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    np.savez("bunny_slice.npz", sdf=sdf, X=X.astype(np.float32),
             Y=Y.astype(np.float32), N=NC, H=H, pitch=sp)
    print(f"wrote bunny_slice.npz  N={NC}  H={H:.3f}  "
          f"d range [{sdf.min():.3f}, {sdf.max():.3f}]  solid frac {cube.mean():.3f}")
    return sdf


if __name__ == "__main__":
    build()
