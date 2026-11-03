"""Per-vertex OPENNESS of a sectioned result frame (conda `mechfoundry-warp117`, CPU warp).

    python ../_shared/void_open.py FULL.stl|FULL.npz SHELL.stl[,SHELL2.stl ...] GHOST.stl OUT_PREFIX

shape_opt_merged's b (round 11): the voids have to stand out from the body, and a void is
not a separate mesh component in general -- the hidden-phi run's cavity is enclosed at 43
but has broken through to the underside by ~57, so hp_cut's component test
(`cavity_faces`) finds it on some frames and not on others.  What every void shares is
that it is ENCLOSED: from a point on its wall almost no direction leads out of the body.
So each vertex of the frame's far-half shell and of its near-half ghost gets

    open = the fraction of 256 cosine-weighted hemisphere rays about its normal that
           leave the WHOLE (uncut) design surface without hitting it

-- one deterministic rule for both runs and every frame, nothing picked per frame.  An
enclosed cavity wall scores 0, the hidden run's tunnels a few hundredths, the open outside
mostly 0.6-1 (the crevices under the table's top bottom out at ~0.25-0.45).  The rays go against the uncut surface, so a
void that the y = 0 section happens to slice open is still a void.  render_so.py /
render_hp.py's `m_evo_*` slots colour by `open` (VOID_* in evo_style.py).

Writes OUT_PREFIX_shell_open.vtk and OUT_PREFIX_ghost_open.vtk (legacy polydata, point
array "open") and prints each mesh's openness quantiles.
"""
import sys

import numpy as np
import trimesh
import warp as wp

N_RAYS = 256
wp.config.quiet = True
wp.init()
DEV = "cpu"                      # the GPUs are shared with long FEM batches; ~1 s on CPU


def hemisphere(n=N_RAYS):
    """Fixed cosine-weighted Fibonacci directions about +z: deterministic, stratified."""
    i = np.arange(n) + 0.5
    u = i / n
    r = np.sqrt(u)
    th = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n)
    return np.stack([r * np.cos(th), r * np.sin(th), np.sqrt(1.0 - u)], 1)


@wp.kernel
def _open(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
          dirs: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    n = nrm[i]
    # an orthonormal frame about n
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    t = wp.normalize(wp.cross(n, a))
    b = wp.cross(n, t)
    o = pts[i] + n * 2.0e-3          # off the wall (h = 0.03)
    free = float(0.0)
    for k in range(dirs.shape[0]):
        d = dirs[k]
        w = t * d[0] + b * d[1] + n * d[2]
        q = wp.mesh_query_ray(mesh, o, w, 1.0e6)
        if not q.result:
            free += 1.0
    out[i] = free / float(dirs.shape[0])


def load_full(path):
    if path.endswith(".npz"):                       # hp_bc_masks.npz: the restored surface
        d = np.load(path)
        return trimesh.Trimesh(d["verts"].astype(np.float64), d["faces"].astype(np.int64))
    return trimesh.load(path, process=True)


def openness(full, m):
    mesh = wp.Mesh(points=wp.array(full.vertices.astype(np.float32), dtype=wp.vec3, device=DEV),
                   indices=wp.array(full.faces.astype(np.int32).ravel(), dtype=int, device=DEV))
    out = wp.zeros(len(m.vertices), dtype=float, device=DEV)
    wp.launch(_open, dim=len(m.vertices), device=DEV, inputs=[
        mesh.id, wp.array(m.vertices.astype(np.float32), dtype=wp.vec3, device=DEV),
        wp.array(m.vertex_normals.astype(np.float32), dtype=wp.vec3, device=DEV),
        wp.array(hemisphere().astype(np.float32), dtype=wp.vec3, device=DEV), out])
    return out.numpy()


def write_vtk(path, m, val):
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nopenness\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(m.vertices)} float\n")
        np.savetxt(f, m.vertices, fmt="%.6f")
        f.write(f"POLYGONS {len(m.faces)} {4 * len(m.faces)}\n")
        np.savetxt(f, np.hstack([np.full((len(m.faces), 1), 3), m.faces]), fmt="%d")
        f.write(f"POINT_DATA {len(m.vertices)}\nSCALARS open float 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, val, fmt="%.4f")


if __name__ == "__main__":
    full_p, shells, ghost_p, prefix = sys.argv[1:5]
    full = load_full(full_p)
    shell = trimesh.util.concatenate([trimesh.load(p, process=True) for p in shells.split(",")])
    shell.merge_vertices()
    for tag, m in (("shell", shell), ("ghost", trimesh.load(ghost_p, process=True))):
        v = openness(full, m)
        write_vtk(f"{prefix}_{tag}_open.vtk", m, v)
        q = np.quantile(v, [0.01, 0.05, 0.25, 0.5])
        print(f"{prefix}_{tag}_open.vtk: {len(v)} vertices, open q01/05/25/50 "
              + " ".join(f"{x:.2f}" for x in q)
              + f", < 0.15: {(v < 0.15).mean() * 100:.1f} %")
