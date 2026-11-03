"""Prep helper (conda GPU): fill the 4 small bolt through-holes of GE bracket
#493 so the fixed-BC footprint is a clean solid washer disk.

The exact zip BC-restoration (Fuchs-Kedem-Uselton) needs each BC patch to be a
simple manifold-with-boundary.  The four bolt holes are tiny (~5 mm) relative to
the reinit grid, so their marching-cubes rims are degenerate (bowtie vertices),
which makes the zip patch boundary non-simple.  We union a short solid cylinder
into each bolt hole (keeping the real bracket, the real interface locations, and
the real load-lug hole), then re-extract a watertight mesh.  Physically this is
the clamped washer pad under each bolt head.

Reads  bracket_493.obj  (mm, from SimJEB, doi:10.7910/DVN/XFUWJG, CC0/ODC-BY)
Writes bracket_493_filled.obj  (mm)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("/home/xl2254/implicit_optimize/src/warp_wrapper"); sys.path.insert(0, ".")
import numpy as np, torch, trimesh
from skimage import measure
from core.analytical_sdf import build_warp_mesh, compute_nodal_sdf

BOLTS_MM = np.array([[51.9, 1.6, 4.3], [0.0, -0.3, 4.3],
                     [-0.2, -148.1, 4.2], [38.1, -147.0, 4.2]])
R_FILL = 7.0          # mm; > hole radius (~5 mm) to plug it
Z_LO, Z_HI = -3.0, 16.0   # mm; span the base flange (bracket base at z=0)
PITCH = 0.8           # mm grid

m = trimesh.load(os.path.join(HERE, "bracket_493.obj"), process=False)
V = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
Fc = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device="cuda")
probe = torch.tensor(np.asarray(m.center_mass), dtype=torch.float32, device="cuda")
diag = float(np.linalg.norm(m.extents))
wp_mesh, negate, _ka = build_warp_mesh(V, Fc, probe, diag, device="cuda")

lo = m.bounds[0] - 4.0; hi = m.bounds[1] + 4.0
gx = [np.arange(lo[i], hi[i] + PITCH, PITCH, dtype=np.float32) for i in range(3)]
GX, GY, GZ = np.meshgrid(*gx, indexing="ij")
pts = torch.tensor(np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1),
                   dtype=torch.float32, device="cuda")
sdf = compute_nodal_sdf(wp_mesh, pts, negate, max_dist=2.0 * diag, device="cuda")
sdf = sdf.cpu().numpy().reshape(GX.shape)
print("mesh sdf range", round(float(sdf.min()), 2), round(float(sdf.max()), 2),
      "frac<0", round(float((sdf < 0).mean()), 3))

# union solid cylinders (negative inside) into each bolt hole
P = np.stack([GX, GY, GZ], -1)
for b in BOLTS_MM:
    rad = np.sqrt((P[..., 0] - b[0]) ** 2 + (P[..., 1] - b[1]) ** 2)
    dz = np.maximum(Z_LO - P[..., 2], P[..., 2] - Z_HI)      # <0 within z-span
    cyl = np.maximum(rad - R_FILL, dz)                        # solid cylinder SDF
    sdf = np.minimum(sdf, cyl)
print("filled sdf frac<0", round(float((sdf < 0).mean()), 3))

verts, faces, _n, _v = measure.marching_cubes(sdf, level=0.0, spacing=(PITCH,) * 3)
verts = verts + lo
out = trimesh.Trimesh(verts, faces, process=True)
trimesh.repair.fix_normals(out)
comps = out.split(only_watertight=False)
if len(comps) > 1:
    out = max(comps, key=lambda c: len(c.faces))
print("filled mesh: V%d F%d watertight=%s genus~euler=%d comps=%d"
      % (len(out.vertices), len(out.faces), out.is_watertight, out.euler_number, len(comps)))
outpath = os.path.join(HERE, "bracket_493_filled.obj")
out.export(outpath)
print("wrote", outpath)
