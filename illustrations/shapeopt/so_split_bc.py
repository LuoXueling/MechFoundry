"""Split the BC-restored surface into three meshes for panel (g), coloured by BC
TYPE using stitch_bc_patches' OWN per-face labels (so_bc_masks.npz, written by
so_prep from the production BC-zip restoration):

  label 1 = restored BC patch  -> the surfaces actually held FIXED in the run
  label 0 = free base,  label 2 = zip bridge  -> free body

The BC patches are the frozen SDF region (feet + top load); split by z into:
  so_bc_dir.stl  — Dirichlet feet   (teal,   patch faces with z<0)
  so_bc_neu.stl  — Neumann top load (orange, patch faces with z>0)
  so_body.stl    — everything else  (off-white)

No proximity guess and no re-selection: these are the exact faces the pipeline
restores as BC, so the caps are clean and consistent with the computation.
Run in conda (trimesh).
"""
import os
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
_p = lambda n: os.path.join(HERE, n)

d = np.load(_p("so_bc_masks.npz"))
V = d["verts"].astype(np.float64)
F = d["faces"].astype(np.int64)
labels = d["labels"]
cz = V[F].mean(1)[:, 2]                             # face-centroid z

is_bc = labels == 1                                 # the restored (fixed) BC faces
is_dir = is_bc & (cz < 0.0)                         # bottom feet -> Dirichlet
is_neu = is_bc & (cz > 0.0)                         # top load   -> Neumann


def dump(name, mask):
    f = F[mask]
    if len(f) == 0:
        trimesh.Trimesh(V[:3], np.array([[0, 1, 2]]), process=False).export(_p(name))
        return
    used = np.unique(f)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    trimesh.Trimesh(V[used], remap[f], process=False).export(_p(name))


dump("so_body.stl", ~is_dir & ~is_neu)
dump("so_bc_dir.stl", is_dir)
dump("so_bc_neu.stl", is_neu)
print(f"split (bc-zip patch labels): body {int((~is_dir & ~is_neu).sum())}  "
      f"dirichlet {int(is_dir.sum())}  neumann {int(is_neu.sum())}  "
      f"(label1 total {int(is_bc.sum())})")

# --------------------------------------------------------------------------
# Evolution strip (panel f): cut each so_evo_<it>.stl in half on y=0 (through two
# legs) so the frame shows the PROFILE, like void_nucleation's strip.  The kept
# far half is capped into a solid (so the cross-section reads as material); the
# near half is exported as a translucent GHOST.  Body material is unchanged — the
# renderer draws the kept half in the same off-white as the (e) surface.
# --------------------------------------------------------------------------
import glob
import re

CUT_AXIS = 1                                        # y = 0, the plane through two feet
GHOST_PAD = 0.05
MIN_BODY_VOL = 1e-4


def as_volume(mesh, tag):
    """Drop open/tiny MC debris before passing a sampled design to manifold3d."""
    if mesh.is_volume:
        return mesh
    parts = mesh.split(only_watertight=False)
    good = [p for p in parts if p.is_watertight and abs(p.volume) >= MIN_BODY_VOL]
    if not good:
        raise ValueError(f"{tag}: no watertight design component survived cleanup")
    keep = trimesh.util.concatenate(good) if len(good) > 1 else good[0]
    if not keep.is_volume:
        keep.merge_vertices()
        keep.fix_normals()
    if not keep.is_volume:
        raise ValueError(f"{tag}: cleaned design is still not a volume")
    return keep


def half_space(mesh, keep_positive):
    """`mesh` intersected with {y >= 0} (keep_positive) or {y <= 0}, capped."""
    lo, hi = mesh.bounds
    ext = (hi - lo) + 2.0 * GHOST_PAD
    span = (hi[CUT_AXIS] + GHOST_PAD) if keep_positive else (GHOST_PAD - lo[CUT_AXIS])
    ext[CUT_AXIS] = span
    ctr = 0.5 * (lo + hi)
    ctr[CUT_AXIS] = 0.5 * span if keep_positive else -0.5 * span
    box = trimesh.creation.box(extents=ext.tolist())
    box.apply_translation(ctr.tolist())
    return trimesh.boolean.intersection([mesh, box])


def split_cap(half):
    """Separate the flat y=0 cut face (normal -y) from the rest of the shell."""
    fc, fn = half.triangles_center, half.face_normals
    is_cap = (fn[:, CUT_AXIS] < -0.99) & (np.abs(fc[:, CUT_AXIS]) < 1e-6)
    if not is_cap.any():
        return None, half
    return half.submesh([is_cap], append=True), half.submesh([~is_cap], append=True)


for _f in sorted(glob.glob(_p("so_evo_*.stl"))):
    if not re.search(r"so_evo_\d+\.stl$", _f):      # skip already-derived _shell/_face/_ghost
        continue
    _it = re.search(r"so_evo_(\d+)\.stl$", _f).group(1)
    _surf = as_volume(trimesh.load(_f, process=True), f"evo iter {_it}")
    _keep = half_space(_surf, True)                 # far half -> capped solid
    _ghost = half_space(_surf, False)               # near half -> translucent
    _cap, _shell = split_cap(_keep)
    _shell.export(_p(f"so_evo_{_it}_shell.stl"))
    (_cap if _cap is not None else _shell).export(_p(f"so_evo_{_it}_face.stl"))
    _ghost.export(_p(f"so_evo_{_it}_ghost.stl"))
    print(f"evo iter {_it}: shell {len(_shell.faces)} + cap "
          f"{0 if _cap is None else len(_cap.faces)} + ghost {len(_ghost.faces)} faces")
