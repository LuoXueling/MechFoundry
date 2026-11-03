"""Cut step (conda `mechfoundry-warp117`, CPU): turn the prep's raw surfaces into the
half-sectioned meshes panels (e) and (f) are drawn from.

Split out of `hp_prep.py` on purpose: the optimiser run behind it takes ~15
minutes, and every experiment with the section — where to cut, how to colour the
BC patches, which components are debris — needs only these few seconds of
trimesh.  (`so_split_bc.py` is split off for exactly the same reason, §5f.)

Panel (e), the restored surface, is cut like the strip and coloured like
`shape_opt_loop`'s (e):

  hp_e_body.stl   free base + zip bridge (labels 0, 2)  -> off-white
  hp_e_dir.stl    restored BC patch, z < 0 (the feet)   -> teal
  hp_e_neu.stl    restored BC patch, z > 0 (the load)   -> orange
  hp_e_face.stl   the flat y = 0 cut face               -> darker fill
  hp_e_ghost.stl  the half that was cut away            -> translucent

The BC faces come from `stitch_bc_patches`' own per-face labels (recorded by
`hp_prep.py`), never from a proximity guess against the frozen nodes — §5f
measured that route as a ragged zigzag.  The labels are carried across the
boolean by a nearest-original-face lookup, which is exact everywhere except on
the one ring of triangles the cut itself creates.
"""

import glob
import os
import re

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
_p = lambda n: os.path.join(HERE, n)                          # noqa: E731

CUT_AXIS = 1                      # y = 0, surface section through opposite feet
GHOST_PAD = 0.05
MIN_BODY_VOL = 1e-4               # below this a component is MC debris


def as_volume(mesh, tag):
    """Make `mesh` something manifold3d will accept as a solid.

    The raw marching-cubes surface is NOT reliably a closed volume: late in the
    run it carries a handful of specks, some of them open, and manifold3d then
    refuses the whole boolean with `Not all meshes are volumes!`.  Which
    iterations are affected is data-dependent, so this runs on every frame.

    An enclosed CAVITY is a closed component too, and it is the one thing this
    figure cannot afford to drop, so components are kept on |volume| — a cavity
    surface carries the opposite orientation and therefore a negative signed
    volume.  Only genuinely open or sub-`MIN_BODY_VOL` pieces go.
    """
    if mesh.is_volume:
        return mesh
    parts = mesh.split(only_watertight=False)
    good = [p for p in parts if p.is_watertight and abs(p.volume) >= MIN_BODY_VOL]
    assert good, f"{tag}: the surface has no closed component to cut"
    keep = trimesh.util.concatenate(good) if len(good) > 1 else good[0]
    print(f"  [{tag}] dropped {len(parts) - len(good)} debris component(s), "
          f"kept {len(good)} (volumes "
          f"{[round(float(p.volume), 4) for p in good]})")
    if not keep.is_volume:
        keep.merge_vertices()
        keep.fix_normals()
    assert keep.is_volume, f"{tag}: still not a volume after dropping debris"
    return keep


def half_space(mesh, keep_positive):
    """`mesh` intersected with {y >= 0} or {y <= 0}, capped.

    `slice_plane(cap=True)` cannot do this here — its capping path needs
    shapely, which is not installed; the boolean backend is manifold3d and is.
    A capped half is a closed solid, so a void inside the part becomes a genuine
    hole in the cut face instead of a dent in a shell.
    """
    lo, hi = mesh.bounds
    ext = (hi - lo) + 2.0 * GHOST_PAD
    span = (hi[CUT_AXIS] + GHOST_PAD) if keep_positive else (GHOST_PAD - lo[CUT_AXIS])
    ext[CUT_AXIS] = span
    ctr = 0.5 * (lo + hi)
    ctr[CUT_AXIS] = 0.5 * span if keep_positive else -0.5 * span
    box = trimesh.creation.box(extents=ext.tolist())
    box.apply_translation(ctr.tolist())
    return trimesh.boolean.intersection([mesh, box])


def split_cap(half, sign=-1.0):
    """Separate the flat cut face from the rest of the shell.

    They have to be separate meshes rather than one mesh with a recoloured face
    group: coplanar actors z-fight, and nudging one toward the camera to break
    the tie shows as a bright rim at this scale.
    """
    fc, fn = half.triangles_center, half.face_normals
    is_cap = (sign * fn[:, CUT_AXIS] > 0.99) & (np.abs(fc[:, CUT_AXIS]) < 1e-6)
    assert is_cap.any(), "no cut face — the half-space boolean did not cap"
    return half.submesh([is_cap], append=True), half.submesh([~is_cap], append=True)


def cavity_faces(mesh):
    """Per-face mask of the ENCLOSED CAVITY walls.

    An interior void is a closed component of the marching-cubes surface that is
    not the outer boundary, so this is an exact classification and not a
    proximity test: take the connected components of the face graph, keep the
    largest as the body, and call every other component with more than a
    handful of faces a cavity wall.  Panel (e) colours those faces so the void
    reads as a hole rather than as a smudge on the section, and the ghost drops
    them so the near half's copy of the cavity does not print over it.
    """
    comps = trimesh.graph.connected_components(mesh.face_adjacency,
                                               nodes=np.arange(len(mesh.faces)))
    mask = np.zeros(len(mesh.faces), bool)
    if len(comps) < 2:
        return mask
    big = max(comps, key=len)
    outer = mesh.submesh([big], append=True)
    for c in comps:
        if c is big or len(c) < 24:
            continue
        # Being a separate component is not enough — the MC surface also throws
        # debris specks that sit ON the outer boundary, and painting one of
        # those as a void would be a straight fabrication.  A cavity is a
        # component the body CONTAINS.
        inside = outer.contains(mesh.submesh([c], append=True).triangles_center)
        if inside.mean() > 0.9:
            mask[c] = True
    return mask


def open_ghost(ghost):
    """The near half with its OWN cut face removed.

    Both halves come out of the boolean capped, and the ghost's cap lies exactly
    on the section plane — directly between the camera and the face the panel is
    about.  Even at 13 % it prints as a translucent sheet over the whole
    cross-section, and the cavity, which is the one thing the section exists to
    show, reads as haze instead of a hole.  The ghost only has to describe the
    outline of the half that was removed, so it is drawn as an open shell.
    """
    _, shell = split_cap(ghost, sign=+1.0)      # its cap faces +y, away from us
    return shell


def check_halves(whole, keep, ghost, tag):
    """The two halves must add back up to the whole: the boxes meet EXACTLY on
    y = 0 and GHOST_PAD only pushes their far sides past the geometry, so there
    is no overlap to double count and no gap to lose."""
    s = keep.volume + ghost.volume
    assert abs(s - whole.volume) < 1e-3 * abs(whole.volume), (
        f"{tag}: halves sum to {s:.4f}, whole is {whole.volume:.4f} — "
        "the cut is not clean")


# ---------------------------------------------------------------------------
# Panel (e): the BC-restored surface, half-sectioned and split by BC type
# ---------------------------------------------------------------------------
d = np.load(_p("hp_bc_masks.npz"))
V, F, labels = d["verts"].astype(np.float64), d["faces"].astype(np.int64), d["labels"]
full = trimesh.Trimesh(V, F, process=False)
print(f"(e) stitched surface: {len(F)} faces, labels "
      f"{np.bincount(labels.astype(np.int64), minlength=3)[:3].tolist()}")

# label 1 = restored BC patch (the surfaces the run holds FIXED), split by z
# into Dirichlet feet and the Neumann load; 0 and 2 are free body.  3 is the
# wall of the cavity this iteration carved.
cz = V[F].mean(1)[:, 2]
kind = np.zeros(len(F), np.int64)          # 0 body, 1 dirichlet, 2 neumann, 3 void
kind[(labels == 1) & (cz < 0.0)] = 1
kind[(labels == 1) & (cz > 0.0)] = 2
_void = cavity_faces(full)
kind[_void] = 3
print(f"(e) cavity walls: {int(_void.sum())} faces in "
      f"{len(trimesh.graph.connected_components(full.face_adjacency, nodes=np.arange(len(F))))} "
      "surface component(s)")
assert _void.any(), (
    "the restored surface has no enclosed cavity — panel (e) has nothing to "
    "show; is this really the nucleating iteration?")

solid = as_volume(full, "e")
keep = half_space(solid, True)
ghost = half_space(solid, False)
check_halves(solid, keep, ghost, "e")
cap, shell = split_cap(keep)

# Carry the labels across the boolean.  Everything except the ring of triangles
# the cut itself created is a face of the original mesh (possibly retriangulated
# in place), so its centroid still sits on the original face it came from.
tree = cKDTree(full.triangles_center)


def transfer(mesh):
    dist, idx = tree.query(mesh.triangles_center, k=1)
    return kind[idx], float(dist.max())


skind, dmax = transfer(shell)
print(f"(e) label transfer: {len(skind)} shell faces, max centroid distance "
      f"{dmax:.4f} (h = 0.03); body {int((skind == 0).sum())} dirichlet "
      f"{int((skind == 1).sum())} neumann {int((skind == 2).sum())} void "
      f"{int((skind == 3).sum())}")
assert (skind == 1).any() and (skind == 2).any(), (
    "the section lost one of the BC types — is the cut plane through the feet?")
assert (skind == 3).any(), "the section misses the cavity — wrong cut plane?"

for name, k in (("hp_e_body.stl", 0), ("hp_e_dir.stl", 1), ("hp_e_neu.stl", 2),
                ("hp_e_void.stl", 3)):
    m = shell.submesh([skind == k], append=True)
    m.export(_p(name))
    print(f"wrote {name} ({len(m.faces)} faces)")
cap.export(_p("hp_e_face.stl"))
# The ghost describes the OUTLINE of the half that was removed and nothing
# else: its own cut face is dropped (it would sit between the camera and the
# section) and so is its half of the cavity (which printed as a ragged pale
# band exactly where the void is).
gkind, _ = transfer(ghost)
_g = open_ghost(ghost.submesh([gkind != 3], append=True))
_g.export(_p("hp_e_ghost.stl"))
print(f"wrote hp_e_face.stl ({len(cap.faces)} faces, area {cap.area:.4f}) + "
      f"hp_e_ghost.stl ({len(_g.faces)} of {len(ghost.faces)} faces; cut face "
      "and cavity dropped)")

# ---------------------------------------------------------------------------
# Panel (f): the evolution strip, cut the same way (no BC colouring)
# ---------------------------------------------------------------------------
for f in sorted(glob.glob(_p("hp_evo_*.stl")),
                key=lambda q: int(re.search(r"hp_evo_(\d+)", q).group(1))):
    if not re.search(r"hp_evo_\d+\.stl$", f):          # skip derived _shell/_face/_ghost
        continue
    it = re.search(r"hp_evo_(\d+)\.stl$", f).group(1)
    surf = as_volume(trimesh.load(f, process=True), f"evo {it}")
    k = half_space(surf, True)
    g = half_space(surf, False)
    check_halves(surf, k, g, f"evo {it}")
    c, s = split_cap(k)
    # Same ghost cleanup as (e).  The strip keeps its cavities in body colour
    # though — after the void breaks through to the underside (~iter 57) they
    # stop being separate components, and a colour that appears on some frames
    # and not others would read as a change in the design, not in the topology.
    _t = cKDTree(surf.triangles_center)
    _vm = cavity_faces(surf)
    g = open_ghost(g.submesh([~_vm[_t.query(g.triangles_center, k=1)[1]]],
                             append=True))
    s.export(_p(f"hp_evo_{it}_shell.stl"))
    c.export(_p(f"hp_evo_{it}_face.stl"))
    g.export(_p(f"hp_evo_{it}_ghost.stl"))
    print(f"evo iter {it}: shell {len(s.faces)} + cap {len(c.faces)} + "
          f"ghost {len(g.faces)} faces, half-vol {k.volume:.4f}")

# ---------------------------------------------------------------------------
# The loop figure's "Geometry + BC" station: the WHOLE restored surface (not
# sectioned), split by stitch_bc_patches' own face labels exactly as
# shapeopt/so_split_bc.py does (label 1 = restored BC patch; z < 0 feet = Dirichlet,
# z > 0 top = Neumann; labels 0, 2 = body).
# ---------------------------------------------------------------------------
_d = np.load(_p("hp_bc_masks.npz"))
_V, _F, _lab = _d["verts"].astype(np.float64), _d["faces"].astype(np.int64), _d["labels"]
_cz = _V[_F].mean(1)[:, 2]
_dir = (_lab == 1) & (_cz < 0.0)
_neu = (_lab == 1) & (_cz > 0.0)
for _name, _mask in (("hp_full_body.stl", ~_dir & ~_neu), ("hp_full_dir.stl", _dir),
                     ("hp_full_neu.stl", _neu)):
    _f = _F[_mask]
    _used = np.unique(_f)
    _remap = -np.ones(len(_V), np.int64)
    _remap[_used] = np.arange(len(_used))
    trimesh.Trimesh(_V[_used], _remap[_f], process=False).export(_p(_name))
print(f"wrote hp_full_*.stl: body {int((~_dir & ~_neu).sum())} dirichlet {int(_dir.sum())} "
      f"neumann {int(_neu.sum())}")
