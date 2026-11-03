"""Section every design-evolution FRAME (conda `mechfoundry-warp117`, CPU trimesh).

The frame series (HP_FRAMES=1 hp_prep) writes one hp_fr_<it>.stl per optimiser
iteration.  This cuts each on y=0 exactly like the shipped evolution strip
(hp_cut.py's evo branch, verbatim, cavity handling and all): drop MC debris, cap
the far half into a solid, keep an OPEN near-half ghost with its cut face and its
copy of any cavity removed, and split out the flat cut face.  Same functions as
hp_cut so the frame bodies match panel (f).

Writes hp_fr_<it>_{shell,face,ghost}.stl.  No-op when there are no hp_fr_*.stl.
"""
import glob
import os
import re

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
_p = lambda n: os.path.join(HERE, n)                              # noqa: E731

CUT_AXIS = 1                      # y = 0, surface section through opposite feet
GHOST_PAD = 0.05
MIN_BODY_VOL = 1e-4               # below this a component is MC debris


def as_volume(mesh, tag):
    if mesh.is_volume:
        return mesh
    parts = mesh.split(only_watertight=False)
    good = [p for p in parts if p.is_watertight and abs(p.volume) >= MIN_BODY_VOL]
    assert good, f"{tag}: the surface has no closed component to cut"
    keep = trimesh.util.concatenate(good) if len(good) > 1 else good[0]
    if not keep.is_volume:
        keep.merge_vertices()
        keep.fix_normals()
    assert keep.is_volume, f"{tag}: still not a volume after dropping debris"
    return keep


def half_space(mesh, keep_positive):
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
    fc, fn = half.triangles_center, half.face_normals
    is_cap = (sign * fn[:, CUT_AXIS] > 0.99) & (np.abs(fc[:, CUT_AXIS]) < 1e-6)
    assert is_cap.any(), "no cut face — the half-space boolean did not cap"
    return half.submesh([is_cap], append=True), half.submesh([~is_cap], append=True)


def cavity_faces(mesh):
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
        inside = outer.contains(mesh.submesh([c], append=True).triangles_center)
        if inside.mean() > 0.9:
            mask[c] = True
    return mask


def open_ghost(ghost):
    _, shell = split_cap(ghost, sign=+1.0)      # its cap faces +y, away from us
    return shell


n = skipped = 0
for f in sorted(glob.glob(_p("hp_fr_*.stl")),
                key=lambda q: int(re.search(r"hp_fr_(\d+)", q).group(1))):
    if not re.search(r"hp_fr_\d+\.stl$", f):          # skip derived _shell/_face/_ghost
        continue
    it = re.search(r"hp_fr_(\d+)\.stl$", f).group(1)
    try:
        surf = as_volume(trimesh.load(f, process=True), f"frame {it}")
        k = half_space(surf, True)
        g = half_space(surf, False)
        c, s = split_cap(k)
        _t = cKDTree(surf.triangles_center)
        _vm = cavity_faces(surf)
        g = open_ghost(g.submesh([~_vm[_t.query(g.triangles_center, k=1)[1]]],
                                 append=True))
        s.export(_p(f"hp_fr_{it}_shell.stl"))
        c.export(_p(f"hp_fr_{it}_face.stl"))
        g.export(_p(f"hp_fr_{it}_ghost.stl"))
        n += 1
        print(f"frame iter {it}: shell {len(s.faces)} + cap {len(c.faces)} + "
              f"ghost {len(g.faces)} faces")
    except Exception as exc:                            # noqa: BLE001
        skipped += 1                                    # a dropped frame just leaves
        print(f"frame iter {it}: SKIPPED ({exc})")      # a gap in the movie
print(f"sectioned {n} hp_fr_*.stl frame(s), skipped {skipped}")
