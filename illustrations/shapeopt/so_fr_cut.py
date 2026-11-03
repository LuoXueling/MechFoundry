"""Section every design-evolution FRAME (conda `mechfoundry-warp117`, CPU trimesh).

The frame series (SO_FRAMES=1 so_prep) writes one so_fr_<it>.stl per optimiser
iteration.  This cuts each on y=0 exactly like the shipped evolution strip
(so_split_bc.py's evo branch, verbatim): the far half is capped into a solid so
the cross-section reads as material, the near half is kept as a translucent
ghost, and the flat cut face is split out to be drawn a step darker.  Same
functions as so_split_bc so the frame bodies match panel (f) pixel-for-pixel.

Writes so_fr_<it>_{shell,face,ghost}.stl next to the inputs.  Idempotent and a
no-op when there are no so_fr_*.stl.
"""
import glob
import os
import re

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
_p = lambda n: os.path.join(HERE, n)                              # noqa: E731

CUT_AXIS = 1                                        # y = 0, the plane through two feet
GHOST_PAD = 0.05
MIN_BODY_VOL = 1e-4                                  # below this a component is MC debris


def as_volume(mesh, tag):
    """Make `mesh` something manifold3d will accept as a solid (== hp_cut).

    so_split_bc only ever cut six hand-picked evo iters, all clean; across ALL
    101 frames the forming table sheds open/tiny MC specks and the boolean then
    refuses the whole mesh with `Not all meshes are volumes!`.  Drop the debris
    (keep closed components over MIN_BODY_VOL) so the frame can be cut; a clean
    frame is already a volume and passes straight through unchanged, so the cut
    stays identical to so_split_bc's on the frames it could handle."""
    if mesh.is_volume:
        return mesh
    parts = mesh.split(only_watertight=False)
    good = [p for p in parts if p.is_watertight and abs(p.volume) >= MIN_BODY_VOL]
    if not good:
        return None
    keep = trimesh.util.concatenate(good) if len(good) > 1 else good[0]
    if not keep.is_volume:
        keep.merge_vertices()
        keep.fix_normals()
    return keep if keep.is_volume else None


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


n = skipped = 0
for _f in sorted(glob.glob(_p("so_fr_*.stl")),
                 key=lambda q: int(re.search(r"so_fr_(\d+)", q).group(1))):
    if not re.search(r"so_fr_\d+\.stl$", _f):       # skip derived _shell/_face/_ghost
        continue
    _it = re.search(r"so_fr_(\d+)\.stl$", _f).group(1)
    try:
        _surf = as_volume(trimesh.load(_f, process=True), f"frame {_it}")
        if _surf is None:
            raise ValueError("no closed component to cut")
        _keep = half_space(_surf, True)             # far half -> capped solid
        _ghost = half_space(_surf, False)           # near half -> translucent
        _cap, _shell = split_cap(_keep)
        _shell.export(_p(f"so_fr_{_it}_shell.stl"))
        (_cap if _cap is not None else _shell).export(_p(f"so_fr_{_it}_face.stl"))
        _ghost.export(_p(f"so_fr_{_it}_ghost.stl"))
        n += 1
        print(f"frame iter {_it}: shell {len(_shell.faces)} + cap "
              f"{0 if _cap is None else len(_cap.faces)} + ghost {len(_ghost.faces)}")
    except Exception as exc:                          # noqa: BLE001
        # One un-cuttable frame must not kill the series; the compose step pairs
        # only iterations that have BOTH panels, so a missing body just drops
        # that one frame from the movie.
        skipped += 1
        print(f"frame iter {_it}: SKIPPED ({exc})")
print(f"sectioned {n} so_fr_*.stl frame(s), skipped {skipped}")
