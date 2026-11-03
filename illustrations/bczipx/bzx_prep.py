"""Prep step (conda `mechfoundry-warp117`, GPU): the three BC-zip branches the main figure
never reaches.

``bc_zip_workflow`` follows one wooden table through the seven steps, and on that
geometry — as on any ordinary one — three sub-steps of the paper's enumeration
are simply not exercised, because nothing asks them to run:

  step 3, second half   fan-cap a nested loop left by the removal
  step 4, all of it     cluster + bridge surplus hole loops, cap the smaller
                        of the last two

(Step 1's other half — refusing a cached patch that itself contains nested loops
— had a panel here once: a plate whose clamp was the annular washer seat around
a bolt hole, which ``_check_no_nested_loops`` rejects at cache time.  It was cut.
Its outcome is a refusal, so it had no "after" to pair with, and it left the
figure with one odd panel and a colour drawn nowhere else.  It belongs in the
text.)

Each gets the smallest geometry that forces it, and nothing else:

  A  a round bar clamped on its end face, through which the optimiser has run a
     coaxial BORE.  The candidate leaves two open loops — the bar's rim and,
     concentric inside it, the bore's mouth — and the inner one is fan-capped,
     plugging the bore where it meets the region the exact patch is about to be
     restored into.  A through bore, not a blind pocket: a pocket's remains are
     sealed inside the solid and the panel after the cap has nothing to show,
     whereas a bore reads as a hole on sight and the cap visibly closes it.
  B  a clamped round trunk the optimiser has SPLIT INTO THREE STRUTS just inside
     the removal box.  The removal leaves three loops where the patch has one,
     so the loops are clustered onto that single patch loop, the closest two are
     bridged, and the smaller of the two survivors is capped.

The bar in A is tilted 18 degrees on purpose.  Nesting is tested by axis-aligned
bounding boxes, and an axis-aligned bar is cut by the (axis-aligned) removal box
in a planar rim whose AABB is degenerate along the axis: the bore's mouth would
then fail the containment test by a fraction of an element, for a reason that has
nothing to do with the geometry being nested.  The tilt makes both rims oblique
ellipses about the same axis, so containment holds by construction.

Both surfaces are UNIFORM CSG MESHES rather than marching cubes (see ``_tube``
for the two reasons).  The BC nodes are still background-grid nodes and the patch
radius is still 2 x the element size, and the algorithm neither knows nor cares
where its input mesh came from.

The ALGORITHM is production code, untouched: ``remeshing.compute_bc_patch`` and
``remeshing.stitch_bc_patches``, wrapped only to record what they were handed.
Writes ``bzx_*.vtk`` + ``bzx_stats.txt``.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.append(os.path.abspath("../src/warp_wrapper"))

from core import remeshing as R                                       # noqa: E402

dev = "cuda"
L = 2.0
stats = {}


def set_grid(n):
    """Rebuild the background node lattice the BC nodes are drawn from.

    This is the FEM background grid, not the surface: both surfaces here are
    built as uniform CSG meshes (see ``_tube``).  Only ``radius = 2h`` -- the
    production rule, 2 x the smallest element size -- still comes from it, so
    the two cases can carry different element sizes.
    """
    global N, h, RADIUS, X, Y, Z, P, PTS
    N = n
    h = L / (N - 1)
    RADIUS = 2.0 * h         # production: 2 x the smallest element size
    lin = torch.linspace(-L / 2, L / 2, N, device=dev)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    P = torch.stack([X, Y, Z], -1)
    PTS = P.reshape(-1, 3)


set_grid(129)


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def T(v):
    return torch.tensor(v, device=dev, dtype=PTS.dtype)


# --- VTK writers (same format as bz_prep, so render_bzx shares its LUT) ------
def write_poly(path, verts, faces, labels, name):
    v = _np(verts).astype(np.float64)
    f = _np(faces).astype(np.int64).reshape(-1, 3)
    lb = _np(labels).astype(np.int64).reshape(-1)
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\n{name}\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n")
        np.savetxt(fh, v, fmt="%.6f")
        fh.write(f"\nPOLYGONS {len(f)} {4 * len(f)}\n")
        np.savetxt(fh, np.hstack([np.full((len(f), 1), 3, np.int64), f]), fmt="%d")
        fh.write(f"\nCELL_DATA {len(f)}\nSCALARS stage float 1\n"
                 "LOOKUP_TABLE default\n")
        np.savetxt(fh, lb, fmt="%d")
    print(f"  wrote {path} ({len(f)} tris)")


def write_lines(path, verts, loops, name):
    """Loops as closed polylines, one cell scalar per loop."""
    v = _np(verts).astype(np.float64)
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\n{name}\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n")
        np.savetxt(fh, v, fmt="%.6f")
        tot = sum(len(l) + 2 for l in loops)
        fh.write(f"\nLINES {len(loops)} {tot}\n")
        for l in loops:
            fh.write(f"{len(l) + 1} " + " ".join(str(int(i)) for i in l)
                     + f" {int(l[0])}\n")
        fh.write(f"\nCELL_DATA {len(loops)}\nSCALARS loop float 1\n"
                 "LOOKUP_TABLE default\n")
        for i in range(len(loops)):
            fh.write(f"{i}\n")
    print(f"  wrote {path} ({len(loops)} loops)")


def taubin(V, F, pin, iters=10, lam=0.5, mu=-0.53):
    """Production's Taubin pass (``stitch_bc_patches``' own lambda/mu block),
    run over a whole display mesh instead of just the bridge triangles.

    ``pin`` is a vertex-index array held fixed -- here, every vertex on a drawn
    boundary loop, so the loops the panels are about do not move by a hair while
    the body around them relaxes.  The alternating +lambda / -mu steps are what
    keep it from shrinking.

    NOTE this is a DISPLAY smoothing: it runs after the workflow, so the surface
    drawn is no longer exactly the one ``stitch_bc_patches`` was handed.  The
    loops, the fan caps and the bridge -- everything the figure actually
    claims -- are untouched, being pinned or built on pinned vertices.
    """
    V = torch.as_tensor(np.asarray(V), dtype=torch.float64, device=dev).clone()
    Fi = torch.as_tensor(np.asarray(F), dtype=torch.int64, device=dev)
    n = V.shape[0]
    e = torch.cat([Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [2, 0]]], 0)
    e, _ = e.sort(dim=1)
    e = torch.unique(e, dim=0)
    rows = torch.cat([e[:, 0], e[:, 1]])
    cols = torch.cat([e[:, 1], e[:, 0]])
    # INVERSE EDGE LENGTH weights (the scale-dependent umbrella), not the
    # uniform ones production uses on the bridge strip.  These meshes are
    # anisotropic quads -- rings 0.022 apart, sections 0.014 -- and a uniform
    # Laplacian drags vertices along the long direction: at 6 iterations the head
    # visibly rippled and at 40 the whole part melted.
    w = 1.0 / (V[rows] - V[cols]).norm(dim=1).clamp(min=1e-12)
    wsum = torch.zeros(n, dtype=V.dtype, device=dev)
    wsum.scatter_add_(0, rows, w)
    W = torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                w / wsum.clamp(min=1e-12)[rows], (n, n)).coalesce()
    free = torch.zeros(n, dtype=torch.bool, device=dev)
    free[torch.unique(Fi.reshape(-1))] = True       # untouched verts would drift
    free[torch.as_tensor(np.asarray(pin, dtype=np.int64), device=dev)] = False
    fi = free.nonzero(as_tuple=True)[0]
    for _ in range(iters):
        for step in (lam, mu):
            V[fi] += step * (torch.sparse.mm(W, V) - V)[fi]
    return _np(V)


def _cent_keys(V, F):
    return [tuple(np.round(p, 9)) for p in V[F].mean(axis=1)]


# ===========================================================================
# Capture wrappers — the real helpers do the work, we only record.
# ===========================================================================
cap = {}
_orig = {n: getattr(R, n) for n in (
    "_cap_inner_hole_loops", "_cluster_and_bridge_hole_loops",
    "_classify_inner_hole_loops", "_zipper_stitch")}


def cap_wrap(verts, hole_loops):
    v, outer, cap_t = _orig["_cap_inner_hole_loops"](verts, hole_loops)
    cap["loops_removed"] = [list(l) for l in hole_loops]
    cap["loops_after_inner_cap"] = [list(l) for l in outer]
    cap["inner_cap_tris"] = _np(cap_t).reshape(-1, 3)
    cap["verts_after_inner_cap"] = _np(v).copy()
    # Which of the removal's loops the production classifier called nested --
    # the figure must mark the same one the algorithm acted on, not one this
    # script re-derives with its own rule.
    cap["is_inner"] = _np(
        _orig["_classify_inner_hole_loops"](verts, hole_loops)).tolist()
    return v, outer, cap_t


def cluster_wrap(verts, hole_loops, n_target, patch_centroids, **kw):
    cap["loops_before_merge"] = [list(l) for l in hole_loops]
    v, merged, bridge_t, cap_t = _orig["_cluster_and_bridge_hole_loops"](
        verts, hole_loops, n_target, patch_centroids, **kw)
    cap["loops_after_merge"] = [list(l) for l in merged]
    cap["merge_tris"] = np.asarray(bridge_t, dtype=np.int64).reshape(-1, 3)
    cap["cluster_cap_tris"] = np.asarray(cap_t, dtype=np.int64).reshape(-1, 3)
    cap["verts_after_merge"] = _np(v).copy()
    return v, merged, bridge_t, cap_t


def zip_wrap(h_loop, p_loop_nat, h_pos_t, p_pos_nat_t):
    tris = _orig["_zipper_stitch"](h_loop, p_loop_nat, h_pos_t, p_pos_nat_t)
    cap.setdefault("zip_tris", []).extend([list(t) for t in tris])
    return tris


R._cap_inner_hole_loops = cap_wrap
R._cluster_and_bridge_hole_loops = cluster_wrap
R._zipper_stitch = zip_wrap


def run_workflow(verts_upd, faces_upd, comps, **kw):
    """Steps 2-7, one production call, with the capture dict reset."""
    cap.clear()
    msgs = []
    base_v, base_f = R._close_open_loops(verts_upd, faces_upd)
    out_v, out_f, labels = R.stitch_bc_patches(
        base_verts_phys=base_v, base_faces=base_f, bc_patches=comps,
        log=msgs.append, **kw)
    for m in msgs:
        print("   ", m.strip())

    # Watertightness — the point of the whole workflow.  Every edge in exactly
    # two triangles (extract_boundary_loops_strict is for OPEN boundaries and
    # reports "branching-out-degree" on a closed mesh, which is not a failure).
    F = _np(out_f).astype(np.int64)
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
                .reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert int((cnt == 1).sum()) == 0 and int((cnt > 2).sum()) == 0, \
        "stitched result is not watertight"
    print(f"    watertight: yes  ({out_f.shape[0]} tris out)")

    # The removed candidate, found by face CENTROID: stitch_bc_patches compacts
    # vertices before returning, so output indices live in a different space
    # from the intermediates captured inside it, but base faces never move.
    Vin, Fin = _np(base_v), _np(base_f).astype(np.int64)
    kept_keys = set(_cent_keys(_np(out_v), F[_np(labels) == 0]))
    removed_mask = np.array([k not in kept_keys for k in _cent_keys(Vin, Fin)])
    base_kept = Fin[~removed_mask]
    assert len(base_kept) == int((_np(labels) == 0).sum()), "kept-face mismatch"
    print(f"    removed candidate: {int(removed_mask.sum())} of {len(Fin)} tris")
    return base_v, base_f, base_kept, out_v, out_f, labels


import trimesh                                                      # noqa: E402


def _tube(r, x0, x1, cy=0.0, cz=0.0, tip=False, sections=20, ring=0.03,
          r_end=None, flare=0.07):
    """A rod along +x with a UNIFORM quad mesh: ``sections`` around, rings every
    ~``ring`` along, a fan cap at ``x0`` and either a fan cap or a hemisphere at
    ``x1``.

    BOTH cases are built from this, not from marching cubes, and for two
    different reasons:

    - the bridge is as wide as whatever loop edge the closest-pair search
      lands on, and an MC loop carries edges anywhere from ~0 to h (see case B);
    - an MC cylinder's silhouette is scalloped at the element scale, and these
      panels are mostly circles, which is exactly where that shows.

    Built by hand rather than with ``trimesh.creation.cylinder``: that primitive
    is a single quad ring along the WHOLE length, and a face is a removal
    candidate if any one of its vertices is in the box -- so an unsplit rod is
    removed end to end (874 of 1174 faces went, and the panel had nothing left
    to show).  ``subdivide_to_size`` is not the fix either: it splits triangles
    independently and leaves the mesh non-watertight.
    """
    th = 2.0 * np.pi * np.arange(sections) / sections
    circ = np.stack([np.cos(th), np.sin(th)], 1)                  # (S, 2)
    xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / ring)) + 1))

    def _r(x):
        """Constant, or a smoothstep FLARE over the last ``flare`` of the run.

        A boolean union of a rod into a trunk leaves a hard intersection circle;
        the optimiser does not make joints like that, and the panel is meant to
        look like a part.  Flaring the rod as it approaches stands in for the
        fillet CSG cannot give: smoothstep leaves the rod parallel to its axis
        where the straight run ends, so there is no kink where the flare starts.
        """
        if r_end is None:
            return r
        t = float(np.clip((x - (x1 - flare)) / flare, 0.0, 1.0))
        return r + (r_end - r) * t * t * (3.0 - 2.0 * t)

    V = [np.stack([np.full(sections, x), cy + _r(x) * circ[:, 0],
                   cz + _r(x) * circ[:, 1]], 1) for x in xs]
    r = _r(x1)   # the tip, if any, sits on whatever radius the flare ended at
    if tip:      # hemisphere: rings of a quarter turn, ending at the pole
        # ...open at both ends: phi=0 repeats the last barrel ring and phi=pi/2
        # collapses the whole ring onto the pole, which the fan below closes.
        phi = np.linspace(0, np.pi / 2,
                          max(3, int(round(0.5 * np.pi * r / ring)) + 1))[1:-1]
        V += [np.stack([np.full(sections, x1 + r * np.sin(p)),
                        cy + r * np.cos(p) * circ[:, 0],
                        cz + r * np.cos(p) * circ[:, 1]], 1) for p in phi]
    V = np.vstack(V)
    n_rings = len(V) // sections
    F = []
    for k in range(n_rings - 1):                                  # barrel quads
        a, b = k * sections, (k + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            F += [[a + i, b + i, b + j], [a + i, b + j, a + j]]
    def _flat_cap(base, x, r_out, flip):
        """Close a ring with CONCENTRIC RINGS, not one fan.

        A fan puts a vertex of valence ``sections`` at the centre, and an
        umbrella-weighted smoother pulls a flat face with such a pole into a
        warped cone -- visible as radiating ridges across the head's front.
        Rings keep every interior vertex at valence 6.
        """
        nonlocal V, F
        n_r = max(1, int(round(r_out / ring)))
        prev = base
        for k in range(1, n_r):
            rr = r_out * (1.0 - k / n_r)
            new = len(V)
            V = np.vstack([V, np.stack([np.full(sections, x), cy + rr * circ[:, 0],
                                        cz + rr * circ[:, 1]], 1)])
            for i in range(sections):
                j = (i + 1) % sections
                q = [[prev + i, new + i, new + j], [prev + i, new + j, prev + j]]
                F += [t[::-1] for t in q] if flip else q
            prev = new
        c = len(V)
        V = np.vstack([V, [[x, cy, cz]]])
        for i in range(sections):
            t = [c, (i + 1) % sections + prev, i + prev]
            F.append(t if flip else t[::-1])

    last = (n_rings - 1) * sections
    if tip:
        c1 = len(V)                                               # hemisphere pole
        V = np.vstack([V, [[x1 + r, cy, cz]]])
        F += [[c1, last + i, last + (i + 1) % sections] for i in range(sections)]
    else:
        _flat_cap(last, x1, r, flip=False)
    _flat_cap(0, x0, _r(x0), flip=True)
    m = trimesh.Trimesh(vertices=V, faces=np.asarray(F), process=True)
    m.fix_normals()          # manifold3d refuses anything not consistently wound
    assert m.is_watertight and m.is_volume, "hand-built rod is not a volume"
    return m


def _to_torch(m):
    assert m.is_watertight, "fabricated surface is not watertight"
    return (torch.as_tensor(m.vertices, dtype=torch.float64, device=dev),
            torch.as_tensor(m.faces, dtype=torch.int32, device=dev))



# ===========================================================================
# A — the removal leaves a nested hole loop: fan-capped at step 3
# ===========================================================================
print("\n=== A  a bore drilled through the clamped end face")
TILT = np.deg2rad(18.0)
U = [float(np.cos(TILT)), 0.0, float(np.sin(TILT))]
R_BAR, S_BAR = 0.20, 0.19
# A coaxial BORE RIGHT THROUGH a SHORT bar, not the shallow spherical pocket
# this case started with.  Two reasons, both about what the panels can show:
# a bowl one element deep reads as a dimple, and whatever it orphans is sealed
# inside the solid where the panel after the cap cannot show it.  And a bar
# short enough to see daylight through -- 2*S_BAR*tan(view angle) under the bore
# diameter -- makes (b) unmistakably a hole and (c) unmistakably a plug, which
# no amount of shading does: ParaView casts no shadows here, so a bore lit like
# everything else reads as a rod standing in a tube.
# Nesting does not care either way -- it is a test on the two loops' AABBs, not
# on what the removal happens to disconnect.
BORE_R = 0.085
SECT_A, RING_A = 96, 0.0035
# 96 sections is round; the RINGS are what make the loops smooth.  The candidate
# boundary follows mesh edges, and the box's cut plane is oblique to a tilted
# bar, so the loop zigzags between adjacent rings by exactly the ring spacing.
# At 0.012 that is 0.6 mm on the page against a 0.34 mm loop tube -- visibly
# serrated.  0.005 puts it under the tube and the loop reads as a curve.
_tilt = trimesh.transformations.rotation_matrix(-TILT, [0, 1, 0])
_bar = _tube(R_BAR, -S_BAR, S_BAR, sections=SECT_A, ring=RING_A)
_bore = _tube(BORE_R, -S_BAR - 0.05, S_BAR + 0.05,   # past both ends to cut clean
              sections=SECT_A, ring=RING_A)
_init_a = _bar.copy()
_upd_a = trimesh.boolean.difference([_bar, _bore])
for _m in (_init_a, _upd_a):
    _m.apply_transform(_tilt)
va0, fa0 = _to_torch(_init_a)
va1, fa1 = _to_torch(_upd_a)
_u = T(U) / float(np.linalg.norm(U))
s_axis = PTS @ _u                              # axial coordinate, tilted frame
_rad = (PTS - s_axis.unsqueeze(1) * _u).norm(dim=1)
frozen_a = PTS[(_rad < R_BAR) & (s_axis > -S_BAR)
               & (s_axis < -S_BAR + 0.05)].double()
print(f"    init {len(va0)}V {len(fa0)}F, updated {len(va1)}V {len(fa1)}F, "
      f"frozen {len(frozen_a)}")
patch_a = R.compute_bc_patch(va0, fa0, frozen_a, RADIUS)
comps_a = patch_a["components"]
assert len(comps_a) == 1 and len(comps_a[0]["boundary_loops_local"]) == 1, \
    "case A wants one single-loop patch"
bv_a, bf_a, kept_a, ov_a, of_a, lb_a = run_workflow(va1, fa1, comps_a)
assert len(cap.get("inner_cap_tris", [])) > 0, \
    "no inner loop was capped — geometry does not exercise step 3's cap branch"
n_inner_a = int(np.sum(cap["is_inner"]))
print(f"    step 3: {len(cap['loops_removed'])} hole loops, {n_inner_a} nested, "
      f"{len(cap['inner_cap_tris'])} cap tris -> "
      f"{len(cap['loops_after_inner_cap'])} loop for the zip")
stats.update(a_loops_removed=len(cap["loops_removed"]), a_inner=n_inner_a,
             a_cap_tris=len(cap["inner_cap_tris"]),
             a_loops_after=len(cap["loops_after_inner_cap"]))

_Va0 = cap["verts_after_inner_cap"]                   # pre-compaction indices
# 120 sweeps, twelve times case B's.  The two need wildly different amounts and
# it is not a matter of taste: case B has union SEAMS, which are high-frequency
# and gone in 10, while this case is a cylinder minus a cylinder -- developable
# everywhere, nothing for a smoother to bite on.  Its only creases are the bore
# mouth and the crater-floor edge, which need many sweeps before they relax into
# the cast-looking funnel the panel now shows.  Measured: 10 sweeps move the
# surface 0.0014 (0.7 % of the bar radius) and change nothing visible, 120 move
# 0.0020 and do, 300 move 0.0035 and dish the crater floor into a bowl.
Va = taubin(_Va0, kept_a, np.concatenate([np.asarray(l, dtype=np.int64)
                                          for l in cap["loops_removed"]]),
            iters=120)
print(f"    taubin: moved {np.abs(Va - _Va0).max():.5f} max, "
      f"{np.linalg.norm(Va - _Va0, axis=1).mean():.5f} mean")
# The removal leaves ONE piece here -- the bore runs through, so its wall stays
# attached to the bar at the far end.  Checked rather than assumed, because a
# blind bore would leave two and the panels would then be drawing an orphan with
# no colour for it.
_ccs = R._connected_components_tri_faces(
    torch.as_tensor(kept_a, dtype=torch.int32, device=dev))
print(f"    kept mesh: {len(_ccs)} component(s)")
assert len(_ccs) == 1, \
    f"the bore should stay attached at the far end; got {len(_ccs)} components"
write_poly("bzx_a_removed.vtk", Va, kept_a, np.zeros(len(kept_a)), "after removal")
write_lines("bzx_a_loop_outer.vtk", Va,
            [np.asarray(l) for l, i in zip(cap["loops_removed"], cap["is_inner"])
             if not i], "outer hole loop")
write_lines("bzx_a_loop_inner.vtk", Va,
            [np.asarray(l) for l, i in zip(cap["loops_removed"], cap["is_inner"])
             if i], "nested hole loop")
# The cap is its own mesh so the renderer can draw ITS triangulation and nothing
# else's: the fan is the subject, and edges on the whole bar would bury it.
write_poly("bzx_a_cap.vtk", Va, cap["inner_cap_tris"],
           np.full(len(cap["inner_cap_tris"]), 4), "inner-loop fan cap")

# ===========================================================================
# B — more hole loops than patch loops: clustered, bridged, capped at step 4
# ===========================================================================
print("\n=== B  the clamped trunk has split into three struts")
# The surfaces here are UNIFORM CSG MESHES, not marching cubes.  The bridge
# is one base EDGE wide -- _bridge_two_hole_loops spans the closest edge pair,
# and subdivision only lengthens the strip, never widens it -- and a
# marching-cubes loop carries edges anywhere from ~0 to h, so the bridge is as
# wide as whatever edge the search happened to land on.  It landed on a 0.003
# one (12 % of h), which is a hairline no camera or grid can rescue: the strip
# tapered to a point.  A cylinder tessellated in even segments gives loops whose
# every edge is the same length, so the bridge is a proper strip of known width
# (~0.028 here) at both ends.  Nothing else about the case changes -- the BC
# nodes are still background-grid nodes and the radius is still 2 x the cell
# size, and the algorithm neither knows nor cares where the mesh came from.
set_grid(73)
TRUNK_R, TRUNK_X0, X_SPLIT = 0.21, -0.55, -0.52
STRUT_X1, HEAD_X0, HEAD_X1 = -0.26, -0.30, -0.18
RING = 0.022                 # axial spacing
# NOT equilateral: the two fat struts are the closest pair, so they are the ones
# bridged, and the thin one is the smallest perimeter, so it is the one capped --
# both branches of the step land where the eye expects.  Equidistant centroids
# would leave the choice to argmin.
# (angle deg, axis offset, radius, flared radius at the head).  The flared
# radius is set by the head: offset + flared radius must stay inside TRUNK_R, or
# the blend bulges out of the body it is supposed to run into.
STRUTS = ((90.0, 0.12, 0.07, 0.085), (210.0, 0.12, 0.07, 0.085),
          (330.0, 0.15, 0.045, 0.055))
SECT_B = 32                  # -> strut loop edge ~0.0137, which IS the bridge:
                             # the ONE knob that trades loop roundness against how
                             # wide the bridge strip comes out

# The struts run back into a common HEAD past the cut, rather than ending in
# three rounded tips.  The removal leaves them as three separate pieces either
# way -- the algorithm is indifferent -- but three rods floating in space is not
# a part, and the panel is meant to look like something the optimiser produced.
_struts = [_tube(r, X_SPLIT, STRUT_X1, d * np.cos(np.deg2rad(a)),
                 d * np.sin(np.deg2rad(a)), sections=SECT_B, r_end=re, flare=0.06)
           for a, d, r, re in STRUTS]
# The head carries its own, finer tessellation: it is the biggest thing in
# the panel, and it is nowhere near the loops, so its sections cost nothing.
# The STRUTS have to stay coarse -- their loop edge is the bridge's width.
_head = _tube(TRUNK_R, HEAD_X0, HEAD_X1, tip=True, sections=72, ring=0.015)
vb0, fb0 = _to_torch(trimesh.boolean.union(
    [_tube(TRUNK_R, TRUNK_X0, HEAD_X0, sections=72, ring=0.015), _head]))
vb1, fb1 = _to_torch(trimesh.boolean.union(
    [_tube(TRUNK_R, TRUNK_X0, X_SPLIT, sections=SECT_B)] + _struts + [_head]))
frozen_b = PTS[(PTS[:, 0] > TRUNK_X0) & (PTS[:, 0] < TRUNK_X0 + 0.05)
               & (PTS[:, 1:].norm(dim=1) < TRUNK_R)].double()
print(f"    init {len(vb0)}V {len(fb0)}F, updated {len(vb1)}V {len(fb1)}F, "
      f"frozen {len(frozen_b)}")
patch_b = R.compute_bc_patch(vb0, fb0, frozen_b, RADIUS)
comps_b = patch_b["components"]
assert len(comps_b) == 1 and len(comps_b[0]["boundary_loops_local"]) == 1, \
    "case B wants one single-loop patch"
bv_b, bf_b, kept_b, ov_b, of_b, lb_b = run_workflow(vb1, fb1, comps_b)
assert len(cap.get("merge_tris", [])) > 0, \
    "nothing was bridged — geometry does not exercise step 4"
print(f"    step 4: {len(cap['loops_before_merge'])} hole loops -> "
      f"{len(cap['loops_after_merge'])} ({len(cap['merge_tris'])} bridge + "
      f"{len(cap['cluster_cap_tris'])} cap tris)")
stats.update(b_loops_before=len(cap["loops_before_merge"]),
             b_loops_after=len(cap["loops_after_merge"]),
             b_bridge_tris=len(cap["merge_tris"]),
             b_cap_tris=len(cap["cluster_cap_tris"]),
             b_patch_loops=len(comps_b[0]["boundary_loops_local"]))

# No display transform here.  The bridge lands in the plane of the cut, in
# the gap between the two struts it joins, so a camera looking at the cut faces
# sees it head on.  (The rectangular-legged version of this case put the bridge
# on the plate's underside and had to be turned over to be seen at all.)
_Vb0 = cap["verts_after_merge"]
Vb = taubin(_Vb0, kept_b, np.concatenate([np.asarray(l, dtype=np.int64)
                                          for l in cap["loops_before_merge"]]))
print(f"    taubin: moved {np.abs(Vb - _Vb0).max():.5f} max, "
      f"{np.linalg.norm(Vb - _Vb0, axis=1).mean():.5f} mean")
# The cached patch is NOT drawn here.  It sits exactly where the removed pad
# was -- i.e. between the camera and the three cut legs -- and hid the bridge
# and the cap, which are what these two panels exist to show.  What it would
# have contributed (that the patch has one loop) is in the panel titles.
write_poly("bzx_b_removed.vtk", Vb, kept_b, np.zeros(len(kept_b)), "after removal")
write_lines("bzx_b_loops.vtk", Vb,
            [np.asarray(l) for l in cap["loops_before_merge"]], "hole loops")
# Bridge and cap as their own meshes, over the same base as the panel before
# it: the pair then differs by exactly what step 4 added, and only those
# triangles are drawn with their edges.  One file per material, so the renderer
# can colour them solid -- scalar colouring overrides the backface colour that
# every panel here needs to make "inside" read as inside.
write_poly("bzx_b_bridge.vtk", Vb, cap["merge_tris"],
           np.full(len(cap["merge_tris"]), 3), "bridge")
write_poly("bzx_b_cap.vtk", Vb, cap["cluster_cap_tris"],
           np.full(len(cap["cluster_cap_tris"]), 4), "fan cap")
write_lines("bzx_b_loops_after.vtk", Vb,
            [np.asarray(l) for l in cap["loops_after_merge"]], "loop after merge")

# ===========================================================================
with open("bzx_stats.txt", "w") as fh:
    fh.write(f"grid_n {N}\nh {h:.6f}\nradius {RADIUS:.6f}\n")
    for k, v in stats.items():
        fh.write(f"{k} {v}\n")
print("\nwrote bzx_stats.txt")

# Five marching-cubes contexts in one process makes Warp's interpreter-shutdown
# teardown race with Torch's CUDA teardown; it SIGSEGVs after the last line of
# real work about half the time, which under `set -e` would abort build.sh long
# after this script had actually succeeded.  Everything above is on disk by now
# and every failure mode above is an assert, so leave before the race can start.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
