"""Prep step (conda `mechfoundry-warp117`, GPU): capture every stage of the BC-zip workflow.

The GEOMETRY is fabricated — a small Y-bracket clamped at its hub, whose two
lower legs the optimiser has replaced by a webbed foot that splits into three
slender toes.  Chosen to be readable in one glance and, unlike the cantilever of
the SDF shape-optimization example, to actually reach every branch of the
algorithm: on that beam the removal region is a one-loop collar that validates
immediately, so neither the ring expansion nor the clustering path ever runs.

The bracket is built so the CACHED PATCH HAS MORE LOOPS (3, one per original
arm) THAN THE INITIAL REMOVAL CANDIDATE (2 — the webbed foot leaves a single
opening across the whole bottom of the box).  That is precisely the condition
``_validate`` rejects with ``cand-loops=… < required=…``, so step 2's BFS ring
expansion has to run; it stops 13 rings later at 4 loops, which is one more than
the patch has, so step 4's clustering runs as well.

The ALGORITHM is the production code, untouched.  Both surfaces come out of
``analytical_sdf.extract_surface`` (the same marching cubes the solver uses),
the patch out of ``remeshing.compute_bc_patch``, and the workflow out of
``remeshing.stitch_bc_patches``; this file only wraps its helpers to record what
they were handed and what they returned.  The paper's seven steps map to:

  1 cache patch + nested check  compute_bc_patch / _check_no_nested_loops
  2 grow the candidate by       B.1 1.1x AABB -> B.2 candidate -> B.3 BFS rings
    facet rings until valid     (the box is fixed; only the ring count grows)
  3 remove; fan-cap inner loops _cap_inner_hole_loops
  4 cluster + bridge        _cluster_and_bridge_hole_loops
                                (_bridge_two_hole_loops, _cap_loop_fan)
  5 match + zip to patch    _zipper_stitch
  6 subdivide long bridges      _subdivide_bridge_faces
  7 Taubin-smooth the new tris  the smoothing block of stitch_bc_patches

Writes ``bz_capture.npz`` + ``bz_stats.txt``.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.append(os.path.abspath("../src/warp_wrapper"))

from core.analytical_sdf import extract_surface                       # noqa: E402
from core import remeshing as R                                       # noqa: E402

# Two cases share this capture machinery.  ``table`` is the FIGURE: a real CC0
# CAD model (Poly Haven WoodenTable_02) whose legs the optimiser has twisted 45
# degrees, exercising the ordinary path end to end.  ``bracket`` is a fabricated
# mini example kept for the two branches the ordinary path never reaches -- the
# BFS ring expansion of step 2 and the loop clustering of step 4.
CASE = os.environ.get("BZ_CASE", "table")
assert CASE in ("table", "bracket"), f"unknown BZ_CASE {CASE}"

dev = "cuda"
if CASE == "table":
    N, L = 161, 1.2
else:
    N, L = 113, 2.0
h = L / (N - 1)

HZ = 0.09                    # plate half-thickness
R_HUB = 0.15                 # clamped hub
R_ARM = 0.075                # arm radius of the initial Y
LEN = 0.62                   # arm / toe length
ARMS = (90, 210, 330)        # initial arm directions, degrees
FOOT = dict(x=0.18, y=-0.12, r=0.12)     # updated: webbed foot (a capsule)
TOE_X = (-0.14, 0.0, 0.14)               # updated: three toes below the web
R_TOE = 0.045
BCH = 0.14                   # BC clamp: frozen nodes with |x|,|y| <= BCH

lin = torch.linspace(-L / 2, L / 2, N, device=dev)
X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")


def extrude(d2, hz):
    """Exact SDF of a 2-D shape swept to a plate of half-thickness ``hz``."""
    w = torch.stack([d2, Z.abs() - hz], -1)
    return torch.clamp(w.amax(-1), max=0) + torch.clamp(w, min=0).norm(dim=-1)


def seg2(ax, ay, bx, by, r):
    """2-D capsule: distance to segment a->b, minus r."""
    pax, pay = X - ax, Y - ay
    bax, bay = bx - ax, by - ay
    t = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0, 1)
    return torch.hypot(pax - t * bax, pay - t * bay) - r


def disc2(r):
    return torch.hypot(X, Y) - r


def arm2(deg, ln, r):
    a = np.deg2rad(deg)
    return seg2(0.0, 0.0, ln * np.cos(a), ln * np.sin(a), r)


def union(*ds):
    out = ds[0]
    for d in ds[1:]:
        out = torch.minimum(out, d)
    return out


# --- the two shapes ---------------------------------------------------------
if CASE == "table":
    # Real CAD (CC0).  "Before" is the model as downloaded; "after" is the same
    # model with each leg rotated 45 degrees about ITS OWN axis -- a twist about
    # the table's axis would swing the legs sideways until the patch and the leg
    # above it no longer overlap, leaving the zipper nothing to bridge.  The
    # rotation is applied by re-querying the BVH at rotated points rather than
    # resampling the SDF grid, which would round off the legs' corners -- the
    # very feature that makes the twist legible.
    import trimesh                                                # noqa: E402
    from core.analytical_sdf import build_warp_mesh, compute_nodal_sdf  # noqa: E402

    CUT = -0.40                  # BC region: the bottom 10% of the legs
    TWIST = np.deg2rad(45.0)
    TWIST_TOP, FLAT_TO = 0.23, CUT + 0.12
    m = trimesh.load(f"{HERE}/cad/WoodenTable_02/WoodenTable_02_1k.gltf",
                     force="mesh", process=True)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.vertices -= m.bounds.mean(0)
    m.vertices /= m.extents.max()
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], -1)
    wpm, neg, _keep = build_warp_mesh(
        torch.as_tensor(m.vertices, dtype=torch.float32, device=dev),
        torch.as_tensor(m.faces, dtype=torch.int32, device=dev),
        torch.tensor([0.0, 0.0, 0.45], device=dev),
        float(np.linalg.norm(m.extents)), dev)
    sdf_init = compute_nodal_sdf(wpm, pts, neg, max_dist=L).reshape(N, N, N)

    # The twist ramp is FLAT through the whole zip band and fades out before the
    # apron: the legs stay four isolated columns only up to z = +0.225, and a
    # per-leg rotation applied to shared apron material would shear the skirt.
    t = ((TWIST_TOP - Z) / (TWIST_TOP - FLAT_TO)).clamp(0, 1)
    ramp = (t * t * t * (t * (t * 6 - 15) + 10)).reshape(-1)

    solid_lo = pts[(sdf_init.reshape(-1) < 0) & (pts[:, 2] <= CUT)]
    legs = solid_lo[:, :2].norm(dim=1).mean()
    ang = torch.arange(4, device=dev, dtype=pts.dtype) * (np.pi / 2)
    legs = torch.stack([legs * torch.cos(ang), legs * torch.sin(ang)], -1)
    for _ in range(40):                       # k-means onto the four leg axes
        lab = torch.cdist(solid_lo[:, :2], legs).argmin(1)
        for jj in range(4):
            sel = solid_lo[lab == jj, :2]
            if sel.numel():
                legs[jj] = sel.mean(0)

    ctr = legs[torch.cdist(pts[:, :2], legs).argmin(1)]
    th = TWIST * ramp
    rel = pts[:, :2] - ctr
    ct, st = torch.cos(th), torch.sin(th)
    pts_tw = torch.stack([ctr[:, 0] + ct * rel[:, 0] + st * rel[:, 1],
                          ctr[:, 1] - st * rel[:, 0] + ct * rel[:, 1],
                          pts[:, 2]], -1)
    sdf_upd = compute_nodal_sdf(wpm, pts_tw, neg, max_dist=L).reshape(N, N, N)
else:
    # before: a plain Y bracket — hub + three arms.
    sdf_init = extrude(union(disc2(R_HUB), *[arm2(a, LEN, R_ARM) for a in ARMS]), HZ)
    # after: the top arm survives; the two legs are now a webbed foot that splits
    # into three toes.  Inside the clamp box the foot is solid, so the removal
    # candidate starts with ONE opening across the bottom where the patch has two.
    sdf_upd = extrude(union(
        disc2(R_HUB), arm2(90, LEN, R_ARM),
        seg2(-FOOT["x"], FOOT["y"], FOOT["x"], FOOT["y"], FOOT["r"]),
        *[seg2(x, FOOT["y"], x, -LEN, R_TOE) for x in TOE_X]), HZ)

verts_i, faces_i, _ = extract_surface(sdf_init.contiguous(), domain_length=L, device=dev)
verts_u, faces_u, _ = extract_surface(sdf_upd.contiguous(), domain_length=L, device=dev)
print(f"case {CASE}: initial surface {verts_i.shape[0]}V {faces_i.shape[0]}F")
print(f"case {CASE}: updated surface {verts_u.shape[0]}V {faces_u.shape[0]}F")

# --- frozen BC nodes ---------------------------------------------------------
radius = 2.0 * h             # production: 2 x the smallest element size
if CASE == "table":
    # every background node inside the solid, below the cut plane
    frozen = pts[(sdf_init.reshape(-1) < 0) & (pts[:, 2] <= CUT)].double()
else:
    g = torch.arange(-1.0, 1.0, h, device=dev)
    gxy = g[g.abs() <= BCH]
    gz = g[g.abs() <= HZ + h]
    FX, FY, FZ = torch.meshgrid(gxy, gxy, gz, indexing="ij")
    frozen = torch.stack([FX.reshape(-1), FY.reshape(-1), FZ.reshape(-1)], -1).double()
print(f"frozen BC nodes {frozen.shape[0]}, radius = 2h = {radius:.4f}")

# ===========================================================================
# Capture wrappers — the real helpers do the work, we only record.
# ===========================================================================
cap = {}
_orig = {n: getattr(R, n) for n in (
    "_cap_inner_hole_loops", "_cluster_and_bridge_hole_loops",
    "_zipper_stitch", "_subdivide_bridge_faces", "_classify_inner_hole_loops")}


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def classify_wrap(verts, loops):
    """``_validate`` calls this once per attempt, on the CANDIDATE's vertices
    only, so its call log is the step-2 ring-expansion trace.  ``_cap_inner_
    hole_loops`` calls it too, on the whole mesh — told apart by array size."""
    out = _orig["_classify_inner_hole_loops"](verts, loops)
    if len(verts) < 0.5 * len(base_v):                   # a _validate attempt
        cap.setdefault("rings", []).append(
            {"n_loops": len(loops), "n_inner": int(out.sum().item()),
             "verts": _np(verts).copy(),
             "loops": [list(l) for l in loops]})
    return out


def cap_wrap(verts, hole_loops):
    v, outer, cap_t = _orig["_cap_inner_hole_loops"](verts, hole_loops)
    cap["loops_removed"] = [list(l) for l in hole_loops]
    cap["loops_after_inner_cap"] = [list(l) for l in outer]
    cap["inner_cap_tris"] = _np(cap_t).tolist()
    return v, outer, cap_t


def cluster_wrap(verts, hole_loops, n_target, patch_centroids, **kw):
    cap["loops_before_merge"] = [list(l) for l in hole_loops]
    cap["verts_before_merge"] = _np(verts).copy()
    v, merged, bridge_t, cap_t = _orig["_cluster_and_bridge_hole_loops"](
        verts, hole_loops, n_target, patch_centroids, **kw)
    cap["loops_after_merge"] = [list(l) for l in merged]
    cap["merge_tris"] = [list(t) for t in bridge_t]
    cap["cluster_cap_tris"] = [list(t) for t in cap_t]
    cap["verts_after_merge"] = _np(v).copy()
    return v, merged, bridge_t, cap_t


def zip_wrap(h_loop, p_loop_nat, h_pos_t, p_pos_nat_t):
    tris = _orig["_zipper_stitch"](h_loop, p_loop_nat, h_pos_t, p_pos_nat_t)
    cap.setdefault("zip_pairs", []).append(
        {"hole": list(h_loop), "patch": list(p_loop_nat),
         "tris": [list(t) for t in tris]})
    return tris


def subdiv_wrap(st_t, verts, bm, threshold):
    # Called ONCE PER PATCH COMPONENT.  Assigning here (rather than appending)
    # kept only the last component's strip, so panel 6 showed one leg bridged
    # and the rest bare -- accumulate.
    cap["verts_before_subdiv"] = _np(verts).copy()
    cap.setdefault("subdiv_before_list", []).append(_np(st_t).copy())
    cap["subdiv_threshold"] = float(threshold)
    out_t, out_v, out_m = _orig["_subdivide_bridge_faces"](st_t, verts, bm, threshold)
    cap.setdefault("subdiv_after_list", []).append(_np(out_t).copy())
    cap["verts_after_subdiv"] = _np(out_v).copy()
    cap["tris_before_subdiv"] = np.vstack(cap["subdiv_before_list"])
    cap["tris_after_subdiv"] = np.vstack(cap["subdiv_after_list"])
    return out_t, out_v, out_m


R._cap_inner_hole_loops = cap_wrap
R._cluster_and_bridge_hole_loops = cluster_wrap
R._zipper_stitch = zip_wrap
R._subdivide_bridge_faces = subdiv_wrap
R._classify_inner_hole_loops = classify_wrap

# ===========================================================================
# Step 1 — cache the patch (production; raises if it contains nested loops)
# ===========================================================================
patch = R.compute_bc_patch(verts_i.double(), faces_i, frozen, radius)
assert patch is not None, "no surface vertices within the BC radius"
comps = patch["components"]
n_loops = [len(c["boundary_loops_local"]) for c in comps]
print(f"step 1: patch {patch['n_verts']} verts, {len(comps)} component(s), "
      f"loops = {n_loops}")
if CASE == "bracket":
    assert len(comps) == 1 and n_loops == [3], \
        f"expected one 3-loop hub patch, got {len(comps)} comps {n_loops}"
else:
    assert len(comps) == 4 and n_loops == [1, 1, 1, 1], \
        f"expected four 1-loop feet, got {len(comps)} comps {n_loops}"

# The scaled AABB of step 2, exactly as stitch_bc_patches computes it.  ONE
# constant: the box drawn in panel 2, the box handed to the production call and
# the B.2 port that colours the panel must all use the same scale, or the
# self-check below fires (it did, at 936 vertices, when this stayed at 1.1
# while the table's call ran at 3.0).
pv = comps[0]["verts_phys"]
pv_min, pv_max = pv.min(0).values, pv.max(0).values
AABB_SCALE = 3.0 if CASE == 'table' else 1.1
half = (pv_max - pv_min) * 0.5 * AABB_SCALE
ctr = (pv_min + pv_max) * 0.5
aabb_lo, aabb_hi = (ctr - half), (ctr + half)
# ...and the same box for EVERY component: the removal box is per patch, so a
# figure that draws one of four implies the other three were never selected.
aabb_all = []
for _c in comps:
    _v = _c["verts_phys"]
    _lo, _hi = _v.min(0).values, _v.max(0).values
    _h = (_hi - _lo) * 0.5 * AABB_SCALE
    _m = (_lo + _hi) * 0.5
    aabb_all.append(np.stack([_np(_m - _h), _np(_m + _h)]))
aabb_all = np.stack(aabb_all)

# ===========================================================================
# Steps 2-7 — one production call
# ===========================================================================
msgs = []
base_v, base_f = R._close_open_loops(verts_u.double(), faces_u)
# The table needs a bigger removal box than the 1.1x default: the loop gap is
# (aabb_scale - 1)/2 of the PATCH's height, and a bottom-10% patch is short, so
# at the default the strip has almost no axial run and the seam creases.  Taubin
# mu is the knob that smooths the strip interior (pass-band 1/lambda + 1/mu).
KW = (dict(aabb_scale=AABB_SCALE, smooth_iters=600, taubin_mu=-0.51)
      if CASE == "table" else {})
out_v, out_f, labels = R.stitch_bc_patches(
    base_verts_phys=base_v, base_faces=base_f, bc_patches=comps, log=msgs.append,
    **KW)
for m in msgs:
    print("   ", m.strip())

rings = cap["rings"]
n_rings = len(rings) - 1                       # attempt 0 is the un-grown box
if CASE == "bracket":
    # the bracket exists precisely to reach these two branches
    assert n_rings > 0, ("the 1.1x AABB validated straight away — geometry does "
                         "not exercise step 2's ring expansion")
    assert rings[0]["n_loops"] < len(comps[0]["boundary_loops_local"]), (
        f"initial candidate already had {rings[0]['n_loops']} loops; the point "
        f"of this case is that it has FEWER than the patch's {n_loops[0]}")
if CASE == "bracket":
    assert cap.get("merge_tris"), \
        "clustering/bridging never ran — geometry does not exercise step 4"
assert cap.get("zip_pairs"), "zipper stitch never ran"
print(f"step 2: {n_rings} BFS ring(s); candidate loops "
      f"{rings[0]['n_loops']} -> {rings[-1]['n_loops']} "
      f"(patch needs {n_loops[0]})")
# ``_cap_inner_hole_loops`` and ``_cluster_and_bridge_hole_loops`` are only
# reached when there is something for them to do — a nested loop to cap, or more
# hole loops than the patch has.  On the ordinary path (the table) neither runs,
# so their capture keys are simply absent.  Fill the ones the figure needs from
# the geometry itself rather than special-casing every downstream reader.
if "loops_removed" not in cap:
    _base_after = base_v.shape[0]  # noqa: F841  (kept for symmetry of intent)
    cap["loops_removed"] = []      # filled below, once base_kept is known
cap.setdefault("inner_cap_tris", [])
cap.setdefault("loops_after_inner_cap", cap["loops_removed"])
cap.setdefault("loops_before_merge", cap["loops_removed"])
cap.setdefault("loops_after_merge", cap["loops_removed"])
cap.setdefault("merge_tris", [])
cap.setdefault("cluster_cap_tris", [])

# Step 4 is SKIPPED whenever the removal already leaves as many loops as the
# patch has -- which is the ordinary case, and the table's.  Report it as
# skipped rather than crashing: an absent capture key is the algorithm taking
# the short path, not a failure.
_n = lambda k: len(cap[k]) if k in cap else 0                      # noqa: E731
print(f"step 3: removal left {_n('loops_removed')} open loop(s); "
      f"inner-capped {_n('inner_cap_tris')} tri(s)")
if "loops_before_merge" in cap:
    print(f"step 4: {_n('loops_before_merge')} -> {_n('loops_after_merge')} "
          f"loops, {_n('merge_tris')} bridge + {_n('cluster_cap_tris')} cap tris")
else:
    print("step 4: skipped — removal already left one loop per patch loop")
print(f"step 5: {len(cap['zip_pairs'])} loop pair(s), "
      f"{sum(len(z['tris']) for z in cap['zip_pairs'])} zip tris")
print(f"step 6: {len(cap['tris_before_subdiv'])} -> {len(cap['tris_after_subdiv'])} "
      f"tris at threshold {cap['subdiv_threshold']:.4f}")
print(f"out: {out_v.shape[0]}V {out_f.shape[0]}F  base/bc/bridge = "
      f"{int((labels==0).sum())}/{int((labels==1).sum())}/{int((labels==2).sum())}")

# Watertight check — the whole point of the workflow.  Test it the standard
# way: every edge in exactly two triangles.  (extract_boundary_loops_strict is
# for OPEN boundaries and reports "branching-out-degree" on a closed mesh, which
# reads like a failure and is not one.)
def _edge_report(F):
    F = _np(F).astype(np.int64)
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
                .reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int((cnt > 2).sum())


n_open_edges, n_nonmanifold = _edge_report(out_f)
watertight = (n_open_edges == 0 and n_nonmanifold == 0)
print(f"result watertight: {watertight} "
      f"(open edges {n_open_edges}, non-manifold {n_nonmanifold})")
assert watertight, "stitched result is not watertight"
assert _edge_report(base_f) == (0, 0), "input surface was not watertight"

# ===========================================================================
# Save
# ===========================================================================
out = {
    "h": np.float64(h), "radius": np.float64(radius), "bch": np.float64(BCH),
    "aabb_lo": _np(aabb_lo), "aabb_hi": _np(aabb_hi),
    "n_rings": np.int32(n_rings),
    "cand_loops_first": np.int32(rings[0]["n_loops"]),
    "cand_loops_last": np.int32(rings[-1]["n_loops"]),
    "verts_init": _np(verts_i), "faces_init": _np(faces_i),
    "verts_upd": _np(base_v), "faces_upd": _np(base_f),
    "patch_verts": _np(comps[0]["verts_phys"]), "patch_faces": _np(comps[0]["faces"]),
    "verts_out": _np(out_v), "faces_out": _np(out_f), "labels_out": _np(labels),
    "verts_pre_smooth": cap["verts_after_subdiv"],
    "tris_before_subdiv": cap["tris_before_subdiv"],
    "tris_after_subdiv": cap["tris_after_subdiv"],
    "subdiv_threshold": np.float64(cap["subdiv_threshold"]),
    "merge_tris": np.asarray(cap["merge_tris"], dtype=np.int64),
    "cluster_cap_tris": np.asarray(cap["cluster_cap_tris"], dtype=np.int64),
    "inner_cap_tris": np.asarray(cap["inner_cap_tris"], dtype=np.int64).reshape(-1, 3),
    "frozen": _np(frozen),
}
for li, lp in enumerate(comps[0]["boundary_loops_local"]):
    out[f"patch_loop{li}"] = np.asarray(lp, dtype=np.int64)
out["n_patch_loops"] = np.int32(len(comps[0]["boundary_loops_local"]))
for key in ("loops_removed", "loops_after_inner_cap", "loops_before_merge",
            "loops_after_merge"):
    for li, lp in enumerate(cap[key]):
        out[f"{key}_{li}"] = np.asarray(lp, dtype=np.int64)
    out[f"n_{key}"] = np.int32(len(cap[key]))
for pi, zp in enumerate(cap["zip_pairs"]):
    out[f"zip{pi}_hole"] = np.asarray(zp["hole"], dtype=np.int64)
    out[f"zip{pi}_patch"] = np.asarray(zp["patch"], dtype=np.int64)
    out[f"zip{pi}_tris"] = np.asarray(zp["tris"], dtype=np.int64)
out["n_zip_pairs"] = np.int32(len(cap["zip_pairs"]))
np.savez_compressed("bz_capture.npz", **out)
print("wrote bz_capture.npz")

with open("bz_stats.txt", "w") as f:
    f.write(f"grid_n {N}\nh {h:.6f}\nradius {radius:.6f}\n")
    f.write(f"init_faces {faces_i.shape[0]}\nupd_faces {base_f.shape[0]}\n")
    f.write(f"patch_verts {patch['n_verts']}\n")
    f.write(f"patch_faces {comps[0]['faces'].shape[0]}\n")
    f.write(f"patch_loops {len(comps[0]['boundary_loops_local'])}\n")
    f.write(f"rings {n_rings}\n")
    f.write(f"cand_loops_first {rings[0]['n_loops']}\n")
    f.write(f"cand_loops_last {rings[-1]['n_loops']}\n")
    f.write(f"loops_removed {len(cap['loops_removed'])}\n")
    f.write(f"inner_cap_tris {len(cap['inner_cap_tris'])}\n")
    f.write(f"loops_after_merge {len(cap['loops_after_merge'])}\n")
    f.write(f"merge_tris {len(cap['merge_tris'])}\n")
    f.write(f"cluster_cap_tris {len(cap['cluster_cap_tris'])}\n")
    f.write(f"case {CASE}\n")
    f.write(f"zip_pairs {len(cap['zip_pairs'])}\n")
    f.write(f"zip_tris {sum(len(z['tris']) for z in cap['zip_pairs'])}\n")
    f.write(f"subdiv_before {len(cap['tris_before_subdiv'])}\n")
    f.write(f"subdiv_after {len(cap['tris_after_subdiv'])}\n")
    f.write(f"subdiv_threshold {cap['subdiv_threshold']:.6f}\n")
    f.write(f"out_faces {out_f.shape[0]}\n")
    f.write(f"n_base {int((labels==0).sum())}\nn_bc {int((labels==1).sum())}\n")
    f.write(f"n_bridge {int((labels==2).sum())}\n")
    f.write(f"watertight {int(watertight)}\n")
    f.write("aabb_lo " + " ".join(f"{v:.6f}" for v in _np(aabb_lo)) + "\n")
    f.write("aabb_hi " + " ".join(f"{v:.6f}" for v in _np(aabb_hi)) + "\n")
print("wrote bz_stats.txt")


# ===========================================================================
# Per-panel geometry for the renderer.  One POLYDATA per panel, each with a
# per-cell ``stage`` label the renderer maps to a colour.
# ===========================================================================
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
    print(f"wrote {path} ({len(f)} tris)")


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
    print(f"wrote {path} ({len(loops)} loops)")


# stitch_bc_patches COMPACTS vertices before returning (drops unreferenced ones
# and remaps out_faces), so output indices live in a different space from the
# intermediates captured inside it.  Everything below therefore works in the
# PRE-compaction space, and identifies the removed candidate by face CENTROID —
# a position-space test, immune to the remap.  Base faces keep their positions
# (Taubin only moves bridge vertices), so the match is exact.
V_pre = cap["verts_after_subdiv"]
V_post, F_out, LB = _np(out_v), _np(out_f).astype(np.int64), _np(labels)
Fin = _np(base_f).astype(np.int64)
Vin = _np(base_v)


def _cent_keys(V, F):
    return [tuple(np.round(p, 9)) for p in V[F].mean(axis=1)]


kept_keys = set(_cent_keys(V_post, F_out[LB == 0]))
removed_mask = np.array([k not in kept_keys for k in _cent_keys(Vin, Fin)])
removed, base_kept = Fin[removed_mask], Fin[~removed_mask]
assert len(base_kept) == int((LB == 0).sum()), (
    f"kept-face count mismatch: {len(base_kept)} vs {int((LB == 0).sum())}")
print(f"removed candidate: {len(removed)} tris of {len(Fin)}")

if not cap["loops_removed"]:
    # The open loops step 3 leaves ARE the boundary loops of the kept base mesh;
    # read them off it with the production extractor instead of inventing them.
    _lp = R.extract_boundary_loops(
        torch.as_tensor(base_kept, dtype=torch.int32, device=dev))
    cap["loops_removed"] = [list(_np(l)) for l in _lp]
    cap["loops_after_inner_cap"] = cap["loops_removed"]
    cap["loops_before_merge"] = cap["loops_removed"]
    cap["loops_after_merge"] = cap["loops_removed"]
    print(f"step 3 loops read back from the kept base mesh: "
          f"{len(cap['loops_removed'])}")


# Step 2's B.2 initial candidate, ported from stitch_bc_patches: active faces
# with >=1 vertex in the 1.1x AABB, restricted to the connected component whose
# centroid is nearest the patch centre.  The port is checked against
# production's own intermediate — the vertex set _validate handed to
# _classify_inner_hole_loops on attempt 0 — before it is drawn.
_bf = base_f.to(torch.int64)
_vin = ((base_v >= aabb_lo) & (base_v <= aabb_hi)).all(-1)
_cand = _vin[_bf[:, 0]] | _vin[_bf[:, 1]] | _vin[_bf[:, 2]]
_ci = _cand.nonzero(as_tuple=True)[0]
_ccs = R._connected_components_tri_faces(base_f[_ci])
if len(_ccs) > 1:
    _pc = (pv.min(0).values + pv.max(0).values) * 0.5
    _best = min(_ccs, key=lambda cc: float(torch.linalg.norm(
        base_v[_bf[_ci[torch.tensor(cc, dtype=torch.int64, device=_bf.device)]]]
        .mean(dim=1).mean(0) - _pc)))
    _keep = torch.zeros_like(_cand)
    _keep[_ci[torch.tensor(_best, dtype=torch.int64, device=_bf.device)]] = True
    _cand = _keep
cand0 = _np(_cand)
_have = {tuple(np.round(p, 9)) for p in Vin[np.unique(Fin[cand0].reshape(-1))]}
_want = {tuple(np.round(p, 9)) for p in rings[0]["verts"]}
assert _have == _want, (
    f"B.2 port disagrees with production's attempt-0 candidate: "
    f"{len(_have - _want)} extra / {len(_want - _have)} missing vertices")
print(f"initial {AABB_SCALE}x AABB candidate: {int(cand0.sum())} tris "
      f"({rings[0]['n_loops']} loops) -> grown {len(removed)} tris "
      f"({rings[-1]['n_loops']} loops) after {n_rings} rings")
out["cand_first_faces"] = Fin[cand0]

# Patch faces in PRE-compaction global indices: stitch_bc_patches appends the
# used patch vertices at V0 = len(verts) just after the cap/merge stage.
# KNOWN LIMITATION: every per-panel VTK below is assembled from comps[0]
# only.  That was exact for the bracket (one patch component) and is WRONG for
# any multi-component case such as the table's four feet -- panels 1/3 then
# highlight one foot of four, and panels 5/6 carry no patch at all because the
# other three components' faces are never remapped into pre-compaction indices.
# The final panel (7) is read straight off the production output, so it is
# correct regardless.  Fixing this means looping the remap over all components.
if len(comps) > 1:
    print(f"WARNING: {len(comps)} patch components — per-panel VTKs cover only "
          f"the first; panels 1/3/5/6 are incomplete for this case")
Pf_local = _np(comps[0]["faces"]).astype(np.int64)
used = np.unique(Pf_local.reshape(-1))
# When steps 3/4 add nothing (the ordinary path), the patch is appended
# straight after the base mesh, so V0 is just the base vertex count.
V0 = len(cap["verts_after_merge"]) if "verts_after_merge" in cap \
    else int(base_v.shape[0])
p2g = np.full(int(Pf_local.max()) + 1, -1, dtype=np.int64)
p2g[used] = np.arange(V0, V0 + len(used))
patch_glob = p2g[Pf_local]
assert patch_glob.min() >= 0 and patch_glob.max() < len(V_pre), "patch remap out of range"

mt = np.asarray(cap["merge_tris"], dtype=np.int64).reshape(-1, 3)
ct = np.asarray(cap["cluster_cap_tris"], dtype=np.int64).reshape(-1, 3)
zt = np.vstack([np.asarray(z["tris"], dtype=np.int64).reshape(-1, 3)
                for z in cap["zip_pairs"]])
st_after = cap["tris_after_subdiv"].astype(np.int64).reshape(-1, 3)
out["removed_faces"] = removed
out["aabb_all"] = aabb_all
np.savez_compressed("bz_capture.npz", **out)

# panel 1 — initial surface, patch highlighted (patch faces are in patch-local
# indices, so write the patch as its own mesh)
Vi, Fi = _np(verts_i), _np(faces_i).astype(np.int64)
_pkeys = set()
for _c in comps:
    _cv, _cf = _np(_c["verts_phys"]), _np(_c["faces"]).astype(np.int64)
    _pkeys |= set(_cent_keys(_cv, _cf))
from scipy.spatial import cKDTree                                  # noqa: E402
_pc = np.vstack([_np(_c["verts_phys"])[_np(_c["faces"]).astype(np.int64)].mean(1)
                 for _c in comps])
_bc = Vi[Fi].mean(1)
_body_only = cKDTree(_pc).query(_bc)[0] > 0.35 * h
write_poly("bz_p1_body.vtk", Vi, Fi[_body_only], np.zeros(int(_body_only.sum())),
           "initial surface")
print(f"panel 1 body: dropped {int((~_body_only).sum())} faces the patch overlay "
      f"redraws (coincident triangles z-fight)")
# EVERY component, concatenated into one mesh with its own vertex array.  The
# cached patch is exact and never moves, so this same file is also what panels
# 5 and 6 overlay -- which sidesteps the pre-compaction index remap entirely
# (that remap is what only ever handled comps[0]).
_pv, _pf, _pl, _off = [], [], [], 0
for _c in comps:
    _v = _np(_c["verts_phys"])
    _f = _np(_c["faces"]).astype(np.int64)
    _pv.append(_v)
    _pf.append(_f + _off)
    _pl.extend(_np(l) + _off for l in _c["boundary_loops_local"])
    _off += len(_v)
Pv, Pf = np.vstack(_pv), np.vstack(_pf)
write_poly("bz_patch_all.vtk", Pv, Pf, np.full(len(Pf), 5), "cached BC patch")
write_lines("bz_p1_loops.vtk", Pv, _pl, "patch loops")
print(f"patch overlay: {len(comps)} component(s), {len(Pf)} tris, {len(_pl)} loops")

# panel 2 — updated surface; the 1.1x AABB candidate (stage 1) inside the grown
# one (stage 2), so the reader sees what the BFS rings added.
_grew = removed_mask & ~cand0
write_poly("bz_p2.vtk", Vin, np.vstack([base_kept, Fin[removed_mask]]),
           np.concatenate([np.zeros(len(base_kept)),
                           np.full(int(removed_mask.sum()), 1)]),
           "updated surface + candidate")
write_lines("bz_p2_loops.vtk", Vin,
            [np.asarray([int(np.abs(Vin - rings[0]["verts"][i]).sum(1).argmin())
                         for i in lp]) for lp in rings[0]["loops"]],
            "initial candidate loops")

# Two box sets: (a) shows each patch's OWN extent (scale 1), which is what the
# cache establishes; (b) shows the same boxes scaled up, which select the
# initial candidate -- the BFS facet rings then grow the removed region past
# it.  Drawing the scaled box in (a) would claim the workflow had already
# decided how far to reach.
def _write_boxes(path, boxes, name):
    _bv, _be, _o = [], [], 0
    for _lo, _hi in boxes:
        _c8 = np.array([[x, y, z] for x in (_lo[0], _hi[0])
                        for y in (_lo[1], _hi[1]) for z in (_lo[2], _hi[2])])
        _bv.append(_c8)
        for _i, _j in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
                       (4,5),(4,6),(5,7),(6,7)):
            _be.append([_i + _o, _j + _o])
        _o += 8
    write_lines(path, np.vstack(_bv), [np.asarray(e) for e in _be], name)


aabb_raw = []
for _c in comps:
    _v = _c["verts_phys"]
    aabb_raw.append(np.stack([_np(_v.min(0).values), _np(_v.max(0).values)]))
_write_boxes("bz_p1_boxes.vtk", np.stack(aabb_raw), "patch extents")
_write_boxes("bz_p2_boxes.vtk", aabb_all, "removal boxes")

# panel 3 — after removal, with the open loops
write_poly("bz_p3.vtk", V_pre, base_kept, np.zeros(len(base_kept)), "after removal")
write_lines("bz_p3_loops.vtk", V_pre,
            [np.asarray(l) for l in cap["loops_removed"]], "open loops")

# panel 4 — cluster + bridge + cap.  Only written when step 4 actually ran:
# on the ordinary path the removal already leaves one loop per patch loop, so
# there is nothing to cluster and panel 4 would be a copy of panel 3.  A figure
# that shows the same picture twice under two different captions is worse than
# a figure that shows six steps and says where the seventh lives.
STEP4_RAN = len(mt) + len(ct) > 0
if STEP4_RAN:
    write_poly("bz_p4.vtk", V_pre,
               np.vstack([base_kept, mt, ct]),
               np.concatenate([np.zeros(len(base_kept)), np.full(len(mt), 3),
                               np.full(len(ct), 4)]),
               "bridged + capped")
    write_lines("bz_p4_loops.vtk", V_pre,
                [np.asarray(l) for l in cap["loops_after_merge"]],
                "loops after merge")
else:
    for _f in ("bz_p4.vtk", "bz_p4_loops.vtk"):
        if os.path.exists(_f):
            os.remove(_f)                      # never leave a stale panel behind
    print("panel 4: not written — step 4 did not run on this geometry")

# panel 5 — patch inserted and zipped
write_poly("bz_p5.vtk", V_pre,
           np.vstack([base_kept, patch_glob, mt, ct, zt]),
           np.concatenate([np.zeros(len(base_kept)), np.full(len(patch_glob), 5),
                           np.full(len(mt), 3), np.full(len(ct), 4),
                           np.full(len(zt), 6)]),
           "patch stitched in")

# panel 6 — after subdivision (same vertices, refined stitch triangles)
write_poly("bz_p6.vtk", V_pre,
           np.vstack([base_kept, patch_glob, st_after]),
           np.concatenate([np.zeros(len(base_kept)), np.full(len(patch_glob), 5),
                           np.full(len(st_after), 6)]),
           "after subdivision")

# panel 7 — after Taubin smoothing (same faces, moved vertices)
write_poly("bz_p7.vtk", V_post, F_out,
           np.where(LB == 0, 0, np.where(LB == 1, 5, 6)),
           "after Taubin smoothing")
