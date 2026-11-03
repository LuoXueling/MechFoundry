"""Prep (conda `mechfoundry-warp117`, GPU): one real optimiser run, captured as a LOOP.

This figure is `shape_opt_loop`'s structure carrying `void_nucleation`'s content:
the same forward/backward cycle — design, solve, sensitivity, update, remesh —
but run in PHASE-FIELD mode, so every stage has two halves and the extra half is
the hidden phase field that ordinary shape optimisation does not have:

  (a) the design      d  and the hidden phi that rides along with it
  (b) the solve       diffused-Nitsche |u| on the trimmed material mesh
  (c) the sensitivity dJ/dd (a thin shell) and dJ/dphi (the whole interior)
  (d) the update      hidden phi steps on dJ/dphi and dips below 0.5;
                      d steps on dJ/dd and moves only in its band
  (e) the remesh      marching cubes + BC restore — where the dip is CARVED
                      into the SDF, so the new surface has an interior void

`(e)` closes back onto `(a)`, which is why the loop is the right frame for this:
the void only exists because the cycle came round again.

The geometry and the schedule are `so_prep.py`/`vn_prep.py`'s, unchanged — the
cylinder blank on four booleaned-on feet, pressed over its whole top, 50 %
volume target, hidden field switched on at a fixed iteration.  The panels are
captured at the NUCLEATING iteration, which is not known in advance: the run
records every iteration into a rolling buffer and freezes it on the iteration
where the thin-feature filter commits and an enclosed cavity appears.

`ShapeOptTrimmed` is subclassed only to RECORD.  Nothing is recomputed and no
loop is re-implemented; even the BC-patch face labels of panel (e) come from the
production `stitch_bc_patches` call itself, through a pure recorder wrapper.

Writes (all `hp_`-prefixed, so `void_nucleation`'s files are untouched):
  hp_pre.vtk       outer octree + sdf, hidden, gphi, gsdf, sdf_c, hidden_c
  hp_solve.vtk     trimmed inner mesh + |u|
  hp_bc_masks.npz  the stitched surface + its BC-patch face labels (panel e)
  hp_evo_<it>.stl  the design surface at the strip iterations
  hp_stats.txt     every number the figure quotes
"""

import os
import sys
import time

import numpy as np
import torch
import trimesh
import warp as wp
from scipy import ndimage

sys.path.append(os.path.abspath("../src/warp_wrapper"))

import core.remeshing as _remeshing  # noqa: E402
from core.bc_surface_selectors import (  # noqa: E402
    HighestSelector, LowestSelector)
from core.shape_opt import ShapeOptTrimmed, _sdf_to_phi  # noqa: E402

# FRAMES mode (HP_FRAMES=1): skip the void-nucleation recorder entirely and
# instead dump every iteration's octree+d (hp_fr_<it>.vtk) and design surface
# (hp_fr_<it>.stl) plus a cumulative optimiser wall-clock (dump I/O excluded) to
# hp_fr_times.txt — the design-evolution frame series' inputs.  HP_OUT_DIR
# redirects the frame files to a scratch folder so hiddenphi/'s loop cache is
# never touched.
_HERE = os.path.dirname(os.path.abspath(__file__))
FR_DIR = os.path.abspath(os.environ.get("HP_OUT_DIR", _HERE))
FRAMES = os.environ.get("HP_FRAMES", "0") == "1"


def _pf(n):
    return os.path.join(FR_DIR, n)


def _write_hex_fr(path, nd, el, sdf):
    """Legacy ASCII VTK hexes + one nodal scalar (numpy in; usable mid-run,
    unlike the module's write_hexes which is defined after optimize())."""
    nd = np.asarray(nd, np.float64)
    el = np.asarray(el, np.int64)
    sdf = np.asarray(sdf, np.float64).reshape(-1)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\noctree hex frame\nASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nd)} float\n")
        np.savetxt(f, nd, fmt="%.6f")
        f.write(f"\nCELLS {len(el)} {len(el) * 9}\n")
        np.savetxt(f, np.hstack([np.full((len(el), 1), 8, np.int64), el]), fmt="%d")
        f.write(f"\nCELL_TYPES {len(el)}\n")
        np.savetxt(f, np.full(len(el), 12, np.int64), fmt="%d")
        f.write(f"\nPOINT_DATA {len(nd)}\n")
        f.write("SCALARS sdf float 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, sdf, fmt="%.6g")

# --- the design problem, taken verbatim from so_prep.py / vn_prep.py ----------
R_CYL = float(os.environ.get("SO_R", 1.0))
H_CYL = float(os.environ.get("SO_H", 1.0))
R_FOOT = float(os.environ.get("SO_RFOOT", 0.6))      # bolt circle
FOOT_R = float(os.environ.get("SO_FOOTR", 0.24))     # foot cylinder radius
FOOT_H = float(os.environ.get("SO_FOOT_H", 0.22))    # protrusion below the disk
FOOT_OV = float(os.environ.get("SO_FOOT_OV", 0.06))  # overlap up into the disk
FROZEN_H = float(os.environ.get("SO_FROZEN_H", 0.02))
TOP_FRAC = float(os.environ.get("SO_TOP_FRAC", 0.06))
TRACTION = (0.0, 0.0, float(os.environ.get("SO_TZ", -0.02)))

# --- production knobs (== vn_prep.py's shipped run) ---------------------------
OCTREE_DEPTH = int(os.environ.get("HP_DEPTH", 7))
INIT_DEPTH = int(os.environ.get("HP_INIT_DEPTH", 4))
EPS_RATIO = 0.5
BAND_K = 3.0
PF_ON_ITER = int(os.environ.get("HP_PF_ON", 35))
PF_LR = float(os.environ.get("HP_PF_LR", 12.0))
GD_LR = float(os.environ.get("HP_GD_LR", 8.0))
N_ITERS = int(os.environ.get("HP_ITERS", 91))
EVO_ITERS = [int(x) for x in
             os.environ.get("HP_EVO", "0,20,35,45,60,80,90").split(",")]
MIN_VOID_CELLS = 8
TARGET_VF = float(os.environ.get("HP_TARGET_VF", 0.5))
W_VOL = float(os.environ.get("HP_W_VOL", 5.0))

h_min = 3.84 / (2 ** OCTREE_DEPTH)
eps = EPS_RATIO * h_min
NOMINAL_PAD = 0.3 + 0.5 * h_min


def _coarse_aligned_clip_bbox(bounds, h_leaf, depth, init_depth, nominal_pad):
    """Expand the nominal box to coarse planes without shifting its cube.

    ``clip_octree_to_bbox`` keeps cells by centroid.  If a requested box face
    cuts through mixed-depth cells, the surviving outer boundary alternates by
    one cell between refinement regions.  In a section render that looks like
    a missing outer layer, and it is also the actual FEM outer domain.  Recover
    the cube origin chosen for the old nominal box, then expand each face to
    the nearest existing initial-octree plane.  This restores the omitted
    boundary cells while preserving every lattice sample near the geometry --
    essential because shifting the cube changes marching-cubes and BC-zip
    topology.
    """
    lo, hi = (np.asarray(v, dtype=np.float64) for v in bounds)
    nominal_lo, nominal_hi = lo - nominal_pad, hi + nominal_pad
    cube_length = h_leaf * (2 ** depth)
    half_length = 0.5 * cube_length
    coarse_h = h_leaf * (2 ** (depth - init_depth))
    half_gap = half_length - 0.5 * (nominal_hi - nominal_lo)
    k = np.maximum(0, np.floor(half_gap / h_leaf + 1.0e-12)).astype(np.int64)
    cube_lo = nominal_lo - k * h_leaf
    lower_index = np.floor((nominal_lo - cube_lo) / coarse_h + 1.0e-12)
    upper_index = np.ceil((nominal_hi - cube_lo) / coarse_h - 1.0e-12)
    box_lo = cube_lo + lower_index * coarse_h
    box_hi = cube_lo + upper_index * coarse_h
    assert np.all(box_lo <= nominal_lo + 1.0e-12)
    assert np.all(box_hi >= nominal_hi - 1.0e-12)
    # Re-running the snapping arithmetic above (a replica of the production
    # routine, not a call to it) on the expanded box must infer the same
    # full-cube origin, so every lattice sample near the geometry stays put.
    # The outer FEM domain does grow by the restored boundary cells.
    new_half_gap = half_length - 0.5 * (box_hi - box_lo)
    new_k = np.maximum(0, np.floor(new_half_gap / h_leaf + 1.0e-12)).astype(np.int64)
    new_cube_lo = box_lo - new_k * h_leaf
    assert np.allclose(new_cube_lo, cube_lo, atol=1.0e-12, rtol=0.0), \
        f"coarse bbox would shift cube origin {cube_lo} -> {new_cube_lo}"
    return tuple(box_lo.tolist()), tuple(box_hi.tolist())


print(f"h_min={h_min:.5f}  eps={eps:.5f}  band |d|<{BAND_K * eps:.5f}")
print(f"pf_phi_lr={PF_LR} (on at iter {PF_ON_ITER})  gd_lr={GD_LR}  "
      f"iters={N_ITERS}")

Z_TOP, Z_BOT = H_CYL / 2.0, -H_CYL / 2.0
Z_FOOT_BOT = Z_BOT - FOOT_H
_base = trimesh.creation.cylinder(radius=R_CYL, height=H_CYL, sections=96)
_ang = np.deg2rad([0.0, 90.0, 180.0, 270.0])   # feet ON THE AXES -> y=0 cuts two
_feet = []
for _a in _ang:
    _fm = trimesh.creation.cylinder(radius=FOOT_R, height=FOOT_H + FOOT_OV,
                                    sections=48)
    _fm.apply_translation([R_FOOT * np.cos(_a), R_FOOT * np.sin(_a),
                           Z_BOT - FOOT_H + (FOOT_H + FOOT_OV) / 2.0])
    _feet.append(_fm)
mesh = trimesh.boolean.union([_base] + _feet)
trimesh.repair.fix_normals(mesh)
assert mesh.is_watertight, "blank + feet union is not watertight"
print(f"blank: {len(mesh.vertices)}V {len(mesh.faces)}F  vol={mesh.volume:.4f}")

CLIP_BBOX = _coarse_aligned_clip_bbox(
    mesh.bounds, h_min, OCTREE_DEPTH, INIT_DEPTH, NOMINAL_PAD
)
print(f"coarse-aligned clip bbox: {CLIP_BBOX[0]} -> {CLIP_BBOX[1]}")


def _assert_complete_outer_box_numpy(nodes, elements, tag):
    """Gate frame exports against a missing bbox layer."""
    corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    extents = corners.max(axis=1) - corners.min(axis=1)
    hex_volume = float(np.prod(extents, axis=1).sum())
    box_lo = np.asarray(CLIP_BBOX[0], dtype=np.float64)
    box_hi = np.asarray(CLIP_BBOX[1], dtype=np.float64)
    box_volume = float(np.prod(box_hi - box_lo))
    assert np.allclose(corners.min(axis=(0, 1)), box_lo, atol=1.0e-7, rtol=0.0)
    assert np.allclose(corners.max(axis=(0, 1)), box_hi, atol=1.0e-7, rtol=0.0)
    assert abs(hex_volume - box_volume) <= 1.0e-7 * box_volume, \
        f"{tag}: outer octree covers {hex_volume:.9g}, expected {box_volume:.9g}"

FOOT_FRAC = float(os.environ.get(
    "SO_FOOT_FRAC", f"{FROZEN_H / (Z_TOP - Z_FOOT_BOT):.4f}"))
fix_sel = LowestSelector("z", fraction=FOOT_FRAC)
load_sel = HighestSelector("z", fraction=TOP_FRAC)

# --- panel (e): record the BC-patch face labels the restoration itself makes --
# `stitch_bc_patches` returns (verts, faces, face_labels) with labels 0 = free
# base, 1 = restored BC patch, 2 = zip bridge, and production discards the third
# (`out_verts_phys, faces, _ = stitch_bc_patches(...)`, shape_opt.py).  Label 1
# is the EXACT set of surfaces the run holds fixed, and §5f is emphatic that a
# proximity guess or a re-resolved selector is not an acceptable substitute.  So
# wrap the function in a recorder: it calls the real one, keeps a copy of all
# three returns, and hands the originals back untouched.  This changes nothing
# the optimiser sees — unlike copying the surrounding sequence into a subclass
# (so_prep's route), which has to be kept in step with production by hand.
_real_stitch = _remeshing.stitch_bc_patches
_STITCH = {}


def _stitch_recorder(*a, **kw):
    out = _real_stitch(*a, **kw)
    _STITCH["last"] = tuple(x.detach().cpu().numpy().copy() for x in out[:3])
    return out


_remeshing.stitch_bc_patches = _stitch_recorder


def _t(x):
    return None if x is None else x.detach().to(torch.float64).cpu().clone()


class Capture(ShapeOptTrimmed):
    """The production optimiser, instrumented to record (never to alter)."""

    def __init__(self, *a, **kw):
        self._it = -1
        self.buf_bw = None          # latest backward snapshot (a, b, c)
        self.buf_flt = None         # latest thin-filter snapshot (d)
        self.snap_pre = None        # frozen at the nucleating iteration
        self.snap_final = None
        self.n_comp = []
        self.trace = []
        self.evo = {}
        self.pf_on_iter = None
        self._t0 = None            # optimiser wall clock start (frames mode)
        self._dump_t = 0.0         # cumulative frame-I/O time, excluded from the clock
        self._frame_times = []     # [(iter, cumulative_compute_seconds)]
        self._traj = []            # [(iter, se_raw, vol_raw)] -> hp_traj.txt (hp_objective)
        super().__init__(*a, **kw)

    def _dump_frame(self):
        """One evolution frame: this iteration's octree+d and design surface, the
        SAME state, timed so its I/O does not inflate the reported wall clock."""
        t = time.perf_counter()
        nd = self._prev_octree_nodes.detach().cpu().numpy()
        el = self._prev_octree_elements.detach().cpu().long().numpy()
        _assert_complete_outer_box_numpy(nd, el, f"frame {self._it}")
        sdf = wp.to_torch(self.e.dof_values).detach().cpu().numpy()
        _write_hex_fr(_pf(f"hp_fr_{self._it}.vtk"), nd, el, sdf)
        s = self._mc_surface()
        if s is not None:
            s.export(_pf(f"hp_fr_{self._it}.stl"))
        self._dump_t += time.perf_counter() - t
        self._frame_times.append(
            (self._it, time.perf_counter() - self._t0 - self._dump_t, len(nd), len(el)))
        print(f"[frame] iter {self._it}: {len(nd)} nodes {len(el)} elems, "
              f"surf={'ok' if s is not None else 'MISSING'}, "
              f"t={self._frame_times[-1][1]:.1f}s")

    # -- (a) the design, (b) the solve, (c) the two sensitivities -------------
    def _post_tape_backward(self):
        super()._post_tape_backward()
        self._it += 1
        if FRAMES:
            if self._t0 is None:
                self._t0 = time.perf_counter()
            self._dump_frame()
            return
        sdf = _t(wp.to_torch(self.e.dof_values))
        # The trimmed INNER mesh is the material domain the FEM actually solved
        # on this iteration; it is read here, before the remesh rebuilds it.
        u = wp.to_torch(self._inner.u.dof_values).detach()
        u = u.reshape(self._inner._prev_octree_nodes.shape[0], -1)
        snap = {
            "it": self._it,
            "nodes": _t(self._prev_octree_nodes),
            "elements": self._prev_octree_elements.detach().cpu().long().clone(),
            "sdf": sdf,
            "gphi": _t(getattr(self, "_last_e_phi_grad", None)),
            "gsdf": _t(getattr(self, "_last_e_grad", None)),
            "hidden": _t(getattr(self, "_hidden_phi", None)),
            # Nodes of every boundary element touching a Dirichlet/Neumann
            # patch: their SDF is FROZEN (`_build_e_fixed_mask`), so the
            # sensitivity printed there is never acted on.
            "bcfix": _t(getattr(self, "_e_fixed_bool", None)),
            "inodes": _t(self._inner._prev_octree_nodes),
            "ielements": self._inner._prev_octree_elements.detach().cpu().long().clone(),
            "umag": torch.linalg.norm(u, dim=1).double().cpu().clone(),
            "surf": self._mc_surface(),
        }
        self.buf_bw = snap
        self.snap_final = snap
        if self._it in EVO_ITERS:
            self.evo[self._it] = snap["surf"]
        deep = sdf < -BAND_K * eps
        hid = snap["hidden"]
        sf = snap["surf"]
        row = {"it": self._it, "n": int(sdf.numel()), "deep": int(deep.sum()),
               "vol": float(sf.volume) if sf is not None else float("nan"),
               "umax": float(snap["umag"].max()),
               "dips": int(((hid < 0.5) & deep).sum()) if hid is not None else -1,
               "flt": "", "void": 0}
        for nm in ("gphi", "gsdf"):
            g = snap[nm]
            ok = g is not None and bool(deep.any())
            row[nm] = float(g.abs()[deep].max()) if ok else float("nan")
            row[nm + "_m"] = float(g.abs().max()) if g is not None else float("nan")
        self.trace.append(row)
        # objective trajectory for hp_objective (compliance + volume vs iter),
        # from the SAME raw compliance/volume the optimiser used, like so_prep.
        self._traj.append(
            (self._it, float(getattr(self._inner, "_last_se_raw", float("nan"))),
             float(getattr(self._inner, "_last_vol_raw", float("nan")))))

    # -- the schedule: phase 1 is pure shape optimisation ---------------------
    def _pf_update_hidden_phi(self):
        if self.pf_on_iter is None and self._it >= PF_ON_ITER:
            self.pf_on_iter = self._it
            vol = self.trace[-1]["vol"] if self.trace else float("nan")
            print(f"  [capture] phase 2: hidden field ON at iteration "
                  f"{self._it}, volume {vol:.4f}")
        self.params["pf_phi_lr"] = PF_LR if self.pf_on_iter is not None else 0.0
        return super()._pf_update_hidden_phi()

    # -- (d) the updated fields: hidden phi after the filter, d after its step -
    def _pf_filter_thin_features(self, phi):
        out = super()._pf_filter_thin_features(phi)
        if self.trace and self.trace[-1]["it"] == self._it:
            self.trace[-1]["flt"] = (
                f"{int((phi < 0.5).sum())}->{int((out < 0.5).sum())}")
        self.buf_flt = {
            "it": self._it,
            "nodes": _t(self._prev_octree_nodes),
            "elements": self._prev_octree_elements.detach().cpu().long().clone(),
            "sdf": _t(wp.to_torch(self.e.dof_values)),
            "phi_in": _t(phi),
            "phi_out": _t(out),
        }
        return out

    # -- (e) the remesh: freeze everything on the iteration that carves -------
    def _post_optimizer_step_remesh_impl(self, iteration):
        r = super()._post_optimizer_step_remesh_impl(iteration)
        if FRAMES:                       # frames mode records nothing here
            return r
        self.n_comp.append((iteration, self._find_voids()))
        found = [v for v in self.n_comp[-1][1] if v[0] >= MIN_VOID_CELLS]
        if self.trace and self.trace[-1]["it"] == self._it:
            self.trace[-1]["void"] = sum(v[0] for v in found)
        committed = (self.buf_flt is not None and self.buf_flt["it"] == self._it)
        if found and committed and self.snap_pre is None:
            self.snap_pre = {"bw": self.buf_bw, "flt": self.buf_flt,
                             "iteration": iteration, "voids": found,
                             "stitch": _STITCH.get("last")}
            print(f"  [capture] *** VOID at optimiser iter {iteration}: "
                  f"{len(found)} enclosed cavity(ies), "
                  f"{[v[0] for v in found]} bg cells ***")
        return r

    def _mc_surface(self):
        try:
            v, f = self._get_or_build_mc_surface_physical()
            return trimesh.Trimesh(vertices=v.detach().cpu().numpy(),
                                   faces=f.detach().cpu().numpy(), process=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [capture] MC surface unavailable: {exc}")
            return None

    def _find_voids(self):
        """Enclosed positive-SDF components of the background grid.

        Counting MC surface components does NOT work — this run carries several
        pieces of boundary debris throughout.  A void is unambiguous in the
        FIELD: a region of sdf > 0 that does not reach the domain boundary.
        """
        g = (self._sussman_grid_sdf.detach()
             .reshape(self._sussman_grid_shape).cpu().numpy())
        lab, k = ndimage.label(g > 0.0)
        if k == 0:
            return []
        outside = set(np.unique(np.concatenate([
            lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
            lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, k + 1))
        return sorted([(int(sizes[lb - 1]), np.argwhere(lab == lb).mean(0))
                       for lb in range(1, k + 1) if lb not in outside],
                      key=lambda v: -v[0])


overrides = {
    "phase_field_epsilon": eps,
    "remeshing": True,
    "remeshing_method": "marchingcubes",
    "reinit_sdf_method": "marchingcubes",
    "sdf_bc_fix_method": "zip",
    "remesh_octree_depth": OCTREE_DEPTH,
    "remesh_init_octree_depth": INIT_DEPTH,
    "remeshing_trimesh": {
        "mesh": mesh,
        "mesh_center": [0.0, 0.0, 0.0],
        "h_target": h_min,
        "clip_bbox": CLIP_BBOX,
    },
    "bc_zip_patch_grow": int(os.environ.get("SO_PATCH_GROW", 0)),
    "design_variable": "phase_field",
    "phi_reproject_band_k": BAND_K,
    "pf_phi_lr": PF_LR,
    "optimizer": "gd",
    "gd_lr": GD_LR,
    "max_optimize_iters": N_ITERS,
    "optimize_relative_tol": 1e-8,
    # Match the void-nucleation capture: after topology change the nonlinear
    # residual can settle at about 1.5e-7 because of GPU reduction order.  This
    # figure-local tolerance lets the run continue without changing production
    # solver defaults, and remains overrideable for diagnostics.
    "absolute_tol": float(os.environ.get("HP_ABSOLUTE_TOL", 5e-7)),
    # default anchored to repo-root output/ via this file's location (see vn_prep).
    "output_root": os.path.abspath(os.environ.get(
        "HP_OUTPUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output"))),
    "output_per_optimize_step": int(os.environ.get("HP_SAVE_EVERY", 1)),
    "output_surface_glb": True,
    "target_volume_fraction": TARGET_VF,
    "shape_opt_w_vol": W_VOL,
    "dirichlet_bcs_dict": {fix_sel: 0.0},
    "neumann_bcs_dict": {load_sel: TRACTION},
}

t0 = time.perf_counter()
opt = Capture("configs.config_shape_opt_cantilever", override_params=overrides)
opt._t0 = time.perf_counter()          # optimiser wall clock starts at optimize()
try:
    opt.optimize()
except RuntimeError as exc:                                        # noqa: BLE001
    print(f"\n!! optimiser stopped early at iteration {opt._it}: {exc}")
print(f"\nrun: {time.perf_counter() - t0:.0f}s, {opt._it + 1} backward passes")

if FRAMES:
    ft = opt._frame_times
    with open(_pf("hp_fr_times.txt"), "w") as f:
        f.write("# iter cum_seconds nodes elems\n")
        for it, ts, nn, ne in ft:
            f.write(f"{it} {ts:.4f} {nn} {ne}\n")
    with open(_pf("hp_fr_stats.txt"), "w") as f:
        f.write(f"gpu={torch.cuda.get_device_name(0)}\n")
        f.write(f"n_frames={len(ft)}\n")
        f.write(f"total_seconds={ft[-1][1] if ft else 0.0:.2f}\n")
    print(f"FRAMES: wrote {len(ft)} hp_fr_*.vtk/.stl + hp_fr_times.txt "
          f"({ft[-1][1] if ft else 0.0:.1f}s optimiser wall clock) to {FR_DIR}")
    sys.exit(0)

opt.evo = {k: v for k, v in opt.evo.items() if v is not None}
print(f"{'it':>3} {'nodes':>6} {'deep':>6} {'dphi_deep':>10} {'dd_deep':>9} "
      f"{'|u|max':>9} {'dips':>5} {'vol':>7} {'filter':>12} {'void':>6}")
for r in opt.trace:
    print(f"{r['it']:>3} {r['n']:>6} {r['deep']:>6} {r['gphi']:>10.2e} "
          f"{r['gsdf']:>9.2e} {r['umax']:>9.2e} {r['dips']:>5} "
          f"{r['vol']:>7.4f} {r['flt']:>12} {r['void']:>6}")

assert opt.snap_pre is not None, (
    "no void formed in this run — raise HP_PF_LR or HP_ITERS.  Enclosed-cavity "
    f"sizes were {[(i, [v[0] for v in vs]) for i, vs in opt.n_comp]}")

pre, flt = opt.snap_pre["bw"], opt.snap_pre["flt"]
i_nuc = pre["it"]
assert flt is not None and flt["it"] == i_nuc, "thin filter did not run here"
assert flt["nodes"].shape == pre["nodes"].shape and \
    torch.allclose(flt["nodes"], pre["nodes"]), \
    "the mesh moved inside the nucleating iteration"

# The outer octree must tile the requested clip box exactly.  A centroid clip
# through mixed-depth cells used to leave alternating one-cell-wide holes at
# the box boundary; the figure's wireframe exposed those holes as a missing
# outer layer.  Hexes are axis-aligned, so summed bbox volumes are exact here.
_corners = pre["nodes"][pre["elements"].long()]
_cell_extents = _corners.amax(dim=1) - _corners.amin(dim=1)
_hex_volume = float(_cell_extents.prod(dim=1).sum())
_bbox_lo = torch.as_tensor(CLIP_BBOX[0], dtype=pre["nodes"].dtype)
_bbox_hi = torch.as_tensor(CLIP_BBOX[1], dtype=pre["nodes"].dtype)
_bbox_volume = float((_bbox_hi - _bbox_lo).prod())
_node_lo, _node_hi = pre["nodes"].amin(dim=0), pre["nodes"].amax(dim=0)
assert torch.allclose(_node_lo, _bbox_lo, atol=1.0e-7, rtol=0.0) and \
    torch.allclose(_node_hi, _bbox_hi, atol=1.0e-7, rtol=0.0), \
    f"outer octree bounds {_node_lo.tolist()} -> {_node_hi.tolist()} do not match {CLIP_BBOX}"
assert abs(_hex_volume - _bbox_volume) <= 1.0e-7 * _bbox_volume, \
    f"outer octree covers {_hex_volume:.9g}, expected full bbox {_bbox_volume:.9g}"
print(f"outer octree tiles clip bbox exactly: {_hex_volume:.6f} volume")

# ---------------------------------------------------------------------------
# What each panel shows, and the assertions that keep it honest.
# ---------------------------------------------------------------------------
sdf, gphi, gsdf = pre["sdf"], pre["gphi"], pre["gsdf"]
assert gphi is not None and gsdf is not None, "gradients were not captured"
hidden_a = pre["hidden"]            # (a): the hidden field the iteration began with
hidden_d = flt["phi_out"]           # (d): after the step and the thin filter
sdf_d = flt["sdf"]                  # (d): after the gradient step on d
deep = sdf < -BAND_K * eps
band = sdf.abs() < BAND_K * eps
elem = pre["elements"]

# (1) THE CLAIM: dJ/dd is negligible in the deep interior, dJ/dphi is not.
a_phi, a_sdf = gphi.abs(), gsdf.abs()
r_phi = float(a_phi[deep].mean() / a_phi.max())
r_sdf = float(a_sdf[deep].mean() / a_sdf.max())
print(f"\ndeep-interior mean/max:  dJ/dphi {r_phi:.4f}   dJ/dd {r_sdf:.2e}")
assert r_sdf < 0.01 * r_phi, f"dJ/dd is not interior-negligible ({r_sdf:.2e})"
assert float(a_sdf[deep].max()) == 0.0, "dJ/dd is not exactly zero deep inside"

# (2) The update: hidden phi dips in the interior, d does not move there at all.
dips = (hidden_d < 0.5) & deep
n_dips = int(dips.sum())
n_in = int(((flt["phi_in"] < 0.5) & deep).sum())
print(f"interior dips after the update: {n_dips} / {int(deep.sum())} deep "
      f"({n_in} before the thin filter, threshold 2 eps = {2 * eps:.4f})")
assert n_dips > 0, "no interior dip at the nucleating iteration"
assert int((_sdf_to_phi(sdf_d, eps) < 0.5).mul(deep).sum()) == 0, (
    "the VISIBLE phi already dips in the interior — d is not 'interior-unchanged'")
d_step = (sdf_d - sdf).abs()
print(f"|delta d| this iteration: {float(d_step.max()):.3e} overall, "
      f"{float(d_step[band].max()):.3e} in the band, "
      f"{float(d_step[deep].max()):.3e} deep interior")
assert float(d_step.max()) > 0.0, "d did not move at all — is gd_lr zero?"
assert float(d_step[deep].max()) == 0.0, (
    "the gradient step changed d in the deep interior; the figure's claim that "
    "shape optimisation cannot reach the interior does not hold here")

# (3) The production viability test, re-derived: some element is fully interior
#     and owns a dip, which is what makes the carve fire.
n_affected = int((deep[elem].all(dim=1) & (hidden_d[elem] < 0.5).any(dim=1)).sum())
print(f"viable elements (all 8 nodes interior AND one dipped): {n_affected}")
assert n_affected > 0, "no element passed the viability test"

# (4) The schedule did what it was set up to do.
vol0, vol_nuc = opt.trace[0]["vol"], opt.trace[i_nuc]["vol"]
assert opt.pf_on_iter is not None, "the hidden field never switched on"
assert i_nuc >= opt.pf_on_iter, (
    f"a void formed at iteration {i_nuc}, before the hidden field switched on "
    f"at {opt.pf_on_iter} — the schedule leaked")
print(f"phase 1: iterations 0-{opt.pf_on_iter - 1}; phase 2 from "
      f"{opt.pf_on_iter}, void at {i_nuc} "
      f"({i_nuc - opt.pf_on_iter} iterations later); volume "
      f"{vol0:.4f} -> {vol_nuc:.4f}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def write_hexes(path, nodes_t, elems_t, name, point_scalars):
    """Legacy ASCII VTK UNSTRUCTURED_GRID of hexes (sdf2octree's corner order
    already matches VTK_HEXAHEDRON, AGENTS.md §5)."""
    v = nodes_t.numpy().astype(np.float64)
    e = elems_t.numpy().astype(np.int64)
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\n{name}\nASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(v)} float\n")
        np.savetxt(f, v, fmt="%.6f")
        f.write(f"\nCELLS {len(e)} {len(e) * 9}\n")
        np.savetxt(f, np.hstack([np.full((len(e), 1), 8, np.int64), e]), fmt="%d")
        f.write(f"\nCELL_TYPES {len(e)}\n")
        np.savetxt(f, np.full(len(e), 12, np.int64), fmt="%d")
        f.write(f"\nPOINT_DATA {len(v)}\n")
        for nm, arr in point_scalars.items():
            f.write(f"SCALARS {nm} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, arr.numpy(), fmt="%.6f")
    print(f"wrote {path} ({len(e)} hexes, {len(v)} nodes, "
          f"{', '.join(point_scalars)})")


# Panel (c) compares SUPPORT, not size, so each gradient is scaled to its own
# maximum and the two share one bar, on a log ramp floored at four decades:
# |dJ/dd| spans three decades WITHIN its shell (the traction patch dominates),
# so on a linear ramp the shell falls in the palest step and the panel argues
# nothing.  The deep interior is EXACTLY zero and sits at the floor.
LOG_FLOOR = 4.0
MASK_SENTINEL = -1.0     # below the ramp -> the LUT's below-range colour

bcfix = pre["bcfix"].bool()
_frac = int((bcfix & deep).sum()) / max(int(deep.sum()), 1)
print(f"BC region (frozen SDF): {int(bcfix.sum())} / {int(bcfix.numel())} nodes; "
      f"{100 * _frac:.1f} % of the deep interior (the feet are solid)")
assert int(bcfix.sum()) > 0, "the BC freeze mask is empty"
assert _frac < 0.5, "the BC mask covers half the deep interior — (c) would be blank"
assert not bool((bcfix & dips).any()), (
    "the BC mask covers nodes where the hidden field dipped — the void region "
    "itself would be masked out of panel (c)")


def log_norm(a):
    """log10(|g| / max over the FREE nodes), floored at LOG_FLOOR decades.

    The BC region is masked OUT rather than zeroed: its SDF is frozen, so that
    sensitivity is never acted on, and zeroing would claim it IS zero, which is
    a different and false statement.  The masked value is a SENTINEL below the
    ramp, not NaN — the legacy ASCII VTK reader cannot parse `nan` and abandons
    the array with a warning instead of an error, so the panel would come out
    blank with nothing failing.
    """
    r = (a / a[~bcfix].max()).clamp_min(10.0 ** (-2.0 * LOG_FLOOR))
    out = (r.log10().clamp_min(-LOG_FLOOR) / LOG_FLOOR + 1.0).clamp(0.0, 1.0)
    out[bcfix] = MASK_SENTINEL
    return out


# (a), (c) and (d) are the SAME mesh and the same camera, so they share one file:
# sdf/hidden are the state the iteration began with, sdf_c/hidden_c the state its
# two updates left behind.
write_hexes("hp_pre.vtk", pre["nodes"], elem,
            f"table at iteration {i_nuc}: d, hidden phi, both gradients, "
            "and both updated fields",
            {"sdf": sdf,
             "hidden": hidden_a,
             "gphi": log_norm(a_phi),
             "gsdf": log_norm(a_sdf),
             "sdf_c": sdf_d,
             "hidden_c": hidden_d})
write_hexes("hp_solve.vtk", pre["inodes"], pre["ielements"],
            f"trimmed material mesh at iteration {i_nuc}: |u|",
            {"umag": pre["umag"]})

# Panel (e): the surface the remesh produced on the nucleating iteration — the
# first one that carries the void — with the restoration's own face labels.
_st = opt.snap_pre["stitch"]
assert _st is not None, "no stitch_bc_patches call was recorded for this iteration"
_sv, _sf, _sl = _st
np.savez("hp_bc_masks.npz", verts=_sv, faces=_sf, labels=_sl)
print(f"wrote hp_bc_masks.npz ({len(_sf)} faces; labels "
      f"{np.bincount(_sl.astype(np.int64), minlength=3)[:3].tolist()} "
      "= [free base, restored BC patch, zip bridge])")
assert int((_sl == 1).sum()) > 0, "the restoration reported no BC-patch faces"

for _it, _surf in sorted(opt.evo.items()):
    _surf.export(f"hp_evo_{_it}.stl")
    print(f"wrote hp_evo_{_it}.stl ({len(_surf.faces)} faces, "
          f"vol {_surf.volume:.4f})")

evo_its = sorted(opt.evo)
_trace = {r["it"]: r for r in opt.trace}
void_cells = [v[0] for v in opt.snap_pre["voids"]]
fin = opt.snap_final

with open("hp_stats.txt", "w") as f:
    f.write(f"h_min {h_min:.7f}\n")
    f.write(f"sdf_truncation {opt._sdf_truncation:.7f}\n")
    f.write(f"sdf_refinement_band {opt._sdf_refinement_band:.7f}\n")
    f.write(f"eps {eps:.7f}\n")
    f.write(f"depth {OCTREE_DEPTH}\n")
    f.write("clip_bbox_lo " + " ".join(f"{x:.7f}" for x in CLIP_BBOX[0]) + "\n")
    f.write("clip_bbox_hi " + " ".join(f"{x:.7f}" for x in CLIP_BBOX[1]) + "\n")
    f.write(f"outer_bbox_volume {_bbox_volume:.7f}\n")
    f.write(f"outer_hex_volume {_hex_volume:.7f}\n")
    f.write(f"band_k {BAND_K}\n")
    f.write(f"band_halfwidth {BAND_K * eps:.7f}\n")
    f.write(f"thin_threshold {2 * eps:.7f}\n")
    f.write(f"pf_phi_lr {PF_LR}\n")
    f.write(f"gd_lr {GD_LR}\n")
    f.write(f"iters {N_ITERS}\n")
    f.write(f"iter_capture {i_nuc}\n")
    f.write(f"iter_pf_on {opt.pf_on_iter}\n")
    f.write(f"iter_final {fin['it']}\n")
    f.write("evo_iters " + " ".join(str(i) for i in evo_its) + "\n")
    f.write("evo_vol " + " ".join(f"{_trace[i]['vol']:.4f}" for i in evo_its) + "\n")
    f.write(f"vol_initial {vol0:.6f}\n")
    f.write(f"vol_at_capture {vol_nuc:.6f}\n")
    f.write(f"vol_shrink_pct {100 * (1.0 - vol_nuc / vol0):.3f}\n")
    f.write(f"nodes {int(sdf.numel())}\n")
    f.write(f"elements {int(elem.shape[0])}\n")
    f.write(f"solve_nodes {int(pre['inodes'].shape[0])}\n")
    f.write(f"solve_elements {int(pre['ielements'].shape[0])}\n")
    f.write(f"umag_max {float(pre['umag'].max()):.6e}\n")
    f.write(f"deep_nodes {int(deep.sum())}\n")
    f.write(f"band_nodes {int(band.sum())}\n")
    f.write(f"gphi_deep_ratio {r_phi:.6f}\n")
    f.write(f"gsdf_deep_ratio {r_sdf:.3e}\n")
    f.write(f"gphi_max {float(a_phi.max()):.6e}\n")
    f.write(f"gsdf_max {float(a_sdf.max()):.6e}\n")
    f.write(f"grad_log_floor {LOG_FLOOR:.0f}\n")
    f.write(f"grad_mask_sentinel {MASK_SENTINEL}\n")
    f.write(f"bc_masked_nodes {int(bcfix.sum())}\n")
    f.write(f"d_step_max {float(d_step.max()):.6e}\n")
    f.write(f"d_step_deep_max {float(d_step[deep].max()):.6e}\n")
    f.write(f"dips_in {n_in}\n")
    f.write(f"dips_kept {n_dips}\n")
    f.write(f"affected_elements {n_affected}\n")
    f.write(f"n_voids {len(void_cells)}\n")
    f.write(f"void_cells {max(void_cells)}\n")
    f.write(f"bc_patch_faces {int((_sl == 1).sum())}\n")
    f.write(f"stitch_faces {len(_sf)}\n")
print("wrote hp_stats.txt")

with open("hp_traj.txt", "w") as f:
    f.write("# iter se_raw vol_raw\n")
    for it, se, vol in opt._traj:
        f.write(f"{it} {se:.8e} {vol:.8e}\n")
print(f"wrote hp_traj.txt ({len(opt._traj)} iterations)")
