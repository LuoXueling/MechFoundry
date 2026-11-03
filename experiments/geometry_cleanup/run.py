"""Run one object's clean-up through the WORKFLOW's own run-forward, with the early stop (GPU).

    PYTHONPATH=<checkout> XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
        python run.py <object> [--threshold H] [--step F] [--cap N] [--reinit on|off] [--tag TAG]

The run IS `workflow_tool.run_forward` on output/geometry_cleanup/<object>/case.json
(setup.py): the same case parsing, selector resolution, material, body force and
ShapeOptTrimmed parameters as the workflow's "Run FEA" with run_mode
"shape_optimization".  For this process `core.shape_opt.ShapeOptTrimmed` is replaced
by `CleanupOpt`, a subclass that changes these things and records everything else:

  * STEP.  The design variable s is a signed distance in metres and dJ/ds scales with
    the body, so one gd_lr cannot serve a 3 cm USB stick and a 200 m skyscraper.  At
    iteration 0, after the backward pass, the largest |dJ/ds| over the free nodes is
    read and the gradient is scaled, from then on by that same constant, so the first
    update's largest surface move is STEP_FRACTION leaves.  That is exactly the run
    gd_lr_eff = gd_lr * scale would be; gd_lr_eff is what results.json reports.
  * NO REINIT.  The case runs with reinit_sdf off (common.REINIT): the octree stays
    the one built around the starting surface and gradient descent moves the nodal
    SDF on it.  The solver's per-iteration reinit + octree rebuild (marching cubes ->
    exact distance -> new octree) is not idempotent on members a few leaves thick: on
    the school chair it alone, with a zero step, moved the surface 0.56 h (p99) in 8
    iterations and beaded the legs (README, "Trials").  With it off the solver never
    re-extracts its surface, so step() does (_cu_extract_surface) before each solve.
  * EARLY STOP.  Every step() solves a new shape; its marching-cubes surface (the one
    save() writes) is compared with iteration 0's by a symmetric surface distance
    (surface_distance below).  A shape whose distance exceeds THRESHOLD leaves, or
    whose maximum nearest-point distance exceeds MAX_THRESHOLD leaves, is REJECTED: its gradient is zeroed, the design is set back to the last accepted
    shape's SDF (saved before every update; the octree never changes, so it is a copy),
    and one more iteration solves that restored shape, which is the final one.  At the
    cap of MAX_UPDATES accepted updates the shape just solved is the final one.  Either
    way the last iteration makes no update and the loop's convergence check returns
    "converged": the loop ends cleanly on a solved shape, so its strain energy is
    known.  No iteration past that is computed.
  * OUTPUTS.  save() is kept for iteration 0 and the final iteration only (VTU, GLB);
    every iteration's surface is kept small (surfaces/it_<k>.npz).

Writes <OUT>/<object>[/<tag>]/: results.json, traj.tsv, surfaces/, solve/, run.log (the
caller's redirect).  Exit status 0 on a completed run (threshold or cap), 1 on failure;
results.json says which, and why.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--threshold", type=float, default=C.STOP_THRESHOLD)
ap.add_argument("--max-threshold", type=float, default=C.STOP_MAX)
ap.add_argument("--step", type=float, default=C.STEP_FRACTION)
ap.add_argument("--cap", type=int, default=C.MAX_UPDATES)
ap.add_argument("--percentile", type=float, default=C.STOP_PERCENTILE)
ap.add_argument("--tag", default="", help="subfolder for a trial variant")
ap.add_argument("--reinit", choices=["on", "off"], default=C.REINIT,
                help="the solver's per-iteration SDF reinit + octree rebuild")
ap.add_argument("--extra", default="", help="JSON solver parameters added to SOLVER_EXTRA")
A = ap.parse_args()

OD = os.path.join(C.obj_dir(A.name), A.tag) if A.tag else C.obj_dir(A.name)
SURF = os.path.join(OD, "surfaces")
os.makedirs(SURF, exist_ok=True)
SETUP = json.load(open(os.path.join(C.obj_dir(A.name), "setup.json")))
H = float(SETUP["h"])

sys.path.insert(0, C.CODE)
sys.path.insert(0, os.path.join(C.CODE, "src", "warp_wrapper"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import core.shape_opt as so_mod  # noqa: E402
from src.agents_wrapper.fem_workflow.scripts import workflow_tool as WT  # noqa: E402


# ---------------------------------------------------------------- surface distance
@wp.kernel
def _closest_dist(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_d: float,
                  out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], max_d)
    if q.result:
        out[i] = wp.length(pts[i] - wp.mesh_eval_position(mesh, q.face, q.u, q.v))
    else:
        out[i] = max_d


def _vertex_areas(v, f):
    a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    w = np.zeros(len(v))
    for k in range(3):
        np.add.at(w, f[:, k], a / 3.0)
    return w


def _wpct(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, q / 100.0 * c[-1]), len(x) - 1)])


class Surface:
    """A triangle surface with a BVH (warp) and per-vertex areas, for distances."""

    def __init__(self, v, f, device):
        self.v = np.asarray(v, np.float32)
        self.f = np.asarray(f, np.int32)
        self.w = _vertex_areas(self.v.astype(np.float64), self.f)
        self.pts = wp.array(self.v, dtype=wp.vec3, device=device)
        self.mesh = wp.Mesh(points=self.pts,
                            indices=wp.array(self.f.ravel(), dtype=int, device=device))
        self.diag = float(np.linalg.norm(self.v.max(0) - self.v.min(0)))

    def dist_from(self, other):
        """Distance from each vertex of `other` to this surface."""
        out = wp.zeros(len(other.v), dtype=float, device=self.pts.device)
        wp.launch(_closest_dist, dim=len(other.v),
                  inputs=[self.mesh.id, other.pts, 2.0 * max(self.diag, other.diag), out],
                  device=self.pts.device)
        return out.numpy().astype(np.float64)


def surface_distance(a, b, pct):
    """Symmetric surface distance between surfaces a and b.

    Both directed distance sets are taken -- every vertex of a to its nearest point on
    b, every vertex of b to its nearest point on a -- each weighted by its vertex's
    share of the surface area (marching-cubes vertices are not evenly spread), and
    the reported value is the larger of the two area-weighted `pct` percentiles.  A
    high percentile rather than the maximum (the Hausdorff distance) so that a speck
    the clean-up removes, a tiny fraction of the area that is "far" from everything,
    does not stop the run by itself; the maximum is recorded too.
    """
    dab, dba = b.dist_from(a), a.dist_from(b)
    return {
        "p": max(_wpct(dab, a.w, pct), _wpct(dba, b.w, pct)),
        "p95": max(_wpct(dab, a.w, 95), _wpct(dba, b.w, 95)),
        "mean": max(float((dab * a.w).sum() / a.w.sum()), float((dba * b.w).sum() / b.w.sum())),
        "max": max(float(dab.max()), float(dba.max())),
    }


# ---------------------------------------------------------------- the optimizer
_Base = so_mod.ShapeOptTrimmed
_made = []
# Solver parameters the workflow card cannot carry.  The Krylov cap: the school chair's
# 7th solve stopped at ||r||/||b|| 1.16e-6 against the 1e-6 tolerance after the default
# 1000 iterations, still converging (jax_amg_solver's own advice: raise max_iters).
SOLVER_EXTRA = {"jaxamg_max_iters": 4000}
if A.extra:
    SOLVER_EXTRA.update(json.loads(A.extra))


class CleanupOpt(_Base):
    def __init__(self, config, override_params=None):
        _made.append(self)
        self._cu_it = -1
        self._cu_rows = []            # one dict per evaluated shape
        self._cu_stop = None          # "threshold" | "cap" once decided
        # run -> (reject -> restored) -> final; see the module docstring
        self._cu_state = "run"
        self._cu_accepted = 0         # iteration index of the last accepted shape
        self._cu_prev = None          # design SDF of the last accepted shape
        self._cu_reject_by = None     # "p" (percentile) | "max"
        self._cu_scale = None
        self._cu_s0 = None
        self._cu_t0 = time.perf_counter()
        p = dict(override_params or {})
        p.update(SOLVER_EXTRA)
        super().__init__(config, override_params=p)
        self.log(f"geometry_cleanup: {A.name}, h {H:.4e}, step {A.step} h, stop at "
                 f"p{A.percentile:g} > {A.threshold} h or {A.cap} updates")

    # --- the shape solved this iteration -------------------------------------
    def _cu_extract_surface(self):
        """Marching cubes of the CURRENT design SDF (no reinit, no octree rebuild).

        With reinit_sdf off the solver never rebuilds its surface, so the cached one
        (what save() writes, what step() snapshots) would stay the starting surface.
        This is the extraction half of _post_optimizer_step_remesh_impl: the octree
        nodal SDF scattered onto the background lattice, then the solver's own MC.
        """
        from core.remeshing import scatter_octree_to_cube_grid
        with torch.no_grad():
            for k in ("_mc_wp_mesh", "_mc_wp_negate", "_mc_wp_keep_alive",
                      "_mc_verts_norm", "_mc_faces", "_mc_bbox_diag_phys"):
                self.__dict__.pop(k, None)
            grid = scatter_octree_to_cube_grid(
                self._prev_octree_nodes, wp.to_torch(self.e.dof_values).detach(),
                self._sussman_grid_sdf, self._remesh_axis_scale, self._remesh_geo_min,
                device=str(self.device))
            self._build_mc_surface_and_bvh(grid)

    def step(self, tape=None):
        if tape is not None and not self._reinit_sdf:
            self._cu_extract_surface()
        r = super().step(tape)
        if tape is None:              # only the optimize loop's taped solves count
            return r
        self._cu_it += 1
        it = self._cu_it
        v = self._normalized_to_physical(self._save_mc_verts_norm.to(torch.float64))
        v = v.detach().cpu().numpy()
        f = self._save_mc_faces.detach().cpu().numpy()
        np.savez_compressed(os.path.join(SURF, f"it_{it}.npz"),
                            v=v.astype(np.float32), f=f.astype(np.int32))
        s = Surface(v, f, str(self.device))
        if self._cu_s0 is None:
            self._cu_s0 = s
            d = {"p": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
        else:
            d = surface_distance(self._cu_s0, s, A.percentile)
        self._cu_rows.append({"it": it, **{f"d_{k}_h": d[k] / H for k in d},
                              "faces": int(len(f)), "t": time.perf_counter() - self._cu_t0})
        row = self._cu_rows[-1]
        if self._cu_state == "restored":
            # the solve of the restored last accepted shape: the final one
            row["role"] = f"restored it {self._cu_accepted}"
            self._cu_state, self._cu_stop = "final", (
                "threshold" if self._cu_reject_by == "p" else "max")
        elif d["p"] > A.threshold * H or d["max"] > A.max_threshold * H:
            row["role"] = "rejected"
            self._cu_state = "reject"
            self._cu_reject_by = "p" if d["p"] > A.threshold * H else "max"
        else:
            row["role"] = "accepted"
            self._cu_accepted = it
            if it >= A.cap:
                self._cu_state, self._cu_stop = "final", "cap"
        print(f"[cleanup] {A.name} it {it}: d_p{A.percentile:g} {d['p'] / H:.3f} h "
              f"(p95 {d['p95'] / H:.3f}, max {d['max'] / H:.3f}) {row['role']}"
              + (f" -> STOP ({self._cu_stop})" if self._cu_stop else ""), flush=True)
        return r

    def _objective_function(self, tape):
        r = super()._objective_function(tape)
        row = self._cu_rows[-1] if self._cu_rows else None
        if row is not None and "se" not in row:
            row["se"] = float(self._last_se_raw)
            row["vol"] = float(self._last_vol_raw)
        return r

    # --- step size, early stop -------------------------------------------------
    def _post_tape_backward(self):
        super()._post_tape_backward()
        g = wp.to_torch(self.e.dof_values.grad)
        if self._cu_state in ("final", "reject"):     # no gradient update
            g.zero_()
            return
        self._cu_prev = wp.to_torch(self.e.dof_values).detach().clone()
        if self._cu_scale is None:
            free = wp.to_torch(self._e_free_mask).reshape(g.shape[0], -1)[:, 0] > 0
            a = g.reshape(g.shape[0], -1)[free].abs().double().flatten()
            lr = float(self.params.get("gd_lr"))
            gmax = float(a.max()) if a.numel() else 0.0
            if gmax <= 0:
                raise RuntimeError("zero design gradient at iteration 0")
            self._cu_scale = A.step * H / (lr * gmax)
            nz = a[a > 0]
            q = torch.quantile(nz[torch.randperm(nz.numel(), device=nz.device)[:1000000]],
                               torch.tensor([0.5, 0.9, 0.99], dtype=nz.dtype, device=nz.device))
            self._cu_step_info = {
                "gd_lr_case": lr, "grad_scale": self._cu_scale,
                "gd_lr_eff": lr * self._cu_scale, "max_abs_grad": gmax,
                "first_move_h": {"max": A.step, **{f"p{k}": float(x) * lr * self._cu_scale / H
                                                   for k, x in zip((50, 90, 99), q)}},
            }
            self.log(f"geometry_cleanup: gd_lr_eff {lr * self._cu_scale:.4e} "
                     f"(first move: max {A.step} h, p50/p90/p99 of nonzero "
                     f"{self._cu_step_info['first_move_h']})")
        g.mul_(self._cu_scale)

    def _post_optimizer_step(self, iteration):
        if self._cu_state == "final":          # nothing changed: no remesh, no reinit
            return
        if self._cu_state == "reject":         # back to the last accepted design
            with torch.no_grad():
                wp.to_torch(self.e.dof_values).copy_(self._cu_prev)
            self._cu_state = "restored"
            self.log(f"  geometry_cleanup: shape {iteration} over the threshold, "
                     f"restored shape {self._cu_accepted}")
            return
        return super()._post_optimizer_step(iteration)

    def _check_convergence(self, it, obj_history, best_obj_iter, rel_tol, constraint_str="none"):
        verdict = super()._check_convergence(it, obj_history, best_obj_iter, rel_tol,
                                             constraint_str)
        if self._cu_stop:
            self.log(f"  geometry_cleanup: stopped ({self._cu_stop}) at iteration {it}")
            return "converged"
        return "continue" if verdict == "converged" else verdict

    def save(self, **kwargs):
        if self._cu_it <= 0 or self._cu_stop:
            return super().save(**kwargs)
        self.__dict__.pop("_last_e_grad", None)
        self.__dict__.pop("_last_e_phi_grad", None)


so_mod.ShapeOptTrimmed = CleanupOpt   # run_forward imports it at call time


def write_results(err=None):
    p = _made[0] if _made else None
    rows = p._cu_rows if p is not None else []
    with open(os.path.join(OD, "traj.tsv"), "w") as f:
        keys = ["it", "role", "d_p_h", "d_p95_h", "d_mean_h", "d_max_h", "se", "vol", "faces", "t"]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    se0 = rows[0].get("se") if rows else None
    v0 = rows[0].get("vol") if rows else None
    last = rows[-1] if rows else {}
    res = {
        "object": A.name, "tag": A.tag, "h": H,
        "threshold_h": A.threshold, "max_threshold_h": A.max_threshold,
        "percentile": A.percentile, "step_h": A.step, "cap": A.cap,
        "reinit": A.reinit,
        "status": "failed" if err else "ok",
        "stop_reason": (p._cu_stop if p is not None else None) if not err else "error",
        "error": err or "",
        # design updates in the final shape (a rejected one and its restore excluded)
        "updates": p._cu_accepted if p is not None else 0,
        "final_surface": f"surfaces/it_{last['it']}.npz" if rows else None,
        "rejected": [r["it"] for r in rows if r.get("role") == "rejected"],
        "final_d_h": last.get("d_p_h"), "final_d_p95_h": last.get("d_p95_h"),
        "final_d_max_h": last.get("d_max_h"),
        "se_ratio": (last["se"] / se0) if se0 and "se" in last else None,
        "vol_ratio": (last["vol"] / v0) if v0 and "vol" in last else None,
        "standin": SETUP.get("standin", False), "solver_extra": SOLVER_EXTRA,
        "step": getattr(p, "_cu_step_info", None) if p is not None else None,
        "seconds": time.perf_counter() - T0,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(OD, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("RESULTS", json.dumps(res))


T0 = time.perf_counter()
case = json.load(open(C.case_path(A.name)))
cpath = C.case_path(A.name)
if A.tag or A.cap != C.MAX_UPDATES or A.reinit != C.REINIT:   # a variant: own case, own solve/
    case["output"] = os.path.join(OD, "solve")
    case["shape_opt"]["max_optimize_iters"] = A.cap + 2
    case["shape_opt"]["reinit_sdf"] = A.reinit == "on"
    cpath = os.path.join(OD, "case_variant.json")
    with open(cpath, "w") as f:
        json.dump(case, f, indent=2)
err = None
try:
    WT.run_forward(SimpleNamespace(case=Path(cpath)))
except Exception as exc:                                           # noqa: BLE001
    traceback.print_exc()
    err = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:600]
    print("!! run stopped:", err, flush=True)
write_results(err)
sys.exit(1 if err else 0)
