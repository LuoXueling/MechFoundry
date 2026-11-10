"""Stage 0 (env mechfoundry-warp117, GPU, cached): the park bench's geometry clean-up, re-run
with the solver's output kept at EVERY iteration, and a per-part forward solve of its
final shape.

    python dv_cleanup.py            # runs only what its cache lacks
    DV_CLEANUP_FORCE=1 python dv_cleanup.py

The demo video's optimisation beat plays the 41 shapes of the park bench's clean-up
(experiments/geometry_cleanup, archived in output/archives/geometry_cleanup.tar.zst) and
plots their strain energy.  The archived run kept the solver's output for iterations 0
and 40 only, so the clean-up is RE-RUN here, unchanged (its own setup.py and run.py, same
step 0.5 h, cap 40, reinit off, uniform stand-in material E 57.2 GPa, same stop rule),
with GEOMETRY_CLEANUP_SAVE_EVERY=1 (run.py's opt-in: save() at every iteration) and
GEOMETRY_CLEANUP_OUT pointing at its own folder, so the archived extract is never touched:

  CACHE/park-bench/              setup.py + run.py output (results.json, traj.tsv,
                                 surfaces/it_<k>.npz, solve/api_fem_forward-*/vtu_surface/
                                 surface_t00kk.vtu, run.log), ~11 min on one A6000
  CACHE/park-bench/parts_final/  the final shape (it_40) solved FORWARD with the session's
                                 own per-part materials (steel 120 GPa / wood 12 GPa field),
                                 for the shrink and the finale tile: case.json, surface.ply,
                                 segments.npz (labels moved over from the session's surface),
                                 solve/, coverage.json, run.log (~1 min)
  CACHE/dv_cleanup.json          the gate's record

The per-part material field (materials/parts/field, 64^3 over the original bbox's longest
side) is sampled at nodes by nearest cell; its unoccupied cells hold the nearest occupied
cell's values (vomp_wrapper.artifact.dense_grid_from_sparse) and points outside its box
clamp to the edge cells, so material the clean-up added outside the original body takes
the nearest part's E, nu, rho -- never a void or default value.  coverage.json records
how much of the final shape that is.

Gate: the re-run's traj.tsv must reproduce the ARCHIVED one (output/geometry_cleanup/
park-bench/traj.tsv): the same iterations and roles, the p99 distance within 0.05 h and
the maximum within 0.3 h at every iteration (the shapes), strain energy within 3 % (the
numbers).  A larger drift stops the build: the movie must not show a different
optimisation than the paper's.  Measured 2026-11-07: p99 within 0.012 h, max within
0.22 h, SE +1.6 % at iteration 0 drifting to -2.25 % at 40 (SE/SE0 0.575 against the
archive's 0.598).  The SE offset is a code change, not noise: commit 964ca7c (2026-10-15,
the day after the archived run) resolves saved-segment BC selectors with the solver's BC
tolerance (0.75 h, was h), so the BC band the optimizer freezes shrank (38 933 -> 33 991
background cells) and the same starting shape solves 1.6 % softer.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GC = os.path.join(REPO, "experiments", "geometry_cleanup")
OBJ = "park-bench"
ARCHIVED = os.path.join(REPO, "output", "geometry_cleanup", OBJ)       # the archive's extract
CACHE = os.environ.get("DV_CLEANUP_ROOT", os.path.join(REPO, "output", "geometry_cleanup_demovideo"))
RUN = os.path.join(CACHE, OBJ)
PARTS = os.path.join(RUN, "parts_final")
FORCE = os.environ.get("DV_CLEANUP_FORCE", "0") == "1"
PY = sys.executable
TOL_SE_REL, TOL_D_H, TOL_DMAX_H = 0.03, 0.05, 0.3


# -- surface queries on the GPU (warp BVH, as the clean-up's run.py measures distances;
# trimesh's proximity/ray/contains need rtree, which mechfoundry-warp117 lacks) ----------
def _wp():
    import warp as wp
    wp.config.quiet = True
    wp.init()
    return wp


_KERNELS = {}


def _kernels():
    if not _KERNELS:
        wp = _wp()

        @wp.kernel
        def closest(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_d: float,
                    face: wp.array(dtype=int), dist: wp.array(dtype=float),
                    sign: wp.array(dtype=float), near: wp.array(dtype=wp.vec3)):
            i = wp.tid()
            q = wp.mesh_query_point(mesh, pts[i], max_d)
            if q.result:
                face[i] = q.face
                near[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
                dist[i] = wp.length(pts[i] - near[i])
                sign[i] = q.sign
            else:
                face[i] = -1
                near[i] = pts[i]
                dist[i] = max_d
                sign[i] = 1.0

        @wp.kernel
        def ray(mesh: wp.uint64, orig: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
                max_t: float, t: wp.array(dtype=float)):
            i = wp.tid()
            q = wp.mesh_query_ray(mesh, orig[i], dirs[i], max_t)
            if q.result:
                t[i] = q.t
            else:
                t[i] = max_t
        _KERNELS.update(closest=closest, ray=ray)
    return _KERNELS


class BVH:
    """A closed triangle surface on the GPU: nearest face, distance and inside sign of
    query points; first hit along rays."""

    def __init__(self, V, F):
        wp = _wp()
        self.wp = wp
        self.dev = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.pts = wp.array(np.asarray(V, np.float32), dtype=wp.vec3, device=self.dev)
        self.mesh = wp.Mesh(points=self.pts, indices=wp.array(np.asarray(F, np.int32).ravel(),
                                                                dtype=int, device=self.dev))
        self.diag = float(np.linalg.norm(np.ptp(np.asarray(V, float), axis=0)))

    def closest(self, P, points=False):
        """(nearest face, distance, sign: < 0 inside) of each point; with points=True
        also the nearest surface point."""
        wp, n = self.wp, len(P)
        out = [wp.zeros(n, dtype=d, device=self.dev) for d in (int, float, float, wp.vec3)]
        wp.launch(_kernels()["closest"], dim=n, device=self.dev,
                  inputs=[self.mesh.id, wp.array(np.asarray(P, np.float32), dtype=wp.vec3, device=self.dev),
                          4.0 * self.diag, *out])
        face, dist, sign, near = (a.numpy() for a in out)
        r = (face, dist.astype(np.float64), sign)
        return r + (near.astype(np.float64),) if points else r

    def inside(self, P):
        return self.closest(P)[2] < 0

    def first_hit(self, orig, dirs, max_t):
        wp, n = self.wp, len(orig)
        t = wp.zeros(n, dtype=float, device=self.dev)
        wp.launch(_kernels()["ray"], dim=n, device=self.dev,
                  inputs=[self.mesh.id, wp.array(np.asarray(orig, np.float32), dtype=wp.vec3, device=self.dev),
                          wp.array(np.asarray(dirs, np.float32), dtype=wp.vec3, device=self.dev),
                          float(max_t), t])
        return t.numpy().astype(np.float64)


def read_traj(path):
    rows = [ln.rstrip("\n").split("\t") for ln in open(path)]
    keys = rows[0]
    return [dict(zip(keys, r)) for r in rows[1:]]


def env(**kw):
    e = dict(os.environ, GEOMETRY_CLEANUP_OUT=CACHE, XLA_PYTHON_CLIENT_PREALLOCATE="false",
             PYTHONPATH=REPO)
    e.update(kw)
    return e


def run_cleanup():
    """setup.py + run.py of experiments/geometry_cleanup, saving every iteration."""
    if os.path.isdir(RUN):
        shutil.rmtree(RUN)
    os.makedirs(CACHE, exist_ok=True)
    subprocess.run([PY, "setup.py", OBJ], cwd=GC, env=env(), check=True)
    with open(os.path.join(RUN, "run.log"), "w") as log:
        subprocess.run([PY, "run.py", OBJ], cwd=GC, env=env(GEOMETRY_CLEANUP_SAVE_EVERY="1"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def surface_vtus():
    """The re-run's surface_t0000..t00NN.vtu, one per iteration (it_k <-> t000k)."""
    runs = sorted(d for d in os.listdir(os.path.join(RUN, "solve")) if d.startswith("api_fem_forward-"))
    assert len(runs) == 1, runs
    sd = os.path.join(RUN, "solve", runs[0], "vtu_surface")
    return [os.path.join(sd, f) for f in sorted(os.listdir(sd)) if f.endswith(".vtu")]


def gate():
    """The re-run against the archived run, iteration by iteration."""
    new, old = read_traj(os.path.join(RUN, "traj.tsv")), read_traj(os.path.join(ARCHIVED, "traj.tsv"))
    assert [r["it"] for r in new] == [r["it"] for r in old], "iterations differ"
    assert [r["role"] for r in new] == [r["role"] for r in old], "accept/reject roles differ"
    dse = [abs(float(a["se"]) / float(b["se"]) - 1) for a, b in zip(new, old)]
    dd = [abs(float(a["d_p_h"]) - float(b["d_p_h"])) for a, b in zip(new, old)]
    dmax = [abs(float(a["d_max_h"]) - float(b["d_max_h"])) for a, b in zip(new, old)]
    res_new = json.load(open(os.path.join(RUN, "results.json")))
    res_old = json.load(open(os.path.join(ARCHIVED, "results.json")))
    rec = {"archived": os.path.relpath(ARCHIVED, REPO), "rerun": os.path.relpath(RUN, REPO),
           "iterations": len(new), "max_se_rel_diff": max(dse), "max_d_p99_diff_h": max(dd),
           "max_d_max_diff_h": max(dmax),
           "se_ratio": [res_old["se_ratio"], res_new["se_ratio"]],
           "final_d_p99_h": [res_old["final_d_h"], res_new["final_d_h"]],
           "final_d_max_h": [res_old["final_d_max_h"], res_new["final_d_max_h"]],
           "vol_ratio": [res_old["vol_ratio"], res_new["vol_ratio"]],
           "gd_lr_eff": [res_old["step"]["gd_lr_eff"], res_new["step"]["gd_lr_eff"]],
           "stop_reason": res_new["stop_reason"], "updates": res_new["updates"],
           "seconds": res_new["seconds"], "tolerance": {"se_rel": TOL_SE_REL, "d_p99_h": TOL_D_H, "d_max_h": TOL_DMAX_H},
           "se_rel_diff_it0_it40": [float(new[0]["se"]) / float(old[0]["se"]) - 1,
                                    float(new[-1]["se"]) / float(old[-1]["se"]) - 1],
           "why_se_differs": "commit 964ca7c (2026-10-15): saved-segment BC selectors resolved "
                             "with the solver's BC tolerance (0.75 h, was h); frozen BC cells "
                             "38933 (archive) -> 33991 (re-run)"}
    print("gate:", json.dumps(rec))
    assert max(dse) < TOL_SE_REL and max(dd) < TOL_D_H and max(dmax) < TOL_DMAX_H, \
        f"the re-run drifted from the archive: {rec}"
    assert len(surface_vtus()) == len(new), "not every iteration was saved"
    return rec


def session_dir():
    return json.load(open(os.path.join(RUN, "setup.json")))["session"]


def run_parts():
    """it_40 solved forward with the session's per-part material field."""
    import trimesh
    sess = session_dir()
    if os.path.isdir(PARTS):
        shutil.rmtree(PARTS)
    os.makedirs(PARTS)
    res = json.load(open(os.path.join(RUN, "results.json")))
    fin = np.load(os.path.join(RUN, res["final_surface"]))
    V1, F1 = fin["v"].astype(np.float64), fin["f"].astype(np.int64)
    m1 = trimesh.Trimesh(V1, F1, process=False)
    assert m1.is_watertight, "final surface not watertight"
    m1.export(os.path.join(PARTS, "surface.ply"))
    # the session's segment labels (per face of its physical.glb), moved to the new faces:
    # each new face takes the label of the original face nearest its centroid
    m0 = trimesh.load(os.path.join(sess, "geometry", "physical.glb"), force="mesh", process=False)
    lab0 = np.load(os.path.join(sess, "boundary_conditions", "segments.npz"))["segment_ids"]
    tri, _, _ = BVH(m0.vertices, m0.faces).closest(m1.triangles_center)
    assert (tri >= 0).all()
    np.savez(os.path.join(PARTS, "segments.npz"), segment_ids=np.asarray(lab0)[tri])
    case = json.load(open(os.path.join(sess, "case", "case.json")))
    case["surface"] = os.path.join(PARTS, "surface.ply")
    case["segments_npz"] = os.path.join(PARTS, "segments.npz")
    art = os.path.join(sess, "materials", "parts", "field")
    case["material"]["artifact"] = art
    case["output"] = os.path.join(PARTS, "solve")
    # the session's own solve did the same: JAX-AMG stalled at its rounding floor
    # (solve/failed.json) and the retry on pyamgx solved; this shape stalls likewise
    case["shape_opt"]["linear_solver"] = "pyamgx"
    assert case["material"]["mode"] == "parts" and case["shape_opt"]["run_mode"] == "forward"
    with open(os.path.join(PARTS, "case.json"), "w") as f:
        json.dump(case, f, indent=2)
    json.dump(coverage(m0, m1, art), open(os.path.join(PARTS, "coverage.json"), "w"), indent=1)
    code = ("from pathlib import Path; from types import SimpleNamespace; "
            "from src.agents_wrapper.fem_workflow.scripts import workflow_tool as WT; "
            f"WT.run_forward(SimpleNamespace(case=Path({os.path.join(PARTS, 'case.json')!r})))")
    with open(os.path.join(PARTS, "run.log"), "w") as log:
        subprocess.run([PY, "-c", code], cwd=REPO, env=env(PYTHONPATH=REPO + os.pathsep +
                       os.path.join(REPO, "src", "warp_wrapper")),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def coverage(m0, m1, art):
    """How much of the final shape lies outside the cells the original body occupied in
    the part field (those take the nearest occupied cell's part), by surface area and by
    volume (a 4 mm lattice, inside tests by warp's mesh_query_point sign); and how much
    of its volume is new material (outside the original body), by the E it gets."""
    man = json.load(open(os.path.join(art, "manifest.json")))
    valid = np.load(os.path.join(art, "valid_mask.npy"))
    grid = np.load(os.path.join(art, "material_grid.npy"))
    lo, hi = (np.asarray(b, float) for b in man["bounds"])

    def cells(P):
        raw = np.floor((P - lo) / (hi - lo) * 64.0).astype(np.int64)
        return np.clip(raw, 0, 63), np.any((raw < 0) | (raw > 63), axis=1)

    def stats(P, w):
        idx, out = cells(P)
        inval = ~valid[tuple(idx.T)]
        return float(w[inval].sum() / w.sum()), float(w[out].sum() / w.sum())

    area = m1.area_faces
    s_unocc, s_out = stats(m1.triangles_center, area)
    s0_unocc, _ = stats(m0.triangles_center, m0.area_faces)
    step = 0.004
    blo, bhi = m1.bounds
    axes = [np.arange(blo[k] + step / 2, bhi[k], step) for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    Pin = P[BVH(m1.vertices, m1.faces).inside(P)]
    new = ~BVH(m0.vertices, m0.faces).inside(Pin)
    v_unocc, v_out = stats(Pin, np.ones(len(Pin)))
    idx, _ = cells(Pin[new])
    E = grid[tuple(idx.T)][:, 0]
    return {"field_bounds": man["bounds"], "lattice_m": step,
            "surface_area_fraction_in_unoccupied_cells": s_unocc,
            "surface_area_fraction_in_unoccupied_cells_original_shape": s0_unocc,
            "surface_area_fraction_outside_field_box": s_out,
            "volume_samples": int(len(Pin)),
            "volume_fraction_in_unoccupied_cells": v_unocc,
            "volume_fraction_outside_field_box": v_out,
            "volume_fraction_new_material": float(new.mean()),
            "new_material_E_GPa_fractions": {f"{e / 1e9:g}": float(n / max(1, new.sum())) for e, n in
                                             zip(*np.unique(np.round(E, -8), return_counts=True))},
            "handling": "unoccupied cells hold the nearest occupied cell's values "
                        "(dense_grid_from_sparse); points outside the box clamp to the edge cells"}


def parts_vtu():
    """The per-part solve's volume VTU, or None."""
    import glob
    hits = sorted(glob.glob(os.path.join(PARTS, "solve", "api_fem_forward-*", "vtu", "solution_t*.vtu")))
    return hits[-1] if hits else None


def main():
    t0 = time.time()
    if FORCE or not os.path.exists(os.path.join(RUN, "results.json")):
        run_cleanup()
    rec = gate()
    if FORCE or parts_vtu() is None:
        run_parts()
    rec["parts_final"] = {"dir": os.path.relpath(PARTS, REPO), "vtu": os.path.relpath(parts_vtu(), REPO),
                          "coverage": json.load(open(os.path.join(PARTS, "coverage.json")))}
    json.dump(rec, open(os.path.join(CACHE, "dv_cleanup.json"), "w"), indent=1)
    print(f"dv_cleanup: {time.time() - t0:.0f} s; record in {CACHE}/dv_cleanup.json")


if __name__ == "__main__":
    main()
