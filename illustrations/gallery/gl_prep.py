"""Stage 1 (env mechfoundry-warp117, small GPU use): the 50 agentic runs -> one record each.

    python gl_prep.py --batch-root output/agentic_batch50_opus --work _work

Every object is drawn as the agentic-results figure's LAST column draws it, and that
column's own code computes what is drawn (imported, not copied):

  surface        <session>/geometry/physical.glb       (ar_prep.load_mesh)
  displacement   the solve ar_prep.chosen_run picks, its last solution_t*.vtu, moved to
                 the surface by the production octree_hex_interpolate
                 (ar_prep._prep_solve, called as is: the same >95 % inside check)
  x N            ar_prep._prep_solve's rule (1/2/3/5 x 10^k nearest to 0.12 L / |u|max,
                 standing bodies stepped down to a 2 % dip); then GATED: it must equal
                 the object's entry in its category figure's record
                 ../agentic_results/ar_stats_agentic_results_<slug>_opus.json
                 (factor exactly, |u|max to 1e-6), or the build stops
  heading        that record's chosen camera heading (view.dir), so the gallery shows
                 the side of each object its category figure shows; fallback rule in
                 heading_of()

Session folders are only read.  Writes, per object k (size order is the layout's job):
  <work>/obj_<k>.npz     V (undeformed, metres, rotated to the heading, floor y = 0,
                         footprint centred in x, front edge at z = 0), F, u (same frame),
                         factor, u_rel = |u| / |u|max
  <work>/gl_objects.json the record (instruction, category, session, L, N, |u|max,
                         heading source, gate results)
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "agentic_results")
sys.path.insert(0, AR)
import ar_prep  # noqa: E402  (imports ar_figure before trimesh, as it must)
import ar_categories as C  # noqa: E402
import ar_runs  # noqa: E402

ELEV = 45.0                    # the gallery camera's elevation (orthographic, azimuth 0)
SIL_FALLBACK = 0.6             # the category heading is kept unless its silhouette at the
#                                gallery elevation is under this share of the best heading's
AXIS_MIN_DEG = ar_prep.AXIS_MIN_DEG     # choose_view filter 2, reused for the fallback


def category_records():
    """instruction -> the run's record in its Opus category figure's ar_stats."""
    out = {}
    for f in sorted(glob.glob(os.path.join(AR, "ar_stats_agentic_results_*_opus.json"))):
        for r in json.load(open(f))["runs"]:
            out[r["meta"]["instruction"]] = dict(r, _file=os.path.basename(f))
    return out


def bc_regions(session, mesh):
    """The Dirichlet/Neumann regions with their area-weighted mean normals, as prep_run
    computes them: all _prep_solve needs to tell a body standing on its support."""
    rz, rj = (os.path.join(session, "boundary_conditions", n) for n in ("resolved.npz", "resolved.json"))
    if not (os.path.exists(rz) and os.path.exists(rj)):
        return []
    masks = np.load(rz)
    regions = json.load(open(rj)).get("regions", [])
    cat = {r["key"]: r["category"] for r in regions if "key" in r}
    out = []
    for key in masks.files:
        c = cat.get(key, key.split("_")[0])
        m = np.asarray(masks[key], bool)
        fn = mesh.face_normals[m] * mesh.area_faces[m][:, None]
        n = fn.sum(0)
        n = (n / np.linalg.norm(n)).round(3).tolist() if np.linalg.norm(n) > 0 else None
        out.append({"key": key, "category": c, "mean_normal": n, "faces": int(m.sum())})
    return out


def heading_deg(view_dir):
    """The horizontal heading (deg) of a view direction (object -> camera), atan2(x, z)."""
    return math.degrees(math.atan2(view_dir[0], view_dir[2]))


def view_at(heading, elev=ELEV):
    a, e = math.radians(heading), math.radians(elev)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def heading_of(mesh, cat_heading):
    """The category figure's heading, unless its silhouette seen from the gallery's 45 deg
    is under SIL_FALLBACK of the best of 24 headings (every 15 deg, AXIS_MIN_DEG off the
    body's principal horizontal axes, choose_view's filter 2); then that best one."""
    L = float(np.ptp(mesh.vertices, axis=0).max())
    centre = mesh.bounds.mean(0)
    R = 0.5 * float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))) * 1.02
    sil = lambda h: ar_prep.silhouette_area(mesh, view_at(h), L, centre, R)[0]  # noqa: E731
    axes = ar_prep.principal_headings(mesh)

    def off_axes(h):
        return min(abs(((h - a + 90.0) % 180.0) - 90.0) for a in axes) >= AXIS_MIN_DEG - 1e-6
    cands = [h for h in range(0, 360, 15) if off_axes(h)]
    areas = {h: sil(h) for h in cands}
    a_cat = sil(cat_heading)
    best = max(areas, key=areas.get)
    ratio = a_cat / areas[best] if areas[best] > 0 else 1.0
    if ratio >= SIL_FALLBACK:
        return cat_heading, {"source": "category figure", "silhouette_vs_best": round(ratio, 3)}
    return float(best), {"source": "silhouette fallback", "silhouette_vs_best": round(ratio, 3),
                         "category_heading": cat_heading}


def rot_y(deg):
    """Rotation about +y taking a heading a to a + deg (x' = x cos + z sin)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch-root", required=True)
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    t0 = time.time()

    sys.path.insert(0, os.path.join(ar_prep.REPO, "src", "warp_wrapper"))
    sys.path.insert(0, ar_prep.REPO)
    import torch
    from core.remeshing import octree_hex_interpolate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch, octree_hex_interpolate, device)

    latest = C.latest_runs(args.batch_root)
    cats = category_records()
    objs, missing = [], []
    for k, (instr, catname) in enumerate(C.objects()):
        path = latest.get(instr)
        if not path:
            missing.append(instr)
            continue
        run = ar_runs.load_run(path)
        s = run.session
        phys = os.path.join(s, "geometry", "physical.glb")
        if not os.path.exists(phys):
            missing.append(instr)
            continue
        mesh = ar_prep.load_mesh(phys)
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
        L = float(np.ptp(V, axis=0).max())
        tf = os.path.join(s, "geometry", "physical.glb.transform.json")
        mat_p = os.path.join(s, "case", "material.json")
        mat = json.load(open(mat_p)) if os.path.exists(mat_p) else {}
        units = (json.load(open(tf)).get("units") if os.path.exists(tf) else None) or mat.get("unit_system")
        rec = {"notes": [], "unit_system": units, "bc": {"regions": bc_regions(s, mesh)}}
        out = {}
        ar_prep._prep_solve(rec, out, run, s, V, L, mat, ctx)
        m_per = rec["m_per_length_unit"]
        cr = cats.get(instr)
        assert cr is not None, f"{instr!r}: no Opus category record to gate against"
        assert os.path.normpath(cr["session"]) == os.path.normpath(s), \
            f"{instr!r}: the category figure used {cr['session']}, the latest run is {s}"
        gate = {"record": cr["_file"]}
        if out.get("u") is not None:
            f_, f_cat = rec["scale"]["factor"], cr["scale"]["factor"]
            um, um_cat = rec["solve"]["umax_surface"], cr["solve"]["umax_surface"]
            assert f_ == f_cat, f"{instr!r}: x{f_:g} here, x{f_cat:g} in {cr['_file']}"
            assert abs(um / um_cat - 1) < 1e-6, f"{instr!r}: |u|max {um:.6g} vs {um_cat:.6g}"
            gate.update(factor_equal=True, umax_rel_diff=abs(um / um_cat - 1))
            u = out["u"]
            factor = f_
        else:
            u, factor = np.zeros_like(V), 1.0
            gate["not_solved"] = True
        cat_h = heading_deg(cr["view"]["dir"])
        h, hrec = heading_of(mesh, cat_h)
        # turn the body so its chosen side faces the gallery camera (heading 0)
        Rm = rot_y(-h)
        Vr, ur = (V * m_per) @ Rm.T, (u * m_per) @ Rm.T
        Vd = Vr + factor * ur
        both = np.vstack([Vr, Vd])
        floor = float(Vr[:, 1].min())
        cx = 0.5 * (both[:, 0].min() + both[:, 0].max())
        zf = float(both[:, 2].max())
        shift = np.array([cx, floor, zf])
        Vr -= shift
        umag = np.linalg.norm(ur, axis=1)
        umax = float(umag.max()) if umag.max() > 0 else 1.0
        np.savez(os.path.join(args.work, f"obj_{k}.npz"), V=Vr.astype(np.float64), F=F.astype(np.int32),
                 u=ur, factor=factor, u_rel=(umag / umax).astype(np.float32))
        objs.append({"k": k, "instruction": instr, "category": catname, "session": s,
                     "run": path, "unit_system": units, "L_m": L * m_per,
                     "dims_m": (np.ptp(V, axis=0) * m_per).tolist(), "faces": int(len(F)),
                     "factor": factor, "umax_m": float(umag.max()),
                     "standing": (rec.get("scale") or {}).get("standing_on_support"),
                     "heading_deg": h, "heading": hrec, "gate": gate, "notes": rec["notes"]})
        print(f"[{k:2d}] {instr[:44]:44s} L={L * m_per:9.4f} m  x{factor:g}  "
              f"heading {h:6.1f} ({hrec['source']}, {hrec['silhouette_vs_best']:.2f})")
    json.dump({"batch_root": os.path.abspath(args.batch_root), "objects": objs, "missing": missing,
               "elevation_deg": ELEV, "prep_seconds": time.time() - t0},
              open(os.path.join(args.work, "gl_objects.json"), "w"), indent=1, default=float)
    print(f"wrote {len(objs)} objects ({len(missing)} missing: {missing}) in {time.time() - t0:.0f} s")
    assert not missing, f"objects without a run or surface: {missing}"


if __name__ == "__main__":
    main()
