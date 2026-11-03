#!/usr/bin/env python
"""Tabulate a run_batch.py batch: one row per session, as Markdown, CSV and xlsx.

    /home/xl2254/anaconda3/bin/python experiments/bodyfitted_vs_nitsche/summarize.py

Reads ``<output>/<object>/comparison.json`` (its flat ``row``, its TetGen
summary and failures) and the job records in ``<output>/_batch/jobs.jsonl``.
A session without a report or with failed stages is still a row, with its
reason.  Writes ``<output>/_summary/summary.{md,csv,xlsx,json}``.  Summary rows:
median, min and max over all sessions with a value, and the median, min and
max over the unflagged ones, with n per column.  xlsx needs openpyxl (the base
conda env has it).
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _main_checkout(root: Path) -> Path:
    import subprocess
    try:
        common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
        if common:
            main = (root / common).resolve().parent
            if (main / "output").is_dir():
                return main
    except Exception:  # noqa: BLE001
        pass
    return root


DATA_ROOT = _main_checkout(REPO)
DEPTHS = (7, 8)
MODES = (("dofs", "P1 dofs"), ("surface", "P1 surf"), ("graded", "P1 graded"))

SESSION_COLUMNS = [
    ("object", "object", "s"), ("standin", "stand-in", "s"), ("E", "E [Pa]", "e2"),
    ("case_depth", "case depth", "i"),
    ("flag_any", "flagged", "b"), ("flag_thin", "thin", "b"), ("flag_crevices", "crevices", "b"),
    ("flag_load_near_support", "load<3h support", "b"), ("flag_support_issue", "support issue", "b"),
    ("flag_wall_thickness_over_h_p50", "wall p50 /h", "r2"), ("flag_max_depth_over_h", "max depth /h", "r2"),
    ("flag_crevice_area_share_below_h", "crevice area", "r3"),
    ("flag_load_to_support_gap_over_h", "load gap /h", "r2"),
    ("flag_support_rigid_mode_ratio", "support rigid-mode ratio", "e2"),
    ("tetgen_failed_meshes", "TetGen failed attempts", "s"), ("tetgen_unmeshed", "TetGen all attempts failed", "s"),
]
for d in DEPTHS:
    SESSION_COLUMNS += [(f"d{d}_nitsche_dofs", f"d{d} Nitsche DOFs", "i")]


def depth_columns(d):
    cols = []
    for m, ml in MODES:
        p = f"d{d}_{m}"
        cols += [(f"{p}_p1_dofs", f"d{d} {ml}: DOFs", "i"),
                 (f"{p}_relL2_rigid_removed", f"d{d} {ml}: rel L2 w/o rigid", "pct"),
                 (f"{p}_relL2", f"d{d} {ml}: rel L2", "pct"),
                 (f"{p}_compliance", f"d{d} {ml}: compl. N/P1", "r3"),
                 (f"{p}_stress_relL2", f"d{d} {ml}: rel L2 Cauchy", "pct"),
                 (f"{p}_stress_relL2_away", f"d{d} {ml}: rel L2 Cauchy >= 2h from patches", "pct"),
                 (f"{p}_vm_relL2", f"d{d} {ml}: rel L2 von Mises", "pct"),
                 (f"{p}_vm_p95", f"d{d} {ml}: vM p95 N/P1", "r3"),
                 (f"{p}_vm_p99", f"d{d} {ml}: vM p99 N/P1", "r3"),
                 (f"{p}_loose", f"d{d} {ml}: LOOSER TOLERANCE (metrics moved here)", "s")]
    return cols


def time_columns(d):
    cols = [(f"d{d}_nitsche_prep_s", f"d{d} Nitsche prep s", "t"),
            (f"d{d}_nitsche_solve_s", f"d{d} Nitsche solve s (tight)", "t"),
            (f"d{d}_nitsche_solve_session_tol_s", f"d{d} Nitsche solve s (rel 1e-2)", "t")]
    for m, ml in MODES:
        p = f"d{d}_{m}"
        cols += [(f"{p}_p1_prep_s", f"d{d} {ml}: prep s", "t"),
                 (f"{p}_p1_prep_incl_failed_s", f"d{d} {ml}: prep s incl. failed TetGen", "t"),
                 (f"{p}_p1_solve_s", f"d{d} {ml}: solve s (tight)", "t"),
                 (f"{p}_p1_solve_session_tol_s", f"d{d} {ml}: solve s (rel 1e-2)", "t")]
    return cols


LEGEND = """
Error: e = u_Nitsche - u_P1 on the P1 mesh's own quadrature (order 2), over the true body.
rel L2 w/o rigid (headline): ||e - r|| / ||u_P1||, r = the best-fit infinitesimal rigid motion of e (6 modes,
        weighted least squares at the quadrature points). rel L2: ||e|| / ||u_P1||.
Stress: each method's Cauchy stress from its own displacement gradient and the same (neo-Hookean) material, on the
        same quadrature: rel L2 of the tensor ||s_N - s_P1|| / ||s_P1||, of von Mises, and the volume-weighted p95/p99
        von Mises ratios N/P1; also the tensor rel L2 over the points >= 2 h_min from every Dirichlet and Neumann
        patch (the clamp band and the loads distort stress locally). No rigid removal: a rigid motion has no strain.
compl. N/P1: l(u_N) / l(u_P1), l = the P1 load functional (traction and body force at the P1 nodes); < 1 = Nitsche stiffer.
Nitsche: workflow run_forward, float64, diffuse volume, remesh_octree_depth d, BC tolerance bc_match_tolerance
        (dev 8c2f794). P1: TetGen tets, clamp band = every node within bc_match_tolerance(eps, h_min of depth d) of a
        Dirichlet region (its components), Nitsche load definition t max(0, n.n_r) with total t x loaded_area.
P1 dofs: vertex count ~ the Nitsche node count at depth d. P1 surf: surface triangles ~ h_min of depth d.
        P1 graded: ~h_min at the surface, interior sized by the depth-d octree's leaf sizes (TetGen -m).
Times: wall clock, GPU-synchronized, after a warm-up in the same process (JIT excluded). Nitsche prep = surface load,
        case parse, BC resolution, SDF + octree, solver build, trimming, build-case; P1 prep = surface remesh, sizing,
        TetGen, boundary mapping, BC transfer, solver setup (and the same plus failed TetGen attempts). Solve (tight):
        Newton to the common rel tolerance (default 1e-6); solve (rel 1e-2): the same solve up to the first iterate at
        the session's own tolerance, from the Newton log.
Flags: thin = area-median wall thickness < 4 h (inward rays); crevices = > 1 % of the surface faces an exterior gap
        < h; load<3h support; support issue = a support piece < 50 h^2 or a partial clamp on a section < 4 h. h is the
        session's own analysis leaf.
"""


def fmt(v, kind):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return ""
    try:
        if kind == "pct":
            return f"{100 * v:.2f}%"
        if kind == "r3":
            return f"{v:.3f}"
        if kind == "r2":
            return f"{v:.2f}"
        if kind == "e2":
            return f"{v:.2e}"
        if kind == "i":
            return f"{int(v):,}"
        if kind == "t":
            return f"{v:.2f}" if v < 100 else f"{v:.0f}"
        if kind == "b":
            return "yes" if v else "no"
    except (TypeError, ValueError):
        pass
    return str(v)


def load_rows(out: Path):
    jobs = {}
    jp = out / "_batch" / "jobs.jsonl"
    if jp.is_file():
        for line in jp.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("phase", "solve") == "solve":
                    jobs[rec["object"]] = rec
    names = sorted(set(jobs) | {p.parent.name for p in out.glob("*/comparison.json") if not p.parent.name.startswith("_")})
    rows = []
    for name in names:
        row = {"object": name}
        notes = []
        path = out / name / "comparison.json"
        if path.is_file():
            rep = json.loads(path.read_text())
            row.update(rep.get("row") or {})
            row["object"] = name
            row["_tetgen"] = rep.get("tetgen_summary") or {}
            row["_tetgen_log"] = rep.get("tetgen_log") or []
            row["_depths"] = rep.get("depths") or {}
            row["_failures"] = rep.get("failures") or []
            notes += [f"{f['stage']}: {f['reason'][:240]}" for f in row["_failures"]]
        else:
            notes.append("no report")
        job = jobs.get(name, {})
        if job:
            row["job_seconds"] = job.get("seconds")
            row["peak_vram_mib"] = job.get("peak_vram_mib")
            if job.get("timed_out"):
                notes.append(f"timeout (last stage {job.get('last_stage')})")
            elif job.get("returncode") not in (0, None) and not job.get("completed"):
                notes.append(f"exit {job.get('returncode')} (last stage {job.get('last_stage')})")
            if job.get("note"):
                row["exit_note"] = job["note"]
        row["status"] = "; ".join(notes) if notes else "ok"
        rows.append(row)
    return rows


TIGHT = 1e-6  # the common tight relative Newton tolerance (--tight-rtol)
METRIC_SUFFIXES = ("relL2_rigid_removed", "relL2", "compliance", "stress_relL2", "stress_relL2_away", "vm_relL2",
                   "vm_p95", "vm_p99")


def flag_loose(rows):
    """Entries whose Nitsche or P1 solve converged only to a looser tolerance
    than ``TIGHT`` (a looser-tolerance retry, or a final relative residual above
    it).  Their metrics move out of the numeric columns (and the medians) into a
    flagged text column ``d<D>_<mode>_loose``, with the achieved residuals, so a
    difference at ~1e-2 solver precision is not read as a tight-tolerance one."""
    flagged = []
    for r in rows:
        for d, D in (r.get("_depths") or {}).items():
            nit = D.get("nitsche") or {}
            n_rtol = nit.get("rtol")
            n_rel = (nit.get("log") or {}).get("final_relative_residual")
            nit_loose = bool(nit.get("ok")) and ((n_rtol or 0) > TIGHT * 1.0001 or (n_rel or 0) > TIGHT * 1.0001)
            for m, res in (D.get("modes") or {}).items():
                if not res.get("ok"):
                    continue
                s = res.get("solve") or {}
                p_rtol = (s.get("solver_extra") or {}).get("relative_tol")
                p_rel = s.get("final_relative_residual")
                p_loose = (p_rtol or 0) > TIGHT * 1.0001 or (p_rel or 0) > TIGHT * 1.0001
                if not (nit_loose or p_loose):
                    continue
                p = f"d{d}_{m}"
                why = []
                if nit_loose:
                    why.append(f"Nitsche rtol {n_rtol:g}, achieved {n_rel if n_rel is None else f'{n_rel:.1e}'}")
                if p_loose:
                    why.append(f"P1 rtol {p_rtol or TIGHT:g}, achieved {p_rel if p_rel is None else f'{p_rel:.1e}'}")
                vals = {k: r.pop(f"{p}_{k}", None) for k in METRIC_SUFFIXES}
                r[f"{p}_loose"] = "; ".join(why) + " | " + ", ".join(
                    f"{k}={v:.4g}" for k, v in vals.items() if isinstance(v, (int, float)))
                flagged.append((r["object"], d, m, "; ".join(why)))
    return flagged


def numeric(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out.append(float(v))
    return out


def aggregates(rows, keys):
    groups = {"all": rows, "unflagged": [r for r in rows if r.get("flag_any") is False]}
    agg = {}
    for g, rs in groups.items():
        for stat in ("median", "min", "max", "n"):
            agg[f"{stat} ({g})"] = {}
        for k in keys:
            v = numeric(rs, k)
            agg[f"n ({g})"][k] = len(v)
            if v:
                agg[f"median ({g})"][k] = statistics.median(v)
                agg[f"min ({g})"][k] = min(v)
                agg[f"max ({g})"][k] = max(v)
    return agg


AGG_ROWS = ("median (all)", "min (all)", "max (all)", "n (all)",
            "median (unflagged)", "min (unflagged)", "max (unflagged)", "n (unflagged)")


def markdown_table(rows, agg, columns, title):
    lines = [f"### {title}", "", "| " + " | ".join(h for _, h, _ in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k), kind) for k, _, kind in columns) + " |")
    for label in AGG_ROWS:
        cells = [f"**{label}**"]
        for k, _, kind in columns[1:]:
            v = agg[label].get(k)
            cells.append("" if kind in ("s", "b") else (str(v) if label.startswith("n ") and v is not None
                                                          else fmt(v, kind)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def tetgen_rows(rows):
    out = []
    for r in rows:
        for mesh, s in sorted((r.get("_tetgen") or {}).items()):
            for f in s["failures"]:
                out.append({"object": r["object"], "mesh": mesh, "stage": f.get("stage"),
                            "switches": f.get("switches"), "type": f.get("type"),
                            "message": (f.get("message") or "")[:300],
                            "recovered": f"yes: {s['success_switches']}" if s["succeeded"] else "no: mesh failed"})
    return out


def fallback_counts(rows):
    """How many meshes / solves used each retry or fallback."""
    c = {"tetgen: meshes needing a switch fallback (-Y toggled / looser quality)": 0,
         "tetgen: graded meshes retried without -Y": 0,
         "tetgen: meshes with background scaling k != 1 (TetGen 1.6 -m workaround)": 0,
         "remesh: redone at maxsurfdist 0.05 s": 0,
         "remesh: meshed at source resolution (remeshed surface self-intersects)": 0,
         "source surface self-intersects (no P1)": 0,
         "P1 solve: 5000 Krylov iterations": 0, "P1 solve: unit-diagonal scaling": 0,
         "P1 solve: SIZE_2 aggregation": 0, "P1 solve: pyamgx backend": 0,
         "P1 solve: sparse direct (SuperLU)": 0, "P1 solve: looser tolerance": 0,
         "Nitsche: other retries (iterations / pyamgx / direct / looser tolerance)": 0,
         "Nitsche: unit-diagonal retry": 0, "Nitsche: load continuation (20 / 50 load steps)": 0,
         "Nitsche: converged on pyamgx (operator-skill order)": 0,
         "Nitsche: converged on pyamgx + beta_max_over_e 500": 0,
         "Nitsche: converged on the PARDISO direct solve": 0,
         "P1 solve: pyamgx backend (operator-skill order)": 0,
         "P1 solve on the CPU: modified Newton (u=0 tangent, PARDISO or pyamg)": 0,
         "P1 solve on the CPU: full Newton (PARDISO refactorized)": 0,
         "P1 solve on the GPU, lean path: full Newton (torch tangent + JAX-AMG)": 0,
         "sessions done despite a non-zero exit (AMGX exit abort)": 0}
    for r in rows:
        by_mesh = {}
        for a in r.get("_tetgen_log") or []:
            by_mesh.setdefault(a["mesh"], []).append(a)
        for mesh, att in by_mesh.items():
            tg = [a for a in att if a.get("stage") == "tetgen"]
            if mesh == "source" and any(not a["ok"] for a in att if a.get("stage") == "selfint_check"):
                c["source surface self-intersects (no P1)"] += 1
            if any(a.get("stage") == "remesh_retry" for a in att):
                c["remesh: redone at maxsurfdist 0.05 s"] += 1
            if any(a.get("stage") == "remesh_fallback_to_source" for a in att):
                c["remesh: meshed at source resolution (remeshed surface self-intersects)"] += 1
            if any((a.get("bg_scale") or 1.0) != 1.0 for a in tg):
                c["tetgen: meshes with background scaling k != 1 (TetGen 1.6 -m workaround)"] += 1
            ok_sw = [a["switches"] for a in tg if a["ok"]]
            failed = [a for a in tg if not a["ok"]]
            if failed and ok_sw:
                if "graded" in mesh and ok_sw and "Y" not in (ok_sw[-1] or "") and any("Y" in (a["switches"] or "") for a in failed):
                    c["tetgen: graded meshes retried without -Y"] += 1
                else:
                    c["tetgen: meshes needing a switch fallback (-Y toggled / looser quality)"] += 1
        for d, D in (r.get("_depths") or {}).items():
            nit = D.get("nitsche") or {}
            if nit.get("rung"):  # the ladder's records: count the accepted rung
                if nit.get("ok") and nit["rung"] != "jaxamg":
                    key = f"Nitsche: accepted on rung {nit['rung']}"
                    c[key] = c.get(key, 0) + 1
                for m, res in (D.get("modes") or {}).items():
                    s = res.get("solve") or {}
                    if res.get("ok") and s.get("rung") and s["rung"] != "jaxamg":
                        key = f"P1 GPU: accepted on rung {s['rung']}"
                        c[key] = c.get(key, 0) + 1
                    elif res.get("ok") and s.get("solver") in ("cpu", "lean-gpu"):
                        key = ("P1 solve on the CPU: full Newton (PARDISO refactorized)" if "full Newton" in
                               (s.get("method") or "") and s.get("solver") == "cpu" else
                               "P1 solve on the CPU: modified Newton (u=0 tangent, PARDISO or pyamg)"
                               if s.get("solver") == "cpu" else
                               "P1 solve on the GPU, lean path: full Newton (torch tangent + JAX-AMG)")
                        c[key] = c.get(key, 0) + 1
                continue
            if (nit.get("fallback") or 0) == 1:
                c["Nitsche: unit-diagonal retry"] += 1
            elif (nit.get("fallback") or 0) > 1:
                c["Nitsche: other retries (iterations / pyamgx / direct / looser tolerance)"] += 1
            if nit.get("ok"):
                fin = [a for a in nit.get("attempts") or [] if a.get("error") is None and not a.get("skipped")]
                fb = (fin[-1] if fin else {}).get("backend")
                if fb == "pyamgx":
                    c["Nitsche: converged on pyamgx + beta_max_over_e 500" if (fin[-1].get("shape_opt_extra") or {})
                      .get("beta_max_over_e") else "Nitsche: converged on pyamgx (operator-skill order)"] += 1
                elif fb == "pardiso-direct" or nit.get("direct_solver"):
                    c["Nitsche: converged on the PARDISO direct solve"] += 1
            if any(a.get("load_steps") and a.get("error") is None and a.get("load_steps") >= 20
                   for a in nit.get("attempts") or []):
                c["Nitsche: load continuation (20 / 50 load steps)"] += 1
            for m, res in (D.get("modes") or {}).items():
                solve = res.get("solve") or {}
                extra = solve.get("solver_extra") or {}
                if res.get("ok") and solve.get("solver") == "cpu":
                    c["P1 solve on the CPU: full Newton (PARDISO refactorized)" if "full Newton" in
                      (solve.get("method") or "") else
                      "P1 solve on the CPU: modified Newton (u=0 tangent, PARDISO or pyamg)"] += 1
                    continue
                if res.get("ok") and solve.get("solver") == "lean-gpu":
                    c["P1 solve on the GPU, lean path: full Newton (torch tangent + JAX-AMG)"] += 1
                    continue
                if extra.get("cg_backend_assembled") == "pyamgx" and res.get("ok"):
                    c["P1 solve: pyamgx backend (operator-skill order)"] += 1
                    continue
                if extra.get("relative_tol"):
                    c["P1 solve: looser tolerance"] += 1
                elif extra.get("direct_solver"):
                    c["P1 solve: sparse direct (SuperLU)"] += 1
                elif extra.get("cg_backend_assembled") == "pyamgx":
                    c["P1 solve: pyamgx backend"] += 1
                elif extra.get("amg_selector"):
                    c["P1 solve: SIZE_2 aggregation"] += 1
                elif (res.get("solve") or {}).get("unit_diagonal"):
                    c["P1 solve: unit-diagonal scaling"] += 1
                elif extra.get("jaxamg_max_iters"):
                    c["P1 solve: 5000 Krylov iterations"] += 1
        if r.get("exit_note"):
            c["sessions done despite a non-zero exit (AMGX exit abort)"] += 1
    return c


def gap_lines(rows, out):
    """Every Nitsche entry that did not converge: each rung's outcome, the support
    rigid-mode ratio, the worst direct (PARDISO) linear residual, the P1 side, and
    the session's ``notes.json`` (manual probes, superseded results)."""
    lines = []
    for r in rows:
        for d, D in (r.get("_depths") or {}).items():
            nit = D.get("nitsche") or {}
            if nit.get("ok") or not nit:
                continue
            att = [a for a in nit.get("attempts") or [] if not a.get("skipped")]
            rungs = "; ".join(f"{a.get('rung') or a.get('index')}: {str(a.get('error'))[:80]}" for a in att)
            direct = [a.get("direct_rel_residual_max") for a in att if a.get("direct_rel_residual_max")]
            p1 = [(m, res) for m, res in (D.get("modes") or {}).items()]
            p1_txt = ", ".join(f"{m} {'solved' if res.get('ok') else 'failed'}"
                               f" ({((res.get('solve') or {}).get('solver')) or '-'})" for m, res in p1)
            ratio = r.get("flag_support_rigid_mode_ratio")
            lines.append(f"- **{r['object']} Nitsche d{d}**: no rung converged ({len(att)} run). Rungs: {rungs}. "
                         f"Support rigid-mode ratio {('-' if ratio is None else f'{ratio:.1e}')} (lambda_min/lambda_max "
                         f"of the supports' rigid-mode Gram matrix). Worst PARDISO linear residual "
                         f"{('-' if not direct else f'{max(direct):.1e}')}. P1 at d{d}: {p1_txt}.")
            notes = out / r["object"] / "notes.json"
            if notes.is_file():
                for k, v in json.loads(notes.read_text()).items():
                    lines.append(f"  - {k}: {v}")
    return lines


def note_lines(rows, out):
    """``notes.json`` of every session that is not a documented gap: remarks on
    entries that are solved and compared (e.g. a convergence that a previous
    run did not reproduce).  They are notes, not flags: nothing is excluded."""
    gap = {r["object"] for r in rows for D in (r.get("_depths") or {}).values()
           if (D.get("nitsche") or {}) and not (D.get("nitsche") or {}).get("ok")}
    lines = []
    for r in rows:
        notes = out / r["object"] / "notes.json"
        if r["object"] not in gap and notes.is_file():
            lines += [f"- **{r['object']}**, {k}: {v}" for k, v in json.loads(notes.read_text()).items()]
    return lines


def failure_reasons(rows):
    """Failed (depth, mode) pairs and sessions grouped by reason."""
    groups = {}
    for r in rows:
        for d, D in (r.get("_depths") or {}).items():
            nit = D.get("nitsche") or {}
            if not nit.get("ok"):
                groups.setdefault(f"Nitsche d{d}: " + reason_class(nit.get("error")), []).append(r["object"])
            for m, res in (D.get("modes") or {}).items():
                if not res.get("ok"):
                    groups.setdefault(f"P1 {m}: " + reason_class(res.get("error")), []).append(f"{r['object']} d{d}")
        if not r.get("_depths"):
            groups.setdefault("no report", []).append(r["object"])
    return groups


def reason_class(err):
    e = (err or "").lower()
    for key, label in (("self-intersects", "generated surface self-intersects"),
                       ("no nitsche run", "no Nitsche run at this depth"),
                       ("no tetgen variant", "TetGen failed for every variant"),
                       ("no mesh could be built", "TetGen failed at every size"),
                       ("out of memory", "out of memory"), ("failed to allocate", "out of memory"),
                       ("illegal memory", "CUDA illegal memory access"),
                       ("linear solver failed", "linear solver did not converge"),
                       ("did not converge", "Newton did not converge")):
        if key in e:
            return label
    return (err or "unknown")[:80]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", default=str(DATA_ROOT / "output" / "bodyfitted_vs_nitsche"))
    args = ap.parse_args()
    out = Path(args.output)
    rows = load_rows(out)
    loose = flag_loose(rows)
    err_cols = {d: depth_columns(d) for d in DEPTHS}
    t_cols = {d: time_columns(d) for d in DEPTHS}
    all_cols = (SESSION_COLUMNS + [c for d in DEPTHS for c in err_cols[d]] + [c for d in DEPTHS for c in t_cols[d]]
                + [("job_seconds", "job s", "t"), ("peak_vram_mib", "peak VRAM MiB", "i"), ("status", "status", "s")])
    agg = aggregates(rows, [k for k, _, kind in all_cols if kind not in ("s", "b")])
    tg = tetgen_rows(rows)
    n_fail_any = len({t["object"] for t in tg})
    n_unmeshed = sum(1 for r in rows if r.get("tetgen_unmeshed"))
    first = [SESSION_COLUMNS[0]]
    sd = out / "_summary"
    sd.mkdir(parents=True, exist_ok=True)
    n_unfl = sum(1 for r in rows if r.get("flag_any") is False)
    n_p1 = sum(1 for r in rows for D in (r.get("_depths") or {}).values()
               for res in (D.get("modes") or {}).values() if res.get("ok") and "rel_L2" in res)
    n_nit = sum(1 for r in rows for D in (r.get("_depths") or {}).values() if (D.get("nitsche") or {}).get("ok"))
    n_solved = sum(1 for r in rows for D in (r.get("_depths") or {}).values()
                   for res in (D.get("modes") or {}).values() if res.get("ok"))
    n_sessions = len(rows)
    true_tg = [(r["object"], d, m, (res.get("error") or "")[:300]) for r in rows
               for d, D in (r.get("_depths") or {}).items() for m, res in (D.get("modes") or {}).items()
               if not res.get("ok") and any(k in (res.get("error") or "") for k in
                                            ("self-intersects", "no TetGen variant meshed", "no mesh could be built"))]
    head_fc = fallback_counts(rows)
    md = ["# Diffused Nitsche vs body-fitted P1 (clamp band, Nitsche load), depths 7 and 8", "",
          f"**Coverage: {n_p1} of {6 * n_sessions} P1 entries compared ({n_solved} solved) and {n_nit} of "
          f"{2 * n_sessions} Nitsche entries present.** A P1 entry is compared only where its depth's Nitsche "
          "solve converged; the others are outside every median.",
          "", "**True TetGen failures:**", ""]
    md += [f"- {o} d{d} {m}: {e}" for o, d, m, e in true_tg] or ["- none"]
    # Evidence per failed source (``_batch/tetgen_evidence.py``): TetGen -d on
    # the cleaned source, the pymeshlab count, every variant on the source as-is.
    for obj in sorted({o for o, *_ in true_tg}):
        ep = out / obj / "tetgen_evidence.json"
        if not ep.is_file():
            md.append(f"  - {obj}: no evidence file")
            continue
        e = json.loads(ep.read_text())
        c = e.get("tetgen_d_counts") or {}
        variants = ", ".join(f"{v['switches']}: {(v.get('error') or 'ok').split(': ')[-1][:60]}"
                             for v in e.get("tetgen_variants_on_source") or [])
        md.append(f"  - **{obj}** (generator: {e.get('generator', 'hunyuan3d')}; cleaned source {e['cleaned_faces']} faces): "
                  f"TetGen -d: {c.get('facet_facet_warnings')} facet-facet and {c.get('segment_facet_warnings')} "
                  f"segment-facet intersections, skipped triangles {c.get('skipped_triangles')}; pymeshlab: "
                  f"{e['pymeshlab_self_intersecting_faces']} self-intersecting faces; every variant on the source "
                  f"as-is failed: {variants}")
    gaps = gap_lines(rows, out)
    md += ["", "**Documented non-TetGen gaps** (a Nitsche solve no ladder rung converged; its depth's P1 entries are "
           "solved but not compared):", ""] + (gaps or ["- none"])
    notes = note_lines(rows, out)
    if notes:
        md += ["", "**Notes** (solved and compared entries; not flags, nothing excluded):", ""] + notes
    off = [(r["object"], d, m, res["solve"]) for r in rows for d, D in (r.get("_depths") or {}).items()
           for m, res in (D.get("modes") or {}).items()
           if res.get("ok") and (res.get("solve") or {}).get("solver") in ("cpu", "lean-gpu")]
    md += ["", f"**Off the standard GPU path:** {len(off)} P1 entries "
           f"({sum(1 for o in off if o[3].get('solver') == 'cpu')} on the CPU, "
           f"{sum(1 for o in off if o[3].get('solver') == 'lean-gpu')} lean GPU; "
           f"{sum(1 for o in off if 'full Newton' in (o[3].get('method') or ''))} of them by full Newton). "
           f"**Looser than the common rel {TIGHT:g} tolerance:** {len(loose)} entries "
           "(their metrics are in the flagged LOOSER TOLERANCE column, outside the medians)."]
    md += ["", "**Retries and fallbacks:** " + "; ".join(f"{k}: {v}" for k, v in head_fc.items() if v), "",
          f"{len(rows)} sessions, {n_unfl} unflagged. Error = Nitsche vs P1 directly on the P1 mesh. "
          f"TetGen: {n_fail_any} sessions with a failed attempt, {n_unmeshed} with a mesh that never meshed.", ""]
    md.append(markdown_table(rows, agg, SESSION_COLUMNS, "Sessions, flags and Nitsche sizes"))
    for d in DEPTHS:
        md.append(markdown_table(rows, agg, first + err_cols[d], f"Depth {d}: Nitsche vs P1"))
    for d in DEPTHS:
        md.append(markdown_table(rows, agg, first + t_cols[d], f"Depth {d}: timing (s)"))
    fc = fallback_counts(rows)
    md.append("### Retries and fallbacks used\n\n| retry / fallback | count |\n|---|---|")
    md += [f"| {k} | {v} |" for k, v in fc.items()]
    md.append("\n### P1 entries not solved on the standard GPU path\n\n"
              "Their solve times are CPU (or lean-GPU) times and are not comparable with the GPU times.\n\n"
              "| entry | device | method | why | final rel residual | unit-diagonal round |\n|---|---|---|---|---|---|")
    for o, d, m, why in loose:
        md.append(f"| {o} d{d} {m} | (tolerance) | converged looser than rel {TIGHT:g} | {why} | |")
    for r in rows:
        for d, D in (r.get("_depths") or {}).items():
            for m, res in (D.get("modes") or {}).items():
                s = res.get("solve") or {}
                if res.get("ok") and s.get("solver") in ("cpu", "lean-gpu"):
                    ud = s.get("unit_diagonal_round")
                    ud = ud if isinstance(ud, str) else (
                        "" if not ud else f"not accepted: {(ud.get('attempt') or {}).get('error', '')[:90]}")
                    md.append(f"| {r['object']} d{d} {m} | {s.get('solver')} | {s.get('method')} | "
                              f"{(s.get('reason') or '')[:120]} | {s.get('final_relative_residual')} | {ud} |")
    md.append("\n### Nitsche entries off the first attempt (backend per attempt)\n\n"
              "Operator-skill order for a stalled JAX-AMG solve: pyamgx, then beta_max_over_e 500 (uniform "
              "material only), then the direct solve; never unit-diagonal scaling.\n\n"
              "| entry | final backend | attempts (index: backend, load steps, result) |\n|---|---|---|")
    for r in rows:
        for d, D in (r.get("_depths") or {}).items():
            nit = D.get("nitsche") or {}
            att = [dict(a) for a in nit.get("attempts") or []
                   if "process died" not in str(a.get("error"))]
            for a in att:  # records older than the backend field
                a.setdefault("backend", None)
                if a["backend"] is None and not a.get("skipped"):
                    a["backend"] = ("pardiso-direct" if a.get("direct_solver") else
                                    (a.get("overrides") or {}).get("cg_backend_assembled") or "jaxamg")
            tried = [dict(a, tried_in_skill_order=True)
                     for a in ((nit.get("skill_chain") or {}).get("attempts") or [])]
            tried += [dict(a) for a in ((nit.get("unit_diagonal_round") or {}).get("attempts") or [])]
            if len(att) <= 1 and not tried and (att[:1] or [{}])[0].get("backend") in (None, "jaxamg"):
                continue
            desc = "; ".join(
                f"{a.get('rung') or a.get('index', '?')}: {a.get('backend') or '-'}"
                f"{' beta500' if (a.get('shape_opt_extra') or {}).get('beta_max_over_e') else ''}"
                f"{' ud' if a.get('unit_diagonal') else ''}, ls {a.get('load_steps')}, "
                f"{'skipped' if a.get('skipped') else ('ok' if a.get('error') is None else (a.get('error') or '')[:60])}"
                for a in att)
            fin = [a for a in att if a.get("error") is None and not a.get("skipped")]
            if tried:
                desc += " || skill order, not converged (stored result kept): " + "; ".join(
                    f"{a.get('rung') or a.get('index')}: {a.get('backend')}"
                    f"{' beta500' if (a.get('shape_opt_extra') or {}).get('beta_max_over_e') else ''}"
                    f"{' unit-diagonal' if a.get('unit_diagonal') else ''}, "
                    f"ls {a.get('load_steps')}, {(a.get('error') or 'ok')[:70]}" for a in tried)
            md.append(f"| {r['object']} d{d} | {(fin[-1].get('backend') if fin else 'failed')} | {desc} |")
    groups = failure_reasons(rows)
    md.append("\n### Failures by reason\n\n| reason | n | which |\n|---|---|---|")
    md += [f"| {k} | {len(v)} | {', '.join(v)} |" for k, v in sorted(groups.items())]
    md.append("")
    md.append("### Failures\n\n| object | status |\n|---|---|")
    for r in rows:
        if r["status"] != "ok":
            md.append(f"| {r['object']} | {r['status'].replace('|', '/')[:700]} |")
    md.append("\n### TetGen failures (every failed attempt)\n\n| object | mesh | stage | switches | type | message | recovered |\n"
              "|---|---|---|---|---|---|---|")
    for t in tg:
        md.append("| " + " | ".join(str(t[k] or "").replace("|", "/") for k in
                                     ("object", "mesh", "stage", "switches", "type", "message", "recovered")) + " |")
    md.append("\n### Legend\n\n" + LEGEND.strip().replace("\n", "  \n") + "\n")
    (sd / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    keys = [k for k, _, _ in all_cols]
    with open(sd / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h, _ in all_cols])
        for r in rows:
            w.writerow([r.get(k) for k in keys])
        for label in AGG_ROWS:
            w.writerow([label] + [agg[label].get(k) for k in keys[1:]])
    try:
        import openpyxl
        from openpyxl.styles import Font
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "summary"
        ws.append([h for _, h, _ in all_cols])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([r.get(k) for k in keys])
        ws.append([])
        for label in AGG_ROWS:
            ws.append([label] + [agg[label].get(k) for k in keys[1:]])
            for c in ws[ws.max_row]:
                c.font = Font(bold=True)
        for j, (_, _, kind) in enumerate(all_cols, start=1):
            code = {"pct": "0.00%", "r3": "0.000", "r2": "0.00", "e2": "0.00E+00", "i": "#,##0", "t": "0.00"}.get(kind)
            if code:
                for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                    for c in row:
                        if not isinstance(c.value, str):
                            c.number_format = code
        ws.freeze_panes = "B2"
        wt = wb.create_sheet("tetgen_failures")
        wt.append(["object", "mesh", "stage", "switches", "type", "message", "recovered"])
        for t in tg:
            wt.append([t[k] for k in ("object", "mesh", "stage", "switches", "type", "message", "recovered")])
        wf = wb.create_sheet("failures")
        wf.append(["object", "status"])
        for r in rows:
            if r["status"] != "ok":
                wf.append([r["object"], r["status"]])
        wl = wb.create_sheet("legend")
        for line in LEGEND.strip().splitlines():
            wl.append([line])
        wb.save(sd / "summary.xlsx")
    except ImportError:
        print("openpyxl not available; no xlsx")
    (sd / "summary.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
         "aggregates": agg, "tetgen_failures": tg, "fallback_counts": fc,
         "failures_by_reason": groups,
         "run_info": json.loads((out / "_summary" / "run_info.json").read_text())
         if (out / "_summary" / "run_info.json").is_file() else None}, indent=2, default=str), encoding="utf-8")
    print(f"{len(rows)} rows -> {sd}")


if __name__ == "__main__":
    main()
