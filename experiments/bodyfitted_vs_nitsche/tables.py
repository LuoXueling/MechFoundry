#!/usr/bin/env python
"""Per-depth tables of the diffused Nitsche vs body-fitted P1 run, as Markdown, xlsx and LaTeX.

    /home/xl2254/anaconda3/bin/python experiments/bodyfitted_vs_nitsche/tables.py

Six tables, one row per object (50), for octree depths 7 and 8:

* degrees of freedom of the Nitsche octree and of the three P1 meshes;
* accuracy: the discrepancy measures against each P1 mesh;
* cost: retries, preparation and solve wall times.

The LaTeX output is written for a journal's Supplementary Information: one
fragment per table (``tex/tab_si_bf_<kind>_d<depth>.tex``, for ``\\input``): the
accuracy and cost tables are landscape ``longtable``s with booktabs rules and a
continued caption, the DOF table a one-page landscape table (two halves side by
side).  No notes under the tables: each caption explains its symbols, flags,
units and remaining markers, and refers to the procedure section for the rest.
Accuracy rows with eta_rig > 10 % against any mesh are shaded.  All six are in
``tables.tex``; ``tables_preview.tex`` is a standalone wrapper that inputs the
fragments (the packages are listed in it).
Notation follows the SI text (``output/paper_si/si_procedures.md``, "Error
comparison with body-fitted FEM").

Records.  Everything is computed from the run's stored per-entry records, not
from ``summary.md``: ``<output>/<object>/comparison.json`` (its ``depths`` records:
the Nitsche ladder record with every rung, each P1 entry's result with its
timing and its GPU rungs, the TetGen log, the flags), ``cpu_jobs/<tag>.log``
(the CPU worker's failed runs), ``tetgen_evidence.json`` and ``notes.json``; the
object names come from the batch's prompts.  The accuracy metrics are the ones
``recompute_metrics.py`` re-evaluated from the stored solutions (the P1 npz and
the Nitsche VTU) with the production ``evaluate_p1``, which also gives the H1
error; it checks each against comparison.json.

The collected records are saved to ``summary/tables/collected.json``, and when
the run's outputs are not on disk (they are archived:
``output/archives/bodyfitted_vs_nitsche.tar.zst``) the tables are rendered from
that file, so a presentation change needs no extraction.  The committed
collected.json was rebuilt from this script's previous outputs (commit ae39f1c:
tables.xlsx, plus the full-precision summary.json and recomputed_metrics.json)
and re-renders that commit's tables.md and tables.tex byte for byte with its
own renderer.

Medians follow ``summarize.py``: over every object with a value, and over the
unflagged ones; an entry solved only to a looser tolerance (``summarize.flag_loose``)
is shown, marked, and left out of the metric medians.

Writes ``summary/tables/{tables.md,tables.xlsx,tables.tex,tables_preview.tex,tex/}``
next to this script.
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize  # noqa: E402

DEPTHS = ("7", "8")
MODES = (("dofs", "Matched DOFs"), ("surface", "Matched surface"), ("graded", "Graded"))
# Flag letters, in the order printed; the captions define them (flags_clause).
FLAG_KEYS = (("flag_thin", "T"), ("flag_crevices", "G"), ("flag_load_near_support", "L"),
             ("flag_support_issue", "S"))
DEVICE = {"gpu": "", None: "", "cpu": "c", "lean-gpu": "L"}
P1_RUNGS = ("jaxamg", "jaxamg_4000", "pyamgx", "jaxamg_unit_diagonal")  # compare_bodyfitted_nitsche.P1_LADDER

# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def sig(v, n=2):
    """v to n significant figures, never in exponent form (values here are < 1e5)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if v == 0:
        return "0"
    digits = n - 1 - int(math.floor(math.log10(abs(v))))
    r = round(v, digits)
    if r and int(math.floor(math.log10(abs(r)))) > int(math.floor(math.log10(abs(v)))):  # 9.96 -> 10, not 10.0
        digits -= 1
        r = round(v, digits)
    return f"{r:.{max(digits, 0)}f}"


def pct(v):
    """A relative discrepancy in percent, two significant figures."""
    return sig(100.0 * v, 2)


def ratio(v):
    return f"{v:.3f}"


def secs(v):
    """Three significant figures; whole seconds from 1000 s (1496, not 1500)."""
    return f"{v:.0f}" if v >= 1000 else sig(v, 3)


def kdofs(v):
    """DOFs in thousands, three significant figures."""
    return sig(v / 1e3, 3)


def count(v):
    return str(int(v)) if float(v).is_integer() else sig(v, 2)


def differs(last, total, eps=0.005):
    """The all-attempts total is shown only where failed attempts add > 0.5 %."""
    return total is not None and last is not None and abs(total - last) > eps * max(abs(last), 1e-12)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def wall_seconds(text):
    """GNU time's 'Elapsed (wall clock)' field, h:mm:ss or m:ss, in seconds."""
    parts = [float(p) for p in text.strip().split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def failed_cpu_runs(obj_dir: Path, tag: str):
    """Wall times of the CPU worker's runs of this entry that ended non-zero
    (``cpu_jobs/<tag>.log``; each run is one '=====' block under GNU time)."""
    log = obj_dir / "cpu_jobs" / f"{tag}.log"
    if not log.is_file():
        return []
    out = []
    for block in re.split(r"^===== ", log.read_text(errors="replace"), flags=re.M)[1:]:
        status = re.search(r"Exit status: (\d+)", block)
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", block)
        if status and int(status.group(1)) != 0:
            out.append(wall_seconds(wall.group(1)) if wall else None)
    return out


def mesh_log(rep, tag):
    return [a for a in rep.get("tetgen_log") or [] if a.get("mesh") == tag]


REMESH_EVENTS = ("remesh_retry", "remesh_fallback_to_source")


def segments(att):
    """The mesh's TetGen log cut into build attempts, each ending at its TetGen
    call (its self-intersection checks and remesh redos come before it)."""
    out, cur = [], []
    for a in att:
        cur.append(a)
        if a.get("stage") in ("tetgen", "surface_remesh"):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def mesh_retries(att, mode, mesh_meta):
    """Retries of the meshing chain, for the accepted mesh and every failed attempt:

    * each failed TetGen call (crash, error or 30-min timeout), which moves to
      the next switch variant (surf), to a 15 % larger size (size-search
      trial), or repeats the graded search without surface preservation;
    * each remesh redo at maxsurfdist 0.05 s and fallback to the source surface
      in the accepted build or a failed one.

    Successful but unused size-search trials (dofs: each re-remeshes its
    surface) are not retries, and their remesh events are not counted.  The
    graded mesh remeshes its surface once, before its search, so its remesh
    events always belong to the accepted mesh."""
    segs = segments(att)
    n_failed = sum(1 for a in att if a.get("stage") in ("tetgen", "surface_remesh") and not a.get("ok"))
    if mode == "graded":
        return n_failed + sum(1 for a in att if a.get("stage") in REMESH_EVENTS)
    ok_segs = [s for s in segs if s[-1].get("ok") and s[-1].get("stage") == "tetgen"]
    final = ok_segs[-1] if ok_segs else None
    if mode == "dofs" and mesh_meta:
        good = [t for t in mesh_meta.get("search_trials") or [] if t.get("vertices")]
        hits = [i for i, t in enumerate(good) if t["vertices"] == mesh_meta.get("n_nodes")]
        if hits and len(good) == len(ok_segs):
            final = ok_segs[hits[-1]]
    counted = [s for s in segs if s is final or not (s[-1].get("ok") and s[-1].get("stage") == "tetgen")]
    return n_failed + sum(1 for s in counted for a in s if a.get("stage") in REMESH_EVENTS)


def nitsche_cost(nit, tight):
    run = [a for a in nit.get("attempts") or [] if not a.get("skipped")]
    # Every rung repeats the prep, but build-case is measured once (the
    # orchestrator's replay) and added into every rung's prep: count it once.
    bc = [(a.get("phases") or {}).get("build_case") or 0.0 for a in run]
    prep_all = sum(a.get("prep_seconds") or 0.0 for a in run) - sum(bc) + (bc[-1] if bc else 0.0)
    solve_all = sum(a.get("solve_seconds") or 0.0 for a in run)
    loose = bool(nit.get("ok")) and ((nit.get("rtol") or 0) > tight * 1.0001
                                     or ((nit.get("log") or {}).get("final_relative_residual") or 0) > tight * 1.0001)
    return {"ok": bool(nit.get("ok")), "rung": nit.get("rung"), "retries": max(len(run) - 1, 0),
            "rungs": [a.get("rung") for a in run], "prep": nit.get("prep_seconds"), "prep_all": prep_all,
            "solve": nit.get("solve_seconds"), "solve_all": solve_all, "solve_1e2": nit.get("solve_seconds_session_tol"),
            "device": "c" if nit.get("direct_solver") else "", "loose": loose, "dofs": nit.get("dofs"),
            "unrecorded": [a.get("rung") for a in run if a.get("prep_seconds") is None or a.get("solve_seconds") is None]}


def p1_cost(rep, obj_dir, d, m, res):
    tag = f"d{d}_{m}"
    att = mesh_log(rep, tag)
    t = res.get("timing") or {}
    s = res.get("solve") or {}
    # all = the accepted mesh's prep + every failed TetGen call (switch variants,
    # failed size-search trials, the graded search's -Y pass).  Unused successful
    # size-search trials are in neither.  Failed self-intersection checks are not
    # added: the accepted build's are inside its surface_remesh phase already,
    # the others belong to unused trials (the stored
    # prep_seconds_incl_failed_tetgen adds them; see ``prep_all_stored``).
    failed_tg = sum(a.get("seconds") or 0.0 for a in att if a.get("stage") == "tetgen" and not a.get("ok"))
    mm_path = obj_dir / "meshes" / f"{tag}.json"
    mesh_meta = json.loads(mm_path.read_text()) if mm_path.is_file() else None
    # The GPU rungs' own records (``attempts/p1_<tag>_<rung>.json``): a CPU or
    # lean entry's result keeps the CPU solve record, not its GPU rungs.
    rungs = [json.loads(p.read_text()) for p in (obj_dir / "attempts" / f"p1_{tag}_{r}.json" for r in P1_RUNGS)
             if p.is_file()]
    failed_gpu = [a for a in rungs if not a.get("ok")]
    cpu_failed = failed_cpu_runs(obj_dir, tag) if s.get("solver") in ("cpu", "lean-gpu") else []
    solve = t.get("solve_seconds")
    unrecorded = [a.get("rung") for a in failed_gpu if a.get("seconds") is None] + [
        "cpu" for v in cpu_failed if v is None]
    return {"mesh_retries": mesh_retries(att, m, mesh_meta),
            "trial_meshes": sum(1 for a in att if a.get("stage") == "tetgen" and a.get("ok")),
            "prep_all_stored": t.get("prep_seconds_incl_failed_tetgen"),
            "solve_retries": len(failed_gpu) + len(cpu_failed),
            "solve_path": [a.get("rung") for a in failed_gpu] + ["cpu"] * len(cpu_failed)
            + [s.get("rung") or s.get("solver")],
            "prep": t.get("prep_seconds"),
            "prep_all": None if t.get("prep_seconds") is None else t["prep_seconds"] + failed_tg,
            "solve": solve, "solve_all": None if solve is None else solve + sum(
                a.get("seconds") or 0.0 for a in failed_gpu) + sum(v or 0.0 for v in cpu_failed),
            "solve_1e2": t.get("solve_seconds_session_tol"), "device": DEVICE.get(s.get("solver"), "?"),
            "unrecorded": unrecorded, "p1_dofs": res.get("p1_dofs")}


def missing_reason(err, evidence):
    """Why a P1 entry has no mesh: a self-intersecting source surface (with the
    TetGen -d facet-facet + segment-facet intersection count on the cleaned
    source and pymeshlab's self-intersecting face count), else summarize's class."""
    if "self-intersects" in (err or "").lower():
        c = (evidence or {}).get("tetgen_d_counts") or {}
        return {"code": "self_intersection",
                "tetgen_d": (c.get("facet_facet_warnings") or 0) + (c.get("segment_facet_warnings") or 0)
                if evidence else None,
                "pymeshlab_faces": (evidence or {}).get("pymeshlab_self_intersecting_faces")}
    return {"code": "other", "text": summarize.reason_class(err)}


def prompts(batch: Path):
    """The user's instruction of each batch session, from its ``prompt.txt``."""
    out = {}
    if batch.is_dir():
        for run in sorted(batch.iterdir()):
            p = run / "prompt.txt"
            m = re.search(r'whole instruction is: "([^"]*)"', p.read_text()) if p.is_file() else None
            if m:
                out[re.sub(r"-\d{8}-\d{6}$", "", run.name)] = m.group(1)
    return out


def collect(out: Path, recomputed: dict, batch: Path):
    rows = summarize.load_rows(out)
    loose = {(o, str(d), m) for o, d, m, _ in summarize.flag_loose(rows)}
    tight = summarize.TIGHT
    instr = prompts(batch)
    table = []
    for r in rows:
        name = r["object"]
        obj_dir = out / name
        rep = json.loads((obj_dir / "comparison.json").read_text())
        ev_path = obj_dir / "tetgen_evidence.json"
        evidence = json.loads(ev_path.read_text()) if ev_path.is_file() else None
        notes = json.loads((obj_dir / "notes.json").read_text()) if (obj_dir / "notes.json").is_file() else {}
        row = {"object": name, "prompt": instr.get(name), "flag_any": r.get("flag_any"),
               "flags": [k for k, _ in FLAG_KEYS if r.get(k)] if r.get("flag_any") else [],
               "support_rigid_mode_ratio": r.get("flag_support_rigid_mode_ratio"),
               "notes": notes, "depths": {}}
        for d in DEPTHS:
            D = rep["depths"][d]
            nit = nitsche_cost(D.get("nitsche") or {}, tight)
            entry = {"nitsche": nit, "modes": {},
                     "note": any(f"d{d}" in k or f"d{d}" in str(v) for k, v in notes.items())}
            for m, _ in MODES:
                res = D["modes"].get(m) or {}
                e = {"ok": bool(res.get("ok")), "compared": "rel_L2" in res, "loose": (name, d, m) in loose}
                if not res.get("ok"):
                    e["reason"] = missing_reason(res.get("error"), evidence)
                else:
                    e.update(p1_cost(rep, obj_dir, d, m, res))
                    rc = recomputed.get(f"{name}/d{d}/{m}")
                    if rc:
                        v = rc["recomputed"]
                        e.update(L2r=v["rel_L2_rigid_removed"], H1r=rc["h1"]["rel_H1_rigid_removed"],
                                 H1s=rc["h1"]["rel_H1_seminorm_rigid_removed"], L2=v["rel_L2"],
                                 C=v["compliance_ratio"], p95=v["stress.von_mises_p95_ratio"],
                                 p99=v["stress.von_mises_p99_ratio"], verified=rc["bitwise_equal"],
                                 max_rel_diff=max((x for x in rc["rel_diff"].values() if x is not None), default=0.0))
                    elif e["compared"]:
                        raise RuntimeError(f"{name} d{d} {m}: compared but not recomputed; run recompute_metrics.py")
                entry["modes"][m] = e
            row["depths"][d] = entry
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Names, flags, markers and captions
# ---------------------------------------------------------------------------


# The object names and their order are those of the model-evaluation table
# (experiments/model_eval/evaluation_table.md, built by
# combine.py): the instruction verbatim, grouped by the category headings of the
# batch's object list, in its order.  Each session's slug is ab_helper.slug of
# its instruction (the batch's run-folder naming), so the map is exact.
OBJECT_LIST = HERE.parents[1] / "experiments" / "agentic_batch" / "objects_50.txt"
_CATALOGUE = {}


def categories(path=OBJECT_LIST):
    """[(category heading, [instruction, ...])], parsed as
    experiments/model_eval/build_packets.categories() parses the same file."""
    cats, cur = [], None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("# ") and len(s) < 40 and not s.startswith("# Demonstration"):
            cur = (s[2:], [])
            cats.append(cur)
        elif s and not s.startswith("#") and cur:
            cur[1].append(s)
    assert len(cats) == 10 and all(len(o) == 5 for _, o in cats), cats
    return cats


def catalogue():
    """slug -> (category index, category heading, index in category, instruction)."""
    if not _CATALOGUE:
        sys.path.insert(0, str(OBJECT_LIST.parent))
        from ab_helper import slug  # noqa: E402  (standard library only)
        for i, (cat, objs) in enumerate(categories()):
            for j, o in enumerate(objs):
                assert slug(o) not in _CATALOGUE, o
                _CATALOGUE[slug(o)] = (i, cat, j, o)
    return _CATALOGUE


def check_names(table):
    """Fail unless every session maps to one instruction of the list, all 50
    are present, and the session's own prompt (when recorded) is that instruction."""
    cat = catalogue()
    names = [r["object"] for r in table]
    unmapped = [n for n in names if n not in cat]
    if unmapped or len(set(names)) != len(cat):
        raise SystemExit(f"object names do not map onto {OBJECT_LIST}: unmapped {unmapped}, "
                         f"{len(set(names))} sessions vs {len(cat)} listed")
    bad = [(r["object"], r.get("prompt")) for r in table if r.get("prompt") and r["prompt"] != cat[r["object"]][3]]
    if bad:
        raise SystemExit(f"session prompts differ from the object list: {bad}")


def object_name(row):
    """The instruction verbatim, as in the model-evaluation table ("a TV remote control")."""
    return catalogue()[row["object"]][3]


def category(row):
    return catalogue()[row["object"]][1]


def flag_letters(row):
    return ", ".join(a for k, a in FLAG_KEYS if k in row["flags"])


def missing(row, d):
    """The reason of the first missing P1 entry at depth d (None: all built)."""
    return next((e["reason"] for e in row["depths"][d]["modes"].values() if not e["ok"]), None)


def loose_any(row, d):
    D = row["depths"][d]
    return D["nitsche"]["loose"] or any(e.get("loose") for e in D["modes"].values())


class Fmt:
    """Markup for one output format: inline math, superscript markers, the
    not-applicable dash, escaping."""

    def __init__(self, kind):
        self.kind = kind

    def esc(self, s):
        if self.kind != "tex":
            return s
        return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
                .replace("_", r"\_").replace("#", r"\#"))

    def mark(self, kinds):
        """The cell markers that remain in the tables (no table notes; the
        caption gives each one a clause): CPU solve, reduced-GPU solve, and a
        solve that converged only to the session tolerance."""
        out = ""
        for k in kinds:
            out += {"tex": {"cpu": r"\textsuperscript{C}", "reduced": r"\textsuperscript{R}",
                            "loose": r"$^{\dagger}$"},
                    "md": {"cpu": "<sup>C</sup>", "reduced": "<sup>R</sup>", "loose": "<sup>†</sup>"},
                    "plain": {"cpu": " [C]", "reduced": " [R]", "loose": " [†]"}}[self.kind][k]
        return out

    @property
    def dash(self):
        return {"tex": "---", "md": "—", "plain": "—"}[self.kind]


def T(tex, md=None, plain=None):
    """A label in each format: TeX, Markdown (Unicode), plain (xlsx)."""
    md = tex if md is None else md
    return {"tex": tex, "md": md, "plain": md if plain is None else plain}


def stack(*lines):
    """A header label on several lines (TeX: a one-column tabular)."""
    return T(r"\begin{tabular}[b]{@{}r@{}}" + r"\\".join(t["tex"] for t in lines) + r"\end{tabular}",
             " ".join(t["md"] for t in lines))


def m_(f, tex, md):
    """Inline math in format f (md and plain get the Unicode form)."""
    return {"tex": f"${tex}$", "md": md, "plain": md}[f.kind]


# The tables carry no notes: their explanations are in the SI procedure text
# ("Error comparison with body-fitted FEM", label si:error-comparison), and each
# caption says compactly what the reader needs to read the table.
GRAY_ETA = 0.10  # accuracy rows shaded where any mesh's eta_rig exceeds this


def gray_row(row, d):
    """An accuracy row is shaded when eta_rig exceeds 10 % against any P1 mesh."""
    return any((e.get("L2r") or 0.0) > GRAY_ETA for e in row["depths"][d]["modes"].values() if e["ok"])


def flags_clause(f):
    h = m_(f, r"h_{\min}", "h_min")
    return (f"Flags, at the session's {h}: T, wall thinner than {m_(f, r'4h_{\min}', '4h_min')}; G, gaps narrower "
            f"than {h} on over 1{f.esc('%')} of the surface; L, a load within {m_(f, r'3h_{\min}', '3h_min')} of a "
            f"support; S, a support piece below {m_(f, r'50h_{\min}^2', '50h_min²')} or a partial clamp on a "
            f"section thinner than {m_(f, r'4h_{\min}', '4h_min')}; blank, unflagged.")


def caption(f, kind, d):
    """The full caption of a table, in format f."""
    dash = f"{f.dash}, no body-fitted mesh (the input surface self-intersects)."
    med = (f"Medians over all objects and over the unflagged ones; {m_(f, 'n', 'n')}, number of values.")
    ref = {"tex": r"Details in Section~S\ref{si:error-comparison}.",
           "md": "Details in the SI section “Error comparison with body-fitted FEM”.",
           "plain": "Details in the SI section “Error comparison with body-fitted FEM”."}[f.kind]
    if kind == "dofs":
        head = (f"Degrees of freedom (in thousands) of diffused Nitsche (three per free node of the trimmed octree) "
                f"and of the three body-fitted P1 meshes (three per vertex) at octree depth {d}. Matched DOFs: vertex "
                f"count targeted at the octree's node count (secant search, closest of up to six trials kept); "
                f"matched surface: surface edge length "
                f"{m_(f, r'h_{\min}', 'h_min')}; graded: {m_(f, r'h_{\min}', 'h_min')} at the surface, coarser "
                f"inside like the octree.")
        return " ".join([head, flags_clause(f), dash, med, ref])
    if kind == "accuracy":
        eta = m_(f, r"\eta_{\mathrm{rig}}", "η_rig")
        head = (f"Discrepancy of diffused Nitsche from each body-fitted P1 solution at octree depth {d}. {eta} and "
                f"{m_(f, r'\eta^{H^1}_{\mathrm{rig}}', 'η_rig^H¹')}: relative {m_(f, 'L^2', 'L²')} and "
                f"{m_(f, 'H^1', 'H¹')} norms of the displacement difference after removing its best-fit rigid "
                f"motion (in {f.esc('%')}); {m_(f, r'\ell_{\mathrm N}/\ell_{\mathrm{P1}}', 'ℓ_N/ℓ_P1')}: compliance "
                f"ratio (below 1: Nitsche stiffer); {m_(f, 's_{95}', 's₉₅')}, {m_(f, 's_{99}', 's₉₉')}: "
                f"Nitsche-to-P1 ratios of the 95th and 99th percentiles of the von Mises stress.")
        gray = {"tex": "Gray rows", "md": "Rows in bold (gray in the PDF and xlsx)",
                "plain": "Gray rows"}[f.kind]
        return " ".join([head, flags_clause(f),
                         f"{gray}: {eta} above 10{f.esc('%')} against at least one mesh.", dash,
                         f"{f.mark(['loose']).strip()} Nitsche solve converged only to "
                         f"{m_(f, '10^{-2}', '10⁻²')}; excluded from the medians." if d == "8" else "",
                         med, ref]).replace("  ", " ")
    t2 = m_(f, "10^{-2}", "10⁻²")
    head = (f"Computational cost at octree depth {d}: retries and wall-clock times in seconds (GPU, after a "
            f"warm-up) of the successful attempt; in parentheses, the total over all attempts where failed attempts "
            f"add more than 0.5{f.esc('%')}. Solver retries: attempts after the first; mesh retries: failed TetGen "
            f"calls plus rejected remeshes of the surface the kept mesh was built from (rejections inside unkept "
            f"size-search trials not counted). Prep.: preparation (Nitsche: boundary conditions, SDF, octree, "
            f"trimming, solver setup; P1: surface resizing, TetGen, boundary mapping, solver setup). Solve: Newton "
            f"iteration to relative residual {m_(f, '10^{-6}', '10⁻⁶')}; to {t2}: the same solve until it first "
            f"reaches {t2}. Superscripts: {f.mark(['cpu']).strip()}, CPU solve; {f.mark(['reduced']).strip()}, "
            f"reduced GPU solver; both not comparable with GPU times"
            + (f"; {f.mark(['loose']).strip()}, converged only to {t2}, excluded from the median." if d == "8" else ".")
            )
    return " ".join([head, flags_clause(f), dash, med + " Time medians are of successful attempts.", ref])


# ---------------------------------------------------------------------------
# Tables.  A column is (headers, cell(row, d) -> Cell, median formatter or None, align);
# headers is one label per header level, top first ("" spans nothing).
# ---------------------------------------------------------------------------


class Cell:
    """main [+ marks] [+ ' (extra)'] in the table; value for the medians (None: left out)."""

    def __init__(self, main="", value=None, marks=(), extra=None, na=False):
        self.main, self.value, self.marks, self.extra, self.na = main, value, list(marks), extra, na


def object_marks(row, d, table):
    """Only the looser-tolerance marker stays on an object (accuracy tables);
    missing meshes show as dashes, remarks went to the procedure text."""
    return ["loose"] if table == "accuracy" and loose_any(row, d) else []


def col_object(levels, table):
    return ([T("")] * (levels - 1) + [T("Object")], lambda r, d: Cell(object_name(r), marks=object_marks(r, d, table)),
            None, "name")


def col_flags(levels):
    return ([T("")] * (levels - 1) + [T("Flags")], lambda r, d: Cell(flag_letters(r)), None, "c")


def na_cell():
    return Cell(na=True)


def mode_metric(m, key, fmt, loose_out=True):
    """A P1 entry's value; a looser-tolerance entry's metric is shown but left
    out of the medians (loose_out; not for its DOFs)."""
    def get(r, d):
        e = r["depths"][d]["modes"][m]
        if not e["ok"]:
            return na_cell()
        v = e.get(key)
        if v is None:
            return Cell()
        return Cell(fmt(v), None if loose_out and e.get("loose") else v)  # a looser-tolerance entry: shown, outside the medians
    return get


KDOFS = T(r"($10^3$)", "(10³)")
P1 = T("Body-fitted P1")


def table_dofs():
    def nit(r, d):
        v = r["depths"][d]["nitsche"]["dofs"]
        return Cell(kdofs(v), v)
    cols = [col_object(2, "dofs"), col_flags(2), ([T(""), stack(T("Diffused"), T("Nitsche"))], nit, kdofs, "r:1")]
    for m, ml in MODES:
        cols.append(([P1, stack(*map(T, ml.split()))], mode_metric(m, "p1_dofs", kdofs, loose_out=False), kdofs,
                     "r:1"))
    return cols


ACC = (("L2r", stack(T(r"$\eta_{\mathrm{rig}}$", "η_rig"), T(r"(\%)", "(%)")), pct),
       ("H1r", stack(T(r"$\eta^{H^1}_{\mathrm{rig}}$", "η_rig^H¹"), T(r"(\%)", "(%)")), pct),
       ("C", T(r"$\ell_{\mathrm N}/\ell_{\mathrm{P1}}$", "ℓ_N/ℓ_P1"), ratio),
       ("p95", T(r"$s_{95}$", "s₉₅"), ratio),
       ("p99", T(r"$s_{99}$", "s₉₉"), ratio))


def table_accuracy():
    cols = [col_object(3, "accuracy"), col_flags(3)]
    for m, ml in MODES:
        for key, head, fmt in ACC:
            cols.append(([P1, T(ml), head], mode_metric(m, key, fmt), fmt, "r:1"))
    return cols


def time_cell(v, total, marks=(), value=True):
    if v is None:
        return Cell()
    return Cell(secs(v), v if value else None, marks, secs(total) if differs(v, total) else None)


def device_marks(dev):
    return {"c": ["cpu"], "L": ["reduced"]}.get(dev, [])


def nit_value(kind):
    def get(r, d):
        n = r["depths"][d]["nitsche"]
        if not n["ok"]:
            return na_cell()
        dm = device_marks(n["device"])
        if kind == "retries":
            return Cell(str(n["retries"]), n["retries"])
        if kind == "prep":
            return time_cell(n["prep"], n["prep_all"])
        if kind == "solve":
            return time_cell(n["solve"], n["solve_all"], dm + (["loose"] if n["loose"] else []), not n["loose"])
        return time_cell(n["solve_1e2"], None, dm)
    return get


def p1_value(m, kind):
    def get(r, d):
        e = r["depths"][d]["modes"][m]
        if not e["ok"]:
            return na_cell()
        dm = device_marks(e["device"])
        if kind in ("mesh_retries", "solve_retries"):
            return Cell(str(e[kind]), e[kind])
        if kind == "prep":
            return time_cell(e["prep"], e["prep_all"])
        if kind == "solve":
            return time_cell(e["solve"], e["solve_all"], dm)
        return time_cell(e["solve_1e2"], None, dm)
    return get


SOLVER_R = stack(T("Solver"), T("retries"))
MESH_R = stack(T("Mesh"), T("retries"))
PREP = stack(T("Prep."), T("(s)"))
SOLVE = stack(T("Solve"), T("(s)"))
TO1E2 = stack(T("To"), T(r"$10^{-2}$", "10⁻²"), T("(s)"))


# Relative widths of the cost columns ("r:<weight>"): a time with its
# all-attempts total in parentheses wraps onto a second line where it must.
W_R, W_PREP, W_SOLVE, W_TO = "r:0.82", "r:1.1", "r:1.4", "r:0.8"


def table_cost():
    cols = [col_object(3, "cost"), col_flags(3)]
    nit = (T(""), T("Diffused Nitsche"))
    cols += [([*nit, SOLVER_R], nit_value("retries"), count, W_R), ([*nit, PREP], nit_value("prep"), secs, W_PREP),
             ([*nit, SOLVE], nit_value("solve"), secs, W_SOLVE), ([*nit, TO1E2], nit_value("s2"), secs, W_TO)]
    for m, ml in MODES:
        g = (P1, T(ml))
        cols += [([*g, MESH_R], p1_value(m, "mesh_retries"), count, W_R),
                 ([*g, SOLVER_R], p1_value(m, "solve_retries"), count, W_R),
                 ([*g, PREP], p1_value(m, "prep"), secs, W_PREP), ([*g, SOLVE], p1_value(m, "solve"), secs, W_SOLVE),
                 ([*g, TO1E2], p1_value(m, "s2"), secs, W_TO)]
    return cols


TABLES = (("dofs", table_dofs, "Degrees of freedom of the diffused Nitsche octree discretization and of the three "
                               "body-fitted P1 meshes at octree depth {d}"),
          ("accuracy", table_accuracy, "Discrepancy between the diffused Nitsche and body-fitted P1 solutions at "
                                       "octree depth {d}, for each P1 mesh"),
          ("cost", table_cost, "Computational cost at octree depth {d}: retries and wall-clock times of preparation "
                               "and solve for diffused Nitsche and for each body-fitted P1 mesh"))
SHORT = {"dofs": "Degrees of freedom at octree depth {d}", "accuracy": "Discrepancy at octree depth {d}",
         "cost": "Computational cost at octree depth {d}"}


def label(kind, d):
    return f"tab:si-bf-{kind}-d{d}"


def median_rows(table, cols, d):
    """(label, cells, ns) for all objects and for the unflagged ones."""
    groups = (("Median (all objects)", table), ("Median (unflagged)", [r for r in table if r["flag_any"] is False]))
    out = []
    for lab, rows in groups:
        cells, ns = [], []
        for _, get, fmt, _ in cols[2:]:
            vals = [c.value for r in rows for c in [get(r, d)]
                    if isinstance(c.value, (int, float)) and math.isfinite(c.value)]
            cells.append(fmt(statistics.median(vals)) if vals else "")
            ns.append(str(len(vals)))
        out.append((lab, cells, ns))
    return out


def ordered(table):
    """The rows in the evaluation table's order: by category, then as listed."""
    return sorted(table, key=lambda r: catalogue()[r["object"]][::2])


def render_cell(c, f):
    if c.na:
        return f.dash
    s = f.esc(c.main) + f.mark(c.marks)
    if c.extra:
        s += f" ({c.extra})"
    return s


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def spans(labels):
    """[(label, first, count)] of runs of equal labels ("" runs kept, unlabelled)."""
    out, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i] and labels[i]:
            j += 1
        out.append((labels[i], i, j - i + 1))
        i = j + 1
    return out


def md_table(kind, d, table, cols, f):
    lines = [f"### {SHORT[kind].format(d=d)}", "", f"LaTeX label `{label(kind, d)}`.", "",
             "*Caption.* " + caption(f, kind, d), ""]
    levels = len(cols[0][0])
    group = []
    for i in range(len(cols)):
        parts = [cols[i][0][lv]["md"] for lv in range(levels - 1) if cols[i][0][lv]["md"]]
        group.append(parts[-1] if parts else "")  # the innermost group label
    shown = [g if (i == 0 or group[i - 1] != g) else "" for i, g in enumerate(group)]
    lines.append("| " + " | ".join(shown) + " |")
    lines.append("|" + "|".join(":---" if a == "name" else ":---:" if a == "c" else "---:" for *_, a in cols) + "|")
    lines.append("| " + " | ".join(f"**{h[-1]['md']}**" for h, *_ in cols) + " |")
    for r in ordered(table):
        cells = [render_cell(get(r, d), f) for _, get, _, _ in cols]
        if kind == "accuracy" and gray_row(r, d):  # gray in the PDF and xlsx
            cells[0] = f"**{cells[0]}**"
        lines.append("| " + " | ".join(cells) + " |")
    for lab, cells, ns in median_rows(table, cols, d):
        lines.append("| " + " | ".join([f"*{lab}*", ""] + cells) + " |")
        lines.append("| " + " | ".join(["*n*", ""] + ns) + " |")
    return "\n".join(lines) + "\n"


DEFINITIONS = r"""
## Definitions and provenance

The tables carry no notes; these definitions complement the captions, and symbols follow the SI text ("Error comparison with body-fitted FEM").

**Degrees of freedom.** Diffused Nitsche: 3 × the free (inner) nodes of the trimmed octree at the depth compared.
Body-fitted P1: 3 × the tetrahedral mesh's vertices. Matched DOFs: vertex count targeted at the Nitsche node
count (secant search of up to six trials over the edge length s*, the closest trial kept). Matched surface: surface triangles of edge
length h_min. Graded: h_min at the surface, interior sized by c × the local octree leaf size (TetGen -m), vertex
count targeted as for matched DOFs. Eleven entries (eight matched-DOF, three graded) end outside 5 % of the
node count.

**Discrepancy measures** (against that P1 solution, on the P1 mesh's second-order quadrature over the body Ω;
e = u_N − u_P1). r* is the best-fit infinitesimal rigid motion of e (six modes, weighted least squares at the
quadrature points).
- η_rig = ‖e − r*‖_L² / ‖u_P1‖_L², the rigid-removed relative L² discrepancy (the headline measure).
- η_rig^H¹ = ‖e − r*‖_H¹ / ‖u_P1‖_H¹, with ‖v‖²_H¹ = ∫|v|² + ∫|∇v|²_F, the same r* (fitted in L², not refitted),
  the same quadrature points and each method's own displacement gradient (P1: constant per tetrahedron; diffused
  Nitsche: the trilinear gradient of its octree cell), the arrays the stress measures use. An infinitesimal
  rotation has a constant skew gradient, so removing r* also changes the gradient part. The norm mixes units
  (lengths in metres): for parts much smaller than 1 m the gradient term dominates. The gradient-only ratio and
  the discrepancy without rigid removal (η) are in the xlsx.
- ℓ_N/ℓ_P1 = ℓ(u_N)/ℓ(u_P1), the compliance ratio, ℓ the work of the P1 nodal loads; below 1 means diffused
  Nitsche is stiffer.
- s₉₅, s₉₉: Nitsche-to-P1 ratios of the volume-weighted 95th and 99th percentiles of the von Mises stress (Cauchy
  stress from each method's own gradient, the same neo-Hookean material).

**Cost** (wall-clock seconds, GPU-synchronized, after an untimed warm-up in the same process). A value is that of
the successful attempt (for P1, on the kept mesh); a value in parentheses is the total over all attempts, shown
where failed attempts add more than 0.5 %. Successful size-search trials that were not kept are in neither.
- Diffused Nitsche solver retries: fallback paths run after the first (GPU multigrid, more Krylov iterations,
  a second multigrid implementation, the same with a 500E penalty cap, diagonal scaling, direct linear solves on
  the CPU, the same with 10 load steps and line search, the same at the session tolerance; skipped paths not
  counted). Every path is a full rerun, so a Nitsche retry repeats preparation and solve. Preparation: surface
  load, case parse, boundary-condition resolution, SDF and octree, solver build and trimming, case build; over
  all attempts, the sum over the paths run with the case build counted once (it is measured once and added into
  every path's preparation).
- P1 mesh retries: failed TetGen calls (crash, error or 30-min time-out, followed by the next switch variant on
  the matched-surface mesh — surface preservation toggled, then a looser quality bound —, by a 15 % larger size
  in a size-search trial, or by the graded search repeated without surface preservation), plus remeshes redone
  with a tighter distance bound and fallbacks to the source surface, in the kept mesh's build or a failed one. A
  rejected remesh counts as one retry. Failed size-search trials (30-min TetGen time-outs included) are retries;
  successful trials that were not kept are not, and nothing in them is counted. Meshing is the only part of P1
  preparation that is retried.
- P1 solver retries: failed GPU solver paths (multigrid, more Krylov iterations, a second multigrid
  implementation, diagonal scaling; each its own record) plus failed CPU-solver runs (before the reduced GPU
  solver).
- P1 preparation: surface resizing (with its self-intersection checks and any remesh redo), sizing field,
  TetGen, boundary mapping, boundary-condition transfer and solver setup of the kept mesh; over all attempts,
  plus the recorded time of every failed TetGen call. (The stored `prep_seconds_incl_failed_tetgen` also adds
  failed self-intersection checks, which are either inside the kept build's resizing time already or belong to
  unused trials; it is in the xlsx as "preparation, all attempts, stored".)
- Solve: Newton iteration to the common relative tolerance 10⁻⁶; over all attempts, plus every failed attempt's
  recorded time (diffused Nitsche: its solve phases; P1: solver setup and solve until it failed, without the
  boundary mapping and transfer the attempt repeated) plus the wall time of every failed CPU-solver run (GNU time;
  includes its assembly and factorization). To 10⁻²: the same successful solve up to its first Newton iterate at
  the session tolerance 10⁻² (from the Newton log).
- CPU solves (P1: direct-factorization Newton solver; diffused Nitsche: direct linear solves) and reduced-GPU
  solves (full Newton, torch tangent and JAX-AMG) are marked in the cost tables; their times are not comparable
  with GPU times.

**Not recorded** (so not in the all-attempts totals): the boundary mapping and transfer a failed P1 solve attempt
redid before its setup; the resizing and setup time of a failed TetGen call's build; the warm-ups (excluded by
design); a session rerun of the batch driver (natural stone arch: its orchestrator was killed after a heap
corruption and an idle hang, and it resumed from the recorded attempts, so the wall time lost to the crashed job
is in no field); the CPU solver's in-run switch from modified to full Newton is inside its one run's time; the
time to 10⁻² of a failed attempt. No repair round ran.

**Medians** as in `summarize.py`: over every object with a value and over the unflagged ones; n per column.
Entries that converged only to a looser tolerance are left out of the metric medians; CPU and reduced-GPU times
are in the time medians. Time and retry medians are of the successful-attempt values. One deviation from
summarize.py: the loaf of bread's depth-8 Nitsche solve time (to 10⁻² only) is left out of the Nitsche solve
median (n 49 instead of 50; median over all objects 7.60 s instead of 7.61 s, over unflagged 5.10 s instead of
5.27 s).
"""


def write_md(path, tables, verification):
    md = ["# Diffused Nitsche vs body-fitted P1: comparison tables", "",
          "Computed by `tables.py` from the stored per-entry records; accuracy measures re-evaluated from the stored "
          "solutions by `recompute_metrics.py`. " + verification, ""]
    f = Fmt("md")
    for kind, d, table, cols in tables:
        md.append(md_table(kind, d, table, cols, f))
    md.append(DEFINITIONS)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# TeX.  Each fragment: \begin{landscape} ... \end{landscape}, for \input in an
# SI document that loads PACKAGES; no notes under the tables (the captions say
# what is needed, the procedure text the rest).  The accuracy and cost tables
# are longtables of exactly WIDTH: longtable does not stretch
# \extracolsep{\fill}, so every column has a fixed width, the object and flag
# columns absolute, the numeric columns a weight ("r:<w>") times a unit.  The
# DOF table is short enough for one page when its 50 rows are set in two
# halves side by side, as a plain table float.
PACKAGES = (("geometry", "page margins (preview only)"), ("pdflscape", "landscape pages"),
            ("longtable", "tables across pages"), ("booktabs", "rules"), ("array", "column types"),
            ("amsmath", "math"), ("xcolor", "gray rows; load as \\usepackage[table]{xcolor}"))
NAME_WIDTH = {"dofs": "4.6cm", "accuracy": "4.6cm", "cost": "3.0cm"}
FLAG_WIDTH = "1.05cm"
WIDTH = {"dofs": r"0.68\linewidth", "accuracy": r"\linewidth", "cost": r"\linewidth"}
FONT = {"dofs": r"\footnotesize", "accuracy": r"\footnotesize", "cost": r"\footnotesize"}
COLSEP = {"dofs": "5pt", "accuracy": "2.5pt", "cost": "2pt"}
UNIT = r"\SIbfColUnit"  # a length register the fragments share (allocated once)


def tex_header(cols, levels, f):
    out = []
    for lv in range(levels):
        labs = [c[0][lv]["tex"] for c in cols]
        if lv == levels - 1:
            out.append(" & ".join(r"\multicolumn{1}{@{}l}{%s}" % lab if i == 0 else lab
                                  for i, lab in enumerate(labs)) + r" \\")
            break
        if not any(labs):
            continue
        sp = spans(labs)
        out.append(" & ".join((r"\multicolumn{%d}{c}{%s}" % (k, lab) if k > 1 else lab) if lab else
                              " & ".join([""] * k) for lab, _, k in sp) + r" \\")
        # a rule reaching the table's right edge (@{}) is trimmed on the left only
        out.append(" ".join(r"\cmidrule(%s){%d-%d}" % ("l" if s + k == len(cols) else "lr", s + 1, s + k)
                            for lab, s, k in sp if lab))
    return out


def weight(a):
    return float(a.split(":")[1])


def tex_table(kind, d, table, cols):
    f = Fmt("tex")
    if kind == "dofs":
        return tex_dofs(d, table, cols)
    n, levels = len(cols), len(cols[0][0])
    wsum = sum(weight(a) for *_, a in cols if a.startswith("r:"))
    spec = []
    for *_, a in cols:
        if a == "name":
            spec.append(r">{\raggedright\arraybackslash}p{%s}" % NAME_WIDTH[kind])
        elif a == "c":
            spec.append(r">{\centering\arraybackslash}p{%s}" % FLAG_WIDTH)
        else:
            spec.append(r">{\raggedleft\arraybackslash}p{%g%s}" % (weight(a), UNIT))
    head = tex_header(cols, levels, f)
    out = [f"% {SHORT[kind].format(d=d)}: generated by experiments/bodyfitted_vs_nitsche/tables.py; do not edit.",
           "% Requires " + ", ".join(p for p, _ in PACKAGES if p != "geometry") + ".",
           r"\begin{landscape}", FONT[kind], r"\setlength{\tabcolsep}{%s}" % COLSEP[kind],
           r"\ifdefined%s\else\newlength{%s}\fi" % (UNIT, UNIT),
           # unit = (WIDTH - fixed columns - inter-column space) / sum of the weights
           r"\setlength{%s}{\dimexpr(%s-%s-%s-%d\tabcolsep)*100/%d\relax}" % (
               UNIT, WIDTH[kind], NAME_WIDTH[kind], FLAG_WIDTH, 2 * n - 2, round(100 * wsum)),
           r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
           r"\setlength{\LTcapwidth}{%s}" % WIDTH[kind],
           r"\begin{longtable}{@{}" + "".join(spec) + "@{}}",
           r"\caption{%s}\label{%s}\\" % (caption(f, kind, d), label(kind, d)), r"\toprule", *head, r"\midrule",
           r"\endfirsthead",
           r"\caption[]{%s (continued).}\\" % SHORT[kind].format(d=d), r"\toprule", *head, r"\midrule",
           r"\endhead",
           r"\midrule", r"\multicolumn{%d}{@{}r@{}}{\textit{Continued on next page}}\\" % n, r"\endfoot",
           r"\bottomrule", r"\endlastfoot"]
    for r in ordered(table):
        gray = r"\rowcolor{gray!15}" if kind == "accuracy" and gray_row(r, d) else ""
        out.append(gray + " & ".join(render_cell(get(r, d), f) for _, get, _, _ in cols) + r" \\")
    # The medians do not start a page on their own: the last three rows end in
    # \\* and the rule above the medians is a booktabs \midrule made unbreakable
    # (\midrule's \belowrulesep glue after the rule is a legal page break, and a
    # \nopagebreak after it comes too late).
    for i in (-1, -2, -3):
        out[i] += "*"
    out.append(r"\noalign{\penalty10000\vskip\aboverulesep\hrule height\lightrulewidth"
               r"\penalty10000\vskip\belowrulesep}")
    for i, (lab, cells, ns) in enumerate(median_rows(table, cols, d)):
        if i:
            out.append(r"\addlinespace")
        out.append(" & ".join([lab, ""] + [c or f.dash for c in cells]) + r" \\*")
        out.append(" & ".join([r"\hspace{1em}$n$", ""] + ns) + (r" \\*" if not i else r" \\"))
    out += [r"\end{longtable}", r"\end{landscape}", ""]
    return "\n".join(out)


def tex_dofs(d, table, cols):
    """The DOF table on one landscape page: the 50 rows in two halves side by
    side (left the first 25 in the evaluation table's order, right the rest),
    the two median blocks under them, left over all objects, right over the
    unflagged ones."""
    f = Fmt("tex")
    rows = ordered(table)
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    k = len(cols)
    one = r">{\raggedright\arraybackslash}p{3.7cm}c" + "r" * (k - 2)
    spec = "@{}" + one + r"@{\hspace{1.6em}}" + one + "@{}"
    head2 = (" & ".join([""] * 3) + r" & \multicolumn{3}{c}{Body-fitted P1}") * 1
    head = [head2 + " & " + head2 + r" \\",
            r"\cmidrule(lr){4-6} \cmidrule(l){%d-%d}" % (k + 4, k + 6),
            " & ".join([r"\multicolumn{1}{@{}l}{Object}"] + [c[0][-1]["tex"] for c in cols[1:]]
                       + ["Object"] + [c[0][-1]["tex"] for c in cols[1:]]) + r" \\"]
    body = []
    for i in range(half):
        a = [render_cell(get(left[i], d), f) for _, get, _, _ in cols]
        b = [render_cell(get(right[i], d), f) for _, get, _, _ in cols] if i < len(right) else [""] * k
        body.append(" & ".join(a + b) + r" \\")
    (la, ca, na), (lb, cb, nb) = median_rows(table, cols, d)
    meds = [" & ".join([la, ""] + [c or f.dash for c in ca] + [lb, ""] + [c or f.dash for c in cb]) + r" \\",
            " & ".join([r"\hspace{1em}$n$", ""] + na + [r"\hspace{1em}$n$", ""] + nb) + r" \\"]
    out = [f"% {SHORT['dofs'].format(d=d)}: generated by experiments/bodyfitted_vs_nitsche/tables.py; do not edit.",
           "% Requires " + ", ".join(p for p, _ in PACKAGES if p != "geometry") + ".",
           r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\caption{%s}\label{%s}" % (caption(f, "dofs", d), label("dofs", d)), r"\smallskip",
           r"\begin{tabular}{%s}" % spec, r"\toprule", *head, r"\midrule", *body, r"\midrule", *meds,
           r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{landscape}", ""]
    return "\n".join(out)


PREVIEW = r"""%% Standalone preview of the SI tables of the diffused Nitsche vs body-fitted P1
%% comparison: pdflatex tables_preview.tex (twice, for the longtable widths).
%% Generated by experiments/bodyfitted_vs_nitsche/tables.py.
%%
%% The fragments in tex/ are meant for \input in the SI document, which needs:
%s
\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pdflscape}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath}
\usepackage[table]{xcolor}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{3}  %% the SI text has Tables S2 and S3 before these
\begin{document}
%s
\end{document}
"""


def write_tex(dest, tables):
    texdir = dest / "tex"
    texdir.mkdir(exist_ok=True)
    frags, inputs = [], []
    for kind, d, table, cols in tables:
        name = f"tab_si_bf_{kind}_d{d}"
        body = tex_table(kind, d, table, cols)
        (texdir / f"{name}.tex").write_text(body, encoding="utf-8")
        frags.append(f"% {'=' * 76}\n% {name}.tex\n% {'=' * 76}\n" + body)
        inputs.append(r"\input{tex/%s}" % name)
    (dest / "tables.tex").write_text(
        "% All six SI tables of the diffused Nitsche vs body-fitted P1 comparison (the fragments of tex/).\n"
        "% Generated by experiments/bodyfitted_vs_nitsche/tables.py; requires "
        + ", ".join(p for p, _ in PACKAGES if p != "geometry") + ".\n\n" + "\n".join(frags), encoding="utf-8")
    pk = "\n".join(f"%%   \\usepackage{{{p}}}  %% {why}" if p != "xcolor" else "%%   \\usepackage[table]{xcolor}  %% gray rows"
                   for p, why in PACKAGES if p != "geometry")
    (dest / "tables_preview.tex").write_text(PREVIEW % (pk, "\n".join(inputs)), encoding="utf-8")


# xlsx: raw values, one per cell, formal headers.  A column is (header,
# value(row, d), number format, excluded-from-median(row, d)).
GPU_NAME = {"": "GPU", "c": "CPU", "L": "Reduced GPU"}


def remarks(row, d):
    out = []
    reason = missing(row, d)
    if reason:
        out.append("No body-fitted mesh: input surface self-intersects" if reason["code"] == "self_intersection"
                   else f"No body-fitted mesh: {reason['text']}")
    if loose_any(row, d):
        out.append("Diffused Nitsche converged only to the session tolerance 10⁻² (excluded from the medians)")
    if row["depths"][d]["note"]:
        out += list(row["notes"])
    return "; ".join(out) or None


def xlsx_columns(kind):
    base = [("Object", lambda r, d: object_name(r), None, None),
            ("Identifier", lambda r, d: r["object"], None, None),
            ("Flags", lambda r, d: flag_letters(r) or None, None, None),
            ("Flagged", lambda r, d: r["flag_any"], None, None),
            ("Remarks", remarks, None, None)]
    nit = lambda r, d: r["depths"][d]["nitsche"]  # noqa: E731
    mode = lambda m: (lambda r, d: r["depths"][d]["modes"][m])  # noqa: E731
    if kind == "dofs":
        cols = base + [("Diffused Nitsche: DOFs", lambda r, d: nit(r, d)["dofs"], "#,##0", None)]
        for m, ml in MODES:
            cols.append((f"Body-fitted P1, {ml.lower()}: DOFs", lambda r, d, m=m: mode(m)(r, d).get("p1_dofs"),
                         "#,##0", None))
        return cols
    if kind == "accuracy":
        cols = list(base)
        for m, ml in MODES:
            p = f"{ml}: "
            ex = (lambda r, d, m=m: mode(m)(r, d).get("loose"))
            for key, head, fmt in (("L2r", "η_rig (rigid-removed relative L² discrepancy)", "0.00%"),
                                   ("H1r", "η_rig^H¹ (rigid-removed relative H¹ discrepancy)", "0.00%"),
                                   ("H1s", "rigid-removed relative H¹-seminorm discrepancy", "0.00%"),
                                   ("L2", "η (relative L² discrepancy, no rigid removal)", "0.00%"),
                                   ("C", "compliance ratio ℓ_N/ℓ_P1", "0.000"),
                                   ("p95", "von Mises 95th-percentile ratio s₉₅", "0.000"),
                                   ("p99", "von Mises 99th-percentile ratio s₉₉", "0.000")):
                cols.append((p + head, lambda r, d, m=m, key=key: mode(m)(r, d).get(key), fmt, ex))
            cols += [(p + "converged only to a looser tolerance",
                      lambda r, d, m=m: mode(m)(r, d).get("loose") if mode(m)(r, d)["ok"] else None, None, None),
                     (p + "recomputed equals stored (bitwise)", lambda r, d, m=m: mode(m)(r, d).get("verified"),
                      None, None)]
        return cols
    t = "0.00"
    cols = base + [
        ("Diffused Nitsche: solver paths run", lambda r, d: nit(r, d)["retries"] + 1 if nit(r, d)["ok"] else None,
         "0", None),
        ("Diffused Nitsche: solver retries", lambda r, d: nit(r, d)["retries"], "0", None),
        ("Diffused Nitsche: preparation, successful attempt (s)", lambda r, d: nit(r, d)["prep"], t, None),
        ("Diffused Nitsche: preparation, all attempts (s)", lambda r, d: nit(r, d)["prep_all"], t, None),
        ("Diffused Nitsche: solve, successful attempt (s)", lambda r, d: nit(r, d)["solve"], t,
         lambda r, d: nit(r, d)["loose"]),
        ("Diffused Nitsche: solve, all attempts (s)", lambda r, d: nit(r, d)["solve_all"], t, None),
        ("Diffused Nitsche: solve to 10⁻², successful attempt (s)", lambda r, d: nit(r, d)["solve_1e2"], t, None),
        ("Diffused Nitsche: solver device", lambda r, d: GPU_NAME.get(nit(r, d)["device"]), None, None),
        ("Diffused Nitsche: converged only to a looser tolerance", lambda r, d: nit(r, d)["loose"], None, None)]
    for m, ml in MODES:
        p = f"Body-fitted P1, {ml.lower()}: "
        g = (lambda key, m=m: (lambda r, d: mode(m)(r, d).get(key) if mode(m)(r, d)["ok"] else None))
        cols += [(p + "mesh retries", g("mesh_retries"), "0", None),
                 (p + "TetGen meshes built", g("trial_meshes"), "0", None),
                 (p + "solver retries", g("solve_retries"), "0", None),
                 (p + "solver path", lambda r, d, m=m: " > ".join(map(str, mode(m)(r, d)["solve_path"]))
                  if mode(m)(r, d)["ok"] else None, None, None),
                 (p + "preparation, successful attempt (s)", g("prep"), t, None),
                 (p + "preparation, all attempts (s)", g("prep_all"), t, None),
                 (p + "preparation, all attempts, stored (s)", g("prep_all_stored"), t, None),
                 (p + "solve, successful attempt (s)", g("solve"), t, None),
                 (p + "solve, all attempts (s)", g("solve_all"), t, None),
                 (p + "solve to 10⁻², successful attempt (s)", g("solve_1e2"), t, None),
                 (p + "solver device", lambda r, d, m=m: GPU_NAME.get(mode(m)(r, d)["device"])
                  if mode(m)(r, d)["ok"] else None, None, None)]
    return cols


SHEET = {"dofs": "DOFs", "accuracy": "Accuracy", "cost": "Cost"}


def write_xlsx(path, table, tables):
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    grey = PatternFill("solid", fgColor="D9D9D9")
    for kind, d, _, _ in tables:
        ws = wb.create_sheet(f"Depth {d} {SHEET[kind]}")
        cols = xlsx_columns(kind)
        ws.append([h for h, *_ in cols])
        for c in ws[1]:
            c.font = Font(bold=True)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        rows = ordered(table)
        for r in rows:
            ws.append([get(r, d) for _, get, _, _ in cols])
            if kind == "accuracy" and gray_row(r, d):  # eta_rig > 10 % against some mesh, as in the PDF
                for c in ws[ws.max_row]:
                    c.fill = grey
        ws.append([])
        for lab, keep in (("Median (all objects)", lambda r: True),
                          ("Median (unflagged)", lambda r: r["flag_any"] is False)):
            med, cnt = [lab], ["n"]
            for _, get, _, ex in cols[1:]:
                vals = [v for r in rows if keep(r) and not (ex and ex(r, d)) for v in [get(r, d)]
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
                med.append(statistics.median(vals) if vals else None)
                cnt.append(len(vals) if vals else None)
            for line in (med, cnt):
                ws.append(line)
                for c in ws[ws.max_row]:
                    c.font = Font(bold=True)
        for j, (_, _, code, _) in enumerate(cols, start=1):
            if code:
                for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=j, max_col=j):
                    for c in row:
                        if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
                            c.number_format = code
        ws.freeze_panes = "B2"
        ws.column_dimensions["A"].width = 44
        ws.row_dimensions[1].height = 60
    wn = wb.create_sheet("Captions")
    f = Fmt("plain")
    for kind, d, _, _ in tables:
        wn.append([f"Depth {d} {SHEET[kind]} ({label(kind, d)})"])
        wn[wn.max_row][0].font = Font(bold=True)
        wn.append([caption(f, kind, d)])
        wn.append([])
    for line in DEFINITIONS.strip().splitlines():
        wn.append([line])
    for row in wn.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    wn.column_dimensions["A"].width = 160
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", default=str(summarize.DATA_ROOT / "output" / "bodyfitted_vs_nitsche"))
    ap.add_argument("--batch", default=str(summarize.DATA_ROOT / "output" / "agentic_batch50_opus"))
    ap.add_argument("--recomputed", default=str(HERE / "summary" / "tables" / "recomputed_metrics.json"))
    ap.add_argument("--collected", default=str(HERE / "summary" / "tables" / "collected.json"),
                    help="the collected records: written when --output exists, else read")
    ap.add_argument("--dest", default=str(HERE / "summary" / "tables"))
    args = ap.parse_args()
    out, dest, coll = Path(args.output), Path(args.dest), Path(args.collected)
    dest.mkdir(parents=True, exist_ok=True)
    rec = json.loads(Path(args.recomputed).read_text())
    if out.is_dir():
        table = collect(out, rec["entries"], Path(args.batch))
        coll.write_text(json.dumps({"source": f"tables.collect({out})", "rows": table}, indent=1) + "\n")
    else:
        print(f"{out} not found (archived): rendering from {coll}")
        table = json.loads(coll.read_text())["rows"]
    check_names(table)
    verification = (f"Recomputed vs stored (`comparison.json`): {rec['n_bitwise_equal']} of {rec['n_entries']} "
                    f"compared entries bit-for-bit equal in every checked metric (relative L² with and without "
                    f"rigid removal, rigid share, compliance, outside-octree fraction, Cauchy and von Mises "
                    f"relative L², s₉₅, s₉₉); maximum relative difference {rec['max_rel_diff']:.1e}.")
    tables = [(kind, d, table, make()) for d in DEPTHS for kind, make, _ in TABLES]
    write_md(dest / "tables.md", tables, verification)
    write_tex(dest, tables)
    write_xlsx(dest / "tables.xlsx", table, tables)
    print(f"{len(table)} sessions -> {dest}/tables.{{md,tex,xlsx}}, tables_preview.tex, tex/")


if __name__ == "__main__":
    main()
