"""Combine the three evaluators' ratings into one table with a majority vote.

    python combine.py          (a Python with openpyxl, e.g. the base conda env)
    python combine.py --evals astra_high --out astra_high --runs-fallback /tmp/batch_extract
    python combine.py --round open_models --family astra

A round (rounds/<name>.json, default the paper round) names the graded models in column order,
their batch roots and display names, the label mappings and each evaluator family's ratings
folder.  Reads eval_<k>.json from --evals (default: the family's ratings folder; for the paper
round's default family, Fable, that is this folder) and writes the tables to --out (default: the
--evals folder); the label mappings are the round's mapping_<k>.json, which every family of a
round shares since all of them read the same packets.  A batch root missing from the repo
(archived) is read from --runs-fallback, a folder holding its extract under the same
repo-relative path (output/<batch>/_batches and the runs' metrics.json are all this script needs).

Reads eval_<k>.json + mapping_<k>.json and each run's metrics.json cost_usd and wall_clock_s.
What cost_usd is depends on the harness, and the round's cost_note says it under the table: for a
Claude Code batch the CLI's total_cost_usd for the coding agent's session at API prices; for a
Codex batch the batch runner's price from the token counts and its dated price table (no bill
exists), which is null for a model or date the table has no price for.  Such a run shows "n/p"
and is left out of the cost total and the mean (the total says how many it misses).  The
workflow's own DeepSeek calls are in neither.  Writes evaluation_table.md and evaluation_table.xlsx.

A cell reads "H/M/H → H": evaluators 1/2/3, then the vote.  The vote is the rating at least two
evaluators gave; when all three differ it is the median (the lower one if only two graded), marked with *."""
import argparse, json
from pathlib import Path
from build_packets import HERE, REPO, latest_runs
from round_config import DEFAULT_ROUND, NUMBER_WORDS, Round

ORDER = {"Low": 0, "Moderate": 1, "High": 2}
ABBR = {"High": "H", "Moderate": "M", "Low": "L", "n/a": "–", None: "?"}
CATS = ("geometry", "bc", "material")


def vote(rs):
    for r in set(rs):
        if rs.count(r) > len(rs) / 2:           # of three evaluators: at least two
            return r, False
    graded = sorted((r for r in rs if r in ORDER), key=ORDER.get)
    return (graded[(len(graded) - 1) // 2] if graded else "n/a"), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=DEFAULT_ROUND, help="rounds/<name>.json, or a path to one")
    ap.add_argument("--family", default=None, help="evaluator family (default: the round's first)")
    ap.add_argument("--evals", type=Path, default=None,
                    help="folder with eval_<k>.json (default: the family's ratings folder)")
    ap.add_argument("--out", type=Path, default=None,
                    help="folder for evaluation_table.* (default: the --evals folder)")
    ap.add_argument("--runs-fallback", type=Path, default=None,
                    help="repo-like folder with extracts of archived batch roots")
    a = ap.parse_args()
    rnd = Round(a.round)
    evals = (HERE / a.evals).resolve() if a.evals else rnd.ratings_dir(a.family or rnd.default_family())
    out = (HERE / a.out).resolve() if a.out else evals

    def base(root):
        # A live batch root is always read from the repo (the original behaviour); only an
        # archived one falls back, and without --runs-fallback that is a clear error rather
        # than a traceback.
        if (REPO / root / "_batches").exists():
            return REPO
        if a.runs_fallback and (a.runs_fallback / root / "_batches").exists():
            return a.runs_fallback
        raise SystemExit(
            f"combine.py: {REPO / root} has no _batches/ (archived to output/archives/"
            f"{root.split('/')[-1]}.tar.zst?). Extract its _batches/ and */metrics.json, e.g.\n"
            f"  tar -I zstd -xf output/archives/{root.split('/')[-1]}.tar.zst -C DIR/output "
            f"--wildcards '{root.split('/')[-1]}/_batches/*' '{root.split('/')[-1]}/*/metrics.json'\n"
            f"and pass --runs-fallback DIR.")
    cats = rnd.categories()
    runs = {m: latest_runs(rnd.root(m), base(rnd.root(m))) for m in rnd.models}
    ev = {}                                     # (k, model, object) -> entry
    ks = sorted(int(p.stem.split("_")[1]) for p in evals.glob("eval_*.json"))
    for k in ks:
        inv = {lab: m for m, lab in rnd.load_mapping(k).items()}
        for e in json.load(open(evals / f"eval_{k}.json")):
            ev[(k, inv[e["label"]], e["object"])] = e
    models, name = rnd.models, rnd.display
    rows, md = [], []
    md.append("| Category | Object | " + " | ".join(
        f"{name(m)} {c}" for m in models for c in ("geom.", "BC", "mat.", "cost")) + " |")
    md.append("|---|---|" + "---|" * (4 * len(models)))
    tally = {m: {c: {"High": 0, "Moderate": 0, "Low": 0, "n/a": 0} for c in CATS} for m in models}
    cost = {m: 0.0 for m in models}
    unpriced = {m: 0 for m in models}
    wall = {m: 0.0 for m in models}
    for slug, objs in cats:
        for o in objs:
            row, cells = [slug, o], []
            for m in models:
                for c in CATS:
                    rs = [ev.get((k, m, o), {}).get(c) for k in ks]
                    v, tie = vote(rs)
                    tally[m][c][v] = tally[m][c].get(v, 0) + 1
                    s = "/".join(ABBR.get(r, "?") for r in rs) + " → " + ABBR[v] + ("*" if tie else "")
                    row.append(s); cells.append(s)
                run, _ = runs[m][o]
                met = json.load(open(run / "metrics.json"))
                usd = met.get("cost_usd")
                wall[m] += met.get("wall_clock_s") or 0.0
                if usd is None:                 # no price for this run (Codex price table)
                    unpriced[m] += 1
                    row.append("n/p"); cells.append("n/p")
                else:
                    cost[m] += usd
                    row.append(f"{usd:.2f}"); cells.append(f"${usd:.2f}")
            rows.append(row)
            md.append(f"| {slug.replace('_', ' ')} | {o} | " + " | ".join(cells) + " |")

    def hml(t):
        return f"{t['High']}/{t['Moderate']}/{t['Low']}" + (f" (+{t['n/a']} –)" if t["n/a"] else "")

    def hms(sec):
        h, m_ = divmod(round(sec / 60), 60)
        return f"{h} h {m_:02d} min"

    def np_(m):                                 # a cost total that misses k unpriced runs
        return f" (+{unpriced[m]} n/p)" if unpriced[m] else ""
    n = sum(len(o) for _, o in cats)
    # Two bottom rows: the votes counted per column (High/Moderate/Low, "+k –" not
    # reached) with the model's total cost, then the model's total wall-clock time
    # over all its runs (metrics.json wall_clock_s), under its cost column.
    tot, tim = ["total", "H/M/L; cost"], ["total", f"time, {n} objects"]
    tot_md, tim_md = [], []
    for m in models:
        tot += [hml(tally[m][c]) for c in CATS] + [f"{cost[m]:.2f}" + np_(m)]
        tim += ["", "", "", f"{wall[m]:.0f}"]
        tot_md += [f"**{hml(tally[m][c])}**" for c in CATS] + [f"**${cost[m]:.2f}**" + np_(m)]
        tim_md += ["", "", "", f"**{hms(wall[m])}**"]
    rows += [tot, tim]
    md.append("| **Total** | **H/M/L; cost** | " + " | ".join(tot_md) + " |")
    md.append(f"| **Total** | **time, {n} objects** | " + " | ".join(tim_md) + " |")
    md.append("")
    legend = (f"Cells: evaluator {'/'.join(map(str, ks))} → majority vote (H high, M moderate, "
              f"L low, – not reached; * all {NUMBER_WORDS[len(ks)].lower()} differed, median taken).")
    md.append(f"{legend} {rnd.cost_note}")
    md.append("")
    md.append("| Model | Geometry H/M/L | BC H/M/L | Material H/M/L | Total cost | Mean cost |")
    md.append("|---|---|---|---|---|---|")
    for m in models:
        t = tally[m]
        priced = n - unpriced[m]
        mean = f"${cost[m] / priced:.2f}" if priced else "n/p"
        md.append(f"| {name(m)} | " + " | ".join(
            f"{t[c]['High']}/{t[c]['Moderate']}/{t[c]['Low']}" + (f" (+{t[c]['n/a']} n/a)" if t[c]['n/a'] else "")
            for c in CATS) + f" | ${cost[m]:.2f}{np_(m)} | {mean} |")
    (out / "evaluation_table.md").write_text("\n".join(md) + "\n")
    note = f"{legend} {rnd.cost_note} Time: the sum of the {n} runs' wall-clock times."
    write_xlsx(out / "evaluation_table.xlsx", models, name, rows, tally, cost, unpriced, wall, n, note)
    print("\n".join(md[-(len(models) + 2):]))


def write_xlsx(path, models, name, rows, tally, cost, unpriced, wall, n, note):
    """The table as a workbook: sheet "Evaluation" (two header rows, the model group
    merged over its four columns; costs as numbers, "n/p" where a run has no price; the two
    total rows bold) and sheet "Summary" (per model: votes per category, total and mean cost
    over the priced runs, total time)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    bold, centre = Font(bold=True), Alignment(horizontal="center", vertical="center")
    head_fill = PatternFill("solid", fgColor="E8EAF0")
    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluation"
    ws.cell(1, 1, "Category"); ws.cell(1, 2, "Object")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=2)
    for i, m in enumerate(models):
        c0 = 3 + 4 * i
        ws.cell(1, c0, name(m))
        ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c0 + 3)
        for j, sub in enumerate(("Geometry", "BC", "Material", "Cost (USD)")):
            ws.cell(2, c0 + j, sub)
    for r in (1, 2):
        for c in range(1, 3 + 4 * len(models)):
            ws.cell(r, c).font, ws.cell(r, c).alignment, ws.cell(r, c).fill = bold, centre, head_fill
    for k, row in enumerate(rows):
        r = 3 + k
        total = row[0] == "total"
        for c, v in enumerate(row, 1):
            if c > 2 and (c - 2) % 4 == 0 and v != "":
                if row[1].startswith("time"):
                    h, m_ = divmod(round(float(v) / 60), 60)
                    v = f"{h} h {m_:02d} min"
                elif "n/p" not in v:            # an unpriced run (or a total missing some) stays text
                    v = float(v)
                    ws.cell(r, c).number_format = '"$"#,##0.00'
            elif c == 1 and not total:
                v = v.replace("_", " ")
            ws.cell(r, c, v)
            if total:
                ws.cell(r, c).font = bold
            if c > 2:
                ws.cell(r, c).alignment = Alignment(horizontal="center")
    ws.cell(4 + len(rows), 1, note)
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 20, 46
    for c in range(3, 3 + 4 * len(models)):
        ws.column_dimensions[get_column_letter(c)].width = 15 if (c - 2) % 4 else 11
    ws.freeze_panes = "C3"
    sm = wb.create_sheet("Summary")
    # The unpriced-runs column only exists when some run has no price (a Codex round), so a
    # round where every run is priced (the paper round) keeps its sheet as it was.
    any_np = any(unpriced.values())
    sm.append(["Model", "Geometry H/M/L", "BC H/M/L", "Material H/M/L", "Total cost (USD)",
               "Mean cost (USD)", "Total time"] + (["Unpriced runs"] if any_np else []))
    for m in models:
        t = tally[m]
        hm = divmod(round(wall[m] / 60), 60)
        priced = n - unpriced[m]
        sm.append([name(m)] + [f"{t[c]['High']}/{t[c]['Moderate']}/{t[c]['Low']}"
                               + (f" (+{t[c]['n/a']} not reached)" if t[c]["n/a"] else "")
                               for c in CATS]
                  + [round(cost[m], 2), round(cost[m] / priced, 2) if priced else "n/p",
                     f"{hm[0]} h {hm[1]:02d} min"] + ([unpriced[m]] if any_np else []))
    for c in sm[1]:
        c.font, c.fill = bold, head_fill
    for r in sm.iter_rows(min_row=2):
        r[4].number_format = r[5].number_format = '"$"#,##0.00'
    for c, w in zip("ABCDEFG" + ("H" if any_np else ""), (10, 24, 20, 24, 16, 15, 14, 14)):
        sm.column_dimensions[c].width = w
    wb.save(path)


if __name__ == "__main__":
    main()
