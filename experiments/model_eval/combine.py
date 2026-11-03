"""Combine the three evaluators' ratings into one table with a majority vote.

    python combine.py          (a Python with openpyxl, e.g. the base conda env)
    python combine.py --evals astra_high --out astra_high --runs-fallback /tmp/batch_extract

Reads eval_<k>.json from --evals (default: this folder) and writes the tables to --out (default:
this folder); the label mappings are always mappings/mapping_<k>.json, since every round reuses
the same packets.  A batch root missing from the repo (archived) is read from --runs-fallback, a
folder holding its extract under the same repo-relative path (output/<batch>/_batches and the
runs' metrics.json are all this script needs).

Reads eval_<k>.json + mappings/mapping_<k>.json (k = 1..3) and each run's metrics.json cost_usd
(the Claude Code CLI's total_cost_usd for the coding agent's session at API prices; the workflow's
own DeepSeek calls are not in it).  Writes evaluation_table.md and evaluation_table.xlsx.

A cell reads "H/M/H → H": evaluators 1/2/3, then the vote.  The vote is the rating at least two
evaluators gave; when all three differ it is the median (the lower one if only two graded), marked with *."""
import argparse, json
from pathlib import Path
from build_packets import HERE, MAPPINGS, MODELS, REPO, categories, latest_runs

ORDER = {"Low": 0, "Moderate": 1, "High": 2}
ABBR = {"High": "H", "Moderate": "M", "Low": "L", "n/a": "–", None: "?"}
CATS = ("geometry", "bc", "material")


def vote(rs):
    for r in set(rs):
        if rs.count(r) >= 2:
            return r, False
    graded = sorted((r for r in rs if r in ORDER), key=ORDER.get)
    return (graded[(len(graded) - 1) // 2] if graded else "n/a"), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, default=HERE, help="folder with eval_<k>.json")
    ap.add_argument("--out", type=Path, default=HERE, help="folder for evaluation_table.*")
    ap.add_argument("--runs-fallback", type=Path, default=None,
                    help="repo-like folder with extracts of archived batch roots")
    a = ap.parse_args()
    evals, out = (HERE / a.evals).resolve(), (HERE / a.out).resolve()

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
    cats = categories()
    runs = {m: latest_runs(root, base(root)) for m, (root, _) in MODELS.items()}
    ev = {}                                     # (k, model, object) -> entry
    ks = sorted(int(p.stem.split("_")[1]) for p in evals.glob("eval_*.json"))
    for k in ks:
        inv = {lab: m for m, lab in json.load(open(MAPPINGS / f"mapping_{k}.json")).items()}
        for e in json.load(open(evals / f"eval_{k}.json")):
            ev[(k, inv[e["label"]], e["object"])] = e
    models = list(MODELS)
    hdr = ["category", "object"] + [f"{m}_{c}" for m in models for c in CATS + ("cost_usd",)]
    rows, md = [], []
    md.append("| Category | Object | " + " | ".join(
        f"{m.capitalize()} {c}" for m in models for c in ("geom.", "BC", "mat.", "cost")) + " |")
    md.append("|---|---|" + "---|" * (4 * len(models)))
    tally = {m: {c: {"High": 0, "Moderate": 0, "Low": 0, "n/a": 0} for c in CATS} for m in models}
    cost = {m: 0.0 for m in models}
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
                usd = met.get("cost_usd") or 0.0
                cost[m] += usd
                wall[m] += met.get("wall_clock_s") or 0.0
                row.append(f"{usd:.2f}"); cells.append(f"${usd:.2f}")
            rows.append(row)
            md.append(f"| {slug.replace('_', ' ')} | {o} | " + " | ".join(cells) + " |")

    def hml(t):
        return f"{t['High']}/{t['Moderate']}/{t['Low']}" + (f" (+{t['n/a']} –)" if t["n/a"] else "")

    def hms(sec):
        h, m_ = divmod(round(sec / 60), 60)
        return f"{h} h {m_:02d} min"
    # Two bottom rows: the votes counted per column (High/Moderate/Low, "+k –" not
    # reached) with the model's total cost, then the model's total wall-clock time
    # over all 50 runs (metrics.json wall_clock_s), under its cost column.
    tot, tim = ["total", "H/M/L; cost"], ["total", "time, 50 objects"]
    tot_md, tim_md = [], []
    for m in models:
        tot += [hml(tally[m][c]) for c in CATS] + [f"{cost[m]:.2f}"]
        tim += ["", "", "", f"{wall[m]:.0f}"]
        tot_md += [f"**{hml(tally[m][c])}**" for c in CATS] + [f"**${cost[m]:.2f}**"]
        tim_md += ["", "", "", f"**{hms(wall[m])}**"]
    rows += [tot, tim]
    md.append("| **Total** | **H/M/L; cost** | " + " | ".join(tot_md) + " |")
    md.append("| **Total** | **time, 50 objects** | " + " | ".join(tim_md) + " |")
    md.append("")
    md.append("Cells: evaluator 1/2/3 → majority vote (H high, M moderate, L low, – not reached; "
              "* all three differed, median taken). Cost: the coding agent's Claude session at API "
              "prices (CLI total_cost_usd); the FEM workflow's own API calls are excluded.")
    md.append("")
    md.append("| Model | Geometry H/M/L | BC H/M/L | Material H/M/L | Total cost | Mean cost |")
    md.append("|---|---|---|---|---|---|")
    n = sum(len(o) for _, o in cats)
    for m in models:
        t = tally[m]
        md.append(f"| {m.capitalize()} | " + " | ".join(
            f"{t[c]['High']}/{t[c]['Moderate']}/{t[c]['Low']}" + (f" (+{t[c]['n/a']} n/a)" if t[c]['n/a'] else "")
            for c in CATS) + f" | ${cost[m]:.2f} | ${cost[m] / n:.2f} |")
    (out / "evaluation_table.md").write_text("\n".join(md) + "\n")
    write_xlsx(out / "evaluation_table.xlsx", models, rows, tally, cost, wall, n)
    print("\n".join(md[-(len(models) + 2):]))


def write_xlsx(path, models, rows, tally, cost, wall, n):
    """The table as a workbook: sheet "Evaluation" (two header rows, the model group
    merged over its four columns; costs as numbers; the two total rows bold) and sheet
    "Summary" (per model: votes per category, total and mean cost, total time)."""
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
        ws.cell(1, c0, m.capitalize())
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
                else:
                    v = float(v)
                    ws.cell(r, c).number_format = '"$"#,##0.00'
            elif c == 1 and not total:
                v = v.replace("_", " ")
            ws.cell(r, c, v)
            if total:
                ws.cell(r, c).font = bold
            if c > 2:
                ws.cell(r, c).alignment = Alignment(horizontal="center")
    note = 4 + len(rows)
    ws.cell(note, 1, "Cells: evaluator 1/2/3 → majority vote (H high, M moderate, L low, – not reached; "
                     "* all three differed, median taken). Cost: the coding agent's Claude session at "
                     "API prices (CLI total_cost_usd); the FEM workflow's own API calls are excluded. "
                     "Time: the sum of the 50 runs' wall-clock times.")
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 20, 46
    for c in range(3, 3 + 4 * len(models)):
        ws.column_dimensions[get_column_letter(c)].width = 15 if (c - 2) % 4 else 11
    ws.freeze_panes = "C3"
    sm = wb.create_sheet("Summary")
    sm.append(["Model", "Geometry H/M/L", "BC H/M/L", "Material H/M/L", "Total cost (USD)",
               "Mean cost (USD)", "Total time"])
    for m in models:
        t = tally[m]
        hm = divmod(round(wall[m] / 60), 60)
        sm.append([m.capitalize()] + [f"{t[c]['High']}/{t[c]['Moderate']}/{t[c]['Low']}"
                                      + (f" (+{t[c]['n/a']} not reached)" if t[c]["n/a"] else "")
                                      for c in CATS]
                  + [round(cost[m], 2), round(cost[m] / n, 2), f"{hm[0]} h {hm[1]:02d} min"])
    for c in sm[1]:
        c.font, c.fill = bold, head_fill
    for r in sm.iter_rows(min_row=2):
        r[4].number_format = r[5].number_format = '"$"#,##0.00'
    for c, w in zip("ABCDEFG", (10, 24, 20, 24, 16, 15, 14)):
        sm.column_dimensions[c].width = w
    wb.save(path)


if __name__ == "__main__":
    main()
