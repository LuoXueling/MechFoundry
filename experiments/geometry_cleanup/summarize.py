"""Summary table of all clean-up runs (CPU).

    python summarize.py [--tag TAG]

Reads <OUT>/<object>[/<tag>]/results.json for every object of the batch and writes
<OUT>/summary[_<tag>].tsv and .md: iterations run (design updates), stop reason
(threshold = the p99 bound, max = the maximum bound, cap, error, missing), final distance/h (the stop metric: symmetric
area-weighted p99), its p95 and max, V/V0, SE/SE0 of the final (solved) shape, the
effective gd_lr, and whether a uniform material stand-in replaced per-part materials.
An object without results is listed as "missing", never dropped.
"""

import argparse
import json
import os

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
a = ap.parse_args()

COLS = ["object", "updates", "stop", "d_p99/h", "d_p95/h", "d_max/h", "V/V0", "SE/SE0",
        "gd_lr_eff", "h [m]", "standin", "error"]
rows = []
for name in sorted(C.runs()):
    rd = os.path.join(C.obj_dir(name), a.tag) if a.tag else C.obj_dir(name)
    p = os.path.join(rd, "results.json")
    st = json.load(open(os.path.join(C.obj_dir(name), "setup.json")))
    if not os.path.isfile(p):
        rows.append([name, "", "missing", "", "", "", "", "", "", f"{st['h']:.3g}",
                     "yes" if st["standin"] else "no", ""])
        continue
    r = json.load(open(p))
    f = lambda x, fmt="%.3f": "" if x is None else fmt % x  # noqa: E731
    rows.append([name, r["updates"], r["stop_reason"], f(r["final_d_h"]), f(r["final_d_p95_h"]),
                 f(r["final_d_max_h"]), f(r["vol_ratio"]), f(r["se_ratio"]),
                 f((r.get("step") or {}).get("gd_lr_eff"), "%.3g"), f"{r['h']:.3g}",
                 "yes" if r["standin"] else "no", r["error"][:160]])

suf = f"_{a.tag}" if a.tag else ""
with open(os.path.join(C.OUT, f"summary{suf}.tsv"), "w") as fh:
    fh.write("\t".join(COLS) + "\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(C.OUT, f"summary{suf}.md"), "w") as fh:
    fh.write("| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS) + "\n")
    for r in rows:
        fh.write("| " + " | ".join(str(x) for x in r) + " |\n")
from collections import Counter  # noqa: E402
print(open(os.path.join(C.OUT, f"summary{suf}.md")).read())
print("stop reasons:", dict(Counter(r[2] for r in rows)),
      "| stand-ins:", sum(r[10] == "yes" for r in rows))
