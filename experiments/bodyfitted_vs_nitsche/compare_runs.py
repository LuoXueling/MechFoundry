#!/usr/bin/env python
"""Compare two output roots of this experiment entry by entry (e.g. a fresh
verification run against the stored results): final status, the path each
entry took (Nitsche rung / backend / flags; P1 device and rung), and the
change of every headline metric.  An entry matches when its status (present or
not, looser-tolerance flag) is the same and the metrics agree to ``--tol``
(absolute; the solves stop at rel 1e-6).

    python experiments/bodyfitted_vs_nitsche/compare_runs.py NEW_ROOT [--reference ROOT] [--out FILE.md]
"""
import argparse
import json
import sys
from pathlib import Path

METRICS = ("rel_L2_rigid_removed", "rel_L2", "compliance_ratio")
STRESS = ("rel_L2_cauchy", "rel_L2_cauchy_away_from_patches")


def nitsche_path(nit):
    if not nit:
        return "-"
    if not nit.get("ok"):
        return "failed"
    acc = [x for x in nit.get("attempts") or [] if x.get("error") is None and not x.get("skipped")]
    last = acc[-1] if acc else {}
    backend = nit.get("backend") or last.get("backend") or (
        "pardiso-direct" if nit.get("direct_solver") else
        (nit.get("overrides") or {}).get("cg_backend_assembled") or "jaxamg")
    parts = [nit.get("rung") or backend]
    if nit.get("unit_diagonal"):
        parts.append("unit-diagonal")
    if last.get("load_steps"):
        parts.append(f"ls {last['load_steps']}")
    if (nit.get("rtol") or 1e-6) > 1e-6:
        parts.append(f"rtol {nit['rtol']:g}")
    return " ".join(parts)


def p1_path(res):
    if not res.get("ok"):
        return "failed: " + str(res.get("error"))[:60]
    s = res.get("solve") or {}
    solver = s.get("solver") or "gpu"
    if solver == "gpu":
        return f"gpu {s.get('rung') or s.get('backend') or 'jaxamg'}{' unit-diagonal' if s.get('unit_diagonal') else ''}"
    return f"{solver} ({'full Newton' if 'full Newton' in str(s.get('method')) else 'modified Newton'})" \
        if solver == "cpu" else solver


def entries(root: Path, name: str):
    path = root / name / "comparison.json"
    if not path.is_file():
        return None
    rep = json.loads(path.read_text())
    out = {}
    for d, D in (rep.get("depths") or {}).items():
        nit = D.get("nitsche") or {}
        out[f"d{d} nitsche"] = {"ok": bool(nit.get("ok")), "path": nitsche_path(nit),
                                "looser": (nit.get("rtol") or 1e-6) > 1e-6, "m": {}}
        for m, res in (D.get("modes") or {}).items():
            vals = {k: res.get(k) for k in METRICS}
            vals.update({k: (res.get("stress") or {}).get(k) for k in STRESS})
            out[f"d{d} {m}"] = {"ok": bool(res.get("ok")) and res.get("rel_L2") is not None, "path": p1_path(res),
                                "looser": (nit.get("rtol") or 1e-6) > 1e-6, "m": vals,
                                "error": res.get("error")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_batch import DATA_ROOT
    ap.add_argument("--reference", default=str(DATA_ROOT / "output" / "bodyfitted_vs_nitsche"))
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out")
    args = ap.parse_args()
    new_root, ref_root = Path(args.new), Path(args.reference)
    lines = ["| session | entry | path (new) | path (reference) | status same | max abs metric change | matches |",
             "|---|---|---|---|---|---|---|"]
    n_ok = n_all = 0
    for sdir in sorted(p for p in new_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        new, ref = entries(new_root, sdir.name), entries(ref_root, sdir.name)
        if new is None:
            lines.append(f"| {sdir.name} | (no comparison.json in the new run) | | | no | | no |")
            n_all += 1
            continue
        for key in sorted(set(new) | set(ref or {})):
            a, b = new.get(key), (ref or {}).get(key)
            same = a is not None and b is not None and a["ok"] == b["ok"] and a["looser"] == b["looser"]
            diffs = [abs(a["m"][k] - b["m"][k]) for k in a["m"] if a and b and a["m"].get(k) is not None
                     and b["m"].get(k) is not None]
            dmax = max(diffs) if diffs else None
            match = same and (dmax is None or dmax <= args.tol)
            n_all += 1
            n_ok += match
            lines.append(f"| {sdir.name} | {key} | {a['path'] if a else '-'} | {b['path'] if b else '-'} | "
                         f"{'yes' if same else 'no'} | {'-' if dmax is None else f'{dmax:.1e}'} | "
                         f"{'yes' if match else 'NO'} |")
    text = f"{n_ok} of {n_all} entries match (status, and metrics within {args.tol:g} absolute)\n\n" + "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text("# Verification against the reference run\n\n" + text + "\n")


if __name__ == "__main__":
    main()
