#!/usr/bin/env python
"""Re-evaluate every compared (depth, mode) entry from the stored solve outputs,
adding the H1 errors, and check the stored metrics against the recomputation.

    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      python experiments/bodyfitted_vs_nitsche/recompute_metrics.py [--objects a b ...]

Env mechfoundry-warp117, one idle GPU.  For each session folder under
``--output`` (default output/bodyfitted_vs_nitsche) with a ``comparison.json``,
every entry that comparison.json compared is evaluated again by the production
``compare_bodyfitted_nitsche.evaluate_p1`` on the stored files only: the P1
solution ``p1_solution_d<D>_<mode>.npz`` and the last Nitsche VTU of the depth's
accepted run.  ``evaluate_p1`` now also returns the H1 errors
(``h1_errors``), from the same quadrature points, the same per-method
displacement gradients and the same best-fit rigid motion r* as the existing
metrics.

The session's BC regions and source surface are not loaded: they enter only the
"away from patches" stress subset and the mesh/source volume ratio, which are
not recomputed here (the stub gives no regions, so its "away" metric equals the
full-body one and is ignored).  Everything else is compared with
comparison.json, value by value (max relative difference reported).

Writes ``--dest`` (default experiments/bodyfitted_vs_nitsche/summary/tables/
recomputed_metrics.json): per entry the recomputed metrics, the H1 errors, the
stored values and the relative differences.  Resumable: entries already in the
file are kept unless ``--redo``.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_bodyfitted_nitsche as cbn  # noqa: E402

import torch  # noqa: E402

# (recomputed key path, stored key path) of every metric checked.
CHECKED = (
    ("rel_L2_rigid_removed",), ("rel_L2",), ("rigid_share_of_err2",), ("compliance_ratio",),
    ("outside_octree_volume_fraction",),
    ("stress", "rel_L2_cauchy"), ("stress", "rel_L2_von_mises"),
    ("stress", "von_mises_p95_ratio"), ("stress", "von_mises_p99_ratio"),
)


def _get(d, path):
    for k in path:
        d = (d or {}).get(k)
    return d


def recompute_entry(out_dir: Path, depth: str, mode: str, nit: dict, order: int):
    vtus = sorted((Path(nit["run_dir"]) / "vtu").glob("solution_t*.vtu"))
    field = cbn.OctreeQ1Field(cbn.read_vtu(vtus[-1]), "cuda")
    stub = SimpleNamespace(args=SimpleNamespace(eval_order=order), inputs=SimpleNamespace(regions=[]),
                           h_by_depth={int(depth): 1.0}, depth=int(depth), depth_stats={},
                           source=SimpleNamespace(volume=1.0, faces=None, source_face=None))
    t0 = time.perf_counter()
    res = cbn.evaluate_p1(stub, f"d{depth}_{mode}", cbn.stored_p1(out_dir / f"p1_solution_d{depth}_{mode}.npz"),
                          field)
    res["seconds"] = time.perf_counter() - t0
    res["vtu"] = str(vtus[-1])
    del field
    cbn.free_gpu()
    return res


def check_rigid_gradient():
    """grad r* from ``rigid_gradient`` against r* itself: r(x_i) - r(x_j) = W (x_i - x_j)."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand((200, 3), generator=g, dtype=torch.float64)
    d = torch.rand((200, 3), generator=g, dtype=torch.float64)
    w = torch.rand(200, generator=g, dtype=torch.float64)
    p, r = cbn.rigid_fit(x, d, w)
    W = cbn.rigid_gradient(p)
    err = (r[1:] - r[:1]) - (x[1:] - x[:1]) @ W.T
    return float(err.abs().max() / r.abs().max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", default=str(cbn.DATA_ROOT / "output" / "bodyfitted_vs_nitsche"))
    ap.add_argument("--dest", default=str(HERE / "summary" / "tables" / "recomputed_metrics.json"))
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(dest.read_text()) if dest.is_file() and not args.redo else {"entries": {}}
    cbn.wp.init()
    data["rigid_gradient_check_max_rel"] = check_rigid_gradient()
    print(f"[check] rigid_gradient vs rigid_fit: max rel {data['rigid_gradient_check_max_rel']:.1e}", flush=True)
    names = args.objects or sorted(p.parent.name for p in out.glob("*/comparison.json")
                                   if not p.parent.name.startswith("_"))
    for name in names:
        rep = json.loads((out / name / "comparison.json").read_text())
        order = int(((rep.get("environment_evaluate") or {}).get("options") or {}).get("eval_order") or 2)
        for d, D in rep["depths"].items():
            nit = D.get("nitsche") or {}
            for m, stored in (D.get("modes") or {}).items():
                key = f"{name}/d{d}/{m}"
                if key in data["entries"] or not (stored.get("ok") and "rel_L2" in stored and nit.get("ok")):
                    continue
                res = recompute_entry(out / name, d, m, nit, order)
                diff = {}
                for path in CHECKED:
                    a, b = _get(res, path), _get(stored, path)
                    diff[".".join(path)] = (None if a is None or b is None else
                                            0.0 if a == b else abs(a - b) / max(abs(b), 1e-300))
                data["entries"][key] = {
                    "recomputed": {".".join(p): _get(res, p) for p in CHECKED},
                    "h1": res["h1"], "stored": {".".join(p): _get(stored, p) for p in CHECKED},
                    "rel_diff": diff, "bitwise_equal": all(v == 0.0 for v in diff.values() if v is not None),
                    "seconds": res["seconds"], "npz": str(out / name / f"p1_solution_d{d}_{m}.npz"),
                    "vtu": res["vtu"]}
                worst = max((v for v in diff.values() if v is not None), default=0.0)
                print(f"[{key}] H1r {res['h1']['rel_H1_rigid_removed']:.4g}  L2r {res['rel_L2_rigid_removed']:.4g}"
                      f"  max rel diff vs stored {worst:.1e}  ({res['seconds']:.1f} s)", flush=True)
                dest.write_text(json.dumps(data, indent=1))
    ent = data["entries"].values()
    data["n_entries"] = len(data["entries"])
    data["n_bitwise_equal"] = sum(1 for e in ent if e["bitwise_equal"])
    data["max_rel_diff"] = max((v for e in ent for v in e["rel_diff"].values() if v is not None), default=0.0)
    dest.write_text(json.dumps(data, indent=1))
    print(f"{data['n_entries']} entries, {data['n_bitwise_equal']} bit-for-bit equal, "
          f"max rel diff {data['max_rel_diff']:.1e} -> {dest}")


if __name__ == "__main__":
    main()
