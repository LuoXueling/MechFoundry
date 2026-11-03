"""Prep step 2 (conda ``mechfoundry-warp117``, GPU): the cost numbers the overview quotes.

The figure carries two wall-clock claims, and both are measured here rather than
copied out of a notebook:

  A. **Octree extraction** — the production
     ``src.triposg_wrapper.trimesh_sdf.trimesh_to_octree`` on the same bunny the
     SDF/octree thumbnails show, at the depth they show (3→6) and one level
     denser.  Warm-up call first (Warp JITs its kernels once), then a
     ``torch.cuda.synchronize()``-fenced timed call — ``sdfoctree/
     prep_octree_sdf.py``'s pattern, which is what put "48 ms" on
     ``bunny_sdf_octree``.

  B. **One design iteration** — 4 iterations of the production optimiser on the
     *same problem the bottom band shows*, by running ``shapeopt/so_prep.py``
     with ``SO_CAPTURE_ITER`` out of range (so it captures nothing) and
     ``SO_OUT_DIR`` pointed at a scratch folder (so it overwrites none of the
     shape-opt figure's artefacts).  Timing is read from the run's own
     ``log.txt``: production already timestamps every line, so the iteration
     boundaries and the phase boundaries inside one are in there.  Iteration 0
     pays Warp's JIT and is dropped; the median of the rest is reported.

Writes ``ov_stats.txt``.  Nothing here re-renders a figure.
"""

import datetime as dt
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ILL = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ILL, ".."))
sys.path.insert(0, REPO)

from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree  # noqa: E402

INIT_DEPTH = 3          # == bunny_sdf_octree's octree panel
OCTREE_DEPTH = 6
BENCH_ITERS = int(os.environ.get("OV_BENCH_ITERS", "4"))
RUN_DIR = os.path.join(REPO, "src", "warp_wrapper", "output")
SCRATCH = os.path.join(HERE, "ov_bench_run")

stats = {}


# ---------------------------------------------------------------------------
# A. octree extraction
# ---------------------------------------------------------------------------
def bench_octree():
    meta = os.path.join(HERE, "bunny_sdf.meta")
    stl = os.path.join(HERE, "bunny.stl")
    if not (os.path.isfile(meta) and os.path.isfile(stl)):
        raise SystemExit(
            "missing bunny.stl / bunny_sdf.meta — run ../bunny/prep_bunny_sdf.py "
            "in this folder first (build.sh overview does)."
        )
    H = float(open(meta).readline())
    mesh = trimesh.load(stl)

    def gen(depth):
        return trimesh_to_octree(
            mesh,
            bounds=(-H, -H, -H, H, H, H),
            init_octree_depth=INIT_DEPTH,
            octree_depth=depth,
            domain_length=2 * H,
            sdf_truncation=0.01,
        )

    def timed(depth):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        oc = gen(depth)
        torch.cuda.synchronize()
        return oc, (time.perf_counter() - t0) * 1e3

    gen(OCTREE_DEPTH)                                   # warm up the JIT
    oc, ms = timed(OCTREE_DEPTH)
    oc2, ms2 = timed(OCTREE_DEPTH + 1)
    stats.update(
        octree_init_depth=INIT_DEPTH,
        octree_depth=OCTREE_DEPTH,
        octree_nodes=len(oc["nodes"]),
        octree_hexes=len(oc["elements"]),
        octree_ms=f"{ms:.1f}",
        octree_nodes_denser=len(oc2["nodes"]),
        octree_hexes_denser=len(oc2["elements"]),
        octree_ms_denser=f"{ms2:.1f}",
    )
    print(f"  octree {INIT_DEPTH}->{OCTREE_DEPTH}: {len(oc['nodes'])} nodes, "
          f"{len(oc['elements'])} hexes, {ms:.1f} ms")
    print(f"  octree {INIT_DEPTH}->{OCTREE_DEPTH + 1}: {len(oc2['nodes'])} nodes, "
          f"{len(oc2['elements'])} hexes, {ms2:.1f} ms")


# ---------------------------------------------------------------------------
# B. one design iteration of the production optimiser
# ---------------------------------------------------------------------------
_TS = re.compile(r"^\[(\d\d-\d\d-\d{4},\d\d:\d\d:\d\d\.\d+)\]\s*(.*)$")


def _parse_log(path):
    """[(seconds_since_start, message)] for every timestamped line."""
    out, t0 = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = _TS.match(line)
            if not m:
                continue
            t = dt.datetime.strptime(m.group(1), "%m-%d-%Y,%H:%M:%S.%f")
            t0 = t0 or t
            out.append(((t - t0).total_seconds(), m.group(2)))
    return out


def _phases(events):
    """Per-iteration wall time, split at the markers production already logs.

    solve   : iteration start        -> the "SE=..., Vol=..." objective line
    adjoint : "Backward for objective" -> the first remesh line
    remesh  : "BC zip: stitching"    -> "Start saving outputs" / iteration end
    """
    starts = [i for i, (_, m) in enumerate(events)
              if m.startswith("===== Optimization Iteration")]
    rows = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(events)
        seg = events[i:j]
        t_start = seg[0][0]
        it = int(seg[0][1].split()[3])

        def first(pred, default=None):
            for t, m in seg:
                if pred(m):
                    return t
            return default

        t_obj = first(lambda m: m.startswith("SE="))
        t_bwd = first(lambda m: m.startswith("Backward for objective"))
        t_zip = first(lambda m: m.lstrip().startswith(("BC zip:", "SDF reinit")))
        t_save = first(lambda m: m.startswith("Start saving outputs"))
        t_end = first(lambda m: m.lstrip().startswith("Opt iter"), seg[-1][0])
        rows.append({
            "iter": it,
            "total": t_end - t_start,
            "solve": (t_obj - t_start) if t_obj else float("nan"),
            "adjoint": (t_zip - t_bwd) if (t_zip and t_bwd) else float("nan"),
            "remesh": ((t_save or t_end) - t_zip) if t_zip else float("nan"),
        })
    return rows


def bench_iteration():
    before = set(os.listdir(RUN_DIR)) if os.path.isdir(RUN_DIR) else set()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)

    env = dict(os.environ)
    env.update(
        SO_N_ITERS=str(BENCH_ITERS),
        SO_CAPTURE_ITER="999999",   # capture nothing
        SO_EVO="-1",                # dump no evolution frames
        SO_SAVE_EVERY="0",          # no per-iteration VTU/GLB: measure compute
        SO_OUT_DIR=SCRATCH,         # never touch shapeopt/'s artefacts
    )
    prep = os.path.join(ILL, "shapeopt", "so_prep.py")
    print(f"  running {BENCH_ITERS} production iterations ({prep}) ...")
    r = subprocess.run([sys.executable, prep], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.stdout.write(r.stdout.decode(errors="replace"))
        raise SystemExit(f"so_prep.py failed ({r.returncode}) — see the log above")

    new = sorted(set(os.listdir(RUN_DIR)) - before)
    if not new:
        raise SystemExit(f"no new run directory appeared under {RUN_DIR}")
    run = os.path.join(RUN_DIR, new[-1])
    events = _parse_log(os.path.join(run, "log.txt"))
    rows = _phases(events)
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} iterations in {run}/log.txt")

    warm = rows[1:]                       # iteration 0 pays the Warp JIT
    med = {k: statistics.median(r[k] for r in warm)
           for k in ("total", "solve", "adjoint", "remesh")}
    for r in rows:
        print(f"    iter {r['iter']}: {r['total']:.2f} s  "
              f"(solve {r['solve']:.2f} · adjoint {r['adjoint']:.2f} · "
              f"remesh {r['remesh']:.2f})")

    # mesh size of the benchmarked problem, from the run's own log
    m = None
    for _, msg in events:
        if msg.startswith("ShapeOptTrimmed: outer"):
            m = re.search(r"outer (\d+) nodes, inner (\d+) nodes", msg)
            break
    if m:
        stats.update(bench_outer_nodes=int(m.group(1)),
                     bench_inner_nodes=int(m.group(2)))
    bench_stats_path = os.path.join(SCRATCH, "so_stats.txt")
    if not os.path.isfile(bench_stats_path):
        raise SystemExit("benchmark did not write its scratch so_stats.txt")
    bench_stats = {}
    with open(bench_stats_path) as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep:
                bench_stats[key] = value
    required = ("h_target", "sdf_truncation", "sdf_refinement_band")
    missing = [key for key in required if key not in bench_stats]
    if missing:
        raise SystemExit(f"benchmark stats missing {missing}")
    stats.update(
        bench_sdf_truncation=bench_stats["sdf_truncation"],
        bench_sdf_refinement_band=bench_stats["sdf_refinement_band"],
        bench_h_target=bench_stats["h_target"],
    )

    stats.update(
        bench_iters=len(rows),
        bench_iters_timed=len(warm),
        iter_s=f"{med['total']:.2f}",
        iter_solve_s=f"{med['solve']:.2f}",
        iter_adjoint_s=f"{med['adjoint']:.2f}",
        iter_remesh_s=f"{med['remesh']:.2f}",
        iter_s_all=" ".join(f"{r['total']:.2f}" for r in rows),
        bench_run=new[-1],
    )
    # Keep the evidence, drop the run: a 4-iteration benchmark has nothing worth
    # 50 MB of VTU, but the log is what the numbers were read out of.
    shutil.copyfile(os.path.join(run, "log.txt"),
                    os.path.join(HERE, "ov_bench_log.txt"))
    shutil.rmtree(run, ignore_errors=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)


def main():
    print("A. octree extraction")
    bench_octree()
    print("B. one design iteration")
    bench_iteration()
    stats["gpu"] = torch.cuda.get_device_name(0)

    # Cross-check against the run the bottom-band thumbnails came from. Mesh
    # counts are iteration-dependent, so compare the physical discretization
    # rather than an initial benchmark mesh with the iteration-43 capture.
    hp = os.path.join(ILL, "hiddenphi", "hp_stats.txt")
    if os.path.isfile(hp) and "bench_outer_nodes" in stats:
        ref = {k: v for k, v in
               (l.split(None, 1) for l in open(hp) if len(l.split(None, 1)) == 2)}
        checks = {
            "h_min": float(stats["bench_h_target"]),
            "sdf_truncation": float(stats["bench_sdf_truncation"]),
            "sdf_refinement_band": float(stats["bench_sdf_refinement_band"]),
        }
        for key, got in checks.items():
            want = float(ref[key])
            assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (
                f"benchmark {key}={got} differs from hp_stats {want}"
            )
        assert int(ref["depth"]) == 7 and float(ref["gd_lr"]) == 8.0
        print(
            "  cross-check: benchmark and hidden-phi capture share "
            f"h={checks['h_min']:.3f}, truncation={checks['sdf_truncation']:.3f}, "
            f"band={checks['sdf_refinement_band']:.3f}; "
            f"mesh counts are iteration-specific ({stats['bench_outer_nodes']} initial vs "
            f"{ref['nodes']} captured)"
        )

    with open(os.path.join(HERE, "ov_stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k} {v}\n")
    print(f"wrote ov_stats.txt ({len(stats)} keys) on {stats['gpu']}")


if __name__ == "__main__":
    main()
