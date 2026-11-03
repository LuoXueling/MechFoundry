"""Which agent runs the agentic-results figures show, and one figure per object category.

    python3 ar_categories.py --batch-root output/agentic_batch50 [--suffix _haiku]
                             [--prefix agentic_results_] [--only furniture ...]
                             [--dry-run] [--watch] [--segmentation]

Plain Python (no GPU, no conda env): it picks runs and calls ``./build.sh agenticresults``
once per category, which is where the GPU work happens.

OBJECTS.  ``experiments/agentic_batch/objects_50.txt`` (``--list``): one instruction per
line, grouped under ``# <Category>`` headings.  A heading is a comment line followed
directly by an instruction; the file's opening comment paragraph is not one.

RUN PICKING (``latest_runs``).  A batch root (``output/agentic_batch50``) holds one folder
per run and ``_batches/<id>/summary.csv`` per batch launch (columns ``instruction``,
``path``, ``status``, ...).  Batches are read in name order (their ids are timestamps), and
for every instruction the LAST row whose run folder exists wins, so a rerun of one object in
a later batch replaces its earlier run.  Every status counts: a run that stopped early is
shown, with the stages it did not reach marked, never skipped.  The run folder's
``metrics.json`` names its primary workflow session (``session_id``), which ``ar_runs.py``
follows.

PER-CATEGORY BUILDS.  A category is built when every one of its instructions has a run.
Each figure is ``illustrations/<prefix><slug><suffix>.{pdf,png}``, its slug the heading in
lower case with non-letters as ``_`` ("Machines and tools" -> ``machines_and_tools``);
``--suffix _haiku`` / ``_opus`` names one model's set.  Builds run one at a time (the target
reuses ``agentic_results/_work``).  After each build the figure's record
``agentic_results/ar_stats.json`` is copied to ``agentic_results/ar_stats_<name>.json`` and the
shared ``ar_stats.json`` is put back as it was, so a category build never replaces the
record of the main figure.  ``--watch`` polls a running batch every minute and builds each
category as soon as its five runs exist, until all are built or the batch log says
``done:``.  ``--dry-run`` prints each category's run list and builds nothing.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ILLU = HERE.parent
REPO = ILLU.parent
OBJECTS = REPO / "experiments" / "agentic_batch" / "objects_50.txt"
DEFAULT_PREFIX = "agentic_results_"


def slug(name):
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def categories(path=OBJECTS):
    """[(name, slug, [instruction, ...]), ...] in file order."""
    lines = [s.strip() for s in Path(path).read_text().splitlines()]
    cats = []
    for k, s in enumerate(lines):
        if s.startswith("#"):
            nxt = lines[k + 1] if k + 1 < len(lines) else ""
            if nxt and not nxt.startswith("#"):          # a heading: an instruction follows
                name = s.lstrip("#").strip()
                cats.append((name, slug(name), []))
        elif s and cats:
            cats[-1][2].append(s)
    return cats


def objects(path=OBJECTS):
    """[(instruction, category name), ...] in file order."""
    return [(o, name) for name, _, objs in categories(path) for o in objs]


def _resolve(batch_root, rel):
    """A summary row's ``path`` (relative to the repository the batch ran from) as an
    absolute path: tried against the batch root and each of its ancestors."""
    p = Path(rel)
    if p.is_absolute():
        return p
    for base in (batch_root, *batch_root.parents):
        if (base / p).exists():
            return base / p
    return None


def latest_runs(batch_root):
    """{instruction: absolute run folder}: the latest run of each instruction across every
    batch of ``batch_root`` (later batches win; rows whose folder is gone are skipped)."""
    batch_root = Path(batch_root).resolve()
    out = {}
    bdir = batch_root / "_batches"
    if not bdir.is_dir():
        return out
    for b in sorted(bdir.iterdir(), key=lambda p: p.name):
        sc = b / "summary.csv"
        if not sc.exists():
            continue
        with open(sc, newline="") as fh:
            for row in csv.DictReader(fh):
                p = _resolve(batch_root, row["path"])
                if p is not None:
                    out[row["instruction"]] = str(p)
    return out


def batch_done(batch_root):
    """The newest batch's log says ``done:`` (the batch script's last line)."""
    bdir = Path(batch_root) / "_batches"
    batches = sorted(bdir.iterdir(), key=lambda p: p.name) if bdir.is_dir() else []
    if not batches:
        return False
    log = batches[-1] / "batch.log"
    return log.exists() and "done:" in log.read_text()


def build(name, paths, log, segmentation=False):
    """``./build.sh agenticresults`` for one figure; returns its exit code.  Keeps the
    shared ar_stats.json, and saves this figure's as ar_stats_<name>.json.
    ``segmentation``: the figure gets the SEGMENTATION column (AR_SEGMENTATION=1)."""
    stats = HERE / "ar_stats.json"
    keep = stats.read_bytes() if stats.exists() else None
    gpu = os.environ.get("AR_GPU", "1")
    env = dict(os.environ, AR_RUNS=" ".join(paths), AR_NAME=name,
               AR_FLOOR=os.environ.get("AR_FLOOR", "warn"), AR_GPU=gpu,
               PV_GPU=os.environ.get("PV_GPU", gpu),
               AR_SEGMENTATION="1" if segmentation else os.environ.get("AR_SEGMENTATION", "0"))
    try:
        with open(log, "w") as fh:
            r = subprocess.run(["./build.sh", "agenticresults"], cwd=ILLU, env=env,
                               stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 0 and stats.exists():
            shutil.copy(stats, stats.with_name(f"ar_stats_{name}.json"))
        return r.returncode
    finally:
        if keep is not None:
            stats.write_bytes(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch-root", required=True, help="e.g. output/agentic_batch50")
    ap.add_argument("--list", default=str(OBJECTS), help="the object list (default: %(default)s)")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="figure-name prefix (default: %(default)s)")
    ap.add_argument("--suffix", default="", help="figure-name suffix, e.g. _haiku or _opus")
    ap.add_argument("--only", nargs="*", help="category slugs to build (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the run lists, build nothing")
    ap.add_argument("--watch", action="store_true",
                    help="poll every minute and build each category once its runs exist")
    ap.add_argument("--segmentation", action="store_true",
                    help="add the SEGMENTATION column before the supports and loads "
                         "(AR_SEGMENTATION=1; the Opus set is built with it)")
    ap.add_argument("--log-dir", default=str(HERE / "_logs"), help="one build log per figure")
    a = ap.parse_args()
    root = Path(a.batch_root)
    if not root.is_absolute():
        root = (REPO / root) if (REPO / root).exists() else root.resolve()
    cats = [c for c in categories(a.list) if not a.only or c[1] in a.only]
    assert cats, "no categories selected"
    built, failed = set(), set()
    while True:
        runs = latest_runs(root)
        for name, sl, objs in cats:
            if sl in built:
                continue
            missing = [o for o in objs if o not in runs]
            fig = f"{a.prefix}{sl}{a.suffix}"
            if missing:
                if a.dry_run or not a.watch:
                    print(f"{fig}: not built, no run yet for {missing}")
                continue
            paths = [runs[o] for o in objs]
            if a.dry_run:
                print(f"{fig}: AR_RUNS=\"{' '.join(paths)}\"")
                built.add(sl)
                continue
            os.makedirs(a.log_dir, exist_ok=True)
            rc = build(fig, paths, os.path.join(a.log_dir, f"{fig}.log"), a.segmentation)
            print(f"{time.strftime('%H:%M:%S')} built {fig}: rc={rc}", flush=True)
            built.add(sl)
            if rc:
                failed.add(sl)
        if not a.watch or a.dry_run or len(built) == len(cats) or batch_done(root) and all(
                sl in built or any(o not in runs for o in objs) for _, sl, objs in cats):
            break
        time.sleep(60)
    left = [sl for _, sl, _ in cats if sl not in built]
    if not a.dry_run:
        print(f"{time.strftime('%H:%M:%S')} finished; not built: {left}; failed: {sorted(failed)}",
              flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
