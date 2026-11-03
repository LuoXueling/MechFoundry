"""Shared constants and paths of the geometry clean-up pipeline (plain Python, no solver).

The clean-up is a short, volume-as-is shape optimization of each agentic-batch body,
stopped as soon as the surface has moved a fixed fraction of a leaf away from where it
started.  Everything that defines it lives here so setup.py, run.py, summarize.py and
the render scripts agree:

  STEP_FRACTION   the first update's largest surface move, in leaves h (run.py sets
                  each body's gd_lr from it at iteration 0)
  STOP_PERCENTILE the symmetric surface distance is this area-weighted percentile of
                  the nearest-point distances, taken in both directions between the
                  current and the starting surface (the larger of the two)
  STOP_THRESHOLD  the run stops at the first shape whose distance exceeds this many h
  STOP_MAX        ... or whose largest nearest-point distance (either direction)
                  exceeds this many h: a small feature erased or a local growth
                  beyond "a little thicker", which a percentile cannot see
  MAX_UPDATES     the cap: at most this many design updates

The values were fixed on the trial set (README.md, "Trials").
"""

import json
import os
import re

REPO = "/home/xl2254/implicit_optimize"            # sessions and output/ live here
HERE = os.path.dirname(os.path.abspath(__file__))
# Solver and workflow code are imported from the checkout THIS file is in.
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
BATCH = os.path.join(REPO, "output", "agentic_batch50_opus")   # read-only
OUT = os.path.join(REPO, "output", "geometry_cleanup")

STEP_FRACTION = 0.5
STOP_PERCENTILE = 99.0
STOP_THRESHOLD = 1.5
STOP_MAX = 3.0      # ... or whose maximum nearest-point distance exceeds this many h
MAX_UPDATES = 40
REINIT = "off"      # "on"|"off": the solver's per-iteration SDF reinit (README, "Trials")

# Settings every clean-up shares (validated by the workflow's ShapeOptSettings).  The
# rest of each case's shape_opt block -- octree depths, Newton tolerances, load steps,
# precision, w_vol -- is the session's own, as are its supports, loads and body force.
COMMON = {
    "run_mode": "shape_optimization",
    "volume_as_is": True,
    "sdf_bc_fix_method": "mask",
    "optimize_relative_tol": 1e-9,       # never converge early on the objective
    "reinit_sdf": REINIT == "on",
    # the cap, plus one iteration that only evaluates the final shape, plus one for a
    # rejected shape (run.py: a shape over the threshold is rolled back and re-solved)
    "max_optimize_iters": MAX_UPDATES + 2,
    "output_per_optimize_step": 1,       # run.py's save() keeps only the first and last
}


def object_name(run_name):
    """bar-stool-20261013-213800 -> bar-stool."""
    return re.sub(r"-\d{8}-\d{6}$", "", run_name)


def runs():
    """{object: (run folder, session folder)} for every run of the batch."""
    out = {}
    for run in sorted(os.listdir(BATCH)):
        mpath = os.path.join(BATCH, run, "metrics.json")
        if not os.path.isfile(mpath):
            continue
        sid = json.load(open(mpath))["session_id"]
        out[object_name(run)] = (os.path.join(BATCH, run),
                                 os.path.join(BATCH, run, "workflow", sid))
    return out


def obj_dir(name):
    return os.path.join(OUT, name)


def case_path(name):
    return os.path.join(OUT, name, "case.json")
