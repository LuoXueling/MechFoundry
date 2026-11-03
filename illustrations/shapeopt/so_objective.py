"""Objective-trajectory plot for the shape-opt example (conda `mechfoundry-warp117`).

Split out of the loop figure: compliance C/C0 (top) and relative volume V/V0 (bottom)
vs iteration, with the shown/captured design marked.  Reads so_traj.txt + so_stats.txt;
writes so_objective.{png,pdf}, 88 x 55 mm.  Layout, style and the text gate are shared
with hp_objective.py through `_shared/objective_style.py` so the two cannot drift.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import objective_style as O  # noqa: E402

stats = {}
for line in open(os.path.join(HERE, "so_stats.txt")):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        stats[k] = v
cap_it = int(stats.get("capture_iter", "40"))

it, comp, vol = O.read_traj(os.path.join(HERE, "so_traj.txt"))
O.draw("so_objective", it, comp, vol, cap_it, f"Shown design (iteration {cap_it})")
