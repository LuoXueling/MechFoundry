"""Objective-trajectory plot for the hidden-phi example (conda `mechfoundry-warp117`).

The sibling of so_objective.py: compliance C/C0 (top) and relative volume V/V0 (bottom)
vs iteration for the void-nucleation run, with the captured/nucleating iteration marked
and the run's second phase (hidden field switched on) shaded.  Reads hp_traj.txt +
hp_stats.txt; writes hp_objective.{png,pdf}, 88 x 55 mm.  Layout, style and the text gate
are shared with so_objective.py through `_shared/objective_style.py`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import objective_style as O  # noqa: E402

stats = {}
for line in open(os.path.join(HERE, "hp_stats.txt")):
    k, *v = line.split()
    stats[k] = v[0] if len(v) == 1 else v
cap_it = int(stats.get("iter_capture", "43"))     # the run captured iteration 43
pf_on = int(stats.get("iter_pf_on", "35"))

it, comp, vol = O.read_traj(os.path.join(HERE, "hp_traj.txt"))
O.draw("hp_objective", it, comp, vol, cap_it, f"Nucleating design (iteration {cap_it})",
       active_from=pf_on)
