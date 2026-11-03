"""Render the design-evolution FRAME series (pvbatch + EGL).

For every optimiser iteration captured by SO_FRAMES=1 so_prep this renders the
two views the frame shows side by side, each with the SHIPPED panel code so they
match the loop figure exactly:

  panel_so_fr_<it>.png      octree + d, coloured section cut   (== panel a)
  panel_so_frbody_<it>.png  design surface, sectioned solid    (== panel f/evo)

    pvbatch render_so_frames.py            # every so_fr_*.vtk
    pvbatch render_so_frames.py 0 20 40    # just those iterations (smoke test)
"""
import glob
import os
import re
import sys

import render_so as R    # main-guarded: importing does not render the loop


def _iters():
    fs = glob.glob("so_fr_*.vtk")
    return sorted(int(re.search(r"so_fr_(\d+)\.vtk", f).group(1)) for f in fs)


want = [int(a) for a in sys.argv[1:]] or _iters()
print(f"rendering {len(want)} frame(s): {want[:5]}{' ...' if len(want) > 5 else ''}")
for it in want:
    R.panel_octree_field(f"so_fr_{it}.vtk", "sdf", f"fr_{it}",
                         R.SDF_MIN, R.SDF_MAX, cmap="sdf")
    shell = f"so_fr_{it}_shell.stl"
    if os.path.exists(shell):
        R.panel_evo(shell, f"frbody_{it}")
    else:
        print("  (no shell for)", it)
print("FRAMES RENDER DONE", len(want))
