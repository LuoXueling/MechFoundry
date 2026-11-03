"""Render the design-evolution FRAME series (pvbatch + EGL).

For every optimiser iteration captured by HP_FRAMES=1 hp_prep this renders the
two views the frame shows side by side, each with the SHIPPED render_hp actors so
they match the loop figure exactly:

  hp_panel_fr_<it>.png      octree + d, coloured section cut   (== panel a_d)
  hp_panel_frbody_<it>.png  design surface, sectioned solid    (== panel f/evo)

    pvbatch render_hp_frames.py            # every hp_fr_*.vtk
    pvbatch render_hp_frames.py 0 20 40    # just those iterations (smoke test)
"""
import glob
import os
import re
import sys

import render_hp as R    # main-guarded: importing does not render the loop


def _render_octree(it):
    rv = R.new_view()
    R.outline(f"hp_fr_{it}.vtk", rv)                 # octree-domain cube frame
    src = R.OpenDataFile(f"hp_fr_{it}.vtk")
    src.UpdatePipeline()
    node = R.half_cut(src)
    R.mesh_actor(rv, node, "sdf")
    R.iso(rv, node, "sdf", R.INK, 4.0)              # the d = 0 contour
    R.Render(rv)
    R.fit_cube(rv)
    R.Render(rv)
    R.SaveScreenshot(f"hp_panel_fr_{it}.png", rv, ImageResolution=R.SIZE,
                     OverrideColorPalette="WhiteBackground")
    R.Delete(rv)


def _render_body(it):
    rv = R.new_view()
    R.section(rv, f"hp_fr_{it}")                     # _shell/_face/_ghost.stl
    R.Render(rv)
    R.fit(rv)
    R.Render(rv)
    R.SaveScreenshot(f"hp_panel_frbody_{it}.png", rv, ImageResolution=R.SIZE,
                     OverrideColorPalette="WhiteBackground")
    R.Delete(rv)


def _iters():
    fs = glob.glob("hp_fr_*.vtk")
    return sorted(int(re.search(r"hp_fr_(\d+)\.vtk", f).group(1)) for f in fs)


want = [int(a) for a in sys.argv[1:]] or _iters()
print(f"rendering {len(want)} frame(s): {want[:5]}{' ...' if len(want) > 5 else ''}")
for it in want:
    _render_octree(it)
    if os.path.exists(f"hp_fr_{it}_shell.stl"):
        _render_body(it)
    else:
        print("  (no shell for)", it)
    print(f"wrote hp_panel_fr_{it}.png + frbody")
print("FRAMES RENDER DONE", len(want))
