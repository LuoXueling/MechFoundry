"""shape_opt_merged row c, stage 2 (pvbatch + EGL): the clean-up objects' panels.

    pvbatch render_som_cleanup.py [object ...]     # after som_cleanup_prep.py
    pvbatch render_som_cleanup.py --check OBJECT   # the change panel == the clean-up's own

Per object of som_cleanup.json, both from ONE camera, the geometry clean-up's own
left-hand view (experiments/geometry_cleanup/render.camera, VIEWS[0]: azimuth 35,
elevation 25, framed on the starting surface's box) dollied back by DOLLY:

  panel_som_cu_<object>_bc.png      agentic_results' supports-and-loads panel
                                    (render_ar.bc_panel: clay body, flat-lit teal /
                                    orange overlays, solid-lit load arrows, the teal
                                    outline tube of a support facing away)
  panel_som_cu_<object>_change.png  the final body coloured by its signed change from
                                    the start in leaves h: geometry_cleanup/render.py's
                                    "change" mode line for line (its view(), lut(),
                                    normals, finish), on the clean-up's own
                                    <run>/render/it_<final>.vtk

Why not the clean-up's PNGs as they are: its camera (0.85 x the fit distance) cuts
the tall bodies off at the frame -- the claw hammer's head, both ends of the femur,
the park bench's feet.  DOLLY 1.45 shows every body whole (1.3 still cut the arrow
over the femur's head).  --check renders the change panel at DOLLY 1 and the
clean-up's 800 px and asserts it equals the contact sheet's tile (max pixel
difference 1 of 255 on 2026-10-14), so the map is the clean-up's exactly.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
sys.path.insert(0, os.path.join(REPO, "experiments", "geometry_cleanup"))

_argv, sys.argv = sys.argv, sys.argv[:1]   # both modules read sys.argv at import
import render_ar as RA  # noqa: E402  (its __main__ loop is guarded)
import render as CR  # noqa: E402     (prints RENDERER; renders nothing with no args)
sys.argv = _argv

from paraview.simple import (ColorBy, Delete, GenerateSurfaceNormals,  # noqa: E402
                             GetOpenGLInformation, LegacyVTKReader, Render,
                             SaveScreenshot, Show)

WORK = os.path.join(HERE, "_cleanup")
DOLLY = 1.45                # x the clean-up camera's distance: no body (or arrow) cut by the frame
SIZE = [1150, 1150]         # the clean-up's 800 px x DOLLY, rounded: same pixels per body


def place(rv, cam, dolly):
    """The clean-up's camera (at the object's recorded view: VIEWS[0], or the prep's
    declared VIEW_OVERRIDE), then moved back along its own axis."""
    CR.camera(rv, cam, cam["azimuth"], cam["elevation"])
    fp, pos = rv.CameraFocalPoint, rv.CameraPosition
    rv.CameraPosition = [fp[i] + dolly * (pos[i] - fp[i]) for i in range(3)]


def change_panel(vtk, size):
    """geometry_cleanup/render.render_run, mode "change", one view."""
    src = LegacyVTKReader(FileNames=[vtk])
    nrm = GenerateSurfaceNormals(Input=src)
    nrm.FeatureAngle = 60
    rv = CR.view()
    rv.ViewSize = size
    d = Show(nrm, rv)
    d.Specular = 0.15
    ColorBy(d, ("POINTS", "change_h"))
    d.LookupTable = CR.lut()
    d.SetScalarBarVisibility(rv, False)
    Render(rv)               # the first Show resets the camera: set ours after it
    return rv


def bc_panel(o, size):
    f = lambda k: os.path.join(WORK, f"{o['name']}_{k}.vtk")  # noqa: E731
    sup = f("support") if os.path.exists(f("support")) else None
    arw = f("arrows") if os.path.exists(f("arrows")) else None
    rv = RA.bc_panel(f("bc"), sup, arw, o["tube_r"])
    rv.ViewSize = size
    rv.Background = [1.0, 1.0, 1.0]
    RA._try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))
    Render(rv)
    return rv


def shoot(rv, cam, dolly, size, out):
    place(rv, cam, dolly)
    Render(rv)
    SaveScreenshot(out, rv, ImageResolution=size, TransparentBackground=0)
    Delete(rv)


def main():
    renderer = GetOpenGLInformation().GetRenderer()
    print("RENDERER:", renderer, flush=True)
    assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env (AGENTS.md §5)"
    rec = json.load(open(os.path.join(HERE, "som_cleanup.json")))
    assert tuple(CR.VIEWS[0]) == (rec["view"]["azimuth"], rec["view"]["elevation"])
    for o in rec["objects"]:   # only a declared override may leave the clean-up's view
        c = o["camera"]
        assert c.get("view_override") or (c["azimuth"], c["elevation"]) == tuple(CR.VIEWS[0])
    objs = {o["name"]: o for o in rec["objects"]}
    if sys.argv[1:2] == ["--check"]:
        from PIL import Image
        o = objs[sys.argv[2]]
        out = os.path.join(WORK, "_check.png")
        shoot(change_panel(o["change_vtk"], CR.SIZE), o["camera"], 1.0, CR.SIZE, out)
        a = np.asarray(Image.open(out).convert("RGB")).astype(int)
        b = np.asarray(Image.open(o["change_png"]).convert("RGB")).astype(int)
        diff = np.abs(a - b).max()
        print(f"check {o['name']}: max pixel difference {diff}")
        assert diff <= 2, "the change panel no longer matches the clean-up's render"
        return
    for name, o in objs.items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        shoot(bc_panel(o, SIZE), o["camera"], DOLLY, SIZE,
              os.path.join(HERE, f"panel_som_cu_{name}_bc.png"))
        shoot(change_panel(o["change_vtk"], SIZE), o["camera"], DOLLY, SIZE,
              os.path.join(HERE, f"panel_som_cu_{name}_change.png"))
        print("wrote panel_som_cu_%s_{bc,change}.png" % name, flush=True)


main()
