"""Render clean-up frames (pvbatch + EGL); render_prep.py writes the inputs.

    pvbatch render.py <run dir> [<run dir> ...]

For every <run>/render/it_<k>.vtk writes, from one camera per run and per view
(camera.json: the starting surface's box; +y up, the generators' convention,
which every session keeps -- its gravity is along -y):

  it_<k>_v<j>_plain.png    the surface, off-white
  it_<k>_v<j>_change.png   the surface coloured by change_h, the signed distance to the
                           starting surface in leaves h: blue = material removed, red =
                           added, white within +-0.25 h, saturated at +-CHANGE_MAX h

Views: v0 from azimuth 35 deg, v1 from the opposite side (215 deg), both 25 deg above
the horizon.  Render with the NVIDIA EGL vendor (the ~/.local/bin/pvbatch wrapper);
the renderer string is printed so a llvmpipe fallback is visible in the log.
"""

import glob
import json
import math
import os
import re
import sys

from paraview.simple import (ColorBy, Delete, GetColorTransferFunction,  # noqa: E402
                             GetOpenGLInformation, LegacyVTKReader, CreateView,
                             Render, SaveScreenshot, Show, Hide, GenerateSurfaceNormals)

SIZE = [800, 800]
CHANGE_MAX = 3.0
VIEWS = [(35.0, 25.0), (215.0, 25.0)]
BODY = [0.92, 0.90, 0.86]
AXES = {"x": 0, "y": 1, "z": 2}


def camera(rv, cam, az, el):
    u = AXES[cam["up"]]
    e1, e2 = [i for i in range(3) if i != u]
    if u == 1:          # y up: a right-handed horizontal pair (z toward the viewer at az 90)
        e1, e2 = 0, 2
    a, b = math.radians(az), math.radians(el)
    d = [0.0, 0.0, 0.0]
    d[e1] = math.cos(b) * math.cos(a)
    d[e2] = math.cos(b) * math.sin(a)
    d[u] = math.sin(b)
    lo, hi = cam["lo"], cam["hi"]
    c = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    r = 0.5 * math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    dist = r / math.tan(math.radians(12.5)) * 0.85
    up = [0.0, 0.0, 0.0]
    up[u] = 1.0
    rv.CameraFocalPoint = c
    rv.CameraPosition = [c[i] + d[i] * dist for i in range(3)]
    rv.CameraViewUp = up
    rv.CameraViewAngle = 25
    rv.CameraParallelProjection = 0


def view():
    rv = CreateView("RenderView")
    rv.ViewSize = SIZE
    rv.OrientationAxesVisibility = 0
    rv.Background = [1, 1, 1]
    for k, v in (("UseColorPaletteForBackground", 0), ("UseFXAA", 1), ("UseToneMapping", 0)):
        try:
            setattr(rv, k, v)
        except AttributeError:
            pass
    return rv


def lut():
    t = GetColorTransferFunction("change_h")
    m = CHANGE_MAX
    t.RGBPoints = [-m, 0.02, 0.19, 0.55, -1.0, 0.25, 0.50, 0.85, -0.25, 0.97, 0.97, 0.97,
                   0.25, 0.97, 0.97, 0.97, 1.0, 0.90, 0.45, 0.25, m, 0.60, 0.05, 0.05]
    t.AutomaticRescaleRangeMode = "Never"
    t.ColorSpace = "RGB"
    return t


def render_run(rd):
    cam = json.load(open(os.path.join(rd, "render", "camera.json")))
    for p in sorted(glob.glob(os.path.join(rd, "render", "it_*.vtk"))):
        k = int(re.search(r"it_(\d+)\.vtk", p).group(1))
        src = LegacyVTKReader(FileNames=[p])
        nrm = GenerateSurfaceNormals(Input=src)
        nrm.FeatureAngle = 60
        for j, (az, el) in enumerate(VIEWS):
            for mode in ("plain", "change"):
                rv = view()
                d = Show(nrm, rv)
                d.Specular = 0.15
                if mode == "plain":
                    d.ColorArrayName = ["POINTS", ""]
                    d.DiffuseColor = BODY
                    d.AmbientColor = BODY
                else:
                    ColorBy(d, ("POINTS", "change_h"))
                    d.LookupTable = lut()
                    d.SetScalarBarVisibility(rv, False)
                Render(rv)            # the first Show resets the camera: set ours after it
                camera(rv, cam, az, el)
                Render(rv)
                out = os.path.join(rd, "render", f"it_{k}_v{j}_{mode}.png")
                SaveScreenshot(out, rv, ImageResolution=SIZE, TransparentBackground=0)
                Delete(rv)
        Delete(nrm)
        Delete(src)
    print("rendered", rd, flush=True)


print("RENDERER:", GetOpenGLInformation().GetRenderer(), flush=True)
for rd in sys.argv[1:]:
    try:
        render_run(rd)
    except Exception as exc:                       # noqa: BLE001
        print("!! render failed", rd, exc, flush=True)
