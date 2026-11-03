"""Render step (pvbatch + EGL env): the image input of the TripoSG lane.

    OV_INPUT_OUT=assets/input/bunny_image.png pvbatch render_ov_input.py

A warm clay render of the display bunny (``assets/bunny_display.stl``: bunny.stl
mirrored as pfbunny mirrors it), standing
in for a photograph of the figurine.  It is a real input, not only a card: the
overview's TripoSG thumbnail is TripoSG's reconstruction FROM this image
(``ov_assets.py tripo``), and the scribble card is traced from its silhouette.
Rendering it in house clears no third-party image rights (§5i).

The camera and light kit are ``pfbunny/render_pf_bunny.py``'s (via ``ov_style``),
so the card sits beside the surface panels; the clay keeps the plain material
§5g settled on for a saturated colour.
"""

import os
import sys

from paraview.simple import *  # noqa: F403

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ov_style as S  # noqa: E402

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

SIZE = [1100, 1100]
CLAY = [0.898, 0.702, 0.541]      # warm ceramic — the figurine the scan is of
OUT = os.environ.get("OV_INPUT_OUT", "ov_input_image.png")
SURFACE = os.environ.get("OV_INPUT_SURFACE", "assets/bunny_display.stl")


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


rv = CreateRenderView()  # noqa: F405
rv.ViewSize = SIZE
rv.Background = [1, 1, 1]
rv.OrientationAxesVisibility = 0
for attr, val in S.SURF_VIEW:
    _try(lambda a=attr, v=val: setattr(rv, a, v))

src = GenerateSurfaceNormals(Input=OpenDataFile(SURFACE))  # noqa: F405
_try(lambda: setattr(src, "FeatureAngle", 50))
d = Show(src, rv)  # noqa: F405
d.Representation = "Surface"
d.DiffuseColor = CLAY
# NO PBR: §5g — PBR (no environment map) pulls a saturated colour toward brown.
d.Ambient = 0.14
d.Diffuse = 1.0
d.Specular = 0.30
d.SpecularPower = 40

# Fitted to the bunny alone: an input photograph is framed on its subject.
rv.CameraFocalPoint = [0.0, 0.0, 0.0]
rv.CameraViewUp = [0.0, 0.0, 1.0]
rv.CameraPosition = S.VIEW_DIR
rv.CameraViewAngle = S.VIEW_ANGLE
Render(rv)  # noqa: F405                     ParaView resets on 1st render
ResetCamera(rv)  # noqa: F405
fp, cp = list(rv.CameraFocalPoint), list(rv.CameraPosition)
rv.CameraPosition = [fp[i] + (cp[i] - fp[i]) * 1.06 for i in range(3)]
Render(rv)  # noqa: F405
renderer = GetOpenGLInformation().GetRenderer()  # noqa: F405
print("RENDERER:", renderer)
assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env"
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
SaveScreenshot(OUT, rv, ImageResolution=SIZE,  # noqa: F405
               OverrideColorPalette="WhiteBackground")
print(f"wrote {OUT}")
