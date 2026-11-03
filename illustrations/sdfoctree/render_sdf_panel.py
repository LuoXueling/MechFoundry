"""Render one SDF-cube panel (run with pvbatch + EGL env):

    pvbatch render_sdf_panel.py uniform   # smooth uniform-grid SDF cube
    pvbatch render_sdf_panel.py octree     # octree hex mesh (SDF + cell frames)

Both use the SAME camera / lighting / colormap as bunny_sdf_pv: the SDF cube is
cut in half (keep Y>=0, faces colored by SDF), the near half of the bunny shows
through translucent, and the cube is outlined.  No per-panel colorbar (a shared
one is added when the two panels are composed).  Writes panel_<name>.png.
"""

import sys
from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

PANEL = sys.argv[1] if len(sys.argv) > 1 else "octree"
ARIAL = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
VMIN, VMAX = -0.26, 0.72                      # fixed so both panels share colors


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


H = float(open("bunny_sdf.meta").readline())

if PANEL == "uniform":
    src = OpenDataFile("bunny_sdf.vtk")
    edges = False
else:
    src = OpenDataFile("octree.vtk")
    edges = True
bunny = OpenDataFile("bunny.stl")
src.UpdatePipeline()

# ---------------------------------------------------------------------------
rv = GetActiveViewOrCreate("RenderView")
rv.ViewSize = [1700, 1700]
rv.Background = [1, 1, 1]
rv.OrientationAxesVisibility = 0
_try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))
_try(lambda: setattr(rv, "UseFXAA", 1))
_try(lambda: setattr(rv, "UseToneMapping", 1))
_try(lambda: setattr(rv, "KeyLightAzimuth", -35))
_try(lambda: setattr(rv, "KeyLightElevation", 45))
_try(lambda: setattr(rv, "KeyLightIntensity", 0.85))
_try(lambda: setattr(rv, "KeyLightWarmth", 0.55))
_try(lambda: setattr(rv, "FillLightKFRatio", 2.6))
_try(lambda: setattr(rv, "BackLightKBRatio", 3.0))
_try(lambda: setattr(rv, "HeadLightKHRatio", 3.0))

# --- solid half of the cube (keep Y >= 0), faces colored by sdf --------------
clipV = Clip(Input=src)
clipV.ClipType = "Plane"
# Octree: nudge the plane a hair BEHIND (+Y, into the cells) so it doesn't sit
# exactly on the cell faces (Y=0 is a cell boundary at every level).  That
# face-coincidence is what produced the degenerate sliver / "X" edge artifacts;
# cutting through the cell interiors gives clean quad cross-sections instead.
clipV.ClipType.Origin = [0.0, 0.009 if edges else 0.0, 0.0]
clipV.ClipType.Normal = [0.0, 1.0, 0.0]
clipV.Invert = 0
dV = Show(clipV, rv)
dV.Representation = "Surface With Edges" if edges else "Surface"
ColorBy(dV, ("POINTS", "sdf"))
lut = GetColorTransferFunction("sdf")
lut.ColorSpace = "Lab"
lut.RGBPoints = [
    VMIN,   0.106, 0.196, 0.478,
    -0.130, 0.235, 0.443, 0.718,
    -0.040, 0.557, 0.706, 0.851,
    0.0,    0.894, 0.894, 0.894,
    0.040,  0.968, 0.805, 0.576,
    0.150,  0.944, 0.600, 0.282,
    0.380,  0.827, 0.329, 0.152,
    VMAX,   0.478, 0.129, 0.094,
]
lut.NanColor = [0.6, 0.6, 0.6]
dV.Ambient = 0.28
dV.Diffuse = 0.85
dV.Specular = 0.0
if edges:
    _try(lambda: setattr(dV, "EdgeColor", [0.03, 0.03, 0.05]))
    _try(lambda: setattr(dV, "LineWidth", 3.5))
    _try(lambda: setattr(dV, "RenderLinesAsTubes", 0))
dV.SetScalarBarVisibility(rv, False)

# --- cube outline (thick frame) ----------------------------------------------
dO = Show(Outline(Input=src), rv)
dO.DiffuseColor = [0.12, 0.13, 0.16]
dO.LineWidth = 4.0

# --- revealed near half of the bunny (keep Y <= 0), translucent --------------
bunnyN = GenerateSurfaceNormals(Input=bunny)
_try(lambda: setattr(bunnyN, "FeatureAngle", 50))
clipB = Clip(Input=bunnyN)
clipB.ClipType = "Plane"
clipB.ClipType.Origin = [0.0, 0.0, 0.0]
clipB.ClipType.Normal = [0.0, 1.0, 0.0]
clipB.Invert = 1
dB = Show(clipB, rv)
dB.Representation = "Surface"
dB.DiffuseColor = [0.92, 0.90, 0.86]
dB.Opacity = 0.32
dB.Specular = 0.35
dB.SpecularPower = 30
if edges:
    # Panel (b) only.  Over the octree the ghost has to compete with a dense
    # black wireframe as well as the field, and at 0.32 opacity it loses; the
    # shell reads as a smudge rather than as the surface the cells resolve.
    # Self-illumination in its OWN colour lifts the whole shell at once
    # (AmbientColor: ParaView's default is white, which would wash it out).
    # PBR has to go for that — VTK's PBR shader ignores the classic
    # Ambient/Diffuse coefficients outright, measured in diffused_nitsche as
    # EXACTLY 0.0 mean pixel difference between ambient 0.55 and 1.0.  Panel (a)
    # keeps PBR: its backdrop is the smooth field, with no wireframe to fight,
    # and the ghost already reads there.
    dB.Ambient = 0.24                        # == render_dn.py's body
    dB.Diffuse = 1.0
    _try(lambda: setattr(dB, "AmbientColor", [0.92, 0.90, 0.86]))
else:
    dB.Ambient = 0.12
    _try(lambda: setattr(dB, "Interpolation", "PBR"))
    _try(lambda: setattr(dB, "Roughness", 0.45))
    _try(lambda: setattr(dB, "Metallic", 0.0))

# --- camera (identical for both panels so they align) ------------------------
rv.CameraFocalPoint = [0.0, 0.0, 0.0]
rv.CameraViewUp = [0.0, 0.0, 1.0]
rv.CameraPosition = [1.5 * H, -3.0 * H, 1.25 * H]
ResetCamera(rv)
_fp = list(rv.CameraFocalPoint)
_cp = list(rv.CameraPosition)
rv.CameraPosition = [_fp[i] + (_cp[i] - _fp[i]) * 1.16 for i in range(3)]
rv.CameraViewAngle = 25
Render()

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
SaveScreenshot(f"panel_{PANEL}.png", rv, ImageResolution=[1700, 1700],
               OverrideColorPalette="WhiteBackground")
print(f"wrote panel_{PANEL}.png")
