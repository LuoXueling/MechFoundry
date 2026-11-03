"""Render step (run with pvbatch + EGL env): SDF cube cut in half, bunny inside.

  pvbatch render_bunny_sdf.py

Publication-quality: smooth-shaded bunny, ambient occlusion, soft light kit,
high resolution + FXAA, and a cleanly styled colorbar.  Writes bunny_sdf_pv.png.
"""

from paraview.simple import *  # noqa: F403
import matplotlib

# The colorbar title carries the math symbol s.  VTK hands a string with $...$ to
# pvbatch's bundled matplotlib mathtext, so give mathtext the house Arial (roman AND
# italic) the way the matplotlib figures do; left alone it falls back to DejaVu.
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


with open("bunny_sdf.meta") as f:
    H = float(f.readline())
    Vin = float(f.readline())

sdf = OpenDataFile("bunny_sdf.vtk")               # ImageData, scalar 'sdf'
bunny = OpenDataFile("bunny.stl")

sdf.UpdatePipeline()
_ai = sdf.GetDataInformation().GetPointDataInformation().GetArrayInformation("sdf")
dmin, dmax = _ai.GetComponentRange(0)
print(f"sdf data range: [{dmin:.3f}, {dmax:.3f}]")

# ---------------------------------------------------------------------------
# Render view — white, high-res, ambient occlusion + FXAA, soft light kit
# ---------------------------------------------------------------------------
ARIAL = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"

rv = GetActiveViewOrCreate("RenderView")
rv.ViewSize = [2500, 1850]                          # extra width = clear right margin
rv.Background = [1, 1, 1]
rv.OrientationAxesVisibility = 0
_try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))
_try(lambda: setattr(rv, "UseFXAA", 1))
_try(lambda: setattr(rv, "UseSSAO", 1))            # screen-space ambient occlusion
_try(lambda: setattr(rv, "UseToneMapping", 1))
# soft three-point light kit (key from upper-left, gentle fill)
_try(lambda: setattr(rv, "KeyLightAzimuth", -35))
_try(lambda: setattr(rv, "KeyLightElevation", 45))
_try(lambda: setattr(rv, "KeyLightIntensity", 0.85))
_try(lambda: setattr(rv, "KeyLightWarmth", 0.55))
_try(lambda: setattr(rv, "FillLightKFRatio", 2.6))
_try(lambda: setattr(rv, "BackLightKBRatio", 3.0))
_try(lambda: setattr(rv, "HeadLightKHRatio", 3.0))

# --- solid half of the SDF cube (keep Y >= 0); faces colored by sdf ----------
clipV = Clip(Input=sdf)
clipV.ClipType = "Plane"
clipV.ClipType.Origin = [0.0, 0.0, 0.0]
clipV.ClipType.Normal = [0.0, 1.0, 0.0]
clipV.Invert = 0
dV = Show(clipV, rv)
dV.Representation = "Surface"
ColorBy(dV, ("POINTS", "sdf"))
# Diverging map: two hues + NEUTRAL-GRAY midpoint, CVD-safe blue<->orange,
# 0-centered independent slopes (full range, near-surface stays saturated).
lut = GetColorTransferFunction("sdf")
lut.ColorSpace = "Lab"
lut.RGBPoints = [
    dmin,   0.106, 0.196, 0.478,
    -0.130, 0.235, 0.443, 0.718,
    -0.040, 0.557, 0.706, 0.851,
    0.0,    0.894, 0.894, 0.894,
    0.040,  0.968, 0.805, 0.576,
    0.150,  0.944, 0.600, 0.282,
    0.380,  0.827, 0.329, 0.152,
    dmax,   0.478, 0.129, 0.094,
]
lut.NanColor = [0.6, 0.6, 0.6]
dV.Ambient = 0.28                                  # faces read as a matte block
dV.Diffuse = 0.85
dV.Specular = 0.0
dV.SetScalarBarVisibility(rv, True)

# --- cube outline (subtle) ---------------------------------------------------
dO = Show(Outline(Input=sdf), rv)
dO.DiffuseColor = [0.30, 0.32, 0.35]
dO.LineWidth = 1.2

# --- revealed near half of the bunny (keep Y <= 0), smooth + translucent ------
bunnyN = GenerateSurfaceNormals(Input=bunny)       # smooth (Gouraud) shading
_try(lambda: setattr(bunnyN, "FeatureAngle", 50))
clipB = Clip(Input=bunnyN)
clipB.ClipType = "Plane"
clipB.ClipType.Origin = [0.0, 0.0, 0.0]
clipB.ClipType.Normal = [0.0, 1.0, 0.0]
clipB.Invert = 1
dB = Show(clipB, rv)
dB.Representation = "Surface"
dB.DiffuseColor = [0.92, 0.90, 0.86]
dB.Opacity = 0.42
dB.Specular = 0.35
dB.SpecularPower = 30
dB.Ambient = 0.12
_try(lambda: setattr(dB, "Interpolation", "PBR"))
_try(lambda: setattr(dB, "Roughness", 0.45))
_try(lambda: setattr(dB, "Metallic", 0.0))

# --- styled colorbar (true Arial, right margin, clear of the cube) -----------
sb = GetScalarBar(lut, rv)
sb.Title = "Signed distance  $s$"
sb.ComponentTitle = ""
sb.TitleColor = [0.1, 0.1, 0.1]
sb.LabelColor = [0.1, 0.1, 0.1]
# use the real Arial .ttf (ParaView's built-in "Arial" is a look-alike)
for fam, fil in (("TitleFontFamily", "TitleFontFile"),
                 ("LabelFontFamily", "LabelFontFile")):
    _try(lambda f=fam: setattr(sb, f, "File"))
    _try(lambda f=fil: setattr(sb, f, ARIAL))
sb.TitleFontSize = 64
sb.LabelFontSize = 54
sb.TitleBold = 0
_try(lambda: setattr(sb, "ScalarBarThickness", 90))        # px
_try(lambda: setattr(sb, "ScalarBarLength", 0.52))         # frac of viewport
_try(lambda: setattr(sb, "WindowLocation", "Any Location"))
_try(lambda: setattr(sb, "Position", [0.80, 0.26]))        # right margin, mid-height
_try(lambda: setattr(sb, "AddRangeLabels", 0))
_try(lambda: setattr(sb, "UseCustomLabels", 1))
_try(lambda: setattr(sb, "CustomLabels", [-0.2, 0.0, 0.2, 0.4, 0.6]))
sb.AutomaticLabelFormat = 0
sb.LabelFormat = "%.1f"
_try(lambda: setattr(sb, "DrawTickMarks", 1))
_try(lambda: setattr(sb, "DrawTickLabels", 1))

# (OSPRay ray tracing tried — its correct multi-layer transparency turns the
# translucent bunny into busy "x-ray glass"; depth-peeled GL reads cleaner for
# a scientific figure, so we keep the raster GL path.)

# --- camera ------------------------------------------------------------------
rv.CameraFocalPoint = [0.0, 0.0, 0.0]
rv.CameraViewUp = [0.0, 0.0, 1.0]
rv.CameraPosition = [1.5 * H, -3.0 * H, 1.25 * H]
ResetCamera(rv)
# dolly the camera back so the cube shrinks and leaves a clear gap to the
# colorbar on the right (ResetCamera fills the frame edge-to-edge otherwise)
_fp = list(rv.CameraFocalPoint)
_cp = list(rv.CameraPosition)
rv.CameraPosition = [_fp[i] + (_cp[i] - _fp[i]) * 1.32 for i in range(3)]
rv.CameraViewAngle = 25
Render()

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
SaveScreenshot("bunny_sdf_pv.png", rv, ImageResolution=[2500, 1850],
               OverrideColorPalette="WhiteBackground")
print("wrote bunny_sdf_pv.png")
