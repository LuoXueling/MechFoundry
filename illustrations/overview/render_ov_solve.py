"""Render the deformed bunny surface: diffused_nitsche's panel (f), and the
Nitsche-solve panel of method_overview — the same recipe on its loop's solve
(DN_SOLVE_PREFIX=ov_loop_solve, written in the display frame), framed as the
overview's other bunnies are: DN_SOLVE_FIT names the surface the camera is
fitted to and DN_SOLVE_DOLLY its dolly (defaults: panel (f)'s octree cube, 1.16),
onto its block's tint, DN_SOLVE_BG (default white)."""
import os
from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

rv = CreateRenderView()  # noqa: F405
rv.ViewSize = [1500, 1500]
# White, as diffused_nitsche prints it; method_overview renders this panel
# opaque onto its block's tint ("r,g,b" in 0..1) instead of keying out white.
rv.Background = [float(c) for c in os.environ.get("DN_SOLVE_BG", "1,1,1").split(",")]
rv.OrientationAxesVisibility = 0
rv.CameraViewAngle = 25
for attr, val in (("UseColorPaletteForBackground", 0),
                  ("UseFXAA", 1), ("UseToneMapping", 1),
                  ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                  ("KeyLightIntensity", 0.85), ("FillLightKFRatio", 2.6)):
    try:
        setattr(rv, attr, val)
    except Exception:
        pass

PREFIX = os.environ.get("DN_SOLVE_PREFIX", "ov_bunny_solve")
OUTPUT = os.environ.get("DN_SOLVE_RENDER", f"{PREFIX}.png")
src = XMLUnstructuredGridReader(FileName=[f"{PREFIX}.vtu"])  # noqa: F405
src.UpdatePipeline()
warp = WarpByVector(Input=src)  # noqa: F405
warp.Vectors = ["POINTS", "displacement"]
# The increased load makes the deformation legible without visual scaling.
warp.ScaleFactor = 1.0

d = Show(warp, rv)  # noqa: F405
d.Representation = "Surface"
ColorBy(d, ("POINTS", "displacement_magnitude"))  # noqa: F405
lut = GetColorTransferFunction("displacement_magnitude")  # noqa: F405
umax = src.PointData["displacement_magnitude"].GetRange()[1]
u1, u2 = 0.35 * umax, 0.70 * umax
lut.ColorSpace = "Lab"
# Low displacement retains the warm body colour from panel (d); colour only
# emerges where the solved motion is appreciable.
lut.RGBPoints = [0.0, 0.920, 0.900, 0.860,
                 u1, 0.920, 0.900, 0.860,
                 u2, 0.867, 0.518, 0.322,
                 umax, 0.478, 0.129, 0.094]
pwf = GetOpacityTransferFunction("displacement_magnitude")  # noqa: F405
pwf.Points = [0.0, 0.24, 0.5, 0.0,
              u1, 0.30, 0.5, 0.0,
              u2, 0.72, 0.5, 0.0,
              umax, 1.0, 0.5, 0.0]
d.ScalarOpacityFunction = pwf
d.Opacity = 1.0
d.SetScalarBarVisibility(rv, False)
d.Interpolation = "Gouraud"
d.Ambient = 0.42
d.Diffuse = 0.88
d.Specular = 0.28
d.SpecularPower = 28

# Exact camera construction used by render_dn.fit_all.  Reset against the same
# full octree cube as panels (a)-(e), not against the deformed surface: the
# latter preserves direction but silently changes focal point and distance.
camera_ref = OpenDataFile(os.environ.get("DN_SOLVE_FIT", "dn_full.vtk"))  # noqa: F405
DOLLY = float(os.environ.get("DN_SOLVE_DOLLY", "1.16"))
camera_outline = Outline(Input=camera_ref)  # noqa: F405
camera_actor = Show(camera_outline, rv)  # noqa: F405
view_dir = [1.5, -3.0, 1.25]
norm = sum(c * c for c in view_dir) ** 0.5
direction = [c / norm for c in view_dir]
rv.CameraFocalPoint = [0.0, 0.0, 0.0]
rv.CameraPosition = [c * 3.0 for c in direction]
rv.CameraViewUp = [0.0, 0.0, 1.0]
ResetCamera(rv)  # noqa: F405
fp, pos = list(rv.CameraFocalPoint), list(rv.CameraPosition)
dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
rv.CameraPosition = [fp[i] + direction[i] * dist * DOLLY for i in range(3)]
Hide(camera_outline, rv)  # noqa: F405
Render(rv)  # noqa: F405
SaveScreenshot(OUTPUT, rv, ImageResolution=[1500, 1500],  # noqa: F405
               TransparentBackground=0)
print(f"wrote {OUTPUT}")
