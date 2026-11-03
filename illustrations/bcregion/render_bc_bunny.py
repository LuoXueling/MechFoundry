"""Render step (pvbatch, GPU): the three panels of the BC-region identification figure.

Run via ``build.sh`` (needs the NVIDIA glvnd override, else it silently falls back
to llvmpipe).  Consumes the files written by ``bc_bunny_select.py``.

  bc_panel_a.png — seed facets from a coordinate inquiry (ear tips)
  bc_panel_b.png — flood fill grown from that seed (both ear surfaces)
  bc_panel_c.png — close-up: the selected facets of both ears
"""

from paraview.simple import *   # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()   # noqa: F405

# seaborn-deep palette (AGENTS.md house style)
ORANGE = [0.867, 0.518, 0.322]
# warm off-white body, matching bunny_sdf_pv.png
BODY = [0.920, 0.900, 0.860]
BODY_PALE = [0.940, 0.928, 0.906]
PANEL_AB = [1800, 1800]
PANEL_C = [1800, 1800]
VIEW_ANGLE = 25                  # bunny_sdf_pv uses 25 deg, not the 30 deg default

# The selected inner-ear facets face nearly straight up (their area-weighted mean
# normal is ~(0.39, 0.91, -0.12)), so a side view shows none of the selection.
# This direction is tilted up enough to expose both ear cups while keeping the
# bunny in a recognizable three-quarter pose.  Shared by all three panels.
# The angle bunny_sdf_octree and diffused_nitsche use, so the three figures
# show the bunny the same way round (render_bunny_sdf.py's camera).
VIEW_DIR = (1.5, -3.0, 1.25)

stats = {}
for line in open("bc_stats.txt"):
    k, *v = line.split()
    stats[k] = v
focus = [float(q) for q in stats["focus"]]
focus_r = float(stats["focus_radius"][0])
focus_dir = [float(q) for q in stats["focus_dir"]]
ears_center = [float(q) for q in stats["ears_center"]]
ears_radius = float(stats["ears_radius"][0])
view_dir = [float(q) for q in stats["view_dir"]]   # ears' mean facet normal


def new_view(size):
    """Render view styled exactly like ``render_bunny_sdf.py``."""
    rv = CreateRenderView()                      # noqa: F405
    rv.ViewSize = size
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = VIEW_ANGLE
    for attr, val in (("UseColorPaletteForBackground", 0),
                      ("UseFXAA", 1), ("UseSSAO", 1),
                      # Tone mapping OFF: on an opaque body it crushes the warm
                      # off-white to mid-grey.  bunny_sdf_pv gets away with it
                      # because its bunny is translucent over a bright field.
                      ("UseToneMapping", 0),
                      ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                      ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
                      ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0),
                      ("HeadLightKHRatio", 3.0)):
        try:
            setattr(rv, attr, val)
        except Exception as e:
            print("  (skip)", attr, e)
    return rv


def show_surface(src, rv, color, opacity=1.0):
    """Smooth-shaded surface, bunny_sdf_pv colours and highlights.

    Deliberately NOT ``Interpolation="PBR"``.  ParaView's PBR path expects
    image-based lighting; with no environment map an opaque body renders a dull
    mid-grey no matter how bright the light kit is.  bunny_sdf_pv sets PBR on a
    42%-opaque bunny in front of a vivid SDF slice, so the darkening never shows
    there.  Standard Gouraud shading keeps the warm off-white actually warm.
    """
    disp = Show(src, rv)                          # noqa: F405
    disp.Representation = "Surface"
    disp.DiffuseColor = color
    disp.Opacity = opacity
    disp.Ambient = 0.12
    disp.Diffuse = 1.0
    disp.Specular = 0.35
    disp.SpecularPower = 30
    return disp


def smooth(src):
    """Gouraud-smooth a surface, as bunny_sdf_pv does before display."""
    nrm = GenerateSurfaceNormals(Input=src)       # noqa: F405
    try:
        nrm.FeatureAngle = 50
    except Exception:
        pass
    return nrm


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit_camera(rv, direction, dolly=1.02):
    """Aim along ``direction``, let ResetCamera fit the scene, then dolly back."""
    d = _unit(direction)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [c * 3.0 for c in d]
    rv.CameraViewUp = [0.0, 0.0, 1.0]
    rv.CameraParallelProjection = 0
    ResetCamera(rv)                                       # noqa: F405
    fp = list(rv.CameraFocalPoint)
    pos = list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * dolly for i in range(3)]


def aim_at(rv, direction, focal, dist):
    """Point the camera at ``focal`` from ``dist`` away along ``direction``."""
    d = _unit(direction)
    rv.CameraFocalPoint = list(focal)
    rv.CameraPosition = [focal[i] + d[i] * dist for i in range(3)]
    rv.CameraViewUp = [0.0, 0.0, 1.0]
    rv.CameraParallelProjection = 0


# ---------------------------------------------------------------------------
# Panels (a) and (b): whole body, shared camera
# ---------------------------------------------------------------------------
for tag, rest, overlay in (("a", "bc_rest_seed.stl", "bc_seed.stl"),
                           ("b", "bc_rest_ears.stl", "bc_ears.stl")):
    rv = new_view(PANEL_AB)
    # Body = the complement of the highlight, NOT the full surface: overlaying
    # the selected facets on a copy of themselves z-fights and speckles.
    body = STLReader(FileNames=[rest])                    # noqa: F405
    show_surface(smooth(body), rv, BODY)

    sel = STLReader(FileNames=[overlay])                  # noqa: F405
    show_surface(smooth(sel), rv, ORANGE)

    Render(rv)                                            # noqa: F405
    fit_camera(rv, VIEW_DIR, dolly=1.02)
    Render(rv)                                            # noqa: F405
    SaveScreenshot(f"bc_panel_{tag}.png", rv, ImageResolution=PANEL_AB,  # noqa: F405
                   OverrideColorPalette="WhiteBackground")
    print(f"wrote bc_panel_{tag}.png")
    Delete(rv)                                            # noqa: F405

# ---------------------------------------------------------------------------
# Panel (c): close-up — the selected facets of both ears
# ---------------------------------------------------------------------------
rv = new_view(PANEL_C)
# Only the ears' local neighbourhood, not the whole body: the unselected facets
# still occlude any gap inside the selected patch, but the head no longer fills
# the corner of the frame as a featureless grey slab.  The collar carries a
# per-facet `fade` scalar (0 at the patch, 1 at its outer ring) driving both
# colour -> white and opacity -> 0, so it dissolves instead of ending on a
# ragged, zigzag facet boundary.
ctx = LegacyVTKReader(FileNames=["bc_ears_context.vtk"])  # noqa: F405
dctx = show_surface(smooth(ctx), rv, BODY_PALE)
# Lit, but ambient-heavy.  At the panels' normal 0.12/1.0 the collar wrapping
# into the hollow between the ear roots goes so dark it reads as dirt; fully
# unshaded (1.0/0.0) removes the smudge but also every trace of form, so the
# mask looks like a flat cut-out.  0.65/0.45 keeps the shading that tells the
# reader it is a surface while holding the shadowed side pale.
dctx.Ambient = 0.65
dctx.Diffuse = 0.45
dctx.Specular = 0.0
ColorBy(dctx, ("CELLS", "fade"))                          # noqa: F405
# The inner half of the collar stays FULLY OPAQUE.  Fading it from the patch
# outward leaves nothing to hide the far side of the selection, and the back of
# the other ear shows through the head as a stray orange patch.  Opaque to
# fade=0.5 (~15 rings), then colour -> white and opacity -> 0 by 0.85, so the
# edge dissolves well away from the selection.
OPAQUE_TO, FADE_END = 0.5, 0.85
fade_lut = GetColorTransferFunction("fade")               # noqa: F405
fade_lut.ColorSpace = "RGB"
fade_lut.RGBPoints = [0.0, *BODY_PALE, OPAQUE_TO, *BODY_PALE,
                      FADE_END, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
fade_pwf = GetOpacityTransferFunction("fade")             # noqa: F405
fade_pwf.Points = [0.0, 1.0, 0.5, 0.0, OPAQUE_TO, 1.0, 0.5, 0.0,
                   FADE_END, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
fade_lut.EnableOpacityMapping = 1
dctx.SetScalarBarVisibility(rv, False)

ears = STLReader(FileNames=["bc_ears.stl"])               # noqa: F405
show_surface(smooth(ears), rv, ORANGE)

Render(rv)                                                # noqa: F405
# Both ears along the SAME direction as (a) and (b), so the reader can match the
# close-up to the ears they just saw.  A different viewpoint here (e.g. the
# selection's own mean normal) shows the patch more head-on but flips the
# ears' apparent arrangement and breaks that correspondence.
# At the 25-deg vertical FOV a sphere of radius r needs r/tan(12.5 deg) ~= 4.51 r.
aim_at(rv, VIEW_DIR, ears_center, ears_radius * 4.7)
Render(rv)                                                # noqa: F405
SaveScreenshot("bc_panel_c.png", rv, ImageResolution=PANEL_C,   # noqa: F405
               OverrideColorPalette="WhiteBackground")
print("wrote bc_panel_c.png")

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
