"""Render step (pvbatch + EGL, GPU): the three discretisations of the twisting box.

    pvbatch render_nc.py [std nit rot]      # default: all

  nc_panel_std.png     — body-fitted hex mesh, deformed
  nc_panel_nit.png     — axis-aligned box embedded in the octree, deformed
  nc_panel_rot.png     — 45deg-rotated box in the same octree, deformed
  nc_panel_nit_phi.png — the raw trimmed aligned octree, cut open over phi
  nc_panel_rot_phi.png — the same for the rotated case

All three are the mesh the solver actually used, warped by its own solution and
coloured by |u| on ONE shared scale from ONE shared camera, so the panels can be
compared cell for cell.  `nc_prep.py` rotates the rotated case back into the box
frame, so its BODY lines up with the other two and its OCTREE is the thing that
comes out oblique — which is the point: same geometry, same physics, a
background mesh that shares none of its faces.
"""

import os
import sys

from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

os.chdir(os.path.abspath(os.environ.get("NC_WORKDIR", os.path.dirname(__file__))))

PANELS = sys.argv[1:] or ["std", "nit", "rot", "nit_phi", "rot_phi"]
SIZE = [1500, 1500]

# Two fields, two ramps.  Both are MAGNITUDES, so both are sequential — one hue,
# light -> dark (dataviz skill, color-formula.md) — and they must not be the same
# hue, or a reader flicking between the rows reads one bar for the other.
#
# phi keeps diffused_nitsche's validated sequential BLUE (steps 100 -> 700), so
# the phase field looks the same in both figures.  |u| therefore takes the warm
# one, generated to the SAME lightness ladder as that blue (OKLCH L 0.905 ->
# 0.338 in even 0.095 steps) at a fixed hue of 52 deg and 92 % of the in-gamut
# chroma at each step — so equal fractions of the two fields carry equal
# perceptual weight, and neither bar shouts over the other.
PHI_RGB_STOPS = [(0.804, 0.886, 0.984), (0.620, 0.773, 0.957),
                 (0.427, 0.655, 0.925), (0.224, 0.529, 0.898),
                 (0.145, 0.416, 0.749), (0.094, 0.310, 0.584),
                 (0.051, 0.212, 0.420)]
# Six steps, not seven: |u| covers the whole of every solution panel, and with
# the darkest step (OKLCH L 0.34) in the ramp the twisted end of the box went to
# a near-black brown that swallowed its own cell frames.  Stopping at L 0.43
# keeps the same ladder and the same hue and leaves the frames legible.  phi
# keeps all seven — its dark end is a flat interior, not a shaded surface.
U_RGB_STOPS = [(0.991, 0.843, 0.757), (0.982, 0.679, 0.493),
               (0.956, 0.500, 0.155), (0.791, 0.410, 0.121),
               (0.633, 0.323, 0.088), (0.483, 0.241, 0.057)]

UMAX = 0.82        # = 2 sin(theta/2) * 0.75*sqrt(2), the twisted face's corners


def _ramp(stops, vmax=1.0):
    out = []
    for i, c in enumerate(stops):
        out += [vmax * i / (len(stops) - 1), *c]
    return out


U_RGB = _ramp(U_RGB_STOPS, UMAX)
PHI_RGB = _ramp(PHI_RGB_STOPS, 1.0)

# The phi panels are cut open on this plane to show the diffuse interface INSIDE
# the background mesh (the look of bunny_sdf_octree).  The camera sits on -y, so
# the far half (y >= origin) is the one whose cut face points at the viewer.
# Nudged a hair off 0: y = 0 is a cell boundary at every octree level, and
# clipping exactly on one makes ParaView emit degenerate slivers that read as
# broken frames (AGENTS.md §5 "Clip gotcha").
CUT_Y = 0.006
# The phi panels keep the solution panels' DIRECTION and frame the half that is
# left after the cut — the bunny_sdf_octree look, where the cut cube fills its
# panel.  (A true close-up on the interface was tried: the band and the cells
# crossing it read beautifully and the object stops being recognisable, which
# for a panel whose job is "this is the mesh the solver saw" is the wrong
# trade.)  Focus is the kept half's centroid, not the box centre.
# Centred on the domain cube (which the frame draws), not on the kept half.
# 2.05 covers the rotated domain's corners (1.125*sqrt(3) = 1.95); a tighter fit
# CLIPS geometry against the viewport, which the ink crop downstream cannot undo
# (AGENTS.md §5) — nc_figure.py asserts no panel's ink touches its border, and
# that assert is what caught it at 1.08.
PHI_FOCUS = (0.0, 0.0, 0.0)
PHI_RADIUS = 2.05

# The twist axis is +z here (the bunny figures are y-up), so the camera looks
# slightly down a long the side: the clamped face, the twisted face and the
# shear between them are all visible in one view.
VIEW_DIR = [1.15, -0.95, 0.62]
VIEW_UP = [0.0, 0.0, 1.0]
VIEW_ANGLE = 25
DOLLY = 1.10


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(size=SIZE):
    """bunny_sdf_pv's view, verbatim (see AGENTS.md §5): white, FXAA, tone
    mapping, soft LightKit.  ``UseSSAO`` is absent in 5.13.3 and just skips."""
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = size
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = VIEW_ANGLE
    for attr, val in (("UseColorPaletteForBackground", 0), ("UseFXAA", 1),
                      ("UseSSAO", 1), ("UseToneMapping", 1),
                      ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                      ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
                      ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0),
                      ("HeadLightKHRatio", 3.0)):
        _try(lambda a=attr, v=val: setattr(rv, a, v))
    return rv


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv, radius=1.35, focus=(0.0, 0.0, 0.0)):
    """Frame a sphere of `radius` about `focus` at 25 deg (needs 4.51 r,
    AGENTS.md §5b).  The three solution panels all use the default — one camera,
    one scale, whatever their own bounds are.  The phi panels keep the same
    DIRECTION and move in (§5b: a close-up keeps its parent's direction, or the
    reader can no longer place it)."""
    d = _unit(VIEW_DIR)
    dist = 4.51 * radius * DOLLY
    rv.CameraFocalPoint = list(focus)
    rv.CameraPosition = [focus[i] + d[i] * dist for i in range(3)]
    rv.CameraViewUp = VIEW_UP


def mesh_actor(rv, src, array="umag", line_width=0.8):
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface With Edges"
    ColorBy(d, ("POINTS", array))  # noqa: F405
    lut = GetColorTransferFunction(array)  # noqa: F405
    lut.ColorSpace = "Lab"
    lut.RGBPoints = U_RGB if array == "umag" else PHI_RGB
    lut.NanColor = [0.6, 0.6, 0.6]
    d.Ambient = 0.28          # diffused_nitsche's matte block
    d.Diffuse = 0.85
    d.Specular = 0.0
    _try(lambda: setattr(d, "EdgeColor", [0.10, 0.10, 0.13]))
    _try(lambda: setattr(d, "LineWidth", line_width))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    return d


def cut_open(src, near=False):
    """Halve on CUT_Y.  By default keep the far half, whose cut face points at
    the camera; ``near=True`` keeps the other one (the void that leaves)."""
    c = Clip(Input=src)  # noqa: F405
    c.ClipType = "Plane"
    c.ClipType.Origin = [0.0, CUT_Y, 0.0]
    c.ClipType.Normal = [0.0, 1.0, 0.0]
    c.Invert = 1 if near else 0       # keep y >= CUT_Y (camera is on -y)
    c.UpdatePipeline()
    return c


def domain_frame(rv, tag):
    """The background domain cube as a wireframe — bunny_sdf_octree's outline.

    It is the box the OCTREE was built in (2.0 aligned, 2.25 rotated), not the
    geometry, so it says how much of the background the trimmer threw away; and
    in the rotated panel it is the frame, not the body, that comes out oblique.
    Flat-shaded: under the light kit a thin shaded stroke goes grey-blue and
    disappears (AGENTS.md §5b).
    """
    src = OpenDataFile(f"nc_{tag.split('_')[0]}_frame.vtk")  # noqa: F405
    src.UpdatePipeline()
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = [0.30, 0.32, 0.35]
    d.Ambient = 1.0
    d.Diffuse = 0.0
    d.Specular = 0.0
    _try(lambda: setattr(d, "LineWidth", 1.3))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    return d


def ghost_geometry(rv):
    """The embedded geometry itself — the 1.5^3 box — translucent and WHOLE.

    This is the bunny figure's move: the octree shows what the solver meshed,
    the ghost shows what it was meshing.  Whole, not clipped to the removed
    half: the camera's right is +y, so the two halves sit side by side rather
    than one in front of the other, and a half box seen into from its open side
    reads as two flat grey panels, not as a body.  The far half is inside the
    opaque cells and hides itself.
    """
    box = Box(Center=[0.0, 0.0, 0.0], XLength=1.5, YLength=1.5, ZLength=1.5)  # noqa: F405
    d = Show(box, rv)  # noqa: F405
    # With edges: a translucent BOX with no edges is just a grey area — the
    # bunny's ghost reads because a bunny has form, and this one has to be given
    # some.  The Box source is one quad per face, so this draws exactly its 12
    # edges.
    d.Representation = "Surface With Edges"
    _try(lambda: setattr(d, "EdgeColor", [0.45, 0.44, 0.42]))
    _try(lambda: setattr(d, "LineWidth", 1.0))
    d.DiffuseColor = [0.92, 0.90, 0.86]      # bunny_sdf_pv's warm off-white
    d.Opacity = 0.30
    d.Ambient = 0.25
    d.Diffuse = 0.90
    d.Specular = 0.10
    return d


# Confirm the GPU: without the glvnd override this silently renders on llvmpipe
# (AGENTS.md §5), which is slow AND shades differently.
print("renderer:", GetOpenGLInformation().GetRenderer())  # noqa: F405

def crop_to_geometry(src, half=0.75):
    """Cut the octree down to the geometry — the 1.5^3 box — with six planes.

    Six explicit plane clips rather than a Box clip type: they are exact, and
    their properties do not move between ParaView versions.  This runs on the
    UNDEFORMED mesh and is followed by the warp, never the other way round: a
    whole-cell crop of a warped mesh gives the staircase, which is a picture of
    the mesh and not of the body.
    """
    out = src
    for axis in range(3):
        for sign in (+1.0, -1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            c = Clip(Input=out)  # noqa: F405
            c.ClipType = "Plane"
            c.ClipType.Origin = [sign * half if i == axis else 0.0
                                 for i in range(3)]
            c.ClipType.Normal = n
            c.Invert = 1                    # keep the side the normal points away from
            out = c
    out.UpdatePipeline()
    return out


def warp(src):
    """Deform by the nodal displacement the prep stored as a vector."""
    w = WarpByVector(Input=src)  # noqa: F405
    w.Vectors = ["POINTS", "u"]
    w.ScaleFactor = 1.0
    w.UpdatePipeline()
    return w


for tag in PANELS:
    rv = new_view()
    src = OpenDataFile(f"nc_{tag}.vtk")  # noqa: F405
    src.UpdatePipeline()
    n_all = src.GetDataInformation().GetNumberOfCells()
    if tag.endswith("_phi"):
        # The RAW trimmed octree, undeformed, cut open over phi: no cell filter,
        # because the cells the interface passes THROUGH are the subject.
        body = cut_open(src)
        mesh_actor(rv, body, array="phi", line_width=1.0)
        domain_frame(rv, tag)
        ghost_geometry(rv)
    else:
        body = src if tag == "std" else warp(crop_to_geometry(src))
        mesh_actor(rv, body, array="umag")
    print(f"[{tag}] {n_all} cells -> "
          f"{body.GetDataInformation().GetNumberOfCells()} drawn")
    Render(rv)  # noqa: F405
    if tag.endswith("_phi"):
        fit(rv, radius=PHI_RADIUS, focus=PHI_FOCUS)
    else:
        fit(rv)
    Render(rv)  # noqa: F405
    SaveScreenshot(f"nc_panel_{tag}.png", rv, ImageResolution=SIZE,  # noqa: F405
                   OverrideColorPalette="WhiteBackground")
    print(f"wrote nc_panel_{tag}.png")
    Delete(rv)  # noqa: F405
