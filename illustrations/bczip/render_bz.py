"""Render step (pvbatch + EGL, GPU): the panels of the BC-zip figure.

    pvbatch render_bz.py [panel ...]        # default: all

One shared camera and one shared material for the seven step panels, so the
reader tracks a single object through the workflow; panels ``5z``/``6z``/``7z``
repeat the same three stages under a tight camera on one zip strip, which is the
only scale at which the long-edge subdivision and the Taubin pass are visible.
Lighting is bunny_sdf_pv's (AGENTS.md §5).
"""

import sys

import os
from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

PANELS = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "5z", "6z", "7z"]
SIZE = [1800, 1800]
# Camera per case -- the two geometries do not share an up axis, let alone a
# framing.  The bracket is a flat plate in the xy plane (y up); the table is a
# z-up solid whose subject is a leg seam near the floor.  Reusing the bracket
# camera on the table renders the tabletop face-on and puts the zoom on its rim.
CASE = "table"
for _line in open("bz_stats.txt"):
    if _line.startswith("case "):
        CASE = _line.split()[1]
if CASE == "table":
    VIEW_DIR = [0.55, -1.0, 0.30]   # three-quarter, from below the tabletop
    VIEW_UP = [0.0, 0.0, 1.0]
    FOCUS = [0.0, 0.0, -0.05]
    FOCUS_R = 0.756   # holds the removal boxes of (b) too
    # One foot's zip collar: the band is z in [-0.390, -0.263] on a leg whose
    # axis sits at radius 0.306.
    ZOOM_FOCUS = [0.268, 0.004, -0.344]
    ZOOM_R = 0.086
else:
    VIEW_DIR = [0.34, 0.20, 1.0]    # near face-on: the bracket is a flat plate
    VIEW_UP = [0.0, 1.0, 0.0]
    FOCUS = [0.0, -0.02, 0.0]
    FOCUS_R = 0.68
    # The top arm's zip strip: one hole loop, one patch loop, no bridging — the
    # cleanest place to read step 6 against step 7.
    ZOOM_FOCUS = [0.0, 0.24, 0.0]
    ZOOM_R = 0.19
VIEW_ANGLE = 25

BODY = [0.922, 0.902, 0.859]        # #ebe6db  untouched base surface
CAND0 = [0.631, 0.239, 0.255]       # #a13d41  the 1.1x AABB candidate
CANDG = [0.851, 0.565, 0.576]       # #d99093  what the BFS rings added
BLUE = [0.298, 0.447, 0.690]        # #4c72b0  bridges / zip strips
VIOLET = [0.663, 0.612, 0.847]      # #a99cd8  fan caps (none on the
                                    # table, which leaves no nested loop;
                                    # kept equal to render_bzx's, where
                                    # caps do appear and the old mid
                                    # violet collided with BLUE under CVD)
GREEN = [0.333, 0.659, 0.408]       # #55a868  the restored BC patch;
                                    # the house green of bc_region_id
INK = [0.10, 0.10, 0.13]

stats = {}
for line in open("bz_stats.txt"):
    k, *v = line.split()
    stats[k] = v


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view():
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = VIEW_ANGLE
    for a, v in (("UseColorPaletteForBackground", 0), ("UseFXAA", 1),
                 ("UseSSAO", 1), ("UseToneMapping", 0),
                 ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                 ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
                 ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0),
                 ("HeadLightKHRatio", 3.0)):
        _try(lambda a=a, v=v: setattr(rv, a, v))
    return rv


def show_mesh(rv, path, line_width, edges):
    """Surface coloured by the per-cell `stage` tag, with cell frames."""
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    src.UpdatePipeline()
    d = Show(src, rv)  # noqa: F405
    # Cell frames only where the triangulation is the subject -- steps 5 and 6
    # are read by how the strip is triangulated; everywhere else the edges are
    # noise over what the panel is actually about.
    # Frames only where the triangulation is the subject -- steps 5 and 6.  This
    # is decided by PANEL, not by filename: the patch overlay is shared with (a),
    # where its frame is noise over the cached region.
    d.Representation = "Surface With Edges" if edges else "Surface"
    ColorBy(d, ("CELLS", "stage"))  # noqa: F405
    lut = GetColorTransferFunction("stage")  # noqa: F405
    lut.ColorSpace = "RGB"
    # Hard steps, one per stage tag written by bz_prep.write_poly:
    #   0 base · 1 AABB candidate · 2 ring growth · 3 bridge · 4 cap
    #   5 BC patch · 6 zip
    lut.RGBPoints = [0.0, *BODY, 0.5, *BODY,
                     0.51, *CAND0, 1.5, *CAND0,
                     1.51, *CANDG, 2.5, *CANDG,
                     2.51, *BLUE, 3.5, *BLUE,
                     3.51, *VIOLET, 4.5, *VIOLET,
                     4.51, *GREEN, 5.5, *GREEN,
                     5.51, *BLUE, 6.0, *BLUE]
    lut.NanColor = [0.6, 0.6, 0.6]
    d.Ambient = 0.16
    d.Diffuse = 0.95
    d.Specular = 0.22
    d.SpecularPower = 30
    _try(lambda: setattr(d, "EdgeColor", [0.42, 0.42, 0.46]))
    _try(lambda: setattr(d, "LineWidth", line_width))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    return src, d


def show_loops(rv, path, width):
    """Open boundary loops — flat-shaded thick tubes so they read at print size."""
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface"
    d.AmbientColor = INK
    d.DiffuseColor = INK
    d.Ambient = 1.0
    d.Diffuse = 0.0
    d.Specular = 0.0
    d.LineWidth = width
    _try(lambda: setattr(d, "RenderLinesAsTubes", 1))
    return d


def show_aabb(rv, path):
    """EVERY removal box.  One box per patch component, drawn from the line mesh
    bz_prep writes -- drawing only the first implied the other components were
    never selected."""
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    src.UpdatePipeline()
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface"
    ColorBy(d, None)  # noqa: F405   solid, not through the stage LUT
    # A box is the outline of the component it removes, so it wears that colour.
    d.AmbientColor = GREEN
    d.DiffuseColor = GREEN
    d.LineWidth = 3.0
    return d


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv, focus, r, fit_r=4.7):
    """At a 25 deg FOV a sphere of radius r needs 4.51 r (AGENTS.md §5b); 4.7
    leaves margin for the AABB outline and the loop tubes."""
    d = _unit(VIEW_DIR)
    rv.CameraFocalPoint = list(focus)
    rv.CameraPosition = [focus[i] + d[i] * r * fit_r for i in range(3)]
    rv.CameraViewUp = VIEW_UP


def save(rv, tag):
    Render(rv)  # noqa: F405
    SaveScreenshot(f"bz_panel_{tag}.png", rv, ImageResolution=SIZE,  # noqa: F405
                   OverrideColorPalette="WhiteBackground")
    print(f"wrote bz_panel_{tag}.png")


# Panel -> (mesh files, loop file or None, draw the AABB?)
SPEC = {
    "1": (["bz_p1_body.vtk", "bz_patch_all.vtk"], None, True),
    "2": (["bz_p2.vtk"], None, True),
    "3": (["bz_p3.vtk", "bz_patch_all.vtk"], None, False),
    "4": (["bz_p4.vtk"], None, False),
    "5": (["bz_p5.vtk", "bz_patch_all.vtk"], None, False),
    "6": (["bz_p6.vtk", "bz_patch_all.vtk"], None, False),
    "7": (["bz_p7.vtk"], None, False),
}

for tag in PANELS:
    zoom = tag.endswith("z")
    meshes, loops, aabb = SPEC[tag[:-1] if zoom else tag]
    # A step that did not run on this geometry leaves no VTK (bz_prep removes a
    # stale one).  Skip its panel rather than rendering an empty view.
    if not all(os.path.exists(m) for m in meshes):
        print(f"skip panel {tag}: {meshes[0]} not written for this case")
        continue
    rv = new_view()
    for m in meshes:
        show_mesh(rv, m, 1.6 if zoom else 0.6,
                  edges=(tag[:-1] if zoom else tag) in ('5', '6'))
    if loops and not zoom:
        show_loops(rv, loops, 5.0)
    if aabb and not zoom:
        show_aabb(rv, f"bz_p{tag}_boxes.vtk")
    Render(rv)  # noqa: F405
    fit(rv, ZOOM_FOCUS if zoom else FOCUS, ZOOM_R if zoom else FOCUS_R)
    save(rv, tag)
    Delete(rv)  # noqa: F405

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
