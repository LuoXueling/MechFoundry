"""Render step (pvbatch + EGL, GPU): the panels of the BC-zip special-case figure.

Four panels, two geometries, one camera each -- unlike ``render_bz.py`` there is
no shared camera to keep, because the panels do not follow one object: each pair
is its own minimal example (see ``bzx_prep.py``).  Within a pair the camera IS
shared, so the reader sees the same view before and after the branch runs.
Materials, lighting and the ``stage`` LUT are ``render_bz.py``'s, so the two
figures read as one family.  Lighting is bunny_sdf_pv's (AGENTS.md §5).
"""

import sys

import os
from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

PANELS = sys.argv[1:] or ["a", "b", "c", "d"]
SIZE = [1800, 1800]
VIEW_ANGLE = 25

BODY = [0.922, 0.902, 0.859]        # #ebe6db  untouched base surface
BLUE = [0.298, 0.447, 0.690]        # #4c72b0  bridges
VIOLET = [0.663, 0.612, 0.847]      # #a99cd8  fan caps.  A LIGHT violet, not
                                    # the mid one this figure started with:
                                    # #8171b3 sits at CVD dE 1.6 from the house
                                    # blue beside it in (e), which no amount of
                                    # captioning fixes.  Lightness is the only
                                    # axis left once cream, green, blue and red
                                    # are spoken for.
RED = [0.631, 0.239, 0.255]         # #a13d41  the loop that gets the region refused
INK = [0.10, 0.10, 0.13]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(light=()):
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
                 ("HeadLightKHRatio", 3.0)) + tuple(light):
        _try(lambda a=a, v=v: setattr(rv, a, v))
    return rv


def show_mesh(rv, path, color, edges, line_width=1.2):
    """One mesh, one material.  SOLID colour, not the `stage` LUT: with scalar
    colouring on, VTK paints backfaces from the scalars too and the backface
    material below has no effect -- and every panel in this figure looks into
    something the removal opened."""
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    src.UpdatePipeline()
    # Legacy polydata carries no normals, so VTK shades every triangle flat and
    # a 16-gon rod reads as a 16-sided prism -- the "faceted" look is the
    # SHADING, not the mesh: the silhouette of that rod is off a circle by
    # 0.0013, which is 0.05 mm on the page.  Generating vertex normals at a 45
    # deg feature angle rounds the barrels while keeping every genuine crease
    # (the cut faces meet the barrel at 90 deg, the flare at ~5).  It moves
    # nothing: Taubin on the base mesh would, and then the panels would no
    # longer be the surface the algorithm ran on.
    src = GenerateSurfaceNormals(Input=src)  # noqa: F405
    src.FeatureAngle = 45
    src.UpdatePipeline()
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface With Edges" if edges else "Surface"
    ColorBy(d, None)  # noqa: F405
    d.AmbientColor = color
    d.DiffuseColor = color
    d.Ambient = 0.16
    d.Diffuse = 0.95
    d.Specular = 0.22
    d.SpecularPower = 30
    # Anything seen from INSIDE gets the same material, dimmed.  Every panel here
    # looks into something the removal opened -- a crater, a bore, a cut leg --
    # and with backfaces shaded like frontfaces a lit bore wall reads as a solid
    # rod standing in a tube.  Dimming (not darkening) keeps the object its own
    # colour: at 0.4 the bar went black, and the fan cap -- whose winding faces
    # away from this camera -- went black with it.
    # ...but only the base meshes, which are what one looks INTO.  The overlays
    # (cap, bridge) are single sheets whose back is not a different thing --
    # and a backface REPRESENTATION of "Surface" is a plain surface, so it drops
    # the edges, which is exactly what the fan is drawn for.  Case C is turned
    # over for the figure, so its cap is seen from behind: leave overlays
    # following the frontface.  Drawn-with-edges == overlay.
    if edges:
        _try(lambda: setattr(d, "BackfaceRepresentation", "Follow Frontface"))
    else:
        back = [0.82 * c for c in color]
        for a, v in (("BackfaceRepresentation", "Surface"),
                     ("BackfaceAmbientColor", back),
                     ("BackfaceDiffuseColor", back)):
            _try(lambda a=a, v=v: setattr(d, a, v))
    # The only meshes drawn with edges are the ones a step ADDED, and they are
    # small: a hairline frame on 47 triangles is invisible at 1.3 in wide.
    _try(lambda: setattr(d, "EdgeColor", [0.22, 0.22, 0.27]))
    _try(lambda: setattr(d, "LineWidth", line_width))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    return src, d


def show_loops(rv, path, color, width):
    """Open boundary loops — flat-shaded thick tubes so they read at print size."""
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    src.UpdatePipeline()
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface"
    ColorBy(d, None)  # noqa: F405
    d.AmbientColor = color
    d.DiffuseColor = color
    d.Ambient = 1.0
    d.Diffuse = 0.0
    d.Specular = 0.0
    d.LineWidth = width
    _try(lambda: setattr(d, "RenderLinesAsTubes", 1))
    return d


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv, cam, focus, r, fit_r=4.55):
    """At a 25 deg FOV a sphere of radius r needs 4.51 r (AGENTS.md §5b); 4.55
    is just enough margin for the loop tubes, which stand proud of the surface.
    Every 0.1 above that is white space inside every panel."""
    d = _unit(cam["dir"])
    rv.CameraFocalPoint = list(focus)
    rv.CameraPosition = [focus[i] + d[i] * r * fit_r for i in range(3)]
    rv.CameraViewUp = cam.get("up", [0.0, 0.0, 1.0])


# One camera per GEOMETRY, not per panel: a before/after pair that moved the
# camera would ask the reader to re-find the subject between two panels whose
# whole point is what changed.
CAM = {
    # A: 19 degrees off the clamped face's own normal -- the widest angle that
    # still sees daylight through the bore (the stub is 0.38 long, the bore 0.17
    # across), and enough obliquity to read as a solid rather than a disc.
    "bar": dict(dir=[-1.013, 0.28, -0.119], focus=[-0.130, 0.0, -0.042], r=0.28),
    # B: 34 degrees off the cut plane's normal.  Everything step 4 added lies in
    # that plane and wants a head-on view; the struts and the head they run back
    # into want the opposite, and since the bridge became a proper strip (a
    # uniform mesh, not marching cubes) it survives the foreshortening.
    "pad": dict(dir=[-0.82, -0.40, 0.41], focus=[-0.30, 0.0, 0.0], r=0.36),
}

# panel -> (camera, [(mesh, colour, edges)], [(loop file, colour, width)]).
# Edges are on for the triangles a step ADDED and off for everything else: the
# fan and the bridge strip are read by how they are triangulated, and the base
# mesh's own 11k triangles would bury them.  Each "after" panel redraws its
# "before" panel's base file unchanged, so the pair differs by exactly what the
# step added.
# Loop tubes at 14 px of an 1800 px render -- 0.34 mm on the printed page.  At
# the 6 px they started at they were a hairline and the panels read as if the
# loops were not drawn at all; they are the subject of every one of them.
SPEC = {
    "a": ("bar", [("bzx_a_removed.vtk", BODY, False)],
          [("bzx_a_loop_outer.vtk", INK, 14.0),
           # Red is "the nesting test flagged this one".  Colouring it violet to
           # rhyme with the cap it is about to get would have made the same
           # thing two colours in one figure.
           ("bzx_a_loop_inner.vtk", RED, 15.0)]),
    "b": ("bar", [("bzx_a_removed.vtk", BODY, False),
                  ("bzx_a_cap.vtk", VIOLET, True)],
          [("bzx_a_loop_outer.vtk", INK, 14.0)]),
    # The cached patch is deliberately absent: it sits between the camera and
    # the cut struts and hides the bridge and the cap.
    "c": ("pad", [("bzx_b_removed.vtk", BODY, False)],
          [("bzx_b_loops.vtk", INK, 14.0)]),
    "d": ("pad", [("bzx_b_removed.vtk", BODY, False),
                  ("bzx_b_bridge.vtk", BLUE, True),
                  ("bzx_b_cap.vtk", VIOLET, True)],
          [("bzx_b_loops_after.vtk", INK, 14.0)]),
}

for tag in PANELS:
    cam_key, meshes, loops = SPEC[tag]
    missing = [m for m, *_ in meshes if not os.path.exists(m)]
    if missing:
        print(f"skip panel {tag}: {missing[0]} not written — run bzx_prep.py")
        continue
    cam = CAM[cam_key]
    rv = new_view(cam.get("light", ()))
    for path, color, edges in meshes:
        show_mesh(rv, path, color, edges)
    for path, color, width in loops:
        if os.path.exists(path):
            show_loops(rv, path, color, width)
    Render(rv)  # noqa: F405
    fit(rv, cam, cam["focus"], cam["r"])
    Render(rv)  # noqa: F405
    SaveScreenshot(f"bzx_panel_{tag}.png", rv, ImageResolution=SIZE,  # noqa: F405
                   OverrideColorPalette="WhiteBackground")
    print(f"wrote bzx_panel_{tag}.png")
    Delete(rv)  # noqa: F405

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
