"""Stage 2 (pvbatch + EGL, GPU): one PNG per panel of agentic_results.

    pvbatch render_ar.py [WORK_DIR]          # default ./_work, after ar_prep.py

Each panel is drawn exactly as its method_overview counterpart (ar_style holds
both recipes, taken from ov_style and render_ov_solve):

  painted / bc / materials   render_ov_panels.surface(): SURF_VIEW light kit
                             (tone mapping off), SURF_FINISH, normals at a 50 deg
                             feature angle, colours straight from the data
                             (COLOR_SCALARS, MapScalars 0); paint at specular 0.08;
                             seg (AR_SEGMENTATION=1) the same, its tints per facet
  solve                      render_ov_solve.py: WarpByVector, |u| through its
                             body -> orange -> brick ramp (0.35 / 0.70 / 1 of the
                             row's |u|max), its light kit with tone mapping on, its
                             Gouraud finish.  One addition: normals at the same
                             50 deg feature angle after the warp, as the other three
                             panels of the row have; the overview's bunny is 160 k
                             facets and smooth without them, a CAD body is not.

Both kits are Figure 1's with one change, the key light's angles (ar_style.KEY_LIGHT,
AR_SURF_VIEW / AR_SOLVE_VIEW): on the view axis at elevation 30 instead of 35 deg to
the camera's left at 45, so the faces looking at the viewer are the lit ones whichever
side of the object the row's camera stands on.

All four panels of a row share one camera: the paper's bunny view rotated into
the workflow's y-up frame (ar_style.VIEW_DIR), ResetCamera on ar_<i>_fit.vtk
(the undeformed AND the scaled deformed surface, so neither leaves the frame),
dolly 1.02.  A row whose record gives ``draw_scale`` f < 1 (ar_runs: the per-object
size override) then widens the view angle so the body projects exactly f x as
large (tan of the half angle / f; perspective unchanged): the compose crops
the row f x smaller, and the wider frame keeps the same room around the body
for the ground and its fade.  Each panel is opaque on white (the overview's tints were dropped
for this figure at the user's request).
"""

import json
import math
import os
import sys

from paraview.simple import *  # noqa: F403

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ar_style as A  # noqa: E402

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405
WORK = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_work"))
ROW = {"draw_scale": 1.0}      # the row being rendered (its per-object size override)
SIZE = [A.PANEL_PX, A.PANEL_PX]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(kit):
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = A.VIEW_ANGLE
    for attr, val in kit:
        _try(lambda a=attr, v=val: setattr(rv, a, v))
    return rv


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv, fit_path, view_dir):
    """render_ov_panels.fit: ResetCamera on a hidden reference, dolly back."""
    ref = Outline(Input=LegacyVTKReader(FileNames=[fit_path]))  # noqa: F405
    Show(ref, rv)  # noqa: F405
    d = _unit(view_dir)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [3.0 * c for c in d]
    rv.CameraViewUp = A.VIEW_UP
    rv.CameraParallelProjection = 0
    Render(rv)  # noqa: F405
    ResetCamera(rv)  # noqa: F405
    fp, pos = list(rv.CameraFocalPoint), list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * A.DOLLY for i in range(3)]
    f = ROW["draw_scale"]
    if f < 1:       # after the fit: ResetCamera would move in to the wider angle
        rv.CameraViewAngle = math.degrees(2 * math.atan(math.tan(math.radians(A.VIEW_ANGLE) / 2) / f))
    Hide(ref, rv)  # noqa: F405


def save(rv, out, panel, fit_path, view_dir, transparent=True, after_fit=None):
    """Visible panels are saved with a TRANSPARENT background: the compose takes
    an exact alpha for the ink box, the factor-label clearance and the contact
    shadow (SPEC_ledger, renderer change 1).  The id pass stays opaque."""
    fit(rv, fit_path, view_dir)
    if after_fit:       # added after the camera is fitted: ResetCamera frames every
        after_fit(rv)   # visible actor, and the ground must not move the camera
    rv.Background = A.BG       # white: the figure's panels carry no tint
    _try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))
    Render(rv)  # noqa: F405
    SaveScreenshot(out, rv, ImageResolution=SIZE,  # noqa: F405
                   TransparentBackground=1 if transparent else 0)
    print(f"wrote {os.path.basename(out)}")
    Delete(rv)  # noqa: F405


def surface_panel(path, assoc, specular=None):
    rv = new_view(A.AR_SURF_VIEW)
    normals = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[path]))  # noqa: F405
    _try(lambda: setattr(normals, "FeatureAngle", A.FEATURE_ANGLE))
    d = Show(normals, rv)  # noqa: F405
    d.Representation = "Surface"
    for k, v in A.SURF_FINISH.items():
        setattr(d, k, v)
    if specular is not None:
        d.Specular = specular
    ColorBy(d, (assoc, "rgb"))  # noqa: F405
    d.MapScalars = 0
    d.SetScalarBarVisibility(rv, False)
    return rv


def _threshold(src, lo, hi):
    t = Threshold(Input=src)  # noqa: F405
    t.Scalars = ["CELLS", "bcid"]
    _try(lambda: setattr(t, "LowerThreshold", lo))
    _try(lambda: setattr(t, "UpperThreshold", hi))
    _try(lambda: setattr(t, "ThresholdMethod", "Between"))
    return t


def _tube(path, radius):
    t = Tube(Input=LegacyVTKReader(FileNames=[path]))  # noqa: F405
    t.Radius = radius
    _try(lambda: setattr(t, "NumberofSides", 12))
    _try(lambda: setattr(t, "Capping", 1))
    return t


def bc_panel(path, support, arrows=None, tube_r=None):
    """The supports-and-loads panel.  Normals are taken on the WHOLE surface
    first, then the surface is split by region id, so body and overlay share
    their seam normals (split meshes each got their own normals and the seams
    sparkled, illustrations §5i).  The clay body keeps the house finish; the BC
    overlays (and the support markers) are lit flatter -- A.BC_OVERLAY_FINISH (ambient-heavy,
    no specular; design review 2, M3) -- so a region that faces
    away from the key light keeps its legend colour instead of going brown."""
    rv = new_view(A.AR_SURF_VIEW)
    normals = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[path]))  # noqa: F405
    _try(lambda: setattr(normals, "FeatureAngle", A.FEATURE_ANGLE))
    parts = [(_threshold(normals, -0.5, 0.5), dict(A.SURF_FINISH)),
             (_threshold(normals, 0.5, 1e6), dict(A.BC_OVERLAY_FINISH))]
    if support:      # the support outline: a thin flat teal tube (a marker, not a body)
        parts.append((_tube(support, tube_r), {"Ambient": 1.0, "Diffuse": 0.0, "Specular": 0.0}))
    if arrows:       # load arrows: lit as solids (A.ARROW_FINISH), occluded by the body
        nrm = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[arrows]))  # noqa: F405
        _try(lambda: setattr(nrm, "FeatureAngle", 60))
        parts.append((nrm, dict(A.ARROW_FINISH)))
    for src, finish in parts:
        d = Show(src, rv)  # noqa: F405
        d.Representation = "Surface"
        for k, v in finish.items():
            setattr(d, k, v)
        ColorBy(d, ("CELLS", "rgb"))  # noqa: F405
        d.MapScalars = 0
        d.SetScalarBarVisibility(rv, False)
    return rv


def bc_id_pass(path, support, arrows=None, tube_r=None):
    """The same geometry and camera, every face its region's id colour, unlit,
    no FXAA and no tone mapping: a pixel decodes to the region it shows."""
    rv = new_view(tuple((a, v) for a, v in A.AR_SURF_VIEW
                        if a not in ("UseFXAA", "UseToneMapping", "UseSSAO")))
    _try(lambda: setattr(rv, "UseFXAA", 0))
    _try(lambda: setattr(rv, "UseToneMapping", 0))
    srcs = [LegacyVTKReader(FileNames=[path])]  # noqa: F405
    if support:
        srcs.append(_tube(support, tube_r))
    if arrows:
        srcs.append(LegacyVTKReader(FileNames=[arrows]))  # noqa: F405
    for src in srcs:
        d = Show(src, rv)  # noqa: F405
        d.Representation = "Surface"
        d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
        ColorBy(d, ("CELLS", "idrgb"))  # noqa: F405
        d.MapScalars = 0
        d.SetScalarBarVisibility(rv, False)
    return rv


def solve_panel(path, scale, umax):
    rv = new_view(A.AR_SOLVE_VIEW)
    src = LegacyVTKReader(FileNames=[path])  # noqa: F405
    warp = WarpByVector(Input=src)  # noqa: F405
    warp.Vectors = ["POINTS", "displacement"]
    warp.ScaleFactor = float(scale)
    normals = GenerateSurfaceNormals(Input=warp)  # noqa: F405
    _try(lambda: setattr(normals, "FeatureAngle", A.FEATURE_ANGLE))
    d = Show(normals, rv)  # noqa: F405
    d.Representation = "Surface"
    ColorBy(d, ("POINTS", "displacement_magnitude"))  # noqa: F405
    lut = GetColorTransferFunction("displacement_magnitude")  # noqa: F405
    lut.ColorSpace = "Lab"
    lut.RGBPoints = A.solve_rgb_points(umax)
    d.Opacity = 1.0
    d.SetScalarBarVisibility(rv, False)
    for k, v in A.SOLVE_FINISH.items():
        setattr(d, k, v)
    return rv


GROUND = os.environ.get("AR_GROUND", "none")      # none | shadow | grid
SHADOW_GREY = [0.80, 0.80, 0.81]                  # the floor where only fill light reaches


def add_ground(rv, f, deformed):
    """The ground layer, in the same view and camera (ar_prep writes the geometry):
    'shadow' = the body projected along the key light onto the floor, flat grey,
    unlit (a shadow is the absence of the key light, not a lit surface);
    'grid' = the UI viewer's floor grid, its own RGBA per vertex (app.py
    placeGround), unlit, as three.js draws LineBasicMaterial."""
    if GROUND == "shadow":
        path = f("shadow_def.vtk") if deformed and os.path.exists(f("shadow_def.vtk")) \
            else f("shadow.vtk")
        d = Show(LegacyVTKReader(FileNames=[path]), rv)  # noqa: F405
        d.Representation = "Surface"
        d.DiffuseColor = SHADOW_GREY
        d.AmbientColor = SHADOW_GREY
        d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    elif GROUND == "grid":
        d = Show(LegacyVTKReader(FileNames=[f("grid.vtk")]), rv)  # noqa: F405
        d.Representation = "Surface"
        ColorBy(d, ("POINTS", "rgba"))  # noqa: F405
        d.MapScalars = 0
        d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
        d.LineWidth = 2.0          # ~1 CSS px of the UI's ~600 px viewer at this render size
        d.SetScalarBarVisibility(rv, False)


if __name__ == "__main__":
    _probe = CreateRenderView()  # noqa: F405
    Render(_probe)  # noqa: F405
    renderer = GetOpenGLInformation().GetRenderer()  # noqa: F405
    Delete(_probe)  # noqa: F405
    print("RENDERER:", renderer)
    assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env (illustrations/AGENTS.md §5)"

    stats = json.load(open(os.path.join(WORK, "ar_stats.json")))
    for rec in stats["runs"]:
        i = rec["index"]
        fit_path = os.path.join(WORK, f"ar_{i}_fit.vtk")
        if not os.path.exists(fit_path):
            print(f"[{i}] nothing to render: {rec['notes']}")
            continue
        view_dir = (rec.get("view") or {}).get("dir") or rec.get("view_dir") or A.VIEW_DIR
        ROW["draw_scale"] = float(rec.get("draw_scale") or 1.0)
        f = lambda name: os.path.join(WORK, f"ar_{i}_{name}")  # noqa: E731
        painted = (rec.get("painted") or {}).get("source")
        sup = f("support.vtk") if os.path.exists(f("support.vtk")) else None
        builders = {"painted": lambda: surface_panel(f("painted.vtk"), "POINTS",
                                                     specular=A.PAINT_SPECULAR if painted else None)}
        if os.path.exists(f("seg.vtk")):     # AR_SEGMENTATION=1: the parts, per facet
            builders["seg"] = lambda: surface_panel(f("seg.vtk"), "CELLS")
        if os.path.exists(f("bc.vtk")):
            arw = f("arrows.vtk") if os.path.exists(f("arrows.vtk")) else None
            tube_r = 0.006 * rec["extent"]          # ar_prep.MARKER_TUBE
            builders["bc"] = lambda: bc_panel(f("bc.vtk"), sup, arw, tube_r)
        if os.path.exists(f("materials.vtk")):
            builders["materials"] = lambda: surface_panel(f("materials.vtk"), "POINTS")
        if os.path.exists(f("solve.vtk")):
            # umax_colour: a row that shares one |u| range (../showcase); else the run's own
            builders["solve"] = lambda: solve_panel(f("solve.vtk"), rec["scale"]["factor"],
                                                    rec["solve"].get("umax_colour")
                                                    or rec["solve"]["umax_surface"])
        # a record may name the panels it needs (../showcase draws two per item);
        # agentic_results' records do not, and get all four
        if rec.get("render_panels"):
            builders = {k: v for k, v in builders.items() if k in rec["render_panels"]}
        for name, build in builders.items():
            save(build(), f(f"{name}.png"), name, fit_path, view_dir)
            if GROUND != "none":
                # the ground ALONE, same fitted camera: the compose lays it UNDER the
                # object.  Drawn with the object, an opaque shadow plane cut through
                # any part the magnified displacement pushes below the floor (a ring
                # across the clog's toe at x187,000); under it, the object always wins,
                # which is exact while the camera is above the floor.
                rv = new_view(A.AR_SURF_VIEW)
                save(rv, f(f"{name}_g.png"), name, fit_path, view_dir,
                     after_fit=lambda rv, d=(name == "solve"): add_ground(rv, f, deformed=d))
        if "bc" in builders:
            save(bc_id_pass(f("bc.vtk"), sup, arw, tube_r), f("bcid.png"), "bc", fit_path, view_dir,
                 transparent=False)
        if "materials" in builders and os.path.exists(f("segid.vtk")):
            # ../showcase only: the surface's segment ids, same camera, unlit -> the
            # compose draws the part seams where neighbouring pixels' ids differ
            save(bc_id_pass(f("segid.vtk"), None), f("segid.png"), "materials", fit_path, view_dir,
                 transparent=False)
        # far-face loads and their arrows, each in a pass of its OWN: drawn together,
        # a far wall hides the arrows that stand outside it
        for src_name, png in () if "bc" not in builders else (("bc_hidden.vtk", "bchidden.png"),
                              ("arrows_hidden.vtk", "bchidden_arrows.png"),
                              ("support_hidden.vtk", "bchidden_support.png")):
            hidden_srcs = [f(src_name)] if os.path.exists(f(src_name)) else []
            if not hidden_srcs:
                continue
            # loads on faces turned away from the camera, drawn ALONE (nothing in
            # front of them), unlit in their id colours: the compose lays them over
            # the BC panel as a see-through tint and measures each one
            rv = new_view(tuple((a, v) for a, v in A.AR_SURF_VIEW
                                if a not in ("UseFXAA", "UseToneMapping", "UseSSAO")))
            _try(lambda: setattr(rv, "UseFXAA", 0))
            _try(lambda: setattr(rv, "UseToneMapping", 0))
            for q in hidden_srcs:
                src = _tube(q, 0.006 * rec["extent"]) if q.endswith("support_hidden.vtk") \
                    else LegacyVTKReader(FileNames=[q])  # noqa: F405
                d = Show(src, rv)  # noqa: F405
                d.Representation = "Surface"
                d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
                ColorBy(d, ("CELLS", "idrgb"))  # noqa: F405
                d.MapScalars = 0
                d.SetScalarBarVisibility(rv, False)
            save(rv, f(png), "bc", fit_path, view_dir)
