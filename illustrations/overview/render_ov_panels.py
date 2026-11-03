"""Render step (pvbatch + EGL, GPU): every panel of method_overview.

    pvbatch render_ov_panels.py [panel ...]     # default: every panel whose inputs exist

Two kinds of panel, each drawn EXACTLY as the figure it comes from draws it
(``ov_style`` holds both recipes; the camera is the paper's (1.5, -3, 1.25)):

  surface panels  ``pfbunny/render_pf_bunny.py``: the display (mirrored) bunny,
                  its light kit with tone mapping off, its matte finish, fitted
                  to the display bunny with a 1.02 dolly — so every surface panel
                  shows the bunny at one size and place.
                  ov_p_tripo ov_p_cad ov_p_painted ov_p_seg_pf ov_p_bc
                  ov_p_texture ov_p_materials
  solve panel     not here: ``render_ov_solve.py`` — diffused_nitsche's panel
                  (f), unchanged — on ov_loop_solve.vtu, framed as these are,
                  onto the FEA tint (DN_SOLVE_BG)
  cube panels     ``nitsche/render_dn.py``, exactly: dn_prep's octree in
                  bunny.stl's own frame (seen from its back, as diffused_nitsche
                  shows it), cut at y = +0.009 keeping the far half, framed on the
                  cube (1.16).  ov_p_octree = its (a) · ov_p_phibc = its (e) with
                  the BC cells, and the ear / base facets translucent and
                  self-lit as (d) draws its body — no body, so phi shows beneath ·
                  ov_p_grad_{d,phi} = (a) over |dJ/dd|, |dJ/dphi|, each cut through
                  its own sensitivity band (ov_loop.json)

A panel whose inputs are missing is skipped (the CAD file is downloaded by hand).
"""

import os
import sys

from paraview.simple import *  # noqa: F403

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ov_style as S  # noqa: E402

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

SIZE = [S.PANEL_PX, S.PANEL_PX]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(kind):
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = S.VIEW_ANGLE
    for attr, val in (S.SURF_VIEW if kind == "surf" else S.CUBE_VIEW):
        _try(lambda a=attr, v=val: setattr(rv, a, v))
    return rv


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv, reference, dolly):
    """ResetCamera on a hidden reference, then dolly back along the view."""
    Show(reference, rv)  # noqa: F405
    d = _unit(S.VIEW_DIR)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [3.0 * c for c in d]
    rv.CameraViewUp = S.VIEW_UP
    rv.CameraParallelProjection = 0
    Render(rv)  # noqa: F405
    ResetCamera(rv)  # noqa: F405
    fp, pos = list(rv.CameraFocalPoint), list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * dolly for i in range(3)]
    Hide(reference, rv)  # noqa: F405


def save(rv, name, kind):
    if kind == "surf":       # render_pf_bunny.fit, on the one display bunny
        fit(rv, OpenDataFile("assets/bunny_display.stl"), S.DOLLY_SURF)  # noqa: F405
    else:                    # render_dn.fit_all, on the full cube
        fit(rv, Outline(Input=LegacyVTKReader(FileNames=["dn_full.vtk"])),  # noqa: F405
            S.DOLLY_CUBE)
    # opaque, on the tint of the workflow part this panel belongs to
    rv.Background = S.fill_rgb(S.PANEL_PART[name])
    Render(rv)  # noqa: F405
    SaveScreenshot(f"ov_p_{name}.png", rv, ImageResolution=SIZE)  # noqa: F405
    print(f"wrote ov_p_{name}.png")
    Delete(rv)  # noqa: F405


def exists(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  skip: missing {missing}")
    return not missing


# --------------------------------------------------------------------------
# surface panels: render_pf_bunny's surface_actor
# --------------------------------------------------------------------------
def surface(rv, path, colour=None, rgb=None):
    normals = GenerateSurfaceNormals(Input=OpenDataFile(path))  # noqa: F405
    _try(lambda: setattr(normals, "FeatureAngle", 50))
    d = Show(normals, rv)  # noqa: F405
    d.Representation = "Surface"
    for k, v in S.SURF_FINISH.items():
        setattr(d, k, v)
    if rgb:
        ColorBy(d, (rgb, "rgb"))  # noqa: F405
        d.MapScalars = 0
        d.SetScalarBarVisibility(rv, False)
    else:
        d.DiffuseColor = colour
    return d


def lines(rv, path, colour=S.EDGE, width=2.0):
    d = Show(OpenDataFile(path), rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = colour
    d.AmbientColor = colour
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    d.LineWidth = width
    _try(lambda: setattr(d, "RenderLinesAsTubes", 1))
    return d


def p_plain(name, path, colour):
    if exists(path):
        rv = new_view("surf")
        surface(rv, path, colour=colour)
        save(rv, name, "surf")


def p_rgb(name, path, assoc, seams=None, specular=None):
    if exists(path, *([seams] if seams else [])):
        rv = new_view("surf")
        d = surface(rv, path, rgb=assoc)
        if specular is not None:
            d.Specular = specular
        if seams:
            lines(rv, seams)
        save(rv, name, "surf")


def p_bc():
    """The TripoSG panel's clay bunny (one surface, so one set of normals),
    the ears and the base in their BC colours — bcautobunny (a)'s recipe."""
    p_rgb("bc", "ov_s_bc.vtk", "CELLS")


# --------------------------------------------------------------------------
# cube panels: render_dn's cut / mesh_actor / surface_actor / outline
# --------------------------------------------------------------------------
def cube(path):
    """Cube data as render_dn reads it: bunny.stl's own frame, not mirrored."""
    reader = OpenDataFile(path)  # noqa: F405
    reader.UpdatePipeline()
    return reader


def cut(src, origin_y=S.CUT_Y, keep_far=True):
    c = Clip(Input=src)  # noqa: F405
    c.ClipType = "Plane"
    c.ClipType.Origin = [0.0, origin_y, 0.0]
    c.ClipType.Normal = S.CUT_N
    # camera on -y: the far half is y >= origin, the side Invert=0 keeps
    c.Invert = 0 if keep_far else 1
    return c


def mesh_actor(rv, src, array, rgb_points, below=None, origin_y=S.CUT_Y):
    d = Show(cut(src, origin_y=origin_y), rv)  # noqa: F405
    d.Representation = "Surface With Edges"
    ColorBy(d, ("POINTS", array))  # noqa: F405
    lut = GetColorTransferFunction(array)  # noqa: F405
    lut.ColorSpace = "Lab"
    lut.RGBPoints = rgb_points
    lut.NanColor = [0.6, 0.6, 0.6]
    if below is not None:
        lut.UseBelowRangeColor = 1
        lut.BelowRangeColor = below
    d.Ambient, d.Diffuse, d.Specular = 0.28, 0.85, 0.0
    _try(lambda: setattr(d, "EdgeColor", S.EDGE))
    _try(lambda: setattr(d, "LineWidth", 0.9))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    return d


def outline(rv):
    d = Show(Outline(Input=cube("dn_full.vtk")), rv)  # noqa: F405
    d.DiffuseColor = [0.30, 0.32, 0.35]
    d.LineWidth = 2.4


def dn_surface(rv, path, colour, opacity=0.72, ambient=0.45):
    """render_dn.surface_actor as panel (d) draws its translucent body: near half
    only (a translucent closed shell would show its own far wall), Gouraud,
    self-lit in its own colour (AmbientColor) so it lifts off the section.
    Denser and brighter than (d)'s body (0.42 / 0.24): at overview size the BC
    sets did not read."""
    nrm = GenerateSurfaceNormals(Input=STLReader(FileNames=[path]))  # noqa: F405
    _try(lambda: setattr(nrm, "FeatureAngle", 50))
    d = Show(cut(nrm, origin_y=0.0, keep_far=False), rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = colour
    d.Opacity = opacity
    d.Specular = 0.35
    d.SpecularPower = 30
    d.Ambient = ambient
    _try(lambda: setattr(d, "AmbientColor", colour))
    d.Diffuse = 1.0
    return d


def p_octree():
    if exists("dn_full.vtk"):
        rv = new_view("cube")
        mesh_actor(rv, cube("dn_full.vtk"), "sdf", S.SDF_RGB)
        outline(rv)
        save(rv, "octree", "cube")


def p_phibc():
    """render_dn (e) — the trimmed mesh over phi — carrying the BC sets: the BC
    cells in their colours (a partition of the one grid, never an overlay) and
    the ear / base facets translucent and self-lit, as (d) draws its body.  No
    body of its own: phi shows beneath."""
    if not exists("ov_trim_bc.vtk", "ov_bc_ears.stl", "ov_bc_base.stl"):
        return
    rv = new_view("cube")
    trim = cube("ov_trim_bc.vtk")

    def tags(lo, hi):
        t = Threshold(Input=trim)  # noqa: F405
        t.Scalars = ["CELLS", "tag"]
        _try(lambda: setattr(t, "LowerThreshold", lo))
        _try(lambda: setattr(t, "UpperThreshold", hi))
        _try(lambda: setattr(t, "ThresholdMethod", "Between"))
        return t

    mesh_actor(rv, tags(-0.5, 1.5), "phi", S.PHI_RGB)
    d = Show(cut(tags(1.5, 3.5)), rv)  # noqa: F405
    d.Representation = "Surface With Edges"
    ColorBy(d, ("CELLS", "tag"))  # noqa: F405
    tl = GetColorTransferFunction("tag")  # noqa: F405
    tl.ColorSpace = "RGB"
    tl.RGBPoints = [2.0, *S.NEUMANN, 2.5, *S.NEUMANN, 2.51, *S.DIRICHLET,
                    3.0, *S.DIRICHLET]
    d.Ambient, d.Diffuse, d.Specular = 0.22, 0.90, 0.0
    _try(lambda: setattr(d, "EdgeColor", S.EDGE))
    _try(lambda: setattr(d, "LineWidth", 0.9))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    dn_surface(rv, "ov_bc_ears.stl", S.NEUMANN)
    dn_surface(rv, "ov_bc_base.stl", S.DIRICHLET)
    outline(rv)
    save(rv, "phibc", "cube")


def p_grad(name, array):
    if not exists("ov_grad_full.vtk", "ov_loop.json"):
        return
    import json
    loop = json.load(open("ov_loop.json"))
    rv = new_view("cube")
    # its own cut plane, through the band this sensitivity lives in (the
    # camera is unchanged); flatter light than the d panel, so a near-white
    # field stays white and the band reads
    d = mesh_actor(rv, cube("ov_grad_full.vtk"), array, S.GRAD_RGB,
                   below=S.MASKED_RGB, origin_y=float(loop[f"cut_y_{array}"]))
    d.Ambient, d.Diffuse = 0.80, 0.35
    _try(lambda: setattr(rv, "UseToneMapping", 0))
    outline(rv)
    save(rv, name, "cube")


PANELS = {
    "tripo": lambda: p_plain("tripo", "assets/bunny_display.stl", S.CLAY),
    "cad": lambda: p_rgb("cad", "ov_s_cad.vtk", "CELLS", "ov_s_cad_edges.vtk"),
    # fur is matte: render_pf_bunny's 0.35 specular made the paint look plastic
    "painted": lambda: p_rgb("painted", "ov_s_painted.vtk", "POINTS", specular=0.08),
    # the same painted surface, as the (pre-processing) texture step
    "texture": lambda: p_rgb("texture", "ov_s_painted.vtk", "POINTS", specular=0.08),
    "seg_pf": lambda: p_rgb("seg_pf", "ov_s_seg_pf.vtk", "CELLS"),
    "bc": p_bc,
    "materials": lambda: p_rgb("materials", "ov_s_materials.vtk", "CELLS"),
    "octree": p_octree,
    "phibc": p_phibc,
    "grad_d": lambda: p_grad("grad_d", "gd"),
    "grad_phi": lambda: p_grad("grad_phi", "gphi"),
}

if __name__ == "__main__":
    _probe = CreateRenderView()  # noqa: F405
    Render(_probe)  # noqa: F405
    renderer = GetOpenGLInformation().GetRenderer()  # noqa: F405
    Delete(_probe)  # noqa: F405
    print("RENDERER:", renderer)
    assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env"
    for name in sys.argv[1:] or list(PANELS):
        print(f"== {name}")
        PANELS[name]()
