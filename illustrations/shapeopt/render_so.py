"""Render the shape-opt loop panels (pvbatch + EGL), in the SAME style as
diffused_nitsche: the octree cut by a centered coordinate plane (y=0, through two
on-axis feet), cell frames + cube outline, one shared camera fit to the full cube.
Writes panel_so_*.png.

Panels (single row):  a design SDF  ->  b solve |u|  ->  d sensitivity  ->
e SDF update  ->  g marching cubes & restore-BC (surface, BC coloured by type).
"""
from paraview.simple import *  # noqa: F403
paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(globals().get("__file__", "render_so.py")))
_sys.path.insert(0, _os.path.join(_HERE, "..", "agentic_results"))
import ar_style as _AR  # noqa: E402  the load-arrow colour + finish
_sys.path.insert(0, _os.path.join(_HERE, "..", "_shared"))
import evo_style as _ES  # noqa: E402  shape_opt_merged b's frames (ghost opacity, voids)

_HOUSE = [0.106, 0.196, 0.478, 0.235, 0.443, 0.718, 0.557, 0.706, 0.851,
          0.894, 0.894, 0.894, 0.968, 0.805, 0.576, 0.944, 0.600, 0.282,
          0.827, 0.329, 0.152, 0.478, 0.129, 0.094]
BODY = [0.920, 0.900, 0.860]      # == diffused_nitsche
TEAL = [0.071, 0.647, 0.580]      # Dirichlet (feet)
ORANGE = [0.867, 0.518, 0.322]    # Neumann   (top load)

# centered cut plane (y=0), nudged a hair off the cell boundary so the cross
# section is clean quads (== diffused_nitsche's CUT_Z gotcha); keep the far half
# (camera is on -y) so the cut face faces the camera.
CUT_N = [0.0, 1.0, 0.0]
CUT_O = [0.0, 0.009, 0.0]
VIEW_DIR = [0.42, -1.1, 0.40]     # moderate oblique, SHARED with hidden_phi_loop's
#                                   octree panels (render_hp.CUBE_VIEW_DIR) so the
#                                   two figures' cube panels sit at the same angle
CAM_UP = [0.0, 0.0, 1.0]

# --- evolution strip (panel f): rendered like hidden_phi_loop's (f) ----------
# A sectioned SOLID, not a flat off-white shell: a DARKER cut face (the
# engineering convention that reads as filled material), PBR + SSAO shading, and
# ONE fixed near-side elevation camera for every frame so the strip does not jump
# in scale as the table forms.  Constants copied from render_hp.py; the design
# blank is so_prep's verbatim, so its bounds fit hidden_phi's frame box.
CUT_FACE = [0.850, 0.828, 0.795]  # section face, only a HAIR darker than the body
EVO_VIEW_DIR = [0.30, -0.90, 0.30]
EVO_VIEW_UP = [0.0, 0.0, 1.0]
EVO_BOX_U = 1.14                  # fixed frame half-width (x/y); geometry is +-1.0
EVO_BOX_Z = (-0.80, 0.62)         # fixed frame z span; geometry is [-0.72, 0.50]
EVO_DOLLY = 0.88                  # zoom IN so the sectioned table fills more of the panel


# Line weights in RENDER pixels, sized for print: the ink crop of a 1500 px octree
# panel is ~1340 px and is printed 30.8 mm wide (loop_chain's 5-column grid), so
# 1 px = 0.023 mm and the reviewers' 0.3 pt (0.106 mm) floor for cell lines is ~4.6 px.
LW_EDGE = 4.6                     # octree cell frames (0.3 pt in print)
EDGE_RGB = [0.50, 0.51, 0.54]     # mid grey: at 0.3 pt a black frame fills the finest
#                                   cells (0.3 mm in print) and hides the field
LW_FRAME = 4.6                    # the octree-domain cube outline
LW_ISO = 7.0                      # d = 0
# |u|: the paper's displacement ramp (agentic_results / showcase, ar_style.SOLVE_*):
# body cream up to 0.35 max, Neumann orange at 0.70, brick at max.
SOLVE_BODY, SOLVE_MID, SOLVE_TOP = [0.920, 0.900, 0.860], [0.867, 0.518, 0.322], \
    [0.478, 0.129, 0.094]


def _solve_rgbpoints(umax):
    return [0.0, *SOLVE_BODY, 0.35 * umax, *SOLVE_BODY, 0.70 * umax, *SOLVE_MID,
            umax, *SOLVE_TOP]


def _diverging_rgbpoints(vmin, vmax):
    stops = [0.0, 0.18, 0.36, 0.5, 0.58, 0.72, 0.86, 1.0]
    pts = []
    for s, i in zip(stops, range(8)):
        val = vmin + s * (vmax - vmin)
        pts += [val, _HOUSE[3 * i], _HOUSE[3 * i + 1], _HOUSE[3 * i + 2]]
    return pts


# plasma (matplotlib) for the |u| magnitude, PRGn (colorbrewer) for the signed
# sensitivity -- so (b) and (d) are each visibly distinct from the SDF panels.
_PLASMA = [(0.0, 0.050, 0.030, 0.528), (0.1, 0.255, 0.014, 0.615),
           (0.2, 0.418, 0.001, 0.658), (0.3, 0.563, 0.052, 0.642),
           (0.4, 0.691, 0.165, 0.564), (0.5, 0.798, 0.280, 0.470),
           (0.6, 0.881, 0.393, 0.383), (0.7, 0.949, 0.518, 0.296),
           (0.8, 0.988, 0.652, 0.211), (0.9, 0.989, 0.810, 0.145),
           (1.0, 0.940, 0.975, 0.131)]
_PRGN = [(0.0, 0.251, 0.0, 0.294), (0.1, 0.463, 0.165, 0.514),
         (0.2, 0.600, 0.439, 0.671), (0.3, 0.761, 0.647, 0.812),
         (0.4, 0.906, 0.831, 0.910), (0.5, 0.969, 0.969, 0.969),
         (0.6, 0.851, 0.941, 0.827), (0.7, 0.651, 0.859, 0.627),
         (0.8, 0.353, 0.682, 0.380), (0.9, 0.106, 0.471, 0.216),
         (1.0, 0.0, 0.267, 0.106)]


def _stops_rgbpoints(vmin, vmax, stops):
    return [c for s, r, g, b in stops for c in (vmin + s * (vmax - vmin), r, g, b)]


def _set(obj, **kw):
    for k, v in kw.items():
        try:
            setattr(obj, k, v)
        except Exception:
            pass


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


# Background of every render.  White for the standalone loop figure; the merged
# figure (shape_opt_merged) renders each panel OPAQUE onto its block's tint, as
# Figure 1 does (a white render composited onto a tint keeps a light halo).
BG = [1.0, 1.0, 1.0]
# The merged figure's stations were first drawn on Figure 1's part tints; the user
# rejected the tinted fills (2026-10-13), so every station renders on WHITE again.
PRE_TINT = [1.0, 1.0, 1.0]
FEA_TINT = [1.0, 1.0, 1.0]


def new_view():
    rv = CreateView("RenderView")
    rv.ViewSize = [1500, 1400]
    rv.Background = list(BG)
    rv.OrientationAxesVisibility = 0
    _set(rv, UseColorPaletteForBackground=0, UseFXAA=1, UseToneMapping=1,
         KeyLightAzimuth=-35, KeyLightElevation=45, KeyLightIntensity=0.9,
         KeyLightWarmth=0.55, FillLightKFRatio=2.6, BackLightKBRatio=3.0,
         HeadLightKHRatio=3.0)
    return rv


def set_camera(rv, dolly=1.05):
    d = _unit(VIEW_DIR)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [c * 3.0 for c in d]
    rv.CameraViewUp = CAM_UP
    ResetCamera(rv)                    # fit every actor -> full cube frame shown
    fp = list(rv.CameraFocalPoint); pos = list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * dolly for i in range(3)]
    rv.CameraViewAngle = 25
    Render(rv)


def save(rv, name):
    print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
    if BG == [1.0, 1.0, 1.0]:
        SaveScreenshot(f"panel_so_{name}.png", rv, ImageResolution=[1500, 1400],
                       OverrideColorPalette="WhiteBackground", TransparentBackground=0)
    else:
        SaveScreenshot(f"panel_so_{name}.png", rv, ImageResolution=[1500, 1400],
                       TransparentBackground=0)
    print("wrote panel_so_%s.png" % name)
    Delete(rv)


def _cut(src):
    c = Clip(Input=src)
    c.ClipType = "Plane"
    c.ClipType.Origin = CUT_O
    c.ClipType.Normal = CUT_N
    c.Invert = 0                       # keep +y (far) half; cut face faces camera
    return c


def outline(node, rv):
    o = Show(Outline(Input=node), rv)
    o.DiffuseColor = [0.30, 0.32, 0.35]     # == diffused_nitsche
    o.LineWidth = LW_FRAME


def panel_octree_field(vtk, field, name, vmin, vmax, cmap="sdf", frame_vtk=None,
                       cut=True, prev_vtk=None, edges=True):
    """Octree profile, diffused_nitsche style: coordinate-plane cut coloured by
    `field`, cell frames, and the cube outline (of frame_vtk, default the octree).
    cut=False shows the mesh in full (used for the trimmed solve mesh in panel b).
    cmap: 'sdf' (house diverging), 'prgn' (sensitivity), 'plasma' (displacement)."""
    rv = new_view()
    src = OpenDataFile(vtk); src.UpdatePipeline()
    fr = src if frame_vtk is None else OpenDataFile(frame_vtk)
    outline(fr, rv)
    node = _cut(src) if cut else src
    d = Show(node, rv)
    # The uncut solve mesh is seen at grazing angles where print-weight frames
    # merge into a grey lid; its voxel staircase reads from the shading alone.
    d.Representation = "Surface With Edges" if edges else "Surface"
    _set(d, EdgeColor=EDGE_RGB, LineWidth=LW_EDGE, RenderLinesAsTubes=0)
    d.Ambient = 0.28; d.Diffuse = 0.85; d.Specular = 0.0
    ColorBy(d, ("POINTS", field))
    if cut and field == "sdf":
        _iso_d0(rv, node)
    if prev_vtk:                      # the previous interface, dashed: the step itself
        _iso_d0(rv, _cut(OpenDataFile(prev_vtk)), dashed=True, colour=PREV_RGB, width=5.0)
    lut = GetColorTransferFunction(field)
    if cmap == "sdf":
        lut.ColorSpace = "RGB"; lut.RGBPoints = _diverging_rgbpoints(vmin, vmax)
    elif cmap == "prgn":
        lut.ColorSpace = "Lab"; lut.RGBPoints = _stops_rgbpoints(vmin, vmax, _PRGN)
    elif cmap == "solve":
        lut.ColorSpace = "RGB"; lut.RGBPoints = _solve_rgbpoints(vmax)
    else:  # plasma
        lut.ColorSpace = "RGB"; lut.RGBPoints = _stops_rgbpoints(vmin, vmax, _PLASMA)
    lut.NanColor = [0.6, 0.6, 0.6]
    d.SetScalarBarVisibility(rv, False)
    set_camera(rv)
    save(rv, name)


INK = [0.10, 0.10, 0.13]
# phi in [0,1]: the sequential blue every phase-field panel in the paper uses
# (render_dn / ov_style / render_hp PHI_RGB).
_PHI = [(0.0, 0.804, 0.886, 0.984), (0.1667, 0.620, 0.773, 0.957),
        (0.3333, 0.427, 0.655, 0.925), (0.5, 0.224, 0.529, 0.898),
        (0.6667, 0.145, 0.416, 0.749), (0.8333, 0.094, 0.310, 0.584),
        (1.0, 0.051, 0.212, 0.420)]
# |dJ/d.| / max over GRAD_DECADES decades: Figure 1's white -> magenta ramp
# (ov_style.GRAD_RGB), so the backward quantities share one colour with the
# backward arrows across the paper.  The masked BC region and the floor are the
# ramp's own near-white.
_GRAD_MAG = [(0.00, 0.975, 0.975, 0.975), (0.15, 0.992, 0.878, 0.937),
             (0.40, 0.945, 0.714, 0.855), (0.70, 0.871, 0.467, 0.682),
             (1.00, 0.773, 0.106, 0.490)]
GRAD_DECADES = 4.0
EPS = 0.015                       # so_prep: phase_field_epsilon = 0.5 * h_target


PREV_RGB = [0.62, 0.63, 0.66]    # the previous d = 0: light dashes, drawn OVER the new
#                                   curve, so where the step is sub-cell the two read as
#                                   a black/grey dashed line and where it moved, apart


LIFT = [0.00675, -0.01769, 0.00643]   # 0.02 along the view direction: no screen shift
EDGE_LIGHT = [0.76, 0.77, 0.79]       # gradient panels: frames at ~50 % so the thin band reads
LW_CASED, LW_CASING = 5.4, 9.5        # d = 0 on gradient panels: 0.35 pt ink in a white casing


def _iso_d0(rv, src, array="sdf", width=LW_ISO, dashed=False, colour=None, lift=0.0):
    """The d = 0 interface as an ink curve on the cut face (render_hp.iso): contour
    the extracted SURFACE so it returns curves, not an isosurface in front of it."""
    ctr = Contour(Input=ExtractSurface(Input=src))
    ctr.ContourBy = ["POINTS", array]
    ctr.Isosurfaces = [0.0]
    if dashed:
        # VTK lines cannot be dashed, so cut the curve into dashes: keep the parts
        # where a sine of (x + z) is positive (period 0.16 = ~1.7 mm in print).
        cal = Calculator(Input=ctr)
        cal.ResultArrayName = "dash"
        cal.Function = "sin(40*(coordsX+coordsZ))"
        ctr = Clip(Input=cal)
        ctr.ClipType = "Scalar"
        ctr.Scalars = ["POINTS", "dash"]
        ctr.Value = 0.0
        _set(ctr, Invert=0)
        # lift the dashes a hair toward the camera (along the view direction) so they draw over
        # the new d = 0 curve instead of z-fighting with it where the two coincide
        lift = lift or 1.0
    if lift:
        ctr = Transform(Input=ctr)  # noqa: F405
        ctr.Transform = "Transform"
        ctr.Transform.Translate = [lift * c for c in LIFT]
    d = Show(ctr, rv)
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", ""]
    c = colour if colour is not None else INK
    d.DiffuseColor = c; d.AmbientColor = c
    d.Ambient = 1.0; d.Diffuse = 0.0
    d.LineWidth = width
    _set(d, RenderLinesAsTubes=1)
    d.SetScalarBarVisibility(rv, False)


def _mesh(rv, node, field, rgbpoints, space="RGB", below=None, edge_rgb=None):
    d = Show(node, rv)
    d.Representation = "Surface With Edges"
    _set(d, EdgeColor=EDGE_RGB if edge_rgb is None else edge_rgb, LineWidth=LW_EDGE,
         RenderLinesAsTubes=0)
    d.Ambient = 0.28; d.Diffuse = 0.85; d.Specular = 0.0
    ColorBy(d, ("POINTS", field))
    lut = GetColorTransferFunction(field)
    lut.ColorSpace = space; lut.RGBPoints = rgbpoints
    if below is not None:
        _set(lut, UseBelowRangeColor=1, BelowRangeColor=below)
    d.SetScalarBarVisibility(rv, False)
    return d


def panel_phi(design_vtk, solve_vtk, name):
    """The phase field the solve integrates: phi = (1 - tanh(d/eps))/2 sampled on the
    TRIMMED solve mesh (Figure 1's 'Trimmed mesh + phi'), cut like every octree
    panel and framed by the same octree cube.  The solve mesh's nodes are a subset
    of the design octree's, so resampling d onto them is exact."""
    rv = new_view()
    des = OpenDataFile(design_vtk); des.UpdatePipeline()
    sol = OpenDataFile(solve_vtk); sol.UpdatePipeline()
    outline(des, rv)
    rs = ResampleWithDataset(SourceDataArrays=des, DestinationMesh=sol)
    calc = Calculator(Input=rs)
    calc.ResultArrayName = "phi"
    calc.Function = f"1/(1+exp(2*sdf/{EPS}))"       # == (1 - tanh(d/eps)) / 2
    node = _cut(calc)
    # Cell frames on the CUT FACE only (a Slice a hair in front of the clip plane):
    # the far half's top is seen at a grazing angle where print-weight frames
    # merge into a grey lid.
    _mesh(rv, node, "phi", _stops_rgbpoints(0.0, 1.0, _PHI)).Representation = "Surface"
    sl = Slice(Input=calc)
    sl.SliceType = "Plane"
    sl.SliceType.Origin = [0.0, CUT_O[1] - 0.003, 0.0]
    sl.SliceType.Normal = [0.0, 1.0, 0.0]
    _set(sl, Triangulatetheslice=0)
    _mesh(rv, sl, "phi", _stops_rgbpoints(0.0, 1.0, _PHI))
    _iso_d0(rv, node)
    set_camera(rv)
    save(rv, name)


def panel_grad_mag(vtk, name, design_vtk=None, field="grad", frame_vtk=None):
    """|dJ/dd| / p99 on a log axis over GRAD_DECADES decades, clipped at 1: the
    scale's top is the 99th percentile of the NON-ZERO |dJ/dd| (BC nodes were
    written as 0 by so_prep), not the maximum, so a few extreme nodes no longer
    wash the band out.  Figure 1's magenta; frames drawn light so the thin band
    reads; d = 0 at 0.35 pt in a white casing so the band shows on both sides."""
    import numpy as np
    from paraview import servermanager
    from vtk.numpy_interface import dataset_adapter as dsa
    rv = new_view()
    src = OpenDataFile(vtk); src.UpdatePipeline()
    g = np.abs(np.asarray(dsa.WrapDataObject(servermanager.Fetch(src)).PointData[field]))
    p99 = float(np.percentile(g[g > 0], 99))
    print(f"  |dJ/dd|: max {g.max():.4g}, p99 of non-zero {p99:.4g} (x{g.max() / p99:.2f})")
    outline(src if frame_vtk is None else OpenDataFile(frame_vtk), rv)
    calc = Calculator(Input=src)
    calc.ResultArrayName = "glog"
    calc.Function = (f"max(min(log10(abs({field})/{p99:.10g}+1e-30)/{GRAD_DECADES}+1,1),0)")
    node = _cut(calc)
    # the field WITHOUT frames, the frames over it as a 40 % wireframe: at 0.3 pt
    # opaque frames cover the finest cells, which is exactly where the band is
    _mesh(rv, node, "glog", _stops_rgbpoints(0.0, 1.0, _GRAD_MAG)).Representation = "Surface"
    wf = Show(ExtractSurface(Input=node), rv)   # own proxy; surface edges only
    wf.Representation = "Wireframe"
    wf.ColorArrayName = ["POINTS", ""]
    wf.AmbientColor = wf.DiffuseColor = EDGE_RGB
    wf.LineWidth = LW_EDGE
    wf.Opacity = 0.4
    if design_vtk:                    # the current interface, cased
        des = _cut(OpenDataFile(design_vtk))
        _iso_d0(rv, des, width=LW_CASING, colour=[1.0, 1.0, 1.0], lift=0.5)
        _iso_d0(rv, des, width=LW_CASED, lift=1.0)
    set_camera(rv)
    save(rv, name)


def _bunny_surface(src, rv, color):
    n = GenerateSurfaceNormals(Input=src)
    _set(n, FeatureAngle=45)
    d = Show(n, rv)
    d.Representation = "Surface"
    d.DiffuseColor = list(color); d.AmbientColor = list(color)
    d.Ambient = 0.14; d.Diffuse = 1.0; d.Specular = 0.30; d.SpecularPower = 30
    return d


def load_arrows(rv, path):
    """The load arrows (../_shared/table_load_arrows.py) in agentic_results' arrow
    colour and finish (ar_style.LOAD_ARROW, ARROW_FINISH): the Neumann hue darkened
    so a head standing on the orange top keeps its outline, lit as a solid so shaft
    and cone read round."""
    n = GenerateSurfaceNormals(Input=OpenDataFile(path))
    _set(n, FeatureAngle=60)
    d = Show(n, rv)
    d.Representation = "Surface"
    d.DiffuseColor = list(_AR.LOAD_ARROW); d.AmbientColor = list(_AR.LOAD_ARROW)
    for k, v in _AR.ARROW_FINISH.items():
        setattr(d, k, v)
    d.SetScalarBarVisibility(rv, False)
    return d


def panel_restore_bc(name):
    rv = new_view()
    _bunny_surface(OpenDataFile("so_body.stl"), rv, BODY)
    _bunny_surface(OpenDataFile("so_bc_dir.stl"), rv, TEAL)      # Dirichlet feet
    _bunny_surface(OpenDataFile("so_bc_neu.stl"), rv, ORANGE)    # Neumann top
    load_arrows(rv, "so_load_arrows.stl")                         # the top's traction
    set_camera(rv)
    save(rv, name)


GHOST_OPACITY = 0.34             # 0.20 printed ~10 % grey: frame 0 nearly vanished
GHOST = [0.80, 0.785, 0.76]      # a step darker than BODY (the 10 % darkening was reverted)


def _evo_view():
    """new_view() (bright: no SSAO, key light 0.9 — same as panel (g)/(a-e)), just
    the strip's view angle.  SSAO was greying the sectioned body below the (g)
    panel's white; the darker CUT_FACE alone carries the 'sectioned solid' cue."""
    rv = new_view()
    rv.CameraViewAngle = 25
    # enough depth-peeling layers for the translucent ghost: with the default
    # four, the darker ghost printed diagonal hatching where its layers overlap
    _set(rv, UseDepthPeeling=1, MaximumNumberOfPeels=24)
    return rv


def _evo_body(rv, path, color, opacity=1.0, pbr=False, open_cut=False):
    """render_hp.body_actor.  pbr=False (plain, like panel (g)) keeps the off-white
    body BRIGHT; PBR without an environment map reads duller/greyer.  open_cut drops
    the ghost's own cut cap (it lies ON y = 0, coincident with the far half's cut face,
    and the two z-fight into diagonal hatching -- §5g's open_ghost, done here by
    keeping only y < -0.003)."""
    src = OpenDataFile(path)
    if open_cut:
        c = Clip(Input=src)
        c.ClipType = "Plane"
        c.ClipType.Origin = [0.0, -0.003, 0.0]
        c.ClipType.Normal = [0.0, 1.0, 0.0]
        c.Invert = 1                           # keep the side the normal points away from
        src = ExtractSurface(Input=c)
    n = GenerateSurfaceNormals(Input=src); _set(n, FeatureAngle=50)
    d = Show(n, rv); d.Representation = "Surface"
    d.DiffuseColor = list(color); d.AmbientColor = list(color)
    d.Opacity = opacity; d.Specular = 0.35 if pbr else 0.30; d.SpecularPower = 30
    if pbr:
        d.Ambient = 0.12; d.Diffuse = 1.0
        _set(d, Interpolation="PBR", Roughness=0.45, Metallic=0.0)
    else:
        d.Ambient = 0.14; d.Diffuse = 1.0
    d.SetScalarBarVisibility(rv, False)
    return d


def _evo_face(rv, path):
    """render_hp.face_actor: the flat cut face as an UNLIT fill a step DARKER than
    the body — the cue that says 'filled material', and unlit so the cap's sliver
    normals don't print as diagonal scratches."""
    d = Show(OpenDataFile(path), rv); d.Representation = "Surface"
    d.DiffuseColor = list(CUT_FACE); d.AmbientColor = list(CUT_FACE)
    d.Ambient = 1.0; d.Diffuse = 0.0; d.Specular = 0.0
    d.SetScalarBarVisibility(rv, False)
    return d


def _evo_camera(rv):
    """render_hp.fit: ONE camera sized off the FIXED frame box (not the data), so
    frames whose table is a little larger do not jump."""
    d = _unit(EVO_VIEW_DIR)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [c * 3.0 for c in d]
    rv.CameraViewUp = EVO_VIEW_UP
    ResetCamera(rv)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    r = 0.5 * ((2.0 * EVO_BOX_U) ** 2 + (EVO_BOX_Z[1] - EVO_BOX_Z[0]) ** 2) ** 0.5
    rv.CameraPosition = [d[i] * 4.51 * r * EVO_DOLLY for i in range(3)]


def panel_evo(shell_path, name):
    """One evolution-strip frame, rendered like hidden_phi_loop's (f): the design
    half-cut on y=0 into a BRIGHT capped solid + a DARKER sectioned cut face + a
    translucent near half, from the fixed elevation camera zoomed to fill more."""
    rv = _evo_view()
    base = shell_path[:-10]                                     # strip "_shell.stl"
    _evo_body(rv, shell_path, BODY, pbr=False)                 # capped far-half solid
    _evo_face(rv, base + "_face.stl")                          # darker section face
    _evo_body(rv, base + "_ghost.stl", GHOST, opacity=GHOST_OPACITY, pbr=False,
              open_cut=True)
    Render(rv)                                                  # 1st render resets cam
    _evo_camera(rv)                                            # so set it afterwards
    save(rv, name)


def panel_m_evo(it):
    """shape_opt_merged b's frame (round 11): panel_evo with evo_style's fainter ghost and
    the voids darkened by openness (so_evo_<it>_{shell,ghost}_open.vtk, void_open.py)."""
    rv = _evo_view()
    base = f"so_evo_{it}"
    _ES.open_actor(rv, base + "_shell_open.vtk", BODY, name="open_body")
    _evo_face(rv, base + "_face.stl")
    _ES.open_actor(rv, base + "_ghost_open.vtk", GHOST, opacity=_ES.M_GHOST_OPACITY,
                   clip_y=-0.003, name="open_ghost")
    Render(rv)
    _evo_camera(rv)
    save(rv, f"m_evo_{it}")


# ---- render every panel -----------------------------------------------------
#   pvbatch render_so.py            # all panels (a,b,d,e,g + the evo strip)
#   pvbatch render_so.py evo        # just the evolution strip (panel f)
#   pvbatch render_so.py a e        # just those loop panels
import sys, glob, re

SDF_MIN, SDF_MAX = -0.30, 0.30
GRAD_MIN, GRAD_MAX = -0.001, 0.001
_ST = dict(l.strip().split("=", 1) for l in open("so_stats.txt") if "=" in l) \
    if __name__ == "__main__" else {}
UMAG_MIN, UMAG_MAX = 0.0, float(_ST.get("umag_max", 0.0125))

# Guarded so the panel functions above can be imported (by render_so_frames.py)
# without this script rendering the whole loop; pvbatch runs it as __main__.
if __name__ == "__main__":
    SEL = set(sys.argv[1:])
    def _sel(tag): return (not SEL) or (tag in SEL)

    if _sel("a"):
        panel_octree_field("so_design.vtk", "sdf", "a", SDF_MIN, SDF_MAX, cmap="sdf")
    # (b): |u| on the trimmed material mesh, shown IN FULL (not cut), framed by the
    # SAME octree cube as the other panels.
    if _sel("b"):
        panel_octree_field("so_solve.vtk", "umag", "b", UMAG_MIN, UMAG_MAX,
                           cmap="solve", frame_vtk="so_design.vtk", cut=False,
                           edges=False)
    # (phi): the phase field on the trimmed solve mesh -- the chain's 2nd station
    if _sel("phi"):
        panel_phi("so_design.vtk", "so_solve.vtk", "phi")
    # (d): |dJ/dd| / max, log over GRAD_DECADES decades, Figure 1's magenta
    if _sel("d"):
        _st = dict(l.strip().split("=", 1) for l in open("so_stats.txt") if "=" in l)
        panel_grad_mag("so_grad.vtk", "d", design_vtk="so_design.vtk")
    if _sel("e"):
        panel_octree_field("so_updated.vtk", "sdf", "e", SDF_MIN, SDF_MAX, cmap="sdf",
                           prev_vtk="so_design.vtk")
    if _sel("g"):
        panel_restore_bc("g")

    # shape_opt_merged: the six loop renders, each on its block's tint
    #   (pvbatch render_so.py merged)       # all of them
    #   (pvbatch render_so.py m_g)          # just that one
    if (not SEL) or ("merged" in SEL) or any(t.startswith("m_") for t in SEL):
        for _tag, _bg, _fn in (
                ("m_a", PRE_TINT, lambda: panel_octree_field("so_design.vtk", "sdf", "m_a", SDF_MIN,
                                                      SDF_MAX, cmap="sdf")),
                ("m_phi", FEA_TINT, lambda: panel_phi("so_design.vtk", "so_solve.vtk", "m_phi")),
                ("m_b", FEA_TINT, lambda: panel_octree_field("so_solve.vtk", "umag", "m_b", UMAG_MIN,
                                                      UMAG_MAX, cmap="solve",
                                                      frame_vtk="so_design.vtk", cut=False,
                                                      edges=False)),
                ("m_d", FEA_TINT, lambda: panel_grad_mag("so_grad.vtk", "m_d",
                                                  design_vtk="so_design.vtk")),
                ("m_e", FEA_TINT, lambda: panel_octree_field("so_updated.vtk", "sdf", "m_e", SDF_MIN,
                                                      SDF_MAX, cmap="sdf",
                                                      prev_vtk="so_design.vtk")),
                ("m_g", PRE_TINT, lambda: panel_restore_bc("m_g")),
                # dJ/du on the trimmed solve mesh (so_dJdu.py), cut like the others
                ("m_gu", FEA_TINT, lambda: panel_grad_mag("so_dJdu.vtk", "m_gu", field="gu",
                                                  design_vtk="so_dJdu.vtk",
                                                  frame_vtk="so_design.vtk"))):
            if SEL and "merged" not in SEL and _tag not in SEL:
                continue
            BG = list(_bg)
            _fn()
        BG = [1.0, 1.0, 1.0]

    # evolution strip (panel f): one half-cut frame per so_evo_<it>_shell.stl
    if (not SEL) or ("evo" in SEL):
        for _f in sorted(glob.glob("so_evo_*_shell.stl"),
                         key=lambda p: int(re.search(r"so_evo_(\d+)_shell", p).group(1))):
            _it = int(re.search(r"so_evo_(\d+)_shell", _f).group(1))
            panel_evo(_f, f"evo_{_it}")
    # shape_opt_merged b: pvbatch render_so.py m_evo   (needs void_open.py's *_open.vtk)
    if "m_evo" in SEL:
        for _f in sorted(glob.glob("so_evo_*_shell_open.vtk"),
                         key=lambda p: int(re.search(r"so_evo_(\d+)_shell", p).group(1))):
            panel_m_evo(int(re.search(r"so_evo_(\d+)_shell", _f).group(1)))
    print("ALL PANELS DONE")
