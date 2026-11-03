"""Render step (pvbatch + EGL, GPU): every panel of the hidden-phi loop.

    pvbatch render_hp.py [slot ...]      # default: all
    pvbatch render_hp.py evo             # just the evolution strip

  hp_panel_a_d / a_hid     the design: d, and the hidden phi riding with it
  hp_panel_b               the diffused-Nitsche solve, |u| on the material mesh
  hp_panel_c_d / c_phi     the two sensitivities, each over its own maximum
  hp_panel_d_hid / d_d     the two updates: hidden phi dips, d moves in its band
  hp_panel_e               marching cubes + BC restore, sectioned to show the void
  hp_panel_evo_<it>        the strip

Every octree panel is the same mesh, cut by the same plane, from the same fixed
camera, so the reader tracks one object around the loop.  `hp_panel_b` is the
exception the sibling figure also makes: the trimmed material mesh is shown IN
FULL, uncut, because what it has to say is which region the FEM solved on.
Lighting and materials are `bunny_sdf_pv`'s (AGENTS.md §5).
"""

import os
import sys

from paraview.simple import *  # noqa: F403
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(
    globals().get("__file__", "render_hp.py"))), "..", "agentic_results"))
import ar_style as _AR  # noqa: E402  the load-arrow colour + finish
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(
    globals().get("__file__", "render_hp.py"))), "..", "_shared"))
import evo_style as _ES  # noqa: E402  shape_opt_merged b's frames (ghost opacity, voids)

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

SIZE = [1700, 1700]

stats = {}
try:                              # resilient so the render fns can be imported
    for line in open("hp_stats.txt"):
        k, *v = line.split()
        stats[k] = v[0] if len(v) == 1 else v
except FileNotFoundError:
    pass
EVO_ITERS = [int(x) for x in stats.get("evo_iters", [])]

# A VERTICAL profile: the reader has to see the blank standing on its feet with
# the load on top.  The feet are on the x/y axes, so the plane through two of
# them is the coordinate plane y = 0 — the bunny's cut.  A diagonal plane would
# cross every octree cell off its faces and ParaView would triangulate each one,
# turning the cell frames into a mesh of triangles instead of the quad grid the
# octree actually is.  The half that is KEPT is y >= 0, the one the camera does
# NOT sit on, so what faces the reader is the cut face.
CUT_N = [0.0, -1.0, 0.0]
# The coarse-aligned capture has a lattice face at y=0.  Nudge the visualization
# plane into the retained half only to make the chosen face unambiguous to VTK;
# this offset is unrelated to the outer-domain boundary, whose completeness is
# asserted by hp_prep.py before export.
CUT_ORIGIN = [0.0, 0.009, 0.0]

# The octree carries a shell of air out to the clip bbox; cropping to the blank
# plus one band keeps the panels from being mostly empty cells.  z must reach
# below the FEET (caps at -0.72) or the legs are sliced off flat at the panel
# edge and the blank looks like it is standing on nothing.
BOX_LO = [-1.14, -1.14, -0.80]
BOX_HI = [1.14, 1.14, 0.62]
CUT_U = [1.0, 0.0, 0.0]
BOX_U = 1.14

# Looking nearly along the cut normal, from a little above and a little to the
# side: the cut face fills the frame and the blank's full height reads as an
# elevation, while the few degrees off-axis keep it reading as a SOLID and not a
# flat drawing.  Panels this small (~1.4 in in the loop) cannot afford a more
# oblique view — the foreshortened cut face is where the void has to be visible.
VIEW_DIR = [0.30, -0.90, 0.30]
VIEW_UP = [0.0, 0.0, 1.0]
VIEW_ANGLE = 25
FOCUS = [0.0, 0.0, 0.0]
DOLLY = 0.88                       # zoom the cropped (e)/evo views IN so the
#                                   sectioned mesh fills more of the panel

INK = [0.10, 0.10, 0.13]
TEAL = [0.071, 0.647, 0.580]      # Dirichlet, and the hidden phi = 0.5 contour
ORANGE = [0.867, 0.518, 0.322]    # Neumann
BODY = [0.920, 0.900, 0.860]      # bunny_sdf_pv's warm off-white
CUT_FACE = [0.850, 0.828, 0.795]  # section face, only a HAIR darker than the body
#                                   (was 0.76 — too dark; the sections read muddy)
# The wall of the cavity the loop just carved, in (e).  A cold dark slate: it
# has to be unmistakably NOT one of the three BC/body tones (which are the warm
# family) and it has to read as a shadowed inside rather than as a painted
# patch, since what the panel claims is that material is missing there.
VOID = [0.310, 0.345, 0.400]
MASK_RGB = [1.0, 1.0, 1.0]        # the BC region, masked out of (c) — see below
FRAME_COLOR = [0.30, 0.32, 0.35]  # the octree-domain cube wireframe (== shape_opt_loop)

# d is SIGNED -> the house diverging map, neutral grey at 0 (AGENTS.md §5c),
# clamped to +-0.30.  Uses `shape_opt_loop`'s FULL house map verbatim (positive
# end to the dark brick, not a clamped mid-orange) so the two figures' d panels
# are coloured identically — the octree cube frame keeps the red "outside" band
# reading as domain, not object.
SDF_VMIN, SDF_VMAX = -0.30, 0.30
SDF_RGB = [
    -0.300, 0.106, 0.196, 0.478,
    -0.192, 0.235, 0.443, 0.718,
    -0.084, 0.557, 0.706, 0.851,
    0.000, 0.894, 0.894, 0.894,
    0.048, 0.968, 0.805, 0.576,
    0.132, 0.944, 0.600, 0.282,
    0.216, 0.827, 0.329, 0.152,
    0.300, 0.478, 0.129, 0.094,
]
# phi in [0,1] is a MAGNITUDE -> sequential blue, 100 -> 700 (AGENTS.md §5c).
PHI_RGB = [
    0.0000, 0.804, 0.886, 0.984,
    0.1667, 0.620, 0.773, 0.957,
    0.3333, 0.427, 0.655, 0.925,
    0.5000, 0.224, 0.529, 0.898,
    0.6667, 0.145, 0.416, 0.749,
    0.8333, 0.094, 0.310, 0.584,
    1.0000, 0.051, 0.212, 0.420,
]
# |dJ/d.| / max is a magnitude too, so also sequential — but it must not be
# confused with phi, which shares the row.  Single-hue orange, the house
# chromatic slot d's diverging map already owns on its positive side.  The ramp
# starts at PAGE WHITE and MASK_RGB is white too, so the ramp's floor, the
# masked BC region and the paper are one tone: below the floor the sensitivity
# is nil and the BC region's is never acted on, and neither deserves ink.
GRAD_RGB = [
    0.0000, 1.000, 1.000, 1.000,
    0.1667, 0.992, 0.839, 0.729,
    0.3333, 0.988, 0.737, 0.565,
    0.5000, 0.949, 0.588, 0.396,
    0.6667, 0.867, 0.435, 0.259,
    0.8333, 0.733, 0.310, 0.161,
    1.0000, 0.549, 0.176, 0.075,
]
# |u| is a third magnitude and needs a third hue family, or (b) reads as one
# more phi panel: plasma (dark purple -> yellow), as in `shape_opt_loop`.
UMAG_MAX = float(stats.get("umag_max", 1.0))
_PLASMA = [(0.0, 0.050, 0.030, 0.528), (0.1, 0.255, 0.014, 0.615),
           (0.2, 0.418, 0.001, 0.658), (0.3, 0.563, 0.052, 0.642),
           (0.4, 0.691, 0.165, 0.564), (0.5, 0.798, 0.280, 0.470),
           (0.6, 0.881, 0.393, 0.383), (0.7, 0.949, 0.518, 0.296),
           (0.8, 0.988, 0.652, 0.211), (0.9, 0.989, 0.810, 0.145),
           (1.0, 0.940, 0.975, 0.131)]
UMAG_RGB = [c for s, r, g, b in _PLASMA for c in (s * UMAG_MAX, r, g, b)]

# Figure 1's |dJ/dd| ramp (ov_style.GRAD_RGB): white -> magenta, the colour of the
# backward arrows.  Used by the g_d / g_phi slots that hidden_phi_loop shows; the
# orange c_d / c_phi slots stay for void_nucleation, which aliases them.
GRAD_MAG_RGB = [
    0.00, 0.975, 0.975, 0.975,
    0.15, 0.992, 0.878, 0.937,
    0.40, 0.945, 0.714, 0.855,
    0.70, 0.871, 0.467, 0.682,
    1.00, 0.773, 0.106, 0.490,
]
MAG_MASK_RGB = [0.975, 0.975, 0.975]
EPS = float(stats.get("eps", 0.015))

RAMP = {"sdf": SDF_RGB, "sdf_c": SDF_RGB, "hidden": PHI_RGB,
        "hidden_c": PHI_RGB, "gphi": GRAD_RGB, "gsdf": GRAD_RGB,
        "umag": UMAG_RGB}

# slot -> (file, array, d=0 iso source, hidden=0.5 iso source, cut?)
SLOT = {
    "a_d":   ("hp_pre.vtk", "sdf", "sdf", None, True),
    "a_hid": ("hp_pre.vtk", "hidden", "sdf", "hidden", True),
    "b":     ("hp_solve.vtk", "umag", None, None, False),
    "c_d":   ("hp_pre.vtk", "gsdf", "sdf", None, True),
    "c_phi": ("hp_pre.vtk", "gphi", "sdf", None, True),
    # (d) is the state the iteration's two updates left behind: hidden phi after
    # the gradient step and the thin-feature filter, d after its own step.  The
    # teal 0.5 curve is the prospective void wall — the threshold the production
    # viability test actually reads — so without it a reader sees a uniformly
    # paler block and cannot tell which part of it has crossed.
    "d_hid": ("hp_pre.vtk", "hidden_c", "sdf_c", "hidden_c", True),
    "d_d":   ("hp_pre.vtk", "sdf_c", "sdf_c", None, True),
}

# ---- hidden_phi_loop's own slots (prefix l_) ---------------------------------
# The loop figure is printed on loop_chain's 5-column grid (30.8 mm per octree
# panel, ~1380 px of ink here, 1 px = 0.022 mm), so its lines are sized for print:
# cell frames 0.3 pt (4.7 px) in mid grey (black at that weight fills the finest
# cells), RGB interpolation so each panel matches its matplotlib colour bar,
# Figure 1's magenta for |dJ/d.|, the paper's cream -> brick |u| ramp, phibar
# greyed outside the body, the previous d = 0 dashed on the stepped d, and a
# darker strip ghost.  void_nucleation aliases the UN-prefixed slots, which keep
# their original look.
L_EDGE_LW, L_FRAME_LW, L_ISO, L_ISO_H, L_ISO_H_THICK = 4.7, 4.7, 7.5, 9.0, 15.0
L_EDGE_RGB = [0.50, 0.51, 0.54]
L_EDGE_LIGHT = [0.76, 0.77, 0.79]          # gradient panels: frames at ~50 %
L_CASED, L_CASING = 5.5, 9.7               # d = 0 on gradient panels: 0.35 pt in white
L_OUTSIDE_RGB = [0.87, 0.87, 0.87]         # phibar outside the body: not acted on
L_GHOST = ([0.80, 0.785, 0.76], 0.34)     # strip ghost (the 10 % darkening was reverted)
SOLVE_BODY, SOLVE_MID, SOLVE_TOP = [0.920, 0.900, 0.860], [0.867, 0.518, 0.322], \
    [0.478, 0.129, 0.094]
L_UMAG_RGB = [0.0, *SOLVE_BODY, 0.35 * UMAG_MAX, *SOLVE_BODY, 0.70 * UMAG_MAX,
              *SOLVE_MID, UMAG_MAX, *SOLVE_TOP]
# shape_opt_merged's optional-branch renders (prefix m_): the l_ panel drawn OPAQUE
# onto the merged figure's FEA block tint (ov_style.PARTS["fea"]), as Figure 1 does.
BG = [1.0, 1.0, 1.0]
FEA_TINT = [1.0, 1.0, 1.0]        # tinted fills rejected (2026-10-13): white
MERGED_SLOTS = ["m_g_phi", "m_d_hid"]
# the loop figures' strips render EVERY sectioned frame hp_cut wrote (hp_prep's evo
# iterations plus any extra, e.g. 80), not only hp_stats' list
import glob as _glob
import re as _re
L_EVO = sorted({int(_re.search(r"hp_evo_(\d+)_shell", f).group(1))
                for f in _glob.glob("hp_evo_*_shell.stl")} | set(EVO_ITERS))
LOOP_SLOTS = ["l_geom", "l_a_d", "l_a_hid", "l_phi", "l_b", "l_g_d", "l_g_phi", "l_d_d",
              "l_d_hid", "l_evo_cap"]


def merged_evo(rv, key):
    """shape_opt_merged b's frame (round 11, slot m_evo_<it> / m_evo_cap): the l_evo frame
    with evo_style's fainter ghost and the voids darkened by openness
    (<base>_{shell,ghost}_open.vtk from ../_shared/void_open.py; for cap the shell is
    hp_e's four parts, all drawn in the body colour as l_evo_cap does)."""
    _try(lambda: setattr(rv, "UseDepthPeeling", 1))
    _try(lambda: setattr(rv, "MaximumNumberOfPeels", 24))
    base = "hp_e" if key == "evo_cap" else f"hp_evo_{key[4:]}"
    _ES.open_actor(rv, base + "_shell_open.vtk", BODY, name="open_body")
    face_actor(rv, base + "_face.stl")
    _ES.open_actor(rv, base + "_ghost_open.vtk", L_GHOST[0], opacity=_ES.M_GHOST_OPACITY,
                   name="open_ghost")
    return "strip frame, voids by openness"


def loop_panel(rv, key):
    """Render one hidden_phi_loop panel (key = slot without the l_ prefix)."""
    kw = dict(space="RGB", edge_rgb=L_EDGE_RGB, lw=L_EDGE_LW)
    if key.startswith("evo_"):
        # enough depth-peeling layers for the translucent ghost (render_so._evo_view)
        _try(lambda: setattr(rv, "UseDepthPeeling", 1))
        _try(lambda: setattr(rv, "MaximumNumberOfPeels", 24))
        # voids in the body colour, seen through the translucent near half (the
        # user, 2026-10-13: no separate void colour)
        if key == "evo_cap":
            # the captured iteration's own restored surface (panel e's mesh),
            # drawn as a strip frame, so the strip shows the panels' iteration
            section(rv, "hp_e", ghost=L_GHOST,
                    parts=[(p, BODY) for p in ("_body.stl", "_dir.stl", "_neu.stl",
                                               "_void.stl")])
        else:
            section(rv, f"hp_evo_{key[4:]}", ghost=L_GHOST)
        return "strip frame"
    if key == "geom":
        # the whole restored surface with its BC regions (hp_cut.py: hp_full_*), the
        # plain material of render_so.panel_restore_bc, fitted like the cube panels
        for part, colour in (("body", BODY), ("dir", TEAL), ("neu", ORANGE)):
            body_actor(rv, f"hp_full_{part}.stl", colour, pbr=False)
        # the top's downward traction: agentic_results' load arrows, built on this
        # surface by ../_shared/table_load_arrows.py, in ar_style's arrow colour + finish
        # (render_so.load_arrows draws the shape run's the same way)
        nrm = GenerateSurfaceNormals(Input=STLReader(FileNames=["hp_load_arrows.stl"]))  # noqa: F405
        _try(lambda: setattr(nrm, "FeatureAngle", 60))
        d = Show(nrm, rv)  # noqa: F405
        d.Representation = "Surface"
        d.DiffuseColor = list(_AR.LOAD_ARROW)
        d.AmbientColor = list(_AR.LOAD_ARROW)
        for k, v in _AR.ARROW_FINISH.items():
            setattr(d, k, v)
        d.SetScalarBarVisibility(rv, False)
        return "restored surface + BC + load arrows"
    outline("hp_pre.vtk", rv, lw=L_FRAME_LW)
    if key == "b":
        src = OpenDataFile("hp_solve.vtk")  # noqa: F405
        mesh_actor(rv, src, "umag", ramp=L_UMAG_RGB, edges=False, **kw)
        return "solve |u|"
    if key == "phi":
        # phi = (1 - tanh(d/eps))/2 on the TRIMMED solve mesh; its nodes are a
        # subset of the outer octree's, so resampling d onto them is exact.
        des = OpenDataFile("hp_pre.vtk")  # noqa: F405
        sol = OpenDataFile("hp_solve.vtk")  # noqa: F405
        rs = ResampleWithDataset(SourceDataArrays=des, DestinationMesh=sol)  # noqa: F405
        calc = Calculator(Input=rs)  # noqa: F405
        calc.ResultArrayName = "phi"
        calc.Function = f"1/(1+exp(2*sdf/{EPS}))"
        node = half_cut(calc)
        # Cell frames on the CUT FACE only: the far half's top is seen at a grazing
        # angle where print-weight frames merge into a grey lid.  The face is a
        # Slice a hair in front of the clip plane, drawn with edges over the
        # edge-less clipped solid.
        mesh_actor(rv, node, "phi", ramp=PHI_RGB, edges=False, **kw)
        sl = Slice(Input=calc)  # noqa: F405
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [0.0, CUT_ORIGIN[1] - 0.003, 0.0]
        sl.SliceType.Normal = [0.0, 1.0, 0.0]
        _try(lambda: setattr(sl, "Triangulatetheslice", 0))
        mesh_actor(rv, sl, "phi", ramp=PHI_RGB, **kw)
        iso(rv, node, "sdf", INK, L_ISO)
        return "trimmed mesh phi"
    src = OpenDataFile("hp_pre.vtk")  # noqa: F405
    src.UpdatePipeline()
    if key in ("a_hid", "d_hid"):
        h, sd = ("hidden", "sdf") if key == "a_hid" else ("hidden_c", "sdf_c")
        calc = Calculator(Input=src)  # noqa: F405
        calc.ResultArrayName = "hvis"
        calc.Function = f"{h}*({sd}<=0) - ({sd}>0)"   # outside the body -> grey
        node = half_cut(calc)
        mesh_actor(rv, node, "hvis", ramp=PHI_RGB, below=L_OUTSIDE_RGB, **kw)
        iso(rv, node, sd, INK, L_ISO)
        iso(rv, node, h, TEAL, L_ISO_H_THICK if key == "d_hid" else L_ISO_H)
        return h
    node = half_cut(src)
    if key in ("g_d", "g_phi"):
        # hp_prep stores v = log10(|g| / own max) / 4 + 1 (floored at 0, BC = -1).
        # Re-anchor the top at the 99th percentile of each field's OWN non-floor
        # values, clipped at 1, so a few extreme nodes no longer wash the band out;
        # floor and BC nodes keep their value.  Frames light, d = 0 cased in white.
        import numpy as np
        from paraview import servermanager
        from vtk.numpy_interface import dataset_adapter as dsa
        arr = "gsdf" if key == "g_d" else "gphi"
        v = np.asarray(dsa.WrapDataObject(servermanager.Fetch(src)).PointData[arr])
        shift = 1.0 - float(np.percentile(v[v > 0], 99))
        print(f"  {arr}: p99 sits {4 * shift:.2f} decades below the max")
        calc = Calculator(Input=src)  # noqa: F405
        calc.ResultArrayName = arr + "_p99"
        calc.Function = f"({arr}>0)*min({arr}+{shift:.6f},1) + ({arr}<=0)*{arr}"
        node = half_cut(calc)
        # the field drawn WITHOUT frames, the frames over it as a 40 % wireframe:
        # at 0.3 pt opaque frames cover the finest cells, which is where the band is
        mesh_actor(rv, node, arr + "_p99", ramp=GRAD_MAG_RGB, mask=MAG_MASK_RGB,
                   below=MAG_MASK_RGB, edges=False, space="RGB")
        wf = Show(ExtractSurface(Input=node), rv)  # noqa: F405  (own proxy, surface edges only)
        wf.Representation = "Wireframe"
        wf.ColorArrayName = ["POINTS", ""]
        wf.AmbientColor = wf.DiffuseColor = L_EDGE_RGB
        wf.LineWidth = L_EDGE_LW
        wf.Opacity = 0.4
        iso(rv, node, "sdf", [1.0, 1.0, 1.0], L_CASING, lift=0.5)
        iso(rv, node, "sdf", INK, L_CASED, lift=1.0)
    elif key == "a_d":
        mesh_actor(rv, node, "sdf", ramp=SDF_RGB, **kw)
        iso(rv, node, "sdf", INK, L_ISO)
    elif key == "d_d":
        mesh_actor(rv, node, "sdf_c", ramp=SDF_RGB, **kw)
        iso(rv, node, "sdf_c", INK, L_ISO)
        # the interface before the step: light dashes OVER the new curve (where the
        # step is sub-cell they read as one black/grey dashed line)
        iso(rv, node, "sdf", [0.62, 0.63, 0.66], 5.0, dashed=True)
    return key

SLOTS = sys.argv[1:] or (list(SLOT) + ["e"] + [f"evo_{i}" for i in EVO_ITERS]
                          + LOOP_SLOTS + [f"l_evo_{i}" for i in L_EVO] + MERGED_SLOTS)
if sys.argv[1:] == ["evo"]:
    SLOTS = [f"evo_{i}" for i in EVO_ITERS]
if sys.argv[1:] == ["merged"]:
    SLOTS = MERGED_SLOTS
if sys.argv[1:] == ["loop"]:
    SLOTS = LOOP_SLOTS + [f"l_evo_{i}" for i in L_EVO]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view():
    """bunny_sdf_pv's view, verbatim: white, FXAA, tone mapping, soft LightKit."""
    rv = CreateRenderView()  # noqa: F405
    rv.ViewSize = SIZE
    rv.Background = list(BG)
    rv.OrientationAxesVisibility = 0
    rv.CameraViewAngle = VIEW_ANGLE
    # SSAO OFF and key light 0.9: matches shape_opt_loop's panel material, which
    # renders the off-white body BRIGHT.  SSAO's ambient occlusion was greying the
    # sectioned (e)/(f) bodies well below shape_opt's white.
    for attr, val in (("UseColorPaletteForBackground", 0), ("UseFXAA", 1),
                      ("UseSSAO", 0), ("UseToneMapping", 1),
                      ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
                      ("KeyLightIntensity", 0.90), ("KeyLightWarmth", 0.55),
                      ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0),
                      ("HeadLightKHRatio", 3.0)):
        _try(lambda a=attr, v=val: setattr(rv, a, v))
    return rv


def _plane(src, origin, normal, invert):
    c = Clip(Input=src)  # noqa: F405
    c.ClipType = "Plane"
    c.ClipType.Origin = origin
    c.ClipType.Normal = normal
    c.Invert = invert                  # 1 keeps the side the normal points AWAY from
    return c


def half_cut(src):
    """Halve the octree near y=0 and keep the FAR half (the side the
    camera does not sit on), so the cut face faces the reader.

    No box crop: the WHOLE octree-domain cube is shown and framed by `outline`,
    exactly like `shape_opt_loop` (whose octree domain is identical).  The cube
    wireframe is the cue that tells the reader this is a profile through a cubic
    domain — without it a cropped brick reads as the object itself, not a cut.
    The 0.009 offset makes VTK choose the far-side layer unambiguously, as
    described at ``CUT_ORIGIN``.  A single PLANE clip cuts cells exactly otherwise (its implicit function is linear;
    ParaView's Box clip is not, and chamfers big cells into triangles)."""
    return _plane(src, CUT_ORIGIN, CUT_N, 1)


def outline(path, rv, lw=2.4):
    """The octree-domain cube as a wireframe (== shape_opt_loop / diffused_nitsche
    `outline`): drawn from the FULL, uncut octree so the cut mesh sits inside it."""
    o = Show(Outline(Input=OpenDataFile(path)), rv)  # noqa: F405
    o.DiffuseColor = FRAME_COLOR
    o.LineWidth = lw
    return o


def mesh_actor(rv, src, array, ramp=None, mask=None, space="Lab", edge_rgb=None,
               lw=0.9, edges=True, below=None):
    """The octree mesh, cut, coloured by a nodal array, with its cell frames."""
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface With Edges" if edges else "Surface"
    ColorBy(d, ("POINTS", array))  # noqa: F405
    lut = GetColorTransferFunction(array)  # noqa: F405
    lut.ColorSpace = space
    lut.RGBPoints = RAMP[array] if ramp is None else ramp
    mask = MASK_RGB if mask is None else mask
    lut.NanColor = mask
    if array in ("gphi", "gsdf"):
        # The BC region is written as a sentinel BELOW the ramp (hp_prep.py: its
        # SDF is frozen, so that sensitivity is never acted on).  Catch it with
        # the below-range colour — NaN would be the natural mechanism, but the
        # legacy ASCII reader cannot parse it and abandons the array silently.
        lut.RescaleTransferFunction(0.0, 1.0)
        _try(lambda: setattr(lut, "UseBelowRangeColor", 1))
        _try(lambda: setattr(lut, "BelowRangeColor", mask))
    if below is not None:
        # pin the range: with a below-range sentinel ParaView otherwise re-fits the
        # map to the data ([-1, 1]) at render time and 0 prints mid-ramp
        _try(lambda: setattr(lut, "AutomaticRescaleRangeMode", "Never"))
        lut.RescaleTransferFunction(0.0, 1.0)
        _try(lambda: setattr(lut, "UseBelowRangeColor", 1))
        _try(lambda: setattr(lut, "BelowRangeColor", below))
    d.Ambient = 0.28          # bunny_sdf_pv: faces read as a matte block
    d.Diffuse = 0.85
    d.Specular = 0.0
    _try(lambda: setattr(d, "EdgeColor", INK if edge_rgb is None else edge_rgb))
    _try(lambda: setattr(d, "LineWidth", lw))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    return d


def iso(rv, src, array, colour, width, dashed=False, lift=0.0):
    """A nodal iso-curve, drawn over the cut face.

    Contour the extracted SURFACE, not the volume: on the volume it returns the
    part's boundary as an isosurface, which stands in front of the cut face and
    hides it.  On the surface it returns curves — the part's outline on the cut
    face and its silhouette on the sides.  On the sensitivity panels it is what
    makes them legible at all (|dJ/dd| is zero over the whole interior AND the
    whole exterior, so without it a reader cannot tell which side is material).
    """
    ctr = Contour(Input=ExtractSurface(Input=src))  # noqa: F405
    ctr.ContourBy = ["POINTS", array]
    ctr.Isosurfaces = [0.5 if array.startswith("hidden") else 0.0]
    if dashed:
        # VTK cannot dash a line: keep the parts of the curve where a sine of
        # (x + z) is positive (period 0.16, ~1.7 mm at the loop figure's size).
        cal = Calculator(Input=ctr)  # noqa: F405
        cal.ResultArrayName = "dash"
        cal.Function = "sin(40*(coordsX+coordsZ))"
        ctr = Clip(Input=cal)  # noqa: F405
        ctr.ClipType = "Scalar"
        ctr.Scalars = ["POINTS", "dash"]
        ctr.Value = 0.0
        _try(lambda: setattr(ctr, "Invert", 0))
        # lift the dashes a hair toward the camera (along the view direction) so they draw over
        # the new d = 0 curve instead of z-fighting with it where the two coincide
        lift = lift or 1.0
    if lift:
        # 0.02 * lift along the view direction: draws over coincident geometry
        # without shifting on screen
        ctr = Transform(Input=ctr)  # noqa: F405
        ctr.Transform = "Transform"
        ctr.Transform.Translate = [lift * c for c in (0.00675, -0.01769, 0.00643)]
    d = Show(ctr, rv)  # noqa: F405
    d.Representation = "Surface"
    # The contour inherits its input's arrays, so ParaView colours it from one
    # of their LUTs and the "ink" line comes out mid-blue; scalar colouring has
    # to be switched off for DiffuseColor to take effect.  Set ColorArrayName
    # directly rather than calling ColorBy(d, None): that helper reads the rep's
    # CURRENT association and hands it back as a string, so on a rep with no
    # colouring yet it raises "invalid association string 'NONE'".
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = colour
    d.AmbientColor = colour
    d.Ambient = 1.0
    d.Diffuse = 0.0
    d.LineWidth = width
    _try(lambda: setattr(d, "RenderLinesAsTubes", 1))
    d.SetScalarBarVisibility(rv, False)
    return d


GHOST_OPACITY = 0.20


def body_actor(rv, path, colour, opacity=1.0, pbr=True):
    """`bunny_sdf_pv`'s material.

    With `pbr`, PBR included: §5b's warning that an opaque PBR surface with no
    environment map goes dull does NOT bite on the off-white body — under this
    light kit and tone mapping the solid half reads warmer and holds its form
    better than the matte version.  It DOES bite on a coloured actor: PBR pulls
    the Neumann orange down to a brown and the Dirichlet teal to a slate, so
    panel (e)'s three BC surfaces no longer matched their own legend swatches.
    They use `render_so.py`'s plain material instead, which is what the sibling
    figure's (e) uses and what the legend colours were picked against.
    """
    nrm = GenerateSurfaceNormals(Input=STLReader(FileNames=[path]))  # noqa: F405
    _try(lambda: setattr(nrm, "FeatureAngle", 50))
    d = Show(nrm, rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = list(colour)
    d.AmbientColor = list(colour)
    d.Opacity = opacity
    d.Specular = 0.35 if pbr else 0.30
    d.SpecularPower = 30
    if pbr:
        d.Ambient = 0.12
        d.Diffuse = 1.0
        _try(lambda: setattr(d, "Interpolation", "PBR"))
        _try(lambda: setattr(d, "Roughness", 0.45))
        _try(lambda: setattr(d, "Metallic", 0.0))
    else:
        d.Ambient = 0.14
        d.Diffuse = 1.0
    d.SetScalarBarVisibility(rv, False)
    return d


def face_actor(rv, path):
    """The flat cut face: an UNLIT fill, a step darker than the outer surface.

    Two reasons for unlit: a section face is a fill, not a lit form, so shading
    it adds nothing; and the boolean's cap triangulation is a fan of long
    slivers whose STL facet normals wobble by a degree or two, which under any
    diffuse term prints as a spray of diagonal scratches across the face.
    Darker than the body is the engineering convention for a sectioned solid,
    and the only cue that says "this is filled material" — which is exactly what
    makes the hole in it read as a void rather than a dent in a shell.
    """
    d = Show(STLReader(FileNames=[path]), rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = CUT_FACE
    d.AmbientColor = CUT_FACE
    d.Ambient = 1.0
    d.Diffuse = 0.0
    d.Specular = 0.0
    d.SetScalarBarVisibility(rv, False)
    return d


def section(rv, base, bc=False, ghost=None, parts=None):
    """A half-sectioned design: capped solid + flat cut face + translucent ghost.

    The ghost is what keeps the frame legible as a table — without it the
    profile alone loses the round tabletop and the far legs — and it is the same
    device `render_bunny_sdf.py` uses for its near half.  With `bc`, the solid
    half is split into body / Dirichlet / Neumann by the restoration's own face
    labels (`hp_cut.py`), which is what makes this panel the loop's
    "restore BC regions" step and not just another strip frame.
    """
    if parts:                      # explicit (suffix, colour) list
        for part, colour in parts:
            body_actor(rv, base + part, colour, pbr=False)
    elif bc:
        for part, colour in (("_body.stl", BODY), ("_dir.stl", TEAL),
                             ("_neu.stl", ORANGE), ("_void.stl", VOID)):
            body_actor(rv, base + part, colour, pbr=False)
    else:
        body_actor(rv, base + "_shell.stl", BODY, pbr=False)   # plain = bright
    face_actor(rv, base + "_face.stl")
    # (e)'s ghost is thinner than the strip's.  The strip has no interior to
    # hide, but (e) is the frame where the cavity first exists, and at ~1.1 k
    # background cells it is small enough that a 0.20 ghost over the section
    # reads as haze on the very feature the panel is for.
    if ghost is not None:          # (colour, opacity)
        body_actor(rv, base + "_ghost.stl", ghost[0], opacity=ghost[1], pbr=False)
        return
    body_actor(rv, base + "_ghost.stl", BODY,
               opacity=0.13 if bc else GHOST_OPACITY, pbr=False)


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit(rv):
    """ONE camera for every panel, sized off the geometry and not off the data,
    so panels whose octree is a little larger do not jump."""
    d = _unit(VIEW_DIR)
    rv.CameraFocalPoint = FOCUS
    rv.CameraPosition = [FOCUS[i] + d[i] * 3.0 for i in range(3)]
    rv.CameraViewUp = VIEW_UP
    ResetCamera(rv)  # noqa: F405
    rv.CameraFocalPoint = FOCUS
    # A sphere of radius r needs r / tan(FOV/2) = 4.51 r at 25 deg (§5b).  r is
    # the half-diagonal of the CUT FACE (2*BOX_U by the z extent), not half the
    # box's largest side: the face is what fills the frame.
    r = 0.5 * ((2.0 * BOX_U) ** 2 + (BOX_HI[2] - BOX_LO[2]) ** 2) ** 0.5
    rv.CameraPosition = [FOCUS[i] + d[i] * 4.51 * r * DOLLY for i in range(3)]


CUBE_VIEW_DIR = [0.42, -1.1, 0.40]  # a MODERATE oblique — enough off-axis to see
#   the cube's front + top + side (so the wireframe frame reads clearly as a 3-D
#   domain), but far less than shape_opt_loop's old [1.5,-3,1.5], which offset the
#   cut-away half into a big empty block.  SHARED with shape_opt_loop.
CUBE_DOLLY = 1.05                   # a little farther back, so the whole frame shows


def fit_cube(rv):
    """ResetCamera fit + dolly (== shape_opt_loop.set_camera): frame the WHOLE
    octree-domain cube (outline included) from shape_opt_loop's oblique camera, so
    the field panels match it exactly.  The evo strip and (e) keep `fit` (the fixed
    box, near-side elevation) so their sectioned frames do not jump in scale."""
    d = _unit(CUBE_VIEW_DIR)
    rv.CameraFocalPoint = FOCUS
    rv.CameraPosition = [FOCUS[i] + d[i] * 3.0 for i in range(3)]
    rv.CameraViewUp = VIEW_UP
    ResetCamera(rv)  # noqa: F405
    fp = list(rv.CameraFocalPoint)
    pos = list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * CUBE_DOLLY for i in range(3)]


# Guarded so render_hp_frames.py can import the actors/cameras above without this
# script rendering the whole loop; pvbatch runs it as __main__.
if __name__ == "__main__":
    for slot in SLOTS:
        BG = FEA_TINT if slot.startswith("m_") else [1.0, 1.0, 1.0]
        rv = new_view()
        if slot.startswith("m_evo_"):
            what = merged_evo(rv, slot[2:])
        elif slot.startswith("l_") or slot.startswith("m_"):
            what = loop_panel(rv, slot[2:])
        elif slot == "e":
            section(rv, "hp_e", bc=True)
            what = "restored surface, sectioned"
        elif slot.startswith("evo_"):
            section(rv, f"hp_evo_{slot[4:]}")
            what = "design surface, sectioned"
        else:
            path, array, iso_d, iso_h, do_cut = SLOT[slot]
            outline("hp_pre.vtk", rv)             # the octree-domain cube frame
            src = OpenDataFile(path)  # noqa: F405
            src.UpdatePipeline()
            node = half_cut(src) if do_cut else src
            mesh_actor(rv, node, array)
            if iso_d:
                iso(rv, node, iso_d, INK, 4.0)
            if iso_h:
                iso(rv, node, iso_h, TEAL, 8.0)
            what = f"{path}: {array}"
        Render(rv)  # noqa: F405   ParaView resets the camera on the FIRST render
        _strip = slot == "e" or slot.startswith("evo_") or slot.startswith("l_evo_") \
            or slot.startswith("m_evo_")
        (fit if _strip else fit_cube)(rv)
        Render(rv)  # noqa: F405
        print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
        if BG == [1.0, 1.0, 1.0]:
            SaveScreenshot(f"hp_panel_{slot}.png", rv, ImageResolution=SIZE,  # noqa: F405
                           OverrideColorPalette="WhiteBackground")
        else:
            SaveScreenshot(f"hp_panel_{slot}.png", rv, ImageResolution=SIZE)  # noqa: F405
        print(f"wrote hp_panel_{slot}.png  ({what})")
        Delete(rv)  # noqa: F405
    print("ALL PANELS DONE")
