"""Render step (pvbatch + EGL, GPU): the panels of the diffused-Nitsche figure.

    pvbatch render_dn.py [panel ...]      # default: all

  dn_panel_a.png — the full octree mesh coloured by signed distance d
  dn_panel_b.png — the same mesh over phi = 1/2 (1 - tanh(d/eps))
  dn_panel_c.png — the same, boundary region in grey
  dn_panel_d.png — the two BC regions painted apart, plus the embedded surface
  dn_panel_e.png — the production-TRIMMED octree mesh over phi

Every panel is the octree mesh with its cell frames, cut by the same plane, from
the same camera.  Camera, cut and light kit are bunny_sdf_pv's, mapped into this
figure's frame (that one is z-up, this one y-up), so the five sit beside it.
"""

import sys

from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405

PANELS = sys.argv[1:] or ["a", "b", "c", "d", "e"]
SIZE = [1700, 1700]

# seaborn-deep (AGENTS.md §3 house style)
# Overlay identity colours, validated all-pairs against the CVD gates (OKLab
# dE x100, Machado-Oliveira-Fernandes 2009 severity 1.0): worst pair 8.7 (>= 8
# target), worst normal-vision pair 17.9 (>= 15 floor), worst vs the phi field
# 12.3.  The house seaborn green (#55A868) had to go: against the house orange
# it collapses to dE 4.5 under deuteranopia — below the hard floor of 6 — at
# EVERY neutral, which is the classic red/green failure on the figure's two most
# important labels.  Teal clears it; the orange is unchanged so the ears still
# match bc_region_id.  The neutral is re-stepped so LIGHTNESS, not hue, separates
# it from both chromatic slots — any grey level with them (L 0.58-0.77) collapses
# under CVD however warm or cool it is, so the passing band is either well below
# or well above.  Above is the one to take: the boundary region is the base state
# and should recede, and a dark grey turns most of (c)/(d) into a heavy mass.
GRAY = [0.729, 0.729, 0.729]      # #bababa  boundary region (neutral base)
ORANGE = [0.867, 0.518, 0.322]    # #DD8452  Neumann  (house, == bc_region_id)
TEAL = [0.071, 0.647, 0.580]      # #12a594  Dirichlet
BODY = [0.920, 0.900, 0.860]      # bunny_sdf_pv's warm off-white

# Matching bunny_sdf_octree's view takes BOTH a mirror and a change of basis,
# This figure works in bunny_sdf_octree's own frame now (dn_prep applies NO
# transform to bunny.stl — see its WHICH BUNNY), so camera, view-up and cut are
# that figure's, verbatim from render_bunny_sdf.py: z-up, camera on -y, the cut
# plane's normal +y with the SOLID half kept on +y, away from the camera.  Do
# not re-derive these by mapping between frames — a 90 deg rotation of the mesh
# perturbs warp's distances enough to move the octree by 42 hexes, which is why
# the geometry stays put and the camera is what moved.
CUT_N = [0.0, 1.0, 0.0]
# y=0 is a cell boundary at every octree level, and clipping exactly on one
# makes ParaView emit degenerate slivers (AGENTS.md §5 "Clip gotcha"); nudge a
# hair into the cells so the cut face is a clean quad cross-section.  Sign and
# magnitude are sdfoctree/render_sdf_panel.py:63's, the panel this figure sits
# beside — it nudges its EDGED panel the same +0.009 and leaves the smooth one
# at 0, and matching it means the two show the very same cross-section.
CUT_Y = 0.009
# (c)/(d) draw two datasets on this one plane.  They used to be the whole mesh
# plus the tagged cells ON TOP of it — coincident geometry, so one had to be
# offset 0.02 to stop the cut faces z-fighting, and the overlay then stood that
# far proud of the backdrop.  0.02 was 0.64 h_min on the old, coarser mesh and
# went unnoticed; at this mesh's h_min = 0.0181 it is 1.10 h_min, a full cell of
# grey shelf sticking out of the cut face with its own lit side walls, which
# reads as an uncropped mesh.  So they are a PARTITION now — tag == 0 carries
# phi, tag >= 1 carries the region colours — sharing CUT_Y with nothing to
# fight, and nothing protrudes.  Do not reintroduce an offset.
VIEW_DIR = [1.5, -3.0, 1.25]      # == render_bunny_sdf.py's camera
VIEW_UP = [0.0, 0.0, 1.0]
VIEW_ANGLE = 25

SDF_VMIN, SDF_VMAX = -0.26, 0.72
SDF_RGB = [
    SDF_VMIN, 0.106, 0.196, 0.478,
    -0.130, 0.235, 0.443, 0.718,
    -0.040, 0.557, 0.706, 0.851,
    0.0, 0.894, 0.894, 0.894,
    0.040, 0.968, 0.805, 0.576,
    0.150, 0.944, 0.600, 0.282,
    0.380, 0.827, 0.329, 0.152,
    SDF_VMAX, 0.478, 0.129, 0.094,
]
# phi in [0,1] is a MAGNITUDE, not a signed deviation, so it takes a
# sequential ramp: one hue, light -> dark (dataviz skill, color-formula.md).
# The documented sequential blue, steps 100 -> 700, evenly placed; validated
# single-hue (4.1 deg spread) with a monotone OKLCH lightness step of 0.095.
# Blue also leaves orange and teal free for the BC overlays.  The `d` panel
# keeps the house DIVERGING map — d is signed, phi is not.
PHI_RGB = [
    0.0000, 0.804, 0.886, 0.984,
    0.1667, 0.620, 0.773, 0.957,
    0.3333, 0.427, 0.655, 0.925,
    0.5000, 0.224, 0.529, 0.898,
    0.6667, 0.145, 0.416, 0.749,
    0.8333, 0.094, 0.310, 0.584,
    1.0000, 0.051, 0.212, 0.420,
]


def _try(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        print("  (skip)", e)


def new_view(size=SIZE):
    """bunny_sdf_pv's view, verbatim: white, FXAA, tone mapping, soft LightKit.

    ``UseSSAO`` really is absent from this ParaView build (5.13.3); it stays in
    the guarded loop and just prints "(skip)".
    """
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


def cut(src, origin_y=CUT_Y, keep_far=True):
    """Halve on the shared plane.  ``keep_far`` keeps the side away from the
    camera (the solid block); the surface layer keeps the near side instead."""
    c = Clip(Input=src)  # noqa: F405
    c.ClipType = "Plane"
    c.ClipType.Origin = [0.0, origin_y, 0.0]
    c.ClipType.Normal = CUT_N
    # Camera sits on -y, so the far half is y >= origin — which is the side the
    # normal points at, i.e. the one Invert=0 keeps.
    c.Invert = 0 if keep_far else 1
    return c


def mesh_actor(rv, src, array, origin_y=CUT_Y):
    """The octree mesh, cut, coloured by a nodal array, with its cell frames."""
    d = Show(cut(src, origin_y=origin_y), rv)  # noqa: F405
    d.Representation = "Surface With Edges"
    ColorBy(d, ("POINTS", array))  # noqa: F405
    lut = GetColorTransferFunction(array)  # noqa: F405
    lut.ColorSpace = "Lab"
    lut.RGBPoints = SDF_RGB if array == "sdf" else PHI_RGB
    lut.NanColor = [0.6, 0.6, 0.6]
    d.Ambient = 0.28          # bunny_sdf_pv: faces read as a matte block
    d.Diffuse = 0.85
    d.Specular = 0.0
    _try(lambda: setattr(d, "EdgeColor", [0.10, 0.10, 0.13]))
    _try(lambda: setattr(d, "LineWidth", 0.9))
    _try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    d.SetScalarBarVisibility(rv, False)
    return d


def outline(rv, src):
    """The cube frame.  bunny_sdf_pv uses 1.2 px, but these panels are printed
    at 1.3 in, where a hairline drops out — same colour, a little heavier."""
    d = Show(Outline(Input=src), rv)  # noqa: F405
    d.DiffuseColor = [0.30, 0.32, 0.35]
    d.LineWidth = 2.4
    return d


def surface_actor(rv, path, colour, opacity, pbr=True, clip=True,
                  ambient=0.12):
    """A piece of the embedded surface, near half.  The pieces are disjoint by
    construction, so no two of them z-fight.

    ``pbr`` is off for every piece now.  The OPAQUE facets never could use it:
    ParaView's PBR path wants image-based lighting, and with no environment map
    an opaque surface renders a dull mid-tone whatever the light kit does — it
    turned the selected facets brown instead of orange, so they took
    bc_region_id's plain Gouraud material (AGENTS.md §5b).  The translucent body
    kept bunny_sdf_pv's PBR until it had to be made self-illuminating, and PBR
    cannot do that: VTK's PBR shader ignores the classic Ambient/Diffuse
    coefficients outright.  Measured, not assumed — rendering the panel at
    ambient 0.55 and at 1.0 under PBR gives a mean absolute pixel difference of
    EXACTLY 0.0.  So the body is Gouraud too, and its old PBR look is gone.

    ``clip`` is what keeps the OPAQUE pieces from going brown a second way.
    ParaView does not cap a clipped polydata, so cutting a closed shell leaves
    it open, and any piece whose opening faces the camera shows its own unlit
    interior: with the ears in the near half (dn_prep's mirror) the near ear
    rendered dark brown next to its orange twin, contradicting the key.  These
    pieces do not need the cut at all — the octree block is opaque and covers
    the whole cube up to CUT_Y, so it already hides everything behind the
    plane.  Leaving the shell CLOSED gives the same silhouette with no interior
    to look into.  The translucent body keeps its cut: it does not occlude, so
    an uncut body really would show its own far wall through the near one.

    ``ambient`` is how much the piece lights ITSELF, in its own colour (hence
    ``AmbientColor``; ParaView's default is white, which would wash it out).
    The translucent body needs it or the panel loses the bunny: at 0.42 opacity
    over a section of nearly its own value, only the light kit separates them,
    and the ghost's silhouette very nearly coincides with the section's, so
    there is no outline to catch either.  Self-illumination lifts the whole
    shell off the section at once, which shading cannot, and its shading then
    reads as the 3-D form it is.  0.24 is chosen against the phi field it sits
    on: by 0.45 the body has washed the interior out, and the panel has to show
    the cells too.
    """
    src = STLReader(FileNames=[path])  # noqa: F405
    nrm = GenerateSurfaceNormals(Input=src)  # noqa: F405
    _try(lambda: setattr(nrm, "FeatureAngle", 50))
    geom = cut(nrm, origin_y=0.0, keep_far=False) if clip else nrm
    d = Show(geom, rv)  # noqa: F405
    d.Representation = "Surface"
    d.DiffuseColor = colour
    d.Opacity = opacity
    d.Specular = 0.35
    d.SpecularPower = 30
    d.Ambient = ambient
    _try(lambda: setattr(d, "AmbientColor", colour))
    if pbr:
        _try(lambda: setattr(d, "Interpolation", "PBR"))
        _try(lambda: setattr(d, "Roughness", 0.45))
        _try(lambda: setattr(d, "Metallic", 0.0))
    else:
        d.Diffuse = 1.0
    return d


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    return [c / n for c in v]


def fit_all(rv, dolly=1.16):
    d = _unit(VIEW_DIR)
    rv.CameraFocalPoint = [0.0, 0.0, 0.0]
    rv.CameraPosition = [c * 3.0 for c in d]
    rv.CameraViewUp = VIEW_UP
    ResetCamera(rv)  # noqa: F405
    fp = list(rv.CameraFocalPoint)
    pos = list(rv.CameraPosition)
    dist = sum((pos[i] - fp[i]) ** 2 for i in range(3)) ** 0.5
    rv.CameraPosition = [fp[i] + d[i] * dist * dolly for i in range(3)]


def save(rv, tag):
    Render(rv)  # noqa: F405
    SaveScreenshot(f"dn_panel_{tag}.png", rv, ImageResolution=SIZE,  # noqa: F405
                   OverrideColorPalette="WhiteBackground")
    print(f"wrote dn_panel_{tag}.png")


# ---------------------------------------------------------------------------
# (a) / (b): the full octree mesh over d, then over phi
# ---------------------------------------------------------------------------
for tag, array in (("a", "sdf"), ("b", "phi")):
    if tag not in PANELS:
        continue
    rv = new_view()
    full = OpenDataFile("dn_full.vtk")  # noqa: F405
    full.UpdatePipeline()
    mesh_actor(rv, full, array)
    outline(rv, full)
    Render(rv)  # noqa: F405
    fit_all(rv)
    save(rv, tag)
    Delete(rv)  # noqa: F405

# ---------------------------------------------------------------------------
# (c) / (d): the same mesh with the tagged regions drawn over it
# ---------------------------------------------------------------------------
for tag in [p for p in ("c", "d") if p in PANELS]:
    rv = new_view()
    full = OpenDataFile("dn_full.vtk")  # noqa: F405
    full.UpdatePipeline()

    def tag_range(lo, hi):
        """The cells whose tag is in [lo, hi] — the two halves of the partition.
        Thresholding the one grid keeps a single source of truth, and both keep
        the canonical CUT_Y, so their cross-section is the very slice (e) shows.
        """
        t = Threshold(Input=full)  # noqa: F405
        t.Scalars = ["CELLS", "tag"]
        _try(lambda: setattr(t, "LowerThreshold", lo))
        _try(lambda: setattr(t, "UpperThreshold", hi))
        _try(lambda: setattr(t, "ThresholdMethod", "Between"))
        return t

    mesh_actor(rv, tag_range(-0.5, 0.5), "phi")     # backdrop: untagged cells
    dR = Show(cut(tag_range(0.5, 3.5)), rv)  # noqa: F405
    dR.Representation = "Surface With Edges"
    ColorBy(dR, ("CELLS", "tag"))  # noqa: F405
    tl = GetColorTransferFunction("tag")  # noqa: F405
    tl.ColorSpace = "RGB"
    # Hard steps: 1 = boundary region, 2 = Neumann (ears), 3 = Dirichlet (base).
    # (c) paints every tag one grey; (d) separates the two BC subsets.
    c2 = ORANGE if tag == "d" else GRAY
    c3 = TEAL if tag == "d" else GRAY
    tl.RGBPoints = [1.0, *GRAY, 1.5, *GRAY,
                    1.51, *c2, 2.5, *c2,
                    2.51, *c3, 3.0, *c3]
    tl.NanColor = [0.6, 0.6, 0.6]
    dR.Ambient = 0.22
    dR.Diffuse = 0.90
    dR.Specular = 0.0
    _try(lambda: setattr(dR, "EdgeColor", [0.10, 0.10, 0.13]))
    _try(lambda: setattr(dR, "LineWidth", 0.9))
    _try(lambda: setattr(dR, "RenderLinesAsTubes", 0))
    dR.SetScalarBarVisibility(rv, False)

    if tag == "d":
        # The embedded surface itself, near half, translucent over the cut —
        # the facets the selectors picked, in the colours their cells carry.
        # The body is translucent at bunny_sdf_pv's 0.42; the selected facets
        # are OPAQUE, or the dark exterior behind them muddies the orange into
        # a brown that no longer matches the cells they produced.
        surface_actor(rv, "dn_surf_rest.stl", BODY, 0.42, ambient=0.24,
                      pbr=False)
        surface_actor(rv, "dn_surf_ears.stl", ORANGE, 1.0, pbr=False, clip=False)
        surface_actor(rv, "dn_surf_base.stl", TEAL, 1.0, pbr=False, clip=False)

    outline(rv, full)
    Render(rv)  # noqa: F405
    fit_all(rv)
    save(rv, tag)
    Delete(rv)  # noqa: F405

# ---------------------------------------------------------------------------
# (e) the trimmed octree mesh — the domain the solver actually assembles on
# ---------------------------------------------------------------------------
if "e" in PANELS:
    rv = new_view()
    trim = OpenDataFile("dn_trim.vtk")  # noqa: F405
    trim.UpdatePipeline()
    mesh_actor(rv, trim, "phi")
    full = OpenDataFile("dn_full.vtk")  # noqa: F405
    outline(rv, full)           # the FULL cube, so (e) frames like (a)-(d)
    Render(rv)  # noqa: F405
    fit_all(rv)
    save(rv, "e")
    Delete(rv)  # noqa: F405

print("RENDERER:", GetOpenGLInformation().GetRenderer())  # noqa: F405
