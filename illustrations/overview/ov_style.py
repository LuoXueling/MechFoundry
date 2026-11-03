"""One camera, one cut, one colour table for every panel of method_overview.

Plain Python — imported by the pvbatch renderer, by ``ov_surfaces.py`` and by
``ov_figure.py`` alike, so none of the three can drift from the others.

CAMERA.  The paper's convention, unchanged: every bunny figure looks from
(1.5, -3, 1.25), z up.  The two kinds of panel keep their own figures' rules:

  * SURFACE panels are ``pfbunny/render_pf_bunny.py`` verbatim — the bunny
    MIRRORED in y (``MIRROR``, as bcregion / bcautobunny / pfbunny do), so the
    camera sees its face; its light kit with tone mapping OFF (on, it greys every
    painted colour); its matte finish; fitted to the bunny with a 1.02 dolly.
    The overview's generative assets are made ON that mirrored surface, so each
    input card shows exactly what its model was given.
  * CUBE panels are ``nitsche/render_dn.py`` verbatim — the UNmirrored mesh
    ``dn_prep`` builds (its asserted 45,082-node octree), seen from its back as
    diffused_nitsche shows it, cut at y = +0.009 keeping the far (+y) half,
    fitted to the cube with a 1.16 dolly.  The cut face is an x-z section, which
    the y-mirror does not change, so it faces the way the surface panels do.

COLOURS.  d and phi are ``nitsche/render_dn.py``'s ramps, copied (render_dn
renders on import, so it cannot be imported) and asserted equal to its source
text by ``check_render_dn()``.  Everything that carries identity is listed in
``PALETTE`` and gated by ``ov_palette_check.py``.
"""

import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))

VIEW_DIR = [1.5, -3.0, 1.25]      # every bunny figure's camera
VIEW_UP = [0.0, 0.0, 1.0]
VIEW_ANGLE = 25
MIRROR = [1.0, -1.0, 1.0]         # surface panels: bcregion / pfbunny's mirror
DOLLY_SURF = 1.02                 # render_pf_bunny.fit
DOLLY_CUBE = 1.16                 # render_dn.fit_all
CUT_N = [0.0, 1.0, 0.0]
CUT_Y = 0.009                     # render_dn's nudge; the far (+y) half is kept
PANEL_PX = 1400                   # every panel rendered square at this size
# render_pf_bunny's light kit and finish (tone mapping OFF), and render_dn's
# (tone mapping ON) — each panel kind keeps its own figure's
SURF_VIEW = (("UseColorPaletteForBackground", 0), ("UseFXAA", 1), ("UseSSAO", 1),
             ("UseToneMapping", 0), ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
             ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
             ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0), ("HeadLightKHRatio", 3.0))
SURF_FINISH = {"Ambient": 0.12, "Diffuse": 1.0, "Specular": 0.35, "SpecularPower": 30}
CUBE_VIEW = (("UseColorPaletteForBackground", 0), ("UseFXAA", 1), ("UseSSAO", 1),
             ("UseToneMapping", 1), ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
             ("KeyLightIntensity", 0.85), ("KeyLightWarmth", 0.55),
             ("FillLightKFRatio", 2.6), ("BackLightKBRatio", 3.0), ("HeadLightKHRatio", 3.0))

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
PHI_RGB = [
    0.0000, 0.804, 0.886, 0.984,
    0.1667, 0.620, 0.773, 0.957,
    0.3333, 0.427, 0.655, 0.925,
    0.5000, 0.224, 0.529, 0.898,
    0.6667, 0.145, 0.416, 0.749,
    0.8333, 0.094, 0.310, 0.584,
    1.0000, 0.051, 0.212, 0.420,
]


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# Identity colours (hex), gated all-pairs by ov_palette_check.py.
PALETTE = {
    "ink": "#1a1a21",
    "grey text": "#5f6672",
    "page": "#ffffff",
    "Neumann (ears)": "#dd8452",
    "Dirichlet (base)": "#12a594",
    "agent": "#6b4fbb",
    "backward": "#a3195b",
    "gpu": "#3b7d23",          # the GPU mark; NVIDIA's own #76b900 fails CVD (5.9)
}
FRAME = "#c3c9d2"                 # hairlines only: structure, not identity (ungated)
CARD = "#f1f2f4"                  # input-card fill: structure, not identity
NEUMANN = _hex(PALETTE["Neumann (ears)"])
DIRICHLET = _hex(PALETTE["Dirichlet (base)"])
BODY = [0.920, 0.900, 0.860]      # bunny_sdf_pv's warm off-white
CLAY = [0.760, 0.760, 0.770]      # an untextured generator output: neutral grey, so paint reads as paint
EDGE = [0.10, 0.10, 0.13]

# |dJ/dd| / max over three decades, white -> magenta (PiYG's pink arm), the
# stops pulled toward the low end: the sensitivity is sharply localised at the
# neck, and an even spacing left the rest of the band indistinguishable from 0.  Stops at
# #c51b7d: the next PiYG step, #8e0152, sits CVD 4.6 from the SDF ramp's brick.
GRAD_RGB = [
    0.00, 0.975, 0.975, 0.975,
    0.15, 0.992, 0.878, 0.937,
    0.40, 0.945, 0.714, 0.855,
    0.70, 0.871, 0.467, 0.682,
    1.00, 0.773, 0.106, 0.490,
]
MASKED_RGB = [0.93, 0.93, 0.93]   # frozen BC nodes: never updated, drawn neutral

# Segment tints: pastel (the user preferred them to saturated colours), eight
# so each of the loop's eight PartField parts has its own; ov_surfaces assigns
# them so that parts which touch get the most different pair.  All >= 16 CVD
# from the Neumann orange and Dirichlet teal one station later.
SEGMENT_TINTS = ("#c9c3e8", "#e8c3dc", "#c8dca8", "#ead9ad", "#b7d6ea", "#dcc9b8",
                 "#bfe0d8", "#cfd8e8")
GPU = _hex("#3b7d23")

# Young's modulus: the d ramp's own inside arm, light blue = compliant, blue =
# stiff — in the figure's family rather than a new hue (viridis's light green
# read as out of place next to it).
# (2026-10-08, the user: "people can tell from the colour that somewhere is soft or
# stiff") -- replaced by "steel", a SEMANTIC map shared with agentic_results: a
# straight line in CIELAB from a pale ice blue (soft: light, airy) to a deep steel
# (stiff: dark, dense), lightness monotone so greyscale keeps the reading; muted and
# cool, clear of the |u| cream, the BC teal/orange and the tags in all three visions
# (agentic_results/e_colormap_options.py).  E_RGB samples the Lab line at 9 stops;
# piecewise-linear sRGB between them strays <= 0.3 DeltaE00 from the exact line
# (ov_palette_check asserts the stops against E_LAB).
E_LAB = ((85.0, -4.0, -12.0), (29.0, -1.0, -29.0))   # soft -> stiff (L*, a*, b*)
E_RGB = ((0.7518, 0.8452, 0.9199), (0.6698, 0.7686, 0.8575), (0.5888, 0.6935, 0.7959),
         (0.5084, 0.6199, 0.735), (0.4284, 0.5479, 0.675), (0.348, 0.4777, 0.6158),
         (0.2658, 0.4094, 0.5574), (0.1775, 0.3432, 0.5), (0.0624, 0.2793, 0.4434))
E_SOFT, E_STIFF = list(E_RGB[0]), list(E_RGB[-1])


def e_rgb(t):
    """The E map at t in [0, 1] (0 = soft): an (..., 3) array for array t."""
    import numpy as np
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    x = np.linspace(0.0, 1.0, len(E_RGB))
    stops = np.asarray(E_RGB)
    return np.stack([np.interp(t, x, stops[:, k]) for k in range(3)], axis=-1)

# |u|: matplotlib's plasma sampled from 0.28 to 1 (its purple on), nine stops;
# pvbatch's Python has no matplotlib, so the samples are written out here.
PLASMA_FROM_PURPLE = [
    (0.535, 0.031, 0.65), (0.656, 0.13, 0.592), (0.756, 0.232, 0.509),
    (0.84, 0.334, 0.427), (0.91, 0.439, 0.35), (0.963, 0.554, 0.272),
    (0.992, 0.681, 0.195), (0.987, 0.822, 0.144), (0.94, 0.975, 0.131)]

# The three parts of the workflow, one light tint each: every step is a block
# of its part's tint, no edge (edged blocks read heavy, and a thin coloured rule
# under each title read worse — the user, 2026-09-29).  Shape generation = the
# three shape sources; pre-processing = segmentation, BCs, texture, materials;
# FEA and optimization = octree, trimmed mesh, solve, objective, gradients (the
# NVIDIA Warp loop; the octree joined it on 2026-10-16, since it is rebuilt
# every iteration).  Pastel tints do not separate under CVD simulation
# (generation / FEA 0.8, pre / FEA 2.3): the parts are also told apart by place
# and by the legend, and the caption names them.
PARTS = {
    "generation": {"label": "Shape generation", "fill": "#fcf0e6"},
    "pre": {"label": "Pre-processing", "fill": "#eaf1fa"},
    "fea": {"label": "FEA & optimization", "fill": "#ebf5e6"},
}
# Which part each rendered panel belongs to: it is rendered straight onto that
# tint (opaque), so it melts into its block — a panel rendered transparent and
# composited later carries the render background in its anti-aliased rim.
PANEL_PART = {"tripo": "generation", "cad": "generation", "painted": "generation",
              "octree": "fea", "seg_pf": "pre", "bc": "pre", "texture": "pre",
              "materials": "pre", "phibc": "fea", "solve": "fea", "grad_d": "fea",
              "grad_phi": "fea"}


def fill_rgb(part):
    """A part's tint as ParaView's 0..1 RGB."""
    return _hex(PARTS[part]["fill"])


def check_render_dn():
    """Assert the copied ramps equal render_dn.py's, read from its source."""
    src = open(os.path.join(HERE, "..", "nitsche", "render_dn.py")).read()
    tree = ast.parse(src)
    found = {}
    consts = {"SDF_VMIN": SDF_VMIN, "SDF_VMAX": SDF_VMAX}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("SDF_RGB", "PHI_RGB"):
                vals = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Name):
                        vals.append(consts[elt.id])
                    else:
                        vals.append(ast.literal_eval(elt))
                found[name] = vals
    assert found.get("SDF_RGB") == SDF_RGB, "SDF ramp drifted from render_dn.py"
    assert found.get("PHI_RGB") == PHI_RGB, "phi ramp drifted from render_dn.py"


if __name__ == "__main__":
    check_render_dn()
    print("ov_style: ramps match render_dn.py")
