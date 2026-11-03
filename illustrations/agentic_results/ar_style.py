"""One look for agentic_results, taken from method_overview rather than re-made.

Plain Python, imported by the prep, the pvbatch renderer and the compose, so the
three cannot drift.  Two sources, neither copied by eye:

  * ``overview/ov_style.py`` is imported as is: the surface-panel recipe
    (``SURF_VIEW`` light kit with tone mapping off, ``SURF_FINISH``), the view
    angle, the 1.02 surface dolly, the panel size, the clay, the Neumann /
    Dirichlet colours, the Young's-modulus ramp, the part tints and the ink.
  * ``overview/render_ov_solve.py`` (the overview's deformed bunny, which is
    diffused_nitsche's panel (f)) renders on import, so its recipe is written
    out here and ``check_render_ov_solve()`` asserts every number against that
    file's source text, the way ``ov_style.check_render_dn()`` pins its ramps.

FRAME.  The workflow's frame is y up with the object facing +z
(operator_skill/references/concepts.md).  The paper's bunny camera
(1.5, -3, 1.25) with z up is the same view after the z-up -> y-up rotation
(x, y, z) -> (x, z, -y): (1.5, 1.25, 3.0) with y up, which is also the
camera diffused_nitsche uses for the un-mirrored bunny.
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OVERVIEW = os.path.join(HERE, "..", "overview")
sys.path.insert(0, OVERVIEW)
import ov_style as S  # noqa: E402

VIEW_DIR = [1.5, 1.25, 3.0]       # S.VIEW_DIR, rotated into the workflow's y-up frame
VIEW_UP = [0.0, 1.0, 0.0]
VIEW_ANGLE = S.VIEW_ANGLE
DOLLY = S.DOLLY_SURF              # every panel of a row shares the surface panels' fit
PANEL_PX = S.PANEL_PX
SURF_VIEW = S.SURF_VIEW           # Figure 1's kit, as is (the gallery still renders with it)
SURF_FINISH = dict(S.SURF_FINISH)
# BC overlays in the supports-and-loads panel only: lit flatter than the clay, so
# a region facing away from the key light keeps its legend colour (review 2, M3).
# The review suggested 0.55 / 0.45; measured, regions facing away still sat at
# DeltaE00 11.6-15.1 from their chips.  Review 3 asked for the most shading the
# DeltaE <= 10 gate allows: swept, diffuse 0.35 is the largest with margin (worst
# region 8.4, the horse flank); 0.37 already fails (10.2).  Planar load faces carry
# one normal, so relief shows only on curved regions at any setting.
BC_OVERLAY_FINISH = {"Ambient": 0.70, "Diffuse": 0.35, "Specular": 0.0}
BC_MARKER_FINISH = {"Ambient": 0.75, "Diffuse": 0.30, "Specular": 0.0}   # the slab: a marker, kept flat
BC_MAX_DE00 = 10.0                # a region's median rendered colour vs its legend chip
PAINT_SPECULAR = 0.08            # render_ov_panels "painted": 0.35 made paint look plastic
CLAY = S.CLAY
NEUMANN = S.NEUMANN
# The 3-D load arrows (and the legend's arrow): the load colour's hue, darkened to
# CIELAB L* 44 from 64 (chroma x1.05).  Drawn in the region's own #dd8452, an arrow
# whose head sits on its orange patch (the umbrella's canopy, the rook's merlons,
# the pram's handle; arrow review 2026-10-12) lost its head in the patch; this shade
# is DeltaE00 19 from the patch and 39 from the clay, and still reads as "load".
LOAD_ARROW = S._hex("#a45123")
# Lit as a solid, not like the flat overlays: an ambient-heavy finish made the
# cylinder and cone print as one flat orange cut-out.  The diffuse term models the
# shaft round and the cone's facets; a little specular marks the head's tip.
ARROW_FINISH = {"Ambient": 0.45, "Diffuse": 0.65, "Specular": 0.18, "SpecularPower": 24}
DIRICHLET = S.DIRICHLET
E_SOFT, E_STIFF = S.E_SOFT, S.E_STIFF    # the ends of the shared "steel" E map
E_RGB, e_rgb = S.E_RGB, S.e_rgb          # the map itself (one definition: ov_style)
INK = S.PALETTE["ink"]
GREY_TEXT = S.PALETTE["grey text"]
FEATURE_ANGLE = 50                # render_ov_panels.surface / render_pf_bunny

# Each panel is rendered opaque onto the tint of the workflow part its step
# belongs to in method_overview (ov_style.PANEL_PART): the painted surface is
# generation, BC regions and materials are pre-processing, the solve is FEA.
PANEL_PART = {"painted": "generation", "bc": "pre", "materials": "pre", "solve": "fea"}
PANELS = ("painted", "bc", "materials", "solve")


BG = [1.0, 1.0, 1.0]              # this figure renders on white, not on part tints


def tint(panel):
    return S.fill_rgb(PANEL_PART[panel])


def tint_hex(panel):
    return S.PARTS[PANEL_PART[panel]]["fill"]


# -- render_ov_solve.py, written out (it renders on import) --------------------
SOLVE_VIEW = (("UseColorPaletteForBackground", 0), ("UseFXAA", 1), ("UseToneMapping", 1),
              ("KeyLightAzimuth", -35), ("KeyLightElevation", 45),
              ("KeyLightIntensity", 0.85), ("FillLightKFRatio", 2.6))
SOLVE_STOPS = (0.35, 0.70)        # u1, u2 as fractions of the row's own |u| max
SOLVE_BODY = [0.920, 0.900, 0.860]
SOLVE_MID = [0.867, 0.518, 0.322]
SOLVE_TOP = [0.478, 0.129, 0.094]
SOLVE_FINISH = {"Interpolation": "Gouraud", "Ambient": 0.42, "Diffuse": 0.88,
                "Specular": 0.28, "SpecularPower": 28}


# -- this figure's key light ----------------------------------------------------
# ParaView's light kit is camera-relative: azimuth is measured from the view axis,
# negative to the camera's left.  Figure 1's key (azimuth -35, elevation 45) suits
# its one bunny, whose face turns to the camera's left; here each row's camera is
# ar_prep.choose_view's own pick, so the object's front lands on either side, and
# a face turned to the camera's right got a Lambert cosine of only 0.12 from the key
# (0.58 head-on).  The user (2026-10-22): "change the light direction so that the
# front face is lightened".  The rule, one for every row and every panel: the key
# sits ON the view axis (azimuth 0) and lower (elevation 30), so the face looking
# at the viewer gets the key's full cosine 0.87 (was 0.58), faces turned 45 deg
# to either side 0.61 (were 0.70 left / 0.12 right) and an up-facing top 0.50 (0.71):
# the light now models the body top-versus-sides instead of left-versus-right.  The
# intensity, warmth, fill / back / head ratios and every finish stay Figure 1's,
# and the brightest face is no brighter than before (the key's best cosine on a
# visible face was already ~0.82-0.87), so nothing new clips.
KEY_LIGHT = (("KeyLightAzimuth", 0), ("KeyLightElevation", 30))


def relit(kit):
    """``kit`` with this figure's key-light angles (KEY_LIGHT) in place of its own."""
    key = dict(KEY_LIGHT)
    return tuple((a, key.get(a, v)) for a, v in kit)


def solve_rgb_points(umax):
    """render_ov_solve's |u| transfer function: the warm body colour up to
    0.35 umax, then orange, then brick at umax."""
    u1, u2 = SOLVE_STOPS[0] * umax, SOLVE_STOPS[1] * umax
    return [0.0, *SOLVE_BODY, u1, *SOLVE_BODY, u2, *SOLVE_MID, umax, *SOLVE_TOP]


# render_ar's kits: Figure 1's, re-aimed (the gates below still pin the sources)
AR_SURF_VIEW = relit(SURF_VIEW)
AR_SOLVE_VIEW = relit(SOLVE_VIEW)


def check_render_ov_solve():
    """Assert the solve recipe above equals render_ov_solve.py's source."""
    path = os.path.join(OVERVIEW, "render_ov_solve.py")
    tree = ast.parse(open(path).read())
    view, finish, rgb, stops = {}, {}, None, None

    def value(node):
        if isinstance(node, ast.Name):
            return node.id
        return ast.literal_eval(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            for elt in node.iter.elts:      # the guarded (attr, val) light-kit loop
                if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                    view[value(elt.elts[0])] = value(elt.elts[1])
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Attribute) and getattr(tgt.value, "id", None) == "d":
            if tgt.attr in SOLVE_FINISH:
                finish[tgt.attr] = value(node.value)
        if isinstance(tgt, ast.Attribute) and tgt.attr == "RGBPoints":
            rgb = [value(e) for e in node.value.elts]
        if isinstance(tgt, ast.Tuple) and [getattr(e, "id", None) for e in tgt.elts] == ["u1", "u2"]:
            stops = tuple(e.left.value for e in node.value.elts)
    assert view == dict(SOLVE_VIEW), f"solve light kit drifted: {view}"
    for src, ar in ((SURF_VIEW, AR_SURF_VIEW), (SOLVE_VIEW, AR_SOLVE_VIEW)):
        changed = {a for (a, v), (_, w) in zip(src, ar) if v != w}
        assert [a for a, _ in src] == [a for a, _ in ar] and changed <= set(dict(KEY_LIGHT)), \
            f"the figure's kit differs from Figure 1's beyond the key-light angles: {changed}"
    assert finish == SOLVE_FINISH, f"solve finish drifted: {finish}"
    assert stops == SOLVE_STOPS, f"solve stops drifted: {stops}"
    assert rgb == [0.0, *SOLVE_BODY, "u1", *SOLVE_BODY, "u2", *SOLVE_MID,
                   "umax", *SOLVE_TOP], f"solve ramp drifted: {rgb}"


if __name__ == "__main__":
    S.check_render_dn()
    check_render_ov_solve()
    print("ar_style: solve recipe == render_ov_solve.py; ramps == render_dn.py")
