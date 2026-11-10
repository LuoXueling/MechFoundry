"""The optimisation beat's surface-change colour map (numpy only; dv_prep and dv_compose).

The signed change from it_0 in h (experiments/geometry_cleanup/render_prep.py's
``change_h``: the distance to it_0's surface, + outside it = material added), coloured
by a NAVY-CYAN diverging map (YlGnBu tones; the user, 2026-11-08: the red end of the
clean-up figure's lut read like the |u| deformation colour, and PRGn's purple-green was
rejected).  Its MECHANICS are experiments/geometry_cleanup/render.py's ``lut()``
(shape_opt_merged panel c), READ from that file's source (it imports paraview at load):
CHANGE_MAX and the knots -- white 0.97 held within +-0.25 h, the +-1 h stops, saturation
at +-CHANGE_MAX -- interpolated in RGB as ParaView does there.  The end colours are this
video's own, and the cyan side has one extra knot:

    -3 h  #0c7c99   -1 h  #41b6c4   -0.5 h  #9cd7de   +-0.25 h  white 0.97
    +1 h  #253494   +3 h  #081d58

Cyan is light: on the plain ramp it would sit only dE76 16.8 off white at -0.5 h (the
threshold where arrows start, dv_prep END_MIN_H), less than the old red end's 21.6, so the
-0.5 h knot puts it half-way (dE76 24.6; navy's +0.5 h is 28.5 on the plain ramp).

render.py's lut itself (blue ... red) stays as it is, for the paper's figure.
"""

import ast
import os

import numpy as np

_RENDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments",
                       "geometry_cleanup", "render.py")


def _cleanup_lut():
    """(CHANGE_MAX, knots (n,), rgb (n, 3)) of render.py's lut(), from its source."""
    tree = ast.parse(open(_RENDER).read())
    cmax = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "CHANGE_MAX" for t in n.targets))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lut")
    pts = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(getattr(t, "attr", None) == "RGBPoints" for t in n.targets))
    vals = np.asarray(eval(compile(ast.Expression(pts), _RENDER, "eval"),
                           {"__builtins__": {}}, {"m": cmax}), float).reshape(-1, 4)
    return float(cmax), vals[:, 0], vals[:, 1:]


CHANGE_MAX, KNOTS, _RGB_PAPER = _cleanup_lut()
assert np.all(np.diff(KNOTS) > 0) and KNOTS[0] == -CHANGE_MAX and KNOTS[-1] == CHANGE_MAX
assert len(KNOTS) == 6 and np.allclose(KNOTS[1:5], [-1.0, -0.25, 0.25, 1.0]), KNOTS


def _hex(h):
    return [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]


# render.py's knots and white band with this video's ends (render.py's RGBPoints order),
# plus the cyan side's half-way knot at -0.5 h (the docstring: cyan is too pale there)
ENDS = {"-max": "#0c7c99", "-1": "#41b6c4", "+1": "#253494", "+max": "#081d58"}
CYAN_MID_H, CYAN_MID_K = -0.5, 0.5
_WHITE = _RGB_PAPER[2]
_CYAN_MID = _WHITE + CYAN_MID_K * (np.array(_hex(ENDS["-1"])) - _WHITE)     # #9cd7de
MAP_KNOTS = np.array([KNOTS[0], KNOTS[1], CYAN_MID_H, KNOTS[2], KNOTS[3], KNOTS[4], KNOTS[5]])
RGB = np.array([_hex(ENDS["-max"]), _hex(ENDS["-1"]), _CYAN_MID, _RGB_PAPER[2], _RGB_PAPER[3],
                _hex(ENDS["+1"]), _hex(ENDS["+max"])])
WHITE_TO = float(KNOTS[3])           # the white band's edge (h), +-
# the arrows: navy for added material (the +1 h stop itself: contrast 9.8 on white; on the
# navy surface the fill alone merges, 1.0-1.5, so dv_compose rings every arrow with a 4 px
# WHITE halo), deep teal for removed (the -3 h end: 4.5 on white, 2.0 on the 1 h cyan).
# Dark PRGn green / purple (#238b45 / #7b3294) and the mid tones #5aae61 / #9970ab came first.
ARROW_GROW, ARROW_SHRINK = "#253494", "#0c7c99"


def change_rgb(change_h):
    """sRGB (..., 3) of the signed change (h), clamped to +-CHANGE_MAX."""
    x = np.clip(np.asarray(change_h, float), MAP_KNOTS[0], MAP_KNOTS[-1])
    return np.stack([np.interp(x, MAP_KNOTS, RGB[:, k]) for k in range(3)], -1)
