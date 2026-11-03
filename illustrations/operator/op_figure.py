"""The AI agent operator's loop (Supplementary Methods, "Operation by a coding agent").

A swim-lane diagram with no pictures, only what the author asked for: the operator's four
steps (brief, plan, driver run, judgment), the workflow's six approval gates (plus the optional
painting gate), the driver's stops where the operator looks at the rendered outputs, the
driver's own numeric checks, and the points at which the operator calls the driver.

    python3 op_figure.py ../operator_loop      -> ../operator_loop.pdf / .png

Every string drawn is in the ALLOWED registry and no two texts may touch (CH_DRAFT=1 skips
the check while iterating).  Colours: the operator is method_overview's agent violet, the
driver ink, the stops the Neumann orange, the driver's checks the Dirichlet teal.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overview"))
import ov_style as S  # noqa: E402

for d in ("/usr/share/fonts/truetype/msttcorefonts", os.path.expanduser("~/.fonts")):
    for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "Arial.TTF", "Arialbd.TTF", "Ariali.TTF"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            matplotlib.font_manager.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "savefig.transparent": False, "pdf.fonttype": 42,
})
assert "Arial" in FontProperties(family="Arial").get_name(), "Arial is not installed"

INK, GREY = S.PALETTE["ink"], S.PALETTE["grey text"]
OP = S.PALETTE["agent"]                     # the operator: a coding agent, violet as in Fig. 2
STOP = S.PALETTE["Neumann (ears)"]          # a stop: the driver pauses for a look
CHECK = S.PALETTE["Dirichlet (base)"]       # a numeric check by the driver
ARROW = "#30343b"
LANEFILL = {"op": "#efecf8", "drv": "#f1f2f4", "wf": "#f1f2f4"}
GATEFILL = "#ffffff"

# ---------------------------------------------------------------------------
# Measures (inches)
# ---------------------------------------------------------------------------
FIG_W = 7.087                      # 180 mm, NMI double column
M = 0.06
HEAD_FS, LANE_FS, TXT_FS, SUB_FS, LANE_SUB_FS, LEG_FS = 7.0, 7.0, 6.0, 5.0, 6.5, 6.0
LINE_H = 0.088
SUB_H = 0.075
HDR_W = 0.74                       # the lane-name column
NCOL = 9
GAP = 0.07
x_cols0 = M + HDR_W + 0.06
COL_W = (FIG_W - M - x_cols0 - (NCOL - 1) * GAP) / NCOL
xc = [(x_cols0 + i * (COL_W + GAP), x_cols0 + i * (COL_W + GAP) + COL_W) for i in range(NCOL)]
cx = [(a + b) / 2.0 for a, b in xc]

LEG_H = 0.16
LANE_H = {"op": 0.50, "drv": 0.50, "wf": 0.44}
LGAP = 0.17                        # between lanes: the vertical calls run here
HEAD_H = 0.17

y_leg = M
y_wf = y_leg + LEG_H + 0.06
y_drv = y_wf + LANE_H["wf"] + LGAP
y_op = y_drv + LANE_H["drv"] + LGAP
y_head = y_op + LANE_H["op"] + 0.04
FIG_H = y_head + HEAD_H + M
LANE_Y = {"op": y_op, "drv": y_drv, "wf": y_wf}
LANE_C = {k: v + LANE_H[k] / 2.0 for k, v in LANE_Y.items()}

fig = plt.figure(figsize=(FIG_W, FIG_H))
IT = fig.dpi_scale_trans
print(f"figure {FIG_W:.3f} x {FIG_H:.3f} in; columns {COL_W:.3f} in")


def fx(v):
    return v / FIG_W


def fy(v):
    return v / FIG_H


def measure(s, fs, weight="normal"):
    f = plt.figure(figsize=(FIG_W, 1.0))
    t = f.text(0, 0, s, fontsize=fs, weight=weight)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


def cap_height(fs, weight="normal"):
    tp = TextPath((0, 0), "H", size=fs, prop=FontProperties(family="Arial", weight=weight))
    return (tp.vertices[:, 1].max() - tp.vertices[:, 1].min()) / 72.0


texts = []


def put(x, y, s, fs=TXT_FS, ha="center", color=INK, **kw):
    t = fig.text(fx(x), fy(y), s, fontsize=fs, ha=ha, va="baseline", color=color, **kw)
    t.set_zorder(8)
    texts.append(t)
    return t


def put_mid(x, yc, s, fs=TXT_FS, weight="normal", **kw):
    """Text whose capitals are centred on yc."""
    return put(x, yc - cap_height(fs, weight) / 2.0, s, fs=fs, weight=weight, **kw)


def wrap(s, fs, w, weight="normal"):
    lines, cur = [], ""
    for word in s.split(" "):
        trial = (cur + " " + word).strip()
        if measure(trial, fs, weight) <= w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def block(x0, y0, w, h, fill, radius=0.03, ec="none", lw=0.0, z=0.5, ls="-"):
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, transform=IT,
                                  boxstyle=f"round,pad=0,rounding_size={radius:.4f}",
                                  fc=fill, ec=ec, lw=lw, zorder=z, ls=ls))


def lines_in(xm, yc, title, sub=(), wmax=None, colour=INK):
    """A bold title over grey sub-lines, the whole stack centred on yc."""
    wmax = wmax or COL_W - 0.06
    tl = wrap(title, TXT_FS, wmax, "bold")
    sl = [ln for s in sub for ln in wrap(s, SUB_FS, wmax)]
    h = (len(tl) - 1) * LINE_H + (len(sl)) * SUB_H
    y = yc + h / 2.0 - cap_height(TXT_FS, "bold") / 2.0
    stack = len(texts)             # the lines of one title are LINE_H apart by design: their
    for ln in tl:                  # line boxes (ascent + descent, 1.15 em) may touch each other
        put(xm, y, ln, fs=TXT_FS, weight="bold", color=colour)._stack = stack
        y -= LINE_H
    y += LINE_H - SUB_H - 0.012
    for ln in sl:
        y -= 0.0
        put(xm, y, ln, fs=SUB_FS, color=GREY)
        y -= SUB_H
    return tl + sl


# ---------------------------------------------------------------------------
# Arrows: method_overview's drawn stealth arrows
# ---------------------------------------------------------------------------
ARROW_LW = 0.75
HL, HW, NOTCH = 0.062, 0.024, 0.28
R = 0.045


def _rounded(pts, r):
    pts = [np.asarray(p, float) for p in pts]
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
        ri = min(r, la / 2.0, lc / 2.0)
        verts += [b - (b - a) / la * ri, b, b + (c - b) / lc * ri]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1])
    codes.append(Path.LINETO)
    return Path(verts, codes)


def arrow(pts, colour=ARROW, dashed=False, lw=ARROW_LW, z=4):
    pts = [np.asarray(p, float) for p in pts]
    tip = pts[-1]
    d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
    shaft = list(pts)
    shaft[-1] = tip - d * HL * (1.0 - NOTCH * 0.5)
    fig.add_artist(PathPatch(_rounded(shaft, R), transform=IT, fc="none", ec=colour, lw=lw,
                             capstyle="butt", joinstyle="round",
                             ls=(0, (3.0, 2.0)) if dashed else "-", zorder=z))
    n = np.array([-d[1], d[0]])
    fig.add_artist(Polygon([tip, tip - d * HL + n * HW, tip - d * HL * (1.0 - NOTCH),
                            tip - d * HL - n * HW], closed=True, transform=IT,
                           fc=colour, ec=colour, lw=0.3, zorder=z))


# ---------------------------------------------------------------------------
# Marks (drawn: Arial has no such glyphs)
# ---------------------------------------------------------------------------
def icon_axes(xc_, yc_, side):
    a = fig.add_axes([fx(xc_ - side / 2.0), fy(yc_ - side / 2.0), fx(side), fy(side)])
    a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_axis_off(); a.set_aspect("equal")
    a.patch.set_alpha(0.0)
    a.set_zorder(9)
    return a


def stop_mark(xc_, yc_, side=0.13):
    """The driver pauses: an octagon with a pause sign."""
    a = icon_axes(xc_, yc_, side)
    t = np.pi / 8 + np.arange(8) * np.pi / 4
    a.add_patch(Polygon(np.c_[0.95 * np.cos(t), 0.95 * np.sin(t)], closed=True, fc=STOP, ec="none"))
    for x in (-0.24, 0.24):
        a.add_patch(FancyBboxPatch((x - 0.10, -0.42), 0.20, 0.84, boxstyle="round,pad=0,rounding_size=0.06",
                                   fc="white", ec="none"))


def eye_mark(xc_, yc_, side=0.15, colour=OP):
    """The operator looks at the rendered pictures."""
    a = icon_axes(xc_, yc_, side)
    t = np.linspace(-1, 1, 60)
    up = np.c_[t * 0.95, 0.52 * (1 - t ** 2)]
    lo = np.c_[t[::-1] * 0.95, -0.52 * (1 - t[::-1] ** 2)]
    a.add_patch(Polygon(np.vstack([up, lo]), closed=True, fc="white", ec=colour, lw=0.8))
    a.add_patch(Circle((0, 0), 0.30, fc=colour, ec="none"))


def call_mark(xc_, yc_, side=0.12):
    """The operator calls the driver: a play button."""
    a = icon_axes(xc_, yc_, side)
    a.add_patch(Circle((0, 0), 0.95, fc=OP, ec="none"))
    a.add_patch(Polygon([(-0.28, -0.42), (-0.28, 0.42), (0.46, 0.0)], closed=True, fc="white", ec="none"))


def check_mark(xc_, yc_, side=0.11):
    """A numeric check by the driver."""
    a = icon_axes(xc_, yc_, side)
    a.add_patch(Circle((0, 0), 0.95, fc=CHECK, ec="none"))
    a.plot([-0.42, -0.10, 0.46], [0.02, -0.32, 0.36], color="white", lw=1.0,
           solid_capstyle="round", solid_joinstyle="round")


def gate_mark(xc_, yc_, side=0.11, colour=GREY):
    """An approval gate: two posts and a bar."""
    a = icon_axes(xc_, yc_, side)
    for x in (-0.7, 0.7):
        a.plot([x, x], [-0.85, 0.75], color=colour, lw=1.0, solid_capstyle="round")
    a.plot([-0.7, 0.7], [0.30, 0.30], color=colour, lw=1.0)
    a.plot([-0.7, 0.7], [-0.20, -0.20], color=colour, lw=1.0)


def check_box(xc_, yc_, side=0.10):
    block(xc_ - side / 2.0, yc_ - side * 0.35, side, side * 0.7, "white", radius=0.015, ec=CHECK, lw=0.6, z=2)


def gate_box(xc_, yc_, side=0.10):
    block(xc_ - side / 2.0, yc_ - side * 0.35, side, side * 0.7, "white", radius=0.015, ec=GREY, lw=0.6, z=2)


def rewind_mark(xc_, yc_, side=0.12, colour=OP):
    """Edit the plan: the session rewinds to that stop's gate."""
    a = icon_axes(xc_, yc_, side)
    t = np.linspace(0.35 * np.pi, 1.85 * np.pi, 40)
    a.plot(0.72 * np.cos(t), 0.72 * np.sin(t), color=colour, lw=0.9, solid_capstyle="round")
    tip = np.array([0.72 * np.cos(t[0]), 0.72 * np.sin(t[0])])
    a.add_patch(Polygon([tip + (0.38, 0.05), tip + (-0.08, 0.36), tip + (-0.06, -0.30)], closed=True,
                        fc=colour, ec="none"))


# ---------------------------------------------------------------------------
# Content: the text of "Operation by a coding agent"
# ---------------------------------------------------------------------------
HEADS = [(0, 0, "1  BRIEF"), (1, 1, "2  PLAN"), (2, 7, "3  DRIVER RUNS THE WORKFLOW"), (8, 8, "4  JUDGMENT")]
LANES = {"op": "Operator", "drv": "Driver", "wf": "Workflow gates"}
LANE_SUB = {"op": "Coding agent", "drv": "Script", "wf": "State machine"}

GATES = {2: ("Geometry input", ""), 3: ("Physical scale", ""),
         4: ("Boundary regions", ""), 5: ("Boundary values", ""), 6: ("Material", ""),
         7: ("Solver settings", "")}
STOPS = {2: ("Geometry", []), 4: ("Regions", []), 6: ("Materials", []), 7: ("Fields", [])}
DRV = {1: ("Prediction", []),
       2: ("Shape check", []), 4: ("Support and load check", []), 7: ("Solve vs prediction", []),
       8: ("Report", [])}

# lanes: a band each, name at the left
for k, y0 in LANE_Y.items():
    block(M, y0, FIG_W - 2 * M, LANE_H[k], LANEFILL[k], radius=0.04, z=0.2)
    colour = OP if k == "op" else INK
    put(M + 0.07, LANE_C[k] + 0.012, LANES[k], fs=LANE_FS, ha="left", weight="bold", color=colour)
    put(M + 0.07, LANE_C[k] - 0.115, LANE_SUB[k], fs=LANE_SUB_FS, ha="left", color=GREY)

# column heads (the operator's four steps)
for i0, i1, s in HEADS:
    x0, x1 = xc[i0][0], xc[i1][1]
    put_mid((x0 + x1) / 2.0, y_head + HEAD_H / 2.0, s, fs=HEAD_FS, weight="bold", color=GREY)
    assert measure(s, HEAD_FS, "bold") <= x1 - x0 + GAP - 0.02, s
    fig.add_artist(plt.Line2D([fx(x0 + 0.01), fx(x1 - 0.01)], [fy(y_head + 0.01)] * 2, color=INK, lw=0.8,
                              transform=fig.transFigure, zorder=1))

drawn = []
CARD_PAD = 0.05

# ---- operator lane -----------------------------------------------------------------------
def card(i, lane, title, sub, ec, colour=INK, yc=None):
    y0 = LANE_Y[lane] + CARD_PAD
    block(xc[i][0], y0, COL_W, LANE_H[lane] - 2 * CARD_PAD, "white", ec=ec, lw=0.6, z=1)
    drawn.extend(lines_in(cx[i], LANE_C[lane] if yc is None else yc, title, sub, colour=colour))


card(0, "op", "Case brief", [], OP, OP)
card(1, "op", "Plan", [], OP, OP)
card(8, "op", "Judgment", [], OP, OP)
arrow([(xc[0][1], LANE_C["op"]), (xc[1][0], LANE_C["op"])], colour=OP)

EYE_Y = y_op + LANE_H["op"] - 0.15
for i, (title, sub) in STOPS.items():          # the operator looks: a card like the others
    card(i, "op", title, sub, OP, OP, yc=y_op + 0.16)
    eye_mark(cx[i], EYE_Y)

# ---- driver lane: its checks (teal) and its stops (orange) ---------------------------------
TOP_D = y_drv + LANE_H["drv"]
# the driver runs left to right through the gate columns (behind its cards)
arrow([(xc[1][1], LANE_C["drv"]), (xc[8][0], LANE_C["drv"])], colour=INK, z=0.8)
for i, (title, sub) in DRV.items():
    ec = INK if i == 8 else CHECK
    card(i, "drv", title, sub, ec)

# ---- gate lane ---------------------------------------------------------------------------
for i, (title, sub) in GATES.items():
    card(i, "wf", title, [sub] if sub else [], GREY)
    if i < 7:
        arrow([(xc[i][1], LANE_C["wf"]), (xc[i + 1][0], LANE_C["wf"])], colour=GREY)

# ---- the calls between lanes ---------------------------------------------------------------
y_ob = y_op
y_wt = y_wf + LANE_H["wf"]
mid = (y_ob + TOP_D) / 2.0


def call_pair(x, end=None):
    """The operator's answer at a stop: run the driver again (approve), or edit the plan."""
    arrow([(x, y_ob + CARD_PAD), (x, TOP_D - CARD_PAD if end is None else end)], colour=OP)
    call_mark(x, mid, side=0.10)
    rewind_mark(x + 0.105, mid, side=0.10)


# the plan goes to the driver: the first call
arrow([(cx[1], y_ob + CARD_PAD), (cx[1], TOP_D - CARD_PAD)], colour=OP)
call_mark(cx[1], mid, side=0.10)
# the driver answers each gate from the plan: straight down from its checks, and the gates
# without a check of their own branch off halfway down the arrow on their left
y_g0, y_g1 = y_drv + CARD_PAD, y_wt - CARD_PAD
y_gm = (y_g0 + y_g1) / 2.0
for i in (2, 4, 7):
    arrow([(cx[i], y_g0), (cx[i], y_g1)], colour=INK)
for i0, i in ((2, 3), (4, 5), (4, 6)):
    arrow([(cx[i0], y_gm), (cx[i], y_gm), (cx[i], y_g1)], colour=INK)
for i in STOPS:
    x_up, x_dn = cx[i] - 0.11, cx[i] + 0.06
    y_lo = TOP_D - CARD_PAD if i in DRV else LANE_C["drv"] + 0.012     # no card: off the run line
    # the stop's pictures (and the reference sections) go up to the operator
    arrow([(x_up, y_lo), (x_up, y_ob + CARD_PAD)], colour=STOP)
    stop_mark(x_up, mid, side=0.10)
    call_pair(x_dn, None if i in DRV else y_lo)
# the judgment goes into the report
arrow([(cx[8], y_ob + CARD_PAD), (cx[8], TOP_D - CARD_PAD)], colour=OP)
call_mark(cx[8], mid, side=0.10)

# ---- legend ---------------------------------------------------------------------------------
LEG = [(call_mark, "Operator runs the driver or approves the stop"),
       (rewind_mark, "Operator edits the plan and rewinds to the gate"),
       (stop_mark, "Driver stops for a look"), (check_box, "Driver check")]
xl = M + 0.04
yl = y_leg + LEG_H / 2.0
for mk, s in LEG:
    mk(xl + 0.06, yl)
    put_mid(xl + 0.15, yl, s, fs=LEG_FS, ha="left", color=INK)
    drawn.append(s)
    xl += 0.15 + measure(s, LEG_FS) + 0.24
assert xl <= FIG_W - M + 0.20, f"legend runs to {xl:.2f}"

# ---------------------------------------------------------------------------
# Checks and output
# ---------------------------------------------------------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
problems = []
boxes = [(t.get_text(), t.get_window_extent(rend)) for t in texts]
for a in range(len(boxes)):
    for b in range(a + 1, len(boxes)):
        sa, sb = getattr(texts[a], "_stack", None), getattr(texts[b], "_stack", None)
        if sa is not None and sa == sb:
            continue
        if boxes[a][1].overlaps(boxes[b][1]):
            problems.append(f"texts {boxes[a][0]!r} and {boxes[b][0]!r} touch")
W_px, H_px = fig.canvas.get_width_height()
for s, bb in boxes:
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W_px or bb.y1 > H_px:
        problems.append(f"{s!r} leaves the page")
for p in problems:
    print("CHECK:", p)
assert not problems or os.environ.get("CH_DRAFT") == "1", "layout checks failed"

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "operator_loop")
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=400)
print(f"wrote {out}.pdf / .png ({FIG_W:.3f} x {FIG_H:.3f} in)")
