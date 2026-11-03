"""Compose step (CPU): challenge_overview, the introductory figure before method_overview.

Reads ``ch_geom.npz`` (``ch_prep.py``); writes ``challenge_overview.{pdf,png}``.

THE STORY, read left to right.  One prompt, the object's name only ("a sitting
rabbit"), is taken up by three lanes, and each meets the same five steps:

  Human engineer  models the shape in CAD; the mesher fails, so the model is
                partitioned and meshed by hand; support and load faces picked and
                their values looked up in handbooks;
                a material set part by part; every design change remeshed by
                hand (the clock)
  AI agent     has a generator make the shape (an island; the tail pushed
                through the body); the meshing algorithm fails on it; supports become
                coded masks over nodes; one material for all; each design
                change makes the mesher fail again (the "!"); a chat-bubble icon,
                since the robot is MechFoundry's agents only
  MechFoundry   starts from the SAME generated shape (one cell across both
                lanes); adaptive meshing on an octree of the signed distance
                (refined round the island, which is not solid), parts picked
                by name, a material per part, and an automatic remesh after a
                design change that moves the surface and opens holes, the
                supports and loads kept

Only what MechFoundry actually changes is shown (the author, 2026-10-17); the
shape step is shared, not solved, and the solve itself is not a column, since commercial solvers already solve
nonlinear elasticity well.  Every pictogram is the bunny's own silhouette,
computed in ``ch_prep.py``; no rendered panel of method_overview is reused, so
this figure stays an illustration and Figure 2 stays the method.

House rules kept from method_overview (illustrations/AGENTS.md §5i): 180 mm
wide, Arial only, method_overview's own type sizes (6.0 / 6.5 pt), drawn arrows with stealth heads, identity colours
from ``overview/ov_style.PALETTE``, and a registry of every string the page may
carry.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import (Circle, FancyBboxPatch, PathPatch, Polygon,  # noqa: E402
                                Rectangle)
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
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "savefig.transparent": False, "pdf.fonttype": 42,
})
assert "Arial" in FontProperties(family="Arial").get_name(), "Arial is not installed"

INK, GREY = S.PALETTE["ink"], S.PALETTE["grey text"]
AGENT, BACK = S.PALETTE["agent"], S.PALETTE["backward"]
NEU, DIR = S.PALETTE["Neumann (ears)"], S.PALETTE["Dirichlet (base)"]
ARROW = "#30343b"
HUMAN = "#4a515b"                  # method_overview's human tab colour
FAIL = "#c0392b"                   # where an agent breaks (gated in ch_palette_check)
LINE = "#8c939d"                   # pictogram outlines and mesh lines (structure)
CADFILL = "#dfe2e7"                # a CAD part's flat fill
CLAYFILL = "#d6d6d9"               # S.CLAY, the generator output's grey, flattened
ROWFILL = {"human": "#f3f4f6", "agent": "#f3f4f6", "mf": "#efecf8"}   # violet is MechFoundry's alone
HEADFILL = {"human": "#eceef1", "agent": "#eceef1", "mf": "#ffffff"}
# the signed distance: method_overview's own ramp (blue inside, orange outside)
_st = np.asarray(S.SDF_RGB).reshape(-1, 4)
SDF_CMAP = LinearSegmentedColormap.from_list(
    "sdf", [((v - _st[0, 0]) / (_st[-1, 0] - _st[0, 0]), tuple(c)) for v, *c in _st])
TINT = dict(zip(("ears", "head", "body", "tail", "feet"), S.SEGMENT_TINTS))

G = np.load(os.path.join(HERE, "ch_geom.npz"), allow_pickle=True)
STATS = json.load(open(os.path.join(HERE, "ch_stats.json")))

# ---------------------------------------------------------------------------
# Measures (inches)
# ---------------------------------------------------------------------------
FIG_W = 7.087                      # 180 mm, NMI double column
M = 0.06

# Type sizes are method_overview's, like for like (the user, 2026-10-17), read from
# ov_figure.py's own assignment so the two figures can never drift apart (importing it would
# run its whole compose).  Stacked in intro_overview, the two must read as one page:
#   cell labels                -> its block titles  TITLE_FS (6.5; was 6.0)
#   column heads, lane names   -> its bold tabs     TAB_FS   (6.0 bold; were 6.5 and 7.0 bold)
#   the prompt                 -> its prompt card   CARD_FS  (6.0 italic; was 6.5 italic)
#   badges, every note inside a picture (defect names, TPU/PLA, the mask)
#                              -> its legend text   LEGEND_FS (6.0; notes were 5.0)
# The "?" marks are glyph-marks, not text; they take the title size (6.5 bold; were 7.0).
def _ov_sizes():
    import ast
    src = open(os.path.join(HERE, "..", "overview", "ov_figure.py")).read()
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                and [t.id for t in node.targets[0].elts][:1] == ["TITLE_FS"]):
            return dict(zip([t.id for t in node.targets[0].elts],
                            [ast.literal_eval(v) for v in node.value.elts]))
    raise RuntimeError("ov_figure.py no longer assigns TITLE_FS, CARD_FS, ...")


OV_FS = _ov_sizes()
HEAD_FS = ROW_FS = OV_FS["TAB_FS"]
TXT_FS = MARK_FS = OV_FS["TITLE_FS"]
PROMPT_FS = OV_FS["CARD_FS"]
SUB_FS = PIC_FS = OV_FS["LEGEND_FS"]
FORK = 0.24                        # bubble -> the three lanes: room for a bend and a head
HGAP = 0.10                        # header column -> first step column: as wide as the step gaps, for its arrow
GAP = 0.10                         # between step columns (the lane arrows sit in it)
BP = 0.04                          # a cell's inner padding
PIC_H = 0.50                       # a cell's picture height
LEAD_PT = 0.85                     # white space between two label lines' boxes (pt)
COL_HEAD = 0.17                    # the column-head band
RGAP = 0.07                        # between the two challenged lanes
SGAP = 0.12                        # challenged lanes -> the solved lane



def line_box(fs, weight="normal"):
    """Height (inches) of the box matplotlib gives EVERY line of text: from the top of
    'l' to the foot of 'p' (Text._get_layout sizes each line by that "lp" pair, whatever
    its own letters), measured on the glyph outlines, so no renderer and no dpi enter."""
    tp = TextPath((0, 0), "lp", size=fs, prop=FontProperties(family="Arial", weight=weight))
    return (tp.vertices[:, 1].max() - tp.vertices[:, 1].min()) / 72.0


# The pitch of a two-line label is that box plus a lead, not a bare number: 0.088 in was
# chosen by eye, and it is 0.0762 in of Arial 6 pt plus 0.85 pt.  Rounded to 0.001 in so
# the page does not move by outline noise (unrounded, the 1.7e-5 in excess shifts every
# row by a fraction of a pixel and re-antialiases 750 PNG pixels for nothing).  The two
# lines stay separate texts so the gate below checks each against every other text.
LINE_H = round(line_box(TXT_FS) + LEAD_PT / 72.0, 3)   # 0.088 in at 6 pt; 0.094 in at 6.5 pt
assert LINE_H - line_box(TXT_FS) >= 0.5 / 72.0, "label lines closer than half a point"

def _text_w(s, fs, weight="normal", style="normal"):
    f = plt.figure(figsize=(FIG_W, 1.0))
    t = f.text(0, 0, s, fontsize=fs, weight=weight, style=style)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


# The two left columns are fitted to what they hold, so the type sizes set the step columns'
# width (at method_overview's 6.5 pt cell titles, the engineer's Design label, then "Manual/Constrained
# remesh", needed 0.818 in for "Manual/Constrained", and with the old fixed 0.84 / 0.92 in left
# columns a cell's picture area was 0.813 in; the label is now "Remesh", the widths are unchanged):
#   the prompt bubble = its text + BUBBLE_PAD a side (0.107 in a side was the old look);
#   the header column = icon inset HDR_IN + the wider of the lane name and (badge mark + badge)
#   + HDR_OUT on the right (the old 0.07 / 0.15 / 0.03 in).
PROMPT = "“A sitting rabbit”"
LANES = (("Human engineer", "Slow and manual"), ("AI agent", "Error-prone"),
         ("MechFoundry", "Fully automatable"))
BUBBLE_PAD, HDR_IN, HDR_BADGE, HDR_OUT = 0.10, 0.07, 0.15, 0.03
CARD_W = round(_text_w(PROMPT, PROMPT_FS, style="italic") + 2 * BUBBLE_PAD, 3)
HDR_W = round(HDR_IN + max(max(_text_w(n, ROW_FS, "bold"), HDR_BADGE + _text_w(b, SUB_FS))
                           for n, b in LANES) + HDR_OUT, 3)
COL_W = (FIG_W - 2 * M - CARD_W - FORK - HDR_W - HGAP - 4 * GAP) / 5.0
x_card0, x_card1 = M, M + CARD_W
x_hdr0 = x_card1 + FORK
x_hdr1 = x_hdr0 + HDR_W
x_cols0 = x_hdr1 + HGAP
xc = [(x_cols0 + i * (COL_W + GAP), x_cols0 + i * (COL_W + GAP) + COL_W) for i in range(5)]
cx = [(a + b) / 2.0 for a, b in xc]
CELL = 2 * BP + PIC_H + 0.03 + 2 * LINE_H        # a label of up to two lines

# vertical, from the bottom up
yC = M
yB = yC + CELL + SGAP
yA = yB + CELL + RGAP
y_head = yA + CELL + 0.035
FIG_H = y_head + COL_HEAD + M
fig = plt.figure(figsize=(FIG_W, FIG_H))
IT = fig.dpi_scale_trans
print(f"figure {FIG_W:.3f} x {FIG_H:.3f} in; bubble {CARD_W:.3f}, headers {HDR_W:.3f}, columns {COL_W:.3f} in")


def fx(v):
    return v / FIG_W


def fy(v):
    return v / FIG_H


measure = _text_w


def cap_height(fs, weight="normal"):
    tp = TextPath((0, 0), "H", size=fs, prop=FontProperties(family="Arial", weight=weight))
    return (tp.vertices[:, 1].max() - tp.vertices[:, 1].min()) / 72.0


texts = []


def put(x, y, s, fs=TXT_FS, ha="center", va="baseline", color=INK, **kw):
    t = fig.text(fx(x), fy(y), s, fontsize=fs, ha=ha, va=va, color=color, **kw)
    t.set_zorder(max(t.get_zorder(), 6))
    texts.append(t)
    return t


def wrap(s, fs, w):
    """Greedy word wrap to width w (inches) at fs."""
    lines, cur = [], ""
    for word in s.split(" "):
        trial = (cur + " " + word).strip()
        if measure(trial, fs) <= w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def block(x0, y0, w, h, fill, radius=0.03, ec="none", lw=0.0, z=0.5):
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, transform=IT,
                                  boxstyle=f"round,pad=0,rounding_size={radius:.4f}",
                                  fc=fill, ec=ec, lw=lw, zorder=z))
    return (x0, y0, x0 + w, y0 + h)


# ---------------------------------------------------------------------------
# Arrows: method_overview's drawn stealth arrows, unchanged
# ---------------------------------------------------------------------------
ARROW_LW = 0.75
HL, HW, NOTCH = 0.062, 0.024, 0.28
R = 0.045
MIN_RUN = 0.035


def _rounded(pts, r):
    pts = [np.asarray(p, float) for p in pts]
    verts, codes = [pts[0]], [Path.MOVETO]
    radii = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
        ri = min(r, la / 2.0, lc / 2.0)
        radii.append(ri)
        verts += [b - (b - a) / la * ri, b, b + (c - b) / lc * ri]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1])
    codes.append(Path.LINETO)
    return Path(verts, codes), radii


def arrow(pts, colour=ARROW, dashed=False, head=True, r=R, lw=ARROW_LW, ax=None, scale=1.0):
    """Shaft through pts (rounded corners), a stealth head at pts[-1].  In
    inches on the page, or in an axes' data units (ax, scale = data per inch)."""
    pts = [np.asarray(p, float) for p in pts]
    tip = pts[-1]
    d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
    hl, hw = HL * scale, HW * scale
    shaft = list(pts)
    if head:
        shaft[-1] = tip - d * hl * (1.0 - NOTCH * 0.5)
    path, radii = _rounded(shaft, r * scale)
    if head and radii:
        run = np.linalg.norm(tip - pts[-2]) - radii[-1] - hl
        assert run >= MIN_RUN * scale, f"arrow into {tuple(np.round(tip, 3))}: head off a bend"
    tr = IT if ax is None else ax.transData
    tgt = fig if ax is None else ax
    add = tgt.add_artist if ax is None else tgt.add_patch
    add(PathPatch(path, transform=tr, fc="none", ec=colour, lw=lw, capstyle="butt",
                  joinstyle="round", ls=(0, (3.2, 2.2)) if dashed else "-", zorder=4))
    if head:
        n = np.array([-d[1], d[0]])
        add(Polygon([tip, tip - d * hl + n * hw, tip - d * hl * (1.0 - NOTCH),
                     tip - d * hl - n * hw], closed=True, transform=tr,
                    fc=colour, ec=colour, lw=0.3, joinstyle="miter", zorder=4))


# ---------------------------------------------------------------------------
# Icons (Arial has none of these glyphs; drawn, as in method_overview)
# ---------------------------------------------------------------------------
icons = []


def icon_axes(x, yc, side):
    a = fig.add_axes([fx(x), fy(yc - side / 2.0), fx(side), fy(side)])
    a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_axis_off()
    a.patch.set_alpha(0.0)
    a.set_zorder(7)
    icons.append(a)
    return a


def robot_mark(x, yc, side, colour):
    a = icon_axes(x, yc, side)
    a.add_patch(FancyBboxPatch((-0.78, -0.72), 1.56, 1.22, boxstyle="round,pad=0,rounding_size=0.3",
                               fc=colour, ec="none"))
    a.plot([0.0, 0.0], [0.5, 0.8], color=colour, lw=0.8, solid_capstyle="butt")
    a.add_patch(Circle((0.0, 0.86), 0.14, fc=colour, ec="none"))
    for ex in (-0.34, 0.34):
        a.add_patch(Circle((ex, -0.08), 0.17, fc="white", ec="none"))
    for sx in (-0.95, 0.95):
        a.add_patch(Rectangle((sx - 0.1, -0.33), 0.2, 0.36, fc=colour, ec="none"))


def chat_mark(x, yc, side, colour):
    """The AI agent: a chat bubble (the robot is MechFoundry's agents only)."""
    a = icon_axes(x, yc, side)
    a.add_patch(FancyBboxPatch((-0.92, -0.42), 1.84, 1.24, boxstyle="round,pad=0,rounding_size=0.32",
                               fc=colour, ec="none"))
    a.add_patch(Polygon([(-0.55, -0.40), (-0.05, -0.40), (-0.70, -0.95)], closed=True, fc=colour,
                        ec="none"))
    for ex in (-0.45, 0.0, 0.45):
        a.add_patch(Circle((ex, 0.20), 0.13, fc="white", ec="none"))


def person_mark(x, yc, side, colour):
    a = icon_axes(x, yc, side)
    a.add_patch(Circle((0.0, 0.42), 0.36, fc=colour, ec="none"))
    t = np.linspace(0.0, np.pi, 40)
    a.add_patch(Polygon(np.c_[0.78 * np.cos(t), -0.98 + 0.9 * np.sin(t)], closed=True,
                        fc=colour, ec="none"))


def clock_mark(x, yc, side, colour=HUMAN):
    a = icon_axes(x, yc, side)
    a.add_patch(Circle((0, 0), 0.86, fc="white", ec=colour, lw=0.8))
    a.plot([0, 0, 0.42], [0.55, 0, 0], color=colour, lw=0.8, solid_capstyle="round",
           solid_joinstyle="round")


def fail_mark(x, yc, side, colour=HUMAN):
    a = icon_axes(x, yc, side)
    a.add_patch(Circle((0, 0), 0.95, fc=colour, ec="none"))
    a.add_patch(FancyBboxPatch((-0.11, -0.12), 0.22, 0.72, boxstyle="round,pad=0,rounding_size=0.1",
                               fc="white", ec="none"))
    a.add_patch(Circle((0, -0.46), 0.13, fc="white", ec="none"))


def check_mark(x, yc, side):
    a = icon_axes(x, yc, side)
    a.add_patch(Circle((0, 0), 0.95, fc=AGENT, ec="none"))
    a.plot([-0.45, -0.12, 0.48], [0.02, -0.32, 0.36], color="white", lw=1.1,
           solid_capstyle="round", solid_joinstyle="round")


def cursor(ax, x, y, size, colour=INK):
    """A mouse pointer whose tip is at (x, y), in ax's data units."""
    p = np.array([(0, 0), (0, -1.0), (0.26, -0.76), (0.44, -1.12), (0.58, -1.05),
                  (0.40, -0.70), (0.74, -0.70)]) * size
    ax.add_patch(Polygon(p + (x, y), closed=True, fc="white", ec=colour, lw=0.6,
                         joinstyle="round", zorder=9))


# ---------------------------------------------------------------------------
# Pictograms: one axes per cell, equal aspect, data units of the geometry
# ---------------------------------------------------------------------------
PICS = []                          # (axes, x0, x1): every pictogram and its picture area's x span


def pic_axes(col, y0, w, h, lim):
    """An axes over the cell's picture box, fitted to lim = (x0, x1, y0, y1)."""
    x0, x1 = (xc[col] if isinstance(col, int) else col)
    x0, x1 = x0 + BP, x1 - BP
    a = fig.add_axes([fx(x0), fy(y0), fx(x1 - x0), fy(h)])
    a.set_axis_off()
    a.patch.set_alpha(0.0)
    a.set_zorder(3)
    lx0, lx1, ly0, ly1 = lim
    sc = max((lx1 - lx0) / (x1 - x0), (ly1 - ly0) / h)       # data per inch
    mx, my = (lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0
    a.set_xlim(mx - sc * (x1 - x0) / 2.0, mx + sc * (x1 - x0) / 2.0)
    a.set_ylim(my - sc * h / 2.0, my + sc * h / 2.0)
    a._sc = sc
    PICS.append((a, x0, x1))
    return a


def poly(ax, xy, holes=(), fc=CADFILL, ec=INK, lw=0.6, z=2, **kw):
    verts, codes = [], []
    for ring_ in [xy, *holes]:
        r_ = np.asarray(ring_, float)
        verts += list(r_) + [r_[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(r_) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=fc, ec=ec, lw=lw, joinstyle="round",
                           zorder=z, **kw))


def ptext_mid(ax, x, y, s, per_inch, fs=PIC_FS, **kw):
    """Text whose capitals are centred on y (va='center' centres the full box, descent
    included, so capitals sit high in a field); per_inch = data units per inch in y."""
    return ptext(ax, x, y - cap_height(fs) * per_inch / 2.0, s, fs=fs, va="baseline", **kw)


def ptext(ax, x, y, s, fs=PIC_FS, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    t = ax.text(x, y, s, fontsize=fs, zorder=10, **kw)
    texts.append(t)
    return t


SIL = G["sil"]
BN_LIM = (-0.08, SIL[:, 0].max() + 0.10, -0.06, 1.06)
BN_LIM_R = (-0.08, SIL[:, 0].max() + 0.40, -0.06, 1.06)
MASK = "z < 0.05"
DEFECT = ("Self-intersection", "Isolated", "components")    # each named where it sits
NAMES = [str(n) for n in G["part_names"]]
PART = {k: list(G[f"part_{k}"]) for k in NAMES}
TRI_PART = G["tri_part"]
BC_COL = {"feet": DIR, "ears": NEU}
MAT = {"ears": S.e_rgb(0.05), "body": S.e_rgb(0.75)}      # soft ears, stiff body: the same two in every row


def parts(ax, fill, ec="white", lw=0.4):
    for k in NAMES:
        for r_ in PART[k]:
            poly(ax, r_, fc=fill(k), ec=ec, lw=lw, z=3)
    poly(ax, SIL, fc="none", ec=INK, lw=0.5, z=4)


PRESS_LEN = GAP        # (in) every load arrow, head included: as long as the lane arrows
PRESS_GAP = 0.012      # (in) between the arrow head (tip AND wings) and the ear's surface
FIELD_PAD = 0.018      # (in) a text field's margin round its capitals (the mask chip, TPU/PLA)
TOP_PAD = 0.015        # (in) the tails keep at least this far below the cell's top edge


def _upper(ring, x):
    """The polygon's upper edge at x: the highest crossing of the vertical line x with its edges."""
    p, q = ring, np.roll(ring, -1, axis=0)
    lo, hi = np.minimum(p[:, 0], q[:, 0]), np.maximum(p[:, 0], q[:, 0])
    on = (lo <= x) & (x <= hi) & (hi > lo)
    t = (x - p[on, 0]) / (q[on, 0] - p[on, 0])
    return (p[on, 1] + t * (q[on, 1] - p[on, 1])).max(initial=-np.inf)


def ear_presses(scale):
    """Where the load arrows press, one per ear, all in data units (scale = data per inch).
    The ears' upper edge has two humps (one per ear) and a dip between them; each arrow sits
    over the middle of its hump's x-span (the left ear's tip, the right ear's slanted top halfway
    out, so it stays off the cell's centre line, where the incoming violet arrow points) and its
    tip stops PRESS_GAP above the highest edge point under the whole head, so neither the tip
    nor the wings touch the orange ear (on it an arrow would vanish, or seem to stab it)."""
    r_ = np.vstack([np.asarray(p, float) for p in PART["ears"]])
    xs = np.linspace(r_[:, 0].min(), r_[:, 0].max(), 401)[1:-1]
    upper = np.array([_upper(r_, x) for x in xs])
    half = len(xs) // 2
    lo, hi = np.argmax(upper[:half]), half + np.argmax(upper[half:])
    dip = lo + np.argmin(upper[lo:hi + 1])
    reach = (HW + PRESS_GAP) * scale            # the head's half width, plus the gap
    out = []
    for x_ in (xs[:dip + 1], xs[dip:]):
        xm = 0.5 * (x_[0] + x_[-1])
        under = [_upper(r_, x) for x in np.linspace(xm - reach, xm + reach, 41)]
        out.append((xm, max(under) + PRESS_GAP * scale))
    return out


def press(ax, scale):
    """The load: the ears pressed DOWN, by orange arrows of one length that stop just above each
    ear's upper surface (rule: ear_presses).  The cell's top margin above the picture box is
    theirs too, so the patches are unclipped; the tails must stay TOP_PAD below the cell's top
    edge (the cell top is the axes' top + BP), clear of the violet arrow coming in above it."""
    y_cell = ax.get_ylim()[1] + BP * scale
    n0 = len(ax.patches)
    for x, y in ear_presses(scale):
        tail = y + PRESS_LEN * scale
        assert tail <= y_cell - TOP_PAD * scale + 1e-6, "a load arrow leaves its cell's top margin"
        arrow([(x, tail), (x, y)], colour=NEU, ax=ax, scale=scale, lw=1.1)
    for p_ in ax.patches[n0:]:
        p_.set_clip_on(False)


def press_lim(x0, x1, y0):
    """The frame of the support-and-load cell: (x0, x1, y0) as given, the top just high enough
    that the load arrows (PRESS_LEN, PRESS_GAP and TOP_PAD above the higher landing point, less
    the cell's own BP above the picture box) fit in the cell.  With the frame fitted by its
    height (data per inch sc = (top - y0) / PIC_H), top - y_land = k sc solves to
    top = (y_land - y0 k / PIC_H) / (1 - k / PIC_H); the landing point depends on sc a little
    (the head's width), so iterate."""
    k = PRESS_GAP + PRESS_LEN + TOP_PAD - BP
    top = 1.0
    for _ in range(4):
        sc = (top - y0) / PIC_H
        y_land = max(y for _, y in ear_presses(sc)) - PRESS_GAP * sc
        top = (y_land - y0 * k / PIC_H) / (1.0 - k / PIC_H)
    assert (x1 - x0) / (COL_W - 2 * BP) <= (top - y0) / PIC_H, "the frame is fitted by its width"
    return (x0, x1, y0, top)


def pins(ax, scale):
    """The support: pinned triangles under the teal feet, on a hatched ground line."""
    x0, x1 = 0.22, 0.86
    h = 0.09
    for x in np.linspace(x0, x1, 3):
        ax.add_patch(Polygon([(x, 0.0), (x - 0.055, -h), (x + 0.055, -h)], closed=True,
                             fc="white", ec=DIR, lw=0.6, zorder=5))
    ax.plot([x0 - 0.10, x1 + 0.10], [-h, -h], color=DIR, lw=0.7, zorder=5)
    for x in np.arange(x0 - 0.08, x1 + 0.10, 0.07):
        ax.plot([x, x - 0.04], [-h, -h - 0.045], color=DIR, lw=0.4, zorder=5)


def mesh(ax, P_, T_, fc=CLAYFILL, ec="white", lw=0.35, skip=()):
    keep = np.array([i for i in range(len(T_)) if i not in set(skip)])
    ax.add_collection(PolyCollection(P_[T_[keep]], fc=fc, ec=ec, lw=lw, zorder=3))


def cross(ax, x, y, r):
    """A red X in a ring: the meshing algorithm stopped."""
    ax.add_patch(Circle((x, y), r, fc=FAIL, ec="none", zorder=8))
    for sgn in (1, -1):
        ax.plot([x - 0.5 * r, x + 0.5 * r], [y - sgn * 0.5 * r, y + sgn * 0.5 * r], color="white",
                lw=0.9, solid_capstyle="round", zorder=9)


def redo_mark(ax, x, y, r, colour):
    """A circular 'again' arrow, drawn (Arial has no such glyph)."""
    t = np.linspace(np.radians(110), np.radians(400), 40)
    pts = np.c_[x + r * np.cos(t), y + r * np.sin(t)]
    ax.plot(pts[:-3, 0], pts[:-3, 1], color=colour, lw=0.8, solid_capstyle="butt", zorder=8)
    d = pts[-1] - pts[-4]
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tip = pts[-1]
    hl, hw = 0.55 * r, 0.32 * r
    ax.add_patch(Polygon([tip, tip - d * hl + n * hw, tip - d * hl - n * hw], closed=True,
                         fc=colour, ec="none", zorder=8))


def cad_part(ax, handles=True):
    """The bunny as a CAD part: flat faces split along the part edges, spline handles."""
    for k in NAMES:
        for r_ in PART[k]:
            poly(ax, r_, fc=CADFILL, ec=LINE, lw=0.35, z=3)
    poly(ax, SIL, fc="none", ec=INK, lw=0.6, z=4)
    if handles:
        for p_ in SIL[::7]:
            ax.add_patch(Rectangle(p_ - 0.022, 0.044, 0.044, fc="white", ec=INK, lw=0.4, zorder=5))


# --- row one: the engineer -----------------------------------------------------------
def h_shape(ax):
    cad_part(ax)
    x_ = SIL[:, 0].max() + 0.14             # a height dimension beside the part (keeps the row's frame)
    sc = ax._sc
    arrow([(x_, 0.5), (x_, 1.0)], colour=GREY, ax=ax, scale=sc * 0.6, lw=0.5)
    arrow([(x_, 0.5), (x_, 0.0)], colour=GREY, ax=ax, scale=sc * 0.6, lw=0.5)
    for yy in (0.0, 1.0):
        ax.plot([x_ - 0.40 if yy == 0.0 else 0.55, x_ + 0.05], [yy, yy], color=GREY, lw=0.4, zorder=2)
    p_ = SIL[7 * 3]
    cursor(ax, p_[0] + 0.01, p_[1] - 0.01, 0.17)


def tri_quality(P_, T_):
    """Radius ratio 2 r_in / r_circ, 1 for an equilateral triangle."""
    a = np.linalg.norm(P_[T_[:, 1]] - P_[T_[:, 2]], axis=1)
    b = np.linalg.norm(P_[T_[:, 0]] - P_[T_[:, 2]], axis=1)
    c = np.linalg.norm(P_[T_[:, 0]] - P_[T_[:, 1]], axis=1)
    s_ = (a + b + c) / 2
    area = np.sqrt(np.maximum(s_ * (s_ - a) * (s_ - b) * (s_ - c), 1e-30))
    return 2 * (area / s_) / (a * b * c / (4 * area))


def h_mesh(ax):
    """The mesher failed on the whole part: cut it into blocks by hand, mesh them one by one."""
    BP_, BT_ = G["BP"], G["BT"]
    cad_part(ax, handles=False)
    done = np.isin(TRI_PART, [NAMES.index("ears"), NAMES.index("head"), NAMES.index("tail")])  # blocks meshed so far
    mesh(ax, BP_, BT_, skip=np.where(~done)[0])
    # the partition runs along the CAD model's own part curves, inked over the grey
    for k in NAMES:
        for r_ in PART[k]:
            poly(ax, r_, fc="none", ec=INK, lw=0.6, z=6)
    cursor(ax, 0.66, 0.42, 0.17)


def h_pick(ax):
    cad_part(ax, handles=False)
    for k in BC_COL:                        # CAD faces, clicked one at a time
        for r_ in PART[k]:
            poly(ax, r_, fc=BC_COL[k], ec="white", lw=0.35, z=4)
    cursor(ax, 0.33, 0.83, 0.17)
    book(ax, SIL[:, 0].max() + 0.20, 0.72, 0.17)       # the values: from references and handbooks


def book(ax, x, y, r):
    """An open handbook: two pages, a spine, lines of text."""
    for sgn in (-1, 1):
        pg = [(x, y - 0.55 * r), (x + sgn * r, y - 0.42 * r), (x + sgn * r, y + 0.62 * r), (x, y + 0.50 * r)]
        ax.add_patch(Polygon(pg, closed=True, fc="white", ec=INK, lw=0.5, joinstyle="round", zorder=7))
        for k in range(3):
            yy = y + 0.25 * r - k * 0.24 * r
            ax.plot([x + sgn * 0.18 * r, x + sgn * 0.80 * r], [yy, yy + 0.07 * r], color=GREY, lw=0.4,
                    zorder=8)


def h_mat(ax):
    w_, h_ = ax.get_position().width * FIG_W, ax.get_position().height * FIG_H
    W_ = w_ / h_
    ax.set_xlim(0, W_); ax.set_ylim(0, 1)
    x0, x1 = W_ / 2 - 0.62, W_ / 2 + 0.55
    ax.add_patch(FancyBboxPatch((x0, 0.04), x1 - x0, 0.92, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white", ec=LINE, lw=0.6, zorder=2))
    ax.plot([x0, x1], [0.80, 0.80], color=LINE, lw=0.6, zorder=3)
    for xx in (x0 + 0.08, x0 + 0.15, x0 + 0.22):
        ax.add_patch(Circle((xx, 0.88), 0.022, fc=LINE, ec="none", zorder=3))
    for i, (k, val) in enumerate((("ears", "TPU"), ("body", "PLA"))):
        yy = 0.58 - i * 0.26
        ax.add_patch(Rectangle((x0 + 0.08, yy - 0.07), 0.14, 0.14, fc=MAT[k], ec=LINE, lw=0.4,
                               zorder=3))
        bx0, bx1 = x0 + 0.30, x1 - 0.06
        fh = (cap_height(PIC_FS) + 2 * FIELD_PAD) / h_          # the field: its capitals plus FIELD_PAD
        ax.add_patch(Rectangle((bx0, yy - fh / 2), bx1 - bx0, fh, fc="white", ec=LINE,
                               lw=0.5, zorder=3))
        ptext_mid(ax, bx0 + 0.04, yy, val, 1.0 / h_, ha="left")
        ax.plot([bx1 - 0.11, bx1 - 0.075, bx1 - 0.04], [yy + 0.02, yy - 0.02, yy + 0.02],
                color=GREY, lw=0.5, zorder=4)


REDO_AT = (SIL[:, 0].max() + 0.22, 0.82)    # the redo mark: one place in every Design cell


def h_redesign(ax):
    """The new shape, partitioned and meshed again by hand (the supports stay picked)."""
    poly(ax, SIL, fc="none", ec=LINE, lw=0.5, z=2, ls=(0, (2, 1.4)))
    mesh(ax, G["NP"], G["NT"])
    poly(ax, G["new_ext"], G["new_holes"], fc="none", ec=INK, lw=0.5, z=4)
    redo_mark(ax, *REDO_AT, 0.11, HUMAN)


# --- row two: the AI agent ----------------------------------------------------------
def clipped(ax, p0, p1, body, **kw):
    """The straight line p0-p1, kept where it is inside the outline body."""
    from matplotlib.path import Path as P_
    t = np.linspace(0, 1, 120)[:, None]
    q = np.asarray(p0) * (1 - t) + np.asarray(p1) * t
    q = np.where(P_(body).contains_points(q)[:, None], q, np.nan)
    ax.plot(q[:, 0], q[:, 1], zorder=6, **kw)


TAIL_NOTCH = (0.923, 0.266)        # where the back meets the tail (ch_prep's outline)


def gen_shape(ax, outline, ec, shape=None):
    """The generator's output as it looks: a closed body (no element edges drawn), an
    island off the back, and a self-intersection: the body's surface runs down from the
    notch between the back and the tail, and a small piece of the body crosses it into
    the tail (shaded)."""
    body = SIL if shape is None else shape
    poly(ax, body, *([] if shape is None else [G["new_holes"]]), fc=CLAYFILL, ec="none", z=3)
    poly(ax, body, fc="none", ec=outline, lw=0.6, z=5)
    # the body's surface runs on down from the notch between the back and the tail,
    # halfway into it (the segment), and a sliver of the body pokes through it into the
    # tail (the shaded crescent, as long as the segment)
    p0, p1 = np.array(TAIL_NOTCH), np.array(TAIL_NOTCH) + (0.006, -0.10)
    t = np.linspace(0, 1, 24)
    n = np.array([p1[1] - p0[1], -(p1[0] - p0[0])]) / np.linalg.norm(p1 - p0)   # into the tail
    arc = p0 + np.outer(t, p1 - p0) + np.outer(0.062 * np.sin(np.pi * t), -n if n[0] < 0 else n)
    ax.add_patch(Polygon(np.vstack([arc, [p0]]), closed=True, fc="#80868f", ec="none", zorder=6))
    ax.plot(arc[:, 0], arc[:, 1], color=ec, lw=0.4, zorder=6)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ec, lw=0.6, solid_capstyle="round", zorder=7)
    poly(ax, G["island"], fc=CLAYFILL, ec=ec, lw=0.6, z=5)


def a_shape(ax):
    """The agent and MechFoundry both start from this one shape."""
    gen_shape(ax, INK, INK)
    # each defect named where it sits: the crescent under the tail, the island above it
    ptext(ax, 0.66, -0.13, DEFECT[0], color=GREY, va="top")
    ax.plot([0.88, 0.96], [-0.11, 0.17], color=GREY, lw=0.4, zorder=6)
    ix, iy = G["island"][:, 0].mean(), G["island"][:, 1].max()
    # the two lines of "Isolated components" one label pitch apart at its own size (the
    # pitch of the cell labels: the font's line box plus LEAD_PT), not a typed data step
    pitch = (line_box(PIC_FS) + LEAD_PT / 72.0) * ax._sc
    ptext(ax, 0.98, 0.99 + pitch, DEFECT[1], color=GREY, va="bottom")
    ptext(ax, 0.98, 0.99, DEFECT[2], color=GREY, va="bottom")
    # every grey note in this cell is set at ONE size (the user, 2026-10-17)
    assert len({t.get_fontsize() for t in ax.texts}) == 1, "the Shape-generation notes differ in size"
    ax.plot([ix - 0.03, ix], [0.97, iy + 0.02], color=GREY, lw=0.4, zorder=6)


def a_mesh(ax):
    gen_shape(ax, INK, FAIL)
    cross(ax, 1.30, 0.20, 0.09)             # the meshing algorithm stops on it


def a_pick(ax):
    """A coded mask: a function returning True per node (here: below 5 % of the height).
    It grabs a band of nodes, not the faces the rabbit stands on, and the ears are a guess."""
    BP_ = G["BP"]
    mesh(ax, BP_, G["BT"])                  # the surface mesh's own nodes
    poly(ax, SIL, fc="none", ec=LINE, lw=0.5, z=4)
    on = BP_[:, 1] < 0.05
    ax.scatter(BP_[~on, 0], BP_[~on, 1], s=0.9, c=GREY, lw=0, zorder=4)
    ax.scatter(BP_[on, 0], BP_[on, 1], s=2.2, c=DIR, lw=0, zorder=5)
    xr = SIL[:, 0].max()
    # the chip is sized by its text (FIELD_PAD round the capitals), so it follows the type size
    sc = ax._sc
    cw = measure(MASK, PIC_FS) * sc + 2 * FIELD_PAD * sc
    chh = cap_height(PIC_FS) * sc + 2 * FIELD_PAD * sc
    cx0, cy = xr - 0.10, 0.58
    ax.add_patch(FancyBboxPatch((cx0, cy - chh / 2), cw, chh, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=DIR, lw=0.6, zorder=6))
    ptext_mid(ax, cx0 + cw / 2, cy, MASK, sc, color=INK)
    arrow([(xr + 0.02, cy - chh / 2), (0.80, 0.075)], colour=DIR, ax=ax, scale=sc, lw=0.7)
    ptext(ax, 0.30, 0.90, "?", fs=MARK_FS, color=FAIL, weight="bold")


def a_mat(ax):
    poly(ax, SIL, fc=S.e_rgb(0.45), ec=INK, lw=0.5)
    ptext(ax, 0.62, 0.36, "?", fs=MARK_FS, color=FAIL, weight="bold")


def a_redesign(ax):
    """The redesigned body meshed again: the thinned body meets the kept ears and feet at
    sharp corners, and the slivers there (lowest radius ratio, flagged) stop the mesher."""
    NP_, NT_ = G["NP"], G["NT"]
    poly(ax, SIL, fc="none", ec=LINE, lw=0.5, z=2, ls=(0, (2, 1.4)))
    mesh(ax, NP_, NT_)
    poly(ax, G["new_ext"], G["new_holes"], fc="none", ec=INK, lw=0.5, z=4)
    # the elements below radius ratio 0.3 all sit where the thinned body meets the kept ears;
    # one ring around that cluster flags the site without hiding the slivers under marks
    q = tri_quality(NP_, NT_)
    bad = np.flatnonzero(q < 0.3)
    for i in bad:
        ax.add_patch(Polygon(NP_[NT_[i]], closed=True, fc="#f6d5d1", ec=FAIL, lw=0.5, zorder=5))
    pts = NP_[NT_[bad]].reshape(-1, 2)
    c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    r = np.hypot(*(pts - c).T).max() + 0.03
    ax.add_patch(Circle(c, r, fc="none", ec=FAIL, lw=0.6, zorder=6))
    redo_mark(ax, *REDO_AT, 0.11, FAIL)
    cross(ax, REDO_AT[0], REDO_AT[1] - 0.30, 0.09)


# --- row three: MechFoundry ----------------------------------------------------------
def m_shape(ax):
    """Either input: a CAD part (left) or a generated mesh (right)."""
    w_, h_ = ax.get_position().width * FIG_W, ax.get_position().height * FIG_H
    W_ = w_ / h_ * 1.25
    ax.set_xlim(0, W_); ax.set_ylim(-0.12, 1.13); ax.set_aspect("equal")
    for x_off, cad in ((0.02, True), (W_ / 2 + 0.06, False)):
        s_ = 0.52 * (W_ / 2 - 0.10) / SIL[:, 0].max() / 0.52
        P_ = SIL * s_ + (x_off, 0.18)
        if cad:
            poly(ax, P_, fc=CADFILL, ec=INK, lw=0.5, z=3)
            for p_ in P_[::9]:
                ax.add_patch(Rectangle(p_ - 0.02, 0.04, 0.04, fc="white", ec=INK, lw=0.35, zorder=5))
        else:
            poly(ax, P_, fc=CLAYFILL, ec=INK, lw=0.5, z=3)


def m_mesh(ax):
    Q = G["quad"]
    sq = [[(x, y), (x + h, y), (x + h, y + h), (x, y + h)] for x, y, h, _ in Q]
    inside = Q[:, 3] < 0                    # the sign of the distance only: no ramp to explain
    ax.add_collection(PolyCollection(sq, fc=np.where(inside[:, None], np.array([[0.70, 0.79, 0.90]]),
                                                     np.array([[1.0, 1.0, 1.0]])),
                                     ec=LINE, lw=0.15, zorder=3))
    poly(ax, SIL, fc="none", ec=INK, lw=0.6, z=4)
    # refined round the island, but no island: it is not part of the closed body
    poly(ax, G["island"], fc="none", ec=GREY, lw=0.4, z=5, ls=(0, (1.5, 1.2)))


def m_pick(ax):
    parts(ax, lambda k: BC_COL.get(k, TINT[k]))
    press(ax, ax._sc)
    pins(ax, ax._sc)


def m_mat(ax):
    parts(ax, lambda k: MAT["ears" if k == "ears" else "body"])


def m_redesign(ax):
    poly(ax, SIL, fc="none", ec=LINE, lw=0.5, z=2, ls=(0, (2, 1.4)))
    poly(ax, G["new_ext"], G["new_holes"], fc=CLAYFILL, ec=INK, lw=0.5, z=3)
    for k in ("feet", "ears"):
        for r_ in PART[k]:
            poly(ax, r_, fc=BC_COL[k], ec="white", lw=0.3, z=4)
    sc = ax._sc
    for (x, y, dx, dy) in ((0.68, 0.80, 0.0, -1.0), (1.06, 0.50, -1.0, -0.25)):
        d = np.array([dx, dy]) / np.hypot(dx, dy)          # the surface moving in: magenta, as dJ/ds
        arrow([(x, y), (x + d[0] * 0.17, y + d[1] * 0.17)], colour=BACK, ax=ax, scale=sc, lw=0.9)
    hx, hy, hr = G["hole"]
    t = np.linspace(0, 2 * np.pi, 40)
    ax.plot(hx + (hr + 0.035) * np.cos(t), hy + (hr + 0.035) * np.sin(t), color=BACK, lw=0.5,
            ls=(0, (1.5, 1.2)), zorder=5)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
# Read left to right: the prompt (the object's name only) forks into the three
# lanes; each lane makes the shape (the engineer in CAD; the AI agent and
# MechFoundry share ONE generated shape, one cell across both lanes), then
# meets four set-up steps, each challenge pointing down at MechFoundry's answer.
STEPS = ["Shape", "Mesh", "Support & load", "Materials", "Design"]
ROWS = {
    "human": {"name": LANES[0][0], "badge": LANES[0][1],
              "text": ["CAD modeling", "Partitioning and meshing", "Handbook and manual pick",
                       "Part-by-part setup", "Remesh"],
              "pics": [h_shape, h_mesh, h_pick, h_mat, h_redesign],
              "lims": [BN_LIM_R, BN_LIM_R, BN_LIM_R, None,
                       BN_LIM_R]},
    "agent": {"name": LANES[1][0], "badge": LANES[1][1],
              "text": [None, "Mesher failure", "Mask functions", "One material", "Repeated mesher failure"],
              "pics": [None, a_mesh, a_pick, a_mat, a_redesign],
              "lims": [None, BN_LIM_R, (-0.08, SIL[:, 0].max() + 0.56, -0.06, 1.06), BN_LIM, BN_LIM_R]},
    "mf": {"name": LANES[2][0], "badge": LANES[2][1],
           "text": [None, "Adaptive meshing", "Segmented part selection", "Material per part", "Auto remesh"],
           "pics": [None, m_mesh, m_pick, m_mat, m_redesign],
           "lims": [None, (-0.06, 1.26, -0.08, 1.24), press_lim(-0.08, SIL[:, 0].max() + 0.12, -0.14),
                    BN_LIM, (-0.08, SIL[:, 0].max() + 0.14, -0.06, 1.06)]},
}
SHARED_TEXT = "Shape generation"

row_y = {"human": yA, "agent": yB, "mf": yC}


def label(xm, y0, text):
    """A cell's label: one or two lines, centred in the two-line band at the cell's foot."""
    lines = wrap(text, TXT_FS, COL_W - 2 * BP)
    assert len(lines) <= 2, f"{text!r} needs {len(lines)} lines"
    for ln in lines:               # wrap() keeps an over-long single word whole: catch it
        assert measure(ln, TXT_FS) <= COL_W - 2 * BP, f"label line {ln!r} wider than its cell"
    yb = y0 + BP + 0.012 + (LINE_H / 2.0 if len(lines) == 1 else LINE_H)
    for j, ln in enumerate(lines):
        put(xm, yb - j * LINE_H, ln, fs=TXT_FS)
row_c = {k: v + CELL / 2.0 for k, v in row_y.items()}

# column heads
ch_h = cap_height(HEAD_FS, "bold")
y_hc = y_head + COL_HEAD / 2.0
heads = []
for i, s in enumerate(STEPS):
    heads.append(put(cx[i], y_hc - ch_h / 2.0, f"{i + 1}  {s.upper()}", fs=HEAD_FS,
                     color=GREY, weight="bold"))
# no head over the prompt (the user, 2026-10-17): the bubble speaks for itself, and the rule
# under the heads spans exactly the five step columns it heads
fig.add_artist(Rectangle((xc[0][0], y_head - 0.005), xc[4][1] - xc[0][0], 0.006, transform=IT,
                         fc=INK, ec="none", zorder=1))

# the prompt: a speech bubble with the object's name only, level with the middle lane
y_fork = row_c["agent"]
b_h = 0.30
block(x_card0, y_fork - b_h / 2.0, CARD_W, b_h, S.CARD, radius=0.08)
fig.add_artist(Polygon([(x_card0 + 0.14, y_fork - b_h / 2.0 + 0.01), (x_card0 + 0.26, y_fork - b_h / 2.0 + 0.01),
                        (x_card0 + 0.10, y_fork - b_h / 2.0 - 0.09)], closed=True, transform=IT,
                       fc=S.CARD, ec="none", zorder=0.5))
assert measure(PROMPT, PROMPT_FS, style="italic") <= CARD_W - 2 * BUBBLE_PAD + 1e-3, "the prompt leaves its bubble"
put((x_card0 + x_card1) / 2.0, y_fork - cap_height(PROMPT_FS) / 2.0, PROMPT, fs=PROMPT_FS, style="italic")

# the fork into the three lanes
x_f = x_card1 + 0.07
for key in ROWS:
    pts = ([(x_card1, y_fork), (x_hdr0, y_fork)] if key == "agent" else
           [(x_card1, y_fork), (x_f, y_fork), (x_f, row_c[key]), (x_hdr0, row_c[key])])
    arrow(pts, colour=AGENT if key == "mf" else ARROW)

# the one generated shape, shared by the agent and MechFoundry
block(xc[0][0], yC, COL_W, yB + CELL - yC, ROWFILL["mf"])
g_h = 1.05
a = pic_axes(0, (yC + yB + CELL) / 2.0 - g_h / 2.0 + 0.04, COL_W - 2 * BP, g_h, BN_LIM_R)
a_shape(a)
label(cx[0], yC, SHARED_TEXT)

for key, row in ROWS.items():
    y0 = row_y[key]
    h = CELL
    hx0, hx1 = x_hdr0, x_hdr1
    block(hx0, y0, HDR_W, h, HEADFILL[key], ec=AGENT if key == "mf" else "none",
          lw=0.6 if key == "mf" else 0.0)
    side = 0.15
    ytop = y0 + h - 0.14
    colour = {"human": HUMAN, "agent": HUMAN, "mf": AGENT}[key]
    if key == "human":
        person_mark(hx0 + 0.07, ytop, side, colour)
    elif key == "agent":
        chat_mark(hx0 + 0.07, ytop, side, colour)
    else:
        robot_mark(hx0 + 0.07, ytop, side, colour)
        # the person's feet on the robot's base line (robot body from -0.72, person from -0.98)
        person_mark(hx0 + 0.07 + side + 0.03, ytop + (0.98 - 0.72) * side / 2.0, side, "#cfd2d8")
    y_name = ytop - 0.21
    put(hx0 + 0.07, y_name, row["name"], fs=ROW_FS, ha="left", color=colour, weight="bold")
    assert hx0 + 0.07 + measure(row["name"], ROW_FS, "bold") <= hx1 - 0.03, f"{row['name']} leaves its box"
    yb_ = y0 + 0.11
    {"human": clock_mark, "agent": fail_mark, "mf": check_mark}[key](hx0 + 0.07, yb_, 0.11)
    assert hx0 + HDR_IN + HDR_BADGE + measure(row["badge"], SUB_FS) <= hx1 - HDR_OUT + 1e-3, f"{row['badge']!r} leaves its box"
    put(hx0 + 0.07 + 0.15, yb_ - cap_height(SUB_FS) / 2.0, row["badge"], fs=SUB_FS,
        ha="left", color={"human": HUMAN, "agent": HUMAN, "mf": AGENT}[key])
    for i in range(5):
        py0 = y0 + h - BP - PIC_H
        if row["pics"][i] is not None:      # (the shared shape is drawn above)
            fill = ROWFILL[key]
            block(xc[i][0], y0, COL_W, h, fill)
            lim = row["lims"][i]
            if lim is None:
                a = fig.add_axes([fx(xc[i][0] + BP), fy(py0), fx(COL_W - 2 * BP), fy(PIC_H)])
                a.set_axis_off(); a.patch.set_alpha(0.0); a.set_zorder(3)
                a._sc = 1.0
                PICS.append((a, xc[i][0] + BP, xc[i][1] - BP))
            else:
                a = pic_axes(i, py0, COL_W - 2 * BP, PIC_H, lim)
            row["pics"][i](a)
            label(cx[i], y0, row["text"][i])
        if i < 4:                           # the lane runs on, step to step, level with the boxes' centres
            arrow([(xc[i][1], row_c[key]), (xc[i + 1][0], row_c[key])], colour=AGENT if key == "mf" else ARROW)
    # the lane's header feeds its first step (the agent and MechFoundry both into the shared shape)
    arrow([(hx1, row_c[key]), (xc[0][0], row_c[key])], colour=AGENT if key == "mf" else ARROW)

# ---------------------------------------------------------------------------
# Centring, by one rule for every cell (the user, 2026-10-17: the CAD, partition and
# part-by-part pictograms sat left of centre).  Each frame (BN_LIM_R & co.) fixes a drawing's
# SCALE and its height, so the bunnies of a row stand on one baseline at one size; but a
# frame's x-centre is the bunny's plus room for whatever sits beside it, and that room
# differs per cell (a dimension line, a book, a mask chip, nothing).  So the frame is kept
# and the drawing slid sideways: its VISUAL bounding box (every patch, line, collection and
# text of the axes, cursors, dimension lines, icons and labels included) is centred in the
# cell's picture area, the cell less its padding BP.  Vertically the frames stay as they
# are (a common baseline per row; the load arrows' cell was solved to fit its top margin).
# Nothing in a pictogram is clipped, so the box is what the reader sees.
# ---------------------------------------------------------------------------
def visual_box(ax, rend):
    """The union of every drawn child's display extent (pixels), clipping off."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    boxes = []
    for c in ax.get_children():
        if c is ax.patch or not c.get_visible() or not isinstance(c, (Patch, Line2D, Collection, Text)):
            continue
        if isinstance(c, Text) and not c.get_text():
            continue
        if c in ax.spines.values():
            continue
        c.set_clip_on(False)
        e = c.get_window_extent(renderer=rend)
        if np.isfinite([e.x0, e.x1, e.y0, e.y1]).all() and e.width + e.height > 0:
            boxes.append(e)
    return (min(b.x0 for b in boxes), max(b.x1 for b in boxes),
            min(b.y0 for b in boxes), max(b.y1 for b in boxes))


CENTRE_TOL = 0.002                 # (in) what the centring must reach, a fifth of a 0.5 pt line
fig.canvas.draw()
rend = fig.canvas.get_renderer()
slid = []
for a_, x0_, x1_ in PICS:
    slid.append(0.0)
    for _ in range(2):             # extents of text and line caps do not scale with the shift; twice settles it
        bx0, bx1, _, _ = visual_box(a_, rend)
        off = (x0_ + x1_) / 2.0 - (bx0 + bx1) / 2.0 / fig.dpi          # inches to slide right
        w_in = a_.get_position().width * FIG_W
        per_in = (a_.get_xlim()[1] - a_.get_xlim()[0]) / w_in
        a_.set_xlim(*(np.asarray(a_.get_xlim()) - off * per_in))
        slid[-1] += off
        fig.canvas.draw()
    bx0, bx1, by0, by1 = (v / fig.dpi for v in visual_box(a_, rend))
    assert abs((bx0 + bx1) / 2.0 - (x0_ + x1_) / 2.0) <= CENTRE_TOL, "a pictogram is not centred"
    assert bx0 >= x0_ - BP + 0.01 and bx1 <= x1_ + BP - 0.01, \
        f"a pictogram ({bx1 - bx0:.3f} in) is wider than its cell allows ({x1_ - x0_ + 2 * BP - 0.02:.3f} in)"
print("centring: pictograms slid " + ", ".join(f"{v:+.3f}" for v in slid) + " in (row by row)")

# ---------------------------------------------------------------------------
# Checks: text registry, sizes, nothing off the page, no two texts touching
# ---------------------------------------------------------------------------
ALLOWED = {PROMPT, MASK, *DEFECT, "?", "TPU", "PLA", *wrap(SHARED_TEXT, TXT_FS, COL_W - 2 * BP)}
ALLOWED |= {f"{i + 1}  {s.upper()}" for i, s in enumerate(STEPS)}
for row in ROWS.values():
    ALLOWED |= {row["name"], row.get("badge", "")}
    for t in row["text"]:
        if t:
            ALLOWED |= set(wrap(t, TXT_FS, COL_W - 2 * BP))
# Measure at the PNG's resolution, not at the figure's default 100 dpi: Agg rounds each
# text's metrics to whole device pixels, so at 100 dpi every 6 pt line box comes out
# 9 px = 0.090 in tall (0.0775 in at 400-600 dpi), wider than the 0.088 in pitch, and the
# two lines of every wrapped label "touched" by 0.002 in that exist only in the
# rounding.  At PNG_DPI one pixel is 0.0025 in, a quarter of the real 0.010 in gap.
PNG_DPI = 400
dpi0 = fig.dpi
fig.set_dpi(PNG_DPI)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
W_px, H_px = fig.canvas.get_width_height()
ext, problems = [], []
for t in texts:
    s = t.get_text()
    assert s in ALLOWED, f"text not allowed on this figure: {s!r}"
    assert t.get_fontsize() in OV_FS.values(), f"{s!r} at {t.get_fontsize()} pt: not one of method_overview's sizes"
    e = t.get_window_extent(renderer=rend)
    assert e.x0 >= 0 and e.x1 <= W_px and e.y0 >= 0 and e.y1 <= H_px, f"{s!r} leaves the canvas"
    ext.append((s, e))
for i, (a_, ea) in enumerate(ext):
    for b_, eb in ext[i + 1:]:
        if ea.overlaps(eb):
            problems.append(f"texts {a_!r} and {b_!r} touch")
for h_ in heads:                   # a head may reach into the gaps, never past them
    e = h_.get_window_extent(renderer=rend)
    assert e.width / fig.dpi <= COL_W + GAP - 0.02, f"head {h_.get_text()!r} too wide"

fig.set_dpi(dpi0)                  # savefig sets its own dpi; leave the figure as it was built

for p_ in problems:
    print("CHECK:", p_)
assert not problems or os.environ.get("CH_DRAFT") == "1", "layout checks failed"
out = sys.argv[1] if len(sys.argv) > 1 else "challenge_overview"
fig.savefig(f"{out}.pdf", dpi=600)
fig.savefig(f"{out}.png", dpi=PNG_DPI)
print(f"wrote {out}.pdf / .png ({FIG_W:.3f} x {FIG_H:.3f} in)")
