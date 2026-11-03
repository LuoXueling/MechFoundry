"""Stage 3 (env mechfoundry-warp117): compose agentic_results.{pdf,png} + its token table.

    python ar_figure.py [WORK_DIR] [--out NAME]

The "ledger" hybrid (design_concepts/SPEC_ledger.md) with the user's changes on top:

  * a table: booktabs rules (0.6 / 0.35 / 0.6 pt ink), 0.25 pt hairlines between rows,
    small-capital column heads with Figure 1's badges (purple star = an LLM assists the
    step, green chip = the GPU solver) and light arrows between the four panel heads;
  * the INSTRUCTION column holds, per row, the prompt in Figure 1's prompt card (white,
    hairline, rounded, italic) and directly under it the agent's cost: input tokens with
    their cached and not-cached parts, output tokens, wall clock -- each as a micro bar on
    one scale shared by all rows, only the not-cached sliver saturated (user change 1);
  * no row letters (user change 4), and therefore no per-row marks on the E bar;
  * every key in a footer band under the table, under the column it explains (change 2);
  * each row is exactly as tall as the instruction column's content, and the four panels
    fill that height (change 3);
  * soft contact shadows composited from each render's alpha (SPEC, 75 % of concept A);
  * the displacement factor in the deformed panel's corner, one corner for every row,
    grey, clear of the object (the user's standing rule, not the spec's foot placement).

Coordinates are millimetres, y measured down from the top of the page.  Every number
is read from ``ar_stats.json`` (ar_prep.py).  The build fails on: a clipped render; a BC
region under 0.5 mm^2 in print or more than DeltaE00 10 from its chip; any text under
5 pt, off the page, overlapping other text or leaving its column; a prompt over three
lines (one over two makes every card in that figure three lines tall); a cost line that wraps; a bar longer than its track; a factor label on its
object or nearer the next row than its own; a contact shadow cut by its image edge or
reaching the row hairline; head text crowding the rules.  Under AR_FLOOR=warn the print
floors only warn, among them: a load arrow printing under 40 % of its length because the
camera looks along it, and a corner label (the body load's among them) on its object.
"""

import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ar_style as A  # noqa: E402
import ar_time  # noqa: E402

for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
    try:
        matplotlib.font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/msttcorefonts/{f}")
    except Exception:  # noqa: BLE001
        pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     # TrueType (Type 42), not Type 3: editable text for the art desk
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.transparent": False})

# -- palette (ov_style's, plus the spec's neutrals and the agent-cost family) ----------
INK, GREY = A.INK, A.GREY_TEXT
HAIR = "#d5d9df"             # row hairlines
CARD_STROKE = "#9aa1ab"      # Figure 1's prompt-card outline; also the head arrows
AGENT, GPU_GREEN = "#6b4fbb", "#3b7d23"
COST_TRACK, COST_CACHED, COST_UNCACHED, COST_NEUTRAL = "#eef0f3", "#cdc5ea", "#8a76c9", "#aab1bb"
# the time bar's groups (ar_time.GROUPS), in that order along the bar; designer spec
# SPEC_timebar v2: agent = the AGENT tag's purple a step deeper (clear of "not cached");
# solver = the FEM SOLVER tag's green itself; workflow steps = a light tint of that green (the
# other GPU tools: every free hue on the page is taken); overhead = a warm stone neutral above
# it, clear of the cool track.  L* 33 / 72 / 47 / 82 against the track's 95; adjacent segments
# differ by >= 26 L*; the closest pair under CVD (workflow / overhead, 14.1 dE00) never touches
TIME_COLORS = {"agent": "#553a9e", "workflow": "#8fbf6e", "solver": "#3b7d23", "overhead": "#d2cab9"}
TIME_KEY = {"agent": "Agent", "workflow": "Workflow steps", "solver": "Solver",
            "overhead": "Runner overhead"}   # every key label starts with a capital (user)

# -- grid (mm) ------------------------------------------------------------------------
DPI = 400
W = 180.0                    # Nature full width (double column)
ML = MR = 1.0
INS_W = 27.0                 # INSTRUCTION column: prompt card + cost block
X0 = ML + INS_W + 2.0        # first panel
GAP = 1.0                    # between panels
HEAD = 6.2                   # top rule .. head rule
HEAD_RULE = 5.6
TY = 3.6                     # column-head baseline
ROW_GAP = 1.6                # between row bands; the hairline sits in its middle
FOOT = 14.0                  # key band under the bottom rule
CARD_TOP = 1.4               # row top -> card top
CARD_PAD_H = 2.6             # card height = CARD_PAD_H + PROMPT_PITCH * lines
PROMPT_PITCH = 3.0
CARD_TEXT = 3.75             # card top -> first prompt baseline
COST_FIRST = 3.3             # card bottom -> first cost baseline
COST_PITCH = 4.0
BAR_OFF, BAR_H = 0.8, 0.8    # baseline -> bar top; bar height
ROW_PAD = 1.0                # last bar -> row bottom
PANEL_PAD_TOP = 0.6          # panel layout box inside the row band
SH_ROOM = 1.7                # ink bottom sits this far above the layout box bottom (shadow)
MARGIN = 0.03                # crop margin, fraction of the ink box's longer side

HEAD_FS, PROMPT_FS, META_FS, KEY_FS = 5.6, 7.0, 6.0, 5.6
TAG_FS, TAG_PAD, TAG_GAP, TAG_H = 5.0, 0.7, 1.0, 2.55   # the column-head tags (mm)
CAP_EM, PT_MM = 0.716, 25.4 / 72   # Arial cap height (em); mm per point
# AR_SEGMENTATION=1 adds a SEGMENTATION column before the supports and loads (the Opus
# per-category figures): the parts the BC selectors were resolved against
# (ar_prep.segmentation_of), in the PartField bunny's tints.  The page stays 180 mm wide,
# so the five panels are narrower; ar_prep's camera chooser reads this layout through
# print_layout() and predicts the checks for it.
SEGMENTATION = os.environ.get("AR_SEGMENTATION", "0") == "1"
PANELS = ("painted", "seg", "bc", "materials", "solve") if SEGMENTATION else A.PANELS
HEADS = {"painted": "GENERATED SHAPE", "seg": "SEGMENTATION", "bc": "SUPPORTS & LOADS",
         "materials": "YOUNG’S MODULUS", "solve": "DEFORMED SHAPE"}
COLUMNS = tuple(HEADS[p] for p in PANELS)
NCOL = len(PANELS)
# Heads too wide for one line between the head arrows (tag + title: "FEM SOLVER DEFORMED
# SHAPE" is 32 mm, a five-column panel 29 mm) are stacked: the tag over its title, the
# head band HEAD_STACKED tall instead of HEAD.
HEAD_STACKED, HEAD_RULE_STACKED, TY_STACKED = 7.9, 7.3, 6.15
TAG_TOP_STACKED = 1.3        # the stacked tag pill's top (the top rule is at 0.4)
# Five 29 mm panels leave no corner beside a centred body for a bottom label ("Body
# load 4.4 N" under the femur's condyles, the cane's tip; no camera of 48 freed it).  In
# that layout every row's ink ends above the bottom label band (showcase/sc_figure.py's
# BOT_RES rule), so the bottom corners are always free; the four-panel paper figure
# keeps SH_ROOM.  And the grid cell (bottom left) and the route (bottom right) of the
# generated-shape panel no longer fit side by side ("2 mm" ran into "SDXL → Hunyuan3D"):
# a route that would come within ROUTE_AIR of the cell label goes to the top right
# (route_order), in both layouts.
ROUTE_AIR = 1.5
# Where the grid-cell label goes: the generated-shape panel, beside the route; in the
# five-column layout the segmentation panel, which carries no other label (the grid lies
# under every panel of a row) -- side by side in 29 mm they ran into each other.
CELL_PANEL = "seg" if SEGMENTATION else "painted"


def route_order(route_w, cell_w, P):
    """The route label's corners, in preference order (right_band / ar_prep)."""
    crowded = cell_w and 2 * CORNER_PAD + cell_w + ROUTE_AIR + route_w > P
    return ("tr", "br") if crowded else ("br", "tr")


SEG_METHOD = {"partfield": "PartField", "shape_diameter": "shape diameter",
              "cad_faces": "CAD B-rep faces"}
MIN_REGION_MM2 = 0.5         # a BC region must print at least this much (review 2, N4)
SMALL_REGION_MM2 = 1.0       # ... unless even face-on it would print less than this:
MIN_SMALL_MM2 = 0.25         # then a visible dot is the requirement, and it is reported
SMALL_REGIONS = []
# AR_FLOOR=warn: a BC region or arrow under its print floor, or a region off its chip
# colour, is listed, not fatal -- a
# figure built for looking at a batch, where a hand-picked camera per small face (a
# tip, a sole, a bore) is not worth it.  The paper build keeps the default, strict.
FLOOR_WARN = os.environ.get("AR_FLOOR", "strict") == "warn"
FLOOR_MISSES = []


def _floor_check(ok, msg):
    if ok:
        return
    if not FLOOR_WARN:
        raise AssertionError(msg)
    FLOOR_MISSES.append(msg)
MIN_ARROW_MM2 = 0.3          # a load arrow must print at least this much
OUTLINE_W, OUTLINE_ALPHA = 0.22, 0.9           # mm, opacity of the undeformed outline
OUTLINE_RGB = (154, 161, 171)                  # #9aa1ab, the card stroke's grey
ARROW_THROUGH = 0.85         # a far-face load's arrows, laid over the panel
SEE_THROUGH = 0.45           # opacity of a far-side load drawn through the body
CORNER_PAD = 0.8             # factor label inset from the panel edge (mm)
LABEL_BASE = 2.9             # factor label baseline below the row top: the panel corner
BOTTOM_LABEL = 2.6           # height of a bottom-corner label band (mm)
INK_ROOM = BOTTOM_LABEL + 0.4 if SEGMENTATION else SH_ROOM   # ink -> layout bottom (above)
LIFTED_ROOM = BOTTOM_LABEL + 0.4   # a row no camera passes with INK_ROOM (ar_prep.choose_view)
CELL_ICON = 1.3              # the grid-cell square beside its length (mm)
GRID_RGB = (0.58, 0.64, 0.72)   # the UI grid's line colour (app.py placeGround)
CORNER_CLEAR = 1.0           # free space between the factor label and the object (mm)
# contact shadow: concept A's parameters at 75 % (SPEC_ledger "Contact shadow")
SH_CORE, SH_HALO, SH_FLOOR, SH_STRENGTH = 0.16, 0.07, 0.008, 0.75   # (retired: the user rejected the blob)
GROUND_FADE = 3.0            # mm: the ground layer fades out at its drawn image edges


# -- formatting ------------------------------------------------------------------------
def sig(v, k):
    if v == 0:
        return "0"
    d = k - 1 - int(math.floor(math.log10(abs(v))))
    return f"{round(v, d):.{max(d, 0)}f}"


def fmt_count(n, k=2):
    """Token counts at k (2) significant figures: 76, 5.2 k, 180 k, 5.1 M."""
    if n is None:
        return "n/a"
    n = float(n)
    if n < 1000:
        return f"{int(n)}"
    if n < 1e6:
        return f"{sig(n / 1e3, k)} k"
    return f"{sig(n / 1e6, k)} M"


def fmt_clock(s):
    if s is None:
        return "n/a"
    s = int(round(float(s)))
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60} min"
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def fmt_scale(x):
    """The UI's value at its 3 significant figures (toPrecision(3)), in Nature
    copy style: no separator up to four digits, a comma from five (×53,500)."""
    v = float(f"{x:.3g}")
    if v >= 10000:
        return f"{int(v):,}"
    return f"{int(v)}" if v >= 100 else f"{v:g}"


def eng(value, unit, sig_=2):
    """value in SI base units -> '12 GPa', '3.2 mm', '4.6 µm'."""
    if value == 0:
        return f"0 {unit}"
    prefixes = {12: "T", 9: "G", 6: "M", 3: "k", 0: "", -3: "m", -6: "µ", -9: "n"}
    e = int(math.floor(math.log10(abs(value)) / 3) * 3)
    e = max(min(e, 12), -9)
    v = value / 10 ** e
    v = round(v, sig_ - 1 - int(math.floor(math.log10(abs(v)))))
    txt = f"{v:.0f}" if v >= 10 ** (sig_ - 1) else f"{v:.{sig_}g}"
    return f"{txt} {prefixes[e]}{unit}"


def typographic(s):
    """Straight apostrophes -> U+2019, for display only (the record is untouched)."""
    return s.replace("'", "’")


# -- the page (mm, y down) -------------------------------------------------------------
class Page:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 25.4, h / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_axis_off()
        self.texts = []

    def text(self, x, y, s, fs, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("va", "baseline")
        t = self.ax.text(x, y, s, fontsize=fs, zorder=6, **kw)
        self.texts.append(t)
        return t

    def line(self, xs, ys, lw, color, z=3):
        self.ax.plot(xs, ys, lw=lw, color=color, solid_capstyle="butt", zorder=z)

    def img(self, a, x, y, w, h):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=1)

    def width(self, s, fs, **kw):
        t = self.ax.text(0, 0, s, fontsize=fs, **kw)
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        t.remove()
        return bb.width / DPI * 25.4

    def box(self, t):
        """A text's ink box in mm: x0, top, x1, bottom."""
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        x0, x1 = bb.x0 / DPI * 25.4, bb.x1 / DPI * 25.4
        return x0, self.h - bb.y1 / DPI * 25.4, x1, self.h - bb.y0 / DPI * 25.4


def sparkle(pg, x, y, r):
    """Figure 1's 'LLM / VLM subagent can assist' badge: purple disc, white star
    (outer 0.66 r, inner 0.20 r), as ov_figure draws it."""
    pg.ax.add_patch(Circle((x, y), r, color=AGENT, lw=0, zorder=6))
    t = np.linspace(0, 2 * np.pi, 9)[:-1]
    rr = np.where(np.arange(8) % 2 == 0, 0.66 * r, 0.20 * r)
    pg.ax.add_patch(Polygon(np.c_[x + rr * np.sin(t), y - rr * np.cos(t)], color="white",
                            lw=0, zorder=7))


def gpu_chip(pg, x, y, r):
    """Figure 1's 'differentiable GPU code' badge: green chip, frame, core and pins."""
    pg.ax.add_patch(Rectangle((x - 0.55 * r, y - 0.55 * r), 1.1 * r, 1.1 * r, fc="none",
                              ec=GPU_GREEN, lw=0.9, zorder=6))
    pg.ax.add_patch(Rectangle((x - 0.25 * r, y - 0.25 * r), 0.5 * r, 0.5 * r, fc=GPU_GREEN,
                              ec="none", zorder=6))
    for d in (-0.3, 0.0, 0.3):
        for (ax_, ay_, bx, by) in ((x + d * r, y - 0.55 * r, x + d * r, y - 0.9 * r),
                                   (x + d * r, y + 0.55 * r, x + d * r, y + 0.9 * r),
                                   (x - 0.55 * r, y + d * r, x - 0.9 * r, y + d * r),
                                   (x + 0.55 * r, y + d * r, x + 0.9 * r, y + d * r)):
            pg.ax.plot([ax_, bx], [ay_, by], color=GPU_GREEN, lw=0.6, zorder=6,
                       solid_capstyle="butt")


def arrow(pg, x0, y0, x1, y1):
    pg.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                    arrowstyle="-|>,head_length=0.9,head_width=0.42",
                                    mutation_scale=1 / 0.3528 * 0.9, lw=0.5, color=CARD_STROKE,
                                    shrinkA=0, shrinkB=0, zorder=4, joinstyle="miter"))


def hbar(pg, x, y, w, h, stops):
    cmap = LinearSegmentedColormap.from_list("k", stops)
    pg.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, extent=(x, x + w, y + h, y),
                 aspect="auto", zorder=4, interpolation="bilinear")
    pg.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK, lw=0.35, zorder=5))


MAX_PROMPT_LINES = 3


def balanced_wrap(pg, text, width, fs, **kw):
    """One line if it fits; else the fewest lines (two, then three) that fit, split
    where the longest line is shortest, so no line is a single orphaned word; greedy
    beyond three (an assertion then fails).  A long instruction ("a road roller with a
    steel drum and rubber tyres") takes three lines at the card's 7 pt rather than a
    smaller face: every card in a figure keeps one size."""
    if pg.width(text, fs, **kw) <= width:
        return [text]
    words = text.split(" ")
    wid = {}

    def w(i, j):                       # width of words[i:j] as one line
        if (i, j) not in wid:
            wid[(i, j)] = pg.width(" ".join(words[i:j]), fs, **kw)
        return wid[(i, j)]
    best = None
    for i in range(1, len(words)):
        wa, wb = w(0, i), w(i, len(words))
        if wa <= width and wb <= width and (best is None or max(wa, wb) < best[0]):
            best = (max(wa, wb), [" ".join(words[:i]), " ".join(words[i:])])
    if best:
        return best[1]
    for i in range(1, len(words) - 1):
        if w(0, i) > width:
            break
        for j in range(i + 1, len(words)):
            m = max(w(0, i), w(i, j), w(j, len(words)))
            if m <= width and (best is None or m < best[0]):
                best = (m, [" ".join(words[:i]), " ".join(words[i:j]), " ".join(words[j:])])
    if best:
        return best[1]
    lines, cur = [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if pg.width(t, fs, **kw) > width and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    return lines + [cur]


# -- panels ----------------------------------------------------------------------------
def load_rgba(path):
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert("RGBA")).astype(float)


def ink_box(rgba):
    ys, xs = np.nonzero(rgba[..., 3] > 25)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def row_crop(boxes, P, H, ext, room=None, draw_scale=1.0):
    """The render-pixel window drawn for a row: the union ink box fitted into the
    panel's layout box (P x H mm) with MARGIN at the top and sides and INK_ROOM under
    the ink (room for the contact shadow), then extended ``ext`` mm further down into
    the row gap.  ``room`` overrides INK_ROOM (a row whose view records ``ink_room``).
    ``draw_scale`` f (the row's per-object size override, ar_runs) draws the ink f x
    the size that fit gives: px/mm / f, shrunk about the ink box's CENTRE as the rule
    places it (horizontal and vertical), so the body does not sink in its row (user,
    2026-10-19: "smaller but not moved downwards"); it frees the bottom band for the
    corner labels as much as the top.  f = 1 is the rule exactly.
    Returns (x0, y0, x1, y1_layout, y1_drawn) in render px and px/mm."""
    room = INK_ROOM if room is None else room
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    pad = MARGIN * max(w, h)
    # px per mm so the padded ink fits P wide and (H - SH_ROOM) tall below a top pad
    s1 = max((w + 2 * pad) / P, (h + pad) / (H - room))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    # the rule's placement: the ink bottom ``room`` above the layout box bottom; the
    # ink centre then sits c_mm below the layout box top, and stays there at any f
    c_mm = (cy - (y1 + room * s1 - H * s1)) / s1
    s = s1 / draw_scale
    left = cx - P * s / 2
    top = cy - c_mm * s
    bottom_layout = top + H * s
    return left, top, left + P * s, bottom_layout, bottom_layout + ext * s, s


def cut_rgba(img, x0, y0, x1, y1):
    """A window of an RGBA render; outside the render is transparent."""
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    out = np.zeros((y1 - y0, x1 - x0, 4))
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def contact_shadow(alpha, px_per_mm):
    """SPEC_ledger's contact shadow, in the window's pixels, as an opacity field.
    Footprint = ink columns in the lowest 5 % of the object's height."""
    a = alpha > 0.5
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return np.zeros(alpha.shape)
    ybot, ytop = ys.max(), ys.min()
    band = a[max(ybot - int(0.05 * (ybot - ytop)), 0):ybot + 1]
    cols = np.nonzero(band.any(axis=0))[0]
    wf, wo = cols.max() - cols.min(), xs.max() - xs.min()
    cx = 0.5 * (cols.min() + cols.max())
    rx = 0.55 * wf + 0.06 * wo
    ry = max(0.16 * rx, 0.40 * px_per_mm)
    yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]]
    core = (((xx - cx) / rx) ** 2 + ((yy - (ybot - 0.3 * ry)) / ry) ** 2 <= 1).astype(float)
    halo = (((xx - cx) / (1.35 * rx)) ** 2 + ((yy - (ybot - 0.2 * ry)) / (1.6 * ry)) ** 2
            <= 1).astype(float)
    s = (SH_CORE * gaussian_filter(core, 0.45 * ry + 0.20 * px_per_mm)
         + SH_HALO * gaussian_filter(halo, 1.2 * ry + 0.40 * px_per_mm))
    return SH_STRENGTH * np.clip(s - SH_FLOOR, 0, None)


def compose(win, px_per_mm, shadow=True):
    """Object over white, with its contact shadow multiplied into the white."""
    a = win[..., 3:4] / 255.0
    base = np.full(win.shape[:2] + (3,), 255.0)
    sh = contact_shadow(win[..., 3] / 255.0, px_per_mm) if shadow else np.zeros(win.shape[:2])
    base *= (1 - sh)[..., None]
    out = win[..., :3] * a + base * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8), sh


def over(top_rgba, base_rgb):
    """``top`` (straight-alpha RGBA) over ``base`` (RGB; None = the white page)."""
    a = top_rgba[..., 3:4] / 255.0
    base = np.full(top_rgba.shape[:2] + (3,), 255.0) if base_rgb is None else base_rgb.astype(float)
    return np.clip(top_rgba[..., :3] * a + base * (1 - a), 0, 255).astype(np.uint8)


def fmt_cell(rec):
    """The grid cell's side (ar_prep.ui_grid: a 1/2/5 x 10^k length in the session's
    unit), in metres, as a clean value: 0.5 m, 1 m; under 0.1 m in mm: 50 mm."""
    g = (rec.get("ground") or {}).get("grid")
    if not g:
        return None
    v = g["cell"] * rec.get("m_per_length_unit", 1.0)
    if v >= 0.1:
        return f"{float(f'{v:.3g}'):g} m"
    return f"{float(f'{v * 1000:.3g}'):g} mm"


def outline_under(base_rgb, undeformed_win, px_per_mm):
    """The undeformed silhouette (the same camera's undeformed panel alpha) as a
    thin light line, drawn on the page UNDER the deformed object, so even a
    modest magnification reads as motion (FEM convention; Figure 1 has none)."""
    from scipy.ndimage import binary_dilation
    m = undeformed_win[..., 3] > 128
    r = max(1, int(round(OUTLINE_W * px_per_mm)))
    edge = binary_dilation(m, iterations=r) & ~m
    out = base_rgb.astype(float)
    out[edge] = out[edge] * (1 - OUTLINE_ALPHA) + np.array(OUTLINE_RGB) * OUTLINE_ALPHA
    return out.astype(np.uint8)


def colour_check_rgb(rgba):
    """Opaque pixels' colour, others white (for the BC colour gate)."""
    a = rgba[..., 3:4] / 255.0
    return np.clip(rgba[..., :3] * a + 255 * (1 - a), 0, 255)


# -- data ------------------------------------------------------------------------------
def cost_of(rec):
    m = rec["meta"]
    plain, read, written = (m.get("input_tokens"), m.get("cache_read_input_tokens"),
                            m.get("cache_creation_input_tokens"))
    total = plain + read + written if None not in (plain, read, written) else None
    return {"input": total, "cached": read if total is not None else None,
            "output": m.get("output_tokens"), "time": m.get("wall_clock_s"),
            "status": m.get("status")}


TABLE_NOTE = ("Token use per run of agentic_results (sources: agentic_results/ar_stats.json). "
              "Cached = input tokens read from the prompt cache (cache_read_input_tokens); "
              "tokens written to the cache (cache_creation_input_tokens) were processed in full "
              "and count as not cached, with the plain uncached input (input_tokens).  "
              "Input total = cached + not cached.")


def write_table(runs, out, maxima):
    cols = ["row", "instruction", "session_id", "model", "input_tokens_total",
            "input_tokens_cached", "input_tokens_not_cached",
            "of_which_written_to_cache", "of_which_plain_uncached", "output_tokens",
            "wall_clock_s", "status", "displacement_scale"]
    rows = []
    for k, rec in enumerate(runs):
        m = rec["meta"]
        plain, read, written = (m.get("input_tokens"), m.get("cache_read_input_tokens"),
                                m.get("cache_creation_input_tokens"))
        ok = None not in (plain, read, written)
        rows.append({"row": k + 1, "instruction": m.get("instruction"),
                     "session_id": m.get("session_id"), "model": m.get("model"),
                     "input_tokens_total": plain + read + written if ok else None,
                     "input_tokens_cached": read,
                     "input_tokens_not_cached": plain + written if ok else None,
                     "of_which_written_to_cache": written, "of_which_plain_uncached": plain,
                     "output_tokens": m.get("output_tokens"),
                     "wall_clock_s": m.get("wall_clock_s"), "status": m.get("status"),
                     "displacement_scale": (rec.get("scale") or {}).get("factor")})
    note = TABLE_NOTE + (f"  The figure's cost bars are relative to the largest run: "
                         f"{fmt_count(maxima['input'])} input tokens, "
                         f"{fmt_count(maxima['output'])} output tokens, "
                         f"{fmt_clock(maxima['time'])}.")
    with open(f"{out}_tokens.csv", "w", newline="") as fh:
        fh.write("# " + note + "\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(f"{out}_tokens.md", "w") as fh:
        fh.write("Supplementary table. " + note + "\n\n")
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join("" if r[c] is None else
                                      (f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]))
                                      for c in cols) + " |\n")


def write_time_table(runs, out):
    """``{out}_time.csv``: for every section of ar_time.SECTIONS, its bar group (the figure's
    time bar shows only the groups), what it holds, and the seconds of each object's run."""
    heads = [(r["meta"].get("instruction") or f"row {k + 1}") for k, r in enumerate(runs)]
    splits = [(r["meta"].get("time_split") or {}) for r in runs]
    groups = dict(ar_time.GROUPS)
    with open(f"{out}_time.csv", "w", newline="") as fh:
        fh.write("# Where each run's wall clock went (ar_time.py; sources: metrics.json, the Claude "
                 "transcript's timestamps, the runner's operator/driver_log.jsonl). The sections add "
                 "up to the wall clock; the figure's time bar shows the bar groups.\n")
        w = csv.writer(fh)
        w.writerow(["section", "bar group", "label", "meaning"] + [f"{h} (s)" for h in heads])
        for key, group, label, meaning in ar_time.SECTIONS:
            w.writerow([key, groups[group], label, meaning] +
                       [(s.get("sections") or {}).get(key, "") for s in splits])
        for g, glabel in ar_time.GROUPS:
            w.writerow([f"group:{g}", glabel, f"{glabel} (the sum of its sections)",
                        "the segment the figure's time bar shows"] +
                       [ar_time.groups_of(s.get("sections") or {}).get(g, "") if s else ""
                        for s in splits])
        w.writerow(["wall_clock", "", "Wall clock", "the run's wall clock (metrics.json): the sum "
                    "of every section"] +
                   [round(s["wall_clock_s"], 1) if s.get("wall_clock_s") is not None else ""
                    for s in splits])


def bc_region_check(k, rec, shaded_rgb, idpass, mm_per_px, hidden=None):
    """Per region, on the exact id pass: >= MIN_REGION_MM2 printed, and the median
    rendered colour within A.BC_MAX_DE00 of its chip (design review 2, M3 / N4).
    A load on a far face (ar_prep "seen_through") is checked on its own see-through
    layer instead: its printed area there must reach the same floor; its colour is
    the see-through tint by design, so the DeltaE gate does not apply to it."""
    from scipy.ndimage import binary_erosion
    from skimage.color import deltaE_ciede2000, rgb2lab
    bc = rec.get("bc")
    if not bc or shaded_rgb is None or idpass is None:
        return []
    ids = idpass.astype(int)
    surf = (ids[..., 1] == 255) & (ids[..., 2] == 0)
    # print scale (mm^2 of paper per unit^2 of body), from the whole body's printed
    # area against its projected area toward the camera
    body_px = (surf & (ids[..., 0] >= 0)).sum()
    scale2 = body_px * mm_per_px ** 2 / max(bc.get("surface_proj_area") or 0, 1e-30)
    report = []
    for j, g in enumerate(bc["regions"]):
        if not g["faces"]:
            continue
        if g.get("seen_through") or g.get("support_seen_through"):
            assert hidden is not None, f"row {k + 1}: {g['name']!r} has no see-through layer"
            hm = (hidden[..., 3] > 128) & (hidden[..., 0].round() == j + 1)
            area = binary_erosion(hm, iterations=1).sum() * mm_per_px ** 2
            _floor_check(area >= MIN_REGION_MM2,
                         f"row {k + 1}: far-side load {g['name']!r} prints {area:.2f} mm2 seen through")
            report.append((g["name"] + " (seen through)", round(area, 2), None))
            continue
        m = binary_erosion(surf & (ids[..., 0] == j + 1), iterations=1)
        area = m.sum() * mm_per_px ** 2
        # A region that would print under SMALL_REGION_MM2 even face-on is small at
        # this print scale, not out of view: it must still show MIN_SMALL_MM2 (a
        # visible dot), and the build names it.  Every other region keeps the
        # MIN_REGION_MM2 floor (design review 2, N4).
        face_on = (g.get("area") or 0) * scale2
        floor = MIN_REGION_MM2
        if bc.get("surface_proj_area") and face_on < SMALL_REGION_MM2:
            floor = MIN_SMALL_MM2
            SMALL_REGIONS.append(f"row {k + 1} {g['name']!r}: {area:.2f} mm2 printed "
                                 f"({face_on:.2f} mm2 even face-on)")
        _floor_check(area >= floor,
                     f"row {k + 1}: region {g['name']!r} shows {area:.2f} mm2 in print (< {floor}; "
                     f"face-on it would print {face_on:.2f} mm2)")
        if not m.any():
            report.append((g["name"], round(area, 2), None))
            continue
        chip = A.DIRICHLET if g["category"] == "dirichlet" else A.NEUMANN
        med = np.median(shaded_rgb[m], axis=0) / 255.0
        de = float(deltaE_ciede2000(rgb2lab(np.array([[chip]], float)),
                                    rgb2lab(np.array([[med]], float)))[0, 0])
        # Under AR_FLOOR=warn a colour miss is listed with the floor misses, not fatal:
        # a batch figure met one shaded flank at DeltaE00 10.1 against the 10.0 gate.
        _floor_check(de <= A.BC_MAX_DE00,
                     f"row {k + 1}: region {g['name']!r} renders DeltaE00 {de:.1f} from its chip")
        report.append((g["name"], round(area, 2), round(de, 1)))
    # load arrows: each along its region's traction AS THE SESSION'S case.json STATES
    # IT (re-read here, not trusted from the prep), and each visible in print
    case = {}
    cp = os.path.join(rec.get("session") or "", "case", "case.json")
    if os.path.exists(cp):
        case = {nb["name"]: nb.get("traction") for nb in
                (json.load(open(cp)).get("boundary_conditions") or {}).get("neumann") or []}
    for a_ in bc.get("arrows") or []:
        t_ = np.asarray(case.get(a_["region"]), float)
        ax = np.asarray(a_["tip"]) - np.asarray(a_["tail"])
        cosang = float(ax @ t_ / (np.linalg.norm(ax) * np.linalg.norm(t_)))
        assert cosang > 0.9999, f"row {k + 1}: arrow on {a_['region']!r} is {math.degrees(math.acos(min(cosang, 1))):.2f} deg off its load"
        layer = hidden if a_["seen_through"] else None
        if layer is not None:
            px = ((layer[..., 3] > 128) & (layer[..., 0].round() == a_["id"])).sum()
        else:
            px = (surf & (ids[..., 0] == a_["id"])).sum()
        area = px * mm_per_px ** 2
        _floor_check(area >= MIN_ARROW_MM2,
                     f"row {k + 1}: an arrow on {a_['region']!r} prints {area:.2f} mm2 (hidden?)")
        # an arrow seen near its own axis prints as a stub; recomputed here from its
        # ends and the row's camera rather than trusted from the prep
        pf = projected_len_fraction(ax, (rec.get("view") or {}).get("dir") or A.VIEW_DIR)
        _floor_check(pf >= MIN_ARROW_PROJ,
                     f"row {k + 1}: an arrow on {a_['region']!r} prints at {pf:.0%} of its length "
                     f"(< {MIN_ARROW_PROJ:.0%}: seen along its axis)")
        report.append((f"arrow {a_['id']} on {a_['region']}", round(area, 2), f"dir ok, {pf:.0%} long"))
    return report


def projected_len_fraction(axis, view_dir):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    d = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    return math.sqrt(max(0.0, 1.0 - float(a @ d) ** 2))


# -- the body load ---------------------------------------------------------------------
# A body load (the case's loads.acceleration: a weight) acts on the whole body, so it has
# no face to stand on.  It is printed as text, "Body load 2.0 N", in the supports-and-loads
# panel's lower-right corner (user, 2026-10-12: the hollow arrow beside the body was
# dropped), the force it applies in newtons (ar_prep.body_load_force).
MIN_ARROW_PROJ = 0.40        # a drawn arrow prints at least this share of its 3-D length


def fmt_force(f):
    """A force in newtons at 2 significant figures, kept whole from 100 N to 999 N
    (3 figures, '101 N' rather than '100 N'), in kN from 1000 N: '0.55 N', '2.0 N',
    '42 N', '296 N', '3.9 kN'."""
    if f >= 999.5:
        return f"{sig(f / 1e3, 2)} kN"
    if f >= 99.5:
        return f"{int(round(f))} N"
    return f"{sig(f, 2)} N"


def body_label(body):
    """The corner text: the force the body load applies, 'Body load 12 N' (user,
    2026-10-12).  It was the acceleration in g, but the runner scales gravity so the
    applied weight equals the REAL object's when the generated body is heavier or
    lighter than it, which printed '0.031 g' and '2.5 g' for ordinary self-weight."""
    f = body.get("weight_N")
    return f"Body load {fmt_force(f)}" if f else "Body load"


# -- the Young's-modulus bar's ticks ------------------------------------------------------
E_LABEL_AIR = 1.0            # mm of free space between two neighbouring E labels


def e_ticks(pg, lo, hi, bx, bw, right):
    """The E bar's ticks as (log10 E, x mm, label or '' for an unlabelled decade tick).

    Every decade inside the range gets a tick.  All of them are labelled when the labels
    fit with E_LABEL_AIR between neighbours and the last one ends before ``right`` (the
    next column): the paper figure's MPa-GPa range, where a label keeps its unit only
    where the next label's prefix differs ("10 100 1 10 100 GPa" read with the unit only
    last).  A wider range (the food runs reach down to ~10 kPa) labels every 2nd, 3rd,
    ... decade instead, each label with its unit ("10 kPa  1 MPa  100 MPa  10 GPa"), the
    other decades keeping a short unlabelled tick; of the offsets of one step, the one
    with the most labels and then the widest gaps wins."""
    x_of = lambda d: bx + (d - lo) / (hi - lo) * bw      # noqa: E731
    decades = [d for d in range(math.ceil(lo), math.floor(hi) + 1)]
    if not decades:
        return [(d, x_of(d), eng(10 ** d, "Pa")) for d in (lo, hi)]
    prefix = lambda d: eng(10 ** d, "Pa").split(" ")[1]   # noqa: E731

    def labels(ds, terse):
        out = []
        for j, d in enumerate(ds):
            whole = (not terse or j == len(ds) - 1 or prefix(d) != prefix(ds[j + 1]))
            out.append(eng(10 ** d, "Pa") if whole else eng(10 ** d, "Pa").split(" ")[0])
        return out

    def min_gap(ds, ls):
        """The narrowest free space between neighbouring labels (inf for one label),
        or None when a label runs past the next column."""
        boxes = []
        for d, s_ in zip(ds, ls):
            w_ = pg.width(s_, KEY_FS)
            boxes.append((x_of(d) - w_ / 2, x_of(d) + w_ / 2))
        if boxes[-1][1] > right - 0.5:
            return None
        return min((b[0] - a[1] for a, b in zip(boxes, boxes[1:])), default=math.inf)

    for step in range(1, len(decades) + 1):
        best = None
        for r in range(step if step > 1 else 1):
            ds = decades[r::step]
            ls = labels(ds, terse=step == 1)
            g = min_gap(ds, ls)
            if g is None or g < E_LABEL_AIR:
                continue
            key = (len(ds), g)
            if best is None or key > best[0]:
                best = (key, dict(zip(ds, ls)))
        if best:
            chosen = best[1]
            return [(d, x_of(d), chosen.get(d, "")) for d in decades]
    # not even one label fits (a range of a single decade label wider than the bar)
    return [(d, x_of(d), eng(10 ** d, "Pa") if j == 0 else "") for j, d in enumerate(decades)]


# -- the footer keys (shared with showcase/sc_figure.py) -----------------------------------
# Each draws one column's keys on ``pg`` with its first baseline at ``fy`` and returns
# the texts it placed (the compose checks them for overlap and the footer band).  A
# column is ``x0`` .. ``x0 + P``.
def cost_keys(pg, x, fy, with_time=True):
    """The cost bars' colours: Cached input, Not cached; then the time bar's groups in
    bar order on two lines (Agent, Workflow steps / Solver, Runner overhead).
    (all texts, the time-group texts: they may run into the empty footer under the
    next column)."""
    texts, time_texts = [], []
    x_ = x
    for colr, s_ in ((COST_CACHED, "Cached input"), (COST_UNCACHED, "Not cached")):
        pg.ax.add_patch(Rectangle((x_, fy - 1.25), 1.6, 1.0, fc=colr, ec="none", zorder=5))
        texts.append(pg.text(x_ + 2.2, fy, s_, KEY_FS, color=GREY))
        x_ += 2.2 + pg.width(s_, KEY_FS) + 2.4
    if with_time:
        # the time bar's groups, two lines in bar order (the scale note was dropped: user)
        # (two lines; the second may run into the empty footer under the next column: user)
        for yk, groups in ((fy + 3.0, ("agent", "workflow")), (fy + 6.0, ("solver", "overhead"))):
            x_ = x
            for g in groups:
                pg.ax.add_patch(Rectangle((x_, yk - 1.25), 1.6, 1.0, fc=TIME_COLORS[g], ec="none",
                                          zorder=5))
                t_ = pg.text(x_ + 2.2, yk, TIME_KEY[g], KEY_FS, color=GREY)
                texts.append(t_)
                time_texts.append(t_)
                x_ += 2.2 + pg.width(TIME_KEY[g], KEY_FS) + 2.4
    return texts, time_texts


def bc_keys(pg, x0, P, fy, has_arrows=True, seen_through=False):
    """Support, Load (swatch and arrow) centred under the column; "Load on a far face,
    seen through" on the next line when a panel has one."""
    texts = []
    # one "Load" key: the swatch and the load arrow together (user, 2026-10-12: the
    # separate "Load direction" key merged into it)
    load_glyph_w = 1.9 + (0.7 + 3.2 if has_arrows else 0.0)
    widths = [pg.width(s_, KEY_FS) for s_ in ("Support", "Load")]
    total = (1.9 + 0.8) + (load_glyph_w + 0.8) + sum(widths) + 3.0
    x = x0 + (P - total) / 2
    pg.ax.add_patch(Rectangle((x, fy - 1.6), 1.9, 1.9, fc=A.DIRICHLET, ec=INK, lw=0.35, zorder=5))
    texts.append(pg.text(x + 1.9 + 0.8, fy, "Support", KEY_FS))
    x += 1.9 + 0.8 + widths[0] + 3.0
    pg.ax.add_patch(Rectangle((x, fy - 1.6), 1.9, 1.9, fc=A.NEUMANN, ec=INK, lw=0.35, zorder=5))
    if has_arrows:
        # the 3-D arrows drawn flat: their colour (A.LOAD_ARROW) and ar_prep's
        # proportions -- head 2.5x the shaft wide, 1.16x its own width long, 36 % of
        # the length.  The FancyArrowPatch it replaces had a 0.32 mm line under a
        # 1.8 mm head (5.6x) in the region's lighter orange, unlike any arrow drawn.
        ln_, sw_ = 3.2, 0.46
        hw_, hl_ = 2.55 * sw_, 0.36 * 3.2
        x0_, y0_ = x + 2.6, fy - 0.65
        pg.ax.add_patch(Polygon([(x0_, y0_ - sw_ / 2), (x0_ + ln_ - hl_, y0_ - sw_ / 2),
                                 (x0_ + ln_ - hl_, y0_ - hw_ / 2), (x0_ + ln_, y0_),
                                 (x0_ + ln_ - hl_, y0_ + hw_ / 2), (x0_ + ln_ - hl_, y0_ + sw_ / 2),
                                 (x0_, y0_ + sw_ / 2)], closed=True, fc=A.LOAD_ARROW,
                                ec="none", zorder=5))
    texts.append(pg.text(x + load_glyph_w + 0.8, fy, "Load", KEY_FS))
    if seen_through:
        # the see-through tint as it prints over clay
        tint = np.array(A.CLAY) * (1 - SEE_THROUGH) + np.array(A.NEUMANN) * SEE_THROUGH
        s_ = "Load on a far face, seen through"
        x, yk = x0 + (P - (1.9 + 0.8 + pg.width(s_, KEY_FS))) / 2, fy + 3.0
        pg.ax.add_patch(Rectangle((x, yk - 1.6), 1.9, 1.9, fc=tint, ec=INK, lw=0.35, zorder=5))
        texts.append(pg.text(x + 1.9 + 0.8, yk, s_, KEY_FS))
    return texts


def e_key(pg, x0, P, fy, lo, hi, right, stops=None):
    """The E colour bar (log, decade ticks by ``e_ticks``, no label past ``right``) and
    "Log" at its right end.  ``stops``: the map's evenly spaced RGB stops; None is the
    shared "steel" map (A.E_RGB), every agentic-results figure's -- only the showcase
    passes its own."""
    texts = []
    bx, bw, by, bh = x0 + 3.5, P - 9.0, fy - 1.6, 1.0
    stops = A.E_RGB if stops is None else stops
    hbar(pg, bx, by, bw, bh, [(k / (len(stops) - 1), c) for k, c in enumerate(stops)])
    # light = soft, dark = stiff is carried by the map itself (the words were dropped: user)
    for d, xt, s_ in e_ticks(pg, lo, hi, bx, bw, right):
        pg.line([xt, xt], [by + bh, by + bh + (0.7 if s_ else 0.4)], 0.35, INK, z=5)
        if s_:
            texts.append(pg.text(xt, by + bh + 3.0, s_, KEY_FS, ha="center"))
    texts.append(pg.text(bx + bw + 0.8, by + bh, "Log", KEY_FS, color=GREY))
    return texts


def u_keys(pg, x0, P, fy, outline_rgb=OUTLINE_RGB):
    """The |u| ramp from 0 to max |u|, then "×N: Displacement magnified N times" and the
    undeformed outline (in ``outline_rgb``, the colour the figure draws it in)."""
    texts = []
    by, bh = fy - 1.6, 1.0
    ux, uw = x0 + 3.5, P - 7.0
    hbar(pg, ux, by, uw, bh, [(0.0, A.SOLVE_BODY), (A.SOLVE_STOPS[0], A.SOLVE_BODY),
                              (A.SOLVE_STOPS[1], A.SOLVE_MID), (1.0, A.SOLVE_TOP)])
    for t_, s_, ha in ((0.0, "0", "center"), (1.0, "max $|\\mathbf{u}|$", "center")):
        pg.line([ux + t_ * uw] * 2, [by + bh, by + bh + 0.7], 0.35, INK, z=5)
        texts.append(pg.text(ux + t_ * uw, by + bh + 3.0, s_, KEY_FS, ha=ha))
    lines = u_factor_lines(pg, P)
    s_ = "Undeformed shape"
    yu = by + bh + 9.0 + 3.0 * (len(lines) - 1)
    xk = x0 + (P - (2.2 + 0.8 + pg.width(s_, KEY_FS))) / 2
    pg.ax.add_patch(Rectangle((xk, yu - 1.5), 2.2, 1.6, fc="none",
                              ec=np.array(outline_rgb) / 255, lw=0.6, zorder=5))
    texts.append(pg.text(xk + 2.2 + 0.8, yu, s_, KEY_FS, color=GREY))
    for j, s_ in enumerate(lines):
        texts.append(pg.text(x0 + P / 2, by + bh + 6.0 + 3.0 * j, s_, KEY_FS, ha="center",
                             color=GREY))
    return texts


def u_factor_lines(pg, P):
    """The ×N key's definition: one line, or two when it is wider than the column
    (a five-column page's 29 mm panel)."""
    s_ = "×N:  Displacement magnified N times"
    return [s_] if pg.width(s_, KEY_FS) <= P - 0.4 else ["×N:  Displacement", "magnified N times"]


def seg_keys(pg, x0, P, fy, methods):
    """The segmentation column's key, centred under it: a strip of the bunny's eight
    part tints with "One colour per part", then which segmentation made the parts."""
    texts, tints = [], A.S.SEGMENT_TINTS
    sw, s_ = 0.9, "One colour per part"
    total = sw * len(tints) + 0.8 + pg.width(s_, KEY_FS)
    x = x0 + (P - total) / 2
    for j, c in enumerate(tints):
        pg.ax.add_patch(Rectangle((x + j * sw, fy - 1.6), sw, 1.9, fc=A.S._hex(c), ec="none", zorder=5))
    pg.ax.add_patch(Rectangle((x, fy - 1.6), sw * len(tints), 1.9, fc="none", ec=INK, lw=0.35,
                              zorder=6))
    texts.append(pg.text(x + sw * len(tints) + 0.8, fy, s_, KEY_FS))
    if methods:
        s_ = "Parts: " + ", ".join(methods)
        texts.append(pg.text(x0 + P / 2, fy + 3.0, s_, KEY_FS, ha="center", color=GREY))
    return texts


# -- the figure ------------------------------------------------------------------------
def prompt_lines(pg, instr):
    s_ = f"“{typographic(instr)}”" if instr else "instruction not recorded"
    return balanced_wrap(pg, s_, INS_W - 3.0, PROMPT_FS, style="italic")


PROMPT_MIN_FS = 5.0          # prompt_fit never goes under the figure's 5 pt floor


def prompt_fit(pg, instr, max_lines=None):
    """The prompt's lines and font size.  Without ``max_lines`` (every row by default):
    prompt_lines at the card's PROMPT_FS.  With it (a runs-JSON entry's
    ``prompt_max_lines``, ar_runs): the largest size <= PROMPT_FS, in 0.1 pt steps, at
    which the prompt wraps into at most ``max_lines`` lines, never under PROMPT_MIN_FS
    -- so one long instruction does not make every row of the figure taller (user,
    2026-10-19: "use smaller font size for this box and don't change the figure size")."""
    if max_lines is None:
        return prompt_lines(pg, instr), PROMPT_FS
    s_ = f"“{typographic(instr)}”" if instr else "instruction not recorded"
    for tenth in range(int(round(PROMPT_FS * 10)), int(round(PROMPT_MIN_FS * 10)) - 1, -1):
        fs = tenth / 10
        lines = balanced_wrap(pg, s_, INS_W - 3.0, fs, style="italic")
        if len(lines) <= max_lines:
            return lines, fs
    raise AssertionError(f"prompt does not fit {max_lines} line(s) at {PROMPT_MIN_FS} pt: {s_!r}")


def panel_box(max_lines):
    """The panel layout box (P wide, H tall, mm) and the row height RH, for a figure
    whose longest prompt takes ``max_lines`` lines: every row is as tall as that
    instruction column."""
    P = (W - MR - X0 - (NCOL - 1) * GAP) / NCOL
    card_h = CARD_PAD_H + PROMPT_PITCH * max_lines
    RH = CARD_TOP + card_h + COST_FIRST + 2 * COST_PITCH + BAR_OFF + BAR_H + ROW_PAD
    return P, RH - PANEL_PAD_TOP, RH


def print_layout(instructions, prompt_max_lines=None):
    """What ar_prep's camera chooser needs to predict this compose's checks for a
    figure of these instructions: the panel box (P x H) and the corner labels'
    measured widths (``width(text)``, META_FS, mm).  The constants it predicts with
    (MARGIN, INK_ROOM, the label bands, the print floors) are read off this module."""
    probe = Page(W, 50)
    caps = prompt_max_lines or [None] * len(instructions)
    lines = max(len(prompt_fit(probe, s_, m)[0]) for s_, m in zip(instructions, caps)) \
        if instructions else 1
    P, H, _ = panel_box(lines)

    def extent(s_):
        """A corner label's ink box as the overlap check measures it (Page.box):
        width, top above the baseline, bottom below it (mm)."""
        t = probe.text(0, 25.0, s_, META_FS)
        probe.fig.canvas.draw()
        b = probe.box(t)
        t.remove()
        probe.texts.remove(t)
        return b[2] - b[0], 25.0 - b[1], b[3] - 25.0
    return {"P": P, "H": H, "width": lambda s_: probe.width(s_, META_FS), "extent": extent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", default=os.path.join(HERE, "_work"))
    ap.add_argument("--out", default="agentic_results")
    ap.add_argument("--ground", default=os.environ.get("AR_GROUND", "none"),
                    choices=("none", "shadow", "grid"),
                    help="none; shadow = key-light shadow on the floor; grid = the UI viewer grid")
    args = ap.parse_args()
    stats = json.load(open(os.path.join(args.work, "ar_stats.json")))
    runs = stats["runs"]
    n = len(runs)
    lo, hi = stats["E_log10_range_Pa"]
    P = (W - MR - X0 - (NCOL - 1) * GAP) / NCOL
    xs = [X0 + c * (P + GAP) for c in range(NCOL)]
    colx = {p: PANELS.index(p) for p in PANELS}     # panel -> column index

    # ---- the instruction column sets the row height (user change 3) ----------------
    probe = Page(W, 50)
    card_w = INS_W
    prompts, prompt_fs, costs = [], [], []
    for rec in runs:
        instr = rec["meta"].get("instruction")
        s_ = f"“{typographic(instr)}”" if instr else "instruction not recorded"
        lines, fs_ = prompt_fit(probe, instr, rec.get("prompt_max_lines"))
        if rec.get("prompt"):     # the prep's record (ar_stats.json) must be what is drawn
            assert rec["prompt"]["fs"] == fs_ and rec["prompt"]["lines"] == lines, \
                f"prompt fit differs from the record: {s_!r}"
        prompt_fs.append(fs_)
        # the card is as tall as the figure's longest prompt, in every row (below)
        assert len(lines) <= MAX_PROMPT_LINES, f"a prompt needs {len(lines)} lines: {s_!r}"
        prompts.append(lines)
        c = cost_of(rec)
        if c["input"] is not None:
            # the cached share as a percentage: two totals that round alike needed 3
            # significant figures ("1.65 M input (1.55 M cached)"), wider than the column
            share = 100.0 * float(c["cached"] or 0) / float(c["input"]) if c["input"] else 0.0
            t_in = f"{fmt_count(c['input'])} input ({share:.0f} % cached)"   # "tokens": next line and the key
        else:
            t_in = "input tokens n/a"
        cl = [t_in, f"{fmt_count(c['output'])} output tokens", fmt_clock(c["time"])]
        st = str(c["status"] or "")
        if st and st.lower() not in ("ok", "success", "succeeded", "complete", "completed",
                                     "finished", "done"):
            cl[2] += f" · {st}"
        for s_ in cl:
            assert probe.width(s_, META_FS) <= INS_W, f"cost line wider than its column: {s_!r}"
        costs.append((c, cl))
    plt.close(probe.fig)
    max_lines = max(len(p) for p in prompts)
    _, H, RH = panel_box(max_lines)              # the panel layout box: row band below its top pad
    EXT = ROW_GAP / 2 - 0.2                      # drawn image reaches 0.2 mm short of the hairline
    # each run's time split into the bar's groups, as fractions of its wall clock (None: plain bar)
    splits = []
    for rec in runs:
        g_ = ar_time.groups_of((rec["meta"].get("time_split") or {}).get("sections") or {})
        tot = sum(g_.values())
        splits.append({g: g_.get(g, 0.0) / tot for g, _ in ar_time.GROUPS} if tot > 0 else None)
    maxima = {"input": max(c["input"] or 0 for c, _ in costs),
              "output": max(c["output"] or 0 for c, _ in costs),
              "time": max(c["time"] or 0 for c, _ in costs)}

    # ---- panels: one crop per row, contact shadows, the BC gates -------------------
    rows, bc_report = [], {}
    for k, rec in enumerate(runs):
        i = rec["index"]
        imgs = {p: load_rgba(os.path.join(args.work, f"ar_{i}_{p}.png")) for p in PANELS}
        idp = os.path.join(args.work, f"ar_{i}_bcid.png")
        idpass = np.asarray(Image.open(idp).convert("RGB")) if os.path.exists(idp) else None
        hidden = load_rgba(os.path.join(args.work, f"ar_{i}_bchidden.png"))
        hidden_faces = hidden
        harrows = load_rgba(os.path.join(args.work, f"ar_{i}_bchidden_arrows.png"))
        hsupport = load_rgba(os.path.join(args.work, f"ar_{i}_bchidden_support.png"))
        if hsupport is not None:  # hidden support outlines: their own pass, merged for the checks
            hidden = hsupport if hidden is None else np.where(
                (hsupport[..., 3:4] > 128), hsupport, hidden)
        if harrows is not None:   # far-face arrows: their own pass, laid over the faces' pass
            hidden = harrows if hidden is None else np.where(
                (harrows[..., 3:4] > 128), harrows, hidden)
        boxes = []
        for p in PANELS:
            if imgs[p] is None:
                continue
            b = ink_box(imgs[p])
            assert b is not None, f"row {k + 1} {p}: empty render"
            assert b[0] > 0 and b[1] > 0 and b[2] < imgs[p].shape[1] and b[3] < imgs[p].shape[0], \
                f"row {k + 1} {p}: the object touches the render border (clipped)"
            boxes.append(b)
        drawn = {}
        crop = row_crop(boxes, P, H, EXT, (rec.get("view") or {}).get("ink_room"),
                        float(rec.get("draw_scale") or 1.0)) if boxes else None
        if crop:
            x0, y0, x1, y1l, y1d, s = crop
            for p in PANELS:
                if imgs[p] is None:
                    continue
                win = cut_rgba(imgs[p], x0, y0, x1, y1d)
                if args.ground == "none":
                    rgb, _ = compose(win, s, shadow=False)
                else:
                    # the ground layer (key-light shadow or the UI grid) is drawn from
                    # its own render; the crop stays the object's.  Where there is no
                    # object, the ground fades out over GROUND_FADE mm at the drawn
                    # image's edges, so a long shadow or the grid's reach never ends
                    # in a hard panel-shaped line.
                    # the ground was rendered ALONE (render_ar): it goes under the object
                    g = load_rgba(os.path.join(args.work, f"ar_{i}_{p}_g.png"))
                    assert g is not None, f"row {k + 1} {p}: no ground render (AR_GROUND?)"
                    gw = cut_rgba(g, x0, y0, x1, y1d)
                    hh, ww = gw.shape[:2]
                    fpx = GROUND_FADE * s
                    yy, xx = np.mgrid[0:hh, 0:ww]
                    d = np.minimum.reduce([xx, ww - 1 - xx, yy, hh - 1 - yy]) / fpx
                    gw[..., 3] *= np.clip(d, 0, 1) ** 1.5
                    edge = max(gw[0, :, 3].max(), gw[-1, :, 3].max(), gw[:, 0, 3].max(),
                               gw[:, -1, 3].max())
                    assert edge < 3, f"row {k + 1} {p}: ground cut at its image edge"
                    base = over(gw, None)
                    if p == "solve" and "painted" in drawn:
                        base = outline_under(base, drawn["painted"][1], s)
                    rgb = over(win, base)
                    if p == "bc":   # the ground must not hide or tint the BC regions
                        bc_report.setdefault(f"{k + 1} with ground", bc_region_check(
                            k, rec, over(imgs["bc"], over(g, None)).astype(float), idpass,
                            1.0 / s, hidden))
                if p == "bc":
                    # loads on far faces, seen through the body: their own render laid
                    # over the panel as a SEE_THROUGH tint of the load colour; their
                    # arrows nearly opaque (a tint over a tinted wall vanished)
                    for layer, op, col in ((hidden_faces, SEE_THROUGH, A.NEUMANN),
                                           (harrows, ARROW_THROUGH, A.LOAD_ARROW),
                                           (hsupport, ARROW_THROUGH, A.DIRICHLET)):
                        if layer is None:
                            continue
                        hw = cut_rgba(layer, x0, y0, x1, y1d)[..., 3:4] / 255.0 * op
                        rgb = np.clip(rgb * (1 - hw) + np.array(col) * 255 * hw,
                                      0, 255).astype(np.uint8)
                drawn[p] = (rgb, win)
            if imgs["bc"] is not None:
                bc_report[k + 1] = bc_region_check(k, rec, colour_check_rgb(imgs["bc"]), idpass,
                                                   1.0 / s, hidden)
        rows.append((imgs, crop, drawn))

    # ---- page ---------------------------------------------------------------------
    def tag_of(c):
        return ("FEM SOLVER", GPU_GREEN) if PANELS[c] == "solve" else ("AGENT", AGENT)

    def head_room(c):     # the head's free span between its arrows (and the page edge)
        return (xs[c] + 0.7 + (0.3 if c > 0 else -0.7),
                xs[c] + P - 1.3 - 0.3 if c < NCOL - 1 else W - MR)
    probe = Page(W, 50)
    group_w = [probe.width(tag_of(c)[0], TAG_FS, weight="bold") + 2 * TAG_PAD + TAG_GAP
               + probe.width(h_, HEAD_FS, weight="bold") for c, h_ in enumerate(COLUMNS)]
    # tag beside its title when every head fits between its arrows, else stacked
    stacked = any(xs[c] + P / 2 - g / 2 < head_room(c)[0] or xs[c] + P / 2 + g / 2 > head_room(c)[1]
                  for c, g in enumerate(group_w))
    wrap_u = len(u_factor_lines(probe, P)) > 1
    plt.close(probe.fig)
    head, head_rule, ty = (HEAD_STACKED, HEAD_RULE_STACKED, TY_STACKED) if stacked \
        else (HEAD, HEAD_RULE, TY)
    body_top = head
    row_tops = [body_top + k * (RH + ROW_GAP) for k in range(n)]
    table_bottom = row_tops[-1] + RH + ROW_GAP / 2
    page_h = table_bottom + FOOT + (3.0 if wrap_u else 0.0)   # the ×N key on two lines
    pg = Page(W, page_h)

    # rules (booktabs) and hairlines; no vertical rules
    pg.line([ML, W - MR], [0.4, 0.4], 0.6, INK)
    pg.line([ML, W - MR], [head_rule, head_rule], 0.35, INK)
    hairlines = [row_tops[k] + RH + ROW_GAP / 2 for k in range(n - 1)]
    for yh in hairlines:
        pg.line([ML, W - MR], [yh, yh], 0.25, HAIR)
    pg.line([ML, W - MR], [table_bottom, table_bottom], 0.6, INK)

    # column heads, badges, arrows
    head_texts = [pg.text(ML, ty, "INSTRUCTION", HEAD_FS, weight="bold", color=GREY)]
    # Figure 1's star and chip badges do not read at this print size (the user):
    # words instead -- a solid rounded tag in Figure 1's colour before each title
    # ("agent": the LLM-assisted steps; "FEM solver": the GPU solve); no key needed
    tag_boxes, tag_pills = [], []
    for c, h_ in enumerate(COLUMNS):
        tw = pg.width(h_, HEAD_FS, weight="bold")
        # capitals, as the titles are; centred OPTICALLY: both lines placed by their
        # cap height (Arial 0.716 em), not by the font's ascent/descent box, so the
        # pill's centre line, its text's caps and the title's caps share one centre
        tag, tcol = tag_of(c)
        gw_ = pg.width(tag, TAG_FS, weight="bold") + 2 * TAG_PAD
        if stacked:      # the tag centred over its title, the title on the arrows' line
            x0_, tx_ = xs[c] + P / 2 - gw_ / 2, xs[c] + P / 2 - tw / 2
            cap_mid = TAG_TOP_STACKED + TAG_H / 2
        else:
            x0_ = xs[c] + P / 2 - (gw_ + TAG_GAP + tw) / 2   # tag + title, centred as a group
            tx_ = x0_ + gw_ + TAG_GAP
            cap_mid = ty - CAP_EM * HEAD_FS * PT_MM / 2       # the title's cap-height centre
        pg.ax.add_patch(FancyBboxPatch((x0_, cap_mid - TAG_H / 2), gw_, TAG_H,
                                       boxstyle="round,pad=0,rounding_size=0.9",
                                       fc=tcol, ec="none", zorder=6))
        head_texts.append(pg.text(x0_ + gw_ / 2, cap_mid + CAP_EM * TAG_FS * PT_MM / 2, tag,
                                  TAG_FS, ha="center", weight="bold", color="white"))
        head_texts.append(pg.text(tx_, ty, h_, HEAD_FS, weight="bold", color=GREY))
        tag_boxes.append((min(x0_, tx_), max(x0_ + gw_, tx_ + tw)) if stacked
                         else (x0_, x0_ + gw_ + TAG_GAP + tw))
        tag_pills.append((x0_, cap_mid - TAG_H / 2, x0_ + gw_, cap_mid + TAG_H / 2, tx_, tx_ + tw))
        if c < NCOL - 1:
            arrow(pg, xs[c] + P - 1.3, ty - 0.72, xs[c + 1] + 0.7, ty - 0.72)

    # rows
    corner_labels, row_cost_texts, card_texts, cards = [], [], [], []
    label_w = [pg.width(f"×{fmt_scale(r['scale']['factor'])}", META_FS) if r.get("scale")
               else 0 for r in runs]
    corner = pick_corner(rows, P, H, label_w)
    # the grid's cell size, once per row (the same grid lies under all four panels)
    # in the generated-shape panel, the far end of the row from the factor label
    cell_txt = [fmt_cell(r) if args.ground == "grid" else None for r in runs]
    cell_w = [CELL_ICON + 0.8 + pg.width(t_, META_FS) if t_ else 0 for t_ in cell_txt]
    cell_corner = pick_corner(rows, P, H, cell_w, panel=CELL_PANEL,
                              order=("bl", "tl")) \
        if any(cell_txt) else None
    # where the body came from (user, 2026-10-11): the generated-shape panel's lower
    # right corner in every row (the cell label keeps to the left-hand corners)
    src_txt = [r.get("body_source") for r in runs]

    def right_band(k, text, panel, order=("br", "tr")):
        # lower right unless the body's ink fills that band (a wide body: the crop fits
        # it by width and stands it on the panel floor, leaving the top free), then the
        # upper right; a row with ink in both keeps the lower one and fails the check
        c = pick_corner([rows[k]], P, H, [pg.width(text, META_FS)], panel=panel, order=order,
                        strict=False)
        return (c[0] if isinstance(c, list) else c)[0]
    src_pos = [(right_band(k, t_, "painted", route_order(pg.width(t_, META_FS), cell_w[k] if CELL_PANEL == "painted" else 0, P)),
                P - CORNER_PAD, "right")
               if t_ and "painted" in rows[k][2] else None
               for k, t_ in enumerate(src_txt)]
    # the body load: the force it applies, as text in the supports-and-loads panel's
    # lower right corner, placed as the source label is in its panel (user, 2026-10-12)
    body_txt = [body_label((r.get("bc") or {})["body_load"])
                if "bc" in rows[k][2] and (r.get("bc") or {}).get("body_load") else None
                for k, r in enumerate(runs)]
    body_band = [right_band(k, t_, "bc") if t_ else None for k, t_ in enumerate(body_txt)]
    for k, rec in enumerate(runs):
        y0 = row_tops[k]
        imgs, crop, drawn = rows[k]
        # the prompt card (Figure 1's prompt field): white, hairline, rounded, italic
        lines = prompts[k]
        ct = y0 + CARD_TOP
        ch = CARD_PAD_H + PROMPT_PITCH * max_lines     # one card height for every row
        off = PROMPT_PITCH * (max_lines - len(lines)) / 2   # a shorter prompt is centred
        # a smaller prompt (prompt_fit): its pitch scales with the size, and its block's
        # ink centre (first baseline .. last baseline less a cap height) stays where a
        # 7 pt prompt of as many lines would put it
        fs_ = prompt_fs[k]
        pitch = PROMPT_PITCH * fs_ / PROMPT_FS
        cap = lambda f: CAP_EM * f * PT_MM               # noqa: E731
        base0 = (ct + off + CARD_TEXT + PROMPT_PITCH * (len(lines) - 1) / 2 - cap(PROMPT_FS) / 2
                 - pitch * (len(lines) - 1) / 2 + cap(fs_) / 2)
        pg.ax.add_patch(FancyBboxPatch((ML, ct), card_w, ch,
                                       boxstyle="round,pad=0,rounding_size=1.3",
                                       fc="white", ec=CARD_STROKE, lw=0.6, zorder=4))
        cards.append((ML, ct, ML + card_w, ct + ch))
        for j, s_ in enumerate(lines):
            card_texts.append((k, pg.text(ML + card_w / 2, base0 + pitch * j, s_,
                                          fs_, ha="center", style="italic")))
        # the agent's cost, directly under the instruction (user change 1): one micro bar
        # per metric on a scale shared by every row; only the not-cached sliver saturated
        c, cl = costs[k]
        cb = ct + (CARD_PAD_H + PROMPT_PITCH * max_lines)    # align cost blocks across rows
        fracs = [(c["input"] or 0) / maxima["input"] if maxima["input"] else 0,
                 (c["output"] or 0) / maxima["output"] if maxima["output"] else 0,
                 (c["time"] or 0) / maxima["time"] if maxima["time"] else 0]
        cached_frac = (c["cached"] or 0) / maxima["input"] if maxima["input"] else 0
        assert cached_frac <= fracs[0] + 1e-12 and all(0 <= f_ <= 1 + 1e-12 for f_ in fracs)
        for j, s_ in enumerate(cl):
            yb = cb + COST_FIRST + j * COST_PITCH
            row_cost_texts.append(pg.text(ML, yb, s_, META_FS, color=GREY))
            by = yb + BAR_OFF
            pg.ax.add_patch(Rectangle((ML, by), INS_W, BAR_H, fc=COST_TRACK, ec="none", zorder=3))
            if j == 0:
                pg.ax.add_patch(Rectangle((ML, by), INS_W * fracs[0], BAR_H, fc=COST_UNCACHED,
                                          ec="none", zorder=4))
                pg.ax.add_patch(Rectangle((ML, by), INS_W * cached_frac, BAR_H, fc=COST_CACHED,
                                          ec="none", zorder=5))
            elif j == 2 and splits[k]:
                # the wall clock by where it went: one segment per group, on the shared scale
                x_ = ML
                for g, _ in ar_time.GROUPS:
                    w_ = INS_W * fracs[2] * splits[k][g]
                    pg.ax.add_patch(Rectangle((x_, by), w_, BAR_H, fc=TIME_COLORS[g], ec="none",
                                              zorder=4))
                    x_ += w_
            else:
                pg.ax.add_patch(Rectangle((ML, by), INS_W * fracs[j], BAR_H, fc=COST_NEUTRAL,
                                          ec="none", zorder=4))
        # panels, drawn from the panel layout top, extended EXT into the row gap
        ptop = y0 + PANEL_PAD_TOP
        for cidx, p in enumerate(PANELS):
            if p in drawn:
                pg.img(drawn[p][0], xs[cidx], ptop, P, H + EXT)
            else:
                pg.text(xs[cidx] + P / 2, ptop + H / 2, "not available", META_FS, ha="center",
                        va="center", color=GREY)
        # the displacement factor: corner of the deformed panel, grey, on the card's line
        sc = rec.get("scale")
        if sc and "solve" in drawn:
            ck = corner[k] if isinstance(corner, list) else corner
            cx = xs[colx["solve"]] + (CORNER_PAD if ck == "tl" else P - CORNER_PAD)
            corner_labels.append((k, pg.text(cx, y0 + LABEL_BASE, f"×{fmt_scale(sc['factor'])}",
                                             META_FS, color=GREY,
                                             ha="left" if ck == "tl" else "right"), colx["solve"], "solve"))
        # the grid cell: a square in the grid's own colour, then its side length
        cp_ = CELL_PANEL if CELL_PANEL in drawn else "painted"   # a row with no segmentation panel
        if cell_txt[k] and cp_ in drawn:
            w_ = cell_w[k]
            cc = cell_corner[k] if isinstance(cell_corner, list) else cell_corner
            lx = xs[colx[cp_]] + CORNER_PAD if cc[1] == "l" else xs[colx[cp_]] + P - CORNER_PAD - w_
            yb_ = y0 + (LABEL_BASE if cc[0] == "t"
                        else PANEL_PAD_TOP + H - BOTTOM_LABEL + 2.0)   # baseline in that band
            pg.ax.add_patch(Rectangle((lx, yb_ - CELL_ICON), CELL_ICON, CELL_ICON, fc="none",
                                      ec=GRID_RGB, lw=0.6, zorder=6))
            corner_labels.append((k, pg.text(lx + CELL_ICON + 0.8, yb_, cell_txt[k], META_FS,
                                             color=GREY), colx[cp_], "cell@" + cp_))
        if src_pos[k]:
            band, x_, ha_ = src_pos[k]
            sy = y0 + (LABEL_BASE if band == "t" else PANEL_PAD_TOP + H - BOTTOM_LABEL + 2.0)
            corner_labels.append((k, pg.text(xs[colx["painted"]] + x_, sy, src_txt[k], META_FS, color=GREY,
                                             ha=ha_), colx["painted"], "source"))
        if body_txt[k]:
            by_ = y0 + (LABEL_BASE if body_band[k] == "t" else PANEL_PAD_TOP + H - BOTTOM_LABEL + 2.0)
            corner_labels.append((k, pg.text(xs[colx["bc"]] + P - CORNER_PAD, by_,
                                             body_txt[k], META_FS, color=GREY, ha="right"),
                                  colx["bc"], "bc"))

    # ---- footer: every key under its column (user change 2) -------------------------
    fy = table_bottom + 3.4                      # first key baseline
    key_texts = []
    # instruction column: the cost bars' encoding
    cost_texts, time_key_texts = cost_keys(pg, ML, fy, any(splits))
    key_texts += cost_texts
    # segmentation: the part tints and the method that made the parts
    if "seg" in colx:
        methods = sorted({SEG_METHOD.get(m_, m_) for m_ in
                          ((rec.get("segmentation") or {}).get("method") for rec in runs) if m_})
        key_texts += seg_keys(pg, xs[colx["seg"]], P, fy, methods)
    # supports & loads
    key_texts += bc_keys(pg, xs[colx["bc"]], P, fy, any((rec.get("bc") or {}).get("arrows") for rec in runs),
                         any(g.get("seen_through") for rec in runs
                             for g in ((rec.get("bc") or {}).get("regions") or [])))
    # Young's modulus: the one log scale every row's colours were drawn on
    key_texts += e_key(pg, xs[colx["materials"]], P, fy, lo, hi, xs[colx["materials"] + 1])
    # deformed shape: |u| ramp + the factor's definition
    key_texts += u_keys(pg, xs[colx["solve"]], P, fy)

    # ---- assertions ------------------------------------------------------------------
    pg.fig.canvas.draw()
    bbs = []
    for t in pg.texts:
        s_ = t.get_text()
        if not s_:
            continue
        size = t.get_fontsize() * (0.7 if "$" in s_ and any(ch in s_ for ch in "_^") else 1.0)
        assert size >= 5.0 - 1e-6, f"text below 5 pt ({size:.1f} pt): {s_!r}"
        b = pg.box(t)
        assert b[0] >= -1e-3 and b[2] <= W + 1e-3 and b[1] >= -1e-3 and b[3] <= page_h + 1e-3, \
            f"text off the page: {s_!r}"
        bbs.append((b, s_))
    for a_ in range(len(bbs)):
        for b_ in range(a_ + 1, len(bbs)):
            p_, q_ = bbs[a_][0], bbs[b_][0]
            if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]:
                raise AssertionError(f"texts overlap: {bbs[a_][1]!r} / {bbs[b_][1]!r}")
    for c, (a0, a1) in enumerate(tag_boxes):   # tag + title between the head arrows
        left = xs[c] + 0.7 + (0.3 if c > 0 else -0.7)
        right = xs[c] + P - 1.3 - 0.3 if c < 3 else W - MR
        assert a0 >= left and a1 <= right, f"column {c + 1}: head runs into its arrows"
    for t in head_texts:          # head text clear of both rules by 0.6 mm
        b = pg.box(t)
        assert b[1] >= 0.4 + 0.6 and b[3] <= head_rule - 0.6, f"head text crowds a rule: {t.get_text()!r}"
    for t in row_cost_texts:      # the instruction column's text stays in its column
        assert pg.box(t)[2] <= ML + INS_W + 1e-6, f"cost text leaves its column: {t.get_text()!r}"
    for t in time_key_texts:      # the time bar's key may use the empty footer under the next column
        assert pg.box(t)[2] <= xs[0] + P + 1e-6, f"time key runs past the next column: {t.get_text()!r}"
    for k, t in card_texts:
        x0c, t0c, x1c, b1c = cards[k]
        b = pg.box(t)
        assert b[0] >= x0c + 0.5 and b[2] <= x1c - 0.5 and b[1] >= t0c and b[3] <= b1c, \
            f"prompt leaves its card: {t.get_text()!r}"
    for k, t, c_, pname in corner_labels:   # corner labels: in their panel, off the object,
        b = pg.box(t)                         # with their own row
        assert xs[c_] <= b[0] and b[2] <= xs[c_] + P, f"corner label leaves its panel: {t.get_text()!r}"
        rgb, win = rows[k][2]["painted" if pname == "source" else pname.split("@")[-1]]
        crop = rows[k][1]
        s = crop[5]
        ptop = row_tops[k] + PANEL_PAD_TOP
        x_lo = b[0] - (CELL_ICON + 0.8 if pname.startswith("cell@") else 0)   # the cell icon too
        sl = win[max(int((b[1] - ptop) * s), 0):int((b[3] - ptop) * s) + 1,
                 max(int((x_lo - xs[c_]) * s), 0):int((b[2] - xs[c_]) * s) + 1, 3]
        _floor_check(not (sl > 25).any(), f"row {k + 1}: corner label {t.get_text()!r} lies on the object")
        # with its own row: wholly inside the row's band, clear of the hairline below
        limit = (row_tops[k] + RH + ROW_GAP / 2) if k + 1 < n else table_bottom
        assert b[1] >= row_tops[k] and b[3] <= limit - 0.3, \
            f"corner label {t.get_text()!r} leaves its row band"
    for k in range(n):            # images (with shadows) stop short of the next hairline
        bottom = row_tops[k] + PANEL_PAD_TOP + H + EXT
        limit = hairlines[k] if k < n - 1 else table_bottom
        assert bottom <= limit - 0.15, f"row {k + 1}: drawn panel reaches the rule below"
    for t in key_texts:           # keys sit in the footer band
        assert pg.box(t)[1] >= table_bottom + 0.4, f"key crowds the bottom rule: {t.get_text()!r}"

    write_table(runs, args.out, maxima)
    # the tags, measured in the RENDER (ink, not font boxes): the pill's text has
    # equal margins above and below, and the title's ink shares the pill's centre
    pg.fig.canvas.draw()
    buf = np.asarray(pg.fig.canvas.buffer_rgba())[..., :3].astype(int)
    mm2px = DPI / 25.4
    for (px0, pt, px1, pb, tx0, tx1) in tag_pills:
        r0, r1 = int(pt * mm2px) - 2, int(pb * mm2px) + 3
        inner = buf[r0:r1, int((px0 + 0.9) * mm2px):int((px1 - 0.9) * mm2px)]
        white = (inner.min(axis=2) > 225).any(axis=1)          # the tag's white letters
        rr = np.nonzero(white)[0]
        ink_t, ink_b = (r0 + rr[0]) / mm2px, (r0 + rr[-1] + 1) / mm2px
        assert abs((ink_t - pt) - (pb - ink_b)) <= 0.1, \
            f"tag text off centre: {ink_t - pt:.2f} mm above, {pb - ink_b:.2f} mm below"
        if stacked:      # the title sits on its own line under the tag
            continue
        title = buf[r0 - 10:r1 + 10, int(tx0 * mm2px):int(tx1 * mm2px)]
        dark = (title.max(axis=2) < 170).any(axis=1)            # the grey title's ink
        rr = np.nonzero(dark)[0]
        t_mid = (r0 - 10 + (rr[0] + rr[-1] + 1) / 2) / mm2px
        assert abs(t_mid - (pt + pb) / 2) <= 0.1, \
            f"title and tag off one line by {t_mid - (pt + pb) / 2:.2f} mm"
    pg.fig.savefig(f"{args.out}.pdf", dpi=600)
    pg.fig.savefig(f"{args.out}.png", dpi=DPI)
    obj = []
    for k, (imgs, crop, drawn) in enumerate(rows):
        if crop:
            bx_ = [ink_box(imgs[p]) for p in PANELS if imgs[p] is not None]
            obj.append(((max(b[3] for b in bx_) - min(b[1] for b in bx_)) / crop[5],
                        (max(b[2] for b in bx_) - min(b[0] for b in bx_)) / crop[5]))
    print(f"wrote {args.out}.pdf / .png  ({W:.0f} x {page_h:.1f} mm; rows {RH:.1f} mm; "
          f"panels {P:.1f} x {H:.1f} mm)")
    print("object ink per row (height x width, mm): " +
          ", ".join(f"{h_:.1f} x {w_:.1f}" for h_, w_ in obj))
    print(f"factor corner: {corner}")
    for s_ in sorted(set(SMALL_REGIONS)):
        print("small region (reported, not out of view):", s_)
    for s_ in FLOOR_MISSES:
        print("UNDER THE PRINT FLOOR (AR_FLOOR=warn):", s_)
    print("BC regions (name, printed mm2, DeltaE00):", json.dumps(bc_report))
    print(f"wrote {args.out}_tokens.csv / .md")
    write_time_table(runs, args.out)
    print(f"wrote {args.out}_time.csv")


def pick_corner(rows, P, H, label_w, panel="solve", order=("tr", "tl"), strict=True):
    """The first top corner (in ``order``) whose label box (the label's measured
    width, its inset and 1 mm of air, down to just below the baseline) is free of
    object ink in EVERY row's ``panel``.  Same corner for every row.  ``strict=False``:
    a row with no clear corner takes the first without failing here (the placed
    label's own overlap check, without the 1 mm of air, still decides)."""
    def clear(row, lw, corner):
        imgs, crop, drawn = row
        if panel not in drawn:
            return True
        win = drawn[panel][1]
        s = crop[5]
        px = int(round((lw + CORNER_PAD + CORNER_CLEAR) * s))
        if corner[0] == "t":
            py = int(round((LABEL_BASE + 0.6 - PANEL_PAD_TOP) * s))   # to below the baseline
            rows_ = slice(0, py)
        else:      # the layout box's bottom band (the drawn image runs EXT further)
            hb = int(round(H * s))
            rows_ = slice(hb - int(round(BOTTOM_LABEL * s)), hb)
        sl = win[rows_, -px:] if corner[1] == "r" else win[rows_, :px]
        return not (sl[..., 3] > 25).any()

    for corner in order:
        if all(clear(row, lw, corner) for row, lw in zip(rows, label_w)):
            return corner
    # no corner clear in every row (ten new objects, 2026-10-10: wide bodies fill each
    # corner in some row): each row takes its own first clear corner
    per_row = []
    for row, lw in zip(rows, label_w):
        ok = [c for c in order if clear(row, lw, c)]
        if not ok:
            # every corner has ink: the row's first corner, listed in warn mode (a body
            # filling its whole panel); the paper build stays strict
            if strict:
                _floor_check(False, f"no corner of the {panel} panel is clear in a row")
            ok = [order[0]]
        per_row.append(ok[0])
    return per_row


if __name__ == "__main__":
    main()
