"""Compose step (any env): the seven BC-restoration panels into one figure.

Reads the PNGs ``render_bz.py`` wrote and ``bz_stats.txt`` from ``bz_prep.py``;
writes ``bc_zip_workflow.{pdf,png}``.  Nothing here computes geometry — every
number printed on the figure is read back from the capture, so the caption
cannot drift from the run.

Layout: steps 1-4 across the top as whole-body views, because what they change
is global (which triangles are selected, which loops exist).  Steps 5-7 across
the bottom as close-ups on one toe, because at body scale they are the same
picture -- the strip is already in place after step 5 and the last two steps
only re-triangulate and relax it.  The eighth cell carries the key.
"""

import os

import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image

for f in ("arial.ttf", "arialbd.ttf"):
    try:
        font_manager.fontManager.addfont(
            f"/usr/share/fonts/truetype/msttcorefonts/{f}")
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "savefig.transparent": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# --- palette: must match render_bz.py exactly (it is the same figure) --------
BODY = "#ebe6db"       # untouched base surface
CAND0 = "#a13d41"      # the initial 1.1x AABB candidate
CANDG = "#d99093"      # what the BFS ring expansion added
BLUE = "#4c72b0"       # bridges / zip strips (house blue)
VIOLET = "#a99cd8"     # fan caps — see render_bzx.py (CVD vs BLUE)
GREEN = "#55a868"      # the restored BC patch -- the same house green
                       # bc_region_id and diffused_nitsche use for BC
LOOP = "#1f4fd8"       # open-loop polylines
INK = "#1a1a21"

stats = {}
for line in open("bz_stats.txt"):
    k, _, v = line.strip().partition(" ")
    if k:
        stats[k] = v


def s_int(k):
    return int(float(stats[k]))


# --- panels ------------------------------------------------------------------
# (tag, file, title, sub) -- sub is the one measured fact that step contributes.
ROW_TOP = [
    ("a", "bz_panel_1.png", "Cache BC patch",
     f"{s_int('patch_faces')} tris, {s_int('patch_loops')} loops; no nesting"),
    ("b", "bz_panel_2.png", "Grow by rings",
     f"{s_int('rings')} BFS rings: "
     f"{s_int('cand_loops_first')} → {s_int('cand_loops_last')} loops"),
    ("c", "bz_panel_3.png", "Remove & restore",
     f"{s_int('loops_removed')} open loops left; "
     f"{s_int('inner_cap_tris')} inner-cap tris"),
    ("d", "bz_panel_4.png", "Cluster, bridge",
     f"{s_int('loops_removed')} → {s_int('loops_after_merge')} loops; "
     f"{s_int('merge_tris')}+{s_int('cluster_cap_tris')} tris"),
]
ROW_BOT = [
    ("e", "bz_panel_5z.png", "Auto-match & zip",
     f"{s_int('zip_pairs')} pairs, {s_int('zip_tris')} tris — watertight"),
    ("f", "bz_panel_6z.png", "Subdivide",
     f"{s_int('subdiv_before')} → {s_int('subdiv_after')} tris"),
    ("g", "bz_panel_7z.png", "Taubin-smooth",
     "curvature continuous across both seams"),
    ("h", "bz_panel_7.png", "Updated geometry", "with consistent BC"),
]


# The 4th field was the old per-panel statistic line, which is no longer drawn.
# Only (h) carries a sub-line; leaving the rest populated silently reinstated
# every stat caption the moment a sub-line was rendered at all.
ROW_TOP = [(t, f, ttl, "") for t, f, ttl, _ in ROW_TOP]
ROW_BOT = [(t, f, ttl, sub if t == "h" else "") for t, f, ttl, sub in ROW_BOT]


def ink_box(img, tol=12):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = np.abs(a - 255).sum(axis=2) > tol
    ys, xs = np.where(ink)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def union(boxes):
    b = np.asarray(boxes)
    return (int(b[:, 0].min()), int(b[:, 1].min()),
            int(b[:, 2].max()), int(b[:, 3].max()))


def pad_to(img, aspect, bg="white"):
    """Letterbox onto a canvas of the given width/height ratio."""
    w, h = img.size
    W, H = ((w, int(round(w / aspect))) if w / h > aspect
            else (int(round(h * aspect)), h))
    canvas = Image.new("RGB", (max(W, w), max(H, h)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


# (a) is captured BEFORE the first optimization step; (b)-(g) all happen in a
# later iteration, on geometry the optimiser has since changed.  That is a real
# discontinuity in time, so the layout breaks there rather than running seven
# panels together as if they were one sequence.
PANELS = [r for r in ROW_TOP + ROW_BOT if os.path.exists(r[1])]
MISSING = [(t, ttl) for t, f, ttl, _ in ROW_TOP + ROW_BOT
           if not os.path.exists(f)]
PRE, POST = PANELS[:1], PANELS[1:]

raw = {t: Image.open(f) for t, f, _, _ in PANELS}
PAD = 14
imgs = {}
# Whole-body panels share a camera and so share one crop; the seam close-ups
# share a different one.  Crop each group to its own common ink box.
for grp in ([r for r in PANELS if r[1] in
             ("bz_panel_1.png", "bz_panel_2.png", "bz_panel_3.png",
              "bz_panel_4.png", "bz_panel_7.png")],
            [r for r in PANELS if r[1].endswith("z.png")]):
    if not grp:
        continue
    tags = [t for t, *_ in grp]
    x0, y0, x1, y1 = union([ink_box(raw[t]) for t in tags])
    W, H = raw[tags[0]].size
    box = (max(x0 - PAD, 0), max(y0 - PAD, 0), min(x1 + PAD, W), min(y1 + PAD, H))
    for t in tags:
        imgs[t] = raw[t].crop(box)

# Letterbox EVERY panel to one common aspect.  Different aspects per group made
# the rows different heights, which top-aligned left the titles staggered.
_AR = max(im.width / im.height for im in imgs.values())
for t in list(imgs):
    imgs[t] = pad_to(imgs[t], _AR)

FIG_W = 7.16
FS_TITLE, FS_KEY, FS_GROUP = 6.6, 6.5, 7.2
GAP_W = 0.28                       # holds the 'Optimized' arrow, inches
LEFT_IN = 0.34
RIGHT_IN = 0.14                     # (a) sits in from the edge, near the rest
SUB_IN = 0.11
TITLE_IN = 0.22 + SUB_IN
GROUP_IN = 0.15
LEGEND_IN = 0.13
MARGIN_IN = 0.05

n = len(PANELS)
panel_w = (FIG_W - GAP_W - LEFT_IN - RIGHT_IN) / n
row_h = panel_w / _AR
fig_h = row_h + TITLE_IN + GROUP_IN + LEGEND_IN + 2 * MARGIN_IN

fig = plt.figure(figsize=(FIG_W, fig_h))
widths = [panel_w] * len(PRE) + [GAP_W] + [panel_w] * len(POST)
gs = fig.add_gridspec(
    1, len(widths), width_ratios=widths,
    left=LEFT_IN / FIG_W, right=1.0 - RIGHT_IN / FIG_W,
    top=1.0 - (TITLE_IN + GROUP_IN + MARGIN_IN) / fig_h,
    bottom=(LEGEND_IN + MARGIN_IN) / fig_h, wspace=0.06,
)
COLS = list(range(len(PRE))) + list(range(len(PRE) + 1, len(PRE) + 1 + len(POST)))

axes = {}
for (t, _f, title, sub), col in zip(PANELS, COLS):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(np.asarray(imgs[t]), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"$\\bf{{({t})}}$ {title}", fontsize=FS_TITLE,
                 pad=4 + 72 * SUB_IN * 0.62, va="baseline", color=INK,
                 wrap=False)
    if sub:
        ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=FS_TITLE - 1.2, color="#7c828b")
    axes[t] = ax

# --- key: one centred horizontal row beneath the panels ----------------------
handles = [
    Patch(fc=BODY, ec="none", label="Base surface"),
    Patch(fc=CAND0, ec="none", label="Removal candidate"),
    Patch(fc=GREEN, ec="none", label="Stored & restored BC patch"),
    Patch(fc=BLUE, ec="none", label="Zip strip"),
]
leg = fig.legend(handles=handles, loc="lower center",
                 bbox_to_anchor=(0.5, MARGIN_IN / fig_h * 0.4),
                 ncol=len(handles), frameon=False, fontsize=FS_KEY,
                 handlelength=1.2, handleheight=0.85, columnspacing=1.5,
                 borderpad=0.0, borderaxespad=0.0)
for txt in leg.get_texts():
    txt.set_color(INK)

# --- group labels: (a) is a different moment in time from the rest ----------
GROUP_LABELS = []


def bbox(art):
    return art.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())


def _span(rows):
    ps = [axes[t].get_position() for t, *_ in rows]
    tops = [bbox(axes[t].title).y1 for t, *_ in rows]
    return min(p.x0 for p in ps), max(p.x1 for p in ps), max(tops)


fig.canvas.draw()
for rows, label in ((PRE, "Before optimization"),
                    (POST, "After optimization in a later iteration, before SDF reinitialization")):
    if not rows:
        continue
    x0, x1, y1 = _span(rows)
    yy = y1 + 0.30 * GROUP_IN / fig_h
    fig.add_artist(plt.Line2D([x0, x1], [yy, yy], color="#9aa0a8", lw=0.7,
                              transform=fig.transFigure))
    _t = fig.text(0.5 * (x0 + x1), yy + 0.03 / fig_h,
                  label, ha="center", va="bottom", fontsize=FS_GROUP,
                  color="#5a6069")
    GROUP_LABELS.append((label, _t))

# --- the optimiser step itself: (a) is the cache, (b) onwards is what came
# --- back changed.  The arrow is the only thing in the figure that is not
# --- geometry, so it lives in the break rather than over a panel.
if PRE and POST:
    _a = axes[PRE[-1][0]].get_position()
    _b = axes[POST[0][0]].get_position()
    _y = 0.5 * (_a.y0 + _a.y1)
    # A negative pad: the arrow runs a little past the break into the white
    # margins of (a) and (b), so it reads as a transition rather than a tick --
    # but not so far that it crowds either panel.  It clears the artwork; both
    # neighbours are letterboxed.  (Length is 1.5x the gap; 2.2x was too long,
    # 0.8x too short.)
    _pad = -0.26 * (_b.x0 - _a.x1)
    arrow = FancyArrowPatch((_a.x1 + _pad, _y), (_b.x0 - _pad, _y),
                            transform=fig.transFigure, arrowstyle="-|>",
                            mutation_scale=7, lw=0.9, color="#5a6069",
                            shrinkA=0, shrinkB=0)
    fig.add_artist(arrow)
    arrow_txt = fig.text(0.5 * (_a.x1 + _b.x0), _y + 0.018,
                         "Optimized", ha="center", va="bottom",
                         fontsize=FS_TITLE - 0.4, color="#5a6069")

# --- layout guards: fail loudly rather than ship a figure with a collision ---
fig.canvas.draw()


for t, ax in axes.items():
    p = ax.get_position()
    assert -0.002 <= p.x0 and p.x1 <= 1.002, f"panel {t} overruns x: {p}"
    assert -0.002 <= p.y0 and p.y1 <= 1.002, f"panel {t} overruns y: {p}"

tags = [t for t, *_ in PANELS]
fig.canvas.draw()
for t0, t1 in zip(tags, tags[1:]):
    b0, b1 = axes[t0].get_position(), axes[t1].get_position()
    assert b1.x0 - b0.x1 > 0.003, f"panels {t0}/{t1} too close"
    tb0, tb1 = bbox(axes[t0].title), bbox(axes[t1].title)
    assert tb0.x1 < tb1.x0, f"titles ({t0}) and ({t1}) overlap"

# the break between the pre-optimization cache and the rest must READ as a break
if len(PRE) and len(POST):
    gap = axes[POST[0][0]].get_position().x0 - axes[PRE[-1][0]].get_position().x1
    inner = min(axes[t1].get_position().x0 - axes[t0].get_position().x1
                for t0, t1 in zip(tags[1:], tags[2:]))
    assert gap > 2.0 * inner, \
        f"the (a)/(b) break ({gap:.4f}) is not clearly wider than the rest ({inner:.4f})"

# EVERY text artist must land inside the canvas.  The first cut of this figure
# clipped the row-1 titles off the top and ran the footer past the right edge,
# and the guards above passed both: they only checked the axes, and text lives
# outside its axes.  Check the drawn text itself.
for t, ax in axes.items():
    for name, art in (("title", ax.title),):
        b = bbox(art)
        assert 0.0 <= b.x0 and b.x1 <= 1.0, f"{name} ({t}) runs off x: {b}"
        assert 0.0 <= b.y0 and b.y1 <= 1.0, f"{name} ({t}) runs off y: {b}"
if PRE and POST:
    _ab = bbox(arrow_txt)
    assert _ab.x0 >= _a.x1 - 0.05 and _ab.x1 <= _b.x0 + 0.05, \
        f"'Optimized' label leaves the break: {_ab}"

for name, art in ([("legend", leg)]
                  + [(f"group '{n}'", a) for n, a in GROUP_LABELS]
                  + ([("arrow label", arrow_txt)] if PRE and POST else [])):
    b = bbox(art)
    assert 0.0 <= b.x0 and b.x1 <= 1.0 and 0.0 <= b.y0 and b.y1 <= 1.0, \
        f"{name} runs off the canvas: {b}"

assert s_int("watertight") == 1, "capture is not watertight - do not ship"

fig.savefig("bc_zip_workflow.pdf", dpi=600)
fig.savefig("bc_zip_workflow.png", dpi=400)
print(f"wrote bc_zip_workflow.pdf / .png  ({FIG_W:.2f} x {fig_h:.2f} in)")
