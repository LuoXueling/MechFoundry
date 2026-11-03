"""Compose step (any env): the BC-zip special cases into one figure.

Reads the PNGs ``render_bzx.py`` wrote and ``bzx_stats.txt`` from ``bzx_prep.py``;
writes ``bc_zip_special_cases.{pdf,png}``.  Nothing here computes geometry.

Layout: one row, two groups, one group per branch of the paper's enumeration that
``bc_zip_workflow`` never reaches.  A group is a geometry, so the break between
groups is wider than the gap inside one, and each carries what its branch does on
a rule above it.  Both are before/after pairs under one camera.

Palette, and therefore the key, is ``render_bzx.py``'s.
"""

import os

import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

# --- palette: must match render_bzx.py exactly (it is the same figure) -------
BODY = "#ebe6db"       # untouched base surface
BLUE = "#4c72b0"       # bridges
VIOLET = "#a99cd8"     # fan caps (light: #8171b3 is CVD dE 1.6 from BLUE)
RED = "#a13d41"        # a loop the nesting test flagged
INK = "#1a1a21"

stats = {}
for line in open("bzx_stats.txt"):
    k, _, v = line.strip().partition(" ")
    if k:
        stats[k] = v


def s_int(k):
    return int(float(stats[k]))


# --- panels ------------------------------------------------------------------
# (tag, file, title).  GROUPS keeps them in the three geometries they came from;
# the layout breaks between groups, not between panels.
GROUPS = [
    ("Cap a nested hole loop", [
        ("a", "bzx_panel_a.png", "Removal exposes a bore"),
        ("b", "bzx_panel_b.png", "Fan-cap the inner loop"),
    ]),
    ("Cluster & bridge surplus loops", [
        ("c", "bzx_panel_c.png", "Removal leaves three loops"),
        ("d", "bzx_panel_d.png", "Bridge & cap"),
    ]),
]
PANELS = [p for _, ps in GROUPS for p in ps]
MISSING = [p[0] for p in PANELS if not os.path.exists(p[1])]
assert not MISSING, (
    f"panel PNGs missing: {MISSING} — run bzx_prep.py then render_bzx.py.  "
    f"(build.sh deletes them after composing, so this script is not re-runnable "
    f"on its own after a full build.)")


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


raw = {t: Image.open(f) for t, f, _ in PANELS}
PAD = 8
imgs = {}
# One crop per GROUP: the panels in a group share a camera, so cropping them
# together keeps the object the same size on both sides of the change.  Cropping
# them apart would resize the object between "before" and "after" and read as
# the step having moved it.
for _label, ps in GROUPS:
    tags = [t for t, *_ in ps]
    x0, y0, x1, y1 = union([ink_box(raw[t]) for t in tags])
    W, H = raw[tags[0]].size
    box = (max(x0 - PAD, 0), max(y0 - PAD, 0), min(x1 + PAD, W), min(y1 + PAD, H))
    for t in tags:
        imgs[t] = raw[t].crop(box)

# Then letterbox EVERY panel to one common aspect, so all five are the same
# height and their titles sit on one line.
_AR = max(im.width / im.height for im in imgs.values())
for t in list(imgs):
    imgs[t] = pad_to(imgs[t], _AR)

# The strip is only as wide as it needs to be.  At 7.16 (the house double-column
# width) four 1.05 in panels left 0.8 in of margin on each side and a 0.9 in
# break: a quarter of the figure was empty.  Sized to its content instead, and
# placed at ITS OWN width rather than stretched to \textwidth.
FIG_W = 5.35
FS_TITLE, FS_KEY, FS_GROUP = 6.2, 6.4, 6.6
# The panel is sized DIRECTLY, not by dividing the page between however many
# panels there happen to be.  A figure placed at \textwidth prints at its own
# aspect ratio, so panel width relative to FIG_W *is* printed panel size: when
# this went from five panels to four, filling the width made every panel 30 %
# bigger and the strip half an inch taller.  1.30 in is the size the eight-panel
# bc_zip_workflow settled on for eight panels; the leftover width becomes margin
# and a wide break between the two pairs, which is what keeps the strip short.
PANEL_W = 1.05
GAP_IN = 0.20                      # between panels of one pair, inches;
                                   # set by the TITLES, which are wider than
                                   # a 1.05 in panel and would otherwise touch
GAP_BREAK = 0.42                   # between the two pairs
TITLE_IN = 0.24
GROUP_IN = 0.15
LEGEND_IN = 0.13
MARGIN_IN = 0.05

n = len(PANELS)
n_gaps = len(GROUPS) - 1
n_inner = n - len(GROUPS)          # gaps inside groups
panel_w = min(PANEL_W,
              (FIG_W - n_gaps * GAP_BREAK - n_inner * GAP_IN - 0.2) / n)
row_h = panel_w / _AR
fig_h = row_h + TITLE_IN + GROUP_IN + LEGEND_IN + 2 * MARGIN_IN

fig = plt.figure(figsize=(FIG_W, fig_h))
widths, cols, col = [], {}, 0
for gi, (_label, ps) in enumerate(GROUPS):
    if gi:
        widths.append(GAP_BREAK)
        col += 1
    for pi, (t, *_) in enumerate(ps):
        if pi:
            widths.append(GAP_IN)
            col += 1
        widths.append(panel_w)
        cols[t] = col
        col += 1
# Gaps are their own columns and wspace is 0, so the block's width is exactly
# the sum above -- which is what lets it be centred with the leftover as margin.
side_in = 0.5 * (FIG_W - sum(widths))
assert side_in > 0.02, "panels do not fit the page width"
gs = fig.add_gridspec(
    1, len(widths), width_ratios=widths,
    left=side_in / FIG_W, right=1.0 - side_in / FIG_W,
    top=1.0 - (TITLE_IN + GROUP_IN + MARGIN_IN) / fig_h,
    bottom=(LEGEND_IN + MARGIN_IN) / fig_h, wspace=0.0,
)

axes = {}
for t, _f, title in PANELS:
    ax = fig.add_subplot(gs[0, cols[t]])
    ax.imshow(np.asarray(imgs[t]), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"$\\bf{{({t})}}$ {title}", fontsize=FS_TITLE, pad=4,
                 va="baseline", color=INK, wrap=False)
    axes[t] = ax

# --- key: one centred horizontal row beneath the panels ----------------------
# Four entries, and every one of them names something a reader cannot guess.
# "Bridge", not "bridge": the initials name the algorithm, not the object,
# and a key is read by people who have not got to the method section yet.
# The cream base surface is not among them: it is everything that no step has
# touched, it is never referred to by a title, and naming it only crowds the
# things that matter.  Nor is the cached BC patch -- it is not drawn in any of
# these four panels, and a key entry for a colour that is nowhere on the page
# sends the reader hunting for it.
handles = [
    Patch(fc=BLUE, ec="none", label="Bridge"),
    Patch(fc=VIOLET, ec="none", label="Fan cap"),
    Line2D([], [], color=INK, lw=2.4, label="Open loop"),
    Line2D([], [], color=RED, lw=2.4, label="Nested loop"),
]
leg = fig.legend(handles=handles, loc="lower center",
                 bbox_to_anchor=(0.5, MARGIN_IN / fig_h * 0.4),
                 ncol=len(handles), frameon=False, fontsize=FS_KEY,
                 handlelength=1.2, handleheight=0.85, columnspacing=1.2,
                 borderpad=0.0, borderaxespad=0.0)
for txt in leg.get_texts():
    txt.set_color(INK)

# --- group labels ------------------------------------------------------------
GROUP_LABELS = []


def bbox(art):
    return art.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())


def _span(ps):
    pos = [axes[t].get_position() for t, *_ in ps]
    tops = [bbox(axes[t].title).y1 for t, *_ in ps]
    return min(p.x0 for p in pos), max(p.x1 for p in pos), max(tops)


fig.canvas.draw()
for label, ps in GROUPS:
    x0, x1, y1 = _span(ps)
    yy = y1 + 0.30 * GROUP_IN / fig_h
    fig.add_artist(plt.Line2D([x0, x1], [yy, yy], color="#9aa0a8", lw=0.7,
                              transform=fig.transFigure))
    _t = fig.text(0.5 * (x0 + x1), yy + 0.03 / fig_h, label, ha="center",
                  va="bottom", fontsize=FS_GROUP, color="#5a6069")
    GROUP_LABELS.append((label, _t))

# --- layout guards: fail loudly rather than ship a figure with a collision ---
fig.canvas.draw()

for t, ax in axes.items():
    p = ax.get_position()
    assert -0.002 <= p.x0 and p.x1 <= 1.002, f"panel {t} overruns x: {p}"
    assert -0.002 <= p.y0 and p.y1 <= 1.002, f"panel {t} overruns y: {p}"

tags = [t for t, *_ in PANELS]
for t0, t1 in zip(tags, tags[1:]):
    b0, b1 = axes[t0].get_position(), axes[t1].get_position()
    assert b1.x0 - b0.x1 > 0.003, f"panels {t0}/{t1} too close"
    tb0, tb1 = bbox(axes[t0].title), bbox(axes[t1].title)
    assert tb0.x1 < tb1.x0, f"titles ({t0}) and ({t1}) overlap"

# Text lives outside its axes, so the axes guards above cannot see it overflow.
for t, ax in axes.items():
    for name, art in (("title", ax.title),):
        b = bbox(art)
        assert 0.0 <= b.x0 and b.x1 <= 1.0, f"{name} ({t}) runs off x: {b}"
        assert 0.0 <= b.y0 and b.y1 <= 1.0, f"{name} ({t}) runs off y: {b}"

# Group labels must stay over their own group, and never touch each other.
_gb = [bbox(a) for _n, a in GROUP_LABELS]
for b in _gb:
    assert 0.0 <= b.x0 and b.x1 <= 1.0 and 0.0 <= b.y0 and b.y1 <= 1.0, \
        f"group label runs off the canvas: {b}"
for b0, b1 in zip(_gb, _gb[1:]):
    assert b0.x1 < b1.x0, "group labels overlap"

_lb = bbox(leg)
assert 0.0 <= _lb.x0 and _lb.x1 <= 1.0 and 0.0 <= _lb.y0 and _lb.y1 <= 1.0, \
    f"legend runs off the canvas: {_lb}"

# The group breaks must READ as breaks.
_inner = min(axes[t1].get_position().x0 - axes[t0].get_position().x1
             for _l, ps in GROUPS for t0, t1 in zip([p[0] for p in ps],
                                                    [p[0] for p in ps][1:]))
for (_l0, p0), (_l1, p1) in zip(GROUPS, GROUPS[1:]):
    gap = axes[p1[0][0]].get_position().x0 - axes[p0[-1][0]].get_position().x1
    assert gap > 2.0 * _inner, \
        f"group break ({gap:.4f}) is not clearly wider than the gap inside a group ({_inner:.4f})"

# Both branches must have actually run in the capture this figure is drawn from.
assert s_int("a_cap_tris") > 0, "step 3's cap never ran — do not ship"
assert s_int("a_inner") == 1, "the removal no longer leaves a nested loop"
assert s_int("b_bridge_tris") > 0 and s_int("b_cap_tris") > 0, \
    "step 4's bridge/cap never ran — do not ship"

fig.savefig("bc_zip_special_cases.pdf", dpi=600)
fig.savefig("bc_zip_special_cases.png", dpi=400)
print(f"wrote bc_zip_special_cases.pdf / .png  ({FIG_W:.2f} x {fig_h:.2f} in)")
