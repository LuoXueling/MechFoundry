"""Compose step (conda `mechfoundry-warp117`): assemble the three rendered panels into the figure.

Reads ``bc_panel_{a,b,c}.png`` + ``bc_stats.txt`` and writes
``bc_region_id.{pdf,png}``.  All quoted counts come from ``bc_stats.txt``, i.e.
from the production selectors — nothing here is hand-entered.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch
from PIL import Image

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "savefig.transparent": False,
})

ORANGE = "#DD8452"
GRAY = "#8C8C8C"

stats = {}
for line in open("bc_stats.txt"):
    k, *v = line.split()
    stats[k] = v[0] if len(v) == 1 else v


def trim(img, pad=6):
    """Crop uniform white margin, keeping ``pad`` px."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = (np.abs(a - 255).sum(axis=2) > 12)
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
    return img.crop((x0, y0, x1, y1))


def pad_to(img, aspect):
    """Letterbox onto a white canvas of the given width/height ratio."""
    w, h = img.size
    W, H = (w, int(round(w / aspect))) if w / h > aspect else (int(round(h * aspect)), h)
    canvas = Image.new("RGB", (max(W, w), max(H, h)), "white")
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


# Panels (a) and (b) share a camera, so crop them to a COMMON box or the bunny
# would change size between them.
pa, pb = Image.open("bc_panel_a.png"), Image.open("bc_panel_b.png")


def ink_box(img):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = (np.abs(a - 255).sum(axis=2) > 12)
    ys, xs = np.where(ink)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


ba, bb = ink_box(pa), ink_box(pb)
PAD = 14
common = (max(min(ba[0], bb[0]) - PAD, 0), max(min(ba[1], bb[1]) - PAD, 0),
          min(max(ba[2], bb[2]) + PAD, pa.width),
          min(max(ba[3], bb[3]) + PAD, pa.height))
pa, pb = pa.crop(common), pb.crop(common)

# Panel (c) is trimmed to its ink box so the ear fills the panel width.  Every
# panel is then letterboxed to SQUARE: matching (c)'s wider native aspect instead
# would shrink the bunny in (a) and (b) and unbalance the row.
pc = trim(Image.open("bc_panel_c.png"), pad=10)

ASPECT = 1.0
panels = [pad_to(im, ASPECT) for im in (pa, pb, pc)]

# (tag, single-line title).  Sentence case throughout; "Flood Fill" and "Grow by
# Rings" keep their capitals because they name the primitives, not the step.
# The seed fraction is read from the stats file so the title cannot drift from
# the selector that produced it.
TITLES = [
    ("a", f"Seed selection: top {float(stats['seed_fraction']):.0%} of height"),
    ("b", "Flood Fill & Grow by Rings"),
    ("c", "Close-up of the selected facets"),
]

FIG_W = 7.16                      # double-column width
LEFT, RIGHT, WSPACE = 0.012, 0.988, 0.14   # gutter sized for the step arrows
HEAD_IN = 0.26                    # single-line title band above the panels
FOOT_IN = 0.30                    # legend band below them

# Square panels: derive the panel height from the width the gridspec actually
# gives each column, so the row has no slack to letterbox away.
panel_w = (RIGHT - LEFT) * FIG_W / (3.0 + 2.0 * WSPACE)
panel_h = panel_w / ASPECT
fig_h = panel_h + HEAD_IN + FOOT_IN
fig = plt.figure(figsize=(FIG_W, fig_h))
gs = fig.add_gridspec(
    1, 3, left=LEFT, right=RIGHT, wspace=WSPACE,
    top=1.0 - HEAD_IN / fig_h, bottom=FOOT_IN / fig_h,
)

axes = []
for i, (im, (tag, title)) in enumerate(zip(panels, TITLES)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.asarray(im), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # One line, tag included: 8.2 pt keeps the longest title inside its column
    # (at 9 pt it overruns the gutter and crowds the neighbouring panel).
    ax.set_title(f"$\\bf{{({tag})}}$  {title}", fontsize=8.2, pad=6, va="baseline")
    axes.append(ax)

# Plain arrows between steps — the panel sequence carries the process.
fig.canvas.draw()
for left, right in ((axes[0], axes[1]), (axes[1], axes[2])):
    p0 = left.get_position()
    p1 = right.get_position()
    x = 0.5 * (p0.x1 + p1.x0)
    y = p0.y0 + 0.52 * (p0.y1 - p0.y0)
    # Keep the DEFAULT "-|>" head proportions and grow the shaft with it.  A
    # custom head_length at this mutation_scale eats the whole shaft and the
    # arrow degenerates into a bare triangle (see the marker gotcha in §3).
    fig.patches.append(FancyArrowPatch(
        (x - 0.016, y), (x + 0.016, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=20, lw=2.6,
        color=GRAY, shrinkA=0, shrinkB=0, zorder=5,
    ))

handles = [
    Patch(facecolor=ORANGE, edgecolor="none", label="Selected facets"),
]
fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
           fontsize=7.6, handlelength=1.5, columnspacing=2.0,
           handletextpad=0.55, bbox_to_anchor=(0.5, 0.008))

fig.savefig("bc_region_id.pdf", dpi=600)
fig.savefig("bc_region_id.png", dpi=400)
print(f"wrote bc_region_id.pdf and bc_region_id.png  "
      f"({FIG_W:.2f} x {fig_h:.2f} in, panel aspect {ASPECT:.3f})")
