"""Compose the focused void-nucleation figure from hidden-phi artifacts.

The canonical numerical run is ``../hiddenphi/hp_prep.py``.  ``render_vn.py``
only aliases its rendered panels into this directory; this script lays out a
more detailed account of the nucleation event without executing or inventing a
second optimization.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


HERE = Path(__file__).resolve().parent
HP_STATS = HERE.parent / "hiddenphi" / "hp_stats.txt"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "cmsy10",      # \mathcal{J}: TeX's calligraphic J (the default
    #                                "cursive" family resolves to Comic Sans here)
    "savefig.transparent": False,
})

INK = "#1a1a21"
TEAL = "#12a594"
ORANGE = "#dd8452"
BODY = "#ebe6db"
VOID = "#4f5866"


def read_stats(path):
    out = {}
    for line in path.read_text().splitlines():
        key, *values = line.split()
        out[key] = values[0] if len(values) == 1 else values
    return out


stats = read_stats(HP_STATS)
capture_it = int(stats["iter_capture"])
final_it = int(stats["iter_final"])
evo_iters = [int(x) for x in stats["evo_iters"]]
log_floor = float(stats["grad_log_floor"])

# These are the exact transfer functions used by hiddenphi/render_hp.py.
SDF_RGB = [(-0.300, (0.106, 0.196, 0.478)),
           (-0.192, (0.235, 0.443, 0.718)),
           (-0.084, (0.557, 0.706, 0.851)),
           (0.000, (0.894, 0.894, 0.894)),
           (0.048, (0.968, 0.805, 0.576)),
           (0.132, (0.944, 0.600, 0.282)),
           (0.216, (0.827, 0.329, 0.152)),
           (0.300, (0.478, 0.129, 0.094))]
PHI_RGB = [(0.0000, (0.804, 0.886, 0.984)),
           (0.1667, (0.620, 0.773, 0.957)),
           (0.3333, (0.427, 0.655, 0.925)),
           (0.5000, (0.224, 0.529, 0.898)),
           (0.6667, (0.145, 0.416, 0.749)),
           (0.8333, (0.094, 0.310, 0.584)),
           (1.0000, (0.051, 0.212, 0.420))]
GRAD_RGB = [(0.0000, (1.000, 1.000, 1.000)),
            (0.1667, (0.992, 0.839, 0.729)),
            (0.3333, (0.988, 0.737, 0.565)),
            (0.5000, (0.949, 0.588, 0.396)),
            (0.6667, (0.867, 0.435, 0.259)),
            (0.8333, (0.733, 0.310, 0.161)),
            (1.0000, (0.549, 0.176, 0.075))]


def cmap(name, nodes):
    lo, hi = nodes[0][0], nodes[-1][0]
    return LinearSegmentedColormap.from_list(
        name, [((x - lo) / (hi - lo), colour) for x, colour in nodes])


CM_SDF = cmap("sdf", SDF_RGB)
CM_PHI = cmap("phi", PHI_RGB)
CM_GRAD = cmap("grad", GRAD_RGB)


def ink_box(img, pad=12):
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    mask = np.abs(a - 255).sum(axis=2) > 12
    ys, xs = np.where(mask)
    if not len(xs):
        return (0, 0, img.width, img.height)
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + 1 + pad, img.width),
            min(int(ys.max()) + 1 + pad, img.height))


field_names = [f"{tag}_{field}" for field in ("phi", "d")
               for tag in ("a", "b", "c")]
raw = {name: Image.open(HERE / f"vn_panel_{name}.png").convert("RGB")
       for name in field_names}
boxes = [ink_box(image, pad=0) for image in raw.values()]
pad = 14
w0, h0 = next(iter(raw.values())).size
common = (max(min(b[0] for b in boxes) - pad, 0),
          max(min(b[1] for b in boxes) - pad, 0),
          min(max(b[2] for b in boxes) + pad, w0),
          min(max(b[3] for b in boxes) + pad, h0))
fields = {name: np.asarray(image.crop(common)) for name, image in raw.items()}

surface_raw = {tag: Image.open(HERE / f"vn_panel_{tag}.png").convert("RGB")
               for tag in ("d", "e")}
surfaces = {tag: np.asarray(image.crop(ink_box(image)))
            for tag, image in surface_raw.items()}

evo_raw = {it: Image.open(HERE / f"vn_panel_e{it}.png").convert("RGB")
           for it in evo_iters}
evo_boxes = [ink_box(image, pad=0) for image in evo_raw.values()]
ew, eh = next(iter(evo_raw.values())).size
evo_common = (max(min(b[0] for b in evo_boxes) - pad, 0),
              max(min(b[1] for b in evo_boxes) - pad, 0),
              min(max(b[2] for b in evo_boxes) + pad, ew),
              min(max(b[3] for b in evo_boxes) + pad, eh))
evo = {it: np.asarray(image.crop(evo_common)) for it, image in evo_raw.items()}

FIG_W, FIG_H = 7.16, 5.30
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    2, 5, left=0.012, right=0.988, bottom=0.425, top=0.925,
    width_ratios=(1.0, 1.0, 1.0, 1.30, 1.30), wspace=0.08, hspace=0.04)

titles = {
    "a": "Current state",
    "b": "Sensitivities",
    "c": r"Hidden-$\varphi$ update",
    "d": "Void carved",
    "e": f"Final (iter {final_it})",
}
labels = {
    "a_phi": r"$\overline{\varphi}$",
    "a_d": r"$s$",
    "b_phi": r"$|\mathrm{D}\mathcal{J}/\mathrm{D}\varphi|$",
    "b_d": r"$|\mathrm{D}\mathcal{J}/\mathrm{D}s|$",
    "c_phi": r"$\overline{\varphi}$  (dips)",
    "c_d": r"$s$  (deep interior unchanged)",
}

axes = {}
for col, tag in enumerate(("a", "b", "c")):
    for row, field in enumerate(("phi", "d")):
        name = f"{tag}_{field}"
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(fields[name], interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(f"$\\bf{{({tag})}}$  {titles[tag]}", fontsize=8.0,
                         pad=4, va="baseline")
        ax.text(0.025, 0.975, labels[name], transform=ax.transAxes,
                fontsize=7.1, ha="left", va="top", color=INK,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none",
                          alpha=0.82))
        axes[name] = ax

for col, tag in ((3, "d"), (4, "e")):
    ax = fig.add_subplot(gs[:, col])
    ax.imshow(surfaces[tag], interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"$\\bf{{({tag})}}$  {titles[tag]}", fontsize=8.0,
                 pad=4, va="baseline")
    axes[tag] = ax

# Three bars explain the paired field columns. Their stored gradient values are
# log-normalized independently, hence the shared support-scale label.
def add_bar(ax, cm, ticks, ticklabels, label, vmin=0.0, vmax=1.0):
    p = ax.get_position()
    width = 0.78 * p.width
    cax = fig.add_axes([p.x0 + 0.5 * (p.width - width), 0.350, width, 0.014])
    cax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cm,
               extent=(vmin, vmax, 0, 1))
    cax.set_xlim(vmin, vmax)
    cax.set_yticks([])
    cax.set_xticks(ticks)
    cax.set_xticklabels(ticklabels, fontsize=6.1)
    cax.tick_params(axis="x", length=2.0, width=0.6, pad=1.2, colors=INK)
    for spine in cax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(INK)
    cax.set_xlabel(label, fontsize=6.5, labelpad=1.0, color=INK)
    return cax


bars = [
    add_bar(axes["a_d"], CM_PHI, [0, 0.5, 1], ["0", "0.5", "1"],
            r"$\overline{\varphi}$"),
    add_bar(axes["b_d"], CM_GRAD, [0, 0.5, 1],
            [f"$10^{{-{log_floor:.0f}}}$",
             f"$10^{{-{log_floor / 2:.0f}}}$", "1"],
            r"$|\mathrm{D}\mathcal{J}/\mathrm{D}{\cdot}|$ / max"),
    add_bar(axes["c_d"], CM_SDF, [-0.3, 0, 0.3],
            ["$\u22120.3$", "0", "$\u22650.3$"], r"$s$", -0.3, 0.3),
]

fig.legend(
    handles=[Line2D([], [], color=INK, lw=1.5, label=r"$s=0$"),
             Line2D([], [], color=TEAL, lw=1.8,
                    label=r"$\overline{\varphi}=0.5$"),
             Patch(facecolor=TEAL, edgecolor="#777777", label="Dirichlet"),
             Patch(facecolor=ORANGE, edgecolor="#777777", label="Neumann"),
             Patch(facecolor=BODY, edgecolor="#777777", label="Body"),
             Patch(facecolor=VOID, edgecolor="#777777", label="Carved void")],
    loc="center", bbox_to_anchor=(0.790, 0.356), ncol=3, frameon=False,
    fontsize=6.3, columnspacing=0.9, handlelength=1.4, handletextpad=0.4)

# Evolution strip: every image is another surface from the same hidden-phi run.
strip_left, strip_right = 0.012, 0.988
strip_gap = 0.010
strip_w = ((strip_right - strip_left) - strip_gap * (len(evo_iters) - 1)) / len(evo_iters)
strip_bottom, strip_h = 0.040, 0.215
strip_axes = []
for index, it in enumerate(evo_iters):
    ax = fig.add_axes([strip_left + index * (strip_w + strip_gap),
                       strip_bottom, strip_w, strip_h])
    ax.imshow(evo[it], interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f"iter {it}", fontsize=6.3, labelpad=1.0, color=INK)
    strip_axes.append(ax)
fig.text(0.5, 0.282, r"$\bf{(f)}$  Design evolution", fontsize=8.0,
         ha="center", va="baseline", color=INK)

# Layout gates: titles, bars and legends must remain inside the canvas and no
# adjacent titles may collide after fonts are resolved.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
width_px, height_px = fig.canvas.get_width_height()
title_axes = [axes[f"{tag}_phi"] for tag in ("a", "b", "c")] + [axes["d"], axes["e"]]
title_boxes = [axis.title.get_window_extent(renderer=renderer) for axis in title_axes]
for box in title_boxes:
    assert box.x0 >= -1 and box.x1 <= width_px + 1 and box.y1 <= height_px + 1
for left, right in zip(title_boxes, title_boxes[1:]):
    assert right.x0 > left.x1, "adjacent void-nucleation titles overlap"
for bar in bars:
    assert bar.get_window_extent(renderer=renderer).y0 > 0

fig.savefig(HERE / "void_nucleation.pdf", dpi=600)
fig.savefig(HERE / "void_nucleation.png", dpi=400)
print("wrote void_nucleation.pdf and void_nucleation.png from hidden-phi panels")
print(f"  capture iter {capture_it}; {stats['dips_kept']} of {stats['dips_in']} "
      f"interior dips kept; cavity {stats['void_cells']} background cells")
