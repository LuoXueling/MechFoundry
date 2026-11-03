"""Compose step (conda `mechfoundry-warp117`): assemble the diffused-Nitsche figure.

Reads ``dn_panel_{a,b,c,d,e,f}.png`` + ``dn_stats.txt`` and writes
``diffused_nitsche.{pdf,png}``.  Every quoted number comes from the stats file,
i.e. from the production functions that ran in ``dn_prep.py``.

Layout: one row of six equal panels sharing a camera direction.  Each
colour bar sits under its own panel, titled with its variable — `d` under (a),
`phi` under (b) — and the region key sits under (c)-(e), so the footer is one
row tall (AGENTS.md §5b — one title line per panel).
"""

import matplotlib
matplotlib.use("Agg")

import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from PIL import Image, ImageOps

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

# Validated all-pairs (see render_dn.py): worst CVD dE 8.7, normal-vision 17.9.
GRAY = "#bababa"       # boundary region (neutral base state, recessive)
ORANGE = "#DD8452"     # Neumann (ears) — house orange, == bc_region_id
TEAL = "#12a594"       # Dirichlet (base) — replaces the house green, which
                       # collapsed with the orange under deuteranopia

stats = {}
for line in open("dn_stats.txt"):
    k, *v = line.split()
    stats[k] = v[0] if len(v) == 1 else v
# State the boundary-region test as the code writes it — a phi range — rather
# than as the |d| < 3.4534 eps it inverts to.  Written as plain decimals, not
# 10^-3: mathtext sets superscripts high off the baseline, which at 6.8 pt reads
# as a typo, and "0.001 < phi < 0.999" is both exact and shorter.  Derived from
# the stats file and checked, so the printed pair cannot misstate the real one.
phi_lo = float(stats["phi_lo"])
_e = int(round(math.log10(phi_lo)))
assert abs(10.0 ** _e - phi_lo) < 1e-15 * max(phi_lo, 1.0), \
    f"phi_lo {phi_lo} is not a round power of ten — write it out in full"
_dp = -_e
_lo_s, _hi_s = f"{phi_lo:.{_dp}f}", f"{1.0 - phi_lo:.{_dp}f}"
assert float(_lo_s) == phi_lo and float(_hi_s) == 1.0 - phi_lo, \
    "the printed decimals do not round-trip to the real thresholds"
PHI_RANGE = f"${_lo_s}<\\varphi<{_hi_s}$"

# --- the two colour maps, mirroring render_dn.py's transfer functions --------
SDF_VMIN, SDF_VMAX = -0.26, 0.72
SDF_PTS = [(SDF_VMIN, (0.106, 0.196, 0.478)), (-0.130, (0.235, 0.443, 0.718)),
           (-0.040, (0.557, 0.706, 0.851)), (0.0, (0.894, 0.894, 0.894)),
           (0.040, (0.968, 0.805, 0.576)), (0.150, (0.944, 0.600, 0.282)),
           (0.380, (0.827, 0.329, 0.152)), (SDF_VMAX, (0.478, 0.129, 0.094))]
SDF_CMAP = LinearSegmentedColormap.from_list(
    "sdf", [((v - SDF_VMIN) / (SDF_VMAX - SDF_VMIN), c) for v, c in SDF_PTS])
# Sequential single-hue blue, documented steps 100->700 (see render_dn.py).
PHI_PTS = [(0.0, (0.804, 0.886, 0.984)), (1 / 6, (0.620, 0.773, 0.957)),
           (2 / 6, (0.427, 0.655, 0.925)), (3 / 6, (0.224, 0.529, 0.898)),
           (4 / 6, (0.145, 0.416, 0.749)), (5 / 6, (0.094, 0.310, 0.584)),
           (1.0, (0.051, 0.212, 0.420))]
PHI_CMAP = LinearSegmentedColormap.from_list("phi", PHI_PTS)

TAGS = ["a", "b", "c", "d", "e", "f"]


def ink_box(img, tol=12):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = np.abs(a - 255).sum(axis=2) > tol
    ys, xs = np.where(ink)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def pad_to(img, aspect, bg="white"):
    """Letterbox onto a canvas of the given width/height ratio."""
    w, h = img.size
    W, H = ((w, int(round(w / aspect))) if w / h > aspect
            else (int(round(h * aspect)), h))
    canvas = Image.new("RGB", (max(W, w), max(H, h)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


# All five share one camera, so they must be cropped to a COMMON ink box or the
# cube would change size between panels.  Every panel draws the same cube
# outline, so that box is what really sets the crop — including (e), whose
# trimmed mesh sits inside it.
imgs = {t: Image.open(f"dn_panel_{t}.png").convert("RGB") for t in TAGS}
bs = [ink_box(imgs[t]) for t in TAGS]
PAD = 16
common = (max(min(b[0] for b in bs) - PAD, 0),
          max(min(b[1] for b in bs) - PAD, 0),
          min(max(b[2] for b in bs) + PAD, imgs["a"].width),
          min(max(b[3] for b in bs) + PAD, imgs["a"].height))
for t in TAGS:
    imgs[t] = pad_to(imgs[t].crop(common), 1.0)
# The surface panel has no cube outline, so the shared crop leaves its mostly
# white raster edge coincident with the axes edge; PDF rasterisation exposed
# that edge as a black hairline.  Give it its own ink crop and an explicit white
# guard band before fitting it to the same square station.
_f = Image.open("dn_panel_f.png").convert("RGB").crop(ink_box(
    Image.open("dn_panel_f.png").convert("RGB")))
imgs["f"] = pad_to(ImageOps.expand(_f, border=24, fill="white"), 1.0)

FIG_W = 7.16                      # double-column width
LEFT, RIGHT, WSPACE = 0.010, 0.990, 0.14
HEAD_IN = 0.28                    # one title line per panel
# Footer: the two colour bars sit under their own panel, (a) and (b), each
# titled with its variable; the region key sits under (c)-(e).  They occupy the
# same vertical band but disjoint columns, so the footer is one row tall.
FOOT_BAR_IN = 0.30      # bar bottom; its ticks + variable label hang 0.27 below
FOOT_KEY_IN = 0.10      # key baseline
FOOT_IN = 0.44

panel_w = (RIGHT - LEFT) * FIG_W / (6.0 + 5.0 * WSPACE)
panel_h = panel_w
fig_h = panel_h + HEAD_IN + FOOT_IN

fig = plt.figure(figsize=(FIG_W, fig_h))
gs = fig.add_gridspec(1, 6, left=LEFT, right=RIGHT, wspace=WSPACE,
                      top=1.0 - HEAD_IN / fig_h, bottom=FOOT_IN / fig_h)

TITLES = {
    "a": "Signed distance $s$",
    "b": r"Phase field $\varphi$",
    # Precise, not merely suggestive: the production test is per ELEMENT — a
    # cell joins the region when ANY of its 8 nodes is in range — so the drawn
    # cells reach up to one element beyond it.  Naming the region alone would
    # claim of every cell what only holds at a node (AGENTS.md §6).
    "c": "Boundary-region cells",
    # Likewise: a box hit is only the selector's stage-1 pre-filter, so this is
    # the selectors' intersection with the region, not "the boxes".
    "d": "BC facets & cells",
    "e": "Trimmed octree mesh",
    "f": "Deformed surface",
}
FS_TITLE = 6.3

axes = {}
for i, t in enumerate(TAGS):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.asarray(imgs[t]), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"$\\bf{{({t})}}$  {TITLES[t]}", fontsize=FS_TITLE, pad=4,
                 va="baseline")
    axes[t] = ax

# --- colour bars: d under (a), phi under (b), each titled by its variable ----
fig.canvas.draw()
CB_H = 0.050 / fig_h
CB_Y = FOOT_BAR_IN / fig_h
cbars = []
for tag, cmap, norm, ticks, label in (
        ("a", SDF_CMAP, Normalize(SDF_VMIN, SDF_VMAX), [-0.2, 0.2, 0.6], "$s$"),
        # phi is also what (c)-(e) are drawn over, but the bar belongs under the
        # panel that IS the phi field; the shared map reads from the panels.
        ("b", PHI_CMAP, Normalize(0.0, 1.0), [0.0, 0.5, 1.0], r"$\varphi$")):
    p = axes[tag].get_position()
    width = 0.80 * p.width
    cax = fig.add_axes([p.x0 + 0.5 * (p.width - width), CB_Y, width, CB_H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=6.0, length=2, pad=1.4)
    cb.set_label(label, fontsize=7.4, labelpad=1.0)
    cb.outline.set_linewidth(0.5)
    cbars.append(cb)

# --- region key, under (c)-(f) ------------------------------------------------
# The translucent geometry surface in (d) has no key: it is not a region like
# the other three, and naming it invited "what is that?" rather than answering
# it.  The caption carries it.
handles = [
    Patch(facecolor=GRAY, edgecolor="#9A9A9A", lw=0.5, label="Boundary region"),
    Patch(facecolor=ORANGE, edgecolor="none", label="Neumann (ears)"),
    Patch(facecolor=TEAL, edgecolor="none", label="Dirichlet (base)"),
]
_pc, _pe = axes["c"].get_position(), axes["f"].get_position()
_key_x = 0.5 * (_pc.x0 + _pe.x1)
leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                 fontsize=6.8, handlelength=1.4, columnspacing=1.5,
                 handletextpad=0.5, borderaxespad=0.0,
                 bbox_to_anchor=(_key_x, CB_Y))
# Centre the key on the BAR RECTANGLES, so the footer reads as one aligned row.
# Solve it by measurement, not by arithmetic on the anchor: matplotlib offsets a
# legend from a point anchor by `borderaxespad` (0.5 font-units — 0.024 figure
# heights at 6.8 pt), which silently lifted the key a bar-height above target.
# borderaxespad=0 removes it; measuring the residual and shifting by it is what
# actually guarantees the result, whatever padding conventions apply.
CB_MID = CB_Y + CB_H / 2


def chip_mid():
    """Vertical centre of the legend's colour CHIPS, not of its bbox.

    The bbox includes the labels, whose ascenders/descenders are not symmetric
    about the chip row, so centring the bbox leaves the chips a couple of pixels
    off the bars.  The chips are what the eye lines up.
    """
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    hs = getattr(leg, "legend_handles", None) or leg.legendHandles
    bs = [h.get_window_extent(r).transformed(inv) for h in hs]
    return 0.5 * (min(b.y0 for b in bs) + max(b.y1 for b in bs))


for _ in range(4):
    fig.canvas.draw()
    _resid = CB_MID - chip_mid()
    if abs(_resid) < 1e-4:
        break
    _lb = (leg.get_window_extent(fig.canvas.get_renderer())
           .transformed(fig.transFigure.inverted()))
    leg.set_bbox_to_anchor((_key_x, _lb.y0 + _resid), fig.transFigure)

# --- guard: nothing may run off the canvas or collide (AGENTS.md §7) ---------
fig.canvas.draw()
r = fig.canvas.get_renderer()


def bbox(art):
    return art.get_window_extent(r).transformed(fig.transFigure.inverted())


def cbar_extent(cb):
    """Bar + tick labels + axis label, as one box."""
    parts = [bbox(cb.ax), bbox(cb.ax.xaxis.label)]
    parts += [bbox(t) for t in cb.ax.get_xticklabels()]
    return (min(p.x0 for p in parts), min(p.y0 for p in parts),
            max(p.x1 for p in parts), max(p.y1 for p in parts))


checks = [("legend", leg)] + [(f"title {k}", ax.title) for k, ax in axes.items()]
checks += [(f"cbar tick {i}", t) for i, cb in enumerate(cbars)
           for t in cb.ax.get_xticklabels()]
checks += [(f"cbar label {i}", cb.ax.xaxis.label) for i, cb in enumerate(cbars)]
for name, art in checks:
    bb = bbox(art)
    assert -0.002 <= bb.x0 and bb.x1 <= 1.002, f"{name} overruns x: {bb}"
    assert -0.002 <= bb.y0 and bb.y1 <= 1.002, f"{name} overruns y: {bb}"

_t = sorted(((bbox(axes[k].title), k) for k in TAGS), key=lambda p: p[0].x0)
MIN_GAP = 0.008           # figure widths; touching titles read as one string
for (b0, k0), (b1, k1) in zip(_t, _t[1:]):
    assert b1.x0 - b0.x1 > MIN_GAP, \
        f"titles {k0}/{k1} too close ({b1.x0 - b0.x1:.4f})"

# Bars and key now share a vertical band, so vertical clearance is not enough —
# test the boxes in 2-D.
_lg = bbox(leg)
_boxes = [("cbar d", cbar_extent(cbars[0])), ("cbar phi", cbar_extent(cbars[1])),
          ("key", (_lg.x0, _lg.y0, _lg.x1, _lg.y1))]
for (n0, a), (n1, b) in itertools.combinations(_boxes, 2):
    overlap = (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
    assert not overlap, f"footer: {n0} overlaps {n1}"
# ... the key's chips are centred on the bars, ...
for i, cb in enumerate(cbars):
    _cb = bbox(cb.ax)
    assert abs(chip_mid() - 0.5 * (_cb.y0 + _cb.y1)) < 5e-4, \
        f"key chips are not vertically centred on colour bar {i}"
# ... and each bar must sit under the panel it belongs to.
for cb, tag in zip(cbars, ("a", "b")):
    p = axes[tag].get_position()
    e = cbar_extent(cb)
    assert p.x0 - 0.01 <= 0.5 * (e[0] + e[2]) <= p.x1 + 0.01, \
        f"colour bar is not centred under ({tag})"

fig.savefig("diffused_nitsche.pdf", dpi=600)
fig.savefig("diffused_nitsche.png", dpi=400)
print(f"wrote diffused_nitsche.pdf / .png  ({FIG_W:.2f} x {fig_h:.2f} in, "
      f"panel {panel_w:.2f} in)")
