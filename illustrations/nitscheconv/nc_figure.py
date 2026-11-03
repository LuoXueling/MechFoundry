"""Compose step (conda `mechfoundry-warp117`): the twisting-box convergence figure.

Reads ``nc_panel_{std,nit,rot,nit_phi,rot_phi}.png`` (render_nc.py) and
``nc_data.txt`` / ``nc_stats.txt`` (nc_prep.py); writes
``nitsche_convergence.{pdf,png}``.  Nothing here solves or measures anything —
every number on the page comes out of the sweep files.

Layout: a 3x2 block of panels on the left, the plot spanning its full height on
the right.  Top row, the solution: the body-fitted mesh and the two embedded
ones, deformed and coloured by |u|.  Bottom row, what the embedded scheme is
actually working with: each octree cut open over the phase field.  The cell that
would sit under the body-fitted panel holds the two colour bars — a body-fitted
mesh has no phase field, and the gap says so.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.transforms import Bbox
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

HERE = os.path.abspath(os.environ.get("NC_WORKDIR", os.path.dirname(os.path.abspath(__file__))))
os.chdir(HERE)
FIGURE_NAME = os.environ.get("NC_FIGURE_NAME", "nitsche_convergence")
UNIFORM_FIGURE = os.environ.get("NC_UNIFORM", "0").lower() not in {"0", "false", "no"}
SHARP_VOLUME_FIGURE = os.environ.get("NC_SHARP_VOLUME", "0").lower() not in {
    "0", "false", "no",
}

stats = {}
for line in open("nc_stats.txt"):
    k, _, v = line.strip().partition("=")
    if k:
        stats[k] = v
BOX = float(stats["box_side"])
LOAD_STEPS = int(float(stats["load_steps"]))
SHOW_DEPTH = int(float(stats["show_depth"]))   # the depth the panels are rendered at

COLS = ("variant depth h nelem ndof epsilon err n_nodes newton rel_residual "
        "secs err_nodal").split()
rows = []
for line in open("nc_data.txt"):
    if line.startswith("#") or not line.strip():
        continue
    v = line.split()
    assert len(v) == len(COLS), f"nc_data.txt has {len(v)} columns, expected {COLS}"
    rows.append(dict(zip(COLS, [v[0]] + [float(x) for x in v[1:]])))

# (d)/(e) name their element size because the panels do not look like they share
# one.  They do: at the shown depth the two are 0.0312 and 0.0352, 12.5 % apart
# and the closest a common depth allows (the rotated domain is 2.25 against 2.0,
# and the next aligned level down is 0.0625 — twice as coarse).  What differs is
# the SECTION: (e)'s cut plane meets its cells at 45°, so each cell's
# cross-section is sqrt(2) wider than the cell's edge and the drawn grid reads
# 1.6x coarser (measured: 13 px against 8 px).  The number settles it; the
# colour bar already carries the phi.
def _h_at(variant, depth):
    return next(r["h"] for r in rows
                if r["variant"] == variant and r["depth"] == depth)


ROWS = [
    [("a", "nc_panel_std.png", "Classical body-fitted FEM"),
     ("b", "nc_panel_nit.png", "Diffused Nitsche, aligned"),
     ("c", "nc_panel_rot.png", "Diffused Nitsche, rotated 45°")],
    [None,
     ("d", "nc_panel_nit_phi.png",
      f"Aligned octree, $h$ = {_h_at('nitsche', SHOW_DEPTH):.3f}"),
     ("e", "nc_panel_rot_phi.png",
      f"Rotated octree, $h$ = {_h_at('rotated', SHOW_DEPTH):.3f}")],
]
if UNIFORM_FIGURE:
    ROWS[0][1] = ("b", "nc_panel_nit.png", "Uniform Nitsche, aligned")
    ROWS[0][2] = ("c", "nc_panel_rot.png", "Uniform Nitsche, rotated 45°")
    ROWS[1][1] = ("d", "nc_panel_nit_phi.png",
                  f"Aligned uniform, $h$ = {_h_at('nitsche', SHOW_DEPTH):.3f}")
    ROWS[1][2] = ("e", "nc_panel_rot_phi.png",
                  f"Rotated uniform, $h$ = {_h_at('rotated', SHOW_DEPTH):.3f}")
if SHARP_VOLUME_FIGURE:
    ROWS[0][1] = ("b", "nc_panel_nit.png", "Sharp Nitsche, aligned")
    ROWS[0][2] = ("c", "nc_panel_rot.png", "Sharp Nitsche, rotated")
INK = "#1a1a21"
MUTED = "#5a6069"
# Three series, one per scheme.  The body-fitted baseline takes a dark neutral —
# it is what the other two are measured against, not a third competitor — and
# the two Nitsche variants take the house blue/orange pair.  Validated all-pairs
# by nc_palette_check.py (OKLab dE x100, Machado-Oliveira-Fernandes 2009 at
# severity 1.0): worst CVD pair 18.5, worst normal-vision pair 19.2, every line
# at least 4:1 against the white page.  Two house values had to move, and both
# for the same reason — a thin plot line has far less ink than a filled cell:
# the seaborn orange #DD8452 sits at 2.8:1 contrast (under the 3:1 gate), and a
# mid grey baseline (#59606b) is only dE 10.9 from the blue, under the
# normal-vision floor of 15.  Darkening both fixes both.
C_STD = "#3b4250"
C_NIT = "#4c72b0"
C_ROT = "#c8622a"
GRID = "#d7dade"

# The panels' two ramps, verbatim from render_nc.py (RGB interpolation of the
# same control points; the renderer walks them in Lab, which is a hair smoother
# but indistinguishable on a 0.1 in bar).  phi keeps diffused_nitsche's blue;
# |u| takes the warm ramp built on the same lightness ladder.
UMAX = 0.82
U_CMAP = LinearSegmentedColormap.from_list("umag", [
    "#fdd7c1", "#fbad7e", "#f48028", "#ca691f", "#a15216", "#7b3d0f"])
PHI_CMAP = LinearSegmentedColormap.from_list("phi", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

PANELS = [p for row in ROWS for p in row if p]
MISSING = [f for _, f, _ in PANELS if not os.path.exists(f)]
assert not MISSING, (f"panel PNGs missing: {MISSING} — run nc_prep.py then "
                     f"pvbatch render_nc.py (build.sh deletes them afterwards)")

SERIES = [
    ("standard", C_STD, "^", "Classical body-fitted FEM"),
    ("nitsche", C_NIT, "o", "Diffused Nitsche, aligned"),
    ("rotated", C_ROT, "s", "Diffused Nitsche, rotated"),
]
if SHARP_VOLUME_FIGURE:
    SERIES[1] = ("nitsche", C_NIT, "o", "Sharp-volume Nitsche, aligned")
    SERIES[2] = ("rotated", C_ROT, "s", "Sharp-volume Nitsche, rotated")
data = {k: np.array([[r["h"], r["err"]] for r in rows if r["variant"] == k])
        for k, *_ in SERIES}
for k, arr in data.items():
    assert len(arr) >= 3, f"{k}: only {len(arr)} points — need a curve"
    assert np.isfinite(arr).all(), f"{k}: a run failed (nan in nc_data.txt)"
    arr[:] = arr[np.argsort(arr[:, 0])]


def slope(arr):
    """Least-squares order of convergence over the whole curve."""
    return float(np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)[0])


# ---------------------------------------------------------------------------
# panel images.  One ink box PER ROW: the three solution panels share a camera
# and must stay at one scale, and so do the two phi panels — but the two rows do
# NOT share a framing (the phi panels frame the half that survives the cut), so
# cropping all five together would only shrink everything to the widest.
# ---------------------------------------------------------------------------
def ink_box(img, tol=12):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = np.abs(a - 255).sum(axis=2) > tol
    ys, xs = np.where(ink)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def pad_to(img, aspect, bg="white"):
    w, h = img.size
    W, H = ((w, int(round(w / aspect))) if w / h > aspect
            else (int(round(h * aspect)), h))
    canvas = Image.new("RGB", (max(W, w), max(H, h)), bg)
    canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
    return canvas


PAD = 10
imgs = {}
for row in ROWS:
    ps = [p for p in row if p]
    raw = {t: Image.open(f) for t, f, _ in ps}
    # A render that clipped its object against the viewport cannot be rescued
    # here — the geometry is already gone — so catch it at the seam: ink that
    # reaches a border means the camera was too tight (AGENTS.md §5).
    for (t, _f, _ti), im in zip(ps, raw.values()):
        x0, y0, x1, y1 = ink_box(im)
        assert x0 > 1 and y0 > 1 and x1 < im.width - 1 and y1 < im.height - 1, \
            f"panel {t} is clipped by its own render: ink box {(x0, y0, x1, y1)} " \
            f"in {im.size} — dolly the camera back"
    b = np.array([ink_box(im) for im in raw.values()])
    W, H = next(iter(raw.values())).size
    box = (max(int(b[:, 0].min()) - PAD, 0), max(int(b[:, 1].min()) - PAD, 0),
           min(int(b[:, 2].max()) + PAD, W), min(int(b[:, 3].max()) + PAD, H))
    for t, im in raw.items():
        imgs[t] = im.crop(box)
# ONE aspect for all five, so every panel is the same size — a grid of unequal
# cells reads as a hierarchy that is not there.  It costs the phi panels some
# side padding (their ink is squarer than the solution renders'), which is the
# price of the grid.
_AR = max(im.width / im.height for im in imgs.values())
AR = {ri: _AR for ri in range(len(ROWS))}
for t in list(imgs):
    imgs[t] = pad_to(imgs[t], _AR)

# ---------------------------------------------------------------------------
# layout (inches)
# ---------------------------------------------------------------------------
FIG_W = 7.16              # house double-column width
PANEL_W = 1.18            # the plot is the deliverable; the meshes are its key
GAP = 0.07                # between panel columns
ROW_GAP = 0.05            # between a row's panels and the next row's titles
BREAK = 0.30              # between the panel block and the plots
TITLE_IN = 0.19
MARGIN = 0.06
PLOT_L, PLOT_B, PLOT_T = 0.44, 0.32, 0.19   # room for the plots' own labels
                                            # (0.32 is measured: the tick row plus a
                                            # 7 pt x label hang ~0.30 in below the axes)
G_TITLE, G_GAP = 0.19, 0.10                 # the cost panel's title and the gap above it
LEGEND_IN = 0.36                            # the shared key beneath both plots

panel_w = PANEL_W
row_h = {ri: panel_w / AR[ri] for ri in AR}
block_w = 3 * panel_w + 2 * GAP
fig_h = (sum(TITLE_IN + row_h[ri] for ri in row_h) + ROW_GAP + 2 * MARGIN)
plot_w = FIG_W - block_w - BREAK - 2 * MARGIN
assert plot_w > 1.8, f"plot column is only {plot_w:.2f} in wide"

fig = plt.figure(figsize=(FIG_W, fig_h))
fx, fy = 1.0 / FIG_W, 1.0 / fig_h


def cell(r, c):
    """Bottom-left corner of panel cell (row r from the top, column c), inches."""
    left = MARGIN + c * (panel_w + GAP)
    top = fig_h - MARGIN - TITLE_IN
    for ri in range(r):
        top -= row_h[ri] + ROW_GAP + TITLE_IN
    return left, top - row_h[r]


axes = {}
for r, row in enumerate(ROWS):
    for c, p in enumerate(row):
        if p is None:
            continue
        t, _f, title = p
        left, bottom = cell(r, c)
        ax = fig.add_axes([left * fx, bottom * fy, panel_w * fx, row_h[r] * fy])
        ax.imshow(np.asarray(imgs[t]), interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"$\\bf{{({t})}}$ {title}", fontsize=6.4, pad=3.5,
                     va="baseline", color=INK)
        axes[t] = ax

# --- the two colour bars, in the cell under the body-fitted panel ------------
# A body-fitted mesh has no phase field, so that cell is empty by construction;
# the bars take it.  |u| sits at the top of the cell, against the row it serves,
# and phi below it, against its own.
bars = []
cb_left, cb_bottom = cell(1, 0)
BAR_W, BAR_H = 0.86 * panel_w, 0.075
for i, (cmap, vmax, ticks, label) in enumerate((
        (U_CMAP, UMAX, [0.0, 0.4, 0.8], r"Displacement $|\mathbf{u}|$"),
        (PHI_CMAP, 1.0, [0.0, 0.5, 1.0], r"Phase field $\varphi$"))):
    y = cb_bottom + row_h[1] - 0.30 - i * 0.46
    cax = fig.add_axes([(cb_left + 0.5 * (panel_w - BAR_W)) * fx, y * fy,
                        BAR_W * fx, BAR_H * fy])
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(0.0, vmax)),
                      cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=6.2, color=INK, labelpad=1.5)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=6.0, colors=INK, length=2.0, width=0.6, pad=1.2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#9aa0a8")
    bars.append(cb)

# ---------------------------------------------------------------------------
# the plots: accuracy on top, cost below, sharing one x axis
# ---------------------------------------------------------------------------
plot_l = MARGIN + block_w + BREAK + PLOT_L
plot_wid = plot_w - PLOT_L
# EQUAL heights.  They are two measurements of the same twelve runs, and the key
# that names those runs now sits under both of them rather than inside the top
# one — so neither panel has to carry the key's headroom, and neither is the
# senior partner.
h_avail = fig_h - 2 * MARGIN - LEGEND_IN - PLOT_B - PLOT_T - G_TITLE - G_GAP
h_f = h_g = h_avail / 2.0

ax = fig.add_axes([plot_l * fx,
                   (MARGIN + LEGEND_IN + PLOT_B + h_g + G_TITLE + G_GAP) * fy,
                   plot_wid * fx, h_f * fy])
axg = fig.add_axes([plot_l * fx, (MARGIN + LEGEND_IN + PLOT_B) * fy,
                    plot_wid * fx, h_g * fy])
for key, colour, marker, label in SERIES:
    arr = data[key]
    ax.plot(arr[:, 0], arr[:, 1], color=colour, marker=marker, ms=3.8, lw=1.6,
            mew=0.0, label=label, zorder=3,
            clip_on=False)

# --- the cost panel: the same runs, timed ------------------------------------
# One quantity per axis, never two (dataviz: no dual-axis charts).  The Newton
# count is nearly flat in h (40-72 Nitsche, 73-100 body-fitted across the four
# figures), so a curve for it would be a horizontal line pretending to be data; it
# goes in a short note instead, computed from the same column.
for key, colour, marker, _label in SERIES:
    t = np.array([[r["h"], r["secs"]] for r in rows if r["variant"] == key])
    t = t[np.argsort(t[:, 0])]
    axg.plot(t[:, 0], t[:, 1], color=colour, marker=marker, ms=3.4, lw=1.4,
             mew=0.0, zorder=3, clip_on=False)


def _it_range(*keys):
    n = sorted({int(r["newton"]) for r in rows if r["variant"] in keys})
    return f"{n[0]}" if len(n) == 1 else f"{n[0]}–{n[-1]}"


# Both Nitsche series count: reading the aligned one alone reported "40" on the
# adaptive sharp-volume figure, whose rotated depth-6 run took 74.
_nitsche_kind = "sharp-volume Nitsche" if SHARP_VOLUME_FIGURE else "diffused Nitsche"
# Three short lines: as one line it spans the whole axis and its left end runs
# into the coarsest points.  Where it goes is decided after the draw, against
# the curves themselves (_NOTE_SPOTS below).
newton_note = axg.text(0.0, 0.0, f"Newton iterations\n"
                       f"{_it_range('nitsche', 'rotated')} {_nitsche_kind}\n"
                       f"{_it_range('standard')} body-fitted",
                       transform=axg.transAxes, fontsize=5.8, color=MUTED,
                       linespacing=1.25)

hs = sorted({r["h"] for r in rows if r["variant"] == "standard"})
for a in (ax, axg):
    a.set_xscale("log")
    a.set_yscale("log")
    a.invert_xaxis()          # refinement runs left -> right
    a.xaxis.set_major_locator(FixedLocator(hs))
    a.xaxis.set_major_formatter(FixedFormatter(
        [f"1/{int(round(1/h))}" if h < 1 else f"{h:g}" for h in hs]))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.tick_params(labelsize=6.4, colors=INK, length=2.4, width=0.6, pad=1.6)
    a.tick_params(which="minor", length=1.3, width=0.5)
    a.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_linewidth(0.6)
        a.spines[side].set_color("#9aa0a8")
# The two share an axis, so only the lower one carries its ticks and its label.
ax.tick_params(axis="x", labelbottom=False)
# The box side belongs on the axis, not in the caption: h on its own says
# nothing about how many elements are across the object, and converting between
# the two is the first thing a reader does.
axg.set_xlabel(f"Element size  $h$   (box size {BOX:g}×{BOX:g}×{BOX:g})",
               fontsize=7.0, color=INK, labelpad=2)
_tmax = max(r["secs"] for r in rows)
axg.set_ylim(top=_tmax * 1.5)
axg.set_ylabel("Solve time (s)", fontsize=7.0, color=INK, labelpad=2)
axg.set_title(f"$\\bf{{(g)}}$ Cost of {LOAD_STEPS} load steps",
              fontsize=6.4, pad=3.5, va="baseline", color=INK)
ax.set_ylabel(r"Relative $L^2$ error in $\mathbf{u}$", fontsize=7.0, color=INK, labelpad=2)
# A little headroom, and no more: the key used to live in the top-right corner
# of this panel and needed a decade of empty space bought for it.  Out of the
# axes, the curves get the height back.
_ymax = max(arr[:, 1].max() for arr in data.values())
ax.set_ylim(top=_ymax * 1.7)
# The key names the three schemes and nothing else: the fitted orders used to
# ride along in the labels, which made every row taller than the mathtext it
# carried and pushed the key into the curves.  They belong in the caption and in
# nc_data.txt, not in a legend that has to fit under a 2.6 in column.
# One column, three rows.  Three of these labels side by side need 3.7 in and
# the column is 2.6; and at two columns matplotlib fills column-major, which put
# "rotated" above "aligned" when read across.  A stacked key is unambiguous and
# costs 0.1 in of panel height.
_cx = (plot_l + 0.5 * plot_wid) * fx
leg = fig.legend(*ax.get_legend_handles_labels(), fontsize=6.0, frameon=False,
                 loc="lower center", bbox_to_anchor=(_cx, MARGIN * fy),
                 ncol=1, handlelength=1.4, handletextpad=0.45,
                 columnspacing=1.1, labelspacing=0.3, borderaxespad=0.0)
for txt in leg.get_texts():
    txt.set_color(INK)
ax.set_title("$\\bf{(f)}$ Convergence to the body-fitted reference",
             fontsize=6.4, pad=3.5, va="baseline", color=INK)

fig.canvas.draw()
_r = fig.canvas.get_renderer()


for t, a in axes.items():
    p = a.get_position()
    assert p.x0 >= -0.002 and p.x1 <= 1.002, f"panel {t} overruns x: {p}"
    assert p.y0 >= -0.002 and p.y1 <= 1.002, f"panel {t} overruns y: {p}"


def bbox(art):
    return art.get_window_extent(_r).transformed(fig.transFigure.inverted())


# The Newton note has to sit where no curve is, and three hand-picked spots
# have each failed on some variant: top left under a single-line note (the
# rotated curve's last point), bottom right as one line (the coarsest points),
# bottom right as three lines (it is half the panel tall and ran into the 1/16
# points of the uniform runs).  So measure: the first spot whose box clears every
# plotted segment and marker wins, and none clearing is an error.  Top left comes
# first because cost rises with 1/h, so the coarse end of the top is the corner
# every variant leaves empty.
_NOTE_SPOTS = [((0.03, 0.96), "left", "top"),
               ((0.97, 0.05), "right", "bottom"),
               ((0.50, 0.96), "center", "top")]


def _note_hits_curve(note):
    # Display space, where the log-log polylines really are straight segments.
    # Padded by the largest marker radius plus a point of air.
    pad = (max(l.get_markersize() for l in axg.get_lines()) / 2 + 1) * fig.dpi / 72
    box = note.get_window_extent(_r).padded(pad)
    for line in axg.get_lines():
        xy = line.get_transform().transform(line.get_xydata())
        s = np.linspace(0.0, 1.0, 64)[:, None]
        for p, q in zip(xy[:-1], xy[1:]):
            if box.count_contains(p + s * (q - p)):
                return True
    return False


for (x, y), ha, va in _NOTE_SPOTS:
    newton_note.set(x=x, y=y, ha=ha, va=va)
    if not _note_hits_curve(newton_note):
        break
else:
    raise AssertionError("the Newton note overlaps a curve at every candidate spot")


# The key is out of the axes now, so what has to be checked is that it clears
# the x label above it and the page below it.  (When it lived inside (f) the
# check was that no curve ran through it — three placements failed that, and
# only measuring caught the last one.)
_leg_bb = Bbox.union([a.get_window_extent(_r) for a in
                      list(leg.legend_handles) + list(leg.get_texts())]
                     ).transformed(fig.transFigure.inverted())
assert _leg_bb.y0 > 0.0, f"the key runs off the bottom: {_leg_bb}"
assert _leg_bb.y1 < bbox(axg.xaxis.label).y0, "the key runs into the x label"
assert _leg_bb.x0 > (MARGIN + block_w) * fx, "the key reaches the panel block"

for row in ROWS:
    tags = [p[0] for p in row if p]
    for t0, t1 in zip(tags, tags[1:]):
        assert bbox(axes[t0].title).x1 < bbox(axes[t1].title).x0, \
            f"titles ({t0}) and ({t1}) overlap"
# The plots sit close to the block now (0.30 in), so a panel title on the right
# of a row can reach their y labels.  Check the widest offender directly.
_plot_ink = min(bbox(a.yaxis.label).x0 for a in (ax, axg))
for row in ROWS:
    t_last = [p[0] for p in row if p][-1]
    assert bbox(axes[t_last].title).x1 < _plot_ink, \
        f"title ({t_last}) reaches the plot column"
# A row-2 title must clear the row-1 panel above it.
for t in [p[0] for p in ROWS[1] if p]:
    assert bbox(axes[t].title).y1 < axes[ROWS[0][0][0]].get_position().y0, \
        f"title ({t}) runs into the row above"
arts = ([(a.title, f"title {t}") for t, a in axes.items()]
        + [(ax.title, "plot title"), (ax.yaxis.label, "y label"),
           (axg.title, "cost title"), (axg.xaxis.label, "x label"),
           (axg.yaxis.label, "cost y label")]
        + [(cb.ax.xaxis.label, f"bar {i} label") for i, cb in enumerate(bars)])
for art, name in arts:
    b = bbox(art)
    assert 0.0 <= b.x0 and b.x1 <= 1.0, f"{name} runs off x: {b}"
    assert 0.0 <= b.y0 and b.y1 <= 1.0, f"{name} runs off y: {b}"
# The bars live in a panel cell; they must not spill into what is beside them.
_cb_right = max(bbox(cb.ax).x1 for cb in bars)
assert _cb_right < axes["d"].get_position().x0, "a colour bar reaches panel (d)"
assert bbox(bars[0].ax).y1 < axes["a"].get_position().y0, \
    "the |u| bar runs into panel (a)"

# The two plots share an x axis, so they must actually line up.
assert abs(ax.get_position().x0 - axg.get_position().x0) < 1e-9 and \
    abs(ax.get_position().x1 - axg.get_position().x1) < 1e-9, \
    "(f) and (g) are not on the same x span"
assert bbox(axg.title).y1 < ax.get_position().y0, "(g)'s title runs into (f)"

for ext in ("pdf", "png"):
    fig.savefig(f"{FIGURE_NAME}.{ext}", dpi=400, facecolor="white")
print(f"wrote {FIGURE_NAME}.pdf / .png")
print("  orders: " + "  ".join(f"{k}={slope(data[k]):.2f}" for k, *_ in SERIES))
print("  times:  " + "  ".join(
    f"{k}={max(r['secs'] for r in rows if r['variant'] == k):.0f}s max" for k, *_ in SERIES))
