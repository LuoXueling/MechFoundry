"""One style + layout for the two objective-trajectory plots (so_objective, hp_objective).

Both plots are built by `draw()` so they cannot drift apart: same 88 x 55 mm page, same
two stacked panels sharing x (compliance on top, relative volume below), same ranges,
ticks, palette and fonts.  The tokens match Figure 1 / the loop figures
(`_shared/loop_chain.py`: ink #1a1a21, grey #5f6672, magenta #a3195b, 6.5 / 6 / 5.5 pt,
Type 42 Arial).  They are copied, not imported, so a redesign of loop_chain cannot
break these plots behind their back.

Layout is in mm on a fixed page (no tight_layout / bbox_inches='tight'): the PNG and PDF
are exactly 88 mm wide at final size, and the two plots sit side by side or stack
without rescaling.  `check()` fails the build on text under 5 pt, text off the page
(1 mm pad), two overlapping texts, or an annotation that touches a data curve.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.transforms import Bbox

for _f in ("Arial.ttf", "Arial_Bold.ttf", "Arial_Italic.ttf", "Arial_Bold_Italic.ttf"):
    try:
        font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/msttcorefonts/{_f}")
    except Exception:
        pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "mathtext.sf": "Arial", "mathtext.default": "it",
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2.0,
                     "ytick.major.size": 2.0, "xtick.major.pad": 1.5,
                     "ytick.major.pad": 1.5, "xtick.direction": "out",
                     "ytick.direction": "out", "savefig.transparent": False})

# ---- tokens -------------------------------------------------------------------------
INK = "#1a1a21"          # compliance
MAGENTA = "#a3195b"      # relative volume
GREY = "#5f6672"         # reference lines, annotation text, axes
ACTIVE_TINT = "#e7e2f5"  # "Hidden field active" region (light end of the phi-bar ramp)
LABEL_FS, NOTE_FS, TICK_FS, MIN_FS = 6.5, 6.0, 5.5, 5.0
DATA_LW, REF_LW = 0.9, 0.5
MARKER_PT = 2.5          # open marker on the shown iteration (diameter, pt)
DPI = 600
MM = 1.0 / 25.4

# ---- geometry (mm) --------------------------------------------------------------------
W, H = 88.0, 55.0
LEFT, RIGHT = 11.0, 3.0
BOTTOM, VOL_H, GAP, C_H = 8.4, 20.0, 2.6, 23.0     # top margin = 55 - 54 = 1.0
# VOL_H is 20 (not the review's ~18) so the 22.6 mm "Relative volume, V/V0" label
# overhangs its panel by only ~1.3 mm each side instead of reaching into the x-tick row.
X_MAX = 100
C_LIM, V_LIM = (0.9, 2.2), (0.45, 1.05)
X_TICKS = range(0, 101, 20)
C_TICKS, V_TICKS = (1.0, 1.5, 2.0), (0.5, 0.75, 1.0)
V_TARGET = 0.5

C_LABEL = r"Compliance, $C/C_0$"
V_LABEL = r"Relative volume, $V/V_0$"
X_LABEL = "Iteration"
TARGET_LABEL = r"$V/V_0$ = 0.5"


def read_traj(path):
    """(iterations, C/C0, V/V0) from a run's `iter se_raw vol_raw` trace."""
    t = np.loadtxt(path)
    return t[:, 0], t[:, 1] / t[0, 1], t[:, 2] / t[0, 2]


def _style(ax, lim, ticks):
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(*lim)
    ax.set_xticks(list(X_TICKS))
    ax.set_yticks(list(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v != int(v) else f"{v:.1f}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelcolor=INK, labelsize=TICK_FS)


def draw(stem, it, comp, vol, cap_it, cap_label, active_from=None,
         active_label="Hidden field active"):
    """Write `stem`.{png,pdf}.  `cap_it` is marked on the compliance curve and labelled
    `cap_label` at the foot of its dashed line; `active_from` shades [active_from, last
    iteration] in both panels (the hidden-field phase)."""
    fig = plt.figure(figsize=(W * MM, H * MM))
    fig.patch.set_facecolor("white")
    aw = (W - LEFT - RIGHT) / W
    ax_c = fig.add_axes([LEFT / W, (BOTTOM + VOL_H + GAP) / H, aw, C_H / H])
    ax_v = fig.add_axes([LEFT / W, BOTTOM / H, aw, VOL_H / H], sharex=ax_c)
    _style(ax_c, C_LIM, C_TICKS)
    _style(ax_v, V_LIM, V_TICKS)
    ax_c.tick_params(labelbottom=False)

    if active_from is not None:
        for ax in (ax_c, ax_v):
            ax.axvspan(active_from, it.max(), color=ACTIVE_TINT, lw=0, zorder=0)
        ax_c.text(it.max() - 1.0, C_LIM[1] - 0.04, active_label, fontsize=NOTE_FS,
                  color=GREY, ha="right", va="top")

    ax_c.plot(it, comp, color=INK, lw=DATA_LW, solid_joinstyle="miter", zorder=3)
    ax_v.plot(it, vol, color=MAGENTA, lw=DATA_LW, solid_joinstyle="miter", zorder=3)

    # target line, labelled once at its right end (above it, clear of the zig-zag)
    ax_v.axhline(V_TARGET, color=GREY, ls=(0, (1, 1.5)), lw=REF_LW, zorder=2)
    ax_v.text(X_MAX, V_TARGET + 0.05, TARGET_LABEL, fontsize=NOTE_FS, color=GREY,
              ha="right", va="bottom")

    # the shown / captured iteration: open marker + dashed drop line + label at its foot
    ci = int(np.argmin(np.abs(it - cap_it)))
    ax_c.plot([cap_it, cap_it], [C_LIM[0], comp[ci]], color=GREY, ls=(0, (3, 2)),
              lw=REF_LW, zorder=2)
    ax_c.plot([cap_it], [comp[ci]], ls="none", marker="o", ms=MARKER_PT,
              mfc="white", mec=INK, mew=0.6, zorder=4)
    ax_c.text(cap_it + 1.5, C_LIM[0] + 0.05, cap_label, fontsize=NOTE_FS, color=GREY,
              ha="left", va="bottom")

    ax_c.set_ylabel(C_LABEL, fontsize=LABEL_FS, color=INK)
    ax_v.set_ylabel(V_LABEL, fontsize=LABEL_FS, color=INK)
    ax_v.set_xlabel(X_LABEL, fontsize=LABEL_FS, color=INK, labelpad=1.5)
    for ax in (ax_c, ax_v):                       # one column for both y labels
        ax.yaxis.set_label_coords(-5.8 / (W - LEFT - RIGHT), 0.5)

    check(fig, (ax_c, ax_v))
    fig.savefig(f"{stem}.pdf", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png  ({W:.0f} x {H:.0f} mm)")


def check(fig, axes, pad_mm=1.0):
    """Gate: every text >= 5 pt, inside the page with pad_mm to spare, no two texts
    overlapping, and no free annotation (ax.texts) crossing a data curve."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi * MM                                       # pixels per mm
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.figure is fig]
    boxes = []
    for t in texts:
        assert t.get_fontsize() >= MIN_FS, f"text under {MIN_FS} pt: {t.get_text()!r}"
        b = t.get_window_extent(r)
        assert b.x0 >= pad_mm * px and b.y0 >= pad_mm * px and \
            b.x1 <= fig.bbox.width - pad_mm * px and b.y1 <= fig.bbox.height - pad_mm * px, \
            f"text within {pad_mm} mm of / off the page edge: {t.get_text()!r} {b}"
        boxes.append((t, b))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            assert not a.overlaps(b), \
                f"texts overlap: {boxes[i][0].get_text()!r} / {boxes[j][0].get_text()!r}"
    gap = 0.3 * px                                         # 0.3 mm clearance from curves
    for ax in axes:
        for t in ax.texts:
            b = t.get_window_extent(r)
            b = Bbox.from_extents(b.x0 - gap, b.y0 - gap, b.x1 + gap, b.y1 + gap)
            for ln in ax.lines:
                # the line's own transform: axhline lives in blended axes/data coords
                d = ln.get_transform().transform(ln.get_xydata())
                # densify each segment so a segment crossing the box is caught
                s = np.linspace(0, 1, 50)[:, None, None]
                pts = (d[:-1][None] * (1 - s) + d[1:][None] * s).reshape(-1, 2) \
                    if len(d) > 1 else d
                hit = (pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & \
                      (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)
                assert not hit.any(), \
                    f"annotation {t.get_text()!r} touches a plotted line"
    print(f"check: {len(boxes)} texts, all >= {MIN_FS} pt, on the page, no overlaps")
