"""Shared page, tokens and drawing primitives for the two optimisation-loop figures.

    shapeopt/so_figure.py   -> shape_opt_loop.{pdf,png}
    hiddenphi/hp_figure.py  -> hidden_phi_loop.{pdf,png}

Both figures are composed on ONE grid language so they stack into a single
180 mm page (Nature double column): a forward/backward CHAIN on top (panel a,
the conceptual key), then the forward panels under the chain's stations, then
the backward / update panels, then the design-evolution strip.

Tokens are Figure 1's (``overview/ov_style.py`` + ``ov_figure.py``) and the
agentic_results / showcase figures': Arial; ink #1a1a21, grey text #5f6672;
backward magenta #a3195b; forward arrows #30343b, 0.75 pt shafts with a slender
concave (stealth) head and rounded corners; the FEA & optimisation tint
#ebf5e6 for the chain's stations, edge-less; colour bars as agentic_results'
``hbar`` (0.35 pt ink frame).  Every length here is in millimetres, y DOWN.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overview"))
import ov_style as S  # noqa: E402

for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
    try:
        matplotlib.font_manager.fontManager.addfont(
            f"/usr/share/fonts/truetype/msttcorefonts/{_f}")
    except Exception:  # noqa: BLE001
        pass
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     # the objective is a calligraphic J (\mathcal{J}, the paper's
                     # notation): Computer Modern's cmsy10 holds the TeX calligraphic
                     # capitals at the ASCII letters; the default "cursive" family
                     # resolves to Comic Sans on this machine
                     "mathtext.cal": "cmsy10",
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.transparent": False})

# ---- tokens -------------------------------------------------------------------
INK = S.PALETTE["ink"]                 # #1a1a21
GREY = S.PALETTE["grey text"]          # #5f6672
BACK = S.PALETTE["backward"]           # #a3195b  backward arrows + gradient panels
FWD = "#30343b"                        # Figure 1's forward arrow
FEA_TINT = S.PARTS["fea"]["fill"]      # #ebf5e6  chain stations
TEAL = S.PALETTE["Dirichlet (base)"]   # #12a594  Dirichlet, and phibar = 0.5
ORANGE = S.PALETTE["Neumann (ears)"]   # #dd8452  Neumann
BODY = "#ebe6dc"
VOID = "#4f5866"
FRAME = S.FRAME                        # #c3c9d2  hairlines
PREV = "#9ea1a8"                       # the previous s = 0 (render: 0.62/0.63/0.66)

PAGE_W = 180.0
DPI = 600
PT_MM = 25.4 / 72.0
CAP_EM = 0.716                         # Arial cap height (em)

LETTER_FS = 8.0                        # Nature panel letters: 8 pt bold lower case
TITLE_FS = 6.5                         # station titles == Figure 1's TITLE_FS
CHAIN_FS = 6.5                         # chain station text
NOTE_FS = 6.0                          # arrow labels (chain), keys
TICK_FS = 5.5                          # colour-bar ticks, plot ticks
MIN_FS = 5.0

ARROW_LW = 0.75                        # pt
HL, HW, NOTCH = 1.57, 0.61, 0.28       # mm: Figure 1's 0.062 / 0.024 in head
R_CORNER = 1.14                        # mm: Figure 1's 0.045 in corner radius
DOT_R = 0.42                           # mm: backward-arrow tail dot (the chain sketch)

BAR_H = 1.1                            # colour bar height (mm)


def cmap_from(points):
    """[(value, (r, g, b)), ...] -> a colormap over the values' span."""
    lo, hi = points[0][0], points[-1][0]
    return LinearSegmentedColormap.from_list(
        "c", [((v - lo) / (hi - lo), c) for v, c in points])


def flat_to_points(flat):
    return [(flat[i], tuple(flat[i + 1:i + 4])) for i in range(0, len(flat), 4)]


# ramps, as the renderers draw them
SDF_PTS = [(-0.300, (0.106, 0.196, 0.478)), (-0.192, (0.235, 0.443, 0.718)),
           (-0.084, (0.557, 0.706, 0.851)), (0.000, (0.894, 0.894, 0.894)),
           (0.048, (0.968, 0.805, 0.576)), (0.132, (0.944, 0.600, 0.282)),
           (0.216, (0.827, 0.329, 0.152)), (0.300, (0.478, 0.129, 0.094))]
CM_PHI = cmap_from(flat_to_points(S.PHI_RGB))
CM_GRAD = cmap_from(flat_to_points(S.GRAD_RGB))
CM_U = plt.get_cmap("plasma")
# |u|: the paper's displacement ramp (agentic_results / showcase, ar_style.SOLVE_*)
CM_SOLVE = cmap_from([(0.0, (0.920, 0.900, 0.860)), (0.35, (0.920, 0.900, 0.860)),
                      (0.70, (0.867, 0.518, 0.322)), (1.0, (0.478, 0.129, 0.094))])


# ---- the page ------------------------------------------------------------------
class Page:
    """A matplotlib figure addressed in millimetres, y measured DOWN from the top."""

    def __init__(self, h, w=PAGE_W):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 25.4, h / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_axis_off()
        self.texts = []
        self.rects = {}                 # name -> (x0, y0, x1, y1) of placed images

    # text --------------------------------------------------------------------------
    def text(self, x, y, s, fs, **kw):
        assert fs >= MIN_FS - 1e-9, f"text below {MIN_FS} pt: {s!r}"
        kw.setdefault("color", INK)
        kw.setdefault("va", "baseline")
        t = self.ax.text(x, y, s, fontsize=fs, zorder=6, **kw)
        self.texts.append(t)
        return t

    def width(self, s, fs, **kw):
        t = self.ax.text(0, 0, s, fontsize=fs, **kw)
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        t.remove()
        return bb.width / DPI * 25.4

    def box(self, t):
        """A text's (or artist's) ink box in mm: x0, top, x1, bottom."""
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        return (bb.x0 / DPI * 25.4, self.h - bb.y1 / DPI * 25.4,
                bb.x1 / DPI * 25.4, self.h - bb.y0 / DPI * 25.4)

    def letter(self, x, y, s):
        return self.text(x, y, s, LETTER_FS, weight="bold")

    # images -------------------------------------------------------------------------
    def img(self, a, x, y, w, h, name=None):
        self.ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=1)
        if name:
            self.rects[name] = (x, y, x + w, y + h)

    # shapes -------------------------------------------------------------------------
    def line(self, xs, ys, lw, color, z=3, **kw):
        self.ax.plot(xs, ys, lw=lw, color=color, solid_capstyle="butt", zorder=z, **kw)

    def station(self, cx, cy, w, h, s, fs=CHAIN_FS, dashed=False, pill=False):
        """A chain station: Figure 1's light FEA block, no edge.  An OPTIONAL station
        (the hidden phase field) carries a dashed 0.5 pt ink outline: Figure 1's
        convention, where dashes mean optional ("dJ/dphi (Optional)")."""
        r = h / 2.0 if pill else 0.9
        self.ax.add_patch(FancyBboxPatch(
            (cx - w / 2.0, cy - h / 2.0), w, h,
            boxstyle=f"round,pad=0,rounding_size={r:.3f}",
            fc=FEA_TINT, ec=INK if dashed else "none", lw=0.5 if dashed else 0,
            ls=(0, (2.4, 1.6)) if dashed else "-", zorder=3))
        return self.text(cx, cy + CAP_EM * fs * PT_MM / 2.0, s, fs, ha="center")

    def arrow(self, pts, colour=FWD, dashed=False, head=True, tail_dot=False, r=R_CORNER):
        """Figure 1's arrow: shaft through pts with rounded corners, stealth head."""
        pts = [np.asarray(p, float) for p in pts]
        tip = pts[-1]
        d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
        shaft = list(pts)
        if head:
            shaft[-1] = tip - d * HL * (1.0 - NOTCH * 0.5)
        verts, codes = [shaft[0]], [Path.MOVETO]
        for i in range(1, len(shaft) - 1):
            a, b, c = shaft[i - 1], shaft[i], shaft[i + 1]
            la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
            ri = min(r, la / 2.0, lc / 2.0)
            verts += [b - (b - a) / la * ri, b, b + (c - b) / lc * ri]
            codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
        verts.append(shaft[-1])
        codes.append(Path.LINETO)
        self.ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=colour,
                                    lw=ARROW_LW, capstyle="butt", joinstyle="round",
                                    ls=(0, (3.2, 2.2)) if dashed else "-", zorder=4))
        if head:
            n = np.array([-d[1], d[0]])
            self.ax.add_patch(Polygon([tip, tip - d * HL + n * HW,
                                       tip - d * HL * (1.0 - NOTCH), tip - d * HL - n * HW],
                                      closed=True, fc=colour, ec=colour, lw=0.3,
                                      joinstyle="miter", zorder=4))
        if tail_dot:
            self.ax.add_patch(Circle(tuple(pts[0]), DOT_R, fc=colour, ec="none", zorder=4))

    def hbar(self, x, y, w, cmap, ticks, labels, symbol, vmin=0.0, vmax=1.0):
        """agentic_results' bar: the ramp in a 0.35 pt ink frame, ticks below, the
        quantity's symbol to the LEFT of the bar on its centre line."""
        self.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap,
                       extent=(x, x + w, y + BAR_H, y), aspect="auto", zorder=4,
                       interpolation="bilinear")
        self.ax.add_patch(Rectangle((x, y), w, BAR_H, fc="none", ec=INK, lw=0.35, zorder=5))
        tb = y + BAR_H + 0.8
        for v, lab in zip(ticks, labels):
            xv = x + (v - vmin) / (vmax - vmin) * w
            self.line([xv, xv], [y + BAR_H, y + BAR_H + 0.45], 0.35, INK, z=5)
            self.text(xv, tb, lab, TICK_FS, ha="center", va="top")
        off = max(0.9, self.width(labels[0], TICK_FS) / 2.0 + 0.8)
        sym = self.text(x - off, y + BAR_H / 2.0 + CAP_EM * NOTE_FS * PT_MM / 2.0,
                        symbol, NOTE_FS, ha="right")
        return sym

    def swatch_key(self, x, y, entries, fs=NOTE_FS, gap=2.2):
        """One row of square swatches + labels; returns the right end (mm)."""
        s = 1.5
        for col, lab in entries:
            self.ax.add_patch(Rectangle((x, y - s / 2.0), s, s, fc=col, ec="#888888",
                                        lw=0.35, zorder=5))
            t = self.text(x + s + 0.7, y + CAP_EM * fs * PT_MM / 2.0, lab, fs)
            x = self.box(t)[2] + gap
        return x - gap

    def line_key(self, x, y, entries, fs=NOTE_FS, gap=2.6):
        """Line glyphs (iso-curves) + labels; entries (colour, lw, label[, dashed]);
        returns the right end (mm)."""
        for e in entries:
            col, lw, lab = e[:3]
            dashed = len(e) > 3 and e[3]
            self.line([x, x + 3.2], [y, y], lw, col, z=5,
                      **({"ls": (0, (1.6, 1.0))} if dashed else {}))
            t = self.text(x + 3.9, y + CAP_EM * fs * PT_MM / 2.0, lab, fs)
            x = self.box(t)[2] + gap
        return x - gap

    # checks -------------------------------------------------------------------------
    def check(self, allow_overlap=()):
        """Every text on the page, inside it, and no two texts overlapping."""
        boxes = [(t, self.box(t)) for t in self.texts if t.get_text().strip()]
        for t, (x0, y0, x1, y1) in boxes:
            assert x0 >= -0.05 and x1 <= self.w + 0.05 and y0 >= -0.05 and y1 <= self.h + 0.05, \
                f"text leaves the page: {t.get_text()!r} {x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                if a[0] < b[2] - 0.05 and b[0] < a[2] - 0.05 and a[1] < b[3] - 0.05 \
                        and b[1] < a[3] - 0.05:
                    pair = {boxes[i][0].get_text(), boxes[j][0].get_text()}
                    assert pair in allow_overlap, f"texts overlap: {pair}"

    def save(self, stem):
        self.fig.savefig(f"{stem}.pdf", dpi=DPI)
        self.fig.savefig(f"{stem}.png", dpi=DPI)
        print(f"wrote {stem}.pdf / .png  ({self.w:.1f} x {self.h:.1f} mm)")


# ---- images ----------------------------------------------------------------------
def ink_box(img, thr=12):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ys, xs = np.where(np.abs(a - 255).sum(axis=2) > thr)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def common_crop(paths, pad=10):
    """Crop a set of renders from ONE camera to their common ink box (§5b): panels
    of the same object then stay the same size."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    b = [ink_box(im) for im in ims]
    w0, h0 = ims[0].size
    box = (max(min(x[0] for x in b) - pad, 0), max(min(x[1] for x in b) - pad, 0),
           min(max(x[2] for x in b) + pad, w0), min(max(x[3] for x in b) + pad, h0))
    return [np.asarray(im.crop(box)) for im in ims]


def own_crop(path, pad=10):
    return common_crop([path], pad)[0]


def aspect(a):
    return a.shape[1] / a.shape[0]


def fit(pg, a, x, y, w, h, name=None):
    """An image as large as fits the w x h box, centred in it; returns its rect."""
    r = aspect(a)
    pw, ph = (w, w / r) if w / r <= h else (h * r, h)
    x0, y0 = x + (w - pw) / 2.0, y + (h - ph) / 2.0
    pg.img(a, x0, y0, pw, ph, name=name)
    return x0, y0, x0 + pw, y0 + ph


# ---- the objective station -----------------------------------------------------------
def objective_history(traj_path, w_vol, target, k=500.0):
    """(iterations, J, SE/SE_0) from the run's own (se_raw, vol_raw) trace, J built
    exactly as ShapeOpt._objective_function builds it:
        J = SE/SE_0 + w_vol * softplus_k((V/V_0 - f) / (1 - f)),
    softplus_k(x) = log(1 + exp(k x)) / k with the kernel's +-20 cut-offs."""
    t = np.loadtxt(traj_path)
    it, se, vol = t[:, 0], t[:, 1], t[:, 2]
    x = (vol / vol[0] - target) / (1.0 - target)
    kx = k * x
    sp = np.where(kx > 20, x, np.where(kx < -20, 0.0,
                                       np.log1p(np.exp(np.clip(kx, -20, 20))) / k))
    return it.astype(int), se / se[0] + w_vol * sp, se / se[0]


def draw_objective(pg, x, y, w, h, it, J, se, mark_it):
    """J over the run (ink), SE/SE_0 alone (faint), the captured iteration marked.
    Every label is a Page text, so the page gates see it."""
    ax = pg.fig.add_axes([x / pg.w, 1.0 - (y + h) / pg.h, w / pg.w, h / pg.h])
    ytop = float(np.ceil(J.max()))
    ax.plot(it, se, color=GREY, lw=0.5, alpha=0.75, solid_joinstyle="round")
    ax.plot(it, J, color=INK, lw=0.75, solid_joinstyle="round")
    k = list(it).index(mark_it)
    ax.axvline(mark_it, color=FRAME, lw=0.5, zorder=0)
    ax.plot([mark_it], [J[k]], "o", ms=2.4, mfc=INK, mec="white", mew=0.4, zorder=5)
    ax.set_xlim(0, it.max())
    ax.set_ylim(0, ytop)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_linewidth(0.35)
        ax.spines[sp_].set_color(INK)
    ax.tick_params(length=1.4, width=0.35, labelsize=TICK_FS, pad=1.0, colors=INK)
    # ticks clear of the captured iteration's label (hidden-phi: 43 vs a 45 tick)
    xt = [0, 50, 100] if it.max() >= 100 else [0, 30, 60, 90]
    ax.set_xticks(xt)
    ax.set_yticks([0, ytop])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("none")
    X = lambda v: x + v / it.max() * w            # noqa: E731
    Y = lambda v: y + h - v / ytop * h            # noqa: E731
    for v in xt:
        pg.text(X(v), y + h + 1.0, f"{v}", TICK_FS, ha="center", va="top")
    for v in (0, ytop):
        pg.text(x - 0.9, Y(v) + CAP_EM * TICK_FS * PT_MM / 2.0, f"{v:g}", TICK_FS, ha="right")
    pg.text(x + w / 2.0, y + h + 3.4, "Iteration", TICK_FS, ha="center", va="top")
    # SE/SE_0 labelled where it starts, under its line (it merges with J once the
    # one-sided volume penalty switches off)
    pg.text(X(0.04 * it.max()), Y(se[0]) + 2.2, r"$\mathrm{SE}/\mathrm{SE}_0$", TICK_FS,
            ha="left", color=GREY)
    pg.text(X(mark_it) + 1.0, Y(J[k]) - 1.0, f"{mark_it}", TICK_FS, ha="left",
            weight="bold")
    return ax


# ---- the shared grid ---------------------------------------------------------------
ML, MR = 1.0, 1.0
GUTTER = 4.2                           # the panel-letter column
GAP = 5.0                              # between panel columns (row arrows live here)
NCOLS = 5                              # BOTH figures: their stations line up stacked
PANEL_ASPECT = 1.35                    # panel box w/h; the octree ink box is 1.253, so the
#                                        panels fill the height and sit ~2 mm narrower
TITLE_DROP = 2.2                       # row top -> title baseline
PANEL_DROP = 3.0                       # row top -> panel top
ROW_GAP = 1.0                          # panel bottom -> next row top
EVO_SCALE = 0.62                       # strip frames at 62 % of the full-width height
EVO_GAP = 1.4
BOX_H = 5.0                            # chain station height
LANE_DY = 3.9                          # station centre -> backward lane
LBL_DY = 2.95                          # lane -> its labels' baseline


class Grid:
    """ncols equal columns across the page.  The STACKED pair uses the common
    5-column grid (stations line up); shape_opt_loop standalone uses a compact
    4-column grid with 6 mm gaps (no whitespace columns)."""

    def __init__(self, ncols=NCOLS, gap=GAP, aspect_=PANEL_ASPECT):
        self.n, self.gap = ncols, gap
        self.x0 = ML + GUTTER
        self.W = (PAGE_W - self.x0 - MR - (ncols - 1) * gap) / ncols
        self.H = self.W / aspect_

    def left(self, i):
        return self.x0 + i * (self.W + self.gap)

    def cx(self, i):
        return self.left(i) + self.W / 2.0

    def right(self, i):
        return self.left(i) + self.W


def title(pg, cx, y, s, colour=INK, max_w=None):
    t = pg.text(cx, y, s, TITLE_FS, ha="center", color=colour)
    if max_w is not None:
        x0, _, x1, _ = pg.box(t)
        assert x1 - x0 <= max_w + 0.2, f"title wider than its column: {s!r} {x1 - x0:.1f} mm"
    return t


def row_arrow(pg, x0, x1, y):
    pg.arrow([(x0 + 0.5, y), (x1 - 0.5, y)])


def chain_core(pg, g, cy, cols, names, update_label):
    """The forward line and backward lane every loop figure shares.

    cols = column indices of (d, phi, u, J); names = their station texts.
    Forward: d -> phi -> (Nitsche solve) -> u -> J.  Backward (magenta, dot tails,
    the user's sketch): J -> u auto-diff, u -> phi adjoint state method + auto-diff,
    phi -> d auto-diff, arriving as dJ/ds; then a thin grey RETURN from dJ/ds up
    into d, labelled with the update that closes the iteration.
    Returns {name: (left, right)} of the boxes and the lane / label y."""
    fs = CHAIN_FS
    bw = [pg.width(s, fs) + 3.6 for s in names]
    ext = {}
    for c, s, w in zip(cols, names, bw):
        pg.station(g.cx(c), cy, w, BOX_H, s, pill=(s is names[-1]))
        ext[c] = (g.cx(c) - w / 2.0, g.cx(c) + w / 2.0)
    d, ph, u, J = cols
    for a, b in ((d, ph), (ph, u), (u, J)):
        pg.arrow([(ext[a][1] + 0.6, cy), (ext[b][0] - 0.6, cy)])
    pg.text((ext[ph][1] + ext[u][0]) / 2.0, cy - 0.9, "Nitsche solve", NOTE_FS, ha="center")
    lane, lbl = cy + LANE_DY, cy + LANE_DY + LBL_DY
    cap = CAP_EM * NOTE_FS * PT_MM / 2.0
    for src, dst, lab, lx in ((J, u, "Auto-diff", None),
                              (u, ph, "Adjoint state method + auto-diff", None),
                              (ph, d, "Auto-diff", (g.cx(d + 1) + g.cx(ph)) / 2.0
                               if ph - d > 1 else None)):
        x_tail, x_head = ext[src][0] + 2.0, ext[dst][1] - 2.0
        pg.arrow([(x_tail, lane), (x_head, lane)], colour=BACK, tail_dot=True)
        pg.text(lx if lx is not None else (x_tail + x_head) / 2.0, lbl, lab, NOTE_FS,
                ha="center")
    grad = pg.text(ext[d][1] - 2.0 - HL - 0.5, lane + cap, r"$\mathrm{D}\mathcal{J}/\mathrm{D}s$",
                   NOTE_FS, ha="right", color=BACK)
    gx0 = pg.box(grad)[0]
    x_v = g.left(d) + 0.7
    pg.arrow([(gx0 - 0.5, lane), (x_v, lane), (x_v, cy), (ext[d][0] - 0.5, cy)], colour=GREY)
    # the label sits against the hook: under its horizontal, starting at its
    # vertical stroke (the remesh / re-initialise / BC detail is in legend_notes)
    pg.text(x_v - 0.2, lbl, update_label, NOTE_FS, ha="left", color=GREY)
    return ext, lane, lbl


def evo_height(imgs):
    """Strip frame height: EVO_SCALE of what six frames across the page would get."""
    return (PAGE_W - ML - GUTTER - MR - 5 * 2.0) / 6 / aspect(imgs[0]) * EVO_SCALE


def evo_strip(pg, x0, y, imgs, iters, letter, bold_it=None, labels=None):
    """The design at successive iterations, labelled above; returns (right, bottom)."""
    h = evo_height(imgs)
    w = h * aspect(imgs[0])
    tb = y + TITLE_DROP
    pg.letter(ML, tb, letter)
    for k, (im, it) in enumerate(zip(imgs, iters)):
        xk = x0 + k * (w + EVO_GAP)
        pg.img(im, xk, y + PANEL_DROP, w, h, name=f"evo_{it}")
        lab = labels[k] if labels else f"Iteration {it}"
        pg.text(xk + w / 2.0, tb, lab, TITLE_FS, ha="center",
                weight="bold" if it == bold_it else "normal")
    return x0 + len(imgs) * w + (len(imgs) - 1) * EVO_GAP, y + PANEL_DROP + h


def key_block(pg, x0, x1, y0, y1, bars, lines):
    """The figure's one key band (right of the strip): colour bars in a column on the
    left, line glyphs and swatches on the right.  bars: (symbol, cmap, ticks, labels,
    vmin, vmax, note); lines: ("line", colour, lw, label, dashed) or ("swatch", colour,
    label)."""
    pitch = (y1 - y0) / len(bars)
    sym_w = max(pg.width(b[0], NOTE_FS) for b in bars)
    off = max(max(1.0, pg.width(b[3][0], TICK_FS) / 2.0 + 0.6) for b in bars)
    bx = x0 + sym_w + off
    bw = 13.0
    for k, (sym, cmap, ticks, labels, vmin, vmax, note) in enumerate(bars):
        yb = y0 + k * pitch
        pg.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap,
                     extent=(bx, bx + bw, yb + BAR_H, yb), aspect="auto", zorder=4,
                     interpolation="bilinear")
        pg.ax.add_patch(Rectangle((bx, yb), bw, BAR_H, fc="none", ec=INK, lw=0.35, zorder=5))
        for v, lab in zip(ticks, labels):
            xv = bx + (v - vmin) / (vmax - vmin) * bw
            pg.line([xv, xv], [yb + BAR_H, yb + BAR_H + 0.4], 0.35, INK, z=5)
            pg.text(xv, yb + BAR_H + 0.8, lab, TICK_FS, ha="center", va="top")
        pg.text(bx - off, yb + BAR_H / 2.0 + CAP_EM * NOTE_FS * PT_MM / 2.0, sym, NOTE_FS,
                ha="right")
        if note:
            pg.text(bx + bw + 1.0, yb + BAR_H / 2.0 + CAP_EM * TICK_FS * PT_MM / 2.0, note,
                    TICK_FS, color=GREY)
    lx = bx + bw + 1.0 + max([pg.width(b[6], TICK_FS) for b in bars if b[6]] + [0]) + 2.4
    lp = (y1 - y0) / len(lines)
    for k, e in enumerate(lines):
        yc = y0 + (k + 0.5) * lp
        if e[0] == "line":
            pg.line_key(lx, yc, [(e[1], e[2], e[3], e[4])])
        else:
            pg.swatch_key(lx, yc, [(e[1], e[2])])
    return lx


def key_bars(phi_symbol):
    """The key band's colour bars, shared by both loop figures."""
    return [("$s$", cmap_from(SDF_PTS), [-0.3, 0, 0.3], ["−0.3", "0", "0.3"], -0.3, 0.3, None),
            (phi_symbol, CM_PHI, [0, 0.5, 1], ["0", "0.5", "1"], 0, 1, None),
            (r"$|\mathbf{u}|$", CM_SOLVE, [0, 1], ["0", "max"], 0, 1, None),
            (r"$|\mathrm{D}\mathcal{J}/\mathrm{D}{\cdot}|$/p99", CM_GRAD, [0, 0.5, 1],
             ["$10^{-4}$", "$10^{-2}$", "≥1"], 0, 1, "log")]
