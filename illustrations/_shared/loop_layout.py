"""The optimization-loop figure layout shared by shape_opt_merged, shape_opt_loop and
hidden_phi_loop, so the three cannot drift apart.

Panel a: two rows plus one legend row.  Every station is its render on WHITE inside a
thin light-grey rounded outline, its one-line title INSIDE at the top; stations are as
wide as the longest title ("Trimmed mesh + phi" at TFS).  Row 1, forward: Initial design
-> Geometry + BC -> Octree + s -> Trimmed mesh + phi -> Nitsche solve, u -> Objective J.
Row 2, read right to left: dJ/du, [dJ/dphibar], dJ/ds and the step station, centred under
Geometry + BC, whose return rises from its centre.  The magenta lane in the gap above row
2 is truthful: dJ/du -> (adjoint + auto-diff) -> a dot dJ/dphi -> (auto-diff) -> dJ/ds,
and dJ/dphibar, when present, branches off at the dot (dJ/dphi taken at phi = 1 on the
interior nodes); dJ/dphibar reaches the step's phibar half by a lane UNDER row 2.

phibar: None (shape optimization: no phibar at all), "optional" (dashed, the merged
figure) or "solid" (the hidden-phi method, where phibar is part of the method).

Panels below a: one row of six design-evolution frames each, on a's span, iteration above
each frame, the captured one bold, the frames gamma-lifted -- or (shape_opt_merged) a
PAIRED row, two runs side by side with three frames each (results given as a list of
two).  One legend row under a, justified to a's span.  A subclass may add rows under the
results (results_bottom(), shape_opt_merged's clean-up block).
"""

import os
import sys

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

import loop_chain as L

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "agentic_results"))
import ar_style as AR  # noqa: E402  the load arrows' colour (render_so/render_hp draw them)

OUTLINE = "#c9ced6"                         # the stations' quiet outline (no fills)
PHIBAR = r"$\overline{\varphi}$"

LET = 4.2                                   # panel-letter gutter
GAP = 4.5                                   # between stations: every arrow one glyph
BP = 0.6                                    # outline inner pad
TI = 3.0                                    # title band INSIDE the outline
RG = 9.0                                    # row 1 -> row 2: the backward lane runs here
LANE2 = 3.6                                 # row 2 -> the phibar lane under it
RADIUS = 0.76
TOP = 0.6
FGAP = 1.4
FLBL = 2.6                                  # a frame's label band, ABOVE the frame
TFS = 6.0                                   # station titles: one line each
# The Neumann key's arrow: Geometry + BC's 3-D load arrows (table_load_arrows.py) drawn
# flat, pointing down like them, in their colour and agentic_results' proportions (head
# 2.55x the shaft wide, 36 % of the length; ar_figure.bc_keys), one swatch tall-ish.
KEY_ARROW_LEN, KEY_ARROW_SHAFT = 2.3, 0.34
KEY_ARROW_GAP = 0.6                         # swatch -> arrow
# agentic_results' Support / Load key (ar_figure.bc_keys), for a figure that opts in with
# AR_BC_KEY (shape_opt_merged, round 13): its words, its 1.9 mm swatches edged in ink, and
# its flat load arrow pointing RIGHT (3.2 mm, shaft 0.46, head 2.55x the shaft wide and
# 36 % of the length), 0.7 mm after the swatch, labels 0.8 mm after the glyph.
AR_SW, AR_ARROW_GAP, AR_ARROW_LEN, AR_ARROW_SHAFT, AR_TXT = 1.9, 0.7, 3.2, 0.46, 0.8
LIFT = 0.72                                 # gamma applied to the result frames
PG = 7.0                                    # between the two runs of a paired row
PROW = 2.2                                  # between paired rows
TL2 = 2.9                                   # a paired run's second title line
PAIR_HMAX = 30.0                            # tallest frame in a paired row


def bg_box(img, thr=14, pad=10):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    bg = a[2, 2]
    ys, xs = np.where(np.abs(a - bg).sum(axis=2) > thr)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
            min(xs.max() + 1 + pad, a.shape[1]), min(ys.max() + 1 + pad, a.shape[0]))


def crop_common(paths):
    """One box for renders of ONE renderer/camera.  Never mix the two renderers:
    their images differ in size and PIL pads the smaller with black."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    bs = [bg_box(im) for im in ims]
    box = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs),
           max(b[3] for b in bs))
    return [np.asarray(im.crop(box)) for im in ims]


def lift(a, gamma=LIFT):
    """Brighten a result frame: gamma on every pixel except bluish ones (B - R > 6), so
    the light-grey body and ghost read brighter on white.  White stays white."""
    f = a.astype(np.float64) / 255.0
    out = f ** gamma
    keep = (a[..., 2].astype(int) - a[..., 0].astype(int)) > 6
    out[keep] = f[keep]
    return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def timing(path):
    """(iterations, seconds) from a frames-mode *_fr_stats.txt (save off, I/O out)."""
    st = dict(l.strip().split("=", 1) for l in open(path) if "=" in l)
    return int(st["n_frames"]), float(st["total_seconds"])


def stats_file(path):
    """so_stats.txt is key=value, hp_stats.txt is 'key value ...'."""
    out = {}
    for line in open(path):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
        elif line:
            k, *v = line.split()
            out[k] = v[0] if len(v) == 1 else v
    return out


class LoopFigure:
    AR_BC_KEY = False      # True: agentic_results' Support / Load key (ar_bc_key)
    """R: renders (cropped arrays) with keys init, geom, d, phi, u, gu, g, step and, when
    phibar is set, gphi and stepphi.  results: [(head, imgs, iterations, captured,
    (n_iterations, seconds))], one row each, lettered after a."""

    def __init__(self, R, results, phibar=None, phi_symbol=r"$\varphi$"):
        self.R, self.results, self.phibar, self.phi_symbol = R, results, phibar, phi_symbol
        pg = L.Page(10.0)
        self.B = pg.width(r"Trimmed mesh + $\varphi$", TFS) + 0.6 + 2 * BP
        # the pill is sized on the plain-J label it was designed with: the calligraphic
        # J (2026-10-29 notation pass) is ~0.3 mm wider and still clears the outline,
        # and measuring it instead would widen a's span and move every station
        self.jw = pg.width("Objective $J$", L.NOTE_FS) + 2.8
        B = self.B
        # row 2 sits S2 right of row 1 when the step station has two halves, so its
        # centre is under Geometry + BC's centre and its return rises straight up
        self.S2 = (GAP + B) / 2.0 if phibar else 0.0
        content = 5 * B + 4 * GAP + self.S2 + GAP + 1.0 + self.jw
        self.AL = max(L.ML + LET, (L.PAGE_W - content) / 2.0)
        self.AR = self.AL + content
        self.H = (B - 2 * BP) / L.aspect(R["d"])
        self.BH = TI + self.H + 2 * BP
        self.y = [TOP, TOP + self.BH + RG]
        self.lane2 = self.y[1] + self.BH + (LANE2 if phibar else 0.0)
        self.key_y = self.lane2 + 3.6
        self.key_rows = 1
        self.yb = self.key_y + 5.6
        # results: a row is one run of six frames (a tuple), or a PAIR of runs side by
        # side with three frames each (a list of two tuples, shape_opt_merged)
        self.paired = bool(results) and isinstance(results[0], list)
        if self.paired:
            self._pair_geometry()
            return
        self.fw = (self.AR - self.AL - 5 * FGAP) / 6.0
        self.fh = self.fw / L.aspect(results[0][1][0])
        self.fhs = [self.fh] * len(results)
        self.ph = 3.0 + FLBL + self.fh
        self.phs = [self.ph] * len(results)
        self.height = self.yb + len(results) * self.ph + (len(results) - 1) * 1.6 + 0.6

    def _pair_geometry(self):
        """Two runs per row on a's span, PG apart; each run three frames in a slot
        fwp wide.  A frame is as large as its slot allows up to PAIR_HMAX tall; a row
        is as tall as its taller run.  The title is two lines: the run in bold, then
        its iterations and wall clock in grey."""
        self.gw = (self.AR - self.AL - PG) / 2.0
        self.fwp = (self.gw - 2 * FGAP) / 3.0
        self.fdims = []                        # per row: [(w, h) per run]
        self.phs = []
        for row in self.results:
            dims = []
            for (_, imgs, _, _, _) in row:
                a = L.aspect(imgs[0])
                w, h = self.fwp, self.fwp / a
                if h > PAIR_HMAX:
                    w, h = PAIR_HMAX * a, PAIR_HMAX
                dims.append((w, h))
            self.fdims.append(dims)
            self.phs.append(3.0 + TL2 + FLBL + max(h for _, h in dims))
        self.fhs = [max(h for _, h in d) for d in self.fdims]
        self.fw, self.fh = self.fwp, self.fhs[0]
        self.height = self.yb + sum(self.phs) + (len(self.results) - 1) * PROW + 0.6

    def results_bottom(self):
        """The y (mm) where the last results row ends."""
        gap = PROW if self.paired else 1.6
        return self.yb + sum(self.phs) + (len(self.results) - 1) * gap

    # ---- geometry -----------------------------------------------------------------------
    def col(self, i, r=0):
        x = self.AL + i * (self.B + GAP) + (self.S2 if r == 1 else 0.0)
        return x, x + self.B

    def cx(self, i, r=0):
        return sum(self.col(i, r)) / 2.0

    def mid(self, r):
        return self.y[r] + TI + BP + self.H / 2.0

    # ---- primitives ---------------------------------------------------------------------
    def outline(self, pg, x0, y, w, h, dashed=False):
        pg.ax.add_patch(FancyBboxPatch(
            (x0, y), w, h, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
            fc="white", ec=L.INK if dashed else OUTLINE, lw=0.45,
            ls=(0, (2.4, 1.6)) if dashed else "-", zorder=0.5))

    def title(self, pg, x0, x1, y, s):
        t = pg.text((x0 + x1) / 2.0, y + BP + TI / 2.0 + L.CAP_EM * TFS * L.PT_MM / 2.0 + 0.2,
                    s, TFS, ha="center")
        tb = pg.box(t)
        assert tb[0] >= x0 + 0.3 and tb[2] <= x1 - 0.3, f"title leaves its outline: {s}"

    def block(self, pg, c, r, title, img, dashed=False):
        x0, x1 = self.col(c, r)
        y = self.y[r]
        self.outline(pg, x0, y, self.B, self.BH, dashed)
        self.title(pg, x0, x1, y, title)
        L.fit(pg, img, x0 + BP, y + BP + TI, self.B - 2 * BP, self.H, name=title)

    def above(self, pg, x0, x1, y, s):
        """A backward label directly ABOVE its horizontal arrow, centred on it."""
        return pg.text((x0 + x1) / 2.0, y - 0.9, s, L.TICK_FS, ha="center")

    def below(self, pg, x0, x1, y, s):
        return pg.text((x0 + x1) / 2.0, y + 0.9 + L.CAP_EM * L.TICK_FS * L.PT_MM, s,
                       L.TICK_FS, ha="center")

    # ---- panel a ------------------------------------------------------------------------
    def draw_a(self, pg, letter="a"):
        B, BH, y, R = self.B, self.BH, self.y, self.R
        dashed = self.phibar == "optional"
        m = [self.mid(r) for r in range(2)]
        pg.letter(self.AL - LET, TOP + 2.4, letter)
        for c, t, k in ((0, "Initial design", "init"), (1, "Geometry + BC", "geom"),
                        (2, "Octree + $s$", "d"), (3, r"Trimmed mesh + $\varphi$", "phi"),
                        (4, r"Nitsche solve, $\mathbf{u}$", "u")):
            self.block(pg, c, 0, t, R[k])
        for i in range(4):
            pg.arrow([(self.col(i)[1] + 0.3, m[0]), (self.col(i + 1)[0] - 0.3, m[0])])
        jx = self.AR - self.jw / 2.0
        h = 4.2
        pg.ax.add_patch(FancyBboxPatch((jx - self.jw / 2.0, m[0] - h / 2.0), self.jw, h,
                                       boxstyle=f"round,pad=0,rounding_size={h / 2.0}",
                                       fc="white", ec=OUTLINE, lw=0.45, zorder=0.5))
        pg.text(jx, m[0] + L.CAP_EM * L.NOTE_FS * L.PT_MM / 2.0, r"Objective $\mathcal{J}$", L.NOTE_FS,
                ha="center")
        assert self.AR <= L.PAGE_W - L.MR + 1e-6
        pg.arrow([(self.col(4)[1] + 0.3, m[0]), (jx - self.jw / 2.0 - 0.3, m[0])])
        # ---- row 2, read right to left --------------------------------------------------
        self.block(pg, 4, 1, r"$\partial\mathcal{J}/\partial\mathbf{u}$", R["gu"])
        if self.phibar:
            self.block(pg, 3, 1, r"$\mathrm{D}\mathcal{J}/\mathrm{D}$" + PHIBAR, R["gphi"], dashed=dashed)
        self.block(pg, 2, 1, r"$\mathrm{D}\mathcal{J}/\mathrm{D}s$", R["g"])
        # the step station, centred under Geometry + BC
        hw = B
        if self.phibar:
            sw = 2 * hw + 3 * BP
            sx0 = self.cx(1) - sw / 2.0
            self.outline(pg, sx0, y[1], sw, BH)
            xs_phi, xs_d = sx0 + BP, sx0 + 2 * BP + hw
            if dashed:
                self.outline(pg, xs_phi, y[1] + BP, hw, BH - 2 * BP, dashed=True)
            self.title(pg, xs_phi, xs_phi + hw, y[1] + 0.3, "Step on " + PHIBAR)
            self.title(pg, xs_d, xs_d + hw, y[1] + 0.3, "Step on $s$")
            L.fit(pg, R["stepphi"], xs_phi + BP, y[1] + BP + TI, hw - 2 * BP,
                  self.H - 2 * BP, name="stepphi")
            L.fit(pg, R["step"], xs_d + BP, y[1] + BP + TI, hw - 2 * BP, self.H - 2 * BP,
                  name="stepd")
        else:
            sw = hw
            sx0 = self.cx(1) - sw / 2.0
            self.outline(pg, sx0, y[1], sw, BH)
            self.title(pg, sx0, sx0 + sw, y[1], "Step on $s$")
            L.fit(pg, R["step"], sx0 + BP, y[1] + BP + TI, hw - 2 * BP, self.H, name="stepd")
        sx1 = sx0 + sw
        # J -> dJ/du: down and left into dJ/du's right edge
        xu1 = self.col(4, 1)[1]
        assert jx - xu1 >= 6.0, "the J elbow has no room"
        pg.arrow([(jx, m[0] + h / 2.0 + 0.3), (jx, m[1]), (xu1 + 0.3, m[1])], colour=L.BACK)
        self.above(pg, xu1, jx, m[1], "Auto-diff")
        # the main lane above row 2: dJ/du -> adjoint -> dJ/dphi -> auto-diff -> dJ/ds
        yl = y[0] + BH + RG / 2.0 + 0.4
        xu, xp, xd = (self.cx(i, 1) for i in (4, 3, 2))
        pg.arrow([(xu, y[1] - 0.3), (xu, yl), (xp + 0.9, yl)], colour=L.BACK)
        pg.ax.add_patch(Circle((xp, yl), 0.55, fc=L.BACK, ec="none", zorder=5))
        pg.text(xp, yl - 1.3, r"$\mathrm{D}\mathcal{J}/\mathrm{D}\varphi$", L.NOTE_FS, ha="center",
                color=L.BACK)
        self.above(pg, xp + 4.0, xu, yl, "Adjoint")
        self.below(pg, xp + 4.0, xu, yl, "auto-diff")
        pg.arrow([(xp - 0.9, yl), (xd, yl), (xd, y[1] - 0.3)], colour=L.BACK)
        self.above(pg, xd, xp - 4.0, yl, "Auto-diff")
        # dJ/ds -> the step (its s half)
        pg.arrow([(self.col(2, 1)[0] - 0.3, m[1]), (sx1 + 0.3, m[1])])
        if self.phibar:
            # dJ/dphibar branches off at dJ/dphi; it reaches the step's phibar half by
            # the lane UNDER row 2
            pg.arrow([(xp, yl + 0.9), (xp, y[1] - 0.3)], colour=L.BACK, dashed=dashed)
            xphi = xs_phi + hw / 2.0
            pg.arrow([(xp, y[1] + BH + 0.3), (xp, self.lane2), (xphi, self.lane2),
                      (xphi, y[1] + BH + 0.3)], dashed=dashed)
        # the step returns up into Geometry + BC from its centre, straight up
        xr = (sx0 + sx1) / 2.0
        assert abs(xr - self.cx(1)) < 1e-6
        assert not (xd <= xr <= xu), "the return crosses the lane"
        pg.arrow([(xr, y[1] - 0.3), (xr, y[0] + BH + 0.3)])
        # the legend row, justified to a's span
        n_gaps = 7 if dashed else 6
        scratch = L.Page(self.height)
        sp = 3.0
        if self.keys_a(scratch) == self.key_y:
            sp = 3.0 + (self.AR - self.key_end) / n_gaps
        if self.keys_a(pg, sp) > self.key_y:
            self.key_rows = 2

    def keys_a(self, pg, sp=3.0):
        """ONE legend row under a: the four bars, then the line keys."""
        bars = [("$s$", L.cmap_from(L.SDF_PTS), ["−0.3", "0.3"], None),
                (self.phi_symbol, L.CM_PHI, ["0", "1"], None),
                (r"$|\mathbf{u}|$", L.CM_SOLVE, ["0", "max"], None),
                (r"$|\mathrm{D}\mathcal{J}/\mathrm{D}{\cdot}|$", L.CM_GRAD, ["$10^{-4}$", "1"], "log")]
        bw = 13.0
        x = self.AL
        yb = self.key_y - L.BAR_H / 2.0 - 0.8
        for k, (sym, cmap, labs, note) in enumerate(bars):
            off = max(1.6, pg.width(labs[0], L.TICK_FS) / 2.0 + 0.8)
            t = pg.text(x, yb + L.BAR_H / 2.0 + L.CAP_EM * L.NOTE_FS * L.PT_MM / 2.0, sym,
                        L.NOTE_FS)
            bx = pg.box(t)[2] + off
            pg.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap,
                         extent=(bx, bx + bw, yb + L.BAR_H, yb), aspect="auto", zorder=4,
                         interpolation="bilinear")
            pg.ax.add_patch(Rectangle((bx, yb), bw, L.BAR_H, fc="none", ec=L.INK, lw=0.35,
                                      zorder=5))
            ticks = [(bx, labs[0]), (bx + bw, labs[1])]
            if k == 0:
                ticks.insert(1, (bx + bw / 2.0, "0"))
            for xv, lab in ticks:
                pg.line([xv, xv], [yb + L.BAR_H, yb + L.BAR_H + 0.4], 0.35, L.INK, z=5)
                pg.text(xv, yb + L.BAR_H + 0.6, lab, L.TICK_FS, ha="center", va="top")
            x = bx + bw + pg.width(labs[1], L.TICK_FS) / 2.0 + 0.4
            if note:
                t = pg.text(bx + bw + 0.9,
                            yb + L.BAR_H / 2.0 + L.CAP_EM * L.TICK_FS * L.PT_MM / 2.0, note,
                            L.TICK_FS, color=L.GREY)
                x = pg.box(t)[2]
            x += sp
        optional = self.phibar == "optional"
        if self.AR_BC_KEY:
            bc_w = (2 * (AR_SW + AR_TXT) + AR_ARROW_GAP + AR_ARROW_LEN
                    + pg.width("Support", L.TICK_FS) + sp + pg.width("Load", L.TICK_FS))
        else:
            bc_w = (2.2 + pg.width("Dirichlet", L.TICK_FS) + sp + 2.2 + KEY_ARROW_GAP
                    + 2.55 * KEY_ARROW_SHAFT + pg.width("Neumann", L.TICK_FS))
        need = (3.9 + pg.width("$s = 0$", L.TICK_FS) + sp + bc_w
                + ((sp + 3.9 + pg.width("Optional", L.TICK_FS)) if optional else 0.0))
        y = self.key_y
        if x + need > self.AR + 0.3:
            x, y = self.AL, self.key_y + 4.6
        xe = pg.line_key(x, y, [(L.INK, 0.9, "$s = 0$")], fs=L.TICK_FS)
        if self.AR_BC_KEY:
            xe = self.ar_bc_key(pg, xe + sp, y, gap=sp)
        else:
            xe = pg.swatch_key(xe + sp, y, [(L.TEAL, "Dirichlet")], fs=L.TICK_FS, gap=sp)
            xe = self.load_key(pg, xe + sp, y)
        if optional:
            xe = pg.line_key(xe + sp, y, [(L.INK, 0.75, "Optional", True)], fs=L.TICK_FS)
        self.key_end = xe
        assert xe <= self.AR + 0.05, "a's legend runs past a's right edge"
        return y

    def ar_bc_key(self, pg, x, y, gap=3.0):
        """agentic_results' key, ar_figure.bc_keys' glyphs on this figure's centre line:
        a teal "Support" swatch, then an orange "Load" swatch with the flat load arrow
        (ar_style.LOAD_ARROW) pointing right.  Returns the right end (mm)."""
        yc = y + L.CAP_EM * L.TICK_FS * L.PT_MM / 2.0       # labels' baseline, cap-centred
        pg.ax.add_patch(Rectangle((x, y - AR_SW / 2.0), AR_SW, AR_SW, fc=L.TEAL, ec=L.INK,
                                  lw=0.35, zorder=5))
        t = pg.text(x + AR_SW + AR_TXT, yc, "Support", L.TICK_FS)
        x = pg.box(t)[2] + gap
        pg.ax.add_patch(Rectangle((x, y - AR_SW / 2.0), AR_SW, AR_SW, fc=L.ORANGE, ec=L.INK,
                                  lw=0.35, zorder=5))
        sw, ln = AR_ARROW_SHAFT, AR_ARROW_LEN
        hw, hl = 2.55 * sw, 0.36 * ln
        x0 = x + AR_SW + AR_ARROW_GAP
        pg.ax.add_patch(Polygon([(x0, y - sw / 2), (x0 + ln - hl, y - sw / 2),
                                 (x0 + ln - hl, y - hw / 2), (x0 + ln, y),
                                 (x0 + ln - hl, y + hw / 2), (x0 + ln - hl, y + sw / 2),
                                 (x0, y + sw / 2)], closed=True, fc=AR.LOAD_ARROW,
                                ec="none", zorder=5))
        t = pg.text(x0 + ln + AR_TXT, yc, "Load", L.TICK_FS)
        return pg.box(t)[2]

    def load_key(self, pg, x, y, label="Neumann"):
        """The Neumann swatch (swatch_key's square) and, beside it, the load arrow as
        the Geometry + BC render draws it: pointing down, in ar_style.LOAD_ARROW.
        Returns the right end (mm)."""
        s = 1.5
        pg.ax.add_patch(Rectangle((x, y - s / 2.0), s, s, fc=L.ORANGE, ec="#888888",
                                  lw=0.35, zorder=5))
        sw, ln = KEY_ARROW_SHAFT, KEY_ARROW_LEN
        hw, hl = 2.55 * sw, 0.36 * ln
        xa = x + s + KEY_ARROW_GAP + hw / 2.0
        y0, y1 = y - ln / 2.0, y + ln / 2.0        # tail, tip (y down: the tip is lower)
        pg.ax.add_patch(Polygon([(xa - sw / 2, y0), (xa - sw / 2, y1 - hl), (xa - hw / 2, y1 - hl),
                                 (xa, y1), (xa + hw / 2, y1 - hl), (xa + sw / 2, y1 - hl),
                                 (xa + sw / 2, y0)], closed=True, fc=AR.LOAD_ARROW,
                                ec="none", zorder=5))
        t = pg.text(xa + hw / 2.0 + 0.7, y + L.CAP_EM * L.TICK_FS * L.PT_MM / 2.0,
                    label, L.TICK_FS)
        return pg.box(t)[2]

    # ---- the results rows ----------------------------------------------------------------
    def draw_pairs(self, pg, letters="bcdefg"):
        y = self.yb
        for k, row in enumerate(self.results):
            if k:
                y += self.phs[k - 1] + PROW
            pg.letter(self.AL - LET, y + 2.4, letters[k])
            rh = self.fhs[k]
            for g, (head, imgs, its, cap, tm) in enumerate(row):
                x0 = self.AL + g * (self.gw + PG)
                pg.text(x0, y + 2.4, head, TFS, weight="bold")
                pg.text(x0, y + 2.4 + TL2,
                        f"{tm[0]} iterations, {tm[1] / 60.0:.1f} min wall clock", TFS,
                        color=L.GREY)
                w, h = self.fdims[k][g]
                for j, (im, it) in enumerate(zip(imgs, its)):
                    xc = x0 + j * (self.fwp + FGAP) + self.fwp / 2.0
                    pg.text(xc, y + TL2 + 3.0 + 2.0,
                            f"Iteration {it}" if j == 0 else f"{it}", L.TICK_FS + 0.5,
                            ha="center", weight="bold" if it == cap else "normal")
                    # frames sit on the row's floor, centred in their slot
                    pg.img(im, xc - w / 2.0, y + TL2 + 3.0 + FLBL + (rh - h), w, h,
                           name=f"{letters[k]}{g}_{it}")

    def draw_results(self, pg, letters="bc"):
        if self.paired:
            return self.draw_pairs(pg, letters)
        for k, (head, imgs, its, cap, tm) in enumerate(self.results):
            y = self.yb + k * (self.ph + 1.6)
            pg.letter(self.AL - LET, y + 2.4, letters[k])
            t = pg.text(self.AL, y + 2.4, head, TFS, weight="bold")
            pg.text(pg.box(t)[2] + 1.6, y + 2.4,
                    f"{tm[0]} iterations, {tm[1] / 60.0:.1f} min wall clock", TFS,
                    color=L.GREY)
            for j, (im, it) in enumerate(zip(imgs, its)):
                xk = self.AL + j * (self.fw + FGAP)
                pg.text(xk + self.fw / 2.0, y + 3.0 + 2.0,
                        f"Iteration {it}" if j == 0 else f"{it}", L.TICK_FS + 0.5,
                        ha="center", weight="bold" if it == cap else "normal")
                pg.img(im, xk, y + 3.0 + FLBL, self.fw, self.fh, name=f"{letters[k]}{it}")

    def draw(self, pg):
        self.draw_a(pg)
        self.draw_results(pg)

    def build(self, stem):
        pg = L.Page(self.height)
        self.draw(pg)
        if self.key_rows == 2:                 # the legend wrapped: make room, draw again
            self.yb += 4.6
            self.height += 4.6
            pg = L.Page(self.height)
            self.draw(pg)
        pg.check()
        pg.save(stem)
        print(f"  stations {self.B:.1f} x {self.BH:.1f} mm, span {self.AL:.1f}-{self.AR:.1f} "
              f"mm; frames {self.fw:.1f} x "
              f"{' / '.join(f'{fh:.1f}' for fh in self.fhs)} mm; legend rows {self.key_rows}")
        return pg
