"""Compose shape_opt_merged.{pdf,png} (conda `mechfoundry-warp117`): ONE figure for SDF
shape optimization, with the hidden phase field as an OPTIONAL branch.

    python som_figure.py            # from illustrations/shapeopt (build.sh shapeoptmerged)

Panel a spans the full width: the optimization loop.  Every station is its render on
WHITE inside a thin light-grey rounded outline with its one-line title INSIDE, at the
top; an OPTIONAL station has the same outline, dashed.  Forward arrows ink, backward
magenta.  Rows by variable:

  [iter 0] -> Geometry + BC -> Octree + d -> Trimmed mesh + phi -> Nitsche solve, u -> J
               ^  ^                                                                  |
               :  +--- Step on d  <-  dJ/dd  <-----  dJ/dphi  <-----  dJ/du  <-------+
               :                                       :
               +- -  Step on phibar  <-  dJ/dphibar <- +            (row 3, dashed)

The lower rows sit SHIFT mm right of the forward row, so the two returns enter Geometry
+ BC at separate points through separate lanes: d's step rises straight up, phibar's step
runs left of the lower rows and up.  The backward chain: dJ/du (auto-diff of the taped
integrals), dJ/dphi (adjoint state method -- AbstractField.step's adjoint_fn, K lambda =
-dJ/du -- plus auto-diff of the residual), dJ/dd (auto-diff of the taped tanh map);
dJ/dphibar is dJ/dphi at phi = 1 on interior nodes (legend notes).  Labels: "Auto-diff"
above its arrow; "Adjoint" above / "auto-diff" below its arrow.  a's colour bars sit in
row 3's free right part, the line keys along a's foot.

b, one paired row: the table's shape optimization (0, the bold capture 40, 100) beside
its concurrent topology and shape optimization (20, the bold capture 43, 90), three
frames each, a run's title on two lines (the run in bold, iterations and wall clock in
grey); frames gamma-lifted at compose time.

c, "Refining generated shapes" -- the geometry clean-up (experiments/geometry_cleanup)
of 17 agentic-workflow bodies, each as two panels from one camera -- its supports and
loads on the starting body (agentic_results' look) and the final body coloured by its
signed change from the start in leaves h (the clean-up's own map) -- in the four rows of
ROWS_C on a's span, one panel height for all, each labelled on ONE line: the name, then
SE/SE0 in grey, a small pipeline arrow from the first panel to the second.  c's key
line sits under its rows: Support, Load, and the change bar (-3 h_min ... 3 h_min,
titled "Change in shape" above it).  Renders: som_cleanup_prep.py ->
render_som_cleanup.py (panel_som_cu_*.png, som_cleanup.json).
"""

import itertools
import json
import os
import sys

import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, "..", "hiddenphi")
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import loop_chain as L  # noqa: E402
import loop_layout as LL  # noqa: E402

STEM = "shape_opt_merged"

# ---- c: the clean-up block ----------------------------------------------------------
# Four rows, all at ONE panel height, each justified to a's span.  With one-line labels
# (~20 mm each) the 17 are label-bound: three rows hold them at 10.3 mm (the narrow
# bodies -- femur, hammer, pogo stick -- become slivers under a 21 mm label), four at
# 16.7 mm (c 92.5 mm tall vs 52.1 mm).  Round 9's "SE/SE0 = x" labels shrink these
# rows to 15.6 mm; the user asked to keep them (a fresh search would give 16.4 mm with
# the objects reshuffled).  Round 10's arrow gap (IG 0.4 -> 2.4 mm) takes them to
# 14.6 mm, still kept (a fresh search: 15.5 mm, reshuffled).  The split is the
# one that gives the largest common height under the user's two constraints (round 8,
# 2026-10-14): the robot arm, the femur and the dental crown in the first row, the
# birthday cake and the crocodile NOT in it.  `python som_figure.py --search K` re-runs
# the search (the best K-row split meeting those constraints, slot = max(its two
# panels, its one-line label), OG between objects; search_rows) and prints the rows.
ORDER_C = ["crocodile", "rhinoceros", "claw-hammer", "layered-birthday-cake", "elephant",
           "horse", "human-femur", "school-chair-with-a-steel-frame-and-a-plastic-se",
           "park-bench", "wooden-rocking-horse", "industrial-robot-arm",
           "car-suspension-control-arm-with-rubber-bushings", "natural-stone-arch",
           "pantheon", "dental-crown", "pogo-stick", "truck-leaf-spring"]
FIRST_ROW = ("industrial-robot-arm", "human-femur", "dental-crown")
NOT_FIRST = ("layered-birthday-cake", "crocodile")
ROWS_C = [["claw-hammer", "human-femur", "industrial-robot-arm", "pantheon",
           "dental-crown"],
          ["crocodile", "elephant", "school-chair-with-a-steel-frame-and-a-plastic-se",
           "car-suspension-control-arm-with-rubber-bushings"],
          ["rhinoceros", "park-bench", "wooden-rocking-horse", "pogo-stick"],
          ["layered-birthday-cake", "horse", "natural-stone-arch", "truck-leaf-spring"]]
IG = 2.4                  # between an object's two panels: room for the BC -> change arrow
ARROW_K = 0.6             # that arrow: a's pipeline arrow (loop_chain.Page.arrow) at 0.6 x
ARROW_PAD = 0.35          # its clearance from either panel
OG = 2.6                  # between objects, at least (rows are justified by widening it)
LBL = 2.3                 # an object's one label line (name + SE/SE0), ABOVE its panels
NAME_FS, STAT_FS = L.TICK_FS + 0.5, L.MIN_FS
LBL_GAP = 1.0             # name -> its SE/SE0
# matplotlib anchors the page at its BOTTOM edge, so the page height's fractional pixel
# decides where a and b fall on the pixel grid.  Kept at the round-7 figure's value
# (157.178 mm = 3712.859 px at 600 dpi) by < 1 px of white under c's key: a and b then
# rasterise pixel-identically whatever c's height.
PX_PHASE = 0.859312
CHANGE_MAX = 3.0          # geometry_cleanup/render.CHANGE_MAX: the map saturates at +-3 h
# geometry_cleanup/render.lut() (RGB interpolation): blue = removed, red = added,
# white within +-0.25 h
CHANGE_PTS = [(-3.0, (0.02, 0.19, 0.55)), (-1.0, (0.25, 0.50, 0.85)),
              (-0.25, (0.97, 0.97, 0.97)), (0.25, (0.97, 0.97, 0.97)),
              (1.0, (0.90, 0.45, 0.25)), (3.0, (0.60, 0.05, 0.05))]


def p_so(name):
    return os.path.join(HERE, name)


def p_hp(name):
    return os.path.join(HP, name)


def check_change_lut():
    """CHANGE_PTS == geometry_cleanup/render.lut()'s RGBPoints, read from its source."""
    import ast
    src = os.path.join(HERE, "..", "..", "experiments", "geometry_cleanup", "render.py")
    tree = ast.parse(open(src).read())
    mx = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", None) == "CHANGE_MAX")
    pts = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "attr", None) == "RGBPoints")
    vals = [mx * (-1 if isinstance(e, ast.UnaryOp) else 1) if
            (isinstance(e, ast.Name) or (isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Name)))
            else ast.literal_eval(e) for e in pts.elts]
    want = [v for x, c in CHANGE_PTS for v in (x, *c)]
    assert mx == CHANGE_MAX and np.allclose(vals, want), f"change ramp drifted: {vals}"


def cleanup_objects(rows):
    """[(record, (bc, change) cropped to ONE box)], row by row."""
    recs = {o["name"]: o for o in json.load(open(p_so("som_cleanup.json")))["objects"]}
    order = [n for row in rows for n in row]
    assert sorted(order) == sorted(recs), "the rows and som_cleanup.json name different objects"
    out = []
    for name in order:
        o = recs[name]
        ims = [Image.open(p_so(f"panel_som_cu_{name}_{k}.png")).convert("RGB")
               for k in ("bc", "change")]
        bs = [LL.bg_box(im, pad=6) for im in ims]
        box = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs),
               max(b[3] for b in bs))
        out.append((o, [np.asarray(im.crop(box)) for im in ims]))
    return out


SE_LABEL = r"SE/SE$_0$"
HMIN = r"$h_{\min}$"      # the paper's notation for the smallest octree leaf


def se_text(o):
    return f"{SE_LABEL} = {o['se_ratio']:.2f}"


def label_width(pg, o):
    """One label line: the name, then SE/SE0 of the final body in grey."""
    return (pg.width(o["label"], NAME_FS) + LBL_GAP
            + pg.width(se_text(o), STAT_FS))


def search_rows(fig, k):
    """The k-row split meeting FIRST_ROW / NOT_FIRST with the largest common panel
    height: bisection on the height H, each step an exact bin-packing check (rows of
    capacity span + OG, object size = slot(H) + OG; depth-first over the objects,
    largest first, rows 2..k interchangeable).  Rows 2..k in ORDER_C order of their
    first object; within a row, ORDER_C order."""
    names = [o["name"] for o, _ in fig.cu]
    span = fig.AR - fig.AL

    def pack(H):
        sz = [fig.slot(i, H) + OG for i in range(len(names))]
        order = sorted(range(len(names)), key=lambda i: -sz[i])
        load, row = [0.0] * k, [None] * len(names)

        def dfs(j):
            if j == len(order):
                return True
            i, tried = order[j], set()
            for b in range(k):
                if (b == 0 and names[i] in NOT_FIRST) or (b > 0 and names[i] in FIRST_ROW):
                    continue
                if b > 0 and round(load[b], 9) in tried:
                    continue            # an identical row was tried already
                tried.add(round(load[b], 9)) if b > 0 else None
                if load[b] + sz[i] <= span + OG + 1e-9:
                    load[b] += sz[i]
                    row[i] = b
                    if dfs(j + 1):
                        return True
                    load[b] -= sz[i]
            return False
        return dfs(0), list(row)

    lo, hi, best = 1.0, 60.0, None
    for _ in range(50):
        H = (lo + hi) / 2.0
        ok, row = pack(H)
        lo, hi, best = (H, hi, row) if ok else (lo, H, best)
    rows = [[n for n in ORDER_C if best[names.index(n)] == b] for b in range(k)]
    rows[1:] = sorted(rows[1:], key=lambda r: ORDER_C.index(r[0]))
    return lo, rows


def small_arrow(pg, x0, x1, y, k=ARROW_K):
    """a's forward pipeline arrow (loop_chain.Page.arrow: ink shaft, notched stealth
    head) scaled by k -- head, width and line weight -- straight from x0 to x1 at y.
    Page.arrow's head is fixed in mm, too long for c's 1.7 mm arrows, so it is redrawn
    here rather than changing the shared helper a uses."""
    hl, hw, lw = L.HL * k, L.HW * k, L.ARROW_LW * k
    xs = x1 - hl * (1.0 - L.NOTCH * 0.5)
    pg.line([x0, xs], [y, y], lw, L.FWD, z=4)
    pg.ax.add_patch(Polygon([(x1, y), (x1 - hl, y - hw), (x1 - hl * (1.0 - L.NOTCH), y),
                             (x1 - hl, y + hw)], closed=True, fc=L.FWD, ec=L.FWD,
                            lw=0.3 * k, joinstyle="miter", zorder=4))


class MergedFigure(LL.LoopFigure):
    """LoopFigure (a + the paired row b) plus the clean-up block c under it."""
    AR_BC_KEY = True       # a's and c's keys: agentic_results' Support / Load (round 13)

    def __init__(self, R, results, cleanup, rows, **kw):
        super().__init__(R, results, **kw)
        self.cu = cleanup
        pg = L.Page(10.0)
        self.lw = [label_width(pg, o) for o, _ in cleanup]
        k = list(itertools.accumulate([0] + [len(r) for r in rows]))
        self.rows_c = [list(range(a, b)) for a, b in zip(k[:-1], k[1:])]
        self.HC = min(self.row_h(r) for r in self.rows_c)
        self.c_head = 4.2                      # c's title row
        self.c_row = LBL + 0.6 + self.HC
        self.c_key = 6.8                       # c's key line, under its rows (bar title above)
        n = len(self.rows_c)
        self.c_height = self.c_head + n * self.c_row + (n - 1) * 1.6 + self.c_key
        self.height += 2.6 + self.c_height
        px = self.height * L.DPI / 25.4
        px = np.floor(px) + PX_PHASE + (1.0 if px - np.floor(px) > PX_PHASE else 0.0)
        self.height = px * 25.4 / L.DPI

    # ---- packing --------------------------------------------------------------------
    def slot(self, i, H):
        a = L.aspect(self.cu[i][1][0])
        return max(2 * H * a + IG, self.lw[i])

    def row_h(self, idx):
        """The largest panel height at which these objects fit the span with OG gaps."""
        span = self.AR - self.AL
        lo, hi = 1.0, 60.0
        for _ in range(60):
            H = (lo + hi) / 2.0
            w = sum(self.slot(i, H) for i in idx) + (len(idx) - 1) * OG
            lo, hi = (H, hi) if w <= span else (lo, H)
        return lo

    # ---- drawing --------------------------------------------------------------------
    def draw(self, pg):
        super().draw(pg)
        self.draw_cleanup(pg, self.results_bottom() + 2.6)

    def draw_cleanup(self, pg, y, letter="c"):
        span = self.AR - self.AL
        tb = y + 2.4
        pg.letter(self.AL - LL.LET, tb, letter)
        t = pg.text(self.AL, tb, "Refining generated shapes", LL.TFS, weight="bold")
        pg.text(pg.box(t)[2] + 1.6, tb,
                "(Left: Supports and loads. Right: Change in shape)", LL.TFS, color=L.GREY)
        yr = y + self.c_head
        for r, idx in enumerate(self.rows_c):
            ws = [self.slot(i, self.HC) for i in idx]
            gap = (span - sum(ws)) / max(len(idx) - 1, 1)
            assert gap >= OG - 1e-6
            gap = min(gap, 3 * OG)             # a short row is not stretched apart
            x = self.AL + (span - sum(ws) - gap * (len(idx) - 1)) / 2.0
            for i, w in zip(idx, ws):
                o, (bc, ch) = self.cu[i]
                xc = x + w / 2.0
                xl = xc - self.lw[i] / 2.0
                tn = pg.text(xl, yr + LBL, o["label"], NAME_FS)
                pg.text(pg.box(tn)[2] + LBL_GAP, yr + LBL, se_text(o),
                        STAT_FS, color=L.GREY)
                pw = self.HC * L.aspect(bc)
                x0 = xc - (2 * pw + IG) / 2.0
                yp = yr + LBL + 0.6
                pg.img(bc, x0, yp, pw, self.HC, name=f"c_{o['name']}_bc")
                pg.img(ch, x0 + pw + IG, yp, pw, self.HC, name=f"c_{o['name']}_change")
                small_arrow(pg, x0 + pw + ARROW_PAD, x0 + pw + IG - ARROW_PAD,
                            yp + self.HC / 2.0)
                x += w + gap
            yr += self.c_row + 1.6
        self.keys_c(pg, yr - 1.6 + 3.0)

    def keys_c(self, pg, ym):
        """c's key line, centred under its rows: Support, Load (agentic_results' key, the
        same as row a's) and the change bar on the same centre line, "Change in shape"
        centred ABOVE it, -3 h_min / 0 / 3 h_min below (h_min is defined in the legend)."""
        def draw(pg_, x):
            xe = self.ar_bc_key(pg_, x, ym)
            bw = 16.0
            bx = xe + 3.0 + max(pg_.width("−3" + HMIN, L.TICK_FS) / 2.0,
                                (pg_.width("Change in shape", L.TICK_FS) - bw) / 2.0)
            yb = ym - L.BAR_H / 2.0
            pg_.text(bx + bw / 2.0, yb - 0.5, "Change in shape", L.TICK_FS, ha="center")
            cmap = L.cmap_from(CHANGE_PTS)
            pg_.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap,
                          extent=(bx, bx + bw, yb + L.BAR_H, yb), aspect="auto", zorder=4,
                          interpolation="bilinear")
            pg_.ax.add_patch(Rectangle((bx, yb), bw, L.BAR_H, fc="none", ec=L.INK, lw=0.35,
                                       zorder=5))
            for v, lab in ((-3, "−3" + HMIN), (0, "0"), (3, "3" + HMIN)):
                xv = bx + (v + 3) / 6.0 * bw
                pg_.line([xv, xv], [yb + L.BAR_H, yb + L.BAR_H + 0.4], 0.35, L.INK, z=5)
                pg_.text(xv, yb + L.BAR_H + 0.6, lab, L.TICK_FS, ha="center", va="top")
            return bx + bw + max(pg_.width("3" + HMIN, L.TICK_FS) / 2.0,
                                 (pg_.width("Change in shape", L.TICK_FS) - bw) / 2.0)
        scratch = L.Page(self.height)
        w = draw(scratch, 0.0)
        x = self.AL + (self.AR - self.AL - w) / 2.0
        assert x >= self.AL, "c's key line is wider than the span"
        draw(pg, x)

    def build(self, stem):
        pg = super().build(stem)
        print(f"  c: {len(self.rows_c)} rows "
              f"({' / '.join(str(len(r)) for r in self.rows_c)} objects), panels "
              f"{self.HC:.2f} mm tall, block {self.c_height:.1f} mm")
        return pg


def merged():
    """The merged figure: shape-run loop, hidden phase field OPTIONAL (dashed, its two
    renders from the hidden-phi run), b the two table runs side by side, c the
    geometry clean-up."""
    sst = LL.stats_file(p_so("so_stats.txt"))
    hst = LL.stats_file(p_hp("hp_stats.txt"))
    cap_so, cap_hp = int(sst["capture_iter"]), int(hst["iter_capture"])
    cube = [p_so(f"panel_so_m_{t}.png") for t in ("a", "phi", "b", "d", "e", "gu")]
    R = dict(zip(("d", "phi", "u", "g", "step", "gu"), LL.crop_common(cube)))
    R["geom"] = LL.crop_common([p_so("panel_so_m_g.png")])[0]
    R["gphi"], R["stepphi"] = LL.crop_common(
        [p_hp("hp_panel_m_g_phi.png"), p_hp("hp_panel_m_d_hid.png")])
    # b: start, the bold capture iteration (the loop's), end; the hidden-phi run from
    # 20, before its hidden field switches on (35), so the carve shows.  One crop per
    # run over its three frames.  Round 11: b's own m_evo frames (../_shared/evo_style.py:
    # fainter near half, voids darkened by openness); a's "Initial design" keeps the
    # standalone strip's evo_0, so a stays pixel-identical.
    so_it = [0, cap_so, 100]
    so_evo = [LL.lift(a) for a in L.common_crop(
        [p_so(f"panel_so_m_evo_{i}.png") for i in so_it], pad=14)]
    hp_it = [20, cap_hp, 90]
    hp_evo = [LL.lift(a) for a in L.common_crop(
        [p_hp("hp_panel_m_evo_cap.png" if i == cap_hp else f"hp_panel_m_evo_{i}.png")
         for i in hp_it], pad=14)]
    R["init"] = [LL.lift(a) for a in L.common_crop(
        [p_so(f"panel_so_evo_{i}.png") for i in so_it], pad=14)][0]
    results = [[("Shape optimization", so_evo, so_it, cap_so,
                 LL.timing(p_so("so_fr_stats.txt"))),
                ("Topology optimization", hp_evo, hp_it, cap_hp,
                 LL.timing(p_hp("hp_fr_stats.txt")))]]
    check_change_lut()
    rows = ROWS_C or [ORDER_C]
    return MergedFigure(R, results, cleanup_objects(rows), rows, phibar="optional",
                        phi_symbol=r"$\varphi$, " + LL.PHIBAR)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--search"]:
        H, rows = search_rows(merged(), int(sys.argv[2]))
        print(f"common panel height {H:.2f} mm")
        print("ROWS_C = " + json.dumps(rows, indent=1))
    else:
        assert ROWS_C, "run --search first"
        merged().build(STEM)
