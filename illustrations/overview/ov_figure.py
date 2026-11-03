"""Compose step (env mechfoundry-warp117, CPU): method_overview, Figure 1.

Reads the ``ov_p_*.png`` panels (``render_ov_panels.py`` and, for the solve,
``render_ov_solve.py``; each rendered onto its workflow part's tint) and the
input cut-outs in ``assets/``; writes ``method_overview.{pdf,png}``.

ONE LOOP ON ONE BUNNY, in the workflow's three parts, each on its own tint:
  shape generation    ONE text prompt: straight into Text2CAD / CAD-Coder and
                      into TripoSG (its text mode), and through SDXL, whose
                      drawing is the image TripoSG (image mode) or Hunyuan3D
                      takes
  pre-processing      segmentation, BC regions, optional texture,
                      (heterogeneous) materials -- run once, before the loop
  FEA & optimization  octree carrying s (built after pre-processing, from the
                      shape, its BC regions and materials), trimmed mesh
                      carrying phi, the Nitsche solve, the objective J under
                      it, and back: dJ/ds into the octree and, optionally
                      (dashed), the hidden phase field's dJ/dphi -- the loop
                      closes over these green stages only (the user,
                      2026-10-16)
Every step is a BLOCK of its part's light tint, no edge (edges read heavy; a
thin coloured rule under each title instead read worse — the user,
2026-09-29): its title (and marks) at the top, its panel below.  The blocks sit
on a grid: two rows, one column per forward step, every second-row block in
its partner's column at its width, its panel the SAME size as its partner's
(one crop box per panel kind, one panel height for both rows); one arrow gap
between columns, and the panels grow until the forward row fills the width.  A
violet star marks the steps an LLM/VLM subagent can assist, a green chip the
differentiable GPU code in NVIDIA Warp; the frame's two tabs are the two ways
the whole pass is driven: the AI operator through the workflow CLI, and human
users through the web UI.

ARROWS are drawn, not matplotlib's: a hairline shaft with rounded corners and a
slender concave ("stealth") head, block edge to block edge; where two flows
join, the joining line bends smoothly into the other's shaft (the three lanes
into segmentation, the materials into the BC regions' arrow into the octree), and only the shared
shaft carries a head.  Every head sits at the end of a straight run
(``MIN_RUN``): a head straight off a bend reads as a kink.

TITLES ONLY (the user's rule): each subfigure gets a title and nothing else; the
marks and the three parts get one legend line each.  ``ALLOWED_TEXT`` is the
registry of every string the page may carry.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import (Circle, FancyBboxPatch, PathPatch, Polygon,  # noqa: E402
                                Rectangle)
from matplotlib.path import Path  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from PIL import Image  # noqa: E402

import ov_manifest  # noqa: E402
import ov_style as S  # noqa: E402

for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
    try:
        matplotlib.font_manager.fontManager.addfont(
            f"/usr/share/fonts/truetype/msttcorefonts/{f}")
    except Exception:                                              # noqa: BLE001
        pass
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "savefig.transparent": False,
})

INK, GREY = S.PALETTE["ink"], S.PALETTE["grey text"]
AGENT, BACK, GPU = S.PALETTE["agent"], S.PALETTE["backward"], S.PALETTE["gpu"]
ARROW = "#30343b"                  # forward arrows: a shade off the ink, lighter than the titles
PARTS = S.PARTS

# ---------------------------------------------------------------------------
# Panels: rendered opaque onto their part's tint, so a panel melts into its
# block; each kind cropped to one box, so every bunny is one size and every
# cube another
# ---------------------------------------------------------------------------
PAD = 8


def load(path):
    im = Image.open(path).convert("RGB")
    if im.size != (S.PANEL_PX, S.PANEL_PX):
        im = im.resize((S.PANEL_PX, S.PANEL_PX), Image.Resampling.LANCZOS)
    return im


def ink_box(im):
    """Everything that is not the image's own background (its corner pixel)."""
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    ys, xs_ = np.where(np.abs(a - a[0, 0]).sum(axis=2) > 12)
    return xs_.min(), ys.min(), xs_.max() + 1, ys.max() + 1


def common_box(imgs):
    b = [ink_box(im) for im in imgs]
    w0, h0 = imgs[0].size
    return (max(min(x[0] for x in b) - PAD, 0), max(min(x[1] for x in b) - PAD, 0),
            min(max(x[2] for x in b) + PAD, w0), min(max(x[3] for x in b) + PAD, h0))


# the solve (render_ov_solve.py, framed on the display bunny as these are) is a
# surface panel like any other: same camera, same crop, same size
SURF = {"tripo": "ov_p_tripo.png", "cad": "ov_p_cad.png", "painted": "ov_p_painted.png",
        "seg": "ov_p_seg_pf.png", "bc": "ov_p_bc.png", "texture": "ov_p_texture.png",
        "materials": "ov_p_materials.png", "solve": "ov_p_solve.png"}
CUBE = {"octree": "ov_p_octree.png", "phibc": "ov_p_phibc.png",
        "grad_d": "ov_p_grad_d.png", "grad_phi": "ov_p_grad_phi.png"}
SURF = {k: v for k, v in SURF.items() if os.path.exists(v)}
missing = [k for k in ("tripo", "cad", "painted", "seg", "bc", "texture", "materials", "solve")
           if k not in SURF] + [k for k, v in CUBE.items() if not os.path.exists(v)]
assert not missing, f"panels missing: {missing}"
img = {}
for group in (SURF, CUBE):
    raw = {k: load(v) for k, v in group.items()}
    box = common_box(list(raw.values()))
    for k, im in raw.items():
        img[k] = np.asarray(im.crop(box))
# SDXL's drawing, cut out (ov_assets.py cutout, as draw_image cuts it out with
# RMBG-1.4), on the generation tint it sits on.  The CAD lane is a real B-rep
# (ov_assets.py cadmodel), rendered like the other shapes (ov_p_cad.png)
GEN_FILL = PARTS["generation"]["fill"]
for k, rel in (("in_sdxl", "input/sdxl_cut.png"),):
    cut_ = Image.open(ov_manifest.asset(rel)).convert("RGBA")
    on_tint = Image.new("RGBA", cut_.size, GEN_FILL)
    on_tint.alpha_composite(cut_)
    box_ = cut_.split()[3].point(lambda v: 255 if v > 8 else 0).getbbox()
    img[k] = np.asarray(on_tint.convert("RGB").crop(box_))


def ar(k):
    return img[k].shape[1] / img[k].shape[0]


# ---------------------------------------------------------------------------
# Measures (inches)
# ---------------------------------------------------------------------------
FIG_W = 7.087                      # 180 mm, NMI double column
M = 0.06                           # page margin to the AI-operator frame
TAB_H = 0.16
TITLE_FS, CARD_FS, LEGEND_FS, TAB_FS = 6.5, 6.0, 6.0, 6.0
MARK = 0.085
BP = 0.035                         # a block's inner padding
HEAD = 0.13                        # a block's title band
LINE2 = 0.095                      # a two-line title's line pitch
GAP = 0.18                         # between columns: every forward arrow one length
GAP_SRC = 0.24                     # sources -> segmentation: the three lanes merge in it
VGAP = 0.17                        # row A -> row B: a vertical arrow
LANE_GAP = 0.06                    # between the three source blocks
DROP = 0.107                       # row B -> the backward lane: R_LANE + HL + RUN_LANE (asserted below)


def measure(s, fs, weight="normal", style="normal"):
    f = plt.figure(figsize=(FIG_W, 1.0))
    t = f.text(0, 0, s, fontsize=fs, weight=weight, style=style)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    plt.close(f)
    return w


def cap_height(fs, weight="normal"):
    tp = TextPath((0, 0), "H", size=fs, prop=FontProperties(family="Arial", weight=weight))
    return (tp.vertices[:, 1].max() - tp.vertices[:, 1].min()) / 72.0


def marks_w(agent, gpu):
    return (MARK + 0.03) * (int(agent) + int(gpu))


def lines_of(s):
    return (s,) if isinstance(s, str) else s


# the forward row, and under each step its second-row partner:
# key: (title, agent, gpu, part)
# the octree comes after pre-processing (the user, 2026-10-16): it is rebuilt
# every iteration, so it sits inside the green loop with dJ/ds under it
ROW_A = ["seg", "bc", "octree", "phibc", "solve"]
TITLES = {"seg": ("Segmentation", True, False, "pre"),
          "bc": ("BC reasoning", True, False, "pre"),
          "octree": ("SDF meshing", False, True, "fea"),
          "phibc": ("Diffused body", False, True, "fea"),
          "solve": ("Diffuse Nitsche", False, True, "fea"),
          "grad_d": (("Shape", "sensitivity"), False, True, "fea"),
          "texture": ("Texture paint", False, False, "pre"),
          # two lines, the star left of both: one line would widen the column
          "materials": (("(Per-part)", "Materials"), True, False, "pre"),
          "grad_phi": (("Topology", "sensitivity"), False, True, "fea")}
UNDER = {"seg": "texture", "bc": "materials", "octree": "grad_d", "phibc": "grad_phi"}
OBJ_TXT = ("Design", "objective")    # two lines: one would widen the solve column
# the generators, top down, and SDXL: text on top feeds CAD and TripoSG, SDXL
# below feeds TripoSG and Hunyuan3D, so TripoSG sits between (no arrow crosses)
LANES = (("cad", "Text-to-CAD", True),     # both text-to-CAD generators (Text2CAD, CAD-Coder)
         ("tripo", "TripoSG", True), ("hunyuan", "Hunyuan3D", False))
SDXL_TXT = "SDXL"
# the prompt box: a rounded text field, the prompt wrapped over two lines (on one
# line at a readable size it would widen the generation part past the budget)
PROMPT = "“a sitting\nrabbit”"

title_w = {k: marks_w(a, g) + max(measure(t, TITLE_FS) for t in lines_of(s)) + 0.04
           for k, (s, a, g, _) in TITLES.items()}
HEAD_B = HEAD + LINE2 * (max(len(lines_of(TITLES[u][0])) for u in UNDER.values()) - 1)
OBJ_W = max(measure(t, TITLE_FS) for t in lines_of(OBJ_TXT)) + marks_w(False, True) + 0.10
OBJ_H = 0.22 + LINE2 * (len(lines_of(OBJ_TXT)) - 1)
TXT_H = 0.25                       # the prompt box's height
G1 = 0.23                          # the prompt / SDXL -> the generators: two fans of arrows
RF = 0.035                         # the fans' corner radius (tighter than R: they share G1)
x_src0 = M + 0.10                  # the tab and the parts' legend start here too
x_A1 = FIG_W - M - 0.10


def layout(h):
    """Everything that follows from the panel height h.

    The generation part is two sub-columns: the prompt card and SDXL, then the
    three generators stacked down the two rows' height in three equal blocks
    (the user, 2026-10-17), top and bottom level with the rows, every shape one
    size (lane_c):
        3 (2 BP + HEAD + lane_c) + 2 LANE_GAP = (2 BP + HEAD + h) + VGAP + (2 BP + HEAD_B + h)
    """
    lane_c = (2 * h + VGAP + HEAD_B - 2 * HEAD - 2 * BP - 2 * LANE_GAP) / 3.0
    out_w = lane_c * max(ar("tripo"), ar("painted"), ar("cad"))
    txt_w = max(measure(t, CARD_FS, style="italic") for t in PROMPT.split("\n")) + 0.14
    # SDXL's drawing: the shapes' height, as wide as the prompt card leaves room for
    w1 = max(out_w + 2 * BP, measure(SDXL_TXT, TITLE_FS) + 0.04 + 2 * BP, txt_w)
    w2 = 2 * BP + max(out_w, *(marks_w(a, False) + max(measure(t, TITLE_FS) for t in lines_of(n))
                               + 0.04 for _, n, a in LANES))
    col = {k: 2 * BP + max(h * ar(k), title_w[k],
                           *([title_w[UNDER[k]]] if k in UNDER else [OBJ_W]))
           for k in ROW_A}
    # the FEA & optimization columns one width (the user, 2026-10-17)
    fea = [k for k in ROW_A if TITLES[k][3] == "fea"]
    col.update({k: max(col[j] for j in fea) for k in fea})
    return dict(lane_c=lane_c, out_w=out_w, txt_w=txt_w, w1=w1, w2=w2,
                src_w=w1 + G1 + w2, col=col)


def width(h):
    L = layout(h)
    return L["src_w"] + GAP_SRC + sum(L["col"].values()) + GAP * (len(ROW_A) - 1)


# the panel height that fills the page's width: gaps fixed, panels take the rest
lo_, hi_ = 0.3, 1.5
for _ in range(60):
    mid_ = (lo_ + hi_) / 2.0
    lo_, hi_ = (mid_, hi_) if width(mid_) <= x_A1 - x_src0 else (lo_, mid_)
H = lo_
# the page-height budget: panels no taller than the 2026-10-16 layout's (0.810 in,
# figure 2.875 in).  Shorter titles (the user's 2026-10-17 relabelling) would
# otherwise grow every panel; the width they free goes into the column gaps.
H_MAX = 0.810
if H > H_MAX:
    H = H_MAX
    GAP = GAP + (x_A1 - x_src0 - width(H)) / (len(ROW_A) - 1)
L = layout(H)
OUT_W, SRC_W, LANE_C, COL = L["out_w"], L["src_w"], L["lane_c"], L["col"]
W1, W2, TXT_W = L["w1"], L["w2"], L["txt_w"]
xc, x = {}, x_src0 + SRC_W + GAP_SRC
for k in ROW_A:                    # column k spans [xc[k][0], xc[k][1]]
    xc[k] = (x, x + COL[k])
    x += COL[k] + GAP
for k, u in UNDER.items():
    xc[u] = xc[k]
cx = {k: (a + b) / 2.0 for k, (a, b) in xc.items()}

# vertical, from the bottom up
BLK_A = 2 * BP + HEAD + H
BLK_B = 2 * BP + HEAD_B + H
y_ret = M + 0.06                                   # the backward leg's lane, clear of the legend
yB = y_ret + DROP                                  # row B blocks' bottom
yA = yB + BLK_B + VGAP                             # row A blocks' bottom
y_top = yA + BLK_A
frame_top = y_top + 0.07 + TAB_H / 2.0
FIG_H = frame_top + TAB_H / 2.0 + M
MID_A, MID_B = yA + BP + H / 2.0, yB + BP + H / 2.0   # the rows' panel centre lines
fig = plt.figure(figsize=(FIG_W, FIG_H))
FT, IT = fig.transFigure, fig.dpi_scale_trans     # IT: inches, the unit of this file
RATIO = FIG_W / FIG_H
print(f"figure {FIG_W:.3f} x {FIG_H:.3f} in; panels {H:.3f} in high")


def fx(v):
    return v / FIG_W


def fy(v):
    return v / FIG_H


def put(x, y, s, fs=TITLE_FS, ha="center", va="baseline", color=INK, **kw):
    t = fig.text(fx(x), fy(y), s, fontsize=fs, ha=ha, va=va, color=color, **kw)
    t.set_zorder(max(t.get_zorder(), 3))
    return t


rect = {}                                          # key -> panel (x0, y0, x1, y1)


def fit(k, x, y, w, h, key=None):
    """The image k, as large as fits the box, centred in it."""
    r = ar(k)
    pw, ph = (w, w / r) if w / r <= h else (h * r, h)
    x0, y0 = x + (w - pw) / 2.0, y + (h - ph) / 2.0
    a = fig.add_axes([fx(x0), fy(y0), fx(pw), fy(ph)])
    a.imshow(img[k], interpolation="lanczos", aspect="auto")
    a.set_axis_off()
    a.set_zorder(2)
    rect[key or k] = (x0, y0, x0 + pw, y0 + ph)
    return a


# ---------------------------------------------------------------------------
# Icons, drawn (Arial has neither glyph; a font fallback breaks the one-font rule)
# ---------------------------------------------------------------------------
icons = []


def _icon_axes(x, yc, side=MARK):
    a = fig.add_axes([fx(x), fy(yc - side / 2.0), fx(side), fy(side)])
    a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_axis_off()
    a.patch.set_alpha(0.0)
    a.set_zorder(5)
    icons.append(a)
    return a


def _sparkle(a, colour, r=0.74):
    tips = [(0, r), (r, 0), (0, -r), (-r, 0)]
    verts, codes = [tips[0]], [Path.MOVETO]
    for i in range(4):
        verts += [(0.0, 0.0), tips[(i + 1) % 4]]
        codes += [Path.CURVE3, Path.CURVE3]
    a.add_patch(PathPatch(Path(verts, codes), fc=colour, ec="none"))


def agent_mark(x, yc, inverse=False):
    a = _icon_axes(x, yc)
    a.add_patch(Circle((0, 0), 1.0, fc="white" if inverse else AGENT, ec="none"))
    _sparkle(a, AGENT if inverse else "white")


def gpu_mark(x, yc):
    a = _icon_axes(x, yc)
    for t in (-0.42, 0.0, 0.42):
        for (x0, y0, x1, y1) in ((t, 0.62, t, 0.98), (t, -0.62, t, -0.98),
                                 (0.62, t, 0.98, t), (-0.62, t, -0.98, t)):
            a.plot([x0, x1], [y0, y1], color=GPU, lw=0.7, solid_capstyle="butt")
    a.add_patch(FancyBboxPatch((-0.66, -0.66), 1.32, 1.32,
                               boxstyle="round,pad=0,rounding_size=0.18", fc=GPU, ec="none"))
    a.add_patch(Rectangle((-0.26, -0.26), 0.52, 0.52, fc="white", ec="none"))


def prompt_mark(x, yc, side, colour="white", lw=1.0):
    a = _icon_axes(x, yc, side)
    a.plot([-0.85, -0.15, -0.85], [0.6, 0.0, -0.6], color=colour, lw=lw,
           solid_capstyle="round", solid_joinstyle="round")
    a.plot([0.1, 0.85], [-0.62, -0.62], color=colour, lw=lw, solid_capstyle="round")


CAP = cap_height(TITLE_FS)
titles, title_top = {}, {}


def title_at(key, cx_, y_mid, s, agent=False, gpu=False):
    """A title and its marks, centred on cx_ about y_mid: the marks on the
    midline, left of every line; a two-line title's lines each centred on the
    text (left-aligned, the short line hung off to one side and the title read
    as off-centre), set one LINE2 apart about the midline."""
    lines = lines_of(s)
    text_w = max(measure(t, TITLE_FS) for t in lines)
    x0 = cx_ - (marks_w(agent, gpu) + text_w) / 2.0
    if agent:
        agent_mark(x0, y_mid)
        x0 += MARK + 0.03
    if gpu:
        gpu_mark(x0, y_mid)
        x0 += MARK + 0.03
    n = len(lines)
    for i, line in enumerate(lines):
        yl = y_mid + ((n - 1) / 2.0 - i) * LINE2
        titles[key if i == n - 1 else f"{key}:{i}"] = put(x0 + text_w / 2.0, yl - CAP / 2.0, line)
    title_top[key] = y_mid + (n - 1) / 2.0 * LINE2 + CAP / 2.0


blocks = {}


def block(key, x0, y0, w, h, part, radius=0.03):
    """A step's block: its part's light tint, rounded, no edge."""
    fig.add_artist(FancyBboxPatch((x0, y0), w, h, transform=IT,
                                  boxstyle=f"round,pad=0,rounding_size={radius:.4f}",
                                  fc=PARTS[part]["fill"], ec="none", zorder=0.5))
    blocks[key] = (x0, y0, x0 + w, y0 + h)


def station(key, y0, head):
    """Block + title band + the panel (H high) on the block's floor."""
    s, a, g, part = TITLES[key]
    x0, x1 = xc[key]
    h = 2 * BP + head + H
    block(key, x0, y0, x1 - x0, h, part)
    title_at(key, cx[key], y0 + h - BP - head / 2.0 + 0.005, s, a, g)
    fit(key, x0 + BP, y0 + BP, x1 - x0 - 2 * BP, H)


# ---------------------------------------------------------------------------
# Arrows (inches)
# ---------------------------------------------------------------------------
ARROW_LW = 0.75                    # pt
HL, HW, NOTCH = 0.062, 0.024, 0.28  # head length, half-width, back notch (of HL)
R = 0.045                          # corner radius
MIN_RUN = 0.035                    # the straight run a head sits on, past the last bend
# the gap under row B about halved (the user, 2026-10-17): the lane's arrow into
# dJ/ds turns up on a tight corner and its head sits right after it (no extra run)
R_LANE, RUN_LANE = 0.03, 0.015    # 0.02 / 0 was too tight (the user, 2026-10-17)
assert abs(DROP - (R_LANE + HL + RUN_LANE)) < 1e-9, "DROP is not the lane arrow's minimum"


def _rounded(pts, r):
    """A polyline with every corner replaced by a quadratic arc of radius <= r."""
    pts = [np.asarray(p, float) for p in pts]
    verts, codes = [pts[0]], [Path.MOVETO]
    radii = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
        ri = min(r, la / 2.0, lc / 2.0)
        radii.append(ri)
        verts += [b - (b - a) / la * ri, b, b + (c - b) / lc * ri]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1])
    codes.append(Path.LINETO)
    return Path(verts, codes), radii


def arrow(pts, colour=ARROW, dashed=False, head=True, r=R, min_run=MIN_RUN):
    """Shaft through pts (rounded corners), a stealth head at pts[-1]."""
    pts = [np.asarray(p, float) for p in pts]
    tip = pts[-1]
    d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
    shaft = list(pts)
    if head:                       # the shaft stops inside the head, clear of its notch
        shaft[-1] = tip - d * HL * (1.0 - NOTCH * 0.5)
    path, radii = _rounded(shaft, r)
    if head and radii:             # a straight run between the last bend and the head
        run = np.linalg.norm(tip - pts[-2]) - radii[-1] - HL
        assert run >= min_run - 1e-9, f"arrow into {tuple(np.round(tip, 3))}: head {run:.3f} in off a bend"
    fig.add_artist(PathPatch(path, transform=IT, fc="none", ec=colour, lw=ARROW_LW,
                             capstyle="butt", joinstyle="round",
                             ls=(0, (3.2, 2.2)) if dashed else "-", zorder=4))
    if head:
        n = np.array([-d[1], d[0]])
        fig.add_artist(Polygon([tip, tip - d * HL + n * HW, tip - d * HL * (1.0 - NOTCH),
                                tip - d * HL - n * HW], closed=True, transform=IT,
                               fc=colour, ec=colour, lw=0.3, joinstyle="miter", zorder=4))


def card(x, y, w, h):
    """An input card: a hairline frame; the tint shows through."""
    a = fig.add_axes([fx(x), fy(y), fx(w), fy(h)])
    a.set_xticks([]); a.set_yticks([])
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_facecolor("none")
    a.set_zorder(2)
    for sp in a.spines.values():
        sp.set_color(S.FRAME)
        sp.set_linewidth(0.6)
    return a


def text_card(x, y, w, h, s):
    """The prompt: a rounded text field, its words centred in it."""
    widest = max(measure(t, CARD_FS, style="italic") for t in s.split("\n"))
    assert widest + 0.10 <= w, "the prompt crowds its box"
    fig.add_artist(FancyBboxPatch((x, y), w, h, transform=IT,
                                  boxstyle="round,pad=0,rounding_size=0.045",
                                  fc="white", ec="#9aa1ab", lw=0.7, zorder=2))
    put(x + w / 2.0, y + h / 2.0, s, fs=CARD_FS, va="center", color=INK, style="italic",
        linespacing=1.25)


# ---------------------------------------------------------------------------
# The frame: two ways to drive the whole pass, one tab each on its top line --
# the AI operator through the workflow CLI, a person through the web UI.  The
# three parts' legend (bottom left) and the marks' legend (bottom right) sit on
# its bottom line
# ---------------------------------------------------------------------------
fig.add_artist(FancyBboxPatch(
    (fx(M), fy(M)), fx(FIG_W - 2 * M), fy(frame_top - M),
    boxstyle="round,pad=0,rounding_size=0.012", transform=FT, facecolor="none",
    edgecolor=AGENT, lw=0.6, mutation_aspect=RATIO, zorder=0))
HUMAN = "#4a515b"                  # the human tab: the ink family, so it never reads as an agent
TABS = [("AI operator drives every step via the workflow CLI", AGENT, "#e7e2f5", "agent"),
        ("Human users drive every step via the web UI", HUMAN, "#eceef1", "human")]
TAB_TXT = TABS[0][0]
ch_b = cap_height(TAB_FS, "bold")
PROMPT_SIDE = 1.25 * ch_b
tab_pad = 0.08


def window_mark(x, yc, side, colour):
    """A UI window: a rounded outline, its title bar and two dots."""
    a = _icon_axes(x, yc, side)
    a.add_patch(FancyBboxPatch((-0.9, -0.75), 1.8, 1.5, boxstyle="round,pad=0,rounding_size=0.25",
                               fc="none", ec=colour, lw=0.9))
    a.plot([-0.9, 0.9], [0.3, 0.3], color=colour, lw=0.9)
    for cx_ in (-0.55, -0.25):
        a.add_patch(Circle((cx_, 0.52), 0.1, fc=colour, ec="none"))


def robot_mark(x, yc, colour):
    """The AI operator: a robot head (the star stays the subagents' mark)."""
    a = _icon_axes(x, yc)
    a.add_patch(FancyBboxPatch((-0.78, -0.72), 1.56, 1.22, boxstyle="round,pad=0,rounding_size=0.3",
                               fc=colour, ec="none"))
    a.plot([0.0, 0.0], [0.5, 0.8], color=colour, lw=0.8, solid_capstyle="butt")
    a.add_patch(Circle((0.0, 0.86), 0.14, fc=colour, ec="none"))
    for ex in (-0.34, 0.34):
        a.add_patch(Circle((ex, -0.08), 0.17, fc="white", ec="none"))
    for sx in (-0.95, 0.95):
        a.add_patch(Rectangle((sx - 0.1, -0.33), 0.2, 0.36, fc=colour, ec="none"))


def person_mark(x, yc, colour):
    """A person: a head over a pair of shoulders."""
    a = _icon_axes(x, yc)
    a.add_patch(Circle((0.0, 0.42), 0.36, fc=colour, ec="none"))
    t = np.linspace(0.0, np.pi, 40)
    a.add_patch(Polygon(np.c_[0.78 * np.cos(t), -0.98 + 0.9 * np.sin(t)], closed=True,
                        fc=colour, ec="none"))


tab_titles, xt = [], x_src0
for text, colour, fill, who in TABS:
    w_tab = (tab_pad + PROMPT_SIDE + 0.05 + MARK + 0.05
             + measure(text, TAB_FS, "bold") + tab_pad)
    fig.add_artist(FancyBboxPatch(
        (fx(xt), fy(frame_top - TAB_H / 2.0)), fx(w_tab), fy(TAB_H),
        boxstyle=f"round,pad=0,rounding_size={fx(TAB_H / 2.0):.5f}", transform=FT,
        fc=fill, ec=colour, lw=0.6, mutation_aspect=RATIO, zorder=1))
    xx = xt + tab_pad
    if who == "agent":
        prompt_mark(xx, frame_top, PROMPT_SIDE, colour=colour)
        xx += PROMPT_SIDE + 0.05
        robot_mark(xx, frame_top, colour)
    else:
        window_mark(xx, frame_top, PROMPT_SIDE, colour)
        xx += PROMPT_SIDE + 0.05
        person_mark(xx, frame_top, colour)
    xx += MARK + 0.05
    tab_titles.append(put(xx, frame_top - ch_b / 2.0, text, fs=TAB_FS, ha="left",
                          color=colour, weight="bold"))
    xt += w_tab + 0.12
assert xt - 0.12 <= x_A1, "the two tabs overrun the frame"

LEG_AGENT = "LLM / VLM subagents assisted"
LEG_GPU = "Differentiable GPU code in NVIDIA Warp"
ch_l = cap_height(LEGEND_FS)
legend = []
ITEM_GAP = 0.12


def knockout(x0, x1, yc):
    """One white break in the frame line under a whole legend group."""
    fig.add_artist(Rectangle((fx(x0 - 0.05), fy(yc - 0.055)), fx(x1 - x0 + 0.10), fy(0.11),
                             transform=FT, fc="white", ec="none", zorder=1))


items = [(LEG_AGENT, agent_mark), (LEG_GPU, gpu_mark)]
w_items = [MARK + 0.04 + measure(lbl, LEGEND_FS) for lbl, _ in items]
xr = x_A1 - (sum(w_items) + ITEM_GAP * (len(items) - 1))
x_marks0 = xr
knockout(xr, x_A1, M)
for (label, draw), w in zip(items, w_items):
    draw(xr, M)
    legend.append(put(xr + MARK + 0.04, M - ch_l / 2.0, label, fs=LEGEND_FS, ha="left"))
    xr += w + ITEM_GAP
SW, SH = 0.15, 0.10                # a part's swatch: a small block of its tint
xl = x_src0
parts_w = [SW + 0.05 + measure(PARTS[k]["label"], LEGEND_FS) for k in ("generation", "pre", "fea")]
knockout(xl, xl + sum(parts_w) + ITEM_GAP * 2, M)
assert xl + sum(parts_w) + ITEM_GAP * 2 + 0.2 <= x_marks0, "the two legends meet"
for key, w in zip(("generation", "pre", "fea"), parts_w):
    fig.add_artist(FancyBboxPatch((xl, M - SH / 2.0), SW, SH, transform=IT,
                                  boxstyle="round,pad=0,rounding_size=0.02",
                                  fc=PARTS[key]["fill"], ec="none", zorder=2))
    legend.append(put(xl + SW + 0.05, M - ch_l / 2.0, PARTS[key]["label"], fs=LEGEND_FS,
                      ha="left"))
    xl += w + ITEM_GAP

# ---------------------------------------------------------------------------
# Shape generation: the prompt and SDXL, then the three generators stacked down
# the two rows' height, every shape (and SDXL's drawing) one size
# ---------------------------------------------------------------------------
x_g0 = x_src0 + W1 + G1                           # the generators' blocks
lane_mid = {}
LB = 2 * BP + HEAD + LANE_C                        # a generator block: all three one size
tops = {k: y_top - i * (LB + LANE_GAP) for i, (k, _, _) in enumerate(LANES)}
for key, name, agent in LANES:
    top_ = tops[key]
    y0 = top_ - LB
    block(key, x_g0, y0, W2, top_ - y0, "generation")
    head = HEAD + LINE2 * (len(lines_of(name)) - 1)
    title_at(key, x_g0 + W2 / 2.0, top_ - BP - head / 2.0 + 0.005, name, agent=agent)
    lane_mid[key] = y0 + BP + LANE_C / 2.0
    yc0 = lane_mid[key] - LANE_C / 2.0
    fit({"cad": "cad", "tripo": "tripo", "hunyuan": "painted"}[key],
        x_g0 + BP, yc0, W2 - 2 * BP, LANE_C, key=key)
assert abs(bottom_ := tops["hunyuan"] - LB - yB) < 1e-6, f"the generators miss row B by {bottom_:.4f}"
# the prompt, level with CAD's shape; SDXL midway between the two generators it feeds
x_c1 = x_src0 + W1 / 2.0
Y_P = lane_mid["cad"]
text_card(x_c1 - TXT_W / 2.0, Y_P - TXT_H / 2.0, TXT_W, TXT_H, PROMPT)
y_sdxl = (lane_mid["tripo"] + lane_mid["hunyuan"]) / 2.0
block("sdxl", x_src0, y_sdxl - LB / 2.0, W1, LB, "generation")
title_at("sdxl", x_c1, y_sdxl + LB / 2.0 - BP - HEAD / 2.0 + 0.005, SDXL_TXT)
fit("in_sdxl", x_src0 + BP, y_sdxl - LB / 2.0 + BP, W1 - 2 * BP, LANE_C, key="sdxl")

# ---------------------------------------------------------------------------
# The two rows, column by column; the objective under the solve
# ---------------------------------------------------------------------------
for k in ROW_A:
    station(k, yA, HEAD)
for k in UNDER.values():
    station(k, yB, HEAD_B)
# the objective level with its neighbour's block centre (not the panels' centre
# line, which the title band pushes down), the user 2026-10-17
Y_OBJ = yB + BLK_B / 2.0
obj = (cx["solve"] - OBJ_W / 2.0, Y_OBJ - OBJ_H / 2.0)
block("objective", obj[0], obj[1], OBJ_W, OBJ_H, "fea", radius=OBJ_H / 2.0)
title_at("objective", cx["solve"], Y_OBJ, OBJ_TXT, gpu=True)


def left(k):
    return blocks[k][0]


def right(k):
    return blocks[k][2]


def bottom(k):
    return blocks[k][1]


def top(k):
    return blocks[k][3]


# ---------------------------------------------------------------------------
# Arrows, block edge to block edge
# ---------------------------------------------------------------------------
# the prompt: straight into CAD; a branch down into TripoSG; down into SDXL.
# SDXL: up into TripoSG (joining the prompt's branch: text OR image) and down
# into Hunyuan3D.  The two fans stand at different x, so neither reads as the other's.
x_t1 = x_c1 + TXT_W / 2.0
x_f1 = x_t1 + RF + 0.005                         # the prompt's fan, just past the card
x_f2 = x_f1 + 0.045                              # SDXL's, just right of it
assert right("sdxl") + RF <= x_f2, "SDXL's fan crowds its block"
assert x_f2 + 2 * RF <= left("tripo") - HL - 0.005, "SDXL joins the prompt's arrow inside its head"
arrow([(x_t1, Y_P), (left("cad"), Y_P)])
arrow([(x_f1 - RF - 1e-3, Y_P), (x_f1, Y_P), (x_f1, lane_mid["tripo"]),
       (left("tripo"), lane_mid["tripo"])], r=RF)
arrow([(x_c1, Y_P - TXT_H / 2.0), (x_c1, top("sdxl"))])
arrow([(right("sdxl"), y_sdxl), (x_f2, y_sdxl), (x_f2, lane_mid["tripo"]),
       (x_f2 + 2 * RF, lane_mid["tripo"])], head=False, r=RF)
arrow([(x_f2 - RF - 1e-3, y_sdxl), (x_f2, y_sdxl), (x_f2, lane_mid["hunyuan"]),
       (left("hunyuan"), lane_mid["hunyuan"])], r=RF)
# the three shapes into segmentation (the first step after generation) by one
# bus: CAD drops into it, TripoSG rises into it, Hunyuan3D rises into TripoSG's
x_bus = right("cad") + 0.07
join = (x_bus + 2 * R, MID_A)
assert MID_A - 2 * R > lane_mid["tripo"], "TripoSG's lane is too close to the centre line"
assert lane_mid["cad"] - 2 * R > MID_A, "CAD's lane is too close to the centre line"
arrow([(right("cad"), lane_mid["cad"]), (x_bus, lane_mid["cad"]), (x_bus, MID_A), join],
      head=False)
arrow([join, (left(ROW_A[0]), MID_A)])
arrow([(right("tripo"), lane_mid["tripo"]), (x_bus, lane_mid["tripo"]), (x_bus, MID_A), join],
      head=False)
arrow([(right("hunyuan"), lane_mid["hunyuan"]), (x_bus, lane_mid["hunyuan"]),
       (x_bus, lane_mid["tripo"] + R)], head=False)
# forward
for u, v in zip(ROW_A, ROW_A[1:]):
    arrow([(right(u), MID_A), (left(v), MID_A)])
arrow([(cx["solve"], bottom("solve")), (cx["solve"], top("objective"))])
# segmentation -> texture -> materials, which bend into the BC regions' arrow
# into the octree: both join there, outside the loop (no arrow returns to them)
arrow([(cx["seg"], bottom("seg")), (cx["seg"], top("texture"))])
arrow([(right("texture"), MID_B), (left("materials"), MID_B)])
# the rise sits midway between the materials block and the head it must join
# before, both of its bends as round as that room allows
room = left("octree") - HL - MIN_RUN - right("materials")
r_up = min(R, room / 2.0)
x_up = right("materials") + room / 2.0
assert r_up >= 0.03, f"no room for the materials to join: {r_up:.3f} in"
arrow([(right("materials"), MID_B), (x_up, MID_B), (x_up, MID_A), (x_up + 2 * r_up, MID_A)],
      head=False, r=r_up)
# backward, from the objective: down to the lane under row B, along it and up
# into dJ/ds, then up into the octree; dashed, optional, through dJ/dphi into phi
arrow([(cx["solve"], bottom("objective")), (cx["solve"], y_ret), (cx["octree"], y_ret),
       (cx["octree"], bottom("grad_d"))], colour=BACK, r=R_LANE, min_run=RUN_LANE)
arrow([(cx["octree"], top("grad_d")), (cx["octree"], bottom("octree"))], colour=BACK)
arrow([(left("objective"), Y_OBJ), (right("grad_phi"), Y_OBJ)], colour=BACK, dashed=True)
arrow([(cx["phibc"], top("grad_phi")), (cx["phibc"], bottom("phibc"))], colour=BACK,
      dashed=True)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
ALLOWED_TEXT = {
    *(t for t, *_ in TABS), PROMPT, LEG_AGENT, LEG_GPU, *lines_of(OBJ_TXT),
    *(p["label"] for p in PARTS.values()), SDXL_TXT,
    *(line for _, n, _ in LANES for line in lines_of(n)),
    *(line for s, *_ in TITLES.values() for line in lines_of(s)),
}
AGENT_STEPS = {"tripo", "cad", "seg", "bc", "materials"}
GPU_STEPS = {"octree", "phibc", "solve", "objective", "grad_d", "grad_phi"}
fig.canvas.draw()
rend = fig.canvas.get_renderer()
W_px, H_px = fig.canvas.get_width_height()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
for t in texts:
    assert t.get_text() in ALLOWED_TEXT, f"text not allowed on this figure: {t.get_text()!r}"
    assert 5.0 <= t.get_fontsize() <= 7.0, f"{t.get_text()!r} at {t.get_fontsize()} pt"
    e = t.get_window_extent(renderer=rend)
    assert e.x0 >= 0 and e.x1 <= W_px and e.y0 >= 0 and e.y1 <= H_px, \
        f"{t.get_text()!r} leaves the canvas"
n_icons = len(AGENT_STEPS) + len(GPU_STEPS) + 2 + 2 + 2   # + the two tabs' pairs
assert len(icons) == n_icons, f"{len(icons)} icons, expected {n_icons}"
# blocks: none overlap; each row shares one top and bottom; every second-row
# block in its partner's column, its panel its partner's size
names = list(blocks)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        A, B = blocks[a], blocks[b]
        assert A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1], \
            f"blocks {a} and {b} overlap"
for row in (ROW_A, list(UNDER.values())):
    assert len({(round(blocks[k][1], 5), round(blocks[k][3], 5)) for k in row}) == 1, \
        f"{row} not aligned"
for k, u in UNDER.items():
    a, b = rect[k], rect[u]
    assert abs((a[2] - a[0]) - (b[2] - b[0])) < 1e-3 and abs((a[3] - a[1]) - (b[3] - b[1])) < 1e-3, \
        f"{u} is not {k}'s size"
    assert (blocks[k][0], blocks[k][2]) == (blocks[u][0], blocks[u][2]), f"{u} leaves {k}'s column"
# every bunny one size: the solve's is the BC regions' (same camera, same crop)
assert abs((rect["solve"][3] - rect["solve"][1]) - (rect["bc"][3] - rect["bc"][1])) < 1e-3
# every title inside its block; no two titles touch
boxes = {k: t.get_window_extent(renderer=rend) for k, t in titles.items()}
for k, e in boxes.items():
    X0, Y0, X1, Y1 = (v * fig.dpi for v in blocks[k.split(":")[0]])
    assert X0 <= e.x0 and e.x1 <= X1 and Y0 <= e.y0 and e.y1 <= Y1, f"title '{k}' leaves its block"
tn = list(boxes)
for i, a in enumerate(tn):
    for b in tn[i + 1:]:                           # (a title's own lines are set as one block)
        if a.split(":")[0] != b.split(":")[0]:
            assert not boxes[a].overlaps(boxes[b]), f"titles {a} and {b} touch"
assert abs(xc["solve"][1] - x_A1) < 1e-3, "the forward row does not fill the width"
for t in tab_titles:
    assert t.get_window_extent(renderer=rend).x1 <= x_A1 * fig.dpi, "a tab leaves the frame"

fig.savefig("method_overview.pdf", dpi=600)
fig.savefig("method_overview.png", dpi=400)
print(f"wrote method_overview.pdf / .png ({FIG_W:.3f} x {FIG_H:.3f} in)")
