"""Stage 3 (env mechfoundry-warp117, CPU): the demo video's frames from the pvbatch passes.

    python dv_compose.py WORK [--step N] [--out-scale F] [--procs P] [--only F,F,...]
    python dv_compose.py WORK --sheet SHEET.png [--n 24]      # time-labelled contact sheet
    python dv_compose.py WORK --phone DIR                     # key frames at 360 px wide

Reads WORK/dv_stats.json and WORK/frames/ (render_dv.py), writes WORK/out/f####.png.
Everything 2-D happens here, at 1920 x 1080 whatever the render scale (a draft's
half-size passes are upsampled first), so the draft lays out exactly like the final:

  * looks are crossfaded in sRGB on premultiplied colour (dv_timeline.look_weights);
    every look is the same mesh under the same camera, so the blend is exact; the load
    arrows are their own pass, laid OVER the body with the BC look's weight, so the
    backrest never hides them while the bench turns
  * the bench layer = floor grid (2x render, Lanczos down, faded out toward the far
    field so its thin far lines do not crawl), the undeformed ghost, the body; then
    the layer's rise (scene 2) and its shrink into the top-left cell (scene 8)
  * the ghost (round 2): a 4 px #9aa1ab line just outside the undeformed silhouette,
    drawn only where it is more than 4 px from the deformed silhouette, no fill -- it
    shows where the body has moved, not a sticker halo; tiles use the same rule
  * titles 72 px Arial Bold, ink on white, baseline 100 (Nitsche: two lines, 100/186);
    title changes dip (old out 0.18 s, new in 0.22 s); scene 3 slides " with texture"
    in.  In scene 8 the Nitsche title shrinks into the bench's cell and gives way to
    its prompt; every tile has its object's prompt above it as the agentic-results
    card prints it (curly quotes, Arial Italic, ink, <= 2 lines, 28 px); the scene-8
    title is two lines of one size (<= 64 px, line height 76) within x = 64-1856
  * scale label "xN" 72 px Arial Bold brick #7a2118, right edge x = 1760, baseline 280
  * colour bar (scene 6): upright, 44 x 540 px at x = 1560, right of the bench, on a
    white 90 % backing panel, 2 px #5f6672 outline, the house E map on log E in
    [5, 200] GPa, 44 px ticks on its right ("200 GPa" on top), rising 20 px as it fades in
  * the optimisation beat (2026-11-07): the bench layer as always (the close-up is the
    renderer's camera), the iterates crossfaded in pairs, the change arrows laid over
    the body like the load arrows, the body faded out across x 840-960 while the plot is
    up; on the right the strain-energy plot (draw_plot: no panel, the floor grid faded
    out under it; 4 px #3d424b axes and ticks, 44 px labels, 40 px tick labels, y from 0, a dashed 1.0,
    the 6 px ink curve drawn in step with the iterate, a brick dot, and at the end a
    live bracket and "-NN %" over "Lower -> Stiffer" (42 px #3d424b, medium); the
    surface-change bar bottom-left, in mm); the change arrows (navy / deep teal)
    carry a 4 px white halo and fade by arrow_fades (look-ahead, <= 1/15 alpha a frame)
  * CLI (scene 1): a generic dark terminal box, opaque from frame 0 with "> " and a
    blinking cursor; the line types in with per-key jitter; it leaves (up 300 px,
    out) in 0.3 s, before the rising bench reaches it
"""

import argparse
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
import dv_timeline as TL  # noqa: E402
import dv_camera as CAM  # noqa: E402
import ar_figure as F  # noqa: E402  (typographic, balanced_wrap: the prompt card's rules)
import ar_style as A  # noqa: E402

W, H = TL.W, TL.H
FONT_DIR = "/usr/share/fonts/truetype/msttcorefonts"
FB, FR = os.path.join(FONT_DIR, "Arial_Bold.ttf"), os.path.join(FONT_DIR, "Arial.ttf")
FI = os.path.join(FONT_DIR, "Arial_Italic.ttf")     # the agentic-results prompt card's face
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


INK = _hex(A.INK)
BRICK = _hex("#7a2118")
GHOST_RGB = np.array(_hex("#9aa1ab"), np.float32) / 255
GHOST_A, GHOST_W, GHOST_GAP = 0.9, 4.0, 4.0
BAR_OUTLINE = _hex("#5f6672")
TITLE_PX, TITLE_B1, TITLE_B2 = 72, 100, 186
NEW_PX_MAX, NEW_B, NEW_LH = 64, 72, 76
PROMPT_PX, PROMPT_LH = 28, 1.18           # tile prompt size (px) and line pitch (x size)
LABEL_PX, LABEL_X, LABEL_B = 72, 1760, 280
# The E bar stands upright right of the bench: framed 70 px lower (the user), the
# bench's feet reach y = 980, where the horizontal bar used to sit, and that bar could
# not move lower and keep 40 px from the frame's bottom edge.
BAR_X0, BAR_Y, BAR_W, BAR_H, BAR_TICK_PX = 1560, 380, 44, 540, 44
BACKING = (1528, 340, 1836, 960)          # white 90 % panel behind the bar and its ticks
CLI_W, CLI_H, CLI_PAD = 1760, 480, 64
CLI_BOX = ((W - CLI_W) // 2, (H - CLI_H) // 2, (W + CLI_W) // 2, (H + CLI_H) // 2)
CLI_TEXT = "Look at AGENTS.md, simulate a park bench"
CLI_PX = 64
TYPE0, TYPE1, ENTER = 0.15, 1.75, 1.75
CLI_OUT = (2.2, 2.5)
GRID_FAR = (330.0, 600.0)                 # the floor grid fades out above y = 330 (far field)


# -- small helpers -------------------------------------------------------------------------
_FONTS = {}


def font(path, size):
    key = (path, int(size))
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(path, int(size))
    return _FONTS[key]


def text_lines(lines, size, path=FB, fill=INK, lh=None, align="centre"):
    """RGBA image of centred lines; returns (img, anchor) with anchor = (x of the centre
    line / right edge, y of the first baseline) in image px."""
    f = font(path, size)
    lh = lh or size * 1.2
    asc, desc = f.getmetrics()
    widths = [f.getlength(s) for s in lines]
    wmax = int(math.ceil(max(widths))) + 8
    hh = int(math.ceil(asc + desc + lh * (len(lines) - 1))) + 8
    img = Image.new("RGBA", (wmax, hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k, (s, w) in enumerate(zip(lines, widths)):
        x = (wmax - w) / 2 if align == "centre" else wmax - 4 - w
        d.text((x, 4 + asc + k * lh), s, font=f, fill=fill + (255,), anchor="ls")
    ax = wmax / 2 if align == "centre" else wmax - 4
    return img, (ax, 4 + asc)


def place(canvas, img, anchor_xy, anchor_img, scale=1.0, alpha=1.0):
    """Composite ``img`` scaled by ``scale`` so its point ``anchor_img`` lands on
    ``anchor_xy``; ``alpha`` multiplies its opacity."""
    if alpha <= 0.002 or scale <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        w0, h0 = img.size
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        img = img.convert("RGBa").resize((w, h), Image.LANCZOS).convert("RGBA")
        anchor_img = (anchor_img[0] * w / w0, anchor_img[1] * h / h0)
    if alpha < 0.998:
        a = np.asarray(img).copy()
        a[..., 3] = (a[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        img = Image.fromarray(a, "RGBA")
    x, y = round(anchor_xy[0] - anchor_img[0]), round(anchor_xy[1] - anchor_img[1])
    _paste(canvas, img, x, y)


def _paste(canvas, img, x, y):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.width, x + img.width), min(canvas.height, y + img.height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas.alpha_composite(img.crop((x0 - x, y0 - y, x1 - x, y1 - y)), dest=(x0, y0))


def load_premul(path, size=None):
    """float32 premultiplied RGBA in [0, 1], resized (premultiplied, Lanczos) to size."""
    im = Image.open(path).convert("RGBA")
    if size and im.size != tuple(size):
        im = im.convert("RGBa").resize(size, Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255
    else:
        a = np.asarray(im, np.float32) / 255
        a[..., :3] *= a[..., 3:4]
    return a


def over(dst, src):
    dst *= (1 - src[..., 3:4])
    dst += src
    return dst


def ghost(undeformed_a, deformed_a, width, gap, alpha):
    """The undeformed silhouette's outline (``width`` px just outside it), kept only
    where it lies more than ``gap`` px from the deformed silhouette: premultiplied RGBA."""
    out = np.zeros(undeformed_a.shape + (4,), np.float32)
    m = undeformed_a > 0.5
    if not m.any() or alpha <= 0:
        return out
    ys, xs = np.nonzero(m | (deformed_a > 0.5))
    p = int(width + gap) + 3
    y0, y1 = max(0, ys.min() - p), min(m.shape[0], ys.max() + p + 1)
    x0, x1 = max(0, xs.min() - p), min(m.shape[1], xs.max() + p + 1)
    mu = m[y0:y1, x0:x1]
    md = deformed_a[y0:y1, x0:x1] > 0.5
    d_u = distance_transform_edt(~mu)
    d_d = distance_transform_edt(~md)
    a = np.clip(width + 0.5 - d_u, 0, 1) * ~mu
    a *= np.clip(d_d - gap, 0, 1)                  # 1 - dilate(deformed, gap), soft-edged
    a *= alpha
    sub = out[y0:y1, x0:x1]
    sub[..., 3] = a
    sub[..., :3] = a[..., None] * GHOST_RGB
    return out


def to_image(premul):
    a = np.clip(premul, 0, 1)
    rgb = np.where(a[..., 3:4] > 1e-6, a[..., :3] / np.maximum(a[..., 3:4], 1e-6), 0)
    out = np.concatenate([rgb, a[..., 3:4]], -1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


class _Measure:
    """ar_figure.balanced_wrap's page, measured with PIL's Arial Italic (px for pt)."""
    def width(self, text, fs, **kw):
        return font(FI, int(round(fs))).getlength(text)


def prompt_label(instr):
    """The object's prompt as the agentic-results figure prints it on its card: the
    objects_50 wording, curly quotes, typographic apostrophes, Arial Italic in ink,
    wrapped by ar_figure.balanced_wrap.  One size, PROMPT_PX, for every tile, in at most
    two lines across the cell; a prompt that cannot fit two lines at that size gets
    the largest size that does (none does now: the school chair, which needed 20 px,
    was swapped for a bar stool)."""
    s_ = f"\u201c{F.typographic(instr)}\u201d"
    width = 0.94 * TL.CELL_W
    for px in range(PROMPT_PX, 9, -1):
        lines = F.balanced_wrap(_Measure(), s_, width, px, style="italic")
        if len(lines) <= 2 and all(_Measure().width(x, px) <= width for x in lines):
            return lines, px
    raise AssertionError(f"{s_!r} does not fit its cell in two lines")


def name_anchor(c, r, n_lines, px):
    """(centre x, first baseline) so the label block sits centred in the cell's band."""
    x0, y0, x1, _ = TL.cell_box(c, r)
    pitch, cap = PROMPT_LH * px, 0.716 * px
    return 0.5 * (x0 + x1), y0 + TL.NAME_BAND / 2 - ((n_lines - 1) * pitch + cap) / 2 + cap


# -- fixed artwork -------------------------------------------------------------------------
class Art:
    def __init__(self, stats):
        self.st = stats
        # TL.TITLES' last entry, two lines (1228 / 1104 px at 72 px: >= 346 px side margins)
        self.nitsche, self.nitsche_anchor = text_lines(
            TL.TITLES[-1][1].split("\n"), TITLE_PX, lh=TITLE_B2 - TITLE_B1)
        lines = TL.NEW_TITLE[2].split("\n")       # both lines one size, as large as fits
        w1 = max(font(FB, 100).getlength(s) for s in lines) / 100
        self.new_px = min(NEW_PX_MAX, math.floor((TL.GRID_X1 - TL.GRID_X0) / w1))
        self.new, self.new_anchor = text_lines(lines, self.new_px, lh=NEW_LH)
        self.names = {}
        for instr in [stats["bench"]["instruction"]] + [t["instruction"] for t in stats["tiles"]]:
            lines, px = prompt_label(instr)
            img, anc = text_lines(lines, px, path=FI, lh=PROMPT_LH * px)
            self.names[instr] = (img, anc, len(lines), px)
        self.bar = self._bar()
        self.cli_box = self._cli_box()

    def _bar(self):
        """Upright: 5 GPa at the bottom, 200 GPa at the top, ticks and labels on its
        right (the unit on the top label only)."""
        lo, hi = (math.log10(v) for v in (5.0, 200.0))
        pad = 30
        img = Image.new("RGBA", (BAR_W + 260, BAR_H + 2 * pad), (0, 0, 0, 0))
        t = np.linspace(1, 0, BAR_H)
        rgb = (A.e_rgb(t) * 255 + 0.5).astype(np.uint8)
        img.paste(Image.fromarray(np.repeat(rgb[:, None], BAR_W, 1), "RGB"), (0, pad))
        d = ImageDraw.Draw(img)
        d.rectangle((0, pad - 1, BAR_W, pad + BAR_H), outline=BAR_OUTLINE + (255,), width=2)
        f = font(FR, BAR_TICK_PX)
        for v in (5, 10, 20, 50, 100, 200):
            y = pad + BAR_H - (math.log10(v) - lo) / (hi - lo) * BAR_H
            d.line((BAR_W, y, BAR_W + 10, y), fill=INK + (255,), width=2)
            d.text((BAR_W + 18, y), "200 GPa" if v == 200 else str(v), font=f,
                   fill=INK + (255,), anchor="lm")
        return img, pad

    def _cli_box(self):
        img = Image.new("RGBA", (CLI_W + 8, CLI_H + 8), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((2, 2, CLI_W + 2, CLI_H + 2), 28, fill=_hex("#1f1f1f") + (255,),
                            outline=_hex("#d97757") + (255,), width=3)
        f = font(MONO, 64)
        d.text((CLI_PAD, 118), "✻", font=f, fill=_hex("#d97757") + (255,), anchor="ls")
        d.text((CLI_PAD + f.getlength("✻ "), 118), "Welcome to Claude Code", font=f,
               fill=_hex("#e6e6e6") + (255,), anchor="ls")
        d.line((CLI_PAD, 182, CLI_W - CLI_PAD, 182), fill=_hex("#555555") + (255,), width=2)
        return img


def typed_chars(t):
    """Characters typed by t: 40 keys between TYPE0 and TYPE1 with per-key jitter."""
    n = len(CLI_TEXT)
    gaps = np.array([1.0 + 0.55 * math.sin(2.7 * k + 0.9) * math.cos(1.3 * k) for k in range(n)])
    gaps[[k for k, ch in enumerate(CLI_TEXT) if ch == " "]] *= 1.35      # a beat at word breaks
    cum = TYPE0 + (TYPE1 - TYPE0 - 0.05) * np.cumsum(gaps) / gaps.sum()
    return int((cum <= t).sum())


def draw_cli(canvas, art, t):
    if t >= CLI_OUT[1]:
        return
    a = 1 - TL.smooth(TL.ramp(t, CLI_OUT[0], CLI_OUT[0] + 0.2))     # gone in 0.2 s; moves for 0.3
    dy = -300 * TL.ease_io(TL.ramp(t, *CLI_OUT))
    img = art.cli_box.copy()
    d = ImageDraw.Draw(img)
    f = font(MONO, CLI_PX)
    px, py = CLI_PAD, 300
    d.text((px, py), "> ", font=f, fill=_hex("#9aa1ab") + (255,), anchor="ls")
    x = px + f.getlength("> ")
    s = CLI_TEXT[:typed_chars(t)]
    d.text((x, py), s, font=f, fill=_hex("#e6e6e6") + (255,), anchor="ls")
    typing = TYPE0 <= t < ENTER
    if typing or int(t * 4) % 2 == 0:            # blinks at 2 Hz while idle
        cw = f.getlength("M")
        cx, cy = (x + f.getlength(s), py) if t < ENTER else (px, py + 96)
        d.rectangle((cx + 2, cy - 50, cx + cw - 2, cy + 12), fill=_hex("#e6e6e6") + (255,))
    place(canvas, img, (CLI_BOX[0] - 2, CLI_BOX[1] - 2 + dy), (0, 0), alpha=a)


# -- titles --------------------------------------------------------------------------------
def dip_alpha(t, i):
    """Title i's opacity: in over TITLE_IN after the previous title's TITLE_OUT, out over
    TITLE_OUT when the next one starts."""
    starts = [s for s, _ in TL.TITLES]
    a = TL.smooth(TL.ramp(t, starts[i] + TL.TITLE_OUT, starts[i] + TL.TITLE_OUT + TL.TITLE_IN))
    if i + 1 < len(starts):
        a *= 1 - TL.smooth(TL.ramp(t, starts[i + 1], starts[i + 1] + TL.TITLE_OUT))
    return a


def draw_titles(canvas, art, t, e_move):
    f = font(FB, TITLE_PX)
    titles = TL.TITLES
    # scene 2-3: "Shape generation", then " with texture" slides in as the line re-centres
    a0 = TL.smooth(TL.ramp(t, titles[0][0], titles[0][0] + TL.TITLE_IN)) * (
        1 - TL.smooth(TL.ramp(t, titles[2][0], titles[2][0] + TL.TITLE_OUT)))
    if a0 > 0:
        base, full = titles[0][1], titles[1][1]
        wb, wf = f.getlength(base), f.getlength(full)
        p = TL.ease_io(TL.ramp(t, titles[1][0], titles[1][0] + 0.4))
        xl = W / 2 - (wb + (wf - wb) * p) / 2
        img, anc = text_lines([base], TITLE_PX)
        place(canvas, img, (xl + wb / 2, TITLE_B1), anc, alpha=a0)
        if p > 0:
            suf, sanc = text_lines([full[len(base):]], TITLE_PX)
            ws = f.getlength(full[len(base):])
            place(canvas, suf, (xl + wb + ws / 2, TITLE_B1), sanc, alpha=a0 * p)
    for i in range(2, len(titles) - 1):
        a = dip_alpha(t, i)
        if a > 0:
            img, anc = text_lines(titles[i][1].split("\n"), TITLE_PX, lh=TITLE_B2 - TITLE_B1)
            place(canvas, img, (W / 2, TITLE_B1), anc, alpha=a)
    # the last title (the optimisation beat's): in after the dip; in scene 8 it shrinks
    # into the bench's cell while it fades, and the bench's name takes its place
    a = dip_alpha(t, len(titles) - 1) * (1 - TL.smooth(TL.ramp(e_move, 0.25, 0.8)))
    _, _, bl, bpx = art.names[art.st["bench"]["instruction"]]
    nx, ny = name_anchor(0, 0, bl, bpx)
    if a > 0:
        sc = 1 + (0.3 - 1) * e_move
        ax = W / 2 + (nx - W / 2) * e_move
        ay = TITLE_B1 + (ny - TITLE_B1) * e_move
        place(canvas, art.nitsche, (ax, ay), art.nitsche_anchor, scale=sc, alpha=a)
    a = TL.smooth(TL.ramp(e_move, 0.5, 1.0))
    if a > 0:
        img, anc, _, _ = art.names[art.st["bench"]["instruction"]]
        place(canvas, img, (nx, ny), anc, alpha=a)
    a = TL.smooth(TL.ramp(t, TL.NEW_TITLE[0], TL.NEW_TITLE[1]))
    if a > 0:
        place(canvas, art.new, (W / 2, NEW_B), art.new_anchor, alpha=a)


_BAND = None


def draw_title_backing(canvas, t):
    """While the octree is up, its domain box (taller than the bench) reaches into the
    title band: a white band (92 % to y = 205, gone by 250) keeps the two-line title on
    white."""
    global _BAND
    w = TL.look_weights(t).get("octree", 0.0)
    if w <= 0:
        return
    if _BAND is None:
        y = np.arange(H, dtype=np.float32)
        a = np.clip((250 - y) / 45, 0, 1) * 0.92
        band = np.zeros((H, W, 4), np.uint8)
        band[..., :3] = 255
        band[..., 3] = (a * 255)[:, None].astype(np.uint8)
        _BAND = band
    b = _BAND.copy()
    b[..., 3] = (b[..., 3] * w).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(b, "RGBA"))


def draw_label(canvas, t, k):
    if t < TL.LABEL_IN[0] or t > TL.LABEL_OUT[1]:
        return
    q = TL.ramp(t, *TL.LABEL_IN)
    a = TL.smooth(q) * (1 - TL.smooth(TL.ramp(t, *TL.LABEL_OUT)))
    sc = (0.6 + 0.4 * TL.ease_ob(q)) * (1 - 0.15 * TL.smooth(TL.ramp(t, *TL.LABEL_OUT)))
    n = int(round(TL.bench_osc(t, k)))
    img, anc = text_lines([f"×{n}"], LABEL_PX, fill=BRICK, align="right")
    place(canvas, img, (LABEL_X, LABEL_B), anc, scale=sc, alpha=a)


def draw_bar(canvas, art, t):
    a = TL.smooth(TL.ramp(t, *TL.BAR_IN)) * (1 - TL.smooth(TL.ramp(t, *TL.BAR_OUT)))
    if a <= 0:
        return
    rise = 20 * (1 - TL.ease_o(TL.ramp(t, *TL.BAR_IN)))
    back = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(back).rounded_rectangle(BACKING, 12, fill=(255, 255, 255, int(255 * 0.9 * a)))
    canvas.alpha_composite(back)
    img, pad = art.bar
    place(canvas, img, (BAR_X0, BAR_Y + rise), (0, pad), alpha=a)


# -- the octree box frame ------------------------------------------------------------------
FRAME_RGB = np.array(_hex("#5f6672"), np.float32) / 255     # visible edges, 2 px
_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
              (0, 4), (1, 5), (2, 6), (3, 7)]
# each box face (axis, side) and the 4 corners on it; corner bit i = upper end on axis i
_BOX_FACES = [(ax, sd, [k for k in range(8) if ((k >> ax) & 1) == sd]) for ax in range(3)
              for sd in (0, 1)]


def box_frame(b, heading):
    """The complete (uncut) octree box's 12 edges, projected with the render's camera,
    as premultiplied RGBA (H, W, 4): drawn over the octree look so the removed front
    half's extent shows (the user, 2026-10-26).  An edge with a face of the box turned to
    the camera is visible -- nothing inside a convex box can hide it -- and is drawn
    2 px #5f6672.  The other three lie behind the kept half and are left out: drawn
    dashed and lighter, they crossed the cut face and read as cell outlines.  Drawn at
    2x, averaged down."""
    lo, hi = (np.asarray(v, float) for v in b["octree"]["drawn"]["drawn_box_m"])
    corners = np.array([[hi[i] if (k >> i) & 1 else lo[i] for i in range(3)] for k in range(8)])
    f, _, _ = CAM.basis(heading, b["elev"])
    cam = np.asarray(b["centre"], float) - b["distance"] * f
    front = set()
    for ax, sd, ks in _BOX_FACES:
        nrm = np.zeros(3)
        nrm[ax] = 1.0 if sd else -1.0
        if (cam - corners[ks[0]]) @ nrm > 0:
            front.add((ax, sd))
    x, y = CAM.project(corners, b["centre"], heading, b["elev"], b["distance"], W, H)
    k = 2
    vis = Image.new("L", (W * k, H * k), 0)
    dv = ImageDraw.Draw(vis)
    for i, j in _BOX_EDGES:
        ax = [a for a in range(3) if ((i >> a) & 1) == ((j >> a) & 1)]
        shown = any((a, (i >> a) & 1) in front for a in ax)
        p0, p1 = np.array([x[i], y[i]]) * k, np.array([x[j], y[j]]) * k
        if shown:
            dv.line([tuple(p0), tuple(p1)], fill=255, width=4)
    a = np.asarray(vis.resize((W, H), Image.BOX), np.float32) / 255
    out = np.zeros((H, W, 4), np.float32)
    out[..., 3] = a
    out[..., :3] = a[..., None] * FRAME_RGB
    return out


# -- the bench layer -----------------------------------------------------------------------
_FAR = None


def far_fade():
    """The grid's far-field fade (y in the full frame): 0 above GRID_FAR[0], 1 below [1]."""
    global _FAR
    if _FAR is None:
        y = np.arange(H, dtype=np.float32)
        x = np.clip((y - GRID_FAR[0]) / (GRID_FAR[1] - GRID_FAR[0]), 0, 1)
        _FAR = (x * x * (3 - 2 * x))[:, None, None]
    return _FAR


_PMASK = None
ARROW_OUTLINE_PX = 4.0   # the white halo (2 px left navy arrows lost on the navy junction)
ARROW_Z_EPS = 1e-7       # depth: arrow nearer than the body-only render (same clipping)
ARROW_PIECE_LO, ARROW_PIECE_HI = 15.0, 60.0   # px: a visible arrow piece's weight ramps 0 -> 1
ARROW_OUTLINE_RGB = np.array(_hex("#ffffff"), np.float32) / 255   # a white halo (the user: the
                         # dark arrows need separating from the navy surface AND the white body)
_XMASK = None


def body_clear_mask(a):
    """(1, W, 1) factor on the body (and its arrows) while the plot is up: 1 left of
    BODY_CLEAR_X[0], smoothstep to 0 at BODY_CLEAR_X[1] (a 120 px feather), so the
    close-up's bench, running off to the right at some headings, never meets the plot."""
    global _XMASK
    if _XMASK is None:
        x0, x1 = TL.BODY_CLEAR_X
        q = np.clip((x1 - np.arange(W, dtype=np.float32)) / (x1 - x0), 0, 1)
        _XMASK = (q * q * (3 - 2 * q))[None, :, None]
    return 1.0 - a * (1.0 - _XMASK)


def plot_grid_mask(t):
    """(H, W, 1) factor on the floor grid: 0 under the plot (PLOT_CLEAR) while it is up,
    rising smoothly to 1 over PLOT_FEATHER px around it, so no grid line crosses the
    axes or labels and there is no edge anywhere."""
    global _PMASK
    a = TL.plot_opacity(t)
    if a <= 0:
        return 1.0
    if _PMASK is None:
        x0, y0, x1, y1 = PLOT_CLEAR
        xx = np.arange(W, dtype=np.float32)[None, :]
        yy = np.arange(H, dtype=np.float32)[:, None]
        dx = np.maximum(np.maximum(x0 - xx, xx - x1), 0)
        dy = np.maximum(np.maximum(y0 - yy, yy - y1), 0)
        q = np.clip(np.sqrt(dx * dx + dy * dy) / PLOT_FEATHER, 0, 1)
        _PMASK = (q * q * (3 - 2 * q))[..., None]           # 0 inside, 1 beyond the feather
    return 1.0 - a * (1.0 - _PMASK)


def layer_geom(t):
    """(width, height, pad x, pad y) of the bench layer at t: the frame, or the shrink's
    oversized canvas (dv_timeline.oversize) with the frame centred in it."""
    ov = TL.oversize(t)
    lw, lh = int(round(ov * W)), int(round(ov * H))
    return lw, lh, (lw - W) // 2, (lh - H) // 2


def _padded(m, py, px):
    """A frame-coordinate mask (H or 1, W or 1, 1) extended to the oversized layer by
    its edge values (frame masks are constant beyond the frame)."""
    if py == 0 and px == 0 or np.isscalar(m):
        return m
    return np.pad(m, ((py if m.shape[0] > 1 else 0,) * 2, (px if m.shape[1] > 1 else 0,) * 2, (0, 0)),
                  mode="edge")


_GEDGE = {}


def grid_edge_mask(lw, lh, feather=240):
    """The oversized layer's floor grid fades to 0 over ``feather`` px at the layer's
    borders: late in the shrink (scale < 1/2) those borders still land inside the frame,
    and the grid (fading out with the move) must not end in a straight line there."""
    key = (lw, lh)
    if key not in _GEDGE:
        fx = np.clip(np.minimum(np.arange(lw), lw - 1 - np.arange(lw)) / feather, 0, 1)
        fy = np.clip(np.minimum(np.arange(lh), lh - 1 - np.arange(lh)) / feather, 0, 1)
        sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
        _GEDGE[key] = (sm(fy)[:, None] * sm(fx)[None, :]).astype(np.float32)[..., None]
    return _GEDGE[key]


ARROW_SEEN_OFF, ARROW_SEEN_ON = 0.35, 0.50   # hysteresis on an arrow's smoothed visible share
ARROW_SIGMA_FR = 4.0      # its Gaussian smoothing (frames)
ARROW_DWELL_FR = 12       # >= 0.4 s between two changes of its on/off state
ARROW_RAMP_FR = 9         # a fade-in takes >= 0.3 s (slew limit)
ARROW_PAUSE_MIN = 0.6     # the end pause: an arrow shows only while >= 60 % of it is seen
ARROW_FALL_FR = 15        # a fade-out takes >= 0.5 s: no arrow loses more than 1/15 of its alpha
                          # in a frame (the user: arrows must not disappear suddenly)
_FADES = {}


def arrow_fades(work):
    """{frame: [alpha per arrow]}: the ONE visibility of each arrow, every factor that is not
    the per-pixel occlusion itself (the user: arrows blinked when prep's gating and this fade
    fought; later: they must not disappear suddenly).  Target = (its site has passed the
    threshold, dv_prep's t_cross) AND (its visible pixels -- the flat ID shot, kept where the
    combined depth is nearer than the body-only one -- against its side-on area,
    Gaussian-smoothed over sigma = 4 frames, on at >= 50 %, off when it or the raw share is
    below 35 %, at most one change per 0.4 s).  The frames are rendered offline, so the fade LOOKS AHEAD: a forward pass
    rises at <= 1/9 per frame, then a backward pass caps every frame at (the next frames'
    alpha + 1/15 per frame between), so an arrow about to go behind the body (or off) has
    already faded by the frame it does, and no fall is faster than 1/15 per frame.  The
    beat's global arrow opacity (dv_timeline.arrows_opacity, its own fall <= 1/15 per frame)
    is folded in as a min -- a product of two falling ramps would fall faster than either --
    and the compose then applies none.  Cached in WORK/arrow_fades.json."""
    if "f" in _FADES:
        return _FADES["f"]
    path = os.path.join(work, "arrow_fades.json")
    if os.path.exists(path):
        _FADES["f"] = json.load(open(path))
        return _FADES["f"]
    from scipy.ndimage import gaussian_filter1d
    fr = os.path.join(work, "frames")
    frames = sorted(int(os.path.basename(p_)[1:5]) for p_ in glob.glob(os.path.join(fr, "oarrows_id", "f*.png")))
    meta = json.load(open(os.path.join(work, "arrow_meta.json")))
    nm = len(meta["t_cross"])
    if not frames or not nm:
        _FADES["f"] = {}
        return {}
    times = np.array(frames, float) / TL.FPS
    share = np.zeros((len(frames), nm))
    occ = np.ones((len(frames), nm))       # the share NOT behind the body: kept ID pixels / all
    for n_, fi in enumerate(frames):
        im_ = np.asarray(Image.open(os.path.join(fr, "oarrows_id", f"f{fi:04d}.png")).convert("RGBA"))
        ids = np.where(im_[..., 3] > 128, im_[..., 0].astype(np.int32), 0)
        # render px per compose px squared (the shrink's frames are oversized: same scale)
        sc2 = (im_.shape[1] / (TL.oversize(times[n_]) * W)) ** 2
        za = np.load(os.path.join(fr, "oarrows_z", f"f{fi:04d}.npy"))
        zb = np.load(os.path.join(fr, "oarrows_body_z", f"f{fi:04d}.npy"))
        kept = za < zb - ARROW_Z_EPS
        if kept.shape != ids.shape:
            kept = np.asarray(Image.fromarray(kept.astype(np.uint8) * 255).resize(
                (ids.shape[1], ids.shape[0]), Image.NEAREST)) > 0
        vis_ = np.bincount(ids[kept].ravel(), minlength=256)
        foot_ = np.bincount(ids.ravel(), minlength=256)    # the ID shot has no body: all of it
        occ[n_] = [vis_[m_] / foot_[m_] if foot_[m_] else 0.0 for m_ in range(1, nm + 1)]
        nom = meta["nominal"].get(str(fi), {})
        for m_ in range(1, nm + 1):
            share[n_, m_ - 1] = min(1.0, vis_[m_] / max(nom.get(str(m_), 1.0) * sc2, 1.0))
    sm = gaussian_filter1d(share, ARROW_SIGMA_FR, axis=0, mode="nearest")
    cap = np.array([TL.arrows_opacity(t) for t in times])
    alpha = np.zeros_like(sm)
    for j in range(nm):
        tc = meta["t_cross"].get(str(j + 1))
        on, last = False, -10 ** 6
        target = np.zeros(len(frames))
        for n_ in range(len(frames)):
            want = on
            if n_ - last >= ARROW_DWELL_FR:
                # off on the smoothed OR the raw share: a sudden occlusion (the crossfade's
                # dominant iterate switching to a grown body over the arrow's base) shows
                # in the raw share first, and the look-ahead must finish before it
                if on and min(sm[n_, j], share[n_, j]) < ARROW_SEEN_OFF:
                    want = False
                elif not on and sm[n_, j] >= ARROW_SEEN_ON:
                    want = True
            if want != on:
                on, last = want, n_
            target[n_] = 1.0 if (on and tc is not None and times[n_] >= tc) else 0.0
        # the end pause (the user: every arrow fully on or fully off there): dv_prep's
        # pause_on arrows (>= 60 % unoccluded and not end-on in most of its frames, by ray
        # casting) are on from its start (a fade-in, if any, done 0.3 s in), until the
        # first frame the body hides more than 40 % of one (its ID pixels against the
        # depth test) (the look-ahead fade-out ends
        # there), never back; the others are off through it (faded out ahead).  After
        # the pause nothing turns on again (the end fade).
        pause = (times >= TL.ITERS[1] - 1e-6) & (times < TL.SHRINK0)
        if pause.any():
            ip = np.nonzero(pause)[0]
            pon = meta.get("pause_on", {}).get(str(j + 1), True) and tc is not None and tc <= TL.ITERS[1] + 1e-6
            # full at the pause's end means up to 0.3 alpha 10 frames into the end fade; if
            # the body hides it then, it would vanish at that alpha: off through the pause
            early = np.nonzero((times >= TL.SHRINK0) & (cap > 0.3))[0]
            if pon and len(early) and (occ[early, j] < ARROW_PAUSE_MIN).any():
                pon = False
            if pon:     # full from the pause's first frame: its fade-in ends there
                target[max(0, ip[0] - ARROW_RAMP_FR):ip[0]] = 1.0
            target[ip] = 1.0 if pon else 0.0
            target[ip[-1] + 1:] = target[ip[-1]]      # and so into the end fade
            # (within the pause only: in the shrink the end fade is already taking every
            # arrow out, and a look-ahead there would start in the pause's last frames)
            low = ip[occ[ip, j] < ARROW_PAUSE_MIN]    # exact: its pixels the body hides
            if pon and len(low):
                target[low[0]:] = 0.0
        a_ = np.minimum(target, cap)
        for n_ in range(1, len(frames)):          # rises: slew-limited forward, and none
            a_[n_] = min(a_[n_], a_[n_ - 1] + (1.0 / ARROW_RAMP_FR    # once the end fade has
                                               if times[n_] < TL.ARROWS_OUT[0] else 0.0))  # begun
        for n_ in range(len(frames) - 2, -1, -1):  # falls: started early enough (look-ahead)
            a_[n_] = min(a_[n_], a_[n_ + 1] + 1.0 / ARROW_FALL_FR)
        alpha[:, j] = a_
    out = {str(fi): [float(x) for x in alpha[n_]] for n_, fi in enumerate(frames)}
    json.dump(out, open(path, "w"))
    json.dump({"frames": frames, "share": share.tolist(), "occ": occ.tolist()},   # for the audit
              open(os.path.join(work, "arrow_share.json"), "w"))
    _FADES["f"] = out
    return out


def bench_layer(work, fr, fi, t, b):
    def p(look):
        return os.path.join(fr, look, f"f{fi:04d}.png")
    lw, lh, px, py = layer_geom(t)
    sz = (lw, lh)
    layer = np.zeros((lh, lw, 4), np.float32)
    g = TL.grid_opacity(t)
    if g > 0:
        gm = g * _padded(far_fade(), py, px) * _padded(plot_grid_mask(t), py, px)
        if lw > W:
            gm = gm * grid_edge_mask(lw, lh)
        layer += load_premul(p("grid"), sz) * gm
    weights = TL.look_weights(t)
    passes = {look: load_premul(p(look), sz) for look in weights if look != "iter"}
    if "iter" in weights:
        # two consecutive iterates, crossfaded (their meshes do not correspond)
        _, _, w1 = TL.vm_iterate(t)
        it = load_premul(p("iterA"), sz)
        if w1 > 1e-6:
            it = it * (1 - w1) + load_premul(p("iterB"), sz) * w1
        passes["iter"] = it
    body = np.zeros_like(layer)
    for look, w in weights.items():
        body += passes[look] * w
    for gl, go in TL.ghosts(t).items():
        und = load_premul(p(gl), sz)[..., 3]
        over(layer, ghost(und, body[..., 3], GHOST_W, GHOST_GAP, GHOST_A * go))
    pa = TL.plot_opacity(t)
    if pa > 0:                     # the close-up's body never reaches the plot
        body *= _padded(body_clear_mask(pa), py, px)
    over(layer, body)
    if "octree" in weights:
        over(layer, box_frame(b, TL.bench_heading(t, b["heading0"])) * weights["octree"])
    if "bc" in weights:
        over(layer, load_premul(p("arrows"), sz) * weights["bc"])
    oa = TL.arrows_opacity(t)
    if oa > 0:                     # the change arrows: their own pass, over the body
        arr = load_premul(p("oarrows"), sz)
        # exact occlusion: "oarrows" is the iterate view rendered with body AND arrows
        # (one renderer, true depth testing); its arrow pixels are where that render's
        # depth is nearer than the same view's body-only depth (render_dv: same camera,
        # same bounds and clipping range).  Everything else in it (the body) is dropped.
        zb_p = os.path.join(fr, "oarrows_body_z", f"f{fi:04d}.npy")
        za_p = os.path.join(fr, "oarrows_z", f"f{fi:04d}.npy")
        if os.path.exists(zb_p) and os.path.exists(za_p):
            zb, za = np.load(zb_p), np.load(za_p)
            vis = (za < zb - ARROW_Z_EPS)
            if za.shape != arr.shape[:2]:          # for the halo's own occlusion (below)
                rs_ = lambda z_: np.asarray(Image.fromarray(z_.astype(np.float32)).resize(  # noqa: E731
                    (arr.shape[1], arr.shape[0]), Image.NEAREST))
                za_l, zb_l = rs_(za), rs_(zb)
            else:
                za_l, zb_l = za, zb
            # ... and only inside the arrows' own footprint (the ID shot, 2 px grown): the
            # two depth shots differ slightly along body silhouettes, and those pixels drew
            # black outline polylines across the body (the user: "weird lines")
            idp_ = os.path.join(fr, "oarrows_id", f"f{fi:04d}.png")
            if os.path.exists(idp_):
                foot = np.asarray(Image.open(idp_).convert("RGBA"))[..., 3] > 0
                if foot.shape != vis.shape:
                    foot = np.asarray(Image.fromarray(foot.astype(np.uint8) * 255).resize(
                        (vis.shape[1], vis.shape[0]), Image.NEAREST)) > 0
                vis &= distance_transform_edt(~foot) <= 2.0
            vis = vis.astype(np.float32)
            if vis.shape != arr.shape[:2]:
                vis = np.asarray(Image.fromarray((vis * 255).astype(np.uint8)).resize(
                    (arr.shape[1], arr.shape[0]), Image.BILINEAR), np.float32) / 255
            arr *= vis[..., None]
        else:
            vis = za_l = zb_l = None
        # per arrow: fade out when its unoccluded share drops (_arrow_fades: slivers of a
        # mostly hidden arrow read as specks); id map from the flat ID shot
        idp = os.path.join(fr, "oarrows_id", f"f{fi:04d}.png")
        fades = arrow_fades(work).get(str(fi))
        fac_px = None
        # the arrows' coverage BEFORE their fades: the piece labels and the halo ring are
        # taken from it, not from the faded alpha -- thresholding the faded alpha at 0.5
        # dropped every arrow's halo (and re-cut its pieces) the frame its fade passed 0.5
        cov = arr[..., 3].copy()
        if fades is not None and os.path.exists(idp):
            im_ = np.asarray(Image.open(idp).convert("RGBA"))
            # only fully covered ID pixels: an AA edge blends the id with the background,
            # and a blended value is another arrow's id (28 at half cover reads as 21-23),
            # which drew 1 px dots of hidden arrows at that arrow's fade
            ids = np.where(im_[..., 3] >= 250, im_[..., 0].astype(np.int32), 0)
            if ids.shape != arr.shape[:2]:
                ids = np.asarray(Image.fromarray(ids.astype(np.uint8)).resize(
                    (arr.shape[1], arr.shape[0]), Image.NEAREST), np.int32)
            lut = np.full(256, -1.0, np.float32)     # -1: no valid id (background, AA
            for m_, fv in enumerate(fades):           # fringe, two ids blended at an edge)
                lut[m_ + 1] = fv
            fac_px = lut[ids]
            ok_ = fac_px >= 0
            if ok_.any():
                # every pixel takes its nearest valid-id pixel's fade, so an arrow's AA
                # fringe (and its outline) fades with it instead of staying as a speck
                _, (iy, ix) = distance_transform_edt(~ok_, return_indices=True)
                fac_px = fac_px[iy, ix]
            else:
                fac_px = np.ones_like(fac_px)
            arr *= fac_px[..., None]
        # a 4 px WHITE halo: the dark arrows sit on a surface coloured by the same change
        # map and blended into it without one; white also lifts them off the white body
        # tiny visible pieces (an arrow's slivers beside a slat or post) fade with their
        # size, continuously -- smoothstep((px - 15) / 45) -- and so does their outline:
        # a hard per-frame drop of small pieces made arrows blink, none left black
        # outline squiggles
        from scipy.ndimage import label
        m = cov > 0.5
        piece_w = None
        if m.any():
            lab_, nlab = label(m)
            sizes = np.bincount(lab_.ravel()).astype(np.float32)
            wv = np.clip((sizes - ARROW_PIECE_LO) / (ARROW_PIECE_HI - ARROW_PIECE_LO), 0, 1)
            wv = wv * wv * (3 - 2 * wv)
            wv[0] = 0.0
            piece_w = wv[lab_]
            _, (iy_, ix_) = distance_transform_edt(~m, return_indices=True)
            piece_w = piece_w[iy_, ix_]           # the AA fringe and outline take their piece's
            arr *= piece_w[..., None]      # (the ring takes piece_w below, so m stays)
        if m.any():
            ys, xs = np.nonzero(m)
            y0, y1 = max(0, ys.min() - 6), min(lh, ys.max() + 7)
            x0, x1 = max(0, xs.min() - 6), min(lw, xs.max() + 7)
            d_ = distance_transform_edt(~m[y0:y1, x0:x1])
            ring = np.zeros((lh, lw, 4), np.float32)
            ra = np.clip(ARROW_OUTLINE_PX + 0.5 - d_, 0, 1) * ~m[y0:y1, x0:x1]
            ring[y0:y1, x0:x1, 3] = ra
            ring[y0:y1, x0:x1, :3] = ra[..., None] * ARROW_OUTLINE_RGB
            if za_l is not None:
                # nor its halo where the body is in front: a halo pixel lies OFF the arrow,
                # so it is hidden where the body there is nearer than the arrow's depth at
                # its nearest arrow pixel (it took the arrow pixels' own mask before, which
                # is 0 off the arrow: only a ~1 px fringe of the halo ever showed)
                _, (iy_r, ix_r) = distance_transform_edt(~m[y0:y1, x0:x1], return_indices=True)
                zan = za_l[y0:y1, x0:x1][iy_r, ix_r]
                ring[y0:y1, x0:x1] *= (zb_l[y0:y1, x0:x1] >= zan - ARROW_Z_EPS)[..., None]
            if piece_w is not None:        # ... and with its piece's size
                ring *= piece_w[..., None]
            if fac_px is not None:         # and the outline fades with its arrow
                _, (iy, ix) = distance_transform_edt(~m[y0:y1, x0:x1], return_indices=True)
                ring[y0:y1, x0:x1] *= fac_px[y0:y1, x0:x1][iy, ix][..., None]
            over(arr, ring)
        # the global arrow opacity is folded into the per-arrow fades (arrow_fades: a min,
        # not a product); applied here only when there are none
        arr = arr * (1.0 if fac_px is not None else oa) * _padded(body_clear_mask(pa), py, px)
        log_dir = os.environ.get("DV_ARROW_LOG")
        if log_dir and fac_px is not None:
            # diagnostics: each arrow's visible weight on screen (sum of its final alpha
            # over its ID-shot pixels), for the per-frame fade audit (DV_ARROW_LOG=DIR)
            os.makedirs(log_dir, exist_ok=True)
            vw = np.bincount(ids.ravel(), weights=arr[..., 3].ravel(), minlength=256)
            full = np.bincount(ids.ravel(), minlength=256)
            json.dump({"vis": [float(vw[m_ + 1]) for m_ in range(len(fades))],
                       "full": [float(full[m_ + 1]) for m_ in range(len(fades))],
                       "fade": [float(x) for x in fades], "oa": float(oa), "alpha_is_fade": True},
                      open(os.path.join(log_dir, f"f{fi:04d}.json"), "w"))
        over(layer, arr)
    return layer


def bench_transform(b, e):
    """(scale, offset x, offset y) of the full-frame bench layer at move progress e: into
    the top-left cell by the tiles' own rule (largest side over the headings it shows,
    at rest and at its peak, = TILE_FILL of the body area's short side).  The layer is
    then pulling out of the beat's close-up (closeup -> 0 over the move) and the body is
    the optimised bench, so the box is dv_prep's box_close_px: the optimised bench, at
    rest and at its peak, under the ORBIT camera over the headings left."""
    x0, y0, x1, y1 = b["cleanup"]["box_close_px"]
    bx, by = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ax0, ay0, ax1, ay1 = TL.body_box(0, 0)
    s_end = TL.TILE_FILL * min(ax1 - ax0, ay1 - ay0) / max(x1 - x0, y1 - y0)
    tx, ty = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    sc = 1 + (s_end - 1) * e
    cx, cy = bx + (tx - bx) * e, by + (ty - by) * e
    return sc, cx - bx * sc, cy - by * sc


def draw_bench(canvas, work, fr, fi, t, b):
    if not TL.bench_visible(t):
        return 0.0
    e = TL.move(t)
    dy, op = TL.rise(t)
    img = to_image(bench_layer(work, fr, fi, t, b) * op)
    sc, ox, oy = bench_transform(b, e)
    _, _, px, py = layer_geom(t)          # the frame's origin inside an oversized layer
    place(canvas, img, (ox, oy + dy), (px, py), scale=sc)
    return e


# -- tiles ---------------------------------------------------------------------------------
def draw_tiles(canvas, art, fr, fi, t, st):
    size = (int(round(2 * TL.TILE_BODY_W)), int(round(2 * TL.TILE_BODY_H)))
    for tile in st["tiles"]:
        i = tile["j"]
        if not TL.tile_visible(t, i):
            continue
        body = load_premul(os.path.join(fr, f"t{i}", f"u_f{fi:04d}.png"), size)
        und = load_premul(os.path.join(fr, f"t{i}", f"g_f{fi:04d}.png"), size)[..., 3]
        lay = ghost(und, body[..., 3], 2 * GHOST_W, 2 * GHOST_GAP, GHOST_A)
        over(lay, body)
        q = TL.ramp(t, TL.tile_pop_time(i), TL.tile_pop_time(i) + TL.TILE_POP)
        a = TL.smooth(q)
        pop = 0.85 + 0.15 * TL.ease_ob(q)
        img = to_image(lay * a)
        c, r = TL.cells()[i]
        x0, y0, x1, y1 = TL.body_box(c, r)
        place(canvas, img, (0.5 * (x0 + x1), 0.5 * (y0 + y1)), (img.width / 2, img.height / 2),
              scale=0.5 * pop)
        nimg, nanc, nl, px = art.names[tile["instruction"]]
        place(canvas, nimg, name_anchor(c, r, nl, px), nanc, alpha=a)


# -- the strain-energy plot ----------------------------------------------------------------
# Right half, beside the close-up (the body fades out across x 840-960).  No panel: a
# white panel's edge cut the floor grid in a hard seam (the user), so the grid itself is
# faded to nothing under the plot over a 180 px feather (plot_grid_mask).  House style
# and the designer's review (2026-11-07): Arial, ink, 4 px #3d424b axes and 4 x 16 px
# outward ticks (the user: thin ones vanished at 360 px), no grid, no
# legend; y from 0 (an honest baseline) with ticks 0 / 0.5 / 1.0; a dashed reference at
# 1.0, a live bracket from 1.0 down to the dot, and in the lower right "-NN %" (64 px bold,
# counting with the iterate) over "Lower → Stiffer" (42 px, #3d424b, medium), both
# right-aligned (_live_note); the surface-change colour bar (mm) sits bottom-left.  Every size is a
# constant here; drawn at 2x, downsampled.
PLOT_PANEL = (1000, 250, 1880, 1000)      # the plot's canvas (transparent: no panel edge)
PLOT_CLEAR = (1020, 270, W, H)            # the floor grid is faded to 0 here under the plot
PLOT_FEATHER = 180                        # ... over this many px outside it (no seam)
PLOT_AX = (1190, 290, 1820, 790)          # x0, y0 (top), x1, y1 (bottom) of the axes box
# the surface-change colour bar under the plot (the bench's colouring in the beat):
# render.py's lut on -CHANGE_MAX..+CHANGE_MAX h, labelled in mm
CBAR = (96, 952, 516, 970)                # x0, y0, x1, y1 of the bar: bottom left (the user),
                                          # its labels >= 56 px from the frame's edges
CBAR_TICKS_MM = (-30, -15, 0, 15, 30)    # seven ticks crowded at 32 px
CBAR_TICK_PX, CBAR_LABEL_PX = 32, 32
CBAR_LABEL = "Surface change (mm)"
PLOT_Y, PLOT_YTICKS = (0.0, 1.0), (0.0, 0.5, 1.0)
PLOT_XTICKS = (0, 10, 20, 30, 40)
PLOT_YLABEL, PLOT_XLABEL = "Strain energy (vs. initial)", "Iteration"
PLOT_LABEL_PX, PLOT_TICK_PX = 44, 40
PLOT_NOTE_PX, PLOT_SUB_PX = 64, 42     # "-NN %" / "Lower -> Stiffer" (the user: the latter more
                                       # visible: 32 -> 42 px, the axis colour, medium weight)
PLOT_SUB_STROKE = 1       # Arial has no medium face: regular + a 1 px stroke at the plot's 2x
                          # supersampling (0.5 px on screen) reads as medium, short of bold
PLOT_NOTE_RGB = _hex("#1f1f24")
PLOT_AXIS_RGB = _hex("#3d424b")           # axes and ticks: darker than the refs, lighter than ink
PLOT_REF_RGB = _hex("#5f6672")            # the dashed 1.0 and the bracket
PLOT_AXIS_PX, PLOT_TICK_W, PLOT_TICK_L = 4, 4, 16   # the user: >= 1 px on a 360 px phone
PLOT_REF_PX, PLOT_BRACKET_PX = 2, 3                 # lighter than the axes
PLOT_LINE_PX, PLOT_DOT_PX, PLOT_RING_PX = 6, 22, 3
PLOT_REF_DASH = (14, 10)                  # the dashed 1.0 reference: dash, gap (px)
PLOT_BRACKET_SERIF = 12                   # the live bracket's top serif half-width
PLOT_NOTE_RX, PLOT_NOTE_Y = 10, 0.20      # the live label: right edge 10 px left of x = 40
                                          # (~1810 px), baseline at y = 0.20 (~690 px): the
                                          # lower-right interior, below the curve's end (0.575),
                                          # its dot and bracket at every iteration (the user)
_PLOT = {}


def _plot_axes(k2=2):
    """The static plot (2x, RGBA, canvas-sized): axes, ticks, labels, the dashed 1.0."""
    x0, y0, x1, y1 = PLOT_PANEL
    img = Image.new("RGBA", ((x1 - x0) * k2, (y1 - y0) * k2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    X = lambda v: ((PLOT_AX[0] - x0) + (PLOT_AX[2] - PLOT_AX[0]) * v / TL.N_ITERS) * k2  # noqa: E731
    Y = lambda v: ((PLOT_AX[3] - y0) - (PLOT_AX[3] - PLOT_AX[1]) * (v - PLOT_Y[0]) / (PLOT_Y[1] - PLOT_Y[0])) * k2  # noqa: E731
    axc, refc = PLOT_AXIS_RGB + (255,), PLOT_REF_RGB + (255,)
    xa, dash, gap = X(0), PLOT_REF_DASH[0] * k2, PLOT_REF_DASH[1] * k2
    while xa < X(TL.N_ITERS):                                  # the dashed 1.0 reference
        d.line((xa, Y(1.0), min(xa + dash, X(TL.N_ITERS)), Y(1.0)), fill=refc, width=PLOT_REF_PX * k2)
        xa += dash + gap
    aw, h_ = PLOT_AXIS_PX * k2, PLOT_AXIS_PX * k2 / 2
    d.line((X(0) - h_, Y(PLOT_Y[0]), X(TL.N_ITERS), Y(PLOT_Y[0])), fill=axc, width=aw)
    d.line((X(0), Y(PLOT_Y[0]) + h_, X(0), Y(PLOT_Y[1])), fill=axc, width=aw)
    ft = font(FR, PLOT_TICK_PX * k2)
    tl, tw_ = PLOT_TICK_L * k2, PLOT_TICK_W * k2
    for v in PLOT_XTICKS:                                       # outward ticks
        d.line((X(v), Y(PLOT_Y[0]), X(v), Y(PLOT_Y[0]) + tl), fill=axc, width=tw_)
        d.text((X(v), Y(PLOT_Y[0]) + tl + 8 * k2), str(v), font=ft, fill=INK + (255,), anchor="mt")
    for v in PLOT_YTICKS:
        d.line((X(0) - tl, Y(v), X(0), Y(v)), fill=axc, width=tw_)
        d.text((X(0) - tl - 10 * k2, Y(v)), f"{v:g}" if v in (0.0,) else f"{v:.1f}", font=ft,
               fill=INK + (255,), anchor="rm")
    fl = font(FR, PLOT_LABEL_PX * k2)
    d.text(((X(0) + X(TL.N_ITERS)) / 2, Y(PLOT_Y[0]) + 82 * k2), PLOT_XLABEL, font=fl,
           fill=INK + (255,), anchor="mt")
    tw = int(fl.getlength(PLOT_YLABEL)) + 8
    timg = Image.new("RGBA", (tw, PLOT_LABEL_PX * k2 * 2), (0, 0, 0, 0))
    ImageDraw.Draw(timg).text((4, PLOT_LABEL_PX * k2), PLOT_YLABEL, font=fl, fill=INK + (255,), anchor="lm")
    timg = timg.rotate(90, expand=True)
    cy = (Y(PLOT_Y[0]) + Y(PLOT_Y[1])) / 2
    img.alpha_composite(timg, (int(X(0) - 118 * k2 - timg.width / 2), int(cy - timg.height / 2)))
    return img, X, Y


def _live_note(d, X, Y, v, cx, cy, k2=2):
    """The live reduction (the user: no pause, the label counts with the iterate): a
    bracket from the dashed 1.0 down to the current dot, at the dot's x, and "-NN %"
    with "Lower -> Stiffer" at a FIXED spot in the plot's lower-right interior, below the
    curve's end, right-aligned (Arial's digits are tabular); neither the curve nor the
    dot nor the bracket reaches it.  Top right, the bracket passed behind the text."""
    refc = PLOT_REF_RGB + (255,)
    s_ = PLOT_BRACKET_SERIF * k2
    bw = PLOT_BRACKET_PX * k2
    ya = Y(1.0)
    yb = cy - (PLOT_DOT_PX / 2 + PLOT_RING_PX + 2) * k2
    if yb - ya > 2 * s_:
        d.line((cx, ya, cx, yb), fill=refc, width=bw)
        d.line((cx - s_, ya, cx + s_, ya), fill=refc, width=bw)
    pct = max(0, int(round((1.0 - v) * 100)))
    fb, fs = font(FB, PLOT_NOTE_PX * k2), font(FR, PLOT_SUB_PX * k2)
    xr, yl = X(TL.N_ITERS) - PLOT_NOTE_RX * k2, Y(PLOT_NOTE_Y)
    d.text((xr, yl), f"\u2212{pct} %" if pct else "0 %", font=fb, fill=PLOT_NOTE_RGB + (255,),
           anchor="rs")                      # no minus on zero (the user); right-aligned,
    d.text((xr, yl + PLOT_SUB_PX * 1.3 * k2), "Lower \u2192 Stiffer", font=fs,   # tabular digits
           fill=PLOT_AXIS_RGB + (255,), anchor="rs", stroke_width=PLOT_SUB_STROKE,
           stroke_fill=PLOT_AXIS_RGB + (255,))


def draw_plot(canvas, st, t):
    """Iteration vs strain energy relative to the start (SE / SE0 of the re-run's
    traj.tsv): the curve draws in step with the iterate on screen, a brick dot on it
    (from (0, 1) as the plot fades in), the live bracket and "-NN %" (_live_note)."""
    a = TL.plot_opacity(t)
    if a <= 0:
        return
    k2 = 2
    if "axes" not in _PLOT:
        _PLOT["axes"] = _plot_axes(k2)
    base, X, Y = _PLOT["axes"]
    img = base.copy()
    d = ImageDraw.Draw(img)
    se = st["bench"]["cleanup"]["se_norm"]
    pos = TL.iter_pos(t)
    k0 = int(math.floor(pos))
    pts = [(X(k), Y(se[k])) for k in range(0, min(k0, TL.N_ITERS) + 1)]
    v = se[-1]
    if k0 < TL.N_ITERS:
        v = se[k0] + (se[k0 + 1] - se[k0]) * (pos - k0)      # the iterate crossfade's own mix
        pts.append((X(pos), Y(v)))
    if len(pts) > 1:
        d.line(pts, fill=INK + (255,), width=PLOT_LINE_PX * k2, joint="curve")
    r, rr = PLOT_DOT_PX * k2 / 2, (PLOT_DOT_PX / 2 + PLOT_RING_PX) * k2
    cx, cy = pts[-1]
    _live_note(d, X, Y, v, cx, cy, k2)
    d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(255, 255, 255, 255))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=BRICK + (255,))
    img = img.resize((img.width // k2, img.height // k2), Image.LANCZOS)
    place(canvas, img, PLOT_PANEL[:2], (0, 0), alpha=a)


def draw_change_bar(canvas, st, t):
    """The surface-change colour bar, bottom left over the close-up (the user), on a soft
    white backing (no panel edge), fading with the change colouring (the "iter" look)."""
    a = TL.look_weights(t).get("iter", 0.0)
    if a <= 0:
        return
    if "cbar" not in _PLOT:
        _PLOT["cbar"] = _change_bar(st)
    cb, cb_xy = _PLOT["cbar"]
    place(canvas, cb, cb_xy, (0, 0), alpha=a)


def _change_bar(st):
    """The surface-change colour bar (render.py's lut, -CHANGE_MAX..+CHANGE_MAX h), ticks
    in mm (h = 11.36 mm), its label above it, on a soft white backing (a blurred rounded
    patch: grid and body lines never cross the labels, and there is no hard edge); RGBA
    and its top-left (frame px)."""
    from PIL import ImageFilter
    import dv_change
    hmm = st["bench"]["cleanup"]["change_colour"]["h_mm"]
    x0, y0, x1, y1 = CBAR
    bw, bh = x1 - x0, y1 - y0
    m = 70                                         # the backing's soft margin
    img = Image.new("RGBA", (bw + 2 * m, bh + 2 * m + 90), (0, 0, 0, 0))
    ox, oy = m, m + 44                             # the bar's top-left in img
    back = Image.new("L", img.size, 0)
    ImageDraw.Draw(back).rounded_rectangle((m - 30, m - 14, m + bw + 30, oy + bh + 66), 18, fill=230)
    back = back.filter(ImageFilter.GaussianBlur(16))
    white = Image.new("RGBA", img.size, (255, 255, 255, 0))
    white.putalpha(back)
    img.alpha_composite(white)
    t = np.linspace(-dv_change.CHANGE_MAX, dv_change.CHANGE_MAX, bw)
    rgb = (dv_change.change_rgb(t) * 255 + 0.5).astype(np.uint8)
    img.paste(Image.fromarray(np.repeat(rgb[None], bh, 0), "RGB"), (ox, oy))
    d = ImageDraw.Draw(img)
    d.rectangle((ox - 1, oy - 1, ox + bw, oy + bh), outline=PLOT_REF_RGB + (255,), width=2)
    ft = font(FR, CBAR_TICK_PX)
    span = 2 * dv_change.CHANGE_MAX * hmm
    for v in CBAR_TICKS_MM:
        x = ox + (v + span / 2) / span * bw
        d.line((x, oy + bh, x, oy + bh + 8), fill=PLOT_AXIS_RGB + (255,), width=3)
        d.text((x, oy + bh + 12), (f"+{v}" if v > 0 else ("\u2212" + str(-v) if v < 0 else "0")),
               font=ft, fill=INK + (255,), anchor="mt")
    d.text((ox, oy - 12), CBAR_LABEL, font=font(FR, CBAR_LABEL_PX), fill=INK + (255,), anchor="ls")
    return img, (x0 - ox, y0 - oy)


# -- one frame -----------------------------------------------------------------------------
_ART = {}


def compose(work, fi, t, out_scale):
    if "art" not in _ART:
        st = json.load(open(os.path.join(work, "dv_stats.json")))
        _ART["st"], _ART["art"] = st, Art(st)
    st, art = _ART["st"], _ART["art"]
    fr = os.path.join(work, "frames")
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw_cli(canvas, art, t)       # under the bench: the rising body covers the leaving box
    e = draw_bench(canvas, work, fr, fi, t, st["bench"])
    draw_bar(canvas, art, t)
    draw_tiles(canvas, art, fr, fi, t, st)
    draw_title_backing(canvas, t)
    draw_titles(canvas, art, t, e)
    draw_label(canvas, t, st["bench"]["k"])
    draw_plot(canvas, st, t)
    draw_change_bar(canvas, st, t)
    img = canvas.convert("RGB")
    if out_scale != 1:
        img = img.resize((round(W * out_scale), round(H * out_scale)), Image.LANCZOS)
    return img


def _job(a):
    work, fi, t, out_scale, out_dir = a
    path = os.path.join(out_dir, f"f{fi:04d}.png")
    compose(work, fi, t, out_scale).save(path)
    return path


def contact_sheet(out_dir, sheet, n=24, cols=6, width=480):
    frames = sorted(glob.glob(os.path.join(out_dir, "f*.png")))
    pick = [frames[round(k * (len(frames) - 1) / (n - 1))] for k in range(n)]
    h = round(width * H / W)
    rows = math.ceil(n / cols)
    img = Image.new("RGB", (cols * width + (cols + 1) * 8, rows * (h + 34) + 8), "white")
    d = ImageDraw.Draw(img)
    for k, p in enumerate(pick):
        fi = int(os.path.basename(p)[1:5])
        im = Image.open(p).convert("RGB").resize((width, h), Image.LANCZOS)
        x, y = 8 + (k % cols) * (width + 8), 8 + (k // cols) * (h + 34)
        img.paste(im, (x, y + 26))
        d.rectangle((x - 1, y + 25, x + width, y + 26 + h), outline=(200, 200, 205))
        d.text((x, y + 20), f"t = {fi / TL.FPS:5.2f} s  (frame {fi})", font=font(FR, 20), fill=INK, anchor="ls")
    img.save(sheet)
    print(f"wrote {sheet}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=min(24, os.cpu_count() or 4))
    ap.add_argument("--only", default=None, help="comma-separated frame numbers")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--phone", default=None, help="write key frames at 360 px wide here")
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    out_dir = os.path.join(work, "out")
    if args.sheet:
        contact_sheet(out_dir, args.sheet, args.n)
        return
    if args.phone:
        os.makedirs(args.phone, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(out_dir, "f*.png"))):
            fi = int(os.path.basename(p)[1:5])
            if fi % 30 == 0:
                im = Image.open(p).convert("RGB")
                im.resize((360, round(360 * H / W)), Image.LANCZOS).save(os.path.join(args.phone, f"p{fi:04d}.png"))
        return
    os.makedirs(out_dir, exist_ok=True)
    times = TL.frame_times(args.step)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        times = [(f, t) for f, t in TL.frame_times(1) if f in keep]
    arrow_fades(work)              # once, before the workers: it reads every arrow frame
    jobs = [(work, fi, t, args.out_scale, out_dir) for fi, t in times]
    with Pool(args.procs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            if k % 60 == 0:
                print(f"  {k}/{len(jobs)} frames", flush=True)
    print(f"wrote {len(jobs)} frames to {out_dir}")


if __name__ == "__main__":
    main()
