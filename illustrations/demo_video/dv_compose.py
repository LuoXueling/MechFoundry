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
            img, anc = text_lines([titles[i][1]], TITLE_PX)
            place(canvas, img, (W / 2, TITLE_B1), anc, alpha=a)
    # the Nitsche title: in after the dip; in scene 8 it shrinks into the bench's cell
    # while it fades, and the bench's name takes its place
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
    n = int(round(TL.osc(t, k)))
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


def bench_layer(work, fr, fi, t, b):
    def p(look):
        return os.path.join(fr, look, f"f{fi:04d}.png")
    layer = np.zeros((H, W, 4), np.float32)
    g = TL.grid_opacity(t)
    if g > 0:
        layer += load_premul(p("grid"), (W, H)) * (g * far_fade())
    weights = TL.look_weights(t)
    passes = {look: load_premul(p(look), (W, H)) for look in weights}
    go = TL.ghost_opacity(t)
    if go > 0:
        und = load_premul(p("clay"), (W, H))[..., 3]
        dfm = passes["solve"][..., 3] if "solve" in passes else und
        over(layer, ghost(und, dfm, GHOST_W, GHOST_GAP, GHOST_A * go))
    body = np.zeros_like(layer)
    for look, w in weights.items():
        body += passes[look] * w
    over(layer, body)
    if "octree" in weights:
        over(layer, box_frame(b, TL.bench_heading(t, b["heading0"])) * weights["octree"])
    if "bc" in weights:
        over(layer, load_premul(p("arrows"), (W, H)) * weights["bc"])
    return layer


def bench_transform(b, e):
    """(scale, offset x, offset y) of the full-frame bench layer at move progress e: into
    the top-left cell by the tiles' own rule (largest side over the clip's headings,
    at rest and at its peak, = TILE_FILL of the body area's short side)."""
    x0, y0, x1, y1 = b["box_all_px"]
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
    place(canvas, img, (ox, oy + dy), (0, 0), scale=sc)
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
    jobs = [(work, fi, t, args.out_scale, out_dir) for fi, t in times]
    with Pool(args.procs) as pool:
        for k, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            if k % 60 == 0:
                print(f"  {k}/{len(jobs)} frames", flush=True)
    print(f"wrote {len(jobs)} frames to {out_dir}")


if __name__ == "__main__":
    main()
