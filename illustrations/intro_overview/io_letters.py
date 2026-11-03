"""intro_overview's panel letters: **a** (challenge_overview, top) and **b** (method_overview,
bottom), as one transparent page the size of the stacked page that io_compose.tex lays over it.

The letters are the house style of the lettered figures (``_shared/loop_chain.py``: 8 pt bold
lower case Arial, ink #1a1a21, Type 42) and are drawn here by matplotlib rather than set by
pdfTeX, which has no Arial metrics: plain pdfTeX only embeds this PDF as one more form XObject,
so both sub-figures stay the vector PDFs their own builds wrote.

Placement: each letter's ink starts ML from the page's left edge (loop_chain's letter gutter).
Vertically each letter has its own rule (the user, 2026-10-17; one shared corner rule left "a"
~1.6 mm above the challenge figure's first line of text):
  a  sits ON the challenge figure's column-head baseline ("1  SHAPE" ... "5  DESIGN"), so the
     letter reads as the first word of that header row.  The baseline is measured, not typed:
     pdftotext -bbox finds the word "MESH" (flat-bottomed capitals, no overshoot) and the
     lowest ink row inside that box on the 600 dpi raster is the baseline.  The header row's
     left end is free since its "PROMPT" head went (the rule under the heads starts at column 1).
  b  keeps the old rule: an ascender (bold "b") INSET below method_overview's top edge; its
     violet frame runs up its left side 1.5 mm in and turns into the "AI operator" pill
     ~3.6 mm down, so the corner is the one free spot.
Gate: each sub-figure is rasterised at 600 dpi and
the letter's ink box, grown by PAD on every side, must cover only white there; the clearance to
the nearest ink is printed.  Run by ./build.sh introoverview before pdftex.
"""

import os
import re
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from PIL import Image  # noqa: E402

for d in ("/usr/share/fonts/truetype/msttcorefonts", os.path.expanduser("~/.fonts")):
    for f in ("arial.ttf", "arialbd.ttf", "Arial.TTF", "Arialbd.TTF"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            matplotlib.font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "pdf.fonttype": 42, "savefig.transparent": True})
BOLD = FontProperties(family="Arial", weight="bold")
assert "Arial" in BOLD.get_name(), "Arial is not installed"

MM = 72.0 / 25.4                 # bp per mm
LETTER_FS = 8.0                  # loop_chain.LETTER_FS: Nature panel letters, 8 pt bold lower case
INK = "#1a1a21"                  # ov_style ink
ML = 1.0 * MM                    # loop_chain.ML: the letter's left ink edge from the page edge
INSET = 0.75 * MM                # an ascender's top below its sub-figure's top edge (1 mm leaves
                                 # b 0.45 mm from method_overview's frame corner, under PAD)
PAD = 0.5 * MM                   # the gate: white all round the letter's ink box, this far out
DPI = 600
FIGS = (("a", "../challenge_overview.pdf"), ("b", "../method_overview.pdf"))


def page_size(pdf):
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, check=True).stdout
    w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out).groups()
    return float(w), float(h)


# the gap is io_compose.tex's own, read from it so the two can never disagree
tex = open("io_compose.tex").read()
gap = float(re.search(r"^\\gap=([\d.]+)mm", tex, re.M).group(1)) * MM

sizes = [page_size(p) for _, p in FIGS]
W = sizes[0][0]
assert all(abs(w - W) < 1e-3 for w, _ in sizes), "the two figures differ in width"
tops = [0.0, sizes[0][1] + gap]           # each sub-figure's top edge on the page (bp, y down)
H = sizes[0][1] + gap + sizes[1][1]

fig = plt.figure(figsize=(W / 72.0, H / 72.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
ax.patch.set_alpha(0.0)

ASC = TextPath((0, 0), "b", size=LETTER_FS, prop=BOLD).get_extents().y1   # bp above the baseline
HEAD_WORD = "MESH"               # a column head of challenge_overview with flat-bottomed capitals


def head_baseline(pdf, im, px):
    """challenge_overview's column-head baseline, bp below its top edge: the lowest ink row
    inside the word HEAD_WORD's box (pdftotext -bbox, bp, y down) on the raster im."""
    out = subprocess.run(["pdftotext", "-bbox", pdf, "-"], capture_output=True, text=True,
                         check=True).stdout
    m = re.search(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">'
                  + HEAD_WORD + "<", out)
    assert m, f"{HEAD_WORD!r} not found in {pdf}"
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    sub = im[int(y0 * px):int(np.ceil(y1 * px)) + 1, int(x0 * px):int(np.ceil(x1 * px)) + 1]
    rows = np.flatnonzero((sub < 128).any(axis=1))
    return (int(y0 * px) + rows.max() + 1) / px          # the bottom edge of the lowest ink row


problems = []
with tempfile.TemporaryDirectory() as tmp:
    for (s, pdf), (_, h), top in zip(FIGS, sizes, tops):
        ink = TextPath((0, 0), s, size=LETTER_FS, prop=BOLD).get_extents()   # bp, y up, baseline 0
        png = os.path.join(tmp, s)
        subprocess.run(["pdftocairo", "-png", "-gray", "-r", str(DPI), "-singlefile", pdf, png],
                       check=True)
        im = np.asarray(Image.open(png + ".png").convert("L")).astype(int)
        px = DPI / 72.0
        # each letter's baseline, bp below its sub-figure's top edge (rules in the docstring)
        rel = head_baseline(pdf, im, px) if s == "a" else INSET + ASC
        x, base = ML - ink.x0, top + rel
        ax.text(x, base, s, fontproperties=FontProperties(family="Arial", weight="bold",
                                                          size=LETTER_FS), color=INK,
                ha="left", va="baseline")
        box = (ML, rel - ink.y1, ML + ink.width, rel - ink.y0)   # bp in its sub-figure
        print(f"io_letters: {s} baseline {rel / MM:.2f} mm below its figure's top")
        # the gate, on the sub-figure alone

        def clear(pad):
            x0, y0 = max(int((box[0] - pad) * px), 0), max(int((box[1] - pad) * px), 0)
            x1, y1 = int(np.ceil((box[2] + pad) * px)), int(np.ceil((box[3] + pad) * px))
            return (im[y0:y1, x0:x1] >= 250).all()

        if not clear(PAD):
            problems.append(f"letter {s!r} lies within {PAD / MM:.2f} mm of ink in {pdf}")
        room = next((p for p in np.arange(0.0, 10.0 * MM, 0.05 * MM) if not clear(p)), 10.0 * MM)
        print(f"io_letters: {s} ink box {box[2] - box[0]:.2f} x {box[3] - box[1]:.2f} bp at "
              f"({box[0] / MM:.2f}, {box[1] / MM:.2f}) mm, nearest ink {room / MM - 0.05:.2f} mm away")

for p_ in problems:
    print("CHECK:", p_)
assert not problems, "panel letters touch the figures"
fig.savefig("io_letters.pdf", transparent=True)
print(f"io_letters: wrote io_letters.pdf ({W:.3f} x {H:.3f} bp)")
