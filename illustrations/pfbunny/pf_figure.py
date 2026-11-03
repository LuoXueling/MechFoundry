"""Compose the two single-panel PartField bunny figures.

Both panels come off one camera, so they are cropped to one COMMON ink box:
crop each to its own extent and the bunny changes size between the two figures,
which makes them impossible to read side by side or against
``automatic_bc_bunny``'s panels.  The box is then padded out to a square so the
two deliverables have identical proportions.

No title is drawn.  These are single-panel figures and a LaTeX caption carries
the description; a title band would only cost height (§5b).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DPI = 350
MARGIN = 26          # px of white kept around the common ink box
PANELS = {"partfield_feature": "pf_panel_feature.png",
          "partfield_segmentation": "pf_panel_segments.png"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "savefig.transparent": False})

images = {name: Image.open(path).convert("RGB") for name, path in PANELS.items()}
boxes = []
for image in images.values():
    ink = np.where((np.asarray(image) < 250).any(axis=2))
    boxes.append((ink[1].min(), ink[0].min(), ink[1].max() + 1, ink[0].max() + 1))
left = min(b[0] for b in boxes) - MARGIN
top = min(b[1] for b in boxes) - MARGIN
right = max(b[2] for b in boxes) + MARGIN
bottom = max(b[3] for b in boxes) + MARGIN

# Square it about the centre, then clamp back inside the render.
width, height = right - left, bottom - top
side = max(width, height)
cx, cy = 0.5 * (left + right), 0.5 * (top + bottom)
full_w, full_h = next(iter(images.values())).size
left = int(round(max(0, min(cx - side / 2, full_w - side))))
top = int(round(max(0, min(cy - side / 2, full_h - side))))
side = int(min(side, full_w - left, full_h - top))
crop = (left, top, left + side, top + side)
print(f"common crop {crop} -> {side}px = {side / DPI:.2f} in at {DPI} dpi")

for name, image in images.items():
    panel = image.crop(crop)
    # figsize x dpi == the crop's pixel size, so the render is placed 1:1 and
    # nothing is resampled on the way into either deliverable.
    fig = plt.figure(figsize=(side / DPI, side / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(panel, interpolation="none")
    ax.axis("off")
    fig.savefig(f"{name}.pdf", dpi=DPI, facecolor="white")
    fig.savefig(f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.{{pdf,png}}")
