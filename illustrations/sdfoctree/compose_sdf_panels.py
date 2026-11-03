"""Compose the two rendered SDF-cube panels (conda `mechfoundry-warp117`) into one figure:
(a) the uniform-grid SDF cube (== bunny_sdf_pv) and (b) the octree hex mesh with
nodal SDF, sharing one colorbar.  Writes bunny_sdf_octree.{png,pdf}.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

for f in ["arial.ttf", "arialbd.ttf"]:
    try:
        font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/msttcorefonts/{f}")
    except Exception:
        pass
plt.rcParams.update({"font.family": "Arial", "font.size": 11,
                     "mathtext.fontset": "custom", "mathtext.bf": "Arial:bold"})

VMIN, VMAX = -0.26, 0.72
PTS = [(-0.26, (0.106, 0.196, 0.478)), (-0.130, (0.235, 0.443, 0.718)),
       (-0.040, (0.557, 0.706, 0.851)), (0.0, (0.894, 0.894, 0.894)),
       (0.040, (0.968, 0.805, 0.576)), (0.150, (0.944, 0.600, 0.282)),
       (0.380, (0.827, 0.329, 0.152)), (0.72, (0.478, 0.129, 0.094))]
CMAP = LinearSegmentedColormap.from_list(
    "sdf", [((v - VMIN) / (VMAX - VMIN), c) for v, c in PTS])
NORM = Normalize(VMIN, VMAX)

# crop both panels to the SAME (union) non-white bbox so the cubes stay aligned
imu = Image.open("panel_uniform.png").convert("RGB")
imo = Image.open("panel_octree.png").convert("RGB")
mask = ((np.asarray(imu) < 248).any(2)) | ((np.asarray(imo) < 248).any(2))
ys, xs = np.where(mask)
box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
imu, imo = imu.crop(box), imo.crop(box)

# octree stats written by prep_octree_sdf: depth-6 (N,M,ms,gpu) then the same
# one level denser (depth-7: N2,M2,ms2)
with open("octree_stats.txt") as f:
    n_nodes = int(f.readline()); n_hex = int(f.readline())
    gen_ms = float(f.readline()); gpu = f.readline().strip()
    n2 = int(f.readline()); m2 = int(f.readline()); ms2 = float(f.readline())
STATS = (f"{n_nodes:,} nodes  ·  {n_hex:,} hexes\n"
         f"generated in {gen_ms:.0f} ms on an {gpu}\n"
         f"(one level denser: {n2:,} nodes · {m2:,} hexes · {ms2:.0f} ms)")

fig = plt.figure(figsize=(8.6, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.01,
                      left=0.006, right=0.93, top=0.72, bottom=0.02)
for i, (im, tag, title) in enumerate(
        [(imu, "a", "Underlying signed distance"),
         (imo, "b", "Hierarchically extracted Octree mesh")]):
    ax = fig.add_subplot(gs[0, i])
    # 'lanczos' preserves the thin cell frames when the panel is downsampled;
    # the default 'antialiased' fades 1-2 px lines away in parts
    ax.imshow(im, interpolation="lanczos")
    ax.axis("off")
    ax.text(0.5, 1.15, f"$\\bf{{({tag})}}$  {title}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5)
    if i == 1:  # stats: three lines, directly below the subtitle
        ax.text(0.5, 1.13, STATS, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.3, color="#333333", linespacing=1.4)

cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax,
                  ticks=[-0.2, 0.0, 0.2, 0.4, 0.6])
cb.set_label("Signed distance  $s$", fontsize=11.5)
cb.ax.tick_params(labelsize=10, length=3)
cb.outline.set_linewidth(0.6)

# the panels are raster (ParaView renders) embedded in the vector PDF, so the
# PDF needs the SAME dpi or matplotlib rasterizes them at the default (~100 dpi)
fig.savefig("bunny_sdf_octree.png", dpi=340, bbox_inches="tight",
            facecolor="white", pad_inches=0.04)
fig.savefig("bunny_sdf_octree.pdf", dpi=340, bbox_inches="tight",
            facecolor="white", pad_inches=0.04)
print("wrote bunny_sdf_octree.png / .pdf")
