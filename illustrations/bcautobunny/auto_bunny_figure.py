"""Compose the separate automatic bunny BC workflow figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

stats = {}
for line in open("auto_stats.txt"):
    key, value = line.split()[:2]
    stats[key] = value

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "savefig.transparent": False})

titles = [
    "Seed: top 5% of height",
    "Surface shape diameter",
    f"Surface segments ($k={stats['num_clusters']}$)",
    "Close-up of the selected facets",
]
fig, axes = plt.subplots(1, 4, figsize=(8.15, 2.35))
for i, (ax, title) in enumerate(zip(axes, titles)):
    ax.imshow(Image.open(f"auto_panel_{chr(97+i)}.png"), interpolation="lanczos")
    ax.set_title(f"$\\bf{{({chr(97+i)})}}$  {title}", fontsize=8.0, pad=5)
    ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.02, wspace=0.035)
fig.savefig("automatic_bc_bunny.pdf", dpi=600, facecolor="white")
fig.savefig("automatic_bc_bunny.png", dpi=350, facecolor="white")
print("wrote automatic_bc_bunny.{pdf,png}")
