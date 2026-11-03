"""Octree-mesh-generation workflow, as FOUR separate figures (one per step),
each driven by faithful 2-D ports of the real pipeline functions on a real
Stanford-bunny profile SDF:

  octree_step1 · Hierarchical SDF query   (query - identify - subdivide -
                 subdivide again (unbalanced) - 2:1 balance - final)
  octree_step2 · Noisy SDF filtering      (erosion - components - keep body)
  octree_step3 · SDF reinitialization     (|grad| != 1 -> surface -> |grad| = 1)
  octree_step4 · Bounding-box scaling      (normalized [-1,1] -> physical extents)

Each writes octree_stepN.{png,pdf}.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
from scipy.ndimage import (binary_erosion, binary_dilation, label,
                           distance_transform_edt as edt)
from skimage import measure

import wf_quadtree as qt

# ---------------------------------------------------------------------------
# Style + palette
# ---------------------------------------------------------------------------
for f in ["arial.ttf", "arialbd.ttf"]:
    try:
        font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/msttcorefonts/{f}")
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.linewidth": 0.6, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

INK = "#1a1a1a"
SHADOW = "#000000"      # identify/active cells: neutral gray shaded overlay
SHADOW_A = 0.24
SHADOW_BAL = "#2f78e8"  # 2:1-balance cells: a DISTINCT (blue) shaded overlay.  A vivid
                        # mid blue at high alpha: the navy #1c3f7a at 0.38 mixed with
                        # the orange exterior field (its complement) to a neutral gray
SHADOW_BAL_A = 0.68
LGRAY = "#8f8f8f"       # legend swatch (active cell)
LBLUE = "#6c9be6"       # legend swatch (2:1-balance cell): SHADOW_BAL at SHADOW_BAL_A over the light field
NODE = "#111111"        # queried nodes: black points, white edge (consistent everywhere)
SPUR = "#e7298a"        # spurious object (magenta)
SURF = "#00868b"        # extracted surface (teal)
BBOX = "#2166ac"        # bounding box (blue)
MASK = "#e9e9e9"        # filtered-out (masked) region (light gray)

CMAP_SDF = LinearSegmentedColormap.from_list("sdf", [
    (0.00, "#1b3278"), (0.16, "#3b71b7"), (0.34, "#8fb6da"), (0.50, "#e7e7e7"),
    (0.60, "#f8ce97"), (0.74, "#f0983f"), (0.88, "#cf5327"), (1.00, "#7a2118"),
])
NORM_SDF = TwoSlopeNorm(vcenter=0.0, vmin=-0.30, vmax=0.55)
CMAP_G = plt.get_cmap("PRGn")
NORM_G = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=1.6)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dat = np.load("bunny_slice.npz")
SDF = dat["sdf"].astype(np.float64)
H = float(dat["H"]); N = int(dat["N"])
Lbg = qt._bg_depth(SDF)
XS = np.linspace(-H, H, N)
DX = 2 * H / (N - 1)

Li, Lf = 3, 6
BAND = 2.5 * (2 * H) / (2 ** Lf)
leaves, lg, STAGES, STEPS = qt.build(SDF, Li, Lf, BAND, record=True)
STAGE = {d: lv for d, lv in STAGES}
RD = Li + 1                                    # transition (depth 4 -> 5) to detail


def phys(idx):
    return -H + idx * (2 * H / (2 ** Lbg))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def clean(ax):
    ax.set_xlim(-H, H); ax.set_ylim(-H, H); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.55"); s.set_linewidth(0.6)


def show_field(ax, F, contour=True, cmap=CMAP_SDF, norm=NORM_SDF, alpha=1.0):
    im = ax.imshow(F.T, origin="lower", extent=[-H, H, -H, H], cmap=cmap,
                   norm=norm, interpolation="bilinear", alpha=alpha, zorder=0)
    if contour:
        xs = np.linspace(-H, H, F.shape[0])
        ax.contour(xs, xs, F.T, levels=[0], colors="k", linewidths=1.0, zorder=3)
    return im


def _rects(cells):
    return [Rectangle((-H + i * (2 * H / 2 ** d), -H + j * (2 * H / 2 ** d)),
                      2 * H / 2 ** d, 2 * H / 2 ** d) for (d, i, j) in cells]


def show_mesh(ax, lv, lw=0.3, ec="0.28"):
    ax.add_collection(PatchCollection(_rects(lv), facecolor="none", edgecolor=ec,
                                      linewidth=lw, zorder=2))


def fill_cells(ax, cells, color, alpha=1.0):
    ax.add_collection(PatchCollection(_rects(cells), facecolor=color, alpha=alpha,
                                      edgecolor="none", zorder=1))


def outline_cells(ax, cells, color, lw=1.3):
    ax.add_collection(PatchCollection(_rects(cells), facecolor="none",
                                      edgecolor=color, linewidth=lw, zorder=4))


def scatter_nodes(ax, idxset, color, size=6, zorder=5):
    if not idxset:
        return
    pts = np.array([[phys(I), phys(J)] for (I, J) in idxset])
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, edgecolors="white",
               linewidths=0.3, zorder=zorder)


def cap(ax, tag, title):
    ax.set_title(f"$\\bf{{({tag})}}$ {title}", fontsize=8.6, loc="left", pad=6,
                 color=INK)


def arrow(ax, text=""):
    ax.axis("off")
    # span the full column so the arrow connects the two panels directly
    ax.add_patch(FancyArrowPatch((0.0, 0.5), (1.0, 0.5), transform=ax.transAxes,
                 arrowstyle="-|>", mutation_scale=12, lw=1.4, color="0.4",
                 clip_on=False))
    if text:
        ax.text(0.5, 0.53, text, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.2, color="0.35", linespacing=0.95)


def _fig_legend(fig, entries):
    """entries: list of (marker, color, label); marker 's' = hollow colored
    square (a highlighted cell), 'o' = black dot with white edge (a node)."""
    fw, fh = fig.get_size_inches()
    y = 1 - 0.17 / fh
    sh = 0.12
    swh, sww = sh / fh, sh / fw
    ent_w = [sww + 0.05 / fw + 0.060 * len(lab) / fw + 0.13 / fw
             for _, _, lab in entries]
    x = 0.99 - sum(ent_w)
    for (mk, color, lab), w in zip(entries, ent_w):
        if mk == "s":
            fig.add_artist(Rectangle((x, y - swh / 2), sww, swh,
                           transform=fig.transFigure, facecolor=color,
                           edgecolor="0.45", linewidth=0.6, clip_on=False))
        else:
            fig.add_artist(Ellipse((x + sww / 2, y), sww * 0.85, swh * 0.85,
                           transform=fig.transFigure, facecolor=color,
                           edgecolor="white", linewidth=0.5, clip_on=False))
        fig.text(x + sww + 0.05 / fw, y, lab, fontsize=8.5, va="center",
                 ha="left", color=INK)
        x += w


def make_fig(widths, panel_in, legend=None):
    """One-row step figure: a row of square panels, plus a thin band above
    them only when the figure carries a legend (no figure title: the paper
    caption names the step)."""
    S = sum(widths); tin = 0.28 if legend else 0.0
    figw = panel_in * S / 0.98
    figh = (tin + panel_in) / 0.92
    fig = plt.figure(figsize=(figw, figh))
    gs = fig.add_gridspec(2, len(widths), height_ratios=[tin, panel_in],
                          width_ratios=widths, left=0.008, right=0.992,
                          top=0.99, bottom=0.015, hspace=0.08, wspace=0.0)
    if legend:
        _fig_legend(fig, legend)
    return fig, gs


def save(fig, name):
    fig.savefig(name + ".png", dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none")
    fig.savefig(name + ".pdf", bbox_inches="tight", pad_inches=0.05,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("wrote", name + ".png /", name + ".pdf")


# ===========================================================================
# STEP 1 : Hierarchical SDF query  (6 panels)
# ===========================================================================
def fig_step1():
    #        a   ar  b   ar  c   AR   d   ar  e   AR   f
    w = [1, .18, 1, .18, 1, .46, 1, .18, 1, .46, 1]
    fig, gs = make_fig(w, 1.9,
                       legend=[("s", LGRAY, "Active cell"),
                               ("s", LBLUE, "2:1-balance cell"),
                               ("o", NODE, "Queried node")])

    def nodes(ax, idx):   # consistent symbol: black points, white boundary
        scatter_nodes(ax, idx, NODE, size=6)

    ax = fig.add_subplot(gs[1, 0])
    show_field(ax, qt.background_field(SDF, STAGE[Li]))
    show_mesh(ax, STAGE[Li], lw=0.5)
    nodes(ax, qt.queried_nodes(STAGE[Li], Lbg))
    clean(ax); cap(ax, "a", "Query all nodes")

    arrow(fig.add_subplot(gs[1, 1]))

    # (b) identify: gray SHADOW over the active cells; fade the field a bit so
    # the shaded cells read clearly
    ax = fig.add_subplot(gs[1, 2])
    show_field(ax, qt.background_field(SDF, STAGE[Li]), alpha=0.42)
    fill_cells(ax, STEPS["active_by_depth"][Li], SHADOW, alpha=SHADOW_A)
    show_mesh(ax, STAGE[Li], lw=0.5)
    clean(ax); cap(ax, "b", "Identify cells")

    arrow(fig.add_subplot(gs[1, 3]))

    ax = fig.add_subplot(gs[1, 4])
    show_field(ax, qt.background_field(SDF, STAGE[Li + 1]))
    show_mesh(ax, STAGE[Li + 1], lw=0.4)
    nodes(ax, qt.queried_nodes(STAGE[Li + 1], Lbg) - qt.queried_nodes(STAGE[Li], Lbg))
    clean(ax); cap(ax, "c", "Subdivide & query")

    arrow(fig.add_subplot(gs[1, 5]), "Repeat\n(b), (c)")

    # (d) repeat identify + subdivide once more (depth 5) -- NOT yet balanced;
    #     bridges (c) and the balanced mesh (e).  Show the identified parents
    #     (outlines) + the finer mesh + the newly queried nodes.
    preb = STEPS["prebalance_by_depth"][RD]
    ax = fig.add_subplot(gs[1, 6])
    show_field(ax, qt.background_field(SDF, preb), alpha=0.42)
    fill_cells(ax, STEPS["active_by_depth"][RD], SHADOW, alpha=SHADOW_A)
    show_mesh(ax, preb, lw=0.3)
    nodes(ax, qt.queried_nodes(preb, Lbg) - qt.queried_nodes(STAGE[RD], Lbg))
    clean(ax); cap(ax, "d", "Identify & subdivide")

    arrow(fig.add_subplot(gs[1, 7]))

    # (e) 2:1 balance: a DISTINCT (blue) shadow over the balance cells
    ax = fig.add_subplot(gs[1, 8])
    show_field(ax, qt.background_field(SDF, STAGE[RD + 1]))
    fill_cells(ax, STEPS["ripple_by_depth"][RD], SHADOW_BAL, alpha=SHADOW_BAL_A)
    show_mesh(ax, STAGE[RD + 1], lw=0.3)
    clean(ax); cap(ax, "e", "2:1 balance")

    arrow(fig.add_subplot(gs[1, 9]), "Repeat\n(b), (d), (e)")

    ax = fig.add_subplot(gs[1, 10])
    show_field(ax, qt.background_field(SDF, STAGE[Lf]))
    show_mesh(ax, STAGE[Lf], lw=0.22)
    clean(ax); cap(ax, "f", "Final mesh")

    save(fig, "octree_step1")


# ===========================================================================
# STEP 2 : Noisy SDF filtering  (3 panels)
# ===========================================================================
def disk_sdf(cx, cy, r):
    return np.sqrt((XS[:, None] - cx) ** 2 + (XS[None, :] - cy) ** 2) - r


def capsule_sdf(p0, p1, r):
    p = np.stack(np.meshgrid(XS, XS, indexing="ij"), -1)
    a = np.array(p0); b = np.array(p1); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1) - r


blob_c = (0.40, 0.40)
SDF_NOISY = np.minimum(SDF, disk_sdf(*blob_c, 0.075))
SDF_NOISY = np.minimum(SDF_NOISY, capsule_sdf((0.18, 0.22), blob_c, 0.012))
interior = SDF_NOISY < 0
eroded = binary_erosion(interior, iterations=2)
lab, nlab = label(eroded)
sizes = np.array([(lab == k).sum() for k in range(1, nlab + 1)])
keep = 1 + int(np.argmax(sizes))
body_core = lab == keep
# keep a small band of SDF beyond the surface (the retained positive-SDF band
# reads as a thin reddish rim); clamp everything past it to the HIGHEST SDF that
# survives in the retained region, so the far field is that uniform color.
# (NOTE: the production filter, utils.filter_largest_interior_component, resets
# the far field to sdf_grid.max() -- the GLOBAL max -- not the retained max.)
recovered = binary_dilation(body_core, iterations=9)
RET_MAX = float(SDF_NOISY[recovered].max())
SDF_CLEAN = SDF_NOISY.copy(); SDF_CLEAN[~recovered] = RET_MAX


def fig_step2():
    fig, gs = make_fig([1, .2, 1, .2, 1], 2.05)

    ax = fig.add_subplot(gs[1, 0])
    show_field(ax, SDF_NOISY)
    ax.annotate("Spurious object", xy=(0.335, 0.43), xytext=(0.0, 0.49),
                fontsize=7.6, color=INK, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK,
                                shrinkA=1, shrinkB=1))
    clean(ax); cap(ax, "a", "Noisy $s$")

    arrow(fig.add_subplot(gs[1, 1]))

    ax = fig.add_subplot(gs[1, 2])
    comp = np.zeros(lab.shape)
    comp[interior] = 0.35
    comp[(lab > 0) & (lab != keep)] = 1.0
    comp[body_core] = 0.72
    cc = LinearSegmentedColormap.from_list("cc", [
        (0.0, "#ffffff"), (0.35, "#d3dcea"), (0.72, "#4c78c8"), (1.0, SPUR)])
    ax.imshow(np.ma.masked_where(comp == 0, comp).T, origin="lower",
              extent=[-H, H, -H, H], cmap=cc, vmin=0, vmax=1, zorder=1,
              interpolation="nearest")
    clean(ax); cap(ax, "b", "Keep body")

    arrow(fig.add_subplot(gs[1, 3]))

    # (c) filtered SDF: far field is a uniform color = the highest SDF kept in
    # the retained region (SDF_CLEAN already clamps it), so it reads continuously
    ax = fig.add_subplot(gs[1, 4])
    show_field(ax, SDF_CLEAN)
    clean(ax); cap(ax, "c", "Filtered $s$")

    save(fig, "octree_step2")


# ===========================================================================
# STEP 3 : SDF reinitialization  (3 panels)
# ===========================================================================
Xg, Yg = np.meshgrid(XS, XS, indexing="ij")
m = (1.0
     + 0.50 * np.exp(-((Xg - 0.05) ** 2 + (Yg + 0.12) ** 2) / 0.10)
     - 0.42 * np.exp(-((Xg + 0.24) ** 2 + (Yg - 0.22) ** 2) / 0.10))
PHI = SDF * m                                   # non-eikonal (NOT a truncated SDF)
gx, gy = np.gradient(PHI, DX)
GRAD0 = np.sqrt(gx ** 2 + gy ** 2)
inside = SDF < 0
SDF_REINIT = (edt(~inside) - edt(inside)) * DX
gx2, gy2 = np.gradient(SDF_REINIT, DX)
GRAD1 = np.sqrt(gx2 ** 2 + gy2 ** 2)
contours = measure.find_contours(SDF, 0.0)


def fig_step3():
    #        a   ar  b   ar  c  gap cax
    fig, gs = make_fig([1, .2, 1, .2, 1, .04, .08], 2.05)

    ax = fig.add_subplot(gs[1, 0])
    show_field(ax, GRAD0, contour=False, cmap=CMAP_G, norm=NORM_G)
    ax.contour(XS, XS, SDF.T, levels=[0], colors="k", linewidths=0.9)
    clean(ax); cap(ax, "a", r"$|\nabla s| \neq 1$")

    arrow(fig.add_subplot(gs[1, 1]))

    ax = fig.add_subplot(gs[1, 2])
    show_field(ax, SDF, contour=False)
    for c in contours:
        ax.plot(phys(c[:, 0]), phys(c[:, 1]), color=SURF, lw=2.0)
    clean(ax); cap(ax, "b", "Extract surface")

    arrow(fig.add_subplot(gs[1, 3]))

    # (c) re-distance via a BVH over the extracted surface -> |grad SDF| = 1;
    # colorbar in its OWN column so (c) stays the same size as (a) and (b)
    ax = fig.add_subplot(gs[1, 4])
    imk = show_field(ax, GRAD1, contour=False, cmap=CMAP_G, norm=NORM_G)
    ax.contour(XS, XS, SDF_REINIT.T, levels=[0], colors="k", linewidths=0.9)
    clean(ax); cap(ax, "c", "Re-distance (BVH)")
    cax = fig.add_subplot(gs[1, 6])
    cb = fig.colorbar(imk, cax=cax, ticks=[0, 1, 1.6])
    cb.set_ticklabels(["0", "1", "1.6"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(r"$|\nabla s|$  (1: Eikonal)", fontsize=7.5)
    cb.outline.set_linewidth(0.5)

    save(fig, "octree_step3")


# ===========================================================================
# STEP 4 : Bounding-box scaling  (2 panels)
# ===========================================================================
allc = np.vstack([np.stack([phys(c[:, 0]), phys(c[:, 1])], 1) for c in contours])
bx0, by0 = allc.min(0); bx1, by1 = allc.max(0)
bw, bh = bx1 - bx0, by1 - by0
cxb, cyb = (bx0 + bx1) / 2, (by0 + by1) / 2


def fig_step4():
    fig, gs = make_fig([1, 0.36, 1], 1.85)

    # (a) fill the cell edge-to-edge (square limits) so the arrow connects to it;
    # the [-1,1] label sits just inside the box to avoid stretching the panel
    ax = fig.add_subplot(gs[1, 0])
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#f4f4f0", edgecolor="0.45", lw=1.0))
    for c in contours:
        ax.plot(phys(c[:, 0]), phys(c[:, 1]), color=INK, lw=1.4)
    ax.add_patch(Rectangle((bx0, by0), bw, bh, facecolor="none", edgecolor=BBOX,
                           lw=1.6, ls=(0, (4, 3))))
    ax.text(0, -0.78, r"$[-1,1]^2$", ha="center", va="top", fontsize=8.5, color="0.35")
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cap(ax, "a", "Normalized domain")

    axm = fig.add_subplot(gs[1, 1]); axm.axis("off")
    axm.add_patch(FancyArrowPatch((0.0, 0.5), (1.0, 0.5), transform=axm.transAxes,
                  arrowstyle="-|>", mutation_scale=13, lw=1.4, color="0.35",
                  clip_on=False))
    axm.text(0.5, 0.55, r"$\lambda = L_x / w$", transform=axm.transAxes, ha="center",
             va="bottom", fontsize=7.6, color="0.3")
    axm.text(0.5, 0.45, r"$s \times \lambda$", transform=axm.transAxes,
             ha="center", va="top", fontsize=7.6, color="0.3")

    Lx = 40.0
    s_scale = Lx / bw
    Ly = bh * s_scale
    ax = fig.add_subplot(gs[1, 2])
    for c in contours:
        ax.plot((phys(c[:, 0]) - cxb) * s_scale, (phys(c[:, 1]) - cyb) * s_scale,
                color=INK, lw=1.4)
    ax.add_patch(Rectangle((-Lx / 2, -Ly / 2), Lx, Ly, facecolor="none",
                           edgecolor=BBOX, lw=1.5, ls=(0, (4, 3))))
    y0 = -Ly / 2 - 6
    ax.annotate("", xy=(Lx / 2, y0), xytext=(-Lx / 2, y0),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=INK))
    ax.text(0, y0 - 2.5, f"$L_x = {Lx:.0f}$ mm", ha="center", va="top", fontsize=9)
    x0 = Lx / 2 + 6
    ax.annotate("", xy=(x0, Ly / 2), xytext=(x0, -Ly / 2),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=INK))
    ax.text(x0 + 2.5, 0, f"$L_y = {Ly:.0f}$ mm", ha="left", va="center",
            fontsize=9, rotation=90)
    # left-align the geometry so it sits next to the arrow (only the Ly label
    # needs room on the right)
    ax.set_xlim(-Lx / 2 - 2, Lx / 2 + 17); ax.set_ylim(-Ly / 2 - 13, Ly / 2 + 5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cap(ax, "b", "Physical space")

    save(fig, "octree_step4")


if __name__ == "__main__":
    fig_step1()
    fig_step2()
    fig_step3()
    fig_step4()
