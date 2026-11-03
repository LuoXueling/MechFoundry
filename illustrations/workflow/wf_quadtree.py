"""Faithful 2-D port of ``hierarchical_extraction`` (octree -> quadtree).

Mirrors the real pipeline's identify-subdivide-query loop with 2:1 balancing:
  * a cell is ACTIVE (flagged to refine) if its corner SDFs have mixed sign
    (the boundary passes through) or any corner lies in the narrow band
    |sdf| < band  -- exactly ``_find_candidates_band``.
  * the 2:1 rule (1-irregular mesh) is enforced by a ripple cascade: a leaf
    whose face-neighbour is >= 2 levels finer is itself split -- the same
    invariant ``_expand_edge_region_fast`` maintains.
  * corner SDFs are read from the fine uniform background grid (the object the
    pipeline keeps), i.e. "queried where recorded, interpolated elsewhere".

A leaf is ``(level, i, j)``: the cell at that level with index (i, j) in
[0, 2**level).  ``level_grid[a, b]`` holds the level of the leaf owning fine
cell (a, b) -- the 2-D analogue of ``existing_anchors``.
"""

import numpy as np


def _bg_depth(bg):
    """Background-lattice depth Lbg such that bg has 2**Lbg + 1 nodes/side."""
    return int(round(np.log2(bg.shape[0] - 1)))


def corner_sdf(bg, level, i, j, Lf=None):
    """The 4 corner SDF values of leaf (level, i, j), read off the fine uniform
    background grid bg (shape (2**Lbg + 1,)*2).  Indexing uses the background
    depth Lbg -- NOT the quadtree's final depth -- so a leaf's corners map to
    the correct nodes of the full domain regardless of how deep the tree goes."""
    Lbg = _bg_depth(bg)
    s = 2 ** (Lbg - level)
    I, J = i * s, j * s
    return np.array([bg[I, J], bg[I + s, J], bg[I, J + s], bg[I + s, J + s]])


def is_active(bg, level, i, j, Lf, band):
    c = corner_sdf(bg, level, i, j)
    mixed = (c.min() < 0) and (c.max() > 0)          # boundary crosses cell
    inband = np.abs(c).min() < band                  # near-surface band
    return bool(mixed or inband)


def _split(leaf, leaves, level_grid, Lf):
    d, i, j = leaf
    leaves.discard(leaf)
    s = 2 ** (Lf - d)
    hs = s // 2
    level_grid[i * s:(i + 1) * s, j * s:(j + 1) * s] = d + 1
    for di in (0, 1):
        for dj in (0, 1):
            leaves.add((d + 1, 2 * i + di, 2 * j + dj))


def _neighbor_max_level(level_grid, leaf, Lf):
    """Max leaf level across the 4 face-neighbours of ``leaf`` (or -1 if all
    faces hit the domain boundary)."""
    d, i, j = leaf
    s = 2 ** (Lf - d)
    n = 2 ** Lf
    best = -1
    a0, a1 = i * s, (i + 1) * s
    b0, b1 = j * s, (j + 1) * s
    if a1 < n:
        best = max(best, int(level_grid[a1, b0:b1].max()))
    if a0 > 0:
        best = max(best, int(level_grid[a0 - 1, b0:b1].max()))
    if b1 < n:
        best = max(best, int(level_grid[a0:a1, b1].max()))
    if b0 > 0:
        best = max(best, int(level_grid[a0:a1, b0 - 1].max()))
    return best


def balance(leaves, level_grid, Lf):
    """2:1 ripple cascade: split any leaf with a >=2-levels-finer face-neighbour
    until the whole mesh is 1-irregular.  Returns the set of leaves that were
    split (for illustrating the ripple)."""
    rippled = set()
    changed = True
    while changed:
        changed = False
        for leaf in list(leaves):
            d = leaf[0]
            if _neighbor_max_level(level_grid, leaf, Lf) >= d + 2:
                _split(leaf, leaves, level_grid, Lf)
                rippled.add(leaf)
                changed = True
    return rippled


def queried_nodes(leaves, Lbg):
    """Background-grid indices (on the 2**Lbg lattice) of nodes that have been
    QUERIED, i.e. the corner nodes of every leaf.  Nodes interior to a leaf are
    never queried -- they are interpolated (see ``background_field``)."""
    idx = set()
    for (d, i, j) in leaves:
        s = 2 ** (Lbg - d)
        for di in (0, 1):
            for dj in (0, 1):
                idx.add(((i + di) * s, (j + dj) * s))
    return idx


def background_field(bg, leaves, out_res=None):
    """Reconstruct the maintained background SDF grid at the current stage.

    Each leaf's block is filled by BILINEAR interpolation of its 4 queried
    corner values (read from the true field ``bg`` at the corner nodes) -- the
    2-D analogue of the pipeline's "queried values recorded, the rest
    tri-linearly interpolated".  Coarse cells (far from the surface) are thus a
    blurry low-res interpolation; the field only sharpens where the mesh has
    been refined and new nodes queried.
    """
    Lbg = _bg_depth(bg)
    if out_res is None:
        out_res = 2 ** Lbg + 1
    r = (out_res - 1) // (2 ** Lbg)
    F = np.empty((out_res, out_res), dtype=np.float32)
    for (d, i, j) in leaves:
        s = 2 ** (Lbg - d)
        I, J = i * s, j * s
        c00, c10, c01, c11 = bg[I, J], bg[I + s, J], bg[I, J + s], bg[I + s, J + s]
        m = s * r + 1
        u = np.linspace(0, 1, m)[:, None]
        v = np.linspace(0, 1, m)[None, :]
        block = c00 * (1 - u) * (1 - v) + c10 * u * (1 - v) \
            + c01 * (1 - u) * v + c11 * u * v
        F[I * r:I * r + m, J * r:J * r + m] = block
    return F


def build(bg, Li, Lf, band, record=False):
    """Run the identify-subdivide-query loop from depth Li to Lf.

    With ``record=True`` also return:
      stages: list of (depth, leaves_after_that_depth) for depth Li..Lf,
              i.e. the progressive refinement.
      steps : the one-cycle mechanism at Li -> Li+1 -- init grid, identified
              active cells, the 2:1 ripple set, and the subdivided mesh.
    """
    n = 2 ** Lf
    level_grid = np.full((n, n), Li, dtype=np.int32)
    leaves = {(Li, i, j) for i in range(2 ** Li) for j in range(2 ** Li)}
    stages = [(Li, set(leaves))]
    steps = {"active_by_depth": {}, "ripple_by_depth": {}, "prebalance_by_depth": {}}

    for d in range(Li, Lf):
        active = [lf for lf in leaves
                  if lf[0] == d and is_active(bg, lf[0], lf[1], lf[2], Lf, band)]
        steps["active_by_depth"][d] = set(active)
        for lf in active:
            _split(lf, leaves, level_grid, Lf)
        # snapshot the just-subdivided mesh BEFORE 2:1 balancing (it may hold
        # 2-level jumps) -- used to bridge subdivide -> balance in the figure
        steps["prebalance_by_depth"][d] = set(leaves)
        rippled = balance(leaves, level_grid, Lf)
        steps["ripple_by_depth"][d] = set(rippled)     # pre-split coarse leaves
        stages.append((d + 1, set(leaves)))

    if record:
        # transition with the most 2:1 ripple (empty at the first refinement)
        rb = steps["ripple_by_depth"]
        steps["ripple_depth"] = max(rb, key=lambda k: len(rb[k])) if rb else Li
        return leaves, level_grid, stages, steps
    return leaves, level_grid


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    dat = np.load("bunny_slice.npz")
    sdf, H, N = dat["sdf"], float(dat["H"]), int(dat["N"])
    Li, Lf = 3, 6
    band = 2.5 * (2 * H) / (2 ** Lf)               # ~2.5 finest cells wide

    leaves, lg, stages, steps = build(sdf, Li, Lf, band, record=True)
    print(f"{len(leaves)} leaves; levels {sorted({l[0] for l in leaves})}")
    print("stage leaf counts:", [(d, len(s)) for d, s in stages])

    show = [stages[0], stages[1], stages[len(stages) // 2], stages[-1]]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4))
    for ax, (d, lv) in zip(axes, show):
        F = background_field(sdf, lv)               # reconstructed bg field
        ax.imshow(F.T, origin="lower", extent=[-H, H, -H, H],
                  cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        rects = [Rectangle((-H + i * (2 * H / 2 ** dd), -H + j * (2 * H / 2 ** dd)),
                           2 * H / 2 ** dd, 2 * H / 2 ** dd)
                 for (dd, i, j) in lv]
        ax.add_collection(PatchCollection(rects, facecolor="none",
                                          edgecolor="0.2", linewidth=0.3))
        ax.contour(np.linspace(-H, H, F.shape[0]), np.linspace(-H, H, F.shape[1]),
                   F.T, levels=[0], colors="k", linewidths=0.9)
        ax.set_title(f"depth {d}  ({len(lv)} cells)")
        ax.set_xlim(-H, H); ax.set_ylim(-H, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig("_preview_quadtree.png", dpi=120, bbox_inches="tight")
    print("wrote _preview_quadtree.png")
