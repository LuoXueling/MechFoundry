"""Generate octree_nitsche.svg — an illustration of the diffused-Nitsche shape
optimization method on a non-conforming, 2:1-balanced adaptive octree.

Everything geometric is COMPUTED, not hand-drawn:
  * the optimized cantilever is a signed-distance field (union of capsules = a
    Michell-like truss, which is what compliance-optimal cantilevers look like);
  * the quadtree is built by recursive subdivision that refines toward the
    interface (|SDF| small), then rippled to a true 2:1 balance;
  * hanging nodes are DETECTED (a leaf-edge midpoint that is another leaf's
    corner) — the by-product of 2:1 balance that Nitsche/condensation handles.

Author the SVG here; build.sh renders the PDF/PNG. Font: Arial (project rule).
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# 1. Optimized-cantilever SDF: union of capsules (rounded truss members)
# ---------------------------------------------------------------------------
# Physical domain [0, DW] x [0, DH]; y is up.  Left wall at x≈0.3 is clamped,
# tip load at T on the right.  Nodes/members chosen to read as a compliance-
# optimal cantilever truss (fanned members, tapering toward the tip).
DW, DH = 4.4, 2.0
R = 0.14  # member half-thickness (capsule radius)

# Fewer, cleaner members -> three large triangular voids (no slivers), with
# open exterior margins top/bottom so the octree has room to grade coarse.
N = {
    "A": (0.40, 1.62),  # top support (on the wall)
    "B": (0.40, 0.38),  # bottom support (on the wall)
    "C": (2.05, 1.34),
    "D": (2.05, 0.66),
    "T": (3.92, 1.00),  # tip (load application)
}
MEMBERS = [
    ("A", "C"), ("C", "T"),   # top chord
    ("B", "D"), ("D", "T"),   # bottom chord
    ("A", "B"),               # wall post
    ("C", "D"),               # mid post
    ("B", "C"),               # shear diagonal
]
SEGS = [(np.array(N[a]), np.array(N[b])) for a, b in MEMBERS]


def _seg_dist(px, py, a, b):
    """Distance from points (px,py) to segment a-b (vectorised)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy)


def sdf(px, py):
    """Signed distance to the truss surface (negative inside the solid)."""
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    d = np.full(px.shape, 1e9)
    for a, b in SEGS:
        d = np.minimum(d, _seg_dist(px, py, a, b) - R)
    return d


# ---------------------------------------------------------------------------
# 2. Adaptive quadtree, refined toward the interface, then 2:1 balanced
# ---------------------------------------------------------------------------
H0 = 0.40                      # root leaf size  -> 11 x 5 root grid
MAXDEPTH = 3                   # finest = 0.40 / 2^3 = 0.05; levels 0.40/0.20/0.10/0.05
NROOT_I, NROOT_J = round(DW / H0), round(DH / H0)   # 11, 4


def hL(level):
    return H0 / (2 ** level)


def cell_center(level, i, j):
    h = hL(level)
    return (i + 0.5) * h, (j + 0.5) * h


def should_refine(level, i, j):
    """Refine only cells the interface passes THROUGH (a narrow band).

    A tight band keeps the deep interior / far exterior coarse; the 2:1
    balance pass then inserts the intermediate-level transition rings, so the
    open regions visibly grade 0.40 -> 0.20 -> 0.10 -> 0.05 toward Gamma.
    """
    if level >= MAXDEPTH:
        return False
    h = hL(level)
    cx, cy = cell_center(level, i, j)
    halfdiag = h * math.sqrt(2) / 2
    return abs(float(sdf(cx, cy))) < halfdiag + 0.20 * h


def build_leaves():
    leaves = set()
    stack = [(0, i, j) for i in range(NROOT_I) for j in range(NROOT_J)]
    while stack:
        level, i, j = stack.pop()
        if should_refine(level, i, j):
            for di in (0, 1):
                for dj in (0, 1):
                    stack.append((level + 1, 2 * i + di, 2 * j + dj))
        else:
            leaves.add((level, i, j))
    return leaves


def covering_leaf(leaves, level, i, j):
    """The leaf that actually covers cell (level,i,j) — itself or an ancestor.

    Returns (Lp, ip, jp) or None if the region is finer (subdivided) / outside.
    """
    L, ci, cj = level, i, j
    while L >= 0:
        if (L, ci, cj) in leaves:
            return (L, ci, cj)
        L -= 1
        ci //= 2
        cj //= 2
    return None


def balance_21(leaves):
    """Ripple-subdivide until face-adjacent leaves differ by <= 1 level."""
    changed = True
    while changed:
        changed = False
        for (level, i, j) in list(leaves):
            if (level, i, j) not in leaves:
                continue
            imax = NROOT_I * (2 ** level)
            jmax = NROOT_J * (2 ** level)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if not (0 <= ni < imax and 0 <= nj < jmax):
                    continue
                cov = covering_leaf(leaves, level, ni, nj)
                if cov is None:
                    continue
                Lp, ip, jp = cov
                if level - Lp >= 2:  # coarse neighbour too coarse -> split it
                    leaves.discard((Lp, ip, jp))
                    for di in (0, 1):
                        for dj in (0, 1):
                            leaves.add((Lp + 1, 2 * ip + di, 2 * jp + dj))
                    changed = True
    return leaves


LEAVES = balance_21(build_leaves())


# ---------------------------------------------------------------------------
# 3. Node classification: hanging vs regular
# ---------------------------------------------------------------------------
def _key(x, y):
    return (round(x, 6), round(y, 6))


def classify_nodes(leaves):
    """Return (regular nodes, {hanging node: coarse-cell level}).

    A hanging node is a leaf-edge midpoint that coincides with another leaf's
    corner.  The leaf that owns that edge is the COARSER side of the 2:1
    junction; we record its level so the renderer can show only the coarser
    (sparser, visibly graded) transitions and drop the dense finest ring.
    """
    corners = set()
    edge_owner = {}   # midpoint key -> coarsest leaf level owning that edge
    for (level, i, j) in leaves:
        h = hL(level)
        x0, y0 = i * h, j * h
        cs = [(x0, y0), (x0 + h, y0), (x0 + h, y0 + h), (x0, y0 + h)]
        for c in cs:
            corners.add(_key(*c))
        for k in range(4):
            ax, ay = cs[k]
            bx, by = cs[(k + 1) % 4]
            mid = _key((ax + bx) / 2, (ay + by) / 2)
            edge_owner[mid] = min(edge_owner.get(mid, level), level)
    hanging = {mid: lvl for mid, lvl in edge_owner.items() if mid in corners}
    regular = corners - set(hanging)
    return regular, hanging


REGULAR, HANGING = classify_nodes(LEAVES)
# Show only the coarser transitions (level 0<->1 and 1<->2): these sit in the
# visibly-graded open regions.  The finest 2<->3 ring hugging Gamma is dense
# and would swamp the figure, so it is omitted from the node overlay.
HANG_DRAW = {p: l for p, l in HANGING.items() if l <= 1}


# ---------------------------------------------------------------------------
# 4. SVG emission
# ---------------------------------------------------------------------------
S = 168.0                       # px per physical unit
ML, MT = 96, 112                # left / top margin (px; MT leaves an annotation row)
CAPTION_H = 176                 # room below drawing for legend + inset
CANVAS_W = int(ML + DW * S + 96)
CANVAS_H = int(MT + DH * S + CAPTION_H)
EPS = 0.05                      # diffuse-interface half-width (physical)


def X(x):
    return ML + x * S


def Y(y):
    return MT + (DH - y) * S     # flip: physical y up -> svg y down


COL = dict(
    mesh="#c3c9d1", solid="#4C72B0", edge="#2f4a73", band="#8fb0da",
    hang="#DD8452", hang_edge="#7a3f18", reg="#9aa3ad",
    load="#C44E52", clamp="#8C8C8C", ink="#2b2b2b", faint="#6b7280",
)
FONT = 'font-family="Arial, Helvetica, sans-serif"'

out = []
w = out.append
w(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" '
  f'height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">')

# ---- defs: clamp hatch + load arrowhead ----
w('<defs>')
w('<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
  'patternTransform="rotate(45)">'
  f'<line x1="0" y1="0" x2="0" y2="8" stroke="{COL["clamp"]}" '
  'stroke-width="1.6"/></pattern>')
# Arrowhead points along the marker +x axis, which orient="auto" aligns with
# the path tangent (so a downward line yields a downward-pointing head).
w('<marker id="load" markerWidth="10" markerHeight="9" refX="8.5" refY="4.5" '
  'orient="auto">'
  f'<path d="M0.5,0.5 L8.5,4.5 L0.5,8.5 Z" fill="{COL["load"]}"/></marker>')
w('</defs>')

w(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="#ffffff"/>')


def label(x, y, text, size=15, anchor="start", color=COL["ink"],
          italic=False, weight="normal"):
    st = ' font-style="italic"' if italic else ''
    w(f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" '
      f'text-anchor="{anchor}" fill="{color}"{st} '
      f'font-weight="{weight}">{text}</text>')


def leader(x1, y1, x2, y2):
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
      f'stroke="{COL["faint"]}" stroke-width="1"/>')
    w(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="2" fill="{COL["faint"]}"/>')


# ---- title ----
label(ML, 34, "Non-conforming 2:1-balanced quadtree with the diffuse Nitsche "
      "interface", 18, "start", COL["ink"], weight="bold")
label(ML, 54, "(topology-optimized cantilever; boundary-refined — octree in 3-D)",
      14, "start", COL["faint"])

# ---- domain box ----
w(f'<rect id="domain" x="{X(0):.1f}" y="{Y(DH):.1f}" width="{DW*S:.1f}" '
  f'height="{DH*S:.1f}" fill="none" stroke="{COL["faint"]}" '
  'stroke-width="1.4" stroke-dasharray="6 4"/>')

# ---- octree mesh (leaf rects) ----
w('<g id="mesh" fill="none" '
  f'stroke="{COL["mesh"]}" stroke-width="1.0">')
for (level, i, j) in sorted(LEAVES):
    h = hL(level)
    x0, y0 = i * h, j * h
    w(f'<rect x="{X(x0):.2f}" y="{Y(y0+h):.2f}" '
      f'width="{h*S:.2f}" height="{h*S:.2f}"/>')
w('</g>')


def capsules(width_units, color, gid, opacity=None):
    op = f' opacity="{opacity}"' if opacity is not None else ""
    w(f'<g id="{gid}"{op} stroke="{color}" stroke-width="{width_units*S:.2f}" '
      f'stroke-linecap="round" fill="none">')
    for a, b in SEGS:
        w(f'<line x1="{X(a[0]):.2f}" y1="{Y(a[1]):.2f}" '
          f'x2="{X(b[0]):.2f}" y2="{Y(b[1]):.2f}"/>')
    w('</g>')


# ---- diffuse Nitsche band (group opacity avoids overlap darkening) ----
capsules(2 * (R + EPS), COL["band"], "nitsche_band", opacity=0.55)
# ---- solid: dark interface outline behind, blue material on top ----
capsules(2 * R + 0.03, COL["edge"], "gamma_outline")    # interface Gamma
capsules(2 * R, COL["solid"], "omega_solid")            # material Omega

# ---- hanging nodes at the coarser 2:1 transitions (regular nodes omitted
#      from the main view to keep the mesh legible; both appear in the inset) ----
w('<g id="hanging_nodes">')
for (x, y), lvl in sorted(HANG_DRAW.items()):
    if 0 <= x <= DW and 0 <= y <= DH:
        r = 4.0 if lvl == 0 else 3.2      # coarser junction -> larger marker
        w(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="{r}" '
          f'fill="{COL["hang"]}" stroke="{COL["hang_edge"]}" '
          'stroke-width="1.1"/>')
w('</g>')

# ---- clamp (Dirichlet): hatched wall just LEFT of the beam's left face ----
wall_x = 0.40
wall_top, wall_bot = 1.72, 0.28
w('<g id="clamp">')
w(f'<rect x="{X(wall_x)-30:.1f}" y="{Y(wall_top):.1f}" width="22" '
  f'height="{(wall_top-wall_bot)*S:.1f}" fill="url(#hatch)" '
  f'stroke="{COL["clamp"]}" stroke-width="1.4"/>')
w(f'<line x1="{X(wall_x)-8:.1f}" y1="{Y(wall_top):.1f}" '
  f'x2="{X(wall_x)-8:.1f}" y2="{Y(wall_bot):.1f}" '
  f'stroke="{COL["clamp"]}" stroke-width="2.2"/>')
w('</g>')

# ---- tip load (Neumann): arrow starts at the lower tip surface, points down ----
tx, ty = N["T"]
w(f'<line id="load_arrow" x1="{X(tx):.1f}" y1="{Y(ty-R):.1f}" '
  f'x2="{X(tx):.1f}" y2="{Y(ty-R)+66:.1f}" stroke="{COL["load"]}" '
  'stroke-width="3.2" marker-end="url(#load)"/>')
w(f'<text x="{X(tx)+16:.1f}" y="{Y(ty-R)+34:.1f}" {FONT} font-size="20" '
  f'font-style="italic" fill="{COL["load"]}">F</text>')

# ---- annotation labels ----
# clamp label (two lines, left of the wall)
label(X(wall_x) - 40, Y(1.06), "clamped", 15, "end", COL["clamp"])
# the displacement is a vector, bold upright (the paper's \mathbf{u} = \mathbf{0}):
# Arial Bold tspans, not the Unicode maths-bold letters Arial does not carry
label(X(wall_x) - 40, Y(1.06) + 19,
      '(<tspan font-weight="bold">u</tspan> = <tspan font-weight="bold">0</tspan>)',
      15, "end", COL["clamp"])
# Omega — placed on the solid tip block
label(X(3.40), Y(1.00) + 7, "Ω", 21, "middle", "#ffffff", weight="bold")
# top annotation row (its own strip above the domain) -----------------------
ann_y = MT - 20


def surface_pt(a, b, t, off):
    """Point offset by ``off`` along the upward normal of segment a-b at
    parameter t — off=R lands on Gamma, off in (R, R+EPS) lands in the band."""
    ax, ay = N[a]
    bx, by = N[b]
    px, py = ax + t * (bx - ax), ay + t * (by - ay)
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    if ny < 0:
        nx, ny = -nx, -ny            # make the normal point up (outward)
    return px + off * nx, py + off * ny


# interface Gamma — left, leader lands ON the dark boundary (off = R)
gx, gy = surface_pt("A", "C", 0.62, R)
leader(X(1.35), ann_y + 6, X(gx), Y(gy))
label(X(1.30), ann_y, "interface  Γ : <tspan font-style=\"italic\">s</tspan> = 0", 15, "start", COL["ink"])
# Nitsche band — right, leader lands IN the light halo (off between R and R+EPS)
bx_, by_ = surface_pt("C", "T", 0.55, R + 0.55 * EPS)
leader(X(3.55), ann_y + 6, X(bx_), Y(by_))
label(X(4.32), ann_y, "diffuse Nitsche band  ε", 15, "end", COL["faint"])
# adaptive-grading call-out — point at a clean coarse T-junction in the open
# bottom exterior (short leader; the graded cells there make the 2:1 obvious)
if HANG_DRAW:
    hx0, hy0 = min(HANG_DRAW, key=lambda p: (p[0] - 2.25) ** 2
                   + (p[1] - 0.16) ** 2)
    lxp, lyp = X(hx0) - 14, Y(hy0) + 40
    leader(lxp + 6, lyp - 26, X(hx0), Y(hy0))
    label(lxp, lyp, "hanging node", 13, "end", COL["hang_edge"])
    label(lxp, lyp + 16, "(2:1 grading)", 13, "end", COL["hang_edge"])

# ---------------------------------------------------------------------------
# 5. Legend (2 columns) + 2:1 hanging-node inset (below the drawing)
# ---------------------------------------------------------------------------
base_y = MT + DH * S + 40
w(f'<line x1="{ML}" y1="{base_y-22:.0f}" x2="{CANVAS_W-40:.0f}" '
  f'y2="{base_y-22:.0f}" stroke="#e5e7eb" stroke-width="1"/>')
row = 27


def swatch_rect(x, y, color, text):
    w(f'<rect x="{x}" y="{y-11}" width="20" height="14" rx="2" '
      f'fill="{color}"/>')
    label(x + 28, y, text, 14)


def swatch_dot(x, y, color, edge, text):
    w(f'<circle cx="{x+10}" cy="{y-4}" r="5.5" fill="{color}" '
      f'stroke="{edge}" stroke-width="1.2"/>')
    label(x + 28, y, text, 14)


colA, colB = ML, ML + 250
swatch_rect(colA, base_y, COL["solid"], "optimized material Ω")
swatch_rect(colA, base_y + row, COL["band"], "phase-field band (ε)")
swatch_rect(colA, base_y + 2 * row, "url(#hatch)", "Dirichlet (Nitsche)")
# Horizontal arrow glyph (sits cleanly in the legend row); a slim stroke keeps
# the stroke-scaled arrowhead from swallowing the shaft.
w(f'<line x1="{colB}" y1="{base_y-5}" x2="{colB+30}" y2="{base_y-5}" '
  f'stroke="{COL["load"]}" stroke-width="2.3" marker-end="url(#load)"/>')
label(colB + 40, base_y, "Neumann load F", 14)
swatch_dot(colB, base_y + row, COL["hang"], COL["hang_edge"], "hanging node")
swatch_dot(colB, base_y + 2 * row, COL["reg"], COL["reg"], "regular node")

# ---- inset: schematic 2:1 T-junction (a coarse cell meets two fine cells) ----
u = 42.0
iw = 3 * u + 40
ix = CANVAS_W - 40 - iw
iy = base_y - 10
w('<g id="inset">')
w(f'<rect x="{ix:.0f}" y="{iy-30:.0f}" width="{iw:.0f}" '
  f'height="{2*u+66:.0f}" rx="7" fill="#f7f8fa" stroke="#d7dbe0" '
  'stroke-width="1"/>')
label(ix + iw / 2, iy - 12, "2:1 balance → hanging node", 13, "middle",
      COL["ink"])
mx0 = ix + (iw - 3 * u) / 2
my0 = iy + 4
# coarse cell (left), size 2u
w(f'<rect x="{mx0:.1f}" y="{my0:.1f}" width="{2*u}" height="{2*u}" '
  f'fill="none" stroke="{COL["ink"]}" stroke-width="1.6"/>')
# two fine cells (right), size u each, stacked
for k in range(2):
    w(f'<rect x="{mx0+2*u:.1f}" y="{my0+k*u:.1f}" width="{u}" height="{u}" '
      f'fill="none" stroke="{COL["ink"]}" stroke-width="1.6"/>')
# regular (real) nodes at the cell corners
for (cx, cy) in [(mx0, my0), (mx0, my0 + 2 * u), (mx0 + 2 * u, my0),
                 (mx0 + 2 * u, my0 + 2 * u), (mx0 + 3 * u, my0),
                 (mx0 + 3 * u, my0 + u), (mx0 + 3 * u, my0 + 2 * u)]:
    w(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{COL["reg"]}"/>')
# hanging node = midpoint of the coarse cell's right edge
hxn, hyn = mx0 + 2 * u, my0 + u
w(f'<circle cx="{hxn:.1f}" cy="{hyn:.1f}" r="5.5" fill="{COL["hang"]}" '
  f'stroke="{COL["hang_edge"]}" stroke-width="1.3"/>')
label(ix + iw / 2, my0 + 2 * u + 22, "constrained DOF (interpolated)", 12,
      "middle", COL["hang_edge"])
w('</g>')

w('</svg>')

with open("octree_nitsche.svg", "w") as f:
    f.write("\n".join(out))

print(f"leaves={len(LEAVES)}  regular_nodes={len(REGULAR)}  "
      f"hanging_nodes={len(HANGING)} (drawn={len(HANG_DRAW)})  "
      f"canvas={CANVAS_W}x{CANVAS_H}")
