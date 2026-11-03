"""Geometry behind the challenge figure's pictograms.

Plain CPU Python (numpy, scipy, shapely, trimesh, scikit-image); writes
``ch_geom.npz`` and ``ch_stats.json``, which ``ch_figure.py`` draws.

Every pictogram is a flat 2-D drawing of ONE shape, the display bunny's side
silhouette (overview/bunny.stl, mirrored in y as every surface panel of
method_overview is), and each is computed, not sketched (illustrations/AGENTS.md
§6, "compute the geometry, don't fake it"):

  mesh      a conforming triangle mesh of the silhouette (Delaunay of boundary
            + interior samples, kept where the centroid is inside); as a
            generator hands it over, with an island floating off the back and
            two facets folded through the surface
  parts     the silhouette cut into ears / head / body / tail / feet by a
            line under the ears, two discs and a strip: what a part segmentation returns, drawn in
            method_overview's segment tints; the feet are the support, the
            ears the load, as in Figure 2
  quadtree  a quadtree, uniform to depth 3 and refined to depth 5 where a cell
            meets the outline (not 2:1 balanced: a picture, not the mesh), each
            leaf coloured by the signed distance at its centre: the 2-D
            picture of the octree + SDF that replaces the mesh
  redesign  the same body thinned (an inward offset) with one hole opened,
            the feet and ears kept as they were: what a shape step with a
            hidden phase field returns; meshed again for the rows that must
            remesh
"""

import json
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(3)
stats = {}


def ring(geom):
    return np.asarray(geom.exterior.coords)


def resample(xy, h):
    """A closed polyline resampled at spacing ~h."""
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = max(int(seg[-1] / h), 8)
    t = np.linspace(0.0, seg[-1], n, endpoint=False)
    return np.c_[np.interp(t, seg, xy[:, 0]), np.interp(t, seg, xy[:, 1])]


def mesh_polygon(poly, h_out, h_in, fine=()):
    """Conforming triangles: boundary samples + a jittered interior lattice."""
    pts = [resample(ring(poly), h_out)]
    for hole in poly.interiors:
        pts.append(resample(np.asarray(hole.coords), h_in))
    x0, y0, x1, y1 = poly.bounds
    g = np.mgrid[x0:x1:h_out * 0.95, y0:y1:h_out * 0.95].reshape(2, -1).T
    g = g + RNG.uniform(-0.18, 0.18, g.shape) * h_out
    keep = [p for p in g if poly.buffer(-0.55 * h_out).contains(Point(p))]
    for c, r in fine:                   # a graded ring of nodes round each hole
        for rr in (r + 0.6 * h_in, r + 1.4 * h_in):
            n = int(2 * np.pi * rr / h_in)
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.5 * (rr > r + h_in)
            for p in np.c_[c[0] + rr * np.cos(a), c[1] + rr * np.sin(a)]:
                if poly.buffer(-0.4 * h_in).contains(Point(p)):
                    keep.append(p)
    # drop interior nodes that crowd a graded node or the boundary
    P = np.vstack(pts)
    inner = []
    for p in keep:
        if np.min(np.linalg.norm(P - p, axis=1)) > 0.45 * h_in and \
                (not inner or np.min(np.linalg.norm(np.asarray(inner) - p, axis=1)) > 0.5 * h_in):
            inner.append(p)
    P = np.vstack([P, np.asarray(inner)])
    tri = Delaunay(P).simplices
    cen = P[tri].mean(axis=1)
    inside = np.array([poly.contains(Point(c)) for c in cen])
    tri = tri[inside]
    used = np.unique(tri)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    return P[used], remap[tri]


# ---------------------------------------------------------------------------
# The bunny's side silhouette, and a coarse "generated mesh" of it
# ---------------------------------------------------------------------------
m = trimesh.load(os.path.join(HERE, "..", "overview", "bunny.stl"))
v = m.vertices.copy()
v[:, 1] *= -1.0                       # the display bunny (ov_style.MIRROR)
p = v[:, [0, 2]]
NPX = 900
lo = p.min(0)
sc = (NPX - 20) / (p.max(0) - lo).max()
q2 = (p - lo) * sc + 10
im = Image.new("L", (NPX, NPX), 0)
dr = ImageDraw.Draw(im)
for f in m.faces:
    dr.polygon([tuple(q2[i]) for i in f], fill=255)
mask = np.asarray(im) > 127
cont = max(measure.find_contours(mask.astype(float), 0.5), key=len)    # (row, col)
xy = np.c_[cont[:, 1], cont[:, 0]]
xy = (xy - 10) / sc                   # back to model units, y = z up (rows grew with z)
sil = Polygon(xy).simplify(0.004)
sil = sil.buffer(0)
xy = ring(sil)
xy = (xy - np.array(sil.bounds[:2])) / (sil.bounds[3] - sil.bounds[1])   # height 1
sil = Polygon(xy)
stats["bunny_outline_points"] = int(len(xy))

BP_, BT_ = mesh_polygon(sil, 0.085, 0.085)
# the generated mesh's defects, the kinds a generator leaves (the author): an
# island, a small separate piece floating off the back, and self-intersecting
# facets, a fan folded through the surface; the body itself is closed
ISLAND = np.array([(1.06, 0.70), (1.17, 0.66), (1.20, 0.77), (1.09, 0.81)])
island_T = np.array([[0, 1, 2], [0, 2, 3]])
FOLD = np.array([[(0.63, 0.58), (0.80, 0.72), (0.72, 0.52)],     # folded through the back
                 [(0.70, 0.56), (0.86, 0.60), (0.76, 0.70)]])
stats["bunny_mesh"] = {"nodes": int(len(BP_)), "triangles": int(len(BT_)),
                       "island_triangles": int(len(island_T)), "folded_facets": int(len(FOLD))}


# ---------------------------------------------------------------------------
# Parts: fixed cuts standing in for a segmentation (feet = support, ears = load)
# ---------------------------------------------------------------------------
BIG = 5.0
ears_h = Polygon([(-1, 0.80 + 1 / 6.0), (BIG, 0.80 - BIG / 6.0), (BIG, BIG), (-1, BIG)])
feet_h = box(-1, -1, BIG, 0.06)
head_h = Point(0.15, 0.63).buffer(0.215, 64)          # round cuts read as a segmentation's
tail_h = Point(1.0, 0.21).buffer(0.095, 48)
ears = sil.intersection(ears_h)
feet = sil.intersection(feet_h)
head = sil.intersection(head_h).difference(ears_h)
tail = sil.intersection(tail_h).difference(feet_h)
body = sil.difference(unary_union([ears_h, feet_h, head_h, tail_h]))
PARTS = {"ears": ears, "head": head, "body": body, "tail": tail, "feet": feet}


def rings(geom):
    """Exterior rings of a (multi)polygon, as a list of arrays."""
    gs = getattr(geom, "geoms", [geom])
    return [ring(g) for g in gs if g.area > 1e-4]


part_rings = {k: rings(g) for k, g in PARTS.items()}
stats["parts_area"] = {k: round(float(g.area), 4) for k, g in PARTS.items()}
cen = BP_[BT_].mean(axis=1)
names = list(PARTS)
tri_part = np.array([next((i for i, k in enumerate(names) if PARTS[k].buffer(1e-9).contains(Point(c))),
                          names.index("body")) for c in cen])


# ---------------------------------------------------------------------------
# Quadtree on the signed distance (the 2-D octree)
# ---------------------------------------------------------------------------
edge = sil.exterior
# the octree is refined round the island too (its distance is small there), but
# the island is not solid: the inside/outside test of the closed body drops it
island_edge = Polygon(ISLAND).exterior
cells = []


def split(x, y, h, depth):
    c = box(x, y, x + h, y + h)
    if depth < 3 or (depth < 5 and (c.intersects(edge) or c.intersects(island_edge))):
        for dx in (0, h / 2):
            for dy in (0, h / 2):
                split(x + dx, y + dy, h / 2, depth + 1)
        return
    p = Point(x + h / 2, y + h / 2)
    d = edge.distance(p) * (-1.0 if sil.contains(p) else 1.0)
    cells.append((x, y, h, d))


QX0, QY0, QH = -0.06, -0.08, 1.32                   # the root square round the shape and its island
for i in range(2):                                  # depth 0 already split once
    for j in range(2):
        split(QX0 + i * QH / 2, QY0 + j * QH / 2, QH / 2, 1)
quad = np.array(cells)
stats["quadtree_leaves"] = int(len(quad))


# ---------------------------------------------------------------------------
# A redesign: body thinned, one hole opened, feet and ears kept
# ---------------------------------------------------------------------------
HOLE = (0.62, 0.36, 0.085)
thin = sil.buffer(-0.045, 32)
new = unary_union([thin, ears.buffer(0.004), feet.buffer(0.004)]).buffer(0)
new = new.difference(Point(HOLE[0], HOLE[1]).buffer(HOLE[2], 48))
if new.geom_type == "MultiPolygon":
    new = max(new.geoms, key=lambda g: g.area)
new = new.simplify(0.003)
NP_, NT_ = mesh_polygon(new, 0.085, 0.05)
ncen = NP_[NT_].mean(axis=1)
stats["redesign"] = {"area_ratio": round(float(new.area / sil.area), 3),
                     "mesh_triangles": int(len(NT_))}

np.savez(os.path.join(HERE, "ch_geom.npz"),
         sil=xy, BP=BP_, BT=BT_, island=ISLAND, island_T=island_T, fold=FOLD, tri_part=tri_part,
         part_names=np.array(names),
         **{f"part_{k}": np.array(v, dtype=object) for k, v in part_rings.items()},
         quad=quad, new_ext=ring(new),
         new_holes=np.array([np.asarray(h.coords) for h in new.interiors], dtype=object),
         NP=NP_, NT=NT_, hole=np.array(HOLE))
with open(os.path.join(HERE, "ch_stats.json"), "w") as fh:
    json.dump(stats, fh, indent=1)
print(json.dumps(stats, indent=1))
