"""Stage 1 (env mechfoundry-warp117, small GPU use): agentic runs -> renderable surfaces.

    python ar_prep.py [RUN ...] [--runs-json FILE] [--work DIR]

RUN is an ``output/agentic_batch/<run_name>/`` folder or a bare workflow session
folder; ``--runs-json`` lists runs with their metadata (``ar_dev_runs.json``).
See ``ar_runs.py`` for where each metadata field comes from.

For every run, everything is drawn on ONE surface, the session's
``geometry/physical.glb`` (the scaled body every later step works on), so the
four panels of a row are the same triangles under the same camera:

  ar_<i>_painted.vtk    the textured surface's own colours (materials/texture/
                        textured.glb; else geometry/generated.glb when the generator
                        coloured it; else the plain surface in the overview's clay)
  ar_<i>_seg.vtk        (AR_SEGMENTATION=1) boundary_conditions/segments.npz, the
                        parts the BC selectors were resolved against, as cell colours
                        in the PartField bunny's tints (segmentation_of)
  ar_<i>_bc.vtk         boundary_conditions/resolved.npz face masks as cell colours
                        on the clay: Dirichlet teal, Neumann orange (method_overview's
                        BC panel)
  ar_<i>_materials.vtk  Young's modulus at the surface, from the solve's OWN volume
                        output (vtu/solution_*.vtu ``youngs_modulus``), on the
                        overview's E ramp over one log scale shared by every row
  ar_<i>_solve.vtk      the solve's displacement at the surface, same source
  ar_<i>_fit.vtk        the undeformed and the scaled deformed points: the camera
                        reference all four panels of the row are fitted to

The displacement and E are moved from the octree to the surface by the
production ``core.remeshing.octree_hex_interpolate`` (as ov_bunny_solve.py moves
the bunny's), after checking it on the node coordinates themselves.  It is
cross-checked against the workflow's own surface export (solve/displacement_
magnitude.glb.json, the marching-cubes surface), and the ratio is recorded.

DISPLACEMENT SCALE (the figure's own rule).  s = the largest 1, 2 or 5 x 10^k with
s * |u|max <= FIG_FRACTION (8 %) of the body's largest extent, never below 1; and
for a body standing on its support (a downward-facing support) s steps down the
same ladder until the magnified shape sinks at most DIP_MAX (0.5 %) of the extent
below its base (the solve has no contact).  Chosen on ten runs: displayed peaks
3.2-7.9 % of the extent, every deformation visible, none through its floor.  The
UI's slider top (``src/ui/telemetry.py:deformation_scales``, 0.3 x extent / |u|max,
called through ar_ui_scale.py in env mechfoundry-agents) is kept in ar_stats.json for
reference: right for an interactive slider the reader drags back, too strong for
a still figure (the clog at x187,000 moved 30 % of its length).

Writes ``ar_stats.json`` (the record every number on the figure is read from).
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ar_figure (matplotlib) before trimesh: trimesh first loads the system libstdc++,
# and matplotlib's C extension then fails to import (CXXABI_1.3.15 not found)
import ar_figure  # noqa: E402
import trimesh  # noqa: E402
import ar_runs  # noqa: E402
import ar_style as A  # noqa: E402

REPO = ar_runs.REPO
UI_SCALES = {}               # session dir -> ar_ui_scale.py's answer
# The figure's displacement magnification: the displayed peak at most this fraction
# of the body's largest extent, rounded down to 1/2/5 x 10^k, never below 1.
FIG_FRACTION = float(os.environ.get("AR_FIG_FRACTION", "0.12"))
LADDER = (1.0, 2.0, 3.0, 5.0)   # the clean factors: 1, 2, 3, 5 x 10^k
DIP_MAX = float(os.environ.get("AR_DIP_MAX", "0.02"))   # a standing body may sink this much of its extent


def _ladder(lo, hi):
    k0, k1 = math.floor(math.log10(max(lo, 1e-30))) - 1, math.floor(math.log10(max(hi, 1e-30))) + 1
    return sorted(n * 10 ** k for k in range(k0, k1 + 1) for n in LADDER)


def nice_nearest(x):
    """The 1/2/3/5 x 10^k nearest to x in log ratio; 1 when x < 1 (true scale)."""
    if x < 1.0:
        return 1.0
    return max(1.0, min(_ladder(x, x), key=lambda c: abs(math.log(x / c))))


def ladder_below(x):
    """The next lower step of the 1/2/3/5 ladder (>= 1)."""
    lower = [c for c in _ladder(x / 10, x) if c < x * (1 - 1e-9)]
    return max(1.0, max(lower)) if lower else 1.0


def nice_floor(x):
    """The largest 1, 2 or 5 x 10^k that is <= x; 1 when x < 1 (true scale)."""
    if x < 1.0:
        return 1.0
    k = math.floor(math.log10(x))
    best = 1.0
    for e in (k - 1, k):
        for n_ in (1.0, 2.0, 5.0):
            if n_ * 10 ** e <= x * (1 + 1e-9):
                best = max(best, n_ * 10 ** e)
    return best
# the workflow's unit systems (concepts.md): (Pa per stress unit, m per length unit)
UNITS = {"SI (m-kg-s)": (1.0, 1.0),
         "SI-mm (mm-tonne-s)": (1.0e6, 1.0e-3),
         "US-inch (in-lbf-s)": (6894.757293168, 0.0254)}


# -- small helpers ---------------------------------------------------------------
def load_mesh(path):
    scene = trimesh.load(path, process=False)
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.dump()) if len(scene.geometry) > 1 else list(scene.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    else:
        mesh = scene
    return mesh


def mesh_colours(mesh):
    """Per-vertex RGB in 0..1 when the mesh carries real colour, else None."""
    vis = mesh.visual
    if vis.kind == "texture":
        try:
            vis = vis.to_color()
        except Exception:  # noqa: BLE001
            return None
    if vis.kind != "vertex":
        return None
    vc = np.asarray(vis.vertex_colors, dtype=float)[:, :3] / 255.0
    if len(np.unique(np.round(vc, 3), axis=0)) < 2:
        return None
    return vc


def transfer_vertex_colours(src_mesh, rgb, verts):
    """Colours onto ``verts``.

    * Identity when the vertex sets match.
    * By INDEX when they are the same mesh in another frame: the current workflow
      paints geometry/texture/textured.glb (and Hunyuan3D colours generated.glb)
      BEFORE the scale step, so the paint sits in the generator's frame and
      physical.glb is that mesh under a uniform scale + translation, vertex for
      vertex.  The fitted similarity must reproduce physical.glb to 1e-5 of its
      extent, or the index mapping is refused.
    * Else nearest vertex (same frame, other topology)."""
    sv = np.asarray(src_mesh.vertices, float)
    L = max(float(np.ptp(verts, axis=0).max()), 1e-12)
    if sv.shape == verts.shape and np.abs(sv - verts).max() < 1e-9 * L:
        return rgb, "same vertices"
    if sv.shape == verts.shape:
        a = sv - sv.mean(0)
        b = verts - verts.mean(0)
        s = float((a * b).sum() / (a * a).sum())
        resid = np.abs(s * a - b).max()
        if s > 0 and resid < 1e-5 * L:
            return rgb, f"by index: same mesh, uniform scale {s:.6g} + translation (residual {resid / L:.1e} x extent)"
    from scipy.spatial import cKDTree
    d, idx = cKDTree(sv).query(verts)
    return rgb[idx], f"nearest vertex (max distance {d.max():.3g})"


def write_polydata(path, verts, faces=None, point_rgb=None, cell_rgb=None, vectors=None,
                   scalars=None, cell_ids=None):
    """Legacy ASCII VTK polydata (ov_surfaces.write_polydata's format, plus a
    displacement vector and magnitude for the solve)."""
    v = np.asarray(verts, dtype=np.float64)
    with open(path, "w") as out:
        out.write(f"# vtk DataFile Version 3.0\n{os.path.basename(path)}\nASCII\n"
                  "DATASET POLYDATA\n")
        out.write(f"POINTS {len(v)} double\n")
        np.savetxt(out, v, fmt="%.9g")
        n_cells = 0
        if faces is not None:
            f = np.asarray(faces, dtype=np.int64)
            n_cells = len(f)
            out.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
            np.savetxt(out, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
        else:
            out.write(f"VERTICES {len(v)} {2 * len(v)}\n")
            np.savetxt(out, np.column_stack([np.ones(len(v), int), np.arange(len(v))]), fmt="%d")
        if point_rgb is not None or vectors is not None or scalars is not None:
            out.write(f"POINT_DATA {len(v)}\n")
        if point_rgb is not None:
            out.write("COLOR_SCALARS rgb 3\n")
            np.savetxt(out, np.clip(np.asarray(point_rgb, float), 0, 1), fmt="%.4f")
        if vectors is not None:
            out.write("VECTORS displacement double\n")
            np.savetxt(out, np.asarray(vectors, float), fmt="%.9g")
        if scalars is not None:
            out.write("SCALARS displacement_magnitude double 1\nLOOKUP_TABLE default\n")
            np.savetxt(out, np.asarray(scalars, float), fmt="%.9g")
        if cell_rgb is not None or cell_ids is not None:
            out.write(f"CELL_DATA {n_cells}\n")
        if cell_rgb is not None:
            out.write("COLOR_SCALARS rgb 3\n")
            np.savetxt(out, np.clip(np.asarray(cell_rgb, float), 0, 1), fmt="%.4f")
        if cell_ids is not None:
            out.write("SCALARS bcid int 1\nLOOKUP_TABLE default\n")
            np.savetxt(out, np.asarray(cell_ids, np.int64), fmt="%d")
            # the id pass's colour: (id, 255, 0) in 8 bit, drawn unlit, so a pixel
            # decodes to its region exactly (blue 0 tells it from the white page)
            ids = np.asarray(cell_ids, float)
            assert ids.max() < 255
            out.write("COLOR_SCALARS idrgb 3\n")
            np.savetxt(out, np.column_stack([ids / 255.0, np.ones_like(ids), np.zeros_like(ids)]),
                       fmt="%.6f")


def read_vtu(path):
    """The solver's VTU (inline base64, 8-byte length header, uncompressed):
    the same decoding as operator_skill/scripts/sessionlib.read_vtu."""
    import base64
    dtypes = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
              "UInt8": "u1", "UInt32": "<u4", "UInt64": "<u8"}
    text = open(path, encoding="latin-1").read()
    assert 'compressor="' not in text[:2000], f"{path}: compressed VTU"
    out = {}
    for m in re.finditer(r'<DataArray([^>]*)>\s*([^<]*)</DataArray>', text):
        attrs, payload = m.group(1), m.group(2).strip()
        name = re.search(r'Name="([^"]+)"', attrs)[1]
        kind = re.search(r'type="([^"]+)"', attrs)[1]
        comp = re.search(r'NumberOfComponents="(\d+)"', attrs)
        raw = base64.b64decode(payload)
        size = int(np.frombuffer(raw[:8], dtype="<u8")[0])
        vals = np.frombuffer(raw[8:8 + size], dtype=dtypes[kind])
        if comp and int(comp[1]) > 1:
            vals = vals.reshape(-1, int(comp[1]))
        out[name] = vals
    return out


# -- the support marker -------------------------------------------------------------
# A support on an underside cannot be seen from any camera that also shows the
# loads.  It is marked, not hidden: a thin teal slab standing off the support
# faces along their outward normal, a small gap away, so it reads as a clamp
# plate under the part rather than as part of the body.  Drawn in the
# boundary-condition panel only, and only when the support faces away.
MARKER_BELOW = 0.05     # fraction of the support's area facing the camera
MARKER_GAP = 0.004      # x the body's largest extent
MARKER_TUBE = 0.006          # support outline tube radius, x extent
MARKER_MIN_R = 0.04          # smallest support outline ring radius, x extent
# Arrow proportions (x extent; arrow review 2026-10-12).  At 0.16 / 0.010 / 0.026 /
# 0.055 a 21 mm body printed a 3.4 mm arrow with a 0.42 mm shaft under a 1.1 mm head
# that rendered as a squat cone.  Now 3.8 mm long, a 0.46 mm shaft (1.3 pt) and a
# 1.2 mm x 1.4 mm head: head 2.5x the shaft wide, 1.16x its own width long, 36 % of
# the arrow, so the shaft still shows beside a load patch.
ARROW_LEN, ARROW_SHAFT, ARROW_HEAD_R, ARROW_HEAD_L = 0.18, 0.011, 0.028, 0.065   # x extent
# Arrows of one region keep at least this gap between them AS PRINTED (projected on
# the image plane, x extent; ~2 head widths).  The per-piece placement put 9-13
# arrows on a fragmented top (the pine cone's scales, the rook's merlons) where they
# printed as one orange bundle, some of them hidden behind the others.
ARROW_GAP = 0.10
REGION_MAX_ARROWS = 6     # arrows per region at most, after the gap filter
ARROW_ID0 = 100              # arrow ids in the id pass (regions use 1..99)
ARROW_LIFTS = (0.0, 0.02, 0.04, 0.07, 0.1)   # x extent: off the face when no site shows it whole
MARKER_T = 0.025         # 2.5 %: about 0.6-0.8 mm in print (design review 2, N1)


def support_outline(mesh, mask, L):
    """The support's boundary edges, offset outward along the region's own vertex
    normals by MARKER_GAP: a thin teal line around the contact (drawn as a flat
    tube, MARKER_TUBE x extent in radius).  Lighter than the old slab, which read
    as part of the body (the user, 2026-10-08).  Returns (points, segments)."""
    F = np.asarray(mesh.faces)[mask]
    used, inv = np.unique(F, return_inverse=True)
    f = inv.reshape(-1, 3)
    vn = np.zeros((len(used), 3))
    fn = mesh.face_normals[mask] * mesh.area_faces[mask][:, None]
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-30)
    pts = np.asarray(mesh.vertices)[used] + MARKER_GAP * L * vn
    # weld by POSITION before finding the boundary: the physical mesh duplicates
    # vertices along seams, and unwelded, every interior edge of the support face
    # counted as a boundary (the "outline" of the culvert's base was its whole
    # triangulation: 5200 of 5456 points inside the face)
    key = np.round(np.asarray(mesh.vertices)[used] / (1e-6 * L)).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    fw = weld.reshape(-1)[f]
    e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]])
    eo = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, idx, cnt = np.unique(np.sort(e, axis=1), axis=0, return_index=True, return_counts=True)
    seg = eo[idx[cnt == 1]]
    # a sole a few % of the body across draws a ring too small to print (a frame's
    # foot, 2026-10-10: 0.09 mm2 against the 0.25 floor): such a ring is widened
    # about its centre to MARKER_MIN_R x extent, a marker around the contact, not on it
    ring = np.unique(seg)
    if len(ring):
        c = pts[ring].mean(axis=0)
        r = float(np.linalg.norm(pts[ring] - c, axis=1).mean())
        if 0 < r < MARKER_MIN_R * L:
            pts = c + (pts - c) * (MARKER_MIN_R * L / r)
    return pts, seg


_RAY_KERNEL = None


def occluded(mesh, points, view_dir, L):
    """Per point: is it hidden from the camera by the body?  A ray from far out
    along the view direction to the point (an orthographic stand-in for the
    perspective camera, ~25 deg) hits the body before reaching it.  Warp's mesh
    BVH (wp.mesh_query_ray): trimesh's ray backend needs rtree, absent here."""
    global _RAY_KERNEL
    import warp as wp
    if _RAY_KERNEL is None:
        @wp.kernel
        def _k(mid: wp.uint64, o: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3),
               tmax: wp.array(dtype=float), hit: wp.array(dtype=int)):
            i = wp.tid()
            q = wp.mesh_query_ray(mid, o[i], d[i], tmax[i])
            hit[i] = wp.where(q.result, 1, 0)
        _RAY_KERNEL = _k
    vd = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    pts = np.asarray(points, float)
    far = 4.0 * L
    o = pts + far * vd
    wm = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3),
                 indices=wp.array(np.asarray(mesh.faces, np.int32).ravel(), dtype=wp.int32))
    hit = wp.zeros(len(pts), dtype=int)
    wp.launch(_RAY_KERNEL, dim=len(pts),
              inputs=[wm.id, wp.array(o.astype(np.float32), dtype=wp.vec3),
                      wp.array(np.tile(-vd, (len(pts), 1)).astype(np.float32), dtype=wp.vec3),
                      wp.array(np.full(len(pts), far - 1e-3 * L, np.float32), dtype=float), hit])
    return hit.numpy().astype(bool)


def load_arrows(mesh, mask, traction, L, view_dir, visible_only=True):
    """3-D arrows along a region's applied traction (from case.json, not from the
    geometry): one on a small region, 3-4 spread over a large one, sited on faces
    turned toward the camera when the region has any.  A push (the load points
    into the surface) has its TIP on the surface; a pull or a tangential load has
    its TAIL there.  The length is fixed relative to the body (ARROW_LEN x extent):
    the panel shows direction, not magnitude.  A region's arrows keep ARROW_GAP
    apart as printed, REGION_MAX_ARROWS at most.  Returns [(verts, faces, tail, tip,
    site)] and asserts every axis equals the load direction."""
    # A region in several pieces (pads on separate members, batch 20261011-121534: the
    # user saw small load pieces with no arrow, since a region under 3 % of the surface
    # got ONE arrow wherever it fell) gets arrows per piece: each piece is placed on
    # its own, slivers under PIECE_MIN of the region's area are left out, and a piece
    # the camera cannot see at all is left to the region's seen-through shading.
    pieces = mask_pieces(mesh, mask)
    keep = []      # the arrows kept so far, for the printed-gap filter

    def spaced(got):
        out_ = []
        for a_ in got:
            if len(keep) >= REGION_MAX_ARROWS:
                break
            if all(printed_gap(a_[2], a_[3], b_[2], b_[3], view_dir) >= ARROW_GAP * L for b_ in keep):
                keep.append(a_)
                out_.append(a_)
        return out_

    if len(pieces) > 1:
        reg_area = mesh.area_faces[mask].sum()
        pieces = [p for p in pieces if mesh.area_faces[p].sum() >= PIECE_MIN * reg_area]
        pieces = sorted(pieces, key=lambda p: -mesh.area_faces[p].sum())[:PIECE_MAX_ARROWS]
        out = []
        for p in pieces:
            pm = np.zeros(len(mask), bool)
            pm[p] = True
            got = _region_arrows(mesh, pm, traction, L, view_dir, visible_only, skip_hidden=True)
            out += spaced(got[:max(0, PIECE_MAX_ARROWS - len(out))])
        if out:
            return out
    return spaced(_region_arrows(mesh, mask, traction, L, view_dir, visible_only))


def printed_gap(t0, p0, t1, p1, view_dir):
    """The distance between two arrows (tail, tip) as the camera sees them: both
    segments projected on the image plane, the closest approach of the two (sampled,
    exact enough for a spacing test).  Two parallel arrows one behind the other along
    their own axis overlap in print however far apart their sites are in 3-D."""
    vd = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    pr = lambda x: np.asarray(x, float) - np.outer(np.asarray(x, float) @ vd, vd)  # noqa: E731
    w = np.linspace(0.0, 1.0, 9)[:, None]
    a = pr(t0 + w * (np.asarray(p0) - t0))
    b = pr(t1 + w * (np.asarray(p1) - t1))
    return float(np.linalg.norm(a[:, None] - b[None], axis=2).min())


PIECE_MIN = 0.02          # a load piece under this share of its region's area gets no arrow
PIECE_MAX_ARROWS = 16     # arrows per region at most, however many pieces it has


def mask_pieces(mesh, mask):
    """The region's faces split into edge-connected pieces (lists of face indices)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    adj = mesh.face_adjacency
    adj = adj[mask[adj[:, 0]] & mask[adj[:, 1]]]
    pos = np.full(len(mask), -1)
    pos[idx] = np.arange(len(idx))
    g = coo_matrix((np.ones(len(adj)), (pos[adj[:, 0]], pos[adj[:, 1]])), shape=(len(idx), len(idx)))
    n, lab = connected_components(g, directed=False)
    return [idx[lab == i] for i in range(n)]


def _region_arrows(mesh, mask, traction, L, view_dir, visible_only=True, skip_hidden=False):
    """load_arrows on one connected piece (or a whole region); with skip_hidden, a
    piece no camera-visible site can carry returns no arrow."""
    t = np.asarray(traction, float)
    d = t / np.linalg.norm(t)
    idx = np.nonzero(mask)[0]
    cent = mesh.triangles_center[idx]
    nrm = mesh.face_normals[idx]
    area = mesh.area_faces[idx]
    vd = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    front = nrm @ vd > 0.2
    if front.sum() >= max(3, 0.05 * len(idx)):
        idx, cent, nrm, area = idx[front], cent[front], nrm[front], area[front]
    frac = mesh.area_faces[mask].sum() / mesh.area.sum()
    # 4 on a large face: k-means lays them out across it in both directions (2 sat on
    # one line across the culvert's top)
    k = 1 if (frac < 0.03 or not visible_only) else (3 if frac < 0.08 else 4)

    def ends(j, lift=0.0):
        p, n = cent[j], nrm[j]
        surf = p + (0.004 + lift) * L * n
        if d @ n < -0.1:     # pushes into the face: tip on the surface
            return surf - ARROW_LEN * L * d, surf
        return surf, surf + ARROW_LEN * L * d   # pulls or slides: tail on the surface

    # only sites whose WHOLE arrow the camera sees (tail, shaft, neck, tip): a rim or
    # a wall in front of an arrow hid it on the clog's footbed.  Up to 400 candidates.
    # When no site shows it whole, the arrow is lifted off the face along its normal,
    # in steps, until one does: a load on a recessed patch (the insulator's cupped top,
    # batch 20261009-204015) otherwise sat inside the cup, hidden by its rim.
    cand = np.arange(len(idx))
    if len(cand) > 400:
        cand = np.random.default_rng(0).choice(cand, 400, replace=False)
    lift = 0.0
    if visible_only:
        for lift_ in ARROW_LIFTS:
            samples = []
            for j in cand:
                tl, tp = ends(j, lift_)
                samples += [tl + (tp - tl) * w for w in (0.0, 0.35, 0.65, 0.92)]
            hid = occluded(mesh, np.array(samples), view_dir, L).reshape(len(cand), 4).any(axis=1)
            if (~hid).any():
                cand, lift = cand[~hid], lift_
                break
        else:
            if skip_hidden:
                return []
    # sites spread evenly over the region as the CAMERA sees it: area-weighted k-means
    # of its faces projected on the image plane, seeded at area quantiles along the
    # projection's long axis, each centre then snapped to the nearest visible
    # candidate.  Farthest-point sites (before 2026-10-09) put one arrow mid-face and
    # the rest on its far edges (the user: "not distributed uniformly on the loading
    # surface"); clustering in 3-D stacked the culvert's 2x2 one behind the other.
    w = area / area.sum()
    q = cent - np.outer(cent @ vd, vd)
    c0 = (q * w[:, None]).sum(0)
    axis = np.linalg.svd((q - c0) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
    s = (q - c0) @ axis
    order = np.argsort(s)
    cum = np.cumsum(w[order])
    centres = np.array([q[order[min(np.searchsorted(cum, (i + 0.5) / k), len(order) - 1)]]
                        for i in range(k)])
    for _ in range(25):
        lab = np.argmin(np.linalg.norm(q[:, None] - centres[None], axis=2), axis=1)
        for i in range(k):
            m_ = lab == i
            if m_.any():
                centres[i] = (q[m_] * w[m_, None]).sum(0) / w[m_].sum()
    sites, free = [], list(cand)
    for c in centres:
        if not free:
            break
        j = int(free[np.argmin(np.linalg.norm(q[free] - c, axis=1))])
        sites.append(j)
        free.remove(j)
    out = []
    for j in sites:
        tail, tip = ends(j, lift)
        v, f = arrow_mesh(tail, tip, L)
        axis = tip - tail
        assert np.dot(axis / np.linalg.norm(axis), d) > 0.9999, "arrow off its load direction"
        out.append((v, f, tail, tip, int(idx[j])))
    return out


def arrow_mesh(tail, tip, L, sides=16):
    """Shaft (cylinder) + head (cone), closed, as triangles."""
    ax = tip - tail
    ln = np.linalg.norm(ax)
    a = ax / ln
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    rs, rh, hl = ARROW_SHAFT * L, ARROW_HEAD_R * L, min(ARROW_HEAD_L * L, 0.45 * ln)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    neck = tip - hl * a
    V = np.vstack([ring(tail, rs), ring(neck, rs), ring(neck, rh), [tail, tip]])
    i0, i1, i2, ct, cp = 0, sides, 2 * sides, 3 * sides, 3 * sides + 1
    F = []
    for k in range(sides):
        k1 = (k + 1) % sides
        F += [[i0 + k, i0 + k1, i1 + k1], [i0 + k, i1 + k1, i1 + k],      # shaft
              [ct, i0 + k1, i0 + k],                                      # tail cap
              [i1 + k, i1 + k1, i2 + k1], [i1 + k, i2 + k1, i2 + k],      # head base ring
              [i2 + k, i2 + k1, cp]]                                      # cone
    return V, np.array(F)


def write_lines(path, pts, segs, cell_rgb, cell_ids):
    """Legacy VTK polydata of line segments with a colour and an id per segment
    (the id pass decodes them like the BC faces)."""
    with open(path, "w") as out:
        out.write(f"# vtk DataFile Version 3.0\n{os.path.basename(path)}\nASCII\nDATASET POLYDATA\n")
        out.write(f"POINTS {len(pts)} double\n")
        np.savetxt(out, pts, fmt="%.9g")
        out.write(f"LINES {len(segs)} {3 * len(segs)}\n")
        np.savetxt(out, np.column_stack([np.full(len(segs), 2), segs]), fmt="%d")
        out.write(f"CELL_DATA {len(segs)}\nCOLOR_SCALARS rgb 3\n")
        np.savetxt(out, np.clip(cell_rgb, 0, 1), fmt="%.4f")
        ids = np.asarray(cell_ids, float)
        out.write("SCALARS bcid int 1\nLOOKUP_TABLE default\n")
        np.savetxt(out, ids.astype(int), fmt="%d")
        out.write("COLOR_SCALARS idrgb 3\n")
        np.savetxt(out, np.column_stack([ids / 255.0, np.ones_like(ids), np.zeros_like(ids)]),
                   fmt="%.6f")


def support_slab(mesh, mask, L):
    """The support faces offset outward: near sheet at GAP, far sheet at GAP+T,
    side walls along the region's boundary edges.  Returns (vertices, faces)."""
    F = np.asarray(mesh.faces)[mask]
    used, inv = np.unique(F, return_inverse=True)
    f = inv.reshape(-1, 3)
    vn = np.zeros((len(used), 3))
    fn = mesh.face_normals[mask] * mesh.area_faces[mask][:, None]
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-30)
    v0 = np.asarray(mesh.vertices)[used]
    near, far = v0 + MARKER_GAP * L * vn, v0 + (MARKER_GAP + MARKER_T) * L * vn
    m = len(used)
    faces = [f + m, f[:, ::-1]]                      # far sheet faces out, near sheet faces in
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    b = e[idx[cnt == 1]]                              # boundary edges, in face winding
    faces.append(np.column_stack([b[:, 1], b[:, 0], b[:, 0] + m]))
    faces.append(np.column_stack([b[:, 1], b[:, 0] + m, b[:, 1] + m]))
    tm = trimesh.Trimesh(np.vstack([near, far]), np.vstack(faces), process=False)
    tm.fix_normals()
    return np.asarray(tm.vertices), np.asarray(tm.faces)


# -- the ground: a real key-light shadow, or the UI's grid ---------------------------
def key_light_ray(view_dir, elevation=None, azimuth=None):
    """The direction the scene's key light shines, in world coordinates.

    The light kit (A.AR_SURF_VIEW / A.AR_SOLVE_VIEW: the angles of A.KEY_LIGHT,
    elevation 30 and azimuth 0 since 2026-10-22; Figure 1's 45 / -35 before)
    places the key light in CAMERA coordinates, following the camera:
    vtkLight.SetDirectionAngle puts it at (cos e sin a, sin e, cos e cos a) with x
    right, y up, z toward the viewer, shining at the focal point."""
    z = np.asarray(view_dir, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(A.VIEW_UP, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    key = dict(A.KEY_LIGHT)
    elevation = key["KeyLightElevation"] if elevation is None else elevation
    azimuth = key["KeyLightAzimuth"] if azimuth is None else azimuth
    e, a = math.radians(elevation), math.radians(azimuth)
    pos = math.cos(e) * math.sin(a) * x + math.sin(e) * y + math.cos(e) * math.cos(a) * z
    return -pos / np.linalg.norm(pos)


def planar_shadow(V, F, ray, floor):
    """The mesh projected along the light ray onto the plane y = floor: a hard
    shadow from the key light, the shape the light actually casts."""
    t = (floor - V[:, 1]) / ray[1]
    return V + t[:, None] * ray[None, :], F


def grid_cell(V):
    """ui_grid's span (largest bbox side) and cell (the 1/2/5 x 10^k nearest to a
    fifth of it, in log ratio)."""
    span = max(float(np.ptp(V, axis=0).max()), 1e-6)
    raw = span / 5
    decade = 10 ** math.floor(math.log10(raw))
    return span, min((s * decade for s in (1, 2, 5, 10)), key=lambda c: abs(math.log(raw / c)))


def ui_grid(V, floor):
    """src/ui/app.py placeGround() (lines 2906-2991), as frameCamera() calls it
    (2994-2999): span = largest bbox side; cell = the nearest 1/2/5 x 10^k to
    span/5 (roundCell, nearest in log ratio); rings = max(ceil(2.2 span / cell), 3);
    lines snapped to multiples of the cell around the bbox centre, on the plane
    through the lowest point, drawn cell by cell; colour sRGB (0.58, 0.64, 0.72),
    alpha 0.8 (1 - r / reach)^1.4 per vertex, r = distance from the grid origin."""
    lo, hi = V.min(0), V.max(0)
    span, cell = grid_cell(V)
    rings = max(math.ceil(2.2 * span / cell), 3)
    reach = rings * cell
    c = 0.5 * (lo + hi)
    ox, oz = round(c[0] / cell) * cell, round(c[2] / cell) * cell
    pts, segs, rgba = [], [], []

    def corner(x, z):
        pts.append((ox + x, floor, oz + z))
        fade = max(0.0, 1 - math.hypot(x, z) / reach)
        rgba.append((0.58, 0.64, 0.72, fade ** 1.4 * 0.8))
    for line in range(-rings, rings + 1):
        for piece in range(-rings, rings):
            for (x0, z0, x1, z1) in ((line, piece, line, piece + 1), (piece, line, piece + 1, line)):
                corner(x0 * cell, z0 * cell)
                corner(x1 * cell, z1 * cell)
                segs.append((len(pts) - 2, len(pts) - 1))
    return np.array(pts), np.array(segs), np.array(rgba), {"cell": cell, "rings": rings,
                                                            "reach": reach}


def write_lines_rgba(path, pts, segs, rgba):
    with open(path, "w") as out:
        out.write(f"# vtk DataFile Version 3.0\n{os.path.basename(path)}\nASCII\nDATASET POLYDATA\n")
        out.write(f"POINTS {len(pts)} double\n")
        np.savetxt(out, pts, fmt="%.9g")
        out.write(f"LINES {len(segs)} {3 * len(segs)}\n")
        np.savetxt(out, np.column_stack([np.full(len(segs), 2), segs]), fmt="%d")
        out.write(f"POINT_DATA {len(pts)}\nCOLOR_SCALARS rgba 4\n")
        np.savetxt(out, np.clip(rgba, 0, 1), fmt="%.4f")


# -- the camera --------------------------------------------------------------------
def rotate_about_up(v, deg):
    a = math.radians(deg)
    x, y, z = v
    return [x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a)]


def view_dir_at(azimuth_deg, elevation_deg):
    """A.VIEW_DIR turned about the vertical by ``azimuth_deg`` and set to
    ``elevation_deg`` above the horizontal (the paper view is ~20.4 deg)."""
    h = np.array(rotate_about_up(A.VIEW_DIR, azimuth_deg), float)
    h[1] = 0.0
    h /= np.linalg.norm(h)
    e = math.radians(elevation_deg)
    return (h * math.cos(e) + np.array([0.0, math.sin(e), 0.0])).tolist()


def region_shown(mesh, m, d, L, rng):
    """A region's projected area toward the camera, counting only faces the body
    does not hide (Warp ray casts on up to 300 sampled faces, area-weighted)."""
    idx = np.nonzero(m)[0]
    if len(idx) > 300:
        idx = rng.choice(idx, 300, replace=False)
    d = np.asarray(d, float) / np.linalg.norm(d)
    proj = mesh.area_faces[idx] * np.clip(mesh.face_normals[idx] @ d, 0, None)
    if not (proj > 0).any():
        return 0.0, 0.0
    pts = mesh.triangles_center[idx] + 0.002 * L * mesh.face_normals[idx]
    vis = ~occluded(mesh, pts, d, L)
    scale = mesh.area_faces[m].sum() / mesh.area_faces[idx].sum()
    shown = float((proj * vis).sum() * scale)
    # the share of the region's AREA the camera sees (front-facing and not hidden):
    # what the view choice scores -- projected area would favour any steeper camera
    # for a load on a top face
    a = mesh.area_faces[idx]
    share = float((a * ((proj > 0) & vis)).sum() / a.sum())
    return shown, share


PAPER_ELEV = math.degrees(math.atan2(A.VIEW_DIR[1], math.hypot(A.VIEW_DIR[0], A.VIEW_DIR[2])))
# A load arrow seen within this angle of its own axis prints as a stub or a dot (a
# pull along a plate's axis, a push normal to a panel facing the camera: both sat
# at 33 deg from the paper view, batch 20261012-081647).  At 35 deg its projected
# length is sin 35 = 57 % of its 3-D length.
LOS_MIN_DEG = 35.0
# The candidate cameras: every 15 deg of azimuth (turned from the paper azimuth, so
# the paper view is candidate 0) x four elevations from the paper's ~20 deg to 50.
# Nothing below the horizon and nothing steeper than 50 deg (a near-top view loses
# the three-quarter look; a flat part already shows ~77 % of its face at 50).
AZIMUTHS = tuple(range(0, 360, 15))
ELEVATIONS = (PAPER_ELEV, 30.0, 40.0, 50.0)
# No camera whose horizontal heading is within this angle of a principal horizontal
# axis of the body (the world x / z axes the generators build along, and the
# footprint's own principal axes when it is clearly elongated): that is a pure
# front or side view, which flattens the object into its outline.  With the 15 deg
# grid offset by the paper azimuth (26.6 deg off z), the views kept nearest an axis
# sit 26.6 or 33.4 deg off it: a three-quarter view.
AXIS_MIN_DEG = 20.0
FOOTPRINT_ELONGATED = 2.0   # second-moment ratio above which the footprint's axes count
# The paper view stays when it "shows" (choose_view) at least this share of the best
# allowed view; otherwise the views within AREA_TIE of the best are candidates and
# the one nearest the paper view (angle between view directions) is taken, so the
# row turns no further than it must.  50-object batch 2026-10-13: the paper view of the
# four-legged animals (seen near head-on) and of a leaf spring (a thin band at
# 20 deg) showed 64-77 %; the rows the old choice already liked showed >= 85 %.
AREA_KEEP = 0.85
ARROW_POW = 0.5
SUPPORT_MIN = 0.25
# a support some view shows (>= MARKER_BELOW of its area) must show at least this
# share of its best: a looser bar than a load's 50 % (supports are often feet seen
# edge-on), dropped when no view that passes the load test meets it
AREA_TIE = 0.95
VIS_SAMPLES = 4000
HEIGHT_BINS = 16
# the panel layout box's width over its height (ar_figure: P x H; ~1.4 measured on
# the 50-object category figures' three-line rows); only the fill term uses it
CELL_ASPECT = 1.4
SIL_RES = 160               # silhouette ray grid per side (over the bounding sphere)


def silhouette_area(mesh, d, L, centre, R):
    """The body's projected silhouette area along ``d`` (orthographic): the share
    of a SIL_RES^2 grid of parallel rays over the bounding sphere's square that hit
    the body, times that square; and the share of the panel the silhouette fills
    once its bounding box is fitted into the panel.  Same Warp BVH query as
    occluded()."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    right = np.cross(np.asarray(A.VIEW_UP, float), d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    t = (np.arange(SIL_RES) + 0.5) / SIL_RES * 2.0 - 1.0
    u, v = np.meshgrid(t * R, t * R)
    # occluded() casts from pts + 4L d back to pts: start the rays 2L behind the
    # centre so the whole body lies between origin and end
    pts = centre - 2.0 * L * d + u.reshape(-1, 1) * right + v.reshape(-1, 1) * up
    hit = occluded(mesh, pts, d, L)
    area = float(hit.mean() * (2.0 * R) ** 2)
    if not hit.any():
        return area, 0.0
    # the share of the row's panel the silhouette fills once the compose fits its
    # ink box into the panel (ar_figure.row_crop: CELL_ASPECT wide x 1 tall)
    pu, pv = u.ravel()[hit], v.ravel()[hit]
    cell = 2.0 * R / SIL_RES
    w, h = np.ptp(pu) + cell, np.ptp(pv) + cell
    s = min(CELL_ASPECT / w, 1.0 / h)
    return area, float(area * s * s / CELL_ASPECT)


def visible_area(mesh, d, L, n=VIS_SAMPLES):
    """The body's surface area the camera sees along ``d``: front-facing faces not
    hidden by the body (area-weighted sample of ``n`` faces, ray casts), counted at
    their TRUE area, not projected.  Unlike the silhouette it tells the two faces of
    a disc apart: a spoked or ribbed face shows more surface than a plain one."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    rng = np.random.default_rng(0)
    w = mesh.area_faces / mesh.area_faces.sum()
    idx = rng.choice(len(w), size=min(n, len(w)), replace=True, p=w)
    front = mesh.face_normals[idx] @ d > 0
    pts = mesh.triangles_center[idx] + 0.002 * L * mesh.face_normals[idx]
    vis = np.zeros(len(idx), bool)
    if front.any():
        vis[front] = ~occluded(mesh, pts[front], d, L)
    # the share of the body's HEIGHT the view shows: of HEIGHT_BINS horizontal
    # slices holding surface samples, those holding a visible one.  A steep view
    # of an umbrella shows a big canopy and hides the whole handle under it.
    y = mesh.triangles_center[idx][:, 1]
    b = np.clip(((y - y.min()) / max(np.ptp(y), 1e-30) * HEIGHT_BINS).astype(int), 0, HEIGHT_BINS - 1)
    cover = len(np.unique(b[vis])) / len(np.unique(b))
    return float(vis.mean() * mesh.area_faces.sum()), float(cover)


def principal_headings(mesh):
    """Horizontal headings (deg, mod 180) of the body's principal horizontal axes:
    the world x and z axes always, the footprint's own axes (area-weighted second
    moment of the face centres in x-z) when it is clearly elongated."""
    heads = [0.0, 90.0]
    c = mesh.triangles_center[:, [0, 2]]
    w = mesh.area_faces
    c = c - (c * w[:, None]).sum(0) / w.sum()
    M = (c * w[:, None]).T @ c / w.sum()
    ev, evec = np.linalg.eigh(M)
    if ev[0] > 0 and ev[1] / ev[0] >= FOOTPRINT_ELONGATED:
        for k in range(2):
            heads.append(math.degrees(math.atan2(evec[0, k], evec[1, k])) % 180.0)
    return heads


def projected_fraction(direction, view_dir):
    """A 3-D arrow's printed length over its true length, for a camera along
    ``view_dir`` (orthographic stand-in for the 25-degree perspective)."""
    a = np.asarray(direction, float) / np.linalg.norm(direction)
    d = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
    return float(math.sqrt(max(0.0, 1.0 - float(a @ d) ** 2)))


# -- the print check: the compose's own strict checks, predicted per candidate camera --
# The chooser's score cannot see what the compose fails on: a region that prints under
# its floor once the row is fitted into its panel (a pogo stick's peg, 0.46 mm2 at the
# chosen view), a corner label with ink under it in both its corners (a sofa that fills
# its panel at the paper view).  preview_print() renders the row's panels at
# PREVIEW_RES with Warp ray casts from the renderer's own perspective camera (render_ar
# fit(): ResetCamera on the fit points = the bounding sphere seen at A.VIEW_ANGLE, then
# A.DOLLY), with the overlays prep_run would add for that camera (load arrows, support
# outline tubes, the seen-through layers), crops them as ar_figure.row_crop does and
# runs the compose's checks on the result.  PRINT is ar_figure.print_layout() for the
# figure being built (main() sets it; the showcase and the comparison pages leave it
# None and keep the plain chooser).
PRINT = None
PREVIEW_RES = 350           # rays per side of the square render frame (A.PANEL_PX 1400: 4 render px each)
# a region (or arrow) must be predicted at >= PREVIEW_SAFETY x its floor: the preview
# is a quarter of the render's resolution, and the renderer's anti-aliased edges and
# tube caps are not modelled
PREVIEW_SAFETY = 1.15
# a region predicted under this x MIN_REGION_MM2 is measured again at the render's own
# resolution over its box (fine_area); the safety factor then drops to PREVIEW_SAFETY_FINE
PREVIEW_FINE = 4.0
PREVIEW_SAFETY_FINE = 1.0
PREVIEW_AIR = 1             # preview px of air grown around the ink for the label bands
_FACE_KERNEL = None
PREVIEW_LAST = {}           # the last preview's images (debugging only)


def cast_faces(V, F, origin, dirs, tmax):
    """Per ray (from ``origin`` along ``dirs``), the index of the first face of
    (V, F) it hits, -1 for none.  Warp's mesh BVH, as occluded()."""
    global _FACE_KERNEL
    import warp as wp
    if _FACE_KERNEL is None:
        @wp.kernel
        def _k(mid: wp.uint64, o: wp.vec3, d: wp.array(dtype=wp.vec3), tmax: float,
               face: wp.array(dtype=int)):
            i = wp.tid()
            q = wp.mesh_query_ray(mid, o, d[i], tmax)
            face[i] = wp.where(q.result, q.face, -1)
        _FACE_KERNEL = _k
    wm = wp.Mesh(points=wp.array(np.asarray(V, np.float32), dtype=wp.vec3),
                 indices=wp.array(np.asarray(F, np.int32).ravel(), dtype=wp.int32))
    face = wp.zeros(len(dirs), dtype=int)
    wp.launch(_FACE_KERNEL, dim=len(dirs),
              inputs=[wm.id, wp.vec3(*np.asarray(origin, np.float32)),
                      wp.array(np.asarray(dirs, np.float32), dtype=wp.vec3), float(tmax), face])
    return face.numpy()


def tube_mesh(pts, segs, r, sides=12):
    """Line segments as open prisms of radius r (render_ar draws the support
    outline as a tube of MARKER_TUBE x extent)."""
    a, b = pts[segs[:, 0]], pts[segs[:, 1]]
    ax = b - a
    ln = np.linalg.norm(ax, axis=1)
    a, b, ax = a[ln > 0], b[ln > 0], ax[ln > 0] / ln[ln > 0, None]
    ref = np.where(np.abs(ax[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = np.cross(ax, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(ax, u)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    off = r * (np.cos(th)[None, :, None] * u[:, None] + np.sin(th)[None, :, None] * w[:, None])
    verts = np.concatenate([a[:, None] + off, b[:, None] + off], axis=1).reshape(-1, 3)
    k = np.arange(sides)
    k1 = (k + 1) % sides
    tri = np.concatenate([np.stack([k, k1, sides + k1], 1), np.stack([k, sides + k1, sides + k], 1)])
    faces = (np.arange(len(a)) * 2 * sides)[:, None, None] + tri[None]
    return verts, faces.reshape(-1, 3)


def corner_labels(rec, V):
    """The row's corner labels as the compose sets them, per panel: (text, ink box
    (width, above, below the baseline) in mm, extra width left of the text, the
    corners in the compose's order, strict).  Route and grid cell in the generated
    shape panel, the body load in the supports-and-loads panel, xN in the deformed.
    strict: pick_corner fails a row with no corner clear by CORNER_CLEAR (xN, the
    cell); otherwise (right_band: route, body load) the first such corner, else the
    first corner, is taken and only the text's own box must be off the object."""
    Fm, ext = PRINT["fig"], PRINT["extent"]
    out = {"painted": [], "seg": [], "bc": [], "solve": []}
    cell_w = 0.0
    if os.environ.get("AR_GROUND", "grid") == "grid":
        cell = Fm.fmt_cell({"ground": {"grid": {"cell": grid_cell(V)[1]}},
                            "m_per_length_unit": rec.get("m_per_length_unit", 1.0)})
        out[Fm.CELL_PANEL].append((cell, ext(cell), Fm.CELL_ICON + 0.8, ("bl", "tl"), True))
        if Fm.CELL_PANEL == "painted":
            cell_w = Fm.CELL_ICON + 0.8 + PRINT["width"](cell)
    if rec.get("body_source"):
        src = rec["body_source"]
        out["painted"].insert(0, (src, ext(src), 0.0,
                                  Fm.route_order(PRINT["width"](src), cell_w, PRINT["P"]), False))
    body = (rec.get("bc") or {}).get("body_load")
    if body:
        t_ = Fm.body_label(body)
        out["bc"].append((t_, ext(t_), 0.0, ("br", "tr"), False))
    if rec.get("scale"):
        t_ = f"×{Fm.fmt_scale(rec['scale']['factor'])}"
        out["solve"].append((t_, ext(t_), 0.0, ("tr", "tl"), True))
    return out


def _eroded_px(m, k):
    """A mask's pixel count after the compose's one-render-pixel erosion, at a preview
    k render px per side: each boundary edge of the preview mask loses 1/k of a pixel
    (a corner pixel counts two edges: conservative)."""
    if not m.any():
        return 0.0
    p = np.pad(m, 1)
    c = p[1:-1, 1:-1]
    edges = sum(int((c & ~q).sum()) for q in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]))
    return max(float(m.sum()) - edges / k, 0.0)


def preview_print(mesh, d, pv, room=None):
    """The compose's strict checks on this row, predicted for the camera along ``d``
    (see PRINT above).  Returns (misses, numbers): misses is empty when the row is
    predicted to pass (every BC region and arrow over PREVIEW_SAFETY x its print
    floor, its arrows over MIN_ARROW_PROJ of their length, every corner label with a
    free corner), numbers the predicted printed areas and the corners found free."""
    Fm = PRINT["fig"]
    P, H = PRINT["P"], PRINT["H"]
    room = Fm.INK_ROOM if room is None else room     # ink -> layout box bottom (row_crop)
    N = PREVIEW_RES
    k = A.PANEL_PX / N
    d = np.asarray(d, float) / np.linalg.norm(d)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    L = float(np.ptp(V, axis=0).max())
    regions, masks = pv["regions"], pv["masks"]
    # 1. the overlays prep_run adds for this camera
    through = {}                       # region index -> drawn on the see-through layer
    tubes, arrows, outline_pts = [], [], []   # (verts, faces, id, see-through, ...)
    bcid = np.zeros(len(F), np.int64)
    for j, g in enumerate(regions):
        if not g["faces"]:
            continue
        m = np.asarray(masks[g["key"]], bool)
        bcid[m] = j + 1
        a = mesh.area_faces[m]
        nd = mesh.face_normals[m] @ d
        tot = max(float((a * np.abs(nd)).sum()), 1e-30)
        if g["category"] == "neumann":
            vis_area, _ = region_shown(mesh, m, d, L, np.random.default_rng(1))
            st = vis_area / tot < MARKER_BELOW
            through[j] = st
            t_ = pv["tractions"].get(g["name"])
            if t_ is not None and np.linalg.norm(t_) > 0:
                for v_, f_, tail, tip, _s in load_arrows(mesh, m, t_, L, d, visible_only=not st):
                    arrows.append((v_, f_, ARROW_ID0 + len(arrows), st, tip - tail))
        elif g["category"] == "dirichlet":
            facing = float((a * np.clip(nd, 0, None)).sum()) / tot
            away = float((a[:, None] * mesh.face_normals[m]).sum(0) @ d) < 0
            if facing < MARKER_BELOW or away:
                sv, sf = support_outline(mesh, m, L)
                if not len(sf):
                    continue
                st = float((~occluded(mesh, sv[np.unique(sf)], d, L)).mean()) < 0.25
                through[j] = st
                tv, tf = tube_mesh(sv, sf, MARKER_TUBE * L)
                tubes.append((tv, tf, j + 1, st))
                outline_pts.append(sv)     # write_run_files fits ALL of them, drawn or not
    # 2. the renderer's camera (render_ar fit) and one ray per preview pixel
    Vd = pv.get("Vd")
    fit = np.vstack([V] + outline_pts + [a_[0] for a_ in arrows]
                    + ([Vd] if Vd is not None else []))
    lo, hi = fit.min(0), fit.max(0)
    half = math.radians(A.VIEW_ANGLE) / 2
    dist = 0.5 * float(np.linalg.norm(hi - lo)) / math.sin(half) * A.DOLLY
    # a per-object draw_scale f < 1 (ar_runs): render_ar widens the view angle after
    # the fit so the body projects f x as large, and row_crop divides px/mm by f
    ds = float(pv.get("draw_scale") or 1.0)
    half = math.atan(math.tan(half) / ds)
    pos = 0.5 * (lo + hi) + d * dist
    right = np.cross(np.asarray(A.VIEW_UP, float), d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    tmax = 4.0 * dist

    def ray_grid(n, r0=0, r1=None, c0=0, c1=None):
        """Rays through the centres of pixels [r0:r1, c0:c1] of an n x n frame."""
        t_ = ((np.arange(n) + 0.5) / n * 2 - 1) * math.tan(half)
        X, Y = np.meshgrid(t_[c0:c1], -t_[r0:r1])   # row 0 = the top of the image
        r_ = -d + X.reshape(-1, 1) * right + Y.reshape(-1, 1) * up
        return r_ / np.linalg.norm(r_, axis=1, keepdims=True), X.shape
    rays, shape = ray_grid(N)

    def scene(parts, rays=rays, shape=shape):   # [(verts, faces, ids per face)] -> id image (-1: none)
        vs, fs, ids, off = [], [], [], 0
        for v_, f_, i_ in parts:
            vs.append(v_)
            fs.append(np.asarray(f_) + off)
            ids.append(np.broadcast_to(i_, (len(f_),)))
            off += len(v_)
        hit = cast_faces(np.vstack(vs), np.vstack(fs), pos, rays, tmax)
        img = np.where(hit >= 0, np.concatenate(ids)[np.maximum(hit, 0)], -1)
        return img.reshape(shape)

    body = scene([(V, F, 0)]) >= 0
    bc_img = scene([(V, F, bcid)] + [(tv, tf, i_) for tv, tf, i_, st in tubes if not st]
                   + [(v_, f_, i_) for v_, f_, i_, st, _ in arrows if not st])
    deformed = scene([(Vd, F, 0)]) >= 0 if Vd is not None else None
    hid_img = hid_parts = None     # the see-through layers: drawn alone, nothing hides them
    if any(through.values()):
        hm = [j + 1 for j, st in through.items() if st and regions[j]["category"] == "neumann"]
        hid_parts = []
        if hm:
            sel = np.isin(bcid, hm)
            hid_parts.append((V, F[sel], bcid[sel]))
        hid_parts += [(tv, tf, i_) for tv, tf, i_, st in tubes if st]
        hid_parts += [(v_, f_, i_) for v_, f_, i_, st, _ in arrows if st]
        hid_img = scene(hid_parts)
    # 3. the compose's crop (ar_figure.row_crop): the union ink box of the panels
    inks = {"painted": body, "seg": body, "bc": bc_img >= 0, "solve": deformed}
    boxes = []
    for ink in inks.values():
        if ink is None or not ink.any():
            continue
        rr, cc = np.nonzero(ink)
        boxes.append((cc.min(), rr.min(), cc.max() + 1, rr.max() + 1))
    # the compose's own function (draw_scale's shrink about the ink centre included)
    left, top, _, _, _, s = Fm.row_crop(boxes, P, H, 0.0, room, ds)   # preview px per mm
    misses, nums = [], {"preview_px_per_mm": round(s, 2), "regions": {}, "corners": {}}
    PREVIEW_LAST.update(body=body, bc=bc_img, deformed=deformed, hidden=hid_img,
                        crop=(left, top, s), fit=(lo, hi))   # for calibrating against the renders
    # 4. the BC floors (ar_figure.bc_region_check), on the predicted id pass
    mm2 = 1.0 / (s * s)
    proj = float((mesh.area_faces * np.clip(mesh.face_normals @ d, 0, None)).sum())
    scale2 = (bc_img >= 0).sum() * mm2 / max(proj, 1e-30)
    bc_parts = [(V, F, bcid)] + [(tv, tf, i_) for tv, tf, i_, st in tubes if not st] \
        + [(v_, f_, i_) for v_, f_, i_, st, _ in arrows if not st]

    def fine_area(img, parts, rid):
        """A small region's printed area at the RENDER's own resolution: the scene
        recast at A.PANEL_PX over the region's preview box (+2 px), eroded by one
        pixel as the compose erodes it.  Thin outline tubes and slivers are a few
        preview px wide, where the coarse erosion estimate is 10-20 % low."""
        rr, cc = np.nonzero(img == rid)
        if not len(rr):
            return 0.0
        from scipy.ndimage import binary_erosion
        r0, r1 = max(rr.min() - 2, 0), min(rr.max() + 3, N)
        c0, c1 = max(cc.min() - 2, 0), min(cc.max() + 3, N)
        fr, fs_ = ray_grid(A.PANEL_PX, int(r0 * k), int(r1 * k), int(c0 * k), int(c1 * k))
        fine = scene(parts, fr, fs_) == rid
        return float(binary_erosion(fine, iterations=1).sum()) * mm2 / (k * k)
    for j, g in enumerate(regions):
        if not g["faces"]:
            continue
        img_, parts_ = (hid_img, hid_parts) if through.get(j) else (bc_img, bc_parts)
        area = _eroded_px(img_ == j + 1, k) * mm2
        fine = area < PREVIEW_FINE * Fm.MIN_REGION_MM2
        if fine:
            area = fine_area(img_, parts_, j + 1)
        if through.get(j):
            floor = Fm.MIN_REGION_MM2
        else:
            floor = Fm.MIN_SMALL_MM2 if g["area"] * scale2 < Fm.SMALL_REGION_MM2 else Fm.MIN_REGION_MM2
        nums["regions"][g["name"]] = round(area, 2)
        if area < (PREVIEW_SAFETY_FINE if fine else PREVIEW_SAFETY) * floor:
            misses.append(f"{g['name']!r} {area:.2f} mm2 (floor {floor:g})")
    for v_, f_, i_, st, ax_ in arrows:
        area = float(((hid_img if st else bc_img) == i_).sum()) * mm2
        if area < PREVIEW_SAFETY * Fm.MIN_ARROW_MM2:
            misses.append(f"an arrow {area:.2f} mm2")
        if projected_fraction(ax_, d) < Fm.MIN_ARROW_PROJ:
            misses.append("an arrow seen along its axis")
    # 5. the corner labels (ar_figure.pick_corner / right_band): a free corner each
    from scipy.ndimage import binary_dilation
    cw, ch = int(round(P * s)), int(round(H * s))
    for panel, labels in pv["labels"].items():
        ink = inks.get(panel)
        if ink is None or not labels:
            continue
        win = np.zeros((ch, cw), bool)
        r0, c0 = int(round(top)), int(round(left))
        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r0 + ch, N), min(c0 + cw, N)
        if sr1 > sr0 and sc1 > sc0:
            win[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = ink[sr0:sr1, sc0:sc1]
        if PREVIEW_AIR > 0:        # (iterations=0 would dilate until nothing changes)
            win = binary_dilation(win, iterations=PREVIEW_AIR)
        for text, (tw, above, below), extra, order, strict in labels:
            lw = tw + extra

            def clear(corner):          # ar_figure.pick_corner's test: the band + air
                px = int(round((lw + Fm.CORNER_PAD + Fm.CORNER_CLEAR) * s))
                if corner[0] == "t":
                    rows = slice(0, int(round((Fm.LABEL_BASE + 0.6 - Fm.PANEL_PAD_TOP) * s)))
                else:
                    rows = slice(ch - int(round(Fm.BOTTOM_LABEL * s)), ch)
                return not (win[rows, -px:] if corner[1] == "r" else win[rows, :px]).any()

            def on_object(corner):      # the placed label's own overlap check
                base = (Fm.LABEL_BASE - Fm.PANEL_PAD_TOP if corner[0] == "t"
                        else H - Fm.BOTTOM_LABEL + 2.0)
                x0_ = P - Fm.CORNER_PAD - lw if corner[1] == "r" else Fm.CORNER_PAD
                sl = win[max(int((base - above) * s), 0):int((base + below) * s) + 1,
                         max(int(x0_ * s), 0):int((x0_ + lw) * s) + 1]
                return bool(sl.any())
            free = [c for c in order if clear(c)]
            if strict:
                ok, where = bool(free), (free[0] if free else None)
            else:
                where = free[0] if free else order[0]
                ok = not on_object(where)
            nums["corners"][text] = where if ok else None
            if not ok:
                misses.append(f"label {text!r} on the object ({'/'.join(order)})")
    return misses, nums


def choose_view(mesh, masks, cat, explicit=None, avoid=None, preview=None):
    """The row's camera, from 24 azimuths x 4 elevations (paper ~20 deg, 30, 40, 50;
    view-up always vertical, never from below).  Three filters, then one score:

    1. line of sight: ``avoid`` is the governing load's direction (the largest
       traction resultant, or the body load when there is no traction); no view
       within LOS_MIN_DEG of it, either way along it, since its arrow would print
       as a stub.
    2. no pure front/side view: the view's horizontal heading keeps AXIS_MIN_DEG
       off every principal horizontal axis (principal_headings).
    3. loads (and supports some view shows) seen: "shows" = the share of a region's area that faces the
       camera AND is not hidden by the body (ray casts: a load inside a cavity faces
       the sky yet is hidden by the walls from a low camera).  A view scores the
       WORST load region over that region's best view, so a small patch is not
       outvoted by a large one; views scoring >= 50 % pass (when none does, those
       within 90 % of the best score).
    Score ("shows"): the geometric mean of three ratios, each over its best passing
    view -- the body's projected silhouette area (an edge-on plate, an end-on bar or
    a disc seen on its rim is small), its visible surface area (front-facing, not
    hidden: tells a spoked face from a plain back), and the share of the panel the
    silhouette fills once the compose fits its ink box into the panel (a long flat
    part seen steeply turns diagonal and prints small; this term alone would favour
    an end-on bar, the other two forbid it); with a governing load, times its
    arrow's printed length (projected_fraction), so a load seen steeply does not
    shrink to stubs (a factor ^ARROW_POW on the mean); and times the share of the
    body's height the view shows (visible_area), so a steep view cannot hide a
    whole part under a wide top.  The paper view (A.VIEW_DIR) is kept
    when it passes and shows >= AREA_KEEP of the best; otherwise, of the passing
    views within AREA_TIE of the best, the one nearest the paper view.  (Projected
    area of the LOAD was tried first: it grows with elevation for any load on a top
    face, so the paper view lost on four of five rows for no reason a reader would
    see.  Silhouette + visible area alone sent flat parts -- a USB stick, a mouse,
    a crocodile -- to 50 deg, where they print smaller: hence the fill term; and a
    loaded table top to 50 deg, where its arrows print short: hence the arrow term.)

    With ``preview`` (prep_run passes one when PRINT is set), the views are tried in
    that preference order (then every other passing view, best score first) and the
    first preview_print predicts the compose to pass is taken: no per-object pins.

    AR_VIEW_TABLE=1 prints every candidate's filters and terms."""
    if explicit:
        return {"dir": list(explicit), "azimuth_deg": None, "why": "explicit view_dir"}
    views = [(az, el) for el in ELEVATIONS for az in AZIMUTHS]
    dirs = [list(A.VIEW_DIR) if (az == 0 and el == PAPER_ELEV) else view_dir_at(az, el)
            for az, el in views]
    unit = [np.asarray(d, float) / np.linalg.norm(d) for d in dirs]
    if avoid is not None and np.linalg.norm(avoid) > 0:
        a = np.asarray(avoid, float) / np.linalg.norm(avoid)
        los = np.array([math.degrees(math.acos(min(1.0, abs(float(u @ a))))) for u in unit])
    else:
        los = np.full(len(views), 90.0)
    heads = principal_headings(mesh)
    heading = np.array([math.degrees(math.atan2(u[0], u[2])) for u in unit])
    axis_off = np.array([min(min(abs((h - p) % 180.0), 180.0 - abs((h - p) % 180.0))
                             for p in heads) for h in heading])
    ok = (los >= LOS_MIN_DEG) & (axis_off >= AXIS_MIN_DEG)
    if not ok.any():                      # never seen in practice: drop the axis filter
        ok = los >= LOS_MIN_DEG
    assert ok.any(), "every candidate view looks along the governing load"
    near = np.array([math.degrees(math.acos(min(1.0, float(u @ unit[0])))) for u in unit])

    V = np.asarray(mesh.vertices, float)
    L = float(np.ptp(V, axis=0).max())
    lo, hi = V.min(0), V.max(0)
    centre, R = 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo))
    sil = np.array([silhouette_area(mesh, d, L, centre, R) for d in dirs])
    area, fill = sil[:, 0], sil[:, 1]
    va = np.array([visible_area(mesh, d, L) for d in dirs])
    vis, cover = va[:, 0], va[:, 1]
    area_n = area / (R * R)               # over the bounding size: comparable across rows

    loads = [np.asarray(masks[key], bool) for key in masks.files
             if cat.get(key, key.split("_")[0]) == "neumann" and np.asarray(masks[key]).any()]
    # supports count too, but only those some candidate view shows at all (>=
    # MARKER_BELOW of their area): feet soles face the floor from every camera and
    # are drawn as marker slabs instead.  Without them, an umbrella held at its
    # handle tip went to a 50-deg view whose canopy hides the handle and support.
    sups = [np.asarray(masks[key], bool) for key in masks.files
            if cat.get(key, key.split("_")[0]) == "dirichlet" and np.asarray(masks[key]).any()]
    score = sup_score = None
    if loads:
        rng = np.random.default_rng(0)
        shown = np.zeros((len(loads), len(views)))
        for v, d in enumerate(dirs):
            for r, m in enumerate(loads):
                shown[r, v] = region_shown(mesh, m, d, L, rng)[1]
        frac = shown / shown.max(axis=1, keepdims=True).clip(1e-30)
        score = frac.min(axis=0)
        best_load = float(score[ok].max())
        good = ok & (score >= min(0.5, 0.9 * best_load))
    else:
        good = ok.copy()
    if sups:
        rng = np.random.default_rng(0)
        ss = np.zeros((len(sups), len(views)))
        for v, d in enumerate(dirs):
            for r, m in enumerate(sups):
                ss[r, v] = region_shown(mesh, m, d, L, rng)[1]
        ss = ss[ss.max(axis=1) >= MARKER_BELOW]
        if len(ss):
            sup_score = (ss / ss.max(axis=1, keepdims=True)).min(axis=0)
            if (good & (sup_score >= SUPPORT_MIN)).any():
                good &= sup_score >= SUPPORT_MIN
            else:
                sup_score = None
    # the view's score: the geometric mean of its silhouette and its visible surface,
    # each over its best passing view (1 = the most of the object any view shows)
    shows = np.cbrt(area / area[good].max() * vis / vis[good].max() * fill / fill[good].max())
    if avoid is not None and np.linalg.norm(avoid) > 0:
        # times the governing load's arrow as printed (its length over its true
        # length, over its best) to the ARROW_POW: a steeper camera shortens a
        # vertical load's arrows (a table top loaded downward, seen at 50 deg: 64 %
        # of their length against 94 % at 20).  A factor, not a fourth mean term,
        # so it does not dilute the shape terms where all views print it alike.
        arrow = np.array([projected_fraction(avoid, d) for d in dirs])
        shows = shows * (arrow / arrow[good].max()) ** ARROW_POW
    shows = shows * cover / cover[good].max()
    best = float(shows[good].max())
    if os.environ.get("AR_VIEW_TABLE"):
        for v in range(len(views)):
            print(f"  az{views[v][0]:4d} el{views[v][1]:5.1f} head{heading[v]:7.1f} off{axis_off[v]:5.1f} los{los[v]:5.1f} "
                  f"ok{int(ok[v])} good{int(good[v])} load{(score[v] if score is not None else -1):5.2f} sil{area[v]/area[good].max():5.2f} fill{fill[v]/fill[good].max():5.2f} vis{vis[v]/vis[good].max():5.2f} cover{cover[v]:5.2f} shows{shows[v]:5.2f}")

    def result(v, why):
        az, el = views[v]
        r = {"dir": dirs[v], "azimuth_deg": az, "elevation_deg": round(el, 1),
             "governing_load_dir": None if avoid is None else list(map(float, avoid)),
             "line_of_sight_deg": round(float(los[v]), 1),
             "axis_off_deg": round(float(axis_off[v]), 1),
             "silhouette_area_over_R2": round(float(area_n[v]), 4),
             "visible_area_over_R2": round(float(vis[v] / (R * R)), 4),
             "panel_fill": round(float(fill[v]), 3),
             "height_shown": round(float(cover[v]), 3),
             "shows": round(float(shows[v]), 3), "paper_view_shows": round(float(shows[0]), 3),
             "turn_from_paper_deg": round(float(near[v]), 1), "why": why}
        if score is not None:
            r["worst_load_shown_fraction_of_its_best"] = float(score[v])
        if sup_score is not None:
            r["worst_support_shown_fraction_of_its_best"] = float(sup_score[v])
        return r

    # the preference order: the paper view when it passes and shows >= AREA_KEEP of the
    # best; then the passing views within AREA_TIE of the best, nearest the paper view
    # first; then every other passing view, best score first.  Without a print check
    # the first is taken; with one (preview_print), the first the compose is
    # predicted to pass.
    cand = good & (shows >= AREA_TIE * best)
    keep_paper = bool(good[0] and shows[0] >= AREA_KEEP * best)
    order = [0] if keep_paper else []
    order += [int(v) for v in np.argsort(near, kind="stable") if cand[v] and v not in order]
    order += [int(v) for v in np.argsort(-shows, kind="stable") if good[v] and v not in order]
    checked, v, passes, lifted = [], order[0], None, None
    if preview is not None:
        passes = False
        for u_ in order:
            misses, nums = preview_print(mesh, dirs[u_], preview)
            checked.append({"azimuth_deg": views[u_][0], "elevation_deg": round(views[u_][1], 1),
                            "shows_over_best": round(float(shows[u_] / best), 3),
                            "misses": misses, "predicted": nums})
            if os.environ.get("AR_VIEW_TABLE"):
                print(f"  print check az{views[u_][0]:4d} el{views[u_][1]:5.1f}: "
                      + ("passes" if not misses else "; ".join(misses)) + f"  {nums}")
            if not misses:
                v, passes = u_, True
                break
        # No view passes: try them again with this row's ink lifted above the bottom
        # label band (ar_figure.LIFTED_ROOM; the compose then crops the row so).  A
        # tall centred body leaves no bottom corner for its route or body-load label
        # in any view (the paper figure's leaning tower of Pisa); every other row
        # keeps INK_ROOM and is drawn as before.
        if not passes and PRINT["fig"].LIFTED_ROOM > PRINT["fig"].INK_ROOM:
            for u_ in order:
                misses, nums = preview_print(mesh, dirs[u_], preview, room=PRINT["fig"].LIFTED_ROOM)
                checked.append({"azimuth_deg": views[u_][0], "elevation_deg": round(views[u_][1], 1),
                                "shows_over_best": round(float(shows[u_] / best), 3),
                                "ink_room": PRINT["fig"].LIFTED_ROOM,
                                "misses": misses, "predicted": nums})
                if os.environ.get("AR_VIEW_TABLE"):
                    print(f"  print check (ink lifted) az{views[u_][0]:4d} el{views[u_][1]:5.1f}: "
                          + ("passes" if not misses else "; ".join(misses)) + f"  {nums}")
                if not misses:
                    v, passes, lifted = u_, True, PRINT["fig"].LIFTED_ROOM
                    break
    # why the view was taken: the preference rule for the first preferred view...
    if v == 0:
        why = ["paper view"]
    elif keep_paper:
        why = [f"paper view preferred (shows {shows[0] / best:.0%} of the best)"]
    elif not good[0]:
        why = ["paper view excluded (" + ", ".join(
            w for w, bad in ((f"within {LOS_MIN_DEG:g} deg of the governing load", los[0] < LOS_MIN_DEG),
                             (f"within {AXIS_MIN_DEG:g} deg of a principal axis", axis_off[0] < AXIS_MIN_DEG),
                             ("shows a load region at < 50 % of its best",
                              score is not None and not (score[0] >= min(0.5, 0.9 * best_load))),
                             (f"shows a support at < {SUPPORT_MIN:.0%} of its best",
                              sup_score is not None and sup_score[0] < SUPPORT_MIN))
            if bad) + ")"]
    else:
        why = [f"paper view shows {shows[0] / best:.0%} of the best (< {AREA_KEEP:.0%})"]
    if v != 0:
        why.append(f"the view nearest it within {AREA_TIE:.0%} of the best" if cand[v]
                   else f"a passing view at {shows[v] / best:.0%} of the best")
    # ... and the print check's verdict on the views preferred to it
    if passes is False:
        why.append(f"no candidate is predicted to pass the compose's print checks "
                   f"({len(checked)} tried): the first preferred view kept")
    elif passes and len(checked) > 1:
        why.append(f"after {len(checked) - 1} preferred view(s) predicted to fail the "
                   "compose's print checks")
    if lifted:
        why.append("with the row's ink lifted above the bottom label band (no view passed without)")
    r = result(v, "; ".join(why))
    if lifted:
        r["ink_room"] = lifted
    if preview is not None:
        r["print_check"] = {"passes": passes, "candidates_checked": len(checked),
                            "rejected": [c for c in checked if c["misses"]],
                            "predicted": checked[-1]["predicted"] if passes else None}
    return r


# -- the solve ---------------------------------------------------------------------
# -- the segmentation column (AR_SEGMENTATION=1, ar_figure.SEGMENTATION) ----------------
# The parts the BC selectors were resolved against: resolve-bcs always reads
# boundary_conditions/segments.npz (graph.py passes it for every method: PartField,
# shape diameter, CAD B-rep faces), one label per face of geometry/physical.glb.  They
# are coloured as the PartField bunny (partfield_segmentation, pfbunny/pf_prep.py) and
# the overview's segmentation panel colour theirs: the overview's eight pastel tints
# (ov_style.SEGMENT_TINTS), handed out by ov_palette_check.segment_tints, which tries
# every assignment and keeps the one whose closest TOUCHING pair of parts is farthest
# apart.  That search is over 8!/(8-k)! assignments and needs k <= 8 parts; a run with
# more (the Opus knee implant asked PartField for 14) is coloured by
# greedy_segment_tints instead, the same objective solved greedily, tints reused only
# between parts that do not touch.
def _palette_gate():
    sys.path.insert(0, os.path.join(HERE, "..", "overview"))
    import ov_palette_check as G  # noqa: E402
    return G


def greedy_segment_tints(labels, face_adjacency, tints):
    """segment_tints' objective for more parts than tints: parts in order of most
    touching neighbours (then largest label count), each given the tint farthest
    (min normal-vision dE00) from the tints its already-coloured touching neighbours
    have; ties go to the tint used least so far.  Returns what segment_tints does."""
    G = _palette_gate()
    ids = [int(v) for v in np.unique(labels)]
    la, lb = labels[face_adjacency[:, 0]], labels[face_adjacency[:, 1]]
    cut = la != lb
    touch = sorted({tuple(sorted(p)) for p in zip(la[cut].tolist(), lb[cut].tolist())})
    nb = {v: set() for v in ids}
    for a, b in touch:
        nb[a].add(b)
        nb[b].add(a)
    dE = {(i, j): G.pair("a", tints[i], "b", tints[j])[0]
          for i in range(len(tints)) for j in range(len(tints)) if i != j}
    count = np.bincount(np.asarray(labels, np.int64))
    colour, used = {}, [0] * len(tints)
    for v in sorted(ids, key=lambda v: (-len(nb[v]), -int(count[v]), v)):
        taken = [colour[u] for u in nb[v] if u in colour]
        best = max(range(len(tints)), key=lambda t: (
            min((dE[(t, c)] if t != c else 0.0) for c in taken) if taken else 0.0, -used[t]))
        colour[v] = best
        used[best] += 1
    worst = min((dE[(colour[a], colour[b])] if colour[a] != colour[b] else 0.0)
                for a, b in touch) if touch else float("inf")
    return colour, touch, worst


def segmentation_of(session, mesh, phys):
    """(record, per-face RGB or None): the session's segments.npz on physical.glb's
    faces in the bunny's tints.  None (the panel then reads 'not available') when the
    run has no segmentation, or one over another surface (face count, or the surface
    sha256 that segments.source.json records)."""
    bc = os.path.join(session, "boundary_conditions")
    npz = os.path.join(bc, "segments.npz")
    if not os.path.exists(npz):
        return {"source": None, "why": "no boundary_conditions/segments.npz"}, None
    labels = np.asarray(np.load(npz)["segment_ids"], np.int64)
    F = np.asarray(mesh.faces)
    if len(labels) != len(F):
        return {"source": None, "why": f"segments.npz labels {len(labels)} faces, "
                                       f"physical.glb has {len(F)}"}, None
    rec = {"source": "boundary_conditions/segments.npz", "faces": int(len(labels))}
    sj = os.path.join(bc, "segments.json")
    if os.path.exists(sj):
        meta = json.load(open(sj))
        rec.update(method=meta.get("method"), num_clusters=meta.get("num_clusters"))
    src = os.path.join(bc, "segments.source.json")
    if os.path.exists(src):
        import hashlib
        want = json.load(open(src)).get("sha256")
        got = hashlib.sha256(open(phys, "rb").read()).hexdigest()
        rec["surface_sha256_matches"] = want == got
        if want and want != got:
            return {**rec, "source": None,
                    "why": "segments.source.json names another surface (sha256)"}, None
    G = _palette_gate()
    S = G.S
    tints = S.SEGMENT_TINTS
    adj = np.asarray(mesh.face_adjacency)
    ids = np.unique(labels)
    if not (labels[adj[:, 0]] != labels[adj[:, 1]]).any():
        # one part, or parts that never touch: segment_tints has no pair to separate
        colour, touch, min_de = {int(v): k % len(tints) for k, v in enumerate(ids)}, [], None
        rec["assignment"] = "in label order (no touching parts)"
    elif len(ids) <= len(tints):
        colour, touch, min_de = G.segment_tints(labels, adj, tints)
        rec["assignment"] = "ov_palette_check.segment_tints (every assignment tried)"
    else:
        colour, touch, min_de = greedy_segment_tints(labels, adj, tints)
        rec["assignment"] = f"greedy_segment_tints ({len(ids)} parts > {len(tints)} tints)"
    tint_rgb = np.array([S._hex(c) for c in tints], dtype=float)
    rgb = tint_rgb[np.array([colour[int(c)] for c in labels])]
    area = np.bincount(labels, weights=mesh.area_faces, minlength=int(ids.max()) + 1)
    rec.update(parts=int(len(ids)), touching_pairs=len(touch),
               min_touching_dE00=None if not touch else round(float(min_de), 1),
               tints={str(int(v)): tints[colour[int(v)]] for v in ids},
               area_fraction={str(int(v)): round(float(area[v] / area.sum()), 4) for v in ids})
    return rec, rgb


def chosen_run(session):
    """The solve the session reports: operator/report.md's ``"run":`` when there
    is one (the operator's final run), else the newest run with a volume VTU."""
    runs = sorted(glob.glob(os.path.join(session, "solve", "api_fem_forward-*")))
    runs = [r for r in runs if glob.glob(os.path.join(r, "vtu", "solution_t*.vtu"))]
    report = os.path.join(session, "operator", "report.md")
    if os.path.exists(report):
        m = re.search(r'"run":\s*"(api_fem_forward-[0-9-]+)"', open(report).read())
        if m and os.path.join(session, "solve", m[1]) in runs:
            return os.path.join(session, "solve", m[1]), "operator/report.md"
    return (runs[-1], "newest run with a volume VTU") if runs else (None, None)


def solve_failed(session):
    """failed.json newer than the field export, or no export at all."""
    f = os.path.join(session, "solve", "failed.json")
    g = os.path.join(session, "solve", "displacement_magnitude.glb")
    if not os.path.exists(g):
        return True
    return os.path.exists(f) and os.path.getmtime(f) > os.path.getmtime(g)


def interpolate_from_vtu(vtu_path, verts, torch, octree_hex_interpolate, device):
    d = read_vtu(vtu_path)
    nodes = np.asarray(d["Points"], dtype=np.float64).reshape(-1, 3)
    offsets = np.asarray(d["offsets"])
    types = np.asarray(d["types"])
    assert np.all(types == 12) and np.all(np.diff(np.r_[0, offsets]) == 8), \
        f"{vtu_path}: not an all-hex mesh"
    elems = np.asarray(d["connectivity"], dtype=np.int64).reshape(-1, 8)
    tn = torch.as_tensor(nodes, device=device)
    te = torch.as_tensor(elems, device=device)
    q = torch.as_tensor(verts, dtype=torch.float64, device=device)
    # the interpolant on the node coordinates themselves: a point inside a hex
    # must come back where it is, whatever the corner ordering convention
    xq = octree_hex_interpolate(tn, te, tn, q).cpu().numpy()
    err = np.linalg.norm(xq - verts, axis=1)
    fields = {"displacement": octree_hex_interpolate(
        tn, te, torch.as_tensor(np.asarray(d["displacement"], np.float64), device=device), q
    ).cpu().numpy()}
    if "youngs_modulus" in d:
        fields["youngs_modulus"] = octree_hex_interpolate(
            tn, te, torch.as_tensor(np.asarray(d["youngs_modulus"], np.float64), device=device), q
        ).cpu().numpy()
        fields["E_nodes"] = (float(np.min(d["youngs_modulus"])), float(np.max(d["youngs_modulus"])))
    fields["umax_nodes"] = float(np.linalg.norm(np.asarray(d["displacement"], float), axis=1).max())
    return fields, err, len(nodes), len(elems)


def grid_E(session, verts):
    """Fallback (no VTU): sample the per-part / VoMP 64^3 material grid, nearest."""
    field = os.path.join(session, "materials", "parts", "field")
    if not os.path.isdir(field):
        cand = glob.glob(os.path.join(session, "materials", "*", "field", "manifest.json")) + \
            glob.glob(os.path.join(session, "materials", "**", "manifest.json"), recursive=True)
        if not cand:
            return None
        field = os.path.dirname(cand[0])
    man = json.load(open(os.path.join(field, "manifest.json")))
    grid = np.load(os.path.join(field, "material_grid.npy"))[..., 0]
    valid = np.load(os.path.join(field, "valid_mask.npy")).astype(bool)
    lo, hi = np.array(man["bounds"][0]), np.array(man["bounds"][1])
    n = np.array(grid.shape)
    ijk = np.clip(np.floor((verts - lo) / (hi - lo) * n).astype(int), 0, n - 1)
    from scipy.spatial import cKDTree
    vi = np.argwhere(valid)
    _, nn = cKDTree(vi).query(ijk)
    ijk = np.where(valid[tuple(ijk.T)][:, None], ijk, vi[nn])
    return grid[tuple(ijk.T)].astype(float)


# -- one run -----------------------------------------------------------------------
GENERATORS = {"triposg": "TripoSG", "hunyuan3d": "Hunyuan3D", "cadcoder": "CAD-Coder",
              "text2cad": "Text2CAD"}


def body_source(session):
    """Where the body came from, for the generated-shape panel's corner (user,
    2026-10-11): the generator and what it was given, or the operator's own build."""
    p = os.path.join(session, "session.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    mode = d.get("geometry_mode")
    gen = ((d.get("generation") or {}).get("generator") or "").lower()
    name = str((d.get("input_asset") or {}).get("source_name") or "")
    if mode == "surface":
        m = re.match(r"blank_([a-z]+)_", name)
        if m:
            return f"Built ({m.group(1)})"
        return "Uploaded mesh"
    g = GENERATORS.get(gen, gen or "generator")
    if gen in ("cadcoder", "text2cad"):
        return g
    if mode == "image":
        return f"SDXL → {g}" if d.get("drawn_image") else f"Image → {g}"
    if mode in ("scribble", "text_scribble"):
        return f"Scribble → {g}"
    return f"Text → {g}"


def real_mass(session):
    """The operator's stated real mass in kg (operator/brief.md "Real mass: 4.5 kg"),
    or None."""
    p = os.path.join(session, "operator", "brief.md")
    if not os.path.exists(p):
        return None
    m = re.search(r"Real mass:\s*([0-9.eE+-]+)\s*(kg|g|t)\b", open(p).read())
    if not m:
        return None
    return float(m.group(1)) * {"kg": 1.0, "g": 1e-3, "t": 1e3}[m.group(2)]


def field_mean_density(session):
    """The mean density over the valid voxels of the session's spatial material
    field (per-part or VoMP artifact, in the session's units), or None."""
    cand = glob.glob(os.path.join(session, "materials", "*", "field", "manifest.json")) + \
        glob.glob(os.path.join(session, "materials", "**", "manifest.json"), recursive=True)
    for man_p in cand:
        man = json.load(open(man_p))
        if "density" not in (man.get("channels") or []):
            continue
        root = os.path.dirname(man_p)
        grid = np.load(os.path.join(root, "material_grid.npy"))
        valid = np.load(os.path.join(root, "valid_mask.npy")).astype(bool)
        rho = grid[..., man["channels"].index("density")][valid]
        if rho.size:
            return float(rho.mean()), os.path.relpath(man_p, session)
    return None, None


def body_load_force(session, mesh, case, mat, kind, vec, units):
    """The body load as the force it applies, in newtons (user, 2026-10-12: "not in g").
    The runner may scale gravity so the weight equals the REAL object's, so |a| in g
    says little; the applied weight is the solved body's mass x |a|, the mass being
    the physical surface's enclosed volume (geometry/physical.glb, the case's
    surface, session units) x the case density.  A spatial material (no single
    density) takes the field's mean density over its valid voxels; with no field,
    the operator's stated real mass.  A body_force is a force per volume: x volume.
    Every quantity stays in the session's units until the last step; one force unit
    is Pa_per_stress x m_per_length^2 newtons (1 N in SI and SI-mm, 1 lbf in US-inch)."""
    n_per_force = units[0] * units[1] ** 2
    vol = float(abs(mesh.volume))
    out = {"volume": vol, "surface_watertight": bool(mesh.is_watertight)}
    a = float(np.linalg.norm(vec))
    rm = real_mass(session)
    if kind == "body_force":
        out.update(weight_N=a * vol * n_per_force,
                   weight_method="|loads.body_force| x volume of geometry/physical.glb")
        return out
    rho = (case.get("material") or {}).get("density")
    mode = (case.get("material") or {}).get("mode") or mat.get("mode")
    if rho is not None and mode in (None, "uniform"):
        mass = vol * float(rho)
        out.update(density=float(rho), mass=mass,
                   weight_method="volume of geometry/physical.glb x case material.density x "
                                 "|loads.acceleration|")
    else:
        rho_f, src = field_mean_density(session)
        if rho_f is not None:
            mass = vol * rho_f
            out.update(density=rho_f, mass=mass,
                       weight_method=f"volume of geometry/physical.glb x mean density of the "
                                     f"material field ({src}) x |loads.acceleration|")
        elif rm is not None:
            out.update(mass_kg=rm, weight_N=rm * a * units[1],
                       weight_method="operator/brief.md Real mass x |loads.acceleration| "
                                     "(spatial material, no density field)")
            return out
        else:
            out.update(weight_N=None, weight_method="no density and no stated real mass")
            return out
    # mass x acceleration in session units is a force in its force unit
    out["mass_kg"] = mass * n_per_force / units[1]      # mass unit = force unit s^2 / length unit
    out["weight_N"] = mass * a * n_per_force
    if rm is not None:     # cross-check only: what the runner aimed the weight at
        out["real_mass_kg"] = rm
        out["real_weight_N"] = rm * 9.80665
        out["weight_over_real_weight"] = out["weight_N"] / (rm * 9.80665)
    return out


def _prep_solve(rec, out, run, s, V, L, mat, ctx):
    """prep_run's solve step: |u| and E moved onto the surface, the x N factor.
    View-independent, so it runs BEFORE the camera is chosen: the print check
    (preview_print) needs the magnified deformed shape, which the compose crops with."""
    # 3./4. solve: displacement and E at the surface
    rec["material_mode"] = mat.get("mode")
    run_dir, why = chosen_run(s)
    failed = solve_failed(s)
    E = None
    if run_dir and not failed:
        vtus = sorted(glob.glob(os.path.join(run_dir, "vtu", "solution_t*.vtu")))
        fields, err, nn, ne = interpolate_from_vtu(vtus[-1], V, *ctx)
        inside = err < 1e-6 * L
        rec["solve"] = {"run": os.path.basename(run_dir), "chosen_by": why,
                        "vtu": os.path.relpath(vtus[-1], s), "nodes": nn, "hexes": ne,
                        "coord_check_inside_fraction": float(inside.mean()),
                        "coord_check_max_err_inside": float(err[inside].max()) if inside.any() else None,
                        "outside_points": int((~inside).sum())}
        # nearly every surface point must sit in a hex and come back exactly; under
        # AR_FLOOR=warn (visual-exam builds) a member thinner than a leaf, whose trimmed
        # surface falls partly outside the mesh, is listed instead of stopping the build
        short = f"{run.get('session_id')}: only {inside.mean():.1%} of the surface lies inside the solve mesh"
        if os.environ.get("AR_FLOOR", "strict") == "warn":
            if inside.mean() <= 0.95:
                rec["notes"].append(short)
        else:
            assert inside.mean() > 0.95, short
        u = fields["displacement"]
        umag = np.linalg.norm(u, axis=1)
        umax = float(umag.max())
        gj = os.path.join(s, "solve", "displacement_magnitude.glb.json")
        ref = json.load(open(gj)).get("max_displacement") if os.path.exists(gj) else None
        rec["solve"].update({"umax_surface": umax, "umax_nodes": fields["umax_nodes"],
                             "umax_workflow_export": ref,
                             "ratio_to_workflow_export": umax / ref if ref else None})
        if ref and not 0.8 < umax / ref < 1.25:
            rec["notes"].append(f"surface |u|max {umax:.4g} differs from the workflow's own "
                                f"export {ref:.4g}")
        # The UI's slider top (telemetry.deformation_scales, 0.3 x extent) is kept in the
        # record for reference; the FIGURE uses its own, gentler rule (FIG_FRACTION).
        ui = UI_SCALES.get(s) or {}
        scale = nice_nearest(FIG_FRACTION * L / umax) if umax > 0 else 1.0
        scale_mult = float(os.environ.get("AR_SCALE_MULT", "1"))   # comparison strips only
        # A body that STANDS on its support (a support facing down at its lowest
        # point) must not be pushed through that floor by the magnification: step
        # down the 1/2/5 ladder until the magnified shape dips at most DIP_MAX of
        # the extent below the undeformed base (there is no contact in the solve).
        y0 = V[:, 1].min()
        standing = any(g["category"] == "dirichlet" and (g.get("mean_normal") or [0, 0, 0])[1] < -0.7
                       for g in ((rec.get("bc") or {}).get("regions") or []))
        stepped = []
        while standing and scale > 1.0 and \
                y0 - (V[:, 1] + scale * u[:, 1]).min() > DIP_MAX * L:
            stepped.append(scale)
            scale = ladder_below(scale)
        scale *= scale_mult
        rec["scale"] = {"factor": scale,
                        "rule": f"1/2/3/5 x 10^k nearest (log ratio) to {FIG_FRACTION:g} x largest "
                                f"extent / |u|max, s >= 1; standing bodies step down until they "
                                f"sink <= {DIP_MAX:g} x extent",
                        "ui": ui, "displayed_peak_over_extent": scale * umax / L,
                        "standing_on_support": bool(standing), "stepped_down_from": stepped}
        # how far the magnified shape dips below the undeformed body's lowest point
        # (there is no contact: a load can push it through the floor it stands on)
        dip = float(V[:, 1].min() - (V + scale * u)[:, 1].min())
        rec["scale"]["below_floor_over_extent"] = max(dip, 0.0) / L
        out.update(u=u, umag=umag, umax=umax, scale=scale)
        if "youngs_modulus" in fields:
            E = fields["youngs_modulus"]
            rec["E_source"] = "solve VTU youngs_modulus"
            rec["E_nodes"] = fields["E_nodes"]
    else:
        rec["solve"] = None
        rec["scale"] = None
        rec["notes"].append("no completed solve" + (" (solve/failed.json is the latest)" if failed and run_dir else ""))
    if E is None:
        if mat.get("mode") == "uniform" and mat.get("youngs_modulus"):
            E = np.full(len(V), float(mat["youngs_modulus"]))
            rec["E_source"] = "case/material.json (uniform)"
        else:
            E = grid_E(s, V)
            rec["E_source"] = "material grid (nearest)" if E is not None else None
    units = UNITS.get(rec["unit_system"])
    if units is None:
        rec["notes"].append(f"unknown unit system {rec['unit_system']!r}: SI assumed")
        units = UNITS["SI (m-kg-s)"]
    rec["pa_per_stress_unit"], rec["m_per_length_unit"] = units
    if E is not None:
        rec["E_surface"] = (float(E.min()), float(E.max()))
        rec["E_surface_Pa"] = (float(E.min()) * units[0], float(E.max()) * units[0])
        if mat.get("mode") == "uniform" and mat.get("youngs_modulus"):
            assert np.allclose(E, mat["youngs_modulus"], rtol=1e-5), \
                "uniform material, but the solve's E is not uniform"
        E = E * units[0]        # Pa: one colour scale across rows in any unit system
    else:
        rec["notes"].append("no Young's modulus found")
    if rec.get("solve"):
        rec["solve"]["umax_surface_m"] = rec["solve"]["umax_surface"] * units[1]
    out["E"] = E


def prep_run(i, run, work, ctx):
    rec = {"index": i, **run.record(), "notes": [], "files": {}}
    s = run.session
    if not s or not os.path.isdir(s):
        rec["notes"].append("no workflow session folder found")
        return rec, None
    rec["body_source"] = body_source(s)
    phys = os.path.join(s, "geometry", "physical.glb")
    if not os.path.exists(phys):
        rec["notes"].append("no geometry/physical.glb: nothing to draw")
        return rec, None
    mesh = load_mesh(phys)
    s_dir = s
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    rec["files"]["surface"] = os.path.relpath(phys, s)
    rec["surface"] = {"vertices": len(V), "faces": len(F)}
    L = float(np.ptp(V, axis=0).max())
    rec["extent"] = L
    rec["dims"] = np.ptp(V, axis=0).tolist()      # x, y (up), z: the row text names height
    tf = os.path.join(s, "geometry", "physical.glb.transform.json")
    rec["unit_system"] = json.load(open(tf)).get("units") if os.path.exists(tf) else None
    mat_json = os.path.join(s, "case", "material.json")
    mat = json.load(open(mat_json)) if os.path.exists(mat_json) else {}
    rec["unit_system"] = rec["unit_system"] or mat.get("unit_system")
    out = {"V": V, "F": F}

    # 1. painted / generated surface
    painted = None
    for rel, kind in (("geometry/texture/textured.glb", "texture step"),     # current workflow
                      ("materials/texture/textured.glb", "texture step"),     # earlier sessions
                      ("geometry/generated.glb", "generator colours")):
        p = os.path.join(s, rel)
        if not os.path.exists(p):
            continue
        m = load_mesh(p)
        rgb = mesh_colours(m)
        if rgb is None:
            continue
        if rel.startswith("geometry/generated") and len(m.vertices) != len(V):
            continue        # the generator's own frame/topology: not safely transferable
        rgb, how = transfer_vertex_colours(m, rgb, V)
        if how.startswith("nearest") and float(how.split("distance ")[1].rstrip(")")) > \
                0.01 * float(np.ptp(V, axis=0).max()):
            rec["notes"].append(f"{rel}: not in the surface's frame and not the same mesh; "
                                "its colours were not used")
            continue
        painted = rgb
        # ov_surfaces asserts a paint's spread > 0.02 before it calls it paint;
        # here a flat result is shown as it is and labelled, not dropped
        spread = float(rgb.std(0).max())
        rec["painted"] = {"source": kind, "file": rel, "transfer": how,
                          "rgb_mean": rgb.mean(0).round(3).tolist(), "rgb_spread": spread,
                          "near_uniform": spread < 0.02}
        if spread < 0.02:
            rec["notes"].append(f"{rel}: near-uniform colour (spread {spread:.3f})")
        break
    if painted is None:
        rec["painted"] = {"source": None}
        rec["notes"].append("no painted surface: the plain surface is shown in clay")
    out["painted"] = painted

    # 1b. the segmentation the BC selectors were resolved against (the figure's
    # segmentation column, AR_SEGMENTATION=1 only)
    if ar_figure.SEGMENTATION:
        rec["segmentation"], out["seg_rgb"] = segmentation_of(s, mesh, phys)
        if out["seg_rgb"] is None:
            rec["notes"].append(f"no segmentation panel: {rec['segmentation']['why']}")

    # 2. BC regions
    rz, rj = (os.path.join(s, "boundary_conditions", n) for n in ("resolved.npz", "resolved.json"))
    if os.path.exists(rz) and os.path.exists(rj):
        masks = np.load(rz)
        regions = json.load(open(rj)).get("regions", [])
        cat = {r["key"]: r["category"] for r in regions if "key" in r}
        rgb = np.tile(np.array(A.CLAY), (len(F), 1))
        cover = np.zeros(len(F), int)
        summary = []
        # Dirichlet first, Neumann over it, as method_overview paints ears then base;
        # overlaps are counted, not hidden
        for want, colour in (("dirichlet", A.DIRICHLET), ("neumann", A.NEUMANN)):
            for key in masks.files:
                c = cat.get(key, key.split("_")[0])
                if c != want:
                    continue
                m = np.asarray(masks[key], bool)
                assert len(m) == len(F), f"{key}: mask over {len(m)} faces, surface has {len(F)}"
                rgb[m] = colour
                cover += m
                name = next((r["name"] for r in regions if r.get("key") == key), key)
                # area-weighted mean outward normal: says WHERE a region faces
                # (the compose names a support it cannot see "underside")
                fn = mesh.face_normals[m] * mesh.area_faces[m][:, None]
                n = fn.sum(0)
                n = (n / np.linalg.norm(n)).round(3).tolist() if np.linalg.norm(n) > 0 else None
                summary.append({"key": key, "name": name, "category": c, "faces": int(m.sum()),
                                "mean_normal": n, "area": float(mesh.area_faces[m].sum())})
        rec["bc"] = {"regions": summary, "overlap_faces": int((cover > 1).sum())}
        # the case's loads, as the solve was given them: the traction per Neumann region
        # and the body load (loads.acceleration, a weight; loads.body_force, a force per
        # volume).  A body load has no region, so the BC panel used to show none.
        case_p = os.path.join(s_dir, "case", "case.json")
        case = json.load(open(case_p)) if os.path.exists(case_p) else {}
        tractions = {}
        for nb in (case.get("boundary_conditions") or {}).get("neumann") or []:
            if nb.get("traction") is not None:
                tractions[nb["name"]] = nb["traction"]
        units_ = UNITS.get(rec["unit_system"]) or UNITS["SI (m-kg-s)"]
        body = None
        for kind in ("acceleration", "body_force"):
            vec = np.asarray((case.get("loads") or {}).get(kind) or [0.0, 0.0, 0.0], float)
            if np.linalg.norm(vec) > 0:
                body = {"kind": kind, "vector": vec.tolist(),
                        "dir": (vec / np.linalg.norm(vec)).tolist()}
                if kind == "acceleration":
                    # in standard gravities, the acceleration being in the session's length unit / s^2
                    body["g"] = float(np.linalg.norm(vec) * units_[1] / 9.80665)
                body.update(body_load_force(s, mesh, case, mat, kind, vec, units_))
                break
        rec["bc"]["body_load"] = body
        # the governing load: the traction with the largest resultant (|t| x region area),
        # else the body load; the camera keeps off its line of sight
        gov, gov_f = None, 0.0
        for g in summary:
            t_ = tractions.get(g["name"])
            if g["category"] == "neumann" and t_ is not None and np.linalg.norm(t_) > 0:
                f_ = float(np.linalg.norm(t_)) * g["area"]
                if f_ > gov_f:
                    gov, gov_f = np.asarray(t_, float), f_
        if gov is None and body is not None:
            gov = np.asarray(body["dir"], float)
        _prep_solve(rec, out, run, s, V, L, mat, ctx)
        preview = None
        if PRINT is not None:
            # what the compose will check on this row, for preview_print: the
            # regions, their tractions, the magnified deformed shape, the corner labels
            preview = {"regions": summary, "masks": masks, "tractions": tractions,
                       "Vd": (V + out["scale"] * out["u"]) if "u" in out else None,
                       "labels": corner_labels(rec, V), "draw_scale": run.draw_scale}
        rec["view"] = choose_view(mesh, masks, cat, run.view_dir, avoid=gov, preview=preview)
        if body is not None:
            body["projected_fraction"] = projected_fraction(body["dir"], rec["view"]["dir"])
        # front-facing projected areas toward the chosen camera (no occlusion): the
        # compose turns them into each region's best-case printed size
        dv = np.array(rec["view"]["dir"], float)
        dv /= np.linalg.norm(dv)
        proj = mesh.area_faces * np.clip(mesh.face_normals @ dv, 0, None)
        rec["bc"]["surface_proj_area"] = float(proj.sum())
        hidden = np.zeros(len(F), bool)
        for g in summary:
            mm_ = np.asarray(masks[g["key"]], bool)
            g["proj_area"] = float(proj[mm_].sum())
            # a LOAD on a face turned away from the camera cannot be shown by any
            # camera that also shows its opposite (a culvert loaded on both walls):
            # it is drawn seen-through instead (render_ar bc_hidden, ar_figure)
            if g["category"] == "neumann" and g["faces"]:
                a_ = mesh.area_faces[mm_]
                tot = (a_ * np.abs(mesh.face_normals[mm_] @ dv)).sum()
                g["facing_camera"] = float(g["proj_area"] / max(tot, 1e-30))
                # counted with occlusion: a load inside a cavity faces the camera
                # yet is hidden by the walls
                vis_area, _ = region_shown(mesh, mm_, dv, L, np.random.default_rng(1))
                g["unoccluded_facing"] = float(vis_area / max(tot, 1e-30))
                if g["unoccluded_facing"] < MARKER_BELOW:
                    g["seen_through"] = True
                    hidden |= mm_
        if hidden.any():
            out["bc_hidden"] = hidden
        # region ids per face (0 = clay, k+1 = summary[k], later regions win, as the
        # colours do): the renderer draws an id pass from them, and the compose
        # checks every region's visibility and rendered colour on it
        bcid = np.zeros(len(F), np.int32)
        for k, g in enumerate(summary):
            bcid[np.asarray(masks[g["key"]], bool)] = k + 1
        out["bcid"] = bcid
        # each support facing away from the camera gets its own marker slab
        # (design review 1 item 1; per region, review 2 N4)
        d = np.array(rec["view"]["dir"], float)
        d /= np.linalg.norm(d)
        slabs, hidden_slabs = [], []
        for k, g in enumerate(summary):
            if g["category"] != "dirichlet" or not g["faces"]:
                continue
            m = np.asarray(masks[g["key"]], bool)
            a = mesh.area_faces[m]
            facing = float((a * np.clip(mesh.face_normals[m] @ d, 0, None)).sum() /
                           (a * np.abs(mesh.face_normals[m] @ d)).sum().clip(1e-30))
            g["facing_camera"] = facing
            # a support whose area-weighted normal points away from the camera is
            # marked too: rounded sole edges can face it by 18 % (a bar stool,
            # acceptance batch 20261010-040732) yet print 0 mm2 once eroded
            away = float((a[:, None] * mesh.face_normals[m]).sum(0) @ d) < 0
            if facing < MARKER_BELOW or away:
                sv, sf = support_outline(mesh, m, L)
                # the outline under a sole can itself be hidden (a high camera looks
                # down past the clog's heel): then it is drawn seen-through
                vis_frac = float((~occluded(mesh, sv[np.unique(sf)], d, L)).mean())   # the drawn points only
                if vis_frac < 0.25:     # a base outline is never more than ~half visible (its back runs behind the body)
                    g["support_seen_through"] = True
                    hidden_slabs.append((sv, sf, k + 1))
                else:
                    slabs.append((sv, sf, k + 1))
                g["marker"] = {"kind": "outline", "visible_fraction": vis_frac, "gap": MARKER_GAP * L,
                               "tube_radius": MARKER_TUBE * L,
                               "why": f"faces the camera at {facing:.1%} (< {MARKER_BELOW:.0%})"}
        if slabs:
            off, vs, fs, ids = 0, [], [], []
            for sv, sf, rid in slabs:
                vs.append(sv)
                fs.append(sf + off)
                ids.append(np.full(len(sf), rid, np.int32))
                off += len(sv)
            out["support_marker"] = (np.vstack(vs), np.vstack(fs), np.concatenate(ids))
        if hidden_slabs:
            off, vs, fs, ids = 0, [], [], []
            for sv, sf, rid in hidden_slabs:
                vs.append(sv)
                fs.append(sf + off)
                ids.append(np.full(len(sf), rid, np.int32))
                off += len(sv)
            out["support_hidden"] = (np.vstack(vs), np.vstack(fs), np.concatenate(ids))
        # load arrows, along each Neumann region's traction from case/case.json
        arrows, hidden_arrows, arrow_rec = [], [], []
        for k, g in enumerate(summary):
            if g["category"] != "neumann" or not g["faces"]:
                continue
            t_ = tractions.get(g["name"])
            if t_ is None or np.linalg.norm(t_) == 0:
                rec["notes"].append(f"load {g['name']!r}: no traction in case.json, no arrow")
                continue
            m = np.asarray(masks[g["key"]], bool)
            for v_, f_, tail, tip, site in load_arrows(mesh, m, t_, L, rec["view"]["dir"],
                                                       visible_only=not g.get("seen_through")):
                aid = ARROW_ID0 + len(arrow_rec)
                arrow_rec.append({"id": aid, "region": g["name"], "traction": list(map(float, t_)),
                                  "tail": tail.tolist(), "tip": tip.tolist(),
                                  "projected_fraction": projected_fraction(tip - tail,
                                                                           rec["view"]["dir"]),
                                  "seen_through": bool(g.get("seen_through"))})
                (hidden_arrows if g.get("seen_through") else arrows).append((v_, f_, aid))
        rec["bc"]["arrows"] = arrow_rec
        out["arrows"], out["hidden_arrows"] = arrows, hidden_arrows
        unknown = [k for k in masks.files if cat.get(k, k.split("_")[0]) not in ("dirichlet", "neumann")]
        if unknown:
            rec["notes"].append(f"BC masks of unknown category left clay: {unknown}")
        out["bc"] = rgb
    else:
        rec["bc"] = None
        rec["notes"].append("no boundary_conditions/resolved.npz")
        out["bc"] = None

    if rec.get("bc") is None:
        _prep_solve(rec, out, run, s, V, L, mat, ctx)
    return rec, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs-json")
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    ap.add_argument("--no-print-check", action="store_true",
                    help="choose cameras without predicting agentic_results' compose checks "
                         "(the comparison pages: another layout)")
    args = ap.parse_args()
    runs = ar_runs.load_runs(args.runs, args.runs_json)
    assert runs, "no runs given"
    global PRINT
    if not args.no_print_check:
        # the figure's own layout (its row height follows its longest prompt) and label
        # widths, for preview_print
        PRINT = {"fig": ar_figure, **ar_figure.print_layout([r.get("instruction") for r in runs],
                                                             [r.prompt_max_lines for r in runs])}
    os.makedirs(args.work, exist_ok=True)
    for old in glob.glob(os.path.join(args.work, "ar_*")):
        os.remove(old)          # never render a previous build's panels (illustrations §5d)

    sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))
    sys.path.insert(0, REPO)
    import torch
    from core.remeshing import octree_hex_interpolate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch, octree_hex_interpolate, device)

    # the UI's own deformation scale, from its own function in its own env
    sessions = [r.session for r in runs if r.session]
    cmd = ["conda", "run", "-n", os.environ.get("AR_UI_ENV", "mechfoundry-agents"), "python",
           os.path.join(HERE, "ar_ui_scale.py"), *sessions]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, f"ar_ui_scale.py failed:\n{res.stderr[-2000:]}"
    UI_SCALES.update(json.loads(res.stdout.strip().splitlines()[-1]))

    recs, outs = [], []
    for i, run in enumerate(runs):
        rec, out = prep_run(i, run, args.work, ctx)
        recs.append(rec)
        outs.append(out)
        print(f"[{i}] {run.get('session_id')}: {run.get('instruction')!r}  notes={rec['notes']}")

    # the instruction card's lines and font size (ar_figure.prompt_fit), recorded so
    # the figure's record says which prompt was set smaller and why
    probe = ar_figure.Page(ar_figure.W, 50)
    for rec, run in zip(recs, runs):
        lines, fs = ar_figure.prompt_fit(probe, run.get("instruction"), run.prompt_max_lines)
        rec["prompt"] = {"max_lines": run.prompt_max_lines, "fs": fs, "lines": lines,
                         "rule": "card size 7 pt" if run.prompt_max_lines is None else
                         "largest size <= 7 pt (0.1 pt steps, >= 5 pt) that wraps into max_lines"}
    lo, hi = e_log_range(outs)
    for rec, out in zip(recs, outs):
        if out is not None:
            write_run_files(rec, out, args.work, lo, hi)
    stats = {"E_log10_range_Pa": [lo, hi],
             "view_dir_default": A.VIEW_DIR, "runs": recs}
    json.dump(stats, open(os.path.join(args.work, "ar_stats.json"), "w"), indent=1, default=float)
    print(f"wrote {args.work}/ar_stats.json")


def e_log_range(outs):
    """One log10-E scale (Pa) for the whole figure, so a colour means one modulus in
    every row; half-decade ends, at least one decade wide.  (None, None) without E."""
    Es = [o["E"] for o in outs if o is not None and o.get("E") is not None]
    if not Es:
        return None, None
    lo = float(np.log10(min(e.min() for e in Es)))
    hi = float(np.log10(max(e.max() for e in Es)))
    if hi - lo < 1.0:                     # a single modulus: centre it in one decade
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5, mid + 0.5
    return math.floor(lo * 2) / 2, math.ceil(hi * 2) / 2   # half-decade ticks


def write_run_files(rec, out, work, lo, hi, e_rgb=None):
    """Every ar_<i>_*.vtk render_ar.py reads for one prepped run (``rec``/``out`` from
    prep_run), the E colours on the shared [lo, hi] log10 scale.  Fills rec["ground"]
    and rec["panels"].  Shared with ../showcase/sc_prep.py.  ``e_rgb`` (t in [0, 1] ->
    RGB) colours the E map; None is the shared "steel" map (A.e_rgb), which every
    agentic-results figure uses -- only the showcase passes its own."""
    i, V, F = rec["index"], out["V"], out["F"]
    p = lambda name: os.path.join(work, f"ar_{i}_{name}.vtk")  # noqa: E731
    write_polydata(p("painted"), V, F, point_rgb=out["painted"] if out["painted"] is not None
                   else np.tile(np.array(A.CLAY), (len(V), 1)))
    if out.get("seg_rgb") is not None:
        write_polydata(p("seg"), V, F, cell_rgb=out["seg_rgb"])
    if out["bc"] is not None:
        write_polydata(p("bc"), V, F, cell_rgb=out["bc"], cell_ids=out["bcid"])
    if out.get("bc_hidden") is not None:
        hf = F[out["bc_hidden"]]
        write_polydata(p("bc_hidden"), V, hf, cell_ids=out["bcid"][out["bc_hidden"]])
    if out["E"] is not None:
        t = np.clip((np.log10(out["E"]) - lo) / (hi - lo), 0, 1)[:, None]
        write_polydata(p("materials"), V, F, point_rgb=(e_rgb or A.e_rgb)(t[:, 0]))
    fit = V
    if "support_marker" in out:
        sv, sf, sid = out["support_marker"]
        write_lines(p("support"), sv, sf, np.tile(np.array(A.DIRICHLET), (len(sf), 1)), sid)
        fit = np.vstack([fit, sv])
    if "support_hidden" in out:
        sv, sf, sid = out["support_hidden"]
        write_lines(p("support_hidden"), sv, sf, np.tile(np.array(A.DIRICHLET), (len(sf), 1)), sid)
        fit = np.vstack([fit, sv])
    for key_, name_ in (("arrows", "arrows"), ("hidden_arrows", "arrows_hidden")):
        if out.get(key_):
            off, vs, fs, ids = 0, [], [], []
            for v_, f_, aid in out[key_]:
                vs.append(v_)
                fs.append(f_ + off)
                ids.append(np.full(len(f_), aid))
                off += len(v_)
            vs, fs, ids = np.vstack(vs), np.vstack(fs), np.concatenate(ids)
            write_polydata(p(name_), vs, fs, cell_rgb=np.tile(np.array(A.LOAD_ARROW), (len(fs), 1)),
                           cell_ids=ids)
            fit = np.vstack([fit, vs])
    if "u" in out:
        write_polydata(p("solve"), V, F, vectors=out["u"], scalars=out["umag"])
        fit = np.vstack([fit, V + out["scale"] * out["u"]])
    write_polydata(p("fit"), fit)
    # ground layers, one floor per row (the same under all four panels): the
    # lowest point of the body, or of the support marker where a row has one,
    # so the floor never passes over the teal plate
    floor = float(V[:, 1].min())
    if "support_marker" in out:
        floor = min(floor, float(out["support_marker"][0][:, 1].min()))
    floor -= 1e-4 * rec["extent"]
    vdir = (rec.get("view") or {}).get("dir") or A.VIEW_DIR
    ray = key_light_ray(vdir)
    sv, sf = planar_shadow(V, F, ray, floor)
    write_polydata(p("shadow"), sv, sf)
    if "u" in out:
        dv, df = planar_shadow(V + out["scale"] * out["u"], F, ray, floor)
        write_polydata(p("shadow_def"), dv, df)
    gp, gs, gc, ginfo = ui_grid(V, floor)
    write_lines_rgba(p("grid"), gp, gs, gc)
    rec["ground"] = {"floor": floor, "key_light_ray": ray.round(4).tolist(), "grid": ginfo}
    rec["panels"] = sorted(os.path.basename(q) for q in glob.glob(os.path.join(work, f"ar_{i}_*.vtk")))
    if rec.get("scale"):
        print(f"[{i}] displacement scale x{rec['scale']['factor']:g} "
              f"(|u|max {out['umax']:.4g}, extent {rec['extent']:.4g}, displayed "
              f"{rec['scale']['displayed_peak_over_extent']:.1%} of the extent)")


if __name__ == "__main__":
    main()
