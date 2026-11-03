"""Stage 2 (env mechfoundry-warp117): the 50 objects -> an amphitheater seen from the stage.

    python am_layout.py --work _work

Reads gl_prep.py's ``obj_<k>.npz`` + ``gl_objects.json``.  Writes ``am_scene.json`` (camera,
seats, tiers, the parameter search) and the world-placed surfaces ``scene_def.vtk``,
``scene_und.vtk`` for render_gl.py (no shadows: the user, 2026-10-17).

Layout coordinates are the schematic's (x right, y away from the stage, z up; D0 = 1 = the
smallest object's drawn size); the VTK world is y-up: (x, y, z) -> (x, z, -y).  Base rules:
the approved schematic v5; revisions: the designer's amphitheater eye-check (fixes 1-4) and
the user's requirements of 2026-10-17.

SIZE      D = D0 (L / Lmin)^P, P = 0.25, before projection; rows filled smallest first.
SEATS     a row's capacity is SPAN x r of arc length (r: the row's distance from the stage
          centre on the axis); a seat is max(D, label footprint) x (1 + gap), gap 0.35, or
          GAP_LABEL 0.55 where the label sets the width (fix 4).  The label footprint is the
          world width the fixed printed label needs at that depth x cos(25 deg); layout and
          camera fit are iterated until the rows are stable.  Next row: r + (D_max(row) +
          D_first(next)) (1 + ROW_GAP) / 2.  In a row: largest at the centre, alternating
          outward.  The rows are arcs about a centre ARC_BACK x R0 behind the stage centre
          (flatter rows, fix 3); odd rows are shifted along their arc by STAGGER x the mean
          seat (fix 4).
TIERS     a row's floor rises (a) so the sight line over the row in front hides <= HIDE of
          its tallest object, and (b) so that line clears the near edge of the row's own
          label strip (fix 1: the labels' floor is seen).
CAMERA    the eye CAM_BACK R0 behind the stage centre at height EYE_H; aimed so the scene's
          projected box is centred; horizontal FOV fitted to 97 % of the 16:9 page.
SWAPS     GL_SEAT_SWAPS="a skyscraper:a lighthouse" (the user's choice, 2026-10-17): the named
          objects exchange seats after the rule-based row assignment and before tiers,
          labels and gates.  The one named exception to "no per-object tuning"; it is
          recorded in am_scene.json and exempted from the size-order gate by name.
SEARCH    ARC_BACK, EYE_H, SPAN and STAGGER are not hand-set: every combination of the grid
          below is laid out, its objects Warp-cast from the eye at 0.25 mm and its labels
          placed by am_labels' hard rules; among the layouts where every label passes and
          every object stays >= 25 % visible, the one covering the most of the page wins
          (covered share of 2.5 mm page cells by objects + labels; then the smaller largest
          empty rectangle).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
sys.path.insert(0, HERE)
import ar_prep  # noqa: E402
import am_labels  # noqa: E402

from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

# -- page -------------------------------------------------------------------------------------
PAGE_W = 180.0
PAGE_H = float(os.environ.get("GL_PAGE_H", "101.25"))      # set per candidate from the grid's page_h
#   rev 5 (the user, 2026-10-19): the page height is searched (180 x 110-135 mm, a taller,
#   more compact block like method_overview's full width); GL_PAGE_H only seeds it
FILL = 0.97
# -- fixed rules ------------------------------------------------------------------------------
P = float(os.environ.get("GL_P", "0.20"))       # rev 4 (the user): sizes closer, 0.25 -> 0.20
D0 = 1.0
R0 = 18.0 * D0
GAP = 0.35
GAP_LABEL = 0.55
ROW_GAP = 0.3               # rev 3 (the user: rows closer together)
CAM_BACK = 2.0
HIDE = 0.8
TURN_MAX = 15.0              # rev 6 (the user: long labels read badly at the row ends): 25 -> 15
LABEL_FRONT = 0.6
LANES = 3                   # rev 3: labels below alternate between lanes along a row (rev 6: 3),
#                             so a label competes for row length only with the next-but-one
LABEL_LANE_CLEAR = 0.5      # mm between the two lanes (= am_labels.LABEL_CLEAR)
LABEL_SEAT_CLEAR = 0.8      # mm between same-lane labels along the row (0.5 + raster headroom)
VIS_MIN = 0.25
# -- the searched grid (fix 3/4 knobs; the user: chosen by page coverage) -----------------------
GRID = {"arc_back": (5.0,), "eye_h": (24.0, 32.0), "span_deg": (120.0,),
        "stagger": (0.0,), "row_gap": (0.1,), "page_h": (90.0, 100.0, 110.0),
        "rows_n": (4, 5), "pack": (0.0, 0.3), "balance_rows": (True,), "close": (1,),
        "row_overlap": (0.3, 0.6)}
#                                  rev 8 (the user: "the top row is too far away"): tiers
#                                  rise for one label line of floor, not three (close = 1;
#                                  close = 2, no strip at all, left 14 labels unplaceable);
#                                  the page may get shorter; rows step closer by row_overlap
#                                  of their half-footprints; 48 points
ROW_SPANS_FRAME = True
ROW_FILL = 0.95                    # share of the frame width a row may fill
SEARCH_RES = 0.25           # mm per pixel of the search's silhouettes
SEARCH_SAFETY = 0.25        # mm added to the label clearances in the search (one search pixel)
FOOT = {}                   # k -> drawn footprint size (largest horizontal side), set by load_objects
CELL = 2.5                  # mm, coverage cells
# -- labels -------------------------------------------------------------------------------------
FONT_ITALIC = "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"
PT_MM = 25.4 / 72.0
LABEL_PT = float(os.environ.get("GL_LABEL_PT", "5.0"))     # cap 0.716 x 5 pt = 1.26 mm (the user, rev 5: the journal minimum)
SHORTEN_WITH = os.environ.get("GL_SHORTEN_WITH", "1") == "1"
SWAPS = [tuple(x.split(":")) for x in os.environ.get("GL_SEAT_SWAPS", "").split(";") if ":" in x]
# user-allowed per-object arrangement (2026-10-19): GL_OVERRIDES names a JSON of fixed
# parameters, seat widenings, per-row spacing factors and per-object seat nudges, each with
# its reason; with it the search is skipped (the user kept revision 8's layout)
OVR = json.load(open(os.environ["GL_OVERRIDES"])) if os.environ.get("GL_OVERRIDES") else {}
_FP = FontProperties(fname=FONT_ITALIC)


def label_text(instr):
    """The prompt in curly quotes; a trailing " with ..." clause shortened to " …" (an
    elided quotation, the user's decision); the head noun is never cut."""
    s = instr.replace("'", "’")
    if SHORTEN_WITH and " with " in s:
        s = s.split(" with ", 1)[0] + " …"
    return "“" + s + "”"


def label_size_mm(text, pt=None):
    pt = LABEL_PT if pt is None else pt
    e = TextPath((0, 0), text, size=1, prop=_FP).get_extents()
    return (e.x1 - e.x0) * pt * PT_MM, (e.y1 - e.y0) * pt * PT_MM


def turn_deg(a, span):
    """Page rotation of a label at seat angle a (rad; a < 0 = left): clockwise on the left."""
    return math.copysign(TURN_MAX * min(1.0, abs(a) / (span / 2)), a)


# =====================================================================================
class Camera:
    """Pinhole camera in schematic coordinates; page in mm, y up, origin bottom left."""

    def __init__(self, eye, look, f_mm):
        self.eye = np.asarray(eye, float)
        self.set_look(look)
        self.f = f_mm

    def set_look(self, look):
        look = np.asarray(look, float)
        self.l = look / np.linalg.norm(look)
        r = np.cross(self.l, [0.0, 0.0, 1.0])
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.l)

    def norm(self, P):
        v = np.atleast_2d(P) - self.eye
        zc = v @ self.l
        return np.column_stack([(v @ self.r) / zc, (v @ self.u) / zc]), zc

    def page(self, P):
        n, zc = self.norm(P)
        return np.column_stack([PAGE_W / 2 + self.f * n[:, 0], PAGE_H / 2 + self.f * n[:, 1]]), zc

    def fov_h(self):
        return 2 * math.degrees(math.atan(PAGE_W / 2 / self.f))


def layout(objs, D, H, f_mm, prm, look=None, swaps=(), widen=None):
    """Seats, rows and tiers for one parameter set under focal length f_mm."""
    cam_y = -CAM_BACK * R0
    eye_h, span, arc_back, stagger = prm["eye_h"], math.radians(prm["span_deg"]), prm["arc_back"], prm["stagger"]
    lab_wh = {k: label_size_mm(label_text(o["instruction"])) for k, o in enumerate(objs)}

    def depth_at(r):          # camera depth of a seat at radius r on the axis
        if look is not None:
            return float((np.array([0.0, r, 0.0]) - [0.0, cam_y, eye_h]) @ look)
        return (r - cam_y) * math.cos(math.atan2(eye_h, r - cam_y))

    widen = widen or {}

    def wl_of(k, r):          # world width of k's printed label along the row
        return lab_wh[k][0] * depth_at(r) / f_mm * math.cos(math.radians(TURN_MAX))

    pack = prm.get("pack")      # rev 6 (the user: a packed audience): seat = the object's
    #                             own footprint x (1 + pack), pack may be negative (overlap);
    #                             labels no longer widen seats (lanes / shifts make room)

    def slot(k, r):
        """Row length a seat needs (rev 3): the object with its gap, or HALF its label
        (two lanes: a label competes only with the next-but-one) plus clearance."""
        if pack is not None:
            # repair rounds add printed row length (REPAIR_PAD_MM each), not a factor: a
            # thin object's (toothbrush, cane) footprint x 1.15 gains nothing
            rounds = round(math.log(widen.get(k, 1.0)) / math.log(REPAIR_WIDEN)) if k in widen else 0
            return FOOT.get(k, D[k]) * (1 + pack) + rounds * REPAIR_PAD_MM * depth_at(r) / f_mm
        clr = LABEL_SEAT_CLEAR * depth_at(r) / f_mm
        return max(D[k] * (1 + GAP), 0.5 * wl_of(k, r) + clr) * widen.get(k, 1.0)

    def seat_positions(seq, r):
        """Arc positions along a row: neighbours keep their bodies apart, the next-but-one
        (same lane) keep their labels apart."""
        clr = LABEL_SEAT_CLEAR * depth_at(r) / f_mm
        w = [widen.get(k, 1.0) for k in seq]
        wl = [wl_of(k, r) for k in seq]
        x = [0.0]
        for i in range(1, len(seq)):
            if pack is not None:
                x.append(x[i - 1] + 0.5 * (slot(seq[i - 1], r) + slot(seq[i], r)))
                continue
            xi = x[i - 1] + 0.5 * (D[seq[i - 1]] * w[i - 1] + D[seq[i]] * w[i]) * (1 + GAP)
            if i >= LANES:
                j = i - LANES
                xi = max(xi, x[j] + (0.5 * (wl[j] + wl[i]) + clr) * 0.5 * (w[j] + w[i]))
            x.append(xi)
        x = np.array(x)
        return x - 0.5 * (x[0] + x[-1])

    def row_cap(r, dmax):
        """Row length a row may fill.  Rev 4 (the user: content to both side edges along its
        full height): every row spans the camera's frame width at its own depth, so near and
        far rows reach the same page edges (no fan); before: SPAN x r."""
        if ROW_SPANS_FRAME:
            return ROW_FILL * PAGE_W * depth_at(r) / f_mm
        return span * (r + 0.5 * dmax)

    order = sorted(range(len(objs)), key=lambda k: (objs[k]["L_m"], k))
    def next_radius(r, cur, k):
        # rev 3: rows are spaced by FOOTPRINTS, not by the largest extent (a tall tower
        # needs little floor; the eye sees over it -- overlap of bodies is allowed); the
        # floor strip in front of the next row holds its labels (foreshortened)
        depth = max(FOOT.get(j, D[j]) for j in cur)
        rg = prm.get("row_gap", ROW_GAP)
        r_std = r + depth * (1 + rg) * 0.5 + FOOT.get(k, D[k]) * (1 + rg) * 0.5
        el = math.atan2(eye_h, r_std - cam_y)
        strip_mm = LANES * max(lab_wh[j][1] for j in cur) + LABEL_LANE_CLEAR + 0.6 + 0.3
        if pack is not None:     # rev 6: only one label line of floor is kept between rows
            strip_mm = max(lab_wh[j][1] for j in cur) + 0.6
        strip = (strip_mm * depth_at(r_std) / f_mm) / max(math.sin(el), 0.2)
        if prm.get("close", 0) >= 2:   # rev 8: rows sit straight behind each other
            return r_std
        # rev 8 (the user: rows close behind each other, overlap accepted): the step to the
        # next row shrinks by row_overlap of the two rows' half-footprints
        ov = prm.get("row_overlap", 0.0) * (depth + FOOT.get(k, D[k])) * 0.5
        return max(r_std, r + depth * 0.5 + strip + FOOT.get(k, D[k]) * 0.5) - ov

    if prm.get("rows_n"):
        # rev 5 (the user: a taller, more compact block): the rows are rows_n contiguous
        # size groups of (nearly) equal count -- labels set most seats, so equal counts give
        # rows of about equal printed length, and the row count sets the block's height
        groups = [[int(x) for x in g] for g in np.array_split(np.array(order), prm["rows_n"])]

        def radii(gs):
            rr, r_ = [], R0
            for gi, g in enumerate(gs):
                if gi:
                    r_ = next_radius(r_, gs[gi - 1], g[0])
                rr.append(r_)
            return rr

        def printed(g, r_):           # a packed row's printed width (page mm)
            return sum(slot(k, r_) for k in g) * f_mm / depth_at(r_)

        if prm.get("balance_rows"):
            # rev 7 (the user: more objects in front, no empty lower corners): rows are
            # filled front to back, smallest first, each until it prints W wide; W is the
            # smallest width that seats everyone in rows_n rows (bisection).  The front rows
            # then hold more, smaller objects and the crowd is a rectangle, not a funnel.
            def fill(W):
                gs, r_, cur_ = [], R0, []
                for k in order:
                    if cur_ and printed(cur_ + [k], r_) > W and len(gs) < prm["rows_n"] - 1:
                        gs.append(cur_)
                        r_ = next_radius(r_, cur_, k)
                        cur_ = []
                    cur_.append(k)
                gs.append(cur_)
                return gs, r_
            lo, hi = 1.0, 2000.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                gs, r_last = fill(mid)
                if printed(gs[-1], r_last) > mid:
                    lo = mid             # the last row overflows: rows must be wider
                else:
                    hi = mid
            groups, _ = fill(hi)
        rows = [[r_, g] for r_, g in zip(radii(groups), groups)]
    else:
        rows = None
    rows_frame, cur, used, r = [], [], 0.0, R0
    for k in ([] if rows is not None else order):
        w = slot(k, r)
        cap = row_cap(r, max([D[j] for j in cur] + [D[k]]))
        if cur and used + w > cap:
            rows_frame.append([r, cur])
            r = next_radius(r, cur, k)
            cur, used = [], 0.0
        cur.append(k)
        used += w
    if rows is None:
        rows = rows_frame + [[r, cur]]
    # rev 3: a short last row leaves empty sky; the largest objects of the row before move
    # back into it (size order kept) while it has fewer than the row before and still fits
    while not prm.get("rows_n") and len(rows) > 1 and len(rows[-1][1]) + 1 < len(rows[-2][1]):
        rl, last = rows[-1]
        cand = max(rows[-2][1], key=lambda j: (D[j], j))
        need = sum(slot(j, rl) for j in last + [cand])
        if need > row_cap(rl, max(D[j] for j in last + [cand])):
            break
        rows[-2][1].remove(cand)
        rows[-1][1].append(cand)
    # the in-row order, then the user's seat swaps (by name), then tiers and seats
    seqs = []
    for rk, idx in rows:
        big = sorted(idx, key=lambda k: (-D[k], k))
        centre_out = []
        for m, k in enumerate(big):
            centre_out.insert(0 if m % 2 else len(centre_out), k)
        seqs.append(centre_out)
    name = {o["instruction"]: k for k, o in enumerate(objs)}
    swapped = []
    for a_, b_ in swaps:
        ka, kb = name[a_], name[b_]
        pa = next((i, j) for i, sq in enumerate(seqs) for j, k in enumerate(sq) if k == ka)
        pb = next((i, j) for i, sq in enumerate(seqs) for j, k in enumerate(sq) if k == kb)
        seqs[pa[0]][pa[1]], seqs[pb[0]][pb[1]] = kb, ka
        swapped.append([a_, b_, list(pa), list(pb)])
    natural = [seat_positions(seq, rk) for (rk, _), seq in zip(rows, seqs)]
    widest_page = max((np.ptp(sa) if len(sa) > 1 else 0.0) * f_mm / depth_at(rk)
                      for (rk, _), sa in zip(rows, natural))
    seats, z, prev, out_rows = {}, 0.0, None, []
    for n, ((rk, _), seq) in enumerate(zip(rows, seqs)):
        Hk = max(H[k] for k in seq)
        if prev is not None:
            rp, zp, Hp = prev
            t = (rk - cam_y) / (rp - cam_y)
            z = max(z, eye_h + (zp + Hp - eye_h) * t - HIDE * Hk)
            # fix 1: the sight line over the row in front clears this row's label strip
            # (rev 8, the user: "the top row is too far away" -- with prm["close"] packed rows
            # keep only the body rule; labels whose floor is covered go above / beside)
            dk = max(D[k] for k in seq)
            # close = 1: the strip seen is one label line (the other lanes may hide behind
            # the row in front; those labels go above / beside); 2: no label strip at all
            tiers_done = prm.get("close", 0) >= 2
            lanes_seen = 1 if prm.get("close", 0) >= 1 else LANES
            if not tiers_done:
                hl = (lanes_seen * max(lab_wh[k][1] for k in seq) + LABEL_LANE_CLEAR) * depth_at(rk) / f_mm
                el = math.atan2(eye_h - z, rk - cam_y)
                strip = rk - LABEL_FRONT * dk - hl / max(math.sin(el), 0.2)
                t2 = (strip - cam_y) / (rp - cam_y)
                z = max(z, eye_h + (zp + Hp - eye_h) * t2)
        widths = np.array([slot(k, rk) for k in seq])
        s_arc = natural[n]
        rs = OVR.get("row_spacing", {}).get(str(n))
        if rs:                          # a named per-row override (placement_overrides.json)
            s_arc = s_arc * rs["factor"]
        if prm.get("rows_n") and len(seq) > 1 and prm.get("pack") is None:
            # rev 5: every row is spread evenly to the printed width of the widest row, so
            # all rows reach the same page edges (the widest row sets them; spreading to the
            # camera frame instead fed back into the FOV fit and ran away)
            target = widest_page * depth_at(rk) / f_mm
            ext = s_arc.max() - s_arc.min()
            if ext < target:
                s_arc = s_arc * (target / ext)
        if n % 2:
            s_arc = s_arc + stagger * widths.mean()
        rc = rk + arc_back * R0
        for i, (k, sa) in enumerate(zip(seq, s_arc)):
            g = sa / rc
            seats[k] = {"x": rc * math.sin(g), "y": -arc_back * R0 + rc * math.cos(g), "z": z,
                        # the label turn follows the seat's place along its row: 0 at the
                        # centre, the full turn at the row's ends (span/2 maps to the end)
                        "a": float(sa / max(abs(s_arc).max(), 1e-9) * span / 2) if ROW_SPANS_FRAME
                        else float(sa / rk), "g": float(g), "row": n, "r": rk, "lane": i % LANES}
        for k in seq:                   # named per-object overrides (placement_overrides.json):
            nd = OVR.get("seat_nudge", {}).get(objs[k]["instruction"])   # + radial = away from
            if nd:                                                        # the viewer, - z = lower
                st_ = seats[k]
                st_["x"] += nd.get("radial", 0.0) * math.sin(st_["g"])
                st_["y"] += nd.get("radial", 0.0) * math.cos(st_["g"])
                st_["z"] += nd.get("z", 0.0)
                st_["override"] = nd
        out_rows.append({"r": rk, "z": z, "objects": list(seq), "tallest": Hk,
                         "seat_arc_mm": s_arc.tolist()})
        prev = (rk, z, Hk)
    return seats, out_rows, swapped


def placed_points(o, D_k, seat, eye_xy, both, sub=None):
    """The object's vertices (schematic coords) at its seat, turned toward the eye."""
    P = both if sub is None else both[sub]
    s = D_k / o["L_m"]
    yaw = math.atan2(eye_xy[0] - seat["x"], eye_xy[1] - seat["y"])   # direction to the eye
    # local frame (gl_prep): x right, y up, z = the shown side; to schematic: (x, -z, y)
    # turned so local +z points along (sin yaw, cos yaw) in the schematic's x-y plane
    # local (x, up, front) is right-handed; front -> d = (sin yaw, cos yaw), up -> +z, so
    # x -> up x d = (-cos yaw, sin yaw): not a mirror image
    lx, ly, lz = P[:, 0] * s, P[:, 1] * s, P[:, 2] * s
    c, si = math.cos(yaw), math.sin(yaw)
    X = seat["x"] - lx * c + lz * si
    Y = seat["y"] + lx * si + lz * c
    Z = seat["z"] + ly
    return np.column_stack([X, Y, Z]), yaw


def fit_camera(points, eye, look0, f0):
    """Re-aim so the projected box is centred, then fit the focal length to the page."""
    cam = Camera(eye, look0, f0)
    for _ in range(30):
        n, _ = cam.norm(points)
        cx = 0.5 * (n[:, 0].max() + n[:, 0].min())
        cy = 0.5 * (n[:, 1].max() + n[:, 1].min())
        if abs(cx) < 1e-5 and abs(cy) < 1e-5:
            break
        # rotate the look a little toward the box centre
        look = cam.l + math.tan(math.atan(cx)) * cam.r + math.tan(math.atan(cy)) * cam.u
        cam.set_look(look)
    n, _ = cam.norm(points)
    wx, wy = np.ptp(n[:, 0]), np.ptp(n[:, 1])
    cam.f = min(FILL * PAGE_W / wx, FILL * PAGE_H / wy)
    return cam


def label_corners(anchor, w, h, turn):
    """A label's page rectangle: top-centre at anchor, rotated by turn (deg, CCW)."""
    loc = np.array([[-w / 2, 0], [w / 2, 0], [w / 2, -h], [-w / 2, -h]])
    t = math.radians(turn)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return loc @ R.T + anchor


def write_vtk(path, V, F, point_scalar=None, cell_id=None):
    V = np.asarray(V, ">f4")
    F = np.asarray(F, np.int64)
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\ngallery\nBINARY\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(V)} float\n".encode())
        f.write(V.tobytes())
        f.write(f"\nPOLYGONS {len(F)} {4 * len(F)}\n".encode())
        f.write(np.column_stack([np.full(len(F), 3), F]).astype(">i4").tobytes())
        if point_scalar is not None:
            f.write(f"\nPOINT_DATA {len(V)}\nSCALARS u_rel float 1\nLOOKUP_TABLE default\n".encode())
            f.write(np.asarray(point_scalar, ">f4").tobytes())
        if cell_id is not None:
            ids = np.asarray(cell_id, np.int64)
            f.write(f"\nCELL_DATA {len(F)}\nSCALARS objid int 1\nLOOKUP_TABLE default\n".encode())
            f.write(ids.astype(">i4").tobytes())
            f.write(b"\nCOLOR_SCALARS idrgb 3\n")
            f.write(np.column_stack([ids, np.full(len(ids), 255), np.zeros(len(ids))]).astype(np.uint8).tobytes())
        f.write(b"\n")


def to_world(P):
    """Schematic (x, y away, z up) -> VTK world (x, y up, z toward the eye)."""
    P = np.atleast_2d(P)
    return np.column_stack([P[:, 0], P[:, 2], -P[:, 1]])


def max_empty_rect(occ):
    """Largest all-False axis-aligned rectangle (histogram method): cells."""
    h = np.zeros(occ.shape[1], int)
    best = 0
    for row in occ:
        h = np.where(row, 0, h + 1)
        st = []
        for i in range(len(h) + 1):
            cur = h[i] if i < len(h) else 0
            start = i
            while st and st[-1][1] >= cur:
                start, hh = st.pop()
                best = max(best, hh * (i - start))
            st.append((start, cur))
    return best


def build(prm, objs, data, D, H, subs, swaps=(), widen=None):
    """Layout + camera fit, iterated until the label-bound seat widths are consistent."""
    n = len(objs)
    eye = np.array([0.0, -CAM_BACK * R0, prm["eye_h"]])
    span = math.radians(prm["span_deg"])
    f_mm = (PAGE_W / 2) / math.tan(math.radians(54.2) / 2)
    look_prev, hist = None, []
    for it in range(8):
        seats, rows, swapped = layout(objs, D, H, f_mm, prm, look_prev, swaps, widen)
        far = max(r["r"] for r in rows)
        zmid = rows[len(rows) // 2]["z"]
        look0 = np.array([0.0, far * 0.5, zmid + 3.0]) - eye
        pts = np.vstack([placed_points(objs[k], D[k], seats[k], eye[:2], data[k]["both"], subs[k])[0]
                         for k in range(n)])
        cam = fit_camera(pts, eye, look0, f_mm)
        lab_pts = []
        for k in range(n):
            s_ = seats[k]
            an, _ = cam.page([s_["x"] - LABEL_FRONT * D[k] * math.sin(s_["g"]),
                              s_["y"] - LABEL_FRONT * D[k] * math.cos(s_["g"]), s_["z"]])
            w, h = label_size_mm(label_text(objs[k]["instruction"]))
            lab_pts.append(label_corners(an[0], w, h, turn_deg(s_["a"], span)))
        lab_pts = np.vstack(lab_pts)
        n_obj, _ = cam.norm(pts)
        n_lab = (lab_pts - [PAGE_W / 2, PAGE_H / 2]) / cam.f
        allp = np.vstack([n_obj, n_lab])
        cx, cy = 0.5 * (allp[:, 0].max() + allp[:, 0].min()), 0.5 * (allp[:, 1].max() + allp[:, 1].min())
        cam.set_look(cam.l + cx * cam.r + cy * cam.u)
        n_obj, _ = cam.norm(pts)
        allp = np.vstack([n_obj, n_lab - [cx, cy]])
        f_new = min(FILL * PAGE_W / np.ptp(allp[:, 0]), FILL * PAGE_H / np.ptp(allp[:, 1]))
        cam.f = f_new
        look_prev = cam.l.copy()
        hist.append({"iter": it, "rows": [len(r["objects"]) for r in rows], "fov_h_deg": cam.fov_h()})
        if it > 0 and abs(f_new - f_mm) / f_mm < 0.002:
            break
        f_mm = f_new
    return {"seats": seats, "rows": rows, "cam": cam, "eye": eye, "swapped": swapped, "iterations": hist}


def cast_masks(b, objs, data, D, mm, which=("Vd", "V")):
    """Per object: its silhouettes (deformed, undeformed) Warp-cast from the eye on the page
    grid (rows top -> bottom)."""
    cam, eye = b["cam"], b["eye"]
    shape = (int(round(PAGE_H / mm)), int(round(PAGE_W / mm)))
    out = {}
    for k in range(len(objs)):
        masks = []
        for w in which:
            Pw, _ = placed_points(objs[k], D[k], b["seats"][k], eye[:2], data[k][w])
            pg, _ = cam.page(Pw)
            c0, c1 = max(int(pg[:, 0].min() / mm) - 2, 0), min(int(pg[:, 0].max() / mm) + 3, shape[1])
            r0 = max(int((PAGE_H - pg[:, 1].max()) / mm) - 2, 0)
            r1 = min(int((PAGE_H - pg[:, 1].min()) / mm) + 3, shape[0])
            m = np.zeros(shape, bool)
            if c1 > c0 and r1 > r0:
                cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
                xn = ((cc + 0.5) * mm - PAGE_W / 2) / cam.f
                yn = ((PAGE_H - (rr + 0.5) * mm) - PAGE_H / 2) / cam.f
                dirs = cam.l[None] + xn.reshape(-1, 1) * cam.r[None] + yn.reshape(-1, 1) * cam.u[None]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                hit = ar_prep.cast_faces(Pw, data[k]["F"], eye, dirs, 1e6) >= 0
                m[r0:r1, c0:c1] = hit.reshape(cc.shape)
            masks.append(m)
        out[k] = masks
    return out, shape


def evaluate(b, objs, data, D, prm, mm=SEARCH_RES):
    """Visible ink by seat depth, the labels by am_labels' hard rules, page coverage."""
    n = len(objs)
    masks, shape = cast_masks(b, objs, data, D, mm)
    cam = b["cam"]
    depth = {k: float(cam.norm([b["seats"][k]["x"], b["seats"][k]["y"], b["seats"][k]["z"]])[1][0])
             for k in range(n)}
    ids = np.zeros(shape, np.int16)
    for k in sorted(range(n), key=lambda k: -depth[k]):          # far first, near paints over
        ids[masks[k][0] | masks[k][1]] = k + 1
    vis = {k: float(((ids == k + 1) & masks[k][0]).sum() / max(masks[k][0].sum(), 1)) for k in range(n)}
    ink = {k: ids == k + 1 for k in range(n)}
    last = len(b["rows"]) - 1
    span = math.radians(prm["span_deg"])
    items = [{"k": k, "text": label_text(objs[k]["instruction"]), "depth": depth[k],
              "turn": turn_deg(b["seats"][k]["a"], span), "back": b["seats"][k]["row"] == last,
              "lane": b["seats"][k]["lane"]}
             for k in range(n)]
    labs, failed = am_labels.place(items, ink, mm, PAGE_W, PAGE_H, _FP, LABEL_PT * PT_MM, SEARCH_SAFETY)
    occ = ids > 0
    for L_ in labs.values():
        occ |= L_["mask"]
    f = max(1, int(round(CELL / mm)))
    gh, gw = shape[0] // f, shape[1] // f
    cells = occ[:gh * f, :gw * f].reshape(gh, f, gw, f).any((1, 3))
    cover = float(cells.mean())
    thirds = [float(c.mean()) for c in np.array_split(cells, 3, axis=1)]
    empty = max_empty_rect(cells) * CELL ** 2
    return {"cover": cover, "thirds": thirds, "largest_empty_mm2": empty, "largest_empty_share": empty / (PAGE_W * PAGE_H),
            "failed": [objs[k]["instruction"] for k in failed], "min_visible": min(vis.values()),
            "vis": vis, "labels": labs, "depth": depth}


REPAIR_ROUNDS = 5
REPAIR_WIDEN = 1.15         # a failed label's seat grows by this per repair round
REPAIR_PAD_MM = 2.0         # packed rows (rev 6): ... or by this much printed row length
_W = {}


def load_objects(work):
    meta = json.load(open(os.path.join(work, "gl_objects.json")))["objects"]
    objs, data = [], []
    for m in meta:
        z = np.load(os.path.join(work, f"obj_{m['k']}.npz"))
        V, u, fac = z["V"], z["u"], float(z["factor"])
        Vd = V + fac * u
        both = np.vstack([V, Vd])
        zc = 0.5 * (both[:, 2].min() + both[:, 2].max())      # centre the footprint on the seat
        V, Vd = V - [0, 0, zc], Vd - [0, 0, zc]
        objs.append(m)
        data.append({"V": V, "Vd": Vd, "F": z["F"], "u_rel": z["u_rel"], "both": np.vstack([V, Vd])})
    L = np.array([o["L_m"] for o in objs])
    D = D0 * (L / L.min()) ** P
    H = np.array([np.ptp(d["both"][:, 1]) / o["L_m"] * D[k] for k, (o, d) in enumerate(zip(objs, data))])
    for k, (o, d) in enumerate(zip(objs, data)):
        FOOT[k] = max(np.ptp(d["both"][:, 0]), np.ptp(d["both"][:, 2])) / o["L_m"] * D[k]
    rng = np.random.default_rng(0)
    subs = [rng.choice(len(d["both"]), min(3000, len(d["both"])), replace=False) for d in data]
    return objs, data, L, D, H, subs


def _init_worker(work):
    _W["d"] = load_objects(work)


def _run_candidate(prm):
    """One grid point: layout, evaluate; a label with no legal position widens its own seat
    (x REPAIR_WIDEN, up to REPAIR_ROUNDS rounds: objects move apart by rule)."""
    objs, data, L, D, H, subs = _W["d"]
    global PAGE_H
    PAGE_H = prm.get("page_h", PAGE_H)
    widen = {}
    for rnd in range(REPAIR_ROUNDS + 1):
        b = build(prm, objs, data, D, H, subs, SWAPS, widen)
        ev = evaluate(b, objs, data, D, prm)
        if not ev["failed"]:
            break
        if rnd < REPAIR_ROUNDS:
            name = {o["instruction"]: k for k, o in enumerate(objs)}
            for f in ev["failed"]:
                widen[name[f]] = widen.get(name[f], 1.0) * REPAIR_WIDEN
    ok = not ev["failed"] and ev["min_visible"] >= VIS_MIN
    return {**prm, "rows": [len(r["objects"]) for r in b["rows"]], "fov_h_deg": b["cam"].fov_h(),
            "cover": ev["cover"], "thirds": ev["thirds"], "largest_empty_share": ev["largest_empty_share"],
            "failed_labels": ev["failed"], "min_visible": ev["min_visible"], "ok": ok, "repairs": rnd,
            "widen": {str(k): v for k, v in widen.items()}}


def main():
    global PAGE_H
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    ap.add_argument("--refine", action="store_true",
                    help="no search: keep am_scene.json's parameters and widen the seats the "
                         "compose's labels failed on (label_failed.json) by REPAIR_WIDEN")
    args = ap.parse_args()
    t0 = time.time()
    objs, data, L, D, H, subs = load_objects(args.work)
    n = len(objs)
    names = {o["instruction"] for o in objs}
    for a_, b_ in SWAPS:
        assert a_ in names and b_ in names, f"GL_SEAT_SWAPS names an unknown object: {a_!r} / {b_!r}"
    if OVR and not args.refine:
        name = {o["instruction"]: k for k, o in enumerate(objs)}
        prm = dict(OVR["params"])
        widen = {name[nm]: v for nm, v in OVR.get("widen", {}).items()}
        table = []
        pick = {"widen": {str(k_): v for k_, v in widen.items()}}
        refined = []
        print(f"overrides: {os.environ['GL_OVERRIDES']} (no search)")
    elif args.refine:
        # the compose's finer raster failed some labels: the repair rule, applied once more
        prev = json.load(open(os.path.join(args.work, "am_scene.json")))
        failed = json.load(open(os.path.join(args.work, "label_failed.json")))
        name = {o["instruction"]: k for k, o in enumerate(objs)}
        widen = {name[nm]: v for nm, v in (prev.get("widen") or {}).items()}
        for nm in failed:
            widen[name[nm]] = widen.get(name[nm], 1.0) * REPAIR_WIDEN
        prm = prev["params"]
        table = prev["search"]
        pick = {"widen": {str(k_): v for k_, v in widen.items()}}
        refined = (prev.get("refined") or []) + [failed]
        print(f"refine: widening {failed}")
    else:
        refined = []
    # ---- the parameter search (in parallel; each candidate repairs its own labels) ------------
    prms = [] if (args.refine or OVR) else [{"arc_back": ab, "eye_h": eh, "span_deg": sp, "stagger": st, "row_gap": rg,
                                    "page_h": ph, "rows_n": rn, "pack": pk, "balance_rows": True, "close": 1,
                                    "row_overlap": ro}
            for ab in GRID["arc_back"] for eh in GRID["eye_h"] for sp in GRID["span_deg"]
            for st in GRID["stagger"] for rg in GRID["row_gap"] for ph in GRID["page_h"]
            for rn in GRID["rows_n"] for pk in GRID["pack"] for ro in GRID["row_overlap"]]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    if prms:
        with ctx.Pool(int(os.environ.get("GL_SEARCH_PROCS", "6")), initializer=_init_worker,
                      initargs=(args.work,)) as pool:
            table = pool.map(_run_candidate, prms)
    for row in ([] if (args.refine or OVR) else table):
        print(f"overlap {row.get('row_overlap')} rows_n {row['rows_n']} pack {row['pack']} page_h {row['page_h']:g} arc_back {row['arc_back']} eye {row['eye_h']:g} row_gap {row['row_gap']:g} span {row['span_deg']:g} stagger "
              f"{row['stagger']}: rows {row['rows']} FOV {row['fov_h_deg']:.1f} cover {row['cover']:.1%} "
              f"thirds {'/'.join(f'{t:.0%}' for t in row['thirds'])} "
              f"empty {row['largest_empty_share']:.1%} vis {row['min_visible']:.2f} repairs {row['repairs']} "
              f"failed {row['failed_labels']} {'OK' if row['ok'] else ''}")
    if not args.refine and not OVR:
        good = [r for r in table if r["ok"]]
        assert good, "no parameter set passes the label rules after repair: see the table above"
        # rev 4 (the user: no empty sides): the least-covered page third first, then the whole
        pick = max(good, key=lambda r: (min(r["thirds"]), r["cover"], -r["largest_empty_share"]))
        prm = {k_: pick[k_] for k_ in ("arc_back", "eye_h", "span_deg", "stagger", "row_gap", "page_h", "rows_n",
                                       "pack", "balance_rows", "close", "row_overlap")}
    widen = {int(k_): v for k_, v in pick["widen"].items()}
    PAGE_H = prm.get("page_h", PAGE_H)
    b = build(prm, objs, data, D, H, subs, SWAPS, widen)
    ev = evaluate(b, objs, data, D, prm)
    best = ((not ev["failed"]), prm, b, ev)
    ok_, prm, b, ev = best
    assert ok_ or args.refine or OVR, f"the chosen layout's labels fail on rebuild: {ev['failed']}"
    if not ok_:
        print(f"refine: the search raster now fails {ev['failed']}; the compose decides")
    print(f"chosen {prm}: cover {ev['cover']:.1%}, largest empty {ev['largest_empty_share']:.1%} "
          f"({time.time() - t0:.0f} s)")
    # ---- outputs ----------------------------------------------------------------------------------
    cam, eye, seats, rows = b["cam"], b["eye"], b["seats"], b["rows"]
    span = math.radians(prm["span_deg"])
    Vd_all, Vu_all, F_all, u_all, id_all, rec = [], [], [], [], [], []
    off = 0
    for k in range(n):
        s_ = seats[k]
        Pd, yaw = placed_points(objs[k], D[k], s_, eye[:2], data[k]["Vd"])
        Pu, _ = placed_points(objs[k], D[k], s_, eye[:2], data[k]["V"])
        F = data[k]["F"].astype(np.int64)
        Vd_all.append(to_world(Pd))
        Vu_all.append(to_world(Pu))
        F_all.append(F + off)
        u_all.append(data[k]["u_rel"])
        id_all.append(np.full(len(F), k + 1))
        off += len(Pd)
        rec.append({"k": k, "instruction": objs[k]["instruction"], "category": objs[k]["category"],
                    "L_m": objs[k]["L_m"], "D": float(D[k]), "height": float(H[k]), "row": s_["row"],
                    "seat": s_, "yaw_deg": math.degrees(yaw), "factor": objs[k]["factor"],
                    "heading": objs[k]["heading"], "depth": ev["depth"][k],
                    "label": {"text": label_text(objs[k]["instruction"]), "turn_deg": turn_deg(s_["a"], span),
                              "pt": LABEL_PT, "search_kind": (ev["labels"].get(k) or {}).get("kind")}})
    np.savez(os.path.join(args.work, "scene_world.npz"),
             **{f"V{k}": Vd_all[k] for k in range(n)}, **{f"F{k}": data[k]["F"] for k in range(n)})
    write_vtk(os.path.join(args.work, "scene_def.vtk"), np.vstack(Vd_all), np.vstack(F_all),
              np.concatenate(u_all), np.concatenate(id_all))
    write_vtk(os.path.join(args.work, "scene_und.vtk"), np.vstack(Vu_all), np.vstack(F_all), None,
              np.concatenate(id_all))
    for old in ("scene_shadow.vtk",):
        if os.path.exists(os.path.join(args.work, old)):
            os.remove(os.path.join(args.work, old))
    E, Lw, Uw = to_world(cam.eye[None])[0], to_world(cam.l[None])[0], to_world(cam.u[None])[0]
    camera = {"parallel": False, "position": E.tolist(), "focal": (E + 50.0 * Lw).tolist(),
              "view_up": Uw.tolist(), "view_angle_deg": 2 * math.degrees(math.atan(PAGE_H / 2 / cam.f)),
              "fov_h_deg": cam.fov_h(), "f_mm": cam.f, "page_mm": [PAGE_W, PAGE_H], "parallel_scale": 1.0,
              "eye_schematic": cam.eye.tolist(), "look_schematic": cam.l.tolist(),
              "right_schematic": cam.r.tolist(), "up_schematic": cam.u.tolist()}
    swapped_names = {x for pair in SWAPS for x in pair}
    srt = np.argsort(L)
    strict = all(D[srt[i + 1]] > D[srt[i]] for i in range(n - 1) if L[srt[i + 1]] > L[srt[i]])
    row_order = all(min(L[k] for k in rows[i]["objects"] if objs[k]["instruction"] not in swapped_names) >=
                    max(L[k] for k in rows[i - 1]["objects"] if objs[k]["instruction"] not in swapped_names)
                    for i in range(1, len(rows)))
    assert strict and row_order, "size order broken"
    scene = {"p": P, "D0": D0, "D_range": [float(D.min()), float(D.max())],
             "ratio_drawn": float(D.max() / D.min()), "ratio_real": float(L.max() / L.min()),
             "params": prm, "widen": {objs[int(k_)]["instruction"]: v for k_, v in pick["widen"].items()},
             "search": table, "refined": refined, "overrides": OVR, "rows": rows, "camera": camera, "objects": rec,
             "iterations": b["iterations"], "seat_swaps": {"requested": SWAPS, "applied": b["swapped"],
                                                           "why": "the user's choice, 2026-10-17"},
             "search_eval": {"cover": ev["cover"], "thirds": ev["thirds"], "largest_empty_share": ev["largest_empty_share"],
                             "min_visible": ev["min_visible"]},
             "rules": {"R0": R0, "gap": GAP, "gap_label": GAP_LABEL, "row_gap": ROW_GAP, "cam_back_R0": CAM_BACK,
                       "hide": HIDE, "turn_max_deg": TURN_MAX, "label_front_D": LABEL_FRONT, "label_pt": LABEL_PT,
                       "fill": FILL, "shorten_with": SHORTEN_WITH, "grid": GRID, "cell_mm": CELL,
                       "search_res_mm": SEARCH_RES, "vis_min": VIS_MIN},
             "size_order": {"D_strictly_increasing": strict, "rows_in_size_order_except_swaps": row_order}}
    json.dump(scene, open(os.path.join(args.work, "am_scene.json"), "w"), indent=1, default=float)
    print(f"wrote am_scene.json: rows {[len(r['objects']) for r in rows]}, FOV {cam.fov_h():.1f} deg, "
          f"tiers {[round(r['z'], 2) for r in rows]}, swaps {b['swapped']}")


if __name__ == "__main__":
    main()
