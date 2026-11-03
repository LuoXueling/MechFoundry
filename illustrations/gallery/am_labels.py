"""The gallery's label rule, shared by the layout search (am_layout, Warp silhouettes at
0.25 mm) and the final compose (am_figure, the render's id passes at 0.085 mm).

Every label is placed nearest object first.  Candidates, in this order:
  below   top-centre ANCHOR_GAP under the object's silhouette, centred under its lowest
          stretch, in its seat's lane (odd seats one label height + LABEL_CLEAR further
          toward the viewer: rev 3, so neighbours' labels need not share row length), then
          nudged toward the viewer (0-3 mm) and along its own baseline (to +-4.5 mm),
          smallest offsets first;
  side    beside the base (its near end ANCHOR_GAP from the silhouette, at mid-height of the
          lowest 30 % of the silhouette), the side away from the page centre first;
  above   (back row only, last resort) ANCHOR_GAP above the silhouette's top.
A candidate passes only if ALL hold, else the next:
  * inside the page by >= 1 mm;
  * >= LABEL_CLEAR from every label already placed;
  * off its own object's ink (OBJ_CLEAR around it);
  * at most 30 % of it covered by NEARER objects (rev 4, the user: a nearer body may cover
    part of a farther label; a farther body under a label is behind it) -- the first
    uncovered passing candidate is taken, else the least covered one;
  * owned: its centroid nearer its own object than any other by >= OWN_MARGIN, and its
    nearest glyph <= OWN_MAX from its own object.
No candidate -> the label is reported by name (the search rejects the layout; the final
build fails).
"""

import math

import numpy as np
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.draw import polygon as fill_polygon

LABEL_CLEAR = 0.5     # mm between labels
OBJ_CLEAR = 0.3       # mm between a label and any object's ink
OWN_MARGIN = 1.0      # mm: own object nearer the label centroid than any other by this
OWN_MAX = 3.0         # mm: nearest glyph to its own object at most this
OWN_PREFER = 1.5      # mm: rev 7, the preferred ownership margin (1.0 stays the hard floor)
AMBIG = {}            # the last call's chosen labels: faults left (over another object, margin)
ABOVE_ALL = True      # rev 6: the "above" fallback for every row, not only the back one
LANES_N = 4           # rev 6: label lanes below a row (seats use am_layout.LANES = 3; a 4th
#                       lane is a fallback for packed rows, where labels outrun their seats)
LABEL_VIS_MIN = 0.7   # rev 4 (the user): a label may be partly behind a nearer body, >= 70 % seen
ANCHOR_GAP = 0.6      # mm
REASONS = {}          # the last call's failures: rejected candidates per rule
BELOW = sorted({(d, s) for d in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
                for s in (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.5, 4.5, -6.0, 6.0, -8.0, 8.0)},
               key=lambda ds: (math.hypot(*ds), abs(ds[1]), ds[1]))
SIDE = [(d, s) for s in (0.0, 0.5, 1.0) for d in (0.0, -0.5, 0.5, -1.0, 1.0)]


_TP = {}                # text -> its TextPath and extents (built once: TextPath is slow)
_POLY = {}              # (text, codes id) -> closed loops of the unit TextPath


def glyphs(text, font, pt_mm, anchor, turn_deg, ref="top"):
    """Glyph outlines in page mm (y up), turned by turn_deg, the reference point of the
    unturned text box (top-centre, left-middle, right-middle or bottom-centre) at anchor."""
    key = (text, id(font))
    if key not in _TP:
        p_ = TextPath((0, 0), text, size=1, prop=font)
        _TP[key] = (p_, p_.get_extents(), [(x, None) for x in ()])
    p, e, _ = _TP[key]
    ox = {"top": (e.x0 + e.x1) / 2, "bottom": (e.x0 + e.x1) / 2, "left": e.x0, "right": e.x1}[ref]
    oy = {"top": e.y1, "bottom": e.y0, "left": (e.y0 + e.y1) / 2, "right": (e.y0 + e.y1) / 2}[ref]
    v = (p.vertices - [ox, oy]) * pt_mm
    t = math.radians(turn_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return v @ R.T + anchor, p.codes


def raster(v, codes, shape, mm, page_h):
    m = np.zeros(shape, bool)
    for loop in Path(v, codes).to_polygons(closed_only=True):
        rr, cc = fill_polygon((page_h - loop[:, 1]) / mm - 0.5, loop[:, 0] / mm - 0.5, shape)
        m[rr, cc] = True
    return m


def glyph_loops(text, font, pt_mm, anchor, turn_deg, ref="top"):
    """The glyphs' closed loops (flattened once per text, then moved by the same affine map
    as glyphs()): what raster_crop fills."""
    key = (text, id(font))
    if key not in _POLY:
        p_ = TextPath((0, 0), text, size=1, prop=font)
        _POLY[key] = p_.to_polygons(closed_only=True)
    glyphs(text, font, pt_mm, anchor, turn_deg, ref)          # fills _TP
    p, e, _ = _TP[key]
    ox = {"top": (e.x0 + e.x1) / 2, "bottom": (e.x0 + e.x1) / 2, "left": e.x0, "right": e.x1}[ref]
    oy = {"top": e.y1, "bottom": e.y0, "left": (e.y0 + e.y1) / 2, "right": (e.y0 + e.y1) / 2}[ref]
    t = math.radians(turn_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return [((lp - [ox, oy]) * pt_mm) @ R.T + anchor for lp in _POLY[key]]


def raster_crop(v, codes, shape, mm, page_h, loops=None):
    """The filled glyphs on a local window of the page grid: (mask, (row slice, col slice))."""
    c0 = max(int(math.floor(v[:, 0].min() / mm)) - 1, 0)
    c1 = min(int(math.ceil(v[:, 0].max() / mm)) + 2, shape[1])
    r0 = max(int(math.floor((page_h - v[:, 1].max()) / mm)) - 1, 0)
    r1 = min(int(math.ceil((page_h - v[:, 1].min()) / mm)) + 2, shape[0])
    m = np.zeros((max(r1 - r0, 1), max(c1 - c0, 1)), bool)
    for loop in (loops if loops is not None else Path(v, codes).to_polygons(closed_only=True)):
        rr, cc = fill_polygon((page_h - loop[:, 1]) / mm - 0.5 - r0, loop[:, 0] / mm - 0.5 - c0, m.shape)
        m[rr, cc] = True
    return m, (slice(r0, r1), slice(c0, c1))


def place(items, ink, mm, page_w, page_h, font, pt_mm, safety=0.0):
    """safety (mm) is added to every clearance and margin: the layout search runs on a
    coarser raster than the compose, and its passes must survive the finer one."""
    """items: [{k, text, turn, depth, back}], ink: {k: bool mask (rows top->bottom)}.
    Returns ({k: placement}, [failed instruction keys])."""
    shape = next(iter(ink.values())).shape
    keys = list(ink)
    dts = {k: distance_transform_edt(~ink[k]) * mm if ink[k].any() else np.full(shape, 1e9) for k in keys}
    union = np.zeros(shape, bool)
    for k in keys:
        union |= ink[k]
    r_obj = max(1, int(math.ceil((OBJ_CLEAR + safety) / mm)))
    r_lab = max(1, int(math.ceil((LABEL_CLEAR + safety) / mm)))
    labs_d = np.zeros(shape, bool)
    nearer = np.zeros(shape, bool)      # ink of the objects already processed = nearer ones
    out, failed = {}, []
    cx_page = page_w / 2
    for it in sorted(items, key=lambda i: i["depth"]):
        k = it["k"]
        rr, cc = np.nonzero(ink[k])
        if not len(rr):
            failed.append(k)
            nearer |= ink[k]
            continue
        x0, x1 = cc.min(), cc.max()
        cx = 0.5 * (x0 + x1)
        mid = np.abs(cc - cx) <= 0.25 * (x1 - x0) + 1
        bottom_y = page_h - (rr[mid].max() + 1) * mm            # page mm, y up
        top_y = page_h - rr.min() * mm
        base = rr >= rr.max() - 0.3 * (rr.max() - rr.min())
        side_y = page_h - (rr[base].mean() + 0.5) * mm
        xl, xr = cc[base].min() * mm, (cc[base].max() + 1) * mm
        t = math.radians(it["turn"])
        bdir = np.array([math.cos(t), math.sin(t)])
        down = np.array([math.sin(t), -math.cos(t)])
        # rev 3: two lanes along a row -- odd seats' labels sit one label height (+ clearance)
        # further toward the viewer, so neighbours' labels do not compete for row length
        glyphs(it["text"], font, pt_mm, np.zeros(2), 0.0)
        e = _TP[(it["text"], id(font))][1]
        step = (e.y1 - e.y0) * pt_mm + LABEL_CLEAR
        lane = it.get("lane", 0)
        cands = []
        for ln in [lane] + [x for x in range(LANES_N) if x != lane]:   # its own lane first
            cands += [("below", np.array([(cx + 0.5) * mm, bottom_y - ANCHOR_GAP]) + (ln * step + d) * down
                       + s * bdir, "top", ln * step + d, s) for d, s in BELOW]
        outer_right = (cx + 0.5) * mm >= cx_page
        for side in (("right", "left") if outer_right else ("left", "right")):
            for d, s in SIDE:
                if side == "right":
                    anc = np.array([xr + ANCHOR_GAP + s, side_y + d])
                    cands.append(("side-right", anc, "left", d, s))
                else:
                    anc = np.array([xl - ANCHOR_GAP - s, side_y + d])
                    cands.append(("side-left", anc, "right", d, s))
        if it.get("back") or ABOVE_ALL:
            # rev 6: above the object's top for every row -- it lies over the farther rows,
            # which a label in front of them may (the user: nearer may cover farther)
            for s in (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0):
                cands.append(("above", np.array([(cx + 0.5) * mm + s, top_y + ANCHOR_GAP]), "bottom", 0.0, s))
        chosen, best, ncand = None, None, 0
        others = union & ~ink[k]
        why = {"page": 0, "label": 0, "own_ink": 0, "covered": 0, "own_margin": 0, "own_far": 0}
        own_d = binary_dilation(ink[k], iterations=r_obj)
        for kind, anc, ref, d, s in cands:
            v, codes = glyphs(it["text"], font, pt_mm, anc, it["turn"], ref)
            if v[:, 0].min() < 1.0 or v[:, 0].max() > page_w - 1.0 or v[:, 1].min() < 1.0 or \
                    v[:, 1].max() > page_h - 1.0:
                why["page"] += 1
                continue
            m, sl = raster_crop(v, codes, shape, mm, page_h,
                                glyph_loops(it["text"], font, pt_mm, anc, it["turn"], ref))
            if not m.any() or (m & labs_d[sl]).any():
                why["label"] += 1
                continue
            if (m & own_d[sl]).any():
                why["own_ink"] += 1
                continue
            # rev 4 (the user): a NEARER body may cover part of a label (it stands in front
            # of it); a farther body under a label is simply behind it
            cover = float((m & nearer[sl]).sum() / m.sum())
            if cover > 1.0 - LABEL_VIS_MIN + 1e-9 - safety * 0.1:
                why["covered"] += 1
                continue
            cen = v.mean(0)
            ci = min(max(int(cen[0] / mm), 0), shape[1] - 1)
            ri = min(max(int((page_h - cen[1]) / mm), 0), shape[0] - 1)
            d_own = dts[k][ri, ci]
            d_oth = min(dts[j][ri, ci] for j in keys if j != k)
            if d_own + OWN_MARGIN + safety > d_oth:
                why["own_margin"] += 1
                continue
            near = float(dts[k][sl][m].min())
            if near > OWN_MAX:
                why["own_far"] += 1
                continue
            over = float((m & others[sl]).sum() / m.sum())     # on a farther object's ink
            margin = d_oth - d_own
            cand = {"kind": kind, "anchor": anc.tolist(), "ref": ref, "nudge": [d, s], "v": v,
                    "codes": codes, "mask": (m, sl), "own_centroid_mm": float(d_own),
                    "other_centroid_mm": float(d_oth), "own_near_mm": near, "covered": cover,
                    "over_other": over, "margin": float(margin)}
            # rev 7 (the user: no ambiguous labels): prefer, in this order, a label nothing
            # covers, that lies on no other object, and whose own object is nearer than any
            # other by OWN_PREFER (1.5 mm); then the fewest such faults, smallest offset first
            key = (cover > 0, over > 0.02, margin < OWN_PREFER, cover + over, -min(margin, OWN_PREFER), ncand)
            ncand += 1
            if best is None or key < best[0]:
                best = (key, cand)
            if key[:3] == (False, False, False):
                break                          # the first faultless candidate
        chosen = best[1] if best else None
        if chosen is not None:
            AMBIG[k] = {"over_other": chosen["over_other"], "margin": chosen["margin"],
                        "covered": chosen["covered"]}
        nearer |= ink[k]
        if chosen is None:
            failed.append(k)
            REASONS[k] = why
            continue
        m, sl = chosen["mask"]
        full = np.zeros(shape, bool)
        full[sl] = m
        chosen["mask"] = full
        out[k] = chosen
        labs_d |= binary_dilation(full, iterations=r_lab)
    return out, failed
