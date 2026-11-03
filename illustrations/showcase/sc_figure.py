"""Stage 3 (env mechfoundry-warp117): compose showcase.{pdf,png}.

    python sc_figure.py [WORK_DIR] [--out NAME] [--ground grid|none|shadow]

Three tall rows (shape / material / loading, sc_prep.py) and two columns: the
simulation INPUT of every item on the left, its DEFORMED SHAPE on the right.  Every
row is a grid of tiles 3 across, up to 3 down (nine items), at the same x positions
in both cells (item k of the left cell is item k of the right cell), filled from
the top left.  Tiles are 0.85 as tall as wide (SC_ASPECT; the 1.0 they were left a
band of air over and under most bodies) with 0.5 mm between tile rows.  A row is as
many tile rows deep as its items need (ceil(n / 3)) and never shorter than its label
column; every tile is the same size in every row, so
a row of fewer items is a shorter row, not a sparse one.  A hairline runs down the
gap between the two cells.

The label column carries only what the row was asked: its title, then one block per
kind of Claude call that made the row -- a prompt card (quoted) and that call's cost
bars, agentic_results' three (input with its cached share, output tokens, the wall
clock split by where it went).
  shape      one kind: the N run agents, each given the same instruction (the card),
             their MEAN cost per run ("Mean of N runs"; every run is its own
             headless agent session, so a run is the unit of cost).
  material / loading
             two kinds (sc_prep.sweep_calls): first the ONE authoring call that wrote
             all N cases (the condensed authoring prompt and, with no head over them,
             its own cost bars, summed over its tries when it needed more than one;
             its wall time is all agent time, since no workflow step or solve runs
             inside it), then the N run agents (their condensed prompt.txt, the case
             stood in for in brackets; "Mean of N runs").
The sweep cards are short gists for illustration (sc_prep.condense_authoring /
condense_run_prompt, a few card lines each); they are re-condensed from their source
prompts at compose time (sc_prep.refresh_cards), so a compose-only build of an older
``_work`` prints the current cards.  The shape card is the instruction, whole.
Every cost bar of the figure shares one scale per metric (the largest block fills the
track).  A ``_work`` written before the record had ``calls`` gets them from the run
and batch files here (sc_prep.sweep_calls), so a compose-only build needs no prep.

The keys are agentic_results' (its key helpers, ar_figure.cost_keys / bc_keys /
e_key / u_keys, imported).  In the footer: the cost keys under the label column;
Support, Load and "Load on a far face, seen through" centred under the input cell;
the |u| ramp from 0 to max |u|, "×N: Displacement magnified N times" and "Undeformed
shape" centred under the deformed cell.  The Young's modulus key ("Log", decade
ticks, an italic E in front: this figure has no Young's-modulus column head to name
it) is not in the footer but at the bottom of the material row, centred under that
row's E tiles, at the footer keys' width (E_KEY_BAND), so it sits beside the only
panels it explains.  Its ramp is this figure's own (sc_prep.SC_E_ANCHORS, "cheese,
lime, watermelon, herb": one hue per material class, the blends between them only in
unused decades); the agentic-results figures keep the shared "steel" map.  For
colour-blind readers the classes are lightness steps, and E and |u| are told apart by
column, key and field shape, not by hue (sc_prep's map comment).

In the tiles, only agentic_results' own panel annotations: the generation path
on a generated body ('Text → TripoSG', shape row) and the displacement factor on
every deformed tile ('×30,000'), each in a free corner, in that figure's size
and grey.  The E map carries 0.25 pt dark seams along the session's segment
boundaries (an id pass rendered by render_ar).  The undeformed outline is a
darker grey than agentic_results' (#8a8a8a, apart from the light end of the |u|
ramp for colour-blind readers); this figure only.

The look is agentic_results', imported rather than re-made: its page (mm, y
down), rules and hairlines, column-head tags with the light head arrow, the
prompt card, the renders' crop/ground/outline compositing, the BC see-through
layers, the keys, the number formats.  Everything printed is read from
``_work/ar_stats.json`` (sc_prep.py).  A row flagged ``placeholder`` carries a red
PLACEHOLDER tag in its label column, and the build lists it.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
import ar_figure as F  # noqa: E402  (also sets the rcParams: Arial, Type 42)
import ar_style as A  # noqa: E402
import ar_time  # noqa: E402
import sc_prep  # noqa: E402  (sweep_calls: a _work written before the record had them)

INK, GREY = F.INK, F.GREY
PLACEHOLDER_RED = "#c44e52"     # the house palette's red: a build flag, never data
OUTLINE_RGB = (138, 138, 138)   # #8a8a8a: the undeformed outline, this figure only
SEAM_RGB = (51, 51, 51)         # #333333: part seams on the E map
SEAM_PT = 0.25
VRULE = "#cccccc"               # the hairline between the input and deformed cells

# -- grid (mm) ---------------------------------------------------------------------------
W, ML, MR = F.W, F.ML, F.MR
LAB_W = F.INS_W                  # the label column: agentic_results' instruction column
X_IN = ML + LAB_W + 2.0          # the input cell
MID = 7.0                        # between the input and deformed cells (hairline inside)
CW = (W - MR - X_IN - MID) / 2   # one cell's width
X_DEF = X_IN + CW + MID
COLS, GROWS = 3, 3               # at most a 3 x 3 tile grid per row (nine items)
MAX_ITEMS = COLS * GROWS
IGAP, IGAP_V = 1.0, 0.5          # between tiles, across / down
IW = (CW - (COLS - 1) * IGAP) / COLS
ASPECT = float(os.environ.get("SC_ASPECT", "0.85"))   # tile height / width, at least
ROW_PAD_T, ROW_PAD_B = 0.4, 0.2
# the material row's E key, under its tiles: tile bottom -> the bar's top (E_KEY_GAP),
# the bar and its tick labels (e_key: bar top = baseline - 1.6, labels 2.4 under the
# baseline) and their descenders; the row grows by E_KEY_BAND when its label column
# does not already leave that room
E_KEY_GAP = 1.2
E_KEY_BASE = E_KEY_GAP + 1.6                     # tile bottom -> the key's baseline
E_KEY_BAND = E_KEY_BASE + 2.4 + 0.8              # tile bottom -> the key's last ink
ROW_GAP = F.ROW_GAP
HEAD, HEAD_RULE, TY = F.HEAD, F.HEAD_RULE, F.TY
FOOT = F.FOOT                    # agentic_results' key band under the bottom rule
GROUND_FADE = 3.0                # mm, as agentic_results
GROUND_ALPHA = 0.65              # the floor grid at 65 % of agentic_results' opacity
TITLE_FS = 7.0
CARD_FS, CARD_PITCH = 5.6, 2.35  # the prompt card (long sweep prompts: smaller than F.PROMPT_FS)
CARD_PAD_X = 1.3
LABEL_BASE_T = F.LABEL_BASE - F.PANEL_PAD_TOP     # tile top -> top-corner label baseline
BOTTOM_BASE = F.BOTTOM_LABEL - 2.0                 # bottom-corner baseline above the tile bottom
TOP_RES = LABEL_BASE_T + 0.8                       # ink-free band for a top-corner label (mm)
BOT_RES = F.BOTTOM_LABEL - F.SH_ROOM + 0.4         # extra ink-free room under the ink for a bottom label

ROW_TITLE = {"shape": "Shapes", "material": "Materials", "loading": "Supports & loads"}
COLUMNS = (("AGENT", F.AGENT, "SIMULATION INPUT"), ("FEM SOLVER", F.GPU_GREEN, "DEFORMED SHAPE"))
FLOOR_MISSES = []


# -- renders -> one item's images ----------------------------------------------------------
def crop_box(boxes, P, H, s=None, top_res=0.0, bot_res=0.0):
    """F.row_crop's window (ink centred across, MARGIN on top and sides, SH_ROOM under
    the ink), at a given px/mm ``s`` when one is forced (a row of one body drawn at one
    scale) -- never tighter than the ink needs.  ``top_res`` / ``bot_res`` (mm) keep a
    band above / below the ink free for a corner label."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    pad = F.MARGIN * max(w, h)
    need = max((w + 2 * pad) / P, (h + pad) / (H - F.SH_ROOM - top_res - bot_res))
    s = max(need, s or 0.0)
    cx = 0.5 * (x0 + x1)
    left = cx - P * s / 2
    bottom = y1 + (F.SH_ROOM + bot_res) * s
    return left, bottom - H * s, left + P * s, bottom, s


def ground_under(work, i, p, crop, s):
    """The ground layer rendered alone (render_ar), lightened, faded at the drawn
    image's edges, as RGB over the white page; None without ground renders."""
    g = F.load_rgba(os.path.join(work, f"ar_{i}_{p}_g.png"))
    if g is None:
        return None
    gw = F.cut_rgba(g, *crop)
    hh, ww = gw.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    d = np.minimum.reduce([xx, ww - 1 - xx, yy, hh - 1 - yy]) / (GROUND_FADE * s)
    gw[..., 3] *= GROUND_ALPHA * np.clip(d, 0, 1) ** 1.5
    return F.over(gw, None)


def outline_under(base_rgb, undeformed_win, px_per_mm):
    """F.outline_under in this figure's darker grey."""
    from scipy.ndimage import binary_dilation
    m = undeformed_win[..., 3] > 128
    r = max(1, int(round(F.OUTLINE_W * px_per_mm)))
    edge = binary_dilation(m, iterations=r) & ~m
    out = base_rgb.astype(float)
    out[edge] = out[edge] * (1 - F.OUTLINE_ALPHA) + np.array(OUTLINE_RGB) * F.OUTLINE_ALPHA
    return out.astype(np.uint8)


def seams_over(rgb, idwin, px_per_mm):
    """Dark part seams where neighbouring pixels of the segment-id pass carry
    different segments (the page and silhouettes against it are not seams), SEAM_PT
    wide on paper, anti-aliased by distance."""
    from scipy.ndimage import distance_transform_edt
    r, g, b = idwin[..., 0], idwin[..., 1], idwin[..., 2]
    sid = np.where((g > 250) & (b < 5), np.rint(r).astype(int), -1)
    edge = np.zeros(sid.shape, bool)
    for a_, b_ in ((sid[:, 1:], sid[:, :-1]), (sid[1:, :], sid[:-1, :])):
        diff = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        if a_.shape[1] != sid.shape[1]:
            edge[:, 1:] |= diff
            edge[:, :-1] |= diff
        else:
            edge[1:, :] |= diff
            edge[:-1, :] |= diff
    if not edge.any():
        return rgb, 0
    w = SEAM_PT * F.PT_MM * px_per_mm            # line width in render px
    d = distance_transform_edt(~edge)
    a = np.clip(w / 2 - d, 0, 1)[..., None]     # the flagged pair straddles the boundary
    out = rgb.astype(float) * (1 - a) + np.array(SEAM_RGB) * a
    return np.clip(out, 0, 255).astype(np.uint8), int(edge.sum())


def item_images(work, rec, input_panel, ground, P, H, s_forced=None, res=(0.0, 0.0)):
    """({panel: (RGB, window RGBA)}, crop, px/mm) for one item, or None."""
    i = rec["index"]
    names = [input_panel, "solve"] + (["painted"] if input_panel == "bc" else [])
    imgs = {p: F.load_rgba(os.path.join(work, f"ar_{i}_{p}.png")) for p in names}
    if imgs[input_panel] is None:
        return None
    boxes = []
    for p in (input_panel, "solve"):
        im = imgs[p]
        if im is None:
            continue
        b = F.ink_box(im)
        assert b is not None, f"item {i} {p}: empty render"
        assert b[0] > 0 and b[1] > 0 and b[2] < im.shape[1] and b[3] < im.shape[0], \
            f"item {i} {p}: the object touches the render border (clipped)"
        boxes.append(b)
    x0, y0, x1, y1, s = crop_box(boxes, P, H, s_forced, *res)
    crop = (x0, y0, x1, y1)
    out = {}
    for p in (input_panel, "painted", "solve"):
        im = imgs.get(p)
        if im is None:
            continue
        win = F.cut_rgba(im, *crop)
        base = ground_under(work, i, p, crop, s) if ground != "none" else None
        if base is None:
            base = np.full(win.shape[:2] + (3,), 255, np.uint8)
        if p == "solve":
            sil = out["painted"][1] if "painted" in out else out[input_panel][1]
            base = outline_under(base, sil, s)
        rgb = F.over(win, base)
        if p == "bc":           # far-face loads, arrows and supports, seen through
            for name, op, col in (("bchidden", F.SEE_THROUGH, A.NEUMANN),
                                  ("bchidden_arrows", F.ARROW_THROUGH, A.LOAD_ARROW),
                                  ("bchidden_support", F.ARROW_THROUGH, A.DIRICHLET)):
                layer = F.load_rgba(os.path.join(work, f"ar_{i}_{name}.png"))
                if layer is None:
                    continue
                hw = F.cut_rgba(layer, *crop)[..., 3:4] / 255.0 * op
                rgb = np.clip(rgb * (1 - hw) + np.array(col) * 255 * hw, 0, 255).astype(np.uint8)
        if p == "materials":
            idp = F.load_rgba(os.path.join(work, f"ar_{i}_segid.png"))
            if idp is not None:
                rgb, npx = seams_over(rgb, F.cut_rgba(idp, *crop), s)
                rec.setdefault("showcase", {})["seam_px"] = npx
        out[p] = (rgb, win)
    return out, crop, s


def corner_clear(win, s, P, H, corner, label_w):
    """Is a corner label's box (its width, CORNER_PAD and CORNER_CLEAR of air) free
    of object ink?  (pick_corner's test, per tile.)"""
    px = int(round((label_w + F.CORNER_PAD + F.CORNER_CLEAR) * s))
    if corner[0] == "t":
        rows_ = slice(0, int(round((LABEL_BASE_T + 0.6) * s)))
    else:
        hb = win.shape[0]
        rows_ = slice(hb - int(round(F.BOTTOM_LABEL * s)), hb)
    sl = win[rows_, -px:] if corner[1] == "r" else win[rows_, :px]
    return not (sl[..., 3] > 25).any()


# -- text ----------------------------------------------------------------------------------
def wrap(pg, text, width, fs, **kw):
    """Greedy lines (a card of several sentences), never a single orphaned word."""
    words, lines, cur = text.split(" "), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if pg.width(t, fs, **kw) > width and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    lines.append(cur)
    if len(lines) > 1 and " " not in lines[-1]:
        prev = lines[-2].rsplit(" ", 1)
        if len(prev) == 2 and pg.width(prev[1] + " " + lines[-1], fs, **kw) <= width:
            lines[-2], lines[-1] = prev[0], prev[1] + " " + lines[-1]
    return lines


def head_tag(pg, xc, tag, colour, title):
    """agentic_results' column head: a solid rounded tag, then the title, centred on
    ``xc`` as a group, both placed on the title's cap-height centre."""
    tw = pg.width(title, F.HEAD_FS, weight="bold")
    gw = pg.width(tag, F.TAG_FS, weight="bold") + 2 * F.TAG_PAD
    x0 = xc - (gw + F.TAG_GAP + tw) / 2
    cap_mid = TY - F.CAP_EM * F.HEAD_FS * F.PT_MM / 2
    pg.ax.add_patch(FancyBboxPatch((x0, cap_mid - F.TAG_H / 2), gw, F.TAG_H,
                                   boxstyle="round,pad=0,rounding_size=0.9",
                                   fc=colour, ec="none", zorder=6))
    pg.text(x0 + gw / 2, cap_mid + F.CAP_EM * F.TAG_FS * F.PT_MM / 2, tag, F.TAG_FS,
            ha="center", weight="bold", color="white")
    pg.text(x0 + gw + F.TAG_GAP, TY, title, F.HEAD_FS, weight="bold", color=GREY)
    return x0, x0 + gw + F.TAG_GAP + tw


def pill(pg, x, y_mid, text, colour, fs=5.0):
    gw = pg.width(text, fs, weight="bold") + 2 * F.TAG_PAD
    pg.ax.add_patch(FancyBboxPatch((x, y_mid - F.TAG_H / 2), gw, F.TAG_H,
                                   boxstyle="round,pad=0,rounding_size=0.9",
                                   fc=colour, ec="none", zorder=6))
    pg.text(x + gw / 2, y_mid + F.CAP_EM * fs * F.PT_MM / 2, text, fs, ha="center",
            weight="bold", color="white")


# -- the row's mean agent cost per run -----------------------------------------------------
def row_cost(runs, row):
    """Means per run over the row's distinct runs (a placeholder row repeats one run:
    it is counted once) of agentic_results' cost_of fields, and of the wall clock's
    bar groups.  Each run is a separate agent session, so the per-run mean is the
    figure's cost, not the row's sum.  The divisor is always the N distinct runs: a
    field some run lacks counts 0 for that run, and ``missing`` names it (the build
    prints it: such a mean is a lower bound, not a mean over a smaller set)."""
    seen, costs, groups = set(), [], []
    for i in row["items"]:
        rec = runs[i]
        d = (rec.get("showcase") or {}).get("run_dir")
        if d in seen:
            continue
        seen.add(d)
        costs.append(F.cost_of(rec))
        g_ = ar_time.groups_of(((rec["meta"].get("time_split") or {}).get("sections")) or {})
        groups.append(g_ if sum(g_.values()) > 0 else None)

    def mean(k):
        v = [c[k] for c in costs if c[k] is not None]
        return float(np.sum(v)) / len(costs) if v else None
    out = {k: mean(k) for k in ("input", "cached", "output", "time")}
    out["n"] = len(costs)
    out["missing"] = [k for k in ("input", "output", "time") if any(c[k] is None for c in costs)]
    if groups and all(groups):
        out["groups"] = {g: float(np.mean([gg.get(g, 0.0) for gg in groups])) for g, _ in ar_time.GROUPS}
    else:
        out["groups"] = None
    return out


def cost_lines(c):
    if c["input"] is not None:
        share = 100.0 * (c["cached"] or 0) / c["input"] if c["input"] else 0.0
        t_in = f"{F.fmt_count(c['input'])} input ({share:.0f} % cached)"
    else:
        t_in = "input tokens n/a"
    return [t_in, f"{F.fmt_count(c['output'])} output tokens", F.fmt_clock(c["time"])]


# -- layout ----------------------------------------------------------------------------------
def grid_rows(n):
    """Tile rows a row of n items needs (at least one, at most GROWS)."""
    return min(GROWS, max(1, math.ceil(n / COLS)))


def item_xy(k, cell_x, top, IH):
    """Top-left of tile k: COLS across, filled from the top left."""
    r, c = divmod(k, COLS)
    return cell_x + c * (IW + IGAP), top + ROW_PAD_T + r * (IH + IGAP_V)


def tiles_height(IH, grows):
    return ROW_PAD_T + grows * IH + (grows - 1) * IGAP_V + ROW_PAD_B


# the label column's vertical stack (mm below the row top)
TITLE_BASE = 3.2
CARD_GAP = 1.8                   # title baseline -> card top
CARD_FIRST = 2.75                # card top -> first baseline
CARD_BOTTOM = 1.35               # last baseline -> card bottom
COST_HEAD = 3.0                  # card bottom -> the block's cost head ("Mean of N runs")
COST_FIRST = 3.3                 # that baseline -> first cost line; with no head (the
                                 # authoring call) card bottom -> first, agentic_results' own
BLOCK_GAP = 3.4                  # one block's last bar -> the next block's card
PILL_GAP = 1.4


def label_stack(cards, placeholder):
    """y offsets of the label column's pieces -- one block per kind of call, each a
    card of ``n`` lines (``cards[b] = (n, headed)``), its cost head when ``headed``
    and three bars -- and the column's height.  A block without a head (the
    authoring call) starts its bars right under its card, at agentic_results' card ->
    first cost line, so no gap is left where the head was."""
    y, blocks, bars_end = TITLE_BASE + CARD_GAP, [], None
    for n, headed in cards:
        ch = CARD_FIRST + CARD_PITCH * (n - 1) + CARD_BOTTOM
        hb = y + ch + COST_HEAD if headed else None
        c0 = (hb if headed else y + ch) + COST_FIRST
        blocks.append({"card_top": y, "card_h": ch, "cost_head": hb, "cost_first": c0})
        bars_end = c0 + 2 * F.COST_PITCH + F.BAR_OFF + F.BAR_H
        y = bars_end + BLOCK_GAP
    y = bars_end
    pill_mid = None
    if placeholder:
        pill_mid = y + PILL_GAP + F.TAG_H / 2
        y = pill_mid + F.TAG_H / 2
    return {"blocks": blocks, "pill_mid": pill_mid, "end": y + 0.6}


def call_blocks(runs, row, probe):
    """The row's label blocks, one per kind of Claude call, in the order they ran:
    [{"lines", "kw" (the card's text style), "cost", "cost_lines", "head"}].  A sweep
    row (material / loading) with an authoring call has two -- the authoring call
    (no head: it is one call, and its card says it wrote all N cases), then the run
    agents ("Mean of N runs") -- every other row one, the run agents.  ``calls`` comes from the record (sc_prep), or, for a _work written before
    it had them, from the run and batch files (sc_prep.sweep_calls)."""
    def card(text):
        if text:
            # an editorial insertion ('[one of the 9 cases]') is kept on one line
            text = re.sub(r"\[[^\]]*\]", lambda m: m.group(0).replace(" ", "\u00a0"), text)
            s_, kw = f"“{F.typographic(text)}”", {"style": "italic"}
        else:
            s_, kw = f"{row.get('distinct_prompts') or len(row['items'])} different prompts", {"color": GREY}
        return wrap(probe, s_, LAB_W - 2 * CARD_PAD_X, CARD_FS, **kw), kw

    def block(text, c, head):
        lines, kw = card(text)
        cl = cost_lines(c)
        for t_ in cl:
            assert probe.width(t_, F.META_FS) <= LAB_W, f"cost line wider than its column: {t_!r}"
        return {"lines": lines, "kw": kw, "cost": c, "cost_lines": cl, "head": head}

    rc = row_cost(runs, row)
    run_head = f"Mean of {rc['n']} run{'s' if rc['n'] != 1 else ''}"
    calls = row["calls"] if "calls" in row else (
        None if row.get("placeholder") else sc_prep.sweep_calls(row.get("run_dirs") or [], row["key"]))
    row["calls"] = calls                     # the record behind the cards (printed below)
    auth = (calls or {}).get("authoring") or {}
    if auth.get("cost"):
        a = auth["cost"]
        ac = {k: a[k] for k in ("input", "cached", "output", "time", "groups")}
        ac.update(n=1, missing=[])
        runs_prompt = ((calls or {}).get("runs") or {}).get("prompt")
        return [block(auth["prompt"], ac, None), block(runs_prompt, rc, run_head)]
    return [block(row.get("prompt"), rc, run_head)]


def draw_costs(pg, x, top, blk, b, maxima):
    """One block's cost head and agentic_results' three micro bars, on the figure's
    shared scales (``maxima``)."""
    c, cl = b["cost"], b["cost_lines"]
    if b["head"]:
        pg.text(x, top + blk["cost_head"], b["head"], F.KEY_FS, color=GREY, weight="bold")
    fr = [(c[k] or 0) / maxima[k] if maxima[k] else 0 for k in ("input", "output", "time")]
    cached = (c["cached"] or 0) / maxima["input"] if maxima["input"] else 0
    for j, s_ in enumerate(cl):
        yb = top + blk["cost_first"] + j * F.COST_PITCH
        pg.text(x, yb, s_, F.META_FS, color=GREY)
        by = yb + F.BAR_OFF
        pg.ax.add_patch(Rectangle((x, by), LAB_W, F.BAR_H, fc=F.COST_TRACK, ec="none", zorder=3))
        if j == 0:
            pg.ax.add_patch(Rectangle((x, by), LAB_W * fr[0], F.BAR_H, fc=F.COST_UNCACHED,
                                      ec="none", zorder=4))
            pg.ax.add_patch(Rectangle((x, by), LAB_W * cached, F.BAR_H, fc=F.COST_CACHED,
                                      ec="none", zorder=5))
        elif j == 2 and c["groups"]:
            x_, tot = x, sum(c["groups"].values())
            for g, _ in ar_time.GROUPS:
                w_ = LAB_W * fr[2] * c["groups"][g] / tot if tot else 0.0
                if w_ > 0:
                    pg.ax.add_patch(Rectangle((x_, by), w_, F.BAR_H, fc=F.TIME_COLORS[g],
                                              ec="none", zorder=4))
                x_ += w_
        else:
            pg.ax.add_patch(Rectangle((x, by), LAB_W * fr[j], F.BAR_H, fc=F.COST_NEUTRAL,
                                      ec="none", zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", default=os.path.join(HERE, "_work"))
    ap.add_argument("--out", default="showcase")
    ap.add_argument("--ground", default=os.environ.get("SC_GROUND", "grid"),
                    choices=("none", "shadow", "grid"))
    args = ap.parse_args()
    stats = json.load(open(os.path.join(args.work, "ar_stats.json")))
    runs = {r["index"]: r for r in stats["runs"]}
    rows = stats["rows"]
    had_calls = all("calls" in r for r in rows)
    # the cards re-condensed from their sources with the current sc_prep rules, so a
    # compose-only build of an older _work prints the current (short) cards
    cards_changed = any([sc_prep.refresh_cards(r) for r in rows if r["key"] != "shape"])
    lo, hi = stats["E_log10_range_Pa"]
    if lo is not None:
        sc_prep.check_e_map(lo, hi)    # the material row's own E map (its key's stops)
    for row in rows:
        assert len(row["items"]) <= MAX_ITEMS, f"row {row['key']}: more than {MAX_ITEMS} items"

    # ---- row heights: the row's tile rows, or its label column when that is taller ------
    probe = F.Page(W, 50)
    blocks = [call_blocks(runs, row, probe) for row in rows]
    stacks = [label_stack([(len(b["lines"]), bool(b["head"])) for b in bl], row["placeholder"])
              for bl, row in zip(blocks, rows)]
    F.plt.close(probe.fig)
    mat_k = next((k for k, r in enumerate(rows) if r["input_panel"] == "materials"), None)
    IH = IW * ASPECT
    # a row is as deep as its tiles or its label column, whichever is taller
    # (the material row's tiles carry the E key under them)
    has_e_key = mat_k is not None and lo is not None
    RHS = [max(tiles_height(IH, grid_rows(len(r["items"])))
               + (E_KEY_BAND if has_e_key and k == mat_k else 0.0), st["end"])
           for k, (r, st) in enumerate(zip(rows, stacks))]
    # one scale per metric over every block of the figure (authoring calls and run means)
    maxima = {k: max((b["cost"][k] or 0) for bl in blocks for b in bl) for k in ("input", "output", "time")}

    # ---- the tiles' images; a shared-view row is drawn at one scale -------------------
    # the tile annotations (agentic_results' kinds only): the factor on the deformed
    # tile, the generation path on a generated body; a row whose tiles have no clear
    # corner for them is re-cropped with a free band above / below the ink
    probe = F.Page(W, 50)
    tlabels = {}
    for row in rows:
        for i in row["items"]:
            rec = runs[i]
            sc = f"×{F.fmt_scale(rec['scale']['factor'])}" if rec.get("scale") else None
            src = rec.get("body_source") if row["key"] == "shape" else None
            tlabels[i] = {"solve": (sc, probe.width(sc, F.META_FS)) if sc else None,
                          "input": (src, probe.width(src, F.META_FS)) if src else None}
    F.plt.close(probe.fig)

    def row_images(row, res):
        got = {i: item_images(args.work, runs[i], row["input_panel"], args.ground, IW, IH, res=res)
               for i in row["items"]}
        if row["view"] == "shared":
            s = max(g[2] for g in got.values() if g)
            got = {i: item_images(args.work, runs[i], row["input_panel"], args.ground, IW, IH, s, res)
                   for i in row["items"]}
        return got

    def blocked(row, got, which, corners):
        out = False
        for i in row["items"]:
            lab, g = tlabels[i]["solve" if which == "solve" else "input"], got.get(i)
            if not lab or not g or which not in g[0]:
                continue
            win = g[0][which][1]
            out |= not any(corner_clear(win, g[2], IW, IH, cn, lab[1]) for cn in corners)
        return out

    images, reserves = {}, {}
    for row in rows:
        got = row_images(row, (0.0, 0.0))
        res = (TOP_RES if blocked(row, got, "solve", ("tr", "tl")) else 0.0,
               BOT_RES if blocked(row, got, row["input_panel"], ("br", "bl")) else 0.0)
        if any(res):
            got = row_images(row, res)
        reserves[row["key"]] = res
        images.update(got)

    # ---- page -------------------------------------------------------------------------
    tops = [HEAD]
    for rh in RHS[:-1]:
        tops.append(tops[-1] + rh + ROW_GAP)
    table_bottom = tops[-1] + RHS[-1] + ROW_GAP / 2
    page_h = table_bottom + FOOT
    pg = F.Page(W, page_h)

    pg.line([ML, W - MR], [0.4, 0.4], 0.6, INK)
    pg.line([ML, W - MR], [HEAD_RULE, HEAD_RULE], 0.35, INK)
    hairlines = [tops[k] + RHS[k] + ROW_GAP / 2 for k in range(len(rows) - 1)]
    for yh in hairlines:
        pg.line([ML, W - MR], [yh, yh], 0.25, F.HAIR)
    pg.line([ML, W - MR], [table_bottom, table_bottom], 0.6, INK)
    xm = X_IN + CW + MID / 2
    pg.line([xm, xm], [HEAD_RULE + 0.6, table_bottom - 0.6], 0.3, VRULE, z=2)

    pg.text(ML, TY, "VARIED", F.HEAD_FS, weight="bold", color=GREY)
    for cx, (tag, col, title) in zip((X_IN, X_DEF), COLUMNS):
        head_tag(pg, cx + CW / 2, tag, col, title)
    F.arrow(pg, xm - 5.0, TY - 0.72, xm + 5.0, TY - 0.72)   # the light head arrow, on the gap

    placeholders, key_texts = [], []
    tile_labels = []
    for k, row in enumerate(rows):
        top, st, RH = tops[k], stacks[k], RHS[k]
        pg.text(ML, top + TITLE_BASE, ROW_TITLE[row["key"]], TITLE_FS, weight="bold")
        # one block per kind of call: its prompt, quoted (agentic_results' card), and
        # its cost bars
        for b, blk in zip(blocks[k], st["blocks"]):
            ct = top + blk["card_top"]
            pg.ax.add_patch(FancyBboxPatch((ML, ct), LAB_W, blk["card_h"],
                                           boxstyle="round,pad=0,rounding_size=1.3",
                                           fc="white", ec=F.CARD_STROKE, lw=0.6, zorder=4))
            for j, line in enumerate(b["lines"]):
                pg.text(ML + CARD_PAD_X, ct + CARD_FIRST + CARD_PITCH * j, line, CARD_FS, **b["kw"])
            draw_costs(pg, ML, top, blk, b, maxima)
        if row["placeholder"]:
            pill(pg, ML, top + st["pill_mid"], "PLACEHOLDER", PLACEHOLDER_RED)
            placeholders.append(row["key"])
        assert top + st["end"] <= top + RH + 1e-6, f"row {row['key']}: label column taller than its row"

        # the tiles: input on the left, deformed on the right, same grid
        for j, i in enumerate(row["items"]):
            rec, got = runs[i], images.get(i)
            for cell_x, which in ((X_IN, row["input_panel"]), (X_DEF, "solve")):
                x, yt = item_xy(j, cell_x, top, IH)
                if got and which in got[0]:
                    pg.img(got[0][which][0], x, yt, IW, IH)
                else:
                    pg.text(x + IW / 2, yt + IH / 2, "not available", F.META_FS, ha="center",
                            va="center", color=GREY)
                    continue
                win, s_px = got[0][which][1], got[2]
                if which == "solve" and rec.get("scale"):
                    lab = f"×{F.fmt_scale(rec['scale']['factor'])}"
                    lw = pg.width(lab, F.META_FS)
                    ok = [cn for cn in ("tr", "tl") if corner_clear(win, s_px, IW, IH, cn, lw)]
                    if not ok:
                        FLOOR_MISSES.append(f"item {i}: no clear top corner for {lab}")
                    cn = (ok or ["tr"])[0]
                    tx = x + (F.CORNER_PAD if cn == "tl" else IW - F.CORNER_PAD)
                    tile_labels.append(pg.text(tx, yt + LABEL_BASE_T, lab, F.META_FS, color=GREY,
                                               ha="left" if cn == "tl" else "right"))
                if which != "solve" and row["key"] == "shape" and rec.get("body_source"):
                    lab = rec["body_source"]
                    lw = pg.width(lab, F.META_FS)
                    ok = [cn for cn in ("br", "bl") if corner_clear(win, s_px, IW, IH, cn, lw)]
                    if not ok:
                        FLOOR_MISSES.append(f"item {i}: no clear bottom corner for {lab}")
                    cn = (ok or ["br"])[0]
                    tx = x + (F.CORNER_PAD if cn == "bl" else IW - F.CORNER_PAD)
                    tile_labels.append(pg.text(tx, yt + IH - BOTTOM_BASE, lab, F.META_FS, color=GREY,
                                               ha="left" if cn == "bl" else "right"))

        # the material row: its E key at the bottom of the row, centred under its E
        # tiles, at the footer keys' width (half a cell, as u_keys under the deformed
        # cell), in this figure's own map
        if has_e_key and k == mat_k:
            half = CW / 2
            ky = top + tiles_height(IH, grid_rows(len(row["items"]))) + E_KEY_BASE
            ex = X_IN + (CW - half) / 2
            key_texts += F.e_key(pg, ex, half, ky, lo, hi, X_IN + CW,
                                 stops=sc_prep.sc_e_stops(lo, hi))
            # no Young's-modulus column head here to name the ramp: an italic E before it
            key_texts.append(pg.text(ex + 3.5 - 0.8, ky - 0.6, "E", F.KEY_FS, ha="right",
                                     style="italic"))
            e_key_bottom = ky + 2.4 + 0.8
            assert e_key_bottom <= top + RH + 1e-6, "the E key runs below the material row"

    # ---- footer: agentic_results' keys, each under the column it explains -------------
    fy = table_bottom + 3.4                      # first key baseline (ar_figure's)
    key_texts += F.cost_keys(pg, ML, fy, any(b["cost"]["groups"] for bl in blocks for b in bl))[0]
    # the input cell: supports & loads, centred under it at half a cell's width (the
    # Young's modulus key sits at the bottom of the material row instead)
    half = CW / 2
    bc_items = [runs[i] for r in rows if r["input_panel"] == "bc" for i in r["items"]]
    if bc_items:
        key_texts += F.bc_keys(pg, X_IN + (CW - half) / 2, half, fy,
                               any((rec.get("bc") or {}).get("arrows") for rec in bc_items),
                               any(g.get("seen_through") for rec in bc_items
                                   for g in ((rec.get("bc") or {}).get("regions") or [])))
    # the deformed cell: agentic_results' deformed-shape column keys, centred under it,
    # the outline swatch in this figure's outline grey
    key_texts += F.u_keys(pg, X_DEF + (CW - half) / 2, half, fy, OUTLINE_RGB)

    # ---- assertions (agentic_results' text gates) -----------------------------------------
    pg.fig.canvas.draw()
    bbs = []
    for t in pg.texts:
        s_ = t.get_text()
        if not s_:
            continue
        assert t.get_fontsize() >= 5.0 - 1e-6, f"text below 5 pt: {s_!r}"
        b = pg.box(t)
        assert b[0] >= -1e-3 and b[2] <= W + 1e-3 and b[1] >= -1e-3 and b[3] <= page_h + 1e-3, \
            f"text off the page: {s_!r}"
        bbs.append((b, s_))
    for a_ in range(len(bbs)):
        for b_ in range(a_ + 1, len(bbs)):
            p_, q_ = bbs[a_][0], bbs[b_][0]
            if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]:
                raise AssertionError(f"texts overlap: {bbs[a_][1]!r} / {bbs[b_][1]!r}")
    for t in pg.texts:      # the label column's text stays in the column
        b = pg.box(t)
        if b[0] < X_IN - 1.0 and b[1] > HEAD_RULE and b[3] < table_bottom:
            assert b[2] <= ML + LAB_W + 1e-3, f"label-column text runs into the tiles: {t.get_text()!r}"
    for k, row in enumerate(rows):          # tiles stay inside their row band
        _, yt = item_xy(len(row["items"]) - 1, X_IN, tops[k], IH)
        limit = hairlines[k] if k < len(rows) - 1 else table_bottom
        assert yt + IH <= limit + 1e-6, f"row {row['key']}: tiles reach the rule below"

    if not had_calls or cards_changed:
        # a _work written before the record had the calls (call_blocks added them to
        # the rows) or before the current cards (refresh_cards): write them back,
        # so the record (copied to sc_stats.json) holds every number and card the
        # figure prints
        json.dump(stats, open(os.path.join(args.work, "ar_stats.json"), "w"), indent=1, default=float)
    pg.fig.savefig(f"{args.out}.pdf", dpi=600)
    pg.fig.savefig(f"{args.out}.png", dpi=F.DPI)
    print(f"wrote {args.out}.pdf / .png  ({W:.0f} x {page_h:.1f} mm; tiles {IW:.1f} x {IH:.1f} mm; "
          f"rows {', '.join(f'{h:.1f}' for h in RHS)} mm; label reserves {reserves})")
    for r in rows:
        print(f"row {r['key']}: prompt from {r.get('prompt_source')}; shared scale {r.get('shared_scale')}")
    for bl, r in zip(blocks, rows):
        for b in bl:
            c = b["cost"]
            print(f"row {r['key']} [{b['head'] or 'authoring call'}]: {c['input']} input ({c['cached']} cached), "
                  f"{c['output']} output, {c['time']} s"
                  + (f"; WARNING {c['missing']} missing in some runs: a lower bound" if c["missing"] else ""))
        auth = ((r.get("calls") or {}).get("authoring") or {}).get("cost")
        if auth:
            print(f"row {r['key']}: authoring call {auth['tries']} tr{'y' if auth['tries'] == 1 else 'ies'} "
                  f"({', '.join(auth['files'])}); time from {auth['time_source']} "
                  f"{auth['time']:.1f} s, batch-log span {auth['log_span_s']} s; ${auth['cost_usd']:.4f}")
        if ((r.get("calls") or {}).get("runs") or {}).get("distinct", 1) > 1:
            print(f"row {r['key']}: WARNING the runs' condensed prompts differ beyond their case: "
                  "the card shows the first")
    for m in FLOOR_MISSES:
        print("CORNER:", m)
    for key in placeholders:
        print(f"PLACEHOLDER row {key!r}: replace its runs before the paper build (sc_prep.py)")


if __name__ == "__main__":
    main()
