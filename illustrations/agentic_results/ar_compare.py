"""The model comparison of the agentic results: one row per object, three agents side by side,
on US-letter pages.

    python3 ar_compare.py plan    --haiku ROOT --sonnet ROOT --opus ROOT --work DIR [--only SLUG ...]
    python3 ar_compare.py restrict DIR             # after ar_prep: render three panels, not four
    python3 ar_compare.py compose DIR [--out NAME] [--ground grid|shadow|none]
    python3 ar_compare.py compose DIR --stub-panels crops|synthetic [--stub-from ILLU]

``./build.sh agenticcompare`` runs plan -> ar_prep.py -> restrict -> render_ar.py -> compose.

COLUMNS (13).  The instruction's prompt card, as tall as the row; then, for each model
(Haiku, Sonnet, Opus, from left to right), four columns: the agent's cost (input tokens with
their cached share, output tokens, wall clock split into agent / workflow / solver / runner
overhead, as in agentic_results), the generated shape, the supports and loads, and Young's
modulus.  There is no deformed-shape column.  A model-name header spans each group of four.

PANELS.  Each image panel is as tall as the stack of the three cost bars beside it; every
text size was chosen for that row height.  Under each image a label line carries what
agentic_results prints in the panel's corners: the body's route (Text -> TripoSG, ...) under
the shape, the body load under the supports and loads, and the grid-cell square under
Young's modulus (the grid lies under all three panels; the narrow panels have no free corner
for it).  The renders are agentic_results' own (ar_prep.py + render_ar.py), one camera per
RUN as ar_prep.choose_view picks it: the three models' bodies of one object differ (each
agent generated its own), so a shared per-object camera would favour one of them.  Each run
is cropped to the union ink box of its three panels.

PAGES.  ``--rows`` (17) rows per page at most; the objects, in list order, split as evenly as
possible over the fewest pages, the longer pages placed so a category broken across a page
keeps at least two rows on each side where it can.  A category is marked by a rotated label
in a strip at the left, a bracket along its rows, and a darker rule above its first row.
Each page: 8.5 x 11 in, all headers at the top, the keys under the table.

FAILED RUNS.  A panel whose stage the run never reached (no body; no supports and loads; no
Young's modulus) is a "not reached" cell; "failed" when the agent's own run did not end ok.
Its cost bars are drawn either way.  A missing run (an object with no run of some model)
fails ``plan``, naming the objects; ``plan --allow-missing`` (stub pages only: compose
refuses it otherwise) draws them as "no run yet" cells, to test pagination early.

STUB PANELS (testing without renders).  ``--stub-panels crops`` builds the pages from the
real runs' metrics.json and CPU-only labels (``light_record``: route, body load, grid cell,
Young's-modulus range, reached stages) and fills the image panels with crops of the built
single-model category figures (``<stub-from>/agentic_results_<slug>.png``, their corner
labels blanked); ``--stub-panels synthetic`` writes fake renders (silhouettes, a grid ground)
into DIR and runs the real compose path on them.  Stub pages carry a "stub panels" note.
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ar_categories as CAT  # noqa: E402
import ar_figure as F  # noqa: E402  (rcParams, Arial, Page, formatting, keys)
import ar_runs  # noqa: E402
import ar_style as A  # noqa: E402
import ar_time  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

MODELS = ("haiku", "sonnet", "opus")
SHOWN = ("painted", "bc", "materials")
OK_STATUS = ("ok", "success", "succeeded", "complete", "completed", "finished", "done")

# -- page grid (mm, y down) --------------------------------------------------------------
PAGE_W, PAGE_H = 215.9, 279.4        # US letter, portrait
ML = MR = 6.0
MT, MB = 6.0, 6.0
CAT_W = 4.0                          # category strip: rotated label + bracket
CARD_W = 19.0                        # prompt card
CARD_GAP = 1.6
GROUP_GAP = 2.4                      # between two models' groups
COL_GAP = 0.8                        # between columns of one group
COST_W = 19.5
X_CARD = ML + CAT_W
GX0 = X_CARD + CARD_W + CARD_GAP
GROUP_W = (PAGE_W - MR - GX0 - 2 * GROUP_GAP) / 3
P = (GROUP_W - COST_W - 3 * COL_GAP) / 3          # image panel width
HEAD_RULE = MT + 13.4
ROWS0 = HEAD_RULE + 0.4
FOOT = 11.5                          # key band under the bottom rule
ROWS_MAX = 17
ROW_GAP = 1.2                        # between row bands; the hairline in its middle
ROW_PAD = 0.3                        # row top -> image / cost stack / card
LBL_H = 2.0                          # the label line under the images
BAR_OFF, BAR_H = 0.55, 0.6           # cost: baseline -> bar top; bar height

GROUP_FS, HEAD_FS, TAG_FS = 6.0, 4.6, 4.0
PROMPT_FS, META_FS, LABEL_FS, KEY_FS, CAT_FS = 5.4, 4.6, 4.2, 5.0, 4.6
TAG_PAD, TAG_H, TAG_R = 0.55, 2.05, 0.72
PROMPT_PITCH = 2.15
MAX_PROMPT_LINES = 4
CAP = F.CAP_EM * F.PT_MM              # cap height per pt (mm)
CAT_RULE = "#aab1bb"                 # the rule above a category's first row
TITLES = (("TOKENS", "& TIME"), ("GENERATED", "SHAPE"), ("SUPPORTS", "& LOADS"),
          ("YOUNG’S", "MODULUS"))
PANEL_DPI = 600                      # each panel image is resampled to this at its print size
MARGIN = 0.03                        # crop margin, fraction of the ink box's longer side
INK_ROOM = 0.25                      # mm under the ink, inside the image box
GROUND_FADE = 1.2                    # mm: the ground fades out at the drawn image edges
WARNINGS = []


def fmt_force(f):
    """agentic_results' force format (F.fmt_force), continued past 999 kN into MN and GN,
    so a building's weight fits the narrow label line ('1.2 GN', not '1200000 kN')."""
    for div, unit in ((1e9, "GN"), (1e6, "MN")):
        if f >= 0.9995 * div:
            return f"{F.sig(f / div, 2)} {unit}"
    return F.fmt_force(f)


def body_label(body):
    f = body.get("weight_N")
    return f"Body load {fmt_force(f)}" if f else "Body load"


def model_label(model_id, key):
    """'claude-haiku-5-5' -> 'Claude Haiku 5.5'; the key when the id is unknown."""
    p = (model_id or "").split("-")
    if len(p) >= 3 and p[0] == "claude":
        return f"Claude {p[1].capitalize()} {'.'.join(p[2:])}"
    return key.capitalize()


# =========================================================================================
# plan: which runs, in which order
# =========================================================================================
def make_plan(roots, only=None, list_path=CAT.OBJECTS, allow_missing=False):
    cats = [c for c in CAT.categories(list_path) if not only or c[1] in only]
    assert cats, "no categories selected"
    for m in MODELS:
        if not os.path.isdir(os.path.join(roots[m], "_batches")):
            raise SystemExit(f"{m}: {roots[m]} is not a batch root (no _batches/)")
    picks = {m: CAT.latest_runs(roots[m]) for m in MODELS}
    missing = {m: [o for _, _, objs in cats for o in objs if o not in picks[m]] for m in MODELS}
    if any(missing.values()) and not allow_missing:
        raise SystemExit("missing runs:\n" + "\n".join(
            f"  {m} ({roots[m]}): {len(v)} object(s): " + "; ".join(v)
            for m, v in missing.items() if v))
    objects, order = [], []
    for name, sl, objs in cats:
        for o in objs:
            runs = {}
            for m in MODELS:
                if o in picks[m]:
                    runs[m] = {"path": picks[m][o], "index": len(order)}
                    order.append(picks[m][o])
            objects.append({"instruction": o, "category": name, "slug": sl, "runs": runs})
    labels = {}
    for m in MODELS:
        ids = [json.load(open(os.path.join(r["path"], "metrics.json"))).get("model")
               for ob in objects for r in [ob["runs"].get(m)] if r and
               os.path.exists(os.path.join(r["path"], "metrics.json"))]
        labels[m] = model_label(ids[0] if ids else None, m)
    return {"roots": {m: str(roots[m]) for m in MODELS}, "labels": labels,
            "objects": objects, "order": order}


# =========================================================================================
# CPU-only records (stub pages): what ar_prep records for the labels, without rendering
# =========================================================================================
def light_record(i, run):
    """The fields compose reads from ar_stats.json, computed on the CPU from the session:
    body_source, ground.grid, bc.body_load (+ region categories), E_surface_Pa, and the
    panels the run reached (``_reached``).  Same helpers as ar_prep.prep_run."""
    import ar_prep as PR
    rec = {"index": i, **run.record(), "notes": [], "_reached": []}
    s = run.session
    if not s or not os.path.isdir(s):
        rec["notes"].append("no workflow session folder found")
        return rec, None
    rec["body_source"] = PR.body_source(s)
    phys = os.path.join(s, "geometry", "physical.glb")
    if not os.path.exists(phys):
        rec["notes"].append("no geometry/physical.glb")
        return rec, None
    mesh = PR.load_mesh(phys)
    V = np.asarray(mesh.vertices, float)
    rec["_reached"].append("painted")
    tf = os.path.join(s, "geometry", "physical.glb.transform.json")
    unit = json.load(open(tf)).get("units") if os.path.exists(tf) else None
    mp = os.path.join(s, "case", "material.json")
    mat = json.load(open(mp)) if os.path.exists(mp) else {}
    unit = unit or mat.get("unit_system")
    units = PR.UNITS.get(unit) or PR.UNITS["SI (m-kg-s)"]
    rec["unit_system"], rec["m_per_length_unit"] = unit, units[1]
    rec["ground"] = {"grid": PR.ui_grid(V, float(V[:, 1].min()))[3]}
    rz, rj = (os.path.join(s, "boundary_conditions", n) for n in ("resolved.npz", "resolved.json"))
    if os.path.exists(rz) and os.path.exists(rj):
        rec["_reached"].append("bc")
        regions = json.load(open(rj)).get("regions", [])
        cp = os.path.join(s, "case", "case.json")
        case = json.load(open(cp)) if os.path.exists(cp) else {}
        body = None
        for kind in ("acceleration", "body_force"):
            vec = np.asarray((case.get("loads") or {}).get(kind) or [0.0, 0.0, 0.0], float)
            if np.linalg.norm(vec) > 0:
                body = {"kind": kind}
                body.update(PR.body_load_force(s, mesh, case, mat, kind, vec, units))
                break
        trac = [nb for nb in (case.get("boundary_conditions") or {}).get("neumann") or []
                if nb.get("traction") is not None and np.linalg.norm(nb["traction"]) > 0]
        rec["bc"] = {"regions": [{"name": r.get("name"), "category": r.get("category")}
                                 for r in regions],
                     "arrows": [{"region": nb["name"]} for nb in trac], "body_load": body}
    E = None
    run_dir, _ = PR.chosen_run(s)
    if run_dir and not PR.solve_failed(s):
        import glob
        vtus = sorted(glob.glob(os.path.join(run_dir, "vtu", "solution_t*.vtu")))
        d = PR.read_vtu(vtus[-1])
        if "youngs_modulus" in d:
            E = np.asarray(d["youngs_modulus"], float)
    if E is None:
        if mat.get("mode") == "uniform" and mat.get("youngs_modulus"):
            E = np.array([float(mat["youngs_modulus"])])
        else:
            E = PR.grid_E(s, V)
    if E is not None:
        E = E * units[0]
        rec["E_surface_Pa"] = (float(E.min()), float(E.max()))
        rec["_reached"].append("materials")
    return rec, E


def light_stats(plan, work):
    import ar_prep as PR
    runs = ar_runs.load_runs(plan["order"])
    recs, Es = [], []
    for i, run in enumerate(runs):
        rec, E = light_record(i, run)
        recs.append(rec)
        Es.append(None if E is None else {"E": E})
        print(f"[{i}] {run.get('instruction')!r}: reached {rec['_reached']}")
    lo, hi = PR.e_log_range(Es)
    stats = {"E_log10_range_Pa": [lo, hi], "runs": recs, "light": True}
    json.dump(stats, open(os.path.join(work, "ar_stats.json"), "w"), indent=1, default=float)
    return stats


# =========================================================================================
# panels
# =========================================================================================
def resample(rgb, w_mm, h_mm):
    """``rgb`` (uint8) at PANEL_DPI for a w_mm x h_mm print box."""
    px = (max(1, int(round(w_mm / 25.4 * PANEL_DPI))), max(1, int(round(h_mm / 25.4 * PANEL_DPI))))
    return np.asarray(Image.fromarray(rgb).resize(px, Image.LANCZOS))


def crop_window(boxes, w, h):
    """The render-pixel window for a w x h mm image box: the union ink box with MARGIN at
    the top and sides and INK_ROOM under it, centred.  (x0, y0, x1, y1, px per mm)."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    bw, bh = x1 - x0, y1 - y0
    pad = MARGIN * max(bw, bh)
    s = max((bw + 2 * pad) / w, (bh + 2 * pad) / (h - INK_ROOM))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    # centred vertically when the box is wider than the ink needs; else on its INK_ROOM
    top = cy - h * s / 2 if (bh + 2 * pad) / (h - INK_ROOM) < s * 0.999 else y1 + INK_ROOM * s - h * s
    left = cx - w * s / 2
    return left, top, left + w * s, top + h * s, s


def real_panels(work, rec, w, h, ground):
    """{panel: RGB uint8} for one run from render_ar's PNGs, composed as ar_figure composes
    a row (ground layer faded at the edges, far-face loads laid over the BC panel)."""
    i = rec["index"]
    imgs = {p: F.load_rgba(os.path.join(work, f"ar_{i}_{p}.png")) for p in SHOWN}
    boxes = []
    for p, im in imgs.items():
        if im is None:
            continue
        b = F.ink_box(im)
        if b is None:
            imgs[p] = None
            WARNINGS.append(f"run {i} {p}: empty render")
            continue
        if not (b[0] > 0 and b[1] > 0 and b[2] < im.shape[1] and b[3] < im.shape[0]):
            WARNINGS.append(f"run {i} {p}: the object touches the render border (clipped)")
        boxes.append(b)
    if not boxes:
        return {}
    x0, y0, x1, y1, s = crop_window(boxes, w, h)
    hf = lambda name: F.load_rgba(os.path.join(work, f"ar_{i}_{name}.png"))  # noqa: E731
    hidden_faces, harrows, hsupport = hf("bchidden"), hf("bchidden_arrows"), hf("bchidden_support")
    out = {}
    for p, im in imgs.items():
        if im is None:
            continue
        win = F.cut_rgba(im, x0, y0, x1, y1)
        base = None
        if ground != "none":
            g = F.load_rgba(os.path.join(work, f"ar_{i}_{p}_g.png"))
            if g is not None:
                gw = F.cut_rgba(g, x0, y0, x1, y1)
                hh, ww = gw.shape[:2]
                yy, xx = np.mgrid[0:hh, 0:ww]
                d = np.minimum.reduce([xx, ww - 1 - xx, yy, hh - 1 - yy]) / (GROUND_FADE * s)
                gw[..., 3] *= np.clip(d, 0, 1) ** 1.5
                base = F.over(gw, None)
            else:
                WARNINGS.append(f"run {i} {p}: no ground render")
        rgb = F.over(win, base)
        if p == "bc":
            for layer, op, col in ((hidden_faces, F.SEE_THROUGH, A.NEUMANN),
                                   (harrows, F.ARROW_THROUGH, A.LOAD_ARROW),
                                   (hsupport, F.ARROW_THROUGH, A.DIRICHLET)):
                if layer is None:
                    continue
                hw = F.cut_rgba(layer, x0, y0, x1, y1)[..., 3:4] / 255.0 * op
                rgb = np.clip(rgb * (1 - hw) + np.array(col) * 255 * hw, 0, 255).astype(np.uint8)
        out[p] = resample(rgb, w, h)
    return out


# -- stub panels: crops of the built single-model category figures -------------------------
class CropSource:
    """Panels cut from ``<illu>/agentic_results_<slug>.png`` (the Sonnet category figures),
    their corner labels blanked, fitted into the comparison's image box."""

    def __init__(self, illu):
        self.illu = illu
        self.cache = {}
        self.pg = F.Page(50, 20)

    def figure(self, slug):
        if slug not in self.cache:
            png = os.path.join(self.illu, f"agentic_results_{slug}.png")
            js = os.path.join(self.illu, "agentic_results", f"ar_stats_agentic_results_{slug}.json")
            if not (os.path.exists(png) and os.path.exists(js)):
                self.cache[slug] = None
            else:
                im = np.asarray(Image.open(png).convert("RGB"))
                self.cache[slug] = (im, json.load(open(js)))
        return self.cache[slug]

    def panels(self, slug, instruction, w, h):
        fig = self.figure(slug)
        if fig is None:
            return None
        im, st = fig
        runs = st["runs"]
        ks = [k for k, r in enumerate(runs) if r["meta"].get("instruction") == instruction]
        if not ks:
            return None
        k, rec, n = ks[0], runs[ks[0]], len(runs)
        mm = im.shape[1] / F.W                                 # px per mm of that figure
        page_h = im.shape[0] / mm
        RH = (page_h - F.HEAD - F.FOOT - (n - 0.5) * F.ROW_GAP) / n
        Pf = (F.W - F.MR - F.X0 - 3 * F.GAP) / 4
        Hf = RH - F.PANEL_PAD_TOP
        top = F.HEAD + k * (RH + F.ROW_GAP) + F.PANEL_PAD_TOP
        base = top + Hf - F.BOTTOM_LABEL + 2.0                 # the corner labels' baseline
        out = {}
        for c, p in enumerate(SHOWN):
            x = F.X0 + c * (Pf + F.GAP)
            a = im[int(top * mm):int((top + Hf) * mm), int(x * mm):int((x + Pf) * mm)].copy()
            blank = []                                         # (x0, x1, y0, y1) mm in the panel
            if p == "painted":
                src = rec.get("body_source")
                if src:
                    wd = self.pg.width(src, F.META_FS)
                    blank.append((Pf - F.CORNER_PAD - wd - 0.3, Pf, base - top - 2.0, Hf))
                cell = F.fmt_cell(rec)
                if cell:
                    wd = F.CELL_ICON + 0.8 + self.pg.width(cell, F.META_FS)
                    blank.append((0, F.CORNER_PAD + wd + 0.3, base - top - 2.0, Hf))
                    blank.append((0, F.CORNER_PAD + wd + 0.3, 0, F.LABEL_BASE - F.PANEL_PAD_TOP + 0.6))
            if p == "bc" and (rec.get("bc") or {}).get("body_load"):
                wd = self.pg.width(F.body_label(rec["bc"]["body_load"]), F.META_FS)
                blank.append((Pf - F.CORNER_PAD - wd - 0.3, Pf, base - top - 2.0, Hf))
            for bx0, bx1, by0, by1 in blank:
                a[max(int(by0 * mm), 0):int(by1 * mm), max(int(bx0 * mm), 0):int(bx1 * mm)] = 255
            # fit (contain) into w x h, centred on white
            s = min(w / Pf, h / Hf)
            fitted = resample(a, Pf * s, Hf * s)
            box = np.full((int(round(h / 25.4 * PANEL_DPI)), int(round(w / 25.4 * PANEL_DPI)), 3),
                          255, np.uint8)
            oy = (box.shape[0] - fitted.shape[0]) // 2
            ox = (box.shape[1] - fitted.shape[1]) // 2
            box[oy:oy + fitted.shape[0], ox:ox + fitted.shape[1]] = fitted[:box.shape[0] - oy,
                                                                          :box.shape[1] - ox]
            out[p] = box
        return out


def write_synthetic_renders(work, stats):
    """Fake render_ar output for every run's reached panels: a silhouette over a transparent
    background (wide, tall or square by index), a grid ground, a far-face load layer on
    every third run.  Exercises the real compose path without a renderer."""
    n = 700
    yy, xx = np.mgrid[0:n, 0:n]
    for rec in stats["runs"]:
        i = rec["index"]
        ax_, ay_ = ((0.36, 0.18), (0.16, 0.38), (0.26, 0.26))[i % 3]
        body = ((xx - n * 0.5) / (n * ax_)) ** 2 + ((yy - n * 0.52) / (n * ay_)) ** 2 <= 1
        cols = {"painted": [196, 160, 120], "bc": [194, 194, 196],
                "materials": [int(255 * c) for c in A.E_RGB[-2]]}
        for p in rec.get("_reached", []):
            a = np.zeros((n, n, 4), np.uint8)
            a[body, :3] = cols[p]
            a[body, 3] = 255
            if p == "bc":
                top = body & (yy < n * (0.52 - ay_ * 0.6))
                a[top, :3] = [int(255 * c) for c in A.NEUMANN]
                low = body & (yy > n * (0.52 + ay_ * 0.8))
                a[low, :3] = [int(255 * c) for c in A.DIRICHLET]
            Image.fromarray(a).save(os.path.join(work, f"ar_{i}_{p}.png"))
            g = np.zeros((n, n, 4), np.uint8)
            lines = ((xx % 60) < 2) | ((yy % 60) < 2)
            g[lines] = [148, 163, 184, 120]
            Image.fromarray(g).save(os.path.join(work, f"ar_{i}_{p}_g.png"))
        if "bc" in rec.get("_reached", []) and i % 3 == 1:
            h = np.zeros((n, n, 4), np.uint8)
            h[body & (xx > n * 0.55), 3] = 255
            Image.fromarray(h).save(os.path.join(work, f"ar_{i}_bchidden.png"))


# =========================================================================================
# the pages
# =========================================================================================
def page_split(objects, rows_max):
    """Rows per page: the fewest pages, as even as possible; of the orders of those page
    sizes, the one whose smallest category piece (a category's rows on one page) is
    largest, then the one with the longer pages first."""
    n = len(objects)
    pages = max(1, math.ceil(n / rows_max))
    base, extra = divmod(n, pages)
    sizes = [base + 1] * extra + [base] * (pages - extra)
    best = None
    for perm in sorted(set(itertools.permutations(sizes)), reverse=True):
        start, pieces = 0, []
        for sz in perm:
            cats = [o["slug"] for o in objects[start:start + sz]]
            pieces += [len(list(g)) for _, g in itertools.groupby(cats)]
            start += sz
        key = min(pieces)
        if best is None or key > best[0]:
            best = (key, list(perm))
    return best[1]


def cost_lines(pg, rec):
    c = F.cost_of(rec)
    if c["input"] is not None:
        share = 100.0 * float(c["cached"] or 0) / float(c["input"]) if c["input"] else 0.0
        t_in = f"{F.fmt_count(c['input'])} input ({share:.0f} % cached)"
    else:
        t_in = "input tokens n/a"
    cl = [t_in, f"{F.fmt_count(c['output'])} output tokens", F.fmt_clock(c["time"])]
    st = str(c["status"] or "")
    if st and st.lower() not in OK_STATUS:
        cl[2] += f" · {st}"
    return c, cl


def fit_fs(pg, s, fs, width, **kw):
    """``fs``, or the largest size under it at which ``s`` fits ``width`` (0.1 pt steps)."""
    while pg.width(s, fs, **kw) > width and fs > 3.0:
        fs = round(fs - 0.1, 2)
    return fs


def draw_pill(pg, cx, cy, tag, col):
    gw = pg.width(tag, TAG_FS, weight="bold") + 2 * TAG_PAD
    pg.ax.add_patch(FancyBboxPatch((cx - gw / 2, cy - TAG_H / 2), gw, TAG_H,
                                   boxstyle=f"round,pad=0,rounding_size={TAG_R}",
                                   fc=col, ec="none", zorder=6))
    return pg.text(cx, cy + CAP * TAG_FS / 2, tag, TAG_FS, ha="center", weight="bold",
                   color="white")


def group_x(g):
    """Left edges of group g's four columns."""
    x = GX0 + g * (GROUP_W + GROUP_GAP)
    return [x, x + COST_W + COL_GAP] + [x + COST_W + COL_GAP + c * (P + COL_GAP) for c in (1, 2)]


def compose(work, out, ground, stub=None, stub_from=None, rows_max=ROWS_MAX):
    plan = json.load(open(os.path.join(work, "compare_plan.json")))
    objects = plan["objects"]
    if stub and stub != "crops":
        stats = light_stats(plan, work)
        write_synthetic_renders(work, stats)
    elif stub == "crops":
        stats = light_stats(plan, work)
    else:
        stats = json.load(open(os.path.join(work, "ar_stats.json")))
    recs = {r["index"]: r for r in stats["runs"]}
    assert len(recs) == len(plan["order"]), \
        f"ar_stats.json has {len(recs)} runs, the plan {len(plan['order'])}"
    for k, path in enumerate(plan["order"]):
        sess = recs[k].get("session") or ""
        assert not sess or os.path.realpath(sess).startswith(os.path.realpath(path)), \
            f"run {k}: ar_stats.json's session {sess} is not under the plan's {path}"
    lo, hi = stats["E_log10_range_Pa"]
    crops = CropSource(stub_from) if stub == "crops" else None
    lacking = [ob["instruction"] for ob in objects if len(ob["runs"]) < len(MODELS)]
    assert stub or not lacking, f"objects without a run of every model: {lacking}"

    pitch = (PAGE_H - MB - FOOT - ROWS0) / rows_max
    RH = pitch - ROW_GAP
    HI = RH - 2 * ROW_PAD - LBL_H                 # image box height = the cost stack's
    pitch_cost = (HI - CAP * META_FS - BAR_OFF - BAR_H) / 2
    assert pitch_cost > CAP * META_FS + BAR_OFF + BAR_H + 0.3, "cost lines would touch the bars"

    # the cost bars share one scale over every run of every page
    all_costs = [F.cost_of(r) for r in stats["runs"]]
    maxima = {k: max((c[k] or 0) for c in all_costs) for k in ("input", "output", "time")}

    # each run's panels (real renders, crops or synthetic) and labels
    def panels_of(ob, m):
        r = ob["runs"].get(m)
        if r is None:
            return None, {}
        rec = recs[r["index"]]
        if stub == "crops":
            got = crops.panels(ob["slug"], ob["instruction"], P, HI) or {}
            got = {p: v for p, v in got.items() if p in rec.get("_reached", [])}
            if not got and rec.get("_reached"):
                got = {p: np.full((8, 8, 3), 235, np.uint8) for p in rec["_reached"]}
            return rec, got
        return rec, real_panels(work, rec, P, HI, ground)

    sizes = page_split(objects, rows_max)
    start = 0
    written = []
    for pk, size in enumerate(sizes):
        rows = objects[start:start + size]
        name = f"{out}_p{pk + 1}"
        draw_page(name, rows, plan, panels_of, maxima, lo, hi, pitch, RH, HI, pitch_cost,
                  stub, ground)
        written.append(name)
        start += size
    write_tables(out, plan, recs, maxima)
    print(f"pages: {len(sizes)} ({', '.join(map(str, sizes))} rows; at most {rows_max}); "
          f"panels {P:.2f} x {HI:.2f} mm; rows {pitch:.2f} mm apart")
    for w in WARNINGS:
        print("WARNING:", w)
    return written


def draw_page(name, rows, plan, panels_of, maxima, lo, hi, pitch, RH, HI, pitch_cost, stub,
              ground):
    pg = F.Page(PAGE_W, PAGE_H)
    W = PAGE_W
    n = len(rows)
    row_tops = [ROWS0 + k * pitch for k in range(n)]
    table_bottom = row_tops[-1] + RH + ROW_GAP / 2
    texts = {"head": [], "cost": [], "card": [], "label": [], "key": [], "cell": [], "cat": []}

    # ---- rules ----------------------------------------------------------------------
    pg.line([ML, W - MR], [MT, MT], 0.6, F.INK)
    pg.line([ML, W - MR], [HEAD_RULE, HEAD_RULE], 0.35, F.INK)
    pg.line([ML, W - MR], [table_bottom, table_bottom], 0.6, F.INK)
    for k in range(n - 1):
        y = row_tops[k] + RH + ROW_GAP / 2
        if rows[k + 1]["slug"] != rows[k]["slug"]:
            pg.line([ML, W - MR], [y, y], 0.35, CAT_RULE)
        else:
            pg.line([X_CARD, W - MR], [y, y], 0.25, F.HAIR)

    # ---- heads ------------------------------------------------------------------------
    t1, t2 = MT + 10.4, MT + 12.3                 # the two title baselines
    pill_y = MT + 6.9
    texts["head"].append(pg.text(X_CARD, t2, "INSTRUCTION", HEAD_FS, weight="bold", color=F.GREY))
    for g, m in enumerate(MODELS):
        xs = group_x(g)
        gx0, gx1 = xs[0], xs[3] + P
        texts["head"].append(pg.text((gx0 + gx1) / 2, MT + 3.4, plan["labels"][m], GROUP_FS,
                                     weight="bold", ha="center"))
        pg.line([gx0 + 0.3, gx1 - 0.3], [MT + 4.6, MT + 4.6], 0.35, F.INK)   # cmidrule
        for c, (l1, l2) in enumerate(TITLES):
            cx = xs[c] + (COST_W if c == 0 else P) / 2
            if c == 0:
                texts["head"].append(pg.text(xs[0], t1, l1, HEAD_FS, weight="bold", color=F.GREY))
                texts["head"].append(pg.text(xs[0], t2, l2, HEAD_FS, weight="bold", color=F.GREY))
                continue
            texts["head"].append(draw_pill(pg, cx, pill_y, "AGENT", F.AGENT))
            texts["head"].append(pg.text(cx, t1, l1, HEAD_FS, weight="bold", color=F.GREY,
                                         ha="center"))
            texts["head"].append(pg.text(cx, t2, l2, HEAD_FS, weight="bold", color=F.GREY,
                                         ha="center"))

    # ---- category strip ------------------------------------------------------------------
    for _, grp in itertools.groupby(range(n), key=lambda j: rows[j]["slug"]):
        grp = list(grp)
        y0 = row_tops[grp[0]] + ROW_PAD
        y1 = row_tops[grp[-1]] + RH - ROW_PAD
        pg.line([ML + CAT_W - 0.9] * 2, [y0, y1], 0.5, CAT_RULE)
        label = rows[grp[0]]["category"].upper()
        fs = CAT_FS
        lines = [label]
        if pg.width(label, fs, weight="bold") > (y1 - y0) - 1.0:
            words = label.split(" ")
            cut = min(range(1, len(words)), key=lambda i: abs(
                pg.width(" ".join(words[:i]), fs, weight="bold") -
                pg.width(" ".join(words[i:]), fs, weight="bold")), default=None)
            if cut:
                lines = [" ".join(words[:cut]), " ".join(words[cut:])]
            longest = max(lines, key=lambda s: pg.width(s, fs, weight="bold"))
            fs = fit_fs(pg, longest, fs, (y1 - y0) - 1.0, weight="bold")
        xc = ML + (CAT_W - 0.9) / 2
        step = CAP * fs + 0.6
        for j, s_ in enumerate(lines):
            # rotated 90 deg: reads bottom to top; the lines stack left to right
            x_ = xc + (j - (len(lines) - 1) / 2) * step + CAP * fs / 2
            t = pg.ax.text(x_, (y0 + y1) / 2, s_, fontsize=fs, rotation=90, ha="center",
                           va="baseline", rotation_mode="anchor", weight="bold", color=F.GREY,
                           zorder=6)
            texts["cat"].append(t)

    # ---- rows -------------------------------------------------------------------------------
    probe_w = CARD_W - 2.4
    page_recs = []
    for k, ob in enumerate(rows):
        y0 = row_tops[k]
        # the prompt card: as tall as the row
        ct, cb = y0 + ROW_PAD, y0 + RH - ROW_PAD
        pg.ax.add_patch(FancyBboxPatch((X_CARD, ct), CARD_W, cb - ct,
                                       boxstyle="round,pad=0,rounding_size=1.0",
                                       fc="white", ec=F.CARD_STROKE, lw=0.5, zorder=4))
        s_ = f"“{F.typographic(ob['instruction'])}”"
        lines = F.balanced_wrap(pg, s_, probe_w, PROMPT_FS, style="italic")
        if len(lines) > 3:
            lines = greedy_wrap(pg, s_, probe_w, PROMPT_FS)
        assert len(lines) <= MAX_PROMPT_LINES, f"a prompt needs {len(lines)} lines: {s_!r}"
        mid = (ct + cb) / 2
        first = mid - PROMPT_PITCH * (len(lines) - 1) / 2 + CAP * PROMPT_FS / 2
        for j, line in enumerate(lines):
            texts["card"].append((k, (X_CARD, ct, X_CARD + CARD_W, cb),
                                  pg.text(X_CARD + CARD_W / 2, first + j * PROMPT_PITCH, line,
                                          PROMPT_FS, ha="center", style="italic")))
        itop = y0 + ROW_PAD
        for g, m in enumerate(MODELS):
            xs = group_x(g)
            rec, panels = panels_of(ob, m)
            if rec is not None:
                page_recs.append(rec)
            if rec is None:
                for c in (1, 2, 3):
                    texts["cell"].append(pg.text(xs[c] + P / 2, itop + HI / 2, "no run yet",
                                                 LABEL_FS, ha="center", va="center", color=F.GREY))
                continue
            # cost stack, exactly the image box's height
            c, cl = cost_lines(pg, rec)
            fr = [(c[q] or 0) / maxima[q] if maxima[q] else 0 for q in ("input", "output", "time")]
            cached = (c["cached"] or 0) / maxima["input"] if maxima["input"] else 0
            split = None
            g_ = ar_time.groups_of((rec["meta"].get("time_split") or {}).get("sections") or {})
            if sum(g_.values()) > 0:
                split = {q: g_.get(q, 0.0) / sum(g_.values()) for q, _ in ar_time.GROUPS}
            for j, line in enumerate(cl):
                yb = itop + CAP * META_FS + j * pitch_cost
                fs = fit_fs(pg, line, META_FS, COST_W)
                if fs < META_FS:
                    WARNINGS.append(f"{ob['instruction']!r} {m}: cost line {line!r} set at {fs} pt")
                texts["cost"].append((xs[0], pg.text(xs[0], yb, line, fs, color=F.GREY)))
                by = yb + BAR_OFF
                pg.ax.add_patch(Rectangle((xs[0], by), COST_W, BAR_H, fc=F.COST_TRACK, ec="none",
                                          zorder=3))
                if j == 0:
                    pg.ax.add_patch(Rectangle((xs[0], by), COST_W * fr[0], BAR_H,
                                              fc=F.COST_UNCACHED, ec="none", zorder=4))
                    pg.ax.add_patch(Rectangle((xs[0], by), COST_W * cached, BAR_H,
                                              fc=F.COST_CACHED, ec="none", zorder=5))
                elif j == 2 and split:
                    x_ = xs[0]
                    for q, _ in ar_time.GROUPS:
                        w_ = COST_W * fr[2] * split[q]
                        pg.ax.add_patch(Rectangle((x_, by), w_, BAR_H, fc=F.TIME_COLORS[q],
                                                  ec="none", zorder=4))
                        x_ += w_
                else:
                    pg.ax.add_patch(Rectangle((xs[0], by), COST_W * fr[j], BAR_H,
                                              fc=F.COST_NEUTRAL, ec="none", zorder=4))
            # image panels, or the cell saying why there is none
            st = str((rec.get("meta") or {}).get("status") or "").lower()
            why = "not reached" if (not st or st in OK_STATUS) else "failed"
            for c, p in enumerate(SHOWN, start=1):
                if p in panels:
                    pg.img(panels[p], xs[c], itop, P, HI)
                else:
                    texts["cell"].append(pg.text(xs[c] + P / 2, itop + HI / 2, why, LABEL_FS,
                                                 ha="center", va="center", color=F.GREY))
            # the label line: route | body load | grid cell
            lb = itop + HI + 1.65
            labs = {}
            if "painted" in panels and rec.get("body_source"):
                labs[1] = rec["body_source"]
            body = (rec.get("bc") or {}).get("body_load")
            if "bc" in panels and body:
                labs[2] = body_label(body)
            cell = F.fmt_cell(rec) if ground == "grid" or stub else None
            if cell and "materials" in panels:      # its own column: no label under a void cell
                labs[3] = cell
            for c, s_ in labs.items():
                icon = c == 3
                avail = P + COL_GAP - (F.CELL_ICON * 0.8 + 0.5 if icon else 0)
                fs = fit_fs(pg, s_, LABEL_FS, avail)
                if fs < LABEL_FS:
                    WARNINGS.append(f"{ob['instruction']!r} {m}: label {s_!r} set at {fs} pt")
                wd = pg.width(s_, fs) + (F.CELL_ICON * 0.8 + 0.5 if icon else 0)
                x_ = xs[c] + P / 2 - wd / 2
                if icon:
                    sq = F.CELL_ICON * 0.8
                    pg.ax.add_patch(Rectangle((x_, lb - sq), sq, sq, fc="none", ec=F.GRID_RGB,
                                              lw=0.5, zorder=6))
                    x_ += sq + 0.5
                texts["label"].append((k, c, pg.text(x_, lb, s_, fs, color=F.GREY)))

    # ---- keys ---------------------------------------------------------------------------------
    fy = table_bottom + 3.4
    x = X_CARD
    for colr, s_ in ((F.COST_CACHED, "Cached input"), (F.COST_UNCACHED, "Not cached"),
                     (F.COST_NEUTRAL, "Output tokens")):
        pg.ax.add_patch(Rectangle((x, fy - 1.15), 1.5, 0.95, fc=colr, ec="none", zorder=5))
        texts["key"].append(pg.text(x + 2.1, fy, s_, KEY_FS, color=F.GREY))
        x += 2.1 + pg.width(s_, KEY_FS) + 2.4
    x = X_CARD
    for q, _ in ar_time.GROUPS:
        pg.ax.add_patch(Rectangle((x, fy + 3.0 - 1.15), 1.5, 0.95, fc=F.TIME_COLORS[q], ec="none",
                                  zorder=5))
        texts["key"].append(pg.text(x + 2.1, fy + 3.0, F.TIME_KEY[q], KEY_FS, color=F.GREY))
        x += 2.1 + pg.width(F.TIME_KEY[q], KEY_FS) + 2.4
    key_left_end = max(pg.box(t)[2] for t in texts["key"])
    # supports & loads, centred on the page's middle group
    xs_mid = group_x(1)
    has_arrows = any((r.get("bc") or {}).get("arrows") for r in page_recs)
    seen = any(gg.get("seen_through") for r in page_recs
               for gg in ((r.get("bc") or {}).get("regions") or []))
    load_glyph_w = 1.9 + (0.7 + 3.2 if has_arrows else 0.0)
    wds = [pg.width(s_, KEY_FS) for s_ in ("Support", "Load")]
    total = (1.9 + 0.8) + (load_glyph_w + 0.8) + sum(wds) + 3.0
    x = xs_mid[0] + (GROUP_W - total) / 2
    x = max(x, key_left_end + 4.0)
    pg.ax.add_patch(Rectangle((x, fy - 1.6), 1.9, 1.9, fc=A.DIRICHLET, ec=F.INK, lw=0.35, zorder=5))
    texts["key"].append(pg.text(x + 1.9 + 0.8, fy, "Support", KEY_FS))
    sx = x
    x += 1.9 + 0.8 + wds[0] + 3.0
    pg.ax.add_patch(Rectangle((x, fy - 1.6), 1.9, 1.9, fc=A.NEUMANN, ec=F.INK, lw=0.35, zorder=5))
    if has_arrows:
        ln_, sw_ = 3.2, 0.46
        hw_, hl_ = 2.55 * sw_, 0.36 * 3.2
        x0_, y0_ = x + 2.6, fy - 0.65
        pg.ax.add_patch(Polygon([(x0_, y0_ - sw_ / 2), (x0_ + ln_ - hl_, y0_ - sw_ / 2),
                                 (x0_ + ln_ - hl_, y0_ - hw_ / 2), (x0_ + ln_, y0_),
                                 (x0_ + ln_ - hl_, y0_ + hw_ / 2), (x0_ + ln_ - hl_, y0_ + sw_ / 2),
                                 (x0_, y0_ + sw_ / 2)], closed=True, fc=A.LOAD_ARROW,
                                ec="none", zorder=5))
    texts["key"].append(pg.text(x + load_glyph_w + 0.8, fy, "Load", KEY_FS))
    if seen:
        tint = np.array(A.CLAY) * (1 - F.SEE_THROUGH) + np.array(A.NEUMANN) * F.SEE_THROUGH
        pg.ax.add_patch(Rectangle((sx, fy + 3.0 - 1.6), 1.9, 1.9, fc=tint, ec=F.INK, lw=0.35,
                                  zorder=5))
        texts["key"].append(pg.text(sx + 1.9 + 0.8, fy + 3.0, "Load on a far face, seen through",
                                    KEY_FS))
    # the grid cell's square
    sq = F.CELL_ICON * 0.8
    yk = fy + (6.0 if seen else 3.0)
    pg.ax.add_patch(Rectangle((sx, yk - sq - 0.15), sq, sq, fc="none", ec=F.GRID_RGB, lw=0.5,
                              zorder=5))
    texts["key"].append(pg.text(sx + sq + 0.8, yk, "Floor grid cell", KEY_FS, color=F.GREY))
    # Young's modulus: one log scale over every page
    xs_r = group_x(2)
    bx, bw, by, bh = xs_r[1] + 2.0, GROUP_W - COST_W - COL_GAP - 9.0, fy - 1.6, 1.0
    F.hbar(pg, bx, by, bw, bh, [(k_ / (len(A.E_RGB) - 1), c_) for k_, c_ in enumerate(A.E_RGB)])
    F.KEY_FS = KEY_FS                              # e_ticks measures its labels at KEY_FS
    for d, xt, s_ in F.e_ticks(pg, lo, hi, bx, bw, W - MR):
        pg.line([xt, xt], [by + bh, by + bh + (0.7 if s_ else 0.4)], 0.35, F.INK, z=5)
        if s_:
            texts["key"].append(pg.text(xt, by + bh + 2.8, s_, KEY_FS, ha="center"))
    texts["key"].append(pg.text(bx + bw + 0.8, by + bh, "Log", KEY_FS, color=F.GREY))
    texts["key"].append(pg.text(bx - 1.2, by + bh, "Young’s modulus", KEY_FS, color=F.GREY,
                                ha="right"))
    if stub:
        texts["key"].append(pg.text(W - MR, PAGE_H - MB + 1.0,
                                    f"stub panels ({stub}): layout test, not the figure",
                                    KEY_FS, color="#c0392b", ha="right"))

    check_page(pg, texts, row_tops, RH, HI, table_bottom)
    pg.fig.savefig(f"{name}.pdf", dpi=600)
    pg.fig.savefig(f"{name}.png", dpi=F.DPI)
    import matplotlib.pyplot as plt
    plt.close(pg.fig)
    print(f"wrote {name}.pdf / .png ({n} rows)")


def greedy_wrap(pg, text, width, fs):
    """Four lines at most for the narrow card: balanced like F.balanced_wrap's two-line
    split, extended: the split of the words into k lines (k = 1..4) with the shortest
    longest line, for the fewest k that fits."""
    words = text.split(" ")
    w = lambda i, j: pg.width(" ".join(words[i:j]), fs, style="italic")  # noqa: E731
    for k in range(1, MAX_PROMPT_LINES + 1):
        best = None
        for cuts in itertools.combinations(range(1, len(words)), k - 1):
            b = (0, *cuts, len(words))
            m = max(w(b[q], b[q + 1]) for q in range(k))
            if m <= width and (best is None or m < best[0]):
                best = (m, [" ".join(words[b[q]:b[q + 1]]) for q in range(k)])
        if best:
            return best[1]
    return [text]


def check_page(pg, texts, row_tops, RH, HI, table_bottom):
    """No text overlaps another, leaves the page, its card or its column; keys in the footer."""
    pg.fig.canvas.draw()
    allt = (texts["head"] + [t for _, t in texts["cost"]] + [t for *_, t in texts["card"]] +
            [t for *_, t in texts["label"]] + texts["key"] + texts["cell"] + texts["cat"])
    bbs = []
    for t in allt:
        if not t.get_text():
            continue
        b = pg.box(t)
        assert b[0] >= -1e-3 and b[2] <= PAGE_W + 1e-3 and b[1] >= -1e-3 and b[3] <= PAGE_H + 1e-3, \
            f"text off the page: {t.get_text()!r}"
        bbs.append((b, t.get_text()))
    for a_ in range(len(bbs)):
        p_ = bbs[a_][0]
        for b_ in range(a_ + 1, len(bbs)):
            q_ = bbs[b_][0]
            if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]:
                raise AssertionError(f"texts overlap: {bbs[a_][1]!r} / {bbs[b_][1]!r}")
    for x0, t in texts["cost"]:
        assert pg.box(t)[2] <= x0 + COST_W + 1e-6, f"cost text leaves its column: {t.get_text()!r}"
    for k, (x0, t0, x1, b1), t in texts["card"]:
        b = pg.box(t)
        assert b[0] >= x0 + 0.4 and b[2] <= x1 - 0.4 and b[1] >= t0 + 0.3 and b[3] <= b1 - 0.3, \
            f"prompt leaves its card: {t.get_text()!r}"
    for k, c, t in texts["label"]:
        b = pg.box(t)
        assert b[1] >= row_tops[k] + ROW_PAD + HI - 0.05, f"label on its image: {t.get_text()!r}"
        assert b[3] <= row_tops[k] + RH + ROW_GAP / 2 - 0.1, f"label crosses the hairline: {t.get_text()!r}"
    for t in texts["head"]:
        b = pg.box(t)
        assert b[1] >= MT + 0.6 and b[3] <= HEAD_RULE - 0.4, f"head text crowds a rule: {t.get_text()!r}"
    for t in texts["key"]:
        assert pg.box(t)[1] >= table_bottom + 0.4, f"key crowds the bottom rule: {t.get_text()!r}"


def write_tables(out, plan, recs, maxima):
    """``<out>_tokens.csv/.md`` and ``<out>_time.csv`` over every run, the model as a column
    (agentic_results' tables, one row per run)."""
    runs = []
    for ob in plan["objects"]:
        for m in MODELS:
            r = ob["runs"].get(m)
            if r is not None:
                rec = dict(recs[r["index"]])
                rec.setdefault("meta", {})
                runs.append(rec)
    F.write_table(runs, out, maxima)
    F.write_time_table(runs, out)


# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("plan")
    for m in MODELS:
        a1.add_argument(f"--{m}", required=True, help=f"the {m} batch root")
    a1.add_argument("--work", required=True)
    a1.add_argument("--only", nargs="*", help="category slugs (default: all)")
    a1.add_argument("--list", default=str(CAT.OBJECTS))
    a1.add_argument("--allow-missing", action="store_true",
                    help="stub pages only: plan with the runs that exist")
    a2 = sub.add_parser("restrict")
    a2.add_argument("work")
    a3 = sub.add_parser("compose")
    a3.add_argument("work")
    a3.add_argument("--out", default="agentic_results_compare")
    a3.add_argument("--ground", default=os.environ.get("AR_GROUND", "grid"),
                    choices=("none", "shadow", "grid"))
    a3.add_argument("--rows", type=int, default=ROWS_MAX, help="rows per page at most")
    a3.add_argument("--stub-panels", choices=("crops", "synthetic"))
    a3.add_argument("--stub-from", default=os.path.dirname(HERE),
                    help="the folder holding agentic_results_<slug>.png (crops)")
    a = ap.parse_args()
    if a.cmd == "plan":
        os.makedirs(a.work, exist_ok=True)
        roots = {}
        for m in MODELS:      # relative roots: from here, else from the repository root
            r = getattr(a, m)
            roots[m] = r if os.path.isabs(r) or os.path.isdir(r) else os.path.join(str(CAT.REPO), r)
        plan = make_plan(roots, a.only, a.list, a.allow_missing)
        json.dump(plan, open(os.path.join(a.work, "compare_plan.json"), "w"), indent=1)
        with open(os.path.join(a.work, "runs.txt"), "w") as fh:
            fh.write(" ".join(plan["order"]) + "\n")
        print(f"plan: {len(plan['objects'])} objects, {len(plan['order'])} runs -> "
              f"{a.work}/compare_plan.json, runs.txt")
    elif a.cmd == "restrict":
        p = os.path.join(a.work, "ar_stats.json")
        st = json.load(open(p))
        for r in st["runs"]:
            r["render_panels"] = list(SHOWN)
        json.dump(st, open(p, "w"), indent=1, default=float)
        print(f"restrict: {len(st['runs'])} runs render {', '.join(SHOWN)} only")
    else:
        if a.stub_panels:
            plan = json.load(open(os.path.join(a.work, "compare_plan.json")))
            if any(len(ob["runs"]) < len(MODELS) for ob in plan["objects"]):
                print("stub: objects without a run of every model are drawn as 'no run yet'")
        compose(a.work, a.out, a.ground, a.stub_panels, a.stub_from, a.rows)


if __name__ == "__main__":
    main()
