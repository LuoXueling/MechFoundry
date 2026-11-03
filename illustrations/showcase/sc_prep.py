"""Stage 1 (env mechfoundry-warp117, small GPU use): showcase rows -> renderable surfaces.

    python sc_prep.py [--rows-json FILE] [--work DIR]

The showcase (variation) figure has three rows, each a list of agentic runs laid
out like ``output/agentic_batch/<run>/`` (``workflow/<session>/``, ``metrics.json``,
``prompt.txt``), as experiments/agentic_batch writes them:

  shape      N realizations of ONE instruction (run_realizations.sh: run k gets
             FEM_OPERATOR_SEED=S+k, so the same words make another body)
                                     input panel: bc (grey body, supports, loads)
  material   one body and loading, N per-part material layouts (run_material_sweep.sh)
                                     input panel: materials (E in this figure's own
                                     cheese-lime-watermelon-herb map,
                                     SC_E_ANCHORS; dark part seams)
  loading    one body and material, N load cases (run_bc_sweep.sh)
                                     input panel: bc

Every item is prepped by agentic_results' own ``ar_prep.prep_run`` and written by
its ``write_run_files``, so the files are exactly what ``render_ar.py`` draws for
that figure (``ar_<i>_*.vtk`` + ``ar_stats.json``, ``i`` numbering the items of
all rows in order).  Each record names the panels it needs (``render_panels``):
its input panel, the solve, and, when the input panel carries arrows and support
markers, the plain ``painted`` panel too -- the undeformed outline under the solve
is that panel's silhouette (the BC panel's would trace the arrows).

WHERE THE ROWS COME FROM (first that is set wins, per row):

  SC_SHAPE_RUNS / SC_MATERIAL_RUNS / SC_LOADING_RUNS
      space-separated run folders or globs (absolute, or relative to SC_DATA_ROOT,
      default the checkout holding output/).  A row given this way is real data:
      its placeholder flag is dropped.
  --rows-json / SC_ROWS_JSON
      ``{"rows": [{"key": "shape"|"material"|"loading", "placeholder": bool,
      "runs": [{"path": <folder or glob>, "E_override": ...}, ...],
      "finished_only": bool, "fallback": [<runs>], "fallback_note": str}]}``
      (default ``sc_rows_placeholder.json``).  ``finished_only`` keeps the runs
      whose metrics.json says status ok; when none is left, ``fallback`` is used.

At most MAX_ITEMS runs per row are drawn; the row's total is recorded.

THE ROW'S PROMPT (the label column's card), read from the run and batch files:
  shape      the instruction in the runs' prompt.txt ('The user's whole instruction
             is: "..."'), else metrics.json's; a card only if every run had the
             same one.
  material / loading
             the sweep's authoring prompt, ``<runs>/_batches/<batch>/authoring/
             prompt.txt`` (ab_sweep.sh), condensed to a short gist of a few card
             lines (``condense_authoring``: the words kept are its own, each cut
             an ellipsis).  A placeholder
             row has no sweep batch: its card is ab_helper.author_prompt() for the
             source session -- the text a sweep from that session would send -- and
             the record says so.
  Each sweep row also records ``calls`` (``sweep_calls``): the two kinds of Claude
  call that made it, each with its card -- the authoring call (the condensed
  authoring prompt and its cost from authoring/claude_result_<k>.json, every try
  summed, time from duration_ms) and the run agents (their prompt.txt condensed by
  ``condense_run_prompt``, the case stood in for by '[one of the N cases]').
  sc_figure draws a block per kind.

PLACEHOLDER ONLY: ``E_override`` replaces the input panel's Young's modulus
(``{"uniform_Pa": E}``, or ``{"default_Pa": E, "segments_Pa": {"<segment id>": E}}``
over boundary_conditions/segments.npz).  It does NOT re-solve: the deformed panel
stays the run's own solve, which is why such a row is flagged as a placeholder.

SCALES.  shape and material: every item its own displacement factor and |u| colour
range (ar_prep's rule), so the deformed envelopes come out alike.  loading: one
factor and one |u| range for the row when the items' max|u| differ by less than
SHARED_MAX_RATIO (the cases then compare directly), else per item.
"""

import argparse
import copy
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "agentic_results")
sys.path.insert(0, AR)
import ar_prep  # noqa: E402
import ar_runs  # noqa: E402
import ar_style as A  # noqa: E402

# -- this figure's Young's-modulus map ("cheese, lime, watermelon, herb") --------------
# The showcase's material row is the one place where E is the subject: nine material
# layouts of one hamburger side by side.  User, 2026-10-16: nothing on the burger may
# look blue, the colours should look delicious, and then (rejecting the one-hue
# "golden crust" butter -> espresso browns) "I need multiple colors to make the
# difference more visible, not blue or this deep brown".  So this figure alone uses
# the designer's P4 (output/showcase_cmap_candidates/food_multi/designer_multi.md):
# one fresh-produce hue per material class, each class on a nearly FLAT band, with
# the hue changes only in the decades no layer uses.  Straight CIELAB lines between
# consecutive anchors in log10 E, held past the ends; every other agentic-results
# figure keeps the shared "steel" map.
#   soft        100 kPa - 80 MPa  bun #fcedc0 -> cheese #fdd359   gel, foam, rubber, foam core
#   polymers    1 - 3 GPa         lime #a7c83e -> lettuce #98c13b HDPE, PP, PC, nylon, balsa
#   wood/stone  9 - 70 GPa        watermelon #e67f95 -> tomato pink #e3738a
#                                                                 ice, oak, pine, bone, concrete,
#                                                                 granite, aluminium, glass
#   metals      110 - 200 GPa     herb #438632 -> pickle #3c7e31  titanium, copper, steel
#   The blends between bands (lime -> pink through a short khaki at 3-9 GPa, pink ->
#   green through a greyish band at 70-110 GPa) show only on the key's bar.
#   * Every neighbour changes hue family (yellow, green, pink, green), and L* still
#     falls monotonically 94 -> 86 -> 76 -> 73 -> 65 -> 62 -> 50 -> 47, so greyscale and
#     every colour-blind simulation keep the soft -> stiff order.  No blue, purple or
#     brown; the one red-family colour is a cool pink (h ~8 deg), not the |u|
#     orange -> red.
#   * History of this map: ice-blue -> steel (shared); gold -> blue -> indigo; + lilac;
#     blue -> petrol; "golden crust" browns; then this P4.
#   * WHAT COLOUR-BLIND READERS LOSE.  Under deuteranopia / protanopia the greens and
#     the pink collapse to khaki and grey: the four classes survive as four LIGHTNESS
#     steps (pale yellow -> mustard -> grey-beige -> olive; neighbouring centres 11.5,
#     18.9, 20.5 DeltaE00), so order and separability stay but "four different hues"
#     is lost.  No blue-free food palette can keep E apart from |u| point by point for
#     them either (P4: the metal greens map onto |u|'s olive-brown, 0.9; the pink near
#     its mid tones, 3.4), so check_e_map holds a NORMAL-vision floor against |u| and
#     none for colour-blind vision.  For those readers the two fields are told apart by
#     layout -- E only in the input column, |u| only in the deformed column --, by
#     their separate keys (the E bar under the material row, the |u| bar in the
#     footer), and by field shape (E constant per layer with dark seams, |u| a smooth
#     gradient over a cream body).  Tritanopia keeps the groups distinct.
#   * Floors check_e_map asserts (DeltaE00 normal / worst colour-blind, measured):
#     polymer, wood/stone and metal bands >= 20 from the |u| orange -> dark-red ramp in
#     normal vision (21.6 at worst, the pink); neighbouring band centres >= 18 / >= 10
#     (20.1 / 11.5 at worst, soft -> polymers; then 60.3 / 18.9, 62.4 / 20.5); the
#     support teal >= 15 normal over the whole ramp (22.3) and >= 6 colour-blind over
#     the four bands (6.5, the pink under protanopia).  Changed: the teal colour-blind
#     floor now applies to the bands only -- the unused lime -> pink blend at 3-9 GPa
#     passes 1.9 from teal under protanopia, and no layer has that colour.  Not
#     floored, as before: the soft band against the deformed body's cream (11.3 / 5.7;
#     light = soft puts it there) and the light half of |u| (10.9 / 3.5); the
#     load-orange swatch is the |u| mid colour, under the same colour-blind argument.
SC_E_ANCHORS = ((5.0, (94.0, -2.0, 24.0)),    # 100 kPa: soft, bun          #fcedc0
                (7.9, (86.0, 2.0, 64.0)),     # 80 MPa:  soft, cheese       #fdd359
                (9.0, (76.0, -30.0, 62.0)),   # 1 GPa:   polymers, lime     #a7c83e
                (9.48, (73.0, -33.0, 60.0)),  # 3 GPa:   polymers, lettuce  #98c13b
                (9.95, (65.0, 42.0, 6.0)),    # 9 GPa:   wood/stone, watermelon #e67f95
                (10.85, (62.0, 46.0, 8.0)),   # 70 GPa:  wood/stone, tomato pink #e3738a
                (11.04, (50.0, -38.0, 38.0)), # 110 GPa: metals, herb       #438632
                (11.3, (47.0, -37.0, 35.0)))  # 200 GPa: metals, pickle     #3c7e31
SC_E_HEX = ("#fcedc0", "#fdd359", "#a7c83e", "#98c13b", "#e67f95", "#e3738a", "#438632",
            "#3c7e31")                         # every anchor, in order
# the material classes, each on its band of anchors (log10 Pa spans of the layers)
SC_E_CLUSTERS = {"soft": (5.0, 7.9), "polymers": (9.0, 9.48), "wood/stone": (9.95, 10.85),
                 "metals": (11.04, 11.3)}
SC_E_BAR_STOPS = 65                            # the key's bar: the map sampled this often


def _lab2rgb(lab):
    """CIELAB (D65) -> sRGB in [0, 1], (..., 3); asserts the colours are in gamut."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    m = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]]))
    lin = xyz @ m.T
    assert lin.min() > -1e-3 and lin.max() < 1 + 1e-3, "showcase E map leaves the sRGB gamut"
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def sc_e_map(lo, hi):
    """The showcase E map on the figure's [lo, hi] log10-Pa scale: t in [0, 1] (0 = lo)
    -> (..., 3) sRGB, the CIELAB anchors interpolated in log10 E (held past the ends)."""
    xs = np.array([a[0] for a in SC_E_ANCHORS])
    labs = np.array([a[1] for a in SC_E_ANCHORS])

    def e_rgb(t):
        e = lo + np.clip(np.asarray(t, float), 0.0, 1.0) * (hi - lo)
        return _lab2rgb(np.stack([np.interp(e, xs, labs[:, k]) for k in range(3)], axis=-1))
    return e_rgb


def sc_e_stops(lo, hi):
    """The map as the key's evenly spaced RGB stops (ar_figure.e_key ``stops``)."""
    return [tuple(c) for c in sc_e_map(lo, hi)(np.linspace(0, 1, SC_E_BAR_STOPS))]


def check_e_map(lo=SC_E_ANCHORS[0][0], hi=SC_E_ANCHORS[-1][0]):
    """Hard: the anchors are SC_E_HEX, the map is in gamut, its lightness falls
    monotonically over [lo, hi] in normal vision and every simulated CVD, and the
    DeltaE00 floors of the block comment above hold (|u| in normal vision only)."""
    sys.path.insert(0, os.path.join(HERE, "..", "overview"))
    import ov_palette_check as PC
    got = tuple(PC._hex(_lab2rgb(a[1])) for a in SC_E_ANCHORS)
    assert got == SC_E_HEX, f"showcase E anchors drifted: {got}"
    rgb = sc_e_map(lo, hi)(np.linspace(0, 1, 201))
    for kind in (None, *PC.M):
        L = [PC.rgb2lab(PC.simulate(c, kind) if kind else c)[0] for c in rgb]
        assert all(y <= x + 1e-9 for x, y in zip(L, L[1:])), \
            f"showcase E map lightness not monotone ({kind or 'normal vision'})"
    f = sc_e_map(SC_E_ANCHORS[0][0], SC_E_ANCHORS[-1][0])
    at = lambda e: f((np.asarray(e, float) - SC_E_ANCHORS[0][0])        # noqa: E731
                     / (SC_E_ANCHORS[-1][0] - SC_E_ANCHORS[0][0]))
    lab = lambda c, k=None: PC.rgb2lab(PC.simulate(c, k) if k else c)   # noqa: E731

    def dmin(A, B, kinds):
        return min(PC.de2000(lab(x, k), lab(y, k)) for k in kinds for x in A for y in B)
    cvd = tuple(PC.M)
    # |u|: render_ar's transfer function, orange (SOLVE_MID) -> dark red (SOLVE_TOP) in RGB
    u_hi = [np.array(A.SOLVE_MID) + (np.array(A.SOLVE_TOP) - np.array(A.SOLVE_MID)) * t
            for t in np.linspace(0, 1, 30)]
    for name in ("polymers", "wood/stone", "metals"):
        d = dmin(at(np.linspace(*SC_E_CLUSTERS[name], 8)), u_hi, (None,))
        assert d >= 20.0, f"showcase E {name} only {d:.1f} DeltaE00 from the |u| orange->red ramp"
    teal = [np.array(A.DIRICHLET)]
    bands = np.concatenate([at(np.linspace(*span, 8)) for span in SC_E_CLUSTERS.values()])
    dn, dc = dmin(rgb[::4], teal, (None,)), dmin(bands, teal, cvd)
    assert dn >= 15.0 and dc >= 6.0, f"showcase E map near the support teal ({dn:.1f} / {dc:.1f})"
    names = list(SC_E_CLUSTERS)
    for a_, b_ in zip(names, names[1:]):
        ca, cb = (at([sum(SC_E_CLUSTERS[n]) / 2]) for n in (a_, b_))
        dn, dc = dmin(ca, cb, (None,)), dmin(ca, cb, cvd)
        assert dn >= 18.0 and dc >= 10.0, f"showcase E {a_} / {b_} too close ({dn:.1f} / {dc:.1f})"
    print(f"showcase E map: cheese {SC_E_HEX[1]} (soft), lime {SC_E_HEX[2]} (polymers), "
          f"watermelon {SC_E_HEX[4]} (wood/stone), herb {SC_E_HEX[6]} (metals); L* monotone "
          f"over 10^{lo:g}-10^{hi:g} Pa in all visions; DeltaE00 floors hold")


REPO = ar_runs.REPO
# relative run paths resolve against the checkout that holds output/ (a git worktree
# of this repo has none: its runs live in the main checkout); SC_DATA_ROOT overrides
DATA_ROOT = os.environ.get("SC_DATA_ROOT") or (
    REPO if os.path.isdir(os.path.join(REPO, "output")) else "/home/xl2254/implicit_optimize")
MAX_ITEMS = 9                     # sc_figure's 3 x 3 grid per row
SHARED_MAX_RATIO = 10.0
ROW_KEYS = ("shape", "material", "loading")
ROW_DEFAULTS = {
    "shape": {"input_panel": "bc", "view": "per_item", "env": "SC_SHAPE_RUNS", "scale": "per_item"},
    "material": {"input_panel": "materials", "view": "shared", "env": "SC_MATERIAL_RUNS",
                 "scale": "per_item"},
    "loading": {"input_panel": "bc", "view": "shared", "env": "SC_LOADING_RUNS",
                "scale": "shared_if_close"},
}


# -- the row spec ----------------------------------------------------------------------
def _json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def _expand(entries):
    """Run specs, a glob path expanded into one spec per match (sorted)."""
    out = []
    for e in entries:
        e = {"path": e} if isinstance(e, str) else dict(e)
        path, _, label = e["path"].partition("::")      # 'folder::label' (kept in the record)
        p = path if os.path.isabs(path) else os.path.join(DATA_ROOT, path)
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        assert hits, f"no run folder matches {e['path']!r}"
        out += [{**e, "path": h, **({"label": label} if label else {})} for h in hits]
    return out


def finished(run_dir):
    return _json(os.path.join(run_dir, "metrics.json")).get("status") == "ok"


def load_rows(rows_json):
    doc = json.load(open(rows_json))
    rows = {r["key"]: dict(r) for r in doc["rows"]}
    for key in ROW_KEYS:
        row = rows.setdefault(key, {"key": key, "runs": []})
        for k, v in ROW_DEFAULTS[key].items():
            row.setdefault(k, v)
        env = os.environ.get(row["env"], "").strip()
        row["fallback_used"] = None
        if env:
            row["runs"] = _expand(env.split())
            row["placeholder"] = False
            row["source"] = f"${row['env']}"
        else:
            row["source"] = os.path.relpath(rows_json, HERE)
            try:
                row["runs"] = _expand(row["runs"])
            except AssertionError:
                row["runs"] = []
            if row.get("finished_only"):
                row["runs"] = [r for r in row["runs"] if finished(r["path"])]
            if not row["runs"] and row.get("fallback"):
                row["runs"] = _expand(row["fallback"])
                row["fallback_used"] = row.get("fallback_note") or "fallback runs"
        assert row["runs"], f"row {key!r} has no runs"
        row["total"] = int(row.get("total") or len(row["runs"]))
        row["runs"] = row["runs"][:MAX_ITEMS]
    return [rows[k] for k in ROW_KEYS]


# -- the row's prompt --------------------------------------------------------------------
INSTR_RE = re.compile(r"The user's whole instruction is: \"(.+?)\"\.", re.S)


def realization_instruction(run_dir, meta):
    """The instruction the run was given: prompt.txt's quoted instruction, else
    metrics.json's.  (text, source)"""
    p = os.path.join(run_dir, "prompt.txt")
    if os.path.exists(p):
        m = INSTR_RE.search(open(p, encoding="utf-8").read())
        if m:
            return m.group(1).strip(), "prompt.txt"
    if meta.get("instruction"):
        return meta["instruction"], "metrics.json instruction"
    return None, None


def batch_of(run_dir):
    """The batch folder (``<runs>/_batches/<id>/``) whose ar_runs.txt, summary or
    log names this run, or None."""
    name = os.path.basename(os.path.normpath(run_dir))
    for b in sorted(glob.glob(os.path.join(os.path.dirname(os.path.normpath(run_dir)), "_batches", "*"))):
        for f in ("ar_runs.txt", "summary.csv", "batch.log"):
            p = os.path.join(b, f)
            if os.path.exists(p) and name in open(p, encoding="utf-8", errors="replace").read():
                return b
    return None


def condense_authoring(text, kind):
    """The authoring prompt as a card: a short gist in its own words, three cuts, each
    an ellipsis (the figure is an illustration of what the call was asked, not the
    record; the whole prompt is ``prompt_source``).  Kept, in order:
      1. the first sentence's head, up to the demonstration clause and without a
         parenthetical ('You are writing 9 material assignments' / 'You are writing
         9 load cases');
      2. the head of the rule that frees the cases ('Assign materials freely' /
         'Choose supports and loads freely');
      3. that rule's purpose sentence ('The point is variety, not realism.' /
         'The point is variety.').
    It reads e.g. 'You are writing 9 material assignments ... Assign materials freely
    ... The point is variety, not realism.' -- 4 card lines.  A piece the text
    lacks is left out (the first sentence stands alone when nothing matches)."""
    text = text.strip()
    m = re.match(r"(.+?)(?: for a demonstration\b.*?)?\.(?=\s|$)", text, re.S)
    # a parenthetical in the head is cut too ('9 load cases (supports and loads)'):
    # the next piece names them, and the cut sits at the ellipsis that follows
    parts = [re.sub(r"\s*\([^)]*\)", "", (m.group(1) if m else text.split("\n")[0]).strip())]
    head = r"Assign materials freely" if kind == "material" else r"Choose supports and loads freely"
    r = re.search(r"^- (" + head + r")\b(.*)$", text, re.M)
    if r:
        parts.append(r.group(1))
        v = re.search(r"The point is variety[^.]*\.", r.group(2))
        if v:
            parts.append(v.group(0))
    return " … ".join(parts)


# -- a sweep row's two kinds of Claude call ---------------------------------------------
# A material or BC sweep (ab_sweep.sh) makes two kinds of headless Claude call: ONE
# authoring call that writes all N cases (authoring/prompt.txt, one or more tries
# authoring/claude_result_<k>.json), then N run agents, each told to simulate one case
# (<run>/prompt.txt, the same text but for the case it carries).  sc_figure gives each
# kind its own prompt card and cost bars.
LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (authoring.*)$", re.M)


def authoring_call(batch_dir):
    """The authoring call's cost, summed over every try (claude_result_<k>.json):
    input = input + cache reads + cache writes, cached = cache reads (agentic_results'
    cost_of), output tokens, and the wall time.  The time is the CLI's own
    ``duration_ms`` summed over the tries; the batch log's span from the first
    'authoring ... (try 1' line to 'authoring: ok' is recorded beside it as a check
    (whole seconds; it also holds the gaps between tries).  None when the batch has no
    authoring result."""
    ad = os.path.join(batch_dir, "authoring")
    files = sorted(glob.glob(os.path.join(ad, "claude_result_*.json")),
                   key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1)))
    if not files:
        return None
    tot = {"input": 0, "cached": 0, "output": 0}
    dur, cost_usd, models = 0.0, 0.0, set()
    for p in files:
        d = _json(p)
        u = d.get("usage") or {}
        read, written = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
        tot["input"] += (u.get("input_tokens") or 0) + read + written
        tot["cached"] += read
        tot["output"] += u.get("output_tokens") or 0
        dur += (d.get("duration_ms") or 0) / 1000.0
        cost_usd += d.get("total_cost_usd") or 0.0
        models |= set((d.get("modelUsage") or {}).keys())
    log_s, log_tries = None, None
    lp = os.path.join(batch_dir, "batch.log")
    if os.path.exists(lp):
        from datetime import datetime
        hits = LOG_RE.findall(open(lp, encoding="utf-8", errors="replace").read())
        starts = [t for t, m in hits if "(try " in m]
        ends = [t for t, m in hits if m.startswith("authoring: ok")]
        log_tries = len(starts)
        if starts and ends:
            f_ = "%Y-%m-%d %H:%M:%S"
            log_s = (datetime.strptime(ends[-1], f_) - datetime.strptime(starts[0], f_)).total_seconds()
    time_s, src = (dur, "duration_ms") if dur > 0 else (log_s, "batch.log span")
    return {**tot, "time": time_s, "time_source": src, "log_span_s": log_s, "tries": len(files),
            "log_tries": log_tries, "cost_usd": cost_usd, "models": sorted(models),
            "files": [os.path.relpath(p, DATA_ROOT) for p in files],
            # no workflow step or solve runs inside it: its whole wall time is agent time
            "groups": {"agent": time_s or 0.0, "workflow": 0.0, "solver": 0.0, "overhead": 0.0}}


RUN_KIND = {"material": "material assignment", "loading": "load case"}


def condense_run_prompt(text, kind, n):
    """A sweep run's prompt.txt as a card, condensed like ``condense_authoring``: a
    short gist in its own words, each cut an ellipsis.  The skill boilerplate (read
    SKILL.md, the session's files, the numbered steps, the reply) is cut -- the shape
    row's card leaves it out too -- and the case itself (its name, description and
    JSON, the only part that differs between the runs) is stood in for by '[one of
    the N cases]' in brackets, an editorial insertion, not the prompt's words.  Kept:
    'This run is a <kind> variant of a finished simulation' (its object, 'of "a
    hamburger"', is cut: the tiles beside the card show it), then the case, then the
    head of the rule that has it run as given (material: 'Apply them as given';
    loading: 'Run this case as given').  It reads e.g. 'This run is a
    load-case variant of a finished simulation ... [one of the 9 cases] ... Run this
    case as given ...' -- some 5 card lines.  None when a sentence is missing."""
    m1 = re.search(r"This run is an? [\w-]+ variant of a finished simulation(?= of \"[^\"]+\"\.)", text)
    m2 = re.search(r"The user's whole instruction for this run is the " + re.escape(RUN_KIND[kind]), text)
    m3 = re.search(r"Apply them as given" if kind == "material"
                   else r"Run this case as given", text)
    if not (m1 and m2 and m3):
        return None
    return " ".join([m1.group(0), "…", f"[one of the {n} cases]", "…", m3.group(0), "…"])


def sweep_calls(run_dirs, kind):
    """The two kinds of call of a sweep row (material / loading), or None:
    {"authoring": {"prompt", "prompt_source", "cost"}, "runs": {"prompt",
    "prompt_source", "distinct"}}.  The run card is the condensed prompt.txt of the
    row's runs; ``distinct`` counts the different condensed texts (1 when the runs
    differ only in their case, as ab_sweep writes them)."""
    if kind not in RUN_KIND or not run_dirs:
        return None
    b = batch_of(run_dirs[0])
    out = {}
    ap = os.path.join(b, "authoring", "prompt.txt") if b else None
    if ap and os.path.exists(ap):
        out["authoring"] = {"prompt": condense_authoring(open(ap, encoding="utf-8").read(), kind),
                            "prompt_source": os.path.relpath(ap, DATA_ROOT),
                            "cost": authoring_call(b)}
    texts = []
    for d in dict.fromkeys(run_dirs):               # a run listed twice counts once
        p = os.path.join(d, "prompt.txt")
        if os.path.exists(p):
            texts.append(condense_run_prompt(open(p, encoding="utf-8").read(), kind,
                                             len(set(run_dirs))))
    got = [t for t in texts if t]
    if got:
        out["runs"] = {"prompt": got[0], "distinct": len(set(got)), "missing": len(texts) - len(got),
                       "prompt_source": "the runs' prompt.txt, condensed (sc_prep.condense_run_prompt)"}
    return out or None


def refresh_cards(row):
    """Re-condense a sweep row's two cards from their sources (the authoring
    prompt.txt, the runs' prompt.txt) with the current ``condense_*``, so a
    compose-only build of an older ``_work`` prints the current cards; the costs are
    left as recorded.  True when a card changed (sc_figure then writes the record
    back)."""
    calls = row.get("calls") or {}
    changed = False
    a = calls.get("authoring")
    if a and a.get("prompt_source"):
        p = a["prompt_source"] if os.path.isabs(a["prompt_source"]) else os.path.join(DATA_ROOT, a["prompt_source"])
        if os.path.exists(p):
            t = condense_authoring(open(p, encoding="utf-8").read(), row["key"])
            changed |= t != a.get("prompt") or t != row.get("prompt")
            a["prompt"] = row["prompt"] = t
    if calls.get("runs") and row.get("run_dirs"):
        fresh = (sweep_calls(row["run_dirs"], row["key"]) or {}).get("runs")
        if fresh:
            changed |= fresh["prompt"] != calls["runs"].get("prompt")
            calls["runs"].update(fresh)
    return changed


def template_authoring(kind, session, n):
    """PLACEHOLDER rows: the authoring prompt a sweep from this session would send
    (experiments/agentic_batch/ab_helper.author_prompt, imported read-only)."""
    try:
        sys.path.insert(0, os.path.join(REPO, "experiments", "agentic_batch"))
        import ab_helper
        return ab_helper.author_prompt("material" if kind == "material" else "bc", session, n, [])
    except Exception as exc:  # noqa: BLE001 -- the row then shows its runs' instruction
        print(f"(no authoring template: {type(exc).__name__}: {exc})")
        return None


def row_prompt(row, recs, runs):
    """(card text, its source, distinct prompts) for one row."""
    if row["key"] == "shape":
        got = [realization_instruction(r["path"], rec["meta"]) for r, rec in zip(row["runs"], recs)]
        texts = {t for t, _ in got if t}
        if len(texts) == 1:
            return texts.pop(), got[0][1], 1
        return None, "the runs' instructions differ", len(texts)
    b = batch_of(row["runs"][0]["path"])
    ap = os.path.join(b, "authoring", "prompt.txt") if b else None
    if ap and os.path.exists(ap):
        return condense_authoring(open(ap, encoding="utf-8").read(), row["key"]), \
            os.path.relpath(ap, DATA_ROOT), 1
    t = template_authoring(row["key"], runs[0].session, len(row["runs"]))
    if t:
        return condense_authoring(t, row["key"]), \
            "PLACEHOLDER: ab_helper.author_prompt() for the source session (no sweep batch yet)", 1
    instr = {rec["meta"].get("instruction") for rec in recs}
    return (instr.pop(), "metrics.json instruction", 1) if len(instr) == 1 else (None, None, len(instr))


# -- E override (placeholder rows only) ------------------------------------------------
def segments_of(session, n_faces):
    p = os.path.join(session, "boundary_conditions", "segments.npz")
    if not os.path.exists(p):
        return None
    seg = np.load(p)["segment_ids"]
    return seg if len(seg) == n_faces else None


def override_E(out, session, ov):
    V, F = out["V"], out["F"]
    if "uniform_Pa" in ov:
        return np.full(len(V), float(ov["uniform_Pa"]))
    seg = segments_of(session, len(F))
    assert seg is not None, f"{session}: no segments.npz over the surface's {len(F)} faces"
    Ef = np.full(len(F), float(ov["default_Pa"]))
    for sid, e in (ov.get("segments_Pa") or {}).items():
        Ef[seg == int(sid)] = float(e)
    E = np.full(len(V), float(ov["default_Pa"]))
    for k in range(3):                       # a vertex takes one of its faces' moduli
        E[F[:, k]] = Ef
    return E


# -- main ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows-json", default=os.environ.get("SC_ROWS_JSON") or
                    os.path.join(HERE, "sc_rows_placeholder.json"))
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    args = ap.parse_args()
    check_e_map()
    rows = load_rows(args.rows_json)
    os.makedirs(args.work, exist_ok=True)
    for old in glob.glob(os.path.join(args.work, "ar_*")):
        os.remove(old)

    sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))
    sys.path.insert(0, REPO)
    import torch
    from core.remeshing import octree_hex_interpolate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch, octree_hex_interpolate, device)

    cache = {}            # (session, view) -> (rec, out): a repeated run is prepped once
    recs, outs, row_stats = [], [], []
    for row in rows:
        shared_view = None
        items, row_runs = [], []
        for spec in row["runs"]:
            i = len(recs)
            run = ar_runs.load_run({"path": spec["path"]})
            if row["view"] == "shared" and shared_view is not None:
                run.view_dir = shared_view
            key = (run.session, tuple(run.view_dir) if run.view_dir else None)
            if key not in cache:
                cache[key] = ar_prep.prep_run(i, run, args.work, ctx)
            rec, out = copy.deepcopy(cache[key])
            rec["index"] = i
            if row["view"] == "shared" and shared_view is None and rec.get("view"):
                shared_view = rec["view"]["dir"]
            if out is not None and spec.get("E_override"):
                out["E"] = override_E(out, run.session, spec["E_override"])
                rec["E_source"] = f"PLACEHOLDER override {spec['E_override']}"
                rec["E_surface_Pa"] = (float(out["E"].min()), float(out["E"].max()))
            # part seams on the E map: the session's segmentation, where E varies
            if out is not None and row["input_panel"] == "materials" and out.get("E") is not None \
                    and np.ptp(out["E"]) > 0:
                seg = segments_of(run.session, len(out["F"]))
                if seg is not None:
                    out["segid"] = seg
                    rec["segments"] = int(len(np.unique(seg)))
            panels = [row["input_panel"], "solve"]
            if row["input_panel"] == "bc":
                panels.append("painted")      # its silhouette is the undeformed outline
            rec["render_panels"] = panels
            var = _json(os.path.join(spec["path"], "variant.json")) or rec["meta"].get("variant") or {}
            rec["showcase"] = {"row": row["key"], "run_dir": spec["path"], "label": spec.get("label"),
                               "E_override": spec.get("E_override"), "variant": var or None}
            recs.append(rec)
            outs.append(out)
            items.append(i)
            row_runs.append(run)
            print(f"[{i}] {row['key']}: {run.get('session_id')} {run.get('instruction')!r}  "
                  f"notes={rec['notes']}")

        # one displacement factor and |u| range for a row of one body under N load
        # cases, when their peaks are within SHARED_MAX_RATIO of each other
        shared = None
        if row["scale"] == "shared_if_close":
            got = [(recs[i], outs[i]) for i in items if outs[i] is not None and "u" in outs[i]]
            if got:
                um = [o["umax"] for _, o in got]
                ratio = max(um) / max(min(um), 1e-300)
                ok = ratio < SHARED_MAX_RATIO
                if ok:
                    f = min(r["scale"]["factor"] for r, _ in got)
                    for r, o in got:
                        o["scale"] = f
                        r["scale"] = dict(r["scale"], factor=f, own_factor=r["scale"]["factor"])
                        r["solve"]["umax_colour"] = max(um)     # render_ar's |u| ramp top
                shared = {"applied": ok, "umax_ratio": ratio,
                          "rule": f"shared when the items' max|u| ratio < {SHARED_MAX_RATIO:g}: the "
                                  "smallest item factor, the largest max|u| as the colour top"}
        for i in items:
            recs[i]["showcase"]["shared_scale"] = bool(shared and shared["applied"])
        prompt, psrc, n_prompts = row_prompt(row, [recs[i] for i in items], row_runs)
        row_stats.append({"key": row["key"], "items": items, "placeholder": bool(row.get("placeholder")),
                          "input_panel": row["input_panel"], "view": row["view"],
                          "source": row["source"], "fallback_used": row["fallback_used"],
                          "total": row["total"], "prompt": prompt, "prompt_source": psrc,
                          "distinct_prompts": n_prompts, "shared_scale": shared,
                          "run_dirs": [s["path"] for s in row["runs"]],
                          "calls": None if row.get("placeholder") else
                          sweep_calls([s["path"] for s in row["runs"]], row["key"])})

    # one E colour scale: the material row's E maps (the only panels that show E)
    mat_items = [i for r in row_stats if r["input_panel"] == "materials" for i in r["items"]]
    lo, hi = ar_prep.e_log_range([outs[i] for i in mat_items])
    if lo is None:
        lo, hi = ar_prep.e_log_range(outs)
    for rec, out in zip(recs, outs):
        if out is not None:
            ar_prep.write_run_files(rec, out, args.work, lo, hi, e_rgb=sc_e_map(lo, hi))   # cheese, lime, watermelon, herb
            if out.get("segid") is not None:      # render_ar's id pass -> the part seams
                ar_prep.write_polydata(os.path.join(args.work, f"ar_{rec['index']}_segid.vtk"),
                                       out["V"], out["F"], cell_ids=out["segid"] + 1)
    stats = {"E_log10_range_Pa": [lo, hi], "view_dir_default": A.VIEW_DIR, "runs": recs,
             "rows": row_stats, "rows_json": os.path.abspath(args.rows_json)}
    json.dump(stats, open(os.path.join(args.work, "ar_stats.json"), "w"), indent=1, default=float)
    print(f"wrote {args.work}/ar_stats.json")
    for r in row_stats:
        if r["placeholder"]:
            print(f"PLACEHOLDER row {r['key']!r} ({r['source']}"
                  f"{'; ' + r['fallback_used'] if r['fallback_used'] else ''})")


if __name__ == "__main__":
    main()
