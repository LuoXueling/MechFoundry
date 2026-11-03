"""Where an agent run's wall clock went: fine sections that add up to it exactly, and the few
groups the figure's time bar shows.  Plain Python.

    split_run(run_dir) -> {"sections": {key: s}, "groups": {key: s}, "wall_clock_s": s, "notes": [...]}

Sources, all in the batch script's run folder (output/agentic_batch/<run>/):
  metrics.json                 the wall clock;
  claude_session/*.jsonl       when each of the agent's runner commands was issued and when its
                               result came back (every transcript line is timestamped);
  workflow/<sid>/operator/driver_log.jsonl
                               the runner's own log: each card it answered, with the time it
                               finished and ``elapsed_s``, and each stop.

The agent's time is everything outside its runner commands, so the sections sum to the wall
clock by construction.  Inside a command the card log says what ran.  The rest of a command is
the runner's: its start-up before the first card, GPU selection and idle-GPU waits between
cards, and after the last card the stop's pictures and summaries -- except after a solve, where
it is the solver's fallbacks and cross-checks and the result summary, counted with the solve.
The same accounting measured the five runs of 2026-10-08 to within 1 s of their wall clocks
(output/agentic_batch/timing/).
"""

import glob
import json
import os
import re
from datetime import datetime

# The figure's bar: few groups (the key has room for few).  key, label
GROUPS = (("agent", "Agent"), ("workflow", "Workflow steps"), ("solver", "Solver"),
          ("overhead", "Runner overhead"))

# The CSV: every section, its group, and what it holds.
SECTIONS = (
    ("agent", "agent", "Agent: reading, reasoning, writing",
     "the agent (the LLM) reading the skill and the pictures, reasoning, and writing the case "
     "brief, the plan edits and the judgement: all time outside its runner commands, the CLI's "
     "start-up included"),
    ("geometry_drawing", "workflow", "Geometry: SDXL drawings",
     "drawing the object with SDXL (draw_image), every drawing including rejected ones"),
    ("geometry_generation", "workflow", "Geometry: 3-D generation",
     "TripoSG, Hunyuan3D and CAD-Coder generations (text, drawing or photograph to 3-D), retries "
     "included"),
    ("geometry_revision", "workflow", "Geometry: prompt revision",
     "the API revision of a CAD-Coder description"),
    ("geometry_built", "workflow", "Geometry: built bodies and web photographs",
     "building a body with drive.py blank and staging it; searching licensed web photographs"),
    ("setup_scale", "workflow", "Setup: scaling and description",
     "accepting the body, the API description of its views, and scaling it to its real size"),
    ("setup_segmentation", "workflow", "Setup: segmentation",
     "segmenting the surface into parts (PartField or shape diameter) and naming them"),
    ("setup_regions", "workflow", "Setup: support and load regions",
     "synthesizing, previewing, revising and approving the support and load regions"),
    ("setup_values_material", "workflow", "Setup: values and material",
     "the boundary-condition values and the material (uniform, inferred or per part)"),
    ("paint", "workflow", "Painting",
     "painting the body's surface colours (Hunyuan3D paint), repaints included, and the runner's "
     "wait for a paint it started in the background"),
    ("solve", "solver", "Solve: FEM solves",
     "the solver runs the workflow launched (all attempts)"),
    ("solve_after", "solver", "Solve: fallbacks, cross-checks, result summary",
     "after a solve: the runner's solver fallbacks and cross-checks (solve-backend included), the "
     "result summary and the field pictures"),
    ("overhead_startup", "overhead", "Runner: start-up and GPU choice before the first card",
     "each runner command's process start, imports and plan checks, then sampling the GPUs and "
     "waiting for an idle one, before it answers its first card"),
    ("overhead_between", "overhead", "Runner: GPU choice and waits between cards",
     "between two cards of one command: sampling the GPUs, choosing and waiting for an idle one, "
     "rewinds"),
    ("overhead_stop", "overhead", "Runner: stop pictures and summaries",
     "after the last card of a command that stops elsewhere than at a solve: rendering the stop's "
     "pictures and summaries, and printing them"),
    ("overhead_commands", "overhead", "Runner: start, judge, finish, status",
     "the session start (API probes, workers), judge, finish and the other bookkeeping commands"),
)
KEYS = tuple(s[0] for s in SECTIONS)
GROUP_OF = {s[0]: s[1] for s in SECTIONS}
RUNNER = re.compile(r"(drive|inspect_session|web_images)\.py\s+([\w-]+)?")


def groups_of(sections):
    """The bar groups' seconds from the sections (so a regrouping needs no new split)."""
    return {g: round(sum(v for k, v in sections.items() if GROUP_OF.get(k) == g), 1) for g, _ in GROUPS}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _card_section(event):
    kind, action = event.get("kind"), event.get("action")
    answer = event.get("answer") if isinstance(event.get("answer"), dict) else {}
    if action == "texture_surface":
        return "paint"
    if kind == "geometry_input":
        if action == "draw_image" or answer.get("action") == "draw_image":
            return "geometry_drawing"
        if answer.get("revise_prompt"):
            return "geometry_revision"
        if answer.get("asset_kind") == "surface":
            return "geometry_built"
        return "geometry_generation"
    if kind in ("geometry_texture", "geometry_physics"):
        return "setup_scale"
    if kind == "bc_input":
        return "setup_segmentation" if action == "preview_segmentation" else "setup_regions"
    if kind in ("bc_values", "material_input"):
        return "setup_values_material"
    if kind == "shape_opt_settings":
        return "solve" if action != "finish" else "overhead_commands"
    return "setup_values_material"


def _commands(transcript_files):
    """The agent's runner commands: issued and returned times, script, subcommand, result text."""
    calls, out = {}, []
    for f in transcript_files:
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") not in ("user", "assistant") or not d.get("timestamp"):
                continue
            content = (d.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            t = _ts(d["timestamp"])
            for b in content:
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    m = RUNNER.search((b.get("input") or {}).get("command") or "")
                    if m:
                        calls[b["id"]] = {"t0": t, "t1": t, "script": m.group(1), "sub": m.group(2) or "",
                                          "text": ""}
                        out.append(calls[b["id"]])
                elif b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                    c = calls[b["tool_use_id"]]
                    c["t1"] = t
                    body = b.get("content")
                    c["text"] = body if isinstance(body, str) else " ".join(
                        x.get("text", "") for x in body or [] if isinstance(x, dict))
    return out


def split_run(run_dir):
    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    wall = float(metrics["wall_clock_s"])
    files = sorted(glob.glob(os.path.join(run_dir, "claude_session", "**", "*.jsonl"), recursive=True))
    files = [f for f in files if "/subagents/" not in f] or files
    logs = glob.glob(os.path.join(run_dir, "workflow", "**", "operator", "driver_log.jsonl"), recursive=True)
    dlog = []
    for p in logs:
        dlog += [json.loads(l) for l in open(p) if l.strip()]
    cards = sorted((e for e in dlog if "kind" in e and isinstance(e.get("time"), (int, float))),
                   key=lambda e: e["time"])
    sec = dict.fromkeys(KEYS, 0.0)
    notes = []
    for c in _commands(files):
        dur = max(0.0, c["t1"] - c["t0"])
        if c["script"] == "web_images":
            sec["geometry_built"] += dur
        elif c["sub"] == "solve-backend":
            sec["solve_after"] += dur
        elif c["sub"] == "blank":
            sec["geometry_built"] += dur
        elif c["script"] != "drive" or c["sub"] not in ("run", "answer", "rewind"):
            sec["overhead_commands"] += dur
        else:
            stop = re.search(r'"stop":\s*"(\w+)"', c["text"])
            cursor, first = c["t0"], True
            for e in (e for e in cards if c["t0"] - 0.5 <= e["time"] <= c["t1"] + 0.5):
                start = max(cursor, e["time"] - float(e.get("elapsed_s") or 0.0))
                start = min(start, c["t1"])
                gap = max(0.0, start - cursor)
                # The wait for the background paint (driver_log paint_wait_s) comes before the
                # card's own clock starts: it is painting, not the runner's.
                wait = min(gap, float(e.get("paint_wait_s") or 0.0))
                sec["paint"] += wait
                sec["overhead_startup" if first else "overhead_between"] += gap - wait
                end = min(max(e["time"], start), c["t1"])
                sec[_card_section(e)] += end - start
                cursor, first = end, False
            tail = max(0.0, c["t1"] - cursor)
            if first:
                sec["overhead_startup"] += tail          # no card answered: refused or errored early
            elif stop and stop.group(1) == "solve":
                sec["solve_after"] += tail
            else:
                sec["overhead_stop"] += tail
    runner = sum(v for k, v in sec.items() if k != "agent")
    sec["agent"] = wall - runner
    if sec["agent"] < 0:
        notes.append(f"runner commands ({runner:.0f} s) exceed the wall clock ({wall:.0f} s)")
        sec["agent"] = 0.0
    if not logs:
        notes.append("no driver_log.jsonl: every runner command counts as start-up overhead")
    return {"sections": {k: round(v, 1) for k, v in sec.items()}, "groups": groups_of(sec),
            "wall_clock_s": wall, "notes": notes}
