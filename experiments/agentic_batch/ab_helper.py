"""JSON and file handling for run_batch.sh.  Standard library only (any python3 >= 3.8).

Subcommands:
  slug TEXT                       a filesystem-safe slug of an instruction
  prompt --template F --worktree W --object O
                                  the initial instruction with the object inserted
  finalize --run-dir D ...        after one `claude -p` exits: extract the result,
                                  copy the transcripts and the workflow session(s),
                                  write metrics.json; prints one status line
  summary --batch-dir B RUN...    the batch table (summary.csv, summary.json) and
                                  ar_runs.txt (the run folders, for AR_RUNS=)

Token counts.  The CLI's result has two usage reports, and they differ:
``usage`` is the main session's own API calls only, while ``modelUsage`` (per
model) also counts every sub-agent the session spawned.  Measured on a session
that spawned one general-purpose sub-agent: usage = 4 in / 154 out / 45 408 cache
read / 20 747 cache creation; the main transcript summed gives exactly that, the
sub-agent's transcript adds 4 / 87 / 26 328 / 27 751, and modelUsage is the sum
(8 / 241 / 71 736 / 48 498).  So metrics.json's headline token fields are
modelUsage summed over models, which already includes sub-agents; nothing is
added on top.  The per-part numbers (main ``usage``, sub-agent transcripts) are
recorded beside it as a cross-check, and when a run was killed before the CLI
wrote its result the headline falls back to main + sub-agent transcript sums.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import unicodedata

TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "output_tokens")
# modelUsage spells the same four counts in camelCase.
MODEL_USAGE_KEYS = {"input_tokens": "inputTokens", "cache_read_input_tokens": "cacheReadInputTokens",
                    "cache_creation_input_tokens": "cacheCreationInputTokens",
                    "output_tokens": "outputTokens"}
SID = r"[0-9a-f]{12}"
DRIVE_CMD = re.compile(r"drive\.py\s+(start|run|blank|answer|rewind|judge|finish|status|solve-backend|anchor)"
                       r"(?:\s+(" + SID + r"))?")


def slug(text, limit=48):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"^(a|an|the)\s+", "", text.strip().lower())
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:limit].rstrip("-") or "run")


# ---------------------------------------------------------------- transcripts

def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def transcript_tokens(files):
    """Sum of assistant-message usage over transcript files, each message counted
    once (Claude Code writes one line per content block and repeats the usage on
    each), the same rule as illustrations/agentic_results/ar_runs.py."""
    per_message = {}
    for f in files:
        for rec in read_jsonl(f):
            msg = rec.get("message") if rec.get("type") == "assistant" else None
            if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict):
                continue
            seen = per_message.setdefault((f, msg.get("id") or rec.get("uuid")),
                                          {k: 0 for k in TOKEN_FIELDS})
            for k in TOKEN_FIELDS:
                seen[k] = max(seen[k], int(msg["usage"].get(k) or 0))
    tot = {k: sum(m[k] for m in per_message.values()) for k in TOKEN_FIELDS}
    tot["messages"] = len(per_message)
    return tot


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text_of(c.get("text", c.get("content", ""))) if isinstance(c, dict) else str(c)
                         for c in content)
    return str(content or "")


def scan_transcript(files):
    """Every Bash command and its result, in order, over the transcript files.
    Returns [(command, result_text)]; a result persisted to a file because it was
    too large is read from that file too."""
    calls, results = [], {}
    for f in files:
        for rec in read_jsonl(f):
            msg = rec.get("message")
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    calls.append((block.get("id"), str((block.get("input") or {}).get("command", ""))))
                elif block.get("type") == "tool_result":
                    text = _text_of(block.get("content"))
                    extra = rec.get("toolUseResult")
                    if isinstance(extra, dict):
                        text += "\n" + str(extra.get("stdout", "")) + "\n" + str(extra.get("stderr", ""))
                    for saved in re.findall(r"saved to:\s*(\S+)", text):
                        try:
                            with open(saved, encoding="utf-8", errors="replace") as fh:
                                text += "\n" + fh.read()
                        except OSError:
                            pass
                    results[block.get("tool_use_id")] = results.get(block.get("tool_use_id"), "") + text
    return [(cmd, results.get(tid, "")) for tid, cmd in calls]


def _session_reach(session):
    """How far a session got: 3 a solve run, 2 resolved BCs, 1 a geometry, 0 nothing."""
    solve = os.path.join(session, "solve")
    if os.path.isdir(solve) and any(n.startswith("api_fem_forward-") for n in os.listdir(solve)):
        return 3
    if os.path.exists(os.path.join(session, "boundary_conditions", "resolved.json")):
        return 2
    geometry = os.path.join(session, "geometry")
    return 1 if os.path.isdir(geometry) and os.listdir(geometry) else 0


def workflow_sessions(calls, final_text, workflow_root):
    """The workflow session id(s) this Claude session created, from its own transcript.

    Runs are parallel, so the newest folder means nothing.  The id is read from
    the result of the agent's own `drive.py start` call (it prints
    `session_id: <id>` on stderr and `"session_id": "<id>"` in its JSON); ids
    the agent then passed to other drive.py commands are a fallback for a start
    whose output was lost.  When the agent started more than one session, the
    primary one is the one that got furthest (a solve, resolved BCs, a geometry)
    when exactly one did, else the one its final reply names last, else the one
    most of its later commands drove, else the last started.
    """
    started, used = [], []
    for cmd, out in calls:
        for sub, sid in DRIVE_CMD.findall(cmd):
            if sub == "start":
                for h in re.findall(r"session_id[\"']?\s*[:=]\s*[\"']?(" + SID + ")", out):
                    if h not in started:
                        started.append(h)
            elif sid:
                used.append(sid)
    exists = lambda s: os.path.isdir(os.path.join(workflow_root, s))
    found = [s for s in started if exists(s)]
    how = "drive.py start output in the transcript"
    if not found:
        found = [s for s in dict.fromkeys(used) if exists(s)]
        how = "session ids passed to drive.py commands in the transcript (no start output found)"
    if not found:
        return None, [], "not found"
    # A session the agent abandoned stops short of the one it finished, and a reply that
    # names the abandoned one after the finished one ("a first session, X, was abandoned"),
    # as a Haiku run's did (batch 20261013-205539, usb-stick), fooled the names-last rule
    # below.  So the one session that got furthest wins outright.
    if len(found) > 1:
        reach = {s: _session_reach(os.path.join(workflow_root, s)) for s in found}
        top = max(reach.values())
        if list(reach.values()).count(top) == 1:
            best = max(found, key=reach.get)
            return best, found, how + "; primary = the one session that got furthest"
    named = [s for s in re.findall(SID, final_text or "") if s in found]
    if named:
        return named[-1], found, how + "; primary named in the final reply"
    if len(found) > 1:
        counts = {s: used.count(s) for s in found}
        best = max(found, key=lambda s: (counts[s], found.index(s)))
        return best, found, how + "; primary = most driven of several"
    return found[0], found, how


def effort_observed(files):
    """The effort level each assistant message was made at, {level: messages}.

    Claude Code stamps every assistant line of a transcript with `"effort"` (and
    `"perTurnEffort"`); the stream's init event does not carry it.  Measured: with
    `--effort low` / `medium` / `high` the lines say low / medium / high.  One
    message is counted once although it is written as several lines.

    Thinking tokens: modelUsage's `thinkingTokens` is part of `outputTokens`, not
    extra.  A one-message call at --effort high reported output 660 with 196
    thinking, the transcript's single message (a thinking block and a text block)
    also says output 660, and the CLI's costUSD is reproduced to the last digit by
    2 $/M input + 10 $/M output (660, not 856) + 0.20 $/M cache read + 4 $/M 1-h
    cache write; the same rates reproduce all five batch runs' costUSD exactly.
    (The rates are fitted from those costs, not taken from a price list.)
    """
    seen = {}
    for f in files:
        for rec in read_jsonl(f):
            if rec.get("type") != "assistant" or not rec.get("effort"):
                continue
            msg = rec.get("message") or {}
            seen[(f, msg.get("id") or rec.get("uuid"))] = rec["effort"]
    out = {}
    for level in seen.values():
        out[level] = out.get(level, 0) + 1
    return out or None


# ---------------------------------------------------------------- the case as solved

# The pictures and fields a figure takes from a session copy, relative to it, when present.
FIGURE_FILES = {
    "views": "geometry/views.png",
    "body_glb": "geometry/physical.glb",
    "textured_views": "operator/texture_views.png",
    "segments": "operator/segments.png",
    "regions": "operator/regions.png",
    "parts_montage": "materials/parts/montage.png",
    "displacement_png": "operator/displacement_magnitude.png",
    "stress_png": "operator/stress_von_mises.png",
    "displacement_glb": "solve/displacement_magnitude.glb",
    "stress_glb": "solve/stress_von_mises.glb",
}


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def case_summary(session):
    """What the session's case/case.json solved: the material (per part: each part's E, nu, rho
    from the part field's manifest in this copy, and the E contrast), the held and loaded regions
    and the body loads; plus the figure files present.  None without a case."""
    case = _load_json(os.path.join(session, "case", "case.json"))
    if not isinstance(case, dict):
        return None
    material = case.get("material") or {}
    out = {"material_mode": material.get("mode")}
    if material.get("mode") in ("parts", "vomp"):
        manifest = _load_json(os.path.join(session, "materials", "parts" if material.get("mode") == "parts" else "vomp",
                                           "field", "manifest.json")) or {}
        parts = (manifest.get("provenance") or {}).get("parts")
        if parts is None:
            parts = (_load_json(os.path.join(session, "materials", "parts", "materials.json")) or {}).get("parts")
        out["parts"] = parts
        e_range = (manifest.get("ranges") or {}).get("youngs_modulus")
        if e_range and e_range[0]:
            out["youngs_modulus_range"] = e_range
            out["youngs_modulus_contrast"] = round(e_range[1] / e_range[0], 3)
    else:
        out.update({k: material.get(k) for k in ("youngs_modulus", "poisson_ratio", "density")})
    bcs = case.get("boundary_conditions") or {}
    out["held"] = [{"name": d.get("name"), "components": d.get("components")} for d in bcs.get("dirichlet") or []]
    out["loaded"] = [{"name": n.get("name"), **{k: n[k] for k in ("traction", "force", "moment", "torque") if k in n}}
                     for n in bcs.get("neumann") or []]
    loads = case.get("loads") or {}
    out["body_loads"] = {k: loads.get(k) for k in ("acceleration", "body_force") if loads.get(k)}
    # The seeds the body was made with, as the plan last had them: a realization's batch seed is
    # only where the agent left a seed at 0 (the first real batch's agents set their own), so the
    # label's seed alone does not say which body was generated.
    plan = _load_json(os.path.join(session, "operator", "plan.json")) or {}
    geometry, texture = plan.get("geometry") or {}, plan.get("texture") or {}
    if isinstance(geometry, dict) and isinstance(texture, dict):
        out["generation_seeds"] = {
            "geometry": (geometry.get("generation_settings") or {}).get("seed")
            if isinstance(geometry.get("generation_settings"), dict) else None,
            "image": geometry.get("image_seed"), "texture": texture.get("texture_seed")}
    run = sorted(glob.glob(os.path.join(session, "solve", "api_fem_forward-*")))
    files = {k: v for k, v in FIGURE_FILES.items() if os.path.exists(os.path.join(session, v))}
    if run:
        files["solve_vtu_dir"] = os.path.relpath(os.path.join(run[-1], "vtu"), session)
    out["figure_files"] = files
    return out


# ---------------------------------------------------------------- finalize

def _iso_seconds(a, b):
    try:
        return round(float(b) - float(a), 3)
    except (TypeError, ValueError):
        return None


def extract_result(stream_path, json_path):
    """The CLI's final result object.  With --output-format stream-json it is the
    last `{"type": "result"}` line of the stream (the same object --output-format
    json prints); it is written out as claude_result.json."""
    result = None
    if os.path.exists(json_path):
        try:
            result = json.load(open(json_path))
        except (OSError, json.JSONDecodeError):
            result = None
    if result is None and stream_path and os.path.exists(stream_path):
        for rec in read_jsonl(stream_path):
            if isinstance(rec, dict) and rec.get("type") == "result":
                result = rec
        if result is not None:
            with open(json_path, "w") as fh:
                json.dump(result, fh, indent=1)
    return result


def copy_claude_session(projects_dir, claude_sid, dest):
    """<claude sid>.jsonl and the <claude sid>/ folder beside it (subagents/*.jsonl with
    their .meta.json, tool-results/) into dest.  Returns (main transcript, sub-agent
    transcripts) as paths in dest."""
    os.makedirs(dest, exist_ok=True)
    hits = glob.glob(os.path.join(projects_dir, "*", claude_sid + ".jsonl"))
    main = None
    if hits:
        src = hits[0]
        main = os.path.join(dest, os.path.basename(src))
        shutil.copy2(src, main)
        side = src[:-len(".jsonl")]
        if os.path.isdir(side):
            shutil.copytree(side, os.path.join(dest, claude_sid), symlinks=True, dirs_exist_ok=True)
    subs = sorted(glob.glob(os.path.join(dest, claude_sid, "subagents", "**", "*.jsonl"), recursive=True))
    return main, subs, (hits[0] if hits else None)


def cmd_finalize(a):
    run = a.run_dir
    stream = os.path.join(run, "claude_stream.jsonl")
    result = extract_result(stream, os.path.join(run, "claude_result.json")) or {}
    claude_sid = result.get("session_id") or a.claude_session_id
    main, subs, src = copy_claude_session(a.projects_dir, claude_sid, os.path.join(run, "claude_session"))

    main_tok = transcript_tokens([main]) if main else None
    sub_tok = transcript_tokens(subs) if subs else {k: 0 for k in TOKEN_FIELDS} | {"messages": 0}
    model_usage = result.get("modelUsage") or {}
    if model_usage:
        tokens = {k: sum(int(m.get(MODEL_USAGE_KEYS[k]) or 0) for m in model_usage.values())
                  for k in TOKEN_FIELDS}
        tokens_source = "CLI result modelUsage, summed over models (includes sub-agents)"
    elif main_tok:
        tokens = {k: main_tok[k] + sub_tok[k] for k in TOKEN_FIELDS}
        tokens_source = "transcripts: main + sub-agents (the CLI wrote no result)"
    else:
        tokens, tokens_source = {k: None for k in TOKEN_FIELDS}, "none"
    model = (max(model_usage, key=lambda m: (model_usage[m].get("outputTokens") or 0))
             if model_usage else None)
    if not model and main:
        for rec in read_jsonl(main):
            mm = (rec.get("message") or {}).get("model") if rec.get("type") == "assistant" else None
            if mm and not mm.startswith("<"):
                model = mm
    model = model or a.model

    files = ([main] if main else []) + subs
    calls = scan_transcript(files)
    observed = effort_observed(files)
    final_text = result.get("result") or ""
    wsid, wsids, how = workflow_sessions(calls, final_text, a.workflow_root)
    wf_dest = os.path.join(run, "workflow")
    os.makedirs(wf_dest, exist_ok=True)
    copied = []
    for s in wsids:
        shutil.copytree(os.path.join(a.workflow_root, s), os.path.join(wf_dest, s),
                        symlinks=True, dirs_exist_ok=True)
        copied.append(s)
    runner = {}
    for cmd, _ in calls:
        for sub, _sid in DRIVE_CMD.findall(cmd):
            runner[sub] = runner.get(sub, 0) + 1
    last_stop = None
    if wsid:
        try:
            last_stop = json.load(open(os.path.join(wf_dest, wsid, "operator", "last_run.json"))).get("stop")
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        # `judge --finish` (and `finish`) end the session without a `run`, so last_run.json still
        # says "solve": the driver log's last stop is the session's own
        try:
            for line in open(os.path.join(wf_dest, wsid, "operator", "driver_log.jsonl")):
                entry = json.loads(line)
                if isinstance(entry, dict) and entry.get("stop"):
                    last_stop = entry["stop"]
        except (OSError, json.JSONDecodeError):
            pass

    # The instruction `start` stored (brief_snapshots/instruction.json; use_anchor.json before
    # 2026-10-11) against the list line this run was given: a batch passes it in
    # FEM_OPERATOR_INSTRUCTION.  The metrics keep their names (anchor_instruction*).
    anchor_instruction = anchor_matches = anchor_given = None
    if wsid:
        for name in ("instruction.json", "use_anchor.json"):
            try:
                record = json.load(open(os.path.join(wf_dest, wsid, "brief_snapshots", name)))
            except (OSError, json.JSONDecodeError):
                continue
            anchor_instruction = record.get("instruction")
            anchor_given = record.get("instruction_given")
            anchor_matches = anchor_instruction == a.instruction
            break

    exit_code = int(a.exit_code)
    if a.timed_out == "1":
        status = "timeout"
    elif not result:
        status = "no_result"
    elif exit_code == 0 and not result.get("is_error") and result.get("subtype") == "success":
        status = "ok"
    else:
        status = "cli_error"
    if final_text:
        with open(os.path.join(run, "agent_reply.md"), "w") as fh:
            fh.write(final_text)

    dur = result.get("duration_ms")
    metrics = {
        "run_name": os.path.basename(os.path.normpath(run)),
        "instruction": a.instruction,
        "model": model,
        "model_requested": a.model,
        "session_id": wsid,
        "session_ids": wsids,
        "session_id_found_by": how,
        "claude_session_id": claude_sid,
        **tokens,
        "tokens_source": tokens_source,
        # Part of output_tokens, not on top of it (see effort_observed's note).
        "thinking_tokens": (sum(int(m.get("thinkingTokens") or 0) for m in model_usage.values())
                            if model_usage else None),
        "effort": a.effort or None,
        "effort_observed": observed,
        "system_note": (open(os.path.join(run, "system_note.txt")).read().strip()
                        if os.path.exists(os.path.join(run, "system_note.txt")) else None),
        "main_session_usage": {k: (result.get("usage") or {}).get(k) for k in TOKEN_FIELDS}
                              if result.get("usage") else None,
        "main_transcript_tokens": main_tok,
        "subagent_transcript_tokens": sub_tok,
        "subagent_transcripts": len(subs),
        "subagents_spawned": (result.get("subagent_stats") or {}).get("spawned"),
        "tokens_by_model": model_usage or None,
        "cost_usd": result.get("total_cost_usd"),
        "wall_clock_s": _iso_seconds(a.t0, a.t1),
        "cli_duration_s": round(dur / 1000, 3) if isinstance(dur, (int, float)) else None,
        "cli_duration_api_s": (round(result["duration_api_ms"] / 1000, 3)
                               if isinstance(result.get("duration_api_ms"), (int, float)) else None),
        "num_turns": result.get("num_turns"),
        "status": status,
        "exit_code": exit_code,
        "cli_subtype": result.get("subtype"),
        "cli_is_error": result.get("is_error"),
        "cli_terminal_reason": result.get("terminal_reason"),
        "permission_denials": len(result.get("permission_denials") or []),
        "drive_commands": runner,
        "runner_runs": runner.get("run", 0),
        "workflow_last_stop": last_stop,
        "anchor_instruction": anchor_instruction,
        "anchor_instruction_matches": anchor_matches,
        "anchor_instruction_given": anchor_given,
        "started_at": a.started_at,
        "finished_at": a.finished_at,
        "transcript_source": src,
    }
    # A realization or sweep run (run_realizations.sh, run_material_sweep.sh, run_bc_sweep.sh):
    # what it varied, for the figure's label, and the case it solved, to check it.
    variant = _load_json(a.variant) if getattr(a, "variant", None) else None
    if isinstance(variant, dict):
        metrics["variant"] = variant
        metrics["variant_kind"] = variant.get("kind")
        metrics["varied"] = variant.get("label") or variant.get("description")
    if wsid:
        metrics["case"] = case_summary(os.path.join(wf_dest, wsid))
    with open(os.path.join(run, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    if result.get("permission_denials"):
        with open(os.path.join(run, "permission_denials.json"), "w") as fh:
            json.dump(result["permission_denials"], fh, indent=1)
    print(f"status={status} exit={exit_code} session={wsid} copied={','.join(copied) or '-'} "
          f"claude={claude_sid} wall={metrics['wall_clock_s']}s "
          f"tokens(in/cr/cc/out)={tokens['input_tokens']}/{tokens['cache_read_input_tokens']}/"
          f"{tokens['cache_creation_input_tokens']}/{tokens['output_tokens']} cost={metrics['cost_usd']}")


# ---------------------------------------------------------------- summary

SUMMARY_COLS = ("run_name", "instruction", "status", "exit_code", "session_id", "claude_session_id",
                "workflow_last_stop", "runner_runs", "effort", "effort_observed", *TOKEN_FIELDS,
                "thinking_tokens", "cost_usd", "wall_clock_s",
                "cli_duration_s", "num_turns", "subagents_spawned", "permission_denials",
                "started_at", "finished_at", "variant_kind", "varied")


def cmd_summary(a):
    rows = []
    for run in a.runs:
        try:
            m = json.load(open(os.path.join(run, "metrics.json")))
        except (OSError, json.JSONDecodeError):
            m = {"run_name": os.path.basename(run), "status": "no_metrics"}
        inside = a.repo and os.path.abspath(run).startswith(os.path.abspath(a.repo) + os.sep)
        m["path"] = os.path.relpath(run, a.repo) if inside else os.path.abspath(run)
        rows.append(m)
    os.makedirs(a.batch_dir, exist_ok=True)
    with open(os.path.join(a.batch_dir, "summary.json"), "w") as fh:
        json.dump({"runs": rows}, fh, indent=1)
    with open(os.path.join(a.batch_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("path",) + SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ("path",) + SUMMARY_COLS})
    with open(os.path.join(a.batch_dir, "ar_runs.txt"), "w") as fh:
        fh.write(" ".join(r["path"] for r in rows) + "\n")


# ---------------------------------------------------------------- sweeps

def _brief_sections(text):
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            name = m.group(1).strip()
            out[name] = []
        elif name:
            out[name].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def source_info(session):
    """What an author of variants needs to know about a finished session: the object, its brief,
    its segments (id, the workflow's label, area share, centroid, extents, facing shares; y is up)
    and the case it solved."""
    session = os.path.abspath(session)
    snapshot = _load_json(os.path.join(session, "session.json")) or {}
    labels = {}
    text = str(snapshot.get("segment_description") or "")
    marks = list(re.finditer(r"(?:^|(?<=[\s;.,]))(\d+)\s*:\s*", text))
    for m, n in zip(marks, marks[1:] + [None]):
        labels[int(m.group(1))] = text[m.end():(n.start() if n else len(text))].strip(" ;.,\n")
    segments = []
    for s in (_load_json(os.path.join(session, "boundary_conditions", "segments.json")) or {}).get("segments") or []:
        lo, hi = (s.get("bounds") or [[0, 0, 0], [0, 0, 0]])
        segments.append({"id": s.get("id"), "label": labels.get(s.get("id")),
                         "area_fraction": round(float(s.get("area_fraction") or 0), 3),
                         "centroid_m": [round(float(c), 4) for c in s.get("centroid") or []],
                         "extent_m": [round(float(b) - float(a), 4) for a, b in zip(lo, hi)],
                         "facing": {k: round(float(v), 2) for k, v in (s.get("facing_fraction_45deg") or {}).items()
                                    if float(v) >= 0.05}})
    try:
        brief = open(os.path.join(session, "operator", "brief.md"), encoding="utf-8").read()
    except OSError:
        brief = ""
    instruction = (_load_json(os.path.join(session, "brief_snapshots", "instruction.json"))
                   or _load_json(os.path.join(session, "operator", "instruction.json")) or {}).get("instruction")
    bounds = (_load_json(os.path.join(session, "boundary_conditions", "segments.json")) or {}).get("bounds")
    return {"session_id": os.path.basename(session), "instruction": instruction,
            "unit_system": snapshot.get("unit_system"),
            "body_extent_m": ([round(float(b) - float(a), 4) for a, b in zip(*bounds)] if bounds else None),
            "segments": segments, "brief_sections": _brief_sections(brief),
            "case_as_solved": case_summary(session)}


MATERIAL_AUTHOR = """You are writing {N} material assignments for a demonstration that an automated finite-element workflow handles many different, heterogeneous materials on one body. The body, its segmentation and its supports and loads are fixed; only the materials change.

The object: "{INSTRUCTION}". Its segments (the parts a material is assigned to; y is up, metres):
{SEGMENTS}

The case as it was solved before (for context only): {CASE}
{IMAGES}
Rules:
- Every assignment is heterogeneous: at least two different materials, on different segments, and the stiffest Young's modulus at least 5 times and at most 500 times the softest (the workflow models per-part materials only above a 2x contrast, and stiffens parts more than 1000x softer).
- Assign materials freely and on purpose, WITHOUT considering what the real object or a part of that shape would be made of: e.g. rubber legs on a steel body, a glass shell over foam, a wooden head on a concrete base. The point is variety, not realism.
- Every segment id gets a material. Use real materials with handbook values: E between 1e5 and 3e11 Pa, Poisson's ratio between 0.2 and 0.45 (0.45 for rubber-like and gel-like materials), density between 20 and 20000 kg/m3.
- The {N} assignments differ from one another: different segments soft and stiff, different material families (metals, glass, ceramics, stone, concrete, woods, polymers, rubbers, foams, gels, composites, bone, ice ...), and different patterns (one soft part on a stiff body, alternating parts, a soft core part, ...). No two share the same set of materials.
- The label and the description name exactly the materials the parts list gives, segment by segment (no material in the words that is not in the parts).

Reply with only a JSON array of {N} objects, no prose, each:
{{"label": "<at most 8 words naming the assignment, e.g. rubber legs, glass shell>",
  "description": "<one sentence: which segment is which material>",
  "parts": [{{"segment": <id>, "segment_label": "<the segment's label above>", "material": "<material>", "youngs_modulus": <Pa>, "poisson_ratio": <nu>, "density": <kg/m3>}}, ...]}}"""

BC_AUTHOR = """You are writing {N} load cases (supports and loads) for a demonstration that an automated finite-element workflow handles many different boundary conditions on one body. The body, its segmentation and its materials are fixed; only the supports and loads change.

The object: "{INSTRUCTION}", about {EXTENT} m (x, y, z; y is up). Its segments (id, the workflow's label, area share, centroid and extent in metres, and the share of each segment's area facing each direction within 45 degrees):
{SEGMENTS}

Its brief (the real object's use, size and material, for context):
{BRIEF}

The case as it was solved before (for context only): {CASE}
{IMAGES}
Rules:
- Choose supports and loads freely: they need NOT be the most likely or the governing use of the object. The point is variety.
- They must be physically possible: every support and load sits on a real surface of the named segments (name the segment ids and which way the faces face, e.g. "the faces of segment 2 facing +x"); the supports together stop every rigid-body motion (at least one support held in all components, e.g. a clamp, a bolted or bonded face, or several separate held patches); loads are plausible for the object's size and mass: forces from a fraction of to about 20 times its weight, or what a person can push, pull or twist (up to about 1 kN, a twist up to about 50 N m for hand-sized parts); a body load as an acceleration of at most 3 g.
- Each case has one or two supports and one or two loads besides self-weight (self-weight always acts).
- The {N} cases differ from one another in the support regions, the load regions, the load directions and the kinds of load: among them a point-like load on a small patch, a load distributed over a whole face or segment, a twist (a torsion couple about an axis), a sideways or upward body load (an acceleration), a pull as well as a push; different supports (base, side, top, one end, a single part clamped).
- Never put a load on a held face.

Reply with only a JSON array of {N} objects, no prose, each:
{{"label": "<at most 8 words, e.g. clamped base, sideways push on arm>",
  "description": "<2-4 sentences, the whole case in words: each support (where, which faces, held in which components and why it is possible) and each load (where, which faces, kind, direction in x/y/z with y up, size with its unit and the reason for that size)>",
  "supports": [{{"segments": [<ids>], "faces": "<which way they face>", "held": "all" | [<components 0=x,1=y,2=z>]}}],
  "loads": [{{"kind": "patch" | "distributed" | "torsion" | "body", "segments": [<ids>], "faces": "<which way they face, or none for a body load>", "direction": [x, y, z], "size": <number>, "unit": "N" | "N m" | "m/s2", "why": "<the basis of the size>"}}]}}"""


def author_prompt(kind, session, n, images):
    info = source_info(session)
    segments = "\n".join(json.dumps(s) for s in info["segments"])
    case = json.dumps({k: v for k, v in (info["case_as_solved"] or {}).items() if k != "figure_files"})
    seen = ("\nLook at these pictures of the body and its segments with the Read tool first: "
            + ", ".join(images) + "\n") if images else ""
    if kind == "material":
        return MATERIAL_AUTHOR.format(N=n, INSTRUCTION=info["instruction"], SEGMENTS=segments, CASE=case, IMAGES=seen)
    sections = info["brief_sections"]
    brief = "\n\n".join(f"## {k}\n{v}" for k, v in sections.items() if not k.lower().startswith("prediction"))
    return BC_AUTHOR.format(N=n, INSTRUCTION=info["instruction"], SEGMENTS=segments, CASE=case, IMAGES=seen,
                            EXTENT=" x ".join(f"{e:g}" for e in info["body_extent_m"] or []) or "?", BRIEF=brief)


def _json_array(text):
    """The first JSON array in a reply (a fenced block or bare)."""
    text = text or ""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    for candidate in ([fenced.group(1)] if fenced else []) + [text[text.find("["):text.rfind("]") + 1]]:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, list):
            return value
    raise ValueError("no JSON array in the reply")


def check_variants(kind, variants, n, segment_ids):
    """Problems with authored variants (a list of strings; empty when they are usable)."""
    problems = []
    if len(variants) < n:
        problems.append(f"{len(variants)} variants, {n} asked")
    for i, v in enumerate(variants[:n], 1):
        if not isinstance(v, dict) or not str(v.get("label") or "").strip() or not str(v.get("description") or "").strip():
            problems.append(f"variant {i}: no label or description")
            continue
        if kind == "material":
            parts = v.get("parts") or []
            ids = {p.get("segment") for p in parts if isinstance(p, dict)}
            if segment_ids and set(segment_ids) - ids:
                problems.append(f"variant {i}: segments {sorted(set(segment_ids) - ids)} have no material")
            try:
                es = [float(p["youngs_modulus"]) for p in parts]
                # The authoring rule's own band, 5-500 x: the first real sweep's 1000 x (steel shell,
                # rubber legs) passed a 1000 x cap and the solve's triage then warned of it.
                if len({p.get("material") for p in parts}) < 2 or max(es) < 5 * min(es):
                    problems.append(f"variant {i}: not heterogeneous (E contrast {max(es) / min(es):.3g} x, under 5)")
                if max(es) > 500 * min(es):
                    problems.append(f"variant {i}: E contrast {max(es) / min(es):.3g} x, over 500")
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                problems.append(f"variant {i}: a part without a numeric youngs_modulus")
        else:
            if not v.get("supports") or not v.get("loads"):
                problems.append(f"variant {i}: no supports or no loads")
    return problems


def cmd_variants(a):
    info = source_info(a.session)
    ids = [s["id"] for s in info["segments"]]
    if a.result:
        result = _load_json(a.result) or {}
        text = result.get("result") if isinstance(result, dict) else None
        variants = _json_array(text)
    else:
        variants = _load_json(a.given)
        if isinstance(variants, dict):
            variants = variants.get("variants")
        if not isinstance(variants, list):
            raise SystemExit(f"{a.given}: not a JSON list of variants")
    problems = check_variants(a.kind, variants, a.n, ids)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"kind": a.kind, "source": os.path.abspath(a.session), "variants": variants[:a.n],
                   "problems": problems}, fh, indent=1)
    print("; ".join(problems) if problems else "ok")
    if problems and not a.allow_problems:
        sys.exit(1)


def cmd_variant_prompt(a):
    """The prompt of one sweep run, and its variant.json (what the run varies, for metrics.json)."""
    data = _load_json(a.variants)
    v = data["variants"][a.index - 1]
    info = source_info(data["source"])
    record = {"kind": data["kind"] + "_sweep", "index": a.index, "label": v.get("label"),
              "description": v.get("description"), "spec": v, "source_session": data["source"],
              "source_session_id": info["session_id"], "session_id": a.session_id}
    with open(a.variant_out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    text = open(a.template, encoding="utf-8").read()
    for key, value in (("{WORKTREE}", a.worktree), ("{SESSION}", a.session_id), ("{OBJECT}", info["instruction"] or ""),
                       ("{SOURCE}", info["session_id"]), ("{LABEL}", v.get("label") or ""),
                       ("{VARIANT}", v.get("description") or ""), ("{VARIANT_JSON}", json.dumps(v))):
        text = text.replace(key, value)
    sys.stdout.write(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("slug"); s.add_argument("text")
    s = sp.add_parser("prompt")
    s.add_argument("--template", required=True); s.add_argument("--worktree", required=True)
    s.add_argument("--object", required=True)
    s = sp.add_parser("finalize")
    for k in ("run-dir", "instruction", "model", "claude-session-id", "projects-dir",
              "workflow-root", "exit-code", "started-at", "finished-at", "t0", "t1"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--timed-out", default="0")
    s.add_argument("--effort", default="")
    s.add_argument("--variant", default="", help="a JSON file: what this run varied (metrics.json `variant`)")
    s = sp.add_parser("summary")
    s.add_argument("--batch-dir", required=True); s.add_argument("--repo", default="")
    s.add_argument("runs", nargs="*")
    s = sp.add_parser("source-info"); s.add_argument("--session", required=True)
    s = sp.add_parser("author-prompt")
    s.add_argument("--kind", choices=("material", "bc"), required=True)
    s.add_argument("--session", required=True); s.add_argument("--n", type=int, required=True)
    s.add_argument("--image", action="append", default=[])
    s = sp.add_parser("variants", help="the authored variants (from the author's CLI result, or a given "
                                       "JSON list), checked, as one variants.json")
    s.add_argument("--kind", choices=("material", "bc"), required=True)
    s.add_argument("--session", required=True); s.add_argument("--n", type=int, required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--result"); g.add_argument("--given")
    s.add_argument("--out", required=True)
    s.add_argument("--allow-problems", action="store_true")
    s = sp.add_parser("variant-prompt")
    for k in ("template", "worktree", "variants", "session-id", "variant-out"):
        s.add_argument("--" + k, required=True)
    s.add_argument("--index", type=int, required=True)
    a = p.parse_args(argv)
    if a.cmd == "slug":
        print(slug(a.text))
    elif a.cmd == "prompt":
        text = open(a.template, encoding="utf-8").read()
        sys.stdout.write(text.replace("{WORKTREE}", a.worktree).replace("{OBJECT}", a.object))
    elif a.cmd == "finalize":
        cmd_finalize(a)
    elif a.cmd == "summary":
        cmd_summary(a)
    elif a.cmd == "source-info":
        print(json.dumps(source_info(a.session), indent=1))
    elif a.cmd == "author-prompt":
        sys.stdout.write(author_prompt(a.kind, a.session, a.n, a.image))
    elif a.cmd == "variants":
        cmd_variants(a)
    elif a.cmd == "variant-prompt":
        cmd_variant_prompt(a)


if __name__ == "__main__":
    main()
