"""Which runs the figure shows, and what is known about each.  Plain Python.

A run is given in one of three forms:

  output/agentic_batch/<run_name>/      the batch script's layout: ``workflow/`` (a copy
                                        of the session folder, or a folder holding it),
                                        ``claude_session/`` (the transcript, *.jsonl) and
                                        ``metrics.json``
  output/fem_workflow_operator/<sid>/   a bare session folder (development)
  an entry of a runs JSON               ``{"session": ..., "instruction": ..., "transcript":
                                        ..., "model": ..., "label": ...}`` (development)

Any entry may also give ``draw_scale`` (0 < f <= 1, default 1): the one named per-object
size override.  The row is drawn at f x the size the figure's own sizing rule
(ar_figure.row_crop: the union ink box fitted into the panel) gives it -- for a body
that the rule would draw too large beside its neighbours (the main figure's control
arm: 0.9; the car brake disc it replaced had 0.75).  The renderer widens its view angle so the frame keeps room around the
smaller body (render_ar.fit), the compose divides the panel crop's px/mm by f and
shrinks about the rule's ink centre, so the body does not sink (ar_figure.row_crop) and the camera chooser predicts the print checks on the smaller
drawing (ar_prep.preview_print).  The ground grid is physical, so its cell label
stays true.  It is recorded per row as ``draw_scale`` in ar_stats.json.

An entry may also give ``prompt_max_lines`` (1-3, default none): the row's instruction
card holds its prompt in at most that many lines, set at the largest font size <= the
card's 7 pt (0.1 pt steps, never under 5 pt) at which it wraps so (ar_figure.prompt_fit)
-- a long instruction that would otherwise take a third line and make every row of the
figure taller (the main figure's control arm: 2 lines, 5.8 pt).  Recorded per row as
``prompt`` (max_lines, fs, lines) in ar_stats.json.

Every metadata field is taken from the first source that has it, and the source
is recorded beside the value, so the figure's numbers can be traced:

  explicit (runs JSON / CLI)  >  metrics.json  >  the Claude transcript

Token counts from a transcript are summed over its assistant messages, each
message counted ONCE: Claude Code writes one line per content block, and every
line of a message repeats that message's usage.  Wall clock from a transcript is
its last timestamp minus its first.
"""

import glob
import json
import os
from collections import Counter
from datetime import datetime

TOKEN_FIELDS = ("input_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "output_tokens")
FIELDS = ("instruction", "model", "session_id", *TOKEN_FIELDS, "wall_clock_s", "status")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _abs(path, base):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))


def _find_session(folder, session_id=None):
    """The session folder inside an agentic_batch run's ``workflow/``."""
    if os.path.exists(os.path.join(folder, "session.json")):
        return folder
    hits = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(folder, "**", "session.json"),
                                                      recursive=True))
    if session_id:
        named = [h for h in hits if os.path.basename(h) == session_id]
        hits = named or hits
    if len(hits) > 1:
        raise ValueError(f"{folder}: several session folders {hits}; give session_id")
    return hits[0] if hits else None


def transcript_metrics(path):
    """Tokens, wall clock and model from Claude Code transcript(s) at ``path``
    (one .jsonl, or a folder searched recursively; sub-agent transcripts in it
    are part of the run and are summed with the main one)."""
    files = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    if not files:
        return {}
    per_message, models, stamps = {}, Counter(), []
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("timestamp"):
                stamps.append(datetime.fromisoformat(rec["timestamp"].replace("Z", "+00:00")))
            msg = rec.get("message") if rec.get("type") == "assistant" else None
            if not isinstance(msg, dict) or not isinstance(msg.get("usage"), dict):
                continue
            key = (f, msg.get("id") or rec.get("uuid"))
            seen = per_message.setdefault(key, {k: 0 for k in TOKEN_FIELDS})
            for k in TOKEN_FIELDS:      # a message's later lines may carry the final count
                seen[k] = max(seen[k], int(msg["usage"].get(k) or 0))
            if msg.get("model") and not msg["model"].startswith("<"):
                models[msg["model"]] += 1
    out = {k: sum(m[k] for m in per_message.values()) for k in TOKEN_FIELDS} if per_message else {}
    if stamps:
        out["wall_clock_s"] = (max(stamps) - min(stamps)).total_seconds()
    if models:
        out["model"] = models.most_common(1)[0][0]
    out["_messages"] = len(per_message)
    out["_files"] = len(files)
    return out


class Run:
    def __init__(self, session, meta, source, label=None, view_dir=None, draw_scale=1.0,
                 prompt_max_lines=None):
        self.session = session          # session folder, or None when it cannot be found
        self.meta = meta                # field -> value
        self.source = source            # field -> where the value came from
        self.label = label
        self.view_dir = view_dir
        draw_scale = float(1.0 if draw_scale is None else draw_scale)
        assert 0 < draw_scale <= 1, f"draw_scale must be in (0, 1], got {draw_scale}"
        self.draw_scale = draw_scale    # the per-object size override (module docstring)
        assert prompt_max_lines is None or int(prompt_max_lines) in (1, 2, 3), \
            f"prompt_max_lines must be 1, 2 or 3, got {prompt_max_lines}"
        self.prompt_max_lines = None if prompt_max_lines is None else int(prompt_max_lines)

    def get(self, key, default=None):
        return self.meta.get(key, default)

    def record(self):
        return {"session": self.session, "label": self.label, "view_dir": self.view_dir,
                "draw_scale": self.draw_scale, "prompt_max_lines": self.prompt_max_lines,
                "meta": self.meta, "source": self.source}


def _merge(meta, source, values, where):
    for k, v in values.items():
        if k.startswith("_") or v is None or k in meta:
            continue
        meta[k] = v
        source[k] = where


def load_run(spec, base=REPO):
    """``spec``: a path (str) or a dict entry of a runs JSON."""
    entry = {"path": spec} if isinstance(spec, str) else dict(spec)
    meta, source = {}, {}
    _merge(meta, source, {k: entry.get(k) for k in FIELDS}, "explicit")
    path = _abs(entry.get("path") or entry.get("session") or entry.get("run"), base)
    run_dir, session = None, None
    if os.path.exists(os.path.join(path, "metrics.json")) or os.path.isdir(os.path.join(path, "workflow")):
        run_dir = path
    if run_dir:
        mpath = os.path.join(run_dir, "metrics.json")
        if os.path.exists(mpath):
            _merge(meta, source, json.load(open(mpath)), "metrics.json")
        session = _find_session(os.path.join(run_dir, "workflow"), meta.get("session_id"))
    else:
        session = path if os.path.exists(os.path.join(path, "session.json")) else None
    transcript = entry.get("transcript")
    if transcript:
        transcript = _abs(transcript, base)
    elif run_dir and os.path.isdir(os.path.join(run_dir, "claude_session")):
        transcript = os.path.join(run_dir, "claude_session")
    if transcript and os.path.exists(transcript):
        tm = transcript_metrics(transcript)
        _merge(meta, source, tm, "transcript")
        source["_transcript"] = f"{transcript} ({tm.get('_files', 0)} file(s), " \
                                f"{tm.get('_messages', 0)} assistant messages)"
    if session and "session_id" not in meta:
        meta["session_id"] = os.path.basename(os.path.normpath(session))
        source["session_id"] = "folder name"
    if run_dir and os.path.exists(os.path.join(run_dir, "metrics.json")):
        try:                    # where the wall clock went (ar_time.py): the time bar and its CSV
            import ar_time
            meta["time_split"] = ar_time.split_run(run_dir)
            source["time_split"] = "ar_time.split_run (transcript + driver_log.jsonl)"
        except Exception as exc:  # noqa: BLE001 -- the figure then shows a plain time bar
            source["time_split"] = f"not split: {type(exc).__name__}: {exc}"
    return Run(session, meta, source, label=entry.get("label"), view_dir=entry.get("view_dir"),
               draw_scale=entry.get("draw_scale"), prompt_max_lines=entry.get("prompt_max_lines"))


def load_runs(paths=(), runs_json=None):
    specs = []
    if runs_json:
        doc = json.load(open(runs_json))
        base = os.path.dirname(os.path.abspath(runs_json))
        for e in doc["runs"] if isinstance(doc, dict) else doc:
            e = {"path": e} if isinstance(e, str) else dict(e)
            for k in ("path", "session", "run", "transcript"):
                if e.get(k):
                    e[k] = _abs(e[k], REPO if not str(e[k]).startswith(".") else base)
            specs.append(e)
    specs += list(paths)
    return [load_run(s) for s in specs]
