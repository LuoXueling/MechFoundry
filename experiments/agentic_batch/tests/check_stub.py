"""Checks the output of test_stub.sh.  python3 check_stub.py TMP JOBS"""
import glob
import json
import os
import sys

TMP, JOBS = sys.argv[1], int(sys.argv[2])
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "illustrations", "agentic_results"))
import ar_runs  # noqa: E402

fails = []
check = lambda ok, what: (print(("PASS " if ok else "FAIL ") + what), ok or fails.append(what))

# Concurrency and order, from the stub's own start/end log.
events = sorted((float(t), kind, obj) for kind, t, obj in
                (line.split(" ", 2) for line in open(os.path.join(TMP, "stub.log")).read().splitlines()))
running, peak, starts = 0, 0, []
for t, kind, obj in events:
    running += 1 if kind == "start" else -1
    peak = max(peak, running)
    if kind == "start":
        starts.append(obj.strip())
expected = [l.strip() for l in open(os.path.join(TMP, "objects.txt"))
            if l.strip() and not l.strip().startswith("#")]
check(peak == JOBS, f"peak concurrency {peak} == {JOBS}")
check(starts == expected, "started in list order: " + " | ".join(starts))
# The next run starts as soon as one finishes (within the 2 s start stagger + slack).
ends = [t for t, k, _ in events if k == "end"]
st = [t for t, k, _ in events if k == "start"]
gaps = [min((s - e for s in st if s >= e), default=0) for e in ends[:len(st) - JOBS]]
check(all(g < 4 for g in gaps), f"next run starts within 4 s of a finish (gaps {[round(g, 1) for g in gaps]})")

runs = sorted(glob.glob(os.path.join(TMP, "out", "*", "metrics.json")))
check(len(runs) == len(expected), f"{len(runs)} run folders with metrics.json")
want = {"sleep3": ("ok", True), "fail": ("cli_error", True), "nostart": ("ok", False),
        "crash": ("no_result", False), "twostart": ("ok", True), "hang": ("timeout", False),
        "sleep1": ("ok", True)}
FIELDS = ("instruction", "model", "session_id", "claude_session_id", "input_tokens",
          "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens",
          "wall_clock_s", "status", "started_at", "finished_at")
def linked_worktree(root):
    """ab_common.sh's test: the root's git dir is not the common one, or its path has a
    .claude/worktrees/ part."""
    import subprocess
    ask = lambda flag: subprocess.run(["git", "-C", root, "rev-parse", "--path-format=absolute", flag],  # noqa: E731
                                      capture_output=True, text=True).stdout.strip()
    git_dir = ask("--git-dir")
    return bool(git_dir and git_dir != ask("--git-common-dir")) or "/.claude/worktrees/" in root + "/"


for mp in runs:
    run = os.path.dirname(mp)
    m = json.load(open(mp))
    key = next(k for k in want if k in m["instruction"].split()[-1])
    status, has_session = want[key]
    tag = os.path.basename(run)
    check(all(f in m for f in FIELDS), f"{tag}: every required field present")
    check((m.get("system_note") or "").startswith("Environment note: the repository root"),
          f"{tag}: the path note was appended and recorded")
    note = m.get("system_note") or ""
    root = note.split(" is ", 1)[-1].split(". ", 1)[0]
    check(("It is a git worktree" in note) == linked_worktree(root),
          f"{tag}: the note calls the root a worktree only when it is one ({root})")
    check(m["status"] == status, f"{tag}: status {m['status']} == {status}")
    check(isinstance(m["wall_clock_s"], float) and m["wall_clock_s"] > 0, f"{tag}: wall clock {m['wall_clock_s']}")
    if status == "no_result" or status == "timeout":
        check(m["input_tokens"] is None and not os.path.exists(os.path.join(run, "claude_result.json")),
              f"{tag}: no result -> no tokens, no claude_result.json")
        continue
    res = json.load(open(os.path.join(run, "claude_result.json")))
    check(res["type"] == "result" and res["session_id"] == m["claude_session_id"],
          f"{tag}: claude_result.json is the CLI result of session {m['claude_session_id']}")
    main = os.path.join(run, "claude_session", m["claude_session_id"] + ".jsonl")
    sub = glob.glob(os.path.join(run, "claude_session", m["claude_session_id"], "subagents", "*.jsonl"))
    check(os.path.exists(main) and len(sub) == 1, f"{tag}: transcript and sub-agent transcript copied")
    tm = ar_runs.transcript_metrics(os.path.join(run, "claude_session"))
    check(all(tm[k] == m[k] for k in ar_runs.TOKEN_FIELDS),
          f"{tag}: metrics tokens (modelUsage) == main + sub-agent transcript sums {[m[k] for k in ar_runs.TOKEN_FIELDS]}")
    check(m["effort"] == "medium" and list((m["effort_observed"] or {}).keys()) == ["medium"]
          and m["thinking_tokens"] == m["output_tokens"] // 3,
          f"{tag}: effort medium passed and seen in the transcript {m['effort_observed']}, "
          f"thinking_tokens {m['thinking_tokens']} from modelUsage")
    check(m["main_session_usage"]["output_tokens"] + m["subagent_transcript_tokens"]["output_tokens"]
          == m["output_tokens"], f"{tag}: CLI usage (main only) + sub-agent == total")
    if has_session:
        wf = os.listdir(os.path.join(run, "workflow"))
        ses = json.load(open(os.path.join(run, "workflow", m["session_id"], "session.json")))
        check(ses["object"] == m["instruction"], f"{tag}: copied workflow {m['session_id']} is this run's "
                                                 f"(of {len(wf)} copied)")
        # the batch passed the list line itself (FEM_OPERATOR_INSTRUCTION) and finalize compared it
        check(m["anchor_instruction"] == m["instruction"] and m["anchor_instruction_matches"] is True,
              f"{tag}: start stored this run's instruction")
        if key == "twostart":
            check(len(wf) == 2 and "final reply" in m["session_id_found_by"],
                  f"{tag}: both sessions copied, primary from the final reply")
        check(m["runner_runs"] == 3 * len(wf), f"{tag}: runner_runs {m['runner_runs']}")
        r = ar_runs.load_run(run)
        check(r.session and r.session.endswith(m["session_id"]) and r.source.get("input_tokens") == "metrics.json",
              f"{tag}: ar_runs.load_run finds the session and reads tokens from metrics.json")
    else:
        check(m["session_id"] is None and os.listdir(os.path.join(run, "workflow")) == [],
              f"{tag}: no workflow session -> session_id null, workflow/ empty")

bdir = glob.glob(os.path.join(TMP, "out", "_batches", "*"))[0]
rows = json.load(open(os.path.join(bdir, "summary.json")))["runs"]
csv_lines = open(os.path.join(bdir, "summary.csv")).read().strip().splitlines()
check(len(rows) == len(expected) and len(csv_lines) == len(expected) + 1, "summary.json/csv: one row per run")
check([r["instruction"] for r in rows] == expected, "summary rows in list order")
check(open(os.path.join(bdir, "batch.log")).read().count(" finish ") == len(expected), "batch.log: a finish line per run")
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED")
sys.exit(1 if fails else 0)
