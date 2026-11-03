#!/usr/bin/env python3
"""A stand-in for `claude -p` for testing run_batch.sh without API calls.

It parses the flags run_batch.sh passes, sleeps, and writes what the real CLI
would: a transcript at $AB_PROJECTS_DIR/<encoded cwd>/<session uuid>.jsonl in the
real record format (assistant lines carrying usage, Bash tool_use and tool_result
blocks, `drive.py start` printing `session_id: <id>`), a sub-agent transcript
under <uuid>/subagents/, a workflow session folder under $AB_WORKFLOW_ROOT, and the
result in the CLI's real stream-json/json shape (usage = main only, modelUsage =
main + sub-agent, as measured on the real CLI).

It also stands in for the sweeps' authoring call (`--tools Read`: N variants as a JSON array) and
for a sweep run, whose prompt names a session that exists already (it drives that one, no start).

Behaviour is keyed by words in the object (the instruction inside the prompt):
  sleepN      sleep N seconds (default 2)
  fail        exit 1 with an error result (is_error, subtype error_during_execution)
  crash       exit 3 with no result at all
  hang        sleep 10 000 s (for the timeout test)
  nostart     never call drive.py start (no workflow session)
  twostart    start two sessions; the final reply names the second
Every call is logged to $STUB_LOG as "start|end <epoch> <object>".
"""
import json
import os
import random
import re
import sys
import time
import uuid

args = sys.argv[1:]
opts, prompt, i = {}, None, 0
while i < len(args):
    a = args[i]
    if a in ("--model", "--session-id", "--output-format", "--permission-mode", "--settings", "--effort",
             "--append-system-prompt"):
        opts[a] = args[i + 1]; i += 2
    elif a in ("--allowedTools", "--disallowedTools", "--tools"):
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        opts[a] = args[i + 1:j]; i = j
    elif a in ("-p", "--print", "--verbose", "--strict-mcp-config", "--disable-slash-commands"):
        opts[a] = True; i += 1
    else:
        prompt = a; i += 1
assert opts.get("-p") and prompt, "stub: needs -p PROMPT"
assert opts.get("--permission-mode") == "dontAsk" and opts.get("--allowedTools"), "stub: allowlist missing"
if opts.get("--tools") == ["Read"]:
    # The sweeps' authoring call (ab_sweep.sh): N variants as a JSON array, the segment ids from
    # the prompt's segment lines.  STUB_AUTHOR_BAD=1 makes the first reply refused (one material).
    n = int(re.search(r"JSON array of (\d+) objects", prompt).group(1))
    ids = [int(i) for i in re.findall(r'^\{"id": (\d+)', prompt, re.M)]
    bad = os.environ.get("STUB_AUTHOR_BAD") == "1" and "previous reply was refused" not in prompt
    if "material assignments" in prompt:
        variants = [{"label": f"variant {k} materials", "description": f"variant {k}",
                     "parts": [{"segment": i, "segment_label": f"part {i}", "material": f"m{(i + k) % 2}",
                                "youngs_modulus": 1e9 if bad or (i + k) % 2 else 1e7, "poisson_ratio": 0.3,
                                "density": 1000} for i in ids]} for k in range(1, n + 1)]
    else:
        variants = [{"label": f"load case {k}", "description": f"case {k}",
                     "supports": [{"segments": [ids[0]], "faces": "-y", "held": "all"}],
                     "loads": [{"kind": "patch", "segments": [ids[-1]], "faces": "+y", "direction": [k, 0, 0],
                                "size": 10 * k, "unit": "N", "why": "stub"}]} for k in range(1, n + 1)]
    log = os.environ.get("STUB_LOG")
    if log:
        with open(log + ".author", "a") as fh:
            fh.write(f"{opts.get('--model')} {n}\n")
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "result": "```json\n" + json.dumps(variants) + "\n```",
                      "session_id": opts.get("--session-id")}))
    sys.exit(0)
# Only the built-in tools the skill uses are loaded, and no MCP server (run_batch.sh, BUILTIN_TOOLS).
assert opts.get("--tools") == ["Bash,Read,Write,Edit,Glob,Grep"], f"stub: tools {opts.get('--tools')}"
assert opts.get("--strict-mcp-config"), "stub: MCP servers not shut out"
assert opts.get("--disable-slash-commands"), "stub: skills listing not disabled"
# Auto-memory off and the 5-minute cache by default (run_batch.sh --memory / --cache-ttl).
assert json.loads(opts.get("--settings") or "{}") == {"promptCacheTtl": "5m", "autoMemoryEnabled": False}, \
    f"stub: settings {opts.get('--settings')}"
assert not any(k.startswith("CLAUDE_CODE_SESSION") for k in os.environ), "stub: parent session env leaked"
found = re.search(r'The user\'s whole instruction is: "(.*?)"\.', prompt)
# A sweep run (prompt_material_sweep.txt, prompt_bc_sweep.txt): the session exists already.
given_session = None if found else re.search(r"The workflow session ([0-9a-f]{12}) ", prompt).group(1)
obj = found.group(1) if found else re.search(r'is the (?:material assignment|load case) "(.*?)"', prompt).group(1)
sid_claude = opts.get("--session-id") or str(uuid.uuid4())
log = os.environ.get("STUB_LOG")


def note(what):
    if log:
        with open(log, "a") as fh:
            fh.write(f"{what} {time.time():.3f} {obj}\n")


note("start")
m = re.search(r"sleep(\d+)", obj)
time.sleep(10000 if "hang" in obj else int(m.group(1)) if m else 2)
if "crash" in obj:
    note("end")
    sys.exit(3)

proj = os.path.join(os.environ["AB_PROJECTS_DIR"], re.sub(r"[^A-Za-z0-9]", "-", os.getcwd()))
os.makedirs(os.path.join(proj, sid_claude, "subagents"), exist_ok=True)
wf_root = os.environ["AB_WORKFLOW_ROOT"]
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
DRIVE = "python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py"
rng = random.Random(sid_claude)


def usage(n):
    return {"input_tokens": 3 * n, "cache_read_input_tokens": 1000 * n,
            "cache_creation_input_tokens": 200 * n, "output_tokens": 50 * n}


lines, main_tot, k = [], {k: 0 for k in usage(0)}, 0


def assistant(blocks, n):
    """One API message written as one line per content block, each repeating the usage."""
    global k
    k += 1
    mid = f"msg_{sid_claude[:8]}_{k}"
    for b in blocks:
        lines.append({"type": "assistant", "timestamp": now(), "sessionId": sid_claude, "uuid": str(uuid.uuid4()),
                      "effort": opts.get("--effort", "high"), "perTurnEffort": opts.get("--effort", "high"),
                      "message": {"id": mid, "model": "claude-sonnet-5-5", "role": "assistant",
                                  "content": [b], "usage": usage(n)}})
    for key, v in usage(n).items():
        main_tot[key] += v


def bash(cmd, out):
    tid = f"toolu_{uuid.uuid4().hex[:20]}"
    assistant([{"type": "text", "text": "running"}, {"type": "tool_use", "id": tid, "name": "Bash",
                                                       "input": {"command": cmd}}], 1)
    lines.append({"type": "user", "timestamp": now(), "uuid": str(uuid.uuid4()),
                  "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                           "content": out}]},
                  "toolUseResult": {"stdout": out, "stderr": ""}})


lines.append({"type": "user", "timestamp": now(), "message": {"role": "user", "content": prompt}})
sessions = []
if given_session:
    sessions = [given_session]
    for _ in range(3):
        bash(f"{DRIVE} run {given_session} --plan output/fem_workflow_operator/{given_session}/operator/plan.json",
             json.dumps({"ok": True, "stop": "solve"}))
    json.dump({"stop": "finished"}, open(os.path.join(wf_root, given_session, "operator", "last_run.json"), "w"))
elif "nostart" not in obj:
    for _ in range(2 if "twostart" in obj else 1):
        ws = "%012x" % rng.getrandbits(48)
        os.makedirs(os.path.join(wf_root, ws, "operator"), exist_ok=True)
        json.dump({"session_id": ws, "object": obj, "seed": os.environ.get("FEM_OPERATOR_SEED")},
                  open(os.path.join(wf_root, ws, "session.json"), "w"))
        # `start` stores the instruction for the checks (no API call)
        record = {"instruction": os.environ.get("FEM_OPERATOR_INSTRUCTION", obj)}
        for folder in ("operator", "brief_snapshots"):
            os.makedirs(os.path.join(wf_root, ws, folder), exist_ok=True)
            json.dump(record, open(os.path.join(wf_root, ws, folder, "instruction.json"), "w"))
        bash(f'{DRIVE} start --instruction "{obj}"', f"session_id: {ws}\n" + json.dumps(
            {"ok": True, "session_id": ws}, indent=1))
        sessions.append(ws)
    for ws in sessions:
        for _ in range(3):
            bash(f"{DRIVE} run {ws} --plan output/fem_workflow_operator/{ws}/operator/plan.json",
                 json.dumps({"ok": True, "stop": "geometry"}))
        json.dump({"stop": "finished"}, open(os.path.join(wf_root, ws, "operator", "last_run.json"), "w"))
reply = (f"Session id: {sessions[-1]}\nfinal judgement: the brief's case was run" if sessions
         else "could not start")
assistant([{"type": "text", "text": reply}], 2)
with open(os.path.join(proj, sid_claude + ".jsonl"), "w") as fh:
    for rec in lines:
        fh.write(json.dumps(rec) + "\n")

sub_tot = usage(4)
with open(os.path.join(proj, sid_claude, "subagents", "agent-a0000stub.jsonl"), "w") as fh:
    for blk in ("a", "b"):   # one message, two blocks: counted once
        fh.write(json.dumps({"type": "assistant", "timestamp": now(), "isSidechain": True,
                             "message": {"id": "msg_sub_" + sid_claude[:8], "model": "claude-sonnet-5-5",
                                         "content": [{"type": "text", "text": blk}], "usage": sub_tot}}) + "\n")
json.dump({"agentType": "general-purpose"}, open(os.path.join(proj, sid_claude, "subagents",
                                                            "agent-a0000stub.meta.json"), "w"))

fail = "fail" in obj
total = {key: main_tot[key] + sub_tot[key] for key in main_tot}
result = {
    "type": "result", "subtype": "error_during_execution" if fail else "success", "is_error": fail,
    "duration_ms": 2100, "duration_api_ms": 1900, "num_turns": k, "result": reply,
    "session_id": sid_claude, "total_cost_usd": 0.0123, "stop_reason": "end_turn",
    "usage": {**main_tot, "service_tier": "standard"},
    "modelUsage": {"claude-sonnet-5-5": {"inputTokens": total["input_tokens"], "outputTokens": total["output_tokens"],
                                         "cacheReadInputTokens": total["cache_read_input_tokens"],
                                         "cacheCreationInputTokens": total["cache_creation_input_tokens"],
                                         "thinkingTokens": total["output_tokens"] // 3,   # part of outputTokens
                                         "costUSD": 0.0123}},
    "permission_denials": [], "terminal_reason": "completed",
    "subagent_stats": {"spawned": 1},
}
if opts.get("--output-format") == "stream-json":
    print(json.dumps({"type": "system", "subtype": "init", "session_id": sid_claude}))
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": reply}]}}))
    print(json.dumps(result))
else:
    print(json.dumps(result))
note("end")
sys.exit(1 if fail else 0)
