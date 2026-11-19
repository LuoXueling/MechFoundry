#!/usr/bin/env python3
"""A stand-in for `codex exec --json` for testing run_batch.sh --agent codex without API calls.

It checks what ab_common.sh's ab_launch_codex hands it (the arguments, a closed stdin, the run's own
CODEX_HOME with its config.toml and rules, no inherited CODEX_* session variable, the provider key in
its environment), sleeps, and writes what the real CLI would: the `--json` event stream on stdout
(thread.started, command_execution and agent_message items, turn.completed with usage, in the shape
measured on codex-cli 0.163.0) and the session's rollout under $CODEX_HOME/sessions/ (function calls,
view_image among them, and token_count totals), plus a fake workflow session under $AB_WORKFLOW_ROOT.

Behaviour is keyed by words in the object, as in stub_claude.py:
  sleepN   sleep N seconds (default 2)
  fail     a turn.failed event (the provider refused), exit 1
  crash    exit 3 after thread.started, nothing else
  hang     sleep 10 000 s (for the timeout test)
  nostart  never call drive.py start
Every call is logged to $STUB_LOG as "start|end <epoch> <object>".
"""
import json
import os
import random
import re
import sys
import time
import tomllib

args = sys.argv[1:]
assert args[:2] == ["exec", "--json"], f"stub: args {args[:3]}"
assert args[2] == "-C" and os.path.isdir(args[3]), "stub: -C <repository> missing"
prompt = args[4]
assert len(args) == 5, f"stub: unexpected arguments {args[5:]}"
assert sys.stdin.read() == "", "stub: stdin is not closed (codex exec would wait on it)"
home = os.environ["CODEX_HOME"]
cfg = tomllib.load(open(os.path.join(home, "config.toml"), "rb"))
prov = cfg["model_providers"][cfg["model_provider"]]
assert prov["wire_api"] == "responses" and cfg["approval_policy"] == "never", "stub: config"
assert cfg["sandbox_mode"] == "workspace-write" and cfg["project_doc_max_bytes"] == 0, "stub: config"
assert cfg["features"]["multi_agent"] is False, "stub: extra tools left on"
assert os.environ.get(prov["env_key"]), f"stub: {prov['env_key']} not in the environment"
assert "Tool note:" in cfg.get("developer_instructions", ""), "stub: codex_tools.txt not given"
assert "Batch note:" in cfg.get("developer_instructions", ""), "stub: codex_note.txt not given"
rules = open(os.path.join(home, "rules", "operator.rules")).read()
assert "operator_skill/scripts/drive.py" in rules and 'decision="allow"' in rules, "stub: rules"
assert not os.environ.get("CODEX_THREAD_ID"), "stub: the parent's Codex session leaked"
if os.environ.get("STUB_EXPECT_MODEL"):
    assert cfg["model"] == os.environ["STUB_EXPECT_MODEL"], f"stub: model {cfg['model']}"
images = cfg["features"].get("view_image", True)

obj = re.search(r'The user\'s whole instruction is: "(.*?)"\.', prompt).group(1)
log = os.environ.get("STUB_LOG")


def note(what):
    if log:
        with open(log, "a") as fh:
            fh.write(f"{what} {time.time():.3f} {obj}\n")


rng = random.Random(obj + str(time.time()))
thread = "01a1eac5-%04x-7f73-8475-%012x" % (rng.getrandbits(16), rng.getrandbits(48))
out = sys.stdout


def emit(rec):
    out.write(json.dumps(rec) + "\n")
    out.flush()


note("start")
emit({"type": "thread.started", "thread_id": thread})
emit({"type": "item.completed", "item": {"id": "item_0", "type": "error",
                                         "message": f"Model metadata for `{cfg['model']}` not found."}})
m = re.search(r"sleep(\d+)", obj)
time.sleep(10000 if "hang" in obj else int(m.group(1)) if m else 2)
if "crash" in obj:
    note("end")
    sys.exit(3)
emit({"type": "turn.started"})
DRIVE = "python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py"
wf_root = os.environ["AB_WORKFLOW_ROOT"]
rollout, n = [], 0


def command(cmd, output):
    global n
    n += 1
    shell = f"/bin/bash -lc '{cmd}'"
    emit({"type": "item.started", "item": {"id": f"item_{n}", "type": "command_execution", "command": shell,
                                           "aggregated_output": "", "exit_code": None, "status": "in_progress"}})
    emit({"type": "item.completed", "item": {"id": f"item_{n}", "type": "command_execution", "command": shell,
                                             "aggregated_output": output, "exit_code": 0, "status": "completed"}})
    rollout.append({"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                                                         "arguments": json.dumps({"cmd": cmd}), "call_id": f"c{n}"}})


ws = None
if "nostart" not in obj:
    ws = "%012x" % rng.getrandbits(48)
    os.makedirs(os.path.join(wf_root, ws, "operator"), exist_ok=True)
    json.dump({"session_id": ws, "object": obj}, open(os.path.join(wf_root, ws, "session.json"), "w"))
    record = {"instruction": os.environ.get("FEM_OPERATOR_INSTRUCTION", obj)}
    os.makedirs(os.path.join(wf_root, ws, "brief_snapshots"), exist_ok=True)
    json.dump(record, open(os.path.join(wf_root, ws, "brief_snapshots", "instruction.json"), "w"))
    command(f'{DRIVE} start --instruction "{obj}"', f"session_id: {ws}\n" + json.dumps({"ok": True, "session_id": ws}))
    for _ in range(3):
        command(f"{DRIVE} run {ws} --plan output/fem_workflow_operator/{ws}/operator/plan.json",
                json.dumps({"ok": True, "stop": "geometry"}))
        if images:
            rollout.append({"type": "response_item", "payload": {"type": "function_call", "name": "view_image",
                                                                 "arguments": "{}", "call_id": "v"}})
    json.dump({"stop": "finished"}, open(os.path.join(wf_root, ws, "operator", "last_run.json"), "w"))
usage = {"input_tokens": 100000, "cached_input_tokens": 80000, "cache_write_input_tokens": 5000,
         "output_tokens": 3000, "reasoning_output_tokens": 1000}
rollout.append({"type": "turn_context", "payload": {"model": cfg["model"]}})
rollout.append({"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}})
day = os.path.join(home, "sessions", "2026", "11", "17")
os.makedirs(day, exist_ok=True)
with open(os.path.join(day, f"rollout-2026-11-17T13-30-05-{thread}.jsonl"), "w") as fh:
    for rec in rollout:
        fh.write(json.dumps(rec) + "\n")
if "fail" in obj:
    emit({"type": "error", "message": "Reconnecting... 1/5 (402)"})
    emit({"type": "turn.failed", "error": {"message": "402 Purchase pre-paid credits"}})
    note("end")
    sys.exit(1)
emit({"type": "item.completed", "item": {"id": "item_r", "type": "agent_message",
                                         "text": f"Session id: {ws}\nverdict: stub" if ws else "could not start"}})
emit({"type": "turn.completed", "usage": usage})
note("end")
