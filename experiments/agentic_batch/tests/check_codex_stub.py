"""Checks the output of test_codex_stub.sh.  python3 check_codex_stub.py TMP"""
import glob
import json
import os
import sys

TMP = sys.argv[1]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "illustrations", "agentic_results"))
import ar_runs  # noqa: E402

fails = []
check = lambda ok, what: (print(("PASS " if ok else "FAIL ") + what), ok or fails.append(what))  # noqa: E731

runs = {json.load(open(p))["instruction"].split()[-1]: os.path.dirname(p)
        for p in glob.glob(os.path.join(TMP, "out*", "*", "metrics.json"))}
want = {"sleep2": ("ok", True), "fail": ("cli_error", True), "nostart": ("ok", False),
        "crash": ("no_result", False), "hang": ("timeout", False), "sleep1": ("ok", True)}
check(sorted(runs) == sorted(want), f"one run per object: {sorted(runs)}")
secret = os.environ.get("HF_TOKEN") or ""
for key, run in sorted(runs.items()):
    m = json.load(open(os.path.join(run, "metrics.json")))
    status, has_session = want[key]
    check(m["status"] == status, f"{key}: status {m['status']} == {status}")
    check(bool(m["session_id"]) == has_session, f"{key}: workflow session found = {has_session}")
    vendor = "out_vendor" in run
    check(m["agent"] == "codex" and m["provider"] == ("deepseek" if vendor else "hf"),
          f"{key}: agent codex, provider {m['provider']}")
    # The id without its pinned route, the route beside it; the vendor API has neither route nor price.
    check(m["model"] == ("deepseek-flash" if vendor else "moonshotai/Kimi-K3.1-Pro"), f"{key}: model {m['model']}")
    check(m["route"] == (None if vendor else "moonshotai"), f"{key}: route {m['route']}")
    check((m.get("system_note") or "").startswith("Environment note: the repository root"), f"{key}: path note")
    cfg = open(os.path.join(run, "codex_home", "config.toml")).read()
    check("Tool note:" in cfg and "Environment note:" in cfg, f"{key}: developer instructions = path note + codex note")
    check("view_image = false" not in cfg, f"{key}: view_image on")
    # No --effort given: the model's default from codex_models.json, sent and recorded.
    check('model_reasoning_effort = "medium"' in cfg and m["effort"] == "medium", f"{key}: effort medium by default")
    if status in ("ok", "cli_error") and has_session:
        # 100000 in (80000 cached, 5000 written) -> 15000 uncached; reasoning is part of output
        check((m["input_tokens"], m["cache_read_input_tokens"], m["cache_creation_input_tokens"],
               m["output_tokens"], m["thinking_tokens"]) == (15000, 80000, 5000, 3000, 1000),
              f"{key}: token arithmetic ({m['tokens_source']})")
        check(m["runner_runs"] == 3 and m["drive_commands"].get("start") == 1, f"{key}: drive commands counted")
        check(m["images_viewed"] == 3, f"{key}: images_viewed {m['images_viewed']}")
        # Kimi (miss 2, hit 0.2, write 2, out 10 USD per M): 15000*2 + 80000*0.2 + 5000*2 + 3000*10 = 0.086;
        # DeepSeek through its own API at the same accounting price as on HF (miss 0.3, hit 0.006, out 1.2;
        # writes as misses): 15000*0.3 + 80000*0.006 + 5000*0.3 + 3000*1.2 = 0.01008
        check(m["cost_usd"] == (0.01008 if vendor else 0.086) and m["cost_basis"]["output"] == (1.2 if vendor else 10.0),
              f"{key}: cost_usd {m['cost_usd']} from the model's price")
        check(m["anchor_instruction_matches"] is True, f"{key}: the instruction start stored matches")
        check(os.path.isdir(os.path.join(run, "workflow", m["session_id"])), f"{key}: workflow session copied")
        check(m["transcript_source"].startswith("codex_home/sessions/"), f"{key}: rollout found")
    if status == "ok" and has_session:
        check(open(os.path.join(run, "agent_reply.md")).read().startswith("Session id:"), f"{key}: agent_reply.md")
    if key == "fail":
        check("402" in (m.get("cli_error") or "") and m["cli_errors_retried"] == 1, f"{key}: the provider error kept")
    leaked = [p for p in glob.glob(os.path.join(run, "**", "*"), recursive=True)
              if os.path.isfile(p) and secret and secret in open(p, errors="replace").read()]
    check(not leaked, f"{key}: the provider key is in no run file {leaked}")
    try:
        ar_runs.load_run(run)
        check(True, f"{key}: ar_runs.load_run reads it")
    except Exception as e:  # noqa: BLE001
        check(False, f"{key}: ar_runs.load_run: {e}")

summary = glob.glob(os.path.join(TMP, "out", "_batches", "*", "summary.csv"))
check(len(summary) == 1 and "codex" in open(summary[0]).read(), "summary.csv written, with the agent column")
log = open(glob.glob(os.path.join(TMP, "out", "_batches", "*", "batch.log"))[0]).read()
check("price {'input': 2.0, 'cached_input': 0.2, 'cache_write': 2.0" in log, "the batch log records the price")
check(bool(secret) and secret not in log, "the provider key (HF_TOKEN, set by the test script) is not in the batch log")
vlog = open(glob.glob(os.path.join(TMP, "out_vendor", "_batches", "*", "batch.log"))[0]).read()
check("sk-stubsecret" not in vlog and "key DEEPSEEK_API_KEY from the environment" in vlog,
      "vendor run: DEEPSEEK_API_KEY used, not logged")
check(open(os.path.join(TMP, "nokey.rc")).read().strip() == "2"
      and "HF_TOKEN is not set" in open(os.path.join(TMP, "nokey.err")).read(), "no key: the batch refuses to start")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
