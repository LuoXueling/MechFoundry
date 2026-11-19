"""Per model call (tool result / user message -> the call's token_usage_record; tokens from the token_count after it): duration, output
tokens, uncached input tokens.  Fit duration = a + out/r per batch root (least squares), so the
per-call overhead (queue + prefill + first token) is separated from the decode rate r."""
import glob, json, sys, datetime as dt
import numpy as np
def t(e): return dt.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
for root in sys.argv[1:]:
    D, O = [], []
    for ro in glob.glob(f"{root}/*/codex_home/sessions/**/rollout-*.jsonl", recursive=True):
        start = end = None; pend = False
        for l in open(ro):
            e = json.loads(l); p = e.get("payload", {}); ty = p.get("type")
            if (e.get("type") == "response_item" and ty in ("function_call_output", "custom_tool_call_output")) or ty == "user_message":
                if not pend: start = t(e)
            elif e.get("type") == "token_usage_record" and start:
                end = t(e); pend = True            # the call has ended; its token_count follows the tool output
            elif ty == "token_count" and pend and (p.get("info") or {}).get("last_token_usage"):
                D.append((end - start).total_seconds()); O.append(p["info"]["last_token_usage"]["output_tokens"])
                pend = False; start = t(e)
    D, O = np.array(D), np.array(O)
    A = np.c_[np.ones_like(O), O]; (a, b), *_ = np.linalg.lstsq(A, D, rcond=None)
    big = O > 2000
    print(f"{root.split('/')[-1]:28} calls {len(D):5}  median out/call {int(np.median(O)):5}  "
          f"fit: overhead {a:5.1f} s + decode {1/b:5.0f} tok/s  |  calls >2k tokens: {big.sum():4} at median {np.median(O[big]/D[big]):5.0f} tok/s")
