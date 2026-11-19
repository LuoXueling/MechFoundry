"""Per run: time spent waiting on the model (from a tool result / user message to the model's next
item), the longest single wait, and output tokens per second of model time."""
import csv, glob, json, sys, datetime as dt
from pathlib import Path
root = Path(sys.argv[1])
def t(e): return dt.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
rows = []
for b in sorted((root / "_batches").iterdir()):
    if not (b / "summary.csv").exists(): continue
    for r in csv.DictReader(open(b / "summary.csv")):
        run = root / Path(r["path"]).name
        ro = sorted(glob.glob(str(run / "codex_home/sessions/**/rollout-*.jsonl"), recursive=True))
        if not ro: continue
        ev = [json.loads(l) for l in open(ro[-1])]
        waits, start = [], None
        for e in ev:
            p = e.get("payload", {}); ty = p.get("type")
            if e.get("type") == "response_item" and ty in ("function_call_output", "custom_tool_call_output") \
               or ty == "user_message":
                start = t(e)
            elif e.get("type") == "response_item" and ty in ("reasoning", "message", "function_call", "custom_tool_call", "local_shell_call") and start:
                waits.append((t(e) - start).total_seconds()); start = None
        mt = sum(waits); out = int(r["output_tokens"] or 0)
        rows.append((b.name, run.name[:45], round(float(r["wall_clock_s"])), round(mt), round(max(waits or [0])), len(waits), out, round(out / mt, 1) if mt else 0))
print(f"{'batch':16} {'run':45} {'wall':>5} {'model':>5} {'maxw':>5} {'calls':>5} {'out':>7} {'tok/s':>6}")
for x in rows: print(f"{x[0]:16} {x[1]:45} {x[2]:5} {x[3]:5} {x[4]:5} {x[5]:5} {x[6]:7} {x[7]:6}")
