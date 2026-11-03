"""Tables from a profile_workflow.py trace folder: where the replay's time went, and how much
of it the GPU or the CPU was actually working.

    python experiments/agentic_batch/analyze_profile.py DIR [DIR ...]   (writes DIR/profile.md)

Everything goes on a 0.1 s grid.  Per cell:
- GPU busy fraction = nvidia-smi utilization.gpu / 100: the share of the sample period in which
  at least one kernel ran.  It says nothing about how full the GPU was (one small kernel counts
  as busy), so GPU-busy seconds are an upper bound on GPU work;
- CPU cores = the tagged processes' utime+stime increase over the cell, by component;
- loading = GPU memory growing by > 200 MiB/s while the GPU is under 20 % busy (weights and
  buffers going onto the card).
A cell is "GPU" when the GPU is >= 20 % busy, else "loading" by the memory rule, else "CPU" when
the tagged processes use >= 0.5 core, else "idle" (waiting: the network, a sleep, a pipe).
"""
from __future__ import annotations

import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

DT = 0.1

COMPONENTS = [   # first match on the command line wins
    ("conda run wrapper", ("bin/conda run", "conda run ", "/condabin/conda", "conda.cli")),
    ("solver", ("solver_worker", "run-forward", "run_forward", "solver_server", "--solver")),
    ("SDXL", ("sdxl",)),
    ("TripoSG", ("generate_surface", "geometry_worker", "triposg")),
    ("Hunyuan3D", ("hunyuan",)),
    ("PartField", ("partfield",)),
    ("VoMP", ("vomp",)),
    ("CAD", ("cadcoder", "text2cad")),
    ("render/summary (inspect_session)", ("inspect_session",)),
    ("tool worker", ("tool_server",)),
    ("workflow tool (one-shot)", ("workflow_tool",)),
    ("runner + workflow graph (drive.py)", ("drive.py",)),
    ("ParaView", ("pvbatch", "pvpython")),
    ("nvidia-smi", ("nvidia-smi",)),
]


def component(cmd: str) -> str:
    low = cmd.lower()
    for name, keys in COMPONENTS:
        if any(k in low for k in keys):
            return name
    return "other: " + cmd[:80]


def load(d: Path):
    meta = json.loads((d / "commands.json").read_text())
    gpu = []
    for line in (d / "gpu.csv").read_text().splitlines()[1:]:
        t, util, mutil, mem, power = (float(x) for x in line.split(","))
        gpu.append((t, util, mem, power))
    procs = json.loads((d / "procs.json").read_text())
    tools = json.loads((d / "tools.json").read_text())
    dlog = [json.loads(l) for l in (d / "driver_log.jsonl").read_text().splitlines()] \
        if (d / "driver_log.jsonl").exists() else []
    apps = json.loads((d / "gpu_apps.json").read_text()) if (d / "gpu_apps.json").exists() else []
    return meta, gpu, procs, tools, dlog, apps


def grid(meta, gpu, procs):
    t0, t1 = meta["begin"], meta["end"]
    n = int((t1 - t0) / DT) + 1
    times = [t0 + i * DT for i in range(n)]
    gt = [g[0] for g in gpu]

    def at(t):
        i = bisect.bisect_right(gt, t) - 1
        return gpu[max(i, 0)] if gpu else (t, 0.0, 0.0, 0.0)

    util = [at(t)[1] / 100 for t in times]
    mem = [at(t)[2] for t in times]
    power = [at(t)[3] for t in times]
    cpu = defaultdict(lambda: [0.0] * n)
    names = {}
    for pid, rec in procs.items():
        comp = component(rec["cmd"])
        names[pid] = comp
        samples = rec["cpu"]
        for (ta, ca), (tb, cb) in zip(samples, samples[1:]):
            # The increase happened somewhere in (previous sample, tb]; the sampler keeps only
            # changes, so spread it over the last 0.1 s before tb (the sampling period).
            lo = max(ta, tb - DT)
            rate = (cb - ca) / max(tb - lo, 1e-6)
            for i in range(max(int((lo - t0) / DT), 0), min(int((tb - t0) / DT) + 1, n)):
                a, b = max(times[i], lo), min(times[i] + DT, tb)
                if b > a:
                    cpu[comp][i] += rate * (b - a) / DT
    return times, util, mem, power, cpu, names


def classify(i, util, mem, cpu_total):
    if util[i] >= 0.2:
        return "GPU"
    # Loading before CPU: deserializing weights keeps a core busy while they go onto the card.
    if 0 < i < len(mem) - 5 and (mem[i + 5] - mem[i - 5]) / (10 * DT) > 200:
        return "loading"
    if cpu_total[i] >= 0.5:
        return "CPU"
    return "idle"


def summarize(lo, hi, times, util, mem, power, cpu, cpu_total):
    t0 = times[0]
    a, b = max(int((lo - t0) / DT), 0), min(int((hi - t0) / DT), len(times))
    if b <= a:
        return None
    states = defaultdict(float)
    for i in range(a, b):
        states[classify(i, util, mem, cpu_total)] += DT
    comps = {k: sum(v[a:b]) * DT for k, v in cpu.items()}
    return {"wall": (b - a) * DT, "gpu_busy": sum(util[a:b]) * DT, "cpu_core_s": sum(cpu_total[a:b]) * DT,
            "states": dict(states), "cpu": {k: v for k, v in comps.items() if v >= 0.05},
            "peak_mem": max(mem[a:b]), "mem_start": mem[a], "energy_kj": sum(power[a:b]) * DT / 1000}


def phases(meta, dlog, tools):
    """(label, start, end) for the cards the runner sent, the work after a run's last card
    (stop payload, preload), each command's runner start-up (before its first card) and the
    tool calls."""
    out = []
    cards = [d for d in dlog if d.get("command") == "run" and d.get("t_start")]
    for c in meta["commands"]:
        mine = [d for d in cards if c["start"] <= d["t_start"] <= c["end"]]
        label = c["label"]
        if label == "paused (agent)":
            continue
        if not mine:
            out.append((f"{label}: whole command (stop={c.get('stop')})", c["start"], c["end"], "command"))
            continue
        out.append((f"{label}: start-up to first card", c["start"], mine[0]["t_select"], "runner"))
        prev_end = None
        for d in mine:
            if prev_end is not None and d["t_select"] - prev_end > 0.05:
                out.append((f"{label}: between cards", prev_end, d["t_select"], "runner"))
            if d["t_start"] - d["t_select"] > 0.05:
                out.append((f"{label}: GPU check before {d['kind']}", d["t_select"], d["t_start"], "runner"))
            out.append((f"{label}: card {d['kind']} {d.get('action') or ''}".rstrip(), d["t_start"], d["t_end"], "card"))
            prev_end = d["t_end"]
        out.append((f"{label}: after last card (stop={c.get('stop')})", prev_end, c["end"], "stop"))
    return out


def fmt_row(label, s, comps_shown):
    st = s["states"]
    cpu = "; ".join(f"{k} {v:.1f}" for k, v in sorted(s["cpu"].items(), key=lambda kv: -kv[1])[:comps_shown])
    return (f"| {label} | {s['wall']:.1f} | {s['gpu_busy']:.1f} | {st.get('GPU', 0):.1f} | {st.get('CPU', 0):.1f} | "
            f"{st.get('loading', 0):.1f} | {st.get('idle', 0):.1f} | {s['cpu_core_s']:.1f} | {cpu} |")


HEAD = ("| phase | wall s | GPU-busy s | GPU cells s | CPU-only s | loading s | idle s | CPU core-s | "
        "CPU core-s by component (top) |\n|---|---:|---:|---:|---:|---:|---:|---:|---|")


def analyze(d: Path) -> str:
    meta, gpu, procs, tools, dlog, apps = load(d)
    times, util, mem, power, cpu, names = grid(meta, gpu, procs)
    cpu_total = [sum(v[i] for v in cpu.values()) for i in range(len(times))]
    args = (times, util, mem, power, cpu, cpu_total)
    lines = [f"# Workflow profile: {Path(meta['source']).name} replayed as {meta['session_id']} (GPU {meta['gpu']})", ""]
    whole = None
    for c in meta["commands"]:
        if c["label"] == "paused (agent)":
            continue
        s = summarize(c["start"], c["end"], *args)
        whole = s if whole is None else merge(whole, s) if s else whole
    paused = sum(c["end"] - c["start"] for c in meta["commands"] if c["label"] == "paused (agent)")
    lines += ["## Whole replay (commands only; the agent's pauses, %.0f s, left out)" % paused, "",
              HEAD, fmt_row("all", whole, 6), ""]

    ph = phases(meta, dlog, tools)
    groups = defaultdict(lambda: None)
    lines += ["## Phases (in order)", "", HEAD]
    for label, lo, hi, kind in ph:
        s = summarize(lo, hi, *args)
        if s is None:
            continue
        lines.append(fmt_row(label, s, 4))
        g = groups[kind]
        groups[kind] = s if g is None else merge(g, s)
    lines += ["", "## Phases by kind", "", HEAD]
    for kind, s in groups.items():
        lines.append(fmt_row(kind, s, 5))

    # (tools.json is not tabled: the monitor writes a receipt's logs and its .json together at
    # the end of the call, so the sampler sees no start.)
    lines += ["", "## CPU core-seconds by component, whole replay", "", "| component | core-s | processes |", "|---|---:|---:|"]
    count = defaultdict(int)
    for pid, comp in names.items():
        count[comp] += 1
    for comp, v in sorted(whole["cpu"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {comp} | {v:.1f} | {count[comp]} |")
    lines += ["", f"Peak GPU memory {whole['peak_mem']:.0f} MiB; GPU energy {whole['energy_kj']:.1f} kJ.", ""]
    return "\n".join(lines)


def merge(a, b):
    out = {k: a[k] + b[k] for k in ("wall", "gpu_busy", "cpu_core_s", "energy_kj")}
    out["states"] = {k: a["states"].get(k, 0) + b["states"].get(k, 0) for k in set(a["states"]) | set(b["states"])}
    out["cpu"] = {k: a["cpu"].get(k, 0) + b["cpu"].get(k, 0) for k in set(a["cpu"]) | set(b["cpu"])}
    out["peak_mem"] = max(a["peak_mem"], b["peak_mem"])
    out["mem_start"] = a["mem_start"]
    return out


STEPS = [   # (row, how a phase label is matched)
    ("SDXL drawing (card)", lambda l, k: k == "card" and "draw_image" in l),
    ("generation from the drawing (card)", lambda l, k: k == "card" and l.endswith("card geometry_input")),
    ("PartField segmentation (card)", lambda l, k: k == "card" and "preview_segmentation" in l),
    ("region preview (card)", lambda l, k: k == "card" and "preview_regions" in l),
    ("texture paint (card)", lambda l, k: k == "card" and "texture_surface" in l),
    ("solve (card)", lambda l, k: k == "card" and l.endswith("card shape_opt_settings")),
    ("other cards", None),
    ("geometry stop payload", lambda l, k: k == "stop" and "stop=geometry" in l),
    ("regions stop payload", lambda l, k: k == "stop" and "stop=regions" in l),
    ("solve stop payload", lambda l, k: k == "stop" and "stop=solve" in l),
    ("other stops (drawing, texture, errors, finish)", None),
    ("runner start-up + GPU checks", lambda l, k: k == "runner"),
    ("whole commands (start, judge, errors before a card)", lambda l, k: k == "command"),
]


def compare(dirs: list[Path]) -> str:
    """Wall s / GPU-busy s / CPU core-s per workflow step, one column per trace."""
    per = []
    for d in dirs:
        meta, gpu, procs, tools, dlog, apps = load(d)
        times, util, mem, power, cpu, names = grid(meta, gpu, procs)
        cpu_total = [sum(v[i] for v in cpu.values()) for i in range(len(times))]
        rows = defaultdict(lambda: [0.0, 0.0, 0.0])
        for label, lo, hi, kind in phases(meta, dlog, {}):
            s = summarize(lo, hi, times, util, mem, power, cpu, cpu_total)
            if s is None:
                continue
            for name, match in STEPS:
                if match and match(label, kind):
                    break
            else:
                name = "other cards" if kind == "card" else "other stops (drawing, texture, errors, finish)"
            r = rows[name]
            r[0] += s["wall"]; r[1] += s["gpu_busy"]; r[2] += s["cpu_core_s"]
        per.append(rows)
    head = "| step | " + " | ".join(d.name for d in dirs) + " |\n|---|" + "---:|" * len(dirs)
    lines = ["Each cell: wall s / GPU-busy s / CPU core-s.", "", head]
    total = [[0.0, 0.0, 0.0] for _ in dirs]
    for name, _ in STEPS:
        cells = []
        for j, rows in enumerate(per):
            w, g, c = rows.get(name, [0, 0, 0])
            total[j] = [total[j][0] + w, total[j][1] + g, total[j][2] + c]
            cells.append(f"{w:.1f} / {g:.1f} / {c:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| **total** | " + " | ".join(f"**{w:.1f} / {g:.1f} / {c:.0f}**" for w, g, c in total) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]]
    for d in dirs:
        text = analyze(d)
        (d / "profile.md").write_text(text)
        print(text)
    if len(dirs) > 1:
        print(compare(dirs))
