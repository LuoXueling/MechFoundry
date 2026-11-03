"""Replay one finished operator session's plan through the runner, without the agent, and
record where the time goes: how much of it is the GPU or the CPU actually working.

    conda run --no-capture-output -n mechfoundry-agents python -u \
        experiments/agentic_batch/profile_workflow.py SOURCE_SESSION_DIR --gpu 1 --out DIR

SOURCE_SESSION_DIR is a session folder with operator/brief.md and operator/plan.json (a run
folder's workflow/<session_id>/).  The replay starts a fresh session in its own runner output
root, copies the brief and the final plan in, and runs `drive.py run` until it finishes,
accepting every stop (running again is the approval) and handing in the source's judgement at
the solve stop.  It runs the commands exactly as the agent does (`conda run ... drive.py ...`),
so the runner's own time is in it too; the agent's thinking is not.

Sampled while it runs (stdlib only; nothing here is installed in mechfoundry-agents):
- the GPU: `nvidia-smi -lms 100` (utilization = the fraction of the sample period in which any
  kernel ran, memory, power), and every 0.5 s which processes hold memory on it;
- the CPU: every 0.1 s, utime+stime of every process whose environment carries this replay's
  WFPROFILE_TAG, which the workers inherit even after they detach (workers.py passes
  os.environ on).  A process's CPU after its last sample (under 0.1 s) is lost;
- the tool calls: a receipt's stdout log appearing (the call starts) and its .json (it ends);
- each command's start and end, and the runner's own driver_log.jsonl (per card t_select,
  t_start, t_end).

Writes to DIR: gpu.csv, procs.json, tools.json, commands.json, driver_log.jsonl and the
session id; `analyze_profile.py DIR` turns them into the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
DRIVE = "src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py"
CLK = os.sysconf("SC_CLK_TCK")


class Sampler:
    def __init__(self, tag: str, gpu: int, out: Path):
        self.tag, self.gpu, self.out = tag.encode(), gpu, out
        self.stop = threading.Event()
        self.gpu_rows: list[tuple] = []
        self.apps: list[tuple] = []
        self.procs: dict[int, dict] = {}      # pid -> {cmd, ppid, first, last, cpu: [(t, ticks)]}
        self.foreign: set[int] = set()
        self.tools: dict[str, dict] = {}
        self.receipts: Path | None = None
        self.threads = [threading.Thread(target=f, daemon=True)
                        for f in (self._gpu, self._apps, self._cpu, self._receipts)]

    def start(self):
        for t in self.threads:
            t.start()

    def finish(self):
        self.stop.set()
        self._smi.terminate()
        for t in self.threads:
            t.join(timeout=5)

    def _gpu(self):
        self._smi = subprocess.Popen(
            ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
             "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
        for line in self._smi.stdout:
            if self.stop.is_set():
                break
            try:
                util, mutil, mem, power = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.gpu_rows.append((time.time(), util, mutil, mem, power))

    def _apps(self):
        while not self.stop.wait(0.5):
            try:
                text = subprocess.run(["nvidia-smi", f"--id={self.gpu}", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                      timeout=10).stdout
            except subprocess.TimeoutExpired:
                continue
            now = time.time()
            for line in text.splitlines():
                try:
                    pid, mem = (int(x) for x in line.split(","))
                except ValueError:
                    continue
                self.apps.append((now, pid, mem))

    def _cpu(self):
        uid = os.getuid()
        while not self.stop.wait(0.1):
            now = time.time()
            for entry in os.scandir("/proc"):
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid in self.foreign:
                    continue
                try:
                    if pid not in self.procs:
                        if entry.stat().st_uid != uid:
                            self.foreign.add(pid)
                            continue
                        env = Path(f"/proc/{pid}/environ").read_bytes()
                        if b"WFPROFILE_TAG=" + self.tag not in env:
                            self.foreign.add(pid)
                            continue
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                        self.procs[pid] = {"cmd": cmd.strip(), "first": now, "cpu": []}
                    stat = Path(f"/proc/{pid}/stat").read_text()
                except (OSError, PermissionError):
                    continue
                fields = stat[stat.rindex(")") + 2:].split()
                rec = self.procs[pid]
                rec["ppid"] = int(fields[1])
                ticks = int(fields[11]) + int(fields[12])    # utime + stime
                if not rec["cpu"] or rec["cpu"][-1][1] != ticks:
                    rec["cpu"].append((now, ticks))
                rec["last"] = now

    def _receipts(self):
        while not self.stop.wait(0.1):
            if self.receipts is None or not self.receipts.is_dir():
                continue
            now = time.time()
            for f in self.receipts.iterdir():
                key, _, suffix = f.name.partition(".")
                rec = self.tools.setdefault(key, {})
                if suffix == "stdout.log" and "start" not in rec:
                    rec["start"] = now
                elif suffix == "json" and "end" not in rec:
                    rec["end"] = now
                    try:
                        rec["tool"] = json.loads(f.read_text()).get("tool")
                    except (OSError, ValueError):
                        rec.pop("end")

    def dump(self):
        (self.out / "gpu.csv").write_text("t,util,mem_util,mem_mib,power_w\n" + "".join(
            ",".join(f"{x:.3f}" if i == 0 else f"{x:g}" for i, x in enumerate(r)) + "\n" for r in self.gpu_rows))
        (self.out / "gpu_apps.json").write_text(json.dumps(self.apps))
        (self.out / "procs.json").write_text(json.dumps(
            {pid: {**r, "cpu": [(t, k / CLK) for t, k in r["cpu"]]} for pid, r in self.procs.items()}))
        (self.out / "tools.json").write_text(json.dumps(self.tools, indent=1))


def run(cmd: list[str], env: dict, log: list, label: str) -> dict:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, env=env, capture_output=True, text=True)
    t1 = time.time()
    text = proc.stdout
    value = {}
    if "{" in text:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        except ValueError:
            pass
    log.append({"label": label, "start": t0, "end": t1, "returncode": proc.returncode,
                "stop": value.get("stop"), "ok": value.get("ok"), "error": value.get("error"),
                "steps": value.get("steps")})
    print(f"{t1 - t0:7.1f} s  {label}: stop={value.get('stop')} ok={value.get('ok')} "
          f"{(str(value.get('error')) or '')[:300] if value.get('error') else ''}", flush=True)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-runs", type=int, default=20)
    ap.add_argument("--drop-route", action="append", default=[],
                    help="take this route off the copied plan's geometry_routes (repeatable)")
    ap.add_argument("--pause-on-error", action="store_true",
                    help="on a runner error, wait for DIR/continue (the plan fixed by hand) instead of ending")
    ap.add_argument("--slot", help="runner output root to reuse (its compile caches warm), relative to the "
                                   "worktree; default a new output/fem_workflow_operator/_slots/profile-<tag>")
    ap.add_argument("--generation-settings",
                    help="JSON merged into the copied plan's geometry.generation_settings.  Give "
                         "inference_steps too: a generation_settings block without it takes the "
                         "model's 16 even for an image (whose default, with no block, is 50)")
    ap.add_argument("--drop-key", action="append", default=[],
                    help="remove this top-level section from the copied plan (repeatable)")
    args = ap.parse_args()
    src_op = args.source / "operator"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:10]
    slot = args.slot or f"output/fem_workflow_operator/_slots/profile-{tag}"
    env = {**os.environ, "WFPROFILE_TAG": tag, "FEM_OPERATOR_OUTPUT_ROOT": slot,
           "FEM_OPERATOR_GPUS": str(args.gpu)}
    base = ["conda", "run", "--no-capture-output", "-n", "mechfoundry-agents", "python", "-u", DRIVE]
    sampler = Sampler(tag, args.gpu, out)
    commands: list[dict] = []
    sampler.start()
    t_begin = time.time()
    try:
        value = run(base + ["start"], env, commands, "start")
        sid = value.get("session_id")
        if not sid:
            raise SystemExit(f"start failed: {value}")
        session = WORKTREE / slot / sid
        sampler.receipts = session / "tool_receipts"
        op = session / "operator"
        shutil.copy(src_op / "brief.md", op / "brief.md")
        plan_value = json.loads((src_op / "plan.json").read_text())
        if args.drop_route and plan_value.get("geometry_routes"):
            # The source agent tried these before the route it kept; the replay goes straight
            # to the kept one (the runner refuses a later route while an earlier one is untried).
            plan_value["geometry_routes"] = [r for r in plan_value["geometry_routes"] if r not in args.drop_route]
        if args.generation_settings:
            geometry = plan_value.setdefault("geometry", {})
            geometry["generation_settings"] = {**(geometry.get("generation_settings") or {}),
                                               **json.loads(args.generation_settings)}
        for key in args.drop_key:
            # e.g. region_revisions: they correct the selectors the source session's API call
            # synthesized, which a fresh session's call need not repeat.
            plan_value.pop(key, None)
        (op / "plan.json").write_text(json.dumps(plan_value, indent=1))
        plan = str(op / "plan.json")
        judged = False
        for i in range(args.max_runs):
            value = run(base + ["run", sid, "--plan", plan], env, commands, f"run {i + 1}")
            stop = value.get("stop")
            if stop == "finished" or stop in ("limit", None):
                break
            if stop == "error":
                if not args.pause_on_error:
                    break
                # As the agent would: the plan is fixed (by hand, in the session's operator/plan.json)
                # while the workers stay warm and the sampler runs; the pause is agent time.
                flag = out / "continue"
                t_pause = time.time()
                print(f"PAUSED: fix {plan}, then touch {flag} (or {out / 'abort'})", flush=True)
                while not flag.exists() and not (out / "abort").exists():
                    time.sleep(1.0)
                commands.append({"label": "paused (agent)", "start": t_pause, "end": time.time()})
                if (out / "abort").exists():
                    break
                flag.unlink()
                continue
            if stop == "solve" and not judged and (src_op / "judgement.md").exists():
                shutil.copy(src_op / "judgement.md", op / "judgement.md")
                run(base + ["judge", sid, "--file", str(op / "judgement.md")], env, commands, "judge")
                judged = True
    finally:
        t_end = time.time()
        run(base + ["gpu", "--stop-workers"], env, commands, "stop workers")
        time.sleep(1.0)
        sampler.finish()
        sampler.dump()
        (out / "commands.json").write_text(json.dumps(
            {"begin": t_begin, "end": t_end, "gpu": args.gpu, "tag": tag, "slot": slot,
             "session_id": locals().get("sid"), "source": str(args.source.resolve()),
             "commands": commands}, indent=1))
        if locals().get("sid"):
            shutil.copy(WORKTREE / slot / sid / "operator" / "driver_log.jsonl", out / "driver_log.jsonl")
    print(f"total {t_end - t_begin:.1f} s; traces in {out}")


if __name__ == "__main__":
    main()
