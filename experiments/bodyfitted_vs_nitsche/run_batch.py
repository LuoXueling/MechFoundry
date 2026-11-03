#!/usr/bin/env python
"""Run the whole comparison over a batch: phase 1 (solve), phase 2 (evaluate), summary.

    python experiments/bodyfitted_vs_nitsche/run_batch.py                 # all sessions
    python experiments/bodyfitted_vs_nitsche/run_batch.py yo-yo elephant --gpus 0

Stdlib only.  Phase 1 runs ``compare_bodyfitted_nitsche.py --phase solve`` for
every session, one job per GPU; a job starts only when ``nvidia-smi
--query-compute-apps`` shows no process on its GPU (bounded wait).  A session
whose ``solve_done`` marker exists is skipped (resumable).  The session job is
an orchestrator: it meshes, and runs every solver rung in its own child process
(the ladders in the README), so a crash costs one rung; a job that ends without
the marker is run once more (it resumes from the rung records); a job that ends
with a non-zero code but with the marker written (an AMGX destructor abort at
exit after the work is saved) is done, and its exit code is kept as a note.  When every
session has finished phase 1, phase 2 runs ``--phase evaluate`` for every
session (metrics from the stored files only), then ``summarize.py``.  Every
job is under ``timeout``, logs to ``<output>/<object>/run.log`` and is appended
to ``<output>/_batch/jobs.jsonl``; a sampler records each job's peak GPU memory.

P1 entries the GPU cannot take (large meshes, out-of-memory or CUDA-fatal
failures) are written by the GPU job as ``<object>/cpu_jobs/<tag>.{npz,json}``
and solved by a separate CPU queue (``--cpu-workers`` jobs of
``cpu_p1_worker.py``, no CUDA context, ``--cpu-threads`` MKL/OpenMP threads
each), so a GPU slot never waits on a CPU solve.  After the GPU phases the queue
is drained; a CPU job the worker found too nonlinear for its fixed tangent and
too large for PARDISO goes to the lean GPU full Newton (``--phase attempt
--kind lean``, self-validating) on a free GPU; other failed CPU jobs get one
more repair round; then phase 2 and the summary.

    python experiments/bodyfitted_vs_nitsche/run_batch.py --gpus 0 1 --largest-first   # everything
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent / "compare_bodyfitted_nitsche.py"
SUMMARIZE = Path(__file__).resolve().parent / "summarize.py"
DEFAULT_PY = "/home/xl2254/anaconda3/envs/mechfoundry-warp117/bin/python"
SUMMARY_PY = "/home/xl2254/anaconda3/bin/python"  # has openpyxl
CPU_WORKER = Path(__file__).resolve().parent / "cpu_p1_worker.py"
# numpy, scipy, pyamg and pypardiso (MKL PARDISO); a throwaway venv under output/
CPU_PY = "/home/xl2254/implicit_optimize/output/bodyfitted_vs_nitsche/_cpu_env/bin/python"


def _main_checkout(root: Path) -> Path:
    try:
        common = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
        if common:
            main = (root / common).resolve().parent
            if (main / "output").is_dir():
                return main
    except Exception:  # noqa: BLE001
        pass
    return root


DATA_ROOT = _main_checkout(REPO)


def object_name(run_name):
    return re.sub(r"-\d{8}-\d{6}$", "", run_name)


def sessions(batch: Path):
    out = {}
    for run in sorted(os.listdir(batch)):
        metrics = batch / run / "metrics.json"
        if metrics.is_file():
            sid = json.loads(metrics.read_text())["session_id"]
            out[object_name(run)] = {"run": run, "session": str(batch / run / "workflow" / sid)}
    return out


def size_estimate(info):
    """The session's own Nitsche node count: big sessions run last."""
    for log in sorted(Path(info["session"]).glob("solve/api_fem_forward-*/log.txt"), reverse=True):
        found = re.findall(r"inner (\d+) nodes", log.read_text(errors="replace"))
        if found:
            return int(found[-1])
    return 0


FOREIGN_MEM_MIB, FOREIGN_UTIL = 2048.0, 10.0  # scheduling rule since the fresh run of 2026-10-16
# Hang watchdog (since the same run: natural-stone-arch's orchestrator hit
# "malloc(): unaligned tcache chunk detected" and sat idle for 65 min): a job
# whose log and records stay unchanged for HANG_SECONDS while its whole process
# tree uses under HANG_CPU of one core and its GPU stays under HANG_GPU % is
# killed and retried once.  Long PARDISO/CPU solves burn CPU, GPU rungs use the GPU.
HANG_SECONDS, HANG_CPU, HANG_GPU = 1200.0, 0.02, 5.0


def _tree(pid):
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        out.add(p)
        try:
            kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:  # noqa: BLE001
            kids = []
        todo += [int(k) for k in kids if int(k) not in out]
    return out


def _cpu_seconds(pids):
    tick, total = os.sysconf("SC_CLK_TCK"), 0.0
    for p in pids:
        try:
            f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / tick
        except Exception:  # noqa: BLE001
            pass
    return total


def _activity(out_dir):
    paths = [out_dir / "run.log", out_dir / "status.json", *(out_dir / "attempts").glob("*.json")]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def gpu_utilization(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return 100.0


def compute_apps():
    uuids = {}
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout
    for line in out.strip().splitlines():
        i, u = [p.strip() for p in line.split(",")]
        uuids[u] = int(i)
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
    apps = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[0] in uuids:
            try:
                apps.append((uuids[p[0]], int(p[1]), float(p[2])))
            except ValueError:
                pass
    return apps


class Batch:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        (self.out / "_batch").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.queue = []
        self.active = {}
        self.stop = False
        self.queue_extra = ()
        self.gpu_done = False
        self.cpu_claimed = set()
        self.cpu_busy = 0
        self.cpu_big = 0

    @staticmethod
    def claim_key(job):
        """A job written again (a repair round re-defers it) is a new job."""
        return f"{job}@{job.stat().st_mtime_ns}"

    @staticmethod
    def cpu_job_big(job):
        """A job whose PARDISO factor may need ~60-100 GB (>= 1.2 M nodes) runs
        alone among the big ones (elephant, 0.34 M nodes, peaked at 16 GB)."""
        try:
            return json.loads(job.read_text())["info"]["p1_vertices"] >= 1_200_000
        except Exception:  # noqa: BLE001
            return True

    def cpu_pending(self):
        """CPU jobs written and not yet done, failed, running or claimed, largest first."""
        found = []
        for job in self.out.glob("*/cpu_jobs/*.json"):
            tag = job.stem
            if any(job.with_name(tag + s).exists() for s in (".done", ".failed", ".running")):
                continue
            if self.claim_key(job) in self.cpu_claimed or not job.with_suffix(".npz").is_file():
                continue
            found.append((job.with_suffix(".npz").stat().st_size, job))
        return [j for _, j in sorted(found, reverse=True)]

    def cpu_worker(self, slot):
        while True:
            with self.lock:
                pending = self.cpu_pending()
                if self.cpu_big:
                    pending = [j for j in pending if not self.cpu_job_big(j)]
                job = pending[0] if pending else None
                big = job is not None and self.cpu_job_big(job)
                if job is not None:
                    self.cpu_claimed.add(self.claim_key(job))
                    self.cpu_busy += 1
                    self.cpu_big += big
                elif self.gpu_done:
                    return  # no GPU job can write a new one any more
            if job is None:
                time.sleep(20)
                continue
            # (a worker that finds only a big job while another big one runs waits)
            out_dir, tag = job.parent.parent, job.stem
            job.with_name(tag + ".running").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            threads = str(self.args.cpu_threads)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MKL_NUM_THREADS=threads, OMP_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS=threads, PYTHONUNBUFFERED="1")
            cmd = ["timeout", "--kill-after=60", str(self.args.cpu_timeout), "/usr/bin/time", "-v",
                   self.args.cpu_python, str(CPU_WORKER), str(out_dir), tag]
            self.log(f"CPU {slot}: {out_dir.name} {tag}")
            t0 = time.time()
            with open(job.with_name(tag + ".log"), "a") as logf:
                logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} CPU {slot}: {' '.join(cmd)}\n")
                logf.flush()
                rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
            job.with_name(tag + ".running").unlink(missing_ok=True)
            done = job.with_name(tag + ".done").exists()
            if not done and not job.with_name(tag + ".failed").exists():
                job.with_name(tag + ".failed").write_text(f"worker exit code {rc} (timeout or crash)")
                try:  # the record must say so, so the repair round re-queues it
                    rpath = out_dir / f"result_{tag}.json"
                    rec = json.loads(rpath.read_text())
                    rec.update(ok=False, deferred_cpu=False, error=f"CPU solve failed: worker exit code {rc}")
                    rpath.write_text(json.dumps(rec, indent=2))
                except Exception:  # noqa: BLE001
                    pass
            self.record({"object": out_dir.name, "phase": "cpu", "tag": tag, "returncode": rc,
                         "seconds": time.time() - t0, "completed": done})
            self.log(f"CPU {slot}: {out_dir.name} {tag} rc {rc} done={done} {time.time() - t0:.0f} s")
            with self.lock:
                self.cpu_busy -= 1
                self.cpu_big -= big

    def log(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with self.lock, open(self.out / "_batch" / "driver.log", "a") as f:
            f.write(line + "\n")

    def record(self, rec):
        with self.lock, open(self.out / "_batch" / "jobs.jsonl", "a") as f:
            f.write(json.dumps({**rec, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")

    def sampler(self):
        with open(self.out / "_batch" / "vram.csv", "a") as f:
            while not self.stop:
                try:
                    for gpu, pid, mib in compute_apps():
                        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{gpu},{pid},{mib}\n")
                        with self.lock:
                            job = self.active.get(gpu)
                            if job and pid in job["pids"]:
                                job["peak"] = max(job["peak"], mib)
                    f.flush()
                except Exception as exc:  # noqa: BLE001
                    f.write(f"# sampler error {exc}\n")
                time.sleep(self.args.sample)

    def wait_free(self, gpu):
        """A GPU is busy only if the processes on it (none of them a job of
        ours: one job per GPU) hold more than FOREIGN_MEM_MIB together, or the
        GPU stays above FOREIGN_UTIL % utilization over a 30 s window.  Light,
        idle processes -- the fem_workflow UI's warm tool_server (~0.3 GB),
        pvbatch renders -- do not block; a real solve or generation in the UI
        does.  Our job then still has the GPU's memory less at most 2 GB."""
        deadline = time.time() + self.args.max_wait
        while time.time() < deadline:
            busy = [a for a in compute_apps() if a[0] == gpu]
            if not busy:
                return True
            mem = sum(a[2] for a in busy)
            if mem <= FOREIGN_MEM_MIB:
                util = []
                for _ in range(6):
                    util.append(gpu_utilization(gpu))
                    time.sleep(5)
                if min(util) <= FOREIGN_UTIL:
                    self.log(f"GPU {gpu}: light foreign processes {busy} (utilization {util}); proceeding")
                    return True
            self.log(f"GPU {gpu} busy ({busy}); waiting")
            time.sleep(60)
        return False

    def run_one(self, gpu, name, info, phase, extra=()):
        out_dir = self.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["timeout", "--kill-after=60", str(self.args.timeout), self.args.python, str(SCRIPT),
               "--session", info["session"], "--output", str(out_dir), "--phase", phase, *extra, *self.args.extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
        # No XLA_PYTHON_CLIENT_ALLOCATOR=platform on retries: with it, every
        # retry (sofa, lighthouse, dental-crown) died within 15 s in the warm-up
        # (cusparseCreate cannot initialise its CUDA context, amgx_exception).
        # The large meshes that ran out of memory go to the CPU path instead.
        t0 = time.time()
        with open(out_dir / "run.log", "a") as logf:
            logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu}: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
            with self.lock:
                self.active[gpu] = {"pids": {proc.pid}, "peak": 0.0}
            hang, idle_since, last_cpu, last_check = False, None, None, time.time()
            while proc.poll() is None:
                try:
                    kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True,
                                          timeout=30).stdout.split()
                    with self.lock:
                        self.active[gpu]["pids"].update(int(k) for k in kids)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5)
                if time.time() - last_check < 60:
                    continue
                last_check = time.time()
                tree = _tree(proc.pid)
                cpu = _cpu_seconds(tree)
                quiet = (last_cpu is not None and cpu - last_cpu < HANG_CPU * 60
                         and time.time() - _activity(out_dir) > 60 and gpu_utilization(gpu) < HANG_GPU)
                last_cpu = cpu
                idle_since = (idle_since or time.time()) if quiet else None
                if idle_since and time.time() - _activity(out_dir) > HANG_SECONDS \
                        and time.time() - idle_since > HANG_SECONDS - 120:
                    self.log(f"GPU {gpu}: {name} hung (no log/record change for {HANG_SECONDS / 60:.0f} min, "
                             f"CPU and GPU idle); killing {sorted(tree)}")
                    for p in tree:
                        try:
                            os.kill(p, 9)
                        except Exception:  # noqa: BLE001
                            pass
                    hang = True
                    break
            proc.wait()
            with self.lock:
                peak = self.active.pop(gpu, {}).get("peak", 0.0)
        try:
            stage = json.loads((out_dir / "status.json").read_text()).get("stage")
        except Exception:  # noqa: BLE001
            stage = None
        return {"object": name, "phase": phase, "gpu": gpu, "returncode": proc.returncode,
                "seconds": time.time() - t0, "peak_vram_mib": peak, "last_stage": stage,
                "timed_out": proc.returncode in (124, 137), "extra": list(extra), "hang": hang}

    def worker(self, gpu, phase):
        while True:
            with self.lock:
                if not self.queue:
                    return
                name, info = self.queue.pop(0)
            if not self.wait_free(gpu):
                self.record({"object": name, "phase": phase, "gpu": gpu,
                             "error": f"GPU {gpu} never became free within {self.args.max_wait} s"})
                continue
            self.log(f"GPU {gpu}: {phase} {name}")
            rec = self.run_one(gpu, name, info, phase, self.queue_extra)
            done = (self.out / name / ("solve_done" if phase == "solve" else "comparison.json")).is_file()
            if phase == "solve" and not done:
                self.record({**rec, "note": ("hang: killed by the watchdog; retried" if rec["hang"]
                                             else "no solve_done marker; retried")})
                self.log(f"GPU {gpu}: {name} ended rc {rec['returncode']} at {rec['last_stage']} "
                         "without finishing; retrying once")
                if self.wait_free(gpu):
                    rec = self.run_one(gpu, name, info, phase, self.queue_extra)
                    done = (self.out / name / "solve_done").is_file()
            if done and rec["returncode"] not in (0, None):
                rec["note"] = (f"exit code {rec['returncode']} after every output was saved "
                               "(e.g. an AMGX destructor abort at process exit): done")
            rec["completed"] = done
            self.record(rec)
            self.log(f"GPU {gpu}: {phase} {name} rc {rec['returncode']} done={done} "
                     f"{rec['seconds']:.0f} s, peak {rec['peak_vram_mib']:.0f} MiB")

    def lean_jobs(self, names):
        """CPU jobs that failed as too nonlinear for the fixed tangent ('needs full Newton')."""
        out = []
        for n in names:
            for failed in sorted((self.out / n / "cpu_jobs").glob("*.failed")):
                tag = failed.stem
                rec = self.out / n / f"result_{tag}.json"
                text = failed.read_text(errors="replace") + (rec.read_text() if rec.is_file() else "")
                if "needs full Newton" in text and not failed.with_suffix(".done").exists():
                    out.append((n, tag))
        return out

    def lean_phase(self, found, names):
        queue, lock = self.lean_jobs(names), threading.Lock()
        self.log(f"lean GPU full Newton: {len(queue)} jobs {queue}")

        def worker(gpu):
            while True:
                with lock:
                    if not queue:
                        return
                    name, tag = queue.pop(0)
                if not self.wait_free(gpu):
                    continue
                record = self.out / name / "cpu_jobs" / f"{tag}.lean.json"
                rec = self.run_one(gpu, name, found[name], "attempt",
                                   ("--kind", "lean", "--tag", tag, "--record", str(record)))
                ok = record.is_file() and json.loads(record.read_text()).get("ok")
                self.record({**rec, "phase": "lean", "tag": tag, "completed": bool(ok)})
                self.log(f"GPU {gpu}: lean {name} {tag} ok={ok} {rec['seconds']:.0f} s")

        threads = [threading.Thread(target=worker, args=(g,)) for g in self.args.gpus]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    def run_phase(self, phase, items):
        self.queue = list(items)
        self.log(f"{phase}: {len(self.queue)} jobs on GPUs {self.args.gpus}")
        workers = [threading.Thread(target=self.worker, args=(g, phase)) for g in self.args.gpus]
        for w in workers:
            w.start()
            time.sleep(2)
        for w in workers:
            w.join()


TRUE_TETGEN = ("self-intersects", "no TetGen variant meshed", "no mesh could be built")


def needs_repair(out_dir: Path):
    """A session without its solve marker, or with a failed stage that is not a
    true TetGen failure."""
    if not (out_dir / "solve_done").is_file():
        return True
    for f in list(out_dir.glob("nitsche_d?.json")) + list(out_dir.glob("result_d*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return True
        if rec.get("deferred_cpu") and (f.parent / "cpu_jobs" / (f.stem[len("result_"):] + ".json")).is_file():
            continue  # the CPU queue has it
        if "needs full Newton" in str(rec.get("error")):
            continue  # the lean GPU phase has it
        if not rec.get("ok") and not any(k in (rec.get("error") or "") for k in TRUE_TETGEN):
            if "no Nitsche run" in (rec.get("error") or ""):
                continue  # follows the Nitsche failure, which is repaired itself
            return True
    return False


def run_info(args):
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_experiment": bool(git("status", "--porcelain", "--", "experiments", "src")),
            "driver_options": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hang_watchdog": f"a job with no log/record change for {HANG_SECONDS / 60:.0f} min while its process "
                             f"tree uses < {HANG_CPU:.0%} of a core and its GPU < {HANG_GPU:.0f} % is killed "
                             "and retried once, recorded as hang (scheduling only)",
            "gpu_busy_rule": f"foreign processes block a GPU only above {FOREIGN_MEM_MIB:.0f} MiB together or "
                             f"{FOREIGN_UTIL:.0f} % utilization over 30 s (scheduling only)",
            "gpus": subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=60).stdout.strip().splitlines()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("objects", nargs="*", help="object names (default: every session of the batch)")
    ap.add_argument("--batch", default=str(DATA_ROOT / "output" / "agentic_batch50_opus"))
    ap.add_argument("--output", default=str(DATA_ROOT / "output" / "bodyfitted_vs_nitsche"))
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--timeout", type=int, default=6 * 3600,
                    help="seconds per job (a session's whole ladder; a deep Nitsche ladder took over 2 h)")
    ap.add_argument("--max-wait", type=int, default=6 * 3600, help="seconds to wait for a free GPU")
    ap.add_argument("--sample", type=int, default=10, help="VRAM sampling period [s]")
    ap.add_argument("--phases", nargs="+", choices=("solve", "evaluate", "summary"),
                    default=["solve", "evaluate", "summary"])
    ap.add_argument("--largest-first", action="store_true", help="run the largest sessions first")
    ap.add_argument("--exclude", nargs="*", default=[], help="objects to leave out")
    ap.add_argument("--repair-rounds", type=int, default=3,
                    help="rounds of re-running sessions with non-TetGen failures after phase 1")
    ap.add_argument("--cpu-workers", type=int, default=2, help="concurrent CPU P1 jobs")
    ap.add_argument("--cpu-threads", type=int, default=24, help="MKL/OpenMP threads per CPU job")
    ap.add_argument("--cpu-timeout", type=int, default=6 * 3600, help="seconds per CPU job")
    ap.add_argument("--cpu-python", default=CPU_PY if Path(CPU_PY).is_file() else DEFAULT_PY)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="further arguments for compare_bodyfitted_nitsche.py")
    args = ap.parse_args()
    found = sessions(Path(args.batch))
    names = args.objects or sorted(found, key=lambda n: size_estimate(found[n]))
    if args.largest_first:
        names = sorted(names, key=lambda n: -size_estimate(found[n]))
    names = [n for n in names if n not in set(args.exclude)]
    missing = [n for n in names if n not in found]
    if missing:
        sys.exit(f"unknown objects: {missing}")
    batch = Batch(args)
    (batch.out / "_summary").mkdir(parents=True, exist_ok=True)
    # A resumed driver keeps what an earlier start recorded (hand-written change
    # notes such as scheduling_change) and lists the earlier starts.
    info_path = batch.out / "_summary" / "run_info.json"
    info = run_info(args)
    try:
        old = json.loads(info_path.read_text())
        info = {**old, **info, "previous_starts": old.get("previous_starts", []) + [
            {k: old.get(k) for k in ("started", "git_commit", "git_branch")}]}
    except Exception:  # noqa: BLE001
        pass
    info_path.write_text(json.dumps(info, indent=2))
    sampler = threading.Thread(target=batch.sampler, daemon=True)
    sampler.start()
    cpu = [threading.Thread(target=batch.cpu_worker, args=(k,)) for k in range(args.cpu_workers)]
    for w in cpu:
        w.start()
    if "solve" in args.phases:
        batch.run_phase("solve", [(n, found[n]) for n in names if not (batch.out / n / "solve_done").is_file()])
        # Repair rounds: every recorded failure that is not a true TetGen failure
        # is run again in a fresh process alone on its GPU.
        for round_ in range(args.repair_rounds):
            todo = [n for n in names if needs_repair(batch.out / n)]
            if not todo:
                break
            batch.log(f"repair round {round_ + 1}: {todo}")
            batch.queue_extra = ("--retry-failed",)
            batch.run_phase("solve", [(n, found[n]) for n in todo])
            batch.queue_extra = ()
    # Drain the CPU queue; a CPU job that failed gets one more GPU-side repair
    # (it is deferred again with a fresh job), then the queue is drained again.
    for final in range(2):
        batch.gpu_done = True
        for w in cpu:
            w.join()
        if "solve" in args.phases:
            batch.lean_phase(found, names)
        todo = [n for n in names if needs_repair(batch.out / n)] if "solve" in args.phases else []
        if final == 1 or not todo:
            break
        batch.log(f"final repair round after the CPU queue: {todo}")
        batch.gpu_done = False
        cpu = [threading.Thread(target=batch.cpu_worker, args=(k,)) for k in range(args.cpu_workers)]
        for w in cpu:
            w.start()
        batch.queue_extra = ("--retry-failed",)
        batch.run_phase("solve", [(n, found[n]) for n in todo])
        batch.queue_extra = ()
    if "evaluate" in args.phases:
        batch.run_phase("evaluate", [(n, found[n]) for n in names if (batch.out / n / "solve.json").is_file()])
    batch.stop = True
    if "summary" in args.phases:
        r = subprocess.run([SUMMARY_PY, str(SUMMARIZE), "--output", str(batch.out)], capture_output=True,
                           text=True, timeout=1800)
        batch.log(f"summary: {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    batch.log("all phases finished")


if __name__ == "__main__":
    main()
