"""Checks the output of test_variants_stub.sh.  python3 check_variants_stub.py TMP"""
import glob
import hashlib
import json
import os
import sys

TMP = sys.argv[1]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "illustrations", "agentic_results"))
import ar_runs  # noqa: E402

fails = []
check = lambda ok, what: (print(("PASS " if ok else "FAIL ") + what), ok or fails.append(what))


def runs(out):
    found = []
    for mp in sorted(glob.glob(os.path.join(TMP, out, "*", "metrics.json"))):
        found.append((os.path.dirname(mp), json.load(open(mp))))
    return found


def batch(out):
    bdir = glob.glob(os.path.join(TMP, out, "_batches", "*"))[0]
    return bdir, json.load(open(os.path.join(bdir, "summary.json")))["runs"]


# ------------------------------------------------------------ realizations
real = runs("out_real")
check(len(real) == 3, f"realizations: {len(real)} runs of 3")
seeds = sorted(m["variant"]["seed"] for _, m in real)
check(seeds == [41, 42, 43], f"realizations: seeds {seeds} = seed base + k")
for run, m in real:
    tag = os.path.basename(run)
    check(m["status"] == "ok" and m["variant"]["kind"] == "realization" and m["variant_kind"] == "realization",
          f"{tag}: ok, variant recorded")
    env = open(os.path.join(run, "slot_env.txt")).read()
    check(f"FEM_OPERATOR_SEED={m['variant']['seed']}" in env, f"{tag}: FEM_OPERATOR_SEED passed to the run")
    ses = json.load(open(os.path.join(run, "workflow", m["session_id"], "session.json")))
    check(ses.get("seed") == str(m["variant"]["seed"]), f"{tag}: the agent's commands saw the seed {ses.get('seed')}")
    check(m["instruction"] == "a stub bench sleep1" and m["anchor_instruction_matches"] is True,
          f"{tag}: the instruction is the given one, unchanged")
    r = ar_runs.load_run(run)
    check(r.session and r.session.endswith(m["session_id"]), f"{tag}: ar_runs.load_run reads it")
check(len({m["session_id"] for _, m in real}) == 3, "realizations: three distinct sessions")
_, rows = batch("out_real")
check(len(rows) == 3 and all(r.get("varied", "").startswith("realization") for r in rows),
      "realizations: summary rows carry the varied label")

# ------------------------------------------------------------ the source is never touched
source = os.path.join(TMP, "source_run", "workflow", "aaaaaaaaaaaa")
before = dict(line.split()[::-1] for line in open(os.path.join(TMP, "source_before.md5")))
now = {p: hashlib.md5(open(p, "rb").read()).hexdigest()
       for p in glob.glob(os.path.join(source, "**", "*"), recursive=True) if os.path.isfile(p)}
check(now == before, f"the source session is unchanged ({len(now)} files)")

# ------------------------------------------------------------ material sweep
authored = open(os.path.join(TMP, "stub.log.author")).read().splitlines()
check(len(authored) == 2 and all(a == "sonnet 2" for a in authored),
      f"material sweep: authored once, refused (homogeneous), authored again: {authored}")
bdir, rows = batch("out_mat")
variants = json.load(open(os.path.join(bdir, "variants.json")))
check(variants["kind"] == "material" and len(variants["variants"]) == 2 and not variants["problems"],
      "material sweep: two usable variants kept in the batch folder")
check(os.path.exists(os.path.join(bdir, "authoring", "prompt.txt"))
      and "previous reply was refused" in open(os.path.join(bdir, "authoring", "prompt.txt")).read(),
      "material sweep: the authoring prompt and the refusal are kept")
mat = runs("out_mat")
check(len(mat) == 2, f"material sweep: {len(mat)} runs of 2")
for run, m in mat:
    tag = os.path.basename(run)
    v = m.get("variant") or {}
    check(m["status"] == "ok" and v.get("kind") == "material_sweep" and m["varied"] == v.get("label"),
          f"{tag}: ok, varied = {m.get('varied')!r}")
    check(m["session_id"] not in (None, "aaaaaaaaaaaa") and v.get("session_id") == m["session_id"],
          f"{tag}: its own session {m['session_id']}, the one cloned for it")
    copy = os.path.join(run, "workflow", m["session_id"])
    check(os.path.exists(os.path.join(copy, "operator", "cloned_from.json")), f"{tag}: the session copy is the clone")
    check((m.get("case") or {}).get("material_mode") == "parts", f"{tag}: the case summary is recorded")
    prompt = open(os.path.join(run, "prompt.txt")).read()
    check(m["session_id"] in prompt and v["label"] in prompt and "drive.py start" in prompt
          and "{" + "SESSION}" not in prompt, f"{tag}: the prompt names the session and the variant")
    check(m["runner_runs"] == 3 and "no start output" in m["session_id_found_by"],
          f"{tag}: the session found from the agent's own drive.py commands")
check(len({m["session_id"] for _, m in mat}) == 2, "material sweep: two distinct sessions")

# ------------------------------------------------------------ BC sweep
check(len(open(os.path.join(TMP, "stub.log.author")).read().splitlines()) == 2,
      "bc sweep: --variants given, nothing authored")
bc = runs("out_bc")
check(len(bc) == 2, f"bc sweep: {len(bc)} runs of 2")
labels = sorted(m.get("varied") for _, m in bc)
check(labels == ["clamped legs, side push", "seat twist"], f"bc sweep: the given cases {labels}")
for run, m in bc:
    tag = os.path.basename(run)
    check(m["status"] == "ok" and m["variant"]["kind"] == "bc_sweep" and m["variant"]["spec"]["loads"],
          f"{tag}: ok, the load case recorded")
    check("load case" in open(os.path.join(run, "prompt.txt")).read(), f"{tag}: the BC prompt was used")
drives = open(os.path.join(TMP, "stub.log.drive")).read().splitlines()
check(len(drives) == 4 and all(d.startswith("clone ") and "--fresh-solve" in d for d in drives),
      f"one clone per sweep run ({len(drives)})")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED")
sys.exit(1 if fails else 0)
