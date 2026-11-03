#!/usr/bin/env bash
# Run one full object instruction N times through the FEM operator skill, one headless Claude
# Code session per run, each generating its own realization of the object (another body) and
# simulating it.  Each run is an ordinary full run of run_batch.sh's kind, recorded in the same
# layout (illustrations/agentic_results/ar_runs.py reads it).
#
#   experiments/agentic_batch/run_realizations.sh [options] -n N "INSTRUCTION"
#
# INSTRUCTION: the whole object description with its loading, e.g. "a wooden park bench, 1.8 m
# long, loaded by two adults sitting at mid-span, feet on the ground".
#
# For a shape sweep whose bodies must be generated, not built (agents otherwise build
# simple prismatic objects, giving near-identical realizations), pass
# --template experiments/agentic_batch/prompt_template_generated.txt: prompt_template.txt
# plus one sentence requiring a text, drawing or image route (the showcase's shape row).
#
# Options (beside the common ones of ab_common.sh: -j, --model, --timeout, --out, --template,
# --effort, --no-path-note, --memory, --cache-ttl, --gpus, --dry-run):
#   -n, --runs N        how many realizations (default 2)
#   --seed-base S       run k's generation seed is S + k (default 100; the geometry seed, which the
#                       generators take as 0-100, is folded into 1-100: run k gets k there)
#   -h, --help
#
# What makes the runs differ: every run is a new workflow session, so none reuses another's
# tool or API caches (both are per session), and each run gets FEM_OPERATOR_SEED=S+k: the runner
# writes that seed into the plan for every generation seed the agent left at 0 (TripoSG's
# generation seed, the SDXL drawing's image_seed, the painter's texture_seed; drive.py
# with_batch_seed), so the same words generate a different body.  A seed the agent sets itself
# (a retry) is kept.  Nothing about the physics or the skill's rules changes.
#
# Per run, metrics.json `variant` = {"kind": "realization", "index": k, "seed": S+k, "label": ...};
# per batch, _batches/<id>/ has summary.csv / summary.json / ar_runs.txt as for run_batch.sh.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ab_common.sh
. "$HERE/ab_common.sh"

N=2 SEED_BASE=100 INSTR=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -n|--runs) N="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        -*) if ab_common_opt "$@"; then shift "$AB_SHIFT"; else echo "unknown option: $1" >&2; usage >&2; exit 2; fi ;;
        *) INSTR="$1"; shift ;;
    esac
done
[ -n "$INSTR" ] || { echo "no instruction given" >&2; usage >&2; exit 2; }
[ "$N" -ge 1 ] 2>/dev/null || { echo "bad -n: $N" >&2; exit 2; }
[ "$SEED_BASE" -ge 0 ] 2>/dev/null || { echo "bad --seed-base: $SEED_BASE" >&2; exit 2; }
ab_setup

run_one() {   # $1 = index, $2 = slot
    local idx="$1" slot="$2" run seed
    seed=$((SEED_BASE + idx))
    run="$(ab_new_run_dir "$INSTR")"
    "$PY" "$HELPER" prompt --template "$TEMPLATE" --worktree "$WORKTREE" --object "$INSTR" > "$run/prompt.txt"
    "$PY" - "$run/variant.json" "$idx" "$N" "$seed" <<'PYEOF'
import json, sys
path, k, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
json.dump({"kind": "realization", "index": k, "of": n, "seed": seed,
           "label": f"realization {k} of {n} (seed {seed})"}, open(path, "w"), indent=1)
PYEOF
    ab_launch "$idx" "$INSTR" "$slot" "$run" "FEM_OPERATOR_SEED=$seed"
}

log "batch $BATCH_ID: $N realization(s) of \"$INSTR\", seeds $((SEED_BASE + 1))..$((SEED_BASE + N)), $JOBS at a time, model $MODEL, effort $EFFORT, timeout ${TIMEOUT}s, memory $([ "$NO_MEMORY" = 1 ] && echo off || echo on), cache ttl $CACHE_TTL"
log "runs in $OUT; batch files in $BATCH_DIR"
if [ "$DRY" = 1 ]; then
    for ((i = 1; i <= N; i++)); do log "[dry] $i seed $((SEED_BASE + i)) \"$INSTR\""; done
    exit 0
fi
ab_schedule run_one "$N"
