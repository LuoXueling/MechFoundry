#!/usr/bin/env bash
# Run a list of objects through the FEM operator skill, one headless Claude Code
# session (`claude -p`) per object, N at a time, and record each run for the
# agentic-results figure (illustrations/agentic_results/ar_runs.py reads the layout).
#
#   experiments/agentic_batch/run_batch.sh [options] LIST_FILE
#
# LIST_FILE: one instruction per line; blank lines and lines starting with # are skipped.
#
# Options:
#   -j, --jobs N        runs at once (default: the number of GPUs, `nvidia-smi -L`)
#   --model M           Claude model (default: sonnet)
#   --timeout SEC       wall-clock limit per run, then the run is killed (default: 7200)
#   --out DIR           where run folders go (default: <worktree>/output/agentic_batch)
#   --template FILE     the initial instruction, {WORKTREE} and {OBJECT} replaced
#                       (default: prompt_template.txt beside this script)
#   --effort LEVEL      thinking effort, low|medium|high|xhigh|max (default: medium), passed
#                       as the CLI's --effort; without it a run would take whatever
#                       ~/.claude/settings.json says, unrecorded
#   --no-path-note      do not append the one-line system-prompt note that says where the
#                       repository root is (default: appended; see run_one)
#   --memory            leave auto-memory on (default: off, --settings autoMemoryEnabled
#                       false; with it on, the user's memory index is in every session's
#                       context: about 4 k tokens a call, and rules the skill never wrote)
#   --cache-ttl 5m|1h   prompt-cache TTL, the CLI's documented `promptCacheTtl` setting
#                       (default: 5m; unset, a subscription takes 1h, whose cache writes
#                       cost 2x input against 5m's 1.25x; the longest gap between a run's
#                       calls in batch 20261009-182254 was 165 s)
#   --gpus LIST         comma-separated GPU indices the slots take in turn (default: all)
#   --dry-run           print what would run, start nothing
#   -h, --help
#
# Isolation: each of the N concurrent slots has its own runner output root,
# output/fem_workflow_operator/_slots/<batch_id>-<k>/ (FEM_OPERATOR_OUTPUT_ROOT), so its
# own resident workers, and its own GPU, k modulo the GPU count (FEM_OPERATOR_GPUS); the
# runner links each session to output/fem_workflow_operator/<session_id>, the path the
# skill names.  A slot's workers stay warm from one of its runs to the next and are
# stopped when the batch ends.  Measured on a shared root (batch 20261008-142904): one
# run's preload stopped the other's paint worker, a paint ran out of memory beside the
# sibling's, and the sibling caused 7 waits for an idle GPU.
#
# Environment: CLAUDE_BIN (default `claude`; the tests put a stub here), and for the
# stub test only AB_WORKFLOW_ROOT / AB_PROJECTS_DIR (where sessions and transcripts are).
#
# Per run, <out>/<slug>-<YYYYmmdd-HHMMSS>/ holds: workflow/<session_id>/ (a copy of
# the workflow session the agent created), claude_session/ (<uuid>.jsonl and the
# <uuid>/ folder with subagents/ and tool-results/), claude_stream.jsonl (the CLI's
# stream-json output), claude_result.json (its final result object), claude_stderr.log,
# prompt.txt, agent_reply.md and metrics.json.  Per batch, <out>/_batches/<batch_id>/
# holds batch.log, summary.csv, summary.json and ar_runs.txt (the run folders,
# space-separated, for `AR_RUNS="$(cat ar_runs.txt)" illustrations/build.sh agenticresults`).
#
# The launch, permissions, isolation and scheduling live in ab_common.sh, shared with
# run_realizations.sh, run_material_sweep.sh and run_bc_sweep.sh.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ab_common.sh
. "$HERE/ab_common.sh"

LIST=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -*) if ab_common_opt "$@"; then shift "$AB_SHIFT"; else echo "unknown option: $1" >&2; usage >&2; exit 2; fi ;;
        *) LIST="$1"; shift ;;
    esac
done
[ -n "$LIST" ] && [ -f "$LIST" ] || { echo "list file not found: ${LIST:-<none>}" >&2; usage >&2; exit 2; }
LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
ab_setup

run_one() {   # $1 = index, $2 = slot
    local idx="$1" slot="$2" instr run
    instr="${ITEMS[$((idx - 1))]}"
    run="$(ab_new_run_dir "$instr")"
    "$PY" "$HELPER" prompt --template "$TEMPLATE" --worktree "$WORKTREE" --object "$instr" > "$run/prompt.txt"
    ab_launch "$idx" "$instr" "$slot" "$run"
}

mapfile -t ITEMS < <(grep -v '^[[:space:]]*#' "$LIST" | sed 's/[[:space:]]*$//' | grep -v '^[[:space:]]*$')
log "batch $BATCH_ID: ${#ITEMS[@]} instruction(s) from $LIST, $JOBS at a time, model $MODEL, effort $EFFORT, timeout ${TIMEOUT}s, memory $([ "$NO_MEMORY" = 1 ] && echo off || echo on), cache ttl $CACHE_TTL"
log "runs in $OUT; batch files in $BATCH_DIR"
if [ "$DRY" = 1 ]; then
    for i in "${!ITEMS[@]}"; do log "[dry] $((i + 1)) \"${ITEMS[$i]}\""; done
    exit 0
fi
ab_schedule run_one "${#ITEMS[@]}"
