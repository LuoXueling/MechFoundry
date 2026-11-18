#!/usr/bin/env bash
# Run (or resume) Astra evaluator k of a round through the Codex CLI, isolated.
#
#   bash run_evaluator.sh [--round NAME] K start            # prompt_K.md of the round on stdin
#   bash run_evaluator.sh [--round NAME] K resume "TEXT"    # continue the same Codex session
#   bash run_evaluator.sh [--round NAME] K smoke "TEXT"     # TEXT instead of the prompt; output
#                                                           # and logs under <logs>/smoke/
#
# Every path (packet, output dir, logs, private CODEX_HOME, prompt, folders to hide) comes from
# the round config through round_config.py runner-env; the round defaults to the paper round,
# whose Astra run (2026-10-19) this script made.  That round cannot be re-run as it was (the repo
# and its work dir have moved), and runner-env says so.  SMOKE_PACKET=DIR runs a smoke test on a
# stand-in packet before the round's packets exist.
#
# Model, effort and flags are the paper round's: gpt-6-astra, model_reasoning_effort high,
# approvals never, sub-agents and memories off, workspace-write sandbox, --json transcript.
#
# Isolation, two layers:
#  1. The bubblewrap namespace of ../isolate.sh: everything read-only; $HOME (~/.claude,
#     ~/.codex with the other evaluators' sessions, other checkouts), the repo, /tmp and the
#     round's extra folders empty; the packet's content read-only in a throwaway cwd; the output
#     dir read-write.  Bound back in here: a private CODEX_HOME (read-write) with the real
#     ~/.codex/auth.json bound into it so the ChatGPT login is shared.
#  2. Codex's own workspace-write sandbox (cwd = the packet, writable root = the output dir),
#     no network for model-run commands, approvals never asked (a denied write just fails).
#  Net effect: the only place the evaluator can write that survives the run is its output dir.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROUND=paper
if [ "${1:-}" = --round ]; then ROUND=$2; shift 2; fi
K=$1; MODE=$2; TEXT=${3:-}
SMOKE=(); [ "$MODE" = smoke ] && SMOKE=(--smoke)
eval "$(python3 "$HERE/../round_config.py" runner-env --round "$ROUND" --family astra --k "$K" "${SMOKE[@]}")"
[ "$MODE" = smoke ] && [ -n "${SMOKE_PACKET:-}" ] && PACKET=$SMOKE_PACKET
[ -f "$PACKET/index.json" ] || { echo "no packet at $PACKET (build_packets.py --round $ROUND)" >&2; exit 1; }
mkdir -p "$OUT" "$LOGS" "$CLI_HOME"
touch "$CLI_HOME/auth.json"                    # the mount point of the shared login
STAMP=$(date +%Y%m%d-%H%M%S)

source "$HERE/../isolate.sh"
BWRAP+=(--bind "$HOME/.codex/auth.json" "$CLI_HOME/auth.json" --setenv CODEX_HOME "$CLI_HOME")
COMMON=(-m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"" -c 'approval_policy="never"'
        --disable multi_agent --disable memories)

if [ "$MODE" = start ] || [ "$MODE" = smoke ]; then
  CMD=(codex exec "${COMMON[@]}" -s workspace-write -C "$PACKET" --add-dir "$OUT"
       --skip-git-repo-check --json -o "$OUT/last_message_$STAMP.txt" -)
  if [ "$MODE" = start ]; then
    echo "$STAMP start: ${BWRAP[*]} ${CMD[*]} < $PROMPT" >> "$LOGS/commands_$K.txt"
    "${BWRAP[@]}" "${CMD[@]}" < "$PROMPT" \
      > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
  else
    echo "$STAMP smoke: ${BWRAP[*]} ${CMD[*]} <<< \"$TEXT\"" >> "$LOGS/commands_$K.txt"
    "${BWRAP[@]}" "${CMD[@]}" <<< "$TEXT" \
      > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
  fi
else
  TID=$(head -1 "$(ls "$LOGS"/eval_${K}_*.jsonl | head -1)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["thread_id"])')
  CMD=(codex exec resume "${COMMON[@]}" -c 'sandbox_mode="workspace-write"'
       -c "sandbox_workspace_write.writable_roots=[\"$OUT\"]"
       --skip-git-repo-check --json -o "$OUT/last_message_$STAMP.txt" "$TID" -)
  echo "$STAMP resume: ${BWRAP[*]} ${CMD[*]} <<< \"$TEXT\"" >> "$LOGS/commands_$K.txt"
  "${BWRAP[@]}" "${CMD[@]}" <<< "$TEXT" \
    > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
fi
