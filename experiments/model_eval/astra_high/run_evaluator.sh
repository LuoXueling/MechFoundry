#!/usr/bin/env bash
# Run (or resume) Astra evaluator k through the Codex CLI, isolated.
#
#   bash run_evaluator.sh K start            # first run: prompt_K.md on stdin
#   bash run_evaluator.sh K resume "TEXT"    # continue the same Codex session with TEXT
#
# Isolation, two layers:
#  1. An outer bubblewrap namespace in which the whole filesystem is read-only and the places that
#     hold other ratings are empty tmpfs: the repo (Fable's eval_*.json, the label mappings, the
#     Fable packets), ~/.claude (Claude transcripts), the work dir with the other packets, /tmp,
#     and ~/.codex (other evaluators' Codex sessions).  Bound back in: this evaluator's packet,
#     its output dir (read-write), a private CODEX_HOME (read-write; the real auth.json is bound
#     into it so the login is shared) and ~/.codex/packages (read-only: the codex binary lives
#     there).  The packet dir itself is a fresh tmpfs with index.json, figures/ and runs/ bound
#     into it read-only: Codex's own sandbox (layer 2) must be able to create its protected
#     mount points (.git, .agents, .aws, ...) in the cwd, which a read-only bind of the whole
#     packet refuses (and bubblewrap 0.6 has no tmp-overlay).  So the packet's content is
#     read-only and anything written beside it vanishes with the namespace.
#  2. Codex's own workspace-write sandbox (cwd = the packet, writable root = the output dir),
#     no network for model-run commands, approvals never asked (a denied write just fails).
#  Net effect: the only place the evaluator can write that survives the run is its output dir.
set -euo pipefail
K=$1; MODE=$2; TEXT=${3:-}
REPO=/home/xl2254/implicit_optimize
WORK=/home/xl2254/astra_eval_work
HERE=$(cd "$(dirname "$0")" && pwd)
PACKET=$WORK/packet_$K
OUT=$REPO/output/agentic_object_record/model_eval_astra/evaluator_$K
LOGS=$REPO/output/agentic_object_record/model_eval_astra/logs
CHOME=$WORK/codex_home_$K
MODEL=gpt-6-astra
EFFORT=high
mkdir -p "$OUT" "$LOGS" "$CHOME"
touch "$CHOME/auth.json"
STAMP=$(date +%Y%m%d-%H%M%S)

BWRAP=(bwrap --ro-bind / / --dev-bind /dev /dev --bind /proc /proc
       --tmpfs "$REPO" --tmpfs /home/xl2254/.claude --tmpfs "$WORK" --tmpfs /tmp
       --tmpfs /home/xl2254/.codex
       --ro-bind /home/xl2254/.codex/packages /home/xl2254/.codex/packages
       --tmpfs "$PACKET"
       --ro-bind "$PACKET/index.json" "$PACKET/index.json"
       --ro-bind "$PACKET/figures" "$PACKET/figures"
       --ro-bind "$PACKET/runs" "$PACKET/runs"
       --bind "$OUT" "$OUT"
       --bind "$CHOME" "$CHOME"
       --bind /home/xl2254/.codex/auth.json "$CHOME/auth.json"
       --setenv CODEX_HOME "$CHOME"
       --chdir "$PACKET" --die-with-parent)
COMMON=(-m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"" -c 'approval_policy="never"'
        --disable multi_agent --disable memories)

if [ "$MODE" = start ]; then
  CMD=(codex exec "${COMMON[@]}" -s workspace-write -C "$PACKET" --add-dir "$OUT"
       --skip-git-repo-check --json -o "$OUT/last_message_$STAMP.txt" -)
  echo "$STAMP start: ${BWRAP[*]} ${CMD[*]} < prompt_$K.md" >> "$LOGS/commands_$K.txt"
  "${BWRAP[@]}" "${CMD[@]}" < "$HERE/prompt_$K.md" \
    > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
else
  TID=$(head -1 "$(ls "$LOGS"/eval_${K}_*.jsonl | head -1)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["thread_id"])')
  CMD=(codex exec resume "${COMMON[@]}" -c 'sandbox_mode="workspace-write"'
       -c "sandbox_workspace_write.writable_roots=[\"$OUT\"]"
       --skip-git-repo-check --json -o "$OUT/last_message_$STAMP.txt" "$TID" -)
  echo "$STAMP resume: ${BWRAP[*]} ${CMD[*]} <<< \"$TEXT\"" >> "$LOGS/commands_$K.txt"
  "${BWRAP[@]}" "${CMD[@]}" <<< "$TEXT" \
    > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
fi
