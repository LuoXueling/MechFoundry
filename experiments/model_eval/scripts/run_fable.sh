#!/usr/bin/env bash
# Run (or resume) Fable evaluator k in headless Claude Code, isolated.
#
#   bash scripts/run_fable.sh K start            # results/fable/prompt_K.md on stdin
#   bash scripts/run_fable.sh K resume "TEXT"    # continue the same Claude session
#   bash scripts/run_fable.sh K smoke "TEXT"     # TEXT instead of the prompt; output and logs
#                                                # under <logs>/smoke/
#
# Every path comes from ../grading.json (config.py runner-env, family "fable").
# SMOKE_PACKET=DIR runs a smoke test on a stand-in packet before the packets exist.
#
# What the paper round's Fable evaluators were, and what this keeps (their transcripts are in the
# coordinator session 56b280a0-... of ~/.claude/projects/-home-xl2254-implicit-optimize, the
# three subagents whose first message is "You are evaluator k"):
#  - model claude-fable-5-1, effort high (every assistant turn of the three records
#    "effort": "high"): here --model <the config's model> --effort high.  The model is Claude
#    Code's `fable` alias, not pinned (user, 2026-11-19), so it is the current Fable
#    (claude-fable-5-2 on that day), not the paper's 5.1; the init event records which;
#  - one agent each (general-purpose subagents; none spawned another): no Agent tool here;
#  - the tools they used: Read (the prompt, index.json, the 30 figures, scratch text), Bash
#    (python3 scripts reading five replies at a time, cat, ls, grep, sed, a for loop), Write and
#    Edit (their scratch files and eval_k.json).  So: --tools Read,Write,Edit,Glob,Grep,Bash.
#    They got the prompt as "read prompt_k.md with the Read tool, then follow it"; here, as in
#    the Astra round, the prompt text itself is the first message (the prompt file is in the
#    hidden repo).
#
# Isolation, two layers:
#  1. The bubblewrap namespace of isolate.sh: everything read-only; $HOME (~/.claude with every
#     Claude transcript and the auto-memory, ~/.claude.json, ~/.codex, other checkouts), the
#     repo and /tmp empty; the packet's content read-only in a
#     throwaway cwd; the output dir read-write.  Bound back in here: a private CLAUDE_CONFIG_DIR
#     (read-write; this evaluator's own session, settings and caches) with the real
#     ~/.claude/.credentials.json bound into it, which is where this CLI keeps the claude.ai
#     OAuth login (no keychain on Linux), so the login is shared and nothing else of ~/.claude is.
#  2. Claude Code's permissions: --permission-mode dontAsk refuses every tool call no rule allows,
#     without asking.  Allowed: Read/Glob/Grep under the packet and the output dir, Write/Edit
#     under the output dir only, and Bash for the read-and-script commands the Fable evaluators
#     used (python3, cat, ls, grep, sed, ...).  A Bash command reaches whatever layer 1 shows it,
#     so layer 1 is what bounds python3; Claude Code's own Bash sandbox would be the analogue of
#     Codex's, but it needs socat, which this machine lacks.  Also: --strict-mcp-config with an
#     empty server list (no claude.ai connectors or MCP servers), auto-memory off (setting and
#     CLAUDE_CODE_DISABLE_AUTO_MEMORY), no slash commands or skills, no auto-updater.
#  Net effect, as for Astra: the only place the evaluator can write that survives is its output
#  dir (and its own private config dir).
#
# The login: the CLI refreshes the OAuth token shortly before it expires and then rewrites the
# credentials file.  Through a bind mount a rewrite by rename would fail and leave the real file
# with a refresh token that may already be spent, so the runner refuses to start with less than
# MIN_TOKEN_HOURS (default 3) of token left: run any `claude -p` outside first to refresh it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { echo "usage: run_fable.sh K start|resume|smoke [TEXT]" >&2; exit 2; }
K=$1; MODE=$2; TEXT=${3:-}
SMOKE=(); [ "$MODE" = smoke ] && SMOKE=(--smoke)
# A plain assignment, so that set -e stops here when config.py refuses (eval "$(...)" would not)
ENV=$(python3 "$HERE/config.py" runner-env --family fable --k "$K" "${SMOKE[@]}")
eval "$ENV"
[ "$MODE" = smoke ] && [ -n "${SMOKE_PACKET:-}" ] && PACKET=$SMOKE_PACKET
[ -f "$PACKET/index.json" ] || { echo "no packet at $PACKET (scripts/build_packets.py)" >&2; exit 1; }
CREDS=$HOME/.claude/.credentials.json
python3 - "$CREDS" "${MIN_TOKEN_HOURS:-3}" <<'EOF'
import json, sys, time
left = (json.load(open(sys.argv[1]))["claudeAiOauth"]["expiresAt"] / 1000 - time.time()) / 3600
if left < float(sys.argv[2]):
    sys.exit(f"the Claude login token expires in {left:.1f} h (< {sys.argv[2]} h): run any "
             "`claude -p` outside the sandbox to refresh it, then start again")
print(f"login token valid for {left:.1f} h")
EOF
mkdir -p "$OUT" "$LOGS" "$CLI_HOME"
touch "$CLI_HOME/.credentials.json"            # the mount point of the shared login
STAMP=$(date +%Y%m%d-%H%M%S)

source "$HERE/isolate.sh"
BWRAP+=(--bind "$CREDS" "$CLI_HOME/.credentials.json" --setenv CLAUDE_CONFIG_DIR "$CLI_HOME"
        --setenv CLAUDE_CODE_DISABLE_AUTO_MEMORY 1 --setenv DISABLE_AUTOUPDATER 1)
ALLOW=("Read(/$PACKET/**)" "Read(/$OUT/**)" "Glob(/$PACKET/**)" "Grep(/$PACKET/**)"
       "Write(/$OUT/**)" "Edit(/$OUT/**)")
for c in python3 python cat ls head tail wc grep sed awk sort uniq find jq echo cd; do
  ALLOW+=("Bash($c:*)")
done
SETTINGS='{"autoMemoryEnabled": false, "includeCoAuthoredBy": false,
           "permissions": {"deny": ["Agent", "Task", "WebFetch", "WebSearch"]}}'
COMMON=(-p --model "$MODEL" --effort "$EFFORT" --output-format stream-json --verbose
        --permission-mode dontAsk --tools "Read,Write,Edit,Glob,Grep,Bash"
        --allowedTools "${ALLOW[@]}" --add-dir "$OUT"
        --strict-mcp-config --mcp-config '{"mcpServers": {}}' --settings "$SETTINGS"
        --disable-slash-commands)

case "$MODE" in
  start|smoke)
    SID=$(python3 -c 'import uuid; print(uuid.uuid4())')
    echo "$SID" > "$LOGS/session_$K.txt"
    CMD=(claude "${COMMON[@]}" --session-id "$SID")
    if [ "$MODE" = start ]; then
      echo "$STAMP start: ${BWRAP[*]} ${CMD[*]} < $PROMPT" >> "$LOGS/commands_$K.txt"
      "${BWRAP[@]}" "${CMD[@]}" < "$PROMPT" \
        > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
    else
      echo "$STAMP smoke: ${BWRAP[*]} ${CMD[*]} <<< \"$TEXT\"" >> "$LOGS/commands_$K.txt"
      "${BWRAP[@]}" "${CMD[@]}" <<< "$TEXT" \
        > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr"
    fi ;;
  resume)
    SID=$(cat "$LOGS/session_$K.txt")
    CMD=(claude "${COMMON[@]}" --resume "$SID")
    echo "$STAMP resume: ${BWRAP[*]} ${CMD[*]} <<< \"$TEXT\"" >> "$LOGS/commands_$K.txt"
    "${BWRAP[@]}" "${CMD[@]}" <<< "$TEXT" \
      > "$LOGS/eval_${K}_$STAMP.jsonl" 2> "$LOGS/eval_${K}_$STAMP.stderr" ;;
  *) echo "mode: start, resume or smoke" >&2; exit 2 ;;
esac
