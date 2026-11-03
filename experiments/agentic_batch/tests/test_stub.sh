#!/usr/bin/env bash
# Stub test of run_batch.sh: no API calls.  Everything goes to a temp dir
# (runs, fake sessions, fake transcripts); prints the checker's verdict.
#   bash experiments/agentic_batch/tests/test_stub.sh [JOBS]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d /tmp/agentic_batch_stub.XXXXXX)"
export AB_WORKFLOW_ROOT="$TMP/fem_workflow_operator" AB_PROJECTS_DIR="$TMP/projects"
export STUB_LOG="$TMP/stub.log" CLAUDE_BIN="$HERE/stub_claude.py"
# Pretend to be inside a Claude Code session: the script must not pass these on.
export CLAUDE_CODE_SESSION_ID=parent-session CLAUDECODE=1
mkdir -p "$AB_WORKFLOW_ROOT" "$AB_PROJECTS_DIR"
cat > "$TMP/objects.txt" <<'EOF'
# comment line, skipped

a wooden clog sleep3
a lectern sleep1 fail
a box culvert sleep2 nostart
a corner casting sleep1 crash
a porcelain insulator sleep4 twostart
a hung beam hang
a wooden clog sleep1
EOF
bash "$HERE/../run_batch.sh" -j "${1:-2}" --timeout 8 --out "$TMP/out" "$TMP/objects.txt"
echo "run_batch.sh exit: $?"
python3 "$HERE/check_stub.py" "$TMP" "${1:-2}"
echo "temp dir: $TMP"
