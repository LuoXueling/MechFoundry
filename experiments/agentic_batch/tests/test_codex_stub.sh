#!/usr/bin/env bash
# Stub test of run_batch.sh --agent codex: no API calls.  Everything goes to a temp dir (runs, fake
# sessions, the runs' CODEX_HOMEs); prints the checker's verdict.  Needs python3 >= 3.11 (tomllib).
#   bash experiments/agentic_batch/tests/test_codex_stub.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d /tmp/agentic_batch_codex_stub.XXXXXX)"
export AB_WORKFLOW_ROOT="$TMP/fem_workflow_operator" STUB_LOG="$TMP/stub.log" CODEX_BIN="$HERE/stub_codex.py"
# A provider key the runs must pass to codex and never write down.
export HF_TOKEN="hf_stubsecret_$$" CODEX_THREAD_ID=parent-thread
mkdir -p "$AB_WORKFLOW_ROOT"
cat > "$TMP/objects.txt" <<'EOF'
a wooden clog sleep2
a lectern sleep1 fail
a box culvert sleep1 nostart
a corner casting sleep1 crash
a hung beam hang
EOF
printf 'a park bench sleep1\n' > "$TMP/one.txt"
STUB_EXPECT_MODEL=moonshotai/Kimi-K3.1-Pro:moonshotai bash "$HERE/../run_batch.sh" --agent codex --model kimi-k3.1-pro \
    -j 2 --gpus 0 --timeout 8 --out "$TMP/out" "$TMP/objects.txt"
echo "run_batch.sh (kimi) exit: $?"
# DeepSeek through its own API: no route, the model's same accounting price, the key from the environment.
DEEPSEEK_API_KEY="sk-stubsecret_$$" STUB_EXPECT_MODEL=deepseek-flash bash "$HERE/../run_batch.sh" --agent codex \
    --model deepseek-v4.1-flash --provider vendor -j 1 --gpus 0 --timeout 8 --out "$TMP/out_vendor" "$TMP/one.txt"
echo "run_batch.sh (deepseek vendor) exit: $?"
# No key, no batch.
env -u HF_TOKEN HF_TOKEN_PATH=/nonexistent bash "$HERE/../run_batch.sh" --agent codex --model deepseek-v4.1-flash \
    --out "$TMP/out_nokey" "$TMP/one.txt" 2> "$TMP/nokey.err"
echo "$?" > "$TMP/nokey.rc"
python3 "$HERE/check_codex_stub.py" "$TMP"
echo "temp dir: $TMP"
