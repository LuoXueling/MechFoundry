#!/usr/bin/env bash
# One short Codex session that checks, before a 50-object batch, everything a run needs from the
# provider and the harness, with the batch's own configuration (ab_common.sh, codex-home):
#
#   experiments/agentic_batch/codex_preflight.sh --model kimi-k3.1-pro [--provider hf] [--out DIR]
#
#   1. the provider answers: key, credits (HF: 402 "Purchase pre-paid credits"), the request shape
#      (HF: 400 "invalid_union" when a non-function tool slips in) and the pinned route;
#   2. the runner's own command form runs outside the sandbox: `drive.py gpu` lists the GPUs (it
#      cannot from inside the sandbox) -- the GPU check only, it starts no session and no worker;
#   3. the model sees images: view_image on a picture drawn here (left half one colour, right half
#      another, chosen at random) and names both colours;
#   4. apply_patch writes a file under the scratch folder.
# Prints one PASS/FAIL line per check and the tokens and priced cost; exit 0 only when all pass.
# Costs one session of a few thousand tokens (well under a cent on DeepSeek, a few cents on Kimi).
set -u -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ab_common.sh
. "$HERE/ab_common.sh"
AGENT=codex
OUT="$WORKTREE/output/codex_preflight"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        *) if ab_common_opt "$@"; then shift "$AB_SHIFT"; else echo "unknown option: $1" >&2; usage >&2; exit 2; fi ;;
    esac
done
JOBS=1 GPU_LIST="${GPU_LIST:-0}"
ab_setup
DIR="$BATCH_DIR/probe"
mkdir -p "$DIR"
HOME_DIR="$DIR/codex_home"

# The picture: two flat halves in two of six named colours, as a PNG written with the standard library.
read -r LEFT RIGHT < <("$PY" - "$DIR/halves.png" <<'EOF'
import random, struct, sys, zlib
names = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 210, 20),
         "black": (10, 10, 10), "white": (250, 250, 250)}
left, right = random.sample(sorted(names), 2)
w, h = 128, 64
row = b"".join(bytes(names[left] if x < w // 2 else names[right]) for x in range(w))
raw = b"".join(b"\x00" + row for _ in range(h))
chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
print(left, right)
EOF
)
REL="${DIR#"$WORKTREE"/}"
NOTE="$DIR/preflight_note.txt"
printf 'This is a short preflight check, not a simulation: do exactly the steps the prompt lists, nothing else.\n' > "$NOTE"
effort=(); [ "$EFFORT_GIVEN" = 1 ] && effort=(--effort "$EFFORT")
"$PY" "$HELPER" codex-home --dir "$HOME_DIR" --model "$MODEL" --provider "$PROVIDER" "${effort[@]}" --note "$NOTE" \
    --note "$HERE/codex_tools.txt" --note "$HERE/codex_note.txt" > "$DIR/codex_model.json"
PROMPT="Do these four steps in order, then reply with exactly four lines, one per step, in the form 'N: <answer>'.
1. Run this command exactly: python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu
   Answer with the number of GPUs it lists.
2. Look at the picture $REL/halves.png with the view_image tool. Answer with the colour of its left half and the colour of its right half, as two lowercase words (for example: green yellow).
3. With apply_patch, create the file $REL/written.txt containing the single line: preflight
   Answer: done.
4. Answer with the word: finished"
mapfile -t env_args < <(clean_env_args)
log "preflight: $CODEX_MODEL_ID, picture $LEFT/$RIGHT, folder $DIR"
( cd "$WORKTREE" && env "${env_args[@]}" FEM_OPERATOR_OUTPUT_ROOT="$REL/operator_root" CODEX_HOME="$HOME_DIR" \
    timeout 600 "$CODEX_BIN" exec --json -C "$WORKTREE" "$PROMPT" \
    < /dev/null > "$DIR/codex_stream.jsonl" 2> "$DIR/codex_stderr.log" )
rc=$?
"$PY" - "$DIR" "$LEFT" "$RIGHT" "$rc" "$HELPER" <<'EOF'
import glob, importlib.util, json, os, re, sys
d, left, right, rc, helper = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
spec = importlib.util.spec_from_file_location("ab_helper", helper)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
events = h.read_jsonl(os.path.join(d, "codex_stream.jsonl"))
failed = [e for e in events if e.get("type") == "turn.failed"]
usage = [e.get("usage") or {} for e in events if e.get("type") == "turn.completed"]
cmds = [e["item"] for e in events if e.get("type") == "item.completed"
        and (e.get("item") or {}).get("type") == "command_execution"]
reply = [e["item"].get("text") or "" for e in events if e.get("type") == "item.completed"
         and (e.get("item") or {}).get("type") == "agent_message"]
reply = reply[-1] if reply else ""
rollout = h.codex_rollout(os.path.join(d, "codex_home"), None)
tools = [(r.get("payload") or {}).get("name") for r in h.read_jsonl(rollout)] if rollout else []
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    ok or fails.append(what)
err = (failed[0].get("error") or {}).get("message", "") if failed else ""
check(rc == 0 and usage and not failed, "1 the provider answered" + (f": {err[:200]}" if err else f" (exit {rc})" if rc else ""))
gpu = [c for c in cmds if "drive.py gpu" in (c.get("command") or "")]
check(bool(gpu) and gpu[-1].get("exit_code") == 0 and '"gpus"' in (gpu[-1].get("aggregated_output") or ""),
      "2 drive.py ran outside the sandbox and listed the GPUs")
# The answer line by its content, not its number (one model numbered its lines from 2).
def names_both(line):
    words = re.findall(r"[a-z]+", line.lower())
    return left in words and right in words and words.index(left) < words.index(right)
line2 = next((l for l in reply.splitlines() if names_both(l)), "")
check("view_image" in tools and bool(line2),
      f"2b the model saw the picture ({left} {right}): {(line2 or reply).strip()[:120]!r}")
written = os.path.join(d, "written.txt")
check(os.path.exists(written) and open(written).read().strip() == "preflight", "3 apply_patch wrote the file")
tok = {}
for u in usage:
    for k, v in u.items():
        tok[k] = tok.get(k, 0) + int(v or 0)
launched = json.load(open(os.path.join(d, "codex_model.json")))
mapped = {"input_tokens": tok.get("input_tokens", 0) - tok.get("cached_input_tokens", 0) - tok.get("cache_write_input_tokens", 0),
          "cache_read_input_tokens": tok.get("cached_input_tokens", 0),
          "cache_creation_input_tokens": tok.get("cache_write_input_tokens", 0), "output_tokens": tok.get("output_tokens", 0)}
print(f"tokens {tok}  priced {h.codex_cost(mapped, launched.get('price'))} USD  route {launched.get('route')}")
print("ALL PASS" if not fails else f"{len(fails)} FAILED (see {d})")
sys.exit(1 if fails else 0)
EOF
