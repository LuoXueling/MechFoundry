# The sweep shared by run_material_sweep.sh (SWEEP_KIND=material) and run_bc_sweep.sh
# (SWEEP_KIND=bc): from one finished session, N runs that each vary one stage and reuse the rest.
# Sourced after the caller sets HERE and SWEEP_KIND; never run.
#
# 1. Authoring: one headless Claude call (the run model, no tools but Read on two pictures of the
#    source: its views and its segments) writes the N variants at once, so they differ from one
#    another: ab_helper.py author-prompt / variants, kept in _batches/<id>/authoring/.  The
#    variants are checked (every segment a material and a real contrast, or supports and loads
#    present); a failed check asks once more with the problems listed.  --variants FILE skips it.
# 2. Each run: `drive.py clone SOURCE --fresh-solve` makes a new session (its own id, in the run's
#    slot) that is a copy of the source waiting after its solve: body, segments, regions, values and
#    material exactly as solved, its checkpoints re-keyed; the source is only read.  The agent is
#    told the variant (prompt_material_sweep.txt / prompt_bc_sweep.txt) and edits only that stage
#    of the plan; the runner rewinds to that stage's card by itself and solves.

N=2 VARIANTS_FILE="" SOURCE="" AUTHOR_MODEL=""
case "$SWEEP_KIND" in
    material) TEMPLATE="$HERE/prompt_material_sweep.txt" ;;
    bc) TEMPLATE="$HERE/prompt_bc_sweep.txt" ;;
    *) echo "ab_sweep.sh: SWEEP_KIND must be material or bc" >&2; exit 2 ;;
esac
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -n|--runs) N="$2"; shift 2 ;;
        --variants) VARIANTS_FILE="$2"; shift 2 ;;
        --author-model) AUTHOR_MODEL="$2"; shift 2 ;;
        -*) if ab_common_opt "$@"; then shift "$AB_SHIFT"; else echo "unknown option: $1" >&2; usage >&2; exit 2; fi ;;
        *) SOURCE="$1"; shift ;;
    esac
done
[ "$N" -ge 1 ] 2>/dev/null || { echo "bad -n: $N" >&2; exit 2; }
# The source: a session folder, or a run folder of run_batch.sh (its workflow/<session_id>/).
if [ -n "$SOURCE" ] && [ -f "$SOURCE/metrics.json" ] && [ ! -f "$SOURCE/checkpoints.sqlite" ]; then
    SOURCE="$SOURCE/workflow/$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["session_id"])' "$SOURCE/metrics.json")"
fi
[ -n "$SOURCE" ] && [ -f "$SOURCE/checkpoints.sqlite" ] && [ -f "$SOURCE/case/case.json" ] \
    || { echo "not a solved session folder: ${SOURCE:-<none>}" >&2; usage >&2; exit 2; }
SOURCE="$(cd "$SOURCE" && pwd)"
[ -z "$VARIANTS_FILE" ] || [ -f "$VARIANTS_FILE" ] || { echo "no such file: $VARIANTS_FILE" >&2; exit 2; }
AUTHOR_MODEL="${AUTHOR_MODEL:-$MODEL}"
ab_setup
OBJECT="$("$PY" "$HELPER" source-info --session "$SOURCE" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["instruction"] or "")')"
VARIANTS="$BATCH_DIR/variants.json"

author_variants() {
    local dir="$BATCH_DIR/authoring" uuid img rc problems try
    mkdir -p "$dir"
    local -a images=() allow=() env_args
    for img in "$SOURCE/operator/texture_views.png" "$SOURCE/geometry/views.png" "$SOURCE/operator/segments.png"; do
        [ -f "$img" ] && images+=("$img")
    done
    [ "${#images[@]}" -gt 2 ] && images=("${images[0]}" "${images[2]}")   # the painted views and the segments
    for img in "${images[@]}"; do allow+=("Read(/$img)"); done
    "$PY" "$HELPER" author-prompt --kind "$SWEEP_KIND" --session "$SOURCE" --n "$N" \
        $(for img in "${images[@]}"; do printf -- '--image %s ' "$img"; done) > "$dir/prompt.txt"
    mapfile -t env_args < <(clean_env_args)
    for try in 1 2; do
        uuid="$("$PY" -c 'import uuid; print(uuid.uuid4())')"
        log "authoring $N $SWEEP_KIND variant(s) with $AUTHOR_MODEL (try $try, claude=$uuid)"
        ( cd "$WORKTREE" && env "${env_args[@]}" timeout --signal=TERM --kill-after=30 900 \
            "$CLAUDE_BIN" -p "$(cat "$dir/prompt.txt")" --model "$AUTHOR_MODEL" --effort "$EFFORT" \
                --session-id "$uuid" --output-format json --permission-mode dontAsk \
                --tools Read --allowedTools "${allow[@]:-Read(/nonexistent)}" --strict-mcp-config \
                --disable-slash-commands --settings "{\"promptCacheTtl\": \"$CACHE_TTL\", \"autoMemoryEnabled\": false}" \
                < /dev/null > "$dir/claude_result_$try.json" 2> "$dir/claude_stderr_$try.log" )
        rc=$?
        problems="$("$PY" "$HELPER" variants --kind "$SWEEP_KIND" --session "$SOURCE" --n "$N" \
            --result "$dir/claude_result_$try.json" --out "$VARIANTS" 2>&1)" && { log "authoring: ok"; return 0; }
        log "authoring try $try (rc=$rc): $problems"
        printf '\n\nYour previous reply was refused: %s. Write all %s again, fixing that.\n' "$problems" "$N" >> "$dir/prompt.txt"
    done
    return 1
}

run_one() {   # $1 = index, $2 = slot
    local idx="$1" slot="$2" run label description out sid instr
    label="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["variants"][int(sys.argv[2]) - 1].get("label") or "")' "$VARIANTS" "$idx")"
    description="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["variants"][int(sys.argv[2]) - 1].get("description") or "")' "$VARIANTS" "$idx")"
    instr="$OBJECT | $SWEEP_KIND: $label"
    run="$(ab_new_run_dir "$OBJECT $SWEEP_KIND $label")"
    # The copy's stored instruction is the variant, words and sizes: the runner's checks take a size
    # as "given" only when the stored instruction states it (the first real BC sweep's 60 N pull and
    # 6 N m twist were refused as given while the copy kept the source's "a sewing machine").
    out="$(ab_drive "$slot" clone "$SOURCE" --fresh-solve --note "$SWEEP_KIND variant $idx: $label" \
        --instruction "$instr. $description" 2> "$run/clone_stderr.log")"
    printf '%s\n' "$out" > "$run/clone.json"
    sid="$(printf '%s' "$out" | "$PY" -c 'import json, sys
try:
    v = json.loads(sys.stdin.read())
    print(v["session_id"] if v.get("ok") else "")
except Exception:
    print("")')"
    if [ -z "$sid" ]; then
        log "[$idx] clone of $SOURCE failed: $(tail -c 400 "$run/clone_stderr.log") $out"
        printf '{"run_name": "%s", "status": "clone_failed", "instruction": %s}\n' "$(basename "$run")" \
            "$("$PY" -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$instr")" > "$run/metrics.json"
        echo "$run" > "$BATCH_DIR/.done_$(printf %04d "$idx")"
        return
    fi
    "$PY" "$HELPER" variant-prompt --template "$TEMPLATE" --worktree "$WORKTREE" --variants "$VARIANTS" \
        --index "$idx" --session-id "$sid" --variant-out "$run/variant.json" > "$run/prompt.txt"
    ab_launch "$idx" "$instr" "$slot" "$run"
}

log "batch $BATCH_ID: $N $SWEEP_KIND variant(s) of session $SOURCE (\"$OBJECT\"), $JOBS at a time, model $MODEL, effort $EFFORT, timeout ${TIMEOUT}s, memory $([ "$NO_MEMORY" = 1 ] && echo off || echo on), cache ttl $CACHE_TTL"
log "runs in $OUT; batch files in $BATCH_DIR"
if [ "$DRY" = 1 ]; then
    log "[dry] the variants would be $([ -n "$VARIANTS_FILE" ] && echo "read from $VARIANTS_FILE" || echo "authored by $AUTHOR_MODEL")"
    for ((i = 1; i <= N; i++)); do log "[dry] $i: clone $SOURCE, $SWEEP_KIND variant $i"; done
    exit 0
fi
if [ -n "$VARIANTS_FILE" ]; then
    "$PY" "$HELPER" variants --kind "$SWEEP_KIND" --session "$SOURCE" --n "$N" --given "$VARIANTS_FILE" \
        --out "$VARIANTS" || { log "the given variants are refused"; exit 1; }
else
    author_variants || { log "no usable variants; nothing run"; exit 1; }
fi
ab_schedule run_one "$N"
