# Shared by run_batch.sh, run_realizations.sh, run_material_sweep.sh and run_bc_sweep.sh: the
# options every one of them takes, the per-run Claude Code launch (permissions, isolation, the
# record ab_helper.py finalizes), and the slot scheduler.  Sourced, never run.
#
# The caller sets HERE (its own folder) before sourcing; this file sets WORKTREE (the repository
# the scripts sit in: they run against the checkout they belong to), HELPER, PY and CLAUDE_BIN.
#
# Common options (ab_common_opt):
#   -j, --jobs N        runs at once (default: the number of GPUs, `nvidia-smi -L`)
#   --model M           Claude model (default: sonnet)
#   --timeout SEC       wall-clock limit per run, then the run is killed (default: 7200)
#   --out DIR           where run folders go (default: <worktree>/output/agentic_batch)
#   --template FILE     the initial instruction ({WORKTREE}, {OBJECT} and the script's own fields)
#   --effort LEVEL      thinking effort, low|medium|high|xhigh|max (default: medium)
#   --no-path-note      do not append the one-line repository-root note to the system prompt
#   --memory            leave auto-memory on (default: off)
#   --cache-ttl 5m|1h   prompt-cache TTL (default: 5m)
#   --gpus LIST         comma-separated GPU indices the slots take in turn (default: all)
#   --dry-run           print what would run, start nothing

[ -n "${HERE:-}" ] || { echo "ab_common.sh: HERE is not set" >&2; return 2 2>/dev/null || exit 2; }
WORKTREE="$(cd "$HERE/../.." && pwd)"
HELPER="$HERE/ab_helper.py"
PY="${PYTHON:-python3}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
DRIVE_REL="src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py"

JOBS="" GPU_LIST="" MODEL="sonnet" TIMEOUT=7200 OUT="$WORKTREE/output/agentic_batch"
TEMPLATE="${TEMPLATE:-$HERE/prompt_template.txt}" NO_MEMORY=1 CACHE_TTL=5m DRY=0 EFFORT="medium" PATH_NOTE=1

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# One common option at "$1" (and its value at "$2"): sets AB_SHIFT to the words it took, or
# returns 1 when "$1" is not one of them.
ab_common_opt() {
    AB_SHIFT=2
    case "$1" in
        -j|--jobs) JOBS="$2" ;;
        --gpus) GPU_LIST="$2" ;;
        --model) MODEL="$2" ;;
        --timeout) TIMEOUT="$2" ;;
        --out) OUT="$2" ;;
        --template) TEMPLATE="$2" ;;
        --effort) EFFORT="$2" ;;
        --cache-ttl) CACHE_TTL="$2" ;;
        --no-memory) NO_MEMORY=1; AB_SHIFT=1 ;;   # the default; kept so older commands still run
        --memory) NO_MEMORY=0; AB_SHIFT=1 ;;
        --no-path-note) PATH_NOTE=0; AB_SHIFT=1 ;;
        --dry-run) DRY=1; AB_SHIFT=1 ;;
        *) return 1 ;;
    esac
}

# After the options: check them and set up the batch folder, its log and the GPU list.
ab_setup() {
    if [ -z "$JOBS" ]; then
        JOBS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
        [ "${JOBS:-0}" -ge 1 ] 2>/dev/null || JOBS=1
    fi
    case "$EFFORT" in low|medium|high|xhigh|max) ;; *) echo "bad --effort: $EFFORT" >&2; exit 2 ;; esac
    case "$CACHE_TTL" in 5m|1h) ;; *) echo "bad --cache-ttl: $CACHE_TTL" >&2; exit 2 ;; esac
    command -v "$CLAUDE_BIN" >/dev/null || { echo "claude CLI not found: $CLAUDE_BIN" >&2; exit 2; }
    # Both overridable only so the stub test can keep its fakes out of the real folders.
    WORKFLOW_ROOT="${AB_WORKFLOW_ROOT:-$WORKTREE/output/fem_workflow_operator}"
    PROJECTS_DIR="${AB_PROJECTS_DIR:-$HOME/.claude/projects}"
    # Claude Code keeps a session's transcript under projects/<cwd with every
    # non-alphanumeric character turned into '-'>/<session uuid>.jsonl.
    ENCODED="$(printf '%s' "$WORKTREE" | sed 's/[^A-Za-z0-9]/-/g')"
    BATCH_ID="$(date +%Y%m%d-%H%M%S)"
    BATCH_DIR="$OUT/_batches/$BATCH_ID"
    mkdir -p "$BATCH_DIR"
    LOG="$BATCH_DIR/batch.log"
    if [ -n "$GPU_LIST" ]; then
        IFS=, read -r -a GPUS <<< "$GPU_LIST"
    else
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
    fi
    NGPU="${#GPUS[@]}"
}

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG" >&2; }

# The tools a run may use, as tightly as the skill allows: the runner's one
# command form (SKILL.md: every command is `python3 -u .../drive.py ...`, which
# hands itself to mechfoundry-agents' interpreter, typed literally, no pipes or
# redirects; inspect_session.py's summary/render/report too, since the references a stop's
# `read` names cite them), reading the skill folder and the operator sessions, writing and
# editing only <session>/operator/ files (the brief, the plan, the judgement), and
# reading this session's own persisted large tool outputs.  Everything else is
# refused without a prompt (--permission-mode dontAsk).  (`conda --version` was
# allowed until the skill stopped asking for it: drive.py says itself when neither
# mechfoundry-agents' interpreter nor conda is found.)
allowed_tools() {   # $1 = the Claude session uuid
    local own="//${PROJECTS_DIR#/}/$ENCODED/$1"
    printf '%s\n' \
        "Bash(python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py *)" \
        "Bash(python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py *)" \
        "Read(./src/agents_wrapper/fem_workflow/operator_skill/**)" \
        "Read(./output/fem_workflow_operator/**)" \
        "Read($own/**)" \
        "Glob(./src/agents_wrapper/fem_workflow/operator_skill/**)" \
        "Glob(./output/fem_workflow_operator/**)" \
        "Grep(./src/agents_wrapper/fem_workflow/operator_skill/**)" \
        "Grep(./output/fem_workflow_operator/**)" \
        "Write(./output/fem_workflow_operator/*/operator/**)" \
        "Edit(./output/fem_workflow_operator/*/operator/**)" \
        "Write(./output/fem_workflow_operator/_slots/*/*/operator/**)" \
        "Edit(./output/fem_workflow_operator/_slots/*/*/operator/**)"
}

# What the allowlist alone does not stop (measured with tests/probe_template.txt):
# Claude Code lets Read, Glob and Grep read anything under the working directory,
# and read-only shell commands (ls, cat, head, a pipe into head ...) run, without
# any rule.  So the skill's "stay inside two folders" (SKILL.md rule 3) is also
# written as deny rules, which win over allow rules: every repository entry beside
# the path to the skill folder and to output/fem_workflow_operator, and the shell's
# file readers.  The list is built from the tree as it is when the batch starts.
denied_tools() {
    local base keep e rel
    for pair in ".:src output" "src:agents_wrapper" "src/agents_wrapper:fem_workflow" \
                "src/agents_wrapper/fem_workflow:operator_skill" "output:fem_workflow_operator"; do
        base="${pair%%:*}" keep=" ${pair#*:} "
        for e in "$WORKTREE/$base"/* "$WORKTREE/$base"/.[!.]*; do
            [ -e "$e" ] || continue
            rel="${e#"$WORKTREE"/}"; rel="${rel#./}"
            case "$keep" in *" $(basename "$e") "*) continue ;; esac
            if [ -d "$e" ]; then printf 'Read(./%s/**)\n' "$rel"; else printf 'Read(./%s)\n' "$rel"; fi
        done
    done
    local c
    for c in cat head tail less more grep egrep fgrep rg find sed awk ls tree file stat wc od xxd \
             strings sort uniq cut diff cmp nl tac realpath readlink du git; do
        printf 'Bash(%s)\nBash(%s *)\n' "$c" "$c"
    done
    # Python read from stdin (`python3 - <<'E' ... E`): code typed into the command, outside the
    # allowed runner commands, yet not refused by them.  In the 2026-10-12 BC-sweep tests one run
    # rewrote its brief that way instead of with Edit.  The runner's own `python3 -u <script>` stays.
    for c in "python3 -" "python -"; do
        printf 'Bash(%s)\nBash(%s *)\n' "$c" "$c"
    done
}

# The tools a run is given at all.  The allow and deny lists above only gate calls: every
# other tool's definition (Task, Skill, Workflow, the web and cron tools, the MCP
# connectors) was still sent with each of a run's 26-59 API calls.  Measured on the batch
# transcripts: the claude.ai Docs connector's 8 tools were about 1.7 k tokens of the 35 k
# context at call 1 (30.7 k with them, 29.0 k without, same skill), the memory index about
# 4 k (--no-memory).  --tools keeps the built-in tools the skill uses; --strict-mcp-config
# with no --mcp-config loads no MCP server.  Batch only: the skill tells the agent nothing
# about it.
# --disable-slash-commands drops the skills and slash-command listing (32 skills, 70 commands
# in the init line of batch 20261009-182254): the agent reads SKILL.md as a file, named by
# the prompt (prompt_template.txt), never as a skill.
BUILTIN_TOOLS="Bash,Read,Write,Edit,Glob,Grep"

# A child must be a fresh, standalone session even when this script is itself
# started from inside Claude Code, whose session variables would otherwise leak in.
clean_env_args() {
    local v
    for v in $(env | grep -o '^CLAUDE[A-Z0-9_]*=' | tr -d '='); do
        case "$v" in CLAUDE_CONFIG_DIR) ;; *) printf -- '-u\n%s\n' "$v" ;; esac
    done
}

slot_root() { printf 'output/fem_workflow_operator/_slots/%s-%s' "$BATCH_ID" "$1"; }   # relative to the repository
slot_gpu() { [ "$NGPU" -ge 1 ] && printf '%s' "${GPUS[$(( $1 % NGPU ))]}"; }

# A new run folder <out>/<slug of $1>-<stamp>[-n]; prints its path.
ab_new_run_dir() {
    local slug name run n=2
    slug="$("$PY" "$HELPER" slug "$1")"
    name="$slug-$(date +%Y%m%d-%H%M%S)"
    run="$OUT/$name"
    while ! mkdir "$run" 2>/dev/null; do run="$OUT/$name-$n"; n=$((n + 1)); done
    printf '%s\n' "$run"
}

# One headless Claude Code session on the prompt already in $run/prompt.txt, then ab_helper's
# finalize.  $1 = index, $2 = the instruction recorded (metrics.json `instruction`, and the one
# `drive.py start` stores), $3 = slot, $4 = run folder; further arguments are extra environment
# assignments for the run (NAME=value).  A $run/variant.json, when present, is recorded in
# metrics.json as `variant` (what a realization or a sweep varied).
ab_launch() {
    local idx="$1" instr="$2" slot="$3" run="$4"; shift 4
    local name uuid t0 t1 started finished rc=0 timed_out=0
    name="$(basename "$run")"
    uuid="$("$PY" -c 'import uuid; print(uuid.uuid4())')"
    local -a allow deny env_args extra=()
    mapfile -t allow < <(allowed_tools "$uuid")
    mapfile -t deny < <(denied_tools)
    mapfile -t env_args < <(clean_env_args)
    # One --settings object: the cache TTL, and auto-memory off unless --memory.
    # (CLAUDE_CODE_PROMPT_CACHE_TTL would win over it, but clean_env_args drops every
    # inherited CLAUDE* variable.)  Recorded in the run's settings.json.
    local settings="{\"promptCacheTtl\": \"$CACHE_TTL\""
    [ "$NO_MEMORY" = 1 ] && settings+=', "autoMemoryEnabled": false'
    settings+="}"
    printf '%s\n' "$settings" > "$run/settings.json"
    extra=(--settings "$settings")
    # Measured in batch 20261008-142904: 3 of 5 agents first wrote their brief to
    # <main checkout>/output/... -- the worktree path with its
    # ".claude/worktrees/<name>" part dropped, as if that were a config folder.  The
    # rules refuse it (it is another checkout); two agents then used the right path,
    # one stopped.  The note below says where the repository is; it is the
    # environment's, not the instruction's, and is recorded in system_note.txt.
    # The worktree sentence only when the root is a linked worktree (its git dir differs from the
    # common one, or its path has a .claude/worktrees/ part): run from the main checkout it is not.
    if [ "$PATH_NOTE" = 1 ]; then
        local linked=""
        local git_dir common_dir
        git_dir="$(git -C "$WORKTREE" rev-parse --path-format=absolute --git-dir 2>/dev/null || true)"
        common_dir="$(git -C "$WORKTREE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        if { [ -n "$git_dir" ] && [ "$git_dir" != "$common_dir" ]; } || [[ "$WORKTREE/" == */.claude/worktrees/* ]]; then
            linked=' It is a git worktree; the ".claude/worktrees/..." part is part of the repository path, not a configuration folder.'
        fi
        printf 'Environment note: the repository root and working directory is %s.%s Every file you write goes under this root: a relative path such as output/fem_workflow_operator/<session_id>/operator/brief.md resolves here, and an absolute path must start with %s/.\n' \
            "$WORKTREE" "$linked" "$WORKTREE" > "$run/system_note.txt"
        extra+=(--append-system-prompt "$(cat "$run/system_note.txt")")
    fi
    printf '%s\n' "${allow[@]}" > "$run/allowed_tools.txt"
    printf '%s\n' "${deny[@]}" > "$run/disallowed_tools.txt"
    local -a slot_env=("FEM_OPERATOR_OUTPUT_ROOT=$(slot_root "$slot")")
    [ "$NGPU" -ge 1 ] && slot_env+=("FEM_OPERATOR_GPUS=$(slot_gpu "$slot")")
    # The user's instruction itself, which `drive.py start` stores for the checks: the list line, not
    # the template's text around it, and not what the agent retypes (ab_helper finalize compares them).
    slot_env+=("FEM_OPERATOR_INSTRUCTION=$instr" "$@")
    printf '%s\n' "${slot_env[@]}" > "$run/slot_env.txt"
    log "[$idx] start  $name  claude=$uuid  slot=$slot ${slot_env[*]}  \"$instr\""
    started="$(date -Iseconds)"; t0="$(date +%s.%N)"
    # A runner step can wait for an idle GPU (two runs share two GPUs) and then paint
    # or generate for minutes; the Bash tool's 2-minute default would cut it off.
    ( cd "$WORKTREE" && env "${env_args[@]}" "${slot_env[@]}" \
        BASH_DEFAULT_TIMEOUT_MS="${BASH_DEFAULT_TIMEOUT_MS:-1200000}" \
        BASH_MAX_TIMEOUT_MS="${BASH_MAX_TIMEOUT_MS:-3600000}" \
        timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
        "$CLAUDE_BIN" -p "$(cat "$run/prompt.txt")" \
            --model "$MODEL" --effort "$EFFORT" --session-id "$uuid" \
            --output-format stream-json --verbose \
            --permission-mode dontAsk --allowedTools "${allow[@]}" --disallowedTools "${deny[@]}" \
            --tools "$BUILTIN_TOOLS" --strict-mcp-config --disable-slash-commands \
            "${extra[@]}" \
            < /dev/null > "$run/claude_stream.jsonl" 2> "$run/claude_stderr.log" )
    rc=$?
    t1="$(date +%s.%N)"; finished="$(date -Iseconds)"
    [ "$rc" = 124 ] || [ "$rc" = 137 ] && timed_out=1
    local line
    local -a variant=()
    [ -f "$run/variant.json" ] && variant=(--variant "$run/variant.json")
    line="$("$PY" "$HELPER" finalize --run-dir "$run" --instruction "$instr" --model "$MODEL" \
        --claude-session-id "$uuid" --projects-dir "$PROJECTS_DIR" --workflow-root "$WORKFLOW_ROOT" \
        --exit-code "$rc" --timed-out "$timed_out" --started-at "$started" --finished-at "$finished" \
        --t0 "$t0" --t1 "$t1" --effort "$EFFORT" "${variant[@]}" 2>>"$run/finalize_error.log")" \
        || line="finalize failed (see $run/finalize_error.log)"
    [ -s "$run/finalize_error.log" ] || rm -f "$run/finalize_error.log"
    log "[$idx] finish $name  rc=$rc  $line"
    echo "$run" > "$BATCH_DIR/.done_$(printf %04d "$idx")"
}

# drive.py from a slot (its output root and GPU), as the agent's own commands run.  Prints the
# command's stdout; $1 = slot, the rest drive.py's arguments.  AB_DRIVE_BIN replaces drive.py
# (the stub test only).
ab_drive() {
    local slot="$1"; shift
    local -a env_args slot_env=("FEM_OPERATOR_OUTPUT_ROOT=$(slot_root "$slot")")
    [ "$NGPU" -ge 1 ] && slot_env+=("FEM_OPERATOR_GPUS=$(slot_gpu "$slot")")
    mapfile -t env_args < <(clean_env_args)
    if [ -n "${AB_DRIVE_BIN:-}" ]; then
        ( cd "$WORKTREE" && env "${env_args[@]}" "${slot_env[@]}" "$AB_DRIVE_BIN" "$@" )
    else
        ( cd "$WORKTREE" && env "${env_args[@]}" "${slot_env[@]}" python3 -u "$DRIVE_REL" "$@" )
    fi
}

summarize() {
    local -a runs=() f
    for f in "$BATCH_DIR"/.done_*; do [ -f "$f" ] && runs+=("$(cat "$f")"); done
    "$PY" "$HELPER" summary --batch-dir "$BATCH_DIR" --repo "$WORKTREE" "${runs[@]}" \
        || log "summary failed"
}

# Slot k is free when its last run's pid has exited.
declare -a SLOT_PID=()
free_slot() {
    local k
    for ((k = 0; k < JOBS; k++)); do
        if [ -z "${SLOT_PID[$k]:-}" ] || ! kill -0 "${SLOT_PID[$k]}" 2>/dev/null; then echo "$k"; return 0; fi
    done
    return 1
}
stop_slot_workers() {
    local k
    for ((k = 0; k < JOBS; k++)); do
        [ -d "$WORKTREE/$(slot_root "$k")/.workers" ] || continue
        ( cd "$WORKTREE" && FEM_OPERATOR_OUTPUT_ROOT="$(slot_root "$k")" conda run --no-capture-output -n mechfoundry-agents \
            python -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu --stop-workers > /dev/null 2>&1 ) \
            || log "slot $k: stopping its workers failed"
    done
}

# Run "$1 <index> <slot>" for index 1..$2, JOBS at a time, each in the background on a free slot;
# then stop the slots' workers and write the summary.  Ctrl-C / TERM: write the summary of the runs
# finished so far, then stop the whole process group (the running claude sessions too; those are
# not finalized).
ab_schedule() {
    local fn="$1" count="$2" i slot
    trap 'log "interrupted: summary of the finished runs, then stopping the running sessions"; summarize; stop_slot_workers; trap - INT TERM; kill -TERM 0' INT TERM
    for ((i = 1; i <= count; i++)); do
        while ! slot="$(free_slot)"; do wait -n; summarize; done
        "$fn" "$i" "$slot" &
        SLOT_PID[$slot]=$!
        sleep 2   # distinct start stamps
    done
    wait
    stop_slot_workers
    summarize
    log "batch $BATCH_ID done: $(ls "$BATCH_DIR"/.done_* 2>/dev/null | wc -l) run(s); summary $BATCH_DIR/summary.csv"
}
