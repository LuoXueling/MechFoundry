# agentic_batch: the operator skill over a list of objects

`run_batch.sh` gives each object of a list to its own headless Claude Code session
(`claude -p --model sonnet`), or with `--agent codex` to a headless Codex session on an
open-weight model ([Codex runs](#codex-runs---agent-codex)), which follows the FEM operator skill
(`src/agents_wrapper/fem_workflow/operator_skill/SKILL.md`) end to end. Each run is recorded in
the layout that `illustrations/agentic_results/ar_runs.py` reads for the paper figure.

```shell
# from the worktree root; the env is irrelevant (bash + python3 stdlib); claude on PATH
experiments/agentic_batch/run_batch.sh output/agentic_batch/test_objects.txt          # N = GPUs at once
experiments/agentic_batch/run_batch.sh -j 1 --timeout 3600 output/agentic_batch/objects_50.txt
experiments/agentic_batch/run_batch.sh --dry-run output/agentic_batch/test_objects.txt
```

Options: `-j N` (default: the GPU count from `nvidia-smi -L`), `--model`, `--effort LEVEL`
(default `medium`), `--timeout SEC` per run (default 7200), `--out DIR`, `--template FILE`,
`--memory` (auto-memory is off by default), `--cache-ttl 5m|1h` (the CLI's `promptCacheTtl`
setting, default `5m`; recorded in each run's `settings.json`), `--dry-run`. Every run also
gets `--disable-slash-commands`: the skill is read as a file, so the skills listing is dropped.

The effort is passed as the CLI's own per-run flag `--effort`, so it does not depend on
`~/.claude/settings.json`. The transcript records the level on every assistant line
(`"effort"`, `"perTurnEffort"`), and the stream's init event does not. Real calls with
`--effort low`, `medium` and `high` recorded low, medium and high. `metrics.json` keeps the
requested `effort` and the `effort_observed` counts per level. The list has one
instruction per line. Lines starting with `#` and blank lines are skipped. GPUs are not pinned:
the skill's runner waits for an idle GPU before each GPU step.

## Realizations and sweeps

Three more scripts run many varied simulations of one object, in the same run layout (they share
`ab_common.sh` with `run_batch.sh`: options, permissions, slots, finalize):

```shell
# one full instruction (with its loading), N runs, each generating its own body
experiments/agentic_batch/run_realizations.sh -n 4 --gpus 1 \
    "a wooden park bench, 1.8 m long, loaded by two adults sitting at mid-span, feet on the ground"
# from a solved session: N heterogeneous per-part material assignments on the same body, regions and loads
experiments/agentic_batch/run_material_sweep.sh -n 4 --gpus 1 output/agentic_batch/<run>/workflow/<session_id>
# from a solved session: N different supports and loads on the same body and material
experiments/agentic_batch/run_bc_sweep.sh -n 4 --gpus 1 output/agentic_batch/<run>
```

- **Realizations**: every run is a new session (tool and API caches are per session), and run k
  gets `FEM_OPERATOR_SEED=<seed base + k>` (`--seed-base`, default 100): the runner writes it into
  the plan for each generation seed the agent left at 0 (TripoSG's, the SDXL drawing's
  `image_seed`, the painter's `texture_seed`; the geometry seed, which the generators take as 0-100, is folded
  into 1-100, so run k gets k there), so the same words make another body.
- **Sweeps** (`ab_sweep.sh`): one Claude call (`--author-model`, default `--model`) writes all N
  variants at once, looking at the source's painted views and segments, so they differ from one
  another; `ab_helper.py variants` checks them (a material variant: every segment assigned, at
  least two materials, E contrast 5-500 x; a load case: supports and loads given) and a refused
  reply is asked for once more. `--variants FILE` gives them instead. Each run then clones the
  source (`drive.py clone SOURCE --fresh-solve`: a new session id in the run's slot, the source's
  checkpoints re-keyed, waiting at its settings card; the source is only read) and the agent edits
  only the varied stage of the plan (`prompt_material_sweep.txt`, `prompt_bc_sweep.txt`); the
  runner rewinds to that stage's card itself. Materials are assigned freely and always per part;
  load cases are free but physically possible; the prompts tell the agent the variant is the
  user's given demonstration case, so the skill's preference for the governing case does not
  bind it, while its physical checks still do.
- `metrics.json` gains `variant` (kind, label, description, the spec, the seed or the source
  session), `variant_kind`, `varied` (the figure's label; also in `summary.csv`) and `case`: what
  `case/case.json` solved (material mode; per part each part's E, ν, ρ from the part field and the
  E contrast; held and loaded regions; body loads; `generation_seeds`, the geometry, image and texture seeds
  the plan last had) and `figure_files` (views, painted views,
  segments, regions, parts montage, displacement and stress pictures and GLBs, the VTU folder),
  relative to `workflow/<session_id>/`. A sweep's batch folder keeps `variants.json` and
  `authoring/` (prompt, replies).

## What a run writes: `output/agentic_batch/<slug>-<YYYYmmdd-HHMMSS>/`

| File | What |
|---|---|
| `workflow/<session_id>/` | a copy of the workflow session the agent created (every session it started, if more than one) |
| `claude_session/` | `<uuid>.jsonl` (the transcript) and `<uuid>/` (`subagents/*.jsonl` + `.meta.json`, `tool-results/`) |
| `claude_result.json` | the CLI's final result object (the last line of `claude_stream.jsonl`) |
| `claude_stream.jsonl`, `claude_stderr.log` | the CLI's raw `stream-json` output and its stderr |
| `metrics.json` | the figure's fields, below |
| `prompt.txt`, `agent_reply.md` | the initial instruction, and the agent's final reply |
| `allowed_tools.txt`, `disallowed_tools.txt`, `permission_denials.json` | the permission rules, and what they refused |

Per batch, `output/agentic_batch/_batches/<batch_id>/` holds `batch.log`, `summary.csv`,
`summary.json`, and `ar_runs.txt` (the run folders, for
`AR_RUNS="$(cat .../ar_runs.txt)" illustrations/build.sh agenticresults`).

`metrics.json`: `instruction`, `model` (the model id that did the work, e.g.
`claude-sonnet-5-5`), `model_requested`, `session_id` (workflow), `session_ids`,
`session_id_found_by`, `claude_session_id`, `input_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens`, `output_tokens`, `thinking_tokens` (included in
`output_tokens`, not on top of it), `effort`, `effort_observed`, `cost_usd`, `wall_clock_s` (measured by the
script around the call), `cli_duration_s` and `cli_duration_api_s` (the CLI's own),
`num_turns`, `status` (`ok`, `cli_error`, `no_result`, `timeout`), `exit_code`,
`started_at`, `finished_at`, plus cross-checks: `main_session_usage`,
`main_transcript_tokens`, `subagent_transcript_tokens`, `tokens_by_model`,
`drive_commands`, `runner_runs`, and `workflow_last_stop`.

### Tokens and sub-agents

The CLI's result has two usage reports. `usage` counts the main session only. `modelUsage`
counts the main session and every sub-agent. This was checked on a real session that spawned one
sub-agent: `usage` matched the main transcript's sum exactly, and `modelUsage` matched the main
plus the sub-agent transcript. The headline token fields are therefore `modelUsage` summed over
models, with nothing added. If the CLI was killed before writing a result, the helper uses the
transcript sums (main + sub-agents) instead and says so in `tokens_source`.

`thinking_tokens` is `modelUsage`'s `thinkingTokens`, and it is part of `outputTokens`. A
one-message call reported 660 output tokens with 196 of them thinking. The transcript's one
message (a thinking block and a text block) also says 660. Fitted rates of $2 per million input
tokens, $10 per million output (660, not 856), $0.20 per million cache read and $4 per million
1-hour cache write reproduce that call's `costUSD` to the last digit, and they also reproduce
the five batch runs exactly. These rates are fitted from the reported costs, not taken from a
price list.

### The workflow session id

Runs execute in parallel, so the script never takes the newest folder. The helper reads the
run's own transcript, finds the agent's `drive.py start` Bash call, and takes the id from its
result (`session_id: <id>` on stderr, `"session_id": "<id>"` in the JSON). If the start output
is missing, it falls back to the ids the agent passed to its other `drive.py` commands. If the
agent started more than one session, the primary one is the one its final reply names. The
prompt already asks the agent to give the session id.

## Permissions

Each run uses `--permission-mode dontAsk`, which refuses without prompting anything not allowed.
The allowlist (`allowed_tools.txt`):

- the runner's one command form, `Bash(python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py *)`;
- Read, Glob and Grep on the skill folder and `output/fem_workflow_operator/`;
- Read on the session's own persisted large tool outputs (`~/.claude/projects/<cwd>/<uuid>/`);
- Write and Edit on `output/fem_workflow_operator/*/operator/**` only.

Only the built-in tools the skill uses are loaded at all (`--tools Bash,Read,Write,Edit,Glob,Grep`),
and no MCP server (`--strict-mcp-config`): the lists above gate calls, but every loaded tool's
definition is sent with each API call of the run.

Claude Code still lets Read, Glob and Grep read anything under the working directory, and lets
read-only shell commands run (`ls`, `cat`, `cmd | head`), whatever the allowlist says. For
that reason the skill's "stay inside two folders" is also enforced with deny rules
(`disallowed_tools.txt`). They cover every repository entry that is not on the path to the skill
folder or to `output/fem_workflow_operator`, plus the shell's file readers. A probe
(`tests/probe_template.txt`, run with `--template`) checked all 20 cases:

- Allowed: the runner, writes and edits under `operator/`, reads of the skill and the
  session, and reads of the session's own persisted output.
- Refused: writes outside `operator/`, writes into the skill, reads of the workflow source
  and `illustrations/`, `/etc`, `ls`, `cat`, and a pipe.

Each run also gets a one-line `--append-system-prompt` note giving the repository root
(`system_note.txt`, `metrics.json` `system_note`; turn it off with `--no-path-note`). It was
added after batch `20261008-142904`, where 3 of 5 agents first wrote the brief to
`<main checkout>/output/...`, the worktree path with its `.claude/worktrees/...`
part dropped. The rules refused it, and one agent stopped there. The instruction itself is
unchanged.

The child environment drops the parent's `CLAUDE*` session variables, so starting the batch
from inside Claude Code still gives fresh sessions. It also sets `BASH_DEFAULT_TIMEOUT_MS`
(20 min) and `BASH_MAX_TIMEOUT_MS` (60 min). Without them, the Bash tool's 2-minute default
would cut off a runner step that waits for an idle GPU and then paints or generates.

## Codex runs (`--agent codex`)

`--agent codex --model deepseek-v4.1-flash|kimi-k3.1-pro` runs each object with `codex exec --json`
instead of `claude -p`. The default provider is Hugging Face Inference Providers (`--provider hf`,
`HF_TOKEN` or the `hf auth login` token). Each model is pinned there to one provider, its route,
so every request of a batch is served alike: the id is sent as `<id>:<route>`, and
`AB_CODEX_ROUTE` overrides the route. `--provider vendor` uses the model's own company (DeepSeek,
Moonshot). Codex speaks only the Responses API to a custom provider, and all of these endpoints
offer it.

- **Models, routes and accounting prices:** `codex_models.json`.
- **The three open-weight 50-object batches** (how they were run, provider speed, reruns):
  `open_weight_batches.md`; `codex_model_latency.py` and `codex_call_rate.py` measure a batch's
  model time and decode rate from its Codex rollouts.
- **Endpoints and what was verified:** `ab_helper.py` (`CODEX_PROVIDERS`).
- **User-facing summary:** the top-level README ("Headless batch with Codex and open-weight models").
- **Hugging Face goes through a local translator, `hf_relay.py`.** The router's own Responses
  endpoint (`/v1/responses`, Hugging Face's responses.js) fails Codex in three ways (measured
  2026-11-17):
  - it refuses request bodies over 100 KB (413), and a session's history passes that in its first minute;
  - it drops `developer` messages, which carry Codex's whole system prompt and the runner's notes;
  - it refuses a tool result that carries an image (`view_image`).

  The runner therefore starts the translator, which serves the Responses API on 127.0.0.1 and calls
  the router's Chat Completions endpoint (several-MB bodies). It maps requests and stream events as
  responses.js does, with two differences: `developer` becomes `system`, and an image tool result
  becomes text with the image in a user message after it. Codex-only fields such as
  `client_metadata` (your git remote and paths) are not sent. It logs one line per request to
  `_batches/<id>/hf_relay.log`, with no headers or bodies. `AB_HF_RELAY=0` turns it off, and
  `tests/check_hf_relay.py` tests both directions.
- **Before a batch:** `codex_preflight.sh --model M [--provider P]` runs one short session with the
  batch's configuration. It checks the provider (key, credits, request shape, route), `drive.py`
  outside the sandbox, image input (a two-colour picture drawn at random) and `apply_patch`.

The prompt is `prompt_template.txt`, the same as for Claude. What differs:

| | Claude run | Codex run |
|---|---|---|
| configuration | flags on `claude -p` | the run's own `codex_home/` (`CODEX_HOME`): `config.toml` and `rules/operator.rules`; nothing of `~/.codex` reaches the run |
| system note | `--append-system-prompt` (path note) | `developer_instructions`: the path note, then `codex_tools.txt`, then `codex_note.txt` |
| tools | `--tools Bash,Read,Write,Edit,Glob,Grep`, no MCP | `exec_command`, `write_stdin`, `view_image`, `apply_patch`, `request_user_input`; web search and the other Codex tools off (two of them are refused by the HF router) |
| AGENTS.md / CLAUDE.md | not read | not read (`project_doc_max_bytes = 0`) |
| the runner | allowed by `Bash(python3 -u …/drive.py *)` | a `prefix_rule` with `allow`, which runs it outside the sandbox (GPU, network, conda, workers) |
| other commands | refused unless read-only and allowed | run in the `workspace-write` sandbox: writes only in the repository and `/tmp`, no network, no GPU; `git`, `python -c` and `python -` refused |
| reads outside the two folders | refused by deny rules | possible (Codex reads with shell commands), so the skill's rule 3 holds by instruction only |
| writes | `Write`/`Edit` on `operator/**` only | `apply_patch`, anywhere the sandbox allows; the note confines it to `operator/` |
| long runner steps | `BASH_DEFAULT_TIMEOUT_MS` 20 min | `exec_command` returns while the process keeps running; the note tells the agent to wait on it with `write_stdin` |
| effort | `--effort` (default `medium`) | `model_reasoning_effort`: `--effort`, else the model's `effort` in `codex_models.json` (`medium`). DeepSeek serves `low` and `medium` as `high` (`effort_served`), so a DeepSeek run at `medium` is a high-effort run; `metrics.json` records the effort requested |
| record | `claude_session/`, `claude_stream.jsonl`, `claude_result.json` | `codex_home/sessions/…/rollout-*.jsonl`, `codex_stream.jsonl`, `codex_stderr.log`, `codex_model.json` |

`codex_tools.txt` maps the tools the skill names to Codex's own: Read → `sed -n 'a,bp'`/`cat`,
Write/Edit → `apply_patch`, "the file reader" for pictures → `view_image`. It holds for any Codex
use (an interactive session, the UI, other front ends). `codex_note.txt` follows it in a batch run
only: the runner's command form, waiting on a long `drive.py` with `write_stdin`, and never asking
the user, which fit an unattended run in the batch's own Codex home. Neither `SKILL.md` nor the
prompt changes, so the Claude batches stay comparable. `drive.py` takes Codex's
`CODEX_THREAD_ID` as the agent's identity (its "as before" advice), as it takes
`CLAUDE_CODE_SESSION_ID` from Claude.

`metrics.json` has the same fields, plus:

- `agent`, `provider`, `route` (the id in `model` is without it), `model_label`;
- `codex_thread_id`;
- `tool_calls` (per tool, from the rollout) and `images_viewed`;
- `cost_basis`.

`claude_session_id` is `null`. `cost_usd` is priced from the tokens with the model's `price` in
`codex_models.json` (`ab_helper.codex_cost`), not billed. The price is the same whichever
provider served the run:

- DeepSeek: its API's peak-hour price, cache miss $0.30, cache hit $0.006, output $1.20 per M.
- Kimi: Moonshot's price, input $2, cached input $0.20, cache write (5-min TTL) $2, output $10 per M.

A model outside the table gets `null`.
Tokens follow Claude's convention:

- `input_tokens` is the uncached input.
- `cache_read_input_tokens` is Codex's `cached_input_tokens`.
- `cache_creation_input_tokens` is `cache_write_input_tokens`.
- `thinking_tokens` is `reasoning_output_tokens`, part of `output_tokens`.

The raw counts are kept in `tokens_raw`. A run killed before its `turn.completed` takes its
totals from the rollout's last `token_count`.

Sweeps (`ab_sweep.sh`) author their variants with Claude. With `--agent codex`, give a Claude
`--author-model` or `--variants FILE`. `run_realizations.sh` takes `--agent codex` as is.

## Tests

```shell
bash experiments/agentic_batch/tests/test_stub.sh 2     # no API calls; everything in a temp dir
bash experiments/agentic_batch/tests/test_variants_stub.sh   # the realization and sweep scripts, likewise
bash experiments/agentic_batch/tests/test_codex_stub.sh      # --agent codex (python3 >= 3.11), likewise
python3 experiments/agentic_batch/tests/check_hf_relay.py    # the relay's request and stream conversion
```

`test_codex_stub.sh` runs `tests/stub_codex.py` in place of `codex`. The stub checks what each
run is given: arguments, a closed stdin, its own `CODEX_HOME` config and rules, the key in its
environment, and no inherited Codex session. It writes a stream and a rollout in codex-cli
0.163.0's format. `tests/check_codex_stub.py` checks:

- the statuses (ok, provider refusal, crash, timeout, no session);
- the token arithmetic, the drive commands and the images viewed;
- the pinned route and the priced cost (Kimi on HF), and a vendor run with neither;
- that the provider key is in no run file and not in the log;
- that a batch without a key refuses to start.

`test_variants_stub.sh` runs the three variant scripts on the stub (`drive.py clone` stubbed by
`tests/stub_drive.py`; the real clone and the batch seed are tested in
`operator_skill/tests/test_clone_and_seed.py`) and checks the seeds, the authoring retry, one clone
per sweep run, the recorded variants and case, and that the source session is unchanged.

`tests/stub_claude.py` stands in for `claude` (via `CLAUDE_BIN`). It sleeps, then writes a
transcript, a sub-agent transcript, a fake workflow session and a result in the real CLI's
format. `tests/check_stub.py` checks:

- peak concurrency equals `-j`, runs start in list order, and the next run starts within
  seconds of one finishing;
- metrics fields and the token arithmetic, plus that the effort is passed and appears in the
  transcript and that `thinking_tokens` is read;
- the copies, and that `ar_runs.load_run` reads each run;
- the summary and log;
- the failure cases: CLI error, crash without a result, timeout, no session, and two
  sessions in one run.

Ctrl-C writes the summary of the finished runs and then stops the whole process group. Runs
still in progress are not finalized.
