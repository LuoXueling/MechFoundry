# Six-model grading: how it was run

Notes of this grading (complete: both families graded on 2026-11-19). Commands: `README.md`,
"Running it". Config: `grading.json`.

The six operator sets, all on the 50 objects of `experiments/agentic_batch/objects_50.txt`, 300
simulations under labels A-F per evaluator:
- **DeepSeek-V4.1-Flash-1017, high effort (medium requested)** (`deepseek`,
  `output/agentic_batch50_deepseek`; run with `--effort medium`, which DeepSeek serves as high,
  user 2026-11-19, so it is a high-effort batch;
  Codex CLI, run 2026-11-18 23:24 to 11-19 06:09);
- **DeepSeek-V4.1-Flash-1017, max effort** (`deepseek_max`, `output/agentic_batch50_deepseek_max`,
  Codex CLI, run 2026-11-18 21:32 to 11-19 07:52);
- **Kimi-K3.1-Pro** (`kimi`, `output/agentic_batch50_kimi`, Codex CLI, run 2026-11-18);
- **Claude Haiku 5.5, Sonnet 5.5, Opus 5.5** (`haiku`, `sonnet`, `opus`; `output/agentic_batch50_haiku`,
  `output/agentic_batch50`, `output/agentic_batch50_opus`, Claude Code): the batches the paper
  graded (`records/paper/`), the same 150 sessions.

How the open-weight batches were run (routes, provider speed, reruns):
`experiments/agentic_batch/open_weight_batches.md`.

Both evaluator families grade with the rubric of `evaluator_prompt.md`, and both read **the same
packet** k under mapping k (`results/mappings/mapping_<k>.json`):
- **Fable**: Claude Code's `fable` alias (not pinned; `claude-fable-5-2` on 2026-11-19), effort high,
  headless Claude Code (`scripts/run_fable.sh`);
- **Astra**: `gpt-6-astra`, effort high, Codex CLI (`scripts/run_astra.sh`).

Seed 2198 gives every model one early (A/B), one middle (C/D) and one late (E/F) label over the
three evaluators: deepseek B/E/C, deepseek_max C/B/E, kimi D/F/A, haiku A/D/F, sonnet E/A/D,
opus F/C/B for k = 1, 2, 3.

## Figures

From `output/agentic_object_record/model_eval_figures/` (the config's `figures`): the category
figures built by `illustrations/agentic_results/ar_categories.py` for Kimi (2026-11-18, `_kimi`)
and the two DeepSeek batches (2026-11-19, `_deepseek`, `_deepseek_max`), all in the five-column
form the prompt describes. For Haiku, Sonnet and Opus the exact figures of the paper's packets
(`output/agentic_object_record/model_eval/packet_<k>/figures/<category>_<label>.png`); the Opus
figures in `illustrations/` were rebuilt after the paper's grading and are not used. Checked
byte-identical to the paper's packet figures: all 30 (Haiku, Sonnet, Opus) in all three of the
paper's packets (2026-11-19). A paper Opus figure and a new DeepSeek max figure of the same
category were compared by eye: the same layout, legends, colour scales and fonts.

## Packets

- Built 2026-11-19 (about 10:00, with the code before the refactor; the refactored code
  rebuilds the same files, the same index and the same 300 reply hashes per packet) into
  `output/agentic_object_record/model_eval_round3/packet_<k>`: 300 simulations, 60 figures and
  54549 session files each. The packets hold real files (hard links into the batch folders) and
  no symlinks. They leave out VTU, `versions/`, PartField features, checkpoints, `solve/*.glb`
  and `*.disp.bin`.
- Check after the build (`scripts/leak_check.py`): CLEAN for all three packets, no Codex
  tool-name warning, "deepseek" in agent-written text 0 for every label (at the build and again
  2026-11-19 10:5x with the refactored script).

## Blinding

The 36 `packet_edits` of `grading.json`, applied by `build_packets.py` to every packet after the
files are placed and before the check. Each edited file is written as a new file, so its hard link
into the run folder is broken and the batch's own `agent_reply.md` is unchanged; an edit whose
text does not occur exactly once stops the build. All 36 are in replies. By why:

- **The agent named itself** (1, the one leak_check.py flags): Kimi, car brake disc ("so Kimi
  rewrote" -> "so I rewrote").
- **The Codex harness, in the open-weight replies** (4 lines removed; found by reading all 100
  DeepSeek replies, no leak term matches them):
  - the environment note's tool mapping (`sed`/`cat` instead of Read/Grep): DeepSeek high,
    wooden toy train engine;
  - the Codex batch note (one exact command form gets GPU/network access): DeepSeek high, car
    wheel;
  - `apply_patch` against the skill's Write: DeepSeek high, jackhammer; DeepSeek max, crocodile
    (also "the harness"). Both were a reply's last line; the newline before it goes too.
- **A `path:line` citation**, a Codex output habit (5; `judgement.md:1` -> `judgement.md`): 5 Kimi
  replies have one and none of the 150 Claude replies. Kimi: Leaning Tower of Pisa, coffee mug,
  glass-top dining table, rhinoceros, industrial robot arm.
- **Claude Code's permission rules, in the Haiku replies** (20): a Bash `cat` of SKILL.md (or
  another Bash command) refused and the Read tool or Glob used instead, which only the Claude
  runs could hit. The line or sentence is removed, the rest kept. Haiku: bar stool, car brake
  disc, car suspension control arm, car wheel, coffee mug, crankshaft, dental crown, electric
  toothbrush (`ls` refused, Glob used), giraffe, igloo, jelly dessert, layered birthday cake,
  lighthouse, loaf of bread (`ls` and `cat` refused, Glob used), road roller, slice of
  watermelon, Taj Mahal, teddy bear, USB stick, yo-yo.
- **Other Claude Code tells** (6, the last four borderline or weak):
  - "the harness flagged the change" (a file changed while it was open): Sonnet, elephant;
  - "Bash is limited to the run commands" (the Bash allowlist): Opus, thatched stone cottage;
  - a stray `</content>` tag, an artefact of Claude's file-write tool format -> "a stray closing
    tag": Haiku, pogo stick;
  - how Claude Code's Glob behaved with a relative path (line removed): Haiku, walking cane;
  - "`inspect_session.py summary` was allowed" (an allowlist) -> "I used ...": Sonnet, park bench;
  - "commands it doesn't allow" -> "commands outside the skill's list": Opus, elephant.

Read and left: DeepSeek high robot arm, "The skill file was truncated in one read, so I read
it in sections" (Claude Code's Read truncates long files too); DeepSeek max thatched cottage,
`troubleshooting.md:22-34` (the skill's own `file:a-b` read format, not a Codex citation); Opus
crankshaft, "after a Grep of the skill folder" (a tool name in passing, as many replies of both
harnesses use); Sonnet skyscraper, "I grepped the skill folder"; the DeepSeek max Taj Mahal reply,
a mid-task progress line (the model ended its turn early; not a harness tell). Replies of both
harnesses mostly end without a newline; a few of each end with one.

## Prompts

`python3 scripts/make_prompts.py --check` -> all six `identical` (2026-11-19 10:5x). The Fable and Astra
prompts differ only in the output path. They say six agents A-F and 300 simulations.

## Isolation

Both runners share layer 1 (`scripts/isolate.sh`, bubblewrap 0.6.1):
- the whole filesystem is read-only;
- `$HOME` is an empty tmpfs, except the claude and codex binaries and `~/anaconda3`, bound back
  read-only. This hides `~/.claude`, `~/.claude.json`, `~/.codex` and `~/MechFoundry`;
- the repo (and its main checkout) and `/tmp` are empty;
- the packet's `index.json`, `figures/` and `runs/` are bound read-only into a throwaway cwd;
- the output dir is bound read-write;
- the environment is cleared down to HOME, PATH, USER, LANG and TERM.

Each runner then adds its own login and its own layer 2:
- **Astra**: a private `CODEX_HOME` with the real `~/.codex/auth.json` bound in. Layer 2 is
  Codex's `workspace-write` sandbox (no network for commands), approvals never asked, sub-agents
  and memories off. Flags and model are the paper's Astra grading's.
- **Fable**: a private `CLAUDE_CONFIG_DIR` with the real `~/.claude/.credentials.json` (the
  claude.ai OAuth login) bound in. Layer 2 is `--permission-mode dontAsk` with an allowlist:
  Read/Glob/Grep on the packet and the output dir, Write/Edit on the output dir, and Bash for
  `python3 cat ls head tail wc grep sed awk sort uniq find jq echo cd`. Also: tools limited to
  Read, Write, Edit, Glob, Grep and Bash (no Agent, so one agent, as the paper's Fable
  evaluators were); `--strict-mcp-config` with no servers; auto-memory off; no slash commands.
  The runner refuses to start with less than 3 h left on the login token, so that no refresh can
  happen through the bind mount.

The paper's Fable evaluators were Claude Code subagents of one coordinator session. Their
transcripts record `claude-fable-5-1` with effort `high` on every turn. They used Read (the
prompt, `index.json`, 30 figures, scratch text), Bash (python3 scripts that read five replies at
a time, plus cat, ls, grep, sed and a `for` loop), Write and Edit. Here the prompt text is the
first message, as in the paper's Astra grading, instead of a file the agent is told to read.

**Known difference between the families.** A Fable Bash command (python3) can reach the network,
which a Codex command cannot. Claude Code's own Bash sandbox would close this, but it needs
`socat`, which this machine lacks. Nothing of the grading is visible inside the namespace, so the
network gives an evaluator no way to learn the labels.

### Smoke tests (not repeated for this grading, stand-in packet)

`SMOKE_PACKET=DIR bash scripts/run_fable.sh 1 smoke "TEXT"` and the same for `run_astra.sh`; logs
in `output/agentic_object_record/model_eval_round3/{fable,astra}/logs/smoke/`.

The runners' isolation is the one smoke-tested on 2026-11-17 (Claude Code 2.1.325, Codex CLI
0.163.0, the same versions as here): the repo, `~/.claude`, `~/.codex/sessions` and the other
packets invisible, the packet read-only, only the output dir writable, no network for Codex
commands, Fable's Bash able to reach the network. The refactor changed only the paths the
runners take from the config (checked: the runners' environment is the same but for the prompt
path), so the tests were not repeated.

## Versions

- Claude Code: 2.1.325. Codex CLI: 0.163.0.
- `gpt-6-astra` listed in `~/.codex/models_cache.json` on 2026-11-18; every Astra rollout of this
  grading records `gpt-6-astra`, effort `high`.

What is and is not pinned:

- **Fable:** not pinned (user, 2026-11-19). `run_fable.sh` passes the `fable` alias, which resolved
  to `claude-fable-5-2` on 2026-11-19 (a one-line `claude -p --model fable` call: init and reply
  both `claude-fable-5-2`); the paper's evaluators ran `claude-fable-5-1`. The model each evaluator
  was served is in its init event: `claude-fable-5-2` for all three (init and `modelUsage`). So the
  Fable drift of the Claude sets from the paper's votes includes the change of model.
- **Astra:** pinned by name only. `gpt-6-astra` has no dated snapshot through the ChatGPT login,
  so OpenAI may serve a different model under that name than in the paper's grading
  (2026-10-19). Each rollout records the name it was served under.
- **The CLIs:** not pinned. Both runners call the `claude` and `codex` found on the PATH; both
  update themselves.

## Runs

| Family | Evaluator | Start | Wall time | Tokens in (cache read / cache write) / out | Cost | Resumes | Notes |
|---|---|---|---|---|---|---|---|
| Fable | 1 | 2026-11-19 14:30:00 | 31.2 min | 32.65 M (30.84 M cache read / 1.82 M cache write) / 182.2 k | $42.50 (API-equivalent; claude.ai login) | 0 | all 60 figures read, plus 8 session images; 4 Bash commands denied by the allow list |
| Fable | 2 | 2026-11-19 14:30:00 | 27.3 min | 25.15 M (23.51 M / 1.64 M) / 155.1 k | $37.21 | 0 | all 60 figures read, plus 3 session images; 2 Bash commands denied |
| Fable | 3 | 2026-11-19 14:30:00 | 28.5 min | 29.49 M (27.57 M / 1.93 M) / 161.5 k | $42.79 | 0 | all 60 figures read, plus 3 session images; 4 Bash commands denied |
| Astra | 1 | 2026-11-19 10:59:02 | 4.3 + 24.4 min | 15.79 M (15.28 M cached) / 52.0 k | (no cost: ChatGPT login) | 1 | all 60 figures viewed; 40 entries before the stop |
| Astra | 2 | 2026-11-19 10:59:02 | 4.2 + 20.0 min | 9.50 M (9.01 M cached) / 44.7 k | | 1 | all 60 figures viewed; 65 entries before the stop |
| Astra | 3 | 2026-11-19 10:59:02 | 4.2 + 23.1 min | 11.57 M (11.06 M cached) / 46.7 k | | 1 | all 60 figures viewed; 50 entries before the stop |

Fable figures come from the `result` event of `fable/logs/eval_<k>_*.jsonl`: `usage` and
`duration_ms` are per CLI process and are summed, while `total_cost_usd` is the session's running
total, so the cost is the last process's value. Astra figures come from `turn.completed` in
`astra/logs/eval_<k>_*.jsonl` and from the rollouts in `astra/home_<k>/sessions/`. Codex 0.163
calls `view_image` from inside its code-mode `exec` tool, so figure views are counted in the
`exec` inputs.

**Resumes.** Astra: all three evaluators stopped about 4 min in (11:03) with the ChatGPT plan's
usage limit ("You've hit your usage limit ... try again at Nov 24th, 2026 2:42 PM"), at 40, 65
and 50 entries; the turn that failed records no token counts, so the Astra tokens above are the
resumed turn's. The user reset the limit, and all three were resumed at 11:24:50
(`run_astra.sh K resume`, the same Codex session) with "Your previous turn was cut off by a usage
limit. Continue the task where you left off: keep the entries already in your output file, grade
the simulations not yet in it, and finish with all 300 entries as the instructions say. Any
helper scripts you wrote under /tmp are gone; write them again if you need them." All three
finished (11:44-11:49). Fable: none. The three evaluators were started together at 14:30 (a
scheduled start after the claude.ai limit's reset) and all finished in one turn (14:57-15:01),
without reaching the 5-hour limit.

## Validation

`python3 scripts/validate.py --family astra [--live]`: all three files VALID, 300 entries each, no
rating that differs from the one its criteria give, for the live files and the copies in
`results/astra/` (2026-11-19). Fable (`--family fable [--live]`): the same, all three VALID with 300
entries and no rating that differs from its criteria, live and in `results/fable/` (2026-11-19). The
n/a cases (runs that never produced a setting), as Astra's majority votes: DeepSeek high, the road
roller (all three categories) and the skyscraper (BC, material); DeepSeek max, the Taj Mahal (all
three); Kimi, the pogo stick, the road roller and the lumbar vertebra (all three); Haiku, the
crocodile (all three), the pogo stick (BC, material) and the crankshaft (material). Fable: the same
cases, except the road roller, whose geometry it rates for DeepSeek high and whose geometry and BC
it rates for Kimi.

## Results

- Tables: `results/fable/evaluation_table.{md,xlsx}`, `results/astra/evaluation_table.{md,xlsx}`;
  Fable against Astra: `results/agreement.md`; Haiku, Sonnet and Opus against the paper's grading:
  `results/drift.md`.
- Fable and Astra majority votes: exact match on 44 % (geometry), 54 % (BC) and 57 % (material) of
  the 300, within one step on 97 %, 98 % and 87 % of those both graded; where they differ Fable is
  nearly always the higher (161/5, 133/5, 124/6). Fable's evaluators agree more among themselves
  (unanimous on 84-93 % per category, Astra 71-78 %). Mean majority vote (High 2, Moderate 1, Low 0;
  geometry / BC / material / mean), Fable: DeepSeek high 1.40/1.79/1.65/1.61, DeepSeek max
  1.55/1.88/1.51/1.65, Kimi 1.48/1.69/1.53/1.57, Haiku 1.49/1.67/1.64/1.60, Sonnet
  1.64/1.90/1.60/1.71, Opus 1.72/1.96/1.62/1.77; Astra: 0.92/1.29/1.12/1.11, 1.02/1.51/1.06/1.20,
  1.02/1.17/1.19/1.13, 0.92/1.25/1.09/1.08, 0.98/1.44/0.98/1.13, 1.14/1.48/0.98/1.20.
- Drift of the Claude sets from the paper's votes, Astra: exact match on 42-47 of 50 per model and
  category, mean shifts -0.12 to +0.13 steps, one vote (Opus material) more than one step apart.
  Fable: exact match on 41-48 of 50, mean shifts -0.08 to +0.10 steps, no vote more than one step
  apart (this grading served `claude-fable-5-2`, the paper's `claude-fable-5-1`). These sets sit
  beside three other agents and six labels here, so part of a shift can be calibration against their
  neighbours, not the grader.

## Outputs

- `results/fable/eval_<k>.json`, `results/astra/eval_<k>.json`: copies of
  `output/agentic_object_record/model_eval_round3/<family>/evaluator_<k>/eval_<k>.json`.
- `results/<family>/evaluation_table.{md,xlsx}`: from `scripts/combine.py --family <family>`. The
  cost column is the CLI's `total_cost_usd` for the Claude runs and the batch runner's price from
  token counts for the Codex runs, not a billed amount; `n/p` means the run has no price.
- `results/agreement.md`: from `scripts/agreement.py`; `results/drift.md`: from `scripts/drift.py`.
