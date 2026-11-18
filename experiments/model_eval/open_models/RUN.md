# Open-weight round: how it was run

The same grading as the paper round, applied to three sets of open-weight operator runs, all made
under the **Codex CLI**:
- **DeepSeek-V4.1-Flash-1017, medium effort** (`deepseek`, batch `output/agentic_batch50_deepseek`);
- **DeepSeek-V4.1-Flash-1017, max effort** (`deepseek_max`, `output/agentic_batch50_deepseek_max`,
  started 2026-11-18 10:48);
- **Kimi-K3.1-Pro** (`kimi`, `output/agentic_batch50_kimi`).

All three ran the 50 objects of `experiments/agentic_batch/objects_50.txt`: 150 simulations under
labels A, B and C, the paper round's structure. The round is `../rounds/open_models.json`. Grading
the same model at two reasoning efforts in one round puts both under the same evaluators, packets
and day, so the difference between them is not mixed up with the drift between rounds.

**History.** A first version of this round graded DeepSeek (medium), Kimi and the paper round's
Sonnet batch as an anchor (the same 50 sessions and the exact figures the paper-round evaluators
saw, to measure the drift between rounds). Its packets were built and its six evaluators started
2026-11-18 10:40; the user stopped them at 10:45 (Fable had 10/10/20 entries, Astra 45/35/45) and
changed the round to the three open-weight sets above. That version's packets, partial ratings,
logs, prompts, mappings and config are kept in
`output/agentic_object_record/model_eval_round2_stopped_20261118/` (the repo files under
`repo_files/`). Its ratings are not used.

**Figures.** From `output/agentic_object_record/model_eval_round2_figures/` (the round's
`figures`): the category figures built by `illustrations/agentic_results/ar_categories.py`, for
DeepSeek medium and Kimi on 2026-11-18 (`_deepseek`, `_kimi`), for DeepSeek max once its batch
has finished (`_deepseek_max`). All are the five-column form the prompt describes. The `_sonnet`
figures in the same folder belong to the stopped version and are not used.

Both evaluator families grade it with the paper round's rubric (`../evaluator_prompt.md`), and
both read **the same packet** k:
- **Fable**: `claude-fable-5-1`, effort high, headless Claude Code (`../run_fable.sh`);
- **Astra**: `gpt-6-astra`, effort high, Codex CLI (`../astra_high/run_evaluator.sh`).

Evaluator k of both families gets mapping k (`mappings/mapping_<k>.json`). Seed 2029 gives each
model a different label for each evaluator: deepseek A/B/C, deepseek_max C/A/B, kimi B/C/A for
k = 1, 2, 3. Evaluators grade in label order, so every model is graded first, second and third
once. Commands: `../README.md`, "Running a round end to end".

## Versions

- Claude Code: 2.1.325. Codex CLI: 0.163.0 (both as in the smoke tests).
- `gpt-6-astra` listed in `~/.codex/models_cache.json` on 2026-11-18.

What is and is not pinned (checked 2026-11-17):

- **Fable:** pinned. `claude-fable-5-1` is a fixed Anthropic model ID, passed explicitly by
  `run_fable.sh`, so this round's Fable evaluators run the paper round's model.
- **Astra:** pinned by name only. `gpt-6-astra` is the only Astra entry Codex lists, and the
  ChatGPT login offers no dated snapshot, so OpenAI may serve a different model under that name
  than in the paper round (2026-10-19). "Same Astra" here means the same name and effort
  (`high`), not provably the same weights; each rollout records the name it was served under.
- **The CLIs:** not pinned. Both runners call the `claude` and `codex` found on the PATH; both
  update themselves (Codex releases 0.151.0 and 0.163.0 are installed under
  `~/.codex/packages/standalone/releases/`). The paper's Astra round used codex-cli 0.163.0.

## Packets

- Built `____` with `python3 build_packets.py --round open_models` into
  `output/agentic_object_record/model_eval_round2/packet_<k>`: `____` simulations, `____` figures
  and `____` run files each. The packets hold real files (hard links into the batch folders) and
  no symlinks. They leave out VTU, `versions/`, PartField features, checkpoints, `solve/*.glb` and
  `*.disp.bin`.
- Blinding. The stopped version's check found one hit and one tell by hand, both in replies of runs
  this round grades again. They were first fixed by editing the packet copies by hand, which a
  rebuild would have lost, so they are now `packet_edits` in the round config, applied by
  `build_packets.py` after the files are placed and before the check. Each edited file is written
  as a new file, so its hard link into the run folder is broken and the batch's own
  `agent_reply.md` is unchanged; an edit whose text does not occur exactly once stops the build.
  - Kimi's car-brake-disc reply (`vehicle_parts/4_reply.md`) named itself ("so Kimi rewrote");
    changed to "so I rewrote".
  - DeepSeek medium's crocodile reply (`animals/5_reply.md`) ends with a line saying that the
    skill's tool wording differed from "this run's environment note" and that it followed the
    note's mapping (`sed`/`cat`, `apply_patch`). No term in the check matches it, but it names
    the Codex tool mapping that only the open-weight runs had, so the line is removed (with the
    newline before it, so the reply ends without one, as all 100 replies of the two batches do).

  Check after the build (`leak_check.py`): `____`; "deepseek" in agent-written text per label:
  `____`. The DeepSeek max replies read by hand for tells like the crocodile line: `____`.
- Prompts: `make_prompts.py --round open_models --check` → `____`. The Fable and Astra prompts
  differ only in the output path.

## Isolation

Both runners share layer 1 (`../isolate.sh`, bubblewrap 0.6.1):
- the whole filesystem is read-only;
- `$HOME` is an empty tmpfs, except the claude and codex binaries and `~/anaconda3`, bound back
  read-only. This hides `~/.claude`, `~/.claude.json`, `~/.codex` and `~/MechFoundry`;
- the repo (and its main checkout), `/tmp` and the round's extra folders are empty;
- the packet's `index.json`, `figures/` and `runs/` are bound read-only into a throwaway cwd;
- the output dir is bound read-write;
- the environment is cleared down to HOME, PATH, USER, LANG and TERM.

Each runner then adds its own login and its own layer 2:
- **Astra**: a private `CODEX_HOME` with the real `~/.codex/auth.json` bound in. Layer 2 is
  Codex's `workspace-write` sandbox (no network for commands), approvals never asked, sub-agents
  and memories off. Flags and model are the paper round's.
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
first message, as in the Astra round, instead of a file the agent is told to read.

**Known difference between the families.** A Fable Bash command (python3) can reach the network,
which a Codex command cannot. Claude Code's own Bash sandbox would close this, but it needs
`socat`, which this machine lacks. Nothing of the round is visible inside the namespace, so the
network gives an evaluator no way to learn the labels.

### Smoke tests (2026-11-17, from a development checkout, stand-in packet)

The stand-in packet has one entry (label A, food), with `figures/food_A.png` (a copy of
`illustrations/agentic_results_food_opus.png`), a reply and a session stub. It sat under `/tmp`,
outside the repo; `SMOKE_PACKET` pointed the runners at it. Logs are in that checkout's
`output/agentic_object_record/model_eval_round2/{fable,astra}/logs/smoke/`.

**Fable** (`run_fable.sh --round open_models 1 smoke`). Two calls, $0.34 and $0.09 (CLI
`total_cost_usd`). The init event shows `claude-fable-5-1`, tools
[Bash, Edit, Glob, Grep, Read, Write], no MCP servers and `dontAsk`.

| Probe | Result |
|---|---|
| `ls` of the repo, `~/.claude`, `~/.codex`, `~/MechFoundry`, `/tmp` (a compound Bash command) | refused by `dontAsk` (paths outside the allowed dirs) |
| Glob on the repo or `/tmp` | refused |
| Glob in the cwd | the packet's 4 files only |
| Read `figures/food_A.png` | correct: row 1 "a jelly dessert", "Body load 11 N", ×50 |
| Read `.../experiments/model_eval/mappings/mapping_1.json` | refused |
| Write in the cwd, `~/`, `figures/` | refused |
| Write `smoke_out.txt` to the output dir | written |
| python3 listing (second call) | repo: only the `.claude` mount-point dir (that checkout's bind targets); `~`: `anaconda3 .codex .local`, and `~/.codex` holds only `packages`; no `~/.claude`; `/tmp`: Claude's own `claude-1000`, `cc-socks` and the stand-in packet's dir |
| python3 heredoc reading `index.json` and a reply; `ls runs/A/food` | works |
| python3 writes to the cwd and `figures/` | the cwd write "succeeds" in the tmpfs and is gone afterwards; `figures/`: read-only file system |
| python3 HTTPS request | 200 (see the known difference above) |

The real `~/.claude/.credentials.json` was not modified, and the placeholder in the private
config dir is still empty.

**Astra** (`astra_high/run_evaluator.sh --round open_models 1 smoke`). One call, 35 k input
tokens (24 k cached) and 0.6 k output; the ChatGPT login reports no cost. The rollout records
`gpt-6-astra` and `reasoning_effort: high`.

| Probe | Result |
|---|---|
| `ls` of the repo, `~/.claude`, `~/.codex`, `~/.codex/sessions`, `~/MechFoundry`, `/tmp` | repo: only `.claude`; `~/.claude`, `~/.codex/sessions`, `~/MechFoundry`: absent; `~/.codex`: `packages`; `/tmp`: Codex's own mount points and the stand-in packet's dir |
| `view_image figures/food_A.png` | correct: "a jelly dessert", "Body load 11 N" |
| `cat .../mappings/mapping_1.json` | no such file |
| write `smoke_cwd.txt` / `figures/smoke.txt` | the cwd write is gone afterwards / read-only file system |
| write `smoke_out.txt` to the output dir | written |
| python3 HTTPS request | fails (no network in Codex's sandbox) |

The real `~/.codex/auth.json` was not modified.

## Runs

| Family | Evaluator | Start | Wall time | Tokens in (cached) / out | Cost | Resumes | Notes |
|---|---|---|---|---|---|---|---|
| Fable | 1 | | | | | | |
| Fable | 2 | | | | | | |
| Fable | 3 | | | | | | |
| Astra | 1 | | | | (no cost: ChatGPT login) | | |
| Astra | 2 | | | | | | |
| Astra | 3 | | | | | | |

Fable figures come from the `result` event of `fable/logs/eval_<k>_*.jsonl` (`total_cost_usd`,
`usage`, `duration_ms`). Astra figures come from `turn.completed` in `astra/logs/eval_<k>_*.jsonl`.
Also record: how many figures each evaluator opened (Read of a `.png` / `view_image`), and any
permission denials (`permission_denials` in the Fable result).

## Validation

`astra_high/validate.py --round open_models --family fable|astra [--live]`: `____`. The n/a cases
(runs that never produced a setting): `____`.

## Outputs

- `fable/eval_<k>.json`, `astra/eval_<k>.json`: copies of
  `output/agentic_object_record/model_eval_round2/<family>/evaluator_<k>/eval_<k>.json`.
- `fable/evaluation_table.{md,xlsx}`, `astra/evaluation_table.{md,xlsx}`: from `combine.py
  --round open_models --family <family>`. The cost column is the batch runner's price from token
  counts with its dated price table, not a billed amount; `n/p` means the run has no price.
- `agreement.md`: from `astra_high/agreement.py --round open_models`.
