# Grading the six operator sets

The 50 objects of `experiments/agentic_batch/objects_50.txt`, each run by six operator sets, are
graded together so that they can be ranked in one table:

| Key | Operator | Harness | Batch |
|---|---|---|---|
| `deepseek` | DeepSeek-V4.1-Flash-1017, high effort (medium requested: DeepSeek serves low and medium as high) | Codex CLI | `output/agentic_batch50_deepseek` |
| `deepseek_max` | DeepSeek-V4.1-Flash-1017, max effort | Codex CLI | `output/agentic_batch50_deepseek_max` |
| `kimi` | Kimi-K3.1-Pro | Codex CLI | `output/agentic_batch50_kimi` |
| `haiku` | Claude Haiku 5.5 | Claude Code | `output/agentic_batch50_haiku` |
| `sonnet` | Claude Sonnet 5.5 | Claude Code | `output/agentic_batch50` |
| `opus` | Claude Opus 5.5 | Claude Code | `output/agentic_batch50_opus` |

Labels A-F, 300 simulations per evaluator. Two evaluator families each run three evaluators on
the same blinded packets: **Fable** (Claude Code's `fable` alias, the current Fable, effort high, headless Claude Code) and
**Astra** (`gpt-6-astra`, effort high, Codex CLI). Each rates every simulation on geometry,
boundary conditions and material, High / Moderate / Low, with the rubric in
`evaluator_prompt.md` (eleven criteria scored met / partial / failed; the category rating follows
mechanically). The three Claude batches are the ones the paper graded, with the exact category
figures its evaluators saw; `records/` holds that grading.

```
README.md            this file
RUN.md               notes of this grading (packets, blinding, versions, runs, validation, results)
grading.json         the config: models in order, seed, figures, packets, packet_edits, families
evaluator_prompt.md  the rubric and instructions every evaluator gets (placeholders filled per evaluator)
scripts/             config.py build_packets.py leak_check.py make_prompts.py run_fable.sh
                     run_astra.sh isolate.sh validate.py combine.py agreement.py drift.py
results/             mappings/, fable/ and astra/ (prompt_<k>.md, eval_<k>.json,
                     evaluation_table.{md,xlsx}), agreement.md, drift.md
records/             the paper's grading, frozen (records/README.md)
```

The fields of `grading.json` are documented in `scripts/config.py`. Packets, evaluator outputs,
logs and CLI homes live under `output/agentic_object_record/model_eval_round3/` (git-ignored);
the staged category figures under `output/agentic_object_record/model_eval_figures/`.

## Running it

Run it from the **main checkout**, because the prompts and packets name the repo path they were
made in. `combine.py` needs a Python with openpyxl (the base conda env); any `python3` runs the
rest. The scripts find every path from their own location, so the cwd does not matter; the
commands below are from `experiments/model_eval`.

```shell
cd experiments/model_eval
# 0. All six batch roots in output/. All six are archived:
#      zstd -dc ../../output/archives/<name>.tar.zst | tar -x -C ../../output
#    (agentic_batch50_deepseek, _deepseek_max, _kimi, agentic_batch50_haiku, agentic_batch50,
#    agentic_batch50_opus).
# 1. The 10 category figures per model, staged in output/agentic_object_record/model_eval_figures/:
#    the open-weight ones built by illustrations/agentic_results/ar_categories.py
#    (--batch-root output/agentic_batch50_<x> --suffix _<x>), the Claude ones copied from the
#    paper's packets (output/agentic_object_record/model_eval/packet_k/figures).
# 2. Mappings (results/mappings/) and the self-contained packets, with the packet_edits applied.
#    The build ends with the blinding check, and a hit fails it: read the hit, then add a
#    packet_edit or a reviewed leak_allow to grading.json and rebuild (never edit a packet by
#    hand: a rebuild would lose it).
python3 scripts/build_packets.py
python3 scripts/leak_check.py                 # the same check again, on its own
# 3. Write the six prompts and commit them as sent.
python3 scripts/make_prompts.py
python3 scripts/make_prompts.py --check
# 4. Run 3 Fable + 3 Astra evaluators, each in its own namespace (in sequence or in parallel).
for k in 1 2 3; do bash scripts/run_fable.sh $k start & done
for k in 1 2 3; do bash scripts/run_astra.sh $k start & done; wait
# 5. Validate the files the evaluators wrote, then commit copies beside the prompts.
python3 scripts/validate.py --family fable --live
python3 scripts/validate.py --family astra --live
for f in fable astra; do for k in 1 2 3; do
  cp ../../output/agentic_object_record/model_eval_round3/$f/evaluator_$k/eval_$k.json results/$f/
done; done
python3 scripts/validate.py --family fable    # the committed copies
python3 scripts/validate.py --family astra
# 6. The tables, Fable against Astra, and the Claude sets against the paper's grading.
python scripts/combine.py --family fable
python scripts/combine.py --family astra
python3 scripts/agreement.py
python3 scripts/drift.py
```

| Script | Does |
|---|---|
| `config.py` | loads `grading.json`; `show` prints it, `runner-env` gives the runners their paths |
| `build_packets.py` | writes the mappings and packets: real files (hard links where possible) without the bulky solver data, then the `packet_edits`, each to a new file (the hard link to the batch is broken, so the run's own file stays as it was; an `old` text that does not occur exactly once stops the build), then the blinding check |
| `leak_check.py [DIR ...]` | the blinding check on its own. It fails if any text in a packet (paths, text files, PNG text chunks) names an operator model, a harness, an evaluator or a batch folder. Its docstring says what counts, and why a bare "deepseek" does not |
| `make_prompts.py [--check]` | writes (or checks) each family's `prompt_<k>.md` from `evaluator_prompt.md` |
| `run_fable.sh K start` | Fable evaluator k in headless Claude Code (`claude -p --model fable --effort high`), isolated by bubblewrap and `dontAsk` permissions |
| `run_astra.sh K start` | Astra evaluator k in Codex (`gpt-6-astra`, effort high), isolated by bubblewrap and Codex's sandbox |
| `isolate.sh` | the bubblewrap layer both runners share |
| `validate.py --family F [--live]` | checks every `eval_<k>.json` of a family against its packet |
| `combine.py --family F` | the family's table: six models, cost `n/p` where a run has no price, and a wall-clock row |
| `agreement.py` | Fable against Astra, `results/agreement.md` |
| `drift.py` | Haiku, Sonnet and Opus against the paper's votes for the same runs, per family and category, `results/drift.md` |

## How isolation works

Both runners share layer 1 (`scripts/isolate.sh`, bubblewrap): the whole filesystem read-only;
`$HOME`, the repo and `/tmp` emptied; the packet's `index.json`, `figures/` and `runs/` bound
read-only into a throwaway cwd; the output dir the one writable place that survives; a clean
environment. Each runner then binds in its own login (the real `~/.claude/.credentials.json` or
`~/.codex/auth.json` into a private CLI home) and adds layer 2: for Fable, `--permission-mode
dontAsk` with an allowlist (Read/Glob/Grep on the packet, Write/Edit on the output dir, Bash for
a fixed set of read-and-script commands, no Agent tool, no MCP servers, auto-memory off); for
Astra, Codex's `workspace-write` sandbox with no network for commands, approvals never asked,
sub-agents and memories off. `run_fable.sh` refuses to start with less than 3 h left on the
login token, so that no token refresh happens through the bind mount. Both refuse a prompt that
is not exactly what `make_prompts.py` renders. The details and the known difference between the
families (a Fable python3 command can reach the network) are in `RUN.md`, "Isolation".

A smoke test uses a stand-in packet and a short isolation prompt, and costs a few cents:
`SMOKE_PACKET=DIR bash scripts/run_fable.sh 1 smoke "TEXT"`, and the same for `run_astra.sh`.
Its output and logs go to `<family logs>/smoke/`.

A Fable evaluator stopped by the claude.ai 5-hour limit ("You've hit your 5-hour limit") is
resumed after the reset with `bash scripts/run_fable.sh K resume "TEXT"` (the same Claude
session); its entries so far stay in its output file. A text that worked: "Your previous turn
was cut off by a usage limit. Continue the task where you left off: keep the entries already in
your output file, grade the simulations not yet in it, and finish with all 300 entries as the
instructions say." `run_astra.sh K resume "TEXT"` does the same for Codex.

## Blinding between harnesses

Blinding must hold between the two harnesses, not only between models. Replies saying a Bash
`cat` was refused and the Read tool used instead name Claude Code, as `sed`/`cat` per an
environment note or `apply_patch` name Codex, and so does a `path:line` citation, a Codex output
habit. No leak term matches these, so they were found by reading the replies, and each is a
`packet_edit` in `grading.json` (36, listed by reason in `RUN.md`).

## Seed

Evaluator k shuffles the labels with `random.Random(2198 + k)`. 2198 is the first seed from 2026
on that gives every model one early (A/B), one middle (C/D) and one late (E/F) label over the
three evaluators (deepseek B/E/C, deepseek_max C/B/E, kimi D/F/A, haiku A/D/F, sonnet E/A/D,
opus F/C/B for k = 1, 2, 3). Evaluators grade in label order and a usage-limit stop falls late
in a packet, so no model is always graded when an evaluator is freshest or most tired.

## Grader versions

Neither grader is pinned. Fable runs Claude Code's `fable` alias, the current Fable
(`claude-fable-5-2` on 2026-11-19; the paper's evaluators ran `claude-fable-5-1`), by the user's
choice (2026-11-19). Astra is pinned by name only: `gpt-6-astra` has no dated snapshot through
the ChatGPT login, so it may be served a different model under the same name than in the paper's
grading. So `scripts/drift.py`'s Fable section measures the change of Fable model as well as of
the grading. Neither CLI (`claude`, `codex`) is
pinned. `RUN.md`, "Versions", records what this grading used.

## Records

`records/paper/` is the paper's grading of Haiku, Sonnet and Opus (Fable 2026-10-14, Astra
2026-10-19), kept as committed; `records/README.md` says what is there and at which commit the
code that rebuilt it lives. Only `drift.py` reads it.
