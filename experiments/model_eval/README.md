# The three-model evaluation

Moved from `illustrations/agentic_results/model_eval/` (2026-10-17). The evaluator prompts
`prompt_<k>.md` are kept exactly as sent.

The 50 objects of `experiments/agentic_batch/objects_50.txt`, each run by Haiku, Sonnet and Opus
(`output/agentic_batch50_haiku`, `output/agentic_batch50`, `output/agentic_batch50_opus`), are
rated by three independent evaluator agents on geometry, boundary conditions and material, each
High / Moderate / Low, with the rubric in `evaluator_prompt.md` (eleven criteria scored met /
partial / failed; the category rating follows mechanically).

All three batch roots are archived in `output/archives/` (`agentic_batch50_haiku`,
`agentic_batch50`, `agentic_batch50_opus`, `.tar.zst`). Rebuilding the packets or
re-running `combine.py` without `--runs-fallback` needs them extracted at their original
paths: `zstd -dc output/archives/<name>.tar.zst | tar -x -C output`.

1. `build_packets.py` writes one blinded packet per evaluator to
   `output/agentic_object_record/model_eval/packet_<k>/` (git-ignored): the 30 category figures, the
   agents' replies and links to their sessions, under labels A/B/C drawn at random per evaluator.
   The label mapping is `mappings/mapping_<k>.json`. `prompt_<k>.md` is the prompt evaluator k got;
   `make_prompts.py --check` shows it is still exactly what the template renders for this round.
2. Each evaluator writes `eval_<k>.json`: one entry per simulation, the criteria, the three ratings
   and a short note per category.
3. `combine.py` (a Python with openpyxl) maps the labels back and writes `evaluation_table.md` and
   `evaluation_table.xlsx`: one row per object, per model the three ratings as "evaluator 1/2/3 →
   majority vote" (all three different: the median, marked *) and the coding agent's cost (the CLI's
   `total_cost_usd`, the workflow's own API calls excluded), then a row of High/Moderate/Low counts
   with each model's total cost and a row of each model's total wall-clock time over the 50 runs.

`combine.py --evals DIR --out DIR --runs-fallback DIR` reads another round's ratings, writes its
tables elsewhere, and reads archived batch roots from an extract. With no arguments it behaves as
before while the three batch roots are live. If a root has been archived (Haiku and Sonnet since
2026-10-14, Opus since 2026-10-20, so all three now), it stops with a message naming the archive and the two folders to extract, where
before it raised a traceback.

## Second round: GPT-6-Astra via Codex (`astra_high/`, 2026-10-19)

The same three-evaluator grading, repeated with **GPT-6-Astra** (`gpt-6-astra`) at **high**
reasoning effort in the **Codex CLI 0.163.0** (`codex exec`). Evaluator k got Fable evaluator k's
packet content, labels and rubric text, so only the evaluator model changed. The prompts differ
from `prompt_<k>.md` only in the packet path and the output path. `astra_high/RUN.md` has the
command lines, isolation, run times, tokens and validation. `astra_high/agreement.md` compares the
two rounds.

Procedural differences, none of which change what the evaluator could read:
- the packets are self-contained copies at `/home/xl2254/astra_eval_work/packet_<k>`
  (`astra_high/copy_packets.py`). The Haiku and Sonnet run folders were archived, so their replies
  and session files were extracted from the archives. Files are hard links, not symlinks into the
  batch folders, and the large solver and history data (VTU, `versions/`, PartField features,
  checkpoints) was left out for all three models;
- each Codex job ran in a bubblewrap namespace where the repo, `~/.claude`, the other packets and
  other Codex sessions were invisible, and its only lasting writable place was its own output dir;
- sub-agents were disabled (`--disable multi_agent`).

Headline: Astra is much stricter than Fable. The two rounds' majority votes match exactly on
52 % (geometry), 59 % (BC) and 48 % (material) of the 150 simulations, and lie within one step on
98 / 95 / 85 %. Where they differ, Fable is almost always the higher. Astra marks M1 (part
mapping) failed in 146 of 450 ratings, against 4 for Fable. Under both evaluators Haiku comes
last on all three categories. Fable puts Opus ahead of Sonnet on all three; Astra has them level
on geometry and material and Sonnet ahead on BC.

## Rounds: grading another set of batches (2026-11-17)

**Grader versions.** Fable is pinned: `claude-fable-5-1` is a fixed model ID. Astra is pinned by
name only: `gpt-6-astra` has no dated snapshot through the ChatGPT login, so a later round may be
served a different model under the same name. Neither CLI (`claude`, `codex`) is pinned. Each
round's `RUN.md` records the versions it used (`open_models/RUN.md`, "Versions").

The scripts above now take a **round**: a small JSON config in `rounds/`, whose fields are
documented in `round_config.py`. It lists the graded models in order (key, display name, batch
root, figure suffix, leak-check terms), the number of evaluators and the label seed. It also says
where the mappings, packets, prompts and ratings live for each evaluator family (`fable`,
`astra`), and which blinding redactions the packets get (`packet_edits`: per model and object, the
reply or a session file, an exact old and new text and why). Every script but one defaults to
`rounds/paper.json`, the round above, so the commands above still behave as they did. The
exception is `astra_high/validate.py`: without a round it keeps its old form, `validate.py FILE
INDEX_JSON` (one ratings file against one packet index), so the paper round's families are checked
with `--round paper --family fable|astra`. The scripts give
the same mappings, the same packets, the same six prompts byte for byte, and the same `evaluation_table.{md,xlsx}` and `agreement.md` from the committed
ratings. This was checked on 2026-11-17 against the code before the change; the xlsx files differ
only in their creation timestamp.

Two things did change for every round:
- The repo root is now wherever the scripts sit, instead of the hard-coded
  `/home/xl2254/implicit_optimize`. The paper round keeps that old path as its `prompt_repo`,
  because its prompts name it.
- `evaluator_prompt.md` has four more placeholders (`{N_AGENTS}`, `{LABELS}`, `{N_OBJECTS}`,
  `{N_SIMULATIONS}`) beside `{PACKET}` and `{OUT}`. The rubric and the instructions are unchanged.

| Script | Use in a round |
|---|---|
| `round_config.py` | loads a config; `show --round R` prints a round, and `runner-env` gives the runners their paths |
| `build_packets.py --round R` | writes the mappings and packets. A new round uses `packet_mode: copy`: self-contained real files (hard links where possible) with the `copy_packets.py` exclusions. It then applies the round's `packet_edits`, each to a new file (the hard link to the batch is broken, so the run's own file stays as it was; an `old` text that does not occur exactly once stops the build), and ends with the blinding check |
| `leak_check.py --round R` | runs the blinding check on its own. It fails if any text in a packet (paths, text files, PNG text chunks) names an operator model, a harness, an evaluator or a batch folder. Its docstring says what counts, and why a bare "deepseek" does not |
| `make_prompts.py --round R [--check]` | writes (or checks) each family's `prompt_<k>.md` from `evaluator_prompt.md` |
| `run_fable.sh --round R K start` | runs Fable evaluator k in headless Claude Code (`claude -p --model claude-fable-5-1 --effort high`), isolated by bubblewrap and `dontAsk` permissions |
| `astra_high/run_evaluator.sh --round R K start` | runs Astra evaluator k in Codex (`gpt-6-astra`, effort high), isolated by bubblewrap and Codex's sandbox. It is the paper round's script, generalized |
| `isolate.sh` | the bubblewrap layer both runners share: `$HOME`, the repo and `/tmp` emptied, the packet read-only, the output dir writable, a clean environment |
| `astra_high/validate.py --round R --family F [--live]` | checks every `eval_<k>.json` of a family against its packet |
| `combine.py --round R --family F` | writes the family's table: N models, cost `n/p` where a run has no price, and a wall-clock row |
| `astra_high/agreement.py --round R` | compares Fable and Astra for the round and writes to the round's `agreement.out` |

### Running a round end to end

This example is the open-weight round in `rounds/open_models.json`: three sets of runs made under
the Codex CLI, DeepSeek-V4.1-Flash-1017 at medium and at max reasoning effort and Kimi-K3.1-Pro, so
labels A/B/C and 150 simulations. Its notes, including a stopped first version that graded the
paper round's Sonnet batch as an anchor, are in `open_models/RUN.md`. Run it from the **main
checkout**, because the prompts and packets name the repo path they were made in. `combine.py`
needs a Python with openpyxl (the base conda env); any `python3` runs the rest.

```shell
cd experiments/model_eval
# 0. The batches have finished: output/agentic_batch50_deepseek,
#    output/agentic_batch50_deepseek_max, output/agentic_batch50_kimi (_batches/ + runs).
# 1. Build the 10 category figures per model (env mechfoundry-warp117) and copy them into the
#    round's figures folder (output/agentic_object_record/model_eval_round2_figures/):
#      illustrations/agentic_results/ar_categories.py --batch-root output/agentic_batch50_deepseek --suffix _deepseek
#      illustrations/agentic_results/ar_categories.py --batch-root output/agentic_batch50_deepseek_max --suffix _deepseek_max
#      illustrations/agentic_results/ar_categories.py --batch-root output/agentic_batch50_kimi --suffix _kimi
# 2. Write the mappings and the self-contained packets
#    (output/agentic_object_record/model_eval_round2/packet_<k>), with the round's packet_edits
#    applied. The build ends with the blinding check, and a hit fails it: read the hit, then add a
#    packet_edit or a reviewed leak_allow to the round config and rebuild (never edit a packet
#    by hand: a rebuild would lose it).
python3 build_packets.py --round open_models
python3 leak_check.py --round open_models          # the same check again, on its own
# 3. Write the six prompts and commit them as sent.
python3 make_prompts.py --round open_models
python3 make_prompts.py --round open_models --check
# 4. Run 3 Fable + 3 Astra evaluators, each in its own namespace (in sequence or in parallel).
for k in 1 2 3; do bash run_fable.sh --round open_models $k start & done
for k in 1 2 3; do bash astra_high/run_evaluator.sh --round open_models $k start & done; wait
#    To continue an evaluator that stopped early:
#      bash run_fable.sh --round open_models K resume "Continue ..."
# 5. Validate the files the evaluators wrote, then commit copies beside the prompts.
python3 astra_high/validate.py --round open_models --family fable --live
python3 astra_high/validate.py --round open_models --family astra --live
for f in fable astra; do for k in 1 2 3; do
  cp ../../output/agentic_object_record/model_eval_round2/$f/evaluator_$k/eval_$k.json open_models/$f/
done; done
python3 astra_high/validate.py --round open_models --family fable     # the committed copies
python3 astra_high/validate.py --round open_models --family astra
# 6. Write the tables and the agreement report.
python combine.py --round open_models --family fable
python combine.py --round open_models --family astra
python3 astra_high/agreement.py --round open_models
```

A smoke test uses a stand-in packet and a short isolation prompt, and costs a few cents:
`SMOKE_PACKET=DIR bash run_fable.sh --round R 1 smoke "TEXT"`, and the same for
`astra_high/run_evaluator.sh`. Its logs go to `<family logs>/smoke/`.

A new round is a new `rounds/<name>.json`; copy `open_models.json` to start one. Give it:
- new mappings, packet and output folders;
- a label seed whose draws differ across the evaluators;
- a packet folder name that does not reveal the models. The packet path is in every `index.json`
  entry and in the prompt, so the leak check sees it.
