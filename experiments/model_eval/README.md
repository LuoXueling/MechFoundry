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
   The label mapping is `mappings/mapping_<k>.json`. `prompt_<k>.md` is the prompt evaluator k got.
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
