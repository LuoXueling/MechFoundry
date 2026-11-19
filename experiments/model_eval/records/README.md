# Records: the paper's grading

Frozen. Nothing here is written by the scripts in `../scripts/`, and only `../scripts/drift.py`
reads it (the six-model grading's Haiku, Sonnet and Opus against the paper's votes for the same
runs). The files are kept exactly as they were committed; the evaluator prompts are exactly as
sent.

## `paper/`: Haiku, Sonnet and Opus, graded for the paper

The 50 objects of `experiments/agentic_batch/objects_50.txt`, each run by Claude Haiku, Sonnet and
Opus in Claude Code (`output/agentic_batch50_haiku`, `output/agentic_batch50`,
`output/agentic_batch50_opus`, all three archived in `output/archives/`), rated on geometry,
boundary conditions and material with the rubric of `../evaluator_prompt.md`: 150 simulations under
labels A/B/C, three evaluators per family.

- **Fable** (`claude-fable-5-1`, effort high, 2026-10-14), at the top of `paper/`: three Claude
  Code subagents of one coordinator session, reading symlink packets
  (`output/agentic_object_record/model_eval/packet_<k>`). `prompt_<k>.md` is what evaluator k
  got, `eval_<k>.json` its ratings, `mappings/mapping_<k>.json` its labels (seed 1000),
  `evaluation_table.{md,xlsx}` the voted table.
- **Astra** (`gpt-6-astra`, effort high, Codex CLI 0.163.0, 2026-10-19), in `paper/astra_high/`:
  evaluator k got Fable evaluator k's packet content, labels and rubric, as self-contained copies
  at `/home/xl2254/astra_eval_work/packet_<k>`, each in a bubblewrap namespace. Its `RUN.md` has
  the command lines, isolation, times, tokens and validation, and `agreement.md` compares the two
  families. Headline: Astra is much stricter; the families' votes match exactly on 52 / 59 / 48 %
  (geometry / BC / material) and lie within one step on 98 / 95 / 85 %.
- `round.json`: the config the round scripts read for this grading (models and their keys, batch
  roots, seed, families).

The paths written inside these files stay as they were: the prompts name the repo at
`/home/xl2254/implicit_optimize`, where it lived then, and relative links (`../combine.py`,
`../evaluator_prompt.md`, `../README.md`, `astra_high/...`) refer to the folder's layout of that
time. `round.json` uses the round format of that time, which `../scripts/config.py` does not read
(`drift.py` takes only its model keys, batch roots and evaluator count).

## Rebuilding them

The current scripts serve only the six-model grading and no longer rebuild these records (symlink
packets, the old repo path in the prompts, `copy_packets.py`, the `--round` option). The code
that did, and that was checked to reproduce the mappings, the six prompts byte for byte, and the
tables and `agreement.md` from the committed ratings, is at commit
`1c88502f87999d160c5d7a544cd23a513bf3fc41` (`experiments/model_eval/`: `round_config.py`,
`rounds/paper.json`, `build_packets.py --round paper`, `make_prompts.py --round paper --check`,
`combine.py`, `astra_high/`).
