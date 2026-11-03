# Astra round: how it was run (2026-10-19)

A second, independent grading of the same 150 simulations by **GPT-6-Astra** (`gpt-6-astra`) at
**high** reasoning effort, via the **Codex CLI 0.163.0** (`codex exec`, logged in with ChatGPT).
The only intended difference from the Fable round (`claude-fable-5-1`, effort high) is the
evaluator model. Evaluator k got the same packet content, the same blinded labels
(`../mappings/mapping_<k>.json`) and the same rubric (`../evaluator_prompt.md`) as Fable
evaluator k.

## Model and effort

- Model id from `~/.codex/models_cache.json`: `gpt-6-astra` ("GPT-6-Astra"; input text + image;
  efforts low ... ultra). It is the only Astra listed.
- Effort: `-c model_reasoning_effort="high"`. Every rollout records `"model":"gpt-6-astra"` and
  `"reasoning_effort":"high"`.
- Images: the evaluator opens each figure itself with Codex's built-in `view_image` tool, as the
  Fable agents opened theirs. A smoke test asked for row contents, and the model read them
  correctly (row 1 jelly dessert, row 4 "Body load 26 N"). Each evaluator viewed all 30 figures
  of its packet, according to the rollouts.

## Packets (`copy_packets.py`)

The Fable packets (`output/agentic_object_record/model_eval/packet_<k>`) link into the run
folders by symlink. The Haiku and Sonnet run folders had been archived
(`output/archives/agentic_batch50{_haiku,}.tar.zst`), so 100 of the 150 reply and session links
dangled. From each archive only these were extracted to `/home/xl2254/astra_eval_work/restore/`
(outside the repo, about 1.2 GB each):
- `_batches/`, and every run's `metrics.json` and `agent_reply.md`;
- the workflow session folder of the 50 runs the packets use, without `versions/`,
  `solve/api_fem_forward-*/`, `solve/*.glb`, `*.disp.bin`, `partfield_features/` and
  `checkpoints.sqlite`.

`copy_packets.py` then rebuilt each packet at `/home/xl2254/astra_eval_work/packet_<k>`. It holds
the same `index.json` entries (only the paths changed), byte-identical figures, and hard-linked
replies and session files. The Opus files come from the live `output/agentic_batch50_opus` and
were not modified. The same file exclusions apply to all three models. The copies contain real
files, not symlinks, so no path names the batch folder, which would reveal the model. A grep of
every text file in a packet found no "haiku", "sonnet", "opus" or batch-folder name.

## Isolation (`run_evaluator.sh`)

Each evaluator ran in two layers:
1. **bubblewrap:** `/` is bound read-only. The repo (Fable's ratings, the mappings, the Fable
   packets), `~/.claude`, `/home/xl2254/astra_eval_work` (the other packets), `/tmp` and
   `~/.codex` (other Codex sessions) are replaced by empty tmpfs. Bound back in are the packet's
   `index.json`, `figures/` and `runs/` (read-only, in a tmpfs packet dir, so the cwd is
   writable but nothing written there survives), the output dir
   `output/agentic_object_record/model_eval_astra/evaluator_<k>/` (read-write), a private
   `CODEX_HOME` with the real `auth.json` bound in, and `~/.codex/packages` (read-only; the
   binary).
2. **Codex's own sandbox:** `-s workspace-write`, cwd = the packet, `--add-dir` = the output dir,
   no network for commands, `approval_policy="never"`.

A smoke test (`output/agentic_object_record/model_eval_astra/logs/smoke/`) confirmed what the
evaluator could not do: see the repo's `experiments/`, other packets or `~/.codex/sessions`,
change a figure, or keep a file in its cwd. It could write to its output dir. Switching off
Codex's sandbox (and relying on layer 1 alone) was not used.

## Command lines

Start (evaluator k; the full bwrap line is in `logs/commands_<k>.txt`):

```
bwrap --ro-bind / / --dev-bind /dev /dev --bind /proc /proc --tmpfs <repo> --tmpfs ~/.claude \
  --tmpfs <work> --tmpfs /tmp --tmpfs ~/.codex --ro-bind ~/.codex/packages ~/.codex/packages \
  --tmpfs <packet> --ro-bind <packet>/{index.json,figures,runs} ... --bind <out> <out> \
  --bind <work>/codex_home_k <work>/codex_home_k --bind ~/.codex/auth.json <work>/codex_home_k/auth.json \
  --setenv CODEX_HOME <work>/codex_home_k --chdir <packet> --die-with-parent \
codex exec -m gpt-6-astra -c model_reasoning_effort="high" -c approval_policy="never" \
  --disable multi_agent --disable memories -s workspace-write -C <packet> --add-dir <out> \
  --skip-git-repo-check --json -o <out>/last_message_<stamp>.txt - < prompt_<k>.md
```

Resume (it worked in the smoke test; no evaluator needed it):
`codex exec resume -m gpt-6-astra -c model_reasoning_effort="high" ... --skip-git-repo-check --json <thread-id> -`.

`--disable multi_agent` keeps each evaluator a single agent, as the Fable evaluators were. The
Codex default lets the root agent spawn sub-agents.

## Runs

All three were started in parallel at 19:37 EDT. Each finished in a single turn with no resume,
and each wrote its file after every figure as instructed.

| Evaluator | Wall time | Input tokens (cached) | Output tokens (reasoning) | Tool calls |
|---|---|---|---|---|
| 1 | 15 min 42 s | 9.08 M (8.75 M) | 27.8 k (8.9 k) | 75 |
| 2 | 15 min 52 s | 8.69 M (8.39 M) | 29.0 k (9.4 k) | 73 |
| 3 | 11 min 09 s | 8.73 M (8.36 M) | 22.3 k (5.1 k) | 73 |

The token counts come from the `turn.completed` usage in `logs/eval_<k>_*.jsonl`. The ChatGPT login
reports no dollar cost. Each evaluator read the replies five rows at a time with a small script
of its own, and opened `resolved.json`, `session.json`, `brief.md` and `judgement.md` where needed.

## Validation (`validate.py`)

All three files are VALID. Each has 150 entries, exactly the packet's (label, category, object)
keys, every required field, valid ratings and criterion scores, and every rating equal to the one
derived from its criteria. The n/a cases match the Fable round (the Haiku crocodile: all
categories; the pogo stick: BC and material; the crankshaft: material). The same script also
passes all three Fable files.

## Outputs

- `eval_<k>.json`: copies of `output/agentic_object_record/model_eval_astra/evaluator_<k>/eval_<k>.json`.
- `evaluation_table.{md,xlsx}`: from `python ../combine.py --evals astra_high --out astra_high
  --runs-fallback /home/xl2254/astra_eval_work/restore`.
- `agreement.md`: from `python3 agreement.py`.
