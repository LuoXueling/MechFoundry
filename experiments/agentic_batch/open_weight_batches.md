# The open-weight 50-object batches

The three open-weight batches, run with `run_batch.sh --agent codex` on the skill, workflow and
solver code the Claude batches ran, through Hugging Face Inference Providers, each pinned to one
route:

| Batch | Model, effort | Route |
|---|---|---|
| `output/agentic_batch50_deepseek` | DeepSeek-V4.1-Flash-1017, high (requested as medium) | `:deepinfra` |
| `output/agentic_batch50_deepseek_max` | DeepSeek-V4.1-Flash-1017, max | `:deepinfra` |
| `output/agentic_batch50_kimi` | Kimi-K3.1-Pro, medium | `:moonshotai` |

They are graded beside the paper's Haiku, Sonnet and Opus batches, all six together, in
`../model_eval/` (Fable and GPT-6-Astra evaluators, 2026-11-19; `../model_eval/RUN.md`). Their
category figures are built with `illustrations/agentic_results/ar_categories.py --suffix _deepseek |
_deepseek_max | _kimi`, and the side-by-side pages with `ARC_NAME=agentic_results_compare_open
./build.sh agenticcompare`.

**Effort.** The `deepseek` batch was run with `--effort medium` (its `metrics.json` say
`medium`), but DeepSeek serves low and medium as high (user, 2026-11-19; it matches the
2026-11-18 calls in `codex_models.json`, where low to xhigh reasoned alike and only max reasoned
longer). So it is a high-effort batch, and the two DeepSeek batches compare high with max, not
medium with max. Figures and tables head it "high effort" (`effort_served` in
`codex_models.json`). Below, "medium" names that batch as it was run.

## How they were run

A first attempt started 2026-11-18 10:48. From about 16:20 DeepInfra, the provider HF routes
DeepSeek to, slowed down (11-61 output tok/s against the normal 104-155, one model call of 40 min),
and from 19:07 HF answered 429 `engine_overloaded` (`limit_source: upstream_provider_shared_pool`;
Kimi through moonshotai still answered), so every run from then failed. The user had the whole
attempt discarded and its folder deleted, and the batch restarted from scratch at 21:32, after three
probes at 144-184 tok/s. A run is redone if it failed or was slowed by the provider (under 80 output
tok/s of model time, or one model call over 600 s), in up to five rounds, each gated by probes.

Both clean batches ended with all 50 runs `ok` and none provider-slowed, in the first round
(the second round's selection found nothing to redo). Decode rate from a fit of call time =
overhead + output tokens / rate over all model calls (`codex_call_rate.py ROOT ...`; calls over 2k output tokens in brackets):

| Batch | Runs | Wall (total, mean per run) | Decode | Overhead | Per run tok/s | Longest call | Cost |
|---|---|---|---|---|---|---|---|
| DeepSeek medium (rerun) | 2026-11-18 23:24 to 11-19 06:09, GPU 0 | 6.7 h, 8.1 min | 139 tok/s (138) | 0.7 s | 116-175 | 143 s | $4.84 |
| DeepSeek max | 2026-11-18 21:32 to 11-19 07:52, GPU 1 | 10.1 h, 12.2 min | 143 tok/s (149) | 0.5 s | 92-192 | 312 s | $8.18 |

The two ran side by side on the same route, so their times compare at the same provider speed.
Runs that ended without a solve, kept as they are (rerunning only those would favour a model):
medium 2 (the skyscraper; the road roller, whose verdict is "the brief's case could not be run"),
max 1 (the Taj Mahal: the model ended its turn mid-task at `geometry_physics`, so its reply is a
progress line, not a report; the CLI exited normally and the run counts as `ok`).

The earlier batches, all 50 runs `ok` (`codex_model_latency.py ROOT` over their rollouts, 2026-11-18):
- DeepSeek medium (2026-11-17 17:10 to 11-18 02:02): 87-135 output tok/s, median 118, 10 s per
  model call on average, longest call 189 s, no trend over the night. Steady, but slower than the
  provider was for the clean max batch: a fit of call time = overhead + output tokens / rate over
  all calls gives 112 tok/s (0.8 s overhead) for medium against 146 tok/s (0.5 s) for the first six
  max runs (2026-11-18 21:32-23:05), and calls over 2k tokens give the same 111 against 146. The model
  and the route are the same, so this is DeepInfra's speed on those evenings, not the effort level.
  A one-object check at medium effort on 2026-11-18 23:10 (the TV remote, stopped after 16 calls)
  decoded at about 159 tok/s, so the slower speed was the provider's that evening. To compare the
  two efforts at the same provider speed, the user had the medium batch rerun from scratch, with
  the max batch's probe gating and rerun rule, on GPU 0 beside the max batch on GPU 1. The first
  medium batch (`output/agentic_batch50_deepseek_20261117`) is not graded; its numbers above are
  kept for the record.
- Kimi (2026-11-18 02:02-10:13, moonshotai): 5-8 s per model call for most of the batch, longest
  call 156 s, no stall or error. Its calls are short (about 400 output tokens each), so tok/s
  (28-105, median 62) mostly tracks time to first token, not decoding; the 80 tok/s rule was set
  for DeepSeek and does not apply. One slower stretch: the eight runs started 07:39-09:23 averaged
  8-13 s per call (28-40 tok/s), about 1.5x the rest. That changes their wall time, not what
  the evaluators see.
