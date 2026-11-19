Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 2.7 M input tokens, 90 k output tokens, 10 min 46 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 88c278050f92 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1663947 | 1566848 | 97099 | 0 | 97099 | 52759 | 437.8 | ok | 2000.0 |
| 2 | an elephant | 2a42271b2499 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2695015 | 2575360 | 119655 | 0 | 119655 | 81320 | 645.5 | ok | 5000.0 |
| 3 | a rhinoceros | 396362fa0554 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1546236 | 1449856 | 96380 | 0 | 96380 | 57552 | 439.6 | ok | 5000.0 |
| 4 | a horse | dcf516d0bc25 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2257665 | 2152704 | 104961 | 0 | 104961 | 47065 | 430.3 | ok | 3000.0 |
| 5 | a crocodile | dca357d5ac3e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2317717 | 2211328 | 106389 | 0 | 106389 | 90349 | 606.4 | ok | 200.0 |
