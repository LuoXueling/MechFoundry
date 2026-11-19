Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 5.2 M input tokens, 120 k output tokens, 16 min 14 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | 72a4a98e7c41 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 5128131 | 4963456 | 164675 | 0 | 164675 | 121274 | 974.1 | ok | 1000.0 |
| 2 | a knee implant | c716786b84a3 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 5192306 | 5072896 | 119410 | 0 | 119410 | 59671 | 691.5 | ok | 10.0 |
| 3 | a walking cane with a rubber tip | 84188643a8e7 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2169031 | 2067200 | 101831 | 0 | 101831 | 58813 | 535.4 | ok | 1000.0 |
| 4 | a human femur | 12e43456d4e0 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2886368 | 2755456 | 130912 | 0 | 130912 | 76777 | 663.0 | ok |  |
| 5 | a lumbar vertebra | 490317059c69 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3892534 | 3748096 | 144438 | 0 | 144438 | 83616 | 902.5 | ok | 200.0 |
