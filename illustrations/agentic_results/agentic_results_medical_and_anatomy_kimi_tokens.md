Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 6.1 M input tokens, 30 k output tokens, 16 min 3 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | 209ab0fbde2b | moonshotai/Kimi-K3.1-Pro | codex | 2420090 | 2335229 | 84861 | 0 | 84861 | 15749 | 381.2 | ok | 2000.0 |
| 2 | a knee implant | b5ad6908102f | moonshotai/Kimi-K3.1-Pro | codex | 3049662 | 2826553 | 223109 | 0 | 223109 | 17050 | 597.3 | ok | 1000.0 |
| 3 | a walking cane with a rubber tip | d105a21ad391 | moonshotai/Kimi-K3.1-Pro | codex | 6130515 | 5996471 | 134044 | 0 | 134044 | 29811 | 829.2 | ok | 30.0 |
| 4 | a human femur | c3ab6b31eefd | moonshotai/Kimi-K3.1-Pro | codex | 941121 | 874640 | 66481 | 0 | 66481 | 16674 | 361.4 | ok |  |
| 5 | a lumbar vertebra | 2db0617aed0d | moonshotai/Kimi-K3.1-Pro | codex | 2117883 | 2039388 | 78495 | 0 | 78495 | 18574 | 963.4 | ok |  |
