Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 4.4 M input tokens, 98 k output tokens, 14 min 0 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | 9d069b762ac9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2674799 | 2588416 | 86383 | 0 | 86383 | 44473 | 556.9 | ok | 1000.0 |
| 2 | a knee implant | 27402b75ebc0 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4265538 | 4138624 | 126914 | 0 | 126914 | 70490 | 805.0 | ok | 500000.0 |
| 3 | a walking cane with a rubber tip | 7496ab033876 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4401944 | 4269952 | 131992 | 0 | 131992 | 98333 | 840.4 | ok | 1000.0 |
| 4 | a human femur | 5c2cf66635c9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1442506 | 1363968 | 78538 | 0 | 78538 | 37249 | 354.1 | ok | 1.0 |
| 5 | a lumbar vertebra | e66be1f0b63c | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3522398 | 3394048 | 128350 | 0 | 128350 | 55382 | 688.6 | ok | 3.0 |
