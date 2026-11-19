Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 2.2 M input tokens, 55 k output tokens, 8 min 43 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | 2cf6be1bb7cb | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1426210 | 1360512 | 65698 | 0 | 65698 | 33809 | 328.4 | ok | 100.0 |
| 2 | a loaf of bread | d7b47180ec7f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1074580 | 1007104 | 67476 | 0 | 67476 | 35590 | 359.0 | ok | 20.0 |
| 3 | a hamburger | 0b09456ffeb8 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1127395 | 1049856 | 77539 | 0 | 77539 | 44782 | 404.8 | ok | 50.0 |
| 4 | a layered birthday cake | 6e2f03528b5d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1544228 | 1467648 | 76580 | 0 | 76580 | 39782 | 336.5 | ok | 300.0 |
| 5 | a slice of watermelon | 870282f43e95 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2194150 | 2099712 | 94438 | 0 | 94438 | 54577 | 523.2 | ok | 30.0 |
