Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 6.3 M input tokens, 88 k output tokens, 13 min 35 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 4d61685fabe2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1562455 | 1495296 | 67159 | 0 | 67159 | 31801 | 282.9 | ok | 5000.0 |
| 2 | an elephant | 196d7f843dec | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1654137 | 1572480 | 81657 | 0 | 81657 | 34910 | 339.3 | ok | 200.0 |
| 3 | a rhinoceros | 3dbc42b9fca8 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2092508 | 2014976 | 77532 | 0 | 77532 | 28444 | 295.9 | ok | 200000.0 |
| 4 | a horse | 977c83aff0bc | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1492708 | 1407744 | 84964 | 0 | 84964 | 42018 | 355.2 | ok | 30000.0 |
| 5 | a crocodile | 550607601743 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 6331803 | 6192640 | 139163 | 0 | 139163 | 88239 | 814.9 | ok | 500000.0 |
