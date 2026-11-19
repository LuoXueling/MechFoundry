Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.3 M input tokens, 67 k output tokens, 10 min 37 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | f5a9ec90d5ac | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2881416 | 2782592 | 98824 | 0 | 98824 | 67371 | 637.3 | ok |  |
| 2 | a thatched stone cottage | ae83e6cf8d51 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3335189 | 3232768 | 102421 | 0 | 102421 | 55978 | 565.6 | ok | 50000.0 |
| 3 | a lighthouse | 1c64d7271858 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1185852 | 1122048 | 63804 | 0 | 63804 | 25550 | 257.6 | ok | 30000.0 |
| 4 | an igloo | 530058c40b43 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1617324 | 1554176 | 63148 | 0 | 63148 | 27721 | 297.2 | ok | 1000.0 |
| 5 | a log cabin | 54bdfe167b70 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1248837 | 1183232 | 65605 | 0 | 65605 | 21567 | 256.1 | ok | 500000.0 |
