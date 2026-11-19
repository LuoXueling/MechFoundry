Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.0 M input tokens, 45 k output tokens, 9 min 41 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 0e8bd9197100 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2312162 | 2222080 | 90082 | 0 | 90082 | 34374 | 526.2 | ok | 10000.0 |
| 2 | the Pantheon | 73d2ea4e2c5f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3036738 | 2946816 | 89922 | 0 | 89922 | 45218 | 581.2 | ok | 5000.0 |
| 3 | Himeji Castle | 732eca44dd9d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1513946 | 1447936 | 66010 | 0 | 66010 | 31771 | 339.5 | ok | 20000.0 |
| 4 | a natural stone arch | 43078b7b2f7f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1427666 | 1354368 | 73298 | 0 | 73298 | 27638 | 334.1 | ok | 50000.0 |
| 5 | the Taj Mahal | 2212b1dc44c6 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1994990 | 1913088 | 81902 | 0 | 81902 | 37219 | 488.4 | ok | 30.0 |
