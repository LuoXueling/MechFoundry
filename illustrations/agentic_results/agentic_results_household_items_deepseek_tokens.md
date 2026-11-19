Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.0 M input tokens, 45 k output tokens, 7 min 12 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 075c5e69e9fb | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3038530 | 2948992 | 89538 | 0 | 89538 | 44747 | 432.3 | ok | 200.0 |
| 2 | a USB stick | d79757aed64e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1263276 | 1199104 | 64172 | 0 | 64172 | 27457 | 264.1 | ok | 10000.0 |
| 3 | an electric toothbrush | 9d84135ca26d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1304085 | 1235584 | 68501 | 0 | 68501 | 26313 | 274.9 | ok | 5.0 |
| 4 | a coffee mug | 2d1e773f2e4d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1476903 | 1416832 | 60071 | 0 | 60071 | 24659 | 241.3 | ok | 50000.0 |
| 5 | a computer mouse | 25f161931f83 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1279077 | 1211264 | 67813 | 0 | 67813 | 23400 | 229.6 | ok | 200000.0 |
