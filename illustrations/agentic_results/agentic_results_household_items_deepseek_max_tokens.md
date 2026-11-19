Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.5 M input tokens, 95 k output tokens, 12 min 33 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | dca76f0519f4 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2383693 | 2297088 | 86605 | 0 | 86605 | 43563 | 463.8 | ok | 1000.0 |
| 2 | a USB stick | d49e5afc145f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3464105 | 3189376 | 274729 | 0 | 274729 | 95195 | 752.8 | ok | 100.0 |
| 3 | an electric toothbrush | 34294d1829a4 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2265952 | 2069888 | 196064 | 0 | 196064 | 67491 | 592.3 | ok | 5.0 |
| 4 | a coffee mug | ce53e61c7cbc | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1003591 | 816000 | 187591 | 0 | 187591 | 41358 | 336.7 | ok | 100000.0 |
| 5 | a computer mouse | 1af18cb3a58b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1402257 | 1163648 | 238609 | 0 | 238609 | 55848 | 454.8 | ok | 200000.0 |
