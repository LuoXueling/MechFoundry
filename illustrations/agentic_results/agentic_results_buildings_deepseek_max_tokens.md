Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 5.0 M input tokens, 100 k output tokens, 16 min 5 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | 2165fcdf1490 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3301735 | 3195648 | 106087 | 0 | 106087 | 64912 | 525.1 | ok | 2000.0 |
| 2 | a thatched stone cottage | 7a2267f7ca53 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2356461 | 2257792 | 98669 | 0 | 98669 | 81974 | 647.6 | ok | 200000.0 |
| 3 | a lighthouse | 4b4e0a9a04d2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 5043531 | 4906112 | 137419 | 0 | 137419 | 83987 | 964.8 | ok | 5000.0 |
| 4 | an igloo | 3dd222ad8493 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3601222 | 3472384 | 128838 | 0 | 128838 | 100271 | 794.1 | ok | 5000.0 |
| 5 | a log cabin | c7c0977965aa | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1900286 | 1793664 | 106622 | 0 | 106622 | 87271 | 653.1 | ok | 100000.0 |
