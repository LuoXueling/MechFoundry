Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.5 M input tokens, 20 k output tokens, 16 min 39 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | 2b5b79392575 | moonshotai/Kimi-K3.1-Pro | codex | 3140638 | 3045935 | 94703 | 0 | 94703 | 16123 | 533.6 | ok | 20.0 |
| 2 | a loaf of bread | c43b41af2dda | moonshotai/Kimi-K3.1-Pro | codex | 1432114 | 1360807 | 71307 | 0 | 71307 | 13639 | 389.6 | ok | 50.0 |
| 3 | a hamburger | a9d778947315 | moonshotai/Kimi-K3.1-Pro | codex | 2956365 | 2872325 | 84040 | 0 | 84040 | 17462 | 998.7 | ok |  |
| 4 | a layered birthday cake | e49f68ed7fbf | moonshotai/Kimi-K3.1-Pro | codex | 627679 | 567187 | 60492 | 0 | 60492 | 11318 | 252.0 | ok | 20.0 |
| 5 | a slice of watermelon | 259a7a30d6e5 | moonshotai/Kimi-K3.1-Pro | codex | 3498939 | 3381537 | 117402 | 0 | 117402 | 20422 | 735.9 | ok | 1.0 |
