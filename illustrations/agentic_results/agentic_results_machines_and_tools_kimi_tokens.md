Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 5.8 M input tokens, 29 k output tokens, 12 min 11 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a road roller with a steel drum and rubber tyres | 572c45c73d95 | moonshotai/Kimi-K3.1-Pro | codex | 5673951 | 5565446 | 108505 | 0 | 108505 | 29110 | 730.8 | ok |  |
| 2 | a jackhammer | 0047d3143722 | moonshotai/Kimi-K3.1-Pro | codex | 5756684 | 5652039 | 104645 | 0 | 104645 | 23207 | 608.3 | ok | 30000.0 |
| 3 | an industrial robot arm | cf74c713c1c6 | moonshotai/Kimi-K3.1-Pro | codex | 833435 | 768123 | 65312 | 0 | 65312 | 10973 | 217.7 | ok | 5000.0 |
| 4 | an adjustable wrench | d6694645e83a | moonshotai/Kimi-K3.1-Pro | codex | 1328072 | 1247501 | 80571 | 0 | 80571 | 13923 | 291.4 | ok | 100.0 |
| 5 | a claw hammer | 55ace4d99333 | moonshotai/Kimi-K3.1-Pro | codex | 1412557 | 1337396 | 75161 | 0 | 75161 | 12878 | 274.3 | ok | 50.0 |
