Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 5.7 M input tokens, 23 k output tokens, 11 min 19 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a school chair with a steel frame and a plastic seat | 84a54143155c | moonshotai/Kimi-K3.1-Pro | codex | 5654264 | 5538868 | 115396 | 0 | 115396 | 23024 | 678.9 | ok | 50.0 |
| 2 | a glass-top dining table | 97b3916bd322 | moonshotai/Kimi-K3.1-Pro | codex | 1873869 | 1792893 | 80976 | 0 | 80976 | 16084 | 367.2 | ok | 500.0 |
| 3 | a park bench | 921a1ece04db | moonshotai/Kimi-K3.1-Pro | codex | 5027137 | 4922838 | 104299 | 0 | 104299 | 21383 | 630.2 | ok | 300.0 |
| 4 | a bar stool | df0f348be915 | moonshotai/Kimi-K3.1-Pro | codex | 2904557 | 2802917 | 101640 | 0 | 101640 | 19957 | 606.9 | ok | 30000.0 |
| 5 | a sofa | 5f29d2fd0321 | moonshotai/Kimi-K3.1-Pro | codex | 1655647 | 1577336 | 78311 | 0 | 78311 | 14390 | 338.4 | ok | 5000.0 |
