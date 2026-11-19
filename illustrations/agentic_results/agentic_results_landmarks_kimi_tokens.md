Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 2.8 M input tokens, 22 k output tokens, 14 min 37 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 69a840791953 | moonshotai/Kimi-K3.1-Pro | codex | 2199552 | 2053987 | 145565 | 0 | 145565 | 16527 | 877.5 | ok | 10000.0 |
| 2 | the Pantheon | 20b4190e62f3 | moonshotai/Kimi-K3.1-Pro | codex | 2846102 | 2743886 | 102216 | 0 | 102216 | 21883 | 778.0 | ok | 5000.0 |
| 3 | Himeji Castle | 9a9e09b6c4c4 | moonshotai/Kimi-K3.1-Pro | codex | 2415163 | 2318243 | 96920 | 0 | 96920 | 18456 | 530.1 | ok | 100.0 |
| 4 | a natural stone arch | 9abe48a9d42e | moonshotai/Kimi-K3.1-Pro | codex | 2303852 | 2224080 | 79772 | 0 | 79772 | 15372 | 370.1 | ok | 2.0 |
| 5 | the Taj Mahal | 6f61b696607a | moonshotai/Kimi-K3.1-Pro | codex | 1162758 | 1088159 | 74599 | 0 | 74599 | 15072 | 343.7 | ok | 5000.0 |
