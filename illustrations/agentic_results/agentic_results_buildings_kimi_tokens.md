Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 8.8 M input tokens, 30 k output tokens, 25 min 8 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | 8c507802b1ac | moonshotai/Kimi-K3.1-Pro | codex | 3208537 | 3107087 | 101450 | 0 | 101450 | 17097 | 830.8 | ok | 5000.0 |
| 2 | a thatched stone cottage | 8dc4ddae0e3a | moonshotai/Kimi-K3.1-Pro | codex | 1862091 | 1786823 | 75268 | 0 | 75268 | 13395 | 556.3 | ok | 30000.0 |
| 3 | a lighthouse | c80aa15ac6e2 | moonshotai/Kimi-K3.1-Pro | codex | 4088152 | 3983497 | 104655 | 0 | 104655 | 21413 | 1076.3 | ok | 2000.0 |
| 4 | an igloo | 508df455a189 | moonshotai/Kimi-K3.1-Pro | codex | 8776686 | 8644146 | 132540 | 0 | 132540 | 30073 | 1508.2 | ok | 3000.0 |
| 5 | a log cabin | bb6e1db33309 | moonshotai/Kimi-K3.1-Pro | codex | 2990968 | 2899503 | 91465 | 0 | 91465 | 19370 | 966.5 | ok | 10000.0 |
