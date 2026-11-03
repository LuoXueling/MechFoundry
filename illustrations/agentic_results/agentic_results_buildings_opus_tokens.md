Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.6 M input tokens, 22 k output tokens, 8 min 6 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | 9740276dd004 | claude-opus-5-5 | 1140306 | 1056858 | 83448 | 83406 | 42 | 15855 | 220.1 | ok | 10000.0 |
| 2 | a thatched stone cottage | b2778688a74c | claude-opus-5-5 | 1281098 | 1187488 | 93610 | 93568 | 42 | 19120 | 287.7 | ok | 1000000.0 |
| 3 | a lighthouse | 4018fb655d42 | claude-opus-5-5 | 2020356 | 1915129 | 105227 | 105161 | 66 | 21673 | 429.0 | ok | 30000.0 |
| 4 | an igloo | 0f955499102d | claude-opus-5-5 | 2579046 | 2471703 | 107343 | 107267 | 76 | 20577 | 486.2 | ok | 1000.0 |
| 5 | a log cabin | 2662a8a839a1 | claude-opus-5-5 | 759416 | 681327 | 78089 | 78059 | 30 | 15049 | 227.3 | ok | 100000.0 |
