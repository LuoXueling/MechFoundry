Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 1.5 M input tokens, 23 k output tokens, 5 min 28 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 7117203f0330 | claude-opus-5-5 | 966866 | 883539 | 83327 | 83291 | 36 | 17330 | 235.8 | ok | 2000.0 |
| 2 | an elephant | 82f39e89e2d0 | claude-opus-5-5 | 895161 | 816105 | 79056 | 79022 | 34 | 14970 | 205.0 | ok | 3000.0 |
| 3 | a rhinoceros | 9f81b6cf29d1 | claude-opus-5-5 | 800040 | 722505 | 77535 | 77503 | 32 | 14497 | 220.5 | ok | 3000.0 |
| 4 | a horse | 831002dd369c | claude-opus-5-5 | 891661 | 811782 | 79879 | 79845 | 34 | 15953 | 238.5 | ok | 1000.0 |
| 5 | a crocodile | d5f284defbb1 | claude-opus-5-5 | 1485487 | 1383895 | 101592 | 101546 | 46 | 22514 | 327.8 | ok | 500.0 |
