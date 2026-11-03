Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 7.4 M input tokens, 75 k output tokens, 12 min 28 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 977fd33385ea | claude-haiku-5-5 | 2188898 | 2081871 | 107027 | 106967 | 60 | 47209 | 370.3 | ok | 2000.0 |
| 2 | a USB stick | b042285e3ce7 | claude-haiku-5-5 | 7443139 | 7281490 | 161649 | 161491 | 158 | 75333 | 747.6 | ok | 1000.0 |
| 3 | an electric toothbrush | 33fcad516671 | claude-haiku-5-5 | 1312260 | 1213338 | 98922 | 98878 | 44 | 42473 | 284.1 | ok | 10.0 |
| 4 | a coffee mug | e58a4dca06fe | claude-haiku-5-5 | 1528816 | 1435877 | 92939 | 92889 | 50 | 34778 | 252.1 | ok | 50000.0 |
| 5 | a computer mouse | ebb2cbe4ae90 | claude-haiku-5-5 | 1603674 | 1510535 | 93139 | 93089 | 50 | 31198 | 252.9 | ok | 30000.0 |
