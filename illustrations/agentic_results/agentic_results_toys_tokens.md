Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 3.4 M input tokens, 27 k output tokens, 8 min 8 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | dca313df00a5 | claude-sonnet-5-5 | 870987 | 799929 | 71058 | 71022 | 36 | 10246 | 152.3 | ok | 20.0 |
| 2 | a wooden rocking horse | 4464ff4b071a | claude-sonnet-5-5 | 746166 | 672742 | 73424 | 73392 | 32 | 10828 | 142.7 | ok | 30000.0 |
| 3 | a pogo stick | fbc36e9abb60 | claude-sonnet-5-5 | 3427235 | 3307666 | 119569 | 119473 | 96 | 27470 | 488.1 | ok | 500.0 |
| 4 | a wooden toy train engine | 843b418bdbdc | claude-sonnet-5-5 | 1152036 | 1072260 | 79776 | 79732 | 44 | 12525 | 173.4 | ok | 200000.0 |
| 5 | a yo-yo | 812e2d37f771 | claude-sonnet-5-5 | 1997553 | 1896117 | 101436 | 101376 | 60 | 19018 | 332.2 | ok | 10000.0 |
