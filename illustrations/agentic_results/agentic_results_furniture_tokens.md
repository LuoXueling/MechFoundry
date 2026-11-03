Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.4 M input tokens, 21 k output tokens, 7 min 53 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a school chair with a steel frame and a plastic seat | d2ffbcaa6fed | claude-sonnet-5-5 | 1661533 | 1569190 | 92343 | 92289 | 54 | 14511 | 212.5 | ok | 50.0 |
| 2 | a glass-top dining table | ddb9903ec17c | claude-sonnet-5-5 | 2073382 | 1975520 | 97862 | 97794 | 68 | 17863 | 281.3 | ok | 200.0 |
| 3 | a park bench | 809d3c845254 | claude-sonnet-5-5 | 2419880 | 2305160 | 114720 | 114652 | 68 | 20799 | 472.9 | ok | 300.0 |
| 4 | a bar stool | f598ee8a0160 | claude-sonnet-5-5 | 974021 | 902700 | 71321 | 71279 | 42 | 10478 | 175.2 | ok | 3000.0 |
| 5 | a sofa | ed63e33804b4 | claude-sonnet-5-5 | 1032253 | 953332 | 78921 | 78881 | 40 | 11946 | 180.8 | ok | 5000.0 |
