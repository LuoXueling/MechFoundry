Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.9 M input tokens, 23 k output tokens, 7 min 59 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a road roller with a steel drum and rubber tyres | c831b4e13cdf | claude-sonnet-5-5 | 2888967 | 2771720 | 117247 | 117165 | 82 | 23132 | 479.0 | ok | 50000.0 |
| 2 | a jackhammer | a1f39114ab87 | claude-sonnet-5-5 | 2719964 | 2610458 | 109506 | 109424 | 82 | 22928 | 446.8 | ok | 2000.0 |
| 3 | an industrial robot arm | 8b6a7ddd2504 | claude-sonnet-5-5 | 1017635 | 945804 | 71831 | 71789 | 42 | 12411 | 159.7 | ok | 500.0 |
| 4 | an adjustable wrench | 73cca1c05b5e | claude-sonnet-5-5 | 1101051 | 1023963 | 77088 | 77044 | 44 | 13233 | 166.1 | ok | 100.0 |
| 5 | a claw hammer | 22f07ebe054b | claude-sonnet-5-5 | 1409271 | 1320760 | 88511 | 88461 | 50 | 15518 | 216.8 | ok | 30.0 |
