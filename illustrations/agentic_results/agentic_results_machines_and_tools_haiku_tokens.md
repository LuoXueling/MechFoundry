Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 5.7 M input tokens, 84 k output tokens, 11 min 42 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a road roller with a steel drum and rubber tyres | fa69821708e9 | claude-haiku-5-5 | 5674446 | 5512264 | 162182 | 162070 | 112 | 83675 | 651.3 | ok | 2000.0 |
| 2 | a jackhammer | f683baba82f4 | claude-haiku-5-5 | 5205108 | 5046225 | 158883 | 158777 | 106 | 69080 | 702.0 | ok | 500000.0 |
| 3 | an industrial robot arm | beeb176cdcb5 | claude-haiku-5-5 | 1458777 | 1356634 | 102143 | 102097 | 46 | 42384 | 266.5 | ok | 3000.0 |
| 4 | an adjustable wrench | cf5ceba6da28 | claude-haiku-5-5 | 1750366 | 1647732 | 102634 | 102580 | 54 | 43669 | 271.4 | ok | 50.0 |
| 5 | a claw hammer | 4553fef600b2 | claude-haiku-5-5 | 3948743 | 3799591 | 149152 | 149066 | 86 | 63534 | 553.0 | ok | 300.0 |
