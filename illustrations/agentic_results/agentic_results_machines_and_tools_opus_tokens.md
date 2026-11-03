Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 7.2 M input tokens, 70 k output tokens, 19 min 16 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a road roller with a steel drum and rubber tyres | 03468728e8f2 | claude-opus-5-5 | 7220676 | 7033514 | 187162 | 187026 | 136 | 69985 | 1156.1 | ok | 20000.0 |
| 2 | a jackhammer | 255401db7d1e | claude-opus-5-5 | 3159490 | 3036601 | 122889 | 122803 | 86 | 27947 | 551.6 | ok | 2000.0 |
| 3 | an industrial robot arm | 4d0027d7fd41 | claude-opus-5-5 | 1582421 | 1489119 | 93302 | 93248 | 54 | 19121 | 337.3 | ok | 5000.0 |
| 4 | an adjustable wrench | 06076fc07e03 | claude-opus-5-5 | 1433491 | 1338698 | 94793 | 94747 | 46 | 18945 | 250.0 | ok | 100.0 |
| 5 | a claw hammer | e523575eda4e | claude-opus-5-5 | 1505605 | 1415302 | 90303 | 90255 | 48 | 18877 | 263.9 | ok | 10.0 |
