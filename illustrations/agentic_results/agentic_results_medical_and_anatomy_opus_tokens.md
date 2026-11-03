Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 3.7 M input tokens, 32 k output tokens, 10 min 29 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | 19e2bc98c966 | claude-opus-5-5 | 1952776 | 1852186 | 100590 | 100530 | 60 | 20799 | 480.2 | ok | 2000.0 |
| 2 | a knee implant | dab454dc286c | claude-opus-5-5 | 3597108 | 3462293 | 134815 | 134719 | 96 | 32092 | 628.6 | ok | 20000.0 |
| 3 | a walking cane with a rubber tip | 7a707bc2b7be | claude-opus-5-5 | 3745920 | 3615433 | 130487 | 130389 | 98 | 28698 | 550.6 | ok | 500.0 |
| 4 | a human femur | 66871e3cb342 | claude-opus-5-5 | 2098499 | 1997253 | 101246 | 101180 | 66 | 21778 | 458.0 | ok | 20.0 |
| 5 | a lumbar vertebra | f37f73faaa4a | claude-opus-5-5 | 2926024 | 2812374 | 113650 | 113568 | 82 | 23669 | 556.3 | ok | 200.0 |
