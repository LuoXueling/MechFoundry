Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 6.0 M input tokens, 84 k output tokens, 10 min 12 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | 9a08c9ef8bf5 | claude-haiku-5-5 | 2088821 | 1988369 | 100452 | 100388 | 64 | 35454 | 339.7 | ok | 2000.0 |
| 2 | a knee implant | a3cfe82fec65 | claude-haiku-5-5 | 6016784 | 5847511 | 169273 | 169157 | 116 | 83578 | 608.0 | ok | 300000.0 |
| 3 | a walking cane with a rubber tip | 0a395bb30de1 | claude-haiku-5-5 | 3094530 | 2991802 | 102728 | 102648 | 80 | 44294 | 337.4 | ok |  |
| 4 | a human femur | 1d55dafacb66 | claude-haiku-5-5 | 1628367 | 1517907 | 110460 | 110414 | 46 | 50563 | 291.5 | ok | 200.0 |
| 5 | a lumbar vertebra | 46b4311a3ba3 | claude-haiku-5-5 | 3687428 | 3560644 | 126784 | 126690 | 94 | 48660 | 612.4 | ok | 500.0 |
