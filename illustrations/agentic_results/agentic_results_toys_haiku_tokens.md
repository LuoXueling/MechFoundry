Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 5.7 M input tokens, 78 k output tokens, 8 min 57 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | caa2f7f93fcb | claude-haiku-5-5 | 1548452 | 1462454 | 85998 | 85940 | 58 | 28904 | 255.9 | ok | 300.0 |
| 2 | a wooden rocking horse | a77422267b48 | claude-haiku-5-5 | 2211301 | 2097803 | 113498 | 113434 | 64 | 50202 | 354.0 | ok | 10000.0 |
| 3 | a pogo stick | 5a2622107f6d | claude-haiku-5-5 | 5652700 | 5517392 | 135308 | 135178 | 130 | 77883 | 536.7 | ok |  |
| 4 | a wooden toy train engine | 7403ed98cfe3 | claude-haiku-5-5 | 1950282 | 1835126 | 115156 | 115102 | 54 | 51031 | 364.1 | ok | 10000.0 |
| 5 | a yo-yo | e260972b7483 | claude-haiku-5-5 | 1754728 | 1641939 | 112789 | 112739 | 50 | 50473 | 425.7 | ok | 50000.0 |
