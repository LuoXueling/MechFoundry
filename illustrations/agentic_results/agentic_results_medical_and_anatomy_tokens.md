Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.5 M input tokens, 20 k output tokens, 7 min 13 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a dental crown | c48ecc61d955 | claude-sonnet-5-5 | 1312339 | 1228095 | 84244 | 84196 | 48 | 13401 | 278.7 | ok | 3000.0 |
| 2 | a knee implant | abfc9322bf45 | claude-sonnet-5-5 | 1395616 | 1309033 | 86583 | 86533 | 50 | 14138 | 375.9 | ok | 5000.0 |
| 3 | a walking cane with a rubber tip | a0f02c2ebe32 | claude-sonnet-5-5 | 1675546 | 1583666 | 91880 | 91822 | 58 | 16129 | 298.1 | ok | 20.0 |
| 4 | a human femur | c1a77b68beaf | claude-sonnet-5-5 | 754334 | 682781 | 71553 | 71521 | 32 | 13647 | 212.3 | ok | 10.0 |
| 5 | a lumbar vertebra | 15ad6b962279 | claude-sonnet-5-5 | 2458410 | 2350636 | 107774 | 107696 | 78 | 20262 | 432.5 | ok | 5000.0 |
