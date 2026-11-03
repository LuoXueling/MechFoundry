Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.6 M input tokens, 46 k output tokens, 7 min 11 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | a3c25e8723aa | claude-haiku-5-5 | 1166827 | 1090606 | 76221 | 76177 | 44 | 23786 | 264.5 | ok | 5.0 |
| 2 | a loaf of bread | 1e0d4fb73c2e | claude-haiku-5-5 | 2552172 | 2438840 | 113332 | 113258 | 74 | 46298 | 431.1 | ok | 10.0 |
| 3 | a hamburger | 716ee895a54b | claude-haiku-5-5 | 1322631 | 1228830 | 93801 | 93757 | 44 | 35099 | 275.6 | ok | 500.0 |
| 4 | a layered birthday cake | 32f66c07230a | claude-haiku-5-5 | 1565899 | 1463930 | 101969 | 101919 | 50 | 42776 | 279.0 | ok | 5.0 |
| 5 | a slice of watermelon | 4cdc48fdacc6 | claude-haiku-5-5 | 1954642 | 1864216 | 90426 | 90362 | 64 | 37229 | 287.4 | ok | 1.0 |
