Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.3 M input tokens, 20 k output tokens, 7 min 34 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | 9d1a169f7ab9 | claude-sonnet-5-5 | 2255662 | 2153879 | 101783 | 101713 | 70 | 20357 | 350.1 | ok | 1000.0 |
| 2 | a thatched stone cottage | 4a55ebcb13cb | claude-sonnet-5-5 | 2072736 | 1970545 | 102191 | 102123 | 68 | 17771 | 454.0 | ok | 50000.0 |
| 3 | a lighthouse | 5eb7d1ae8d34 | claude-sonnet-5-5 | 1616623 | 1528778 | 87845 | 87785 | 60 | 15238 | 334.6 | ok | 5000.0 |
| 4 | an igloo | fc81ce53a534 | claude-sonnet-5-5 | 1165873 | 1077310 | 88563 | 88521 | 42 | 14170 | 422.0 | ok | 1000.0 |
| 5 | a log cabin | d7a3ffa841cc | claude-sonnet-5-5 | 2019259 | 1919639 | 99620 | 99552 | 68 | 16314 | 322.7 | ok | 20000.0 |
