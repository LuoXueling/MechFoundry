Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.8 M input tokens, 64 k output tokens, 9 min 13 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a skyscraper | f8ffbcd92e23 | claude-haiku-5-5 | 2375786 | 2268780 | 107006 | 106936 | 70 | 44167 | 454.9 | ok | 1000.0 |
| 2 | a thatched stone cottage | 80e6344274dc | claude-haiku-5-5 | 2801062 | 2676289 | 124773 | 124709 | 64 | 64208 | 553.3 | ok |  |
| 3 | a lighthouse | 64b1430329d6 | claude-haiku-5-5 | 1800206 | 1696021 | 104185 | 104129 | 56 | 43806 | 278.3 | ok | 20000.0 |
| 4 | an igloo | 84500ed423bb | claude-haiku-5-5 | 1216876 | 1125889 | 90987 | 90943 | 44 | 34460 | 252.9 | ok | 10000.0 |
| 5 | a log cabin | 4fc429bc83ce | claude-haiku-5-5 | 1735496 | 1629228 | 106268 | 106212 | 56 | 41302 | 356.1 | ok | 500.0 |
