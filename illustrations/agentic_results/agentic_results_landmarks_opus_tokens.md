Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.4 M input tokens, 27 k output tokens, 8 min 19 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 566b483089a4 | claude-opus-5-5 | 2019884 | 1918755 | 101129 | 101061 | 68 | 23599 | 460.9 | ok | 5000.0 |
| 2 | the Pantheon | 451495907e86 | claude-opus-5-5 | 2350742 | 2242853 | 107889 | 107817 | 72 | 27009 | 488.8 | ok | 20000.0 |
| 3 | Himeji Castle | 284b39c65126 | claude-opus-5-5 | 2246183 | 2133636 | 112547 | 112481 | 66 | 19956 | 499.1 | ok | 10000.0 |
| 4 | a natural stone arch | 6a0218ba407f | claude-opus-5-5 | 1499897 | 1406560 | 93337 | 93287 | 50 | 14026 | 272.8 | ok | 10000.0 |
| 5 | the Taj Mahal | 70159b049582 | claude-opus-5-5 | 862990 | 783718 | 79272 | 79238 | 34 | 16762 | 245.0 | ok | 3000.0 |
