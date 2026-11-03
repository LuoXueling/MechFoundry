Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 1.1 M input tokens, 13 k output tokens, 2 min 52 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 69ecefce4461 | claude-sonnet-5-5 | 921640 | 851229 | 70411 | 70371 | 40 | 11438 | 159.2 | ok | 2000.0 |
| 2 | an elephant | 419757786162 | claude-sonnet-5-5 | 1143585 | 1066764 | 76821 | 76775 | 46 | 12804 | 169.0 | ok | 300.0 |
| 3 | a rhinoceros | d4658181bd0e | claude-sonnet-5-5 | 1149260 | 1072164 | 77096 | 77048 | 48 | 13004 | 172.4 | ok | 10000.0 |
| 4 | a horse | 68bd6be06fac | claude-sonnet-5-5 | 936269 | 859363 | 76906 | 76868 | 38 | 12437 | 170.0 | ok | 5000.0 |
| 5 | a crocodile | 4afab00557b8 | claude-sonnet-5-5 | 1105535 | 1032592 | 72943 | 72897 | 46 | 12986 | 170.5 | ok | 50.0 |
