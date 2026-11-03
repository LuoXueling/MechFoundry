Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.6 M input tokens, 24 k output tokens, 9 min 6 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 71b205af02b2 | claude-sonnet-5-5 | 1833401 | 1741696 | 91705 | 91639 | 66 | 14257 | 380.8 | ok | 5000.0 |
| 2 | the Pantheon | 13539e6f99cf | claude-sonnet-5-5 | 2584187 | 2473915 | 110272 | 110192 | 80 | 24241 | 545.7 | ok | 10000.0 |
| 3 | Himeji Castle | 5b999f37faa9 | claude-sonnet-5-5 | 924571 | 848524 | 76047 | 76009 | 38 | 11481 | 202.9 | ok | 30000.0 |
| 4 | a natural stone arch | 3ca15719c68b | claude-sonnet-5-5 | 1373392 | 1290588 | 82804 | 82752 | 52 | 12171 | 306.8 | ok | 30000.0 |
| 5 | the Taj Mahal | 2c78259d9002 | claude-sonnet-5-5 | 1232493 | 1148430 | 84063 | 84017 | 46 | 13165 | 235.7 | ok | 3000.0 |
