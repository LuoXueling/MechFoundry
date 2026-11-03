Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.3 M input tokens, 22 k output tokens, 7 min 17 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | an umbrella | 921926428397 | claude-sonnet-5-5 | 1758450 | 1663041 | 95409 | 95349 | 60 | 16008 | 398.7 | ok | 30.0 |
| 2 | a mine headframe | 357345d7d714 | claude-sonnet-5-5 | 2075355 | 1979595 | 95760 | 95692 | 68 | 16826 | 397.3 | ok | 20000.0 |
| 3 | an anglepoise desk lamp | f9989653fc05 | claude-sonnet-5-5 | 2261168 | 2163030 | 98138 | 98070 | 68 | 22250 | 313.7 | ok | 10.0 |
| 4 | a steel jerry can | 575e9c3125f0 | claude-sonnet-5-5 | 1250485 | 1168183 | 82302 | 82256 | 46 | 15958 | 227.5 | ok | 2000000.0 |
| 5 | an excavator arm | 039bb1f5f699 | claude-sonnet-5-5 | 1301111 | 1225625 | 75486 | 75438 | 48 | 13812 | 212.3 | ok | 30.0 |
| 6 | a baby pram | 853c06662ab4 | claude-sonnet-5-5 | 1603359 | 1507538 | 95821 | 95769 | 52 | 20323 | 266.7 | ok | 500.0 |
| 7 | a three-panel folding screen | 965d383570d7 | claude-sonnet-5-5 | 1249321 | 1164670 | 84651 | 84605 | 46 | 17056 | 257.4 | ok | 10000.0 |
| 8 | a balloon whisk | 094db6735541 | claude-sonnet-5-5 | 1870163 | 1773858 | 96305 | 96241 | 64 | 15227 | 358.4 | ok | 500.0 |
| 9 | a ferris wheel | 948a0bd1713c | claude-sonnet-5-5 | 1669152 | 1584844 | 84308 | 84248 | 60 | 17348 | 436.9 | ok |  |
| 10 | a rotary clothes line | 1c7186e2d3e9 | claude-sonnet-5-5 | 587992 | 541016 | 46976 | 46944 | 32 | 13188 | 157.2 | ok |  |
