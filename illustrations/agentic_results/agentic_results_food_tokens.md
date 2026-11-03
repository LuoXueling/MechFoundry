Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.9 M input tokens, 19 k output tokens, 8 min 31 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | 283de5cd5937 | claude-sonnet-5-5 | 978922 | 908017 | 70905 | 70863 | 42 | 12427 | 191.5 | ok | 3.0 |
| 2 | a loaf of bread | 0c71d99a58b5 | claude-sonnet-5-5 | 1335031 | 1252945 | 82086 | 82034 | 52 | 13635 | 261.6 | ok | 30.0 |
| 3 | a hamburger | 24c883a572fb | claude-sonnet-5-5 | 2881623 | 2771984 | 109639 | 109555 | 84 | 19026 | 511.2 | ok | 3.0 |
| 4 | a layered birthday cake | ed792af45046 | claude-sonnet-5-5 | 763947 | 692632 | 71315 | 71283 | 32 | 11232 | 167.3 | ok | 20.0 |
| 5 | a slice of watermelon | 89c7cca2de07 | claude-sonnet-5-5 | 1371729 | 1287780 | 83949 | 83897 | 52 | 14528 | 249.4 | ok | 20.0 |
