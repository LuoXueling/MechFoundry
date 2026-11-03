Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.5 M input tokens, 28 k output tokens, 9 min 5 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | 9088600e6b55 | claude-opus-5-5 | 1629513 | 1539498 | 90015 | 89959 | 56 | 16812 | 313.3 | ok | 50.0 |
| 2 | a loaf of bread | a798f81096db | claude-opus-5-5 | 1871587 | 1767201 | 104386 | 104328 | 58 | 22347 | 400.5 | ok | 3.0 |
| 3 | a hamburger | 38db33dd8f0e | claude-opus-5-5 | 2116299 | 2013203 | 103096 | 103036 | 60 | 24202 | 465.5 | ok | 1.0 |
| 4 | a layered birthday cake | bf1d591ef934 | claude-opus-5-5 | 955807 | 874587 | 81220 | 81184 | 36 | 16999 | 245.1 | ok | 5.0 |
| 5 | a slice of watermelon | 880acc4a508a | claude-opus-5-5 | 2486016 | 2369203 | 116813 | 116743 | 70 | 27897 | 544.7 | ok | 30.0 |
