Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 6.9 M input tokens, 88 k output tokens, 13 min 3 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a school chair with a steel frame and a plastic seat | dbc9c1c4554a | claude-haiku-5-5 | 3629792 | 3488480 | 141312 | 141230 | 82 | 67850 | 542.9 | ok | 50.0 |
| 2 | a glass-top dining table | 6cbd9d366139 | claude-haiku-5-5 | 3118628 | 2990575 | 128053 | 127975 | 78 | 56549 | 439.5 | ok | 500.0 |
| 3 | a park bench | d6068d225c0a | claude-haiku-5-5 | 6913137 | 6727589 | 185548 | 185412 | 136 | 87586 | 783.3 | ok | 2000.0 |
| 4 | a bar stool | 70a929c05373 | claude-haiku-5-5 | 1045727 | 976764 | 68963 | 68919 | 44 | 19850 | 163.7 | ok | 5000.0 |
| 5 | a sofa | 2ccc44aeb8e5 | claude-haiku-5-5 | 1954386 | 1837887 | 116499 | 116447 | 52 | 46543 | 348.9 | ok | 5000.0 |
