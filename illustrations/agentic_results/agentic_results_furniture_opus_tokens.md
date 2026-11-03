Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.5 M input tokens, 25 k output tokens, 8 min 36 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a school chair with a steel frame and a plastic seat | c90e732be96e | claude-opus-5-5 | 2475914 | 2361128 | 114786 | 114714 | 72 | 22474 | 423.8 | ok | 30.0 |
| 2 | a glass-top dining table | 9e19e47e3168 | claude-opus-5-5 | 1599488 | 1497648 | 101840 | 101790 | 50 | 23542 | 359.2 | ok | 1000.0 |
| 3 | a park bench | c09fb246962f | claude-opus-5-5 | 1807065 | 1699095 | 107970 | 107912 | 58 | 25353 | 516.4 | ok | 200.0 |
| 4 | a bar stool | fae04d03ee0a | claude-opus-5-5 | 843007 | 761085 | 81922 | 81890 | 32 | 15831 | 237.3 | ok | 500.0 |
| 5 | a sofa | 03af3db6b4df | claude-opus-5-5 | 1003070 | 916998 | 86072 | 86036 | 36 | 15871 | 243.6 | ok | 5000.0 |
