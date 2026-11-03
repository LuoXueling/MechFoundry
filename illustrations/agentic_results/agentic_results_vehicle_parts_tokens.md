Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 3.8 M input tokens, 28 k output tokens, 7 min 52 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a car suspension control arm with rubber bushings | d03344a19339 | claude-sonnet-5-5 | 1575591 | 1483246 | 92345 | 92291 | 54 | 15853 | 272.3 | ok | 10.0 |
| 2 | a truck leaf spring | 8815096b93c8 | claude-sonnet-5-5 | 1448145 | 1368411 | 79734 | 79678 | 56 | 17714 | 227.1 | ok | 1.0 |
| 3 | a car wheel with its rubber tyre | c64de69286fc | claude-sonnet-5-5 | 1907784 | 1808443 | 99341 | 99275 | 66 | 16060 | 421.7 | ok | 20000.0 |
| 4 | a car brake disc | 59f833d18aaa | claude-sonnet-5-5 | 2132804 | 2035221 | 97583 | 97515 | 68 | 21226 | 370.3 | ok | 5000.0 |
| 5 | a crankshaft | 60843d087320 | claude-sonnet-5-5 | 3843563 | 3719921 | 123642 | 123538 | 104 | 27670 | 471.7 | ok | 1000.0 |
