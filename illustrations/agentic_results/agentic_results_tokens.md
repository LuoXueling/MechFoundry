Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 3.1 M input tokens, 34 k output tokens, 8 min 36 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | an electric toothbrush | a3ddf34a5ebf | claude-opus-5-5 | 1408591 | 1321859 | 86732 | 86682 | 50 | 17842 | 273.8 | ok | 3.0 |
| 2 | a park bench | c09fb246962f | claude-opus-5-5 | 1807065 | 1699095 | 107970 | 107912 | 58 | 25353 | 516.4 | ok | 200.0 |
| 3 | a car suspension control arm with rubber bushings | 73e7c5c5a94c | claude-opus-5-5 | 1849766 | 1741425 | 108341 | 108285 | 56 | 22337 | 388.0 | ok | 50.0 |
| 4 | a truck leaf spring | 569d10b3c6e3 | claude-opus-5-5 | 3076303 | 2952617 | 123686 | 123608 | 78 | 33964 | 511.2 | ok | 30.0 |
| 5 | a claw hammer | e523575eda4e | claude-opus-5-5 | 1505605 | 1415302 | 90303 | 90255 | 48 | 18877 | 263.9 | ok | 10.0 |
| 6 | a human femur | 66871e3cb342 | claude-opus-5-5 | 2098499 | 1997253 | 101246 | 101180 | 66 | 21778 | 458.0 | ok | 20.0 |
