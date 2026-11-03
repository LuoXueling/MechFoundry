Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 4.8 M input tokens, 48 k output tokens, 12 min 33 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a car suspension control arm with rubber bushings | 73e7c5c5a94c | claude-opus-5-5 | 1849766 | 1741425 | 108341 | 108285 | 56 | 22337 | 388.0 | ok | 50.0 |
| 2 | a truck leaf spring | 569d10b3c6e3 | claude-opus-5-5 | 3076303 | 2952617 | 123686 | 123608 | 78 | 33964 | 511.2 | ok | 30.0 |
| 3 | a car wheel with its rubber tyre | abb0e9dafe58 | claude-opus-5-5 | 1951018 | 1845909 | 105109 | 105047 | 62 | 23469 | 580.5 | ok | 5000.0 |
| 4 | a car brake disc | ca5daddc59b4 | claude-opus-5-5 | 2306239 | 2194811 | 111428 | 111360 | 68 | 26207 | 512.0 | ok | 3000.0 |
| 5 | a crankshaft | 84d9272ea75d | claude-opus-5-5 | 4814983 | 4659366 | 155617 | 155507 | 110 | 47901 | 753.3 | ok | 300.0 |
