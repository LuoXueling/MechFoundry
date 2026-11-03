Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 5.1 M input tokens, 81 k output tokens, 11 min 32 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a car suspension control arm with rubber bushings | b77383e1737c | claude-haiku-5-5 | 3503040 | 3363528 | 139512 | 139430 | 82 | 60868 | 443.5 | ok | 20.0 |
| 2 | a truck leaf spring | 23c0e20f48c4 | claude-haiku-5-5 | 5055638 | 4883951 | 171687 | 171583 | 104 | 80922 | 692.2 | ok | 10.0 |
| 3 | a car wheel with its rubber tyre | c279cc8bca77 | claude-haiku-5-5 | 1927840 | 1818089 | 109751 | 109695 | 56 | 40422 | 314.1 | ok | 5000.0 |
| 4 | a car brake disc | 2a6120d9a7c7 | claude-haiku-5-5 | 865815 | 784837 | 80978 | 80944 | 34 | 31594 | 225.4 | ok | 10000.0 |
| 5 | a crankshaft | b22752b09a70 | claude-haiku-5-5 | 2359071 | 2262495 | 96576 | 96506 | 70 | 41943 | 396.7 | ok |  |
