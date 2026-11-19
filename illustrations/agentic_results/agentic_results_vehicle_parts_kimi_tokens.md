Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 13 M input tokens, 45 k output tokens, 23 min 1 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a car suspension control arm with rubber bushings | 1015276d2134 | moonshotai/Kimi-K3.1-Pro | codex | 3134585 | 3018799 | 115786 | 0 | 115786 | 27374 | 1373.7 | ok | 20.0 |
| 2 | a truck leaf spring | f54d71e214d1 | moonshotai/Kimi-K3.1-Pro | codex | 13314875 | 12958275 | 356600 | 0 | 356600 | 45249 | 1380.7 | ok | 100.0 |
| 3 | a car wheel with its rubber tyre | 71ae09de8185 | moonshotai/Kimi-K3.1-Pro | codex | 3060570 | 2974330 | 86240 | 0 | 86240 | 16091 | 490.8 | ok | 1000.0 |
| 4 | a car brake disc | 1089687116f4 | moonshotai/Kimi-K3.1-Pro | codex | 1166303 | 1092462 | 73841 | 0 | 73841 | 14399 | 316.0 | ok | 10000.0 |
| 5 | a crankshaft | 019fe1409a1c | moonshotai/Kimi-K3.1-Pro | codex | 4677435 | 4556434 | 121001 | 0 | 121001 | 28770 | 686.6 | ok | 1000.0 |
