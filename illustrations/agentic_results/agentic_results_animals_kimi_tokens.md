Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.0 M input tokens, 19 k output tokens, 9 min 15 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 144d93800057 | moonshotai/Kimi-K3.1-Pro | codex | 2233284 | 2143764 | 89520 | 0 | 89520 | 18112 | 407.1 | ok | 50.0 |
| 2 | an elephant | b5aff75e0cf0 | moonshotai/Kimi-K3.1-Pro | codex | 2978428 | 2890127 | 88301 | 0 | 88301 | 15342 | 433.0 | ok | 10000.0 |
| 3 | a rhinoceros | e9e6d1ee6efb | moonshotai/Kimi-K3.1-Pro | codex | 1690582 | 1612992 | 77590 | 0 | 77590 | 15832 | 316.5 | ok | 50000.0 |
| 4 | a horse | f443de3b382b | moonshotai/Kimi-K3.1-Pro | codex | 1778810 | 1699083 | 79727 | 0 | 79727 | 12610 | 286.6 | ok | 2000.0 |
| 5 | a crocodile | 8cc7cff992a7 | moonshotai/Kimi-K3.1-Pro | codex | 2503325 | 2392670 | 110655 | 0 | 110655 | 19217 | 555.5 | ok | 100.0 |
