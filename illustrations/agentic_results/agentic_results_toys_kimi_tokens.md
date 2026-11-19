Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 3.0 M input tokens, 22 k output tokens, 12 min 23 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | ed7b3ec4214f | moonshotai/Kimi-K3.1-Pro | codex | 1664821 | 1583262 | 81559 | 0 | 81559 | 18750 | 449.3 | ok | 100.0 |
| 2 | a wooden rocking horse | 93b215bfedd8 | moonshotai/Kimi-K3.1-Pro | codex | 1951076 | 1880167 | 70909 | 0 | 70909 | 11626 | 364.7 | ok | 2000.0 |
| 3 | a pogo stick | 968d94e3943d | moonshotai/Kimi-K3.1-Pro | codex | 3041625 | 2956674 | 84951 | 0 | 84951 | 21925 | 742.7 | ok |  |
| 4 | a wooden toy train engine | 469d39072c65 | moonshotai/Kimi-K3.1-Pro | codex | 2396795 | 2290474 | 106321 | 0 | 106321 | 16878 | 384.0 | ok | 3000.0 |
| 5 | a yo-yo | 1e551392754c | moonshotai/Kimi-K3.1-Pro | codex | 1049816 | 978063 | 71753 | 0 | 71753 | 10299 | 312.2 | ok | 300000.0 |
