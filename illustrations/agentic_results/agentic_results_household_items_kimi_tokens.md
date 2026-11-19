Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 2.3 M input tokens, 20 k output tokens, 12 min 32 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 28a47d5cf673 | moonshotai/Kimi-K3.1-Pro | codex | 2264100 | 2171851 | 92249 | 0 | 92249 | 16100 | 752.2 | ok | 500.0 |
| 2 | a USB stick | 64d60bc00ae7 | moonshotai/Kimi-K3.1-Pro | codex | 1531863 | 1440295 | 91568 | 0 | 91568 | 19724 | 410.7 | ok | 5000.0 |
| 3 | an electric toothbrush | 6a4492275165 | moonshotai/Kimi-K3.1-Pro | codex | 1265122 | 1191262 | 73860 | 0 | 73860 | 13627 | 330.8 | ok | 20.0 |
| 4 | a coffee mug | 49c3269dd227 | moonshotai/Kimi-K3.1-Pro | codex | 937740 | 872272 | 65468 | 0 | 65468 | 11450 | 277.5 | ok | 50000.0 |
| 5 | a computer mouse | a6129a879734 | moonshotai/Kimi-K3.1-Pro | codex | 1595670 | 1516342 | 79328 | 0 | 79328 | 15108 | 474.2 | ok | 1000.0 |
