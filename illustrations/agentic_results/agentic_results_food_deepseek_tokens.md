Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 4.3 M input tokens, 68 k output tokens, 16 min 12 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a jelly dessert | dadf067b768e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3018130 | 2926976 | 91154 | 0 | 91154 | 46453 | 479.6 | ok | 10.0 |
| 2 | a loaf of bread | 2f4b91dbf96a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1833236 | 1751040 | 82196 | 0 | 82196 | 40947 | 443.2 | ok | 10.0 |
| 3 | a hamburger | cfb71664cc4e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4328277 | 4216192 | 112085 | 0 | 112085 | 67508 | 972.1 | ok | 2.0 |
| 4 | a layered birthday cake | 607251581e18 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1039973 | 982016 | 57957 | 0 | 57957 | 24976 | 270.3 | ok | 30.0 |
| 5 | a slice of watermelon | bcd8ef51b43a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2526653 | 2433536 | 93117 | 0 | 93117 | 50449 | 431.1 | ok | 2.0 |
