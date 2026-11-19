Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 4.7 M input tokens, 110 k output tokens, 17 min 15 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 758af01c110a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3483429 | 3370240 | 113189 | 0 | 113189 | 54805 | 749.6 | ok | 5000.0 |
| 2 | the Pantheon | 91c2fee7c199 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4723874 | 4584960 | 138914 | 0 | 138914 | 82521 | 840.1 | ok | 5000.0 |
| 3 | Himeji Castle | 2fdaab9245b2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4482484 | 4320640 | 161844 | 0 | 161844 | 105503 | 1035.2 | ok | 500.0 |
| 4 | a natural stone arch | 4df641e26373 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2290353 | 2171392 | 118961 | 0 | 118961 | 57418 | 545.0 | ok | 200000.0 |
| 5 | the Taj Mahal | cc1f28d23b27 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2230019 | 2159872 | 70147 | 0 | 70147 | 60384 | 733.6 | ok |  |
