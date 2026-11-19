Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 5.3 M input tokens, 91 k output tokens, 16 min 54 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | 66c54a4b6668 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2818028 | 2708096 | 109932 | 0 | 109932 | 67189 | 528.9 | ok | 10.0 |
| 2 | a wooden rocking horse | 1c23c4564b5b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2060720 | 1962752 | 97968 | 0 | 97968 | 54215 | 452.4 | ok | 2000.0 |
| 3 | a pogo stick | f7ef7161cc37 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 5258650 | 5118720 | 139930 | 0 | 139930 | 83267 | 965.7 | ok | 1000.0 |
| 4 | a wooden toy train engine | c7a3d6db014b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2567769 | 2455808 | 111961 | 0 | 111961 | 89468 | 728.6 | ok | 500.0 |
| 5 | a yo-yo | a0e602f24696 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 4141499 | 4020736 | 120763 | 0 | 120763 | 90665 | 1014.0 | ok | 200.0 |
