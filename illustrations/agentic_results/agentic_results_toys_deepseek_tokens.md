Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 4.0 M input tokens, 72 k output tokens, 11 min 51 s.

| row | instruction | session_id | model | agent | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | 05dd0f9f80d9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1559778 | 1494912 | 64866 | 0 | 64866 | 32072 | 338.6 | ok | 2000.0 |
| 2 | a wooden rocking horse | 9677deab9824 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1262218 | 1193216 | 69002 | 0 | 69002 | 32096 | 324.6 | ok | 10000.0 |
| 3 | a pogo stick | 895fec3a9082 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 3957576 | 3837184 | 120392 | 0 | 120392 | 71694 | 711.2 | ok | 10000.0 |
| 4 | a wooden toy train engine | 451b8d9f9ef2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 1696355 | 1618432 | 77923 | 0 | 77923 | 27193 | 339.8 | ok | 5000.0 |
| 5 | a yo-yo | 8a00a42c4036 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | 2774917 | 2695168 | 79749 | 0 | 79749 | 52537 | 559.8 | ok | 50000.0 |
