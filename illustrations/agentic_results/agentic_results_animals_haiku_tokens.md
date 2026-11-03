Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.4 M input tokens, 48 k output tokens, 5 min 14 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a giraffe | 624cf5636800 | claude-haiku-5-5 | 2409739 | 2299315 | 110424 | 110354 | 70 | 43812 | 314.5 | ok | 3000.0 |
| 2 | an elephant | 62064c14e70d | claude-haiku-5-5 | 1412084 | 1311488 | 100596 | 100550 | 46 | 37727 | 267.1 | ok | 5000.0 |
| 3 | a rhinoceros | 2d72f34a181e | claude-haiku-5-5 | 1382443 | 1292745 | 89698 | 89648 | 50 | 33618 | 248.8 | ok | 100000.0 |
| 4 | a horse | 5f96963c4f2b | claude-haiku-5-5 | 1593002 | 1481589 | 111413 | 111365 | 48 | 48238 | 310.9 | ok | 1000.0 |
| 5 | a crocodile | d0c05863f7b4 | claude-haiku-5-5 | 1065817 | 989745 | 76072 | 76034 | 38 | 34240 | 235.6 | ok |  |
