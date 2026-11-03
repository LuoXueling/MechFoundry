Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 1.4 M input tokens, 18 k output tokens, 4 min 34 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | e045ee5e9396 | claude-opus-5-5 | 1014605 | 942098 | 72507 | 72469 | 38 | 15110 | 272.4 | ok | 1000.0 |
| 2 | a USB stick | 1853ba6620f0 | claude-opus-5-5 | 1294273 | 1206812 | 87461 | 87417 | 44 | 16005 | 227.9 | ok | 3.0 |
| 3 | an electric toothbrush | a3ddf34a5ebf | claude-opus-5-5 | 1408591 | 1321859 | 86732 | 86682 | 50 | 17842 | 273.8 | ok | 3.0 |
| 4 | a coffee mug | 2c96c95ab93a | claude-opus-5-5 | 812014 | 737812 | 74202 | 74170 | 32 | 13442 | 211.0 | ok | 50000.0 |
| 5 | a computer mouse | fb8d771bf93e | claude-opus-5-5 | 956384 | 873777 | 82607 | 82571 | 36 | 18219 | 247.1 | ok | 10000.0 |
