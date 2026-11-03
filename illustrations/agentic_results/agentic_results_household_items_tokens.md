Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 1.6 M input tokens, 15 k output tokens, 3 min 44 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 1aae26b57dc6 | claude-sonnet-5-5 | 949055 | 866997 | 82058 | 82020 | 38 | 11898 | 206.7 | ok | 300.0 |
| 2 | a USB stick | 882377636038 | claude-sonnet-5-5 | 1571073 | 1484220 | 86853 | 86797 | 56 | 15444 | 224.1 | ok | 5.0 |
| 3 | an electric toothbrush | a108fe4244d9 | claude-sonnet-5-5 | 1229698 | 1148085 | 81613 | 81567 | 46 | 12149 | 197.6 | ok | 10.0 |
| 4 | a coffee mug | cb9c0c364b61 | claude-sonnet-5-5 | 860670 | 783699 | 76971 | 76937 | 34 | 11461 | 169.1 | ok | 3000.0 |
| 5 | a computer mouse | ee62da33a70a | claude-sonnet-5-5 | 742464 | 674406 | 68058 | 68026 | 32 | 10087 | 165.6 | ok | 100000.0 |
