Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 4.9 M input tokens, 79 k output tokens, 12 min 22 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | the Leaning Tower of Pisa | 590fcfcc5017 | claude-haiku-5-5 | 4117906 | 3990917 | 126989 | 126885 | 104 | 52414 | 540.1 | ok | 10000.0 |
| 2 | the Pantheon | c13b3dfbc53c | claude-haiku-5-5 | 4749596 | 4586640 | 162956 | 162852 | 104 | 78570 | 741.6 | ok | 20000.0 |
| 3 | Himeji Castle | 199a6696c020 | claude-haiku-5-5 | 4853958 | 4705797 | 148161 | 148049 | 112 | 51652 | 713.7 | ok | 30000.0 |
| 4 | a natural stone arch | 834670958931 | claude-haiku-5-5 | 1715785 | 1620580 | 95205 | 95151 | 54 | 35298 | 282.4 | ok | 30000.0 |
| 5 | the Taj Mahal | ad2c7bf9dedf | claude-haiku-5-5 | 2060361 | 1946447 | 113914 | 113856 | 58 | 48982 | 406.2 | ok | 5000.0 |
