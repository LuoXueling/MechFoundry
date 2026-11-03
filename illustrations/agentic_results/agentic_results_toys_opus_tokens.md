Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 2.7 M input tokens, 26 k output tokens, 7 min 48 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a teddy bear | f70ceee7ad2d | claude-opus-5-5 | 1024656 | 944869 | 79787 | 79747 | 40 | 17610 | 241.5 | ok | 5.0 |
| 2 | a wooden rocking horse | aaa719040101 | claude-opus-5-5 | 820270 | 743884 | 76386 | 76354 | 32 | 14558 | 202.2 | ok | 5000.0 |
| 3 | a pogo stick | 8ea9a6b89e2f | claude-opus-5-5 | 2663910 | 2544837 | 119073 | 118995 | 78 | 25927 | 468.2 | ok | 500.0 |
| 4 | a wooden toy train engine | 7ce3ebef7805 | claude-opus-5-5 | 1159089 | 1073565 | 85524 | 85482 | 42 | 16118 | 233.0 | ok | 30000.0 |
| 5 | a yo-yo | 2d2b4f1aebe1 | claude-opus-5-5 | 2611948 | 2498183 | 113765 | 113691 | 74 | 22667 | 441.2 | ok | 500000.0 |
