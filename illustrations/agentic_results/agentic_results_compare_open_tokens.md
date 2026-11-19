Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  Codex runs (agent codex): metrics.json records Codex's counts in these fields (cache_read_input_tokens = cached_input_tokens, cache_creation_input_tokens = cache_write_input_tokens, input_tokens = the rest of the input).  The figure's cost bars are relative to the largest run: 16 M input tokens, 290 k output tokens, 41 min 45 s.

| row | instruction | session_id | model | agent | effort | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 075c5e69e9fb | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3038530 | 2948992 | 89538 | 0 | 89538 | 44747 | 432.3 | ok | 200.0 |
| 2 | a TV remote control | dca76f0519f4 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2383693 | 2297088 | 86605 | 0 | 86605 | 43563 | 463.8 | ok | 1000.0 |
| 3 | a TV remote control | 28a47d5cf673 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2264100 | 2171851 | 92249 | 0 | 92249 | 16100 | 752.2 | ok | 500.0 |
| 4 | a USB stick | d79757aed64e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1263276 | 1199104 | 64172 | 0 | 64172 | 27457 | 264.1 | ok | 10000.0 |
| 5 | a USB stick | d49e5afc145f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3464105 | 3189376 | 274729 | 0 | 274729 | 95195 | 752.8 | ok | 100.0 |
| 6 | a USB stick | 64d60bc00ae7 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1531863 | 1440295 | 91568 | 0 | 91568 | 19724 | 410.7 | ok | 5000.0 |
| 7 | an electric toothbrush | 9d84135ca26d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1304085 | 1235584 | 68501 | 0 | 68501 | 26313 | 274.9 | ok | 5.0 |
| 8 | an electric toothbrush | 34294d1829a4 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2265952 | 2069888 | 196064 | 0 | 196064 | 67491 | 592.3 | ok | 5.0 |
| 9 | an electric toothbrush | 6a4492275165 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1265122 | 1191262 | 73860 | 0 | 73860 | 13627 | 330.8 | ok | 20.0 |
| 10 | a coffee mug | 2d1e773f2e4d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1476903 | 1416832 | 60071 | 0 | 60071 | 24659 | 241.3 | ok | 50000.0 |
| 11 | a coffee mug | ce53e61c7cbc | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1003591 | 816000 | 187591 | 0 | 187591 | 41358 | 336.7 | ok | 100000.0 |
| 12 | a coffee mug | 49c3269dd227 | moonshotai/Kimi-K3.1-Pro | codex | medium | 937740 | 872272 | 65468 | 0 | 65468 | 11450 | 277.5 | ok | 50000.0 |
| 13 | a computer mouse | 25f161931f83 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1279077 | 1211264 | 67813 | 0 | 67813 | 23400 | 229.6 | ok | 200000.0 |
| 14 | a computer mouse | 1af18cb3a58b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1402257 | 1163648 | 238609 | 0 | 238609 | 55848 | 454.8 | ok | 200000.0 |
| 15 | a computer mouse | a6129a879734 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1595670 | 1516342 | 79328 | 0 | 79328 | 15108 | 474.2 | ok | 1000.0 |
| 16 | a school chair with a steel frame and a plastic seat | 9e50d7604a8d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1943865 | 1866240 | 77625 | 0 | 77625 | 29366 | 302.9 | ok | 300.0 |
| 17 | a school chair with a steel frame and a plastic seat | 23dd03756749 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 16485376 | 15870976 | 614400 | 0 | 614400 | 285587 | 2504.7 | ok | 20.0 |
| 18 | a school chair with a steel frame and a plastic seat | 84a54143155c | moonshotai/Kimi-K3.1-Pro | codex | medium | 5654264 | 5538868 | 115396 | 0 | 115396 | 23024 | 678.9 | ok | 50.0 |
| 19 | a glass-top dining table | 7c9e52eb2703 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3726551 | 3623168 | 103383 | 0 | 103383 | 54017 | 770.0 | ok | 500.0 |
| 20 | a glass-top dining table | 53478bfa4013 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 6068815 | 5918848 | 149967 | 0 | 149967 | 106688 | 833.1 | ok | 3000.0 |
| 21 | a glass-top dining table | 97b3916bd322 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1873869 | 1792893 | 80976 | 0 | 80976 | 16084 | 367.2 | ok | 500.0 |
| 22 | a park bench | 54ac27bc6e33 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 10532644 | 10377600 | 155044 | 0 | 155044 | 76332 | 952.0 | ok | 2000.0 |
| 23 | a park bench | 3e65802b0a1c | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 5368440 | 5202048 | 166392 | 0 | 166392 | 117033 | 1039.0 | ok | 100.0 |
| 24 | a park bench | 921a1ece04db | moonshotai/Kimi-K3.1-Pro | codex | medium | 5027137 | 4922838 | 104299 | 0 | 104299 | 21383 | 630.2 | ok | 300.0 |
| 25 | a bar stool | 92ade18d3696 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2203310 | 2105344 | 97966 | 0 | 97966 | 43027 | 461.2 | ok | 500.0 |
| 26 | a bar stool | 9e11df628721 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1993300 | 1888512 | 104788 | 0 | 104788 | 64042 | 483.7 | ok | 500.0 |
| 27 | a bar stool | df0f348be915 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2904557 | 2802917 | 101640 | 0 | 101640 | 19957 | 606.9 | ok | 30000.0 |
| 28 | a sofa | d73aa50eefa1 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1385891 | 1317120 | 68771 | 0 | 68771 | 25315 | 299.0 | ok | 5000.0 |
| 29 | a sofa | b83931002c7f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3624784 | 3488128 | 136656 | 0 | 136656 | 86275 | 615.7 | ok | 10000.0 |
| 30 | a sofa | 5f29d2fd0321 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1655647 | 1577336 | 78311 | 0 | 78311 | 14390 | 338.4 | ok | 5000.0 |
| 31 | a teddy bear | 05dd0f9f80d9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1559778 | 1494912 | 64866 | 0 | 64866 | 32072 | 338.6 | ok | 2000.0 |
| 32 | a teddy bear | 66c54a4b6668 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2818028 | 2708096 | 109932 | 0 | 109932 | 67189 | 528.9 | ok | 10.0 |
| 33 | a teddy bear | ed7b3ec4214f | moonshotai/Kimi-K3.1-Pro | codex | medium | 1664821 | 1583262 | 81559 | 0 | 81559 | 18750 | 449.3 | ok | 100.0 |
| 34 | a wooden rocking horse | 9677deab9824 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1262218 | 1193216 | 69002 | 0 | 69002 | 32096 | 324.6 | ok | 10000.0 |
| 35 | a wooden rocking horse | 1c23c4564b5b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2060720 | 1962752 | 97968 | 0 | 97968 | 54215 | 452.4 | ok | 2000.0 |
| 36 | a wooden rocking horse | 93b215bfedd8 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1951076 | 1880167 | 70909 | 0 | 70909 | 11626 | 364.7 | ok | 2000.0 |
| 37 | a pogo stick | 895fec3a9082 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3957576 | 3837184 | 120392 | 0 | 120392 | 71694 | 711.2 | ok | 10000.0 |
| 38 | a pogo stick | f7ef7161cc37 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 5258650 | 5118720 | 139930 | 0 | 139930 | 83267 | 965.7 | ok | 1000.0 |
| 39 | a pogo stick | 968d94e3943d | moonshotai/Kimi-K3.1-Pro | codex | medium | 3041625 | 2956674 | 84951 | 0 | 84951 | 21925 | 742.7 | ok |  |
| 40 | a wooden toy train engine | 451b8d9f9ef2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1696355 | 1618432 | 77923 | 0 | 77923 | 27193 | 339.8 | ok | 5000.0 |
| 41 | a wooden toy train engine | c7a3d6db014b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2567769 | 2455808 | 111961 | 0 | 111961 | 89468 | 728.6 | ok | 500.0 |
| 42 | a wooden toy train engine | 469d39072c65 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2396795 | 2290474 | 106321 | 0 | 106321 | 16878 | 384.0 | ok | 3000.0 |
| 43 | a yo-yo | 8a00a42c4036 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2774917 | 2695168 | 79749 | 0 | 79749 | 52537 | 559.8 | ok | 50000.0 |
| 44 | a yo-yo | a0e602f24696 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 4141499 | 4020736 | 120763 | 0 | 120763 | 90665 | 1014.0 | ok | 200.0 |
| 45 | a yo-yo | 1e551392754c | moonshotai/Kimi-K3.1-Pro | codex | medium | 1049816 | 978063 | 71753 | 0 | 71753 | 10299 | 312.2 | ok | 300000.0 |
| 46 | a jelly dessert | dadf067b768e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3018130 | 2926976 | 91154 | 0 | 91154 | 46453 | 479.6 | ok | 10.0 |
| 47 | a jelly dessert | 2cf6be1bb7cb | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1426210 | 1360512 | 65698 | 0 | 65698 | 33809 | 328.4 | ok | 100.0 |
| 48 | a jelly dessert | 2b5b79392575 | moonshotai/Kimi-K3.1-Pro | codex | medium | 3140638 | 3045935 | 94703 | 0 | 94703 | 16123 | 533.6 | ok | 20.0 |
| 49 | a loaf of bread | 2f4b91dbf96a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1833236 | 1751040 | 82196 | 0 | 82196 | 40947 | 443.2 | ok | 10.0 |
| 50 | a loaf of bread | d7b47180ec7f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1074580 | 1007104 | 67476 | 0 | 67476 | 35590 | 359.0 | ok | 20.0 |
| 51 | a loaf of bread | c43b41af2dda | moonshotai/Kimi-K3.1-Pro | codex | medium | 1432114 | 1360807 | 71307 | 0 | 71307 | 13639 | 389.6 | ok | 50.0 |
| 52 | a hamburger | cfb71664cc4e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4328277 | 4216192 | 112085 | 0 | 112085 | 67508 | 972.1 | ok | 2.0 |
| 53 | a hamburger | 0b09456ffeb8 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1127395 | 1049856 | 77539 | 0 | 77539 | 44782 | 404.8 | ok | 50.0 |
| 54 | a hamburger | a9d778947315 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2956365 | 2872325 | 84040 | 0 | 84040 | 17462 | 998.7 | ok |  |
| 55 | a layered birthday cake | 607251581e18 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1039973 | 982016 | 57957 | 0 | 57957 | 24976 | 270.3 | ok | 30.0 |
| 56 | a layered birthday cake | 6e2f03528b5d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1544228 | 1467648 | 76580 | 0 | 76580 | 39782 | 336.5 | ok | 300.0 |
| 57 | a layered birthday cake | e49f68ed7fbf | moonshotai/Kimi-K3.1-Pro | codex | medium | 627679 | 567187 | 60492 | 0 | 60492 | 11318 | 252.0 | ok | 20.0 |
| 58 | a slice of watermelon | bcd8ef51b43a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2526653 | 2433536 | 93117 | 0 | 93117 | 50449 | 431.1 | ok | 2.0 |
| 59 | a slice of watermelon | 870282f43e95 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2194150 | 2099712 | 94438 | 0 | 94438 | 54577 | 523.2 | ok | 30.0 |
| 60 | a slice of watermelon | 259a7a30d6e5 | moonshotai/Kimi-K3.1-Pro | codex | medium | 3498939 | 3381537 | 117402 | 0 | 117402 | 20422 | 735.9 | ok | 1.0 |
| 61 | a giraffe | 4d61685fabe2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1562455 | 1495296 | 67159 | 0 | 67159 | 31801 | 282.9 | ok | 5000.0 |
| 62 | a giraffe | 88c278050f92 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1663947 | 1566848 | 97099 | 0 | 97099 | 52759 | 437.8 | ok | 2000.0 |
| 63 | a giraffe | 144d93800057 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2233284 | 2143764 | 89520 | 0 | 89520 | 18112 | 407.1 | ok | 50.0 |
| 64 | an elephant | 196d7f843dec | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1654137 | 1572480 | 81657 | 0 | 81657 | 34910 | 339.3 | ok | 200.0 |
| 65 | an elephant | 2a42271b2499 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2695015 | 2575360 | 119655 | 0 | 119655 | 81320 | 645.5 | ok | 5000.0 |
| 66 | an elephant | b5aff75e0cf0 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2978428 | 2890127 | 88301 | 0 | 88301 | 15342 | 433.0 | ok | 10000.0 |
| 67 | a rhinoceros | 3dbc42b9fca8 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2092508 | 2014976 | 77532 | 0 | 77532 | 28444 | 295.9 | ok | 200000.0 |
| 68 | a rhinoceros | 396362fa0554 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1546236 | 1449856 | 96380 | 0 | 96380 | 57552 | 439.6 | ok | 5000.0 |
| 69 | a rhinoceros | e9e6d1ee6efb | moonshotai/Kimi-K3.1-Pro | codex | medium | 1690582 | 1612992 | 77590 | 0 | 77590 | 15832 | 316.5 | ok | 50000.0 |
| 70 | a horse | 977c83aff0bc | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1492708 | 1407744 | 84964 | 0 | 84964 | 42018 | 355.2 | ok | 30000.0 |
| 71 | a horse | dcf516d0bc25 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2257665 | 2152704 | 104961 | 0 | 104961 | 47065 | 430.3 | ok | 3000.0 |
| 72 | a horse | f443de3b382b | moonshotai/Kimi-K3.1-Pro | codex | medium | 1778810 | 1699083 | 79727 | 0 | 79727 | 12610 | 286.6 | ok | 2000.0 |
| 73 | a crocodile | 550607601743 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 6331803 | 6192640 | 139163 | 0 | 139163 | 88239 | 814.9 | ok | 500000.0 |
| 74 | a crocodile | dca357d5ac3e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2317717 | 2211328 | 106389 | 0 | 106389 | 90349 | 606.4 | ok | 200.0 |
| 75 | a crocodile | 8cc7cff992a7 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2503325 | 2392670 | 110655 | 0 | 110655 | 19217 | 555.5 | ok | 100.0 |
| 76 | a car suspension control arm with rubber bushings | e7dc01e7a3f4 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4417454 | 4310144 | 107310 | 0 | 107310 | 48317 | 561.6 | ok | 10.0 |
| 77 | a car suspension control arm with rubber bushings | ba44c5c0487a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 4152666 | 3996672 | 155994 | 0 | 155994 | 102909 | 832.7 | ok | 10.0 |
| 78 | a car suspension control arm with rubber bushings | 1015276d2134 | moonshotai/Kimi-K3.1-Pro | codex | medium | 3134585 | 3018799 | 115786 | 0 | 115786 | 27374 | 1373.7 | ok | 20.0 |
| 79 | a truck leaf spring | 7b9c26c86f8e | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4769290 | 4669568 | 99722 | 0 | 99722 | 62030 | 686.7 | ok | 100.0 |
| 80 | a truck leaf spring | 05f18a435bfe | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 6020320 | 5911552 | 108768 | 0 | 108768 | 80136 | 977.9 | ok | 2.0 |
| 81 | a truck leaf spring | f54d71e214d1 | moonshotai/Kimi-K3.1-Pro | codex | medium | 13314875 | 12958275 | 356600 | 0 | 356600 | 45249 | 1380.7 | ok | 100.0 |
| 82 | a car wheel with its rubber tyre | d41f540d2e41 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1973242 | 1902592 | 70650 | 0 | 70650 | 29029 | 326.3 | ok | 2000.0 |
| 83 | a car wheel with its rubber tyre | dc181cc672e0 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3914182 | 3775104 | 139078 | 0 | 139078 | 77183 | 688.8 | ok | 500.0 |
| 84 | a car wheel with its rubber tyre | 71ae09de8185 | moonshotai/Kimi-K3.1-Pro | codex | medium | 3060570 | 2974330 | 86240 | 0 | 86240 | 16091 | 490.8 | ok | 1000.0 |
| 85 | a car brake disc | 9f88d348f190 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4106248 | 4008960 | 97288 | 0 | 97288 | 55352 | 526.7 | ok | 2000.0 |
| 86 | a car brake disc | 8b54c0ad4cc3 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 4503522 | 4346496 | 157026 | 0 | 157026 | 113695 | 822.7 | ok | 5000.0 |
| 87 | a car brake disc | 1089687116f4 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1166303 | 1092462 | 73841 | 0 | 73841 | 14399 | 316.0 | ok | 10000.0 |
| 88 | a crankshaft | b33df010e457 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4683987 | 4567168 | 116819 | 0 | 116819 | 63506 | 746.4 | ok | 5000.0 |
| 89 | a crankshaft | 9d12d97130a9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 9625602 | 9432448 | 193154 | 0 | 193154 | 132308 | 1074.7 | ok | 1000.0 |
| 90 | a crankshaft | 019fe1409a1c | moonshotai/Kimi-K3.1-Pro | codex | medium | 4677435 | 4556434 | 121001 | 0 | 121001 | 28770 | 686.6 | ok | 1000.0 |
| 91 | a road roller with a steel drum and rubber tyres | f06ee39c0c03 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 9224402 | 9042048 | 182354 | 0 | 182354 | 107080 | 1228.7 | ok |  |
| 92 | a road roller with a steel drum and rubber tyres | 8e29d1b16762 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 14821460 | 14443392 | 378068 | 0 | 378068 | 148657 | 1662.5 | ok | 20000.0 |
| 93 | a road roller with a steel drum and rubber tyres | 572c45c73d95 | moonshotai/Kimi-K3.1-Pro | codex | medium | 5673951 | 5565446 | 108505 | 0 | 108505 | 29110 | 730.8 | ok |  |
| 94 | a jackhammer | a382aba37bf5 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3237973 | 3159424 | 78549 | 0 | 78549 | 61534 | 525.2 | ok | 500000.0 |
| 95 | a jackhammer | 0c5216ec6a91 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 9006826 | 8639360 | 367466 | 0 | 367466 | 191611 | 1503.1 | ok | 100000.0 |
| 96 | a jackhammer | 0047d3143722 | moonshotai/Kimi-K3.1-Pro | codex | medium | 5756684 | 5652039 | 104645 | 0 | 104645 | 23207 | 608.3 | ok | 30000.0 |
| 97 | an industrial robot arm | 270fb4a56dae | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1515688 | 1442176 | 73512 | 0 | 73512 | 33261 | 350.1 | ok | 2000.0 |
| 98 | an industrial robot arm | 8f0ce822392b | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3200883 | 3094016 | 106867 | 0 | 106867 | 56138 | 522.0 | ok | 300.0 |
| 99 | an industrial robot arm | cf74c713c1c6 | moonshotai/Kimi-K3.1-Pro | codex | medium | 833435 | 768123 | 65312 | 0 | 65312 | 10973 | 217.7 | ok | 5000.0 |
| 100 | an adjustable wrench | 565735f3db05 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1241376 | 1177216 | 64160 | 0 | 64160 | 22793 | 257.4 | ok | 100.0 |
| 101 | an adjustable wrench | c6ee337e5cb7 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1562911 | 1466112 | 96799 | 0 | 96799 | 54833 | 422.4 | ok | 50.0 |
| 102 | an adjustable wrench | d6694645e83a | moonshotai/Kimi-K3.1-Pro | codex | medium | 1328072 | 1247501 | 80571 | 0 | 80571 | 13923 | 291.4 | ok | 100.0 |
| 103 | a claw hammer | 85df1551b0f1 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2510729 | 2424448 | 86281 | 0 | 86281 | 31539 | 362.4 | ok | 5.0 |
| 104 | a claw hammer | 6d26b093c8c1 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2485858 | 2385280 | 100578 | 0 | 100578 | 55667 | 463.6 | ok | 50.0 |
| 105 | a claw hammer | 55ace4d99333 | moonshotai/Kimi-K3.1-Pro | codex | medium | 1412557 | 1337396 | 75161 | 0 | 75161 | 12878 | 274.3 | ok | 50.0 |
| 106 | a dental crown | 9d069b762ac9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2674799 | 2588416 | 86383 | 0 | 86383 | 44473 | 556.9 | ok | 1000.0 |
| 107 | a dental crown | 72a4a98e7c41 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 5128131 | 4963456 | 164675 | 0 | 164675 | 121274 | 974.1 | ok | 1000.0 |
| 108 | a dental crown | 209ab0fbde2b | moonshotai/Kimi-K3.1-Pro | codex | medium | 2420090 | 2335229 | 84861 | 0 | 84861 | 15749 | 381.2 | ok | 2000.0 |
| 109 | a knee implant | 27402b75ebc0 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4265538 | 4138624 | 126914 | 0 | 126914 | 70490 | 805.0 | ok | 500000.0 |
| 110 | a knee implant | c716786b84a3 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 5192306 | 5072896 | 119410 | 0 | 119410 | 59671 | 691.5 | ok | 10.0 |
| 111 | a knee implant | b5ad6908102f | moonshotai/Kimi-K3.1-Pro | codex | medium | 3049662 | 2826553 | 223109 | 0 | 223109 | 17050 | 597.3 | ok | 1000.0 |
| 112 | a walking cane with a rubber tip | 7496ab033876 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 4401944 | 4269952 | 131992 | 0 | 131992 | 98333 | 840.4 | ok | 1000.0 |
| 113 | a walking cane with a rubber tip | 84188643a8e7 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2169031 | 2067200 | 101831 | 0 | 101831 | 58813 | 535.4 | ok | 1000.0 |
| 114 | a walking cane with a rubber tip | d105a21ad391 | moonshotai/Kimi-K3.1-Pro | codex | medium | 6130515 | 5996471 | 134044 | 0 | 134044 | 29811 | 829.2 | ok | 30.0 |
| 115 | a human femur | 5c2cf66635c9 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1442506 | 1363968 | 78538 | 0 | 78538 | 37249 | 354.1 | ok | 1.0 |
| 116 | a human femur | 12e43456d4e0 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2886368 | 2755456 | 130912 | 0 | 130912 | 76777 | 663.0 | ok |  |
| 117 | a human femur | c3ab6b31eefd | moonshotai/Kimi-K3.1-Pro | codex | medium | 941121 | 874640 | 66481 | 0 | 66481 | 16674 | 361.4 | ok |  |
| 118 | a lumbar vertebra | e66be1f0b63c | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3522398 | 3394048 | 128350 | 0 | 128350 | 55382 | 688.6 | ok | 3.0 |
| 119 | a lumbar vertebra | 490317059c69 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3892534 | 3748096 | 144438 | 0 | 144438 | 83616 | 902.5 | ok | 200.0 |
| 120 | a lumbar vertebra | 2db0617aed0d | moonshotai/Kimi-K3.1-Pro | codex | medium | 2117883 | 2039388 | 78495 | 0 | 78495 | 18574 | 963.4 | ok |  |
| 121 | a skyscraper | f5a9ec90d5ac | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2881416 | 2782592 | 98824 | 0 | 98824 | 67371 | 637.3 | ok |  |
| 122 | a skyscraper | 2165fcdf1490 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3301735 | 3195648 | 106087 | 0 | 106087 | 64912 | 525.1 | ok | 2000.0 |
| 123 | a skyscraper | 8c507802b1ac | moonshotai/Kimi-K3.1-Pro | codex | medium | 3208537 | 3107087 | 101450 | 0 | 101450 | 17097 | 830.8 | ok | 5000.0 |
| 124 | a thatched stone cottage | ae83e6cf8d51 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3335189 | 3232768 | 102421 | 0 | 102421 | 55978 | 565.6 | ok | 50000.0 |
| 125 | a thatched stone cottage | 7a2267f7ca53 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2356461 | 2257792 | 98669 | 0 | 98669 | 81974 | 647.6 | ok | 200000.0 |
| 126 | a thatched stone cottage | 8dc4ddae0e3a | moonshotai/Kimi-K3.1-Pro | codex | medium | 1862091 | 1786823 | 75268 | 0 | 75268 | 13395 | 556.3 | ok | 30000.0 |
| 127 | a lighthouse | 1c64d7271858 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1185852 | 1122048 | 63804 | 0 | 63804 | 25550 | 257.6 | ok | 30000.0 |
| 128 | a lighthouse | 4b4e0a9a04d2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 5043531 | 4906112 | 137419 | 0 | 137419 | 83987 | 964.8 | ok | 5000.0 |
| 129 | a lighthouse | c80aa15ac6e2 | moonshotai/Kimi-K3.1-Pro | codex | medium | 4088152 | 3983497 | 104655 | 0 | 104655 | 21413 | 1076.3 | ok | 2000.0 |
| 130 | an igloo | 530058c40b43 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1617324 | 1554176 | 63148 | 0 | 63148 | 27721 | 297.2 | ok | 1000.0 |
| 131 | an igloo | 3dd222ad8493 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3601222 | 3472384 | 128838 | 0 | 128838 | 100271 | 794.1 | ok | 5000.0 |
| 132 | an igloo | 508df455a189 | moonshotai/Kimi-K3.1-Pro | codex | medium | 8776686 | 8644146 | 132540 | 0 | 132540 | 30073 | 1508.2 | ok | 3000.0 |
| 133 | a log cabin | 54bdfe167b70 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1248837 | 1183232 | 65605 | 0 | 65605 | 21567 | 256.1 | ok | 500000.0 |
| 134 | a log cabin | c7c0977965aa | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 1900286 | 1793664 | 106622 | 0 | 106622 | 87271 | 653.1 | ok | 100000.0 |
| 135 | a log cabin | bb6e1db33309 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2990968 | 2899503 | 91465 | 0 | 91465 | 19370 | 966.5 | ok | 10000.0 |
| 136 | the Leaning Tower of Pisa | 0e8bd9197100 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 2312162 | 2222080 | 90082 | 0 | 90082 | 34374 | 526.2 | ok | 10000.0 |
| 137 | the Leaning Tower of Pisa | 758af01c110a | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 3483429 | 3370240 | 113189 | 0 | 113189 | 54805 | 749.6 | ok | 5000.0 |
| 138 | the Leaning Tower of Pisa | 69a840791953 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2199552 | 2053987 | 145565 | 0 | 145565 | 16527 | 877.5 | ok | 10000.0 |
| 139 | the Pantheon | 73d2ea4e2c5f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 3036738 | 2946816 | 89922 | 0 | 89922 | 45218 | 581.2 | ok | 5000.0 |
| 140 | the Pantheon | 91c2fee7c199 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 4723874 | 4584960 | 138914 | 0 | 138914 | 82521 | 840.1 | ok | 5000.0 |
| 141 | the Pantheon | 20b4190e62f3 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2846102 | 2743886 | 102216 | 0 | 102216 | 21883 | 778.0 | ok | 5000.0 |
| 142 | Himeji Castle | 732eca44dd9d | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1513946 | 1447936 | 66010 | 0 | 66010 | 31771 | 339.5 | ok | 20000.0 |
| 143 | Himeji Castle | 2fdaab9245b2 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 4482484 | 4320640 | 161844 | 0 | 161844 | 105503 | 1035.2 | ok | 500.0 |
| 144 | Himeji Castle | 9a9e09b6c4c4 | moonshotai/Kimi-K3.1-Pro | codex | medium | 2415163 | 2318243 | 96920 | 0 | 96920 | 18456 | 530.1 | ok | 100.0 |
| 145 | a natural stone arch | 43078b7b2f7f | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1427666 | 1354368 | 73298 | 0 | 73298 | 27638 | 334.1 | ok | 50000.0 |
| 146 | a natural stone arch | 4df641e26373 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2290353 | 2171392 | 118961 | 0 | 118961 | 57418 | 545.0 | ok | 200000.0 |
| 147 | a natural stone arch | 9abe48a9d42e | moonshotai/Kimi-K3.1-Pro | codex | medium | 2303852 | 2224080 | 79772 | 0 | 79772 | 15372 | 370.1 | ok | 2.0 |
| 148 | the Taj Mahal | 2212b1dc44c6 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | medium | 1994990 | 1913088 | 81902 | 0 | 81902 | 37219 | 488.4 | ok | 30.0 |
| 149 | the Taj Mahal | cc1f28d23b27 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | codex | max | 2230019 | 2159872 | 70147 | 0 | 70147 | 60384 | 733.6 | ok |  |
| 150 | the Taj Mahal | 6f61b696607a | moonshotai/Kimi-K3.1-Pro | codex | medium | 1162758 | 1088159 | 74599 | 0 | 74599 | 15072 | 343.7 | ok | 5000.0 |
