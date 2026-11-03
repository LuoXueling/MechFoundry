Supplementary table. Token use per run of agentic_results (sources: agentic_results/ar_stats.json). Cached = input tokens read from the prompt cache (cache_read_input_tokens); tokens written to the cache (cache_creation_input_tokens) were processed in full and count as not cached, with the plain uncached input (input_tokens).  Input total = cached + not cached.  The figure's cost bars are relative to the largest run: 7.4 M input tokens, 88 k output tokens, 19 min 16 s.

| row | instruction | session_id | model | input_tokens_total | input_tokens_cached | input_tokens_not_cached | of_which_written_to_cache | of_which_plain_uncached | output_tokens | wall_clock_s | status | displacement_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a TV remote control | 977fd33385ea | claude-haiku-5-5 | 2188898 | 2081871 | 107027 | 106967 | 60 | 47209 | 370.3 | ok | 2000.0 |
| 2 | a TV remote control | 1aae26b57dc6 | claude-sonnet-5-5 | 949055 | 866997 | 82058 | 82020 | 38 | 11898 | 206.7 | ok | 300.0 |
| 3 | a TV remote control | e045ee5e9396 | claude-opus-5-5 | 1014605 | 942098 | 72507 | 72469 | 38 | 15110 | 272.4 | ok | 1000.0 |
| 4 | a USB stick | b042285e3ce7 | claude-haiku-5-5 | 7443139 | 7281490 | 161649 | 161491 | 158 | 75333 | 747.6 | ok | 1000.0 |
| 5 | a USB stick | 882377636038 | claude-sonnet-5-5 | 1571073 | 1484220 | 86853 | 86797 | 56 | 15444 | 224.1 | ok | 5.0 |
| 6 | a USB stick | 1853ba6620f0 | claude-opus-5-5 | 1294273 | 1206812 | 87461 | 87417 | 44 | 16005 | 227.9 | ok | 3.0 |
| 7 | an electric toothbrush | 33fcad516671 | claude-haiku-5-5 | 1312260 | 1213338 | 98922 | 98878 | 44 | 42473 | 284.1 | ok | 10.0 |
| 8 | an electric toothbrush | a108fe4244d9 | claude-sonnet-5-5 | 1229698 | 1148085 | 81613 | 81567 | 46 | 12149 | 197.6 | ok | 10.0 |
| 9 | an electric toothbrush | a3ddf34a5ebf | claude-opus-5-5 | 1408591 | 1321859 | 86732 | 86682 | 50 | 17842 | 273.8 | ok | 3.0 |
| 10 | a coffee mug | e58a4dca06fe | claude-haiku-5-5 | 1528816 | 1435877 | 92939 | 92889 | 50 | 34778 | 252.1 | ok | 50000.0 |
| 11 | a coffee mug | cb9c0c364b61 | claude-sonnet-5-5 | 860670 | 783699 | 76971 | 76937 | 34 | 11461 | 169.1 | ok | 3000.0 |
| 12 | a coffee mug | 2c96c95ab93a | claude-opus-5-5 | 812014 | 737812 | 74202 | 74170 | 32 | 13442 | 211.0 | ok | 50000.0 |
| 13 | a computer mouse | ebb2cbe4ae90 | claude-haiku-5-5 | 1603674 | 1510535 | 93139 | 93089 | 50 | 31198 | 252.9 | ok | 30000.0 |
| 14 | a computer mouse | ee62da33a70a | claude-sonnet-5-5 | 742464 | 674406 | 68058 | 68026 | 32 | 10087 | 165.6 | ok | 100000.0 |
| 15 | a computer mouse | fb8d771bf93e | claude-opus-5-5 | 956384 | 873777 | 82607 | 82571 | 36 | 18219 | 247.1 | ok | 10000.0 |
| 16 | a school chair with a steel frame and a plastic seat | dbc9c1c4554a | claude-haiku-5-5 | 3629792 | 3488480 | 141312 | 141230 | 82 | 67850 | 542.9 | ok | 50.0 |
| 17 | a school chair with a steel frame and a plastic seat | d2ffbcaa6fed | claude-sonnet-5-5 | 1661533 | 1569190 | 92343 | 92289 | 54 | 14511 | 212.5 | ok | 50.0 |
| 18 | a school chair with a steel frame and a plastic seat | c90e732be96e | claude-opus-5-5 | 2475914 | 2361128 | 114786 | 114714 | 72 | 22474 | 423.8 | ok | 30.0 |
| 19 | a glass-top dining table | 6cbd9d366139 | claude-haiku-5-5 | 3118628 | 2990575 | 128053 | 127975 | 78 | 56549 | 439.5 | ok | 500.0 |
| 20 | a glass-top dining table | ddb9903ec17c | claude-sonnet-5-5 | 2073382 | 1975520 | 97862 | 97794 | 68 | 17863 | 281.3 | ok | 200.0 |
| 21 | a glass-top dining table | 9e19e47e3168 | claude-opus-5-5 | 1599488 | 1497648 | 101840 | 101790 | 50 | 23542 | 359.2 | ok | 1000.0 |
| 22 | a park bench | d6068d225c0a | claude-haiku-5-5 | 6913137 | 6727589 | 185548 | 185412 | 136 | 87586 | 783.3 | ok | 2000.0 |
| 23 | a park bench | 809d3c845254 | claude-sonnet-5-5 | 2419880 | 2305160 | 114720 | 114652 | 68 | 20799 | 472.9 | ok | 300.0 |
| 24 | a park bench | c09fb246962f | claude-opus-5-5 | 1807065 | 1699095 | 107970 | 107912 | 58 | 25353 | 516.4 | ok | 200.0 |
| 25 | a bar stool | 70a929c05373 | claude-haiku-5-5 | 1045727 | 976764 | 68963 | 68919 | 44 | 19850 | 163.7 | ok | 5000.0 |
| 26 | a bar stool | f598ee8a0160 | claude-sonnet-5-5 | 974021 | 902700 | 71321 | 71279 | 42 | 10478 | 175.2 | ok | 3000.0 |
| 27 | a bar stool | fae04d03ee0a | claude-opus-5-5 | 843007 | 761085 | 81922 | 81890 | 32 | 15831 | 237.3 | ok | 500.0 |
| 28 | a sofa | 2ccc44aeb8e5 | claude-haiku-5-5 | 1954386 | 1837887 | 116499 | 116447 | 52 | 46543 | 348.9 | ok | 5000.0 |
| 29 | a sofa | ed63e33804b4 | claude-sonnet-5-5 | 1032253 | 953332 | 78921 | 78881 | 40 | 11946 | 180.8 | ok | 5000.0 |
| 30 | a sofa | 03af3db6b4df | claude-opus-5-5 | 1003070 | 916998 | 86072 | 86036 | 36 | 15871 | 243.6 | ok | 5000.0 |
| 31 | a teddy bear | caa2f7f93fcb | claude-haiku-5-5 | 1548452 | 1462454 | 85998 | 85940 | 58 | 28904 | 255.9 | ok | 300.0 |
| 32 | a teddy bear | dca313df00a5 | claude-sonnet-5-5 | 870987 | 799929 | 71058 | 71022 | 36 | 10246 | 152.3 | ok | 20.0 |
| 33 | a teddy bear | f70ceee7ad2d | claude-opus-5-5 | 1024656 | 944869 | 79787 | 79747 | 40 | 17610 | 241.5 | ok | 5.0 |
| 34 | a wooden rocking horse | a77422267b48 | claude-haiku-5-5 | 2211301 | 2097803 | 113498 | 113434 | 64 | 50202 | 354.0 | ok | 10000.0 |
| 35 | a wooden rocking horse | 4464ff4b071a | claude-sonnet-5-5 | 746166 | 672742 | 73424 | 73392 | 32 | 10828 | 142.7 | ok | 30000.0 |
| 36 | a wooden rocking horse | aaa719040101 | claude-opus-5-5 | 820270 | 743884 | 76386 | 76354 | 32 | 14558 | 202.2 | ok | 5000.0 |
| 37 | a pogo stick | 5a2622107f6d | claude-haiku-5-5 | 5652700 | 5517392 | 135308 | 135178 | 130 | 77883 | 536.7 | ok |  |
| 38 | a pogo stick | fbc36e9abb60 | claude-sonnet-5-5 | 3427235 | 3307666 | 119569 | 119473 | 96 | 27470 | 488.1 | ok | 500.0 |
| 39 | a pogo stick | 8ea9a6b89e2f | claude-opus-5-5 | 2663910 | 2544837 | 119073 | 118995 | 78 | 25927 | 468.2 | ok | 500.0 |
| 40 | a wooden toy train engine | 7403ed98cfe3 | claude-haiku-5-5 | 1950282 | 1835126 | 115156 | 115102 | 54 | 51031 | 364.1 | ok | 10000.0 |
| 41 | a wooden toy train engine | 843b418bdbdc | claude-sonnet-5-5 | 1152036 | 1072260 | 79776 | 79732 | 44 | 12525 | 173.4 | ok | 200000.0 |
| 42 | a wooden toy train engine | 7ce3ebef7805 | claude-opus-5-5 | 1159089 | 1073565 | 85524 | 85482 | 42 | 16118 | 233.0 | ok | 30000.0 |
| 43 | a yo-yo | e260972b7483 | claude-haiku-5-5 | 1754728 | 1641939 | 112789 | 112739 | 50 | 50473 | 425.7 | ok | 50000.0 |
| 44 | a yo-yo | 812e2d37f771 | claude-sonnet-5-5 | 1997553 | 1896117 | 101436 | 101376 | 60 | 19018 | 332.2 | ok | 10000.0 |
| 45 | a yo-yo | 2d2b4f1aebe1 | claude-opus-5-5 | 2611948 | 2498183 | 113765 | 113691 | 74 | 22667 | 441.2 | ok | 500000.0 |
| 46 | a jelly dessert | a3c25e8723aa | claude-haiku-5-5 | 1166827 | 1090606 | 76221 | 76177 | 44 | 23786 | 264.5 | ok | 5.0 |
| 47 | a jelly dessert | 283de5cd5937 | claude-sonnet-5-5 | 978922 | 908017 | 70905 | 70863 | 42 | 12427 | 191.5 | ok | 3.0 |
| 48 | a jelly dessert | 9088600e6b55 | claude-opus-5-5 | 1629513 | 1539498 | 90015 | 89959 | 56 | 16812 | 313.3 | ok | 50.0 |
| 49 | a loaf of bread | 1e0d4fb73c2e | claude-haiku-5-5 | 2552172 | 2438840 | 113332 | 113258 | 74 | 46298 | 431.1 | ok | 10.0 |
| 50 | a loaf of bread | 0c71d99a58b5 | claude-sonnet-5-5 | 1335031 | 1252945 | 82086 | 82034 | 52 | 13635 | 261.6 | ok | 30.0 |
| 51 | a loaf of bread | a798f81096db | claude-opus-5-5 | 1871587 | 1767201 | 104386 | 104328 | 58 | 22347 | 400.5 | ok | 3.0 |
| 52 | a hamburger | 716ee895a54b | claude-haiku-5-5 | 1322631 | 1228830 | 93801 | 93757 | 44 | 35099 | 275.6 | ok | 500.0 |
| 53 | a hamburger | 24c883a572fb | claude-sonnet-5-5 | 2881623 | 2771984 | 109639 | 109555 | 84 | 19026 | 511.2 | ok | 3.0 |
| 54 | a hamburger | 38db33dd8f0e | claude-opus-5-5 | 2116299 | 2013203 | 103096 | 103036 | 60 | 24202 | 465.5 | ok | 1.0 |
| 55 | a layered birthday cake | 32f66c07230a | claude-haiku-5-5 | 1565899 | 1463930 | 101969 | 101919 | 50 | 42776 | 279.0 | ok | 5.0 |
| 56 | a layered birthday cake | ed792af45046 | claude-sonnet-5-5 | 763947 | 692632 | 71315 | 71283 | 32 | 11232 | 167.3 | ok | 20.0 |
| 57 | a layered birthday cake | bf1d591ef934 | claude-opus-5-5 | 955807 | 874587 | 81220 | 81184 | 36 | 16999 | 245.1 | ok | 5.0 |
| 58 | a slice of watermelon | 4cdc48fdacc6 | claude-haiku-5-5 | 1954642 | 1864216 | 90426 | 90362 | 64 | 37229 | 287.4 | ok | 1.0 |
| 59 | a slice of watermelon | 89c7cca2de07 | claude-sonnet-5-5 | 1371729 | 1287780 | 83949 | 83897 | 52 | 14528 | 249.4 | ok | 20.0 |
| 60 | a slice of watermelon | 880acc4a508a | claude-opus-5-5 | 2486016 | 2369203 | 116813 | 116743 | 70 | 27897 | 544.7 | ok | 30.0 |
| 61 | a giraffe | 624cf5636800 | claude-haiku-5-5 | 2409739 | 2299315 | 110424 | 110354 | 70 | 43812 | 314.5 | ok | 3000.0 |
| 62 | a giraffe | 69ecefce4461 | claude-sonnet-5-5 | 921640 | 851229 | 70411 | 70371 | 40 | 11438 | 159.2 | ok | 2000.0 |
| 63 | a giraffe | 7117203f0330 | claude-opus-5-5 | 966866 | 883539 | 83327 | 83291 | 36 | 17330 | 235.8 | ok | 2000.0 |
| 64 | an elephant | 62064c14e70d | claude-haiku-5-5 | 1412084 | 1311488 | 100596 | 100550 | 46 | 37727 | 267.1 | ok | 5000.0 |
| 65 | an elephant | 419757786162 | claude-sonnet-5-5 | 1143585 | 1066764 | 76821 | 76775 | 46 | 12804 | 169.0 | ok | 300.0 |
| 66 | an elephant | 82f39e89e2d0 | claude-opus-5-5 | 895161 | 816105 | 79056 | 79022 | 34 | 14970 | 205.0 | ok | 3000.0 |
| 67 | a rhinoceros | 2d72f34a181e | claude-haiku-5-5 | 1382443 | 1292745 | 89698 | 89648 | 50 | 33618 | 248.8 | ok | 100000.0 |
| 68 | a rhinoceros | d4658181bd0e | claude-sonnet-5-5 | 1149260 | 1072164 | 77096 | 77048 | 48 | 13004 | 172.4 | ok | 10000.0 |
| 69 | a rhinoceros | 9f81b6cf29d1 | claude-opus-5-5 | 800040 | 722505 | 77535 | 77503 | 32 | 14497 | 220.5 | ok | 3000.0 |
| 70 | a horse | 5f96963c4f2b | claude-haiku-5-5 | 1593002 | 1481589 | 111413 | 111365 | 48 | 48238 | 310.9 | ok | 1000.0 |
| 71 | a horse | 68bd6be06fac | claude-sonnet-5-5 | 936269 | 859363 | 76906 | 76868 | 38 | 12437 | 170.0 | ok | 5000.0 |
| 72 | a horse | 831002dd369c | claude-opus-5-5 | 891661 | 811782 | 79879 | 79845 | 34 | 15953 | 238.5 | ok | 1000.0 |
| 73 | a crocodile | d0c05863f7b4 | claude-haiku-5-5 | 1065817 | 989745 | 76072 | 76034 | 38 | 34240 | 235.6 | ok |  |
| 74 | a crocodile | 4afab00557b8 | claude-sonnet-5-5 | 1105535 | 1032592 | 72943 | 72897 | 46 | 12986 | 170.5 | ok | 50.0 |
| 75 | a crocodile | d5f284defbb1 | claude-opus-5-5 | 1485487 | 1383895 | 101592 | 101546 | 46 | 22514 | 327.8 | ok | 500.0 |
| 76 | a car suspension control arm with rubber bushings | b77383e1737c | claude-haiku-5-5 | 3503040 | 3363528 | 139512 | 139430 | 82 | 60868 | 443.5 | ok | 20.0 |
| 77 | a car suspension control arm with rubber bushings | d03344a19339 | claude-sonnet-5-5 | 1575591 | 1483246 | 92345 | 92291 | 54 | 15853 | 272.3 | ok | 10.0 |
| 78 | a car suspension control arm with rubber bushings | 73e7c5c5a94c | claude-opus-5-5 | 1849766 | 1741425 | 108341 | 108285 | 56 | 22337 | 388.0 | ok | 50.0 |
| 79 | a truck leaf spring | 23c0e20f48c4 | claude-haiku-5-5 | 5055638 | 4883951 | 171687 | 171583 | 104 | 80922 | 692.2 | ok | 10.0 |
| 80 | a truck leaf spring | 8815096b93c8 | claude-sonnet-5-5 | 1448145 | 1368411 | 79734 | 79678 | 56 | 17714 | 227.1 | ok | 1.0 |
| 81 | a truck leaf spring | 569d10b3c6e3 | claude-opus-5-5 | 3076303 | 2952617 | 123686 | 123608 | 78 | 33964 | 511.2 | ok | 30.0 |
| 82 | a car wheel with its rubber tyre | c279cc8bca77 | claude-haiku-5-5 | 1927840 | 1818089 | 109751 | 109695 | 56 | 40422 | 314.1 | ok | 5000.0 |
| 83 | a car wheel with its rubber tyre | c64de69286fc | claude-sonnet-5-5 | 1907784 | 1808443 | 99341 | 99275 | 66 | 16060 | 421.7 | ok | 20000.0 |
| 84 | a car wheel with its rubber tyre | abb0e9dafe58 | claude-opus-5-5 | 1951018 | 1845909 | 105109 | 105047 | 62 | 23469 | 580.5 | ok | 5000.0 |
| 85 | a car brake disc | 2a6120d9a7c7 | claude-haiku-5-5 | 865815 | 784837 | 80978 | 80944 | 34 | 31594 | 225.4 | ok | 10000.0 |
| 86 | a car brake disc | 59f833d18aaa | claude-sonnet-5-5 | 2132804 | 2035221 | 97583 | 97515 | 68 | 21226 | 370.3 | ok | 5000.0 |
| 87 | a car brake disc | ca5daddc59b4 | claude-opus-5-5 | 2306239 | 2194811 | 111428 | 111360 | 68 | 26207 | 512.0 | ok | 3000.0 |
| 88 | a crankshaft | b22752b09a70 | claude-haiku-5-5 | 2359071 | 2262495 | 96576 | 96506 | 70 | 41943 | 396.7 | ok |  |
| 89 | a crankshaft | 60843d087320 | claude-sonnet-5-5 | 3843563 | 3719921 | 123642 | 123538 | 104 | 27670 | 471.7 | ok | 1000.0 |
| 90 | a crankshaft | 84d9272ea75d | claude-opus-5-5 | 4814983 | 4659366 | 155617 | 155507 | 110 | 47901 | 753.3 | ok | 300.0 |
| 91 | a road roller with a steel drum and rubber tyres | fa69821708e9 | claude-haiku-5-5 | 5674446 | 5512264 | 162182 | 162070 | 112 | 83675 | 651.3 | ok | 2000.0 |
| 92 | a road roller with a steel drum and rubber tyres | c831b4e13cdf | claude-sonnet-5-5 | 2888967 | 2771720 | 117247 | 117165 | 82 | 23132 | 479.0 | ok | 50000.0 |
| 93 | a road roller with a steel drum and rubber tyres | 03468728e8f2 | claude-opus-5-5 | 7220676 | 7033514 | 187162 | 187026 | 136 | 69985 | 1156.1 | ok | 20000.0 |
| 94 | a jackhammer | f683baba82f4 | claude-haiku-5-5 | 5205108 | 5046225 | 158883 | 158777 | 106 | 69080 | 702.0 | ok | 500000.0 |
| 95 | a jackhammer | a1f39114ab87 | claude-sonnet-5-5 | 2719964 | 2610458 | 109506 | 109424 | 82 | 22928 | 446.8 | ok | 2000.0 |
| 96 | a jackhammer | 255401db7d1e | claude-opus-5-5 | 3159490 | 3036601 | 122889 | 122803 | 86 | 27947 | 551.6 | ok | 2000.0 |
| 97 | an industrial robot arm | beeb176cdcb5 | claude-haiku-5-5 | 1458777 | 1356634 | 102143 | 102097 | 46 | 42384 | 266.5 | ok | 3000.0 |
| 98 | an industrial robot arm | 8b6a7ddd2504 | claude-sonnet-5-5 | 1017635 | 945804 | 71831 | 71789 | 42 | 12411 | 159.7 | ok | 500.0 |
| 99 | an industrial robot arm | 4d0027d7fd41 | claude-opus-5-5 | 1582421 | 1489119 | 93302 | 93248 | 54 | 19121 | 337.3 | ok | 5000.0 |
| 100 | an adjustable wrench | cf5ceba6da28 | claude-haiku-5-5 | 1750366 | 1647732 | 102634 | 102580 | 54 | 43669 | 271.4 | ok | 50.0 |
| 101 | an adjustable wrench | 73cca1c05b5e | claude-sonnet-5-5 | 1101051 | 1023963 | 77088 | 77044 | 44 | 13233 | 166.1 | ok | 100.0 |
| 102 | an adjustable wrench | 06076fc07e03 | claude-opus-5-5 | 1433491 | 1338698 | 94793 | 94747 | 46 | 18945 | 250.0 | ok | 100.0 |
| 103 | a claw hammer | 4553fef600b2 | claude-haiku-5-5 | 3948743 | 3799591 | 149152 | 149066 | 86 | 63534 | 553.0 | ok | 300.0 |
| 104 | a claw hammer | 22f07ebe054b | claude-sonnet-5-5 | 1409271 | 1320760 | 88511 | 88461 | 50 | 15518 | 216.8 | ok | 30.0 |
| 105 | a claw hammer | e523575eda4e | claude-opus-5-5 | 1505605 | 1415302 | 90303 | 90255 | 48 | 18877 | 263.9 | ok | 10.0 |
| 106 | a dental crown | 9a08c9ef8bf5 | claude-haiku-5-5 | 2088821 | 1988369 | 100452 | 100388 | 64 | 35454 | 339.7 | ok | 2000.0 |
| 107 | a dental crown | c48ecc61d955 | claude-sonnet-5-5 | 1312339 | 1228095 | 84244 | 84196 | 48 | 13401 | 278.7 | ok | 3000.0 |
| 108 | a dental crown | 19e2bc98c966 | claude-opus-5-5 | 1952776 | 1852186 | 100590 | 100530 | 60 | 20799 | 480.2 | ok | 2000.0 |
| 109 | a knee implant | a3cfe82fec65 | claude-haiku-5-5 | 6016784 | 5847511 | 169273 | 169157 | 116 | 83578 | 608.0 | ok | 300000.0 |
| 110 | a knee implant | abfc9322bf45 | claude-sonnet-5-5 | 1395616 | 1309033 | 86583 | 86533 | 50 | 14138 | 375.9 | ok | 5000.0 |
| 111 | a knee implant | dab454dc286c | claude-opus-5-5 | 3597108 | 3462293 | 134815 | 134719 | 96 | 32092 | 628.6 | ok | 20000.0 |
| 112 | a walking cane with a rubber tip | 0a395bb30de1 | claude-haiku-5-5 | 3094530 | 2991802 | 102728 | 102648 | 80 | 44294 | 337.4 | ok |  |
| 113 | a walking cane with a rubber tip | a0f02c2ebe32 | claude-sonnet-5-5 | 1675546 | 1583666 | 91880 | 91822 | 58 | 16129 | 298.1 | ok | 20.0 |
| 114 | a walking cane with a rubber tip | 7a707bc2b7be | claude-opus-5-5 | 3745920 | 3615433 | 130487 | 130389 | 98 | 28698 | 550.6 | ok | 500.0 |
| 115 | a human femur | 1d55dafacb66 | claude-haiku-5-5 | 1628367 | 1517907 | 110460 | 110414 | 46 | 50563 | 291.5 | ok | 200.0 |
| 116 | a human femur | c1a77b68beaf | claude-sonnet-5-5 | 754334 | 682781 | 71553 | 71521 | 32 | 13647 | 212.3 | ok | 10.0 |
| 117 | a human femur | 66871e3cb342 | claude-opus-5-5 | 2098499 | 1997253 | 101246 | 101180 | 66 | 21778 | 458.0 | ok | 20.0 |
| 118 | a lumbar vertebra | 46b4311a3ba3 | claude-haiku-5-5 | 3687428 | 3560644 | 126784 | 126690 | 94 | 48660 | 612.4 | ok | 500.0 |
| 119 | a lumbar vertebra | 15ad6b962279 | claude-sonnet-5-5 | 2458410 | 2350636 | 107774 | 107696 | 78 | 20262 | 432.5 | ok | 5000.0 |
| 120 | a lumbar vertebra | f37f73faaa4a | claude-opus-5-5 | 2926024 | 2812374 | 113650 | 113568 | 82 | 23669 | 556.3 | ok | 200.0 |
| 121 | a skyscraper | f8ffbcd92e23 | claude-haiku-5-5 | 2375786 | 2268780 | 107006 | 106936 | 70 | 44167 | 454.9 | ok | 1000.0 |
| 122 | a skyscraper | 9d1a169f7ab9 | claude-sonnet-5-5 | 2255662 | 2153879 | 101783 | 101713 | 70 | 20357 | 350.1 | ok | 1000.0 |
| 123 | a skyscraper | 9740276dd004 | claude-opus-5-5 | 1140306 | 1056858 | 83448 | 83406 | 42 | 15855 | 220.1 | ok | 10000.0 |
| 124 | a thatched stone cottage | 80e6344274dc | claude-haiku-5-5 | 2801062 | 2676289 | 124773 | 124709 | 64 | 64208 | 553.3 | ok |  |
| 125 | a thatched stone cottage | 4a55ebcb13cb | claude-sonnet-5-5 | 2072736 | 1970545 | 102191 | 102123 | 68 | 17771 | 454.0 | ok | 50000.0 |
| 126 | a thatched stone cottage | b2778688a74c | claude-opus-5-5 | 1281098 | 1187488 | 93610 | 93568 | 42 | 19120 | 287.7 | ok | 1000000.0 |
| 127 | a lighthouse | 64b1430329d6 | claude-haiku-5-5 | 1800206 | 1696021 | 104185 | 104129 | 56 | 43806 | 278.3 | ok | 20000.0 |
| 128 | a lighthouse | 5eb7d1ae8d34 | claude-sonnet-5-5 | 1616623 | 1528778 | 87845 | 87785 | 60 | 15238 | 334.6 | ok | 5000.0 |
| 129 | a lighthouse | 4018fb655d42 | claude-opus-5-5 | 2020356 | 1915129 | 105227 | 105161 | 66 | 21673 | 429.0 | ok | 30000.0 |
| 130 | an igloo | 84500ed423bb | claude-haiku-5-5 | 1216876 | 1125889 | 90987 | 90943 | 44 | 34460 | 252.9 | ok | 10000.0 |
| 131 | an igloo | fc81ce53a534 | claude-sonnet-5-5 | 1165873 | 1077310 | 88563 | 88521 | 42 | 14170 | 422.0 | ok | 1000.0 |
| 132 | an igloo | 0f955499102d | claude-opus-5-5 | 2579046 | 2471703 | 107343 | 107267 | 76 | 20577 | 486.2 | ok | 1000.0 |
| 133 | a log cabin | 4fc429bc83ce | claude-haiku-5-5 | 1735496 | 1629228 | 106268 | 106212 | 56 | 41302 | 356.1 | ok | 500.0 |
| 134 | a log cabin | d7a3ffa841cc | claude-sonnet-5-5 | 2019259 | 1919639 | 99620 | 99552 | 68 | 16314 | 322.7 | ok | 20000.0 |
| 135 | a log cabin | 2662a8a839a1 | claude-opus-5-5 | 759416 | 681327 | 78089 | 78059 | 30 | 15049 | 227.3 | ok | 100000.0 |
| 136 | the Leaning Tower of Pisa | 590fcfcc5017 | claude-haiku-5-5 | 4117906 | 3990917 | 126989 | 126885 | 104 | 52414 | 540.1 | ok | 10000.0 |
| 137 | the Leaning Tower of Pisa | 71b205af02b2 | claude-sonnet-5-5 | 1833401 | 1741696 | 91705 | 91639 | 66 | 14257 | 380.8 | ok | 5000.0 |
| 138 | the Leaning Tower of Pisa | 566b483089a4 | claude-opus-5-5 | 2019884 | 1918755 | 101129 | 101061 | 68 | 23599 | 460.9 | ok | 5000.0 |
| 139 | the Pantheon | c13b3dfbc53c | claude-haiku-5-5 | 4749596 | 4586640 | 162956 | 162852 | 104 | 78570 | 741.6 | ok | 20000.0 |
| 140 | the Pantheon | 13539e6f99cf | claude-sonnet-5-5 | 2584187 | 2473915 | 110272 | 110192 | 80 | 24241 | 545.7 | ok | 10000.0 |
| 141 | the Pantheon | 451495907e86 | claude-opus-5-5 | 2350742 | 2242853 | 107889 | 107817 | 72 | 27009 | 488.8 | ok | 20000.0 |
| 142 | Himeji Castle | 199a6696c020 | claude-haiku-5-5 | 4853958 | 4705797 | 148161 | 148049 | 112 | 51652 | 713.7 | ok | 30000.0 |
| 143 | Himeji Castle | 5b999f37faa9 | claude-sonnet-5-5 | 924571 | 848524 | 76047 | 76009 | 38 | 11481 | 202.9 | ok | 30000.0 |
| 144 | Himeji Castle | 284b39c65126 | claude-opus-5-5 | 2246183 | 2133636 | 112547 | 112481 | 66 | 19956 | 499.1 | ok | 10000.0 |
| 145 | a natural stone arch | 834670958931 | claude-haiku-5-5 | 1715785 | 1620580 | 95205 | 95151 | 54 | 35298 | 282.4 | ok | 30000.0 |
| 146 | a natural stone arch | 3ca15719c68b | claude-sonnet-5-5 | 1373392 | 1290588 | 82804 | 82752 | 52 | 12171 | 306.8 | ok | 30000.0 |
| 147 | a natural stone arch | 6a0218ba407f | claude-opus-5-5 | 1499897 | 1406560 | 93337 | 93287 | 50 | 14026 | 272.8 | ok | 10000.0 |
| 148 | the Taj Mahal | ad2c7bf9dedf | claude-haiku-5-5 | 2060361 | 1946447 | 113914 | 113856 | 58 | 48982 | 406.2 | ok | 5000.0 |
| 149 | the Taj Mahal | 2c78259d9002 | claude-sonnet-5-5 | 1232493 | 1148430 | 84063 | 84017 | 46 | 13165 | 235.7 | ok | 3000.0 |
| 150 | the Taj Mahal | 70159b049582 | claude-opus-5-5 | 862990 | 783718 | 79272 | 79238 | 34 | 16762 | 245.0 | ok | 3000.0 |
