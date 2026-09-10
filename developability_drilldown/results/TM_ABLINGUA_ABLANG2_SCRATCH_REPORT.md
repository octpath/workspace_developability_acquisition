# TmApp AbLingua / AbLang2 / Scratch report

Platform: `DL_FOLDLOCAL_COSINE_V3`. AbLang2 `PRECONTEXTUALIZED_ACROSS_CHAINS=NO`.

| Representation | Architecture | Merge | TEST_P | TEST_S | TEST_mean | TEST_worst | Pub | Priv | Overall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| None | None | concat | 3.5121 | 3.2919 | 3.4020 | 3.5121 | 3.4788 | 3.2874 | 3.3831 |
| None | None | concat | 3.2431 | 3.4057 | 3.3244 | 3.4057 | 3.5227 | 3.1587 | 3.3407 |
| None | None | concat | 3.3728 | 3.3694 | 3.3711 | 3.3728 | 3.4870 | 3.2859 | 3.3865 |
| None | None | concat | 3.4257 | 3.3591 | 3.3924 | 3.4257 | 3.4570 | 3.2416 | 3.3493 |
| None | None | concat | 3.2363 | 3.2671 | 3.2517 | 3.2671 | 3.5364 | 3.3285 | 3.4324 |
| ABLINGUA | ARCH-1 | mean | 3.2602 | 3.3067 | 3.2835 | 3.3067 | 3.5794 | 3.3329 | 3.4562 |
| ABLINGUA | ARCH-3 | mean | 3.4314 | 3.4798 | 3.4556 | 3.4798 | 3.5145 | 3.2373 | 3.3759 |
| ABLINGUA | ARCH-4 | mean | 3.2432 | 3.3641 | 3.3037 | 3.3641 | 3.4994 | 3.2940 | 3.3967 |
| ABLINGUA | ARCH-5 | mean | 3.3825 | 3.2331 | 3.3078 | 3.3825 | 3.5461 | 3.3220 | 3.4340 |
| ABLINGUA | ARCH-6 | concat | 3.3221 | 3.2645 | 3.2933 | 3.3221 | 3.5280 | 3.2416 | 3.3848 |
| ABLINGUA | ARCH-6 | mean | 3.5997 | 3.3450 | 3.4724 | 3.5997 | 3.4151 | 3.1539 | 3.2845 |
| ABLINGUA | ARCH-7 | concat | 3.3735 | 3.3381 | 3.3558 | 3.3735 | 3.5580 | 3.2083 | 3.3832 |
| ABLINGUA | ARCH-7 | mean | 3.2354 | 3.2764 | 3.2559 | 3.2764 | 3.5127 | 3.2646 | 3.3887 |
| ABLINGUA | ARCH-8 | concat | 3.4453 | 3.3313 | 3.3883 | 3.4453 | 3.5093 | 3.2585 | 3.3839 |
| ABLINGUA | ARCH-8 | mean | 3.3328 | 3.2594 | 3.2961 | 3.3328 | 3.4251 | 3.0929 | 3.2590 |
| SCRATCH | ARCH-1 | concat | 3.4217 | 3.6792 | 3.5504 | 3.6792 | 3.6379 | 3.5281 | 3.5830 |
| SCRATCH | ARCH-1 | mean | 3.3655 | 3.5995 | 3.4825 | 3.5995 | 3.6297 | 3.4156 | 3.5227 |
| SCRATCH | ARCH-2 | — | 3.1551 | 3.4731 | 3.3141 | 3.4731 | 3.5329 | 3.3663 | 3.4496 |
| SCRATCH | ARCH-3 | concat | 3.3321 | 3.3082 | 3.3201 | 3.3321 | 3.4964 | 3.3679 | 3.4322 |
| SCRATCH | ARCH-3 | mean | 3.1880 | 3.4077 | 3.2979 | 3.4077 | 3.7377 | 3.4171 | 3.5774 |
| SCRATCH | ARCH-4 | concat | 3.3084 | 3.4959 | 3.4022 | 3.4959 | 3.5969 | 3.5112 | 3.5541 |
| SCRATCH | ARCH-4 | mean | 3.2524 | 3.2638 | 3.2581 | 3.2638 | 3.6186 | 3.4603 | 3.5394 |
| SCRATCH | ARCH-5 | concat | 3.3784 | 3.3618 | 3.3701 | 3.3784 | 3.5840 | 3.4286 | 3.5063 |
| SCRATCH | ARCH-5 | mean | 3.2085 | 3.4037 | 3.3061 | 3.4037 | 3.5901 | 3.3521 | 3.4711 |
| SCRATCH | ARCH-6 | concat | 3.4227 | 3.2228 | 3.3228 | 3.4227 | 3.5666 | 3.3749 | 3.4708 |
| SCRATCH | ARCH-6 | mean | 3.3656 | 3.4891 | 3.4274 | 3.4891 | 3.5460 | 3.3172 | 3.4316 |
| SCRATCH | ARCH-7 | concat | 3.1904 | 3.4915 | 3.3409 | 3.4915 | 3.5459 | 3.2840 | 3.4149 |
| SCRATCH | ARCH-7 | mean | 3.2751 | 3.4288 | 3.3520 | 3.4288 | 3.6440 | 3.5003 | 3.5722 |
| SCRATCH | ARCH-8 | concat | 3.4752 | 3.4548 | 3.4650 | 3.4752 | 3.6367 | 3.2507 | 3.4437 |
| SCRATCH | ARCH-8 | mean | 3.3127 | 3.3939 | 3.3533 | 3.3939 | 3.4278 | 3.4349 | 3.4313 |
| ABLINGUA | ARCH-6G | concat | 3.3522 | 3.3000 | 3.3261 | 3.3522 | 3.5350 | 3.2098 | 3.3724 |
| ABLINGUA | ARCH-6G | mean | 3.3516 | 3.2777 | 3.3147 | 3.3516 | 3.5742 | 3.2843 | 3.4293 |
| SCRATCH | ARCH-6G | concat | 3.4090 | 3.2267 | 3.3179 | 3.4090 | 3.5201 | 3.3583 | 3.4392 |
| SCRATCH | ARCH-6G | mean | 3.2860 | 3.2921 | 3.2890 | 3.2921 | 3.4788 | 3.3898 | 3.4343 |
| ABLANG2 | ARCH-1 | concat | 3.1965 | 3.3070 | 3.2518 | 3.3070 | 3.1694 | 3.1538 | 3.1616 |
| ABLANG2 | ARCH-1 | mean | 3.2029 | 3.2803 | 3.2416 | 3.2803 | 3.2524 | 3.0645 | 3.1585 |
| ABLANG2 | ARCH-2 | — | 3.1224 | 3.3701 | 3.2463 | 3.3701 | 3.2258 | 3.0463 | 3.1361 |
| ABLANG2 | ARCH-3 | concat | 3.3636 | 3.1883 | 3.2760 | 3.3636 | 3.0349 | 3.0748 | 3.0548 |
| ABLANG2 | ARCH-3 | mean | 2.9951 | 3.2831 | 3.1391 | 3.2831 | 3.1054 | 2.9991 | 3.0523 |
| ABLANG2 | ARCH-4 | concat | 3.1612 | 3.3588 | 3.2600 | 3.3588 | 3.2163 | 3.1317 | 3.1740 |
| ABLANG2 | ARCH-4 | mean | 3.2407 | 3.3118 | 3.2763 | 3.3118 | 3.1967 | 2.9867 | 3.0917 |
| ABLANG2 | ARCH-6 | concat | 3.2651 | 3.4072 | 3.3362 | 3.4072 | 3.3088 | 3.0744 | 3.1916 |
| ABLANG2 | ARCH-6 | mean | 3.2382 | 3.3408 | 3.2895 | 3.3408 | 3.2692 | 3.0780 | 3.1736 |
| ABLANG2 | ARCH-6G | concat | 3.2175 | 3.3819 | 3.2997 | 3.3819 | 3.3729 | 3.1206 | 3.2468 |
| ABLANG2 | ARCH-6G | mean | 3.3647 | 3.2835 | 3.3241 | 3.3647 | 3.3046 | 3.1194 | 3.2120 |
| ABLANG2 | ARCH-7 | concat | 3.1087 | 3.2912 | 3.1999 | 3.2912 | 3.2409 | 2.9866 | 3.1137 |
| ABLANG2 | ARCH-7 | mean | 3.0232 | 3.2521 | 3.1377 | 3.2521 | 3.2474 | 3.1936 | 3.2205 |
| ABLANG2 | ARCH-8 | concat | 3.1770 | 3.5268 | 3.3519 | 3.5268 | 3.2870 | 3.1023 | 3.1946 |
| ABLANG2 | ARCH-8 | mean | 3.3602 | 3.4919 | 3.4260 | 3.4919 | 3.2677 | 3.0623 | 3.1650 |

## Questions (fill after inspection)

1. Does AbLang2 reproduce historical strength under V3?
2. Is AbLang2 consistently stronger, or architecture-dependent?
3. Does Scratch remain competitive?
4. Does AbLang2 prefer MEAN or CONCAT?
5. Does explicit H/L communication help AbLang2?
6. Given PRECONTEXTUALIZED_ACROSS_CHAINS=NO, does downstream H/L communication help?
7. Does geometry help AbLang2?

