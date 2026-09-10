# T090–T104 Scratch architecture report

Historical scratch FULL semantics: AA emb + pos + chain + IMGT + region (additive → d_model).

| Code | Arch | Merge | VAL_mean | TEST_mean | TEST_worst | Overall |
|---|---|---|---:|---:|---:|---:|
| EXP-T090 | ARCH-1 | concat | 2.9491 | 3.5504 | 3.6792 | 3.5830 |
| EXP-T091 | ARCH-1 | mean | 2.9968 | 3.4825 | 3.5995 | 3.5227 |
| EXP-T092 | ARCH-2 | — | 2.9286 | 3.3141 | 3.4731 | 3.4496 |
| EXP-T093 | ARCH-3 | concat | 3.0702 | 3.3201 | 3.3321 | 3.4322 |
| EXP-T094 | ARCH-3 | mean | 3.1087 | 3.2979 | 3.4077 | 3.5774 |
| EXP-T095 | ARCH-4 | concat | 3.0353 | 3.4022 | 3.4959 | 3.5541 |
| EXP-T096 | ARCH-4 | mean | 3.0722 | 3.2581 | 3.2638 | 3.5394 |
| EXP-T097 | ARCH-5 | concat | 2.9972 | 3.3701 | 3.3784 | 3.5063 |
| EXP-T098 | ARCH-5 | mean | 3.0835 | 3.3061 | 3.4037 | 3.4711 |
| EXP-T099 | ARCH-6 | concat | 3.0439 | 3.3228 | 3.4227 | 3.4708 |
| EXP-T100 | ARCH-6 | mean | 3.0040 | 3.4274 | 3.4891 | 3.4316 |
| EXP-T101 | ARCH-7 | concat | 2.9383 | 3.3409 | 3.4915 | 3.4149 |
| EXP-T102 | ARCH-7 | mean | 2.9671 | 3.3520 | 3.4288 | 3.5722 |
| EXP-T103 | ARCH-8 | concat | 3.0871 | 3.4650 | 3.4752 | 3.4437 |
| EXP-T104 | ARCH-8 | mean | 3.0033 | 3.3533 | 3.3939 | 3.4313 |
