# T075–T104 Architecture × Representation Matrix

Platform: `DL_FOLDLOCAL_COSINE_V3`. Primary metric: TEST_mean. Sign: lower MAE better.

| Code | Arch | Rep | Merge | Params | VAL_mean | TEST_mean | TEST_worst | Overall |
|---|---|---|---|---:|---:|---:|---:|---:|
| EXP-T075 | ARCH-1 | ABLINGUA | concat | 501633 | 3.0462 | 3.4020 | 3.5121 | 3.3831 |
| EXP-T076 | ARCH-2 | ABLINGUA | — | 485121 | 2.9997 | 3.3244 | 3.4057 | 3.3407 |
| EXP-T077 | ARCH-3 | ABLINGUA | concat | 501633 | 3.0055 | 3.3711 | 3.3728 | 3.3865 |
| EXP-T078 | ARCH-4 | ABLINGUA | concat | 501633 | 3.0350 | 3.3924 | 3.4257 | 3.3493 |
| EXP-T079 | ARCH-5 | ABLINGUA | concat | 567683 | 2.9596 | 3.2517 | 3.2671 | 3.4324 |
| EXP-T080 | ARCH-1 | ABLINGUA | mean | 485249 | 3.0260 | 3.2835 | 3.3067 | 3.4562 |
| EXP-T081 | ARCH-3 | ABLINGUA | mean | 485249 | 2.9750 | 3.4556 | 3.4798 | 3.3759 |
| EXP-T082 | ARCH-4 | ABLINGUA | mean | 485249 | 3.0146 | 3.3037 | 3.3641 | 3.3967 |
| EXP-T083 | ARCH-5 | ABLINGUA | mean | 551299 | 3.0421 | 3.3078 | 3.3825 | 3.4340 |
| EXP-T084 | ARCH-6 | ABLINGUA | concat | 567681 | 3.0277 | 3.2933 | 3.3221 | 3.3848 |
| EXP-T085 | ARCH-6 | ABLINGUA | mean | 551297 | 3.0148 | 3.4724 | 3.5997 | 3.2845 |
| EXP-T086 | ARCH-7 | ABLINGUA | concat | 567681 | 3.0046 | 3.3558 | 3.3735 | 3.3832 |
| EXP-T087 | ARCH-7 | ABLINGUA | mean | 551297 | 3.0074 | 3.2559 | 3.2764 | 3.3887 |
| EXP-T088 | ARCH-8 | ABLINGUA | concat | 567681 | 3.0270 | 3.3883 | 3.4453 | 3.3839 |
| EXP-T089 | ARCH-8 | ABLINGUA | mean | 551297 | 3.0299 | 3.2961 | 3.3328 | 3.2590 |
| EXP-T090 | ARCH-1 | SCRATCH | concat | 340481 | 2.9491 | 3.5504 | 3.6792 | 3.5830 |
| EXP-T091 | ARCH-1 | SCRATCH | mean | 324097 | 2.9968 | 3.4825 | 3.5995 | 3.5227 |
| EXP-T092 | ARCH-2 | SCRATCH | — | 323969 | 2.9286 | 3.3141 | 3.4731 | 3.4496 |
| EXP-T093 | ARCH-3 | SCRATCH | concat | 340481 | 3.0702 | 3.3201 | 3.3321 | 3.4322 |
| EXP-T094 | ARCH-3 | SCRATCH | mean | 324097 | 3.1087 | 3.2979 | 3.4077 | 3.5774 |
| EXP-T095 | ARCH-4 | SCRATCH | concat | 340481 | 3.0353 | 3.4022 | 3.4959 | 3.5541 |
| EXP-T096 | ARCH-4 | SCRATCH | mean | 324097 | 3.0722 | 3.2581 | 3.2638 | 3.5394 |
| EXP-T097 | ARCH-5 | SCRATCH | concat | 406531 | 2.9972 | 3.3701 | 3.3784 | 3.5063 |
| EXP-T098 | ARCH-5 | SCRATCH | mean | 390147 | 3.0835 | 3.3061 | 3.4037 | 3.4711 |
| EXP-T099 | ARCH-6 | SCRATCH | concat | 406529 | 3.0439 | 3.3228 | 3.4227 | 3.4708 |
| EXP-T100 | ARCH-6 | SCRATCH | mean | 390145 | 3.0040 | 3.4274 | 3.4891 | 3.4316 |
| EXP-T101 | ARCH-7 | SCRATCH | concat | 406529 | 2.9383 | 3.3409 | 3.4915 | 3.4149 |
| EXP-T102 | ARCH-7 | SCRATCH | mean | 390145 | 2.9671 | 3.3520 | 3.4288 | 3.5722 |
| EXP-T103 | ARCH-8 | SCRATCH | concat | 406529 | 3.0871 | 3.4650 | 3.4752 | 3.4437 |
| EXP-T104 | ARCH-8 | SCRATCH | mean | 390145 | 3.0033 | 3.3533 | 3.3939 | 3.4313 |

## Best models (internal TEST_mean; robustness = TEST_worst)

| Scope | Code | Architecture | Merge | TEST_mean | TEST_worst |
|---|---|---|---|---:|---:|
| Best AbLingua | EXP-T079 | ARCH-5 gated residue cross-attn | CONCAT | 3.2517 | 3.2671 |
| Best Scratch | EXP-T096 | ARCH-4 joint chain-specific dual REG | MEAN | 3.2581 | 3.2638 |
| Best overall | EXP-T079 | ARCH-5 gated residue cross-attn | CONCAT | 3.2517 | 3.2671 |
| Strong AbLingua runner-up | EXP-T087 | ARCH-7 REG-only cross-attn | MEAN | 3.2559 | 3.2764 |

## ARCH-6 vs ARCH-8 (true cross-chain vs matched extra attention)

| Rep | ARCH-6 (cross) | TEST_mean | ARCH-8 (within) | TEST_mean | Δ (6−8; neg ⇒ cross better) |
|---|---|---:|---|---:|---:|
| AbLingua CONCAT | T084 | 3.2933 | T088 | 3.3883 | −0.0950 |
| Scratch CONCAT | T099 | 3.3228 | T103 | 3.4650 | −0.1422 |

Cross-chain bridge beats the matched within-chain control on TEST_mean for both representations (CONCAT).

## Geometry readiness (conceptual only; not run)

Residue cross-attention (ARCH-5/6) remains the cleanest locus for a future H–L Cα distance bias on cross-attention logits. REG-only (ARCH-7) is less naturally aligned with pairwise residue geometry.

External columns use Primary mean as the canonical aggregation. Public/Private are post-competition diagnostics and do not rewrite the internal ranking.
