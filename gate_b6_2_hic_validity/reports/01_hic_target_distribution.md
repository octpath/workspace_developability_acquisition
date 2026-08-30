# Phase 1 — HIC Target Distribution

## Summary
HIC (N=324) is **strongly right-skewed** (skew=2.22, excess kurtosis=5.53).
Mean (9.377) exceeds median (9.098) by 0.279.

| Role | N | mean | median | SD | IQR | min | max | skew |
|------|---|------|--------|----|-----|-----|-----|------|
| Full | 324 | 9.377 | 9.098 | 0.832 | 0.705 | 8.465 | 13.856 | 2.22 |
| Train | 162 | 9.395 | 9.115 | 0.810 | 0.720 | 8.523 | 12.740 | 1.93 |
| Public | 81 | 9.406 | 9.164 | 0.912 | 0.694 | 8.571 | 13.856 | 2.63 |
| Private | 81 | 9.314 | 8.993 | 0.801 | 0.656 | 8.465 | 12.449 | 2.16 |

## Central concentration (Train)
- Within median ± 0.25 IQR: 33.3%
- Within median ± 0.50 IQR: 63.0%
- Within median ± 1.00 IQR: 81.5%
- Narrowest 50% width: 0.422 ([8.697, 9.119])
- Narrowest 80% width: 1.208
- Narrowest 90% width: 1.980

## Bin diagnostics (Train)
Sturges≈9, Freedman–Diaconis≈16, Scott≈9.

## Phase-1 answers
1. **Yes — strongly right-skewed.**
2. **Central mass is concentrated** (~63% within median±0.5 IQR).
3. **High-HIC tail is small** (~5–10% of Train; see `hic_tail_counts.csv`).
4. Tail is sparse but present in Train/Public/Private; Public holds the extreme max.
5. **Mean > median** (right tail pulls the mean).
6. **Yes — shape is consistent with MAE favoring central predictors.**
