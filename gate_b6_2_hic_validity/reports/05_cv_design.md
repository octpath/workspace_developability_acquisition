# Phase 5 — Continuous-Target-Aware Grouped CV

## Current Group CV
Existing folds show HIC range/tail imbalance (tail-count SD≈1.3; fold Pearson SD≈0.16).

## Scheme stability (mean across models)

| scheme           |    mae_sd |   pear_sd |   bal_mean_sd |   tail_sd |
|:-----------------|----------:|----------:|--------------:|----------:|
| CURRENT_GROUP_CV | 0.0730304 |  0.161865 |     0.0841756 |  1.26564  |
| SGKF_Q5          | 0.0856129 |  0.152798 |     0.0479403 |  1.30689  |
| SGKF_Q7          | 0.0741702 |  0.122628 |     0.0495467 |  1.1083   |
| SGKF_Q9          | 0.0914067 |  0.151049 |     0.0706456 |  0.894427 |
| TAIL_Q7          | 0.0679075 |  0.162511 |     0.0493792 |  0.894427 |

## Preferred Train-only scheme
**SGKF_Q7**

Sturges (~8–9) motivates 7–9 strata; equal-frequency quantile strata used (not equal-width Sturges bins).
One-shot Public/Private confirmation only after Train-only lock (no re-selection).
