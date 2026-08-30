# Learned pooling results

## Setup
- Encoder: **frozen** ESM-2 t30 150M (fair-esm)
- Used for **both** HIC and TmApp (AbLang2 token API not used; documented limitation)
- Heads: scalar token weighting, single-query attention pooling, heuristic region-gated pooling
- Validation: 3-fold grouped CV × 2 seeds; early stopping on internal group split
- Region masks: **heuristic** length fractions (not ANARCI) — organizer diagnostic only

## Results (Train-CV MAE)

| Target | Best pool | MAE | Pearson (Fisher-z mean) | Beat frozen baseline? |
|--------|-----------|----:|------------------------:|-----------------------|
| HIC | attn h=64 | ≈0.55 | ≈0.19 | **NO** (structure/PLM ≈0.50) |
| TmApp | region h=64 | ≈4.22 | ≈0.16 | **NO** (BIO/fusion ≈3.2) |

## Conclusion
Frozen learned pooling **did not** raise the organizer ceiling under MAE. Prefer B4-style structure/PLM/fusion features for packaging baselines.
