# Ensemble CV integrity audit (Gate B5)

## Verdict

**B4 OOF ensemble CV estimates were OPTIMISTIC (biased high for performance / low MAE).**

They must **not** be treated as unbiased nested-CV ceilings.

## What B4 did

In `gate_b4_absolute/scripts/02_nested_finalists_oneshot.py` → `build_residuals_ensembles`:

1. Base-model OOF predictions were generated (Train-only; legitimate Level-1 OOF).
2. Stackers were fit on **ALL** OOF rows + **ALL** Train labels:
   - `nnls(M[mask], y[mask])` for `ENSEMBLE_NNLS`
   - `Ridge.fit(M[mask], y[mask])` for `ENSEMBLE_RIDGE`
3. Stacker predictions `pred = M @ w` (or Ridge) were then scored by slicing those
   **same** OOF rows into outer folds.

Although Level-1 base predictions are out-of-fold, the **Level-2 stacker is
evaluated in-sample** on the meta-training rows.

Mean/median ensembles of OOF predictions are less biased (no fitted stacker),
but NNLS/Ridge stack scores reported in B4 (~0.469 HIC, ~2.615 TmApp) are
**not** valid outer-CV estimates.

## Correct protocol (this Gate)

For each OUTER fold:

```
OUTER TRAIN
  → inner grouped OOF for each base (OUTER TRAIN only)
  → fit stacker on inner-OOF
  → refit bases on OUTER TRAIN
  → predict OUTER VALIDATION
  → apply frozen stacker
  → score OUTER VALIDATION
```

Outer validation never influences base HPs, membership, weights, or calibration.

## B4 reported (optimistic) values

| Target | Model | Reported CV MAE | Reported CV Pearson |
|--------|-------|----------------:|--------------------:|
| HIC | ENSEMBLE_NNLS/oof_nnls | ≈ 0.469 | ≈ 0.548 |
| TmApp | ENSEMBLE_RIDGE/oof_ridge | ≈ 2.615 | ≈ 0.596 |

Corrected nested values are in `corrected_nested_ensemble_results.md` and
`metrics/corrected_nested_ensemble.csv`.
