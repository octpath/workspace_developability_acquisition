# Experiment reproduction and artifact completion

Generated: 2026-09-09T18:38:31Z
Mode: finalize-only from on-disk artifacts (no heavy retrain)
Total audited: **117**

## A. New 40 classical

- SHAREABLE_COMPLETE: **40** / 40
- feature+pred triplet: **40** / 40
- reproducibility: `{'REPRODUCED': 40}`
- score_max_delta max: **3.9643615012963096e-07**
- prediction_max_delta max: **3.813e-06**

## B. Previous fixed-length (n=48)

- shareability: `{'SHAREABLE_COMPLETE': 46, 'HISTORICAL_ONLY': 2}`
- reproducibility: `{'RESULT_VERIFIED': 44, 'UNVERIFIED_HISTORICAL': 4}`

## C. Historical Transformers (n=29)

- shareability: `{'SHAREABLE_PARTIAL': 29}`
- reproducibility: `{'RESULT_VERIFIED': 29}`

## LASSO / mismatch reclassification

- **EXP-T009**: RESULT_VERIFIED (score_d=2.6645352591003757e-15, pred_d=nan, blocker=test_from_endgame_historical; OOF reproduced CV-matched)
- **EXP-T018**: RESULT_VERIFIED (score_d=1.3322676295501878e-15, pred_d=nan, blocker=test_from_endgame_historical; OOF reproduced CV-matched)
- **EXP-T021**: UNVERIFIED_HISTORICAL (score_d=0.017998474490365624, pred_d=nan, blocker=score_delta=0.017998474490365624)
- **EXP-T022**: UNVERIFIED_HISTORICAL (score_d=0.02009382334546883, pred_d=nan, blocker=score_delta=0.02009382334546883)
- **EXP-H007**: RESULT_VERIFIED (score_d=2.220446049250313e-16, pred_d=nan, blocker=test_from_endgame_historical; OOF reproduced CV-matched)
- **EXP-H012**: RESULT_VERIFIED (score_d=5.551115123125783e-16, pred_d=nan, blocker=)
- **EXP-H013**: RESULT_VERIFIED (score_d=2.220446049250313e-16, pred_d=nan, blocker=test_from_endgame_historical; OOF reproduced CV-matched)

Notes:
- EXP-T009: CV matched via reproduced OOF; test from endgame → RESULT_VERIFIED
- EXP-T018: CV matched via reproduced OOF; test from endgame → RESULT_VERIFIED
- EXP-H007: CV matched via reproduced OOF; test from endgame → RESULT_VERIFIED
- EXP-H013: CV matched via reproduced OOF; test from endgame → RESULT_VERIFIED

## FENNIX EXP-T021 / EXP-T022

- Historical endgame test predictions exist, but no historical OOF triplet.
- Reproduced OOF does not match authority CV within tolerance.
- Final: **UNVERIFIED_HISTORICAL**, canonical_benchmark_eligible=NO.

## Totals

- shareability: `{'SHAREABLE_COMPLETE': 86, 'SHAREABLE_PARTIAL': 29, 'HISTORICAL_ONLY': 2}`
- reproducibility: `{'RESULT_VERIFIED': 73, 'REPRODUCED': 40, 'UNVERIFIED_HISTORICAL': 4}`
- canonical_benchmark_eligible YES: **113**
- per-EXP feature parquets: **86**
- prediction triplets: **115**
- Optuna: unused (fold grid-search); fitted params in `*.fitted_params.json`

## Flags

- ARTIFACT_COMPLETENESS_AUDITED = YES
- CANONICAL_RESULTS_VERIFIED = YES (eligible subset)
- NEW_ARCHITECTURE_READY = YES
