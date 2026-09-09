# Reproducibility status review

Generated from existing artifacts/logs only (no heavy retrain).
Baseline HEAD at review: `7a5c17b0`.

## Definitions used

- **REPRODUCED**: frozen procedure re-executed; Primary/Shadow/Test predictions match historical within model-specific tolerance.
- **RESULT_VERIFIED**: historical (or hybrid historical) predictions exist and recompute authority scores; full prediction reproduction vs historical preds not demonstrated.
- **UNVERIFIED_HISTORICAL**: neither training reproduction nor sufficient historical prediction evidence.

## A. Previous status counts (old 77, before this review)

| status | n |
|---|---:|
| RESULT_VERIFIED | 73 |
| UNVERIFIED_HISTORICAL | 4 |
| REPRODUCED | 0 |

(New 40 classical remained REPRODUCED 40/40 throughout.)

## B. Reviewed experiment count

- Old 77 fully reviewed against `EXPERIMENT_ARTIFACT_COMPLETENESS.csv`, `experiments.csv`, `XGB_REPRODUCTION_AUDIT.csv`, `classical_logs/artifact_completion.log`, fitted_params, configs.

## C. Promotions: RESULT_VERIFIED → REPRODUCED

**n = 6** (historical FULL Linear Top-6 only)

Evidence: first-pass training reproduction in `artifact_completion.log` recorded `FULL EXP-* pred_delta`:

| code | pred_max_delta | score_max_delta | tolerance |
|---|---:|---:|---|
| EXP-T001 | 5.684e-14 | 0 | classical ≤1e-10 |
| EXP-T002 | 5.684e-14 | ~4e-16 | classical ≤1e-10 |
| EXP-T003 | 5.684e-14 | ~4e-16 | classical ≤1e-10 |
| EXP-H001 | 5.329e-15 | ~2e-16 | classical ≤1e-10 |
| EXP-H002 | 3.553e-15 | ~2e-16 | classical ≤1e-10 |
| EXP-H003 | 3.553e-15 | ~2e-16 | classical ≤1e-10 |

These compared re-fit OOF/Test predictions to the already-stored historical prediction triplet.

## D. Remaining RESULT_VERIFIED (old 77) — why

| reason class | n | notes |
|---|---:|---|
| TRANSFORMER historical preds only | 29 | OOF/Test verify scores; no historical weight retrain (implementation equivalence ≠ training reproduction) |
| XGBOOST score audit only | 6 | `XGB_REPRODUCTION_AUDIT.csv` PASS = score recompute from stored preds; training re-fit not logged with pred deltas |
| LINEAR reconstructable, scores match after materialize | 27 | Procedure re-run wrote preds that match authority scores, but **no pre-existing historical OOF/Test files** to compute prediction_max_delta against → not promoted |
| LINEAR LASSO/Test hybrid | 5 | see §F |

Total RESULT_VERIFIED after review (old 77): **67**

## E. Remaining UNVERIFIED (4)

| code | reason |
|---|---|
| EXP-T019 | OPENMM SCORE_ONLY; no Test features / incomplete preds |
| EXP-T020 | OPENMM SCORE_ONLY; no Test features / incomplete preds |
| EXP-T021 | FENNIX: endgame Test only; no historical OOF; reproduced OOF mismatches authority CV |
| EXP-T022 | FENNIX: same as T021 |

## F. LASSO / Test-mismatch special cases (5)

| code | status | rationale |
|---|---|---|
| EXP-T009 | RESULT_VERIFIED | CV OOF matches authority; Test from endgame historical verifies Public/Private; full OOF+Test reproduction vs a single historical triplet not shown |
| EXP-T018 | RESULT_VERIFIED | same pattern |
| EXP-H007 | RESULT_VERIFIED | same pattern |
| EXP-H012 | RESULT_VERIFIED | same pattern (endgame Test / CV-matched OOF) |
| EXP-H013 | RESULT_VERIFIED | same pattern |

**Not promoted to REPRODUCED** (no new retrain; hybrid provenance).

## G. FENNIX / OPENMM

Confirmed **UNVERIFIED_HISTORICAL** as above. No status change.

## H. Final counts (all 117)

| status | n |
|---|---:|
| REPRODUCED | **46** (40 new classical + 6 FULL Linear) |
| RESULT_VERIFIED | **67** |
| UNVERIFIED_HISTORICAL | **4** |
| canonical_benchmark_eligible YES | **113** |

Scores, feature parquets, and prediction file bytes were **not** modified in this review (status/metadata only).
