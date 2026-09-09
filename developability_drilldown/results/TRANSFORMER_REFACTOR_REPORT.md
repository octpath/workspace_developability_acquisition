# Phase 2A Transformer refactor & historical backfill

## Status

Complete. Baseline registry commit: `81c3a361`. No new scientific experiments.

## Old source → new module

| Old (immutable bundle) | New |
|------------------------|-----|
| `advanced_models/models/annotated_transformer.py` | `models/antibody_transformer/model.py` |
| `advanced_models/models/feature_fusion.py` | `models/antibody_transformer/fusion.py` |
| `advanced_models/config.py` (constants) | `models/antibody_transformer/config.py` |
| `advanced_models/data.py` | `models/antibody_transformer/data.py` |
| `advanced_models/cv.py` (transformer parts) | `models/antibody_transformer/training.py` |
| `advanced_models/metrics.py` | `models/antibody_transformer/metrics.py` |
| `advanced_models/presets.json` | `models/antibody_transformer/presets.json` |
| — | `models/antibody_transformer/equivalence.py` |

`top_models_feature_bundle/` was not modified.

## Inventory (29)

- TmApp: 19 (`EXP-T026`…`EXP-T044`)
- HIC: 10 (`EXP-H024`…`EXP-H033`)
- Groups: Phase1 sequence 10, Phase1 fusion 12, AbLang2 follow-up 7

## Scores

Authority: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv` (all 29 `model_id`s).

## Predictions

Local untracked sources under `advanced_outputs/` (and `ablang2_followup/`) copied to
`experiments/predictions/EXP-*/{oof_primary,oof_shadow,test}.csv`.
Source path + sha256 + value deltas recorded in `TRANSFORMER_BACKFILL_AUDIT.csv`.
No retraining.

## Representation

All 29: `representation_status=HISTORICAL_UNAVAILABLE`.
No `experiments/features/EXP-*.parquet` for historical Transformers.
`feature_path` / `feature_space` empty; inputs via `input_space` + `input_asset_ref`.
Future seed aggregation: **UNDECIDED**.

## TRANSFORMER FULL semantics

Family-specific (Linear/XGB FULL unchanged; parquet still required):

- config
- Primary/Shadow OOF + Test predictions
- authoritative scores
- input/residue provenance
- explicit `representation_status`

## Selection policy

All 29: `CV_SELECTED_POSTCOMP_EVALUATED` with evidence from MBS
`public_private_role=POSTMORTEM_ONLY`, ADVANCED notes, AbLang2 follow-up plan.
`current_evaluation_mode=POSTCOMP_EXPLORATORY` for all.

## License (conservative aggregate)

- Scratch-only sequence (5): `OK`
- Frozen PLM and/or fusion FS branches (24): `REVIEW`

## Equivalence

Unit tests compare old vs new: `forward_repr`, final prediction, one optimizer step
(loss / gradients / updated parameters). Tolerance ≤ 1e-6.

## Storage

No checkpoints, no duplicated residue `.npy`, no fusion feature parquet copies.
Expected Git delta ~1–3 MB (predictions + configs + code + audit).

## Next codes

- EXP-T045 / EXP-H034 / EXP-M001 (reserved, untouched)
