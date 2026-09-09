# Registry migration report — permanent EXP codes

**Date:** 2026-09-09  
**Baseline commit:** `48dfdd21` (Phase 1 catalog)  
**Migration script:** `scripts/migrate_registry_exp_codes.py`

## Before → after schema

### Identity
| Before | After |
|--------|-------|
| `experiment_id` (only) | `experiment_code` (canonical) + `experiment_id` (descriptive) |
| `feature_recipe` | `feature_set_id` + `source_recipe_id` |
| paths under descriptive IDs | paths under `EXPxxx` |

### Quality
| Before | After |
|--------|-------|
| `reproducible` | `source_reproducible` + `drilldown_reproducible` |
| `license_status=SEE_feature_manifest` | `license_status` ∈ {OK,REVIEW,RESTRICTED,UNKNOWN} + `license_reference` |

Optional future columns present (empty for Phase 1): `ensemble_type`, `member_experiment_codes`.

## EXP numbering rule

1. Freeze **committed** `results/experiments.csv` row order (48 rows, unique `experiment_id`).
2. Assign `EXP001` … `EXP048` in that order once.
3. Persist mapping in `results/EXPERIMENT_CODES.csv` (`issued_at_phase=PHASE1_BACKFILL`).
4. **Never renumber** existing codes; new experiments use append-only `next_code()`.

Codes encode **no** target / family / performance meaning.

## Assignment freeze (first / last)

| Code | experiment_id |
|------|---------------|
| EXP001 | LIN_TM_ABLINGUA_CDR3_RIDGE |
| EXP002 | LIN_TM_ABLINGUA_GLOBAL_RIDGE |
| EXP003 | LIN_TM_BIOEMU_MPNN_RIDGE |
| EXP004 | LIN_HIC_HYDRO_TITRATION_LASSO |
| EXP005 | LIN_HIC_CONTINUOUS_SURFACE_LASSO |
| … | … |
| EXP044 | XGB_TM_ABLINGUA_GLOBAL |
| EXP045 | XGB_TM_ABLINGUA_CDR3 |
| EXP046 | XGB_HIC_CONTINUOUS_SURFACE |
| EXP047 | XGB_HIC_HYDRO_TITRATION |
| EXP048 | XGB_HIC_ESM2_SEQ_AROMATIC |

## Score preservation

Join by `experiment_id` on score columns. **max abs delta = 0.0** (PASS).

## Prediction preservation

Rename-only of FULL prediction dirs. **max abs delta = 0.0**; file SHA256 unchanged (PASS).

## Feature sets

`results/FEATURE_SETS.csv`: **6** shared RAW_PREPROCESS sets for FULL Linear/XGB pairs, keyed by verified `feature_content_sha256` (not by stripping `__RIDGE`/`__LASSO`).

## Path migration

| Artifact | Count |
|----------|------:|
| `configs/EXPxxx.yaml` | 12 |
| `features/EXPxxx.parquet` | 12 |
| `predictions/EXPxxx/` | 12 |

## Reproducibility semantics

- FULL → `drilldown_reproducible=YES`, `source_reproducible=YES`
- RECONSTRUCTABLE → `drilldown_reproducible=PARTIAL`
- SCORE_ONLY → `drilldown_reproducible=NO`

## License semantics

FULL experiments: conservative aggregation over feature blocks via `feature_manifest.csv` → typically `REVIEW` when PLM embeddings are included. `license_reference` points at the manifest. Legacy `SEE_feature_manifest` status removed.

## Submissions

Regenerated as `sub__EXPxxx__EXPyyy.csv` with manifest columns `tm_experiment_code` / `hic_experiment_code`.

## Validation

`scripts/validate_repository.py` → **PASS**  
`pytest tests/` → run after this report in CI/local.

## Out of scope

Transformer refactor / backfill (next phase). No changes under `top_models_feature_bundle/`.
