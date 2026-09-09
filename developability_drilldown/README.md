# Developability Drilldown

Post-competition research catalog for antibody developability models.

This is **not** a V2 of `top_models_feature_bundle/`. The old bundle remains the immutable authoritative source. This folder organizes experiments for reuse, prediction tracking, CV / Public / Private comparison, and TmApp×HIC submission composition.

## Terminology (strict)

| Term | Meaning |
|------|---------|
| **Experiment** | One target (`TmApp` **or** `HIC`) × one model/feature configuration |
| **Prediction** | Single-target output: `id,TmApp` or `id,HIC`. Never called a submission |
| **Submission** | Competition-ready `id,TmApp,HIC` composed from one Tm experiment + one HIC experiment |
| **Feature parquet** | `RAW_PREPROCESS` recipe matrix (before impute / PCA / scaler / fold-local transforms) |

## Current evaluation policy

- `current_evaluation_mode = POSTCOMP_EXPLORATORY`
- Public / Private may be consulted during exploratory research
- They are **not** independent held-out validation
- Historical `selection_policy_at_creation` (typically `CV_ONLY` for Phase 1 Linear/XGB) is preserved and not rewritten

## Master registry

Single authoritative table:

```text
results/experiments.csv
```

Comparable FEATURE_LINEAR rows from the organizer master registry are cataloged here (scores), with canonical Linear Top-6 and XGBoost Top-6 marked `artifact_status=FULL`.

## Layout

```text
experiments/configs/          per-experiment YAML
experiments/features/         RAW_PREPROCESS parquet (FULL only)
experiments/predictions/      oof_primary / oof_shadow / test (FULL only)
results/experiments.csv       master registry
results/CATALOG(_JA).md       auto-generated views
submissions/                  composed id,TmApp,HIC only
scripts/                      export / backfill / compose / validate
```

## Compose a submission

```bash
python scripts/compose_submission.py \
  --tm LIN_TM_ABLINGUA_CDR3_RIDGE \
  --hic XGB_HIC_CONTINUOUS_SURFACE
```

## Validate

```bash
python scripts/validate_repository.py
pytest tests/
```

## Solution labels

`solution.csv` is **not** distributed. If present locally under the old bundle, scoring helpers may use it. The repository remains usable without it; Public/Private values in `experiments.csv` come from authoritative sources.

## Feature hashing

- `feature_sha256`: SHA256 of parquet file bytes
- `feature_content_sha256`: canonical content hash over sorted IDs + column order + values (NaN-aware)
- `feature_recipe_hash`: hash of ordered blocks + column identities
- Same recipe shared by Linear and XGB MUST share `feature_content_sha256`

## Next phase (not implemented here)

Migrate Transformer / Fusion code into `models/antibody_transformer/` and backfill those experiments into this registry (scratch, AbLingua, AbLang2, REG variants, then dual-PLM / distance-bias research).
