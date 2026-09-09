# Developability Drilldown

Post-competition research catalog for antibody developability models.

This is **not** a V2 of `top_models_feature_bundle/`. The old bundle remains the immutable authoritative source.

## Experiment identifiers

| Field | Role |
|-------|------|
| **`experiment_code`** | Permanent short ID: `EXP001`, `EXP002`, … Canonical primary key |
| **`experiment_id`** | Human-readable descriptive configuration name (e.g. `LIN_TM_ABLINGUA_CDR3_RIDGE`) |

Code policy:

- **immutable** once issued
- **append-only** (`max(existing)+1`)
- **never reused**, even if an experiment is retired
- **never renumbered** by score, family, or target
- Authority file: [`results/EXPERIMENT_CODES.csv`](results/EXPERIMENT_CODES.csv)

Display form: `EXP001 — LIN_TM_ABLINGUA_CDR3_RIDGE`

Future families may include `LINEAR`, `XGBOOST`, `TRANSFORMER`, `ENSEMBLE` (e.g. stacking via `ensemble_type=STACKING` + `member_experiment_codes`). Codes themselves carry **no** family meaning.

## Terminology (strict)

| Term | Meaning |
|------|---------|
| **Experiment** | One target (`TmApp` **or** `HIC`) × one model/feature configuration |
| **Prediction** | Single-target output: `id,TmApp` or `id,HIC`. Never called a submission |
| **Submission** | Competition-ready `id,TmApp,HIC` composed from one Tm + one HIC experiment |
| **Feature set** | Estimator-independent raw feature composition (`feature_set_id`) |
| **Feature parquet** | `RAW_PREPROCESS` matrix (before impute / PCA / scaler / fold-local transforms) |

Artifact paths use codes only:

```text
experiments/configs/EXP012.yaml
experiments/features/EXP012.parquet
experiments/predictions/EXP012/{oof_primary,oof_shadow,test}.csv
submissions/sub__EXP012__EXP044.csv
```

## Current evaluation policy

- `current_evaluation_mode = POSTCOMP_EXPLORATORY`
- Public / Private may be consulted during exploratory research
- They are **not** independent held-out validation
- Historical `selection_policy_at_creation` (typically `CV_ONLY`) is preserved

## Reproducibility / license columns

| Column | Meaning |
|--------|---------|
| `source_reproducible` | Reconstructable from original repo sources (`YES`/`NO`/`UNKNOWN`) |
| `drilldown_reproducible` | Usable from materialized drilldown artifacts (`YES`/`PARTIAL`/`NO`) |
| `artifact_status` | `FULL` / `RECONSTRUCTABLE` / `SCORE_ONLY` / `PARTIAL` / `INCONSISTENT` |
| `license_status` | `OK` / `REVIEW` / `RESTRICTED` / `UNKNOWN` |
| `license_reference` | Pointer (e.g. feature manifest path) |

## Master registry

```text
results/experiments.csv
```

Also: `FEATURE_SETS.csv`, `EXPERIMENT_CODES.csv`, `CATALOG(_JA).md`.

## Compose a submission

```bash
python scripts/compose_submission.py --tm EXP001 --hic EXP004
# descriptive IDs also resolve:
python scripts/compose_submission.py --tm LIN_TM_ABLINGUA_CDR3_RIDGE --hic LIN_HIC_HYDRO_TITRATION_LASSO
```

Output: `submissions/sub__EXP001__EXP004.csv`

## Validate

```bash
python scripts/validate_repository.py
pytest tests/
```

## Solution labels

`solution.csv` is **not** distributed.

## Next phase (not implemented here)

Migrate Transformer / Fusion into `models/antibody_transformer/` and backfill with new append-only EXP codes.
