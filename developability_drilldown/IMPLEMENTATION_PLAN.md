# developability_drilldown — Phase 1 Implementation Plan (FROZEN)

**Status:** FROZEN for Phase 1 implementation; **registry identity migrated** (see `results/REGISTRY_MIGRATION_REPORT.md` — permanent `experiment_code` EXP001–EXP048)  
**Start HEAD:** `047cc0856540b5afec267e408f306e29c22ef0c1`  
**Phase 1 catalog commit:** `48dfdd21`  
**Scope:** Linear + XGBoost catalog, features, predictions, submission composer  
**Out of scope:** Transformer refactor / new Transformer experiments

---

## Design invariants

1. **ONE EXPERIMENT = ONE TARGET** (`TmApp` or `HIC`).
2. Single-target output = **PREDICTION** (`id,TmApp` or `id,HIC`). Never named `submission` / `sub_`.
3. **SUBMISSION** = `id,TmApp,HIC` only, composed from one Tm + one HIC experiment.
4. `top_models_feature_bundle/` and organizer outputs are **read-only authoritative sources** (no move/delete/mutate).
5. New tree: `developability_drilldown/`.

---

## Revisions vs prior draft

| Topic | Old | Frozen |
|-------|-----|--------|
| XGB OOF | 2/6 only; 4 missing | **6/6 FULL**; missing 4 may rerun under frozen config (artifact completion, not search) |
| selection_mode | all `POSTCOMP_EXPLORATORY` | Split: `selection_policy_at_creation` (historical) + `current_evaluation_mode` (now) |
| Registry | Top-6 + separate historical CSV | **`results/experiments.csv` is the single master**; all comparable FEATURE_LINEAR rows + XGB 6 |
| Feature provenance | paths only | `n_features`, `feature_space=RAW_PREPROCESS`, `feature_sha256`, `feature_content_sha256`, `feature_recipe_hash` |

---

## Directory tree

```
developability_drilldown/
├── README.md / README_JA.md / IMPLEMENTATION_PLAN.md / .gitignore
├── data/{dev,test,folds}.csv
├── experiments/{configs,features,predictions}/...
├── results/{experiments.csv,CATALOG.md,CATALOG_JA.md,VALIDATION.txt,XGB_REPRODUCTION_AUDIT.csv}
├── submissions/{submissions.csv,sub__*.csv}
├── models/{linear,xgboost}/README.md
├── scripts/{export_features,backfill_from_sources,rebuild_xgb_oof,build_catalog,
│            compose_submission,score_predictions,validate_repository}.py
├── assets/SOURCE_MAPPING.md
└── tests/
```

---

## Experiment IDs

### FULL Linear Top-6 (canonical friendly IDs)

| experiment_id | source_model_id |
|---------------|-----------------|
| LIN_TM_ABLINGUA_CDR3_RIDGE | TM_PARENT_ABLINGUA_CDR3__RIDGE |
| LIN_TM_ABLINGUA_GLOBAL_RIDGE | TM_PARENT_ABLINGUA_GLOBAL__RIDGE |
| LIN_TM_BIOEMU_MPNN_RIDGE | TM_BASE_BIOEMU_MPNN__RIDGE |
| LIN_HIC_HYDRO_TITRATION_LASSO | HIC_HYDRO_TITRATION__LASSO |
| LIN_HIC_CONTINUOUS_SURFACE_LASSO | HIC_ARO_CONTINUOUS_SURFACE__LASSO |
| LIN_HIC_ESM2_SEQ_AROMATIC_LASSO | HIC_ESM2_SEQ_AROMATIC__LASSO |

### FULL XGB Top-6

| experiment_id | source_model_id |
|---------------|-----------------|
| XGB_TM_BIOEMU_MPNN | XGB__TM_BASE_BIOEMU_MPNN__RIDGE |
| XGB_TM_ABLINGUA_GLOBAL | XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE |
| XGB_TM_ABLINGUA_CDR3 | XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE |
| XGB_HIC_CONTINUOUS_SURFACE | XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO |
| XGB_HIC_HYDRO_TITRATION | XGB__HIC_HYDRO_TITRATION__LASSO |
| XGB_HIC_ESM2_SEQ_AROMATIC | XGB__HIC_ESM2_SEQ_AROMATIC__LASSO |

### Other FEATURE_LINEAR

Deterministic: `LIN_` + `source_model_id` with `__` → `_`, after applying Top-6 overrides. Collision-checked. Always store `source_model_id`.

---

## artifact_status

`FULL` | `SCORE_ONLY` | `PARTIAL` | `RECONSTRUCTABLE` | `INCONSISTENT`

- **FULL (12):** Linear Top-6 + XGB 6 — config, raw feature parquet, OOF primary/shadow, test prediction, scores.
- **SCORE_ONLY / RECONSTRUCTABLE:** remaining FEATURE_LINEAR rows in master registry (no Phase 1 materialization of parquet/preds).

Include all FEATURE_LINEAR with `cv_protocol=canonical_simple_tvt_v1` (~42). CONSTANT / OPENMM stay as SCORE_ONLY with notes (no inventing missing Public/Private).

---

## Feature parquet = RAW_PREPROCESS

- Horizontal concat of recipe blocks from bundle `data/*.parquet`.
- **Before** impute / PCA / StandardScaler / fold-local transforms.
- Schema: `id`, `split` (`dev`|`test`), feature columns (preserve source names).
- 324 rows. Duplicate physically per experiment_id (~80MB OK).
- Same recipe → identical `feature_content_sha256` (LIN vs XGB).

---

## Selection / evaluation policy

- Existing Linear/XGB: `selection_policy_at_creation = CV_ONLY` (unless source evidence says otherwise).
- Now: `current_evaluation_mode = POSTCOMP_EXPLORATORY`.
- Do **not** rewrite historical selection as post-comp.
- Public/Private may be consulted in research; not called independent held-out validation.

---

## XGB OOF completion

- Reuse tracked npz for family winners (2).
- Rerun remaining 4 with frozen protocol via advanced suite code: same recipe, folds, presets search + early stopping semantics, seeds (`random_state=0`), preprocessing.
- **Forbidden:** preset/HP/Optuna/fold/seed/policy changes; Public/Private-guided tuning.
- Authoritative scores never overwritten by reconstructed values.
- Audit: `results/XGB_REPRODUCTION_AUDIT.csv`. Mismatch → `INCONSISTENT` + deltas, or fix only reproducibility bugs.

---

## experiments.csv (single master)

Identity, scores (incl. `public_private_delta = public-private`, `public_private_gap = abs(...)`), history/policy, artifacts, feature metadata, model details, provenance, quality. Derived: `cv_mean`, `cv_worst`, overall, gaps.

Score authority: `MODEL_BENCHMARK_SUMMARY.csv` / `ADVANCED_MODEL_RESULTS.csv` / `LINEAR_MODEL_MASTER_REGISTRY.csv` — never hand-typed.

---

## Predictions / submissions

- FULL: `oof_primary.csv`, `oof_shadow.csv`, `test.csv` with target column names.
- Composer: `compose_submission.py --tm ... --hic ...` → `sub__{tm}__{hic}.csv`.
- Example submissions (2–3) + `submissions.csv` joined from master.

---

## Validation / tests / git

- `validate_repository.py` + pytest covering registry, terminology, features, predictions, XGB 6/6, composer, bundle untouched, solution untracked.
- Path-specific commits; no force push; no mutation of old bundle tracked files.

---

## Phase 2 (note only)

Future: `models/antibody_transformer/` from `advanced_models` AnnotatedTransformer; catalog Transformer/Fusion; then dual PLM / distance bias research.
