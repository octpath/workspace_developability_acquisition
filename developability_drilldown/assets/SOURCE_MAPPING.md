# Source mapping (FULL experiments)

Codes are permanent (`experiment_code`). Descriptive `experiment_id` is human-readable only.

Old bundle / organizer paths are read-only sources.

## `EXP-H001` — `LIN_HIC_HYDRO_TITRATION_LASSO`

- **legacy_experiment_code:** `EXP004`
- **source_model_id:** `HIC_HYDRO_TITRATION__LASSO`
- **feature_set_id:** `FS_HIC_HYDRO_TITRATION`
- **source_recipe_id:** `HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H001.yaml`
- **feature parquet:** `experiments/features/EXP-H001.parquet`
- **oof_primary:** `experiments/predictions/EXP-H001/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H001/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H001/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-H002` — `LIN_HIC_CONTINUOUS_SURFACE_LASSO`

- **legacy_experiment_code:** `EXP005`
- **source_model_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **feature_set_id:** `FS_HIC_CONTINUOUS_SURFACE`
- **source_recipe_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H002.yaml`
- **feature parquet:** `experiments/features/EXP-H002.parquet`
- **oof_primary:** `experiments/predictions/EXP-H002/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H002/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H002/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-H003` — `LIN_HIC_ESM2_SEQ_AROMATIC_LASSO`

- **legacy_experiment_code:** `EXP006`
- **source_model_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **feature_set_id:** `FS_HIC_ESM2_SEQ_AROMATIC`
- **source_recipe_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H003.yaml`
- **feature parquet:** `experiments/features/EXP-H003.parquet`
- **oof_primary:** `experiments/predictions/EXP-H003/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H003/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H003/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-T001` — `LIN_TM_ABLINGUA_CDR3_RIDGE`

- **legacy_experiment_code:** `EXP001`
- **source_model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_CDR3`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T001.yaml`
- **feature parquet:** `experiments/features/EXP-T001.parquet`
- **oof_primary:** `experiments/predictions/EXP-T001/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T001/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T001/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-T002` — `LIN_TM_ABLINGUA_GLOBAL_RIDGE`

- **legacy_experiment_code:** `EXP002`
- **source_model_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_GLOBAL`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T002.yaml`
- **feature parquet:** `experiments/features/EXP-T002.parquet`
- **oof_primary:** `experiments/predictions/EXP-T002/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T002/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T002/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-T003` — `LIN_TM_BIOEMU_MPNN_RIDGE`

- **legacy_experiment_code:** `EXP003`
- **source_model_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **feature_set_id:** `FS_TM_BIOEMU_MPNN`
- **source_recipe_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T003.yaml`
- **feature parquet:** `experiments/features/EXP-T003.parquet`
- **oof_primary:** `experiments/predictions/EXP-T003/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T003/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T003/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP-H021` — `XGB_HIC_CONTINUOUS_SURFACE`

- **legacy_experiment_code:** `EXP046`
- **source_model_id:** `XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **feature_set_id:** `FS_HIC_CONTINUOUS_SURFACE`
- **source_recipe_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H021.yaml`
- **feature parquet:** `experiments/features/EXP-H021.parquet`
- **oof_primary:** `experiments/predictions/EXP-H021/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H021/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H021/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP-H022` — `XGB_HIC_HYDRO_TITRATION`

- **legacy_experiment_code:** `EXP047`
- **source_model_id:** `XGB__HIC_HYDRO_TITRATION__LASSO`
- **feature_set_id:** `FS_HIC_HYDRO_TITRATION`
- **source_recipe_id:** `HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H022.yaml`
- **feature parquet:** `experiments/features/EXP-H022.parquet`
- **oof_primary:** `experiments/predictions/EXP-H022/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H022/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H022/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_HYDRO_TITRATION__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP-H023` — `XGB_HIC_ESM2_SEQ_AROMATIC`

- **legacy_experiment_code:** `EXP048`
- **source_model_id:** `XGB__HIC_ESM2_SEQ_AROMATIC__LASSO`
- **feature_set_id:** `FS_HIC_ESM2_SEQ_AROMATIC`
- **source_recipe_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-H023.yaml`
- **feature parquet:** `experiments/features/EXP-H023.parquet`
- **oof_primary:** `experiments/predictions/EXP-H023/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-H023/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-H023/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP-T023` — `XGB_TM_BIOEMU_MPNN`

- **legacy_experiment_code:** `EXP043`
- **source_model_id:** `XGB__TM_BASE_BIOEMU_MPNN__RIDGE`
- **feature_set_id:** `FS_TM_BIOEMU_MPNN`
- **source_recipe_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T023.yaml`
- **feature parquet:** `experiments/features/EXP-T023.parquet`
- **oof_primary:** `experiments/predictions/EXP-T023/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T023/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T023/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP-T024` — `XGB_TM_ABLINGUA_GLOBAL`

- **legacy_experiment_code:** `EXP044`
- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_GLOBAL`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T024.yaml`
- **feature parquet:** `experiments/features/EXP-T024.parquet`
- **oof_primary:** `experiments/predictions/EXP-T024/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T024/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T024/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP-T025` — `XGB_TM_ABLINGUA_CDR3`

- **legacy_experiment_code:** `EXP045`
- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_CDR3`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP-T025.yaml`
- **feature parquet:** `experiments/features/EXP-T025.parquet`
- **oof_primary:** `experiments/predictions/EXP-T025/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP-T025/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP-T025/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

