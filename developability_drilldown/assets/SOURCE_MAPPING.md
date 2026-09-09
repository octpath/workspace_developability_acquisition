# Source mapping (FULL experiments)

Codes are permanent (`experiment_code`). Descriptive `experiment_id` is human-readable only.

Old bundle / organizer paths are read-only sources.

## `EXP004` — `LIN_HIC_HYDRO_TITRATION_LASSO`

- **source_model_id:** `HIC_HYDRO_TITRATION__LASSO`
- **feature_set_id:** `FS_HIC_HYDRO_TITRATION`
- **source_recipe_id:** `HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP004.yaml`
- **feature parquet:** `experiments/features/EXP004.parquet`
- **oof_primary:** `experiments/predictions/EXP004/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP004/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP004/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP005` — `LIN_HIC_CONTINUOUS_SURFACE_LASSO`

- **source_model_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **feature_set_id:** `FS_HIC_CONTINUOUS_SURFACE`
- **source_recipe_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP005.yaml`
- **feature parquet:** `experiments/features/EXP005.parquet`
- **oof_primary:** `experiments/predictions/EXP005/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP005/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP005/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP006` — `LIN_HIC_ESM2_SEQ_AROMATIC_LASSO`

- **source_model_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **feature_set_id:** `FS_HIC_ESM2_SEQ_AROMATIC`
- **source_recipe_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP006.yaml`
- **feature parquet:** `experiments/features/EXP006.parquet`
- **oof_primary:** `experiments/predictions/EXP006/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP006/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP006/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP001` — `LIN_TM_ABLINGUA_CDR3_RIDGE`

- **source_model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_CDR3`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP001.yaml`
- **feature parquet:** `experiments/features/EXP001.parquet`
- **oof_primary:** `experiments/predictions/EXP001/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP001/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP001/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP002` — `LIN_TM_ABLINGUA_GLOBAL_RIDGE`

- **source_model_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_GLOBAL`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP002.yaml`
- **feature parquet:** `experiments/features/EXP002.parquet`
- **oof_primary:** `experiments/predictions/EXP002/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP002/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP002/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP003` — `LIN_TM_BIOEMU_MPNN_RIDGE`

- **source_model_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **feature_set_id:** `FS_TM_BIOEMU_MPNN`
- **source_recipe_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP003.yaml`
- **feature parquet:** `experiments/features/EXP003.parquet`
- **oof_primary:** `experiments/predictions/EXP003/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP003/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP003/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `EXP046` — `XGB_HIC_CONTINUOUS_SURFACE`

- **source_model_id:** `XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **feature_set_id:** `FS_HIC_CONTINUOUS_SURFACE`
- **source_recipe_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP046.yaml`
- **feature parquet:** `experiments/features/EXP046.parquet`
- **oof_primary:** `experiments/predictions/EXP046/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP046/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP046/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP047` — `XGB_HIC_HYDRO_TITRATION`

- **source_model_id:** `XGB__HIC_HYDRO_TITRATION__LASSO`
- **feature_set_id:** `FS_HIC_HYDRO_TITRATION`
- **source_recipe_id:** `HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP047.yaml`
- **feature parquet:** `experiments/features/EXP047.parquet`
- **oof_primary:** `experiments/predictions/EXP047/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP047/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP047/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_HYDRO_TITRATION__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP048` — `XGB_HIC_ESM2_SEQ_AROMATIC`

- **source_model_id:** `XGB__HIC_ESM2_SEQ_AROMATIC__LASSO`
- **feature_set_id:** `FS_HIC_ESM2_SEQ_AROMATIC`
- **source_recipe_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP048.yaml`
- **feature parquet:** `experiments/features/EXP048.parquet`
- **oof_primary:** `experiments/predictions/EXP048/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP048/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP048/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP043` — `XGB_TM_BIOEMU_MPNN`

- **source_model_id:** `XGB__TM_BASE_BIOEMU_MPNN__RIDGE`
- **feature_set_id:** `FS_TM_BIOEMU_MPNN`
- **source_recipe_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP043.yaml`
- **feature parquet:** `experiments/features/EXP043.parquet`
- **oof_primary:** `experiments/predictions/EXP043/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP043/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP043/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP044` — `XGB_TM_ABLINGUA_GLOBAL`

- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_GLOBAL`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP044.yaml`
- **feature parquet:** `experiments/features/EXP044.parquet`
- **oof_primary:** `experiments/predictions/EXP044/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP044/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP044/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

## `EXP045` — `XGB_TM_ABLINGUA_CDR3`

- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **feature_set_id:** `FS_TM_ABLINGUA_CDR3`
- **source_recipe_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/EXP045.yaml`
- **feature parquet:** `experiments/features/EXP045.parquet`
- **oof_primary:** `experiments/predictions/EXP045/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/EXP045/oof_shadow.csv`
- **test prediction:** `experiments/predictions/EXP045/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`

