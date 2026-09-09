# Source mapping (FULL experiments)

Old bundle / organizer paths are read-only sources. New artifacts live under `developability_drilldown/`.

## `LIN_HIC_CONTINUOUS_SURFACE_LASSO`

- **source_model_id:** `HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_HIC_CONTINUOUS_SURFACE_LASSO.yaml`
- **feature parquet:** `experiments/features/LIN_HIC_CONTINUOUS_SURFACE_LASSO.parquet`
- **oof_primary:** `experiments/predictions/LIN_HIC_CONTINUOUS_SURFACE_LASSO/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_HIC_CONTINUOUS_SURFACE_LASSO/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_HIC_CONTINUOUS_SURFACE_LASSO/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `LIN_HIC_ESM2_SEQ_AROMATIC_LASSO`

- **source_model_id:** `HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_HIC_ESM2_SEQ_AROMATIC_LASSO.yaml`
- **feature parquet:** `experiments/features/LIN_HIC_ESM2_SEQ_AROMATIC_LASSO.parquet`
- **oof_primary:** `experiments/predictions/LIN_HIC_ESM2_SEQ_AROMATIC_LASSO/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_HIC_ESM2_SEQ_AROMATIC_LASSO/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_HIC_ESM2_SEQ_AROMATIC_LASSO/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `LIN_HIC_HYDRO_TITRATION_LASSO`

- **source_model_id:** `HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_HIC_HYDRO_TITRATION_LASSO.yaml`
- **feature parquet:** `experiments/features/LIN_HIC_HYDRO_TITRATION_LASSO.parquet`
- **oof_primary:** `experiments/predictions/LIN_HIC_HYDRO_TITRATION_LASSO/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_HIC_HYDRO_TITRATION_LASSO/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_HIC_HYDRO_TITRATION_LASSO/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/HIC_HYDRO_TITRATION__LASSO__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `LIN_TM_ABLINGUA_CDR3_RIDGE`

- **source_model_id:** `TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_TM_ABLINGUA_CDR3_RIDGE.yaml`
- **feature parquet:** `experiments/features/LIN_TM_ABLINGUA_CDR3_RIDGE.parquet`
- **oof_primary:** `experiments/predictions/LIN_TM_ABLINGUA_CDR3_RIDGE/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_TM_ABLINGUA_CDR3_RIDGE/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_TM_ABLINGUA_CDR3_RIDGE/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `LIN_TM_ABLINGUA_GLOBAL_RIDGE`

- **source_model_id:** `TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_TM_ABLINGUA_GLOBAL_RIDGE.yaml`
- **feature parquet:** `experiments/features/LIN_TM_ABLINGUA_GLOBAL_RIDGE.parquet`
- **oof_primary:** `experiments/predictions/LIN_TM_ABLINGUA_GLOBAL_RIDGE/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_TM_ABLINGUA_GLOBAL_RIDGE/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_TM_ABLINGUA_GLOBAL_RIDGE/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `LIN_TM_BIOEMU_MPNN_RIDGE`

- **source_model_id:** `TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / LINEAR
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `organizer_extension/top3_ensemble_quickcheck/base_predictions/`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/LIN_TM_BIOEMU_MPNN_RIDGE.yaml`
- **feature parquet:** `experiments/features/LIN_TM_BIOEMU_MPNN_RIDGE.parquet`
- **oof_primary:** `experiments/predictions/LIN_TM_BIOEMU_MPNN_RIDGE/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/LIN_TM_BIOEMU_MPNN_RIDGE/oof_shadow.csv`
- **test prediction:** `experiments/predictions/LIN_TM_BIOEMU_MPNN_RIDGE/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - oof_primary: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_primary.csv`
  - oof_shadow: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__oof_shadow.csv`
  - test: `organizer_extension/top3_ensemble_quickcheck/base_predictions/TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - recipes: `top_models_feature_bundle/recipes.csv`
  - alpha_policy: `top_models_feature_bundle/FULL_DEV_ALPHA_POLICY_BUNDLE.json`

## `XGB_HIC_CONTINUOUS_SURFACE`

- **source_model_id:** `XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_HIC_CONTINUOUS_SURFACE.yaml`
- **feature parquet:** `experiments/features/XGB_HIC_CONTINUOUS_SURFACE.parquet`
- **oof_primary:** `experiments/predictions/XGB_HIC_CONTINUOUS_SURFACE/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_HIC_CONTINUOUS_SURFACE/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_HIC_CONTINUOUS_SURFACE/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ARO_CONTINUOUS_SURFACE__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

## `XGB_HIC_ESM2_SEQ_AROMATIC`

- **source_model_id:** `XGB__HIC_ESM2_SEQ_AROMATIC__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_HIC_ESM2_SEQ_AROMATIC.yaml`
- **feature parquet:** `experiments/features/XGB_HIC_ESM2_SEQ_AROMATIC.parquet`
- **oof_primary:** `experiments/predictions/XGB_HIC_ESM2_SEQ_AROMATIC/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_HIC_ESM2_SEQ_AROMATIC/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_HIC_ESM2_SEQ_AROMATIC/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_ESM2_SEQ_AROMATIC__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

## `XGB_HIC_HYDRO_TITRATION`

- **source_model_id:** `XGB__HIC_HYDRO_TITRATION__LASSO`
- **target / family:** HIC / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_HIC_HYDRO_TITRATION.yaml`
- **feature parquet:** `experiments/features/XGB_HIC_HYDRO_TITRATION.parquet`
- **oof_primary:** `experiments/predictions/XGB_HIC_HYDRO_TITRATION/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_HIC_HYDRO_TITRATION/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_HIC_HYDRO_TITRATION/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/HIC__xgboost__HIC_HYDRO_TITRATION__LASSO__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

## `XGB_TM_ABLINGUA_CDR3`

- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_TM_ABLINGUA_CDR3.yaml`
- **feature parquet:** `experiments/features/XGB_TM_ABLINGUA_CDR3.parquet`
- **oof_primary:** `experiments/predictions/XGB_TM_ABLINGUA_CDR3/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_TM_ABLINGUA_CDR3/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_TM_ABLINGUA_CDR3/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_CDR3__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

## `XGB_TM_ABLINGUA_GLOBAL`

- **source_model_id:** `XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_TM_ABLINGUA_GLOBAL.yaml`
- **feature parquet:** `experiments/features/XGB_TM_ABLINGUA_GLOBAL.parquet`
- **oof_primary:** `experiments/predictions/XGB_TM_ABLINGUA_GLOBAL/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_TM_ABLINGUA_GLOBAL/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_TM_ABLINGUA_GLOBAL/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_PARENT_ABLINGUA_GLOBAL__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

## `XGB_TM_BIOEMU_MPNN`

- **source_model_id:** `XGB__TM_BASE_BIOEMU_MPNN__RIDGE`
- **target / family:** TmApp / XGBOOST
- **artifact_status:** FULL
- **score_source:** `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
- **prediction_source:** `advanced_outputs test + rebuild_xgb_oof OOF`
- **feature_source:** `top_models_feature_bundle/data/*.parquet concat RAW_PREPROCESS`
- **config:** `experiments/configs/XGB_TM_BIOEMU_MPNN.yaml`
- **feature parquet:** `experiments/features/XGB_TM_BIOEMU_MPNN.parquet`
- **oof_primary:** `experiments/predictions/XGB_TM_BIOEMU_MPNN/oof_primary.csv`
- **oof_shadow:** `experiments/predictions/XGB_TM_BIOEMU_MPNN/oof_shadow.csv`
- **test prediction:** `experiments/predictions/XGB_TM_BIOEMU_MPNN/test.csv`
- **config.source_paths:**
  - scores: `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`
  - benchmark_summary: `top_models_feature_bundle/results/MODEL_BENCHMARK_SUMMARY.csv`
  - presets: `top_models_feature_bundle/advanced_models/presets.json`
  - test_predictions: `top_models_feature_bundle/advanced_outputs/predictions/TmApp__xgboost__TM_BASE_BIOEMU_MPNN__RIDGE__test.csv`
  - oof_rebuild: `developability_drilldown/scripts/rebuild_xgb_oof.py`
- **OOF note:** family-winner OOFs reused from tracked `cross_family_ensemble/*.npz` where available; otherwise reconstructed via `scripts/rebuild_xgb_oof.py` under frozen advanced-suite protocol (not a new search).

