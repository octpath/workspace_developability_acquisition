# GBDT search results

- Fixed `learning_rate=0.03` for XGBoost, LightGBM, CatBoost
- `n_estimators`/`iterations` NOT Optuna-tuned (ceiling 5000)
- Early stopping: 150 rounds on internal group holdout (~18% of groups)
- Outer validation never used for early stopping
- Nested effective iterations: median 84, ceiling hit rate 0%
- Nested: GBDT did **not** beat best ElasticNet/structure for HIC or AbLang2+BIO fusion for TmApp

See `metrics/screening_summary.csv` and `metrics/nested_cv_results.csv`.
