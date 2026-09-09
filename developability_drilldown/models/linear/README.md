# Linear models

Phase 1 does **not** copy estimator training code into this folder.

Authoritative reproduction lives in:

- `top_models_feature_bundle/reproduce_top_recipes.py`
- `top_models_feature_bundle/bundle_simple_tvt.py`
- `top_models_feature_bundle/examples/train_ridge.py`
- `top_models_feature_bundle/examples/train_lasso.py`

FULL Linear experiments store:

- YAML configs under `experiments/configs/LIN_*`
- RAW feature parquets under `experiments/features/`
- OOF / test predictions under `experiments/predictions/`

Fitted pickles are not distributed; features + hyperparameters + CV protocol are sufficient to refit.
