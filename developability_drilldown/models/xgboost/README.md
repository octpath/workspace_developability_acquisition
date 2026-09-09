# XGBoost models

Phase 1 uses the frozen advanced suite without hyperparameter retuning:

- `top_models_feature_bundle/advanced_models/models/xgboost_model.py`
- `top_models_feature_bundle/advanced_models/features.py`
- `top_models_feature_bundle/advanced_models/presets.json`
- `top_models_feature_bundle/advanced_models/frozen_results/ADVANCED_MODEL_RESULTS.csv`

OOF completion (artifact fill, not model search):

```bash
python scripts/rebuild_xgb_oof.py --device cuda
```

Audit:

```text
results/XGB_REPRODUCTION_AUDIT.csv
```

Authoritative CV / Public / Private scores in `results/experiments.csv` are never overwritten by reconstructed values.
