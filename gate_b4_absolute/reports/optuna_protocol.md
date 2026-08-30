# Optuna protocol (Gate B4)

- Primary objective: minimize inner grouped-CV MAE
- Sampler: TPESampler with recorded seed
- Persistent SQLite under `optuna/*.db`
- No pruning for sklearn/GBDT screening
- GBDT: learning_rate fixed 0.03; n_estimators ceiling 5000; ES rounds 150 on internal group split
- Outer validation never used for early stopping

Software: {"python": "3.12.13", "numpy": "2.5.2", "pandas": "3.0.5", "sklearn": "1.9.0", "xgboost": "3.4.1", "lightgbm": "4.7.0", "catboost": "1.2.10", "optuna": "4.9.0"}
