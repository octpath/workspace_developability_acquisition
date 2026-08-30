# Train CV protocol

- Train N = 162
- 5 Group folds × 4 repeats = 20 outer evaluations
- Groups = sequence_group (90% paired identity clusters)
- Same folds for all representations
- Primary metric: Spearman; secondary Pearson/MAE/RMSE
- Public/Private labels sealed during model search

Fold file: `config/TRAIN_CV_FOLDS.json`
