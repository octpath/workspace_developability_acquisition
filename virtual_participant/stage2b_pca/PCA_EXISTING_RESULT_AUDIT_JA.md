# Stage 2b — 既存PCA結果の監査

## 結論

Stage 2 registry（`stage2_model_registry.csv`）を確認したところ、

- 出現した `pca_dim`: **[8, 16, 32, 48, 64]**
- `pca_dim` が欠損（None / raw）の行数: **0**

したがって、Stage 2本体では **raw embedding（PCAなし）の systematic 比較は実施されていない**。
PCA次元は 8/16/32/48/64 のみが探索対象だった。

## 含意

- 「PCAが必要／不要」は Stage 2 だけでは結論できない。
- Stage 2b で raw vs PCA の controlled comparison を新規に実施する必要がある。
- 既存の PCA32 fixed と Optuna 最適化の差は、PCA次元と regressor hyperparameter の両方の変化を含む。

## Stage 2 best（参考）

| Target | 種別 | PCA | Primary | Shadow |
|---|---|---:|---:|---:|
| TmApp | PLM-only AbLang2 HL_paired SVR | 64 | 2.9231 | 3.0361 |
| TmApp | fusion + SEQ_BASIC | 48 | 2.7756 | 2.8316 |
| HIC | PLM-only ESM-2 H SVR | 48 | 0.4552 | 0.4529 |
| HIC | fusion + SEQ_ALL | 8 | 0.4485 | 0.4510 |
