# Developability Drilldown

抗体 developability モデルの **コンペ後研究用カタログ**です。

これは `top_models_feature_bundle/` の V2 ではありません。旧バンドルは不変の権威ソースとして残し、本フォルダは実験の整理・特徴量再利用・予測追跡・CV/Public/Private 比較・TmApp×HIC submission 合成のための作業空間です。

## 用語（厳格）

| 用語 | 意味 |
|------|------|
| **Experiment** | 1 target（`TmApp` **または** `HIC`）× 1 モデル/特徴設定 |
| **Prediction** | 単一 target 出力: `id,TmApp` または `id,HIC`。submission と呼ばない |
| **Submission** | Tm experiment と HIC experiment を合成した `id,TmApp,HIC` のみ |
| **Feature parquet** | `RAW_PREPROCESS`（impute / PCA / scaler / fold-local 変換の前） |

## 現行評価ポリシー

- `current_evaluation_mode = POSTCOMP_EXPLORATORY`
- Public / Private は探索研究で参照してよい
- independent held-out validation とは呼ばない
- 過去の `selection_policy_at_creation`（Phase 1 の Linear/XGB は原則 `CV_ONLY`）は保持し、post-comp 扱いに書き換えない

## マスターレジストリ

唯一の権威テーブル:

```text
results/experiments.csv
```

organizer の FEATURE_LINEAR（同一 canonical CV）を広く登録し、Linear Top-6 と XGBoost Top-6 のみ `artifact_status=FULL` で成果物を materialize します。

## Submission 合成

```bash
python scripts/compose_submission.py \
  --tm LIN_TM_ABLINGUA_CDR3_RIDGE \
  --hic XGB_HIC_CONTINUOUS_SURFACE
```

## 検証

```bash
python scripts/validate_repository.py
pytest tests/
```

## solution.csv

配布しません。ローカルに旧バンドルの solution がある場合のみスコア再計算に利用可能です。

## 次フェーズ（未実装）

`models/antibody_transformer/` への Transformer 切り出しとカタログ追加（scratch / AbLingua / AbLang2 / Fusion / REG 系、その後 dual PLM・距離バイアス等）。
