# Developability Drilldown

抗体 developability モデルの **コンペ後研究用カタログ**です。`top_models_feature_bundle/` の V2 ではありません。

## 実験識別子

| フィールド | 役割 |
|------------|------|
| **`experiment_code`** | 永久 short ID（`EXP001`…）。canonical primary key |
| **`experiment_id`** | 人間可読な設定名（例: `LIN_TM_ABLINGUA_CDR3_RIDGE`） |

コード方針:

- 一度発行したら **不変**
- **追記のみ**（既存最大番号 + 1）
- 廃止しても **再利用しない**
- スコア・family・target で **振り直さない**
- 権威ファイル: [`results/EXPERIMENT_CODES.csv`](results/EXPERIMENT_CODES.csv)

表示例: `EXP001 — LIN_TM_ABLINGUA_CDR3_RIDGE`

将来の family 例: `LINEAR` / `XGBOOST` / `TRANSFORMER` / `ENSEMBLE`（stacking は `ensemble_type=STACKING`）。番号自体に family 意味は持たせません。

## 用語（厳格）

| 用語 | 意味 |
|------|------|
| **Experiment** | 1 target（`TmApp` または `HIC`）× 1 設定 |
| **Prediction** | 単一 target 出力。submission と呼ばない |
| **Submission** | Tm + HIC を合成した `id,TmApp,HIC` のみ |
| **Feature set** | estimator 非依存の raw 特徴構成（`feature_set_id`） |

成果物パスは code のみ（`experiments/configs/EXP012.yaml` 等）。

## 再現性 / ライセンス列

- `source_reproducible` … 元リポジトリから再構成可能か
- `drilldown_reproducible` … drilldown 内 materialize で扱えるか
- `artifact_status` … FULL / RECONSTRUCTABLE / SCORE_ONLY / …
- `license_status` … OK / REVIEW / RESTRICTED / UNKNOWN
- `license_reference` … manifest 等への pointer

曖昧な単独 `reproducible` 列は使いません。

## Submission 合成

```bash
python scripts/compose_submission.py --tm EXP001 --hic EXP004
```

## 検証

```bash
python scripts/validate_repository.py
pytest tests/
```

## 次フェーズ（未実装）

`models/antibody_transformer/` への切り出しと、追記専用 EXP コードでの Transformer backfill。
