# Developability Drilldown

抗体 developability のコンペ後研究カタログです。`top_models_feature_bundle/` の V2 ではありません。

## Experiment code

| Namespace | 意味 | 例 |
|-----------|------|-----|
| **EXP-Txxx** | TmApp **単一 target** experiment | `EXP-T001`… |
| **EXP-Hxxx** | HIC **単一 target** experiment | `EXP-H001`… |
| **EXP-Mxxx** | **同時学習** multi-target experiment 用（予約） | 次番号 `EXP-M001`（現在 0 件） |

- T/H/M は **予測スコープ** を示す（model family ではない）
- LINEAR / XGBOOST / TRANSFORMER を code に埋め込まない
- namespace ごとに **append-only**。再利用・再採番禁止
- 説明的 `experiment_id` は別途保持
- 旧 `EXP001`–`EXP048` は `LEGACY_EXPERIMENT_CODE_MAP.csv` に保存

表示例: `EXP-T001 — LIN_TM_ABLINGUA_CDR3_RIDGE`

### EXP-M ではないもの

独立した Tm prediction と HIC prediction を組み合わせるのは **submission** であり、EXP-M ではない。

EXP-M は **1 つの joint model/config/学習手続き** が複数 target を同時に予測する場合のみ。本コンペは target 別 metric のみで combined score は無いため、実 joint model が出るまで EXP-M は予約のまま。

## 用語

- **Prediction** … experiment の出力（単一 target なら `id,TmApp` / `id,HIC`）。submission と呼ばない
- **Submission** … `submissions/` に置いた `id,TmApp,HIC` のみ

## 合成

```bash
python scripts/compose_submission.py --tm EXP-T001 --hic EXP-H001
```

出力: `sub__EXP-T001__EXP-H001.csv`（常に T → H）

## 検証

```bash
python scripts/validate_repository.py
pytest tests/
```
