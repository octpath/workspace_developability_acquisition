# Stage 5 Report Patch — 監査記録

## 監査日

2026-09-02（artifact 再読取 + Shadow 完遂）

## 1. HIC simple blend Shadow lock 確認

### 対象

`HIC__SIMPLE_blend_seq_surf_adv`（Primary = 0.4251776987709104）

### 結果: **CASE A**（Shadow lock 含む、Shadow 初回未計算 → 完遂）

`STAGE5_SHADOW_LOCK.json` の HIC finalists:

| role | experiment_id | Primary |
|---|---|---:|
| conservative_incumbent | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` | 0.4373 |
| best_integrated | **`HIC__SIMPLE_blend_seq_surf_adv`** | 0.4252 |
| diverse_alt | `HIC__SIMPLE_blend_seq_surf` | 0.4258 |

初回 `stage5_shadow_results.csv` には HIC simple blend が**未記載**だった（incumbent / nnls_norm / identity のみ）。lock 済み finalist のため、frozen equal-mean configuration で Shadow 評価を完遂:

| experiment_id | Shadow MAE | status |
|---|---:|---|
| `HIC__SIMPLE_blend_seq_surf_adv` | 0.4320764671 | STAGE5_SHADOW_CONFIRMED |
| `HIC__SIMPLE_blend_seq_surf` | 0.4328749670 | STAGE5_EQUIVALENT |

- base model / weight 変更なし（equal mean 固定）
- finalist 追加なし（lock 前から含まれていた）

## 2. Best experiment identity 解決

### TmApp Primary best = Shadow-confirmed best

| 項目 | 値 |
|---|---|
| experiment_id | `TmApp__META_performance__ridge_100.0` |
| branch | ridge_stack |
| meta | Ridge α=100 |
| base models | S3INC+INTERACTIONS, overall+RASA, AbLang2+SEQ_BASIC, ADV_TMAPP_ALL |
| Primary exact | 2.7134783500422097 |
| Shadow exact | 2.759100007677969 |
| status | STAGE5_SHADOW_CONFIRMED |

### HIC Primary best

| 項目 | 値 |
|---|---|
| experiment_id | `HIC__SIMPLE_blend_seq_surf_adv` |
| branch | simple_blend (equal mean) |
| base models | ESM2+SEQ_ALL, SURFACE_CHEM, ADV_SURFACE_PATCH |
| Primary exact | 0.4251776987709104 |
| Shadow exact | 0.4320764671 |
| Shadow lock | **含む**（best_integrated） |
| status | STAGE5_SHADOW_CONFIRMED |

## 3. Historical vs nested identity 差

| Target | Stage4 historical (P) | Stage5 nested identity (P) | diff |
|---|---:|---:|---:|
| TmApp | 2.74260866109489 | 2.743927237333513 | +0.001319 |
| HIC | 0.4372717171539347 | 0.4376955171135285 | +0.000424 |

原因: Stage4 historical OOF と Stage5 nested refit の手順差。性能改善として解釈しない。

## 4. レポート修正箇所

| § | 修正内容 |
|---|---|
| §3 | meta-level leakage 説明を修正（base OOF は OOS、meta in-sample が問題） |
| §6 | TmApp/HIC simple blend 解釈追加 |
| §8 | meta weight stability 表を本文化 |
| §9–10 | calibration Shadow 未再現 / HIC MAE 非改善を明記 |
| §11–12 | residual best candidate 具体化 |
| §13 | historical vs nested identity 分離表 |
| §14–15 | 最終比較表（method, branch, Shadow, status） |
| §16 | HIC high-tail 表を本文化 |
| §17 | Shadow lock 監査 + simple blend Shadow 完遂 |
| §18 | leakage audit 要約 |
| §19 | branch 別結論に整理 |
| §20–21 | finalization shortlist + 次ステップ |

## 5. 更新 artifact

- `STAGE5_REPORT_JA.md`（全面改訂）
- `stage5_shadow_results.csv`（HIC simple blend 2 件追加）
- `stage5_hic_tail_diagnostics.csv`（simple blend primary/shadow 追加）
- `stage5_prediction_diagnostics.csv`（simple blend shadow 追加）

## 6. 未実施（protocol 遵守）

- Primary 再探索なし
- finalist 追加なし
- Test prediction 未作成
- Public / Private 未参照
