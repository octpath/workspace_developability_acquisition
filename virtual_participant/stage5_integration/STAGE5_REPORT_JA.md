# Stage 5 — Cross-fitted Model Integration, Calibration & Residual Modeling

## 1. このStageで何をしたか

Stage4 までで sequence / PLM / basic structure / advanced structure の各情報階層を評価し、新規物理 feature の限界利益が小さくなった。Stage5 では新規 feature 追加ではなく、frozen base learner（各 target 7 モデル）の prediction を対象に、nested cross-fitted な ensemble、calibration、residual modeling を実施した。meta learner 評価では full Primary OOF をそのまま meta-training に使わず、outer test fold を完全隔離する nested procedure を採用した。Primary で finalist を `STAGE5_SHADOW_LOCK.json` に freeze し、Shadow lock 後に監査した。本レポートは既存 artifact の監査に基づき、exact value で再集計した。

## 2. Stage 4までの到達点

**Historical incumbent**（Stage4 registry 記録値）:

| Target | experiment_id | Primary | Shadow |
|---|---|---:|---:|
| TmApp | `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt` | 2.7426 | 2.7704 |
| HIC | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` | 0.4373 | 0.4332 |

## 3. なぜ単純なOOF stackingではいけないか

base model の OOF prediction 自体は、各抗体をその抗体を学習に使っていない model で予測しているため、**base-level では out-of-sample** である。

しかし全 162 件の base OOF prediction を meta feature として使い、同じ 162 件の label で meta learner を fit し、その同じ 162 件で meta learner の MAE を評価すると、**meta learner 自身が in-sample 評価**になる。base OOF prediction 自体は out-of-sample であるため **base-level data leakage ではない**。問題は、meta learner の training rows 上で性能を評価することによる **optimistic / in-sample evaluation bias**（meta-level in-sample evaluation bias）である。厳密な data leakage というより、楽観的な meta-level performance estimate を生む。

Stage5 では outer test fold を完全に隔離し、outer train 内だけで inner OOF prediction を作って meta learner を学習する nested cross-fitting により、この bias を回避した。base prediction 自体に in-sample leakage がある、とは述べない。

## 4. 使用したbase model

各 target 7 モデル（`stage5_base_model_inventory.csv` / `stage5_base_models.json`）。Stage1 classical、Stage2/2b PLM、Stage3 structure / provisional fusion、Stage4 advanced / incumbent fusion。hyperparameter は frozen（Stage5 で再 Optuna なし）。

## 5. Base model間のprediction / residual相関

`stage5_prediction_correlation_matrix_*.csv` / `stage5_residual_correlation_matrix_*.csv` 参照。残差相関は一般に高く（Stage4 と同様 0.87–0.91 程度）、強い error complementarity は限定的。ensemble gain が小さい主要因の一つと解釈する。

## 6. Simple mean ensemble

事前定義の equal-weight blend（weight fitting なし）。各 base model の Stage5 refit OOF（Primary CV）の単純平均。

### TmApp

| experiment_id | Primary | Δ vs nested identity |
|---|---:|---:|
| `TmApp__SIMPLE_blend_plm_struct` | 2.8411 | +0.0971 |
| `TmApp__SIMPLE_blend_seq_adv` | 2.9196 | +0.1756 |
| `TmApp__SIMPLE_blend_seq_struct_adv` | 2.9553 | +0.2114 |

**解釈:** 単純平均では Stage5 nested identity baseline（2.7439）を上回らず、model diversity があるだけで自動的に ensemble gain が得られるわけではなかった。

### HIC

| experiment_id | Primary | Shadow | Shadow status |
|---|---:|---:|---|
| `HIC__SIMPLE_blend_esm2_surf` | 0.4325 | — | 未 lock |
| `HIC__SIMPLE_blend_seq_surf` | 0.4258 | 0.4329 | STAGE5_EQUIVALENT |
| `HIC__SIMPLE_blend_seq_surf_adv` | **0.4252** | **0.4321** | **STAGE5_SHADOW_CONFIRMED** |

**解釈:** Primary では low-complexity な equal-weight blend が learned meta model と同等以上の性能を示した。N=162 の small-N setting では、weight を学習する自由度を増やすより単純平均が安定する可能性を示唆する。`HIC__SIMPLE_blend_seq_surf_adv` は Shadow lock 済み finalist であり、Shadow でも Stage4 historical incumbent（0.4332）をわずかに上回った（0.4321）。ただし改善幅は小さい。

## 7. Convex / NNLS / Ridge stacking

nested cross-fitted meta learner 結果（`stage5_nested_stack_results.csv`）。

### TmApp（performance set, 4 base models）

| experiment_id | meta | Primary | Δ vs nested identity |
|---|---|---:|---:|
| `TmApp__META_performance__ridge_100.0` | Ridge α=100 | **2.7135** | **−0.0304** |
| `TmApp__META_performance__ridge_10.0` | Ridge α=10 | 2.7543 | +0.0104 |
| `TmApp__META_diversity__ridge_100.0` | Ridge α=100 | 2.7433 | −0.0006 |
| `TmApp__META_performance__convex_mae` | convex MAE | 2.7979 | +0.0540 |

Primary 最良 learned stack: **`TmApp__META_performance__ridge_100.0`**（Ridge stack, α=100, performance-focused 4-model set）。

### HIC（diversity set が Primary 最良）

| experiment_id | meta | Primary | Shadow | Shadow status |
|---|---|---:|---:|---|
| `HIC__META_diversity__nnls_norm` | NNLS normalized | **0.4273** | 0.4424 | STAGE5_PRIMARY_ONLY_GAIN |
| `HIC__META_diversity__quantile_0.01` | Quantile α=0.01 | 0.4289 | — | 未 lock |
| `HIC__META_performance__nnls_norm` | NNLS normalized | 0.4318 | — | 未 lock |

Primary 最良 learned stack: **`HIC__META_diversity__nnls_norm`**。Shadow では Stage4 historical incumbent を上回らず（0.4424 vs 0.4332）。

## 8. Meta weight stability

`stage5_stack_weights.csv` より。Primary MAE だけで meta learner を評価しない。

### TmApp best stack — `TmApp__META_performance__ridge_100.0`

base model 順: (0) S3INC+INTERACTIONS, (1) overall+RASA, (2) AbLang2+SEQ_BASIC, (3) ADV_TMAPP_ALL

| base model | mean weight | SD | min | max |
|---|---:|---:|---:|---:|
| S3INC+INTERACTIONS | 0.305 | 0.060 | 0.200 | 0.346 |
| overall+RASA | 0.493 | 0.143 | 0.352 | 0.724 |
| AbLang2+SEQ_BASIC | 0.372 | 0.186 | 0.134 | 0.576 |
| ADV_TMAPP_ALL | 0.361 | 0.075 | 0.298 | 0.484 |

Ridge stack では **negative weight は発生しなかった**（全 fold で非負）。fold 間で overall+RASA と AbLang2+SEQ_BASIC への weight 配分に変動があるが、4 モデルすべてに非ゼロ weight が割当てられ、one-model domination は見られない。intercept は fold ごとに大きな負値（≈−30〜−43）だが、これは Ridge の定数項であり cancellation 問題としては weight 符号の混在ほど深刻ではない。

### HIC best learned stack — `HIC__META_diversity__nnls_norm`

base model 順: (0) SEQ_PLUS_ANTIBODY, (1) ESM2 Heavy, (2) SURFACE_CHEM, (3) S3INC+SURFACE_PATCH

| base model | mean weight | SD | min | max |
|---|---:|---:|---:|---:|
| SEQ_PLUS_ANTIBODY | 0.000 | 0.000 | 0.000 | 0.000 |
| ESM2 Heavy | 0.046 | 0.088 | 0.000 | 0.201 |
| SURFACE_CHEM | 0.451 | 0.119 | 0.324 | 0.592 |
| S3INC+SURFACE_PATCH | 0.503 | 0.164 | 0.278 | 0.668 |

NNLS normalized では **SEQ_PLUS_ANTIBODY の weight は全 fold で 0**（実質 3-model stack）。SURFACE_CHEM と S3INC+SURFACE_PATCH に weight が集中するが、fold 間 SD は 0.12–0.16 程度。Primary 改善（0.4273）に対して Shadow 未再現（0.4424）であり、**score 改善に比して generalization は不安定**と判断する。

## 9. TmApp calibration

nested affine calibration（incumbent base prediction に対する post-hoc 補正）:

| method | Primary | Shadow | Shadow status | Δ vs nested identity (P) |
|---|---:|---:|---|---:|
| identity | 2.7439 | 2.7708 | STAGE5_EQUIVALENT | 0.0000 |
| OLS affine | **2.7278** | 2.7789 | **STAGE5_PRIMARY_ONLY_GAIN** | **−0.0161** |
| quantile affine | 2.7607 | — | 未 lock | +0.0168 |

**結論:** Primary では OLS affine が nested identity を上回ったが、**Shadow では再現せず**（2.7789 > 2.7708 incumbent Shadow）、robust な calibration gain とは判断しない。

## 10. HIC calibrationとprediction shrinkage

| method | Primary | Shadow | Δ vs nested identity (P) |
|---|---:|---:|---:|
| identity | 0.4377 | 0.4338 | 0.0000 |
| OLS affine | 0.4623 | — | +0.0246 |
| quantile affine | 0.4446 | — | +0.0069 |

**shrinkage 診断**（nested identity, Primary）: slope ≈ 1.017, pred SD / true SD ≈ 0.39（`stage5_prediction_diagnostics.csv`）。prediction shrinkage の兆候は確認できる。

**結論:** **simple affine calibration は overall MAE を改善しなかった**（OLS は悪化、quantile も identity より悪い）。bias の存在と、単純な post-hoc calibration の有効性は別問題である。calibration slope=1 自体を目的にはしていない。

## 11. Residual modeling — TmApp

| experiment_id | family | Ridge α | λ | Primary | Δ vs nested identity | Shadow |
|---|---|---|---|---:|---:|---|
| `TmApp__RESID_ADV_INVFOLD__Ridge100.0__lam0.5` | ADV_INVFOLD | 100 | 0.50 | **2.7279** | −0.0160 | 未 lock |
| `TmApp__RESID_ADV_UNSAT_POLAR__Ridge1.0__lam0.5` | ADV_UNSAT_POLAR | 1 | 0.50 | 2.7415 | −0.0024 | 未 lock |

best residual は nested identity をわずかに下回るが、Shadow 未監査。**residual modeling に安定した追加価値は認めなかった**（改善幅小、Shadow 未確認）。

## 12. Residual modeling — HIC

| experiment_id | family | Ridge α | λ | Primary | Δ vs nested identity | Shadow |
|---|---|---|---|---:|---:|---|
| `HIC__RESID_ADV_SURFACE_PATCH__Ridge10.0__lam0.25` | ADV_SURFACE_PATCH | 10 | 0.25 | **0.4351** | −0.0026 | 未 lock |
| `HIC__RESID_COMPACT_SURF_PATCH__Ridge100.0__lam0.25` | COMPACT | 100 | 0.25 | 0.4367 | −0.0010 | 未 lock |

いずれも nested identity（0.4377）を小幅に下回るのみ。**residual modeling に安定した追加価値は認めなかった**。

## 13. Historical incumbentとnested identity baseline

Stage4 historical score と Stage5 nested procedure 内の identity baseline は、refit / fold 手順の差により完全一致しない。

| Target | Stage4 historical (P) | Stage5 nested identity (P) | difference (P) | Stage4 historical (S) | Stage5 nested identity (S) |
|---|---:|---:|---:|---:|---:|
| TmApp | 2.7426 | 2.7439 | +0.0013 | 2.7704 | 2.7708 | 
| HIC | 0.4373 | 0.4377 | +0.0004 | 0.4332 | 0.4338 |

**Stage5 branch の incremental value 評価では、原則 Stage5 nested identity baseline との比較を優先する。** 上記微差（0.001–0.002 程度）は性能改善とは解釈しない。

Stage5 nested identity = `CAL_identity__incumbent`（meta 補正なしの frozen incumbent refit）。

## 14. TmApp最終比較

| method | branch | Primary | Shadow | Δ vs nested identity (P) | Shadow status |
|---|---|---:|---:|---:|---|
| Stage4 historical incumbent | incumbent | 2.7426 | 2.7704 | −0.0013 | — |
| Stage5 nested identity | calibration/identity | 2.7439 | 2.7708 | 0.0000 | STAGE5_EQUIVALENT |
| best simple blend | simple_blend | 2.8411 | — | +0.0971 | 未 lock |
| best learned stack | ridge_stack | **2.7135** | **2.7591** | **−0.0304** | **STAGE5_SHADOW_CONFIRMED** |
| best calibration | OLS affine | 2.7278 | 2.7789 | −0.0161 | STAGE5_PRIMARY_ONLY_GAIN |
| best residual | residual | 2.7279 | — | −0.0160 | 未 lock |
| **Stage5 Primary best** | ridge_stack | **2.7135** | **2.7591** | **−0.0304** | **STAGE5_SHADOW_CONFIRMED** |

Primary best = Shadow-confirmed best: **`TmApp__META_performance__ridge_100.0`**（Ridge stack α=100, performance 4-model set）。

## 15. HIC最終比較

| method | branch | Primary | Shadow | Δ vs nested identity (P) | Shadow status |
|---|---|---:|---:|---:|---|
| Stage4 historical incumbent | incumbent | 0.4373 | 0.4332 | −0.0004 | — |
| Stage5 nested identity | calibration/identity | 0.4377 | 0.4338 | 0.0000 | STAGE5_EQUIVALENT |
| best simple blend | simple_blend | **0.4252** | **0.4321** | **−0.0125** | **STAGE5_SHADOW_CONFIRMED** |
| best learned stack | nnls_norm | 0.4273 | 0.4424 | −0.0104 | STAGE5_PRIMARY_ONLY_GAIN |
| best calibration | identity | 0.4377 | 0.4338 | 0.0000 | STAGE5_EQUIVALENT |
| best residual | residual | 0.4351 | — | −0.0026 | 未 lock |
| **Stage5 Primary best** | simple_blend | **0.4252** | **0.4321** | **−0.0125** | **STAGE5_SHADOW_CONFIRMED** |

Primary best method: **`HIC__SIMPLE_blend_seq_surf_adv`**（equal mean of ESM2+SEQ_ALL fusion / SURFACE_CHEM / ADV_SURFACE_PATCH base OOF）。Shadow lock 済み finalist であり、初回レポート時点で Shadow 未計算だったが、**frozen configuration（equal mean 固定）で Shadow 評価を完遂**した（0.4321, STAGE5_SHADOW_CONFIRMED）。

Primary best と Shadow-confirmed best learned stack は異なる。learned stack（nnls_norm）は Shadow 未再現。

## 16. HIC frozen high-tail診断

凍結定義: **HIC ≥ 10.5372 min, N = 17**。tail MAE は model-selection objective ではない。

**Provenance（Primary CV）:** 下表の tail metric はすべて Stage5 nested refit OOF prediction から算出（`stage5_hic_tail_diagnostics.csv`）。OOF source は Primary CV fold assignment（`virtual_participant/stage0_cv/cv_primary.csv`）。Stage4 registry 記録の overall MAE（incumbent Primary **0.4373** / Shadow **0.4332**）は refit 手順差により nested identity OOF overall（**0.4377** / **0.4338**）と一致しない。**registry score と nested OOF tail metric を同一 prediction として混在させない。**

### Primary CV

| model | experiment_id | OOF source | overall MAE | non-tail MAE | tail MAE | tail bias | underpred | pred SD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Stage5 nested identity (incumbent) | `HIC__CAL_identity__incumbent` | Stage5 nested refit OOF, Primary CV | 0.4377 | 0.2844 | 1.7449 | −1.7449 | 17/17 | 0.3151 |
| best simple blend | `HIC__SIMPLE_blend_seq_surf_adv` | Stage5 refit base OOF mean, Primary CV | 0.4252 | 0.2659 | 1.7839 | −1.7839 | 17/17 | 0.2733 |
| best learned stack | `HIC__META_diversity__nnls_norm` | Stage5 nested stack OOF, Primary CV | 0.4273 | 0.2809 | **1.6756** | −1.6756 | 17/17 | 0.3477 |

**参考（registry のみ、tail metric なし）:** Stage4 historical incumbent registry Primary = 0.4373, Shadow = 0.4332（§2）。Stage4 OOF file（`stage4_advanced_structure/oof/HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt.csv`）は Stage5 nested identity OOF と同一 prediction であるため、tail 表に別行として重複掲載しない。

**所見:**

- 全候補で **underprediction 17/17 が継続**（系統的 high-tail shrinkage は解消していない）。
- learned stack（nnls_norm）は tail MAE のみ nested identity 比で部分的緩和（1.745 → 1.676）するが、overall MAE 改善の主因は non-tail 側（0.284 → 0.281）であり、tail bias の符号は変わらない。
- simple blend は overall / non-tail を改善する一方、Primary tail MAE はやや悪化（1.784）。Shadow tail MAE 1.815（`HIC__SIMPLE_blend_seq_surf_adv`, Shadow CV）。
- center/non-tail MAE の大幅悪化は見られない。

Stage4 から tail underprediction count は改善していない（依然 17/17）。

## 17. Primary / Shadow整合性

| experiment_id | Primary | Shadow | status |
|---|---:|---:|---|
| TmApp incumbent | 2.7439 | 2.7708 | STAGE5_EQUIVALENT |
| TmApp ridge stack α=100 | 2.7135 | 2.7591 | **STAGE5_SHADOW_CONFIRMED** |
| TmApp OLS calibration | 2.7278 | 2.7789 | STAGE5_PRIMARY_ONLY_GAIN |
| HIC incumbent | 0.4377 | 0.4338 | STAGE5_EQUIVALENT |
| HIC simple blend seq_surf_adv | 0.4252 | 0.4321 | **STAGE5_SHADOW_CONFIRMED** |
| HIC simple blend seq_surf | 0.4258 | 0.4329 | STAGE5_EQUIVALENT |
| HIC nnls_norm stack | 0.4273 | 0.4424 | STAGE5_PRIMARY_ONLY_GAIN |
| HIC identity calibration | 0.4377 | 0.4338 | STAGE5_EQUIVALENT |

Shadow は Stage1–4 でも監査に使用しており、**completely untouched test set ではない**。未知 Test への性能保証とは言えない。

Shadow lock 後に finalist を追加していない。初回 Shadow 実行時に lock 済み HIC simple blend の Shadow 評価が漏れていたため、frozen config で完遂した（finalist 追加ではない）。

## 18. Leakage audit

`STAGE5_LEAKAGE_AUDIT_JA.md` / `tests/test_stage5_leakage.py` に基づく結論:

- outer test fold の行は base learner / meta learner / calibrator / residual model の training に含めていない。
- meta learner は outer-train 上の inner OOF predictions + labels のみで fit し、outer-test predictions へ適用。
- calibrator と residual target も nested cross-fitting。
- 全 162 行に exactly-one outer OOF prediction。
- Primary / Shadow fold を混同していない。
- NaN / inf prediction なし。HIC frozen tail N=17 を確認。

**Status: PASS**

## 19. Stage 5で分かったこと

1. TmApp では単純平均 ensemble は nested identity を上回らなかった。
2. TmApp では learned Ridge stack（α=100）に Primary 改善（−0.030）があり、Shadow でも改善方向が維持された（STAGE5_SHADOW_CONFIRMED）。
3. TmApp affine calibration（OLS）は Primary では改善したが Shadow 再現せず、robust gain とは言えない。
4. HIC では Primary 上、simple equal-weight blend（`HIC__SIMPLE_blend_seq_surf_adv`）が最良だった。Shadow lock 済みで、Shadow でも Stage4 historical をわずかに上回った。
5. HIC learned NNLS stack は Primary では改善（0.4273）したが、Shadow では Stage4 incumbent を上回らなかった（STAGE5_PRIMARY_ONLY_GAIN）。
6. HIC affine calibration は overall MAE を改善しなかった。
7. residual modeling の追加価値は限定的（小幅 Primary 改善、Shadow 未監査）。
8. base model 間の residual correlation が高く、大きな ensemble gain は得にくい。
9. high-HIC prediction shrinkage（17/17 underprediction）は Stage5 後も主要な残存課題である。

## 20. Finalizationへ持ち越す候補

### TmApp（最大 3）

| role | experiment | Primary | Shadow | status | reason |
|---|---|---:|---:|---|---|
| conservative incumbent | `TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt` | 2.7426 | 2.7704 | Stage4 historical | 最も保守的、Shadow equivalent |
| best Shadow-confirmed integrated | `TmApp__META_performance__ridge_100.0` | 2.7135 | 2.7591 | STAGE5_SHADOW_CONFIRMED | nested stack, Primary+Shadow 改善 |
| modality-diverse alternative | `TmApp__META_diversity__ridge_100.0` | 2.7433 | — | Primary ≈ identity | diversity set, Shadow 未 lock |

### HIC（最大 3）

| role | experiment | Primary | Shadow | status | reason |
|---|---|---:|---:|---|---|
| conservative incumbent | `HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt` | 0.4373 | 0.4332 | Stage4 historical | Shadow equivalent, 保守的 |
| best Shadow-confirmed integrated | `HIC__SIMPLE_blend_seq_surf_adv` | 0.4252 | 0.4321 | STAGE5_SHADOW_CONFIRMED | Primary+Shadow 改善, equal-weight |
| Primary-best learned alternative | `HIC__META_diversity__nnls_norm` | 0.4273 | 0.4424 | STAGE5_PRIMARY_ONLY_GAIN | learned stack, Shadow 未再現 |

この段階では 1 モデルに決定しない。

## 21. 次に進むべきこと

Stage5 監査後、大規模な Primary 探索は続けない。次は **Finalization / Pre-Test Lock** へ進む:

- Stage1–5 結果の総括
- 各 target 2–3 モデルへの shortlist 確定
- exact model specification freeze
- full Dev fit recipe freeze
- Test prediction 作成（本レポート修正時点では未作成）
- prediction sanity audit
- submission hash freeze
- その後初めて organizer 側 Public/Private 評価

assay-matched electrostatics や authorized energy tool は横枝として保持する。

---

**最終状態:** `STAGE5_FINAL_REPORT_FROZEN`
