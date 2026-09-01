# Round 2 Gate 0 — Diagnostic Report

**状態:** `ROUND2_GATE0_DIAGNOSTICS_COMPLETE_READY_FOR_DEEP_RESEARCH`

Organizer reveal 後・Round 2 訓練前の **diagnostic-only** 解析。新規 model training / Optuna / feature engineering は行っていない。

**HIC high-tail threshold:** ≥ **10.5372 min**（Train/Dev q90 凍結）  
**Dev high-tail N=17 / Test high-tail N=13**

---

## A. HIC high-tail — PRIMARY (`HIC__SIMPLE_blend_seq_surf_adv`)

### Gate0 最重要判定

**CASE A: `RANKABLE_BUT_COMPRESSED`**

high-tail **classification**（ROC-AUC / AP / top-rank 集中度）は良好だが、**予測 absolute magnitude が系統的に低い**（dynamic-range compression + underprediction）。

### A1. 全体 rank correlation

| split | Pearson | Spearman | Kendall |
|-------|--------:|---------:|--------:|
| Dev OOF | 0.672 | 0.623 | 0.464 |
| Test | 0.600 | 0.557 | 0.401 |

### A2. high-tail classification

| metric | Dev | Test |
|--------|----:|-----:|
| ROC-AUC | 0.830 | **0.856** |
| Average Precision | 0.477 | 0.505 |
| Precision@13 | 0.353 | 0.538 |
| Recall@13 | 0.353 | **0.538** |
| Recall@20 | 0.471 | 0.615 |
| Enrichment top 10% | 3.53× | 6.15× |
| Enrichment top 20% | 2.94× | 4.73× |
| **top-13 predicted 中の true high-tail** | — | **7 / 13** |

Test では true high-tail 13 件のうち **7 件**が predicted top-13 に含まれる。

### A3. percentile analysis（true high-tail）

| split | median pred percentile | min | max | top10% 内 | top20% 内 |
|-------|----------------------:|----:|----:|----------:|----------:|
| Dev | 87.7 | 2.5 | 99.4 | 7/17 | 12/17 |
| Test | **92.6** | 35.2 | 100.0 | **8/13** | **9/13** |

詳細: `hic_tail_rankability.csv`

### A4. high-tail 内部 rank

| split | Spearman(true, pred) within tail |
|-------|--------------------------------:|
| Dev | 0.309 |
| Test | 0.132 |

tail 内の強弱識別は **弱い**（特に Test）。

### A5. dynamic-range compression

| split | true SD | pred SD | pred/true SD | tail bias | reg: pred = a + b×true |
|-------|--------:|--------:|-------------:|----------:|------------------------|
| Dev | 0.832 | 0.273 | 0.328 | −1.784 | a=6.89, b=0.039 |
| Test | 0.856 | 0.285 | **0.333** | **−2.132** | a=7.05, b=0.037 |

5-quantile bin 別 MAE / bias: `hic_dynamic_range_diagnostics.csv`

**解釈:** slope ≈ 0.04 の Dev/Test 回帰は、全体レンジに対して pred が **ほぼ flat** に近い。13/13 underprediction（Test tail）、pred SD ≈ 0.29 vs true SD ≈ 0.86。

### A6. 判定根拠

- ✅ classification AUC/AP 良好、true high-tail が predicted 上位 percentile に集中（Test median 92.6%）
- ✅ しかし pred magnitude は true より大幅に低い（compression）
- → **RANKABLE_BUT_COMPRESSED**

---

## B. HIC — model family 別 tail 挙動

`hic_tail_model_comparison.csv` 参照。

| model family | Test ROC-AUC | AP | Recall@13 | Tail MAE | pred SD ratio |
|--------------|-------------:|---:|----------:|---------:|--------------:|
| Stage1 classical | 0.827 | 0.288 | 0.385 | 2.295 | 0.290 |
| ESM2 Heavy | 0.846 | 0.418 | 0.462 | 2.175 | 0.348 |
| ESM2 + SEQ_ALL | 0.770 | 0.302 | 0.385 | 2.213 | 0.359 |
| **SURFACE_CHEM** | 0.811 | 0.483 | **0.538** | 1.953 | **0.509** |
| ADV_SURFACE_PATCH | 0.852 | **0.530** | 0.385 | 2.229 | 0.297 |
| **Round1 primary blend** | **0.856** | 0.505 | **0.538** | 2.132 | 0.333 |

**問いへの回答:**

- **sequence vs surface:** SURFACE_CHEM が Recall@13・dynamic range（pred SD ratio）で最良。patch 単独は AP 高いが Recall@13 は中程度。
- **ensemble:** primary blend は ROC-AUC 最大・Recall@13 は SURFACE_CHEM と同率（0.538）だが、ESM2+SEQ 単独（0.385）より改善。
- **compression:** ensemble は ESM2+SEQ より pred SD ratio **低下**（0.359→0.333）→ **compression やや増加**。

---

## C. TmApp — CV→Test gap 構造

`tmapp_cv_test_gap_models.csv`（111 models with Private score）

### C1–C2. model-level gap & constant-offset hypothesis

**Private = 1.829 + 0.510 × Primary_CV**（R²=0.516, residual SD=0.156）  
**AllTest = 1.829 + 0.510 × Primary_CV**（同傾向）

- intercept **a ≈ +1.83** が dominant
- slope **b ≈ 0.51 < 1** → 純粋な constant offset（b≈1, a>0）ではないが、**dataset-wide affine shift + 弱いスケーリング**に近い
- 強い model（低 Primary CV）ほど Private MAE も低いが、**加算 gap は縮小**（下記 C3）

### C3. gap vs model quality

**corr(gap_private, Primary_CV) = −0.704**（Pearson）, **−0.533**（Spearman）

→ **強い model ほど gap が大きいわけではない**。むしろ Primary が良い model ほど gap は小さくなる（gap 定義上 Primary 自体が小さいため）。

### C4–C5. gap by Stage / modality

| Stage | N | median gap | IQR |
|-------|--:|-----------:|----:|
| Stage1 | 57 | 0.305 | 0.26–0.42 |
| Stage2 | 17 | 0.157 | 最小 |
| Stage5 | 28 | 0.399 | 最大級 |

| modality bucket | median gap |
|-----------------|-----------:|
| ensemble/meta | **0.409** |
| classical | 0.304 |
| sequence+structure fusion | 0.311 |

→ gap は **Stage5 ensemble/meta にやや集中**するが、Stage1 も同程度。特定 modality の catastrophic failure というより **広範な dataset-wide shift**。

### C6. target-range dependency（Test quantile bins）

PRIMARY (`TmApp__META_performance__ridge_100.0`):

| true TmApp bin | signed bias | MAE |
|----------------|------------:|----:|
| low (52–66°C) | **+4.68** | 4.79 |
| mid-low | +0.70 | 1.55 |
| mid | −1.05 | 2.41 |
| mid-high | −2.41 | 2.80 |
| **high (74–82°C)** | **−4.63** | 4.63 |

→ **low-Tm で overprediction、high-Tm で underprediction** の U 字型 bias。HIC tail ほど単純ではないが、range-dependent systematic bias あり。

---

## D. Primary vs Shadow — matched subset

Primary / Shadow / Public / Private / AllTest が **全て non-null の同一 subset** で比較。

| Target | N | Private 相関最良 | All Test 相関最良 | 判定 |
|--------|--:|-----------------|------------------|------|
| TmApp | **27** | Primary (r=0.455) | CV_WORST (r=0.548) | **PRIMARY_BETTER** vs Shadow for Private |
| HIC | **36** | Primary (r=0.770) | Primary (r=0.770) | **PRIMARY_BETTER** |

同一 N subset でも Primary ≧ Shadow。mean / worst-of-two は Private で Primary を上回らず。

詳細: `matched_cv_comparison_tmapp.csv`, `matched_cv_comparison_hic.csv`

**注意:** matched N が小さい（Shadow 欠損 model 多い）。descriptive のみ。

---

## E. Milestone trajectory

主要 milestone のみ（model multiplicity 排除）。`milestone_trajectory_*.csv`

### TmApp

| milestone | Primary CV | Public | Private |
|-----------|----------:|-------:|--------:|
| Baseline median | 3.437 | 3.784 | 3.772 |
| Stage1 classical | 3.101 | 3.587 | 3.352 |
| Stage2 AbLang2+SEQ | 2.776 | 3.239 | 3.263 |
| Stage4 interaction | 2.743 | 3.332 | **3.161** |
| Stage5 final stack | **2.714** | **3.231** | 3.212 |

→ CV は Stage1→5 で monotonic 改善。**Private は Stage4 incumbent (3.16) が最良**で、Stage5 stack (3.21) は僅かに悪化。

### HIC

| milestone | Primary CV | Public | Private |
|-----------|----------:|-------:|--------:|
| Baseline median | 0.518 | 0.535 | 0.510 |
| Stage3 SURFACE_CHEM | 0.440 | 0.483 | 0.438 |
| Stage4 patch fusion | 0.437 | 0.440 | 0.439 |
| Stage5 final blend | **0.425** | **0.420** | **0.424** |

→ CV / Public / Private が milestone 間で **well-aligned** に改善。

Plot: `plots/milestone_trajectory.png`

---

## H. 必須問いへの明示回答

### HIC

| # | 問い | 回答 |
|---|------|------|
| Q1 | high-tail は rankable か？ | **Yes（部分的）**。Test Spearman=0.557, ROC-AUC=0.856, top-13 capture=7/13, tail median percentile=92.6% |
| Q2 | 主問題は compression か？ | **Yes**。pred SD/true SD=0.33, tail bias=−2.13, 13/13 underprediction |
| Q3 | どの family が tail 識別に寄与？ | **SURFACE_CHEM**（Recall@13=0.538, 最大 dynamic range）。patch は AP 最大 (0.530) |
| Q4 | ensemble で ranking / compression？ | Ranking **改善**（Recall@13: 0.385→0.538 vs ESM2+SEQ）。Compression **やや増加**（SD ratio 0.359→0.333） |

### TmApp

| # | 問い | 回答 |
|---|------|------|
| Q1 | constant offset か？ | **近似 affine shift**（intercept≈+1.83, slope≈0.51）。pure constant offset ではない |
| Q2 | 強い model ほど gap 大？ | **No**。corr(gap, Primary)=−0.70（強い model ほど gap 小） |
| Q3 | Stage/modality 偏り？ | **広範 shift**。Stage5/meta median gap やや大きいが Stage1 も同程度 |
| Q4 | target range bias？ | **Yes**。low-Tm overpred (+4.7°C), high-Tm underpred (−4.6°C) |

### CV

| 問い | 回答 |
|------|------|
| matched subset で最良指標 | **PRIMARY_BETTER**（TmApp N=27, HIC N=36）。Shadow / mean / worst は Private で Primary を上回らず |

---

## 出力一覧

| ファイル | 内容 |
|---------|------|
| `hic_tail_rankability.csv` | 抗体別 percentile / rank |
| `hic_tail_model_comparison.csv` | model family 別 tail 指標 |
| `hic_dynamic_range_diagnostics.csv` | quantile bin + regression |
| `tmapp_cv_test_gap_models.csv` | model-level gap |
| `tmapp_gap_by_stage.csv` | Stage 別 gap 要約 |
| `tmapp_gap_by_modality.csv` | modality 別 gap 要約 |
| `tmapp_target_range_diagnostics.csv` | TmApp quantile bias |
| `matched_cv_comparison_*.csv` | matched CV 比較 |
| `milestone_trajectory_*.csv` | milestone 推移 |

### plots/

- `hic_true_vs_pred_scatter.png`
- `hic_tail_diagnostics_panel.png`（rank / calibration / percentile / family 比較）
- `tmapp_gap_diagnostics_panel.png`
- `milestone_trajectory.png`

---

## DeepResearchで調べるべき技術課題

（文献調査は **未開始**。Round 2 Gate 0 後の handoff 候補。）

### HIC

1. Surface aggregation propensity / retention 関連 descriptor
2. Hydrophobic / aromatic patch + spatial neighborhood 記述
3. Chromatographic retention predictor（high-HIC regime 特化）
4. Tail-aware loss / quantile regression / heteroscedastic calibration
5. Rank-preserving dynamic-range expansion（post-hoc calibration）

### TmApp

1. Packing defect / cavity / internal void descriptors
2. Interface energetics beyond contact counts
3. Frustration / local strain proxies
4. Thermodynamic stability predictors（ΔΔG-like, inverse folding 活用方針）
5. Domain-shift robust CV + absolute MAE calibration protocol

---

**状態:** `ROUND2_GATE0_DIAGNOSTICS_COMPLETE_READY_FOR_DEEP_RESEARCH`
