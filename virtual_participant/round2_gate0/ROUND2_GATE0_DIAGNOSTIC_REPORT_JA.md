# Round 2 Gate 0 — Diagnostic Report (Final Audit)

**状態:** `ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH`

Diagnostic-only（新規 training / Optuna / feature engineering なし）。

## Executive Summary

### HIC: `RANKABLE_BUT_COMPRESSED`

- Test ROC-AUC=0.856, top-13 capture=7/13
- OLS slope (Test)=0.204 (audit PASS=True)
- pred SD / true SD=0.333, tail bias=-2.132

### TmApp: `MODEL_RANK_SIGNAL_PRESENT` + `CV_GAIN_ATTENUATION_ON_TEST` + `TARGET_RANGE_COMPRESSION`

- Private ≈ 1.829 + 0.510×Primary (R²=0.516)
- gap=Private−Primary ≈ 1.829 − 0.490×Primary
- corr(gap, Primary)=-0.704 → **強いCV modelほど optimism gap 拡大**
- low-Tm overprediction / high-Tm underprediction（range compression）

HIC high-tail threshold: **≥ 10.5372 min** (Dev N=17, Test N=13)

## A. HIC high-tail (PRIMARY)

**Gate0 判定:** **RANKABLE_BUT_COMPRESSED**

### A1. Rank correlation

| split | Pearson | Spearman |
|-------|--------:|---------:|
| Dev | 0.602 | 0.623 |
| Test | 0.611 | 0.557 |

### A2. Top-k classification metrics（監査済み）

| split | k | TP | Precision@k | Recall@k | enrichment |
|-------|--:|---:|------------:|---------:|-----------:|
| Dev | 13 | 6 | 0.462 | 0.353 | 4.40× |
| Dev | 20 | 8 | 0.400 | 0.471 | 3.81× |
| Test | 13 | 7 | 0.538 | 0.538 | 6.71× |
| Test | 20 | 8 | 0.400 | 0.615 | 4.98× |

Test top-13 predicted 中 true high-tail: **7/13**

監査: `hic_tail_topk_metric_audit.csv`

### A5. Dynamic-range regression（監査済み）

| split | pearson | true_sd | pred_sd | sd_ratio | expected slope | fitted slope | diff |
|-------|--------:|--------:|--------:|---------:|---------------:|-------------:|-----:|
| Dev | 0.602 | 0.807 | 0.273 | 0.339 | 0.204 | 0.204 | 5.55e-17 |
| Test | 0.611 | 0.854 | 0.285 | 0.333 | 0.204 | 0.204 | 5.55e-17 |

**Slope identity check:** PASS

監査: `hic_dynamic_range_regression_audit.csv`

### HIC Q1–Q4

- **HIC Q1: high-tailはrankableか？** Yes（部分的） — Test Spearman=0.557, top13 capture=7/13
- **HIC Q2: rankableなら compressionか？** Yes — pred SD ratio=0.333, OLS slope=0.204 (expected 0.204), tail bias=-2.132
- **HIC Q3: どのfamilyがtail識別に寄与？** Recall@13 最大: SURFACE_CHEM (0.54); ROC-AUC Test 最大: Round1 equal-weight primary (0.856)
- **HIC Q4: ensembleで ranking / compression？** Ensemble vs ESM2+SEQ: Recall@13 0.54 vs 0.38; pred SD ratio 0.333 vs 0.359 → compression 増加

## C. TmApp CV→Test gap

### C2. Regression audit（Private / Public / AllTest）

| metric | intercept | slope | R² | residual SD | N |
|--------|----------:|------:|---:|------------:|--:|
| Private | 1.8293 | 0.5102 | 0.516 | 0.1562 | 111 |
| Public | 2.1748 | 0.4310 | 0.457 | 0.1487 | 111 |
| AllTest | 2.0020 | 0.4706 | 0.548 | 0.1352 | 111 |

監査: `tmapp_regression_audit.csv`（Private / Public / AllTest は **異なる係数**）

### C3. Gap vs Primary CV

- corr(gap_private, Primary_CV): Pearson=-0.704, Spearman=-0.533
- **解釈:** Primary_CV が小さい（= CV 上強い model）ほど Private−Primary gap は **大きい**。
  CV 上の改善幅は Test へ完全には移らず、**CV performance が高い model ほど optimism gap が拡大**する傾向。
- affine compression / gain attenuation（slope≈0.51<1, intercept≈+1.83）。単純 constant offset ではない。

### TmApp Q1–Q4

- **TmApp Q1: affine compression？** Private ≈ 1.829 + 0.510×Primary (R²=0.516)。slope≈0.51<1 → **affine compression / gain attenuation**。単純 constant offset ではない。
- **TmApp Q2: 強いCV modelとgap？** gap=Private−Primary ≈ 1.829 − 0.490×Primary。corr(gap,Primary)=-0.704 → **Primaryが小さい（強いCV model）ほど gap が大きい**。CV改善幅はTestへ完全には移らず、optimism gap が拡大する傾向。
- **TmApp Q3: Stage/modality偏り？** median gap by stage: {'Stage1': 0.3045719980346244, 'Stage2': 0.15722541143009616, 'Stage2b': 0.4216534684098472, 'Stage3': 0.31142878160821263, 'Stage4': 0.30512550474527034, 'Stage5': 0.3987662132583334}; 最高median modality: ensemble/meta
- **TmApp Q4: target range bias？** PRIMARY: low-Tm overprediction (+4.7°C), high-Tm underprediction (−4.6°C) — **TARGET_RANGE_COMPRESSION**

## D. Matched CV comparison

- TmApp matched N=27
  - **Private prediction:** PRIMARY_BETTER（Primary が Shadow/mean/worst より高相関）
  - **All Test:** CV_WORST が最高相関（Pearson=0.548, N=27）
  - ただし N=27 の descriptive result であり、Round2 CV selection rule 変更の根拠とはしない。
- HIC matched N=36: **PRIMARY_BETTER**（Primary 中心維持）

## E. Milestone trajectory（Stage1–5 整列）

### TmApp

| milestone                 | experiment_id                                        |   Primary_CV |   Public |   Private |   All_Test |
|:--------------------------|:-----------------------------------------------------|-------------:|---------:|----------:|-----------:|
| Baseline                  | Median                                               |        3.437 |    3.784 |     3.772 |      3.778 |
| Stage1 classical          | TmApp__SEQ_BASIC__SVROpt                             |        3.101 |    3.587 |     3.352 |      3.469 |
| Stage2 AbLang2+SEQ        | TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt |        2.776 |    3.239 |     3.263 |      3.251 |
| Stage3 structure fusion   | TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt  |        2.754 |    3.346 |     3.205 |      3.275 |
| Stage4 interaction fusion | TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt        |        2.743 |    3.332 |     3.161 |      3.246 |
| Stage5 final stack        | TmApp__META_performance__ridge_100.0                 |        2.713 |    3.231 |     3.212 |      3.221 |

### HIC

| milestone           | experiment_id                                |   Primary_CV |   Public |   Private |   All_Test |
|:--------------------|:---------------------------------------------|-------------:|---------:|----------:|-----------:|
| Baseline            | Median                                       |        0.518 |    0.535 |     0.510 |      0.522 |
| Stage1 classical    | HIC__SEQ_PLUS_ANTIBODY__SVROpt               |        0.473 |    0.485 |     0.463 |      0.474 |
| Stage2 ESM2+SEQ_ALL | HIC__FUSION__esm2__H__SEQ_ALL__SVROpt        |        0.448 |    0.422 |     0.490 |      0.456 |
| Stage3 SURFACE_CHEM | HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt    |        0.440 |    0.483 |     0.438 |      0.461 |
| Stage4 patch fusion | HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt |        0.437 |    0.440 |     0.439 |      0.439 |
| Stage5 final blend  | HIC__SIMPLE_blend_seq_surf_adv               |        0.425 |    0.420 |     0.424 |      0.422 |

Plot: `plots/milestone_trajectory.png`

## 7. Gate0 最終結論

### HIC: `RANKABLE_BUT_COMPRESSED`

- high-tail classification（ROC-AUC / AP / top-k enrichment）は良好
- ただし dynamic-range compression（OLS slope≈0.204, pred/true SD≈0.333）と tail underprediction（bias=-2.132）
- slope identity audit: **PASS**（expected ≈ fitted）

### TmApp: `MODEL_RANK_SIGNAL_PRESENT` + `CV_GAIN_ATTENUATION_ON_TEST` + `TARGET_RANGE_COMPRESSION`

- CV 順位には Test への情報がある（Private regression R²≈0.52）
- ただし **affine compression / gain attenuation**（slope≈0.51<1）であり constant offset ではない
- **強い CV model ほど Private−Primary optimism gap が拡大**（corr≈−0.70）
- low-Tm overprediction / high-Tm underprediction（target range compression）

## DeepResearchで調べるべき技術課題

（文献調査は未開始）

### HIC
- Surface aggregation propensity descriptors beyond total SASA
- Hydrophobic / aromatic patch metrics with spatial neighborhood
- Chromatographic retention proxies for high-HIC regime
- Tail-aware loss / quantile regression for skewed HIC
- Post-hoc calibration preserving rank but expanding dynamic range

### TmApp
- Packing defect / cavity descriptors for thermostability gap
- Interface energetics beyond contact counts
- Frustration / local strain proxies
- Thermodynamic stability predictors (ΔΔG-like)
- Domain-shift robust CV protocols for absolute MAE calibration

---

**状態:** `ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH`
