# HIC Current-State Audit

**Audit type:** repository inventory only（新規学習・Optuna・特徴量再計算・embedding・構造予測・leaderboard最適化なし）  
**Audit HEAD (start):** `2f1e178dedb424f1da60fbd16b8b41e4f35f10ee` (`main`)  
**Deliverables:** `reports/HIC_CURRENT_STATE_AUDIT.md`, `reports/HIC_EXPERIMENT_INVENTORY.csv`, `reports/HIC_EVIDENCE_MATRIX.csv`  
**Scope note:** TmApp factorial / technical report は freeze 済み。本監査は **HIC のみ**。数値は既存ファイルから転記・集計し、推測で埋めない。

---

## 1. Executive summary

HIC（IgG hydrophobic interaction chromatography retention time, **min**）は Shehata et al. 2019 由来の連続回帰タスクとして確立され、競技母集団は **N=324**（HIC∧TmApp）。primary metric は一貫して **MAE**。

リポジトリ内の HIC 研究は大きく **3層** に分かれる。

| Layer | 内容 | 代表パス |
|-------|------|----------|
| A. Gate / split / validity | 連続性妥当性、high-tail、split探索、Trust-CV、仮想コンペ | `gate_b6_*`, `gate_b7_*` |
| B. Organizer feature prospecting / top models | AROMATIC-TOPO、物理バッチ、ensemble、postmortem | `organizer_extension/`, `top_models_feature_bundle/` |
| C. developability_drilldown EXP-H | **EXP-H001–H139**（139 codes）古典〜V3 late fusion | `developability_drilldown/results/` |

**再開時に知っておくべき要点**

1. **プロトコルが複数ある。** `CAND_12528` 上の Gate スコアと、競技本番 `GEN_0001_B_20271100`、drilldown の `canonical_simple_tvt` / `dl_foldlocal_cosine_v3` は **直接比較不可**。
2. **HIGH-HIC 尾部（>11.5 min, n≈13）の系統的過小予測**は複数ゲートで再現。overall MAE 改善でも尾部は十分に直っていない。
3. **HIC の強いレバーは明示的 surface / 物理特徴（特に SURFACE = AROMATIC_TOPO + HYDRO_FIELD）**。V3 の architecture 単独は TEST_mean ≈0.50 で頭打ち（`EXP-H054–H081`）。
4. **露出芳香族は計算上サポートされるが、因果主張は repository が明示拒否**（`GEOMETRY_ADDS_SIGNAL`；「3D芳香族がHICを引き起こす」とはしない）。
5. **TmApp 型 Representation×Topology×Annotation factorial は HIC 未実施**。HIC V3 は annotation=`FULL` 固定。次の HIC 実験は audit 確認後まで起動しない前提。

Machine-readable 詳細: `HIC_EXPERIMENT_INVENTORY.csv`（144行 = 139 EXP-H + 5 gate/organizer 要約）、`HIC_EVIDENCE_MATRIX.csv`（29 claims）。

---

## 2. Dataset / target summary

| 項目 | 値 | Source |
|------|-----|--------|
| Assay | IgG HIC retention time | `gate_b1/reports/assay_definitions.md` |
| Internal column | `hic_rt_min` | 同上 |
| Unit | **min** | 同上 |
| Source study | Shehata et al., Cell Reports 2019（mmc2 `HIC retention time (min)`） | `competition/organizer/PROVENANCE.md` |
| Source complete N | **348**（HIC単独） | assay_definitions |
| Competition N | **324**（HIC ∧ TmApp 非欠損） | `gate_b3/frozen/organizer/final_population.csv` |
| Dev / Test | 162 / 162 | COMPETITION_SPEC / PROVENANCE |
| Public / Private | 81 / 81 | PROVENANCE |
| Production split | **`GEN_0001_B_20271100`** | `competition/organizer/PROVENANCE.md`; `gate_b7_3_principled_split/config/B7_3_RECOMMENDED_SPLIT.json` |
| Diagnostic bands | LOW&lt;10.5 / MEDIUM 10.5–11.5 / HIGH&gt;11.5 min | PROVENANCE / B6.2 |
| Interpretation | 疎水性 / developability **proxy**；**凝集アッセイではない** | assay_definitions |

分布・解釈の詳細は `competition/organizer/SCORE_INTERPRETATION_TECHNICAL_JA.md`（右歪み・高値裾；測定再現性参考 ≈0.12 min は Grade A/B 参考であり Shehata 固有 noise floor ではない）。

---

## 3. Historical experiment map

### 3.1 Timeline（git / gate 文書順）

| 時期・層 | 出来事 | Source |
|----------|--------|--------|
| Gate B6 split search | 候補探索 → 当初 **CAND_04974** → bake-off で **CAND_12528** | `gate_b6_split_search/reports/` |
| Gate B6.2 | HIC 連続回帰妥当性；KEEP continuous；SGKF_Q7 | `gate_b6_2_hic_validity/` |
| Gate B6.4 | HIGH-tail 診断；data-limited | `gate_b6_4_hic_high_tail/` |
| Gate B7 VC / B7.1 / B7.2 | 仮想コンペ・Trust-CV・再評価；多くが CAND_12528 freeze 推奨 | `gate_b7_*` |
| Gate B7.3 | model-blind で **GEN_0001_B_20271100** を推奨 → **競技本番** | PROVENANCE |
| 2026-09-02 | Physical / Advanced / Late batches（AROMATIC 等）post-reveal prospecting | commits `c98cf872`, `6e727c40`, `3136d09e`, `efc1e17f` |
| 2026-09-08 | top model feature bundle | `971d6860` |
| 2026-09-10 | V3 H054–H081 arch + H082–H093 SURFACE fusion | `fc64362c`, `5b8ac2c6`, `3d0e336e` |
| 2026-09-11 | STATIC_SAP / HSP atlas / H102–H113 | `076c03f7`, `210a270d`, `9625151f` |
| 2026-09-12 | residue F1 / global F1 conditioning；H140 DO NOT RUN | `ed6bdbc1`, `58a0e543` |

### 3.2 EXP-H families（drilldown）

| Family | IDs | n | Protocol | 要約（ファイル記載） |
|--------|-----|---|----------|----------------------|
| F01 classical linear | H001–H020 | 20 | `canonical_simple_tvt_primary_shadow` | aromatic / hydro / surface / ESM2 線形 |
| F02 classical XGB | H021–H023 | 3 | 同上 | H021 CONTINUOUS_SURFACE Overall≈0.456 |
| F03 Phase2A Transformer | H024–H033 | 10 | 同上 | H-only ± fusion；H030 Overall≈0.439 |
| F04 classical refinement | H034–H053 | 20 | 同上 | **H047** SVR Overall **0.427473** |
| F05 V3 arch/geometry | H054–H081 | 28 | `dl_foldlocal_cosine_v3_oof_test` | FULL annot；plateau TEST_mean≈0.50 |
| F06 H047 late fusion | H082–H093 | 12 | V3 | **SURFACE** が最一貫；H086 Overall **0.406** |
| F07 STATIC_SAP_KD | H094–H101 | 8 | V3 | **NOT_SUPPORTED** |
| F08 HSP mainline | H102–H113 | 12 | V3 | BM/EIS **supported**；H107 Overall **0.393619** |
| F09 SOURCE24 TRF | H114–H125 | 12 | V3 | **NOT_SUPPORTED** |
| F10 SOURCE_SAP24 XGB | H126–H127 | 2 | V3_xgb | **no_reproduce_0.46** |
| F11 residue F1 | H128–H133 | 6 | V3 | **CASE G**（Ab-level F1 優位） |
| F12 global F1 cond. | H134–H139 | 6 | V3 | **CASE D**；H140 禁止 |

Registry: `developability_drilldown/results/EXPERIMENT_CODES.csv`（139 ACTIVE）、`experiments.csv`。

### 3.3 Gate / organizer（EXP-H 外）

Gate 7ディレクトリ内に **EXP-H\* 記載なし**。モデルは CONST_MEDIAN / SEQ / ESM2 PCA SVR / ESMFN / FUSION / NESTED 等（B5予測の再利用が多い）。organizer 側に AROMATIC-TOPO、cross-family equal-mean、historical Public/Private winners がある。

---

## 4. Evaluation protocol history

### 4.1 Splits

| Split ID | Train / Pub / Priv | 生成法 | 使用箇所 | 備考 |
|----------|-------------------|--------|----------|------|
| CAND_04974 | 162 / 81 / 81 | B6 common search | 初期 freeze manifest | 後に置換 |
| **CAND_12528** | 162 / 81 / 81 | B6.1 bake-off | B6.2, B6.4, B7 VC/1/2 | Pub hash `4d24d595…` |
| **GEN_0001_B_20271100** | 162 / 81 / 81 | B7.3 model-blind SA | **競技本番** | Pub `2a267694…` / Priv `f9d26ae7…`；HIGH 4/3 |
| Dev/Test 162/162 | （role_map） | Gate B3 freeze | 全競技・drilldown 母集団 | sequence group 非重複 |

Stratification / similarity: B7.3 は atomic sequence groups・HIGH∈{3,4}・model-blind q8 L1 等。B7.2 は seed 間で CV↔Pub が大きく不安定（HIC CV↔Pub mean 0.451, std 0.209）。

### 4.2 CV / holdout

| 方式 | どこで | 注意 |
|------|--------|------|
| Outer CV / SGKF_Q7 | B6.2 preferred | Public/Private は選定後確認 |
| Train group CV + one-shot holdout | B6.4 diagnostic | |
| Trust-CV stress / virtual competition | B7.1 / B7 VC | Private は reveal 後診断 |
| `canonical_simple_tvt_primary_shadow` | EXP-H001–H053 等 | Primary/Shadow CV |
| `dl_foldlocal_cosine_v3_oof_test` | EXP-H054+ | fold-local cosine V3；registry の cv_* は V3 では OOF TEST 系 |

### 4.3 Metrics

| Metric | 役割 |
|--------|------|
| **MAE** | Primary（全主要ゲート・EXP-H） |
| Spearman | Secondary / 相関診断 |
| Pearson | **診断のみ**（高裾 leverage で fragile） |
| HIGH-band MAE / HIGH→LOW rate / elev≥10.5 | Tail / developability band |
| Public-winner Private regret | Split / selection 品質 |
| Overall = mean(Public, Private) | drilldown external 要約 |
| Permutation ΔMAE / bootstrap CI | 特徴ブロック寄与 |

### 4.4 直接比較してよいか（最重要）

| 比較ペア | 判定 |
|----------|------|
| 同一 `cv_protocol` 内の EXP-H 同士 | **比較可**（同一 registry） |
| EXP-H canonical_simple vs V3 | **比較不可**（CV定義が異なる） |
| Gate B6.2 Nested MAE（CAND_12528） vs EXP-H Overall（V3/本番系） | **比較不可** |
| CAND_12528 Public/Private vs GEN_0001 Public/Private | **比較不可**（ID集合が異なる） |
| Classical H047 Overall vs H086 Overall | **参考のみ**（モデル級・プロトコル差；レポート自身が caveat） |
| Organizer postmortem Pub/Priv vs pre-reveal prospective | **診断 vs 確認を混同しない** |

---

## 5. Best-known HIC results

**同一プロトコル内の「最良」のみ強調する。**

### 5.1 Gate era（CAND_12528）

| Model | CV MAE | Public | Private | Source |
|-------|-------:|-------:|--------:|--------|
| NESTED_STACK_NNLS | 0.4667 | 0.5135 | 0.4313 | `gate_b6_2_hic_validity/` |
| VC persona C FUSION_ESM2_STRUCT | — | — | **0.4437** (regret 0) | `gate_b7_virtual_competition/` |

### 5.2 Production split GEN_0001（B7.3 再スコア）

| Selection | Model | CV | Pub | Priv | Source |
|-----------|-------|---:|----:|-----:|--------|
| CV-best | NESTED | 0.4667 | 0.4881 | 0.4566 | `07_FINAL_GEN0001_3X3_BENCHMARK.md` |
| Pub-best | FUSION_ESM2_ESMFN | 0.5212 | **0.4650** | 0.4965 | 同上 |
| Priv-best | ESMFN_STRUCTURE | 0.4970 | 0.5163 | **0.4335** | 同上 |

### 5.3 developability_drilldown V3（registry）

| Code | 位置づけ | TEST系 / Overall | Source |
|------|----------|------------------|--------|
| EXP-H086 | F1 SURFACE late fusion（強クラスタ） | Overall **0.406344** | `HIC_H047_FEATURE_FUSION_REPORT.md` |
| EXP-H090 | Scratch + F1 | Overall **0.409375** | 同上 |
| EXP-H107 | SURFACE + HSP EIS | Overall **0.393619** | `HIC_HSP_MAINLINE_REPORT.md` |
| EXP-H137 | FiLM external 強いが CASE D | Overall **0.390740**；TEST_mean は hist F1 より悪い | `HIC_GLOBAL_F1_SURFACE_CONDITIONING_REPORT.md` |
| EXP-H047 | 古典 SVR champion | Overall **0.427473** | `HIC_CLASSICAL_REFINEMENT.md` |

### 5.4 Organizer top models（postmortem 含む）

| 区分 | MAE | Source |
|------|-----|--------|
| BEST_CV cross-family equal-mean | worst≈**0.4346**；Pub/Priv≈0.407/0.456 | `top_models_feature_bundle/results/cross_family_ensemble/` |
| HIST Private winner | Priv≈**0.418** | `HISTORICAL_SINGLE_MODEL_BESTS.md` |
| Stacking | **SKIP**（異種 OOF fold 不整合） | `HIC__STACKING_SKIP.md` |

---

## 6. Failure modes

### 6.1 High-HIC tail

| 項目 | 記録 |
|------|------|
| 定義 | HIGH &gt; **11.5** min |
| n | **13**（Train 6 / Pub 3 / Priv 4）on CAND_12528 |
| 現象 | 系統的過小予測；regression-to-center **STRONG**；sample scarcity **STRONG** |
| Nested upper 10% | MAE **1.4421**；signed bias **−1.4421** |
| HIGH→LOW | Nested Pub 1.00 / Priv 0.50；VC personas 0.75–1.00 |
| Diagnostic winner | PHYSSEQ_STRUCT_Ridge_elevW2：overall MAE 0.5866、HIGH MAE 2.1663；Pub/Priv one-shot 0.6807/0.5597 |
| Verdict | **HIC_VALID_BUT_DATA_LIMITED_COMPETITION_TARGET** |
| Sources | `gate_b6_2_hic_validity/reports/GATE_B6_2_HIC_VALIDITY_FINAL.md`；`gate_b6_4_hic_high_tail/reports/GATE_B6_4_FINAL.md`；`gate_b7_virtual_competition/reports/GATE_B7_VIRTUAL_COMPETITION_FINAL.md` |

再現性: B6.2・B6.4・B7 VC で一貫。overall MAE 改善が尾部を直すという証拠は **弱い／否定的**。

### 6.2 Prediction shrinkage / center regression

B6.2 `02_prediction_shrinkage.md` および B6.4 原因監査で報告。定数中央値からの改善は全体で ~10% 程度（Nested vs const）だが裾は残る。

### 6.3 Metric fragility

Pearson は高裾レバレッジに敏感（B6.2 / B7.3 Pearson sanity）。**MAE 維持**が公式方針。

### 6.4 Split instability

B7.2: seed 間で教育メトリクス不安定 → `SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE`（当時は 12528 維持）。B7.3 で model-blind 近最適 **GEN_0001** へ置換推奨。

---

## 7. Feature evidence

Evidence tier の定義は §9 / `HIC_EVIDENCE_MATRIX.csv` に準拠。

| Feature / family | Evidence | Direction | Reproducibility | Conditions | Source |
|------------------|----------|-----------|-----------------|------------|--------|
| SURFACE (AROMATIC_TOPO+HYDRO_FIELD) late fusion | **STRONG** | ↓MAE vs seq DL | 3 backbones + permutation | V3 H082–H093 | `HIC_H047_FEATURE_FUSION_REPORT.md` |
| CONTINUOUS_SURFACE / HYDRO_TITRATION classical | **STRONG** | competitive MAE | linear+XGB | canonical_simple | `experiments.csv` H002/H021/H022 |
| ESM2_SEQ_AROMATIC | **MODERATE** | useful alone/fusion | multi models | classical + Phase2A | H003/H023/H030 |
| RASA-weighted ESM2 + CDR3 (H047 path) | **MODERATE** | best classical Overall | refinement batch；vs H021 bootstrap CI includes 0 | canonical_simple | `HIC_CLASSICAL_REFINEMENT.md` |
| LOCAL_RASA alone fusion | **NOT SUPPORTED** | flat/worse | H084/088/092 | V3 | H047 fusion report |
| HSP BM-R5 / EIS-R8 (+SURFACE) | **MODERATE→STRONG** | additive ↓MAE | H102–H113；一部 bootstrap CI が0横断 | V3 after atlas | `HIC_HSP_MAINLINE_REPORT.md` |
| HSP FP-R5 | **WEAK / NOT CONFIRMED** | mixed | mainline | V3 | 同上 |
| STATIC_SAP_KD | **NOT SUPPORTED** | no gain vs SURFACE | H094–H101 | V3 | `HIC_STATIC_SAP_KD_REPORT.md` |
| SOURCE24 / SOURCE_SAP24 | **NOT SUPPORTED** | no / worse | H114–H127 | V3 | SOURCE reports |
| Residue F1 compact | **NOT SUPPORTED** vs Ab F1 | worse | H128–H133 | V3 | residue report |
| Generic HYDRO-FIELD / COPATCH / OpenMM dyn. aromatic | **NOT SUPPORTED** as increment | no ΔMAE | prospecting | post-reveal | Physical/Advanced/OpenMM |
| Geometry ARCH-6G | **WEAK** | inconsistent | H064–079 | V3 | `HIC_GEOMETRY_REPORT.md` |

---

## 8. Structure / SASA / RASA / aromatic evidence

### 8.1 What was actually computed

| Artifact | Computed contents | Path |
|----------|-------------------|------|
| AROMATIC-TOPO_v1 | F/W/Y；RASA≥0.20/0.50；patch Cα≤8Å；exposed aromatic SASA 等（His除外は Tm/HIC promising report 記載） | `feature_extension/extractors/extract_aromatic.py`；`aromatic_topology.parquet` |
| continuous_surface / hydro_field / static_sap | FreeSASA・疎水場・静的SAP | `feature_extension/` precomputed |
| Gate B6.4 ESMFN structure feats | `ESMFN_Fv_sasa_aromatic` Spearman **0.426**；Cliffs δ **0.705**（HIGH vs nonHIGH） | `gate_b6_4_hic_high_tail/reports/03_high_hic_structure_mechanism.md` |
| HSP atlas | ~21k spatial hydrophobicity variants；univariate Spearman；Ridge/SVR screen | `feature_research/hic_spatial_hydrophobicity/` |
| Signal-source audit | seq YWF vs AROMATIC vs combo MAE | `organizer_extension/feature_prospecting/SIGNAL_SOURCE_AUDIT/` |

### 8.2 Computed predictive results（aromatic）

| Model | CV / Pub / Priv MAE | Label | Source |
|-------|---------------------|-------|--------|
| seq-only YWF | 0.5406 / 0.5642 / 0.4928 | TEST_ONLY_POSTHOC | SIGNAL_SOURCE_AUDIT |
| AROMATIC-TOPO | 0.5038 / 0.4919 / 0.4661 | REPRODUCIBLE | 同上 + AROMATIC REPORT |
| seq + AROMATIC | 0.4991 / 0.4943 / 0.4597 | REPRODUCIBLE | 同上 |
| combo−seq Δ | **−0.0415 / −0.0699 / −0.0332** | **GEOMETRY_ADDS_SIGNAL** | 同上 |

AROMATIC standalone vs Round1 surface: historical verdict **PROMISING_BUT_REDUNDANT**（NO_INCREMENT）。

### 8.3 Human interpretation（非計算）

- 「広い疎水性一般」より **露出芳香族トポロジー** が効く、という **事後解釈** は postmortem / synthesis 文書に存在。
- 公式監査は **「3D 芳香族幾何が HIC を引き起こす」と断定しない**（SIGNAL_SOURCE_AUDIT 明記）。
- 「芳香族＝疎水だから」という一般論だけで結果を説明した一次解析は、上記分離実験（seq YWF vs geometry）より弱い。

### 8.4 Chain / region

- B6.4: CDRH3 aromatic / hydrophobic / length が HIGH で差；`ESMFN_VH_sasa_hydrophobic` Cliffs δ 0.361 vs VL ≈0（表上 −0.06）。
- Phase2A は H-only Transformer を多用（H024–H033）。**VH支配の確定的 factorial はない。**

---

## 9. Public / Private evidence timeline

### A. Internal-only evidence（外部を選ばない／見ていない段階の設計）

- HIC を連続タスクとして維持する妥当性議論の骨格（分布・shrinkage・CV 設計）— B6.2 の主解析。
- HIGH-tail の data-limited 診断の骨格 — B6.4（holdout は確認用途と明記）。
- EXP-H の多数バッチは `*_PRE_EXTERNAL_FREEZE.yaml` で内部凍結後に external を記録（例: `T105_T123_HIC_PRE_EXTERNAL_FREEZE.yaml`, commit `5b8ac2c6`）。
- SURFACE late fusion / HSP / SAP / residue / global conditioning の **内部 TEST_mean・permutation** は confirmatory に近い（ただし最終 Overall は Pub/Priv を含む）。

### B. Post-Public evidence

- Trust-CV（B7.1）: CV↔Public 不一致時、Private は Public 側に寄る傾向（TrustCV_rate 0.133）→ **PUBLIC_AT_LEAST_AS_RELIABLE**（CAND_12528）。
- Virtual competition live: Public MAE のみ返却；persona が Public 駆動で Private 0.4437。
- Organizer feature prospecting contract: Round1 後 Public/Private **reveal 済み**；以降は ORGANIZER-EXPLORATORY（`ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md`）。

### C. Post-Private diagnostic evidence

- VC Private reveal 後の HIGH-tail 診断（overall 改善≠尾部改善）。
- AROMATIC signal-source audit（`POST_COMPETITION_DIAGNOSTIC`）。
- top_models Pub/Priv **POSTMORTEM ONLY**（選択に未使用と README 系が記載）。
- Physical Batch1–Late Batch3 の増分判定（多くが post-reveal）。

**混同禁止:** confirmatory（事前凍結の内部指標）と post-hoc diagnostic（reveal 後の機序分解・postmortem）を同一の「確立知見」として扱わない。Evidence matrix の `pre_or_post_external` 列を参照。

---

## 10. Comparison with current TmApp pipeline

| 項目 | TmApp（現状） | HIC（現状） |
|------|---------------|-------------|
| Factorial Rep×Topo×Annot | **実施済・freeze**（SEP/JOINT/REG-SEP/XREG/FUSE × BASE/IMGT/REGION/FULL） | **未実施** |
| 主レバー（synthesis） | representation / topology（AbLang2 等） | **明示的 surface / 物理特徴** |
| V3 annotation | factorial で変動 | HIC V3 は **FULL 固定** |
| Architecture-only plateau | AbLang2 クラスタが強い | ESM2/Scratch ARCH は ~0.50 で頭打ち |
| Geometry | Tm でも限定的 | HIC でも clear win なし |
| Late fusion physics | Tm では主線でない | HIC の成功パターン |
| 転用可能なもの | `DL_FOLDLOCAL_COSINE_V3` 基盤、OOF、prereg/freeze 運用、EXP code 管理 | 同上を HIC 用に既使用 |
| HIC 固有に変えるべき点 | — | 尾部・MAE、surface/HSP、Public/Private 汚染段階、HIGH n 制約 |

Sources: `TM_STRUCTURE_VS_HIC_FEATURE_SYNTHESIS.md`；`technical_report/tmapp_factorial/`（HIC 外挿禁止の明記）；`EXPERIMENT_CODES.csv`（H139 まで）。

---

## 11. Established findings（Tier 1）

1. HIC は連続 MAE 回帰として妥当；二値化しない（B6.2 / B6.4）。
2. HIGH 尾部の系統的過小予測と data limitation（B6.2 / B6.4 / B7 VC）。
3. Pearson は脆弱；MAE primary（B6.2 / B7.3）。
4. 本番 split は **GEN_0001_B_20271100**；CAND_12528 スコアと非互換（PROVENANCE / B7.3）。
5. プロトコル横断の数値直比較は不可（本監査 §4.4）。
6. SURFACE late fusion は V3 DL を改善（H082–H093）。
7. LOCAL_RASA 単独 fusion は無効（同）。
8. Architecture-only V3 は ~0.50 で頭打ち（H054–H081）。
9. STATIC_SAP_KD / SOURCE24 / SOURCE_SAP24 / residue F1 は主線として非支持（各レポート）。
10. TmApp factorial は HIC 未実施（absence evidence）。

（詳細・出典は `HIC_EVIDENCE_MATRIX.csv`）

---

## 12. Weak / contradictory findings

### Tier 2 — Supported（制約付き）

- Trust-CV 非実証 / Public 有用（12528 限定）。
- HSP BM/EIS 加法（bootstrap が常に有意とは限らない）。
- GEOMETRY_ADDS_SIGNAL（post-competition；Round1 増分は redundant）。
- B6.4 aromatic SASA 関連（HIGH n=13）。
- HIC=feature problem / Tm=representation problem の非対称仮説（単一批の synthesis）。
- Organizer BEST_CV equal-mean；stacking SKIP。
- Generic hydro / OpenMM 増分なし（post-reveal）。
- Geometry ARCH-6G 非勝利。
- CASE D: late fusion 近最適（H137 external は例外扱い）。

### Tier 3 — Suggestive

- 局所 patch が global SASA を超えるか（計算は多いが決定的勝者不在；residue F1 は失敗）。
- VH/CDRH3 支配（小標本・H-only 歴史）。
- H137 FiLM を「確定アーキ進歩」とみなすこと（レポートは CASE D / H140 禁止）。

### Tier 4 — Unresolved / contradictory

- 「露出芳香族が因果的ドライバー」— **repository は非支持（非主張）**。
- HIC 向け PLM/topology/annotation factorial の優劣 — **未実験**。

---

## 13. Unresolved questions（優先順）

1. **HIGH-tail は表現不足か、損失/分布/標本数（n≈13）の構造的限界か？** — 複数手法で尾部が残るが、介入実験は未決。
2. **TmApp factorial（多PLM×5 topology×4 annotation）を HIC に転用する価値は？** — 未実施；既存 synthesis は「HIC は物理特徴」側。転用は仮説検証であって既定成功ではない。
3. **露出芳香族: 配列カウント vs RASA/SASA 露出 vs local patch — どれが必要十分か？** — GEOMETRY_ADDS_SIGNAL はあるが、Round1 増分 redundancy と residual F1 失敗が並存。
4. **HSP 加法は SURFACE を超えて安定に主線化できるか？** — BM/EIS 支持だが CI・外部選定ルール未固定。
5. **どの評価プロトコルを「今後の正」とするか（GEN_0001 + どの CV）？** — 本番は GEN_0001；drilldown V3 との公式ブリッジ文書が薄い。
6. **VH vs VL / CDR vs framework の因果的寄与** — 示唆のみ。
7. **Stacking / 異種 OOF 統合** — 意図的 SKIP；安全な統合設計は未解決。
8. **測定 noise（~0.12 min 参考）に対する残余誤差の内訳** — 解釈ガイドはあるが、モデル誤差分解は未完。

---

## 14. Recommended next decision points

**この audit を確認するまで新規 HIC 実験は起動しない（依頼どおり）。** 確認後の意思決定ポイントのみ列挙する。

1. **評価の正を宣言する:** GEN_0001 + 採用 CV（V3 OOF vs SGKF 等）を一文で固定する。
2. **目的関数を選ぶ:** overall MAE か HIGH-tail か（両立は歴史的に困難）。
3. **次仮説の型を選ぶ:** (a) surface/HSP 精緻化、(b) HIC 向け factorial、(c) tail-specific loss/weight、(d) データ限界の受け容れ。
4. **芳香族仮説を検証するなら:** 事前登録で「配列組成 / 露出カウント / SASA / patch」を分離し、因果言語を禁止したまま predictive necessity を測る。
5. **H140+ / より複雑な fusion:** 現行レポートは **DO NOT RUN**（CASE D）。覆すなら事前の失敗条件を明記。

---

## 15. Source index

### Gate
- `gate_b6_2_hic_validity/reports/GATE_B6_2_HIC_VALIDITY_FINAL.md`
- `gate_b6_4_hic_high_tail/reports/GATE_B6_4_FINAL.md`
- `gate_b6_split_search/reports/GATE_B6_COMMON_SPLIT_FINAL.md`, `GATE_B6_1_FINAL_SPLIT_BAKEOFF.md`
- `gate_b7_1_trust_cv/reports/GATE_B7_1_TRUST_CV_FINAL.md`
- `gate_b7_2_split_reassessment/reports/GATE_B7_2_SPLIT_REASSESSMENT_FINAL.md`
- `gate_b7_3_principled_split/reports/GATE_B7_3_PRINCIPLED_SPLIT_FINAL.md`, `07_FINAL_GEN0001_3X3_BENCHMARK.md`
- `gate_b7_virtual_competition/reports/GATE_B7_VIRTUAL_COMPETITION_FINAL.md`

### Drilldown
- `developability_drilldown/results/EXPERIMENT_CODES.csv`, `experiments.csv`
- `HIC_*_REPORT.md`（CLASSICAL_REFINEMENT, ESM2_SCRATCH_ARCHITECTURE, GEOMETRY, H047_FEATURE_FUSION, STATIC_SAP_KD, HSP_MAINLINE, SOURCE24*, RESIDUE*, GLOBAL_F1*）
- `TM_STRUCTURE_VS_HIC_FEATURE_SYNTHESIS.md`, `TM_HIC_PROMISING_MODEL_REPORT_FINAL.md`
- `T105_T123_HIC_PRE_EXTERNAL_FREEZE.yaml`（commit `5b8ac2c6`）

### Features / organizer / competition
- `gate_b1/reports/assay_definitions.md`
- `competition/organizer/PROVENANCE.md`, `SCORE_INTERPRETATION_TECHNICAL_JA.md`
- `feature_extension/`, `feature_research/hic_spatial_hydrophobicity/`
- `organizer_extension/feature_prospecting/AROMATIC-TOPO/`, `SIGNAL_SOURCE_AUDIT/`
- `top_models_feature_bundle/results/`

### Git commits（HIC 主要）
`c98cf872`, `6e727c40`, `3136d09e`, `efc1e17f`, `971d6860`, `fc64362c`, `5b8ac2c6`, `3d0e336e`, `076c03f7`, `210a270d`, `9625151f`, `ed6bdbc1`, `58a0e543`

### Companion CSVs
- `reports/HIC_EXPERIMENT_INVENTORY.csv`
- `reports/HIC_EVIDENCE_MATRIX.csv`

---

**Audit status:** COMPLETE — inventory only; no new HIC experiments launched.
