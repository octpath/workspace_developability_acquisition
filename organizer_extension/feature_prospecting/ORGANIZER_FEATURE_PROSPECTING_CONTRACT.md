# Organizer Feature Prospecting — Experimental Contract (Gate 1)

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`

**Gate:** Organizer Extension Gate 1 → 1.1 → **1.2** (structure sources + final contract corrections)

**作成日:** 2026-09-02（Gate1） / Gate1.1 / **Gate1.2 patch:** 2026-09-02

本書は Round1 後の Organizer exploratory extension において、全 PDB-derived feature family 評価に共通する実験規約を凍結する。Gate1–1.2 では ANM/pKa/SAP 等の feature extraction・TmApp/HIC model training・Public/Private score 探索は行わない。

Gate1.2 追加: CANONICAL_RESIDUAL_RIDGE outer leakage 修正、mechanistic relevance **5_… numeric display**、**Boltz-2** 第三 structure generator、crosswalk v2、3-generator robustness。詳細は [GATE1_2_STRUCTURE_SOURCE_EXTENSION_REPORT_JA.md](GATE1_2_STRUCTURE_SOURCE_EXTENSION_REPORT_JA.md)。

---

## 0. Workspace / Evidence boundary

### 0.1 新規ワークスペース

```
organizer_extension/feature_prospecting/
```

Organizer Extension の新規 artifact はこの directory 配下にのみ作成する。Round1 participant artifact とは明確に分離する。

### 0.2 Evidence 区分

| 区分 | 定義 |
|------|------|
| **ORGANIZER-EXPLORATORY** | 本拡張の全実験・報告 |
| **Round1 frozen** | `virtual_participant/` 配下 — **READ-ONLY、変更禁止** |
| **Participant Round2** | 別系統（本契約の対象外） |

Public / Private ラベルは Round1 終了後に reveal 済み。CV / Public / Private の effect direction 一貫性は **post-hoc scientific replication** として利用してよい。

**禁止用語（本拡張の結論に対して）:**
- unseen-test evidence
- competition-valid prospective evidence
- blind test validation

### 0.3 Repository audit — frozen upstream（source of truth）

数値・split・model ID は repository 内 frozen artifact を正とする。手入力で再構築しない。

| 項目 | Repository artifact | 確認値 |
|------|---------------------|--------|
| 母集団 | `gate_b3/frozen/organizer/final_population.csv` | N=324 |
| Dev / Test | 同上 + `competition/data/dev.csv` / `solution.csv` | 162 / 162 |
| Public / Private | `competition/organizer/SPLIT_MANIFEST.json` | 81 / 81 |
| Primary CV | `virtual_participant/stage0_cv/cv_primary.csv` | `opt_joint_group_k5_s42`, Dev N=162 |
| Shadow CV | `virtual_participant/stage0_cv/cv_shadow.csv` | `opt_joint_group_k5_s2026`（補助監査のみ） |
| TmApp reference | `virtual_participant/round1_postmortem/round1_model_rank_comparison.csv` | `TmApp__META_performance__ridge_100.0` |
| HIC reference | 同上 | `HIC__SIMPLE_blend_seq_surf_adv` |
| Gate0 final | `virtual_participant/round2_gate0/ROUND2_GATE0_DIAGNOSTIC_REPORT_JA.md` | `ROUND2_GATE0_DIAGNOSTICS_FINAL_AUDIT_PASS_READY_FOR_DEEP_RESEARCH` |
| PDB inventory | `virtual_participant/round1_pdb_feature_inventory/ROUND1_PDB_FEATURE_INVENTORY_JA.md` | `ROUND1_PDB_FEATURE_INVENTORY_AUDIT_COMPLETE` |

### 0.4 Repository audit — discrepancies（明記）

Gate1 監査で prompt / inventory 記述と実 artifact の差異を確認した。勝手に修正せず、以下を契約上の既知事項とする。

**D1. ESMFold PDB path の二重系統**

- Inventory: `esmfold_native/{id}.pdb`（author segment lengths）
- 実 repository:
  - `esmfold_native/ADI-*.pdb` — 370 files（antibody ID 命名）
  - `gate_b2/cache/structures/esmfold_native/{pair_hash}.pdb` — 371 files（hash 命名）
  - `esmfold_native/esmfold_native.csv` は後者の cache path を参照
- **契約:** feature 実装時に `FEATURE_SPEC.json` で canonical input path を version 単位で凍結。dual-path 存在は既知。

**D2. ABodyBuilder2 cache の superset**

- Inventory: Stage3 ablation で Dev N=162 成功
- 実 repository: `gate_b1/cache/structures/abodybuilder2/` に 400 PDB files
- **契約:** structure robustness / feature 抽出は competition 抗体 ID（Dev+Test 324）に限定。cache 全体は superset。

---

## 1. Core scientific questions

各 feature family について、以下を**別々に**判定する。

| ID | 問い | 判定ラベル |
|----|------|------------|
| Q1 | feature 自体に target information があるか | **STANDALONE SIGNAL** |
| Q2 | incumbent が持たない target information があるか | **COMPLEMENTARY / INCREMENTAL SIGNAL** |
| Q3 | ESMFold と ABodyBuilder2 で同 physical quantity として再現するか | **STRUCTURE-GENERATOR ROBUSTNESS** |
| Q4 | CV / Public / Private で effect direction が再現するか | **SPLIT REPRODUCIBILITY** |

詳細分類は [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md) および [STRUCTURE_ROBUSTNESS_SPEC.md](STRUCTURE_ROBUSTNESS_SPEC.md) を参照。

### 1.1 科学的原則

主目的は「incumbent を 0.01 改善できる feature だけ残す」ことではない。

**PDB 由来 physical quantity が TmApp / HIC について reproducible target information を持つか**を調べる。

| 結果パターン | 科学的解釈 |
|--------------|------------|
| Standalone あり、Incremental なし | `PHYSICAL_SIGNAL_PRESENT` / `BUT_REDUNDANT_WITH_INCUMBENT` — **positive result** |
| Standalone 弱い、Incremental あり | `COMPLEMENTARY_SIGNAL_PRESENT` |
| 両方 strong | 最強候補 |

---

## 2. Targets and metrics

| Target | Metric | Unit |
|--------|--------|------|
| TmApp | MAE | °C |
| HIC | MAE | min |

### 2.1 Frozen reference performance（Round1 Stage5 PRIMARY）

出典: `virtual_participant/round1_postmortem/round1_model_rank_comparison.csv`

| Target | Model ID | Primary CV | Public | Private | All Test |
|--------|----------|------------|--------|---------|----------|
| TmApp | `TmApp__META_performance__ridge_100.0` | 2.7135 | 3.2307 | 3.2116 | 3.2211 |
| HIC | `HIC__SIMPLE_blend_seq_surf_adv` | 0.4252 | 0.4204 | 0.4239 | 0.4222 |

Reference predictions:
- TmApp: `virtual_participant/round1_finalization/predictions/TmApp_PRIMARY_predictions.csv`
- HIC: `virtual_participant/round1_finalization/predictions/HIC_PRIMARY_predictions.csv`

### 2.2 三 split スコアセット（必須保存）

全 experiment について必ず保存:

- **Primary CV**（OOF MAE）
- **Public**（N=81）
- **Private**（N=81）
- **All Test**（N=162）

Public / Private は必ず別列でも保存する。主比較は **CV / Public / Private の三者**。

---

## 3. Frozen CV policy

| 項目 | 値 |
|------|-----|
| Primary CV | `opt_joint_group_k5_s42` |
| Fold assignment | `virtual_participant/stage0_cv/cv_primary.csv` — **変更禁止** |
| Shadow CV | `opt_joint_group_k5_s2026` — supplementary audit のみ、主要 selection criterion にしない |

理由: Gate0 matched analysis で Primary が Private との対応において Shadow より良好だった（`ROUND2_GATE0_DIAGNOSTIC_REPORT_JA.md` §D）。

---

## 4. Reference model と評価モデルの二層

### 4.1 Frozen incumbent（fusion の baseline）

| Target | Incumbent |
|--------|-----------|
| TmApp | `TmApp__META_performance__ridge_100.0` |
| HIC | `HIC__SIMPLE_blend_seq_surf_adv` |

Incumbent の OOF / Test predictions は Round1 frozen artifact から読み込む。再学習しない。

### 4.2 A. Canonical feature-screening model

**全 family 共通。** feature family 自身の情報量を見る。

| 項目 | 凍結値 |
|------|--------|
| Primary regressor | **Ridge regression** |
| Alpha grid | `[0.1, 1.0, 10.0, 100.0]` |
| Alpha 選択 | **Exact nested procedure**（下記 + SIGNAL_CLASSIFICATION_SPEC §3） |
| Naive MAE baseline | **`TRAIN_MEDIAN_BASELINE`**（mean ではない） |
| 目的 | feature quality と model optimization の分離 |

#### Exact nested Ridge（凍結）

- Outer folds: `opt_joint_group_k5_s42`
- Outer held-out = fold `f`; outer training = remaining 4 Primary folds
- **Inner CV:** remaining 4 Primary fold IDs をそのまま使用（新しい random KFold 禁止）
- Select alpha: smallest mean inner MAE; **tie → larger alpha**
- Outer training 全体で選択 alpha を refit → held-out 予測
- Test: full Dev 上で Primary 5 folds により alpha 選択 → full Dev refit → Test 予測
- imputation / standardization / PCA は各 training 分割内のみ fit

**Secondary sensitivity（任意）:** linear SVR, RBF-SVR → `MODEL_DEPENDENT_SIGNAL` のみ。

### 4.3 B. PRIMARY incremental: CANONICAL_RESIDUAL_RIDGE

全 family 共通の **PRIMARY** incremental evaluation。

```
p_candidate = p_ref_baseline + Ridge(X → residual)
```

**Gate1.2 leakage-safe outer CV（必須）:**

- `outer_test` baseline: frozen Round1 reference **OOF**（fold f 除外で学習済みのため可）
- `outer_train` residual targets: **outer_train のみ**で Round1 reference recipe を **cross-fit し直す**  
  （global frozen OOF residual を outer_train に使うのは PRIMARY 禁止 → meta contamination）
- outer_test labels/samples を reference cross-fitting に使わない
- Test: Dev162 frozen OOF residual で fit → Test = Round1 Test pred + residual

詳細: SIGNAL_CLASSIFICATION_SPEC §5。

### 4.4 C. SECONDARY_FUSION（任意）

concat / stacking / SVR / family-specific fusion は `SECONDARY_FUSION` として許可。  
**canonical incremental classification は必ず CANONICAL_RESIDUAL_RIDGE を主結果とする。**

---

## 5. Preprocessing contract

**全 transform は CV fold 内部で fit。**

対象: imputation, standardization, PCA, feature selection, dimensionality reduction

| Feature 種別 | PCA |
|--------------|-----|
| 低次元 handcrafted | 原則 **なし** |
| high-dimensional frozen embedding | fold-internal PCA **許可** |

Public / Private / All Test 全体を使って scaler / PCA 等を fit してはならない。

Test prediction 時は Dev で学習した最終モデルを full Dev refit して適用する（Round1 と同様の refit policy を family ごとに `FEATURE_SPEC.json` で明示）。

---

## 6. Feature family freeze / versioning

### 6.1 Version 命名

```
<FAMILY_ID>_v<N>
```

例: `PKA-SHIFT_v1`, `STATIC-SAP_v1`

### 6.2 凍結対象（target を見る前）

- feature definitions, regions, thresholds, radii
- pH grid, force field, structure preprocessing
- aggregation rules, missing-value rules
- model family, small HP grid

### 6.3 必須 manifest

各 family directory:

```
organizer_extension/feature_prospecting/<family_id>/
    FEATURE_SPEC.json      # 必須
    FEATURE_MANIFEST.csv   # 必須
```

`FEATURE_SPEC.json` 最低限フィールド:

```json
{
  "family_id": "PKA-SHIFT",
  "version": "v1",
  "specification_hash": "<sha256 of frozen spec>",
  "target": "TmApp",
  "structure_sources": ["esmfold", "abodybuilder2"],
  "structure_crosswalk_version": "v1",
  "crosswalk_hash": "<sha256 of STRUCTURE_INPUT_CROSSWALK.csv>",
  "feature_definitions": {},
  "preprocessing": {},
  "canonical_model": {"type": "Ridge", "alpha_grid": [0.1, 1.0, 10.0, 100.0], "nested": "exact_primary_folds"},
  "canonical_incremental_model": "CANONICAL_RESIDUAL_RIDGE",
  "baseline_type": "TRAIN_MEDIAN_BASELINE",
  "secondary_fusion": null,
  "posthoc_after_test_view": false,
  "parent_version": null
}
```

### 6.4 Post-hoc versioning policy

Public / Private を見た**後**に radius / threshold / pH / feature 追加削除 / HP / preprocessing を変更した場合、同一 version の改善とは扱わない。

必ず新 version:

```
STATIC-SAP_v1  →  (Public/Private reveal 後に radius 変更)  →  STATIC-SAP_v2_POSTHOC
```

`posthoc_after_test_view=true`, `parent_version` を必ず記録。

---

## 7. Standalone signal

Feature-only canonical Ridge（exact nested §4.2）について。

### 7.1 必須 metrics（split 別）

- MAE（vs **`TRAIN_MEDIAN_BASELINE`**）
- Pearson(pred, true), Spearman(pred, true)
- prediction SD, prediction range, pred SD / true SD
- `baseline_MAE_*`（CV / Public / Private / AllTest）

### 7.2 Univariate association

pre-specified summary のみ。many-feature fishing 禁止。

---

## 8. Incremental signal（PRIMARY = CANONICAL_RESIDUAL_RIDGE）

```
delta_MAE_s = MAE(candidate)_s - MAE(reference)_s
```

delta < 0 = improvement。必須: `delta_CV`, `delta_Public`, `delta_Private`, `delta_AllTest`。

分類規則（固定 MAE threshold なし）: SIGNAL_CLASSIFICATION_SPEC §6  
（`REPRODUCIBLE_INCREMENT` / `WEAK_OR_MIXED_INCREMENT` = exactly 2/3 + CI contains 0 / `DIRECTION_REVERSAL` 等）

---

## 9. Paired bootstrap policy

```
d_i = |y_i - pred_candidate_i| - |y_i - pred_reference_i|
```

Default B=10,000。保存: mean/median delta, 95% CI, P(delta < 0)。

Formal hypothesis test ではない。WEAK/MIXED 判定に CI が 0 を含むかを使用（§ SIGNAL §6）。

---

## 10. Residual association

Reference OOF / Test residual:

```
r_i = y_i - pred_reference_i
```

candidate feature（または feature-only prediction）との Pearson / Spearman を保存。

pre-specified family-level summary のみ。univariate fishing 禁止。

---

## 11. Dual-structure / three-generator robustness

ESMFold・ABodyBuilder2・**Boltz-2**（`BOLTZ2_FV_STANDARD_v1`）を technical replicates として利用可能。

**Canonical paths:** [STRUCTURE_INPUT_CROSSWALK_v2.csv](STRUCTURE_INPUT_CROSSWALK_v2.csv)（Gate1.2）。v1 は ESMFold+ABB2 のみの legacy。各 family は独自 path discovery 禁止。

- Pairwise: ESMFold–ABB2 / ESMFold–Boltz2 / ABB2–Boltz2
- Family primary summary: **minimum pairwise median Spearman**
- ROBUST: all 3 pairs ≥ 0.8; MODERATE: all ≥ 0.5 かつ 1つ < 0.8; FRAGILE: いずれか < 0.5
- Generator ensemble / median feature は Gate1.2 では作らない

詳細: [STRUCTURE_ROBUSTNESS_SPEC.md](STRUCTURE_ROBUSTNESS_SPEC.md)

Per semantic feature: Pearson, Spearman, rank consistency, sign agreement, top/bottom quantile overlap

Family-level: median Spearman, IQR, fraction ρ>0.8, fraction ρ>0.5, low-consistency feature list

**低一致から自動 discard しない。** 例: OpenMM strain の低一致自体が scientific finding になり得る。

### 11.1 Generator-specific signal

ESMFold / ABB2 それぞれで target association を評価し分類:

`BOTH_STRUCTURES_SIGNAL` | `ESMFOLD_ONLY_SIGNAL` | `ABB2_ONLY_SIGNAL` | `STRUCTURE_DEPENDENT_MIXED` | `NO_SIGNAL`

---

## 12. Confound control

Family に応じ artifact covariates を記録:

- sequence length / residue count
- mean structure confidence (pLDDT 等)
- missing atoms / residues
- clash count
- initial severe-contact count
- total SASA
- structure source

**OpenMM strain では必須:** initial clash proxy, atom count, residue count, mean confidence

strain–target association が clash count だけで説明されないか診断する。

---

## 13. Target-specific diagnostics

### 13.1 TmApp

Gate0: `TARGET_RANGE_COMPRESSION`, low-Tm overprediction / high-Tm underprediction

必須:
- target quintile performance
- low-Tm bias, high-Tm bias
- prediction SD / true SD

### 13.2 HIC

Gate0: `RANKABLE_BUT_COMPRESSED`

Frozen high-tail threshold: **HIC ≥ 10.5372 min**（Dev N=17, Test N=13）

必須:
- overall MAE
- high-tail MAE
- high-tail bias
- ROC-AUC（frozen threshold）
- Average Precision
- Recall@13
- prediction SD / true SD

---

## 14. Signal classification / relevance labels（概要）

詳細: [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md)

- Standalone vs Incremental 二軸判定（Incremental PRIMARY = CANONICAL_RESIDUAL_RIDGE）
- Split classes: REPRODUCIBLE / WEAK_OR_MIXED / DIRECTION_REVERSAL / CV_ONLY / TEST_ONLY_POSTHOC / NO_REPRODUCIBLE
- Pre-experiment: `mechanistic_relevance_*`（LIKELY_RELEVANT 等）
- Post-experiment: `empirical_verdict`（PROMISING / PROMISING_BUT_REDUNDANT / COMPLEMENTARY / MIXED / NO_EVIDENCE_IN_CURRENT_DATA / UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP）
- Master table: [FEATURE_RELEVANCE_SUMMARY.csv](FEATURE_RELEVANCE_SUMMARY.csv)

---

## 15. Multiple testing / researcher DoF

探索研究のため大量 candidate 試行は禁止しない。ただし:

- family 数、version 数、parameter variants、model variants を **全て registry に記録**
- best のみ報告しない
- attempted / successful / failed variants を各 family で残す

Registry: [FEATURE_FAMILY_REGISTRY.csv](FEATURE_FAMILY_REGISTRY.csv), [FEATURE_PROSPECTING_SCORE_REGISTRY.csv](FEATURE_PROSPECTING_SCORE_REGISTRY.csv)

---

## 16. Per-family output contract

```
organizer_extension/feature_prospecting/<family_id>/
    FEATURE_SPEC.json
    FEATURE_MANIFEST.csv
    features_esmfold.parquet          # if applicable
    features_abodybuilder2.parquet    # if applicable
    structure_robustness.csv
    predictions/
        canonical_primary_oof.csv
        canonical_public.csv
        canonical_private.csv
        fusion_primary_oof.csv
        fusion_public.csv
        fusion_private.csv
    metrics.json
    bootstrap.csv
    REPORT_JA.md
```

`predictions/*.csv` 最低列: `id`, `y_true`, `y_pred`, `split`, `fold`（OOF のみ）

`metrics.json` 構造は Gate2 実装時に schema を追加定義。Gate1 では family 未実装のため schema のみ契約上固定。

---

## 17. Workflow constraints

| 禁止（Gate1 / 各 family pre-specify 後） | 理由 |
|------------------------------------------|------|
| 大規模 Optuna-first | feature quality と search luck の分離 |
| Public/Private-driven tuning | post-hoc bias |
| Round1 artifact の変更 | immutability |
| Primary CV fold の変更 | comparability |

必要なら後続 `FEATURE_MODEL_OPTIMIZATION` phase を別管理。

---

## 18. First-wave backlog（未実行）

DeepResearch based priority。Gate1 では登録のみ。

**First wave — TmApp:** ANM-SPECTRUM, VHL-ANGLE, PKA-SHIFT, 3DI-FROZEN

**First wave — HIC:** AROMATIC-TOPO, STATIC-SAP, 3DI-FROZEN

**Second wave — TmApp:** VOID-EXPLICIT, POLAR-SAT, OPENMM-STRAIN, ENCOM-CHEM, INTERFACE-ENERGY

**Second wave — HIC:** HYDRO-FIELD, ELEC-HYDRO-COPATCH, GEARNET-FROZEN

**Late:** SURFACE-DL, SHORT-ENSEMBLE, MLFF

登録: [FEATURE_FAMILY_REGISTRY.csv](FEATURE_FAMILY_REGISTRY.csv)

---

## 19. Gate1.1 validation checklist

| # | 項目 | 状態 |
|---|------|------|
| 1 | TRAIN_MEDIAN_BASELINE に統一 | PASS |
| 2 | Exact nested Ridge documented | PASS |
| 3 | CANONICAL_RESIDUAL_RIDGE documented | PASS |
| 4 | WEAK/MIXED classification deterministic（bootstrap CI） | PASS |
| 5 | STRUCTURE_INPUT_CROSSWALK.csv created | PASS |
| 6 | Structure sequence mapping audited | PASS |
| 7 | Gate1.1 bundle SHA-256 recorded | PASS |
| 8 | repository HEAD / git status recorded | PASS |
| 9 | Mechanistic relevance for all backlog families | PASS |
| 10 | FEATURE_RELEVANCE_SUMMARY.csv created | PASS |
| 11 | Empirical verdict fields = NOT_RUN | PASS |
| 12 | 新規 feature extraction なし | PASS |
| 13 | 新規 model training なし | PASS |
| 14 | Public/Private score exploration なし | PASS |
| 15 | Round1 unchanged | PASS |

---

## 20. Related documents

- [GATE1_1_CORRECTION_REPORT_JA.md](GATE1_1_CORRECTION_REPORT_JA.md)
- [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md)
- [STRUCTURE_ROBUSTNESS_SPEC.md](STRUCTURE_ROBUSTNESS_SPEC.md)
- [FAMILY_REPORT_TEMPLATE.md](FAMILY_REPORT_TEMPLATE.md)
- [STRUCTURE_INPUT_CROSSWALK.csv](STRUCTURE_INPUT_CROSSWALK.csv)
- [FEATURE_RELEVANCE_SUMMARY.csv](FEATURE_RELEVANCE_SUMMARY.csv)
- [GATE1_FREEZE_MANIFEST.json](GATE1_FREEZE_MANIFEST.json)
- Round1 PDB inventory: `virtual_participant/round1_pdb_feature_inventory/ROUND1_PDB_FEATURE_INVENTORY_JA.md`
- Round2 Gate0: `virtual_participant/round2_gate0/ROUND2_GATE0_DIAGNOSTIC_REPORT_JA.md`

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_2_STRUCTURE_SOURCES_FROZEN`
