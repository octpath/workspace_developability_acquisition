# <FAMILY_ID>_<version>

**状態:** （実験完了後に更新）

**Evidence class:** ORGANIZER-EXPLORATORY（unseen-test / prospective evidence ではない）

---

## Bottom line

| Target | Mechanistic prior | Empirical verdict | One-line conclusion |
|--------|-------------------|-------------------|---------------------|
| TmApp | （pre-freeze） | NOT_RUN / … | |
| HIC | （pre-freeze） | NOT_RUN / … | |

**One-line interpretation:**  
（例: 「低周波 flexibility は TmApp には有望、HIC との直接関係は乏しい」）

---

## 1. Scientific hypothesis

（この feature family が TmApp / HIC と関連しうる物理的根拠を 2–4 文）

### Why this may matter for TmApp

（mechanistic prior の根拠）

### Why this may matter for HIC

（mechanistic prior の根拠）

---

## 2. Physical quantity

（定義、単位、文献・software）

---

## 3. Feature specification

- **Version:**
- **Specification hash:**
- **Structure crosswalk version / hash:**
- **Structure sources:** ESMFold / ABodyBuilder2（paths from STRUCTURE_INPUT_CROSSWALK.csv）
- **N features:**
- **Preprocessing:** fold-internal only
- **Canonical model:** Ridge, exact nested Primary folds, alpha `[0.1, 1.0, 10.0, 100.0]`, tie → larger alpha
- **Baseline:** TRAIN_MEDIAN_BASELINE
- **PRIMARY incremental:** CANONICAL_RESIDUAL_RIDGE
- **SECONDARY_FUSION:** （あれば）
- **Post-hoc after test view:** false / true
- **Parent version:**

---

## 4. Round1 overlap

| Aspect | Round1 status | This version |
|--------|---------------|--------------|
| | | |

---

## 5. Structure-generator robustness

**評価タイミング:** target ラベル使用前

| Summary | Value |
|---------|-------|
| Median Spearman | |
| IQR | |
| Fraction ρ > 0.8 | |
| Fraction ρ > 0.5 | |
| **Classification** | ROBUST / MODERATE / FRAGILE |

Low-consistency features (Spearman < 0.5):

-

**Generator-specific signal:** BOTH / ESMFOLD_ONLY / ABB2_ONLY / MIXED / NONE

（低一致を棄却理由としていない旨を明記）

---

## 6. Standalone prediction

Canonical Ridge feature-only（exact nested）。Naive baseline = TRAIN_MEDIAN_BASELINE。

| Split | MAE | Baseline MAE | Pearson | Spearman | pred SD | true SD | pred SD / true SD |
|-------|-----|--------------|---------|----------|---------|---------|-------------------|
| Primary CV OOF | | | | | | | |
| Public | | | | | | | |
| Private | | | | | | | |
| All Test | | | | | | | |

**Standalone classification:** REPRODUCIBLE / WEAK / CV_ONLY / TEST_ONLY_POSTHOC / NO_SIGNAL

MODEL_DEPENDENT_SIGNAL（secondary SVR）: あり / なし

---

## 7. Incremental prediction（CANONICAL_RESIDUAL_RIDGE）

| Split | Reference MAE | Candidate MAE | ΔMAE | Bootstrap P(improve) | 95% CI |
|-------|---------------|---------------|------|----------------------|--------|
| Primary CV | | | | | |
| Public | | | | | |
| Private | | | | | |
| All Test | | | | | |

Reference model ID:  
Selected residual alpha:

**Incremental classification:** REPRODUCIBLE_INCREMENT / WEAK_OR_MIXED_INCREMENT / DIRECTION_REVERSAL / CV_ONLY_INCREMENT / TEST_ONLY_POSTHOC_INCREMENT / NO_INCREMENT / HARMFUL

（3 split 全て改善でも PRACTICALLY_LARGE 等を勝手に付与しない。ΔMAE / CI / P(improve) を併記。）

SECONDARY_FUSION（あれば別表）:

---

## 8. Residual association

| Summary feature | Split | Pearson(r, feat) | Spearman(r, feat) |
|-----------------|-------|------------------|-------------------|
| | CV OOF | | |
| | Public | | |
| | Private | | |

**Residual association:** PRESENT / WEAK / ABSENT

---

## 9. Target-specific diagnostics

### TmApp（該当時）

- Quintile MAE / bias
- Low-Tm / High-Tm bias
- pred SD / true SD

### HIC（該当時）

- High-tail threshold: **≥ 10.5372 min**
- High-tail MAE / bias
- ROC-AUC / AP / Recall@13
- pred SD / true SD

---

## 10. Artifact / confound checks

| Covariate | Association with feature / residual | Notes |
|-----------|--------------------------------------|-------|
| Sequence length | | |
| Mean pLDDT | | |
| Missing residues | | |
| Clash count | | |
| Total SASA | | |
| Structure source | | |

---

## 11. Split reproducibility

| Axis | CV | Public | Private | Class |
|------|-----|--------|---------|-------|
| Standalone vs TRAIN_MEDIAN_BASELINE | | | | |
| Incremental ΔMAE sign（CANONICAL_RESIDUAL_RIDGE） | | | | |

**Split reproducibility:** …

---

## 12. Final classification

| Dimension | Result |
|-----------|--------|
| Mechanistic prior (TmApp / HIC) | |
| Empirical verdict (TmApp / HIC) | |
| Physical association | |
| Standalone signal | |
| Incremental signal | |
| Structure robustness | |
| Generator signal | |
| Split reproducibility | |

**Final statement:**

**Evidence limitations:**

- ORGANIZER-EXPLORATORY
- Public / Private labels revealed post-Round1
- Not competition-valid prospective evidence
- （family 固有）

---

## Appendix

- `FEATURE_SPEC.json`
- `FEATURE_MANIFEST.csv`
- `structure_robustness.csv`
- `metrics.json`
- `bootstrap.csv`
- `predictions/`
