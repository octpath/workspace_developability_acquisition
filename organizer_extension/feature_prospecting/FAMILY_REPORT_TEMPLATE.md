# <FAMILY_ID>_<version>

**状態:** （実験完了後に更新）

**Evidence class:** ORGANIZER-EXPLORATORY（unseen-test / prospective evidence ではない）

---

## Bottom line

| Target | Mechanistic prior (display) | Empirical verdict | One-line conclusion |
|--------|-----------------------------|-------------------|---------------------|
| TmApp | e.g. `5_LIKELY_RELEVANT` | NOT_RUN / … | |
| HIC | e.g. `2_UNLIKELY_PRIMARY` | NOT_RUN / … | |

**One-line interpretation:**  
（例: 「低周波 flexibility は TmApp には有望、HIC との直接関係は乏しい」）

Mechanistic scale (higher = more relevant):  
`5_LIKELY_RELEVANT` > `4_PLAUSIBLY_RELEVANT` > `3_RELATED_BUT_INDIRECT` > `2_UNLIKELY_PRIMARY` > `1_NO_CLEAR_MECHANISTIC_LINK`

---

## 1. Scientific hypothesis

### Why this may matter for TmApp

### Why this may matter for HIC

---

## 2. Physical quantity

---

## 3. Feature specification

- **Structure crosswalk:** v2（ESMFold + ABodyBuilder2 + Boltz-2）
- **Canonical model:** Ridge exact nested Primary folds; baseline `TRAIN_MEDIAN_BASELINE`
- **PRIMARY incremental:** `CANONICAL_RESIDUAL_RIDGE`（Gate1.2 leakage-safe outer residual cross-fit）
- **SECONDARY_FUSION:** （あれば）

---

## 4. Round1 overlap

---

## 5. Structure-generator robustness（3 generators）

Pairwise median Spearman: ESMFold–ABB2 / ESMFold–Boltz2 / ABB2–Boltz2  
Primary class from **minimum pairwise median Spearman**.

| Pair | Median Spearman |
|------|-----------------|
| ESMFold vs ABB2 | |
| ESMFold vs Boltz2 | |
| ABB2 vs Boltz2 | |

**Classification:** ROBUST / MODERATE / FRAGILE  
**Generator signal:** NOT_RUN（Gate1.2）

---

## 6. Standalone prediction

| Split | MAE | Baseline MAE | Pearson | Spearman | pred SD / true SD |
|-------|-----|--------------|---------|----------|-------------------|
| Primary CV OOF | | | | | |
| Public | | | | | |
| Private | | | | | |
| All Test | | | | | |

---

## 7. Incremental prediction（CANONICAL_RESIDUAL_RIDGE）

| Split | Reference | Candidate | ΔMAE | P(improve) | 95% CI |
|-------|-----------|-----------|------|------------|--------|
| Primary CV | | | | | |
| Public | | | | | |
| Private | | | | | |

---

## 8. Residual association

---

## 9. Target-specific diagnostics

HIC high-tail threshold: **≥ 10.5372 min**

---

## 10. Artifact / confound checks

---

## 11. Split reproducibility

---

## 12. Final classification

**Evidence limitations:** ORGANIZER-EXPLORATORY; Public/Private revealed; not prospective evidence.
