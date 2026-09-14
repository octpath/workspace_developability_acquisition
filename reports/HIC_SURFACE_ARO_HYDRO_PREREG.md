# HIC SURFACE ARO19 vs HYDRO16 Block Decomposition — Preregistration

**STATUS: `HIC_SURFACE_ARO_HYDRO_PREREGISTERED`**

This document freezes the ARO19 / HYDRO16 block ablation design **before** any new
ARO_ONLY / HYDRO_ONLY training.

Do **not** edit:

- `reports/HIC_SCIENTIFIC_FREEZE_V3.md`
- `reports/HIC_SCIENTIFIC_FREEZE_V2.md`
- `reports/HIC_SURFACE_PROSPECTIVE_FINAL.md`
- prior factorial / internal freezes

---

## 0. Provenance

| Field | Value |
|-------|--------|
| Freeze v3 SHA | `1bfdf9107f55d14c58117aaa7f64e255dc387ed9` |
| SURFACE prospective prereg | `993b8a9e268f65ed8d203949c34cbe3ef87446e6` |
| Reused SHAM/FULL arms | EXP-H340–H343 (no retrain) |
| Prereg introducing commit |  |

**Holdout note:** Project-wide Test is **not** a pristine unused holdout for all HIC history.
New ARO_ONLY / HYDRO_ONLY predictions are unevaluated at this prereg and are treated as
`prospective_relative_to_ARO_HYDRO_prereg`.

---

## 1. Scientific question

Canonical F1_SURFACE = ARO19 ‖ HYDRO16 (35D).

> Which block carries the predictive gain of F1_SURFACE over SHAM35 —
> aromatic exposure/topology (ARO19), hydrophobic surface-field (HYDRO16),
> both complementary, or largely redundant?

This is a **fixed F1 block ablation**, not a new model / topology / representation search.

---

## 2. Fixed contexts (identical to H340–H343)

| Context | Backbone parent | Topology | Annotation |
|---------|-----------------|----------|------------|
| AbLang1 | EXP-H187 lineage / H340–H341 protocol | JOINT | FULL |
| AbLingua | EXP-H167 lineage / H342–H343 protocol | JOINT | FULL |

No representation / topology / annotation re-search. Same platform, seed 101, splits,
optimizer/LR grid/early-stop/loss as H340–H343.

---

## 3. Four surface conditions (aux_dim always 35)

Canonical slot order: `[ARO19 | HYDRO16]`.

| Condition | Content | Codes |
|-----------|---------|-------|
| SHAM35 | all zeros | **reuse** H340, H342 |
| ARO_ONLY35 | `[ARO19 \| ZERO16]` | **new** (AbLang1, AbLingua) |
| HYDRO_ONLY35 | `[ZERO19 \| HYDRO16]` | **new** (AbLang1, AbLingua) |
| FULL35 | `[ARO19 \| HYDRO16]` | **reuse** H341, H343 (no retrain) |

All four conditions share:

- aux_dim = 35
- same AuxMLP / late_concat_aux32 / parameter count
- same fusion, optimizer protocol, LR grid, early stopping, loss, splits, seed
- two-block preprocessing pathway (ARO + HYDRO) with fold-local impute + StandardScaler

Zero-padding blocks must pass a StandardScaler smoke test before training.
If unsafe → STOP and amend prereg before training (do not silently change design).

---

## 4. New training count

**Exactly 4 new trainings:**

1. AbLang1 + ARO_ONLY35
2. AbLang1 + HYDRO_ONLY35
3. AbLingua + ARO_ONLY35
4. AbLingua + HYDRO_ONLY35

---

## 5. Primary contrasts (Δ = A − B; negative = improvement of A)

Per representation:

| Contrast | Meaning |
|----------|---------|
| ARO_ONLY − SHAM | ARO total block effect |
| HYDRO_ONLY − SHAM | HYDRO total block effect |
| FULL − SHAM | Full F1 effect (known from H340–H343) |
| FULL − HYDRO_ONLY | Incremental ARO given HYDRO |
| FULL − ARO_ONLY | Incremental HYDRO given ARO |

Do **not** explain performance by feature-count 19 vs 16 alone.

---

## 6. Analysis plan

### Internal

- Primary / Shadow / CV mean for all contrasts
- Antibody-level paired OOF bootstrap: N_BOOT=10000, seed=101, AE differences
- No adding experiments after seeing results

### External

- After all four new predictions are fixed, score Public / Private / Test for all four new arms
- Also report reused H340–H343 scores
- No mid-stream design change; score all arms regardless of internal outcomes

### HIGH-tail (diagnostic only)

- Threshold **HIC > 11.5** only
- Compare ARO_ONLY−SHAM, HYDRO_ONLY−SHAM, FULL−SHAM on MAE_high / signed / non-HIGH

### Sample-level heterogeneity (diagnostic)

- Fraction improved/worsened vs SHAM for each condition
- HIGH vs non-HIGH
- ARO-only / HYDRO-only / both responders; cross-representation agreement

---

## 7. Mechanistic classification (to be assigned after results)

One of:

- `ARO_DOMINANT`
- `HYDRO_DOMINANT`
- `ARO_HYDRO_COMPLEMENTARY`
- `SURFACE_BLOCK_REDUNDANT`
- `MIXED_OR_INCONCLUSIVE`

Judgment uses Primary/Shadow/Public/Private **direction and effect size** jointly —
no post-hoc numeric thresholds invented after seeing numbers.

---

## 8. Exposed-aromatic hypothesis

Hypothesis: exposed aromatic residues may be particularly informative for HIC.

- If ARO19 is reproducibly dominant → may propose a **later** ARO19-internal family
  decomposition (not in this phase)
- If HYDRO dominant or complementary → do **not** collapse to an aromatics-only story
- ARO19 win alone does **not** license “specific exposed aromatic feature X is causal”

---

## 9. Stop condition

Stop after ARO19 vs HYDRO16 **block** conclusion. No 35-way feature search in this phase.

---

## 10. Outputs (planned)

- `reports/HIC_SURFACE_ARO_HYDRO_DECOMPOSITION.md`
- `reports/HIC_SURFACE_ARO_HYDRO_SCORES.csv`
- `reports/HIC_SURFACE_ARO_HYDRO_CONTRASTS.csv`
- `reports/HIC_SURFACE_ARO_HYDRO_BOOTSTRAP.csv`
- `reports/HIC_SURFACE_ARO_HYDRO_HIGHTAIL.csv`
