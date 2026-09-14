# HIC Scientific Freeze v3

**STATUS: `HIC_SCIENTIFIC_FREEZE_V3`**

This document freezes the current HIC scientific picture by **retaining Freeze v2 conclusions**
and adding the prospective SURFACE SHAM vs REAL replication as a new evidence stream.

It does **not** edit:

- `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md`
- `reports/HIC_SCIENTIFIC_FREEZE_V2.md`
- `reports/HIC_SURFACE_PROSPECTIVE_FINAL.md`
- `reports/HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md`

---

## 0. Provenance ledger

| Stream | Role in v3 |
|--------|------------|
| Internal factorial freeze | Factor-level conclusions under Pub/Priv embargo |
| Prospective factorial external | Rank-transfer / factor validation on H140–H339 |
| Retrospective historical | H001–H139 ranking / H102–H113 matched ablations (Test-informed selection) |
| **Prospective SURFACE SHAM–REAL** | **New:** architecture-matched SURFACE incremental value on AbLang1/AbLingua |

| Field | Value |
|-------|--------|
| Freeze v3 parent HEAD | `e6448e3b9d27b660c6e6d4f961394ae7b6547999` |
| Freeze v2 | `reports/HIC_SCIENTIFIC_FREEZE_V2.md` |
| SURFACE prospective prereg | `993b8a9e268f65ed8d203949c34cbe3ef87446e6` |
| SURFACE prospective internal freeze | `f53039ffe606a2ca685fe64efaaeafb9c3993c6b` |
| SURFACE prospective external complete | `e6448e3b9d27b660c6e6d4f961394ae7b6547999` |
| New training in this freeze document | **None** |

---

## 1. Surface incremental value (new freeze)

**Label:** `SURFACE_INCREMENTAL_VALUE_PROSPECTIVELY_REPLICATED`

| Arm | Code |
|-----|------|
| AbLang1 SHAM35 | EXP-H340 |
| AbLang1 REAL F1_SURFACE35 | EXP-H341 |
| AbLingua SHAM35 | EXP-H342 |
| AbLingua REAL F1_SURFACE35 | EXP-H343 |

External contrasts (`REAL − SHAM`; negative = SURFACE improves):

| Context | ΔPublic | ΔPrivate | ΔTest |
|---------|---------|----------|-------|
| AbLang1 | ≈ **−0.0934** | ≈ **−0.0625** | ≈ **−0.0779** |
| AbLingua | ≈ **−0.0694** | ≈ **−0.0510** | ≈ **−0.0602** |

Test antibody-level paired bootstrap 95% CIs (mean AE_REAL − AE_SHAM) are **below 0** for both representations.

External prereg verdict: `PROSPECTIVE_SURFACE_EXTERNAL_STRONG`.

**Frozen wording:**

> In preregistered AbLang1 and AbLingua model contexts, explicit antibody-specific surface physicochemical information provided reproducible incremental predictive value over an architecture- and parameter-matched sham auxiliary branch.

**Not frozen:** biological causality of surface physicochemistry for HIC retention.

---

## 2. HIGH-tail association with SURFACE (new freeze)

Threshold fixed: **HIC > 11.5**, Test n_high = **7**.

| Context | ΔMAE_high | ΔMAE_nonHIGH |
|---------|-----------|--------------|
| AbLang1 | ≈ **−0.93** | ≈ **−0.039** |
| AbLingua | ≈ **−0.76** | ≈ **−0.028** |

Prospective HIGH-tail rescue is large relative to non-HIGH and directionally matches retrospective H102–H113.

**Label:** `SURFACE_INFORMATION_STRONGLY_ASSOCIATED_WITH_HIGH_HIC_ERROR_RESCUE`

**Not frozen:** which F1 sub-block (aromatic vs hydrophobic surface field) drives HIGH-tail rescue.

---

## 3. Position of EXP-H341

- H341 Test Overall ≈ **0.390153**
- Historical H137 Test Overall ≈ **0.390740**

H341 slightly edges H137 numerically; the absolute gap is **tiny**.

**Scientific importance of H341** is **not** “new absolute champion,” but that it is the **REAL arm of a preregistered SHAM-matched contrast**. Absolute leaderboard ties must not overshadow the prospective treatment evidence.

---

## 4. Overall HIC picture (v2 retained + v3 additions)

| Theme | Status in v3 |
|-------|----------------|
| Exact model/cell CV ranking unstable | Retained |
| Topology secondary | Retained |
| Annotation secondary / context-dependent | Retained |
| Best PLM identity split-sensitive | Retained |
| Pretrained representations externally useful | Retained |
| Sequence-only HIGH-tail underprediction robust | Retained |
| Explicit surface information | **Prospectively replicated** (new) |
| Next unresolved scientific question | **`WHICH_SURFACE_INFORMATION_MATTERS`** |

---

## 5. Non-claims

- No universal best PLM identity
- No topology winner from single Test-best cells
- No biological mechanism proof for surface chemistry
- No claim that project-wide Test is a pristine unused holdout for all historical work
- Prospective SURFACE evidence is prospective **relative to its own prereg**, not a rewrite of historical contamination narratives

---

## 6. Immediate next scientific step (pointer only)

Block-level decomposition of canonical F1_SURFACE into **ARO19 vs HYDRO16** under the same AbLang1/AbLingua JOINT×FULL contexts (separate prereg). Not a return to broad architecture search.
