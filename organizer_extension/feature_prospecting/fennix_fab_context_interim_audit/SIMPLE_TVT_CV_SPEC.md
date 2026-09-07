# Simple Train / Validation / Test CV — Spec

**Date:** 2026-09-07  
**Role:** Orthogonal robustness check — does **not** replace `STRICT_NESTED_INCREMENT`.  
**Constraint:** Evaluation-only; FeNNix production worker untouched; no new embeddings / structure extraction.

---

## Scientific question

| Protocol | Question |
|----------|----------|
| `STRICT_NESTED_INCREMENT` | Does structure improve the **incumbent** via leakage-safe late-fusion / residual? |
| `SIMPLE_TVT_CV` (this) | Does structure improve the **same direct model** when features are concatenated vs BASE alone? |

No globally precomputed incumbent / AbLang2 / ESM2 **OOF scalars** as inputs.

---

## Fold protocol (frozen)

Source: `virtual_participant/stage0_cv/cv_primary.csv`, `cv_shadow.csv`.

For test fold \(k \in \{0,1,2,3,4\}\):

| Role | Folds |
|------|-------|
| TEST | \(k\) |
| VALIDATION | \((k+1) \bmod 5\) |
| TRAIN | the other three |

Each sample is TEST exactly once. Fold-role mapping is **not** optimized on targets. Run independently for Primary and Shadow.

---

## BASE representations (frozen paths)

### TmApp BASE — `ABLANG2_HL_PAIRED + SEQ_BASIC`

Corresponds to Stage2 fusion inputs underlying `TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt` (representation only; model here is Ridge, not SVR).

| Block | Path | Dim (Dev) |
|-------|------|-----------|
| AbLang2 HL_paired | `virtual_participant/stage2_plm/cache/stage2_embeddings.npz` key `ablang2__HL_paired` | 480 |
| SEQ_BASIC | Stage1 `make_xy(..., "SEQ_BASIC")` numeric | 78 |

**Total BASE dim ≈ 558.**

### HIC BASE — `ESM2_H + SEQ_ALL`

Corresponds to Stage2 fusion inputs underlying `HIC__FUSION__esm2__H__SEQ_ALL__SVROpt`.

| Block | Path | Dim |
|-------|------|-----|
| ESM2 H | same npz key `esm2__H` | 1280 |
| SEQ_ALL | Stage1 `make_xy(..., "SEQ_ALL")` numeric | 115 |

**Total ≈ 1395.**

### HIC BASE+ARO (stronger baseline) — `ESM2_H + SEQ_ALL + AROMATIC_TOPO`

Used as BASE for `CONTINUOUS_SURFACE` (tests continuous surface **beyond** aromatic).

| Block | Path | Dim |
|-------|------|-----|
| above | | |
| Aromatic topo | `structure_gap_closure/cache/aromatic_features_esmfold.csv` | ≤23 numeric |

`HIC_SURFACE_ALL` uses BASE = `ESM2_H+SEQ_ALL` (no aromatic) and STRUCTURE = continuous ∪ aromatic, matching the Gap Closure aggregate package without double-counting aromatic in BASE.

---

## Structure families (frozen; no column selection)

### Gap Closure TmApp

| Family | Path(s) |
|--------|---------|
| `PACKING_CAVITY` | `structure_gap_closure/results/TMAPP_PACKING_CAVITY_FEATURES.csv` |
| `BURIED_UNSAT` | `.../TMAPP_BURIED_UNSAT_FEATURES.csv` |
| `FAB_INTERFACE` | `.../TMAPP_INTERFACE_FEATURES.csv` (drop status col) |
| `CORE_DEFECT` | packing ∪ buried_unsat |
| `GAP_ALL` | packing ∪ buried_unsat ∪ interface |

### Gap Closure HIC

| Family | Definition |
|--------|------------|
| `CONTINUOUS_SURFACE` | `HIC_CONTINUOUS_SURFACE_FEATURES.csv` columns `fv_esmfold__*` |
| `HIC_SURFACE_ALL` | continuous surface ∪ aromatic topo |

### Interim FeNNix (usable Dev subset only)

`fennix_fab_context_interim_audit/results/INTERIM_{FAMILY}_FEATURES.csv`  
Families: `DELTA_GEOM`, `DELTA_ENV`, `CONSTANT`, `INTERFACE`, `FULL_FAB_NORMALIZED`, `COMBINED_PREDECLARED`.

Label: **`PROVISIONAL_SIMPLE_CV`**. Require every rotation `n_train≥20`, `n_val≥5`, `n_test≥5`.

---

## Model

1. Median impute — fit on TRAIN only.  
2. `StandardScaler` — fit on TRAIN only.  
3. Ridge, α ∈ `{0.1, 1, 10, 100}` — choose α by **VALIDATION MAE**.  
4. Refit impute + scaler + Ridge(α*) on **TRAIN ∪ VAL**.  
5. Predict **TEST** once.

Repeat for BASE and BASE+STRUCTURE with identical BASE columns.

**SVR:** not run (would require inventing a small grid for BASE+STRUCTURE; Ridge is the declared primary).

---

## Metrics

- `Delta_MAE = BASE_MAE − (BASE+STRUCTURE)_MAE` (positive = structure helps).  
- Labels: `SIMPLE_CV_CONSISTENT_IMPROVEMENT` / `SIMPLE_CV_MIXED` / `SIMPLE_CV_NO_IMPROVEMENT`.  
- Bootstrap B=10000 only if Primary and Shadow Δ > 0.

---

## Outputs

- `SIMPLE_TVT_CV_SPEC.md` (this file)
- `SIMPLE_TVT_CV_RESULTS.csv`
- `SIMPLE_VS_STRICT_COMPARISON.csv`
- `SIMPLE_TVT_CV_REPORT_JA.md`
