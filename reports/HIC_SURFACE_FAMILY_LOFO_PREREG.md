# HIC SURFACE Physical-Family LOFO — Preregistration

**STATUS: `HIC_SURFACE_FAMILY_LOFO_PREREGISTERED`**

This document freezes the F1_SURFACE35 physical-family taxonomy and leave-one-family-out
(LOFO) design **before** any `FULL_MINUS_F` training.

Do **not** edit:

- `reports/HIC_SCIENTIFIC_FREEZE_V3.md`
- `reports/HIC_SURFACE_PROSPECTIVE_FINAL.md`
- `reports/HIC_SURFACE_ARO_HYDRO_DECOMPOSITION.md`
- prior factorial / freeze documents

Do **not** redefine families after seeing CV / Public / Private / Test / HIGH-tail results.

---

## 0. Provenance

| Field | Value |
|-------|--------|
| Freeze v3 SHA | `1bfdf9107f55d14c58117aaa7f64e255dc387ed9` |
| ARO/HYDRO decomposition package SHA | `ebf2fda9615c12f037914bc5b605a6ec6ba2eabc` |
| ARO/HYDRO prereg SHA | `40f8ac0a7cf9caff59cbae913f99638d8cc93c4b` |
| Taxonomy CSV | `reports/HIC_SURFACE_FAMILY_TAXONOMY.csv` |
| Machine taxonomy | `developability_drilldown/results/HIC_SURFACE_FAMILY_TAXONOMY_FROZEN.json` |
| Prereg introducing commit | *(filled after sole prereg+taxonomy commit)* |

**Holdout note:** Project-wide Test is not pristine for all HIC history.
New `FULL_MINUS_F` predictions are unevaluated at this prereg and are treated as
`prospective_relative_to_FAMILY_LOFO_prereg`.

---

## 1. Scientific question

Canonical F1_SURFACE = ARO19 ‖ HYDRO16 (35D) is prospectively useful and
ARO/HYDRO blocks are complementary.

> Under FULL35, which **physical descriptor families** carry unique conditional
> predictive information (LOFO removal worsens prediction)?

This is **family-level LOFO**, not single-feature ablation and not new model search.

---

## 2. Exact 35D column order (canonical)

Source of truth (implementation, not report prose):

- Loader: `H047AuxFeatureStore("F1_SURFACE")` → `RealF1Surface35AuxFeatureStore`
- Asset: `developability_drilldown/experiments/features/EXP-H047.parquet`
- Slices within H047 feat columns: ARO `feat[1395:1414]`, HYDRO `feat[1414:1430]`
- Concatenation: **ARO19 then HYDRO16** (indices 0–18, 19–34)

Exact columns (0-based):

```
 0 aro_exposed_TYR_count
 1 aro_exposed_TRP_count
 2 aro_exposed_PHE_count
 3 aro_exposed_aromatic_total_count
 4 aro_aromatic_exposed_SASA_total
 5 aro_aromatic_exposed_SASA_fraction
 6 aro_strongly_exposed_aromatic_count
 7 aro_strongly_exposed_aromatic_SASA
 8 aro_CDR_exposed_aromatic_count
 9 aro_CDR_aromatic_SASA
10 aro_CDR_aromatic_fraction
11 aro_aromatic_patch_count
12 aro_largest_aromatic_patch_n_res
13 aro_largest_aromatic_patch_exposed_SASA
14 aro_max_local_aromatic_SASA
15 aro_sequence_aromatic_count
16 aro_sequence_TYR_count
17 aro_sequence_TRP_count
18 aro_sequence_PHE_count
19 mean_H_surface
20 q75_H_surface
21 q90_H_surface
22 q95_H_surface
23 max_H_surface
24 positive_H_area_fraction
25 top10_H_mean
26 top10_H_area_fraction
27 high_H_patch_count
28 largest_high_H_patch_area_fraction
29 largest_high_H_patch_n_vertices
30 CDR_mean_H
31 CDR_q90_H
32 CDR_high_H_area_fraction
33 n_surface_points
34 phi_finite_frac
```

Generators:

- ARO: `organizer_extension/.../extract_physical_batch1.py::aromatic_features`
- HYDRO: `organizer_extension/.../hydro_surface.py::hydro_summaries` (+ QC co-write)

Taxonomy assigned **target-blind** (no HIC correlation / CV / Test / importance used).

---

## 3. Frozen physical families (n=8)

Family count follows generator semantics; not forced to 4–6.

| Family | Block | Indices (0-based) | n | Physical meaning |
|--------|-------|-------------------|---:|------------------|
| `ARO_EXPOSED_AMOUNT` | ARO19 | 0–7 | 8 | Exposed aromatic amount (counts + SASA + strong exposure) |
| `ARO_CDR_LOCALIZATION` | ARO19 | 8–10 | 3 | CDR localization of exposed aromatics |
| `ARO_PATCH_TOPOLOGY` | ARO19 | 11–14 | 4 | Exposed-aromatic Cα patches / local neighborhood |
| `ARO_SEQUENCE_COMPOSITION` | ARO19 | 15–18 | 4 | Exposure-blind sequence aromatic composition (QC channel) |
| `HYDRO_GLOBAL_FIELD` | HYDRO16 | 19–24 | 6 | Whole-Fv H(s) global distribution |
| `HYDRO_LOCAL_HIGH_INTENSITY` | HYDRO16 | 25–29 | 5 | Top-intensity / high-H patch structure |
| `HYDRO_CDR_FIELD` | HYDRO16 | 30–32 | 3 | CDR-restricted H(s) summaries |
| `HYDRO_SURFACE_CONSTRUCTION_QC` | HYDRO16 | 33–34 | 2 | Surface sampling / φ QC co-bundled columns |

Rules applied:

- Every column assigned **exactly once**
- No ARO↔HYDRO cross-block families
- Threshold / summary variants of the same physical quantity stay together
- No post-hoc redefinition after results

---

## 4. Mask implementation (critical)

**Forbidden:** zeroing raw features before StandardScaler fit/transform
(non-zero train means would reintroduce nonzero scaled values).

**Required pipeline:**

1. Load canonical FULL35
2. Fold-local median impute + StandardScaler fit on **train only**
3. Transform train / val / test
4. **After transform**, set removed-family coordinates to **exactly 0** in model-input space
5. Feed AuxMLP

Invariant: `aux_dim=35`, same AuxMLP, same parameter count, same slots.

Smoke tests required before/with training:

- removed coords == 0 exactly
- retained coords identical to unmasked FULL pipeline
- same index mask on train/val/test
- no fold-local scaler leakage from masking

---

## 5. Fixed contexts & experiment count

| Context | FULL baseline (reuse, no retrain) |
|---------|-----------------------------------|
| AbLang1 × JOINT × FULL | `EXP-H341` |
| AbLingua × JOINT × FULL | `EXP-H343` |

New trainings only:

- For each of **8 families** × **2 representations** → **16** `FULL_MINUS_F` arms

Experiment codes: next free HIC codes starting at **EXP-H348**.

Protocol identical to H340–H347 prospective series (seed 101, JOINT, FULL, late_concat_aux32, LR grid, early stop, splits).

---

## 6. Primary contrast

For family F:

`Δ = FULL_MINUS_F − FULL`

- Δ > 0 → removal hurts → unique conditional predictive value
- Δ ≈ 0 → little unique value under FULL background
- Δ < 0 → removal helps → redundant/harmful under FULL background

Endpoints: ΔCV_primary, ΔCV_shadow, ΔCV_mean, ΔPublic, ΔPrivate, ΔTest.

This is **conditional contribution given other families**, not standalone family power.

---

## 7. Analysis plan

### Internal

- Primary / Shadow OOF MAE and antibody-level paired AE
- Bootstrap: N_BOOT=10000, seed=101, metric `AE_FULL_MINUS_F − AE_FULL`

### External

- After **all** 16 arms have fixed predictions, score all arms on Public / Private / Test
- No mid-stream redesign; no selecting families from internal results

### HIGH-tail (secondary only)

- Threshold **HIC > 11.5** only (no new threshold search)
- Diagnostic localization of which family removal loses HIGH-tail rescue

### Heterogeneity

- Sample-level AE deltas; cross-rep agreement; no new families from responders

---

## 8. Predeclared evidence classes (per family)

Assigned from external directional pattern (do not rewrite post-hoc with bootstrap):

| Class | Rule |
|-------|------|
| `ROBUST_CONDITIONAL_CONTRIBUTOR` | Test Δ>0 in 2/2 reps **and** Public Δ>0 in 2/2 **and** Private Δ>0 in 2/2 (all 6 external contrasts worsen on removal) |
| `EXTERNAL_DIRECTIONAL_CONTRIBUTOR` | Test Δ>0 in 2/2 reps, but Public/Private mixed |
| `CONTEXT_DEPENDENT_CONTRIBUTOR` | One rep removal worsens; other neutral/improves |
| `LITTLE_UNIQUE_CONDITIONAL_VALUE` | Test does not show consistent removal-worsening across both reps (neutral/improve) |

Bootstrap CIs are reported but do **not** override these directional classes.

**Contributor ≠ biological causality.**

---

## 9. Global interpretation candidates (choose one after results)

- `DISTRIBUTED_SURFACE_SIGNAL`
- `AROMATIC_FAMILY_ENRICHED`
- `HYDRO_FAMILY_ENRICHED`
- `AROMATIC_AND_HYDRO_FAMILIES_COMPLEMENTARY`
- `CONTEXT_DEPENDENT_SURFACE_SIGNAL`
- `FAMILY_LOCALIZATION_INCONCLUSIVE`

---

## 10. Prohibited in this phase

- Individual-feature / 35-way ablation
- Optuna / topology / annotation / representation search
- Family boundary changes after results
- Threshold search; SHAP-driven or correlation-driven selection
- Ensembles; calibration
- CONTINUOUS_SURFACE / TITRATION_SHAPE / HSP additions
- Editing Freeze v3

---

## 11. Outputs (planned)

- `reports/HIC_SURFACE_FAMILY_LOFO_RESULTS.md`
- `reports/HIC_SURFACE_FAMILY_LOFO_CONFIG_AUDIT.csv`
- `reports/HIC_SURFACE_FAMILY_LOFO_SCORES.csv`
- `reports/HIC_SURFACE_FAMILY_LOFO_CONTRASTS.csv`
- `reports/HIC_SURFACE_FAMILY_LOFO_BOOTSTRAP.csv`
- `reports/HIC_SURFACE_FAMILY_LOFO_HIGHTAIL.csv`
- `reports/HIC_SURFACE_FAMILY_LOFO_HETEROGENEITY.csv`

---

## 12. Stop condition

Stop after family-level LOFO conclusion. No feature-level search in this phase.
