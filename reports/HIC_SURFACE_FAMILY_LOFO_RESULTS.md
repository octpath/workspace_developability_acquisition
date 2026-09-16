# HIC SURFACE Physical-Family LOFO Results

**STATUS: `HIC_SURFACE_FAMILY_LOFO_COMPLETE`**

| Field | Value |
|-------|--------|
| Taxonomy / prereg SHA | `e45d9ea8c4720ac92388f7219e7c0d6a0b92cf3f` |
| Freeze v3 SHA | `1bfdf9107f55d14c58117aaa7f64e255dc387ed9` |
| ARO/HYDRO decomp SHA | `ebf2fda9615c12f037914bc5b605a6ec6ba2eabc` |
| Analysis code SHA | `ed8773ca944dfd8b17f92d62a51c6709caf2095e` |
| n_families | 8 |
| Global interpretation | `HYDRO_FAMILY_ENRICHED` |

## 1–3. Taxonomy

See `HIC_SURFACE_FAMILY_TAXONOMY.csv` (35 columns, exactly-once family membership).

- **`ARO_EXPOSED_AMOUNT`** indices=[0, 1, 2, 3, 4, 5, 6, 7]: aro_exposed_TYR_count, aro_exposed_TRP_count, aro_exposed_PHE_count, aro_exposed_aromatic_total_count, aro_aromatic_exposed_SASA_total, aro_aromatic_exposed_SASA_fraction, aro_strongly_exposed_aromatic_count, aro_strongly_exposed_aromatic_SASA
- **`ARO_CDR_LOCALIZATION`** indices=[8, 9, 10]: aro_CDR_exposed_aromatic_count, aro_CDR_aromatic_SASA, aro_CDR_aromatic_fraction
- **`ARO_PATCH_TOPOLOGY`** indices=[11, 12, 13, 14]: aro_aromatic_patch_count, aro_largest_aromatic_patch_n_res, aro_largest_aromatic_patch_exposed_SASA, aro_max_local_aromatic_SASA
- **`ARO_SEQUENCE_COMPOSITION`** indices=[15, 16, 17, 18]: aro_sequence_aromatic_count, aro_sequence_TYR_count, aro_sequence_TRP_count, aro_sequence_PHE_count
- **`HYDRO_GLOBAL_FIELD`** indices=[19, 20, 21, 22, 23, 24]: mean_H_surface, q75_H_surface, q90_H_surface, q95_H_surface, max_H_surface, positive_H_area_fraction
- **`HYDRO_LOCAL_HIGH_INTENSITY`** indices=[25, 26, 27, 28, 29]: top10_H_mean, top10_H_area_fraction, high_H_patch_count, largest_high_H_patch_area_fraction, largest_high_H_patch_n_vertices
- **`HYDRO_CDR_FIELD`** indices=[30, 31, 32]: CDR_mean_H, CDR_q90_H, CDR_high_H_area_fraction
- **`HYDRO_SURFACE_CONSTRUCTION_QC`** indices=[33, 34]: n_surface_points, phi_finite_frac

## 4. Mask implementation audit

Mask stage: **post_standardscaler_model_input** (removed coords set to exact 0 after fold-local scale).
Smoke: `tests/test_hic_surface_family_lofo_mask.py`. Config audit CSV written.

## 5–6. Experiment codes & config equality

- `EXP-H348` — ablang1 FULL_MINUS_ARO_EXPOSED_AMOUNT (baseline `EXP-H341`)
- `EXP-H349` — ablingua FULL_MINUS_ARO_EXPOSED_AMOUNT (baseline `EXP-H343`)
- `EXP-H350` — ablang1 FULL_MINUS_ARO_CDR_LOCALIZATION (baseline `EXP-H341`)
- `EXP-H351` — ablingua FULL_MINUS_ARO_CDR_LOCALIZATION (baseline `EXP-H343`)
- `EXP-H352` — ablang1 FULL_MINUS_ARO_PATCH_TOPOLOGY (baseline `EXP-H341`)
- `EXP-H353` — ablingua FULL_MINUS_ARO_PATCH_TOPOLOGY (baseline `EXP-H343`)
- `EXP-H354` — ablang1 FULL_MINUS_ARO_SEQUENCE_COMPOSITION (baseline `EXP-H341`)
- `EXP-H355` — ablingua FULL_MINUS_ARO_SEQUENCE_COMPOSITION (baseline `EXP-H343`)
- `EXP-H356` — ablang1 FULL_MINUS_HYDRO_GLOBAL_FIELD (baseline `EXP-H341`)
- `EXP-H357` — ablingua FULL_MINUS_HYDRO_GLOBAL_FIELD (baseline `EXP-H343`)
- `EXP-H358` — ablang1 FULL_MINUS_HYDRO_LOCAL_HIGH_INTENSITY (baseline `EXP-H341`)
- `EXP-H359` — ablingua FULL_MINUS_HYDRO_LOCAL_HIGH_INTENSITY (baseline `EXP-H343`)
- `EXP-H360` — ablang1 FULL_MINUS_HYDRO_CDR_FIELD (baseline `EXP-H341`)
- `EXP-H361` — ablingua FULL_MINUS_HYDRO_CDR_FIELD (baseline `EXP-H343`)
- `EXP-H362` — ablang1 FULL_MINUS_HYDRO_SURFACE_CONSTRUCTION_QC (baseline `EXP-H341`)
- `EXP-H363` — ablingua FULL_MINUS_HYDRO_SURFACE_CONSTRUCTION_QC (baseline `EXP-H343`)

## 7–11. Scores, contrasts, bootstrap, evidence classes

### Scores (selected)

| Rep | Arm | CV_P | Public | Private | Test |
|-----|-----|------|--------|---------|------|
| ablang1 | FULL35 | 0.475665 | 0.372349 | 0.407956 | 0.390153 |
| ablingua | FULL35 | 0.479274 | 0.376478 | 0.429763 | 0.403121 |
| ablang1 | FULL_MINUS_ARO_EXPOSED_AMOUNT | 0.515726 | 0.360856 | 0.411851 | 0.386353 |
| ablingua | FULL_MINUS_ARO_EXPOSED_AMOUNT | 0.465691 | 0.366202 | 0.428945 | 0.397574 |
| ablang1 | FULL_MINUS_ARO_CDR_LOCALIZATION | 0.456243 | 0.366886 | 0.419986 | 0.393436 |
| ablingua | FULL_MINUS_ARO_CDR_LOCALIZATION | 0.457261 | 0.365418 | 0.425760 | 0.395589 |
| ablang1 | FULL_MINUS_ARO_PATCH_TOPOLOGY | 0.464948 | 0.382008 | 0.421825 | 0.401917 |
| ablingua | FULL_MINUS_ARO_PATCH_TOPOLOGY | 0.462859 | 0.363710 | 0.441430 | 0.402570 |
| ablang1 | FULL_MINUS_ARO_SEQUENCE_COMPOSITION | 0.493997 | 0.393126 | 0.445821 | 0.419474 |
| ablingua | FULL_MINUS_ARO_SEQUENCE_COMPOSITION | 0.461346 | 0.362046 | 0.438445 | 0.400246 |
| ablang1 | FULL_MINUS_HYDRO_GLOBAL_FIELD | 0.476374 | 0.383095 | 0.402746 | 0.392921 |
| ablingua | FULL_MINUS_HYDRO_GLOBAL_FIELD | 0.475388 | 0.384348 | 0.425040 | 0.404694 |
| ablang1 | FULL_MINUS_HYDRO_LOCAL_HIGH_INTENSITY | 0.511399 | 0.407671 | 0.417176 | 0.412424 |
| ablingua | FULL_MINUS_HYDRO_LOCAL_HIGH_INTENSITY | 0.469989 | 0.374519 | 0.450299 | 0.412409 |
| ablang1 | FULL_MINUS_HYDRO_CDR_FIELD | 0.502908 | 0.397955 | 0.417730 | 0.407842 |
| ablingua | FULL_MINUS_HYDRO_CDR_FIELD | 0.467153 | 0.361897 | 0.429596 | 0.395747 |
| ablang1 | FULL_MINUS_HYDRO_SURFACE_CONSTRUCTION_QC | 0.520871 | 0.412556 | 0.432883 | 0.422720 |
| ablingua | FULL_MINUS_HYDRO_SURFACE_CONSTRUCTION_QC | 0.454843 | 0.361157 | 0.431185 | 0.396171 |

### Contrasts Δ = FULL_MINUS_F − FULL (positive = family unique value)

| Family | Rep | ΔCV_P | ΔCV_S | ΔPub | ΔPriv | ΔTest |
|--------|-----|-------|-------|------|-------|-------|
| ARO_EXPOSED_AMOUNT | ablang1 | +0.040060 | -0.012897 | -0.011494 | +0.003895 | -0.003799 |
| ARO_EXPOSED_AMOUNT | ablingua | -0.013583 | +0.009931 | -0.010276 | -0.000819 | -0.005547 |
| ARO_CDR_LOCALIZATION | ablang1 | -0.019422 | -0.028352 | -0.005463 | +0.012030 | +0.003283 |
| ARO_CDR_LOCALIZATION | ablingua | -0.022013 | +0.006660 | -0.011060 | -0.004003 | -0.007531 |
| ARO_PATCH_TOPOLOGY | ablang1 | -0.010717 | -0.027403 | +0.009659 | +0.013869 | +0.011764 |
| ARO_PATCH_TOPOLOGY | ablingua | -0.016415 | +0.051196 | -0.012768 | +0.011666 | -0.000551 |
| ARO_SEQUENCE_COMPOSITION | ablang1 | +0.018331 | +0.016905 | +0.020777 | +0.037865 | +0.029321 |
| ARO_SEQUENCE_COMPOSITION | ablingua | -0.017928 | -0.006015 | -0.014432 | +0.008682 | -0.002875 |
| HYDRO_GLOBAL_FIELD | ablang1 | +0.000709 | -0.026907 | +0.010746 | -0.005210 | +0.002768 |
| HYDRO_GLOBAL_FIELD | ablingua | -0.003886 | +0.026875 | +0.007870 | -0.004723 | +0.001573 |
| HYDRO_LOCAL_HIGH_INTENSITY | ablang1 | +0.035733 | -0.025364 | +0.035322 | +0.009220 | +0.022271 |
| HYDRO_LOCAL_HIGH_INTENSITY | ablingua | -0.009285 | +0.004370 | -0.001959 | +0.020535 | +0.009288 |
| HYDRO_CDR_FIELD | ablang1 | +0.027243 | +0.024838 | +0.025605 | +0.009774 | +0.017690 |
| HYDRO_CDR_FIELD | ablingua | -0.012121 | +0.030514 | -0.014581 | -0.000167 | -0.007374 |
| HYDRO_SURFACE_CONSTRUCTION_QC | ablang1 | +0.045206 | -0.013695 | +0.040207 | +0.024927 | +0.032567 |
| HYDRO_SURFACE_CONSTRUCTION_QC | ablingua | -0.024431 | +0.045267 | -0.015321 | +0.001422 | -0.006950 |

### Evidence class per family

- **`ARO_EXPOSED_AMOUNT`** → `LITTLE_UNIQUE_CONDITIONAL_VALUE` (AbLang1 ΔTest=-0.0038, AbLingua ΔTest=-0.0055)
- **`ARO_CDR_LOCALIZATION`** → `CONTEXT_DEPENDENT_CONTRIBUTOR` (AbLang1 ΔTest=+0.0033, AbLingua ΔTest=-0.0075)
- **`ARO_PATCH_TOPOLOGY`** → `CONTEXT_DEPENDENT_CONTRIBUTOR` (AbLang1 ΔTest=+0.0118, AbLingua ΔTest=-0.0006)
- **`ARO_SEQUENCE_COMPOSITION`** → `CONTEXT_DEPENDENT_CONTRIBUTOR` (AbLang1 ΔTest=+0.0293, AbLingua ΔTest=-0.0029)
- **`HYDRO_GLOBAL_FIELD`** → `EXTERNAL_DIRECTIONAL_CONTRIBUTOR` (AbLang1 ΔTest=+0.0028, AbLingua ΔTest=+0.0016)
- **`HYDRO_LOCAL_HIGH_INTENSITY`** → `EXTERNAL_DIRECTIONAL_CONTRIBUTOR` (AbLang1 ΔTest=+0.0223, AbLingua ΔTest=+0.0093)
- **`HYDRO_CDR_FIELD`** → `CONTEXT_DEPENDENT_CONTRIBUTOR` (AbLang1 ΔTest=+0.0177, AbLingua ΔTest=-0.0074)
- **`HYDRO_SURFACE_CONSTRUCTION_QC`** → `CONTEXT_DEPENDENT_CONTRIBUTOR` (AbLang1 ΔTest=+0.0326, AbLingua ΔTest=-0.0069)

### Bootstrap (AE_MINUS − AE_FULL; N=10000, seed=101)

See `HIC_SURFACE_FAMILY_LOFO_BOOTSTRAP.csv`.

## 12. HIGH-tail secondary (HIC>11.5, n=7)

- ARO_EXPOSED_AMOUNT: ablang1 ΔMAE_high=-0.1168; ablingua ΔMAE_high=-0.0217
- ARO_CDR_LOCALIZATION: ablang1 ΔMAE_high=+0.0735; ablingua ΔMAE_high=-0.0957
- ARO_PATCH_TOPOLOGY: ablang1 ΔMAE_high=+0.0766; ablingua ΔMAE_high=-0.0955
- ARO_SEQUENCE_COMPOSITION: ablang1 ΔMAE_high=+0.0824; ablingua ΔMAE_high=-0.1468
- HYDRO_GLOBAL_FIELD: ablang1 ΔMAE_high=+0.0393; ablingua ΔMAE_high=+0.0575
- HYDRO_LOCAL_HIGH_INTENSITY: ablang1 ΔMAE_high=+0.2630; ablingua ΔMAE_high=-0.0658
- HYDRO_CDR_FIELD: ablang1 ΔMAE_high=+0.2016; ablingua ΔMAE_high=+0.0426
- HYDRO_SURFACE_CONSTRUCTION_QC: ablang1 ΔMAE_high=+0.2579; ablingua ΔMAE_high=-0.1068

## 13. Sample-level heterogeneity

See `HIC_SURFACE_FAMILY_LOFO_HETEROGENEITY.csv` (worsen fractions + cross-rep agreement).

## 14–15. ARO vs HYDRO synthesis & global interpretation

- ARO directional/robust contributors: none
- HYDRO directional/robust contributors: ['HYDRO_GLOBAL_FIELD', 'HYDRO_LOCAL_HIGH_INTENSITY']
- **Global interpretation: `HYDRO_FAMILY_ENRICHED`**

Notes (faithful to effect sizes, not post-hoc reclassification):

- Among hydro directional families, **`HYDRO_LOCAL_HIGH_INTENSITY`** has the larger Test effect (AbLang1 ΔTest≈+0.022; AbLingua ≈+0.009).
- **`HYDRO_GLOBAL_FIELD`** meets the Test 2/2 directional rule but magnitude is small and Private Δ is negative in both reps → not ROBUST.
- Several ARO families are context-dependent (AbLang1-only worsening); **`ARO_EXPOSED_AMOUNT`** shows little unique conditional value under FULL (Test Δ < 0 in both reps).
- `HYDRO_SURFACE_CONSTRUCTION_QC` is context-dependent; do not interpret it as a hydrophobicity mechanism (includes constant `phi_finite_frac` + `n_surface_points`).

## 16. Exposed-aromatic hypothesis update

NOT_SUPPORTED_AS_PRIMARY_STORY — under FULL35 LOFO, hydro local-intensity / global-field families show the only cross-representation Test directional contribution; aromatic families do not. Avoid aromatics-only narrative. Still no causal claim for any hydro descriptor.

## 17. Next scientific recommendation

Preferred next (still **no** 35-way search): coarse probes **inside** `HYDRO_LOCAL_HIGH_INTENSITY` (and optionally weak `HYDRO_GLOBAL_FIELD`), not ARO19-internal family search as the primary path.

## Non-claims

Family contribution is predictive conditional information only — not biophysical causality.

Allowed ceiling wording for directional hydro families:

> This physical descriptor family carries reproducible conditional predictive information beyond the remaining F1_SURFACE families in the tested model contexts.

## Stop

Family-level LOFO complete. No 35-way / individual-feature search in this phase.
Do not edit `HIC_SCIENTIFIC_FREEZE_V3.md` in this phase.

