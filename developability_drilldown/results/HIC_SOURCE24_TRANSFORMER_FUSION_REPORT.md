# HIC SOURCE24 Transformer Fusion Ablation (H114–H125)

**Platform:** `DL_FOLDLOCAL_COSINE_V3`  
**Prereg:** `H114_H125_SOURCE24_FUSION_PREREGISTRATION.yaml` (commit `b4cdfec5`)  
**Internal freeze:** `H114_H125_PRE_EXTERNAL_FREEZE.yaml`  
**Controls:** EXP-H071 Scratch ARCH-2 (TEST_mean **0.5017**), EXP-H061 ESM2 ARCH-4 (TEST_mean **0.5068**)

## Auxiliary-branch audit

Historical `LateFusionAuxMLP`: Linear(p,64)→GELU→Dropout(0.2)→Linear(64,32)→GELU then concat with `z_DL`. Introduced for heterogeneous F1–F4 bundles; later reused even for 3D HSP. **Not assumed canonical for 24D physical summaries** — this batch compares it to genuine DIRECT concat.

## Central table (TEST_mean; perm = mean block-permutation ΔMAE)

| Feature | Backbone | DIRECT TEST | AUX32 TEST | Δ DIRECT−AUX | Direct perm | Aux perm | Ext DIRECT overall | Ext AUX overall |
|---------|----------|------------:|-----------:|-------------:|------------:|---------:|-------------------:|----------------:|
| SAP24 | Scratch | 0.5140 | 0.5180 | −0.0040 | +0.0175 | +0.0113 | 0.5015 | 0.4970 |
| SAP24 | ESM2 | 0.5207 | 0.5224 | −0.0017 | +0.0342 | +0.0235 | 0.4784 | 0.4766 |
| SCM24 | Scratch | 0.5185 | 0.5184 | +0.0001 | +0.0228 | +0.0308 | 0.4806 | 0.5090 |
| SCM24 | ESM2 | 0.5098 | 0.5133 | −0.0035 | +0.0253 | +0.0292 | 0.4823 | 0.4792 |
| COMBINED48 | Scratch | 0.5358 | **0.4988** | +0.0370 | +0.0183 | +0.0274 | 0.5097 | 0.4913 |
| COMBINED48 | ESM2 | 0.5287 | 0.5344 | −0.0057 | +0.0326 | +0.0222 | 0.4949 | 0.5075 |

Δ DIRECT−AUX < 0 ⇒ DIRECT better.

## Parameter counts (full trainable)

| Code | Mode | n_trainable |
|------|------|------------:|
| H114/H116 | Scratch DIRECT | 343682 |
| H115/H117 | Scratch AUX32 | 348386 |
| H118 | Scratch DIRECT 48D | 346754 |
| H119 | Scratch AUX32 48D | 349922 |
| H120/H122 | ESM2 DIRECT | 504962 |
| H121/H123 | ESM2 AUX32 | 509666 |
| H124 | ESM2 DIRECT 48D | 508034 |
| H125 | ESM2 AUX32 48D | 511202 |

Fusion-only Δ ≈ +4.7k (24D) / +3.2k (48D) for AUX32 vs DIRECT (matches ~Linear64→32 overhead).

## Answers

1. **SAP24 improve Scratch?** No on TEST_mean (0.514 / 0.518 vs 0.502). Primary OOF improves vs H071 for DIRECT, but Shadow worsens → mean not better.
2. **SAP24 improve ESM2?** No (0.521 / 0.522 vs 0.507).
3. **SCM24 improve either?** No reliable gain (Scratch ~0.518; ESM2 DIRECT 0.510 closest but still ≥ base).
4. **COMBINED48 outperform singles?** Only Scratch **AUX32** H119 (0.4988) edges H071 and beats SAP/SCM alone; DIRECT combined is worse (0.536). ESM2 combined worse than singles.
5. **DIRECT vs AUX32 for SAP24?** Near-tie; DIRECT slightly better TEST_mean; bootstrap CIs include 0.
6. **DIRECT vs AUX32 for SCM24?** Near-tie.
7. **Does the model use these blocks?** **Yes (functional reliance).** Block permutation ΔMAE ≈ +0.01–0.03 across modes/backbones. Score gain ≠ use; use ≠ net improvement over strong sequence backbone.
8. **Was AUX32 inappropriate for small vectors?** Not clearly harmful for 24D (CASE C). For Scratch COMBINED48, AUX32 **helps** vs DIRECT (CASE B locally). No evidence DIRECT uniquely unlocks SAP24.
9. **Reconcile ~0.46 XGBoost?** **No.** Best Transformer+SOURCE external overall here ≈ 0.48; internal TEST_mean stays ~0.50. Separate XGBoost reproduction still required.
10. **SURFACE+SOURCE24 next?** **Not justified** as promotion — SOURCE alone does not beat backbone; SURFACE fusion deferred until a clearer SOURCE win.
11. **Residue-level?** **Not justified** — antibody-level SOURCE blocks do not improve Transformer TEST_mean with reliance-without-gain pattern.

## Fusion-mode verdict

**Closest to CASE D for predictive gain** (neither fusion beats backbone consistently), with **CASE C** for DIRECT vs AUX32 on SAP/SCM 24D, and a **local CASE B** for Scratch COMBINED48 (AUX32 ≫ DIRECT).

## Feature-block verdict

| Block | Verdict |
|-------|---------|
| SOURCE_SAP24 | NOT_SUPPORTED for Transformer promotion |
| SOURCE_SCM24 | NOT_SUPPORTED |
| COMBINED48 | WEAK / exploratory only (H119 slight TEST_mean edge; not robust across backbone/mode) |

## Fixed-feature (Ridge/SVR) consistency

Classical feature-research also found SOURCE24 weak. Transformer late fusion does **not** reverse that; it adds evidence that features are *consumed* (permutation) without improving HIC MAE over H071/H061.

## External (diagnostic only)

Does not alter internal verdicts. Several overall MAEs ~0.48 are numerically interesting but post-competition only.

## STOP

Do **not** run EXP-H126 / SURFACE+SOURCE / XGBoost / residue SAP/SCM / T142 from this batch.
