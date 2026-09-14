# HIC SURFACE ARO19 vs HYDRO16 Block Decomposition

**STATUS: `HIC_SURFACE_ARO_HYDRO_DECOMPOSITION_COMPLETE`**

| Field | Value |
|-------|--------|
| Freeze v3 SHA | `1bfdf9107f55d14c58117aaa7f64e255dc387ed9` |
| Decomposition prereg SHA | `40f8ac0a7cf9caff59cbae913f99638d8cc93c4b` |
| Analysis code SHA | `ebf2fda9615c12f037914bc5b605a6ec6ba2eabc` |
| Mechanistic classification | `ARO_HYDRO_COMPLEMENTARY` |

## 1–3. Provenance & experiment codes

### ablang1
- `SHAM35` → `EXP-H340`
- `ARO_ONLY35` → `EXP-H344`
- `HYDRO_ONLY35` → `EXP-H345`
- `FULL35` → `EXP-H341`
### ablingua
- `SHAM35` → `EXP-H342`
- `ARO_ONLY35` → `EXP-H346`
- `HYDRO_ONLY35` → `EXP-H347`
- `FULL35` → `EXP-H343`

## 4. Config equality audit

All arms: aux_dim=35, late_concat_aux32, JOINT/FULL, seed=101.
n_trainable matched within each representation (see `HIC_SURFACE_ARO_HYDRO_CONFIG_AUDIT.csv`).

## 5–11. Scores and contrasts

### Scores

| Rep | Arm | CV_P | CV_S | Public | Private | Test |
|-----|-----|------|------|--------|---------|------|
| ablang1 | SHAM35 | 0.523699 | 0.537799 | 0.465721 | 0.470414 | 0.468068 |
| ablang1 | ARO_ONLY35 | 0.546931 | 0.503262 | 0.442518 | 0.447825 | 0.445171 |
| ablang1 | HYDRO_ONLY35 | 0.507908 | 0.490314 | 0.409169 | 0.438188 | 0.423678 |
| ablang1 | FULL35 | 0.475665 | 0.516697 | 0.372349 | 0.407956 | 0.390153 |
| ablingua | SHAM35 | 0.495137 | 0.532780 | 0.445832 | 0.480763 | 0.463297 |
| ablingua | ARO_ONLY35 | 0.475712 | 0.503078 | 0.392707 | 0.442326 | 0.417516 |
| ablingua | HYDRO_ONLY35 | 0.484382 | 0.478734 | 0.415265 | 0.439901 | 0.427583 |
| ablingua | FULL35 | 0.479274 | 0.469713 | 0.376478 | 0.429763 | 0.403121 |

### Contrasts (Δ = A − B; negative = A better)

| Rep | Contrast | ΔCV_P | ΔCV_S | ΔPub | ΔPriv | ΔTest |
|-----|----------|-------|-------|------|-------|-------|
| ablang1 | ARO_vs_SHAM | +0.023233 | -0.034537 | -0.023204 | -0.022590 | -0.022897 |
| ablang1 | HYDRO_vs_SHAM | -0.015790 | -0.047486 | -0.056552 | -0.032226 | -0.044389 |
| ablang1 | FULL_vs_SHAM | -0.048033 | -0.021103 | -0.093372 | -0.062458 | -0.077915 |
| ablang1 | FULL_vs_ARO | -0.071266 | +0.013435 | -0.070168 | -0.039869 | -0.055018 |
| ablang1 | FULL_vs_HYDRO | -0.032243 | +0.026383 | -0.036820 | -0.030232 | -0.033526 |
| ablingua | ARO_vs_SHAM | -0.019425 | -0.029702 | -0.053125 | -0.038437 | -0.045781 |
| ablingua | HYDRO_vs_SHAM | -0.010755 | -0.054045 | -0.030567 | -0.040862 | -0.035714 |
| ablingua | FULL_vs_SHAM | -0.015863 | -0.063067 | -0.069354 | -0.050999 | -0.060177 |
| ablingua | FULL_vs_ARO | +0.003562 | -0.033365 | -0.016229 | -0.012563 | -0.014396 |
| ablingua | FULL_vs_HYDRO | -0.005108 | -0.009022 | -0.038787 | -0.010137 | -0.024462 |

### Bootstrap (Primary OOF & Test AE; N=10000, seed=101)

- ablang1/ARO_vs_SHAM/oof_primary: mean=+0.023233 CI=[-0.030399, +0.079056] frac_improve=0.500
- ablang1/ARO_vs_SHAM/test: mean=-0.022897 CI=[-0.053022, +0.005683] frac_improve=0.463
- ablang1/HYDRO_vs_SHAM/oof_primary: mean=-0.015790 CI=[-0.054497, +0.023665] frac_improve=0.506
- ablang1/HYDRO_vs_SHAM/test: mean=-0.044389 CI=[-0.080102, -0.010589] frac_improve=0.562
- ablang1/FULL_vs_SHAM/oof_primary: mean=-0.048033 CI=[-0.104607, +0.008496] frac_improve=0.549
- ablang1/FULL_vs_SHAM/test: mean=-0.077915 CI=[-0.129071, -0.031777] frac_improve=0.593
- ablang1/FULL_vs_ARO/oof_primary: mean=-0.071266 CI=[-0.123509, -0.022517] frac_improve=0.568
- ablang1/FULL_vs_ARO/test: mean=-0.055018 CI=[-0.086408, -0.024795] frac_improve=0.599
- ablang1/FULL_vs_HYDRO/oof_primary: mean=-0.032243 CI=[-0.077170, +0.013948] frac_improve=0.586
- ablang1/FULL_vs_HYDRO/test: mean=-0.033526 CI=[-0.066315, -0.002367] frac_improve=0.543
- ablingua/ARO_vs_SHAM/oof_primary: mean=-0.019425 CI=[-0.069331, +0.028881] frac_improve=0.469
- ablingua/ARO_vs_SHAM/test: mean=-0.045781 CI=[-0.083456, -0.009908] frac_improve=0.580
- ablingua/HYDRO_vs_SHAM/oof_primary: mean=-0.010755 CI=[-0.040167, +0.018518] frac_improve=0.543
- ablingua/HYDRO_vs_SHAM/test: mean=-0.035714 CI=[-0.064383, -0.006633] frac_improve=0.593
- ablingua/FULL_vs_SHAM/oof_primary: mean=-0.015863 CI=[-0.068133, +0.040020] frac_improve=0.531
- ablingua/FULL_vs_SHAM/test: mean=-0.060177 CI=[-0.102852, -0.019716] frac_improve=0.586
- ablingua/FULL_vs_ARO/oof_primary: mean=+0.003562 CI=[-0.038675, +0.050011] frac_improve=0.568
- ablingua/FULL_vs_ARO/test: mean=-0.014396 CI=[-0.037240, +0.007710] frac_improve=0.488
- ablingua/FULL_vs_HYDRO/oof_primary: mean=-0.005108 CI=[-0.055178, +0.048141] frac_improve=0.506
- ablingua/FULL_vs_HYDRO/test: mean=-0.024462 CI=[-0.057986, +0.007666] frac_improve=0.525

## 12–13. HIGH-tail (HIC>11.5, diagnostic)

n_HIGH=7

### ablang1
- SHAM35: MAE_high=2.7416, signed=-2.7416
- ARO_ONLY35: MAE_high=2.3030, signed=-2.3030
- HYDRO_ONLY35: MAE_high=2.1931, signed=-2.1931
- FULL35: MAE_high=1.8098, signed=-1.8090
- Δ ARO_vs_SHAM: ΔMAE_high=-0.4386
- Δ HYDRO_vs_SHAM: ΔMAE_high=-0.5485
- Δ FULL_vs_SHAM: ΔMAE_high=-0.9317
### ablingua
- SHAM35: MAE_high=2.7018, signed=-2.7018
- ARO_ONLY35: MAE_high=2.1661, signed=-2.1368
- HYDRO_ONLY35: MAE_high=2.3984, signed=-2.3984
- FULL35: MAE_high=1.9374, signed=-1.8930
- Δ ARO_vs_SHAM: ΔMAE_high=-0.5357
- Δ HYDRO_vs_SHAM: ΔMAE_high=-0.3034
- Δ FULL_vs_SHAM: ΔMAE_high=-0.7643

## 14. Sample-level heterogeneity

See `HIC_SURFACE_ARO_HYDRO_HETEROGENEITY.csv`.
- FULL responder pattern agreement across AbLang1/AbLingua: **0.710**

## 15. Mechanistic classification

**`ARO_HYDRO_COMPLEMENTARY`**

Judgment summary (Primary/Shadow/Public/Private direction + effect size; no post-hoc numeric threshold):

- Both **ARO_ONLY** and **HYDRO_ONLY** improve over SHAM on Public/Private/Test in **both** representations.
- **FULL** remains better than either single block on Public/Private/Test in both representations (incremental conditional effects present).
- Internal Primary is noisier (AbLang1 ARO_ONLY Primary ΔCV_P > 0 vs SHAM), but Shadow and external streams align with complementary value.
- Neither block alone recovers the FULL external gain; effect sizes do not collapse to a single-block story.
- Feature-count 19 vs 16 is **not** used as an explanation.

Raw classify notes: aro_vs_sham_improve=True mean_primary=+0.0019; hydro_vs_sham_improve=True mean_primary=-0.0133; full_vs_sham_improve=True mean_primary=-0.0319; aro_incremental_on_hydro=True mean_primary=-0.0187; hydro_incremental_on_aro=True mean_primary=-0.0339

HIGH-tail diagnostic: both blocks reduce MAE_high vs SHAM; FULL reduces further in both reps → HIGH-tail rescue is **not** ARO-exclusive.

## 16. Exposed-aromatic hypothesis status

PARTIAL — aromatics informative but not sole; HYDRO also contributes. Avoid aromatics-only causal story.

ARO19 win is **not** claimed. Do **not** assert that a specific exposed-aromatic feature is causal.

## 17. Next experiment recommendation

Because classification is complementary (not ARO-dominant), do **not** immediately collapse to ARO19-internal family search alone.

Preferred next (still block/family level; **no** 35-way feature search):

1. Modest paired ARO-family and HYDRO-family probes, or
2. HIGH-tail–focused localization of which surface descriptors move the n=7 high-HIC errors.

## Stop

This phase stops at ARO19 vs HYDRO16 **block** conclusion.

## Stop

This phase stops at ARO19 vs HYDRO16 **block** conclusion. No 35-way feature search.

