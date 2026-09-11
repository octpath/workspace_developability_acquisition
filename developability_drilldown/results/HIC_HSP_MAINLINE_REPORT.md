# HIC HSP Mainline Report (H102–H113)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- source HSP commit: `210a270d`
- prereg commit: `a167830c`
- analyze git HEAD: `a167830c89e2aff1f8a7af672dccc002b7605130`
- freeze: `H102_H113_PRE_EXTERNAL_FREEZE.yaml`

## Master table

    Code Backbone                                     Aux                                       HSP_family  Aux_dim   TEST_P   TEST_S  TEST_mean  TEST_worst      Pub     Priv  Overall
EXP-H071 EXP-H071                                    NONE                                              NaN        0 0.528430 0.474947   0.501688    0.528430 0.479571 0.480027 0.479799
EXP-H090 EXP-H071                              F1_SURFACE                                              NaN       35 0.457571 0.485454   0.471512    0.485454 0.405876 0.412875 0.409375
EXP-H093 EXP-H071                         F4_H047_AUX_ALL                                              NaN      200 0.498382 0.505869   0.502126    0.505869 0.406960 0.421483 0.414222
EXP-H061 EXP-H061                                    NONE                                              NaN        0 0.533068 0.480523   0.506795    0.533068 0.481635 0.461520 0.471577
EXP-H086 EXP-H061                              F1_SURFACE                                              NaN       35 0.471437 0.501216   0.486326    0.501216 0.401757 0.410931 0.406344
EXP-H089 EXP-H061                         F4_H047_AUX_ALL                                              NaN      200 0.471707 0.467812   0.469759    0.471707 0.401818 0.439192 0.420505
EXP-H102 EXP-H071               FS_HIC_HSP_BM_R5_PROMOTED       HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0        3 0.477691 0.461575   0.469633    0.477691 0.475599 0.426610 0.451104
EXP-H103 EXP-H071  FS_HIC_SURFACE_PLUS_HSP_BM_R5_PROMOTED       HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0       38 0.440935 0.489280   0.465107    0.489280 0.401873 0.413290 0.407581
EXP-H104 EXP-H071               FS_HIC_HSP_FP_R5_PROMOTED HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0        3 0.503093 0.494627   0.498860    0.503093 0.429540 0.446530 0.438035
EXP-H105 EXP-H071  FS_HIC_SURFACE_PLUS_HSP_FP_R5_PROMOTED HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0       38 0.461215 0.485465   0.473340    0.485465 0.403584 0.414375 0.408980
EXP-H106 EXP-H071              FS_HIC_HSP_EIS_R8_PROMOTED     HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0        3 0.472498 0.480827   0.476663    0.480827 0.489533 0.446706 0.468120
EXP-H107 EXP-H071 FS_HIC_SURFACE_PLUS_HSP_EIS_R8_PROMOTED     HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0       38 0.451192 0.444067   0.447629    0.451192 0.391341 0.395898 0.393619
EXP-H108 EXP-H061               FS_HIC_HSP_BM_R5_PROMOTED       HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0        3 0.459880 0.453339   0.456609    0.459880 0.451480 0.426587 0.439034
EXP-H109 EXP-H061  FS_HIC_SURFACE_PLUS_HSP_BM_R5_PROMOTED       HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0       38 0.460091 0.454226   0.457159    0.460091 0.399743 0.402805 0.401274
EXP-H110 EXP-H061               FS_HIC_HSP_FP_R5_PROMOTED HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0        3 0.459349 0.470324   0.464836    0.470324 0.408184 0.426015 0.417099
EXP-H111 EXP-H061  FS_HIC_SURFACE_PLUS_HSP_FP_R5_PROMOTED HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0       38 0.463618 0.500994   0.482306    0.500994 0.399187 0.414017 0.406602
EXP-H112 EXP-H061              FS_HIC_HSP_EIS_R8_PROMOTED     HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0        3 0.484634 0.474815   0.479725    0.484634 0.461959 0.437573 0.449766
EXP-H113 EXP-H061 FS_HIC_SURFACE_PLUS_HSP_EIS_R8_PROMOTED     HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0       38 0.461833 0.493370   0.477601    0.493370 0.390358 0.404945 0.397652

## Central increment table (SURFACE+HSP − SURFACE; negative better)

Family  Scratch_d_over_SURFACE     Scratch_boot_primary  Scratch_perm_HSP_primary  ESM2_d_over_SURFACE        ESM2_boot_primary  ESM2_perm_HSP_primary                   Verdict        Backbone
 BM-R5               -0.006405 -0.0166 [-0.0635,0.0305]                  0.062200            -0.029168 -0.0113 [-0.0509,0.0297]               0.044291 MAINLINE_SUPPORTED_STRONG BACKBONE_ROBUST
 FP-R5                0.001828  0.0036 [-0.0393,0.0493]                  0.017042            -0.004021 -0.0078 [-0.0407,0.0260]               0.028989             NOT_CONFIRMED   NOT_SUPPORTED
EIS-R8               -0.023883 -0.0064 [-0.0411,0.0274]                  0.023737            -0.008725 -0.0096 [-0.0490,0.0299]               0.028927 MAINLINE_SUPPORTED_STRONG BACKBONE_ROBUST

## Scientific answers

1. **BM-R5**: Yes — alone improves bases; SURFACE+HSP improves Scratch (−0.006) and ESM2 (−0.029); HSP permutation ΔMAE>0 on both → **MAINLINE_SUPPORTED_STRONG**.
2. **FP-R5**: Alone modest; SURFACE increment incoherent (Scratch +0.002 / ESM2 −0.004) → **NOT_CONFIRMED** as additive over SURFACE.
3. **EIS-R8**: Strong Scratch SURFACE complement (TEST_mean 0.448 vs H090 0.472); ESM2 also negative Δ; HSP perm>0 → **MAINLINE_SUPPORTED_STRONG**.
4. Strongest HSP-alone Scratch: BM-R5 H102 TEST_mean=0.4696 (vs H071 0.5017); ESM2 alone BM H108=0.4566.
5. **Replace SURFACE?** No. HSP-alone can beat base backbones and sometimes approach SURFACE, but H090/H086 remain competitive; promotion is additive.
6. **Beyond SURFACE:** BM-R5 and EIS-R8 (esp. Scratch EIS H107).
7. **Reliance:** Block permutation shows positive ΔMAE when HSP is shuffled for SURFACE+HSP runs of P1/P3 (and non-trivial for P2 alone).
8. **Cross-backbone:** P1 and P3 BACKBONE_ROBUST; P2 NOT_SUPPORTED.
9. **Screen→DL:** P1 alone signal and P3 SURFACE complementarity replicated; FP less so. Cheap screen is useful but not perfect.
10. **F4+HSP later?** Justified for **BM-R5 and EIS-R8** only (not FP).
11. **Residue-level HSP later?** Conditionally justified for BM-R5 / EIS-R8 after Ab-level confirmation; not for FP.
12. Supports that **explicit exposed spatial hydrophobicity descriptors** carry HIC-associated predictive information under this assay — not that HSP *causes* retention.

## F4 context (not trained with HSP)

- H093 F4 Scratch TEST_mean=0.5021; H089 F4 ESM2=0.4698
- Best SURFACE+HSP Scratch H107=0.4476 beats H093; ESM2 H109=0.4572 vs H089=0.4698

## Guardrails

- Do not call BM-R5 the true SAP.
- Bootstrap 95% CIs for Δ vs SURFACE often include 0; rely jointly on TEST_mean + permutation.
