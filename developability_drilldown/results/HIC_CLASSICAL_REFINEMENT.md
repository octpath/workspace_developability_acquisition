# HIC classical feature refinement

## Summary
- New experiments: 20 (EXP-H034..EXP-H053)
- Best CV-worst: **EXP-H047** (SVR_HIC_HIC_HYDRO_TITRATION_ESM2_RASA_CDR3_SVR, SVR)
- Feature set: FS_HIC_HYDRO_TITRATION+FB_ESM2_RASA_CDR3
- CV: P=0.4529 S=0.4440 W=0.4529
- Test: Public=0.4076 Private=0.4474

## Answers
1. H-CDR/CDR3 ESM2 blocks (FB_ESM2_CDR3, FB_ESM2_RASA_CDR3) helped when added to hydro/surface bases.
2. RASA-weighted ESM2 pooling improved Stage-A scores vs global ESM2 alone.
3. Aromatic RASA summary block (FB_ARO_RASA_SUM) tested; top paths dominated by ESM2+RASA/CDR3 combos.
4. Region-specific aromatic exposure did not outperform ESM2 RASA/CDR3 path on CV-worst.
5. Continuous surface / hydro bases remained complementary; best = Hydro + ESM2 RASA CDR3 with SVR readout.
6. Ablation: removing FB_ESM2_RASA_CDR3 from hydro path reverts to base CV-worst.
7. SVR best CV-worst on top HIC set; Ridge/Lasso/ENet slightly worse; XGB similar/worse on CV-worst.

## Prior classical baseline
- EXP-H021 XGB_HIC_CONTINUOUS_SURFACE CV-worst ≈ 0.446

## Bootstrap vs EXP-H021
target      new baseline  mean_mae_improvement  ci95_low  ci95_high
   HIC EXP-H047 EXP-H021             -0.007125 -0.030813   0.015731
