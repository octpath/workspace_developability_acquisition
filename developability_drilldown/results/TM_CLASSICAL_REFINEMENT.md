# TmApp classical feature refinement

## Summary
- New experiments: 20 (EXP-T045..EXP-T064)
- Best CV-worst: **EXP-T045** (LIN_TM_TM_BIOEMU_MPNN_AL_CDR3_AL2_RASA_CDR_AL_CDR_ALL_RIDGE, RIDGE)
- Feature set: FS_TM_BIOEMU_MPNN+FB_AL_CDR3+FB_AL2_RASA_CDR+FB_AL_CDR_ALL
- CV: P=2.7060 S=2.7289 W=2.7289
- Test: Public=3.0197 Private=3.2736

## Answers
1. Region-aware AbLang2 pooling (FR/CDR split, CDR blocks) improved over global-only when added to strong bases (Stage A/B).
2. CDR emphasis (CDRW2/4, CDR_ALL) showed mixed signal; best path used AbLingua CDR3 + AbLang2 RASA/CDR blocks on BioEmu base.
3. CDR3-specific blocks (FB_AL_CDR3, FB_AL2_CDR3) contributed in forward paths from AbLingua CDR3 and BioEmu bases.
4. RASA-weighted AbLang2 pooling (FB_AL2_RASA_P1/CDR) improved Stage-A add-to-base scores vs unweighted alone.
5. Fixed AbLang2+AbLingua concat tested; did not beat top combined base paths on CV-worst.
6. BioEmu/MPNN base remained essential anchor; best combo = BioEmu+MPNN + AbLingua CDR3 + AbLang2 RASA/CDR + AbLingua CDR_ALL.
7. Ablation: dropping FB_AL_CDR_ALL from top BioEmu path increased CV-worst (block contributes).
8. Nonlinear readout (SVR/XGB) did not beat Ridge on top Tm feature set for CV-worst.

## Prior classical baseline
- EXP-T001 LIN_TM_ABLINGUA_CDR3_RIDGE CV-worst ≈ 2.785

## Bootstrap vs EXP-T001 (primary OOF MAE improvement)
target      new baseline  mean_mae_improvement  ci95_low  ci95_high
 TmApp EXP-T045 EXP-T001              0.026058 -0.037283   0.087513
