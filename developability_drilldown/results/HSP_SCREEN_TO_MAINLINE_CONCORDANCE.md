# HSP screen → mainline DL concordance

Screening used fixed Ridge/SVR on B3 bundles (source commit 210a270d).

## P1 BM-R5 (`HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0`)

- Screen alone Ridge TEST_mean: 0.4829
- Screen SURFACE+desc Ridge TEST_mean: 0.5112
- DL Scratch alone EXP-H102 TEST_mean: 0.4696 (vs H071 0.5017)
- DL Scratch SURFACE+HSP EXP-H103 Δ vs H090: -0.0064
- DL ESM2 SURFACE+HSP EXP-H109 Δ vs H086: -0.0292
- Mainline class: **MAINLINE_SUPPORTED_STRONG** / BACKBONE_ROBUST

## P2 FP-R5 (`HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0`)

- Screen alone Ridge TEST_mean: 0.4885
- Screen SURFACE+desc Ridge TEST_mean: 0.5250
- DL Scratch alone EXP-H104 TEST_mean: 0.4989 (vs H071 0.5017)
- DL Scratch SURFACE+HSP EXP-H105 Δ vs H090: +0.0018
- DL ESM2 SURFACE+HSP EXP-H111 Δ vs H086: -0.0040
- Mainline class: **NOT_CONFIRMED** / NOT_SUPPORTED

## P3 EIS-R8 (`HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0`)

- Screen alone Ridge TEST_mean: 0.5328
- Screen SURFACE+desc Ridge TEST_mean: 0.5095
- DL Scratch alone EXP-H106 TEST_mean: 0.4767 (vs H071 0.5017)
- DL Scratch SURFACE+HSP EXP-H107 Δ vs H090: -0.0239
- DL ESM2 SURFACE+HSP EXP-H113 Δ vs H086: -0.0087
- Mainline class: **MAINLINE_SUPPORTED_STRONG** / BACKBONE_ROBUST

## Summary questions

1. P1 BM-R5 replicate? MAINLINE_SUPPORTED_STRONG
2. P2 FP-R5 replicate? NOT_CONFIRMED
3. P3 EIS-R8 SURFACE complementarity? MAINLINE_SUPPORTED_STRONG
4. Cheap screen vs DL: directional agreement is assessed above; magnitude need not match.

## Final concordance verdict

- P1 BM-R5: screen PROMOTE_STRONG → mainline **SUPPORTED_STRONG** (replicates).
- P2 FP-R5: screen INTERESTING → mainline **NOT_CONFIRMED** for SURFACE increment.
- P3 EIS-R8: screen INTERESTING (SURFACE complement) → mainline **SUPPORTED_STRONG** (complements SURFACE).
- Fixed-feature screening predicted DL usefulness directionally for P1/P3; FP over-called alone/additive value.
