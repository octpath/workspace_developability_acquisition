# Promotion recommendation — HSP spatial hydrophobicity → mainline EXP-H102+

**Do not run EXP-H102 in this task.** This document only recommends descriptor families.

## Classification of shortlist

| Family | Class | Rationale |
|--------|-------|-----------|
| `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0` | **PROMOTE_STRONG** | Best alone TEST (~0.483); beats STATIC_SAP_KD_v1-like; BM+closest+R5 |
| `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0` | **PROMOTE_INTERESTING** | Strong alone TEST/external; side-chain absolute SASA; FP scale |
| `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0` | **PROMOTE_INTERESTING** | Best Ridge incremental vs SURFACE (bootstrap ΔMAE CI excludes 0 on Primary) |
| `HSP_FP_*` other shortlist variants | REDUNDANT-ish / interesting | Related FP family; keep one FP representative |
| `LIT2_STATIC_SAP_BM_R5p0` | NOT_CONFIRMED | Weak alone TEST (~0.535); does not beat SURFACE |
| `LIT3_PSH_KD` | NOT_CONFIRMED | Weak alone / no SURFACE gain |
| `LIT4_POS_SASA_*` | NOT_CONFIRMED / weak | Alone weak; KD positive-SASA external mixed, not robust |

## Recommend ≤3 for future EXP-H102+ late fusion

1. **FS_HSP_BM_CLOSEST_R5_B3** — `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0` (B3: MAX/MEAN/SUM)
2. **FS_HSP_FP_SCABS_CLOSEST_R5_B3** — `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0`
3. **FS_HSP_EIS_SCABS_CENTROID_R8_B3** — `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0` (complementarity candidate)

## Does any REPLACE SURFACE?

**No.** SURFACE (esp. SVR) remains strong; promotion is as **additive physical aux**, not replacement.

## Residue-level injection?

**Not yet as default.** Antibody-level BM/FP/EIS families now show signal under fixed Ridge/SVR, but mainline DL confirmation is required first. Residue-level injection is a **conditional follow-up** after EXP-H102+ late-fusion of the three promoted bundles.

## Mainline next code

`EXP-H102` — **DO NOT RUN here.**
