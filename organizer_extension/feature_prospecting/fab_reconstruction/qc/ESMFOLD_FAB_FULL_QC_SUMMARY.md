# ESMFOLD FAB FULL QC SUMMARY

Pilot verdict: `PASS_FULL_COHORT`  
Completion: **324/324** successful length-matched structures (failures: **0**).

## Confidence / clash / interfaces

| Metric | Mean | Median | P10 | P90 |
|--------|------|--------|-----|-----|
| global_pLDDT | 86.6 | 86.7 | 84.7 | 88.3 |
| VH_pLDDT | 87.4 | 87.7 | 84.4 | 90.4 |
| VL_pLDDT | 88.3 | 88.8 | 84.9 | 90.8 |
| CH1_pLDDT | 83.8 | 84.0 | 82.0 | 85.0 |
| CL_pLDDT | 86.5 | 86.7 | 85.2 | 87.6 |
| CH1_CL_contacts | 39.9 | 39.0 | 32.0 | 47.0 |
| severe clashes | 0.02 | 0.0 | 0.0 | 0.0 |

## H–L Cys Sγ–Sγ distance (proximity ≠ covalent bond)

| Subset | median | q10 | q90 | max | <2.3Å | <2.6Å | <3.0Å | ≥3.0Å |
|--------|--------|-----|-----|-----|-------|-------|-------|-------|
| overall | 2.80 | 2.57 | 3.45 | 52.10 | 3.1% | 12.3% | 74.4% | 25.6% |
| kappa | 2.74 | 2.55 | 2.95 | 52.10 | 4.2% | 16.0% | 93.7% | 6.3% |
| lambda | 3.34 | 2.93 | 3.64 | 3.91 | 0.0% | 2.3% | 20.9% | 79.1% |

See `HL_DISULFIDE_DISTANCE_AUDIT.md`. κ is typically closer than λ (different expected topology). Flag for FeNNix post-relax QC.

## Fv vs Fab (full 324)

Combined Fv CA RMSD median **0.58 Å** (q90 1.09, max 24.72).  
Orientation COM median **0.50 Å**.  
See `comparisons/FV_VS_FAB_STRUCTURE_SUMMARY.md`.
