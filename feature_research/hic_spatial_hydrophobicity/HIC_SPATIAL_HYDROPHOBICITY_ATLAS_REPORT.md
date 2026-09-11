# HIC Spatial Hydrophobicity Atlas — Master Report

**Namespace:** `feature_research/hic_spatial_hydrophobicity`  
**IDs:** `HSP-F****` (not EXP-H)  
**Mainline:** EXP-H102 unused; H054–H101 unmodified; no Transformer training.

## Atlas size

- Antibodies: 324 (Fv ESMFold)
- Generic families: 504 (7 scales × 2 transforms × 3 exposures × 2 neighborhoods × 6 radii)
- Antibody features: ~21,229
- Residue long-form rows: 504 × ~75k
- Redundancy clusters (MEAN/LIT subset): 483
- Jain HIC scale: **UNAVAILABLE**

## Key result vs H094

`STATIC_SAP_KD_v1`-like cell (`KD`+`MINMAX`+`TOTAL_RASA`+`CENTROID`+`R5`+B3) VAL_mean≈0.54 (Ridge) — mediocre.  
Best cell: **BM RAW + TOTAL_RASA + CLOSEST_SC + R5** VAL≈0.485 / TEST≈0.483.

## Factor trends (VAL alone, B3)

| Factor | Prefer |
|--------|--------|
| Scale | **FP, BM** > MIY > WW > KD |
| Transform | **MINMAX** slightly better than RAW overall (but BM RAW wins top cell) |
| Exposure | **SIDECHAIN_SASA_ABS** slightly better |
| Neighborhood | **CLOSEST_SC** ≥ CENTROID (clearer for SVR) |
| Radius | **~5–7.5 Å** better than 4 Å |
| Aggregation | **B3 / BR** better than BH / BALL |

## Stage-2 shortlist (alone Ridge TEST_mean)

1. BM closest R5 — **0.483**
2. FP scAbs closest R5 — **0.489**
3. FP scAbs centroid R7.5 — 0.501
4. Literature SAP/PSH/pos-SASA — 0.52–0.57 (weaker)

## SURFACE complementarity

Ridge bootstrap (Primary): EIS scAbs centroid R8 and BM closest R5 show **negative ΔMAE** vs SURFACE alone (CI excludes 0).  
Does **not** justify replacing SURFACE; supports **additive** testing.

## Promotion (≤3)

See `PROMOTION_RECOMMENDATION.md`.

## Why STATIC_SAP_KD_v1 failed

Combination of **KD + min-max + centroid + total rSASA + R5 global3** is far from the best cell. Failures were **definitional**, not proof that all SAP-like descriptors lack HIC signal.
