# SOURCE SAP / SCM Feature Research Report

**Track:** `feature_research/hic_sap_scm_source/`  
**Prior blocked commit:** `83dfed7d` (auditable)  
**Mainline:** EXP-H114 **not run**

## Source Stage-1 (frozen before extensions)

See `SOURCE24_STAGE1_INTERPRETATION.md`.

| Block | VAL_mean alone | SURFACEΔ | Verdict |
|-------|----------------|----------|---------|
| H094 GLOBAL3 | 0.532 | — | reference |
| SAP24 | 0.598 | −0.065 | **NOT_SUPPORTED** |
| SCM24 | 0.597 | −0.053 | **NOT_SUPPORTED** |
| SAP24+SCM24 | 0.658 | −0.118 | **NOT_SUPPORTED** (not complementary) |

SURFACE alone VAL_mean ≈ 0.515. Positive SURFACEΔ would mean improvement; observed deltas are negative (SURFACE+block worse).

Ridge overfits high-D blocks when fused with SURFACE; SVR is flatter but does not rescue SOURCE24 over SURFACE.

## Extensions (VAL alone)

| Block | VAL_mean | vs source |
|-------|----------|-----------|
| SAP30 | 0.595 | ≈ SAP24 (GLOBAL6 negligible) |
| SAP50 | 0.580 | slight alone gain; SURFACE fusion still worse |
| SCM30 | 0.607 | no gain |
| SCM50 | 0.637 | worse |

**GLOBAL6:** no material incremental value.  
**EXTRA20:** small alone improvement for SAP50 only; not justified as a block (no SURFACE increment; still worse than H094).

## Stage-2 TEST (shortlist; mean Ridge/SVR)

| Block | alone TEST_mean | SURFACE+ TEST_mean |
|-------|-----------------|--------------------|
| SAP24 | 0.596 | 0.570 |
| SCM24 | 0.606 | 0.555 |
| SAP24+SCM24 | 0.660 | 0.619 |
| SAP50 | 0.589 | 0.593 |
| SCM30 | 0.617 | 0.566 |
| SAP30+SCM30 | 0.667 | 0.632 |

Paired bootstrap (SURFACE+SAP24 vs SURFACE): Ridge ΔMAE ≈ +0.12 (harmful); SVR CI includes 0.

## Answers to scientific questions

1. **Does SAP24 rescue H094?** No — H094 GLOBAL3 remains better alone.
2. **Does multi-region / multi-scale / top5 help KD/Tien?** Not under fixed Ridge/SVR; R5↔R10 are only moderately correlated (mean ρ≈0.4) so scales differ, but predictive MAE does not improve vs GLOBAL3.
3. **Is SCM24 predictive?** Weak alone; worse than SURFACE; not supported for promotion.
4. **SAP24 ∩ SCM24 complementary?** No — combined worse (REDUNDANT/harmful under Ridge).
5–7. **SURFACE increments?** No — SURFACE+SOURCE blocks worsen MAE (esp. Ridge).
8. **ALL_FV GLOBAL6?** Negligible.
9–10. **STD / TOP5_SHARE_POSITIVE?** Slight alone SAP50 effect; largely not useful; no near-dup |ρ|≥0.98 vs source but no promotion case.
11. **More robust than BM-R5 / EIS-R8 isolated winners?** Methodologically cleaner (fixed multi-region/scale, no radius cherry-pick) but **weaker predictive evidence** than those HSP late-fusion confirmations.
12. **Suitable for EXP-H114+?** None.
13. **Residue-level justified?** **No.**

## Promotion

No SAP / SCM / SAP+SCM block recommended. Residue-level modeling not justified.

**STOP — do not run EXP-H114.**
