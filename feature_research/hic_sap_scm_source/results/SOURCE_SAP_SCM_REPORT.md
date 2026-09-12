# SOURCE SAP / SCM Feature Research Report

**Track:** `feature_research/hic_sap_scm_source/`  
**Date:** 2026-09-12  
**Mainline:** EXP-H114 **not run**; H054–H113 / HSP results **not modified**.

## Executive verdict

| Question | Answer |
|----------|--------|
| SOURCE_SAP24 | **BLOCKED_SOURCE_UNRESOLVED** (`positive_sum_mean`) |
| SOURCE_SCM24 | **BLOCKED_SOURCE_UNRESOLVED** (charge semantics) |
| SOURCE_SAP24+SCM24 | **unsupported** (both blocked) |
| GLOBAL6 / EXTRA20 | **not evaluated** (depends on SOURCE24) |
| Residue-level modeling | **not justified** |
| Mainline promotion | **none** |

## Why generation stopped

Section 3 (strict fidelity gate) forbids inventing `positive_sum_mean` and forbids labeling any 24-D block as `SOURCE_SAP24` while it is unresolved. SCM charge is independently unresolved.

Closest repo/literature near-misses (explicitly **not** used as source formulas):

- Organizer STATIC-SAP: Black–Mould × CA × R5/R10; stats include `mean_positive` / `sum_positive` / `top5_mean` (18-D).
- STATIC_SAP_KD (H094): KD min-max × centroid; Ab stats MAX/MEAN/SUM (not positive_sum_mean).
- DeepSP: MD SAP/SCM domain sums of positive / negative scores (30-D region set ≠ VH/VL/CDR/FR × 3 stats).

## What was confirmed without inventing

- Shrake–Rupley: probe **1.4 Å**, `n_points` **100** (repo SAP lineage).
- KD for repo KD-SAP: **min-max** over 20 AA.
- RASA: SASA / Tien MaxASA, clip [0,1].
- Neighborhood for KD-SAP lineage: side-chain centroid, self included.
- User-intended layout: 4 regions × {5,10} × {MAX, TOP5_MEAN, POSITIVE_SUM_MEAN}.

## Stage-1 / Stage-2

Skipped. No VAL/TEST numbers. Shortlist freeze records `BLOCKED_NO_STAGE1`.

## Unblocking requirements

1. Authoritative equation for `positive_sum_mean` (numerator, denominator, empty-region rule).  
2. Authoritative SCM per-residue property + sign convention.  
3. Confirm KD min-max vs Black–Mould and centroid vs CA if the external source differs from repo KD-SAP.

After unblocking: implement generic aggregation engine → SOURCE24 → GLOBAL6 → EXTRA20 QC → Stage-1/2 as specified — still without running EXP-H114 until a separate promotion decision.
