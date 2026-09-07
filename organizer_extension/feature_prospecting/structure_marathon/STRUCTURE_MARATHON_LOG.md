# Structure Marathon Log

## 2026-09-06T00:26:51Z — campaign start

- HEAD: `3e832a69ac4387bf68f67263ea572bd536f91d5c`
- GPU0 free ~24 GB; GPU1 free ~11 GB; RAM avail ~58 GB
- Quarantined orphan `ADI-47317_prepared.pdb` → `fennix_fab_context/cache/quarantine/`
- PLAN_LOCK written before target scoring
- Next: FeNNix pilot 12 B/C/M (no TmApp) + inventory + P0 feature extract

## S2/S3/S4 complete
- S2 disagreement 324
- S3/S4 graphs 324
- S1 restarted after antibody_id index fix
## 2026-09-06T01:40:11Z
- M2 ESM-IF1: COMPLETE 324/324 SUCCESS (CPU; CUDA device mismatch avoided)
- M3 SaProt 35M: COMPLETE 324/324 SUCCESS (AA+3Di Foldseek)
- T1 SPURS: DEFERRED_TECHNICAL
- M4 ProSST: DEFERRED_TECHNICAL
- T2 ThermoMPNN: DEFERRED_TECHNICAL
- G1 GearNet: SKIP_DEFERRED
- Score signals so far: no clear INC_vs_INCUMBENT breakthrough; S1_PATCH_NEIGHBORS weakest positive TmApp delta with CI crossing 0

## 2026-09-06T07:08:54Z resume after power loss
- Curvature audit completed: Spearman=0.984, median NAD=0.019 → EQUIVALENT
- Protocol freeze: reuse CUDA preps; remaining on CPU batches
- CPU remaining prep started (210 Abs, batch 15)
