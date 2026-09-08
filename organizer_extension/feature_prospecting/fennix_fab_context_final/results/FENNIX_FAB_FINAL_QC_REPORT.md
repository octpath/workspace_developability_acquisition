# FeNNix Fab — Final target-blind QC

UTC: 2026-09-08T10:31:12.741155+00:00

## Coverage
- expected IDs: 323 (exclude ADI-47265)
- feature rows: 323
- duplicate IDs: 0
- missing IDs vs cohort: 0
- missing feature cells: 0
- ±inf cells: 0
- zero-variance features: 6

## B/C/M integrity
- all accepted: True
- r1 npz ok: True

## Technical associations (flags |ρ|>0.85 vs n_atoms/n_sites)
- none above |ρ|=0.85 among sampled features

## Overall QC verdict

**QC_PASS_WITH_LIMITATIONS**

### Conditions A/B/C/M (SPEC)
- **A**: isolated Fv ESMFold (FeNNix-v2)
- **B**: Fab-geom Fv (from Fab) + R1 relax
- **C**: full prepared Fab + R1
- **M**: matched Fv coords from C (constants removed), no re-relax
- **DELTA_GEOM** = B−A; **DELTA_ENV** = C_var−M_var (environment, not absolute energy across systems)
