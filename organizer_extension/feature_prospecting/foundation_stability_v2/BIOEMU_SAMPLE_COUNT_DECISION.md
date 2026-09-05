# BioEmu sample count decision

**SPEC_FROZEN_BEFORE_TARGET_SCORING**

Selected N = **16**

Rule: smallest of {16,32} with median Spearman vs N=64 ≥ 0.95 and median NAD ≤ 0.10; else 64.

See `BIOEMU_SAMPLE_CONVERGENCE.csv`.

## Runtime notes (target-blind)

- Model: `bioemu-v1.2` via package bioemu 1.4.1
- MSA: local Boltz unpaired H/L CSVs converted to a3m, depth capped at 256 (frozen before TmApp scoring)
- Convergence Abs: 12 target-blind (length quartile × germline)
- All 24 VH/VL chains reached ≥64 valid samples before decision

