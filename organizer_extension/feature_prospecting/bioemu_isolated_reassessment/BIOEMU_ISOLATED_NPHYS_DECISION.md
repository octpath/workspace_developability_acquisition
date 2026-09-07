# BioEmu isolated PHYSICAL-frame N decision

**SPEC_FROZEN_BEFORE_TARGET_SCORING**

- **VH**: Nphys = **8**
  - Nphys=4: median MC_noise_ratio (major) = 0.336
  - Nphys=8: median MC_noise_ratio (major) = 0.231
  - Nphys=16: median MC_noise_ratio (major) = 0.140
  - Nphys=32: median MC_noise_ratio (major) = 0.092
- **VL**: Nphys = **8**
  - Nphys=4: median MC_noise_ratio (major) = 0.289
  - Nphys=8: median MC_noise_ratio (major) = 0.208
  - Nphys=16: median MC_noise_ratio (major) = 0.141
  - Nphys=32: median MC_noise_ratio (major) = 0.089

**Cohort frozen Nphys (max of VH/VL) = 8**

Rule: smallest of {8,16,32} with median Spearman vs N64 ≥ 0.95 and median NAD ≤ 0.10 on major families; else 64.

Seeds: deterministic MC subsets (20 draws) from first 64 physical frames; seed in SPEC.

See `BIOEMU_ISOLATED_PHYSICAL_CONVERGENCE.csv`, `BIOEMU_ISOLATED_MC_NOISE.csv`.

## Contact-family note (target-blind)

Major families met Spearman≥0.95 and NAD≤10% at **Nphys=8**.

Contact-family median MC_noise_ratio remains elevated vs geometry/shape:

| chain | Nphys=4 | Nphys=8 | Nphys=16 | Nphys=32 |
|-------|---------|---------|----------|----------|
| VH    | 0.757   | 0.485   | 0.304    | 0.207    |
| VL    | 0.453   | 0.298   | 0.198    | 0.121    |

Nphys was **not** increased to chase contact TmApp performance (forbidden). Contact signal at Nphys=8 is reported with this residual MC uncertainty.

## Post-scoring note

Full-cohort TmApp reassess completed after this freeze.
Final verdict: **BIOEMU_ISOLATED_CONFIRMED_NO_INCREMENT**
(see `BIOEMU_ISOLATED_REASSESS_REPORT_JA.md`).
