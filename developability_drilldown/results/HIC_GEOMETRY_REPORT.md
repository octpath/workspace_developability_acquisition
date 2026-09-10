# HIC Geometry Report (ARCH-6G)

Platform: `DL_FOLDLOCAL_COSINE_V3`. Δ = MAE(geom) − MAE(control); negative ⇒ geometry better.

| Geom | Control | Rep | Merge | Δ TEST_mean | Δ TEST_P | Δ TEST_S | Overall Δ |
|---|---|---|---|---:|---:|---:|---:|
| EXP-H064 | EXP-H062 | ESM2 | concat | 0.0051 | 0.0079 | 0.0023 | 0.0023 |

### Learned RBF weights — EXP-H064

Saved at `results/EXP-H064_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-H065 | EXP-H063 | ESM2 | mean | -0.0043 | 0.0115 | -0.0201 | -0.0081 |

### Learned RBF weights — EXP-H065

Saved at `results/EXP-H065_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-H078 | EXP-H076 | SCRATCH | concat | -0.0461 | -0.0323 | -0.0599 | -0.0060 |

### Learned RBF weights — EXP-H078

Saved at `results/EXP-H078_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-H079 | EXP-H077 | SCRATCH | mean | 0.0156 | 0.0353 | -0.0041 | 0.0284 |

### Learned RBF weights — EXP-H079

Saved at `results/EXP-H079_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

## D. Inference geometry-zero ablation

Source: `TM_HIC_GEOMETRY_ZERO_AGGREGATE.csv` / `TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.md`.

| code | Primary TEST_OOF Δ | Shadow TEST_OOF Δ |
|---|---:|---:|
| EXP-H064 | -1.31454e-05 | -6.06937e-06 |
| EXP-H065 | -8.59484e-07 | -7.87076e-06 |
| EXP-H078 | 5.09921e-05 | 4.7407e-05 |
| EXP-H079 | -3.75291e-05 | 6.29131e-05 |

## Diagnostics checklist

- A. matched no-geometry control (table above)
- B. TEST delta (table)
- C. paired bootstrap → `TM_HIC_PAIRED_BOOTSTRAP.csv`
- D. inference geometry-zero ablation: see section above (`TM_HIC_GEOMETRY_ZERO_*`)
- E–G. RBF coefficients / magnitude / distance peaks: see per-code `*_GEOMETRY_WEIGHTS.csv`
- H–I. Primary/Shadow + representation consistency: compare rows above
- J. HIC: Fv H/L geometry only (ESMFold Fv Cα); not full Ig

