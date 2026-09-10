# TmApp Geometry Report (ARCH-6G)

Platform: `DL_FOLDLOCAL_COSINE_V3`. Δ = MAE(geom) − MAE(control); negative ⇒ geometry better.

| Geom | Control | Rep | Merge | Δ TEST_mean | Δ TEST_P | Δ TEST_S | Overall Δ |
|---|---|---|---|---:|---:|---:|---:|
| EXP-T105 | EXP-T084 | ABLINGUA | concat | 0.0328 | 0.0301 | 0.0355 | -0.0124 |

### Learned RBF weights — EXP-T105

Saved at `results/EXP-T105_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-T106 | EXP-T085 | ABLINGUA | mean | -0.1577 | -0.2481 | -0.0673 | 0.1448 |

### Learned RBF weights — EXP-T106

Saved at `results/EXP-T106_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-T107 | EXP-T099 | SCRATCH | concat | -0.0049 | -0.0137 | 0.0040 | -0.0316 |

### Learned RBF weights — EXP-T107

Saved at `results/EXP-T107_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-T108 | EXP-T100 | SCRATCH | mean | -0.1383 | -0.0795 | -0.1971 | 0.0027 |

### Learned RBF weights — EXP-T108

Saved at `results/EXP-T108_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-T118 | EXP-T116 | ABLANG2 | concat | -0.0365 | -0.0476 | -0.0253 | 0.0551 |

### Learned RBF weights — EXP-T118

Saved at `results/EXP-T118_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

| EXP-T119 | EXP-T117 | ABLANG2 | mean | 0.0346 | 0.1264 | -0.0573 | 0.0384 |

### Learned RBF weights — EXP-T119

Saved at `results/EXP-T119_GEOMETRY_WEIGHTS.csv` (per-fold × head × basis). Zero-init at start ⇒ ARCH-6 equivalent.

## D. Inference geometry-zero ablation

Source: `TM_HIC_GEOMETRY_ZERO_AGGREGATE.csv` / `TM_HIC_GEOMETRY_ZERO_DIAGNOSTIC.md`.

| code | Primary TEST_OOF Δ | Shadow TEST_OOF Δ |
|---|---:|---:|
| EXP-T105 | 2.83748e-05 | 8.28872e-06 |
| EXP-T106 | 9.4661e-06 | -4.70008e-05 |
| EXP-T107 | -0.000428777 | 1.70013e-05 |
| EXP-T108 | 7.38921e-05 | -2.34062e-05 |
| EXP-T118 | -3.85237e-05 | -0.000212846 |
| EXP-T119 | 0.000139966 | 8.20395e-05 |

## Diagnostics checklist

- A. matched no-geometry control (table above)
- B. TEST delta (table)
- C. paired bootstrap → `TM_HIC_PAIRED_BOOTSTRAP.csv`
- D. inference geometry-zero ablation: see section above (`TM_HIC_GEOMETRY_ZERO_*`)
- E–G. RBF coefficients / magnitude / distance peaks: see per-code `*_GEOMETRY_WEIGHTS.csv`
- H–I. Primary/Shadow + representation consistency: compare rows above
- J. HIC: Fv H/L geometry only (ESMFold Fv Cα); not full Ig

