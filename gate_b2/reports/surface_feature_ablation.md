# Surface feature ablation (S0–S4)

Nested feature sets on **identical** structures, frozen splits, and regressor protocol.

| Set | Contents |
|---|---|
| S0 | Absolute SASA summaries |
| S1 | S0 + RASA summaries |
| S2 | S0 + absolute surface physicochemistry |
| S3 | S0 + RASA-weighted physicochemistry |
| S4 | S0 + RASA summary + absolute + RASA-weighted physchem |

## Mean CV Spearman deltas vs S0 (across canonical + shadows)

```
target          HIC      TmApp
s1_minus_s0  -0.014     +0.040
s2_minus_s0  +0.011     +0.066
s3_minus_s0  -0.104     -0.037
s4_minus_s0  -0.011     +0.060
patch_best    0.160      0.196   (absolute CV of PATCH family, not a delta)
```

## Interpretation

- **HIC:** RASA summary alone does **not** help; absolute surface physchem helps slightly; RASA-weighted physchem alone hurts. Prefer absolute exposure × chemistry annotation over RASA normalization.
- **TmApp:** RASA and surface physchem help modestly in isolation, but structure families remain well below PLM.
- **Patches:** Required modest complexity implemented (RASA≥0.20, CA≤8Å). Useful as part of ALL stacks for HIC; weak alone.

SASA freeze: probe_radius=1.4, n_points=100, MaxASA=Tien2013 (see `cache/structure_features/sasa_params.json`).
