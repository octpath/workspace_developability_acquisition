# HIC Residue-Level F1_SURFACE Fusion Report (H128–H133)
**Status:** COMPLETE — schema freeze → prereg → train → diagnostics → external → freeze
**Case:** **CASE G** — no residue-level model improves even the sequence baseline in a way that justifies escalating complexity; antibody-level F1 aggregation remains superior.
**Cross-attention justified?** **NO**
**Next HIC:** `EXP-H134` (**DO NOT RUN**)

## Schema (frozen)
- Feature set: `FS_HIC_RESIDUE_F1_SURFACE_COMPACT10`
- **p = 10** channels: `rasa, sasa, is_aromatic, is_exposed, is_strongly_exposed, hydro_area_sum, hydro_H_awmean, hydro_pos_H_area, hydro_neg_H_area, availability_hydro`
- Provenance: PASS-EXACT reconstruction; compact schema is **not** claimed ≡ F1_SURFACE35
- Docs: `results/H128_H133_RESIDUE_SURFACE_SCHEMA.{md,yaml}`

## Controls (not retrained)
| Code | Role | TEST_mean | Public | Private | Overall |
|------|------|----------:|-------:|--------:|--------:|
| EXP-H071 | Scratch seq | 0.501688 | 0.479571 | 0.480027 | 0.479799 |
| EXP-H090 | Scratch + Ab F1 | 0.471512 | 0.405876 | 0.412875 | 0.409375 |
| EXP-H061 | ESM2 seq | 0.506795 | 0.481635 | 0.461520 | 0.471577 |
| EXP-H086 | ESM2 + Ab F1 | 0.486326 | 0.401757 | 0.410931 | 0.406344 |

## Central table
| Backbone | Mode | Code | TEST_P | TEST_S | TEST_mean | Δ seq-base | Δ Ab-SURFACE | boot vs seq (P) | boot vs Ab (P) | zero-SURFACE ΔP | residue-shuffle ΔP | Public | Private | Overall | params |
|----------|------|------|-------:|-------:|----------:|-----------:|-------------:|-----------------|----------------|----------------:|-------------------:|-------:|--------:|--------:|-------:|
| Scratch | additive | EXP-H128 | 0.545684 | 0.481759 | 0.513722 | +0.012034 | +0.042210 | +0.017 [-0.046,+0.079] | +0.088 [+0.021,+0.164] | -0.014734 | -0.015015 | 0.494730 | 0.465378 | 0.480054 | 325249 |
| Scratch | gated | EXP-H129 | 0.540204 | 0.495443 | 0.517824 | +0.016135 | +0.046311 | +0.012 [-0.053,+0.074] | +0.083 [+0.014,+0.159] | -0.015149 | -0.015368 | 0.496179 | 0.461532 | 0.478856 | 325644 |
| Scratch | surface_aware_pooling | EXP-H130 | 0.542922 | 0.519220 | 0.531071 | +0.029383 | +0.059559 | +0.014 [-0.029,+0.055] | +0.085 [+0.025,+0.146] | -0.007456 | -0.016607 | 0.491006 | 0.466908 | 0.478957 | 328449 |
| ESM2 | additive | EXP-H131 | 0.533946 | 0.525542 | 0.529744 | +0.022949 | +0.043418 | +0.001 [-0.046,+0.044] | +0.063 [+0.003,+0.125] | -0.010561 | -0.009873 | 0.488846 | 0.472828 | 0.480837 | 486529 |
| ESM2 | gated | EXP-H132 | 0.523550 | 0.490751 | 0.507151 | +0.000355 | +0.020824 | -0.010 [-0.057,+0.035] | +0.052 [-0.009,+0.115] | -0.007072 | -0.007772 | 0.479265 | 0.463161 | 0.471213 | 486924 |
| ESM2 | surface_aware_pooling | EXP-H133 | 0.496128 | 0.518552 | 0.507340 | +0.000545 | +0.021014 | -0.037 [-0.093,+0.016] | +0.025 [-0.041,+0.086] | -0.002707 | -0.005882 | 0.464238 | 0.452519 | 0.458378 | 489729 |

## VAL metrics
| Code | VAL_P | VAL_S | VAL_mean | VAL_worst |
|------|------:|------:|---------:|----------:|
| EXP-H128 | 0.422557 | 0.436548 | 0.429552 | 0.436548 |
| EXP-H129 | 0.421952 | 0.456467 | 0.439209 | 0.456467 |
| EXP-H130 | 0.455252 | 0.460697 | 0.457974 | 0.460697 |
| EXP-H131 | 0.479255 | 0.436776 | 0.458016 | 0.479255 |
| EXP-H132 | 0.477724 | 0.440138 | 0.458931 | 0.477724 |
| EXP-H133 | 0.452221 | 0.440198 | 0.446209 | 0.452221 |

## Diagnostics summary
- **Zero-SURFACE:** continuous channels set to TRAIN mean (standardized 0). Positive ΔMAE ⇒ reliance on SURFACE. Primary often **negative** (zeroing helps) ⇒ models do not beneficially rely on residue SURFACE.
- **Residue shuffle:** within antibody & chain; **n_perm=100 per fold (500 fold×perm pairs)** (budget; protocol asked 100). Mean ΔMAE on primary often near 0 or negative ⇒ destroying residue correspondence does **not** systematically hurt.
- Bootstrap CIs vs sequence baselines generally include 0; vs Ab-SURFACE often significantly **worse** (positive ΔMAE) on primary.

### Gate (H129 / H132)
| code     | mode   |   gate_mean |   gate_median |   gate_sd |      q10 |      q25 |      q50 |      q75 |      q90 |   gate_mean_H |   gate_mean_L |     n |
|:---------|:-------|------------:|--------------:|----------:|---------:|---------:|---------:|---------:|---------:|--------------:|--------------:|------:|
| EXP-H129 | gated  |    0.590433 |      0.610608 |  0.20001  | 0.295047 | 0.450062 | 0.610608 | 0.747025 | 0.843546 |      0.71878  |      0.446357 | 75104 |
| EXP-H132 | gated  |    0.569333 |      0.579975 |  0.213515 | 0.271381 | 0.407419 | 0.579975 | 0.752197 | 0.843437 |      0.690518 |      0.433297 | 75104 |

### Attention (H130 / H133)
| code     | mode                  |   entropy_H_mean |   entropy_L_mean |   top1_H_mean |   top5_H_mean |   top1_L_mean |   top5_L_mean |   n_H |   n_L |   mean_nres_H |   mean_nres_L |
|:---------|:----------------------|-----------------:|-----------------:|--------------:|--------------:|--------------:|--------------:|------:|------:|--------------:|--------------:|
| EXP-H130 | surface_aware_pooling |          3.64831 |          3.99606 |     0.174587  |      0.442383 |     0.10565   |      0.310595 |   324 |   324 |       122.593 |        109.21 |
| EXP-H133 | surface_aware_pooling |          4.15867 |          4.21408 |     0.0974607 |      0.286519 |     0.0785497 |      0.240583 |   324 |   324 |       122.593 |        109.21 |

## Interpretation
**CASE G.** Residue-level compact SURFACE fusion does not improve H071 or meaningfully improve H061, and never beats H090/H086. Successful F1_SURFACE signal appears to depend on engineered **antibody-level** aggregation rather than aligned local residue state under current sample size / fusion modes.

Cross-attention is **not** justified: there is no evidence that correct residue correspondence carries incremental predictive value beyond antibody-level SURFACE.

## Required answers
1. Frozen vector: COMPACT10 channels listed above (see schema docs).
2. **p = 10**.
3. Improve H071? **NO** (all Scratch residue TEST_mean worse).
4. Improve H061? **No meaningful gain** (H132/H133 ≈ flat; H131 worse).
5. Beat H090? **NO**.
6. Beat H086? **NO**.
7. Zeroing SURFACE hurt? **Not consistently**; primary often improves when zeroed.
8. Within-Ab residue permutation hurt? **Not consistently** (n=100/fold).
9. Correct residue correspondence matter? **No strong evidence**.
10. Best mode? None beats controls; among residue modes ESM2 gated/pool least harmful (≈H061), Scratch additive least bad among Scratch but still worse than H071.
11. Backbone-dependent? Mildly — ESM2 closer to flat; Scratch clearly degraded.
12. Ab-level F1 near-optimal? **Yes** relative to these residue modes.
13. Cross-attention next? **NO**.
14. If yes what interaction? **N/A**.

## Artifacts
- Prereg: `results/H128_H133_RESIDUE_SURFACE_PREREGISTRATION.yaml`
- Freeze: `results/H128_H133_PRE_EXTERNAL_FREEZE.yaml`
- OOF/external: `results/EXP-H12{8-9,}_OOF_EVALUATION.yaml` / `EXP-H13{0-3}_*`
