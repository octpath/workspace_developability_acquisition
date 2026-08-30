# Predicted vs Observed Visual Check (Gate B6.2 add-on)

Frozen artifacts only: Train **proper OOF** + CAND_12528 Public/Private one-shot predictions.
No retraining, no Optuna, no split change.

Train-derived thresholds (applied to all roles): **Q90=10.5372**, **Q95=11.1356**

Shared 1:1 axis range across all model panels: **[7.414, 14.006]**

## Plots

- `plots/pred_vs_true/nested_stack_nnls_cv_public_private.png`
- `plots/pred_vs_true/esmfold_structure_cv_public_private.png`
- `plots/pred_vs_true/esm2_pca64_svr_cv_public_private.png`
- `plots/pred_vs_true/fusion_esm2_esmfold_cv_public_private.png`
- `plots/pred_vs_true/seq_simple_cv_public_private.png`
- `plots/pred_vs_true/nested_stack_nnls_tail_labeled.png`
- `plots/pred_vs_true/nested_stack_nnls_residual_vs_true.png`
- `plots/pred_vs_true/nested_stack_nnls_true_vs_pred_distribution.png`

Scatter regression on the figure is **pred = a + b·true** (dashed purple).
Gate B6.2 calibration slope (**obs = c + s·pred**) is annotated separately as `cal s=…` — do not confuse the two.

NESTED_STACK_NNLS snapshot:

| Role | n | MAE | Pearson | pred/obs SD |
|------|---|-----|---------|-------------|
| Train OOF | 162 | 0.467 | 0.564 | 0.561 |
| Public | 81 | 0.513 | 0.393 | 0.430 |
| Private | 81 | 0.431 | 0.685 | 0.577 |

Cross-model pred/obs SD + Pearson:

| Model | Role | pred/obs SD | Pearson | MAE |
|-------|------|-------------|---------|-----|
| NESTED_STACK_NNLS | Train CV OOF | 0.561 | 0.564 | 0.467 |
| NESTED_STACK_NNLS | Public (CAND_12528) | 0.430 | 0.393 | 0.513 |
| NESTED_STACK_NNLS | Private (CAND_12528) | 0.577 | 0.685 | 0.431 |
| ESMFN_STRUCTURE / ElasticNet | Train CV OOF | 0.649 | 0.520 | 0.497 |
| ESMFN_STRUCTURE / ElasticNet | Public (CAND_12528) | 0.464 | 0.479 | 0.503 |
| ESMFN_STRUCTURE / ElasticNet | Private (CAND_12528) | 0.627 | 0.648 | 0.447 |
| PLM_ESM2_PCA64 / SVR | Train CV OOF | 0.577 | 0.529 | 0.490 |
| PLM_ESM2_PCA64 / SVR | Public (CAND_12528) | 0.545 | 0.218 | 0.609 |
| PLM_ESM2_PCA64 / SVR | Private (CAND_12528) | 0.655 | 0.588 | 0.503 |
| FUSION_ESM2_ESMFN / ElasticNet | Train CV OOF | 0.742 | 0.464 | 0.521 |
| FUSION_ESM2_ESMFN / ElasticNet | Public (CAND_12528) | 0.525 | 0.437 | 0.529 |
| FUSION_ESM2_ESMFN / ElasticNet | Private (CAND_12528) | 0.715 | 0.642 | 0.433 |
| SEQ_SIMPLE / Ridge | Train CV OOF | 0.698 | 0.383 | 0.565 |
| SEQ_SIMPLE / Ridge | Public (CAND_12528) | 0.656 | 0.279 | 0.670 |
| SEQ_SIMPLE / Ridge | Private (CAND_12528) | 0.766 | 0.388 | 0.582 |

## Answers (visual)

1. **Central HIC (~8.5–9.5):** Yes — bulk points sit near the identity line (slight scatter around ~8.5–10).
2. **Prediction range vs observed:** Yes — clearly narrower (Nested CV pred/obs SD=0.561; Public 0.430; Private 0.577). Histograms show predictions truncated ~≤11 while observed reaches 12–14.
3. **High-HIC systematic underprediction:** Yes — orange/red (Train Q90/Q95+) sit below y=x; residual slope ≈ −0.6 to −0.8.
4. **Shared across CV/Public/Private:** Yes — center-OK + right-tail-under is common to all three roles.
5. **Public-only anomaly:** Pattern is the same, but Public is the *most compressed* (flattest OLS slope b≈0.17, weakest Pearson) with extreme high-HIC leverage points near 13–14 still predicted ~10. Not a different mechanism — a more extreme version of the same shrinkage.
6. **Private-only anomaly:** No — Private most resembles CV (steeper positive slope, stronger Pearson); same underprediction of the tail.
7. **Other PLM/structure models:** Yes — ESMFN / ESM2-SVR / Fusion / Seq-Ridge all show the same compressed cloud and high-true / low-pred tail.
8. **Flat median band vs rising predictions:** Not a pure horizontal median stripe — predictions do rise with true HIC (b>0, r>0), but too shallow; high measured values are pulled toward ~9–10.5.
9. **Supports Gate B6.2 conclusion?** **Yes** — plots support continuous signal + material high-tail underprediction / under-dispersion.
