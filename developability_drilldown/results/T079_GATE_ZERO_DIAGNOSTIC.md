# T079 gate-zero diagnostic (no retrain)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- source: `results/EXP-T079_SELECTED_LR.csv`
- definition residual ratio: `||g * CrossAttn||_F / ||H_pre||_F` on residue tokens (after layer-1; REG excluded), averaged over up to 4 VAL batches.
- delta MAE = MAE(gate0) − MAE(normal); positive ⇒ zeroing gates worsens MAE.

## Aggregate OOF MAE

| scheme | split | MAE normal | MAE gate0 | Δ |
|---|---|---:|---:|---:|
| primary | VAL OOF | 2.914272 | 2.915135 | 0.000863111 |
| primary | TEST OOF | 3.236327 | 3.236583 | 0.000256621 |
| shadow | VAL OOF | 3.004834 | 3.004648 | -0.000185884 |
| shadow | TEST OOF | 3.267069 | 3.266412 | -0.000657447 |

## Per-fold deltas

| scheme | fold | g_H | g_L | residual_ratio_H_mean | residual_ratio_L_mean | val_mae_delta | test_oof_mae_delta | ext_fold_mae_delta | mean_abs_pred_delta_val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 0 | 0.0105516 | -0.0109775 | 0.00307274 | 0.00287708 | -0.000483195 | -0.00248289 | -0.00020418 | 0.00686969 |
| primary | 1 | 0.00128058 | -0.000376533 | 0.000536123 | 0.000187237 | 1.22533e-05 | 4.60076e-05 | 3.08001e-05 | 0.000186111 |
| primary | 2 | 0.00620285 | -0.00874569 | 0.00169143 | 0.00268223 | 0.000946283 | -0.00015097 | 0.00022827 | 0.00403905 |
| primary | 3 | 0.00683817 | -0.00204728 | 0.00167663 | 0.000540777 | 0.000474215 | -0.000484705 | -0.00010926 | 0.0018332 |
| primary | 4 | 0.0057927 | -0.00436059 | 0.00337993 | 0.00306523 | 0.00343466 | 0.00437498 | 0.00247955 | 0.0205221 |
| shadow | 0 | 0.00740969 | -0.00731549 | 0.00201362 | 0.00214838 | -6.77398e-05 | -3.31402e-05 | 0.000561608 | 0.0051073 |
| shadow | 1 | -0.00363805 | -0.00716889 | 0.00237937 | 0.00487037 | -0.000524759 | -0.00268092 | -0.000333386 | 0.00939631 |
| shadow | 2 | -0.00478644 | -0.00356762 | 0.00209901 | 0.00183542 | -0.000253439 | -0.000112534 | 0.000103703 | 0.00188661 |
| shadow | 3 | -0.00140635 | -0.00224077 | 0.000606902 | 0.000843046 | -4.83195e-05 | -5.36442e-05 | -0.000176465 | 0.000677629 |
| shadow | 4 | -0.00124335 | 0.00468957 | 0.000720888 | 0.00218113 | -4.31538e-05 | -0.000353264 | 3.80528e-05 | 0.000900507 |

## Interpretation notes

- Learned gates on T079 are typically near zero (see `EXP-T079_CROSS_GATES.csv`); small residual ratios and small prediction deltas are expected.
- Residual ratio uses the trained g values (not forced to zero). If instrumentation fails on a fold, rely on mean absolute prediction delta columns.
- This diagnostic does **not** alter T080–T104 configs or checkpoints.

CSV: `results/T079_GATE_ZERO_DIAGNOSTIC.csv`

