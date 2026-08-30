# TmApp scientific analysis

Question: is TmApp primarily a maturation/germline signal, or is there substantial sequence/structure information beyond it?

## Univariate Spearman vs `tm_app_C` (n=346)

```
x                              spearman  pearson
PL_vh_germline_distance        -0.249   -0.265
A2_HL_charge_ph7                0.228    0.244
PL_combined_germline_distance  -0.220   -0.243
ABB_Fv_sasa_hydrophobic        -0.143   -0.156
B_all_FR_gravy                  0.142    0.178
A2_HL_gravy                     0.094    0.126
ABB_BSA                         0.023    0.048
```

## Controlled modeling evidence

- BIO_SHORTCUT canonical CV ≈ **0.505** vs best PLM ≈ **0.576** — germline/family features already capture most linear signal.
- Structure (ABB / native ESMFold) trails PLM by ~0.17–0.20 CV Spearman.
- Residual: PLM on BIO residuals ~0.24; structure residuals weaker (~0.08–0.16).
- Dev-OOF ensemble puts ~85% weight on PLM, ~0% on ESMFN.

## Verdict

**Primarily maturation/germline-accessible**, with modest residual PLM signal beyond BIO_SHORTCUT. Structure does not currently add reliable competition headroom. If used in a competition, publish BIO_SHORTCUT as an official baseline and consider germline-family grouped evaluation diagnostics.
