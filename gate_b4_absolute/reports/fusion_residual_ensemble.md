# Fusion / residual / ensemble (MAE)

| target   | tag                               | model        |      mae |
|:---------|:----------------------------------|:-------------|---------:|
| HIC      | ENSEMBLE_NNLS                     | oof_nnls     | 0.469072 |
| HIC      | ENSEMBLE_RIDGE                    | oof_ridge    | 0.47427  |
| HIC      | ENSEMBLE_MEAN                     | oof_combine  | 0.475016 |
| HIC      | ENSEMBLE_MEDIAN                   | oof_combine  | 0.478135 |
| HIC      | CAL_ENSEMBLE_NNLS                 | linear_oof   | 0.482434 |
| HIC      | RESID_ESMFN_STRUCTURE__PLM_ESM2   | OOF_residual | 0.709243 |
| HIC      | RESID_SEQ_SIMPLE__ESMFN_STRUCTURE | OOF_residual | 0.797557 |
| HIC      | RESID_PLM_ESM2__ESMFN_STRUCTURE   | OOF_residual | 0.808907 |
| TmApp    | ENSEMBLE_RIDGE                    | oof_ridge    | 2.61477  |
| TmApp    | ENSEMBLE_NNLS                     | oof_nnls     | 2.63106  |
| TmApp    | CAL_ENSEMBLE_NNLS                 | linear_oof   | 2.68479  |
| TmApp    | ENSEMBLE_MEDIAN                   | oof_combine  | 2.71815  |
| TmApp    | ENSEMBLE_MEAN                     | oof_combine  | 2.73663  |
| TmApp    | RESID_BIO__PLM_ABLANG2            | OOF_residual | 3.25995  |
| TmApp    | RESID_PLM_ABLANG2__BIO            | OOF_residual | 3.29826  |
| TmApp    | RESID_IMGT_POS_HL__PLM_ABLANG2    | OOF_residual | 3.49752  |

Residuals used OOF bases. Ensembles fit on Train OOF only.
