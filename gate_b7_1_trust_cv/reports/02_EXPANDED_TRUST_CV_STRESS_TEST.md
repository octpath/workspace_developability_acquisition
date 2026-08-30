# Phase 2B — Expanded Trust-CV stress test

| target   | label      |   n_unique_models |   n_pairs |   cv_public_spearman |   cv_public_pearson |   directional_agreement_rate |   n_material_disagreements |   material_disagreement_rate |   n_private_follows_cv |   n_private_follows_public |   trustcv_rate |   trustcv_ci_lo |   trustcv_ci_hi |   follow_cv_mean |   follow_cv_median |   follow_cv_p75 |   follow_cv_p90 |   follow_cv_max |   follow_pub_mean |   follow_pub_median |   follow_pub_p75 |   follow_pub_p90 |   follow_pub_max | verdict                     |
|:---------|:-----------|------------------:|----------:|---------------------:|--------------------:|-----------------------------:|---------------------------:|-----------------------------:|-----------------------:|---------------------------:|---------------:|----------------:|----------------:|-----------------:|-------------------:|----------------:|----------------:|----------------:|------------------:|--------------------:|-----------------:|-----------------:|-----------------:|:----------------------------|
| TmApp    | STRESS_ALL |                38 |       703 |            0.0928196 |            0.138802 |                     0.545845 |                        113 |                    0.161891  |                     53 |                         57 |       0.481818 |        0.392385 |        0.576577 |        0.0625523 |        0.000167403 |       0.0867473 |       0.210701  |       0.520795  |        0.0682056  |                   0 |         0.132962 |      0.203767    |        0.354541  | TRUST_CV_AMBIGUOUS          |
| HIC      | STRESS_ALL |                37 |       666 |            0.849419  |            0.966158 |                     0.845921 |                         17 |                    0.0256798 |                      2 |                         13 |       0.133333 |        0        |        0.355147 |        0.0267394 |        0.024278    |       0.0411699 |       0.0551755 |       0.0753798 |        0.00177873 |                   0 |         0        |      0.000396782 |        0.0292464 | PUBLIC_AT_LEAST_AS_RELIABLE |

## Policies

| target   | policy       |   mean_private |   mean_regret |   median_regret |   p90_regret |
|:---------|:-------------|---------------:|--------------:|----------------:|-------------:|
| HIC      | BALANCED     |       0.447573 |     0.0163135 |       0.0181225 |    0.0181225 |
| HIC      | CV_FIRST     |       0.454847 |     0.023587  |       0.0245988 |    0.0397762 |
| HIC      | PUBLIC_FIRST |       0.447573 |     0.0163135 |       0.0181225 |    0.0181225 |
| TmApp    | BALANCED     |       3.38872  |     0.114469  |       0.155957  |    0.155957  |
| TmApp    | CV_FIRST     |       3.35967  |     0.0854241 |       0.0803973 |    0.155957  |
| TmApp    | PUBLIC_FIRST |       3.38872  |     0.114469  |       0.155957  |    0.155957  |

FINAL: FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED
