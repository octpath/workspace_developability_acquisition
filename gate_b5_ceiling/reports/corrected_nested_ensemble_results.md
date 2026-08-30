# Corrected nested ensemble results

| target   | model               |   corrected_mae |   corrected_pearson_fisher_z |   corrected_pearson_agg |   corrected_rmse |   corrected_spearman |   b4_reported_mae |   b4_reported_pearson |   delta_mae_corrected_minus_b4 |   delta_pearson_corrected_minus_b4 |
|:---------|:--------------------|----------------:|-----------------------------:|------------------------:|-----------------:|---------------------:|------------------:|----------------------:|-------------------------------:|-----------------------------------:|
| HIC      | NESTED_STACK_MEAN   |        0.476115 |                     0.564662 |                0.550158 |         0.670657 |             0.524233 |          nan      |              nan      |                    nan         |                        nan         |
| HIC      | NESTED_STACK_MEDIAN |        0.484024 |                     0.553446 |                0.545434 |         0.676393 |             0.526869 |          nan      |              nan      |                    nan         |                        nan         |
| HIC      | NESTED_STACK_NNLS   |        0.473053 |                     0.575188 |                0.564097 |         0.661811 |             0.557888 |            0.4691 |                0.5479 |                      0.0039531 |                          0.0161973 |
| HIC      | NESTED_STACK_RIDGE  |        0.475967 |                     0.565708 |                0.547943 |         0.673033 |             0.543672 |          nan      |              nan      |                    nan         |                        nan         |
| TmApp    | NESTED_STACK_MEAN   |        2.84925  |                     0.555765 |                0.577027 |         3.71881  |             0.545661 |          nan      |              nan      |                    nan         |                        nan         |
| TmApp    | NESTED_STACK_MEDIAN |        3.02589  |                     0.537507 |                0.565876 |         3.86757  |             0.537033 |          nan      |              nan      |                    nan         |                        nan         |
| TmApp    | NESTED_STACK_NNLS   |        2.88394  |                     0.546239 |                0.565554 |         3.73821  |             0.53592  |          nan      |              nan      |                    nan         |                        nan         |
| TmApp    | NESTED_STACK_RIDGE  |        2.86887  |                     0.538704 |                0.561162 |         3.75655  |             0.527146 |            2.6148 |                0.5956 |                      0.254065  |                         -0.0344378 |

Positive `delta_mae_corrected_minus_b4` means B4 was optimistic (reported MAE too low).
