# HIC PLM + structure nested stacking

Bases: ESMFN_STRUCTURE/ElasticNet, PLM_ESM2/PCA64/SVR, SEQ_SIMPLE/Ridge.

| tag                 | model   |      mae |   pearson |     rmse |   spearman |
|:--------------------|:--------|---------:|----------:|---------:|-----------:|
| NESTED_STACK_MEAN   | mean    | 0.476115 |  0.548811 | 0.670657 |   0.524233 |
| NESTED_STACK_MEDIAN | median  | 0.484024 |  0.532708 | 0.676393 |   0.526869 |
| NESTED_STACK_NNLS   | nnls    | 0.473053 |  0.56008  | 0.661811 |   0.557888 |
| NESTED_STACK_RIDGE  | ridge   | 0.475967 |  0.551955 | 0.673033 |   0.543672 |
