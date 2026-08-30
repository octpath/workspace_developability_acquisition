# TmApp BIO + PLM nested stacking

Bases: BIO/Ridge, PLM_ABLANG2/ElasticNet, FUSION_ABLANG2_BIO/ElasticNet.

| tag                 | model   |     mae |   pearson |    rmse |   spearman |
|:--------------------|:--------|--------:|----------:|--------:|-----------:|
| NESTED_STACK_MEAN   | mean    | 2.84925 |  0.544502 | 3.71881 |   0.545661 |
| NESTED_STACK_MEDIAN | median  | 3.02589 |  0.527803 | 3.86757 |   0.537033 |
| NESTED_STACK_NNLS   | nnls    | 2.88394 |  0.531646 | 3.73821 |   0.53592  |
| NESTED_STACK_RIDGE  | ridge   | 2.86887 |  0.524406 | 3.75655 |   0.527146 |
