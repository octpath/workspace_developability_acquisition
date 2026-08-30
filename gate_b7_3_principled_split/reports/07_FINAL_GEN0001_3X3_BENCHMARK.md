# Final GEN_0001 3×3 Benchmark

Production split:
    GEN_0001_B_20271100

Seed:
    20271100

Public ID hash:
    2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376

Private ID hash:
    f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0

Public N:
    81

Private N:
    81

Primary metric:
    MAE

Sanity:
    Public∩Private empty: True
    |Public∪Private|=162: True
    TmApp/HIC share identical IDs: True
    HIC MEDIUM pub/priv: 3/3
    HIC HIGH pub/priv: 4/3
    Models rescored from frozen B5 predictions only (no retrain)
    CV MAE from frozen meanOOF; max |Δ| vs frozen CV ref = 0.00e+00


## TmApp — 3×3 winner table

| Selected by | Model | CV MAE | Public MAE | Private MAE |
|---|---|---:|---:|---:|
| CV-best | NESTED_STACK_MEAN | 2.7781 | 3.6291 | 3.2823 |
| Public-best | PLM_ABLANG2_PCA32_SVR | 2.9480 | 3.5252 | 3.2134 |
| Private-best | PLM_ABLANG2_PCA32_SVR | 2.9480 | 3.5252 | 3.2134 |

CV-best == Public-best? **False**
CV-best == Private-best? **False**
Public-best == Private-best? **True**

Public-selection Private regret:
    0.0000 °C

CV-selection Private regret:
    0.0689 °C


## HIC — 3×3 winner table

| Selected by | Model | CV MAE | Public MAE | Private MAE |
|---|---|---:|---:|---:|
| CV-best | NESTED_STACK_NNLS | 0.4667 | 0.4881 | 0.4566 |
| Public-best | FUSION_ESM2_ESMFN_ElasticNet | 0.5212 | 0.4650 | 0.4965 |
| Private-best | ESMFN_STRUCTURE_ElasticNet | 0.4970 | 0.5163 | 0.4335 |

CV-best == Public-best? **False**
CV-best == Private-best? **False**
Public-best == Private-best? **False**

Public-selection Private regret:
    0.0630 min

CV-selection Private regret:
    0.0231 min


## Comparison with CAND_12528

| Target | Split | CV-best model | Public-best model | Private-best model | Public-selection Private regret | CV-selection Private regret |
|---|---|---|---|---|---:|---:|
| TmApp | CAND_12528 | NESTED_STACK_MEAN | NESTED_STACK_MEAN | PLM_ABLANG2_PCA32_SVR | 0.1862 | 0.1862 |
| TmApp | GEN_0001_B_20271100 | NESTED_STACK_MEAN | PLM_ABLANG2_PCA32_SVR | PLM_ABLANG2_PCA32_SVR | 0.0000 | 0.0689 |
| HIC | CAND_12528 | NESTED_STACK_NNLS | ESMFN_STRUCTURE_ElasticNet | NESTED_STACK_NNLS | 0.0155 | 0.0000 |
| HIC | GEN_0001_B_20271100 | NESTED_STACK_NNLS | FUSION_ESM2_ESMFN_ElasticNet | ESMFN_STRUCTURE_ElasticNet | 0.0630 | 0.0231 |


## Interpretation

Is Public-best close to Private-best?
    TmApp: YES (same model)
    HIC: NO — Public=FUSION_ESM2_ESMFN_ElasticNet, Private=ESMFN_STRUCTURE_ElasticNet; Public-selection regret=0.0630 min

Is CV-best close to Private-best?
    TmApp: NO — CV=NESTED_STACK_MEAN, Private=PLM_ABLANG2_PCA32_SVR; CV-selection regret=0.0689 °C
    HIC: NO — CV=NESTED_STACK_NNLS, Private=ESMFN_STRUCTURE_ElasticNet; CV-selection regret=0.0231 min

Does the new split look competition-safe for TmApp?
    Public-selection regret=0.0000 °C vs CAND_12528 0.1862 °C; CV-selection regret=0.0689 °C.

Does the new split look competition-safe for HIC?
    Public-selection regret=0.0630 min vs CAND_12528 0.0155 min; CV-selection regret=0.0231 min.


## Appendix — full model scores on GEN_0001

| target   | model                        |   cv_mae |   public_mae |   private_mae |   public_rank |   private_rank |
|:---------|:-----------------------------|---------:|-------------:|--------------:|--------------:|---------------:|
| TmApp    | CONST_MEDIAN                 | 3.54321  |     3.78395  |      3.7716   |             5 |              5 |
| TmApp    | SEQ_SIMPLE_Ridge             | 3.63346  |     3.66246  |      3.56686  |             4 |              3 |
| TmApp    | BIO_Ridge                    | 3.16808  |     3.66044  |      3.65203  |             3 |              4 |
| TmApp    | PLM_ABLANG2_PCA32_SVR        | 2.94796  |     3.52523  |      3.21344  |             1 |              1 |
| TmApp    | NESTED_STACK_MEAN            | 2.77807  |     3.62912  |      3.28233  |             2 |              2 |
| HIC      | CONST_MEDIAN                 | 0.518465 |     0.534864 |      0.510086 |             4 |              4 |
| HIC      | SEQ_SIMPLE_Ridge             | 0.564673 |     0.621716 |      0.630073 |             6 |              6 |
| HIC      | PLM_ESM2_PCA64_SVR           | 0.490091 |     0.544275 |      0.568086 |             5 |              5 |
| HIC      | ESMFN_STRUCTURE_ElasticNet   | 0.497026 |     0.516332 |      0.433534 |             3 |              1 |
| HIC      | FUSION_ESM2_ESMFN_ElasticNet | 0.521222 |     0.465006 |      0.496517 |             1 |              3 |
| HIC      | NESTED_STACK_NNLS            | 0.466698 |     0.488076 |      0.456641 |             2 |              2 |


## Appendix — secondary metrics (winner rows only)

| target   | selected_by   | model                        |   public_spearman |   private_spearman |   public_pearson |   private_pearson |
|:---------|:--------------|:-----------------------------|------------------:|-------------------:|-----------------:|------------------:|
| TmApp    | CV-best       | NESTED_STACK_MEAN            |          0.362077 |           0.568341 |         0.403275 |          0.526831 |
| TmApp    | Public-best   | PLM_ABLANG2_PCA32_SVR        |          0.412391 |           0.620008 |         0.376967 |          0.552056 |
| TmApp    | Private-best  | PLM_ABLANG2_PCA32_SVR        |          0.412391 |           0.620008 |         0.376967 |          0.552056 |
| HIC      | CV-best       | NESTED_STACK_NNLS            |          0.471143 |           0.594335 |         0.514619 |          0.545337 |
| HIC      | Public-best   | FUSION_ESM2_ESMFN_ElasticNet |          0.49961  |           0.504074 |         0.569745 |          0.508197 |
| HIC      | Private-best  | ESMFN_STRUCTURE_ElasticNet   |          0.45137  |           0.566363 |         0.491403 |          0.610267 |
