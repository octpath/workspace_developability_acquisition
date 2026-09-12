# HIC SOURCE_SAP24 Fold-Local Shallow XGBoost (H126–H127)

Estimator-only experiment. Feature: frozen SOURCE_SAP24 (24D). No SCM/SURFACE/Transformer.

## Summary table

| Model | Features | depth | LR | TEST_P | TEST_S | TEST_mean | TEST_worst | best_iter median | Public | Private | Overall |
|-------|----------|------:|---:|-------:|-------:|----------:|-----------:|-----------------:|-------:|--------:|--------:|
| EXP-H126 | SOURCE_SAP24 | 2 | 0.02 | 0.5799 | 0.5572 | 0.5686 | 0.5799 | 43 | 0.6296 | 0.5765 | 0.6031 |
| EXP-H127 | SOURCE_SAP24 | 3 | 0.02 | 0.5804 | 0.5535 | 0.5670 | 0.5804 | 48 | 0.6347 | 0.5737 | 0.6042 |

## Internal VAL (OOF)

| Model | VAL_P | VAL_S | VAL_mean | VAL_worst |
|-------|------:|------:|---------:|----------:|
| EXP-H126 | 0.5640 | 0.5566 | 0.5603 | 0.5640 |
| EXP-H127 | 0.5687 | 0.5539 | 0.5613 | 0.5687 |

## Comparisons

- Ridge SOURCE_SAP24 TEST_mean ≈ 0.644
- RBF-SVR SOURCE_SAP24 TEST_mean ≈ 0.547
- H071 Scratch TEST_mean = 0.5017
- H061 ESM2 TEST_mean = 0.5068
- H114/H115 Scratch+SAP24 = 0.5140 / 0.5180
- H120/H121 ESM2+SAP24 = 0.5207 / 0.5224
- reported approximate-SAP + XGBoost ≈ 0.46 (not same conditions)

## Verdict (canonical depth=2 = H126)

- TEST_mean = **0.5686** → **CASE_C_no_reproduce_0.46**
- depth=3 sensitivity H127 TEST_mean = 0.5670

Interpretation (CASE C): exact SOURCE_SAP24 + shallow XGBoost does not reproduce the reported ~0.46 approximate-SAP XGB result, and does not beat Transformer baselines. Deprioritize exact SOURCE_SAP24 as an HIC mainline feature track.

### Final questions

1. Beat H071/H061? **NO** (H126=0.5686 vs 0.5017/0.5068)
2. Better than Ridge/SVR? Ridge=YES; SVR=NO (H126=0.5686 vs Ridge 0.644 / SVR 0.547)
3. Reproduce ≈0.46? **NO** (0.5686)
4. depth=2 enough vs depth=3? **YES** (nearly identical); Δ(d3−d2)=-0.0016
5. Tree-specific predictive signal? **WEAK** — better than Ridge, worse than RBF-SVR, far from Transformer; no clear tree-unique win.
6. Continue SAP24 for HIC? **NO strong remaining justification** under CASE C for exact SOURCE_SAP24 as HIC mainline.

# Fold-wise best_iteration / VAL MAE

## EXP-H126

| scheme | fold | best_iteration | VAL_MAE |
|--------|-----:|---------------:|--------:|
| primary | 0 | 102 | 0.556558 |
| primary | 1 | 1 | 0.608177 |
| primary | 2 | 2 | 0.548495 |
| primary | 3 | 78 | 0.595072 |
| primary | 4 | 24 | 0.510515 |
| shadow | 0 | 56 | 0.564519 |
| shadow | 1 | 50 | 0.547546 |
| shadow | 2 | 83 | 0.560423 |
| shadow | 3 | 36 | 0.607038 |
| shadow | 4 | 16 | 0.501618 |

## EXP-H127

| scheme | fold | best_iteration | VAL_MAE |
|--------|-----:|---------------:|--------:|
| primary | 0 | 91 | 0.586277 |
| primary | 1 | 1 | 0.609269 |
| primary | 2 | 0 | 0.548988 |
| primary | 3 | 159 | 0.589570 |
| primary | 4 | 24 | 0.507708 |
| shadow | 0 | 100 | 0.550040 |
| shadow | 1 | 46 | 0.551587 |
| shadow | 2 | 51 | 0.562413 |
| shadow | 3 | 61 | 0.603656 |
| shadow | 4 | 18 | 0.500510 |


STOP. Do not run SCM/SURFACE/residue/Transformer follow-ons from this batch.
