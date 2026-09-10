# EXP-T075 Training Diagnostics

- platform: `DL_FOLDLOCAL_COSINE_V3`
- seed: `101`
- lr_grid: `[1e-05, 0.0001, 0.001, 0.01]`
- max_epochs=200, patience=30, min_epochs=0
- scheduler: cosine T_max=200, eta_min=0.01×lr0

## Per candidate

| scheme | fold | initial_lr | eta_min | best_epoch | lr_at_best | best_VAL | selected | final | num_fail |
|--------|------|------------|---------|------------|------------|----------|----------|-------|----------|
| primary | 0 | 1e-05 | 1e-07 | 177 | 4.47606e-07 | 3.127366 |  | 200 | False |
| primary | 0 | 1e-04 | 1e-06 | 37 | 9.22942e-05 | 2.728453 | Y | 67 | False |
| primary | 0 | 1e-03 | 1e-05 | 10 | 0.000995062 | 2.869828 |  | 40 | False |
| primary | 0 | 1e-02 | 0.0001 | 9 | 0.00996097 | 3.351198 |  | 39 | False |
| primary | 1 | 1e-05 | 1e-07 | 200 | 1.00611e-07 | 3.483289 |  | 200 | False |
| primary | 1 | 1e-04 | 1e-06 | 20 | 9.78118e-05 | 3.383720 |  | 50 | False |
| primary | 1 | 1e-03 | 1e-05 | 10 | 0.000995062 | 3.280419 | Y | 40 | False |
| primary | 1 | 1e-02 | 0.0001 | 6 | 0.00998474 | 3.577388 |  | 36 | False |
| primary | 2 | 1e-05 | 1e-07 | 174 | 5.38554e-07 | 3.537271 |  | 200 | False |
| primary | 2 | 1e-04 | 1e-06 | 30 | 9.49524e-05 | 3.190300 |  | 60 | False |
| primary | 2 | 1e-03 | 1e-05 | 54 | 0.000838141 | 3.131482 | Y | 84 | False |
| primary | 2 | 1e-02 | 0.0001 | 73 | 0.00715761 | 3.625140 |  | 103 | False |
| primary | 3 | 1e-05 | 1e-07 | 153 | 1.44161e-06 | 2.949128 |  | 183 | False |
| primary | 3 | 1e-04 | 1e-06 | 19 | 9.80345e-05 | 2.627386 | Y | 49 | False |
| primary | 3 | 1e-03 | 1e-05 | 5 | 0.000999023 | 2.681618 |  | 35 | False |
| primary | 3 | 1e-02 | 0.0001 | 28 | 0.00956145 | 3.223410 |  | 58 | False |
| primary | 4 | 1e-05 | 1e-07 | 49 | 8.65839e-06 | 3.336268 |  | 79 | False |
| primary | 4 | 1e-04 | 1e-06 | 40 | 9.09984e-05 | 3.160979 |  | 70 | False |
| primary | 4 | 1e-03 | 1e-05 | 33 | 0.000938772 | 3.091479 | Y | 63 | False |
| primary | 4 | 1e-02 | 0.0001 | 4 | 0.0099945 | 3.344987 |  | 34 | False |
| shadow | 0 | 1e-05 | 1e-07 | 200 | 1.00611e-07 | 2.993484 |  | 200 | False |
| shadow | 0 | 1e-04 | 1e-06 | 20 | 9.78118e-05 | 2.646367 | Y | 50 | False |
| shadow | 0 | 1e-03 | 1e-05 | 22 | 0.000973312 | 2.666457 |  | 52 | False |
| shadow | 0 | 1e-02 | 0.0001 | 20 | 0.00978118 | 3.257627 |  | 50 | False |
| shadow | 1 | 1e-05 | 1e-07 | 196 | 1.15259e-07 | 3.375323 |  | 200 | False |
| shadow | 1 | 1e-04 | 1e-06 | 23 | 9.70736e-05 | 3.351653 | Y | 53 | False |
| shadow | 1 | 1e-03 | 1e-05 | 10 | 0.000995062 | 3.369153 |  | 40 | False |
| shadow | 1 | 1e-02 | 0.0001 | 11 | 0.00993906 | 3.484375 |  | 41 | False |
| shadow | 2 | 1e-05 | 1e-07 | 159 | 1.13873e-06 | 3.320927 |  | 189 | False |
| shadow | 2 | 1e-04 | 1e-06 | 18 | 9.82456e-05 | 3.298451 | Y | 48 | False |
| shadow | 2 | 1e-03 | 1e-05 | 6 | 0.000998474 | 3.340305 |  | 36 | False |
| shadow | 2 | 1e-02 | 0.0001 | 43 | 0.00896127 | 3.500378 |  | 73 | False |
| shadow | 3 | 1e-05 | 1e-07 | 124 | 3.3003e-06 | 3.523312 |  | 154 | False |
| shadow | 3 | 1e-04 | 1e-06 | 8 | 9.97011e-05 | 3.507045 |  | 38 | False |
| shadow | 3 | 1e-03 | 1e-05 | 12 | 0.000992629 | 3.305144 | Y | 42 | False |
| shadow | 3 | 1e-02 | 0.0001 | 18 | 0.00982456 | 3.554084 |  | 48 | False |
| shadow | 4 | 1e-05 | 1e-07 | 200 | 1.00611e-07 | 3.210890 |  | 200 | False |
| shadow | 4 | 1e-04 | 1e-06 | 21 | 9.75773e-05 | 3.008655 | Y | 51 | False |
| shadow | 4 | 1e-03 | 1e-05 | 10 | 0.000995062 | 3.030318 |  | 40 | False |
| shadow | 4 | 1e-02 | 0.0001 | 9 | 0.00996097 | 3.405696 |  | 39 | False |

## Selected initial-LR winner counts

- 1e-05: 0
- 1e-04: 6
- 1e-03: 4
- 1e-02: 0

## Selected best_epoch

- min=10 q25=18.2 median=20.5 mean=24.70 q75=30.5 max=54
- stop before epoch 50: 9
- stop before epoch 100: 10
- final_epoch reach 200: 0
- final_epoch <50: 4
- final_epoch <100: 10

## Selected lr_at_best

- min=9.22942e-05 q25=9.76359e-05 median=9.81401e-05 q75=0.000913614 max=0.000995062

## Numerical failures

- none
