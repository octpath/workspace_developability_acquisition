# EXP-T074 LR Trajectory Diagnostics

- protocol: `DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2_COARSE_COSINE`
- seed: `101`
- lr_grid: `[1e-05, 0.0001, 0.001, 0.01]`
- cosine_epochs=100, eta_min=0.01×initial_lr, then hold
- min_epochs=100, max_epochs=200, patience=20
- scheduler step timing: `LR set at beginning of each epoch; no post-epoch CosineAnnealingLR.step()`

## Per candidate (best VAL)

| scheme | fold | initial_lr | eta_min | best_epoch | lr_at_best | best_VAL | selected | final_epoch | num_fail |
|--------|------|------------|---------|------------|------------|----------|----------|-------------|----------|
| primary | 0 | 1e-05 | 1e-07 | 200 | 1e-07 | 3.270007 |  | 200 | False |
| primary | 0 | 1e-04 | 1e-06 | 37 | 7.15761e-05 | 2.750634 |  | 100 | False |
| primary | 0 | 1e-03 | 1e-05 | 7 | 0.000991232 | 2.716752 | Y | 100 | False |
| primary | 0 | 1e-02 | 0.0001 | 17 | 0.00938772 | 3.352697 |  | 100 | False |
| primary | 1 | 1e-05 | 1e-07 | 199 | 1e-07 | 3.578890 |  | 200 | False |
| primary | 1 | 1e-04 | 1e-06 | 20 | 9.14405e-05 | 3.383104 |  | 100 | False |
| primary | 1 | 1e-03 | 1e-05 | 5 | 0.000996097 | 3.273595 | Y | 100 | False |
| primary | 1 | 1e-02 | 0.0001 | 7 | 0.00991232 | 3.620307 |  | 100 | False |
| primary | 2 | 1e-05 | 1e-07 | 72 | 2.01611e-06 | 3.609572 |  | 100 | False |
| primary | 2 | 1e-04 | 1e-06 | 28 | 8.32349e-05 | 3.216817 |  | 100 | False |
| primary | 2 | 1e-03 | 1e-05 | 16 | 0.000946048 | 3.065327 | Y | 100 | False |
| primary | 2 | 1e-02 | 0.0001 | 71 | 0.00214046 | 3.630322 |  | 100 | False |
| primary | 3 | 1e-05 | 1e-07 | 87 | 5.71106e-07 | 3.123300 |  | 107 | False |
| primary | 3 | 1e-04 | 1e-06 | 19 | 9.22942e-05 | 2.634455 | Y | 100 | False |
| primary | 3 | 1e-03 | 1e-05 | 6 | 0.000993906 | 2.679253 |  | 100 | False |
| primary | 3 | 1e-02 | 0.0001 | 28 | 0.00832349 | 3.224176 |  | 100 | False |
| primary | 4 | 1e-05 | 1e-07 | 51 | 5.05e-06 | 3.342753 |  | 100 | False |
| primary | 4 | 1e-04 | 1e-06 | 72 | 2.01611e-05 | 3.209078 |  | 100 | False |
| primary | 4 | 1e-03 | 1e-05 | 52 | 0.000489452 | 3.208057 | Y | 100 | False |
| primary | 4 | 1e-02 | 0.0001 | 10 | 0.00980345 | 3.350939 |  | 100 | False |
| shadow | 0 | 1e-05 | 1e-07 | 200 | 1e-07 | 3.157777 |  | 200 | False |
| shadow | 0 | 1e-04 | 1e-06 | 20 | 9.14405e-05 | 2.646921 |  | 100 | False |
| shadow | 0 | 1e-03 | 1e-05 | 7 | 0.000991232 | 2.602089 | Y | 100 | False |
| shadow | 0 | 1e-02 | 0.0001 | 10 | 0.00980345 | 3.262926 |  | 100 | False |
| shadow | 1 | 1e-05 | 1e-07 | 200 | 1e-07 | 3.438866 |  | 200 | False |
| shadow | 1 | 1e-04 | 1e-06 | 18 | 9.31067e-05 | 3.349660 |  | 100 | False |
| shadow | 1 | 1e-03 | 1e-05 | 13 | 0.000965239 | 3.304160 | Y | 100 | False |
| shadow | 1 | 1e-02 | 0.0001 | 9 | 0.00984449 | 3.484375 |  | 100 | False |
| shadow | 2 | 1e-05 | 1e-07 | 96 | 1.60943e-07 | 3.423203 |  | 116 | False |
| shadow | 2 | 1e-04 | 1e-06 | 20 | 9.14405e-05 | 3.290054 |  | 100 | False |
| shadow | 2 | 1e-03 | 1e-05 | 19 | 0.000922942 | 3.212158 | Y | 100 | False |
| shadow | 2 | 1e-02 | 0.0001 | 43 | 0.00628101 | 3.500114 |  | 100 | False |
| shadow | 3 | 1e-05 | 1e-07 | 40 | 6.72675e-06 | 3.548279 |  | 100 | False |
| shadow | 3 | 1e-04 | 1e-06 | 8 | 9.88079e-05 | 3.507259 |  | 100 | False |
| shadow | 3 | 1e-03 | 1e-05 | 11 | 0.000975773 | 3.376395 | Y | 100 | False |
| shadow | 3 | 1e-02 | 0.0001 | 23 | 0.00886404 | 3.548993 |  | 100 | False |
| shadow | 4 | 1e-05 | 1e-07 | 200 | 1e-07 | 3.332420 |  | 200 | False |
| shadow | 4 | 1e-04 | 1e-06 | 21 | 9.05463e-05 | 3.008534 |  | 100 | False |
| shadow | 4 | 1e-03 | 1e-05 | 15 | 0.000952889 | 2.999157 | Y | 100 | False |
| shadow | 4 | 1e-02 | 0.0001 | 2 | 0.00999756 | 3.404285 |  | 100 | False |

## A. INITIAL LR WINNER COUNTS

- 1e-05: 0
- 1e-04: 1
- 1e-03: 9
- 1e-02: 0

## B. LR-AT-BEST DISTRIBUTION (selected)

- min=9.22942e-05 q25=0.000928719 median=0.000959064 q75=0.000987367 max=0.000996097
- log10: min=-4.035 q25=-3.032 median=-3.018 q75=-3.006 max=-3.002

## C. BEST-EPOCH DISTRIBUTION (selected)

- min=5 median=14.0 mean=16.40 max=52
- <=25: 9
- 26-50: 0
- 51-75: 1
- 76-100: 0
- >100: 0
- final_epoch min/median/max: 100/100.0/100
- fraction final_epoch>=100: 1.00
- fraction final_epoch>=200: 0.00

## D. CROSS-CANDIDATE LR-AT-BEST BY INITIAL LR

- init 1e-05: n=10 lr_at_best median=1.30471e-07 [1e-07, 6.72675e-06] best_epoch median=147.5
- init 1e-04: n=10 lr_at_best median=9.14405e-05 [2.01611e-05, 9.88079e-05] best_epoch median=20.0
- init 1e-03: n=10 lr_at_best median=0.000970506 [0.000489452, 0.000996097] best_epoch median=12.0
- init 1e-02: n=10 lr_at_best median=0.00959559 [0.00214046, 0.00999756] best_epoch median=13.5

### Overlap interpretation

- per-initial median lr_at_best: {'1e-05': 1.304713570270343e-07, '1e-04': 9.14404884265908e-05, '1e-03': 0.0009705061680403927, '1e-02': 0.009595585905158997}
- ratio max/min of those medians ≈ 73545.54 (closer to 1 ⇒ trajectories meet in a common useful LR band).

## Numerical failures

- none
