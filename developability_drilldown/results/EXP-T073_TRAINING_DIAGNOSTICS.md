# EXP-T073 Training Diagnostics

- protocol: `DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2`
- seed: `101`
- lr_ref: `0.0003`
- lr_grid: `[2.9999999999999997e-05, 8.999999999999999e-05, 0.0003, 0.0009]`
- max_epochs=200, patience=20
- scheduler: CosineAnnealingLR T_max=200 eta_min=0.0

## Selected LR by fold

| scheme | fold | selected_lr | best_epoch | best_step | best_val_mae | final_epoch | stopped_early |
|--------|------|-------------|------------|-----------|--------------|-------------|---------------|
| primary | 0 | 0.0009 | 12 | 84 | 2.775854 | 32 | True |
| primary | 1 | 0.0009 | 12 | 72 | 3.201210 | 32 | True |
| primary | 2 | 0.0009 | 12 | 84 | 3.146419 | 32 | True |
| primary | 3 | 3e-05 | 67 | 469 | 2.589641 | 87 | True |
| primary | 4 | 9e-05 | 41 | 287 | 3.154061 | 61 | True |
| shadow | 0 | 0.0009 | 9 | 63 | 2.615618 | 29 | True |
| shadow | 1 | 0.0003 | 7 | 49 | 3.301578 | 27 | True |
| shadow | 2 | 0.0009 | 24 | 168 | 3.232650 | 44 | True |
| shadow | 3 | 0.0003 | 25 | 175 | 3.349044 | 45 | True |
| shadow | 4 | 0.0009 | 12 | 84 | 2.915994 | 32 | True |

## All LR candidates (best VAL MAE)

### primary
- fold 0:
  - lr=3e-05: best_epoch=148 step=1036 VAL=2.847697 final=168 stopped=True
  - lr=9e-05: best_epoch=50 step=350 VAL=2.819500 final=70 stopped=True
  - lr=0.0003: best_epoch=18 step=126 VAL=2.789441 final=38 stopped=True
  - lr=0.0009: best_epoch=12 step=84 VAL=2.775854 final=32 stopped=True **SELECTED**
- fold 1:
  - lr=3e-05: best_epoch=45 step=270 VAL=3.425261 final=65 stopped=True
  - lr=9e-05: best_epoch=16 step=96 VAL=3.403451 final=36 stopped=True
  - lr=0.0003: best_epoch=6 step=36 VAL=3.373612 final=26 stopped=True
  - lr=0.0009: best_epoch=12 step=72 VAL=3.201210 final=32 stopped=True **SELECTED**
- fold 2:
  - lr=3e-05: best_epoch=88 step=616 VAL=3.151993 final=108 stopped=True
  - lr=9e-05: best_epoch=26 step=182 VAL=3.172898 final=46 stopped=True
  - lr=0.0003: best_epoch=11 step=77 VAL=3.277020 final=31 stopped=True
  - lr=0.0009: best_epoch=12 step=84 VAL=3.146419 final=32 stopped=True **SELECTED**
- fold 3:
  - lr=3e-05: best_epoch=67 step=469 VAL=2.589641 final=87 stopped=True **SELECTED**
  - lr=9e-05: best_epoch=23 step=161 VAL=2.638329 final=43 stopped=True
  - lr=0.0003: best_epoch=8 step=56 VAL=2.627191 final=28 stopped=True
  - lr=0.0009: best_epoch=6 step=42 VAL=2.701674 final=26 stopped=True
- fold 4:
  - lr=3e-05: best_epoch=14 step=98 VAL=3.330054 final=34 stopped=True
  - lr=9e-05: best_epoch=41 step=287 VAL=3.154061 final=61 stopped=True **SELECTED**
  - lr=0.0003: best_epoch=18 step=126 VAL=3.181999 final=38 stopped=True
  - lr=0.0009: best_epoch=10 step=70 VAL=3.198300 final=30 stopped=True
### shadow
- fold 0:
  - lr=3e-05: best_epoch=60 step=420 VAL=2.642685 final=80 stopped=True
  - lr=9e-05: best_epoch=26 step=182 VAL=2.645430 final=46 stopped=True
  - lr=0.0003: best_epoch=14 step=98 VAL=2.636047 final=34 stopped=True
  - lr=0.0009: best_epoch=9 step=63 VAL=2.615618 final=29 stopped=True **SELECTED**
- fold 1:
  - lr=3e-05: best_epoch=52 step=364 VAL=3.328568 final=72 stopped=True
  - lr=9e-05: best_epoch=18 step=126 VAL=3.329028 final=38 stopped=True
  - lr=0.0003: best_epoch=7 step=49 VAL=3.301578 final=27 stopped=True **SELECTED**
  - lr=0.0009: best_epoch=9 step=63 VAL=3.352837 final=29 stopped=True
- fold 2:
  - lr=3e-05: best_epoch=65 step=455 VAL=3.240358 final=85 stopped=True
  - lr=9e-05: best_epoch=15 step=105 VAL=3.263388 final=35 stopped=True
  - lr=0.0003: best_epoch=7 step=49 VAL=3.244307 final=27 stopped=True
  - lr=0.0009: best_epoch=24 step=168 VAL=3.232650 final=44 stopped=True **SELECTED**
- fold 3:
  - lr=3e-05: best_epoch=37 step=259 VAL=3.517859 final=57 stopped=True
  - lr=9e-05: best_epoch=12 step=84 VAL=3.480213 final=32 stopped=True
  - lr=0.0003: best_epoch=25 step=175 VAL=3.349044 final=45 stopped=True **SELECTED**
  - lr=0.0009: best_epoch=1 step=7 VAL=3.601365 final=21 stopped=True
- fold 4:
  - lr=3e-05: best_epoch=75 step=525 VAL=3.031179 final=95 stopped=True
  - lr=9e-05: best_epoch=28 step=196 VAL=3.009615 final=48 stopped=True
  - lr=0.0003: best_epoch=10 step=70 VAL=3.008958 final=30 stopped=True
  - lr=0.0009: best_epoch=12 step=84 VAL=2.915994 final=32 stopped=True **SELECTED**

## Best-epoch distribution (selected only)

- mean=22.10 median=12.0 min=7 max=67 std=18.84
- fraction stopped before epoch 40: 0.80
- fraction stopped before epoch 80: 1.00
- fraction reaching epoch 200: 0.00
- fraction stopped_early: 1.00

## Cosine annealing vs early stopping

CosineAnnealingLR(T_max=200) only modestly reduces LR in the first ~20–40 epochs.
If most selected best_epoch values are ≪ 200, patience=20 likely terminates before
cosine annealing has materially annealed. This run reports the observation only;
patience/scheduler are NOT changed.

- primary selected LRs: [3e-05, 9e-05, 0.0009] (n_unique=3)
- shadow selected LRs: [0.0003, 0.0009] (n_unique=2)
