# DL_FOLDLOCAL_COSINE_V3 — Shared Deep-Learning Training Platform

Architecture parameters (layers, heads, REG, H/L communication, etc.) are
**experiment-specific** and are NOT part of this shared platform.

## Shared settings

| Item | Value |
|------|-------|
| optimizer | AdamW |
| weight_decay | 0.01 |
| batch_size | 16 |
| loss | SmoothL1(beta=0.5) |
| initial_lr_candidates | 1e-5, 1e-4, 1e-3, 1e-2 |
| lr_selection | fold-local validation MAE |
| scheduler | cosine, T_max=200, eta_min=0.01×initial_lr |
| warmup / restart | none |
| max_epochs | 200 |
| patience | 30 |
| min_epochs | none |
| checkpoint | best validation MAE (from epoch 1) |
| full_Dev refit | disabled |
| external prediction | fold-checkpoint mean + median |
| seed_policy | experiment-specific |

## Scheduler timing

LR for 1-indexed epoch `N` is set at the **beginning** of the epoch to
`lr_at_t(N-1)`. The training-history `learning_rate` column is the LR used
**during** that epoch's gradient updates.

## Split semantics

Existing Primary/Shadow TRAIN/VAL/TEST rotations only. NOT nested CV.

Baseline experiment under this platform: `EXP-T075`.

