# EXP-H055 — ESM-2 ARCH-1 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-1`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `673dd6a6c44f5a800f13355f471e6f373bc625a7784658e3d0423bf741d87699`
- n_trainable: 501633
- VAL_P/S/mean/worst: 0.451691 / 0.433025 / 0.442358 / 0.451691

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          55                  385          0.000083      0.484967           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/primary_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010          31                  186          0.000946      0.503241           61           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          31                  217          0.000946      0.354154           61           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001          82                  574          0.000065      0.443711          112           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0001 0.000001          41                  287          0.000091      0.469732           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/primary_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0010 0.000010          40                  280          0.000910      0.395105           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0010 0.000010          41                  287          0.000905      0.487956           71           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0010 0.000010          18                  126          0.000982      0.458140           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          28                  196          0.000956      0.408144           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0001 0.000001          61                  427          0.000080      0.417743           91           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H055_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

