# EXP-H059 — ESM-2 ARCH-3 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-3`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `0d1c36722271ba4a6fd204c98f4d540c024d0c5347b33d55907608e8ab147eaa`
- n_trainable: 485249
- VAL_P/S/mean/worst: 0.458653 / 0.422429 / 0.440541 / 0.458653

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          63                  441          0.000078      0.486550           93           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/primary_k0_lr1e-04_seed101.pt 649131418599f207   101
primary     1               0.0010 0.000010          19                  114          0.000980      0.517106           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/primary_k1_lr1e-03_seed101.pt 649131418599f207   101
primary     2               0.0010 0.000010          39                  273          0.000914      0.359853           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/primary_k2_lr1e-03_seed101.pt 649131418599f207   101
primary     3               0.0010 0.000010          20                  140          0.000978      0.460950           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/primary_k3_lr1e-03_seed101.pt 649131418599f207   101
primary     4               0.0001 0.000001          48                  336          0.000087      0.466109           78           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/primary_k4_lr1e-04_seed101.pt 649131418599f207   101
 shadow     0               0.0010 0.000010          33                  231          0.000939      0.408696           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 649131418599f207   101
 shadow     1               0.0010 0.000010          22                  154          0.000973      0.463390           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 649131418599f207   101
 shadow     2               0.0010 0.000010          13                   91          0.000991      0.461290           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 649131418599f207   101
 shadow     3               0.0010 0.000010          55                  385          0.000832      0.426286           85           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 649131418599f207   101
 shadow     4               0.0010 0.000010          58                  406          0.000814      0.352791           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H059_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 649131418599f207   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

