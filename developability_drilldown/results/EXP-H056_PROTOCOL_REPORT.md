# EXP-H056 — ESM-2 ARCH-1 MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-1`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `29c052b24e6039700f82e077980baae03061c70f74356359fe7d28eff41ffea8`
- n_trainable: 485249
- VAL_P/S/mean/worst: 0.461046 / 0.436254 / 0.448650 / 0.461046

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          15                  105          0.000988      0.514603           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/primary_k0_lr1e-03_seed101.pt 649131418599f207   101
primary     1               0.0010 0.000010          14                   84          0.000990      0.523928           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/primary_k1_lr1e-03_seed101.pt 649131418599f207   101
primary     2               0.0010 0.000010          32                  224          0.000942      0.347303           62           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/primary_k2_lr1e-03_seed101.pt 649131418599f207   101
primary     3               0.0010 0.000010          26                  182          0.000962      0.449853           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/primary_k3_lr1e-03_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010          13                   91          0.000991      0.465902           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0010 0.000010          26                  182          0.000962      0.411717           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 649131418599f207   101
 shadow     1               0.0010 0.000010          19                  133          0.000980      0.465745           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001          52                  364          0.000085      0.465469           82           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0001 0.000001         179                 1253          0.000004      0.419266          200          False              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 649131418599f207   101
 shadow     4               0.0001 0.000001          70                  490          0.000074      0.420369          100           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H056_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 649131418599f207   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

