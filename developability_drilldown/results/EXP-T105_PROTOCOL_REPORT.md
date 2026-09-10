# EXP-T105 — AbLingua ARCH-6G geometry cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `ABLINGUA`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `True`
- config_hash: `37d210b91613d27d03cbf809818046e4d3c667285b853d879a2148f3c9fdcb56`
- n_trainable: 567713
- VAL_P/S/mean/worst: 2.984888 / 3.025324 / 3.005106 / 3.025324

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          39                  273          0.000091      2.730557           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/primary_k0_lr1e-04_seed101.pt 9b1e9e6632131a81   101
primary     1               0.0010 0.000010           7                   42          0.000998      3.342672           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/primary_k1_lr1e-03_seed101.pt 9b1e9e6632131a81   101
primary     2               0.0001 0.000001          26                  182          0.000096      3.119143           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/primary_k2_lr1e-04_seed101.pt 9b1e9e6632131a81   101
primary     3               0.0001 0.000001          18                  126          0.000098      2.560935           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/primary_k3_lr1e-04_seed101.pt 9b1e9e6632131a81   101
primary     4               0.0001 0.000001          69                  483          0.000074      3.167899           99           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/primary_k4_lr1e-04_seed101.pt 9b1e9e6632131a81   101
 shadow     0               0.0010 0.000010          14                   98          0.000990      2.527459           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     1               0.0001 0.000001          26                  182          0.000096      3.272929           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 9b1e9e6632131a81   101
 shadow     2               0.0010 0.000010          25                  175          0.000965      3.091599           55           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 9b1e9e6632131a81   101
 shadow     3               0.0001 0.000001          17                  119          0.000098      3.306543           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 9b1e9e6632131a81   101
 shadow     4               0.0010 0.000010           6                   42          0.000998      2.934857           36           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T105_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 9b1e9e6632131a81   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

