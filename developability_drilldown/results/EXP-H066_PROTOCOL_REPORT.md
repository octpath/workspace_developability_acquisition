# EXP-H066 — ESM-2 ARCH-8 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-8`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `b164693acc4992e1efd3b063ad5043c1f0670aed40739cf26ad8c42522211541`
- n_trainable: 567681
- VAL_P/S/mean/worst: 0.460318 / 0.441096 / 0.450707 / 0.460318

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          14                   98          0.000990      0.502762           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/primary_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     1               0.0010 0.000010          14                   84          0.000990      0.528367           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/primary_k1_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     2               0.0010 0.000010          20                  140          0.000978      0.360851           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/primary_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
primary     3               0.0001 0.000001          66                  462          0.000076      0.462014           96           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/primary_k3_lr1e-04_seed101.pt 570e590061e5af2a   101
primary     4               0.0010 0.000010          22                  154          0.000973      0.444144           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/primary_k4_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     0               0.0010 0.000010          60                  420          0.000802      0.402809           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     1               0.0001 0.000001         118                  826          0.000037      0.440115          148           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 570e590061e5af2a   101
 shadow     2               0.0010 0.000010          11                   77          0.000994      0.471082           41           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     3               0.0010 0.000010          39                  273          0.000914      0.471272           69           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 570e590061e5af2a   101
 shadow     4               0.0001 0.000001          43                  301          0.000090      0.420456           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H066_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 570e590061e5af2a   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

