# EXP-H074 — Scratch ARCH-4 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-4`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `False`
- config_hash: `af78e79d82828ed37288ce19ae54bac4537ed021cc49009cb25343f82e52971f`
- n_trainable: 340481
- VAL_P/S/mean/worst: 0.461742 / 0.449963 / 0.455852 / 0.461742

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          36                  252          0.000927      0.483690           66           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/primary_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0010 0.000010          23                  138          0.000971      0.514650           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0010 0.000010          54                  378          0.000838      0.416154           84           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/primary_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0010 0.000010          33                  231          0.000939      0.447389           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/primary_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0010 0.000010          34                  238          0.000935      0.444487           64           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/primary_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0010 0.000010          48                  336          0.000871      0.389091           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0100 0.000100           7                   49          0.009978      0.521278           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/shadow_k1_lr1e-02_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0001 0.000001         127                  889          0.000031      0.433616          157           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0001 0.000001          88                  616          0.000061      0.514184          118           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0010 0.000010          19                  133          0.000980      0.391538           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H074_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

