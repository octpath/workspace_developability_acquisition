# EXP-H078 — Scratch ARCH-6G CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6G`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `True`
- config_hash: `988945786780b6e0e217a517af4cad217b797f987e2a79e815d1721ab82ebf9d`
- n_trainable: 406561
- VAL_P/S/mean/worst: 0.462607 / 0.447762 / 0.455185 / 0.462607

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          16                  112          0.000986      0.503757           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     1               0.0010 0.000010          12                   72          0.000993      0.517620           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     2               0.0010 0.000010          21                  147          0.000976      0.407600           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
primary     3               0.0001 0.000001          52                  364          0.000085      0.485123           82           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101
primary     4               0.0010 0.000010          45                  315          0.000886      0.395929           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     0               0.0010 0.000010          45                  315          0.000886      0.375957           75           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     1               0.0010 0.000010          46                  322          0.000881      0.396159           76           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8c1eeca1c36aaf78   101
 shadow     2               0.0001 0.000001          45                  315          0.000089      0.489181           75           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101
 shadow     3               0.0001 0.000001          12                   84          0.000099      0.536064           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101
 shadow     4               0.0001 0.000001          46                  322          0.000088      0.440936           76           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H078_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 8c1eeca1c36aaf78   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

