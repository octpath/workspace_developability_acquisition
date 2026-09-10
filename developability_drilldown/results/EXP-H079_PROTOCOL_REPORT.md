# EXP-H079 — Scratch ARCH-6G MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6G`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `mean` / `HL`
- geometry: `True`
- config_hash: `6598aed53d098d17411f74b7846ae40a6b9bcb9a52927b8c04ac466d4bda6dac`
- n_trainable: 390177
- VAL_P/S/mean/worst: 0.461265 / 0.434060 / 0.447663 / 0.461265

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          27                  189          0.000959      0.482049           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     1               0.0010 0.000010           6                   36          0.000998      0.529183           36           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     2               0.0010 0.000010          18                  126          0.000982      0.404646           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8425f2f579fed963   101
primary     3               0.0001 0.000001          48                  336          0.000087      0.473413           78           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8425f2f579fed963   101
primary     4               0.0010 0.000010          29                  203          0.000953      0.414262           59           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     0               0.0010 0.000010          47                  329          0.000876      0.388298           77           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     1               0.0010 0.000010          47                  329          0.000876      0.450719           77           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     2               0.0001 0.000001          95                  665          0.000055      0.439771          125           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 8425f2f579fed963   101
 shadow     3               0.0010 0.000010          48                  336          0.000871      0.475700           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8425f2f579fed963   101
 shadow     4               0.0001 0.000001          57                  399          0.000082      0.415942           87           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H079_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 8425f2f579fed963   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

