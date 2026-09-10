# EXP-H076 — Scratch ARCH-6 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-6`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `False`
- config_hash: `aac01d1709e2386532e92794ffe596e9e4c4a18994a894bb082291713e60ad64`
- n_trainable: 406529
- VAL_P/S/mean/worst: 0.476862 / 0.464657 / 0.470759 / 0.476862

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          41                  287          0.000091      0.519438           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/primary_k0_lr1e-04_seed101.pt 8462d848103517b5   101
primary     1               0.0010 0.000010          12                   72          0.000993      0.517600           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8462d848103517b5   101
primary     2               0.0010 0.000010          20                  140          0.000978      0.434941           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8462d848103517b5   101
primary     3               0.0001 0.000001          52                  364          0.000085      0.485161           82           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8462d848103517b5   101
primary     4               0.0010 0.000010          22                  154          0.000973      0.424565           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     0               0.0010 0.000010          49                  343          0.000866      0.416114           79           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     1               0.0010 0.000010          18                  126          0.000982      0.491652           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     2               0.0010 0.000010          29                  203          0.000953      0.482508           59           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     3               0.0010 0.000010          14                   98          0.000990      0.514674           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     4               0.0010 0.000010          19                  133          0.000980      0.418289           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H076_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8462d848103517b5   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

