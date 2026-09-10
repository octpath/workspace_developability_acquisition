# EXP-T121 — AbLang2 ARCH-7 REG-only cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-7`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `14b1b797f0a31eb3d37050caa677edcb211808e0cf732f8483f84d2498db0a6a`
- n_trainable: 448897
- VAL_P/S/mean/worst: 2.720867 / 2.806288 / 2.763577 / 2.806288

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          17                  119          0.000984      2.559755           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8d6e8476a7702db3   101
primary     1               0.0010 0.000010          50                  300          0.000860      3.292829           80           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8d6e8476a7702db3   101
primary     2               0.0001 0.000001          49                  343          0.000087      3.173840           79           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     3               0.0001 0.000001         105                  735          0.000047      2.042070          135           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     4               0.0001 0.000001         138                  966          0.000023      2.523001          168           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/primary_k4_lr1e-04_seed101.pt 8d6e8476a7702db3   101
 shadow     0               0.0001 0.000001          58                  406          0.000081      2.283349           88           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8d6e8476a7702db3   101
 shadow     1               0.0001 0.000001          69                  483          0.000074      2.678453           99           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 8d6e8476a7702db3   101
 shadow     2               0.0010 0.000010          29                  203          0.000953      2.866845           59           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     3               0.0100 0.000100           7                   49          0.009978      3.540389           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 8d6e8476a7702db3   101
 shadow     4               0.0010 0.000010          17                  119          0.000984      2.655805           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T121_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8d6e8476a7702db3   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

