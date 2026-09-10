# EXP-T101 — Scratch ARCH-7 REG-only cross CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-7`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `a8cfa1bd19038557d212e00ea2930b5f2fcb9e8ef6837c405635f3ba45e110cc`
- n_trainable: 406529
- VAL_P/S/mean/worst: 2.904491 / 2.972051 / 2.938271 / 2.972051
- TEST_P/S/mean/worst: 3.190352 / 3.491463 / 3.340907 / 3.491463

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          10                   70          0.000995      2.752032           40           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8462d848103517b5   101
primary     1               0.0010 0.000010          13                   78          0.000991      3.190103           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8462d848103517b5   101
primary     2               0.0010 0.000010          25                  175          0.000965      3.205131           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8462d848103517b5   101
primary     3               0.0010 0.000010           8                   56          0.000997      2.775854           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8462d848103517b5   101
primary     4               0.0010 0.000010          29                  203          0.000953      2.595175           59           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     0               0.0001 0.000001          42                  294          0.000090      2.626543           72           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8462d848103517b5   101
 shadow     1               0.0001 0.000001          28                  196          0.000096      3.320779           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 8462d848103517b5   101
 shadow     2               0.0010 0.000010          33                  231          0.000939      2.732622           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     3               0.0001 0.000001          26                  182          0.000096      3.476841           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 8462d848103517b5   101
 shadow     4               0.0010 0.000010          22                  154          0.000973      2.698493           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T101_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8462d848103517b5   101

