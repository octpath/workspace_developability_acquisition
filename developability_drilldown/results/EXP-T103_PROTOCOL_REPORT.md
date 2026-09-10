# EXP-T103 — Scratch ARCH-8 within-chain extra attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-8`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `067b94ef72c7dac477913385755e4ccb034a6d0926e8f9b4ff1fc359220c27aa`
- n_trainable: 406529
- VAL_P/S/mean/worst: 3.153053 / 3.021071 / 3.087062 / 3.153053
- TEST_P/S/mean/worst: 3.475191 / 3.454753 / 3.464972 / 3.475191

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          15                  105          0.000988      3.172211           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/primary_k0_lr1e-03_seed101.pt 8462d848103517b5   101
primary     1               0.0010 0.000010          22                  132          0.000973      3.187087           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8462d848103517b5   101
primary     2               0.0001 0.000001          58                  406          0.000081      3.543725           88           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8462d848103517b5   101
primary     3               0.0010 0.000010          11                   77          0.000994      2.771750           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/primary_k3_lr1e-03_seed101.pt 8462d848103517b5   101
primary     4               0.0010 0.000010          15                  105          0.000988      3.088831           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     0               0.0001 0.000001          55                  385          0.000083      2.441109           85           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8462d848103517b5   101
 shadow     1               0.0010 0.000010          20                  140          0.000978      3.206667           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     2               0.0010 0.000010          21                  147          0.000976      3.001841           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8462d848103517b5   101
 shadow     3               0.0100 0.000100          13                   91          0.009912      3.470135           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 8462d848103517b5   101
 shadow     4               0.0010 0.000010          26                  182          0.000962      2.989691           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T103_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8462d848103517b5   101

