# EXP-T102 — Scratch ARCH-7 REG-only cross MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-7`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `mean`
- config_hash: `9d055782c20728754794d701a8f89f2feb015f3ddf666e4d2a27ca35e528dc0e`
- n_trainable: 390145
- VAL_P/S/mean/worst: 2.924401 / 3.009879 / 2.967140 / 3.009879
- TEST_P/S/mean/worst: 3.275112 / 3.428850 / 3.351981 / 3.428850

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          42                  294          0.000090      2.821322           72           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/primary_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
primary     1               0.0010 0.000010          16                   96          0.000986      3.047864           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     2               0.0010 0.000010          17                  119          0.000984      3.262207           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     3               0.0010 0.000010          23                  161          0.000971      2.589466           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/primary_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
primary     4               0.0001 0.000001          89                  623          0.000060      2.900507          119           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/primary_k4_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     0               0.0001 0.000001          78                  546          0.000068      2.631094          108           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     1               0.0010 0.000010          13                   91          0.000991      3.097477           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     2               0.0010 0.000010          49                  343          0.000866      2.977758           79           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     3               0.0010 0.000010           4                   28          0.000999      3.480796           34           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101
 shadow     4               0.0010 0.000010          27                  189          0.000959      2.859390           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T102_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab87aa9f6cc57b6   101

