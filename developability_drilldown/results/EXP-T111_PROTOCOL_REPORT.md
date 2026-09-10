# EXP-T111 — AbLang2 ARCH-2 joint single REG

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-2`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `None` / `HL`
- geometry: `False`
- config_hash: `d344dff4e3606af7ef5833a2a0f77804540bceb5b0e5576dc62d836de07fd523`
- n_trainable: 382721
- VAL_P/S/mean/worst: 2.815077 / 2.914211 / 2.864644 / 2.914211

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          72                  504          0.000072      2.828576          102           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/primary_k0_lr1e-04_seed101.pt e4849aa9ea4fbcc2   101
primary     1               0.0010 0.000010          22                  132          0.000973      3.502033           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/primary_k1_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
primary     2               0.0010 0.000010          59                  413          0.000808      2.864047           89           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/primary_k2_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
primary     3               0.0001 0.000001          42                  294          0.000090      2.345969           72           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/primary_k3_lr1e-04_seed101.pt e4849aa9ea4fbcc2   101
primary     4               0.0010 0.000010          51                  357          0.000855      2.512871           81           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/primary_k4_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
 shadow     0               0.0010 0.000010          24                  168          0.000968      2.509729           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/shadow_k0_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
 shadow     1               0.0010 0.000010          40                  280          0.000910      2.716638           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/shadow_k1_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
 shadow     2               0.0001 0.000001          49                  343          0.000087      2.923006           79           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/shadow_k2_lr1e-04_seed101.pt e4849aa9ea4fbcc2   101
 shadow     3               0.0010 0.000010          19                  133          0.000980      3.459017           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/shadow_k3_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101
 shadow     4               0.0010 0.000010          17                  119          0.000984      2.958278           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T111_run/checkpoints/shadow_k4_lr1e-03_seed101.pt e4849aa9ea4fbcc2   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

