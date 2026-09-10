# EXP-H071 — Scratch ARCH-2 single REG

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-2`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `None` / `HL`
- geometry: `False`
- config_hash: `df6965bbdaa20d533cbb2b585011ab497f761f1cb44c766884f7c5633afb4471`
- n_trainable: 323969
- VAL_P/S/mean/worst: 0.474522 / 0.461138 / 0.467830 / 0.474522

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          39                  273          0.000091      0.490430           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/primary_k0_lr1e-04_seed101.pt b2759a2692967759   101
primary     1               0.0010 0.000010          19                  114          0.000980      0.505451           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/primary_k1_lr1e-03_seed101.pt b2759a2692967759   101
primary     2               0.0010 0.000010          56                  392          0.000826      0.406108           86           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/primary_k2_lr1e-03_seed101.pt b2759a2692967759   101
primary     3               0.0001 0.000001          61                  427          0.000080      0.501977           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/primary_k3_lr1e-04_seed101.pt b2759a2692967759   101
primary     4               0.0010 0.000010          13                   91          0.000991      0.467180           43           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/primary_k4_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     0               0.0010 0.000010          19                  133          0.000980      0.433525           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/shadow_k0_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     1               0.0010 0.000010          30                  210          0.000950      0.502080           60           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/shadow_k1_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     2               0.0010 0.000010          16                  112          0.000986      0.473058           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/shadow_k2_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     3               0.0010 0.000010          37                  259          0.000923      0.494868           67           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/shadow_k3_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     4               0.0010 0.000010          28                  196          0.000956      0.401968           58           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H071_run/checkpoints/shadow_k4_lr1e-03_seed101.pt b2759a2692967759   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

