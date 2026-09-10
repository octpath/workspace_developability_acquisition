# EXP-H057 — ESM-2 ARCH-2 single REG

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-2`
- representation: `ESM2`
- content_mode / merge / chain: `frozen` / `None` / `HL`
- geometry: `False`
- config_hash: `a4f11774fb0e0b5960661cca3ff9d35c91389fb195385360ed0c2ad7d94b8fcd`
- n_trainable: 485121
- VAL_P/S/mean/worst: 0.443465 / 0.449794 / 0.446629 / 0.449794

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          50                  350          0.000086      0.479302           80           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/primary_k0_lr1e-04_seed101.pt b9ed3af03ebb5577   101
primary     1               0.0001 0.000001          41                  246          0.000091      0.523121           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/primary_k1_lr1e-04_seed101.pt b9ed3af03ebb5577   101
primary     2               0.0010 0.000010          28                  196          0.000956      0.364058           58           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/primary_k2_lr1e-03_seed101.pt b9ed3af03ebb5577   101
primary     3               0.0001 0.000001          72                  504          0.000072      0.456480          102           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/primary_k3_lr1e-04_seed101.pt b9ed3af03ebb5577   101
primary     4               0.0010 0.000010          99                  693          0.000521      0.390753          129           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/primary_k4_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     0               0.0010 0.000010          33                  231          0.000939      0.385980           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/shadow_k0_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     1               0.0010 0.000010          19                  133          0.000980      0.513242           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/shadow_k1_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     2               0.0001 0.000001          37                  259          0.000092      0.458984           67           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/shadow_k2_lr1e-04_seed101.pt b9ed3af03ebb5577   101
 shadow     3               0.0010 0.000010          54                  378          0.000838      0.483684           84           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/shadow_k3_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     4               0.0001 0.000001          57                  399          0.000082      0.408015           87           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H057_run/checkpoints/shadow_k4_lr1e-04_seed101.pt b9ed3af03ebb5577   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

