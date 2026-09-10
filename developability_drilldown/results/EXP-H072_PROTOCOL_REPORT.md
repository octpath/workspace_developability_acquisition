# EXP-H072 — Scratch ARCH-3 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-3`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `False`
- config_hash: `87983158d6cad80723d03da26bc31c66e25a1e25e9cb64347b962561476f225c`
- n_trainable: 340481
- VAL_P/S/mean/worst: 0.465941 / 0.469760 / 0.467850 / 0.469760

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          22                  154          0.000973      0.473686           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/primary_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0010 0.000010          39                  234          0.000914      0.471582           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0010 0.000010          42                  294          0.000901      0.433786           72           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/primary_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0010 0.000010          51                  357          0.000855      0.482712           81           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/primary_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0001 0.000001          80                  560          0.000067      0.467519          110           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/primary_k4_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0010 0.000010          25                  175          0.000965      0.474995           55           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0010 0.000010          33                  231          0.000939      0.456814           63           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0010 0.000010          15                  105          0.000988      0.463105           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0010 0.000010          27                  189          0.000959      0.526353           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0010 0.000010          13                   91          0.000991      0.425600           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H072_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

