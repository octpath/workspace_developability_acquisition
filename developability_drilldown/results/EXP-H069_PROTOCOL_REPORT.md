# EXP-H069 — Scratch ARCH-1 CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `HIC`
- arch_id: `ARCH-1`
- representation: `SCRATCH`
- content_mode / merge / chain: `scratch` / `concat` / `HL`
- geometry: `False`
- config_hash: `4e13895612165b9641aa5b8ff26e592a21ff63d2058f5eb33e40c9cbada8bc5c`
- n_trainable: 340481
- VAL_P/S/mean/worst: 0.470374 / 0.457799 / 0.464086 / 0.470374

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          25                  175          0.000965      0.438738           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/primary_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0001 0.000001          67                  402          0.000076      0.500269           97           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/primary_k1_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0010 0.000010          21                  147          0.000976      0.452602           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/primary_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0010 0.000010          35                  245          0.000931      0.489850           65           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/primary_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0001 0.000001          55                  385          0.000083      0.470463           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/primary_k4_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0010 0.000010          26                  182          0.000962      0.490961           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0010 0.000010          47                  329          0.000876      0.395508           77           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0001 0.000001          66                  462          0.000076      0.476195           96           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0001 0.000001          52                  364          0.000085      0.520168           82           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/shadow_k3_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0001 0.000001          78                  546          0.000068      0.403179          108           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-H069_run/checkpoints/shadow_k4_lr1e-04_seed101.pt 0d84ab04a06e7c54   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

