# EXP-T095 — Scratch ARCH-4 joint chain-specific dual REG CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-4`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `b0875988606c659e04cb530fe8fc544a13cbcae3b4a9885e2f1d11db74647253`
- n_trainable: 340481
- VAL_P/S/mean/worst: 2.962447 / 3.108076 / 3.035262 / 3.108076
- TEST_P/S/mean/worst: 3.308433 / 3.495900 / 3.402166 / 3.495900

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          90                  630          0.000059      2.894862          120           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/primary_k0_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0010 0.000010          22                  132          0.000973      3.373230           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0001 0.000001          84                  588          0.000064      3.343071          114           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/primary_k2_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0010 0.000010          15                  105          0.000988      2.421373           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/primary_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0010 0.000010          57                  399          0.000821      2.768975           87           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/primary_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0001 0.000001         107                  749          0.000046      2.467379          137           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0001 0.000001          49                  343          0.000087      3.425453           79           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0010 0.000010          25                  175          0.000965      3.338099           55           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0010 0.000010          15                  105          0.000988      3.420279           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0010 0.000010          50                  350          0.000860      2.899434           80           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T095_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101

