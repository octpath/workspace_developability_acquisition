# EXP-T093 — Scratch ARCH-3 joint unrestricted dual REG CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-3`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `a035009fe03823c6ec43d6568d59504a4d5f1fe6527321ed344ee7d558b5098d`
- n_trainable: 340481
- VAL_P/S/mean/worst: 3.002952 / 3.137411 / 3.070181 / 3.137411
- TEST_P/S/mean/worst: 3.332103 / 3.308162 / 3.320132 / 3.332103

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          23                  161          0.000971      2.701137           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/primary_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0010 0.000010          39                  234          0.000914      3.285848           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0010 0.000010          21                  147          0.000976      3.406742           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/primary_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0010 0.000010          22                  154          0.000973      2.858592           52           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/primary_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0010 0.000010          40                  280          0.000910      2.763031           70           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/primary_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0001 0.000001          91                  637          0.000058      2.605766          121           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0010 0.000010          32                  224          0.000942      3.343632           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0001 0.000001          82                  574          0.000065      3.045347          112           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0010 0.000010          19                  133          0.000980      3.437142           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0100 0.000100           5                   35          0.009990      3.262414           35           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T093_run/checkpoints/shadow_k4_lr1e-02_seed101.pt 0d84ab04a06e7c54   101

