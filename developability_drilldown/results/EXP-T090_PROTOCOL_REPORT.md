# EXP-T090 — Scratch ARCH-1 separate dual REG CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-1`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `concat`
- config_hash: `7825b3e7e7a3f4a8f9c1aef326fa83dfee593e8833329e8cbed5c5fa1866d6ca`
- n_trainable: 340481
- VAL_P/S/mean/worst: 2.956308 / 2.941925 / 2.949117 / 2.956308
- TEST_P/S/mean/worst: 3.421672 / 3.679165 / 3.550418 / 3.679165

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001         105                  735          0.000047      2.764891          135           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/primary_k0_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     1               0.0010 0.000010          26                  156          0.000962      3.266688           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/primary_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
primary     2               0.0001 0.000001          83                  581          0.000064      3.369218          113           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/primary_k2_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     3               0.0001 0.000001         104                  728          0.000048      2.582127          134           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/primary_k3_lr1e-04_seed101.pt 0d84ab04a06e7c54   101
primary     4               0.0010 0.000010          61                  427          0.000796      2.794901           91           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/primary_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     0               0.0010 0.000010          19                  133          0.000980      2.540637           49           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     1               0.0010 0.000010          32                  224          0.000942      3.124684           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     2               0.0010 0.000010          16                  112          0.000986      2.881226           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     3               0.0010 0.000010          12                   84          0.000993      3.424354           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 0d84ab04a06e7c54   101
 shadow     4               0.0010 0.000010          43                  301          0.000896      2.736189           73           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T090_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 0d84ab04a06e7c54   101

