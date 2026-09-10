# EXP-T076 — Joint H/L + single REG (T068-like)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch: `{'joint_hl_single_reg': True, 'joint_hl_dual_reg': False, 'joint_hl_chain_specific_dual_reg': False, 'use_cross_attention_bridge': False}`
- n_trainable: 485121
- VAL_P/S/mean/worst: 2.987128 / 3.012172 / 2.999650 / 3.012172
- TEST_P/S/mean/worst: 3.243109 / 3.405739 / 3.324424 / 3.405739

## Selected LR

 scheme  fold  selected_initial_lr      eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0              0.00001 1.000000e-07         111                  777          0.000004      2.874455          141           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/primary_k0_lr1e-05_seed101.pt b9ed3af03ebb5577   101
primary     1              0.00100 1.000000e-05          21                  126          0.000976      3.402397           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/primary_k1_lr1e-03_seed101.pt b9ed3af03ebb5577   101
primary     2              0.00010 1.000000e-06          18                  126          0.000098      3.086167           48           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/primary_k2_lr1e-04_seed101.pt b9ed3af03ebb5577   101
primary     3              0.00010 1.000000e-06          21                  147          0.000098      2.600716           51           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/primary_k3_lr1e-04_seed101.pt b9ed3af03ebb5577   101
primary     4              0.00100 1.000000e-05          51                  357          0.000855      2.962451           81           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/primary_k4_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     0              0.00100 1.000000e-05          60                  420          0.000802      2.549775           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/shadow_k0_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     1              0.00010 1.000000e-06          26                  182          0.000096      3.271561           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/shadow_k1_lr1e-04_seed101.pt b9ed3af03ebb5577   101
 shadow     2              0.00100 1.000000e-05          29                  203          0.000953      3.077377           59           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/shadow_k2_lr1e-03_seed101.pt b9ed3af03ebb5577   101
 shadow     3              0.00010 1.000000e-06          13                   91          0.000099      3.312297           43           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/shadow_k3_lr1e-04_seed101.pt b9ed3af03ebb5577   101
 shadow     4              0.00100 1.000000e-05          23                  161          0.000971      2.854920           53           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T076_run/checkpoints/shadow_k4_lr1e-03_seed101.pt b9ed3af03ebb5577   101

