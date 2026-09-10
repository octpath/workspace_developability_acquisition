# EXP-T078 — Joint H/L + chain-specific dual REG masks (T071-like)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch: `{'joint_hl_single_reg': False, 'joint_hl_dual_reg': False, 'joint_hl_chain_specific_dual_reg': True, 'use_cross_attention_bridge': False}`
- n_trainable: 501633
- VAL_P/S/mean/worst: 2.966494 / 3.103569 / 3.035031 / 3.103569
- TEST_P/S/mean/worst: 3.425690 / 3.359062 / 3.392376 / 3.425690

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          37                  259          0.000092      2.705858           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/primary_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010           5                   30          0.000999      3.325410           35           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          28                  196          0.000956      3.072370           58           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001          19                  133          0.000098      2.567116           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0001 0.000001          41                  287          0.000091      3.158644           71           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/primary_k4_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0010 0.000010          12                   84          0.000993      2.654710           42           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0001 0.000001          21                  147          0.000098      3.336143           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0010 0.000010          22                  154          0.000973      3.235546           52           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010           7                   49          0.000998      3.307104           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0010 0.000010           6                   42          0.000998      2.992010           36           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T078_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101

