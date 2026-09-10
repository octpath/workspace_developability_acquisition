# EXP-T077 — Joint H/L + unrestricted dual REG (T070-like)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch: `{'joint_hl_single_reg': False, 'joint_hl_dual_reg': True, 'joint_hl_chain_specific_dual_reg': False, 'use_cross_attention_bridge': False}`
- n_trainable: 501633
- VAL_P/S/mean/worst: 2.986835 / 3.024182 / 3.005509 / 3.024182
- TEST_P/S/mean/worst: 3.372782 / 3.369448 / 3.371115 / 3.372782

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          11                   77          0.000994      2.716992           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/primary_k0_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     1               0.0010 0.000010           4                   24          0.000999      3.540800           34           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/primary_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     2               0.0010 0.000010          45                  315          0.000886      2.857167           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/primary_k2_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
primary     3               0.0001 0.000001          20                  140          0.000098      2.634851           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/primary_k3_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
primary     4               0.0010 0.000010          20                  140          0.000978      3.175487           50           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/primary_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     0               0.0001 0.000001          27                  189          0.000096      2.664221           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     1               0.0010 0.000010          26                  182          0.000962      3.334725           56           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     2               0.0001 0.000001          54                  378          0.000084      2.886685           84           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 1e0cea8ee76ae170   101
 shadow     3               0.0010 0.000010          18                  126          0.000982      3.191524           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 1e0cea8ee76ae170   101
 shadow     4               0.0010 0.000010          16                  112          0.000986      3.049773           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T077_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 1e0cea8ee76ae170   101

