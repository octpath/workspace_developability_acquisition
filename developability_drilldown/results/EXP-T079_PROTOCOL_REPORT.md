# EXP-T079 — Separate H/L + bidirectional residue cross-attn bridge + dual REG (T072-like)

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch: `{'joint_hl_single_reg': False, 'joint_hl_dual_reg': False, 'joint_hl_chain_specific_dual_reg': False, 'use_cross_attention_bridge': True}`
- n_trainable: 567683
- VAL_P/S/mean/worst: 2.914277 / 3.004834 / 2.959555 / 3.004834
- TEST_P/S/mean/worst: 3.236323 / 3.267068 / 3.251696 / 3.267068

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          37                  259          0.000092      2.710728           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/primary_k0_lr1e-04_seed101.pt 8683f4dfb3d9116d   101
primary     1               0.0010 0.000010           7                   42          0.000998      3.132817           37           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8683f4dfb3d9116d   101
primary     2               0.0001 0.000001          25                  175          0.000097      3.088696           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/primary_k2_lr1e-04_seed101.pt 8683f4dfb3d9116d   101
primary     3               0.0001 0.000001          19                  133          0.000098      2.570647           49           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8683f4dfb3d9116d   101
primary     4               0.0010 0.000010          26                  182          0.000962      3.068026           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8683f4dfb3d9116d   101
 shadow     0               0.0001 0.000001          24                  168          0.000097      2.579108           54           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8683f4dfb3d9116d   101
 shadow     1               0.0010 0.000010           8                   56          0.000997      3.216632           38           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8683f4dfb3d9116d   101
 shadow     2               0.0010 0.000010          14                   98          0.000990      3.187544           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8683f4dfb3d9116d   101
 shadow     3               0.0010 0.000010           9                   63          0.000996      3.118899           39           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 8683f4dfb3d9116d   101
 shadow     4               0.0010 0.000010           7                   49          0.000998      2.931727           37           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T079_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8683f4dfb3d9116d   101

