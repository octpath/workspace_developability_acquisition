# EXP-T106 — AbLingua ARCH-6G geometry cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `ABLINGUA`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `True`
- config_hash: `6ba477646abddc990f80ce07c8a9a514ec094e7ab5838a947c96a5006cf5cc94`
- n_trainable: 551329
- VAL_P/S/mean/worst: 3.020849 / 3.089010 / 3.054930 / 3.089010

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          55                  385          0.000083      2.652823           85           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/primary_k0_lr1e-04_seed101.pt 466029b1a7d89082   101
primary     1               0.0001 0.000001          12                   72          0.000099      3.539083           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/primary_k1_lr1e-04_seed101.pt 466029b1a7d89082   101
primary     2               0.0010 0.000010          15                  105          0.000988      3.040616           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/primary_k2_lr1e-03_seed101.pt 466029b1a7d89082   101
primary     3               0.0001 0.000001          23                  161          0.000097      2.577449           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/primary_k3_lr1e-04_seed101.pt 466029b1a7d89082   101
primary     4               0.0010 0.000010           8                   56          0.000997      3.289580           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/primary_k4_lr1e-03_seed101.pt 466029b1a7d89082   101
 shadow     0               0.0001 0.000001          18                  126          0.000098      2.773189           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 466029b1a7d89082   101
 shadow     1               0.0001 0.000001          27                  189          0.000096      3.287119           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 466029b1a7d89082   101
 shadow     2               0.0001 0.000001          66                  462          0.000076      2.921690           96           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 466029b1a7d89082   101
 shadow     3               0.0010 0.000010          32                  224          0.000942      3.399441           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 466029b1a7d89082   101
 shadow     4               0.0010 0.000010          15                  105          0.000988      3.063780           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T106_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 466029b1a7d89082   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

