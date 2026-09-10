# EXP-T085 — AbLingua ARCH-6 ungated cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-6`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `4a42b9e85b566dba8b048498adc84b3c8904363e5473e9a55c91ed571e9c88ea`
- n_trainable: 551297
- VAL_P/S/mean/worst: 2.982046 / 3.047555 / 3.014801 / 3.047555
- TEST_P/S/mean/worst: 3.599715 / 3.344993 / 3.472354 / 3.599715

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          45                  315          0.000886      2.683165           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/primary_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     1               0.0001 0.000001          12                   72          0.000099      3.542422           42           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/primary_k1_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     2               0.0010 0.000010          80                  560          0.000665      2.842934          110           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/primary_k2_lr1e-03_seed101.pt f4efb3ed83fd3486   101
primary     3               0.0001 0.000001          23                  161          0.000097      2.577454           53           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/primary_k3_lr1e-04_seed101.pt f4efb3ed83fd3486   101
primary     4               0.0010 0.000010          37                  259          0.000923      3.256085           67           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/primary_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     0               0.0010 0.000010          14                   98          0.000990      2.684386           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/shadow_k0_lr1e-03_seed101.pt f4efb3ed83fd3486   101
 shadow     1               0.0001 0.000001          34                  238          0.000093      3.259918           64           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/shadow_k1_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     2               0.0001 0.000001          82                  574          0.000065      2.914995          112           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/shadow_k2_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     3               0.0001 0.000001          16                  112          0.000099      3.453801           46           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/shadow_k3_lr1e-04_seed101.pt f4efb3ed83fd3486   101
 shadow     4               0.0010 0.000010          34                  238          0.000935      2.923328           64           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T085_run/checkpoints/shadow_k4_lr1e-03_seed101.pt f4efb3ed83fd3486   101

