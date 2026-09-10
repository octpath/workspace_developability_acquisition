# EXP-T082 — AbLingua ARCH-4 joint chain-specific dual REG MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-4`
- representation: `ABLINGUA`
- content_mode / merge_mode: `frozen` / `mean`
- config_hash: `51c3bd927549f617846daec5c3dbb9101dbf0cb762b3eee16ae3358ac2059149`
- n_trainable: 485249
- VAL_P/S/mean/worst: 3.021031 / 3.008105 / 3.014568 / 3.021031
- TEST_P/S/mean/worst: 3.243248 / 3.364139 / 3.303693 / 3.364139

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          11                   77          0.000994      2.646109           41           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/primary_k0_lr1e-03_seed101.pt 649131418599f207   101
primary     1               0.0001 0.000001          17                  102          0.000098      3.538292           47           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/primary_k1_lr1e-04_seed101.pt 649131418599f207   101
primary     2               0.0001 0.000001          33                  231          0.000094      3.065527           63           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/primary_k2_lr1e-04_seed101.pt 649131418599f207   101
primary     3               0.0010 0.000010          14                   98          0.000990      2.532718           44           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/primary_k3_lr1e-03_seed101.pt 649131418599f207   101
primary     4               0.0010 0.000010           8                   56          0.000997      3.318060           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/primary_k4_lr1e-03_seed101.pt 649131418599f207   101
 shadow     0               0.0010 0.000010          15                  105          0.000988      2.716362           45           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/shadow_k0_lr1e-03_seed101.pt 649131418599f207   101
 shadow     1               0.0001 0.000001          32                  224          0.000094      3.321032           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 649131418599f207   101
 shadow     2               0.0001 0.000001         118                  826          0.000037      2.962714          148           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/shadow_k2_lr1e-04_seed101.pt 649131418599f207   101
 shadow     3               0.0010 0.000010          20                  140          0.000978      3.258433           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/shadow_k3_lr1e-03_seed101.pt 649131418599f207   101
 shadow     4               0.0010 0.000010          45                  315          0.000886      2.783275           75           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T082_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 649131418599f207   101

