# EXP-T120 — AbLang2 ARCH-7 REG-only cross-attn CONCAT

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-7`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `concat` / `HL`
- geometry: `False`
- config_hash: `a2727bbc06c1d477207e129b91216c79f32c245ce84a016235b461dd4ee3a589`
- n_trainable: 465281
- VAL_P/S/mean/worst: 2.701890 / 2.870237 / 2.786063 / 2.870237

## Selected LR

 scheme  fold  selected_initial_lr      eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0              0.00100 1.000000e-05          44                  308          0.000891      2.501995           74           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/primary_k0_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     1              0.00100 1.000000e-05          26                  156          0.000962      3.347585           56           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/primary_k1_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     2              0.00100 1.000000e-05          38                  266          0.000919      2.878881           68           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/primary_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
primary     3              0.00010 1.000000e-06          75                  525          0.000070      2.252493          105           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/primary_k3_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
primary     4              0.00100 1.000000e-05          71                  497          0.000730      2.514565          101           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/primary_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     0              0.00010 1.000000e-06          51                  357          0.000086      2.457770           81           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
 shadow     1              0.00010 1.000000e-06          63                  441          0.000078      2.789933           93           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/shadow_k1_lr1e-04_seed101.pt 5ab9c53eb35c587c   101
 shadow     2              0.00100 1.000000e-05          32                  224          0.000942      3.008220           62           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 5ab9c53eb35c587c   101
 shadow     3              0.00001 1.000000e-07           2                   14          0.000010      3.548414           32           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/shadow_k3_lr1e-05_seed101.pt 5ab9c53eb35c587c   101
 shadow     4              0.00100 1.000000e-05          21                  147          0.000976      2.538543           51           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T120_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 5ab9c53eb35c587c   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

