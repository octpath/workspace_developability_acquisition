# EXP-T117 — AbLang2 ARCH-6 ungated residue cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `False`
- config_hash: `a47918192108f2500c06209886a3b51f9a9cacd6bcc644d65a2dd471f2921173`
- n_trainable: 448897
- VAL_P/S/mean/worst: 2.814131 / 2.944394 / 2.879263 / 2.944394

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          49                  343          0.000087      2.576812           79           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/primary_k0_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     1               0.0010 0.000010           3                   18          0.001000      3.605009           33           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/primary_k1_lr1e-03_seed101.pt 8d6e8476a7702db3   101
primary     2               0.0010 0.000010          27                  189          0.000959      2.913207           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8d6e8476a7702db3   101
primary     3               0.0001 0.000001          45                  315          0.000089      2.414068           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8d6e8476a7702db3   101
primary     4               0.0010 0.000010          25                  175          0.000965      2.544261           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     0               0.0001 0.000001          59                  413          0.000081      2.647489           89           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8d6e8476a7702db3   101
 shadow     1               0.0010 0.000010          17                  119          0.000984      2.879273           47           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     2               0.0010 0.000010          40                  280          0.000910      2.952123           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8d6e8476a7702db3   101
 shadow     3               0.0100 0.000100           8                   56          0.009970      3.517035           38           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 8d6e8476a7702db3   101
 shadow     4               0.0010 0.000010          20                  140          0.000978      2.717434           50           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T117_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8d6e8476a7702db3   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

