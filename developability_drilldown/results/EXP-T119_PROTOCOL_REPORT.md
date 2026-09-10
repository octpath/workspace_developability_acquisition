# EXP-T119 — AbLang2 ARCH-6G geometry cross-attn MEAN

- platform: `DL_FOLDLOCAL_COSINE_V3`
- target: `TmApp`
- arch_id: `ARCH-6G`
- representation: `ABLANG2`
- content_mode / merge / chain: `frozen` / `mean` / `HL`
- geometry: `True`
- config_hash: `b720f35b89e98c147fe862c8e77cc52833163767f1abb21a4e3d199bb33f117d`
- n_trainable: 448929
- VAL_P/S/mean/worst: 2.825817 / 2.869082 / 2.847450 / 2.869082

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0001 0.000001          48                  336          0.000087      2.758269           78           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/primary_k0_lr1e-04_seed101.pt 8cc088ad541eae99   101
primary     1               0.0100 0.000100           8                   48          0.009970      3.542986           38           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/primary_k1_lr1e-02_seed101.pt 8cc088ad541eae99   101
primary     2               0.0010 0.000010          30                  210          0.000950      2.828467           60           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/primary_k2_lr1e-03_seed101.pt 8cc088ad541eae99   101
primary     3               0.0001 0.000001          45                  315          0.000089      2.414034           75           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/primary_k3_lr1e-04_seed101.pt 8cc088ad541eae99   101
primary     4               0.0010 0.000010          25                  175          0.000965      2.565029           55           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/primary_k4_lr1e-03_seed101.pt 8cc088ad541eae99   101
 shadow     0               0.0001 0.000001          60                  420          0.000080      2.589819           90           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/shadow_k0_lr1e-04_seed101.pt 8cc088ad541eae99   101
 shadow     1               0.0010 0.000010          18                  126          0.000982      2.820397           48           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/shadow_k1_lr1e-03_seed101.pt 8cc088ad541eae99   101
 shadow     2               0.0010 0.000010          48                  336          0.000871      2.619846           78           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/shadow_k2_lr1e-03_seed101.pt 8cc088ad541eae99   101
 shadow     3               0.0100 0.000100          14                   98          0.009897      3.515555           44           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/shadow_k3_lr1e-02_seed101.pt 8cc088ad541eae99   101
 shadow     4               0.0010 0.000010          27                  189          0.000959      2.788319           57           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T119_run/checkpoints/shadow_k4_lr1e-03_seed101.pt 8cc088ad541eae99   101

_OOF TEST comparative rankings deferred until batch PRE_EXTERNAL_FREEZE._

