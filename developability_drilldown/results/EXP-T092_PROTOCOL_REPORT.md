# EXP-T092 — Scratch ARCH-2 joint single REG

- platform: `DL_FOLDLOCAL_COSINE_V3`
- arch_id: `ARCH-2`
- representation: `SCRATCH`
- content_mode / merge_mode: `scratch` / `None`
- config_hash: `37ec83967f4cc39c466d70726a8face46bf70eda6ed4d3c19702cedc8e33e255`
- n_trainable: 323969
- VAL_P/S/mean/worst: 2.902729 / 2.954482 / 2.928605 / 2.954482
- TEST_P/S/mean/worst: 3.155071 / 3.473059 / 3.314065 / 3.473059

## Selected LR

 scheme  fold  selected_initial_lr  eta_min  best_epoch  best_optimizer_step  lr_at_best_epoch  best_val_mae  final_epoch  stopped_early  numerical_failure                                                                                                                    checkpoint        init_hash  seed
primary     0               0.0010 0.000010          27                  189          0.000959      2.608250           57           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/primary_k0_lr1e-03_seed101.pt b2759a2692967759   101
primary     1               0.0010 0.000010          29                  174          0.000953      3.151349           59           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/primary_k1_lr1e-03_seed101.pt b2759a2692967759   101
primary     2               0.0010 0.000010          15                  105          0.000988      3.264123           45           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/primary_k2_lr1e-03_seed101.pt b2759a2692967759   101
primary     3               0.0010 0.000010          16                  112          0.000986      2.616481           46           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/primary_k3_lr1e-03_seed101.pt b2759a2692967759   101
primary     4               0.0010 0.000010          39                  273          0.000914      2.874874           69           True              False /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/primary_k4_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     0               0.0010 0.000010          38                  266          0.000919      2.588078           68           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/shadow_k0_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     1               0.0001 0.000001          63                  441          0.000078      3.189087           93           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/shadow_k1_lr1e-04_seed101.pt b2759a2692967759   101
 shadow     2               0.0010 0.000010          74                  518          0.000709      2.706663          104           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/shadow_k2_lr1e-03_seed101.pt b2759a2692967759   101
 shadow     3               0.0001 0.000001          40                  280          0.000091      3.507416           70           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/shadow_k3_lr1e-04_seed101.pt b2759a2692967759   101
 shadow     4               0.0010 0.000010          51                  357          0.000855      2.775335           81           True              False  /workspace_developability_acquisition/developability_drilldown/results/EXP-T092_run/checkpoints/shadow_k4_lr1e-03_seed101.pt b2759a2692967759   101

