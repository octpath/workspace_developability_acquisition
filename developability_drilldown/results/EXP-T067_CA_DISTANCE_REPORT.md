# EXP-T067 — Learnable Cα-distance attention bias

- git: `f79b0b0cbb7a69f23d74ef36a41645d617e3ff3c`
- control: `EXP-T037` (stored canonical OOF; not retrained)
- change: per-head additive attention bias `a_h * exp(-d_ij / ell_h)` from within-chain Cα distances
- topology: H and L encoded separately (no H–L cross-chain distance)
- CV verdict: **NEGATIVE**

## 1–4. vs EXP-T037

- T067 P/S/mean/W: 2.753816 / 2.772998 / 2.763407 / 2.772998
- Δ Primary: +0.000001
- Δ Shadow: +0.000000
- Δ worst: +0.000000
- Both schemes same direction: YES

## 5. Paired bootstrap (MAE_T067 − MAE_T037)

- Primary: Δ=+0.000001 CI95=[-0.000001, +0.000002]
- Shadow: Δ=+0.000000 CI95=[-0.000000, +0.000001]

## 6. Classical ceiling EXP-T045

- T045 worst: 2.728948
- T067−T045 worst: +0.044050

## 7–8. Learned distance parameters

- mean |a_h|: 0.003848674372011374
- median ell_h (Å): 17.10286521911621
- by head: `{"0": {"mean_abs_a": 0.0038843191251241175, "median_ell": 17.102943420410156}, "1": {"mean_abs_a": 0.0028473856427202312, "median_ell": 17.102730751037598}, "2": {"mean_abs_a": 0.004348070093935045, "median_ell": 17.103328704833984}, "3": {"mean_abs_a": 0.004314922626266101, "median_ell": 17.102746963500977}}`
- full table: `results/EXP-T067_DISTANCE_PARAMETERS.csv`

## 9. Consistency / four-way comparison

| EXP | role | Primary | Shadow | worst |
|-----|------|---------|--------|-------|
| T037 | no geometry | 2.753816 | 2.772998 | 2.772998 |
| T065 | token-additive RASA | 2.814704 | 2.874607 | 2.874607 |
| T066 | aggregation RASA | 2.773911 | 2.903064 | 2.903064 |
| T067 | Cα distance bias | 2.753816 | 2.772998 | 2.772998 |

## 10–11. Data-driven bias curves / neighbor–local–nonlocal

- diagnostic: `results/EXP-T067_run/distance_usage_diagnostic.json`
- distance quantiles: `{"q10": 8.658373641967774, "q25": 12.362141132354736, "q50": 17.11317539215088, "q75": 22.441428661346436, "q90": 27.762595939636235}`
- bias at quantiles (summary): `{"q10": 0.00231978835727973, "q25": 0.0018680921172885884, "q50": 0.0014149949143125351, "q75": 0.0010362250804876148, "q90": 0.0007591598921670238}`
- category mean bias: `{"sequence_neighbor_|i-j|=1": 0.0030855463161634728, "local_1<|i-j|<=4": 0.002302309943472548, "nonlocal_|i-j|>4": 0.0013086515802142112}`
- apparent dominant category: `sequence_neighbor_|i-j|=1`

T067 cannot assess H–L cross-chain geometry.

## Test (after freeze)

- Public: 3.263940
- Private: 3.281434
- Overall: 3.272687

## Cα QC

- expected/mapped/missing: 75057/75057/0
- AA mismatches: 0
- structure_source: /workspace_developability_acquisition/feature_extension/data/esmfold_fv
- hashes: {'ca_heavy.npy': '0e58df5fe4d5dc1112db8630fa6067d5b13088ca18d769bd1c44c74717a04eae', 'ca_light.npy': 'e38e3e9c57c40a22a19cafb16878d3f1a6e139cf2733963c08d2fa7f6ac4a902', 'ca_ids.npy': 'ce62b155c7355352d9941f19d3be50263c6835ed0f3ce3429a77f9b0714c665d', 'ca_meta.json': 'cc44895989348c89563b574cdb3cab0495cbdb921b2789fbfb2b9e1e6a0a13a8'}

## Artifacts / reproducibility

- config / features / CA parquet / preds present
- representation: NOT_EXPORTED
- checkpoints: n=3 under `results/EXP-T067_run/checkpoints/` (total_bytes=6759324)
- SHAREABLE_COMPLETE / REPRODUCED / eligible=YES

## Interpretation

CV verdict **NEGATIVE**. Explicit pairwise Cα geometry is a different mechanism from RASA (T065/T066). Do not over-claim from a single experiment; within-chain only.

## STOP — no EXP-T068 / sweeps
