# EXP-T069 — Joint H/L single-REG + Cα-distance bias

- git: `ecc67b3a67d14058f8797d7524a5d1fe34a8513b`
- primary control: `EXP-T068`
- secondary: `EXP-T030-REPLAY-001` and historical `EXP-T030`
- change: joint `[REG,H...,L...]` + per-head `a_h * exp(-d_ij / ell_h)` over full Fv (incl. H–L)
- CV verdict: **POSITIVE**

## Primary vs EXP-T068

- T069 P/S/mean/W: 3.535941 / 3.334679 / 3.435310 / 3.535941
- Δ Primary: -0.003385
- Δ Shadow: +0.029057
- Δ worst: -0.003385
- Both schemes same direction: NO

## Secondary deltas

- vs replay Primary/Shadow/worst: +0.303264 / +0.132721 / +0.303264
- vs historical T030 Primary/Shadow/worst: +0.218172 / +0.120069 / +0.218172

## Paired bootstrap vs T068

- Primary: Δ=-0.003385 CI95=[-0.062121, +0.055972]
- Shadow: Δ=+0.029057 CI95=[-0.032787, +0.090462]

## Learned distance parameters

- mean |a_h|: 0.0033882775682589758
- median ell_h (Å): 22.41041374206543
- by head: `{"0": {"mean_abs_a": 0.004308338894043117, "median_ell": 22.410442352294922}, "1": {"mean_abs_a": 0.002806243170925882, "median_ell": 22.410484313964844}, "2": {"mean_abs_a": 0.0031675889621207414, "median_ell": 22.410423278808594}, "3": {"mean_abs_a": 0.003270939245946162, "median_ell": 22.41038227081299}}`
- full table: `results/EXP-T069_DISTANCE_PARAMETERS.csv`

## Geometry diagnostic (incl. H–L)

- diagnostic: `results/EXP-T069_run/distance_usage_diagnostic.json`
- category counts: `{"sequence_neighbor_|i-j|=1": 74409, "local_1<|i-j|<=4": 219339, "nonlocal_|i-j|>4": 4033746, "H-L_cross_chain": 4331952}`
- category mean bias: `{"sequence_neighbor_|i-j|=1": 0.0028624044525114847, "local_1<|i-j|<=4": 0.0022891829050252404, "nonlocal_|i-j|>4": 0.0014874608397147636, "H-L_cross_chain": 0.0009722465058033893}`
- apparent dominant category: `sequence_neighbor_|i-j|=1`

## Test (after freeze)

- Public: 3.570976
- Private: 3.148981
- Overall: 3.359979

## Cα QC

- expected/mapped/missing: 75057/75057/0
- AA mismatches: 0

## Artifacts

- config / CA parquet / preds; feature_path EMPTY
- checkpoints: n=3 under `results/EXP-T069_run/checkpoints/`
- checkpoint manifest: `results/EXP-T069_CHECKPOINT_MANIFEST.json`
- SHAREABLE_COMPLETE / REPRODUCED

CV verdict **POSITIVE**.
