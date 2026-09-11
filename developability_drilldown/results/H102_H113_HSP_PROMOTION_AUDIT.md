# H102–H113 HSP Promotion Audit

Source commit: `210a270d`
Source track: `feature_research/hic_spatial_hydrophobicity/`

## Ambiguity check

Frozen promotion (`PROMOTION_RECOMMENDATION.md`) explicitly specifies **B3**
(ALL_FV MAX / MEAN / SUM) for all three promoted families.

Stage-1 shortlist ranking and Stage-2 confirmation for GENERIC families
were computed with `bundle == "B3"` in `run_screen.py`.

**Conclusion: column set is uniquely identified. No STOP.**

## P1 — `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0`

- feature_set_id: `FS_HIC_HSP_BM_R5_PROMOTED`
- surface_plus: `FS_HIC_SURFACE_PLUS_HSP_BM_R5_PROMOTED`
- bundle_id: **B3**
- dim: **3**
- columns:
  - `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0__ALL_FV__MAX`
  - `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0__ALL_FV__MEAN`
  - `HSP_BM_RAW_TOTAL_RASA_TIEN_CLOSEST_SC_R5p0__ALL_FV__SUM`
- aggregation scope: ALL_FV
- aggregations: MAX, MEAN, SUM
- Stage-1 contexts: A_ALONE / B_SURFACE / C_PHYS (B3)
- Stage-2 contexts: A_ALONE / B_SURFACE (B3)
- source feature-spec hash: `c58f5a41cb071409`
- property table hash: `91c7bdd90bcd5740`
- structure hash: `1c14c6eab5771b45cf6bb8585ba5c6408caf34f092e88136995e118220289674`
- mainline feature hash: `831d72170d5bd5c7`
- parquet: `experiments/features/hsp_bm_r5_promoted.parquet`

## P2 — `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0`

- feature_set_id: `FS_HIC_HSP_FP_R5_PROMOTED`
- surface_plus: `FS_HIC_SURFACE_PLUS_HSP_FP_R5_PROMOTED`
- bundle_id: **B3**
- dim: **3**
- columns:
  - `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0__ALL_FV__MAX`
  - `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0__ALL_FV__MEAN`
  - `HSP_FP_MINMAX_SIDECHAIN_SASA_ABS_CLOSEST_SC_R5p0__ALL_FV__SUM`
- aggregation scope: ALL_FV
- aggregations: MAX, MEAN, SUM
- Stage-1 contexts: A_ALONE / B_SURFACE / C_PHYS (B3)
- Stage-2 contexts: A_ALONE / B_SURFACE (B3)
- source feature-spec hash: `8fe429d458cc0154`
- property table hash: `5c245bdc00d9f7ed`
- structure hash: `1c14c6eab5771b45cf6bb8585ba5c6408caf34f092e88136995e118220289674`
- mainline feature hash: `02a25f5f72761516`
- parquet: `experiments/features/hsp_fp_r5_promoted.parquet`

## P3 — `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0`

- feature_set_id: `FS_HIC_HSP_EIS_R8_PROMOTED`
- surface_plus: `FS_HIC_SURFACE_PLUS_HSP_EIS_R8_PROMOTED`
- bundle_id: **B3**
- dim: **3**
- columns:
  - `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0__ALL_FV__MAX`
  - `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0__ALL_FV__MEAN`
  - `HSP_EIS_RAW_SIDECHAIN_SASA_ABS_CENTROID_R8p0__ALL_FV__SUM`
- aggregation scope: ALL_FV
- aggregations: MAX, MEAN, SUM
- Stage-1 contexts: A_ALONE / B_SURFACE / C_PHYS (B3)
- Stage-2 contexts: A_ALONE / B_SURFACE (B3)
- source feature-spec hash: `faf5f8cd5c526965`
- property table hash: `18538577f6210fc6`
- structure hash: `1c14c6eab5771b45cf6bb8585ba5c6408caf34f092e88136995e118220289674`
- mainline feature hash: `2be692a559b23c6e`
- parquet: `experiments/features/hsp_eis_r8_promoted.parquet`
