# FENNIX_PARTICIPANT_RELEASE_AUDIT

## ready for participant FeNNix block: YES (with review flags)

| Check | Result |
|---|---|
| Final assembly N=323 | PASS |
| ADI-47265 excluded + documented | PASS |
| Target-blind QC | QC_PASS_WITH_LIMITATIONS (fixed site-count zero-variance cols) |
| Fresh-worker repro | NUMERICALLY_CLOSE (C/M exact; B FIRE drift, corr≥0.99999) |
| Labels / OOF / Public-Private in parquet | PASS (none) |
| License | FeNNol **LGPL-3.0**; decision `PARTICIPANT_ONLY_REVIEW_RECOMMENDED` |
| Dev-CV claim | Do **not** advertise interim CONSTANT win; full cohort DID_NOT_REPLICATE |

## Artifact

- `feature_extension/data/precomputed_features/fennix_fab_context.parquet`
- dictionary CSV alongside
- v1.1 zips via `tools/package_v1_1.py`

## Competition implication

All FeNNix families + BIOEMU_NEW_CONTACT+CONSTANT → **COMP_DROP** on full usable Dev Simple TVT.
Still released as experimental target-blind structure descriptors.
