# antibody_transformer

Behavior-preserving extract of the competition antibody Transformer
(`AnnotatedTransformer` + `FeatureFusionModel` + CV helpers).

## Boundaries

- Stable: batch dict, `forward_repr`, `forward`, fusion fixed branch
- Not implemented here: dual PLM, distance bias, joint REG, multi-target

## Historical representations

Historical experiments do **not** archive pre-head representations.
Registry field: `representation_status = HISTORICAL_UNAVAILABLE`.

Future aggregation of seed-wise hidden states is **UNDECIDED**
(candidates only: `PER_SEED`, `CONCAT_SEEDS`, `CANONICAL_SEED`, `ALIGNED_AGGREGATION`).
Do not equate prediction seed-mean with hidden representation seed-mean.

## Residue assets

Referenced via `assets/transformer/residue_asset_manifest.yaml`
(pointing at immutable `top_models_feature_bundle/residue_level/`).
