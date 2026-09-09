# Transformer residue / feature license notes

Status values in the drilldown registry use the compact enum:

- `OK`
- `REVIEW`
- `RESTRICTED`
- `UNKNOWN`

Source assets often carry finer labels (`REVIEW_MODEL_OUTPUT`, `OK_COMPETITION_DERIVED`, `SEE_FEATURE_EXTENSION`).

## Aggregation rule

Each TRANSFORMER experiment gets a **conservative aggregate** over the assets it actually uses:

1. Scratch AA sequences + IMGT annotations → typically map toward `OK` when competition-derived.
2. AbLingua / AbLang2 / ESM2 residue embeddings → `REVIEW` (do not promote to `OK` without new legal evidence).
3. Fusion fixed-length `FS_*` branches → inherit Phase 1 Feature-set caution (`REVIEW` when any PLM/extension block is involved).

Uncertain cases stay `UNKNOWN` or `REVIEW`. Never upgrade to `OK` without evidence.

See `residue_asset_manifest.yaml` and `top_models_feature_bundle/residue_level/RESIDUE_ASSET_AUDIT.md`.
