# Organizer Top-Model Feature Bundle

## What this is

A compact set of **target-blind** features used by the strongest organizer-side Simple TVT CV recipes (Ridge/Lasso endgame).

## Files

See `feature_manifest.csv` for block → file mapping.

| file | role |
|------|------|
| `folds.csv` | fold_primary / fold_shadow |
| `recipes.csv` | Top-3 per target (CV rank frozen **before** Public/Private) |
| `data/base_sequences.parquet` | id, heavy, light |
| `data/*.parquet` | feature blocks |

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes

See `recipes.csv` (cv_rank 1–3 per target).

## Quick usage

```python
import pandas as pd
train = pd.read_csv("official_dev.csv")  # competition train table
feat = pd.read_parquet("data/ablang2.parquet")
train = train.merge(feat, on="id", how="left")
```

## Ridge / Lasso examples

- `examples/train_ridge.py`
- `examples/train_lasso.py`

## Important

- Fit scaler / PCA / imputation **inside training folds**
- Do **not** use Public/Private metadata for model selection
- Feature files are target-blind
- Some model-derived blocks may have redistribution conditions — see `license_status` in `feature_manifest.csv`
- Organizer Ridge may use PCA32 on AbLingua blocks; this bundle distributes **raw** embeddings

## Blocks in this release

- AROMATIC_TOPO
- AbLang2_HL_paired
- AbLingua_CDR3
- AbLingua_HL_mean
- BIOEMU_NEW_PAIRWISE
- CONTINUOUS_SURFACE
- ESM2_H
- HYDRO_FIELD
- M1_PROTEINMPNN
- SEQ_ALL
- SEQ_BASIC
- TITRATION_SHAPE
