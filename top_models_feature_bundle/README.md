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

## Top recipes (advanced-model freeze; CV only)

### TmApp
1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV 2.7321/2.7850
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding
2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV 2.7466/2.8227
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding
3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV 2.7027/2.8459
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors

### HIC
1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV 0.4872/0.4834
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
2. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV 0.4822/0.4898
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv)
3. `HIC_ESM2_SEQ_AROMATIC__LASSO` — LASSO — CV 0.4927/0.4809
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv)

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes (advanced-model freeze; CV only)

### TmApp
1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV 2.7321/2.7850
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding
2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV 2.7466/2.8227
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding
3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV 2.7027/2.8459
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors

### HIC
1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV 0.4872/0.4834
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
2. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV 0.4822/0.4898
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv)
3. `HIC_ESM2_SEQ_AROMATIC__LASSO` — LASSO — CV 0.4927/0.4809
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv)

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes (advanced-model freeze; CV only)

### TmApp
1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV 2.7321/2.7850
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding
2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV 2.7466/2.8227
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding
3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV 2.7027/2.8459
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors

### HIC
1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV 0.4872/0.4834
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
2. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV 0.4822/0.4898
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv)
3. `HIC_ARO_CONT_TITR__LASSO` — LASSO — CV 0.4822/0.4912
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)

## Recommended folds

- `scheme == primary` → fold_primary
- `scheme == shadow` → fold_shadow

Rotation: TEST=k, VAL=(k+1)%5, TRAIN=other 3.

## Top recipes (advanced-model freeze; CV only)

### TmApp
1. `TM_PARENT_ABLINGUA_CDR3__RIDGE` — RIDGE — CV 2.7321/2.7850
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding + AbLingua-600M CDR3-guided residue pooling embedding
2. `TM_PARENT_ABLINGUA_GLOBAL__RIDGE` — RIDGE — CV 2.7466/2.8227
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors + AbLingua-600M masked-mean global heavy+light concatenated embedding
3. `TM_BASE_BIOEMU_MPNN__RIDGE` — RIDGE — CV 2.7027/2.8459
   Features: AbLang2 paired heavy+light chain protein-language-model embedding + stage-1 SEQ_BASIC sequence descriptors (length, composition, CDR length summaries) + BioEmu isolated VH/VL NEW_PAIRWISE ensemble pairwise C-alpha RMSD descriptors + ProteinMPNN sequence–structure compatibility native-score descriptors

### HIC
1. `HIC_HYDRO_TITRATION__LASSO` — LASSO — CV 0.4872/0.4834
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + HYDRO_FIELD continuous hydrophobic-field surface descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
2. `HIC_ARO_TITRATION__LASSO` — LASSO — CV 0.4894/0.4824
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + TITRATION_SHAPE electrostatic titration-shape descriptors (ESMFold Fv)
3. `HIC_ARO_CONTINUOUS_SURFACE__LASSO` — LASSO — CV 0.4822/0.4898
   Features: ESM-2 Heavy-chain protein-language-model embedding + stage-1 SEQ_ALL sequence descriptors (expanded antibody sequence descriptor set) + AROMATIC-TOPO exposed-aromatic structural topology descriptors (ESMFold Fv) + continuous molecular-surface hydrophobic/chemical descriptors (ESMFold Fv)

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


## Scope clarification
The earlier chat 'Organizer Top-3' referred only to the **restricted** endgame Ridge/Lasso feature-recipe inventory, **not** all historical organizer linear models. Historical HIC ~0.42 results were mainly **prediction blends of SVR bases**, tracked separately in `organizer_extension/linear_model_closure/`.
