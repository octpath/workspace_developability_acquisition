# Pearson evaluation protocol (Gate B5)

## Definitions

- **prediction Pearson r**: Pearson correlation between predicted and measured assay values across antibodies.
- **model-ranking transfer rho**: Spearman/Kendall of *model ranks* across CV/Public/Private (separate diagnostic).

## Summaries computed

1. **Fold-level Pearson** — per outer validation fold; report mean/median/SD/min/max and **Fisher-z mean**.
2. **Repeat-level pooled OOF Pearson** — one OOF vector of length 162 per repeat.
3. **Aggregated repeated-OOF Pearson** — mean OOF across repeats, then one Pearson on Train=162 (descriptive).

## Uncertainty

Fisher-z CI and/or bootstrap (≥5000) on aggregated OOF and holdouts.

## Primary objective

Production selection remains **minimize MAE**. Pearson is a high-priority secondary diagnostic.
