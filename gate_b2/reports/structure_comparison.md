# Structure comparison — ABB vs native ESMFold (and B1 HF/linker)

- Compared antibodies: 370
- ABB COM dist mean±sd: 22.16 ± 0.41 Å
- Native ESMFold COM dist mean±sd: 22.09 ± 0.43 Å
- ΔCOM (ESMN−ABB) mean: -0.07 Å
- Fv CA RMSD ABB vs ESMN (n=100): mean 1.11 Å
- Fv CA RMSD B1-HF/linker vs native ESMN (n=100): mean 0.29 Å

## Interpretation

Purpose is not structure benchmarking; ask whether geometry differences are large enough
to change downstream developability features. See surface/feature ablations for predictive impact.

ESMFold supports multimer inputs through colon-separated chains and chain-aware inference
machinery; it is not a true AlphaFold-Multimer model.

- |Δ| Fv_total_sasa mean=478.79 spearman=0.783
- |Δ| BSA mean=146.90 spearman=0.591
- |Δ| largest_hydrophobic_patch_sasa mean=35.09 spearman=0.855
- |Δ| Fv_mean_rasa mean=0.01 spearman=0.579
