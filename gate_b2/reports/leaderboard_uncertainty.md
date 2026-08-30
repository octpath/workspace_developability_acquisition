# Leaderboard uncertainty / ranking reliability

Primary metric: Spearman ρ. Uncertainty estimated by bootstrap resampling of Public/Private (see `metrics/bootstrap_results.csv`).

## Model-rank Spearman across canonical + 5 shadows

### HIC
- cv_to_public: mean=0.587 median=0.602 sd=0.211 min=0.317 max=0.839
- public_to_private: mean=0.591 median=0.611 sd=0.074 min=0.473 max=0.667
- cv_to_private: mean=0.687 median=0.703 sd=0.167 min=0.510 max=0.919

### TmApp
- cv_to_public: mean=0.820 median=0.798 sd=0.079 min=0.740 max=0.923
- public_to_private: mean=0.725 median=0.731 sd=0.095 min=0.592 max=0.853
- cv_to_private: mean=0.815 median=0.818 sd=0.079 min=0.693 max=0.926

## Bootstrap note

For finalist models, Public/Private Spearman 95% CI width is typically **~0.38–0.40**. With N≈60–90 per role, single-split Private winner claims are noisy; prefer win-rate across shadows + bootstrap pairwise win probabilities.

## Winner stability (Private #1)

- **HIC:** native ESMFold / fusion families dominate across shadows.
- **TmApp:** PLM families (AbLang2 / ESM-1b / ESM-2) dominate.

See also `winner_stability.md`.
