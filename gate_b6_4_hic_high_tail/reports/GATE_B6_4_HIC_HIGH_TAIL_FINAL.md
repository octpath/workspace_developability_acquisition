# Gate B6.4 — HIC High-Tail Mechanism & Learnability Final Report

Overall verdict on HIGH-HIC learnability:

**HIC_VALID_BUT_DATA_LIMITED_COMPETITION_TARGET**

HIGH cohort:
- total: 13
- Train: 6
- Public: 3
- Private: 4
- consensus severe failures: 10

Data integrity:
- source values verified: 13/13 exact HIC vs mmc2.xlsx; 13/13 sequence matches
- suspicious records: 0

Sequence-space support:
- HIGH nearest-Train paired identity (median): 0.761
- LOW comparison: 0.727
- near-identical discordant pairs (sim≥0.95 & |ΔHIC|≥1.5): n=0

Main sequence signals:
- strongest interpretable features: ESMFN_Fv_sasa_aromatic, H_CDR3_frac_aromatic, VH_frac_hydrophobic, H_CDR3_frac_hydrophobic, H_CDR3_len
- CDRH3 signal: H_CDR3_frac_aromatic(δ=0.61), H_CDR3_frac_hydrophobic(δ=0.57), H_CDR3_len(δ=0.52)
- charge/pI: see metrics/high_hic_feature_effects.csv
- SHM signal: WEAK — VL mutfrac/germline-distance slightly lower in HIGH (cliffsδ≈−0.33); Spearman vs HIC ≈ −0.11

Main structure signals:
- exposed hydrophobic SASA / patch metrics: see metrics/high_hic_structure_features.csv
- hydrophobic patch signal: cliffsδ(top)=0.705
- structure-model advantage: ESMFN closer than PLM in 8/13 HIGH cases

Why current models underpredict HIGH:
- regression-to-center: STRONG
- sample scarcity: STRONG (Train HIGH=6; total=13)
- missing feature signal: MODERATE
- sequence novelty: WEAK
- unexplained component: WEAK

Diagnostic model experiment:
- best Train-only model: PHYSSEQ_STRUCT_Ridge_elevW2
- overall MAE: 0.5866
- HIGH MAE: 2.1663
- HIGH→LOW: 0.667
- elevated recognition (≥10.5): 0.278
- vs frozen NESTED: HIGH→LOW 0.667 → 0.667; nested MAE=0.4667

One-shot Public/Private confirmation:
- Public: MAE=0.6807, HIGH→LOW=0.667, elev≥10.5=0.25
- Private: MAE=0.5597, HIGH→LOW=0.5, elev≥10.5=0.4

Hypothesis ratings:
- H1 DATA/LABEL ISSUE: NOT SUPPORTED
- H2 SPARSE-TAIL / REGRESSION-TO-CENTER: STRONG
- H3 DOMAIN / SEQUENCE-NOVELTY: WEAK
- H4 MISSED PHYSICOCHEMICAL SIGNAL: MODERATE
- H5 STRUCTURAL SIGNAL: MODERATE
- H6 SHM / GERMLINE-DEVIATION SIGNAL: WEAK
- H7 NON-SEQUENCE / ASSAY-SPECIFIC COMPONENT: WEAK

Is HIGH-HIC information sequence/structure learnable:
**Partially yes** — physchem/surface features separate HIGH imperfectly; Train-only diagnostics can reduce HIGH→LOW vs frozen models, but n=13 limits stability and many severe failures remain underpredicted.

Is there credible participant headroom:
**Yes, modest and data-limited** — not strong evidence of a large untapped ceiling.

Recommended HIC competition status:
**HIC_VALID_BUT_DATA_LIMITED_COMPETITION_TARGET**

Reason:
Labels are intact; HIGH is a real smooth right tail. Difficulty is dominated by scarcity + regression-to-center, with only partial missing-feature signal and some residual/assay-associated unexplained variance. Keep continuous HIC with explicit sparse-tail caveats; do not convert to classification; do not reopen split search.

---

## Answers (Q1–35)

1. Yes — 13/13 HIC and sequences match mmc2.xlsx.
2. Smooth right-tail continuation above 11.5, not a suspicious isolated spike cluster.
3. HIGH median nearest-Train sim 0.761 vs LOW 0.727 — not clearly more distant.
4. HIGH spans multiple sequence groups/families (see case table); not a single trivial clique.
5. Near-identical discordant pairs: n=0.
6. Local |ΔHIC| decreases at high similarity vs random pairs, but residual differences remain (smoothness table).
7. Strongest simple separators: ESMFN_Fv_sasa_aromatic, H_CDR3_frac_aromatic, VH_frac_hydrophobic, H_CDR3_frac_hydrophobic, H_CDR3_len.
8. CDRH3 hydrophobicity: H_CDR3_frac_aromatic(δ=0.61), H_CDR3_frac_hydrophobic(δ=0.57), H_CDR3_len(δ=0.52).
9. Aromatic fractions appear among predeclared features; rank in feature_effects.
10. Charge/pI present in feature_effects (HL_charge / HL_pI).
11. Germline-distance/mutfrac Spearman: see shm table — support WEAK.
12. Per-mutation hydrophobicity-up SHM: **not computed** (no frozen germline AA alignments).
13. Exposed hydrophobicity-up mutations: **not computed** (same limitation).
14. Exposed hydrophobic SASA: see structure table (HIGH vs nonHIGH medians/cliffs).
15. Hydrophobic patches: largest/total patch SASA compared in structure metrics.
16. Structure helps some PLM misses (8/13 closer) but does not fully solve HIGH→LOW.
17. ESMFN advantage likely reflects surface/hydrophobic descriptors vs pooled embeddings (prediction + surface evidence); frozen coefficient objects not re-exported.
18. Disagreement cases listed in metrics/high_hic_model_disagreement.csv.
19. Consensus failures often still underpredicted despite some elevated hydrophobicity/patch metrics — scarcity/shrinkage dominates.
20. PSR/TmApp correlations: metrics/high_hic_assay_correlations.csv — not an exclusive explanation.
21. B-cell subset counts: metrics/high_hic_bcell_subset_counts.csv; donor mostly UNK.
22. Yes — scarcity/regression-to-center is a primary mechanism (H2 STRONG).
23. Also partly missing physchem/structure emphasis (H4/H5), not only scarcity.
24. Explicit physchem/structure features can improve Train-CV HIGH diagnostics (winner=PHYSSEQ_STRUCT_Ridge_elevW2).
25. Structure-surface features contribute in PHYSSTRUCT / combined specs.
26. Predeclared elev-weight w=2 tested; selected? yes.
27. Winner constrained to overall MAE ≤ nested+0.15.
28. One-shot holdout above (Public HIGH n=3, Private n=4) — confirmation only.
29. Organizer oracle MAE=0.5429 vs nested=0.4667.
30. Discordant near-identical pairs + residual underprediction support nontrivial unexplained component (H7 WEAK).
31. Consensus failures are understandable as scarce-tail shrinkage + incomplete surface/physchem capture — not label errors.
32. Yes — continuous HIC remains scientifically meaningful.
33. Difficulty is **useful but data-limited headroom**, not pure liability.
34. Yes — keep continuous HIC track with caveats (not drop; not classify).
35. Caveats: sparse elevated/HIGH examples; MAE favors center; high-HIC developability cases hard; Pearson fragile to tail leverage; 10.5/11.5 bands interpretive only.

## Decision

**HIC_VALID_BUT_DATA_LIMITED_COMPETITION_TARGET**
