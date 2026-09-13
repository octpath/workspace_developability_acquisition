# Figure 5 — Annotation effects vs BASE

## What the figure directly shows

ΔMAE = MAE(annotation) − MAE(BASE) for IMGT, REGION, and FULL, stratified by representation and topology. Sequence-position and chain-ID embeddings are present in all four annotation conditions.

## Strongest patterns

Annotation effects are not uniform: some representation×topology cells improve with REGION/FULL, others worsen. Topology dependence is visible (columns differ within a row).

## Robustness

Check annotation-gain bootstrap CIs before treating small signed deltas as stable. Primary/Shadow agreement matters for claims.

## What we should NOT claim

That a near-zero IMGT delta means “IMGT is already encoded in the PLM,” or that REGION gains mean the model learned CDR biology.

## Possible technical interpretation

Interpretation: explicit IMGT/CDR-FR features are an optional inductive bias whose value depends on both the residue representation and the H/L topology.
