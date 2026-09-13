# Figure 2 — Best configuration per representation

## What the figure directly shows

For each representation, the single lowest mean(P,S) cell (and separately worst(P,S)) among the 20 annotation×topology combinations. Overall best in this matrix: EXP-T205 = AbLang2 (separate-chain) / XREG / REGION (mean=2.994). Scratch best: REG-SEP/FULL. AbLang2 separate-chain best: XREG/REGION; AbLang2 paired H/L best: XREG/REGION.

## Strongest patterns

AbLang2 contexts occupy the leading end of the ranking. Best topology/annotation is not shared across representations (e.g., Scratch REG-SEP+FULL; ESM-C JOINT+BASE; ESM-2 FUSE+FULL; ESM-1b XREG+BASE).

## Robustness

Ordering by mean(P,S) vs worst(P,S) can differ for mid-ranked representations; small MAE gaps should not be treated as decisive without bootstrap.

## What we should NOT claim

That the top-ranked representation “contains structure” or is universally best outside this OOF matrix.

## Possible technical interpretation

Interpretation: optimal downstream inductive bias (topology/annotation) appears representation-dependent under a frozen training protocol.
