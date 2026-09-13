# Figure 6 — Paired vs separate PLM context

## What the figure directly shows

For matched checkpoints, heatmaps of MAE(paired H/L context) − MAE(separate-chain context) over annotation×topology. Negative = paired context better. AbLang2 and CurrAb share the same color scale. Stars mark primary paired-bootstrap CIs excluding zero (descriptive).

## Strongest patterns

CurrAb shows a more consistently negative (paired-better) field on average. AbLang2 is more mixed by annotation/topology; best cells for both AbLang2 contexts remain near XREG+REGION. Average paired advantage and best-cell near-ties can coexist (especially CurrAb).

## Robustness

Use bootstrap markers cautiously; many cells will have CIs including zero. Require Primary/Shadow agreement for stronger wording.

## What we should NOT claim

That paired inference “learned the interface,” or that downstream XREG substitutes for PLM-internal pairing in a mechanistic sense.

## Possible technical interpretation

Interpretation: holding weights fixed, changing only inference-time H/L context alters the downstream error surface; AbLang2 and CurrAb do not show identical PAIR−SEPARATE maps.
