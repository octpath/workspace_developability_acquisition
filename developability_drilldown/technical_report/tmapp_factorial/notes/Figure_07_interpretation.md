# Figure 7 — Primary / Shadow robustness

## What the figure directly shows

Scatter of all 200 cells in Primary vs Shadow MAE space, plus mean(P,S) vs |P−S|. Overall best and high-disagreement cells are highlighted.

## Strongest patterns

Most cells track near y=x. A minority show larger |P−S|, indicating that low mean alone is insufficient for robustness ranking.

## Robustness

This figure is the direct P/S diagnostic; combine with worst(P,S) rankings when selecting “stable” cells.

## What we should NOT claim

That agreement of P and S validates external generalization, or that disagreement implies a bug.

## Possible technical interpretation

Interpretation: fold-scheme sensitivity remains non-negligible for some representation×topology×annotation combinations even under a frozen protocol.
