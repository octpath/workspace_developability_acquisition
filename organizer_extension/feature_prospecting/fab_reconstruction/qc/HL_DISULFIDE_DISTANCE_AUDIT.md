# Heavy–light interchain cysteine geometry audit (ESMFold Fab)

**N:** 324 successful Fab predictions.

**Important:** Distances are Sγ–Sγ (QC nearest heavy/light Cys SG pair). ESMFold does **not** enforce a covalent disulfide. Proximity ≠ formed bond.

## Distributions (Å)

| Subset | n | q10 | median | q90 | max | <2.3Å | <2.6Å | <3.0Å | ≥3.0Å |
|--------|---|-----|--------|-----|-----|-------|-------|-------|-------|
| overall | 324 | 2.57 | 2.80 | 3.45 | 52.10 | 3.1% | 12.3% | 74.4% | 25.6% |
| kappa | 238 | 2.55 | 2.74 | 2.95 | 52.10 | 4.2% | 16.0% | 93.7% | 6.3% |
| lambda | 86 | 2.93 | 3.34 | 3.64 | 3.91 | 0.0% | 2.3% | 20.9% | 79.1% |

## FeNNix-on-Fab QC note

Treat H–L Cys geometry as an **input-QC / post-relaxation check**, not ground-truth chemistry.
After FeNNix structure preparation / restrained relaxation, verify whether the expected H–L disulfide
geometry becomes chemically reasonable. Do **not** allow unresolved disulfide strain to dominate
energy features used for later TmApp work.
