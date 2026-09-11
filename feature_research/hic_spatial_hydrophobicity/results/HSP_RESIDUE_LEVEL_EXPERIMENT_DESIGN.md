# HSP Residue-Level Experiment Design (DO NOT RUN)

Status: design note only. No Transformer training. EXP-H114 unused.

## Gate result (this audit)

Both BM and EIS were classified **HYPERPARAMETER_SENSITIVE**.

Therefore:

- no `HSP_*_CANONICAL` descriptor was declared
- residue-level injection is **NOT justified** under the preregistered robustness gate
- do not proceed merely because H103/H107 looked good at the exact selected geometries

Mainline P1/P3 remain valid as **frozen experiment-specific** late-fusion blocks, not as
canonical physical families for residue injection.

## Channels

- NONE — residue-level experiment NOT justified.

k = 0 residue HSP channels.

## Comparison arms (future)

| Arm | Aux / injection | Purpose |
|-----|-----------------|--------|
| A | SURFACE only | baseline |
| B | SURFACE + antibody-level canonical HSP (B3) | Ab-level control |
| C | SURFACE + residue-level canonical HSP | location-preserving |
| D | SURFACE + Ab-level HSP + residue-level HSP | complementarity |

## First injection mechanism (fixed, minimal)

```
hsp_i ∈ R^k   # TRAIN-fold StandardScaler on HSP channels
hsp_proj: Linear(k, d_model)  # init weights/bias = 0
token_i ← token_i + hsp_proj(hsp_i)
```

Init-equivalence QC required:

```
pred(residue_HSP at init) == pred(no residue_HSP)
```

## Functional-use diagnostics (future)

1. force hsp_proj = 0 at inference
2. permute HSP among residues within antibody
3. permute HSP profiles across antibodies

## Questions answered by this design

1. Spatial hydrophobicity useful? — largely answered by Ab-level H102–H113
2. Does WHERE the patch occurs help?
3. Does residue-level replace global aggregation?
4. Are global and local HSP complementary?

Do not compare multiple injection mechanisms in the first residue batch.
