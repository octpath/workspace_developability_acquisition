# HSP Canonicalization Report

- BM: **HYPERPARAMETER_SENSITIVE** → canonical: `NONE`
- EIS: **HYPERPARAMETER_SENSITIVE** → canonical: `NONE`

Selection followed interpretability / literature / plateau-centrality; **not** lowest VAL.
Existing mainline P1/P3 geometries retained when robust.

## Exact canonical definitions

```yaml
NONE — no family passed robustness gate
```

## Residue-level modeling justified? **False**
Channels: 0

If neither family is ROBUST_*: do **not** proceed to residue-level injection
merely because H103/H107 looked good under the exact selected geometries.

See `HSP_CANONICAL_RESIDUE_QC.md` and `HSP_RESIDUE_LEVEL_EXPERIMENT_DESIGN.md`.
