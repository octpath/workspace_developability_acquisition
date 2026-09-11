# Generalized spatial property engine (design note — hydrophobicity-only execution)

**Path:** `feature_research/hic_spatial_hydrophobicity`  
**Status:** Conceptual API frozen; **only hydrophobicity scales executed** in this task.

## Core form

```
P_i(R; property, exposure, neighborhood)
  = Σ_j I[neighbor(i,j; R)] * exposure_j * property_j
```

## Factory inputs

| Argument | Role |
|----------|------|
| `residue_property_table` | Per-AA or per-residue continuous property |
| `exposure_mode` | TOTAL_RASA_TIEN / SIDECHAIN_OVER_TIEN / SIDECHAIN_SASA_ABS / … |
| `neighborhood_mode` | CENTROID / CLOSEST_SC / (future atom-SAP) |
| `radius` | Å |
| `aggregation_scope` | ALL_FV / H / L / CDR / HCDR3 / LCDR3 |
| `aggregation_function` | MAX MEAN SUM Q90 Q95 TOP3 TOP5 |

## Future property channels (UNEXECUTED here)

- Positive / negative / absolute charge
- Aromaticity
- H-bond donor/acceptor propensity

Do **not** rely on signed charge sum alone (cancellation). Prefer separate +/−/|q| channels.

## Implementation map

- `src/spatial_engine.py` — generic neighborhood + aggregation
- `src/scales.py` — hydrophobicity property tables only
- `src/literature_anchors.py` — non-generic literature formulas (SAP/PSH/…)

Charge/aromatic screens must open a **new** feature-research batch, not reuse this atlas silently.
