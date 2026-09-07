# precomputed_features/

Expensive or frozen model-derived / descriptor feature tables.
Each parquet has an `id` column and numeric feature columns only.

**Target labels used: NO** for all blocks.

## Feature dictionary

| File | Model / family | Scope | Dim | Suggested target (heuristic) | Status |
|---|---|---|---:|---|---|
| `proteinmpnn.parquet` | ProteinMPNN | Fv ESMFold | 2 | TmApp/HIC | INCLUDED |
| `esm_if1.parquet` | ESM-IF1 | Fv ESMFold | 4 | TmApp/HIC | INCLUDED |
| `saprot.parquet` | SaProt | Fv ESMFold | 1442 | TmApp/HIC | INCLUDED |
| `generator_disagreement.parquet` | cross-generator disagreement | Fv multi-generator | 14 | TmApp/HIC | INCLUDED |
| `continuous_surface.parquet` | FreeSASA continuous surface (Gap Closure) | Fv/Fab ESMFold | 360 | HIC | INCLUDED |
| `packing_cavity.parquet` | packing/cavity (Gap Closure) | Fab ESMFold | 33 | TmApp | INCLUDED |
| `buried_unsatisfied.parquet` | buried unsatisfied polar (Gap Closure) | Fab ESMFold | 20 | TmApp | INCLUDED |
| `fab_interface.parquet` | Fab domain interface (Gap Closure) | Fab ESMFold | 30 | TmApp | INCLUDED |
| `aromatic_topology.parquet` | AROMATIC-TOPO v1 | Fv ESMFold | 19 | HIC | INCLUDED |
| `static_sap.parquet` | STATIC-SAP | Fv ESMFold | 18 | HIC | INCLUDED |
| `hydro_field.parquet` | HYDRO-FIELD | Fv ESMFold | 16 | HIC | INCLUDED |
| `titration_shape.parquet` | TITRATION_SHAPE | Fv ESMFold | 18 | HIC/TmApp | INCLUDED |

## Notes
- Prefer these pooled/scalar tables over re-running ProteinMPNN / ESM-IF1 / SaProt.
- Continuous surface descriptors are **molecular-surface** style from Gap Closure (FreeSASA LR),
  not merely residue-adjacency graphs — see Gap Closure docs for definitions.
- Packing / unsatisfied / interface blocks showed weak organizer CV increments but are scientifically valid for experimentation.

## Target labels used during generation
**NO**
