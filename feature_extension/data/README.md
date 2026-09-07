# data/

Participant-facing structure and precomputed feature assets for **feature_extension v1**.

| Subdirectory | Contents | Molecular scope |
|---|---|---|
| `esmfold_fv/` | Predicted Fv PDBs | Fv |
| `esmfold_fab/` | Reconstructed Fab PDBs | Fab (variable + surrogate constants) |
| `bioemu_isolated/` | Aggregated BioEmu features + QC | Isolated VH / VL monomers |
| `precomputed_features/` | Model / descriptor parquet tables | See per-file README |
| `optional/` | Reserved for large optional archives | — |

**Target labels are not included.** All assets here were generated independently of competition targets.
