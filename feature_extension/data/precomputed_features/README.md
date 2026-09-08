# precomputed_features/

Expensive or frozen model-derived / descriptor feature tables.
Each parquet has an `id` column and numeric feature columns only.

**Target labels used: NO** for all blocks.

## Join guidance

Start from your competition dataframe and **LEFT JOIN** on `id`.

Do **not** use naive inner joins across all blocks — incomplete blocks (e.g.
`buried_unsatisfied`, N=323) will silently drop rows.

Missing feature values: do not invent them in the release tables. For modeling,
impute only with **train-fold-local** statistics under CV.

Coverage detail: see top-level `BLOCK_COVERAGE.csv`.

## Feature dictionary

| File | Model / family | Scope | Dim | N IDs | Suggested target (heuristic) | Status |
|---|---|---|---:|---:|---|---|
| `proteinmpnn.parquet` | ProteinMPNN | Fv ESMFold | 2 | 324 | TmApp/HIC | COMPLETE |
| `esm_if1.parquet` | ESM-IF1 | Fv ESMFold | 4 | 324 | TmApp/HIC | COMPLETE |
| `saprot.parquet` | SaProt | Fv ESMFold | 1442 | 324 | TmApp/HIC | COMPLETE |
| `generator_disagreement.parquet` | cross-generator disagreement | Fv multi-generator | 14 | 324 | TmApp/HIC | COMPLETE |
| `continuous_surface.parquet` | Gap Closure continuous SAS-sampled surface | Fv/Fab ESMFold | 360 | 324 | HIC | COMPLETE_IDS; some Fab-prep cells NaN for ADI-47265 |
| `packing_cavity.parquet` | packing/cavity (Gap Closure) | Fab ESMFold | 33 | 324 | TmApp | COMPLETE |
| `buried_unsatisfied.parquet` | buried unsatisfied polar (Gap Closure) | Fab ESMFold | 20 | **323** | TmApp | Missing **ADI-47265** (Fab prep issue) |
| `fab_interface.parquet` | Fab domain interface (Gap Closure) | Fab ESMFold | 29 | 324 | TmApp | COMPLETE |
| `aromatic_topology.parquet` | AROMATIC-TOPO v1 | Fv ESMFold | 19 | 324 | HIC | COMPLETE |
| `static_sap.parquet` | STATIC-SAP | Fv ESMFold | 18 | 324 | HIC | COMPLETE |
| `hydro_field.parquet` | HYDRO-FIELD | Fv ESMFold | 16 | 324 | HIC | COMPLETE |
| `titration_shape.parquet` | TITRATION_SHAPE | Fv ESMFold | 18 | 324 | HIC/TmApp | COMPLETE |
| `fennix_fab_context.parquet` | FeNNix Fab-context (v1.1) | **Fab** reconstructed | 92 | **323** | TmApp (heuristic) | Missing **ADI-47265** (TECHNICAL_SKIP) |

## FeNNix Fab-context (v1.1)

See `FENNIX_FAB_CONTEXT_FEATURE_DICTIONARY.csv`.

**What:** FeNNix local-curvature / perturbational aggregates on reconstructed prepared Fab
(A/B/C/M protocol). Families: DELTA_GEOM, DELTA_ENV, CONSTANT, INTERFACE,
FULL_FAB_NORMALIZED, PREP_RELAX_SENSITIVITY (+ COMBINED in the wide table).

**Scope:** Fab-like (VH+CH1 / VL+CL), not IgG. Surrogate constant domains.

**Coverage:** 323/324 — `ADI-47265` absent (Fab prep / HL disulfide geometry QC failure).
Do not impute that row in the distributed table. **LEFT JOIN** on `id`.

**Organizer Dev-CV note:** full-cohort Simple TVT did **not** show stable Primary∩Shadow
improvement for these blocks (interim N=100 CONSTANT signal did **not** replicate).
Provided as target-blind experimental descriptors.

## continuous_surface (important)

`continuous_surface.parquet` is a **precomputed organizer feature block** generated with:

FreeSASA Lee–Richards atom SASA + exterior Fibonacci SAS sampling + multi-scale
hydrophobic masks (KD/FP/BM) + connected components on **surface sample points**
(link 2.0 Å), with patch area/perimeter/compactness.

It is **not** the same as `extractors/extract_surface_patch.py`, which uses a
**residue-CA adjacency graph**. They are **not numerically equivalent**; the
extractor does **not** reproduce this block.

Honest scope from Gap Closure: continuous SAS-sampled hydrophobic patches —
**not** classical MSMS SES triangulation, and **not** residue-graph S3.

## Notes
- Prefer these tables over re-running heavy models.
- Packing / unsatisfied / interface blocks showed weak organizer Dev-CV increments
  but remain scientifically meaningful for participant experiments.

## Target labels used during generation
**NO**
