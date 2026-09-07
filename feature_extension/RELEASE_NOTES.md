# RELEASE_NOTES — feature_extension v1

**Version:** v1  
**Status:** participant release package (structures + precomputed features + extractors)  
**FeNNix full-Fab:** **not included** (production job still running; deferred to a later update)

## Included

- ESMFold **Fv** PDBs (N=324)
- Reconstructed ESMFold **Fab** PDBs (N=324) + constant-domain policy/sequences
- BioEmu **isolated VH/VL** aggregated features + QC (`features.parquet`, `qc.csv`)
- Precomputed blocks: ProteinMPNN, ESM-IF1, SaProt, generator disagreement,
  AROMATIC-TOPO, STATIC-SAP, HYDRO-FIELD, TITRATION_SHAPE,
  continuous surface, packing/cavity, buried unsatisfied, Fab interface
- Organizer Stage0 `folds.csv` (DEV N=162; `fold_primary`, `fold_shadow`)
- Lightweight extractors + examples + `validate_release.py` / `package_release.py`

All released feature assets: **`target_used = NO`**.

## Not yet included / deferred

| Item | Reason |
|---|---|
| Full-Fab FeNNix features | Production still running; separate future release |
| Raw BioEmu trajectories | Size / complexity (`OMITTED_SIZE_AND_COMPLEXITY`) |
| ABodyBuilder2 / Boltz2 structure trees | Simplified v1; ESMFold Fv/Fab prioritized |
| Structure-guided pooling (S1) high-dim | Prefer lighter core; optional later |
| Obsolete BioEmu V12 primary tables | Superseded by isolated reassessment |
| VL+CL extended-chain BioEmu | Not part of validated release |
| OpenMM / new ESMFold / ProteinMPNN inference code reruns | Reuse frozen outputs only |

## Planned update

Future **v1.x** may add:

- full-cohort FeNNix Fab features (when frozen and QC’d)
- other newly frozen target-blind feature blocks

No unfinished result is promised.

## Folds (authoritative Stage0)

Recovered from `virtual_participant/stage0_cv/`:

- Primary: `opt_joint_group_k5_s42` (seed **42**)
- Shadow: `opt_joint_group_k5_s2026` (seed **2026**)
- Sequence group rule: **min(VH, VL) identity ≥ 0.9**; groups are atomic
- Common TmApp/HIC 5-fold; assignment optimized for size/distribution/high-tail balance,
  **not** for maximizing a particular model’s MAE

## Fab reconstruction (summary)

From frozen `CONSTANT_DOMAIN_POLICY.md` (POLICY B):

| Role | Sequence ID | Source |
|---|---|---|
| Heavy constant | `IGHG1_CH1_EPKSC_UNIPROT_P01857` | UniProt IGHG1 P01857 |
| C-kappa | `IGKC_UNIPROT_P01834` | UniProt IGKC P01834 |
| C-lambda | `IGLC2_UNIPROT_P0DOY2` | UniProt IGLC2 P0DOY2 |

These are **predicted/reconstructed** Fabs, not experimental structures, and not full IgG.

## Third-party provenance and licensing notes

Do **not** treat this section as legal advice. Organizers should confirm redistribution
before a fully public dump if policies differ from in-repo LICENSE files.

| Tool / model | In-repo / env evidence | Derived artifacts in v1 | Notes |
|---|---|---|---|
| **ProteinMPNN** | `tools/ProteinMPNN/LICENSE` — **MIT** | `proteinmpnn.parquet` | Scores from organizer frozen run |
| **SaProt** | marathon `tools/SaProt/LICENSE` — **MIT** | `saprot.parquet` | Pooled/global features included |
| **ESM / ESMFold / ESM-IF1** | fair-esm package LICENSE — **MIT** (env docs) | Fv/Fab PDBs; `esm_if1.parquet` | Model version strings often `NOT_RECORDED` in historical runs |
| **BioEmu** | Microsoft BioEmu — **MIT** (environment packaging) | `bioemu_isolated/` features | Isolated VH/VL only; model noted as bioemu-v1.2 in reassessment spec |
| **ImmuneBuilder / ABodyBuilder2** | BSD-3 in env dist-info | **Structures omitted from v1** | Available in organizer trees; not required for core v1 |
| **FreeSASA** (Gap Closure continuous surface) | dependency of Gap Closure pipeline | `continuous_surface.parquet` | Precomputed outputs only; extractor not re-shipped |
| **BioPython** | Biopython license | Used by participant extractors | Install via `requirements.txt` |

### OMITTED_PENDING_LICENSE_REVIEW

None identified as blocking for the **included** v1 artifacts above given MIT/BSD evidence
in-tree. If an external counsel/policy review disagrees for ESMFold PDB redistribution or
BioEmu-derived tables, pull those archives before public posting.

ABodyBuilder2 / Boltz2 coordinate trees were omitted from v1 partly to reduce license/surface
area and package size (`OMITTED_SIMPLIFY_V1`), not solely for license failure.

## QC notes

- `buried_unsatisfied.parquet`: N=323 (one antibody missing vs 324)
- SaProt table is high-dimensional (global pooled dims ~1.4k); still a frozen table, not raw per-residue dumps
- Validator: `python tools/validate_release.py`
- Smoke tests: `python -m pytest tests/test_smoke.py` (from package parent on `PYTHONPATH`)

## Packaging

`python tools/package_release.py` writes under `dist/`:

- `feature_extension_v1_code.zip`
- `feature_extension_v1_esmfold_fv.zip`
- `feature_extension_v1_esmfold_fab.zip`
- `feature_extension_v1_bioemu_isolated.zip`
- `feature_extension_v1_precomputed_features.zip`
- `SHA256SUMS.txt`
