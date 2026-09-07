# RELEASE_AUDIT — feature_extension v1

## ready for participant release: YES

Conditional on organizer confirmation of third-party derived-artifact redistribution
(see licensing below). Core technical QC for label leakage / folds / readability: **PASS**.

## Core data blocks included

| Block | Scope | N IDs | Dim / files |
|---|---|---:|---|
| ESMFold Fv PDBs | Fv | 324 | `.pdb` |
| ESMFold Fab PDBs | Fab (reconstructed) | 324 | `.pdb` + constant policy/fasta |
| BioEmu isolated features | VH+VL monomers | 324 | 65 features + `qc.csv` |
| ProteinMPNN | Fv | 324 | 2 |
| ESM-IF1 | Fv | 324 | 4 |
| SaProt | Fv | 324 | 1442 |
| Generator disagreement | Fv multi-gen | 324 | 14 |
| Continuous surface | Fv/Fab | 324 | 360 |
| Packing/cavity | Fab | 324 | 33 |
| Buried unsatisfied | Fab | 323 | 20 |
| Fab interface | Fab | 324 | 29 |
| AROMATIC-TOPO | Fv | 324 | 19 |
| STATIC-SAP | Fv | 324 | 18 |
| HYDRO-FIELD | Fv | 324 | 16 |
| TITRATION_SHAPE | Fv | 324 | 18 |
| folds.csv | DEV only | 162 | fold_primary, fold_shadow |

## Omitted blocks and reasons

| Item | Reason |
|---|---|
| Full-Fab FeNNix | Still running / deferred (`NOT_YET_INCLUDED`) |
| Raw BioEmu trajectories | Size/complexity |
| ABodyBuilder2 / Boltz2 structures | Simplify v1 surface area |
| Structure-guided pooling S1 | High-dim; optional later |
| BioEmu V12 primary | Superseded |
| VL+CL BioEmu | Not validated for release |
| New heavy inference | Resource / FeNNix priority |

## Archive sizes (`dist/`)

| Archive | Size |
|---|---|
| `feature_extension_v1_code.zip` | ~26 KB |
| `feature_extension_v1_esmfold_fv.zip` | ~12 MB |
| `feature_extension_v1_esmfold_fab.zip` | ~21 MB |
| `feature_extension_v1_bioemu_isolated.zip` | ~127 KB |
| `feature_extension_v1_precomputed_features.zip` | ~3.3 MB |
| `SHA256SUMS.txt` | present |

## Validation result

`python tools/validate_release.py` → **PASS**

Checks: forbidden label/split columns in tabular data, unique IDs, readable parquet/csv,
no ±inf, folds schema (162 × primary/shadow ∈ {0..4}), PDB naming, no escaping symlinks.

Warnings (non-fatal): sparse NaNs in some continuous-surface Fab-prep columns (1 ID);
documented incomplete buried-unsatisfied coverage (323/324).

## Test result

Smoke tests (`tests/test_smoke.py`) → **PASS** (imports, one-PDB SASA/aromatic extract,
parquet ID uniqueness, folds, examples present).

`pytest` not installed in organizer `.venv_b1`; tests executed via direct function calls.

## License-review status

| Artifact class | Status |
|---|---|
| ProteinMPNN / SaProt derived tables | MIT evidence in-tree → **included** |
| ESMFold / ESM-IF1 derived | fair-esm MIT evidence in env → **included** |
| BioEmu derived features | BioEmu MIT evidence in env → **included** |
| ABB2 structures | **omitted** (simplify / optional later) |
| Counsel-level public redistribution | **Organizer approval recommended** before wide public hosting |

No `OMITTED_PENDING_LICENSE_REVIEW` blockers for the included set based on in-repo LICENSE files.
Organizers should still confirm competition redistribution policy for predicted PDBs.

## Known limitations

- Molecular scope ≠ assay molecule (Fv / reconstructed Fab / isolated VH-VL vs Fab TmApp / IgG HIC).
- Fab constants are surrogate UniProt sequences (POLICY B), not exact experimental alleles.
- Continuous MS extractor not redistributed; use precomputed table. Residue-graph helper is labeled distinctly.
- FeNNix not packaged.
- Some NaN columns in Fab-prep continuous-surface subset.

## Label / leakage checklist

- [x] No TmApp/HIC values in `data/`
- [x] No Public/Private flags in release tables / folds
- [x] No organizer OOF / residual / incumbent columns
- [x] No absolute organizer paths in participant-facing shipped text (build helper uses package-relative ROOT)
- [x] `target_used=NO` on feature assets in MANIFEST
