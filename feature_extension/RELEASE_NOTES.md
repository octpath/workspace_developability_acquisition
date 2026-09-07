# RELEASE_NOTES — feature_extension v1

**Version:** v1  
**Status:** participant release package (structures + precomputed features + extractors)  
**FeNNix full-Fab:** **not included** (production job still running; deferred to a later update)

## Included

- ESMFold **Fv** PDBs (N=324)
- Reconstructed ESMFold **Fab** PDBs (N=324) + constant-domain policy/sequences
- BioEmu **isolated VH/VL** aggregated features + QC + `FEATURE_DICTIONARY.csv`
- Precomputed blocks: ProteinMPNN, ESM-IF1, SaProt, generator disagreement,
  AROMATIC-TOPO, STATIC-SAP, HYDRO-FIELD, TITRATION_SHAPE,
  continuous surface, packing/cavity, buried unsatisfied, Fab interface
- `BLOCK_COVERAGE.csv` / updated `MANIFEST.csv` coverage fields
- Organizer Stage0 `folds.csv` (DEV N=162; `fold_primary`, `fold_shadow`)
- Lightweight extractors + examples + validation / packaging tools

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

Do **not** treat this section as legal advice.  
**Do not assume** “repository LICENSE is MIT ⇒ model weights and all generated outputs
are automatically MIT.” Below separates code / weights / derived-output status where known.

Decision labels used:

- `CLEAR_FOR_RELEASE` — permissive code+weights evidence; no explicit output ban found
- `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` — permissive upstream; outputs not specially restricted in reviewed sources
- `PARTICIPANT_ONLY_REVIEW_RECOMMENDED` — useful to ship to participants, but organizers should confirm public hosting policy
- `OMIT_PENDING_LICENSE_REVIEW` — concrete uncertainty blocked inclusion (none for current core set)

| artifact | upstream project | code license | weights license | derived-output redistribution status | release decision | source/reference | notes |
|---|---|---|---|---|---|---|---|
| ESMFold Fv/Fab PDBs | Meta fair-esm / ESMFold | MIT (repo LICENSE) | MIT (same project packaging; HF/docs treat ESMFold as MIT) | No explicit ban on redistributing predicted PDBs found in reviewed LICENSE/README | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` + `PARTICIPANT_ONLY_REVIEW_RECOMMENDED` | https://github.com/facebookresearch/esm ; env `fair-esm` LICENSE | Competition-policy review still recommended for wide public hosting |
| `esm_if1.parquet` | ESM-IF1 (fair-esm) | MIT | MIT (project packaging) | Same as above for score/feature tables | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` | fair-esm LICENSE | |
| `proteinmpnn.parquet` | ProteinMPNN | MIT (`tools/ProteinMPNN/LICENSE`) | Weights distributed with project under same MIT tree | No explicit output restriction found | `CLEAR_FOR_RELEASE` | in-repo LICENSE | Native scores only |
| `saprot.parquet` | SaProt | MIT (marathon `tools/SaProt/LICENSE`) | MIT (project LICENSE) | No explicit output restriction found | `CLEAR_FOR_RELEASE` | in-repo LICENSE | High-dim pooled features |
| `bioemu_isolated/*` | Microsoft BioEmu | MIT (GitHub LICENSE) | MIT (`MODEL_CARD.md` license: mit) | MIT covers software/weights; no separate ban on derived scalar features found | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` + `PARTICIPANT_ONLY_REVIEW_RECOMMENDED` | https://github.com/microsoft/bioemu | Aggregated descriptors only; raw trajectories omitted |
| `continuous_surface.parquet` | FreeSASA + organizer Gap Closure code | FreeSASA is typically MIT (verify install); organizer scripts project-owned | N/A (classical geometry library) | Derived numeric table from SAS sampling | `NO_EXPLICIT_OUTPUT_RESTRICTION_FOUND` | Gap Closure SPEC; FreeSASA project | Not MSMS SES triangulation |
| AROMATIC-TOPO / STATIC-SAP / HYDRO-FIELD / TITRATION_SHAPE | Bio.PDB / project extractors | Biopython license + project-owned scripts | N/A | Project-owned descriptors | `CLEAR_FOR_RELEASE` | AROMATIC-TOPO FEATURE_SPEC | |
| packing / buried-unsat / fab_interface | Gap Closure (Bio.PDB/FreeSASA geometry) | project-owned + deps | N/A | Project-owned descriptors | `CLEAR_FOR_RELEASE` | Gap Closure SPEC | |
| ABodyBuilder2 structures | ImmuneBuilder | BSD-3 (env dist-info) | check upstream weights terms | **Omitted from v1** | `OMIT` (simplify v1; not solely license failure) | env ImmuneBuilder LICENSE | |

### Unresolved / organizer judgment

1. Wide **public** redistribution of predicted PDBs and BioEmu-derived tables should get a final competition/legal OK even though no explicit output ban was found.
2. FreeSASA’s exact packaged LICENSE file was not re-copied into this release; continuous_surface ships **numeric features only**, not FreeSASA source.
3. This audit is **not** legal certainty.

### OMITTED_PENDING_LICENSE_REVIEW

No core included artifact was removed solely for this label during the final audit.
ABB2/Boltz2 structures remain omitted for simplify/size (`OMITTED_SIMPLIFY_V1`).

## QC notes

- `buried_unsatisfied.parquet`: N=323 — missing **ADI-47265** (known Fab prep issue)
- `continuous_surface.parquet`: 324 IDs; Fab-prep column subset has NaNs for ADI-47265
- SaProt ~1.4k pooled dims
- BioEmu family dictionary: `data/bioemu_isolated/FEATURE_DICTIONARY.csv`
- Coverage: `BLOCK_COVERAGE.csv`
- Validator: `python tools/validate_release.py`

## Packaging

`python tools/package_release.py` writes under `dist/`:

- `feature_extension_v1_code.zip`
- `feature_extension_v1_esmfold_fv.zip`
- `feature_extension_v1_esmfold_fab.zip`
- `feature_extension_v1_bioemu_isolated.zip`
- `feature_extension_v1_precomputed_features.zip`
- `SHA256SUMS.txt`
