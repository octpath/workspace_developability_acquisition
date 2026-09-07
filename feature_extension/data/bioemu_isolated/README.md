# bioemu_isolated/

## What is this?
Aggregated **BioEmu** ensemble descriptors for **isolated VH and VL** monomeric chains
(final organizer physically QC-filtered reassessment features).

Files:
- `features.parquet` — frozen feature columns keyed by `id`
- `qc.csv` — target-independent QC metadata

## Molecular scope
**Isolated VH** and **Isolated VL** monomers sampled separately.
BioEmu in this release does **not** model the full Fab or IgG.

## Number of antibodies
324

## Feature families (plain language)
Organizer evaluation groups columns from this table into logical families
(see BioEmu isolated reassessment `eval_reassess.py`):
- **NEW_CONTACT** — contact occupancy / persistence / entropy summaries (`*contact*`)
- **NEW_PAIRWISE** — CA-RMSD ensemble summaries (`*ca_rmsd*`)
- **NEW_FLEX** — RMSF / flexibility summaries (`*rmsf*`)
Additional Rg and related ensemble scalars are also included when present.

Exact column list (n=65): starts with VH_ca_rmsd_median, VL_ca_rmsd_median, mean_ca_rmsd_median, max_ca_rmsd_median, absdiff_ca_rmsd_median, VH_ca_rmsd_q90, VL_ca_rmsd_q90, mean_ca_rmsd_q90, max_ca_rmsd_q90, absdiff_ca_rmsd_q90, VH_ca_rmsf_mean, VL_ca_rmsf_mean ...

Do **not** treat obsolete V12 BioEmu feature tables as equivalent; they are not the primary release.

## Sampling notes
- VH and VL ensembles were sampled **separately**.
- Features summarize **physically filtered** frames.
- Usable frame counts may differ by chain and antibody (see `qc.csv`).
- Extended **VL+CL** chain experiments are **not** part of this validated release.

## Target labels used during generation
**NO**

## Citation / model
Microsoft BioEmu (see RELEASE_NOTES licensing section).
