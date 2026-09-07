# bioemu_isolated/

## What is this?
Aggregated **BioEmu** ensemble descriptors for **isolated VH and VL** monomeric chains
(final organizer physically QC-filtered reassessment features).

Files:
- `features.parquet` — frozen feature columns keyed by `id` (324 × 65)
- `qc.csv` — target-independent QC metadata
- `FEATURE_DICTIONARY.csv` — per-column dictionary (family, scope, description)

## Molecular scope
**Isolated VH** and **Isolated VL** monomers sampled **separately**.
BioEmu in this release does **not** model the full Fab or IgG.

## Number of antibodies
324 (complete vs competition structure crosswalk).

## How features were built (authoritative)
From BioEmu isolated reassessment (`analyze_convergence.py` / reassessment spec):

1. Load physically filtered BioEmu frames for each chain (`filter_samples` / official physical filter).
2. Frozen cohort **Nphys = 8** physical frames per chain when available (see `BIOEMU_ISOLATED_NPHYS_DECISION.md`).
3. Per chain, compute descriptors on CA coordinates (MDTraj): pairwise CA-RMSD, CA-RMSF, Rg, contact occupancy stats, CDR/FW RMSF.
4. Combine VH and VL with `mean_`, `max_`, `absdiff_` of the per-chain scalars.

**Target labels used during generation: NO**

VH/VL were sampled independently. Features are **not** z-scored / train-normalized in this table.

## Feature families (use these lists)

Logical families match organizer `eval_reassess.py` naming. Prefer loading columns from
`FEATURE_DICTIONARY.csv` rather than reverse-engineering names.

```python
import pandas as pd
bio = pd.read_parquet("data/bioemu_isolated/features.parquet")
dic = pd.read_csv("data/bioemu_isolated/FEATURE_DICTIONARY.csv")

bio_pairwise_cols = dic.loc[dic.family == "NEW_PAIRWISE", "feature"].tolist()  # 10
bio_contact_cols  = dic.loc[dic.family == "NEW_CONTACT", "feature"].tolist()   # 20
bio_flex_cols     = dic.loc[dic.family == "NEW_FLEX", "feature"].tolist()      # 25
bio_shape_cols    = dic.loc[dic.family == "NEW_SHAPE", "feature"].tolist()     # 10
bio_combined_cols = dic["feature"].tolist()  # NEW_COMBINED = all 65 released columns
```

### NEW_PAIRWISE (10)
Pairwise CA-RMSD summaries over physical frames (ensemble diversity).
Columns: `VH_ca_rmsd_*`, `VL_ca_rmsd_*`, `mean_ca_rmsd_*`, `max_ca_rmsd_*`, `absdiff_ca_rmsd_*`
(metrics: `median`, `q90`).

### NEW_CONTACT (20)
CA–CA contact occupancy statistics (sequence separation ≥ 4, cutoff 0.8 nm):
mean occupancy, persistent / labile fractions, occupancy entropy — plus VH/VL/mean/max/absdiff.
Contact features have higher MC noise at Nphys=8 than geometry families (documented in Nphys decision).

### NEW_FLEX (25)
CA-RMSF whole-chain (`ca_rmsf_mean` / `q90`) and region RMSF (`rmsf_cdr_mean`, `rmsf_fw_mean`, `rmsf_cdr3_mean`).

### NEW_SHAPE (10)
Radius of gyration mean / SD (`rg_mean`, `rg_sd`) with VH/VL/mean/max/absdiff.
(Present in organizer eval as `NEW_SHAPE`.)

### NEW_COMBINED (65)
All released columns above. Organizer eval sometimes used a **prefix-capped subset (≤30)**;
participants may use the full 65 or the family lists.

## QC
See `qc.csv`: `physical_usable_frames`, `vh_usable_frames`, `vl_usable_frames`, statuses.
Minimum-frame rule during extraction: skip antibody if either chain has fewer than
`max(4, Nphys/2)` usable frames (authoritative `build_features`).

## Known limitations
- Isolated monomers ≠ Fab dynamics / experimental TmApp molecule.
- Do not treat as a direct TmApp predictor.
- Obsolete BioEmu V12 tables are **not** equivalent.

## Citation / model
Microsoft BioEmu (reassessment notes `bioemu-v1.2` checkpoint class). See RELEASE_NOTES licensing.
