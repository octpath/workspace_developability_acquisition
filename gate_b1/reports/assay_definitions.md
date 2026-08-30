# Assay definitions — Shehata et al. 2019

Source: Cell Reports 2019 supplement `mmc2.xlsx` column headers and paper terminology.

## PSR — PSR Score

- Column (supplement): **PSR Score**
- Internal column: `psr_score`
- Meaning: polyspecificity reagent / nonspecific binding assay score from the study panel.
- Higher values indicate greater polyspecific / nonspecific binding (generally less desirable for developability).
- Continuous score; N complete = 398 of 400 paired clones.

## HIC — HIC retention time (min)

- Column (supplement): **HIC retention time (min)**
- Internal column: `hic_rt_min`
- Meaning: hydrophobic interaction chromatography retention time in minutes.
- Higher retention generally corresponds to stronger hydrophobic interaction with the HIC resin.
- This is a **hydrophobicity / developability proxy**, **not** an aggregation assay.
- Continuous; units: minutes; N complete = 348.

## TmApp — TmApp (°C)

- Column (supplement): **TmApp (°C)**
- Internal column: `tm_app_C`
- Meaning: apparent thermal transition / conformational stability measure reported by the authors as TmApp.
- Units: **degrees Celsius (°C)**.
- Higher TmApp generally corresponds to greater apparent thermal stability.
- Continuous; N complete = 346.

## Biological structure of the panel

Antibodies originate from human B-cell repertoire subsets reported in the supplement:

- Naïve
- IgM memory
- IgG memory
- LLPCs (long-lived plasma cells)

Author-provided germline annotations (`VH Germline`, `VL Germline`) and B-cell subset are **ORGANIZER_ONLY_AUDIT** variables for confounding diagnostics, not participant features.

## Feature legality

| Class | Contents |
|-------|----------|
| PARTICIPANT_LEGAL | `id` / `antibody_id`, `heavy`, `light` (+ derived sequence/structure features from those alone) |
| ORGANIZER_ONLY_AUDIT | `b_cell_subset`, author `vh_germline` / `vl_germline`, study metadata |
| ILLEGAL_FOR_PARTICIPANTS | Models that use organizer-only columns as inputs (oracle / confounding audits only) |
