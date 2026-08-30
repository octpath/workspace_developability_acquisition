# Dataset audit — Shehata 2019 (Gate B1)

- Source CSV: `/workspace_developability_acquisition/interim/shehata_full_joined.csv` (sha256 `6e3b5fdb00076afb…`)
- Source XLSX: `/workspace_developability_acquisition/raw/shehata/a05/mmc2.xlsx` (sha256 `f06a0849c89792bd…`)

## Counts

| Metric | Value |
|--------|------:|
| N rows | 400 |
| Unique antibody IDs | 400 |
| Unique VH | 400 |
| Unique VL | 400 |
| Unique VH/VL pairs | 400 |
| Duplicate VH/VL pairs | 0 |
| Missing VH | 0 |
| Missing VL | 0 |
| VH with author gap chars | 13 |
| VL with author gap chars | 11 |
| TRIPLE_CORE (PSR∩HIC∩TmApp) | 324 |

## Sequence length distributions

- VH: min=113, median=122.0, max=140
- VL: min=103, median=108.0, max=120

## B-cell subset (ORGANIZER_ONLY)

- IgG memory: 146
- LLPCs: 145
- IgM memory: 65
- Naïve: 44

## Targets

### PSR (`psr_score`)

- N complete: **398** (missing 2)
- mean±SD: 0.03463 ± 0.08833
- median: 0
- min/max: 0 / 0.7107
- IQR outliers: 73 (18.3%)
- quantiles: q01=0, q05=0, q10=0, q25=0, q50=0, q75=0.01695, q90=0.1142, q95=0.1918, q99=0.4581

### HIC (`hic_rt_min`)

- N complete: **348** (missing 52)
- mean±SD: 9.39 ± 0.8332
- median: 9.113
- min/max: 8.465 / 13.86
- IQR outliers: 28 (8.0%)
- quantiles: q01=8.568, q05=8.697, q10=8.756, q25=8.867, q50=9.113, q75=9.6, q90=10.41, q95=11.2, q99=12.48

### TmApp (`tm_app_C`)

- N complete: **346** (missing 54)
- mean±SD: 69.86 ± 4.74
- median: 70
- min/max: 52.5 / 83.5
- IQR outliers: 6 (1.7%)
- quantiles: q01=57.95, q05=61.5, q10=64.5, q25=67, q50=70, q75=73, q90=75.75, q95=77.5, q99=80.5

## Expected count confirmation

| Target | Expected | Observed |
|--------|----------:|----------:|
| PSR | ~398 | 398 |
| HIC | ~348 | 348 |
| TmApp | ~346 | 346 |
| triple | ~324 | 324 |

## Filtering rules

1. Start from interim/shehata_full_joined.csv (400 ADI clones with VH+VL protein).
1. mmc2.xlsx has 402 sheet rows of which 2 are legend footnotes (not clones) → 400 clones.
1. Gap characters '-' in author protein strings are stripped for modeling sequences (clean_aa).
1. Non-standard letters outside ACDEFGHIKLMNPQRSTVWY are stripped if present.
1. Rows are NEVER deleted solely for missing labels; subset tables are created instead.
1. TRIPLE_CORE = rows with non-null PSR, HIC, and TmApp.
1. PARTICIPANT_LEGAL inputs for future competition: antibody_id, heavy, light only.
1. ORGANIZER_ONLY_AUDIT: b_cell_subset, author vh/vl_germline, join metadata.

## Conflicting labels

- Same VH/VL pair with conflicting labels: none

## Participant vs organizer columns

- PARTICIPANT_LEGAL: `antibody_id`, `heavy`, `light`
- ORGANIZER_ONLY_AUDIT: b_cell_subset, vh_germline, vl_germline, label_source, sequence_source, join_confidence

