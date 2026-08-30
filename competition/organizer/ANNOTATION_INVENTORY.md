# Sequence-derived antibody annotation inventory

Competition: **Antibody Developability — TmApp & HIC**  
Gate: Final Release — participant annotation package  
Inventory date: 2026-08-31  

Primary candidate artifact:

- `gate_b1/data/numbering_germline.csv` (N=400 study rows; all 324 competition IDs present)

Supporting sources inspected:

- `gate_b1/scripts/02_numbering_germline.py`
- `gate_b1/scripts/02c_anarci_germline_full.py`
- `gate_b1/reports/numbering_and_germline_audit.md`
- `gate_b3/frozen/organizer/final_population.csv` (also carries families + forbidden metadata)
- `gate_b6_split_search/cache/test_balance_features.csv` (unsafe wholesale; includes targets/roles/donor)

Coverage on competition population (N=324): every preferred ANARCI/PL field below has **0 missing** rows.

---

## Inventory table

| field name | source artifact | annotation method/tool | sequence-derived? | available Dev? | available Test? | missingness (N=324) | participant-safe? | decision |
|---|---|---|---|---|---|---|---|---|
| `PL_vh_family` | `numbering_germline.csv` | ANARCI V-gene assignment → family (`VH#`) | Yes | Yes | Yes | 0 | Yes | **INCLUDE** as `heavy_v_family` |
| `PL_vl_family` | same | ANARCI V-gene → family (`VK#`/`VL#`) | Yes | Yes | Yes | 0 | Yes | **INCLUDE** as `light_v_family` |
| `PL_anarci_vh_v_gene` | same | ANARCI allele-level V gene | Yes | Yes | Yes | 0 | Yes (but allele-precise) | **EXCLUDE** — prefer family level for robustness |
| `PL_anarci_vl_v_gene` | same | ANARCI allele-level V gene | Yes | Yes | Yes | 0 | Yes (but allele-precise) | **EXCLUDE** — prefer family |
| `PL_anarci_vh_j_gene` | same | ANARCI J allele (e.g. `IGHJ4*01`) | Yes | Yes | Yes | 0 | Yes | **INCLUDE** derived `heavy_j_family` (`JH#`) |
| `PL_anarci_vl_j_gene` | same | ANARCI J allele | Yes | Yes | Yes | 0 | Yes | **INCLUDE** derived `light_j_family` (`JK#`/`JL#`) |
| `PL_kappa_lambda` | same | ANARCI light chain type | Yes | Yes | Yes | 0 | Yes | **INCLUDE** as `light_chain_type` |
| `PL_H_CDR1_len` … `PL_L_CDR3_len` | same | Length of author mmc2 IMGT CDR segment strings (`AUTHOR_MMC2_IMGT_SEGMENTS`); concat = VH/VL | Yes (segment lengths of provided sequences) | Yes | Yes | 0 | Yes if provenance clear | **INCLUDE** as `h_cdr*_length` / `l_cdr*_length` |
| `H_CDR1`…`L_FR4` region strings | same | Author mmc2 IMGT segments | Yes | Yes | Yes | 0 | Borderline (full substrings redundant with sequences) | **EXCLUDE** — lengths only; avoid shipping full region text |
| `PL_anarci_vh_v_identity` | same | ANARCI V-allele identity (0–1) | Yes | Yes | Yes | 0 | Yes | **INCLUDE** as `heavy_germline_identity` |
| `PL_anarci_vl_v_identity` | same | ANARCI V-allele identity (0–1) | Yes | Yes | Yes | 0 | Yes | **INCLUDE** as `light_germline_identity` |
| `PL_anarci_*_germline_distance` | same | `1 −` ANARCI V identity | Yes | Yes | Yes | 0 | Yes | **EXCLUDE** — redundant with identity |
| `PL_anarci_*_mutfrac_excl_CDR3` | same | Hamming vs germline AA on IMGT 1–104 | Yes | Yes | Yes | 0 | Yes | **EXCLUDE** — advanced organizer SHM profile; keep identity only |
| `PL_vh_germline_distance` (naive consensus) | same | Hamming vs naïve-within-family consensus | Yes | Yes | Yes | check | Yes but less standard | **EXCLUDE** — superseded by ANARCI identity |
| `ORG_author_*` / author `vh_germline` | same / population | Author table germline labels | Study annotation of same sequences | Yes | Yes | 0 | Prefer ANARCI for reproducibility | **EXCLUDE** — use ANARCI PL fields |
| `b_cell_subset` / donor / cohort | population / triple_core / balance cache | Experimental metadata | **No** | Yes | Yes | — | **No** | **EXCLUDE** |
| `TmApp` / `HIC` | population | Assay | **No** | Dev only / Test secret | — | — | **No** | **EXCLUDE** from annotation files |
| Public/Private / role | SPLIT_MANIFEST / role_map | Split membership | **No** | — | — | — | **No** | **EXCLUDE** |
| PLM embeddings / structure features | B5+ caches | Organizer models | Sequence-input but performance-selected | — | — | — | **No** as organizer baseline | **EXCLUDE** |
| Nearest-train identity / balance features | B6 caches | Split diagnostics | Mixed | — | — | — | **No** | **EXCLUDE** |
| Position-wise SHM one-hots | feature scripts | Engineered | Sequence-derived but organizer-engineered | — | — | — | Keep participant-created | **EXCLUDE** |

---

## Selection summary

**INCLUDE (14 annotation columns + `id`):**

| Distributed column | Source mapping |
|---|---|
| `heavy_v_family` | `PL_vh_family` |
| `heavy_j_family` | family parsed from `PL_anarci_vh_j_gene` |
| `light_v_family` | `PL_vl_family` |
| `light_j_family` | family parsed from `PL_anarci_vl_j_gene` |
| `light_chain_type` | `PL_kappa_lambda` (`kappa`/`lambda`) |
| `h_cdr1_length` … `h_cdr3_length` | `PL_H_CDR*_len` |
| `l_cdr1_length` … `l_cdr3_length` | `PL_L_CDR*_len` |
| `heavy_germline_identity` | `PL_anarci_vh_v_identity` |
| `light_germline_identity` | `PL_anarci_vl_v_identity` |

**EXCLUDE (reasons):** allele-level V genes (prefer family); redundant germline distance; mutfrac / consensus distances; author ORG germline; region AA strings; donor / B-cell / assay / split / organizer model features.

---

## Provenance notes (pre-build)

- Germline V/J + identity + κ/λ: **ANARCI** (`scheme=imgt`, `allowed_species=['human']`) via `gate_b1/scripts/02c_anarci_germline_full.py`; `PL_family_source=anarci` for all 324.
- CDR lengths: frozen scheme **`AUTHOR_MMC2_IMGT_SEGMENTS`** — lengths of Shehata mmc2 IMGT CDR segment columns that concatenate exactly to distributed `heavy`/`light`.
- Same frozen table used for Dev and Test; membership only selects rows.

Inventory complete — proceed to build participant annotation CSVs.
