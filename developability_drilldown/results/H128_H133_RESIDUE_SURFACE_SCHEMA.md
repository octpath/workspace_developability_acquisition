# H128–H133 Residue-Level F1_SURFACE Input Schema (FROZEN)

**Status:** PREREGISTERED / FROZEN before training  
**p = 10** (≤12 target)  
**Provenance:** PASS-EXACT F1 reconstruction artifacts  
**Not equivalent to:** full vertex cloud / exact F1_SURFACE35 (compact Transformer input)

Distinction:

| Layer | Role |
|-------|------|
| A. Full provenance | `residue_surface_aro.parquet` + `residue_surface_hydro_vertices.parquet` — exact F1 round-trip |
| B. Compact schema (this doc) | 10 local channels per sequence residue for Transformer fusion |

---

## Channel table

| idx | name | type | block | definition | source field | F1 relationship | missing | preprocess |
|----:|------|------|-------|------------|--------------|-----------------|---------|------------|
| 0 | `rasa` | continuous | ARO | SASA / MaxASA(Tien) | `residue_surface_aro.rasa` | drives exposure thresholds in ARO19 | none (complete) | TRAIN mean/std over valid TRAIN residues |
| 1 | `sasa` | continuous | ARO | residue SASA (Å²) | `residue_surface_aro.sasa` | aromatic exposed SASA sums | none | TRAIN mean/std |
| 2 | `is_aromatic` | binary | ARO | 1 if AA ∈ {F,W,Y} | `is_aromatic` | selects aromatic class for ARO19 | none | none (0/1) |
| 3 | `is_exposed` | binary | ARO | 1 if RASA ≥ 0.20 | `is_exposed` | exposed aromatic counts | none | none |
| 4 | `is_strongly_exposed` | binary | ARO | 1 if RASA ≥ 0.50 | `is_strongly_exposed` | strongly exposed ARO cols | none | none |
| 5 | `hydro_area_sum` | continuous | HYDRO | Σ area of vertices originating from residue | vertices→residue | contributes to area-weighted HYDRO16 | if no verts | TRAIN mean/std on **available** TRAIN residues only; else 0 after scale |
| 6 | `hydro_H_awmean` | continuous | HYDRO | Σ(H·area)/Σ(area) | vertices→residue | local H field mean | if no verts | same |
| 7 | `hydro_pos_H_area` | continuous | HYDRO | Σ area where H>0 | vertices→residue | positive_H_area_fraction numerator pieces | if no verts | same |
| 8 | `hydro_neg_H_area` | continuous | HYDRO | Σ area where H≤0 | vertices→residue | complement of positive H area | if no verts | same |
| 9 | `availability_hydro` | binary | HYDRO | 1 if residue has ≥1 originating vertex | derived | mask for incomplete surface coverage (~66% residues have verts) | n/a | none (0/1) |

## Explicit non-goals

- No SAP24 / SCM24 / GLOBAL / EXTRA
- No antibody-level F1_SURFACE35 copied onto residues
- No windowed sequence features
- No supervised channel selection
- No `is_cdr` (sequence annotation, not surface state)

## Alignment

```
(antibody_id, chain∈{H,L}, sequence_index)
  ↔ structure residue (exact PASS-EXACT map)
  ↔ Transformer residue token (same index; REG/pad excluded)
```

## Artifact hashes

Recorded in companion YAML at freeze time.
