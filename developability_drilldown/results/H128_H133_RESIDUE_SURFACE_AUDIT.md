# H128–H133 Residue-Level SURFACE Audit

**Status: STOP_BEFORE_TRAINING**

**Verdict: NO scientifically valid residue-aligned SURFACE source underlies historical F1_SURFACE.**

This batch asked whether residue-aligned SURFACE improves Transformer HIC when injected before antibody pooling. Per batch rules, training is allowed only if an already-justified residue-level source underlying the existing SURFACE feature research exists. That condition fails.

---

## 1. Historical antibody-level SURFACE (canonical)

| Item | Value |
|------|-------|
| Late-fusion bundle | `F1_SURFACE` |
| Definition | `AROMATIC_TOPO` (19) + `HYDRO_FIELD` (16) = **35D antibody vector** |
| Canonical loader | `models/antibody_transformer/h047_aux_features.py` → `H047AuxFeatureStore` |
| Canonical parquet | `experiments/features/EXP-H047.parquet` slices `feat[1395:1414]` + `feat[1414:1430]` |
| Standalone twins (bit-identical after id align) | `top_models_feature_bundle/data/aromatic_topo.parquet`, `hydro_field.parquet` |
| Fusion mode historically | antibody-level `late_concat_aux32` (not residue fusion) |

### AROMATIC_TOPO (19)

```
aro_exposed_TYR_count, aro_exposed_TRP_count, aro_exposed_PHE_count,
aro_exposed_aromatic_total_count, aro_aromatic_exposed_SASA_total,
aro_aromatic_exposed_SASA_fraction, aro_strongly_exposed_aromatic_count,
aro_strongly_exposed_aromatic_SASA, aro_CDR_exposed_aromatic_count,
aro_CDR_aromatic_SASA, aro_CDR_aromatic_fraction, aro_aromatic_patch_count,
aro_largest_aromatic_patch_n_res, aro_largest_aromatic_patch_exposed_SASA,
aro_max_local_aromatic_SASA, aro_sequence_aromatic_count,
aro_sequence_TYR_count, aro_sequence_TRP_count, aro_sequence_PHE_count
```

### HYDRO_FIELD (16)

```
mean_H_surface, q75_H_surface, q90_H_surface, q95_H_surface, max_H_surface,
positive_H_area_fraction, top10_H_mean, top10_H_area_fraction,
high_H_patch_count, largest_high_H_patch_area_fraction,
largest_high_H_patch_n_vertices, CDR_mean_H, CDR_q90_H,
CDR_high_H_area_fraction, n_surface_points, phi_finite_frac
```

---

## 2. Upstream residue / surface sources

### AROMATIC_TOPO

- Extract: `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py` → `aromatic_features()`
- Residue table: `structure_utils.build_residue_table()` (in-memory)
- Structure: ESMFold Fv; SASA via Bio.PDB Shrake–Rupley; RASA = SASA / Tien2013 MaxASA
- **Persisted residue artifact for F1: NONE** (only antibody aggregates saved)

### HYDRO_FIELD

- Extract: `organizer_extension/feature_prospecting/scripts/extract_hydro_copatch_batch2.py`
- Engine: `hydro_surface.build_surface_and_field()` / `hydro_summaries()`
- Persisted intermediate: `organizer_extension/feature_prospecting/HYDRO-FIELD/surface_field_esmfold.parquet`
  - Shape `(305590, 10)`
  - Columns: `id, generator, vertex_i, x, y, z, H, phi, area, is_cdr`
  - **Granularity = surface vertices (~943/Ab), not sequence residues**
  - **No residue id / seq_index column**

---

## 3. Residue-level channels / dimensions usable as F1?

| Asset | Granularity | Residue↔token alignable? | Is F1 canonical upstream artifact? |
|-------|-------------|--------------------------|------------------------------------|
| F1 late-fusion input | antibody 35D | N/A | YES (canonical) |
| ARO residue `rows` | residue (ephemeral) | would be, if saved | NO — never persisted |
| `surface_field_*.parquet` | mesh vertex | NO | partial HYDRO intermediate only |
| `rasa_{heavy,light}.npy` | residue seq index | YES (H/L padded) | NO — F3 LOCAL_RASA path |
| `annotations.parquet` | residue | indexing only | NO SURFACE channels |
| `feature_research/.../residue_geometry_sasa.parquet` | residue | research track | NO — not F1 SURFACE |

---

## 4. Aggregation residue → antibody (historical)

**ARO:** exposed (RASA≥0.20) / strongly exposed (≥0.50) aromatic counts & SASA; CDR subsets; Cα patch connectivity (8Å); local aromatic SASA max (10Å).

**HYDRO:** FreeSASA surface sampling → hydrophobicity field H(s) → area-weighted quantiles, top-10, high-H patches, CDR vertex subsets.

Late fusion then median-imputes / scales the **35 antibody scalars** and feeds AuxMLP → concat with pooled Transformer vector.

---

## 5. H/L mapping

- Prospecting: PDB chain sequence exact-match to competition heavy/light → logical H/L.
- Transformer annotations: `chain ∈ {H,L}`, `seq_index` 0-based; IMGT insertions present in annotations (`imgt_insertion` nonempty on 976 rows) but ARO/HYDRO keys are **sequence indices**, not PDB resseq+icode.
- F1 itself is **Fv-wide aggregate**, not chain-split 35D.

---

## 6. Sequence-index ↔ structure mapping

- ARO/HYDRO: standard AA in PDB chain order → `seq_i = 0,1,…`; validated against competition sequences.
- Transformer / RASA cache: same sequence-index convention in `annotations.parquet`.
- **Critical gap:** there is no frozen table `(id, chain, seq_index) → F1 SURFACE channels`.

---

## 7. Missingness / coverage

| Check | Result |
|-------|--------|
| H047 / aromatic_topo / hydro_field ids | 324/324 |
| H047 ARO↔bundle / HYDRO↔bundle max abs (id-aligned) | **0** |
| surface_field ids | 324 |
| RASA cache mapped both chains | 324 (separate asset) |
| F1 NaNs requiring science-level repair | none material |

Coverage of antibody-level F1 is complete. Residue-token SURFACE rows for F1 are simply **absent**.

---

## 8. Insertion codes / numbering

- IMGT insertions exist in annotations; numbering_status OK.
- F1 extractors do not key on insertion codes; they use contiguous sequence indices after AA reconstruction.
- No evidence of silent F1↔Transformer token shift in late fusion (because F1 is not token-aligned at all).

---

## 9. Does every Transformer token have an unambiguous SURFACE row?

**NO.**

- Special tokens: N/A to F1 (antibody vector only).
- Valid residues: no F1 residue SURFACE vector exists to attach.
- Padding: N/A.
- Cannot “shift/truncate” what does not exist; the scientific prerequisite fails earlier.

---

## 10. Related residue assets (must NOT be substituted silently)

### F3 LOCAL_RASA_CDR3 (`rasa_*.npy` + ESM2 residue pack)

- Residue-aligned RASA exists and is scientifically real.
- It is the upstream of **F3**, not **F1_SURFACE**.
- Historical F3 late fusion on H071/H061 **did not help** (e.g. H092 TEST_mean ≈ 0.527).
- Using RASA alone as “SURFACE” for H128–H133 would invent a different feature definition than the batch’s SURFACE research target.

### `residue_geometry_sasa.parquet` (spatial hydrophobicity research)

- Separate research cache; not the frozen F1 SURFACE recipe; must not be promoted as F1 residue SURFACE without a new, explicit feature-research unlock.

---

## 11. Historical antibody-level SURFACE late-fusion controls (reuse, do not retrain)

From `results/HIC_H047_FUSION_TABLE.csv` / `experiments.csv`:

| Code | Backbone | Bundle | TEST_mean | Overall |
|------|----------|--------|----------:|--------:|
| EXP-H071 | Scratch | NONE | 0.5017 | 0.4798 |
| EXP-H061 | ESM2 | NONE | 0.5068 | 0.4716 |
| **EXP-H090** | H071 | **F1_SURFACE** | **0.4715** | **0.4094** |
| **EXP-H086** | H061 | **F1_SURFACE** | **0.4863** | **0.4063** |
| EXP-H082 | H054 | F1_SURFACE | 0.4867 | 0.4221 |

Antibody-level SURFACE late fusion already improves both Scratch and ESM2. The open question was whether **residue correspondence** adds further value — which cannot be tested without residue-aligned F1 channels.

---

## 12. Gate decision

```
scientifically_valid_residue_aligned_SURFACE_for_F1 = NO
reason:
  - canonical F1 is antibody-level 35D only
  - ARO residue intermediates were never persisted
  - HYDRO intermediates are vertex-level, not residue-token-aligned
  - deriving residue vectors from antibody aggregates is forbidden
  - substituting RASA/F3 or research SASA would redefine SURFACE
action:
  STOP before implementing/training EXP-H128..H133
next_HIC_remains: EXP-H128
```

### What would unlock a future residue-SURFACE batch (out of scope here)

1. Explicit feature-research task to **materialize and freeze** residue channels that are definitionally part of F1 (e.g. per-residue SASA/RASA/aromatic/exposure flags used by ARO), with hash + Transformer token audit; **and/or**
2. A justified residue reduction of HYDRO (vertex→residue) with scientific specification — not silent invention in a model batch; **and**
3. Separate decision whether F3 RASA is in-scope as a residue SURFACE proxy (currently it is a different historical family).

Until then, cross-attention and residue fusion modes A/B/C are **not scientifically executable** under this batch’s constraints.

---

## Audit checklist (requested items)

| # | Item | Finding |
|---|------|---------|
| 1 | Exact historical antibody SURFACE set | F1_SURFACE 35D from H047 / aromatic_topo+hydro_field |
| 2 | Upstream residue-level source | ARO ephemeral residue table; HYDRO vertex mesh |
| 3 | Residue-level columns/channels | Not frozen for F1; HYDRO has H/phi/area at vertices |
| 4 | Feature dimension p | Antibody p=35; residue p for F1 = **undefined** |
| 5 | H/L mapping | Sequence exact-match; F1 is Fv aggregate |
| 6 | seq↔structure mapping | Sequence index; no F1 residue table |
| 7 | Missing residues | Antibody F1 complete; residue F1 rows absent |
| 8 | Missing structures/features | None for antibody F1 |
| 9 | Insertion codes | Present in annotations; F1 uses seq index |
| 10 | Every token has SURFACE row? | **NO** |

**STOP.**
