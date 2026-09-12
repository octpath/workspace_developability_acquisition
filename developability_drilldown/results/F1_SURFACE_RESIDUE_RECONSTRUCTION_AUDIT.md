# F1_SURFACE Residue-Level Provenance Reconstruction Audit

**Verdict: PASS-EXACT**  
**H128 issued: NO** (next HIC remains `EXP-H128`)  
**Training: NONE**

Machine-readable twin: `results/F1_SURFACE_RESIDUE_RECONSTRUCTION_AUDIT.yaml`  
PASS criteria (frozen a priori): `feature_research/f1_surface_residue_reconstruction/PASS_CRITERIA.md`

---

## A. Historical F1_SURFACE definition

| Item | Value |
|------|-------|
| Composition | ARO19 + HYDRO16 = **35D** |
| Concat order | AROMATIC_TOPO → HYDRO_FIELD |
| Canonical matrix | `experiments/features/EXP-H047.parquet` slices `feat[1395:1430]` |
| Standalone | `top_models_feature_bundle/data/aromatic_topo.parquet` + `hydro_field.parquet` |
| Loader | `H047AuxFeatureStore("F1_SURFACE")` |
| Structure generator | ESMFold Fv via `STRUCTURE_INPUT_CROSSWALK_v2.csv` → `esmfold_canonical_path` |

### Definition table (executable code)

| block | column | exact definition | upstream source | aggregation | reproducible? |
|-------|--------|------------------|-----------------|-------------|---------------|
| ARO | `aro_exposed_{TYR,TRP,PHE}_count` | count exposed (RASA≥0.20) aromatics of type | Bio.PDB ShrakeRupley residue SASA/RASA | residue→Ab | YES |
| ARO | `aro_exposed_aromatic_total_count` | \|exposed FWY\| | same | residue→Ab | YES |
| ARO | `aro_aromatic_exposed_SASA_total` | Σ SASA(exposed FWY) | same | residue→Ab | YES |
| ARO | `aro_aromatic_exposed_SASA_fraction` | total / Σ SASA(all residues) | same | residue→Ab | YES |
| ARO | `aro_strongly_exposed_*` | RASA≥0.50 aromatic count/SASA | same | residue→Ab | YES |
| ARO | `aro_CDR_exposed_*` / fraction | exposed arom ∩ CDR | CDR map `cdr_sequence_index_imgt.csv` | residue→Ab | YES |
| ARO | `aro_aromatic_patch_*` | Cα CC among exposed arom, PATCH_R=8Å | residue CA | residue→Ab | YES |
| ARO | `aro_max_local_aromatic_SASA` | max neighborhood SASA, LOCAL_R=10Å | residue CA | residue→Ab | YES |
| ARO | `aro_sequence_*_count` | FWY counts ignoring exposure | sequence/structure AA | residue→Ab | YES |
| HYDRO | `mean_H_surface` … `CDR_high_H_area_fraction` | area-weighted / quantile / patch stats of H(s) | FreeSASA Lee–Richards surface points + Fauchère–Pliska field | vertex→Ab | YES |
| HYDRO | `n_surface_points` | \# surface points after cap 2000 | surface construction | vertex→Ab | YES |
| HYDRO | `phi_finite_frac` | QC: fraction finite APBS φ on vertices | historical `surface_field_esmfold.parquet` φ (coords identical) | vertex→Ab | YES (all = 1.0) |

Code paths:

- ARO: `structure_utils.build_residue_table` + `extract_physical_batch1.aromatic_features`
- HYDRO: `hydro_surface.build_surface_and_field` + `hydro_summaries` (reconstruction keeps parent atom/residue)

---

## B. Exact H090 / H086 provenance

```
EXP-H090.yaml / EXP-H086.yaml
  fusion_bundle_id: F1_SURFACE
  fusion_mode: late_concat_aux32
        ↓
H047AuxFeatureStore → EXP-H047.parquet ARO‖HYDRO (35D)
        ↓
LateFusion AuxMLP 35→32 + Transformer backbone (H071 / H061)
```

Bit-identity confirmed: H047 F1 columns ≡ aromatic_topo ≡ hydro_field (id-aligned max abs = 0).

---

## C. ARO19 reconstruction path

```
esmfold_native/{id}.pdb
  → build_residue_table (ShrakeRupley probe=1.4, n_points=100)
  → per-residue channels (sasa, rasa, flags, CA, CDR, …)
  → aromatic_features aggregation
  → ARO19
```

Persisted: `feature_research/f1_surface_residue_reconstruction/features/residue_surface_aro.parquet`  
(75,057 residue rows = annotations length; mapping_status `OK_SEQ_EXACT` for all)

---

## D. HYDRO16 reconstruction path

```
same PDB + sequences + CDR
  → heavy-atom FreeSASA Lee–Richards
  → Fibonacci SAS points from each atom with SASA≥0.5
  → keep originating atom → (chain, seq_index, pdb_resseq, atom_name)
  → H(s) MLP field (α=1, cutoff=7Å)
  → hydro_summaries → HYDRO15 + phi_finite_frac
```

**Assignment method:** `originating_atom_during_surface_construction`  
(not post-hoc nearest-residue). Vertex coordinates match historical `surface_field_esmfold.parquet` with **max abs = 0** for all 324 antibodies (305,590 vertices).

`phi_finite_frac`: historical values are identically 1.0; attached from historical φ on identical vertices.

Persisted:

- `residue_surface_hydro_vertices.parquet` (vertex + residue provenance; required for exact reaggregation)
- `residue_surface_hydro.parquet` (residue rollups for inspection; **not** sufficient alone for quantiles/patches)

---

## E. Residue ↔ sequence alignment

- Chain map: PDB chain sequence **exact match** to competition `heavy` / `light`
- Sequence index: contiguous 0-based over standard AA in chain order
- Cross-check vs `annotations.parquet`: **0** AA mismatches; 324/324 H and L present
- Insertion codes: column recorded; ESMFold Fv vertices show empty icode in this set
- Transformer: `transformer_chain_token_index = sequence_index`; `special_token=False` (special tokens never receive SURFACE)

Aligned table: `residue_surface_aligned.parquet`

---

## F. Residue-level artifact schema (summary)

**ARO residue row:** antibody_id, chain, sequence_index, sequence_aa, structure ids, sasa, rasa, exposure/CDR flags, CA, mapping_status, availability_aro  

**HYDRO vertex row:** vertex coords, H, area, is_cdr, phi, chain, sequence_index, structure_residue_id, source_atom_name/element, assignment_method  

**Aligned:** ARO residue ⊕ optional hydro residue rollups + availability_hydro

---

## G–I. Round-trip results

| Metric | Value |
|--------|------:|
| Antibodies reconstructed | **324 / 324** |
| Failures | **0** |
| F1 max abs error | **2.27e-13** |
| Antibodies with max‖err‖≤1e-10 | **324 / 324** |
| ARO max abs | 2.27e-13 |
| HYDRO max abs | **0.0** |
| Per-column Pearson r | ≈ 1.0 (all) |

No antibody-wise hard failures. Residual ARO errors are float noise on SASA-derived floats.

---

## J. Mapping / coverage

| Check | Result |
|-------|--------|
| Structures available | 324/324 |
| H / L mapping | 324/324 |
| ARO residue coverage | 75057 rows, all OK |
| HYDRO vertex coverage | 305590, all with originating atom |
| Full 35D round-trip exact | 324/324 |

---

## K. Verdict

**PASS-EXACT**

Both ARO and HYDRO blocks reconstruct from provenance-preserving intermediates within numerical precision for all antibodies.

---

## L. Faithful residue-level F1 source exists?

**YES.**

- ARO: true residue table sufficient to reaggregate ARO19  
- HYDRO: surface vertices with **native originating-atom→residue** labels sufficient to reaggregate HYDRO16  
- Combined reaggregation matches historical F1_SURFACE used by H090/H086  

Note: HYDRO quantiles/patches remain **vertex-level** operations; residue rollups alone do not replace the vertex cloud. The scientifically faithful residue-linked source is the **vertex+residue** artifact (plus ARO residue table).

---

## M. Is H128–H133 residue fusion now technically justified?

**Technically: YES (PASS-EXACT + full coverage).**  
**This task does NOT run H128–H133.** Reconsider only under a **new instruction**.

Cross-attention still not authorized by this audit alone; simpler residue fusion modes may now be executable if requested.

---

## N. Git / tests / hashes

| Artifact | Role |
|----------|------|
| `feature_research/f1_surface_residue_reconstruction/scripts/reconstruct_f1_residue_provenance.py` | reconstruction |
| `features/*.parquet` | residue / vertex / round-trip tables |
| `developability_drilldown/tests/test_f1_surface_residue_reconstruction.py` | regression tests |

Historical H090/H086 metrics untouched. Next HIC: **EXP-H128** (unissued).

**STOP.**
