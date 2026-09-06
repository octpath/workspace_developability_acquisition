# Structure Feature Inventory (pre–Structure Marathon)

**Generated:** 2026-09-06  
**Scope:** Read-only survey of `organizer_extension/feature_prospecting/` and related `virtual_participant/` artifacts.  
**Authority for campaign plan:** `STRUCTURE_MARATHON_PLAN_LOCK.md` / `STRUCTURE_MARATHON_STATE.json`  
**Note:** This file inventories existing work only. Source family directories were not modified.

**Status legend**

| Status | Meaning |
|--------|---------|
| **COMPLETED** | SPEC + features/eval artifacts + report (or explicit complete manifest) present |
| **PARTIAL** | SPEC and/or report exist, but extraction/eval blocked, incomplete, or scaffolding-only |
| **NOT_FOUND** | No executed prospecting report/CSV/SPEC package for this item |

---

## Summary counts (requested items)

| Status | Count |
|--------|------:|
| COMPLETED | **21** |
| PARTIAL | **1** (SURFACE-DL) |
| NOT_FOUND | **1** (structure-guided PLM pooling) |

**S1–S4 useful gaps (see § Gaps):** all four marathon P0 families are still `PENDING` with empty workdirs; Round1 RASA/SASA/patch/contact and prospecting AROMATIC/HYDRO/STATIC-SAP are reusable priors, not S1–S4 deliverables.

---

## Per-item inventory

| Item | Status | Key path(s) | One-line note |
|------|--------|-------------|---------------|
| **ANM** (`ANM-SPECTRUM`) | COMPLETED | `.../ANM-SPECTRUM/` — `FEATURE_SPEC.json`, `features_{esmfold,abodybuilder2,boltz2}.parquet`, `EVALUATION_RESULTS.json`, `REPORT_JA.md`, `GATE2A_ANM_SPECTRUM_V1_FREEZE_MANIFEST.json` | Gate2A complete; TmApp/HIC → `NO_EVIDENCE_IN_CURRENT_DATA` (softness univariate only). |
| **VHL** (`VHL-ANGLE`) | COMPLETED | `.../VHL-ANGLE/` — `FEATURE_SPEC.json`, features parquet×3, `EVALUATION_RESULTS.json`, `REPORT_JA.md`, `GATE2B_VHL_ANGLE_V1_FREEZE_MANIFEST.json`; review `.../VHL-ANGLE_v1_TECHNICAL_REVIEW.md` | Gate2B complete; ESMFold/Boltz2 valid; **ABB2 ABangle mapping `TECHNICAL_MAPPING_CONCERN`** (do not silently “fix” v1). |
| **Corrected PKA** | COMPLETED | `.../PKA-SHIFT/TECHNICAL_CORRECTION/` — `GATE2C1_*_FREEZE_MANIFEST.json`, corrected `features_*.parquet`, `EVALUATION_RESULTS.json`, `PKA_SENTINEL_AUDIT_{JA.md,json}`; parent report `PKA-SHIFT/TECHNICAL_CORRECTION_REPORT_JA.md` | Disulfide Cys PROPKA `99.99` sentinel excluded; TmApp ESMFold still **MIXED**; HIC **NO_EVIDENCE**. |
| **Original sentinel-contaminated PKA** | COMPLETED | `.../PKA-SHIFT/` — `FEATURE_SPEC.json`, original `features_*.parquet`, `REPORT_JA.md`, `GATE2C_PKA_SHIFT_V1_FREEZE_MANIFEST.json`, `EVALUATION_RESULTS.json` | Immutable v1 kept; labeled **`TECHNICALLY_COMPROMISED_BY_PROPKA_DISULFIDE_SENTINEL`** (~85–89% of \|ΔpKa\| mass from sentinel). |
| **AROMATIC-TOPO** | COMPLETED | `.../AROMATIC-TOPO/` — `FEATURE_SPEC.json`, features parquet×3, `EVALUATION_RESULTS.json`, `REPORT_JA.md`, `GATE2_BATCH1_AROMATIC-TOPO_V1_FREEZE_MANIFEST.json`; master `PHYSICAL_BATCH1_REPORT_JA.md` | Physical Batch1; HIC standalone **REPRODUCIBLE** but residual **NO_INCREMENT** → promising-but-redundant; strongest fusion partner later. |
| **STATIC-SAP** | COMPLETED | `.../STATIC-SAP/` — `FEATURE_SPEC.json`, features parquet×3, eval/report/freeze; master `PHYSICAL_BATCH1_REPORT_JA.md` | Batch1 complete; primary HIC claim **NO_EVIDENCE** on ESMFold despite mechanistic prior. |
| **POLAR-SAT** | COMPLETED | `.../POLAR-SAT/` — `FEATURE_SPEC.json`, features parquet×3, `residue_level.parquet`, eval/report/freeze; master `PHYSICAL_BATCH1_REPORT_JA.md` | Batch1 complete; TmApp standalone REPRODUCIBLE / **NO_INCREMENT** (redundant); robustness **FRAGILE**. |
| **HYDRO-FIELD** | COMPLETED | `.../HYDRO-FIELD/` — `FEATURE_SPEC.json`, features + `surface_field_*.parquet`, eval; master `ADVANCED_BATCH2_REPORT_JA.md` | Advanced Batch2; continuous Fauchère MLP/FreeSASA; HIC Ridge **NO_EVIDENCE** (does not replace AROMATIC-TOPO). |
| **ELEC-HYDRO-COPATCH** | COMPLETED | `.../ELEC-HYDRO-COPATCH/` — `FEATURE_SPEC.json`, features parquet×3, eval; master `ADVANCED_BATCH2_REPORT_JA.md` | Batch2; APBS×hydro co-patch adds no HIC beyond hydrophobicity; **FRAGILE**. |
| **INTERFACE-ENERGY** | COMPLETED | `.../INTERFACE-ENERGY/` — `FEATURE_SPEC.json`, features parquet×3, `interface_energy_qc.csv`, `geometry_artifact_audit.json`; master `ADVANCED_BATCH2_REPORT_JA.md` | Batch2; OpenMM fixed-coord VH–VL proxy; TmApp **UNLIKELY** + **GEOMETRY_ARTIFACT_CONCERN**. |
| **3DI** (`3DI-FROZEN`) | COMPLETED | `.../3DI-FROZEN/` — `FEATURE_SPEC.json`, `embeddings_*.parquet`, `tokens_*.parquet`, features parquet×3, eval; master `ADVANCED_BATCH2_REPORT_JA.md` | Foldseek3Di→ProstT5 HL_CONCAT PCA32; TmApp/HIC **NO_EVIDENCE**; emb robustness MODERATE. |
| **TITRATION** (`TITRATION-SHAPE`) | COMPLETED | `.../TITRATION-SHAPE/` — `FEATURE_SPEC.json`, features + `titration_curves_*.parquet`, `REPORT_JA.md`, eval; `LATE_BATCH3_COMPLETE_MANIFEST.json` | Late Batch3; PROPKA HH Q(pH); generator **ROBUST**; TmApp **MIXED** / HIC **NO_EVIDENCE**. |
| **OPENMM-STRAIN** | COMPLETED | `.../OPENMM-STRAIN/` — `FEATURE_SPEC.json`, features parquet×3, `geometry_artifact_audit.json`, `REPORT_JA.md`; Late Batch3 manifest | Local minimizer strain; **FRAGILE**; vacuum force heavy-tail; TmApp **MIXED**. |
| **SURFACE-DL** | PARTIAL | `.../SURFACE-DL/` — `FEATURE_SPEC.json`, `REPORT_JA.md`, `EVALUATION_RESULTS.json` (no feature parquet) | **`METHOD_BLOCKED_DEPENDENCY_OR_LICENSE`** (dMaSIF); acceptable blocked result, not a scored family. |
| **Fusion Closure** | COMPLETED | `.../fusion_closure/` — `FUSION_CLOSURE_SPEC.json`, `FUSION_CLOSURE_REPORT_JA.md`, `FEATURE_FUSION_SUMMARY_CURRENT.csv`, `FUSION_EXPERIMENT_REGISTRY.csv`, `results/` | Gate 3A–3E complete; HIC×AROMATIC concat strongest complementary physics story. |
| **Preprocessing audit** | COMPLETED | `.../fusion_closure/preprocessing_stability_audit/` — `PREPROCESSING_STABILITY_AUDIT_JA.md`, `PREPROCESSING_IMPLEMENTATION_AUDIT.md`, `STABILITY_AUDIT_THRESHOLDS.json`, `results/*.csv` | AROMATIC scaler stable; ESM2 PCA32 fold-dependent; preprocessing not the fusion bottleneck. |
| **Modulation fusion** | COMPLETED | `.../fusion_closure/modulation_fusion/` — `MODULATION_FUSION_SPEC.json`, `MODULATION_FUSION_REPORT_JA.md`, `results/PRIMARY_RESULTS.csv`, `M1_FEATURE_IMPORTANCE.csv` | Explicit interactions/FiLM worse than concat → `CONCAT_ALREADY_SUFFICIENT`. |
| **FeNNix v1** | COMPLETED | `.../foundation_stability/` — `FOUNDATION_FEATURE_SPEC.json`, `FOUNDATION_STABILITY_REPORT_JA.md`, `FOUNDATION_*_RESULTS.csv`, `results/FOUNDATION_ALL_EVAL.csv` | FeNNix-Bio1S full Fv eval; best ABB2 standalone looked strong but **clash-artifact** risk; not recommended for final model. |
| **FeNNix v2** | COMPLETED | `.../foundation_stability_v2/` — `FOUNDATION_STABILITY_V2_SPEC.{md,json}`, `FENNIX_V2_{CURVATURE_FEATURES,RESULTS,RELAXATION_QC,GENERATOR_ROBUSTNESS}.csv`, `FOUNDATION_STABILITY_V2_REPORT_JA.md`, `FENNIX_V1_V2_ARTIFACT_COMPARISON.csv` | Clash dependence largely removed vs v1; **NO_INCREMENT** / FRAGILE across generators. *(Fab-context follow-on `fennix_fab_context/` is in-progress pilot — not v2 Fv package.)* |
| **BioEmu v1.2** | COMPLETED | `.../foundation_stability_v2/` — `BIOEMU_V12_FEATURES.csv`, `BIOEMU_V12_RESULTS.csv`, `BIOEMU_SAMPLE_COUNT_DECISION.md`, `BIOEMU_*CONVERGENCE*`, v2 report | N=16 raw samples frozen; 324 Abs VH+VL; standalone≈median; **NO_INCREMENT**; ΔG proxy scientifically unjustified. |
| **BioEmu physical-frame reassessment** | COMPLETED | `.../bioemu_isolated_reassessment/` — `BIOEMU_ISOLATED_REASSESS_SPEC.{md,json}`, `BIOEMU_ISOLATED_REASSESS_REPORT_JA.md`, `*_STANDALONE/INCREMENTAL/RESIDUAL/BOOTSTRAP.csv`, `BIOEMU_ISOLATED_PHYSICAL_CONVERGENCE.csv`, `BIOEMU_ISOLATED_NPHYS_DECISION.md` | Corrected Nphys=8 framing; **`BIOEMU_ISOLATED_CONFIRMED_NO_INCREMENT`**. |
| **RASA / SASA features** | COMPLETED | Round1: `virtual_participant/round1_pdb_feature_inventory/ROUND1_PDB_FEATURE_INVENTORY_JA.md`; Stage3: `virtual_participant/stage3_structure/` (`features_ESMFold.csv` / packing caches); extractors `gate_b2/scripts/03_structure_features.py` | Shrake–Rupley SASA + RASA (Tien MaxASA); **STRUCT_SASA / STRUCT_RASA / STRUCT_SURFACE_CHEM** (~84+ region chem); not a prospecting family but production Round1 features. |
| **Structure-guided PLM pooling** | NOT_FOUND | Plan only: `structure_marathon/STRUCTURE_MARATHON_PLAN_LOCK.md` (S1 masks); empty `structure_marathon/structure_guided_pooling/`; state `S1_structure_guided_pooling: PENDING` | No SPEC/features/report of executed pooling; closest priors = Round1 mean PLM embeds + RASA thresholds in plan lock (not implemented). |

---

## Structure generators (Fv)

Canonical three-generator stack used across prospecting families:

| Generator | Role | Primary PDB / structure roots | Crosswalk |
|-----------|------|-------------------------------|-----------|
| **ESMFold** | Primary Fv | `/workspace_developability_acquisition/esmfold_native/{id}.pdb` (~370 PDBs); hash-mirrored under `gate_b2/cache/structures/esmfold_native/`; also `gate_b1/cache/structures/esmfold/` | `STRUCTURE_INPUT_CROSSWALK_v2.csv` cols `esmfold_*` |
| **ABodyBuilder2 (ABB2)** | Secondary Fv | `gate_b1/cache/structures/abodybuilder2/{pair_hash}.pdb` (~400); mirrored `gate_b2/cache/structures/abodybuilder2/` | col `abodybuilder2_path` |
| **Boltz2** | Tertiary Fv (`BOLTZ2_FV_STANDARD_v1`) | `organizer_extension/feature_prospecting/structure_sources/boltz2_fv_standard_v1/structures_pdb/{id}.pdb` (324) + `structures_mmcif/`; SPEC/manifest/audit in same dir | cols `boltz2_pdb_path`, `boltz2_native_cif_path` |

**Registry note:** Almost all prospecting families declare `structure_source=ESMFold+ABodyBuilder2+Boltz2` in `FEATURE_FAMILY_REGISTRY.csv` (registry `status` column may still say `BACKLOG_*` even when directories are fully executed — trust family freeze manifests / reports).

**Cohort mapping:** `STRUCTURE_INPUT_CROSSWALK_v2.csv` (324 Abs + header) + audit JSON `STRUCTURE_INPUT_CROSSWALK_V2_AUDIT.json`.

---

## Where PDBs live — Fv vs Fab

### Fv (variable domain only)

| Source | Path pattern | Approx. count |
|--------|--------------|---------------|
| ESMFold (antibody-id canonical) | `/workspace_developability_acquisition/esmfold_native/*.pdb` | ~370 |
| ESMFold (pair-hash cache) | `gate_b2/cache/structures/esmfold_native/*.pdb` | (subset/hash map) |
| ABB2 | `gate_b1/cache/structures/abodybuilder2/*.pdb` | ~400 |
| Boltz2 PDB | `.../structure_sources/boltz2_fv_standard_v1/structures_pdb/*.pdb` | **324** |
| Boltz2 mmCIF | `.../structure_sources/boltz2_fv_standard_v1/structures_mmcif/*.cif` | 324 |

Feature extraction for classical prospecting families consumed these Fv generators (not Fab).

### Fab (reconstructed experimental-like)

| Artifact | Path | Notes |
|----------|------|-------|
| ESMFold Fab PDBs | `.../fab_reconstruction/structures/esmfold_fab/{id}.pdb` | **324** full Fab |
| Manifest / QC | `.../fab_reconstruction/structures/ESMFOLD_FAB_STRUCTURE_MANIFEST.csv`, `qc/ESMFOLD_FAB_FULL_QC.csv`, reports under `fab_reconstruction/` | `RECONSTRUCTED_EXPERIMENTAL_LIKE_FAB` |
| Sequences | `.../fab_reconstruction/sequences/` | heavy/light FASTA, junction audit |
| FeNNix-prepared Fab (disulfide OpenMM prep) | `.../fennix_fab_context/cache/prepared_fab/{id}_prepared.pdb` | **~113** prepared so far (`FAB_PREP_QC.csv`); marathon Workstream A pilot |

There is **no** full-cohort ABB2/Boltz2 Fab PDB set in this tree (Fab path is ESMFold reconstruction + optional FeNNix prep).

---

## Related completed packages (not in the numbered list, useful context)

| Package | Path | Status |
|---------|------|--------|
| Physical Batch1 master | `PHYSICAL_BATCH1_{REPORT_JA.md,COMPLETE_MANIFEST.json,...}` | COMPLETED (incl. VOID-EXPLICIT) |
| Advanced Batch2 master | `ADVANCED_BATCH2_{REPORT_JA.md,COMPLETE_MANIFEST.json,...}` | COMPLETED |
| Late Batch3 + postmortem | `LATE_BATCH3_*`, `FEATURE_PROSPECTING_POSTMORTEM_V1_JA.md` | COMPLETED |
| Signal / relevance registries | `FEATURE_RELEVANCE_SUMMARY_CURRENT.csv`, `METHOD_REFERENCE_REGISTRY_CURRENT.csv` | Current empirical labels |
| BioEmu constant-context (Fab chains) | `bioemu_constant_context/` | Separate context/filter workstream (not isolated reassessment) |
| FeNNix Fab context | `fennix_fab_context/` | **PARTIAL / RUNNING** — SPEC frozen; prep CUDA ok≈113; pilot phase (`STRUCTURE_MARATHON_STATE`) |
| Round1 PDB inventory | `virtual_participant/round1_pdb_feature_inventory/` | COMPLETED inventory of Stage3/4 structure features |

---

## Gaps useful for S1–S4

Marathon state (`STRUCTURE_MARATHON_STATE.json`): **S1–S4 all `PENDING`**; workdirs `structure_guided_pooling/`, `generator_disagreement/`, `surface_patch_graph/`, `contact_graph/` are **empty**.

| ID | Family | Gap vs existing assets |
|----|--------|------------------------|
| **S1** | Structure-guided PLM pooling | **No executed pooling.** Need ESM2 residue embeds × geometry masks (CDR/exposed/interface/… from PLAN_LOCK). Reuse Round1 **RASA≥0.25/0.40** thresholds and Stage3 RASA/SASA residue logic; Fab masks need Fab PDBs + domain annotation. |
| **S2** | Cross-generator disagreement | Prospecting already computed many **3-generator** feature sets + robustness CSVs, but **no dedicated disagreement feature family**. Watch **VHL ABB2 mapping concern** and FeNNix/PKA **FRAGILE** generator splits when defining disagreement descriptors. |
| **S3** | Surface-patch graph (HIC) | Round1 `STRUCT_PATCH` / `ADV_SURFACE_PATCH` + prospecting **AROMATIC-TOPO / STATIC-SAP / HYDRO-FIELD** are priors; marathon wants **graph** featurization (empty). Prefer aromatic/exposed hydrophobic patches; full-Fab HIC surface = exploratory per plan lock. |
| **S4** | Contact/packing graph (TmApp) | Round1 `STRUCT_PACKING` / `ADV_INTERACTIONS` / interface cols exist; prospecting **POLAR-SAT / INTERFACE-ENERGY / OPENMM-STRAIN / ANM** are related but not contact-graphs. Build Cβ–Cβ≤8Å graphs fresh per PLAN_LOCK. |

**Do not re-run as S1–S4:** classical prospecting families above, Fusion/Modulation/Preproc audits, FeNNix v1/v2 Fv packages, BioEmu v1.2 + isolated reassessment (plan lock forbids BioEmu rerun).

---

## Quick path index (absolute roots)

```
/workspace_developability_acquisition/organizer_extension/feature_prospecting/
/workspace_developability_acquisition/organizer_extension/feature_prospecting/structure_marathon/
/workspace_developability_acquisition/virtual_participant/
/workspace_developability_acquisition/esmfold_native/
/workspace_developability_acquisition/gate_b1/cache/structures/
/workspace_developability_acquisition/gate_b2/cache/structures/
```
