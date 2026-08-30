# Acquisition Audit Final Report — Developability Gate A0

**Date:** 2026-08-28  
**Working directory:** `/workspace_developability_acquisition`  
**Isolation:** `/workspace` used as read-only reference only; no edits, installs, or interference with C3a Gate 6.

---

## Executive table

| Candidate | Feasibility category | Grade | Complete seq+label (best target) | Notes |
| --------- | -------------------- | ----- | -------------------------------: | ----- |
| **Jain 2017** | READY_FOR_MODELING_GATE | A/B | 137 × many assays | Cleanest antibody multi-assay join already on disk |
| **eSOL / Niwa** | READY_AFTER_MINOR_CURATION | B | 3167 / 3173 solubility | Original labels + UniProt sequences |
| **Shehata 2019 (PSR)** | READY_AFTER_MINOR_CURATION | B | 398 paired + PSR | Tm/HIC still need original mmc2 |
| **NbThermo** | READY_AFTER_MINOR_CURATION | B/C | 514 seq+Tm | Method heterogeneity must be preserved |
| **Meltome Atlas** | READY_AFTER_MINOR_CURATION | B | large Tm index; seq via UniProt | Condition columns mandatory |
| **Ginkgo GDPa1** | PROMISING_BUT_ACCESS_BLOCKED | D | 0 (auth only blocker) | Best Ab-specific multi-assay candidate post-token |
| **Ginkgo GDPa3** | PROMISING_BUT_ACCESS_BLOCKED | D | 0 | ~80 Abs; location gated / not on HF org API |
| **DeepViscosity** | INSUFFICIENT_PUBLIC_DATA | E | ~0 high-confidence joins | AZ proprietary 229; partial viscosity leak only |

**Advance to future modeling Gate (recommended 2–4):**  
1. **Jain 2017** (multi-assay Ab developability)  
2. **eSOL** (diverse proteins → solubility)  
3. **Shehata PSR** (diverse human Abs → polyreactivity)  
4. **GDPa1** (once HF access approved) as the leading antibody multi-assay expansion  

---

## Ginkgo GDPa1 / GDPa3

### 1. Can we actually obtain the files?
**Not in this session without authentication.**  
Public HF metadata is readable. File bytes require accepting gated terms + login.  
Visible after gate (tree listing works for GDPa1 when listing metadata):

- `GDPa1_v1.2_20250814.csv` (~862 KB)  
- `GDPa1_v1.2_20250814_full.xlsx` (~978 KB)  
- `structures/`  
- `LICENSE.md`, `README.md`

Downloader prepared: `scripts/download_ginkgo.py` (uses `HF_TOKEN`).

### 2. Is authentication the only blocker?
**For GDPa1: yes → `ACCESS_BLOCKED_ONLY_BY_AUTHENTICATION`.**  
Automated download should work once a token with accepted terms is provided.

**For GDPa3:** not listed under `ginkgo-datapoints` HF org datasets API (only GDPa1 + GDPx1–4). Competition paper points to `https://datapoints.ginkgo.bio/dataset-access`. Treat as gated / location-unclear until an approved account can enumerate files.

### 3. Are VH/VL directly provided?
**GDPa1:** Yes, per README — full Excel includes antibody sequences.  
**GDPa3:** Expected (OAS-sampled paired Abs); not verified on disk.

### 4. Which developability assays are available?
GDPa1 (10 assays per README): titer, rCE-SDS purity, SEC aggregation, nanoDSF/DSF thermostability, SMAC, HIC, HAC, AC-SINS, polyreactivity (CHO SMP / ovalbumin), DLS-kD (subset ~10).  
GDPa3 (competition materials): HIC, SEC, nanoDSF, titer, polyreactivity (CHO).

### 5. Complete sequence+label antibodies per assay?
**0 downloaded.** After access, expect ~242 (processed CSV) or up to 246 (full batch) with per-assay missingness; build assay completeness matrix from Excel tidy + summary sheets.

### 6. Can we redistribute?
**LICENSE_REVIEW_REQUIRED / restricted.** LICENSE.md: CC BY 4.0 **for commercial use not involving sale/transfer/licensing of the Dataset itself**. Personal download ≠ competition redistribution. Gating + contact-share terms affect logistics.

### 7. What explains 242 vs 246?
Do **not** silently pick one:

| Count | Where it appears | Meaning |
| ----: | ---------------- | ------- |
| **246** | Paper abstract; production-batch note; HF blog | Unique Abs in study / first production batch (106 approved + 135 clinical + 5 prereg/withdrawn) |
| **242** | HF README intro; processed averaged CSV description | Release used for modeling table after averaging |

README processing: keep first production batch (contained all 246), median across replicates. The **4-Ab gap is not itemized in public metadata**; likely incomplete-assay filtering in the processed CSV. Confirm against full Excel `Versioning` sheet after access.

### 8. Targets for a future modeling Gate (post-access)?
Priority: **VH+VL → Tm2 (nanoDSF)**; also AC-SINS, HIC, SEC, polyreactivity as separate targets (do not collapse). Use `hierarchical_cluster_IgG_isotype_stratified_fold` if provided.

**Placeholder:** `interim/ginkgo_antibody_index.csv` (schema only until download succeeds).

---

## eSOL

### 1. Original experimental table obtained?
**Yes.** `raw/esol/esol.zip` → `esol.csv` from LSDB Archive (NBDC00440).

### 2. How many experimental labels?
**3173** with `Solubility (%)` (of 4132 archive rows). 959 lack solubility (unquantified ORFs per Niwa et al.).

### 3. Unambiguous sequence mappings?
**3167 / 3173** via B-number → UniProt reviewed proteome (UP000000625).  
**6 unresolved; 0 ambiguous** after priority join (B > JW > gene).

Artifact: `interim/esol_reconstructed.csv`, `interim/esol_join_audit.csv`.

### 4. Why 3173 / 3198 / 2943 etc.?
| Count | Likely origin |
| ----: | ------------- |
| **4132** | Full ASKA/ORF archive entries |
| **3173** | Quantified solubility in Niwa PNAS 2009 / eSOL |
| **788** | Aggregation-prone subset with chaperone add-on (2012) |
| **3198 / 2943** | Downstream ML filters (length, mapping failures, membrane exclusion, etc.) — not original experimental N |

Prefer **3173 original labels**; document any filter when deriving competition subsets.

### 5. Clean sequence→solubility table?
**Yes, Grade B.** Join is reproducible; provenance columns included. Note: 9 distinct sequences associated with >1 solubility (multi-locus identical sequence) — keep as separate IDs or flag.

### 6. Annotation/version mismatches?
Modern UniProt sequences used. Gene renames cause residual unresolved (6). Material isoform conflicts not dominant after B-number priority. For strict historical fidelity, optionally pin UniProt release; not done here.

**License:** CC BY-SA 2.1 Japan — redistribution of derivatives allowed with attribution + share-alike.

---

## DeepViscosity

### 1. Complete 229-mAb sequence set recoverable?
**No.** Paper: *“DV_mAb_229 datasets … are proprietary data and were therefore not shared.”*  
Repo references `data/AZ_seq_vis.csv` etc. — **directory never committed**.

### 2. Paired VH/VL recoverable?
Only **16** demo/example Abs in `DeepViscosity_input.csv` / FASTA (predictor inputs), not the training set.

### 3. Continuous viscosity recoverable?
**Partial only:** 153 continuous values mined from `sequence_grouping.ipynb` outputs (`interim/deepviscosity_labels_partial.csv`). No joinable public sequences for those anonymized `mAbN` IDs.

### 4. Binary labels?
Threshold **20 cP** is defined, but no public binary table for 229. Binary is not independently recoverable beyond thresholding the partial continuous leak.

### 5. Labels in figures vs tables/files?
Primary continuous labels are **proprietary files**, not SI tables. Notebook outputs accidentally expose a subset. **Plot digitization not used** (per policy).

### 6. Reconstruct from IDs/PDB/public DBs?
Anonymized `mAbN` IDs — **no auditable identity** to therapeutics/PDB.

### 7. High-confidence joined rows?
**≈0 for competition use** (status reports ≤6 demo overlaps with viscosity IDs).

### 8. Comparable formulation?
Paper reports uniform condition: **20 mM His-HCl, pH 6.0, 150 mg/mL, 25 °C** — scientifically clean *if* data were public.

### 9. Final status
**`PUBLIC_SEQUENCE_LABEL_JOIN_NOT_RECOVERABLE`** for full 229 continuous.  
Also: `CONTINUOUS_TARGET_NOT_RECOVERED` (full); `PARTIAL_DATASET_RECONSTRUCTABLE` (153 labels without sequences).  
**Not** `FULL_229_BINARY_ONLY_RECONSTRUCTABLE`.

---

## Shehata

### 1. Exact antibody count?
**400** Abs with sequences in supplement lineage; **398** with numeric PSR after dropping 2 missing-PSR rows (HF/Obstacle processing). Archive Excel reportedly 402 rows including legend lines.

### 2. Paired sequences public?
**Yes** in derivatives: `interim/shehata_labels.csv` + `interim/shehata_sequences.fasta` (398 VH+VL).

### 3. Assay labels joinable?
**PSR:** yes (continuous + binary at >0.33 → 7 high / 391 low).  
**Tm / HIC / charge:** reported in original `shehata-mmc2.xlsx` but **raw xlsx not present** in public git and Cell Reports is **not open access** (EuropePMC: no PMC/PDF/Suppl). Status: **PARTIALLY_RECONSTRUCTABLE** for non-PSR assays.

### 4. Complete row counts
| Target | n labels | n paired VH/VL | n complete |
| ------ | -------: | -------------: | ---------: |
| PSR score | 398 | 398 | 398 |
| PSR binary | 398 | 398 | 398 |
| Tm / HIC | unknown public | — | 0 |

### 5. Ambiguity / leakage metadata
B-cell subset (`IgG memory`, `IgM memory`, `Naïve`, `LLPCs`) is both scientific signal and potential **leakage/stratification** covariate — preserve, do not ignore. Extreme class imbalance for binary PSR.

### 6. Cleanest competition assay?
**Continuous PSR** (or carefully stratified binary) with paired VH+VL. Prefer continuous over 7/391 binary.

---

## Jain 2017 / NbThermo / Meltome

### Jain et al. 2017
- **137** clinical-stage Abs; SD02 VH+VL + SD03 12 assays joined → `interim/jain2017_joined.csv`.  
- Direct PNAS supplement URLs returned **403** here; CSVs recovered from public Obstacle pipeline (converted from official SD files).  
- Assays kept separate: AC-SINS, CSI-BLI, HIC, PSR, Fab Tm, accelerated stability slope, SMAC, CIC, ELISA, BVP, titer, SGAC-SINS.  
- **Grade A/B.** Ready for modeling Gate.  
- Jain 2024 / HEL discovery set: **not deeply audited** this session (Tier 2 timebox).

### NbThermo
- JSON dump: **548** entries (paper cites **564** — discrepancy recorded).  
- **514** sequence+Tm; methods mixed (CD 304, DSF 254, nanoDSF 72, DSC 16, …) across **62** DOIs.  
- Do not reject for heterogeneity — **condition/method columns required**.  
- Artifact: `interim/nbthermo_flattened.csv`.

### Meltome Atlas
- Nature MOESM3–11 downloaded (no raw PRIDE MS).  
- MOESM4: **358,672** rows; **65,239** protein IDs; **305,237** non-null Tm across 77 dataset sheets.  
- MOESM3 metadata: organism, strain/tissue, lysate vs cells, OGT, etc.  
- Sequences: join Protein ID → UniProt (batch fetch deferred; scripts ready).  
- **Grade B** with mandatory condition separation.

---

## Minimal integrity stats (reconstructed sets)

| Dataset | raw label rows | unique IDs | unique seqs | exact dup seqs | conflicting dup labels | missing seq | missing target |
| ------- | -------------: | ---------: | ----------: | -------------: | ---------------------: | ----------: | -------------: |
| eSOL | 3173 | 3173 | 3157 | 10 | 9 | 6 | 0 |
| Shehata PSR | 398 | 398 | 398 H / 398 L / 398 pairs | 0 | 0 | 0 | 0 |
| Jain 2017 | 137 | 137 | 137 pairs | 0 | 0 | 0 | 0 (primary assays) |
| NbThermo | 548 | 548 | 501 | 18 | method-dependent | 29 | 9 |
| DeepViscosity partial | 153 | 153 | 0 joined | — | 0 | 153 | 0 |
| Meltome Tm index | 358672 | 65239 prot IDs | pending | — | multi-context | pending | 53435 null Tm rows |

---

## Licensing / redistribution (summary)

| Source | Personal download | Redistribute to participants | Flag |
| ------ | ----------------- | ---------------------------- | ---- |
| GDPa1 | after HF gate | Restricted / review CC BY commercial clause + gate | LICENSE_REVIEW_REQUIRED |
| GDPa3 | unclear | unclear | LICENSE_REVIEW_REQUIRED |
| eSOL | yes | yes with BY-SA | OK with attribution |
| UniProt | yes | yes (CC BY 4.0) | OK |
| DeepViscosity data | no | no | proprietary |
| Shehata derivatives | yes | review upstream Cell terms | LICENSE_REVIEW_REQUIRED |
| Jain SD CSV | yes (OA paper) | review PNAS supplement terms | LICENSE_REVIEW_REQUIRED |
| NbThermo / Meltome | yes | review | LICENSE_REVIEW_REQUIRED |

---

## Final recommendation (acquisition + target cleanliness only)

### READY_FOR_MODELING_GATE
- **Jain 2017** — paired sequences + multiple clean experimental developability assays.

### READY_AFTER_MINOR_CURATION
- **eSOL** — finalize unresolved 6; decide duplicate-sequence policy; freeze UniProt release.  
- **Shehata PSR** — optional retrieval of mmc2 for Tm/HIC; stratify by B-cell subset.  
- **NbThermo** — method-aware splits; resolve 548 vs 564.  
- **Meltome** — UniProt sequence join + strict condition columns (species / lysate vs cells).

### PROMISING_BUT_ACCESS_BLOCKED
- **GDPa1** (auth only) — top Ab-specific multi-assay follow-up.  
- **GDPa3** — small OAS diversity panel; obtain via Datapoints after access.

### PROMISING_BUT_RECONSTRUCTION_NEEDED
- Shehata **Tm/HIC** (need original supplement).

### INSUFFICIENT_PUBLIC_DATA
- **DeepViscosity 229** — do not advance until AstraZeneca/authors release sequence+label tables.

---

## Deliverables map

```text
/workspace_developability_acquisition/
├── manifests/source_manifest.tsv
├── manifests/reconstruction_manifest.tsv
├── interim/
│   ├── ginkgo_antibody_index.csv          # schema placeholder
│   ├── ginkgo_access_status.json
│   ├── esol_reconstructed.csv
│   ├── esol_join_audit.csv
│   ├── deepviscosity_labels_partial.csv
│   ├── deepviscosity_sequences_partial.fasta
│   ├── deepviscosity_join_audit.csv
│   ├── deepviscosity_status.json
│   ├── shehata_labels.csv
│   ├── shehata_sequences.fasta
│   ├── shehata_join_audit.csv
│   ├── jain2017_joined.csv
│   ├── jain2017_join_audit.csv
│   ├── jain2017_assay_coverage.csv
│   ├── nbthermo_flattened.csv
│   ├── nbthermo_summary.json
│   ├── meltome_tm_index.csv
│   └── meltome_dataset_metadata.csv
├── scripts/
│   ├── download_ginkgo.py
│   ├── reconstruct_esol.py
│   ├── audit_deepviscosity.py
│   ├── reconstruct_shehata.py
│   ├── reconstruct_jain.py
│   ├── reconstruct_nbthermo.py
│   ├── reconstruct_meltome.py
│   ├── fetch_uniprot_sequences.py
│   └── fetch_pdb_sequences.py
└── reports/
    ├── acquisition_scorecard.md
    └── ACQUISITION_AUDIT_FINAL.md
```

**Stop condition met:** acquisition/reconstruction feasibility established. No train/test splits, embeddings, baselines, or `/workspace` modifications.
