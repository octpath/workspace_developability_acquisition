# Antibody Developability Dataset Deep Dive — Gate A0.5

**Date:** 2026-08-28  
**Scope:** Antibody-specific sequence → developability acquisition only  
**Exclusions from ranking:** Jain 2017 (prior use); FLAb expression local-mutant landscapes; DeepViscosity AZ-229 proprietary core  

Artifacts under `/workspace_developability_acquisition` (`interim/`, `interim/a05/`, `raw/`).

---

## Master comparison table (serious candidates only)

| Candidate | Target | N complete antibodies | Paired VH/VL | Distinct molecules? | Assay consistency | Public access | Internal-use license | Reconstruction effort | Competition potential |
| --------- | ------ | --------------------: | ------------ | ------------------- | ----------------- | ------------- | -------------------- | --------------------- | --------------------- |
| **GDPa1** | Tm2 / AC-SINS / HIC / SEC / PSR / titer (10 assays) | ~242–246 (gated) | Yes (Excel) | High (clinical panel) | Single platform PROPHET-Ab | Form + HF gate | PERMISSION_REQUIRED for redistribution | Low after access | **Class A** (blocked) |
| **GDPa3** | HIC / SEC / nanoDSF / titer / polyreactivity | 80 | Expected yes | High (OAS-sampled) | Same platform family | Form (+ likely HF) | PERMISSION_REQUIRED | Low after access | **Class A as OOD** / Class C standalone |
| **GDPa5** | 10 VHH developability assays | 160 | VHH only | Diverse clinical+nonclinical | Single platform | Form | PERMISSION_REQUIRED | Low after access | **Class B/A** nanobody track |
| **Shehata 2019** | PSR continuous | **398** | **Yes** | **High** (400 unique pairs; 364 H-prefix40 families) | Single Adimab HT panel | **Yes** (mmc2 recovered) | LIKELY_OK (CC BY 4.0) | Done | **Class A** |
| **Shehata 2019** | TmApp (°C) | **346** | Yes | High | Same study | Yes (mmc2) | LIKELY_OK | Done | **Class A** |
| **Shehata 2019** | HIC RT | **348** | Yes | High | Same study | Yes (mmc2) | LIKELY_OK | Done | **Class A** |
| **NbThermo** | Tm (method-split) | 514 seq+Tm; **DSF-only 241 unique** | VHH | Moderate (400 prefix70 families; some families ≤19) | **Heterogeneous** (CD/DSF/nanoDSF/…) | Yes (`database.json`) | AMBIGUOUS→LIKELY_OK for research | Low | **Class B** if single-method |
| **DOTAD / Lecerf** | Fe/heme polyreactivity panel | **115** paired | Yes (INN→seq) | Therapeutic diversity | Single study | Yes (DOTAD xlsx) | AMBIGUOUS (aggregated DB) | Low | **Class C** |
| **DOTAD / Makowski** | SMP/SCP/OVA/HSA | **29** | Yes | Therapeutic | Single study | Yes | AMBIGUOUS | Low | **Class C** |
| **Kraft 2019 (FLAb)** | Heparin Rel RT | **128** | Yes | High (~125 families) | Single study | Yes (FLAb CSV) | Check paper license | Low | **Class C** (adjacent to colloidal/self-assoc) |
| **Garbinski 2023 Tm** | nanoDSF Tm1 | 86 | Mostly | Moderate | Single study | FLAb CSV | Check | Low | **Class C** |
| **DeepViscosity 229** | viscosity @150 mg/mL | 0 joinable | No public | — | Uniform (reported) | Proprietary | RESTRICTED | Impossible currently | **Class D** |
| **Lai ~27 / Apgar viscosity** | viscosity | ~16–38 public test-ish | Named mAbs → reconstructable? | Limited | Study-specific | Partial SI | AMBIGUOUS | Medium | **Class C/D** (too small) |
| **GDPa4** | bispecific developability | 160 bsAb + 65 IgG | Nonstandard | Specialized | Platform | Form | PERMISSION_REQUIRED | Low | **Class C** (task mismatch) |
| **DOTAD as whole** | mixed assays | ~3.5k points / many papers | Mostly via INN | Mixed | **Not comparable** | Yes | AMBIGUOUS | High to clean | **Class C/D** as pooled mix |

Jain 2017 omitted from ranking (explicit exclusion) though still present inside DOTAD sheets.

---

## Part highlights

### GDPa series (see also `reports/GDPA1_ACCESS_RECHECK.md`)

- Official path: **Datapoints form → email**, not anonymous CDN.  
- HF remains individually gated (401).  
- GDPa3 ideal **OOD** for GDPa1; GDPa5 best Ginkgo nanobody panel; GDPa4 bispecific-specialized.

### Shehata 2019 — upgraded this Gate

Original `mmc2.xlsx` recovered via Elsevier CDN using Crossref PII `S2211124719311040` (prior Gate used wrong PII).

| Target | Complete paired Abs | Continuous? | Missingness | Consistency |
| ------ | ------------------: | ----------- | ----------: | ----------- |
| PSR | 398 | Yes | 0.5% | Single study |
| TmApp | 346 | Yes | 13.5% | Single study |
| HIC | 348 | Yes | 13.0% | Single study |
| All three | 324 | Yes | — | Single study |

Provenance: `raw/shehata/a05/mmc2.xlsx` → `interim/shehata_full_joined.csv`, FASTA updated.  
License: Crossref lists **CC BY 4.0** for VOR — **LIKELY_OK** for internal competition with attribution (still confirm Cell/Elsevier redistribution norms for packaged splits).

**Best Shehata competition targets:** continuous **HIC** or **TmApp** (cleaner than severely imbalanced PSR binary); PSR continuous remains valid.

### DOTAD deep audit

- **v1 downloads OK:** `sequence.xlsx` (12,939 rows / **963** INN; **942** paired), `experimental_data.xlsx` (per-paper sheets).  
- Experimental measurements are **literature aggregates**, not one protocol.  
- Dominant experimental multi-assay sheet = **Jain** (excluded for new competition).  
- Hebditch sheet = **predicted** — do not use as experimental targets.  
- Non-Jain experimental subsets are **small** (Avery AC-SINS 13; Fekete HIC 15; Lai agg 21; Makowski 29; Lecerf 115).  
- **DOTAD2.0** homepage metrics claim huge scales (324k Abs / 54M points) — largely metadata/annotation graph, **not** a clean single-assay competition table.  
- Verdict: **discovery index / reconstruction helper**, not Class A by itself.

Artifacts: `interim/a05/dotad_assay_by_assay.csv`, joined non-Jain subsets under `interim/a05/dotad_*_joined.csv`.

### NbThermo deep audit

| Method | Rows with seq | Unique sequences |
| ------ | ------------: | ---------------: |
| Circular dichroism | 283 | 267 |
| DSF (SYPRO) | 245 | **241** |
| nanoDSF | 72 | 72 |
| DSC | 15 | 15 |

- JSON **548** vs paper **564**.  
- prefix70 families: **400** (350 singletons; largest family 19) — better than a pure local-mutant library, but not fully independent.  
- **Recommendation:** prefer **DSF-only ~241** or **CD-only ~267** single-method subsets over pooled 514.

### FLAb / FLAb2 as discovery index

Used category READMEs + selective small CSV pulls only.

- Many large N sets are **one-parent mutational landscapes** (expression Adams/Koenig; jetha HIC largely clustered).  
- **Reject** those for this project’s diversity goal.  
- Useful pointers: Shehata (now sourced from primary mmc2), Kraft heparin (~128 diverse), Garbinski Tm (~86), NbThermo, Ginkgo links.  
- Do **not** treat FLAb CSVs as license-cleared redistribution packages without checking each study.

### Viscosity alternatives

- AZ **229** still proprietary (**Class D**).  
- Public named panels (Lai ~27; Apgar ~38 as DeepViscosity external tests) are **too small** for a primary 2-week competition even if sequences reconstructed from INN/Thera-SAbDab.  
- Best viscosity path remains **GDPa-adjacent industrial release** (none public continuous viscosity table found) or wait for author data.

### Aggregation / self-association

| Source | Assay | N paired (approx.) | Notes |
| ------ | ----- | -----------------: | ----- |
| GDPa1 | AC-SINS, SEC | ~246 | Best if unlocked |
| Shehata | HIC (hydrophobicity proxy) | 348 | Available now |
| Kraft/FLAb | Heparin Rel RT | 128 | Diverse therapeutics |
| DOTAD Avery | AC-SINS | 13 | Too small |
| Jain | AC-SINS/CSI-BLI | 137 | Excluded from ranking |

---

## Ranking (antibody candidates only; Jain excluded)

Criteria: acquisition → distinct Abs → paired seq → target cleanliness → continuous → assay consistency → N → ≠ C3a → headroom → license practicality.

1. **Shehata HIC or TmApp** — best **currently obtainable** Class A  
2. **Shehata PSR continuous** — obtainable; watch imbalance / B-cell covariates  
3. **GDPa1 multi-assay** — best scientific ceiling if form/HF + redistribution permission resolved  
4. **GDPa3** — OOD companion to GDPa1 (not primary alone)  
5. **NbThermo DSF-only (~241)** — best non-IgG Tm track  
6. **GDPa5 (160 VHH)** — Ginkgo nanobody alternative if IgG gated  
7. **Kraft heparin / Lecerf** — Class C backups  

---

## Candidate classes

### Class A — Strong competition candidate
- Shehata **TmApp** (346) or **HIC** (348) or **PSR** (398)  
- GDPa1 (post-access)  

### Class B — Promising / reconstruction or access
- NbThermo single-method subsets  
- GDPa3 (OOD) / GDPa5  
- DOTAD INN joins for niche assays (small N)

### Class C — Interesting but small / specialized / heterogeneous
- GDPa4 bispecifics; Kraft; Garbinski; pooled DOTAD; Lai/Apgar viscosity  

### Class D — Unusable for now
- DeepViscosity 229 core  
- Predicted-only sheets (Hebditch)  
- One-parent mutational expression landscapes  

---

## Final recommendations

### Best currently obtainable antibody dataset
**Shehata 2019 paired VH/VL + continuous HIC or TmApp** (`interim/shehata_full_joined.csv`; N=348 / 346; 324 triple-complete).

### Best dataset if GDPa1 access can be resolved
**GDPa1** (VH/VL → Tm2 and/or AC-SINS/HIC/SEC), with **GDPa3** as external OOD on shared assays — after **written internal-redistribution permission**.

### Best non-Ginkgo alternative
**Shehata** (above). Secondary: **NbThermo DSF-only**.

### Best Tm / thermostability candidate
1. Shehata TmApp (346, IgG, obtainable)  
2. GDPa1 nanoDSF Tm2 (post-access)  
3. NbThermo DSF-only (~241 VHH)

### Best aggregation / self-association candidate
1. GDPa1 AC-SINS / SEC (post-access)  
2. Shehata HIC (348) as hydrophobicity/aggregation-risk proxy  
3. Kraft heparin Rel RT (~128) Class C

### Best viscosity candidate
**None competition-ready.** DeepViscosity 229 blocked; public named panels too small.

### Best candidate requiring reconstruction
**DOTAD non-Jain sheets** (already partially joined) and **named viscosity INN→Thera-SAbDab** — scientifically limited by N.

---

## Shortlist for future modeling Gate (2–4)

1. **Shehata HIC** (or TmApp) — run immediately  
2. **Shehata PSR continuous** — optional second target / ablation  
3. **GDPa1** — parallel track pending Datapoints form + license email  
4. **NbThermo DSF-only** — nanobody Tm diversity track (different molecule class from C3a)

---

## License snapshot (internal private competition)

| Dataset | Personal download | Company internal use | Internal redistribution | Derived splits | Flag |
| ------- | ----------------- | -------------------- | ----------------------- | -------------- | ---- |
| Shehata mmc2 | Yes | LIKELY_OK (CC BY 4.0) | LIKELY_OK with attribution | LIKELY_OK | CLEAR→LIKELY_OK |
| GDPa1/3/5 | Form/HF | LIKELY_OK under commercial CC BY carve-out | **PERMISSION_REQUIRED** | **PERMISSION_REQUIRED** | PERMISSION_REQUIRED |
| NbThermo | Yes | LIKELY_OK research | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS |
| DOTAD | Yes | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS | AMBIGUOUS |
| FLAb component studies | Yes | per-study | per-study | per-study | Check each |

---

## Stop condition

A0.5 complete: GDPa1 direct-access recheck, antibody deep search, reconstruction (Shehata mmc2 + DOTAD subsets + NbThermo method split), licensing review, shortlist.  
No modeling, embeddings, or competition splits performed.
