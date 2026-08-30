# Acquisition Scorecard — Developability Gate A0

Access date: 2026-08-28  
Workspace: `/workspace_developability_acquisition` only (`/workspace` untreated / read-only).

| Candidate | Target | Experimental labels found | Sequences found | Complete joined molecules | Direct download? | Reconstructable? | Main reconstruction source | License status | Grade |
| --------- | ------ | ------------------------: | --------------: | ------------------------: | ---------------- | ---------------- | -------------------------- | -------------- | ----- |
| GDPa1 | nanoDSF Tm2 | gated (expected ~242–246) | gated (VH+VL in Excel) | 0 (blocked) | No (HF gated) | Yes after auth | Hugging Face `ginkgo-datapoints/GDPa1` | CC BY 4.0 + commercial restriction; gated | D |
| GDPa1 | AC-SINS | gated | gated | 0 | No | Yes after auth | same | same | D |
| GDPa1 | HIC | gated | gated | 0 | No | Yes after auth | same | same | D |
| GDPa1 | SEC aggregation | gated | gated | 0 | No | Yes after auth | same | same | D |
| GDPa1 | Polyreactivity | gated | gated | 0 | No | Yes after auth | same | same | D |
| GDPa1 | Titer | gated | gated | 0 | No | Yes after auth | same | same | D |
| GDPa3 | nanoDSF / HIC / SEC / titer / polyreactivity | claimed ~80 | claimed paired OAS | 0 (not retrieved) | No | Likely after access | datapoints.ginkgo.bio (not on HF org API) | LICENSE_REVIEW_REQUIRED | D |
| eSOL | cell-free solubility % | **3173** | **3167** mapped | **3167** | Yes (labels); sequences via UniProt | Yes | LSDB eSOL + UniProt UP000000625 | CC BY-SA 2.1 JP (+ UniProt CC BY 4.0) | B |
| DeepViscosity | viscosity @ 150 mg/mL (continuous) | **153** partial (notebook leak) | **0** for training 229 | **0–6** demo overlap only | No | No for full 229 | GitHub + paper (proprietary AZ) | Code MIT; data proprietary | E |
| DeepViscosity | viscosity binary ≤20 cP | not publicly tabled | 0 for 229 | 0 | No | No | same | same | E |
| Shehata2019 | PSR continuous | **398** | **398** paired VH+VL | **398** | Yes (derivative CSV) | Yes | Obstacle/HF derivatives of mmc2 | LICENSE_REVIEW_REQUIRED (Cell Reports upstream) | B |
| Shehata2019 | Tm / HIC | in original mmc2 (reported) | in mmc2 | **0 public** | No (raw xlsx absent / not OA) | Partial / blocked | Cell Reports supplement | LICENSE_REVIEW_REQUIRED | C/D |
| Jain2017 | AC-SINS | **137** | **137** VH+VL | **137** | Yes (CSV derivative of SD03/SD02) | Yes | PNAS SD02+SD03 via Obstacle CSV | PNAS OA; LICENSE_REVIEW_REQUIRED for redistribution | A/B |
| Jain2017 | CSI-BLI | 137 | 137 | 137 | Yes | Yes | same | same | A/B |
| Jain2017 | HIC | 137 | 137 | 137 | Yes | Yes | same | same | A/B |
| Jain2017 | PSR | 137 | 137 | 137 | Yes | Yes | same | same | A/B |
| Jain2017 | Fab Tm (DSF) | 137 | 137 | 137 | Yes | Yes | same | same | A/B |
| Jain2017 | Accelerated stability slope | 137 | 137 | 137 | Yes | Yes | same | same | A/B |
| Jain2017 | SMAC / CIC / ELISA / BVP / titer | 137 each | 137 | 137 | Yes | Yes | same | same | A/B |
| NbThermo | Tm (heterogeneous methods) | **539** / 548 | **519** | **514** | Yes | Yes | NbThermo `database.json` | LICENSE_REVIEW_REQUIRED | B/C |
| Meltome | Tm (multi-species/context) | **305237** Tm rows; **65239** protein IDs | via UniProt IDs (not batch-fetched) | reconstructable | Yes (Nature MOESM4) | Yes | Nature Methods supplements + UniProt | LICENSE_REVIEW_REQUIRED | B |

## Grade legend

- **A** DIRECT_DOWNLOAD_READY  
- **B** RECONSTRUCTABLE  
- **C** RECONSTRUCTABLE_WITH_GAPS  
- **D** ACCESS_BLOCKED  
- **E** NOT_CURRENTLY_RECONSTRUCTABLE  
