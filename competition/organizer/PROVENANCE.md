# Provenance — competition packaging (1.0-rc1)

## Primary citation (verified from Crossref VoR metadata)

Shehata, L., Maurer, D. P., Wec, A. Z., Lilov, A., Champney, E., Sun, T., Archambault, K., Burnina, I., Lynaugh, H., Zhi, X., Xu, Y., & Walker, L. M. (2019). Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports*, *28*(13), 3300–3308.e4.

- DOI: https://doi.org/10.1016/j.celrep.2019.08.056  
- PMID: 31553901  
- Metadata artifact: `raw/shehata/a05/crossref.json`

## Attribution / license

This competition dataset is derived from data reported in the citation above.

The Version of Record is distributed under the Creative Commons Attribution 4.0 International (**CC BY 4.0**) license:

- Crossref VoR license entry: `content-version = vor`  
- URL: https://creativecommons.org/licenses/by/4.0/  
  (Crossref records `http://creativecommons.org/licenses/by/4.0/`; prefer the canonical HTTPS form in attribution text.)

The same Crossref record also lists Elsevier TDM licenses; packaging attribution for redistributed VoR-derived content follows **CC BY 4.0**.

The source data have been reformatted and split for this machine-learning competition.

**The original authors are not affiliated with or responsible for this competition.** Use of the data does not imply endorsement.

## Assay columns used

From Shehata supplementary antibody table (**mmc2**), as recovered into the acquisition workspace:

| Competition column | Source column (supplement) | Unit |
|---|---|---|
| `TmApp` | `TmApp (°C)` | °C |
| `HIC` | `HIC retention time (min)` | min |
| `heavy` / `light` | VH / VL sequences | AA string |
| `id` | Author / Adimab-style clone IDs (e.g. `ADI-…`) | string |

Local source copies include:

- `raw/shehata/a05/mmc2.xlsx` (and csv/xls derivatives)  
- SI figures: `raw/shehata/a05/mmc1.pdf`  

## Assay method notes (finalized)

### TmApp

Confirmed / retained for participant-facing documentation:

- TmApp = **apparent melting temperature** (°C).  
- Measurement concerns antibody **Fab** fragments.  
- Assay method: **differential scanning fluorimetry (DSF)**.  
- Purified Fab samples heated while fluorescence monitored; apparent transition assigned from the thermal fluorescence curve (or its derivative) per study method.  
  (Local audits establish DSF / Fab TmApp; they do **not** independently lock “first derivative only,” so participant docs avoid that stronger claim.)  
- Used as an indicator related to conformational / thermal stability and resistance to unfolding.  

Supporting artifacts:

- SI figure labels (**Fab TmApp**) in `mmc1.pdf`  
- Gate B4 assay audit citing STAR Methods DSF wording  
- Human/external review confirmation for launch finalization  

Do not over-attribute generic dye-mechanism details (e.g. SYPRO Orange) as explicit Shehata claims; participant text uses the simpler DSF fluorescence wording.

### HIC

- Continuous **IgG** HIC retention time in minutes from mmc2.  
- Hydrophobicity-related developability readout; associated with self-association / aggregation-related risk, but **not** a direct aggregation assay.  

Interpretive diagnostic bands (not competition classes): LOW < 10.5, MEDIUM 10.5–11.5, HIGH > 11.5 (minutes).

## Extraction / filtering path

1. Recover Shehata sequences + assay labels from mmc2 into Gate B1 join tables / `triple_core`.  
2. Competition population requires **both** HIC and TmApp non-null.  
3. Gate B3 freeze: `gate_b3/frozen/organizer/final_population.csv` — **N=324**.  
4. Train/Test membership (162/162) frozen in Gate B3 `role_map.csv` (`Train` vs non-Train).  
5. Exact heavy/light pairs do not overlap Train vs Test; sequence groups do not overlap.  
6. Sequences stored uppercase, AA20 alphabet, stripped of whitespace (already frozen).  

## Public / Private split provenance

- Production split: **`GEN_0001_B_20271100`**  
- Seed metadata: `20271100` (not a reconstruction source)  
- Construction: Gate B7.3 **model-blind statistical balancing** (simulated annealing method tag `B_simulated_annealing`), then model-bank safety checks on frozen finalists.  
- Source artifact: `gate_b7_3_principled_split/config/B7_3_RECOMMENDED_SPLIT.json`  
- Packaged authoritative lists: `competition/organizer/SPLIT_MANIFEST.json`  

Hashes:

- Public: `2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376`  
- Private: `f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0`  

HIC diagnostic band balance under this split: MEDIUM 3/3, HIGH 4/3.

## Final file generation

Built by `competition/organizer/scripts/build_competition_data.py` from:

- frozen population + role map  
- `SPLIT_MANIFEST.json`  

Outputs:

- `data/distribution/dev.csv`  
- `data/distribution/test_features.csv`  
- `data/distribution/sample_submission.csv` (Train medians)  
- `data/secret/solution.csv`  

No rescaling of assay values. No regeneration of Public/Private from seed.

## What participants receive

Only:

- `dev.csv`  
- `test_features.csv`  
- `sample_submission.csv`  
- `DATA_DICTIONARY.md`  
- `README.md`  

They do **not** receive Test labels, Public/Private flags, or organizer benchmark results.

## Original study vs ML task

The Shehata study investigated affinity maturation, specificity, and biophysical properties. This package repurposes measured TmApp and HIC into supervised sequence→property regression tasks. ML claims should not be attributed to the original authors.
