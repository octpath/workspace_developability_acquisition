# Assay scale audit — Shehata 2019 (Gate B4)
Source: Shehata et al., *Cell Reports* 28:3300–3308.e4 (2019), DOI 10.1016/j.celrep.2019.08.056;
supplementary `mmc2` assay columns; Adimab/study STAR-methods description (HIC + DSF).
Frozen population: N=324 antibodies with both HIC and TmApp non-null (Gate B3 triple-core).

## HIC

| Field | Finding |
|---|---|
| Measurement method | Hydrophobic interaction chromatography (HIC); high-throughput Adimab panel |
| Reported quantity | Retention time |
| Units | minutes (supplement column: `HIC retention time (min)`) |
| Protocol consistency | Single published study / single platform panel; all 324 values from same Shehata supplement table |
| Instrument / assay | Study STAR Methods: HIC as hydrophobicity / aggregation-propensity proxy (not an aggregation kinetic assay) |
| Rounding / discretization | Continuous-looking floats; N_unique=230 / 324; sample decimals to 0.001 |
| Replicates | No per-antibody replicate table in the public supplement used here |
| Batch information | Not provided as a participant/organizer column in mmc2 join |
| Train distribution | mean=9.395, SD=0.810, IQR=0.720, range=[8.523,12.740] |

**Absolute-error meaningful?** **YES** — all values share one study protocol and minute scale.
Caveat: HIC RT is **protocol-dependent**; model predicts assay RT under Shehata conditions, not a universal physical constant.

## TmApp

| Field | Finding |
|---|---|
| Measurement method | Differential scanning fluorimetry (DSF) / thermal melt; reported as TmApp |
| Reported quantity | Apparent thermal transition temperature |
| Units | °C (supplement column: `TmApp (°C)`) |
| Protocol consistency | Single study / single panel; all 324 from same table |
| Instrument / assay | DSF-based apparent Tm (TmApp) as conformational stability readout |
| Rounding / discretization | Half-degree grid common; N_unique=51 / 324; fraction exactly *.0 or *.5 = 1.00 |
| Replicates | No public replicate SD table for TmApp in the materials used here |
| Batch information | Not available as a column |
| Train distribution | mean=69.895, SD=4.502, IQR=5.750, range=[57.5,83.5] |

**Absolute-error meaningful?** **YES** — common °C assay scale within this study.
Caveat: heavy discrete ties (half-degree reporting) inflate rank-ties; MAE in °C remains interpretable.

## Track-level decision

| Track | Common assay scale? | Absolute-value scoring | Action |
|---|---|---|---|
| HIC | YES | GO | Rebuild under MAE (minutes) |
| TmApp | YES | GO | Rebuild under MAE (°C) |

No HOLD. Frozen B3 split unchanged.

## Experimental noise ceiling

**NO EMPIRICAL ASSAY NOISE CEILING AVAILABLE** — no replicate measurements with SD in the public materials reused here.
