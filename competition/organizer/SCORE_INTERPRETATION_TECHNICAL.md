# Scientific Interpretation of Prediction Scores

日本語版: [SCORE_INTERPRETATION_TECHNICAL_JA.md](SCORE_INTERPRETATION_TECHNICAL_JA.md)

Competition: **Antibody Developability — TmApp & HIC**  
Audience: senior scientific / managerial / audit review  
Companion stats: `SCORE_INTERPRETATION_STATS.json`  
Status: documentation only (does not change scoring or release data)

---

## Organizational position (one paragraph)

For this competition, **TmApp MAE around 2.8–3.0 °C** and **HIC MAE around 0.45–0.48 min** (local cross-validation) should be interpreted as **strong benchmark-level prediction**. These levels show meaningful sequence-to-property signal relative to constant baselines (~21.6% and ~10.0% MAE reduction, respectively, versus Train-CV median baselines). Scores below approximately **2.5 °C** and **0.40 min** would represent clear improvement beyond the current organizer benchmark. These landmarks are **competition-relative** rather than universal developability thresholds. Published repeatability data from related DSF/HIC assays indicate that analytical measurements can reproduce substantially more tightly than current ML errors, so present models should be viewed as **early screening / prioritization tools**, not replacements for experimental characterization.

---

## 1. Purpose

This document answers a recurring review question:

> What prediction scores should we aim for in *this* competition, and what scientific meaning can safely be attached to those scores?

It separates two rulers that must not be collapsed into one “useful / clinical / manufacturing” scale:

1. **Competition-relative performance** — improvement over baselines and organizer benchmarks on this Shehata-derived dataset.  
2. **Scientific / assay-relative interpretation** — how prediction error compares qualitatively with technical variation reported for related assays.

Machine-readable numbers used throughout are frozen in `SCORE_INTERPRETATION_STATS.json`.

---

## 2. Executive conclusion

1. **Strong competition-relative results** are approximately **TmApp ~2.8–3.0 °C** and **HIC ~0.47 min** on **local CV**.  
2. These errors remain **several-fold larger** than technical variation reported in relevant related DSF/HIC contexts.  
3. Therefore they demonstrate **meaningful sequence→property prediction**, but **do not** justify describing models as DSF/HIC substitutes.  
4. Scores below **~2.5 °C** (TmApp) or **~0.40 min** (HIC) would be clear advances beyond the current organizer benchmark.  
5. Scores approaching **~1 °C** (TmApp) or **~0.1 min** (HIC) would be scientifically notable because they approach repeatability *scales* reported in related experiments; assay equivalence would still require **direct experimental validation**.  
6. Participant-facing guidance should emphasize **local CV** landmarks. Public N=81 is noisy; do not equate Public MAE bands with local-CV bands.

---

## 3. Competition targets and metrics

| Track | Target | What was measured | Unit | Primary metric |
|---|---|---|---|---|
| 1 | `TmApp` | Apparent melting temperature of **Fab** by **DSF** | °C | MAE (↓ better) |
| 2 | `HIC` | **IgG** hydrophobic interaction chromatography retention time | min | MAE (↓ better) |

Population: **N=324** (both labels). Dev **162**, Test **162**, Public/Private **81/81**, split `GEN_0001_B_20271100`.  
Two independent leaderboards; **no** combined score. Tie policy: exact equal Private MAE → shared rank.

Source study: Shehata et al., *Cell Reports* (2019), DOI `10.1016/j.celrep.2019.08.056` (CC BY 4.0 VoR-derived packaging).

---

## 4. Empirical competition benchmark

### 4.1 Local CV (primary guidance ruler)

From frozen organizer Train-CV / aggregated OOF artifacts (B5 calibration + GEN_0001 rescored CV):

| Target | Constant / median baseline MAE | Strong organizer level | Approximate MAE skill |
|---|---:|---:|---:|
| TmApp | ≈ **3.543 °C** | ≈ **2.78–2.95 °C** (best nested ≈ **2.778 °C**) | ≈ **21.6%** error reduction |
| HIC | ≈ **0.518 min** | ≈ **0.45–0.48 min** (CV-best ≈ **0.467 min**) | ≈ **10.0%** error reduction |

**MAE skill** = `1 − MAE_model / MAE_baseline`.  
Skill 0 = no improvement over constant baseline; 0.10 = 10% lower MAE; 0.20 = 20% lower MAE.

Skill measures **competition-relative predictive improvement**. It does **not** measure clinical success, developability probability, assay replacement, or manufacturing success.

### 4.2 Final-split examples (diagnostic transfer only)

On production split GEN_0001 (organizer models rescored; not participant targets):

| Target | Setting | Public MAE | Private MAE |
|---|---|---:|---:|
| TmApp | Median baseline | ≈ 3.784 | ≈ 3.772 |
| TmApp | Strong frozen models | ≈ 3.5–3.6 | ≈ 3.2–3.3 |
| HIC | Median baseline | ≈ 0.535 | ≈ 0.510 |
| HIC | CV-best frozen model | ≈ 0.488 | ≈ 0.457 |

**Do not mix** local-CV landmarks with Public leaderboard thresholds. TmApp local CV ~2.8–3.0 can transfer to Public mid-3s; compare like with like.

---

## 5. Actual target-distribution statistics (N=324)

Computed directly from frozen competition labels (`final_population.csv`), not from baseline-MAE heuristics.

| Statistic | TmApp (°C) | HIC (min) |
|---|---:|---:|
| Mean | 69.89 | 9.38 |
| SD | **4.69** | **0.83** |
| Median | 70.0 | 9.10 |
| IQR | **6.0** | **0.70** |
| MAD (around median) | **3.0** | **0.27** |
| P10–P90 | 64.5–75.5 | 8.75–10.40 |
| Min–Max | 52.5–83.5 | 8.47–13.86 |

HIC is **right-skewed** with a **sparse high tail**; SD alone understates the screening importance of rare HIGH-band antibodies.

Descriptive normalization (not universal interpretation):

| Landmark | TmApp MAE/SD | TmApp MAE/IQR | HIC MAE/SD | HIC MAE/IQR |
|---|---:|---:|---:|---:|
| Local-CV baseline | ≈ 0.76 | ≈ 0.59 | ≈ 0.62 | ≈ 0.74 |
| Local-CV best | ≈ 0.59 | ≈ 0.46 | ≈ 0.56 | ≈ 0.66 |

---

## 6. Competition-relative performance scale

Approximate **local-CV** landmarks for *this* dataset and organizer benchmark (use “~ / around / roughly”):

### TmApp MAE (°C)

| Band | Approximate MAE |
|---|---|
| Baseline-like | ~3.5 or worse |
| Clear predictive signal | below ~3.3 |
| Strong organizer-benchmark level | **~2.8–3.0** |
| Exceptional vs current benchmark | below ~2.5 |
| Very high predictive precision | below ~2.0 |

### HIC MAE (min)

| Band | Approximate MAE |
|---|---|
| Baseline-like | ~0.52 |
| Clear predictive signal | below ~0.50 |
| Strong organizer-benchmark level | **~0.45–0.48** |
| Exceptional vs current benchmark | below ~0.40 |
| Very high predictive precision | below ~0.30 |

These are **empirical competition landmarks**, not natural scientific cutoffs and not industrial acceptance criteria.

---

## 7. Scientific evidence hierarchy

| Grade | Meaning |
|---|---|
| **A** | Directly Shehata or extremely close protocol with clear mapping |
| **B** | Closely related antibody assay (same class of measurement) |
| **C** | General method evidence |
| **Inference** | Organizer synthesis / competition-relative interpretation |

**Critical gap:** Shehata-specific technical repeatability was **not** established from replicate statistics in the available packaging evidence. Related-assay numbers must not be rewritten as “Shehata noise floor.”

---

## 8. TmApp assay interpretation

TmApp is an **apparent** thermal transition / melting temperature measured on **Fab** fragments by **differential scanning fluorimetry (DSF)** under study conditions.

- Higher TmApp generally indicates greater thermal / conformational stability **in that assay**.  
- TmApp is **assay-dependent**, not a universal thermodynamic constant.  
- It is **not** a direct aggregation measurement.  
- It does **not** by itself determine developability or manufacturing success.

---

## 9. TmApp reproducibility evidence

Preferred wording for documentation:

> Related antibody / therapeutic-protein DSF studies report technical variation on approximately the **sub-degree to ~1 °C** scale depending on protocol and replicate design, **although Shehata-assay-specific repeatability was not directly established from the available evidence.**

Examples (Grade B/C — not Shehata floors):

- Therapeutic-protein **nanoDSF** work reports Fab-domain repeatability on the order of **~0.2 °C** in some comparability settings (related methods literature).  
- Broader DSF protocol literature notes that multi-day / instrument / plate-location effects can approach **~0.5–1 °C** in some designs.

**Do not write:** “Shehata assay noise floor = 0.2–0.5 °C.”

---

## 10. HIC assay interpretation

HIC retention time reflects **effective hydrophobic interaction** with a chromatography stationary phase under a specific protocol.

Higher retention can be associated with stronger effective hydrophobicity and with risks linked to self-association / nonspecific interactions in developability discussions. However:

- HIC is **not** an aggregation assay.  
- Higher HIC does **not** imply that an antibody necessarily aggregates.  
- Absolute retention times are **protocol-dependent**.

---

## 11. HIC reproducibility evidence

Closest defensible comparison identified for documentation:

- Jain et al., *Bioinformatics* (2017), DOI `10.1093/bioinformatics/btx519`  
- Reference IgG1 (adalimumab variable-region control) run periodically in an Adimab HIC setup: **8.6 ± 0.12 min** over **127** measurements.

Treat **~0.12 min** as a **useful closely matched protocol variation scale** (Grade A/B), **not** a universal HIC noise floor and **not** a directly proven Shehata replicate statistic.

Highly optimized fixed-system HIC inject-to-inject precision can be substantially smaller in other laboratory contexts; repeatability depends strongly on system, column, and protocol. That contrast reinforces caution against calling any single number “the” noise floor.

---

## 12. Why assay repeatability ≠ ML error floor

Analytical repeatability describes how tightly the **same** experimental procedure reproduces a measurement under controlled conditions.

ML MAE describes average absolute error of a **sequence→property predictor** across **different antibodies**.

Even if ML MAE approached a related-assay repeatability scale:

- it would not automatically equal Shehata assay equivalence;  
- it would not guarantee correct pairwise ordering at that same delta;  
- it would not authorize replacing experimental characterization without prospective validation.

---

## 13. MAE skill relative to baseline

Using frozen local-CV values:

| Target | Baseline | Best | Skill | Error reduction |
|---|---:|---:|---:|---:|
| TmApp | 3.543 °C | 2.778 °C | ≈ 0.216 | ≈ **21.6%** |
| HIC | 0.518 min | 0.467 min | ≈ 0.100 | ≈ **10.0%** |

Interpretation: strong organizer models extract real signal, with larger relative headroom on TmApp than HIC under current feature/model families.

---

## 14. MAE vs ranking / discrimination

- **MAE**: absolute assay-value fidelity (competition primary metric).  
- **Spearman**: rank-order consistency (diagnostic).  

The competition metric remains **MAE**. Do not change it.

For internal scientific discussion of screening utility, useful diagnostics include:

- high-tail recall / top-*k* enrichment (especially HIC HIGH band),  
- pairwise ranking accuracy at chosen deltas,  
- calibration / residual structure.

### MAE is not “resolution”

`TmApp MAE = 2 °C` does **not** imply every pair differing by 2 °C can be reliably separated.  
`HIC MAE = 0.3 min` does **not** imply every pair differing by 0.3 min can be reliably ordered.

Pairwise discrimination depends on error distribution, bias, calibration, prediction-error correlation, and true separation.

---

## 15. Competition-relative score bands

See §6. Participant guidance should cite **local CV**. Public leaderboard comparisons should use Public baselines / Public peers (N=81; ordering can fluctuate).

---

## 16. Scientific interpretation of score landmarks

### TmApp

| Approximate MAE | Competition reading | Scientific caution |
|---|---|---|
| ~3.5–3.8 °C | Baseline-like | Little improvement over constant prediction |
| ~3.0 °C | Strong competition-level | Meaningful sequence→TmApp signal; still several× larger than technical variation in some related DSF studies; can capture **broad** stability variation — **not** “all 3 °C differences resolved,” **not** assay replacement |
| <2.5 °C | Exceptional vs organizer benchmark | Error substantially smaller than major inter-antibody spread — **not** automatically manufacturing-/clinically-useful |
| <2.0 °C | Very high predictive precision here | Scientifically interesting narrowing toward related DSF repeatability *scales* — still insufficient for assay substitution |
| ~1.0 °C | Approaches order of some related DSF repeatability | **Not** validated DSF replacement; **not** Shehata noise floor; **not** experimentally interchangeable with DSF |

### HIC

| Approximate MAE | Competition reading | Scientific caution |
|---|---|---|
| ~0.52 min | Baseline-like | Constant-level |
| ~0.47 min | Strong competition-level | Genuine sequence→HIC signal (~10% CV error reduction); still several-fold larger than ~0.12 min close-protocol reference variation — **not** uninformative, **not** assay-level accuracy |
| <0.40 min | Exceptional vs organizer benchmark | Clear advance beyond current demonstrated organizer models |
| <0.30 min | Very high predictive precision here | Approaches same broad order as close-protocol analytical variation — still **not** pairwise resolution of 0.3 min |
| ~0.1 min | Order of magnitude of closely matched reference-control variation | **Not** proven assay replacement; **not** universal noise floor; **not** sufficient for final developability decisions |

---

## 17. Claims that are supported

- Local-CV MAE ~2.8–3.0 °C (TmApp) and ~0.45–0.48 min (HIC) are **strong relative to this competition’s baselines and organizer benchmark**.  
- These levels show **reproducible predictive signal** from sequence.  
- Related DSF/HIC literature reports **substantially tighter** analytical variation than current ML errors.  
- Public N=81 is small enough that leaderboard ordering can fluctuate.  
- HIC label distribution is skewed with a sparse high tail relevant to screening utility.

---

## 18. Claims that are NOT supported

- These MAE bands are **industrial acceptance thresholds**.  
- Models at strong benchmark level **replace** DSF or HIC.  
- Models certify developability, manufacturing readiness, clinical utility, or “safe antibodies.”  
- Any cited related-assay SD is the **Shehata noise floor**.  
- MAE equals pairwise resolution at the same numeric delta.  
- Public MAE bands equal local-CV bands.

---

## 19. Evidence ledger

| Claim | Source | Assay / sample | Numerical observation | Grade | Relevance to Shehata | Interpretation |
|---|---|---|---|---|---|---|
| TmApp is Fab DSF apparent Tm | Shehata et al. 2019; packaging PROVENANCE | Fab DSF | TmApp (°C) in mmc2 | A | Direct | Assay definition |
| HIC is IgG retention time | Shehata et al. 2019; PROVENANCE | IgG HIC | minutes in mmc2 | A | Direct | Assay definition |
| Local-CV TmApp baseline ≈ 3.543 °C | B5 `calibration_analysis.md` CONST_MEDIAN | Train OOF | MAE 3.54321 | Inference | Competition population | Baseline landmark |
| Local-CV TmApp best ≈ 2.778 °C | B5 calibration / GEN_0001 3×3 | Train OOF | MAE 2.778 | Inference | Competition population | Strong benchmark |
| Local-CV HIC baseline ≈ 0.518 min | B5 calibration CONST_MEDIAN | Train OOF | MAE 0.518465 | Inference | Competition population | Baseline landmark |
| Local-CV HIC best ≈ 0.467 min | GEN_0001 3×3 CV-best | Train CV | MAE 0.4667 | Inference | Competition population | Strong benchmark |
| TmApp N=324 SD ≈ 4.69 °C | `final_population.csv` | Competition labels | SD 4.691 | A | Direct labels | Biological spread |
| HIC N=324 SD ≈ 0.83 min; skewed | `final_population.csv` | Competition labels | SD 0.832; skew > 2 | A | Direct labels | Biological spread / tail |
| Related nanoDSF Fab repeatability ~0.2 °C | Therapeutic-protein nanoDSF comparability literature | nanoDSF Fab/domain Tm | ~0.2 °C repeatability | B | Related method, not Shehata protocol | Qualitative assay-tightness scale |
| Multi-day / protocol DSF variation ~0.5–1 °C | General DSF methods literature | DSF protocols | up to ~1 °C class effects | C | Method class | Caution on overclaiming floors |
| Close-protocol HIC reference 8.6 ± 0.12 min (n=127) | Jain et al. 2017 Bioinformatics | Adimab HIC reference IgG1 | 8.6 ± 0.12 min | A/B | Closely related Adimab HIC family | Closest HIC variation scale |
| Strong ≠ assay replacement | Organizer synthesis | — | — | Inference | Policy | Required caveat |

---

## 20. Key references

1. Shehata L, et al. (2019). Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports* 28:3300–3308.e4. DOI: [10.1016/j.celrep.2019.08.056](https://doi.org/10.1016/j.celrep.2019.08.056).  
2. Jain T, et al. (2017). Prediction of delayed retention of antibodies in hydrophobic interaction chromatography from sequence using machine learning. *Bioinformatics* 33:3758–3766. DOI: [10.1093/bioinformatics/btx519](https://doi.org/10.1093/bioinformatics/btx519). (Reference-control HIC 8.6 ± 0.12 min, n=127.)  
3. Jain T, et al. (2017). Biophysical properties of the clinical-stage antibody landscape. *PNAS* 114:944–949. DOI: [10.1073/pnas.1616408114](https://doi.org/10.1073/pnas.1616408114). (Developability assay landscape context.)  
4. Related therapeutic-protein **nanoDSF** comparability / repeatability literature reporting Fab-domain repeatability on the order of ~0.2 °C (Grade B; protocol differs from Shehata DSF).  
5. General DSF protocol literature documenting intraplate / multi-day / instrument contributions that can reach sub-degree to ~1 °C scales (Grade C).  
6. Organizer frozen artifacts: `gate_b5_ceiling/reports/calibration_analysis.md`, `gate_b5_ceiling/reports/GATE_B5_CEILING_FINAL.md`, `gate_b7_3_principled_split/reports/07_FINAL_GEN0001_3X3_BENCHMARK.md`, `competition/organizer/SCORE_INTERPRETATION_STATS.json`.

---

## Appendix — reviewer Q&A (quality test)

**Q1. Why is TmApp 3.0 °C considered strong?**  
Because it is near the organizer local-CV benchmark (~2.8–3.0) and well below the ~3.54 °C median baseline (~15–20%+ MAE reduction), i.e., clear competition-relative signal — not because 3 °C is an industrial acceptance criterion.

**Q2. Why is HIC 0.47 min considered strong?**  
Same logic: near organizer local-CV best (~0.467) and ~10% below the ~0.518 baseline.

**Q3. Are these industrial acceptance thresholds?**  
**No.** Competition-relative landmarks only.

**Q4. Are models as accurate as the assays?**  
**No.** Related assay repeatability is substantially tighter than current ML MAE.

**Q5. Why can’t we call 0.1 min the HIC noise floor?**  
~0.1–0.12 min is a **closely matched reference-control variation scale** from related Adimab HIC work, not a proven Shehata replicate floor and not universal.

**Q6. Does MAE 0.3 mean two antibodies 0.3 min apart can be distinguished?**  
**No.** MAE is average absolute error, not pairwise resolution.

**Q7. What evidence supports experimental repeatability comparison?**  
Related DSF literature (sub-degree–~1 °C) and Jain 2017 HIC reference-control (±0.12 min); see evidence ledger. Shehata-specific replicates are unavailable in packaging evidence.

**Q8. What is directly measured versus organizer inference?**  
Direct: Shehata assay definitions and N=324 label distributions. Inference: score bands, skill, and “strong/exceptional” language relative to organizer CV.
