# Antibody Developability Competition

日本語版: [README_ja.md](README_ja.md)

**Competition title:** Antibody Developability — TmApp & HIC  
**Package version:** 1.0  

**Tracks:** TmApp · HIC retention time  
**Primary metric (each track):** Mean Absolute Error (MAE) — lower is better  
**Input:** paired antibody variable-region sequences (`heavy`, `light`)

This README is written for participants with a computer-science / machine-learning background who may have little or no antibody-development experience.

---

## 1. What is antibody developability?

Discovering an antibody that binds its biological target is only part of making a drug.

A therapeutic antibody candidate must also be practical to **express, purify, formulate, store, transport, and manufacture reproducibly**. It should ideally avoid problematic physicochemical properties such as:

- poor structural stability  
- excessive self-association  
- aggregation tendency  
- problematic surface hydrophobicity  
- poor solubility  
- very high viscosity  
- nonspecific interactions  

**Developability** is a broad concept describing whether a biologically promising molecule also has physicochemical properties compatible with successful pharmaceutical development and manufacturing.

There is **no single universal “developability score.”** Developability is assessed through multiple experimental readouts. This competition focuses on **two** of those readouts.

---

## 2. “Does it work?” vs “Can we develop it?”

A useful intuition for non-biologists:

| Question type | Everyday meaning | Examples |
|---|---|---|
| **Biological** | “Does the antibody do what we want?” | Binding, specificity, biological activity |
| **Developability** | “Can we turn this molecule into a practical drug product?” | Stability, solubility, hydrophobicity, self-association, aggregation risk, viscosity, manufacturability |

These questions are related but distinct. **This competition concerns a small subset of the second category.**

---

## 3. Why early risk prediction matters

Early discovery can produce many candidate antibodies. Experimental burden and cost rise substantially as candidates move toward:

- detailed characterization  
- process development  
- purification development  
- formulation development  
- scale-up  
- stability studies  
- external manufacturing (for example via a CMO/CDMO)  

If undesirable physicochemical properties are discovered only **after** major process-development or manufacturing resources have already been committed, the consequences can include additional experiments, reformulation, purification-process changes, candidate redesign, delays, and increased cost.

An important goal of early developability assessment is therefore:

> Identify potential physicochemical risks **before** expensive downstream development and manufacturing decisions are made.

**Careful wording:** TmApp and HIC do **not**, by themselves, decide whether an antibody “can” or “cannot” be manufactured. They are **early developability-related risk indicators** under defined assay conditions—not direct predictors of manufacturing success.

---

## 4. Antibody sequence primer (short)

Antibodies contain **heavy** and **light** chains. In this dataset:

- `heavy` = variable-region amino-acid sequence of the heavy chain (VH)  
- `light` = variable-region amino-acid sequence of the light chain (VL)  

Variable regions are central to antigen recognition. They contain relatively conserved **framework** regions and more variable **complementarity-determining regions (CDRs)**. CDRs contribute strongly to antigen-binding geometry.

A useful mental model for why sequence→property prediction can work:

> sequence → local chemistry → structure / surface properties → measurable physicochemical behavior

You do **not** need to be an immunologist to participate. Many strong approaches start from sequence descriptors or protein language models alone.

---

## 5. Sequence-derived antibody annotations

Antibody sequences can also be described using antibody-specific biological annotations. For convenience, we provide a small optional set of **sequence-derived antibody annotations based on the provided VH/VL sequences**.

| File | Role | N |
|---|---|---:|
| `data/dev_annotations.csv` | Annotations for Dev IDs | 162 |
| `data/test_annotations.csv` | Annotations for Test IDs | 162 |

Join to `dev.csv` / `test_features.csv` by `id`. These files do **not** duplicate `heavy`, `light`, `TmApp`, or `HIC`.

### What is included (where available)

- Inferred **V family** and **J gene** for heavy and light chains  
- **Light-chain type** (`kappa` / `lambda`)  
- **CDR lengths** (HCDR1–3, LCDR1–3)  
- **Germline identity** for heavy and light V regions (0–1 scale)

See `DATA_DICTIONARY.md` for exact definitions, tools, and numbering conventions.

### Germline (short intuition)

Antibody variable regions are assembled from inherited **germline** gene segments (V, D, and J; often summarized as **V(D)J recombination**) and are then further diversified. By comparing an observed antibody sequence with reference germline sequences, one can infer a likely **V family / J gene**. For the V region, **germline identity** measures how similar the observed sequence is to its ANARCI-assigned germline V reference (scale 0–1; 1.0 = identical under that calculation).

Lower germline identity means greater sequence divergence from the inferred germline V reference. That divergence can include changes accumulated through **somatic hypermutation (SHM)** — sequence diversification that can accumulate during antibody maturation.

**Important:** lower identity is **not** inherently good or bad; more SHM is **not** inherently better or worse; germline identity is **not** affinity and **not** a developability score. Treat these fields as descriptors and investigate predictive relationships yourself.

### CDR lengths

**CDRs** (complementarity-determining regions) are highly variable loops involved in antigen recognition. Their lengths (and sequence composition) can affect local geometry, structure, and exposed surface chemistry, so CDR lengths are useful sequence-derived descriptors. We do **not** claim that any particular CDR determines TmApp or HIC.

The distributed CDR lengths use the IMGT-segmented CDR regions provided with the source-study sequences (Shehata mmc2). Other IMGT/ANARCI implementations may place some boundaries slightly differently. See `DATA_DICTIONARY.md` for the exact convention.

### Light-chain type

Human antibodies commonly use one of two light-chain types: **kappa** or **lambda**. The annotation records which type was inferred from the light-chain sequence. Neither type is intrinsically “better.”

### Optional — not privileged label information

> These annotation files are **optional convenience resources**. They are sequence-derived antibody annotations based on the provided VH/VL sequences and do **not** contain additional experimental TmApp or HIC information.

Participants are free to ignore them and work directly from `heavy` / `light` sequences. The official scorer does **not** use the annotation files.


## 6. Target 1 — TmApp (apparent melting temperature, °C)

### Intuition

Proteins normally adopt folded three-dimensional structures. As temperature increases, those structures eventually undergo thermal unfolding transitions. **Differential scanning fluorimetry (DSF)** monitors temperature-dependent fluorescence changes associated with that process, and **TmApp** summarizes the characteristic **apparent** transition temperature observed under the assay conditions.

- **Unit:** degrees Celsius (°C)  
- **Higher TmApp** generally indicates **greater thermal / conformational stability** under those measurement conditions.

### What was measured in the source study?

In the Shehata study:

- **TmApp** means **apparent melting temperature**.  
- The measurement concerns antibody **Fab** fragments (the antigen-binding arms).  
- Thermal stability was assayed by **differential scanning fluorimetry (DSF)**.  
- Purified Fab samples were heated while fluorescence was monitored.  
- The apparent melting / thermal transition temperature was assigned from the thermal fluorescence curve (or its derivative) according to the study method.  
- TmApp was used as an indicator related to conformational / thermal stability and resistance to unfolding.

**Note on mechanism wording:** some dye-based DSF assays elsewhere use dyes that respond when normally buried hydrophobic regions become exposed during unfolding. Treat that as **general dye-based DSF intuition**, not a mechanistic detail you must attribute specifically to this paper. For this competition it is enough that:

> DSF monitors temperature-dependent fluorescence changes associated with protein unfolding.

### Why “apparent”?

**TmApp is not:**

- a universal thermodynamic constant  
- an absolute destruction temperature  
- a pass/fail threshold  

The measured transition depends on the assay context, including construct, buffer / solution conditions, measurement protocol, and thermal scan conditions. Protein unfolding may not behave as an ideal reversible equilibrium.

Therefore:

> **TmApp = apparent melting / thermal transition temperature** under defined assay conditions.

A value such as **TmApp = 70 °C** means approximately that a characteristic thermal transition was observed around that temperature under the assay conditions—**not** that the antibody is completely stable at 69.9 °C and destroyed at 70.0 °C.

### Why TmApp matters for developability

Greater conformational stability can make a molecule more resistant to structural perturbation. Lower thermal stability can indicate a less robust folded state and may be associated with increased development risk.

**Mandatory caveats:**

- TmApp is **not** itself an aggregation assay.  
- TmApp is **not** by itself a pass/fail criterion.  
- High TmApp does **not** guarantee an otherwise ideal drug candidate.  

Treat TmApp as **one developability-related axis**.

---

## 7. Target 2 — HIC retention time (minutes)

### Intuition

**HIC** = **Hydrophobic Interaction Chromatography**.

The experiment measures how strongly a molecule interacts with a **hydrophobic stationary phase** under a specified chromatographic protocol. In this dataset the HIC values correspond to **IgG** hydrophobic interaction chromatography retention times reported by the source study.

The competition target is **retention time**:

- **Unit:** minutes  
- **Longer retention** generally indicates **stronger hydrophobic interactions** / greater effective exposed hydrophobic character under the assay conditions.

### So what? Why does hydrophobicity matter?

Conceptual chain:

1. exposed / effective hydrophobic character  
2. → stronger hydrophobic interactions (longer HIC retention under the protocol)  
3. → potentially greater unwanted protein–protein interaction / self-association tendency  
4. → possible aggregation-related, solubility, formulation, or other developability risk  

Longer HIC retention time indicates stronger effective hydrophobic interaction under the assay protocol. Excessive exposed hydrophobic character can promote unwanted protein–protein interactions and self-association and can therefore be **associated with** aggregation-related or formulation risks.

**Mandatory caveat:**

> HIC is **NOT** a direct aggregation assay.  
> A high HIC value does **NOT** imply that aggregation must occur.

Do **not** equate “high HIC” with “aggregating antibody.”

### Assay-specific nature

HIC retention time is **protocol-dependent**. It is not a universal intrinsic constant of an antibody. The absolute numerical scale depends on column chemistry, mobile phase, salt conditions, gradient, flow, and related protocol choices.

This competition predicts:

> the HIC retention time **under the assay context represented by this dataset**

not a universal “hydrophobicity number.”

### Interpretive bands (not competition classes)

To help interpret the scale, organizer diagnostics use these **interpretive / diagnostic** bands:

| Band | HIC retention time |
|---|---|
| LOW | < 10.5 min |
| MEDIUM | 10.5–11.5 min |
| HIGH | > 11.5 min |

These are **not** competition classes, not universal clinical thresholds, and not official pass/fail rules.

The competition target remains **continuous**. Scoring uses the **numerical** HIC value. Do not turn this into a classification task unless you are doing so only as your own optional analysis.

---

## 8. Side-by-side target summary

| Target | Measurement context | Unit | Higher value roughly indicates | Developability intuition |
|---|---|---|---|---|
| **TmApp** | Fab thermal transition measured by DSF | °C | Greater thermal/conformational stability under assay conditions | More robust folded state |
| **HIC** | IgG hydrophobic interaction chromatography retention | min | Stronger effective hydrophobic interaction | Greater exposed hydrophobic character; can be associated with self-association / aggregation-related risk |

Neither measurement alone determines whether an antibody is developable or manufacturable.  
**TmApp is not an aggregation assay.**  
**HIC is not an aggregation assay.**

They represent **two distinct physicochemical axes** relevant to early developability assessment.

---

## 9. Why sequence may predict these properties

Amino-acid sequence controls side-chain chemistry, charge, hydrophobicity, aromatic content, hydrogen-bonding possibilities, loop composition, and structural packing. Those factors influence folding stability, exposed surface chemistry, hydrophobic patches, and self-interaction tendencies.

Therefore VH/VL sequence contains information relevant to both TmApp and HIC.

Participants may use methods ranging from simple sequence descriptors to protein language models, antibody-specific language models, structural predictions, and ensembles. Choose what you can validate robustly.

---

## 10. Original study vs this competition

The Shehata study was **not** originally designed as a machine-learning sequence-to-property benchmark. It investigated biological and biophysical properties of a collection of human antibodies, including relationships involving antibody maturation and measured physicochemical characteristics.

This competition **repurposes** experimentally measured values into supervised prediction tasks:

- VH + VL sequence → **TmApp**  
- VH + VL sequence → **HIC retention time**

Do not attribute machine-learning claims (PLMs, sequence models, or this competition formulation) to the original authors.

---

## 11. Dataset

Exact counts from the distributed files:

| File | Role | N |
|---|---|---:|
| `data/dev.csv` | Labeled development / training set | 162 |
| `data/test_features.csv` | Unlabeled test sequences | 162 |
| `data/dev_annotations.csv` | Optional sequence-derived annotations (Dev) | 162 |
| `data/test_annotations.csv` | Optional sequence-derived annotations (Test) | 162 |
| `data/sample_submission.csv` | Example submission template | 162 |

Every antibody in this package has **both** TmApp and HIC labels available to organizers (and both labels are present in `dev.csv`).

### Columns

| Column | Where | Meaning |
|---|---|---|
| `id` | all | Antibody identifier |
| `heavy` | dev, test_features | VH amino-acid sequence |
| `light` | dev, test_features | VL amino-acid sequence |
| `TmApp` | dev, submission | Apparent melting temperature (°C) |
| `HIC` | dev, submission | HIC retention time (min) |

See `DATA_DICTIONARY.md` for formal definitions.

---

## 12. Two independent leaderboards

There are **two** competition tracks:

1. **TmApp**  
2. **HIC**

Each is scored **independently**.

**Primary metric:** Mean Absolute Error

\[
\mathrm{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat{y}_i\rvert
\]

Lower is better.

There is:

- **no** combined score  
- **no** weighted average  
- **no** overall grand metric  

Possible champions: **TmApp Champion** and **HIC Champion** (possibly different teams).

### Tie policy (scientific)

Final ranking for each track is by **Private MAE** (lower is better).

If two participants have **exactly the same Private MAE** for a track, they are **tied** (shared scientific rank).

Exact Private-MAE ties are **not** broken by Pearson, Spearman, RMSE, Public score, submission timestamp, or submission ID.

If a hosting platform later renders ties differently for display, that is a platform presentation detail—the scientific competition rule remains shared rank for equal Private MAE.

---

## 13. Public / Private leaderboard

- Test set size: **N = 162**  
- Public leaderboard subset: **N = 81**  
- Private / final ranking subset: **N = 81**  

Public and Private use the **same** partition for both TmApp and HIC.

The Public leaderboard uses one fixed subset of Test. Final ranking uses the held-out Private subset.

**Do not** attempt to reverse-engineer which Test rows are Public vs Private from participant files—those flags are organizer-only.

---

## 14. Suggested validation strategy

The live leaderboard is finite and therefore noisy. Tiny Public movements can be misleading.

Practical advice:

1. Build a reliable **local cross-validation** procedure on `dev.csv`.  
2. Compare CV evidence with Public feedback—do not trust either blindly.  
3. Avoid chasing tiny Public leaderboard changes.  

For **HIC**, the label distribution contains a relatively **sparse high-HIC tail**, so naïve random folds can show noticeable fold-to-fold variability. A practical starting strategy is **sequence-group-aware** splitting combined with **target-distribution-aware stratification**.

Local CV is not guaranteed to beat Public; use both as incomplete evidence.

---

## 15. Submission format

Submit a CSV with columns:

```text
id,TmApp,HIC
```

Requirements:

- Exact Test ID set (same as `test_features.csv` / `sample_submission.csv`)  
- Unique IDs  
- Both `TmApp` and `HIC` columns present and **finite** (no NaN / Inf)  
- Scorer joins by **`id`** (do not rely on row order)

**One-track experiments:** even if you focus on only one target, **both** prediction columns are required. For the track you are not actively modeling, a trivial training-set median prediction is acceptable.

`sample_submission.csv` uses Train-set target medians as placeholders for both columns.

---

## 16. Important scientific caveats

- TmApp is assay-dependent and is not a universal stability constant.  
- HIC retention time is chromatographic-protocol-dependent.  
- HIC is not a direct aggregation assay.  
- TmApp is not a direct aggregation assay.  
- Neither target alone defines developability.  
- Sequence-based predictions are screening / prioritization tools and do **not** replace experimental characterization.

---

## 17. Data provenance and attribution

This competition dataset is derived from data reported in:

> Shehata, L., Maurer, D. P., Wec, A. Z., Lilov, A., Champney, E., Sun, T., Archambault, K., Burnina, I., Lynaugh, H., Zhi, X., Xu, Y., & Walker, L. M. (2019). Affinity maturation enhances antibody specificity but compromises conformational stability. *Cell Reports*, *28*(13), 3300–3308.e4.  
> DOI: [10.1016/j.celrep.2019.08.056](https://doi.org/10.1016/j.celrep.2019.08.056)

The Version of Record is distributed under the Creative Commons Attribution 4.0 International (**CC BY 4.0**) license:  
https://creativecommons.org/licenses/by/4.0/

The source data have been reformatted and split for this machine-learning competition. Relative to the source materials, this package:

- uses author-provided antibody IDs and VH/VL sequences  
- keeps sequences as uppercase amino-acid strings  
- filters to antibodies with both TmApp and HIC present  
- reorganizes rows into labeled development (`dev.csv`) and unlabeled test (`test_features.csv`) tables  
- applies a fixed Public/Private partition of the Test set for leaderboard scoring  

The original authors are **not affiliated with or responsible for** this competition. Use of the data does not imply endorsement.

---

## Files in this distribution

```text
README.md
README_ja.md
data/dev.csv
data/test_features.csv
data/dev_annotations.csv
data/test_annotations.csv
data/sample_submission.csv
data/DATA_DICTIONARY.md
```

Good luck—and remember: **two tracks, MAE each, no combined score.**
