# TmApp Technical Report

**VH/VL sequence → TmApp prediction**  
Representation × H/L topology × annotation factorial

**Scope statement.** This report focuses exclusively on TmApp prediction. The prediction target is TmApp from VH/VL sequence. Topology and annotation conclusions here must **not** be assumed to transfer automatically to HIC, viscosity, aggregation, or antibody developability as a whole; those endpoints require separate evaluation.

**Terminology.** Downstream H/L designs are referred to as **SEP** (Separate H/L), **JOINT** (Full Joint), **REG-SEP** (Joint Residues, Separate REGs), **XREG** (Cross-REG Read), and **FUSE** (REG Fusion). Legacy labels A/B1/B2/C/D appear only in reproducibility notes. A **REG** is a learned chain-level token that gathers information from residues processed by the downstream Transformer; we use REG / REG_H / REG_L thereafter.

Figures and tables live under `developability_drilldown/technical_report/tmapp_factorial/`. Numerical values are taken from the frozen 200-cell internal matrix (`tmapp_factorial_human_master.csv` and supporting bootstrap / DiD tables). Lower MAE is better; negative ΔMAE indicates improvement relative to the stated baseline.

---

## 1. Introduction

### 1.1 Antibody structure and the VH/VL inputs

Antibodies contain paired heavy and light chains. In the Fab arm, the heavy- and light-chain variable domains (VH and VL) form a paired variable region. Not all antibody variation is confined to VH/VL, but **this prediction task uses VH/VL sequences as inputs**. Individual chain sequence properties matter, and the relationship between the two chains may also matter for a VH/VL-conditioned regressor.

### 1.2 Why model Heavy and Light explicitly?

A VH/VL → TmApp predictor must make design choices about:

1. how Heavy and Light are represented at the residue level;
2. whether and where information is exchanged between them downstream;
3. whether antibody-specific positional or region annotations are supplied explicitly.

These choices motivate a controlled comparison of Transformer H/L topologies under a shared training protocol.

### 1.3 Transformer H/L design space

We compare five frozen downstream topologies (Figure 1):

| ID | Name | Conceptual role |
| --- | --- | --- |
| **SEP** | Separate H/L | No explicit downstream H/L communication before final merge |
| **JOINT** | Full Joint | Unrestricted joint processing of H residues, L residues, REG_H, and REG_L |
| **REG-SEP** | Joint Residues, Separate REGs | Biological residues communicate jointly; REG_H / REG_L remain chain-specific |
| **XREG** | Cross-REG Read | Chains encoded separately; then REG_H reads Light residues and REG_L reads Heavy residues |
| **FUSE** | REG Fusion | Chains processed independently; then REG_H ↔ REG_L interact via the frozen D3 module |

### 1.4 Residue representation matters

Residue inputs may come from learned amino-acid embeddings trained from scratch, general-protein protein language models (PLMs), antibody-oriented PLMs, and—within matched checkpoints—paired versus separate-chain inference contexts. The most useful downstream H/L inductive bias may therefore depend on what is already present in the residue representation. We do **not** assume a priori that any PLM “contains structural information.”

### 1.5 Explicit antibody annotations

Sequence-position and chain-ID embeddings are present in all conditions. On top of that BASE, we ablate:

- **BASE** — no IMGT position, no CDR/FR region embedding;
- **IMGT Position** — explicit IMGT-position embedding;
- **CDR/FR Region** — explicit region embedding;
- **IMGT + Region (FULL)** — both.

This asks whether explicit antibody-specific annotation still helps when pretrained residue representations are used.

### 1.6 Main experimental question

We study the factorial

\[
\text{Representation} \times \text{H/L topology} \times \text{Annotation}
\]

for **TmApp** under a frozen downstream platform, testing how these design choices interact.

---

## 2. Methods (brief)

### 2.1 Task and data

**Prediction task.** Predict a continuous **TmApp** value from paired **VH/VL** amino-acid sequences (`heavy`, `light`).

**Source study.** Labels and sequences originate from Shehata et al. (2019), *Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability* (*Cell Reports*; see References). Repository assay metadata (`gate_b1/reports/assay_definitions.md`) define **TmApp** as the authors’ reported **apparent thermal transition / conformational stability** measure with units **°C** (supplement column **TmApp (°C)**; internal column `tm_app_C`). Higher TmApp corresponds to greater apparent thermal stability. That frozen assay note does **not** separately record an operational method string such as “Fab fragment + DSF”; this report therefore follows the repository wording rather than adding assay details absent from that metadata.

**Cohort size.** This drilldown / competition cohort contains **exactly 324 antibodies** with both TmApp and HIC labels available (162 Dev + 162 Test unique IDs; residue assets likewise use `n_ids = 324`). The Shehata supplement has a larger TmApp-complete count (**N = 346** in the same assay-definition note); the modeling tables used here are the frozen **324-ID** intersection, not the full TmApp-complete supplement.

**Evaluation.** Internal evaluation uses the existing TmApp Primary and Shadow OOF protocols on these VH/VL tables (same folds as the frozen drilldown platform). Metrics reported here are Primary MAE, Shadow MAE, mean(P,S), worst(P,S), and |P−S| (MAE in °C on the TmApp scale).

### 2.2 Representations (10)

| Display name | Role | Context |
| --- | --- | --- |
| Scratch | Learned AA embedding | LEARNED_AA |
| AbLingua | Antibody PLM | SEPARATE_CHAIN |
| AbLang1 | Antibody PLM | SEPARATE_CHAIN |
| AbLang2 (separate-chain) | Antibody PLM | SEPARATE_CHAIN |
| AbLang2 (paired H/L) | Antibody PLM | PAIRED_NATIVE |
| ESM-1b / ESM-2 / ESM-C 600M | General protein PLMs | SEPARATE_CHAIN |
| CurrAb (separate-chain) | Antibody PLM | SEPARATE_CHAIN |
| CurrAb (paired H/L) | Antibody PLM | PAIRED_NATIVE |

AbLang2 display names follow `representation_context`, not historical allocation labels (`ablang2_paired` is separate-chain; `ablang2_unpaired` is paired H/L inference with the same checkpoint). CurrAb paired/separate comparisons use the same CurrAb revision. Full definitions: Table 1.

### 2.3 Downstream platform (frozen)

All new cells used `DL_FOLDLOCAL_COSINE_V3`, seed 101, `d_model=128`, MEAN merge, fold-local LR selection on VAL only, no full-Dev refit. Topologies and annotations were not tuned mid-matrix. Historical FULL cells were reused when preregistered. Protocol and registry details are in the factorial preregistration and internal freeze.

### 2.4 Contrasts

- Topology effect: \(\Delta = \mathrm{MAE}(\text{topology}) - \mathrm{MAE}(\mathrm{SEP})\) (negative = helps vs SEP).
- Annotation effect: \(\Delta = \mathrm{MAE}(\text{annotation}) - \mathrm{MAE}(\mathrm{BASE})\) (negative = helps vs BASE).
- Matched context: \(\Delta = \mathrm{MAE}(\text{paired H/L}) - \mathrm{MAE}(\text{separate-chain})\) (negative = paired context better).

Paired bootstrap on OOF predictions supports selected contrasts; multi-comparison DiDs remain exploratory unless Primary and Shadow agree.

---

## 3. Results

### 3.1 Overall landscape

Across 200 cells, the leading internal configurations are AbLang2-centered (Figure 2; Table 5). The best mean(P,S) cell is **AbLang2 (separate-chain) / XREG / REGION** (EXP-T205; mean≈2.994; P≈3.005, S≈2.984). The matched AbLang2 (paired H/L) best cell is also **XREG / REGION** (EXP-T225; mean≈3.002). Absolute 4×5 landscapes differ markedly by representation (Figure 3): there is no single “winning grid” shared by all inputs.

Best cell per representation (Table 4):

| Representation | Best topology | Best annotation | mean(P,S) |
| --- | --- | ---: |
| Scratch | REG-SEP | FULL | 3.258 |
| AbLingua | FUSE | IMGT | 3.245 |
| AbLang1 | JOINT | BASE | 3.298 |
| AbLang2 (separate-chain) | XREG | REGION | 2.994 |
| AbLang2 (paired H/L) | XREG | REGION | 3.002 |
| ESM-1b | XREG | BASE | 3.208 |
| ESM-2 | FUSE | FULL | 3.332 |
| ESM-C 600M | JOINT | BASE | 3.123 |
| CurrAb (separate-chain) | XREG | REGION | 3.248 |
| CurrAb (paired H/L) | XREG | FULL | 3.250 |

Tiny MAE gaps are not treated as decisive without uncertainty assessment (Section 3.6).

### 3.2 Finding 1 — H/L topology requirements are representation-dependent

**The predictive value of explicit downstream H/L communication depended strongly on the residue representation.**

Winner labels alone already disagree (Scratch → REG-SEP; AbLang2 → XREG; ESM-C 600M → JOINT; ESM-2 → FUSE; ESM-1b → XREG). More importantly, topology ΔMAE versus SEP (Figure 4) shows different signed landscapes:

- **Scratch, FULL:** JOINT −0.185, REG-SEP −0.224, XREG −0.131, FUSE −0.043 — interaction topologies help substantially relative to SEP.
- **AbLang2 (separate-chain), FULL:** JOINT −0.103, XREG −0.104; REG-SEP and FUSE do not help.
- **ESM-2, FULL:** all four interaction topologies improve on SEP, with FUSE −0.194 among the largest gains.
- **ESM-C 600M, FULL:** all interaction deltas are **positive** (SEP preferred under FULL); the representation’s best cell is instead JOINT+BASE elsewhere in the grid.

Averaging Δ versus SEP across annotations still separates families (e.g., ESM-1b/ESM-2 tend to benefit from XREG; ESM-C averages near non-negative interaction deltas; AbLang2 contexts favor XREG on average). Primary/Shadow agreement and bootstrap topology-gain tables should be preferred over isolated cells when emphasizing robustness.

### 3.3 Finding 2 — Explicit antibody annotation is context-dependent

**Antibody-specific positional and region annotations acted as representation- and topology-dependent inductive biases rather than universally beneficial features.**

Figure 5 shows signed ΔMAE versus BASE for IMGT, REGION, and FULL that change across both rows (representations) and columns (topology). REGION/FULL are **not** universal improvements.

A sharp example under **XREG**: REGION versus BASE improves AbLang2 (separate-chain) by ≈−0.15 mean(P,S) and AbLang2 (paired H/L) by ≈−0.32, while harming Scratch (+0.12), ESM-1b (+0.17), and ESM-C (+0.16) under the same topology. Several strong general-protein cells use **BASE** (ESM-1b XREG+BASE; ESM-C JOINT+BASE). AbLang1’s best cell is also JOINT+BASE.

We do **not** interpret a weak IMGT or FULL effect as evidence that “the PLM already knows IMGT.” The ablation concerns information passed into the downstream model.

### 3.4 Finding 3 — Paired PLM inference and downstream H/L interaction are not equivalent

**Changing H/L context during PLM inference altered the downstream error landscape, but its relationship with downstream H/L interaction differed between PLMs.**

Matched-checkpoint PAIR−SEPARATE heatmaps (Figure 6; same color scale) show:

- **CurrAb:** mean PAIR−SEPARATE ≈ −0.07 (paired H/L context better on average). Best separate and paired cells remain close in absolute MAE (Table 4: 3.248 vs 3.250).
- **AbLang2:** mean PAIR−SEPARATE ≈ +0.04 (separate-chain context better on average in this matrix), with a more mixed annotation×topology surface. Both contexts still peak at **XREG+REGION**.

Thus average context advantage and near-tied best cells can coexist. Pairing × topology interaction contrasts (three-way tables) are exploratory; they do not support a claim that downstream XREG (or any topology) is a literal substitute for PLM-internal paired contextualization.

### 3.5 Scratch as a scientific control

Scratch is not only a weak baseline. Without pretrained residues, the predictor’s dependence on explicit topology and annotation is large: under FULL, moving from SEP to REG-SEP improves mean(P,S) from ≈3.483 to ≈3.258 (EXP-T091 → EXP-T096), with Primary and Shadow both near 3.25–3.26 and very small |P−S|.

With suitable REG-SEP+FULL inductive bias, Scratch becomes competitive with several PLM cells that use mismatched topology/annotation combinations. Preferred reading:

> Pretrained residue representations did not simply remove the need for downstream inductive bias; they changed which downstream biases were useful.

### 3.6 Robustness and uncertainty

Figure 7 shows Primary versus Shadow for all 200 cells. Most track near the diagonal; a minority have large |P−S| (including some strong mean cells such as ESM-C JOINT+BASE and ESM-2 FUSE+FULL). Ranking by worst(P,S) therefore differs from ranking by mean alone (Table 5b). Bootstrap CIs for matched paired-minus-separate contrasts often include zero; stars on Figure 6 are descriptive markers, not a binary significance map. External Public/Private scores were computed only after the internal freeze and are diagnostic only (they must not rewrite internal conclusions or trigger redesign).

### 3.7 Secondary observations (descriptive)

- **ESM-C 600M** can be strong (JOINT+BASE, mean≈3.123) despite weaker performance in a narrow XREG+FULL smoke reading (≈3.345) — a single smoke cell is not a substitute for the full factorial.
- **ESM-1b** outperforming **ESM-2** on this TmApp matrix (best means 3.208 vs 3.332) shows no simple generation ordering.
- Antibody-oriented models do not form a simple monotonic hierarchy: AbLang2 leads, but AbLang1/AbLingua/CurrAb best cells are not uniformly ahead of all general-protein configurations.

---

## 4. Limitations

- Endpoint is **TmApp only**; no claim about HIC or broader developability.
- Frozen platform and preregistered design space; no claim of global optimality over all architectures.
- OOF Primary/Shadow are internal; external diagnostics are post-hoc and exploratory.
- Multiplicity: many representation×topology×annotation contrasts; isolated CI exclusions are not definitive.
- AbLang2 human labels require `representation_context` because historical allocation names are inverted relative to inference context.
- Predictive inductive-bias evidence is not mechanistic evidence about PLM internals, REG geometry, or physical H/L interfaces.

---

## 5. Conclusions

For VH/VL → **TmApp** under a shared downstream protocol:

1. **Topology:** Explicit downstream H/L communication is not universally helpful; its value depends strongly on the residue representation.
2. **Annotation:** IMGT / CDR-FR features are context-dependent inductive biases, not universal upgrades—especially clear under XREG for AbLang2 REGION gains versus BASE-favoring ESM configurations.
3. **Paired inference:** Matched paired versus separate-chain PLM contexts change the error surface, but AbLang2 and CurrAb patterns differ, and downstream topologies are not interchangeable with PLM-internal pairing.
4. **Scratch:** Without pretrained residues, topology/annotation choices matter a great deal; pretrained inputs reshuffle which biases help rather than erasing the need for design choices.

### Non-claims

This report does not claim that PLMs “understand antibody structure,” that REG tokens correspond to physical H/L contacts, that paired PLMs “learned the interface,” or that IMGT information is encoded internally—unless separately demonstrated. It also does not extend conclusions beyond TmApp.

---

## 6. References

1. Shehata, L., Maurer, D. P., Wec, A. Z., Lilov, A., Champney, E., Sun, T., Archambault, K., Burnina, I., Lynaugh, H., Zhi, X., Xu, Y., & Walker, L. M. (2019). Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports*, *28*(13), 3300–3308.e4. https://doi.org/10.1016/j.celrep.2019.08.056

2. Repository assay metadata used for TmApp wording and supplement completeness counts: `gate_b1/reports/assay_definitions.md` (Shehata et al. 2019 supplement column **TmApp (°C)**).

---

## 7. Reproducibility pointers

| Item | Location |
| --- | --- |
| Human master (200 cells) | `technical_report/tmapp_factorial/data/tmapp_factorial_human_master.csv` |
| Figures 1–7 | `technical_report/tmapp_factorial/figures/` |
| Tables 1–5 | `technical_report/tmapp_factorial/tables/` |
| Terminology | `technical_report/tmapp_factorial/terminology.yaml` |
| Frozen machine results | `results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv` |
| Internal freeze | `results/TMAPP_REP_TOPO_ANNOT_INTERNAL_FREEZE.yaml` |
| Platform | `DL_FOLDLOCAL_COSINE_V3`, seed 101 |
| Legacy topology map | A→SEP, B1→JOINT, B2→REG-SEP, C→XREG, D→FUSE |

---

**TmApp REPORT FROZEN.** No further TmApp experiments, retraining, or redesign are launched from this document.

*End of TmApp technical report. No new experiments were run for this document.*
