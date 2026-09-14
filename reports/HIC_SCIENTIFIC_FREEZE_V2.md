# HIC Scientific Freeze v2

**STATUS: HIC_SCIENTIFIC_FREEZE_V2**

This document freezes the **current scientific conclusions** for HIC by integrating:

1. Internal factorial freeze (Public/Private embargoed at the time)
2. Prospective factorial external diagnosis (EXP-H140–H339)
3. Retrospective historical evidence (pre-factorial HIC experiments, including surface/HSP lineages)

It does **not** rewrite or edit:

- `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md`
- `reports/HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md`

---

## 0. Provenance ledger (must remain separated)

| Stream | Scope | Role in v2 | Holdout purity |
|--------|-------|------------|----------------|
| **Internal factorial** | EXP-H140–H339 CV metrics; scientific freeze `559980e3` / factorial freeze `cca9bd50` | Factor-level conclusions under embargo | Internal only; Pub/Priv not used for freeze v1 |
| **Prospective factorial external** | Same 200 cells scored after authorization; diagnosis `4a2568b4` | External validation of frozen internal claims | Prospective **relative to this factorial freeze**; not a pristine project-wide holdout |
| **Retrospective historical** | H001–H139 and prior Test-scored rows (e.g. H102–H113, H137) | Reference candidates, surface association, all-history ranking | Test may already have informed project history — **not** unused holdout |

| Field | Value |
|-------|--------|
| Freeze v2 parent HEAD | `4a2568b454e0b039a5566576b50768e7c0595938` |
| Freeze v2 introducing commit | *(this commit on main)* |
| Internal scientific freeze | `559980e37576eaffaeba9603cee591544e52d991` |
| External diagnosis | `4a2568b454e0b039a5566576b50768e7c0595938` |
| Evaluation freeze | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |
| Formal factorial prereg | `4e175ab4f98864ba7f7c6496f832e06ea9730519` |
| New training in this freeze | **None** |

---

## Claim 1 — Fine-grained model selection is unstable

**Provenance:** prospective factorial external vs internal CV (200 cells).

| Metric | Value |
|--------|-------|
| Pearson(cv_mean, test_overall) | ≈ **0.115** |
| Spearman | ≈ **0.127** |
| Kendall τ | ≈ **0.085** |
| Internal Top10 ∩ External Top10 | **∅** |
| mean \|rank change\| | ≈ 62 |

Internal best cell **EXP-H266** (ESM-2 × JOINT × REGION) falls to external rank **26/200**.

**Frozen conclusion:**

> Internal CV is weak for selecting the exact best HIC architecture/cell.

**Not frozen:** “CV is useless.” Factor-level aggregates (topology, annotation direction, HIGH-tail) still transfer better than cell ranks.

---

## Claim 2 — Topology is secondary

**Provenance:** internal matched Rep×Annot contrasts + prospective external Test Overall contrasts.

External SEP contrasts (mean Δ, n=40 strata each):

| Contrast | External mean Δ |
|----------|-----------------|
| JOINT − SEP | ≈ **−0.0044** |
| REG-SEP − SEP | ≈ **+0.0018** |
| XREG − SEP | ≈ **+0.0008** |
| FUSE − SEP | ≈ **+0.0028** |

Internal mean |Δ| was similarly ≲ 0.004 with low both-scheme rates.

**Frozen conclusion:**

> More sophisticated H/L interaction modeling is not supported as a major bottleneck for HIC in the tested design space.

**Not frozen:** “H/L interaction is irrelevant.” Effects can be representation-dependent and nonzero.

---

## Claim 3 — Annotation is secondary and context-dependent

**Provenance:** internal + prospective external matched Rep×Topo contrasts vs BASE.

External BASE contrasts:

| Contrast | External mean Δ |
|----------|-----------------|
| IMGT − BASE | ≈ **+0.0031** |
| REGION − BASE | ≈ **−0.0010** |
| FULL − BASE | ≈ **−0.0040** |

Direction roughly matches internal (FULL slightly favorable, IMGT slightly unfavorable).

**Frozen conclusion:**

> Region annotation provides at most modest average benefit and is representation-dependent.

---

## Claim 4 — Representation ranking is split-sensitive

**Provenance:** internal factorial averages vs prospective external averages (same 200 cells).

| Split | Best average | Scratch rank (of 10) |
|-------|--------------|----------------------|
| Internal CV | **AbLingua** | **3** |
| External Test | **AbLang1** | **10** |

Externally, in matched topology×annotation comparisons, **all pretrained representations beat Scratch on average** (improve fractions 0.50–1.00). Internally, several antibody PLMs were worse than Scratch on average.

**Frozen conclusion:**

> Pretraining is externally useful in this factorial, but the identity of the best PLM is not stable enough to freeze as a universal winner.

---

## Claim 5 — HIGH-tail failure is robust

**Provenance:** internal factorial + prospective external Test; threshold fixed at **HIC > 11.5** (no new thresholds).

| Split | n_high | mean signed error (pred − true) |
|-------|--------|----------------------------------|
| Internal (train/OOF HIGH set) | **6** | ≈ **−2.66** |
| External Test | **7** | ≈ **−2.87** |

All 200 sequence-only factorial cells retain severe HIGH underprediction.

**Frozen labels:**

- `ROBUST_HIGH_HIC_SYSTEMATIC_UNDERPREDICTION`
- `SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE`

**Candidate causes (not adjudicated):** sample scarcity; missing physicochemical / surface information; loss / calibration; assay noise.

---

## Claim 6 — Retire “~0.50 plateau”; adopt performance-gap framing

**Do not use** the v1 phrasing “sequence-only plateau near 0.50” as a ceiling claim.

Prospective factorial external:

| Statistic | Value |
|-----------|-------|
| Best Test Overall | ≈ **0.4418** (EXP-H185) |
| Median Test Overall | ≈ **0.4809** |

Retrospective references:

| Model | Role | Test Overall |
|-------|------|--------------|
| EXP-H107 | balanced CV+external reference | ≈ **0.3936** |
| EXP-H137 | historical Test-best | ≈ **0.3907** |

**Frozen concept:** `SEQUENCE_ONLY_TO_SURFACE_PERFORMANCE_GAP`

**Frozen conclusion:**

> Sequence-only models can clearly break 0.50, but a substantial performance gap remains relative to the strongest explicit surface/physicochemical models observed so far.

---

## Claim 7 — Scientific reference candidate is H107 (not H137)

| Code | Role | CV mean | Test | \|Pub−Priv\| |
|------|------|---------|------|-------------|
| **EXP-H107** | **scientific reference (balanced)** | ≈ **0.4476** | ≈ **0.3936** | ≈ **0.0046** |
| EXP-H137 | historical Test-best (descriptive) | ≈ 0.5112 | ≈ 0.3907 | ≈ 0.0108 |

H107 = Scratch × ARCH-2 × **SURFACE (F1) + HSP EIS-R8** (retrospective lineage).

**Surface causality is not frozen.** Matched ablation reaudit (`reports/HIC_SURFACE_MATCHED_ABLATION_REAUDIT.md`) supports association with contamination caveats.

**Frozen wording:**

> Explicit surface/physicochemical information is the leading remaining explanatory hypothesis for the observed performance gap, but causality is not yet established.

---

## Cross-claim status vs Freeze v1 (internal)

| Topic | v2 status vs internal freeze |
|-------|------------------------------|
| Topology secondary | **SUPPORTED** externally |
| Annotation modest / rep-dependent | **SUPPORTED** externally |
| AbLingua universal average-best | **WEAKENED** (external average → AbLang1) |
| Scratch surprisingly competitive | **CONTRADICTED** externally (rank 10) |
| Several Ab-PLMs worse than Scratch | **CONTRADICTED** externally |
| HIGH-tail unresolved | **SUPPORTED** / strengthened |
| ~0.50 plateau as ceiling | **RETIRED** → gap framing |
| AbLang2 / CurrAb context | Internal “no robust advantage” **not rewritten**; external additive notes only |

---

## Non-claims (explicit)

- No universal best PLM identity.
- No topology winner from a single Test-best cell.
- No causal proof that SURFACE features cause lower HIC error.
- No claim that historical Test rankings are pristine holdout evidence.
- No new training / architecture search authorized by this freeze alone.

---

## Companion artifacts

- External diagnosis: `reports/HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md`
- Surface matched reaudit: `reports/HIC_SURFACE_MATCHED_ABLATION_REAUDIT.md`
- Surface pair CSV: `reports/HIC_SURFACE_MATCHED_ABLATION_PAIRS.csv`
- Internal freeze (immutable): `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md`

---

## Next scientific step (decision pointer)

Surface matched reaudit decision: **`PROSPECTIVE_SURFACE_REPLICATION_RECOMMENDED` (C2)** — small AbLang1 / AbLingua (± Scratch) SURFACE add-on replication with frozen JOINT×FULL; details in the reaudit report. **Not executed in this freeze.**
