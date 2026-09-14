# HIC Factorial — Scientific Conclusion Freeze (Internal)

**STATUS: SCIENTIFIC_CONCLUSIONS_FROZEN_INTERNAL**

**PUBLIC_PRIVATE_NOT_CONSULTED**

This document freezes the final *internal* scientific interpretation of the
HIC Representation × Topology × Annotation 200-cell factorial.
Subsequent Public/Private diagnostics must not rewrite these conclusions.

## 1. Freeze status

| Field | Value |
|-------|--------|
| Evaluation freeze | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |
| Formal prereg | `4e175ab4f98864ba7f7c6496f832e06ea9730519` |
| Internal factorial freeze | `cca9bd5016ecd14f06727f285bd26bf58fc46647` |
| Manifest SHA256 | `b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3` |
| Cells | EXP-H140 … EXP-H339 (200/200 COMPLETE) |
| Scientific freeze HEAD (at writing) | `cca9bd5016ecd14f06727f285bd26bf58fc46647` |
| Public/Private consulted | **No** |
| New training / embeddings | **None** |

## 2. Provenance

- Internal report: `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_REPORT.md`
- Cell metrics: `developability_drilldown/results/HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv`
- Bootstrap (N_BOOT=2000, seed=101, antibody-level paired): `HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv`
- Validated contrasts: `reports/HIC_FACTORIAL_VALIDATED_CONTRASTS.csv`
- Interaction robustness: `reports/HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv`
- Validation: PASS; missing OOF files = 0

## 3. Executive scientific conclusions

1. **Best on average under this factorial (representation):** **AbLingua** (mean TEST_mean=0.5180).
2. **Best single internal cell (descriptive):** **EXP-H266** = ESM-2 × JOINT × REGION (TEST_mean=0.4945).
3. Best average representation (**AbLingua**) and best single cell (**ESM-2**) are **different families**.
4. Topology effects vs SEP are **small** (mean |Δ| ≲ 0.004); both-scheme improvement rates ≈ 0.24 — **no general strong topology win**.
5. Annotation effects vs BASE are **small**; FULL is slightly better on average, IMGT slightly worse; **representation-dependent**.
6. AbLang2 / CurrAb PAIRED vs SEPARATE show **no robust context advantage** (P/S direction mismatch).
7. **SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE** (n_high=6, mean signed error ≈ −2.66 across all 200 cells).
8. Broad Rep×Topo×Annot search modestly improved the best internal sequence-only score (≈0.4945) but **did not fundamentally break** the previously observed ~0.50 regime **within this tested design space**.
9. Cross-target: HIC and TmApp favor **different** representation patterns (descriptive only; absolute MAE not compared).
10. Sequence-only headroom appears limited → **justifies re-examining independent surface/physics evidence**, without claiming surface superiority from this factorial alone.

## 4. Representation

Ordering is **mean TEST_mean across 20 Topo×Annot conditions** — `best on average under this factorial`, not a universal ranking.

| Representation | mean | median | min | max | vs Scratch | mean\|P−S\| | best cell |
|----------------|------|--------|-----|-----|------------|-----------|-----------|
| AbLingua | 0.5180 | 0.5126 | 0.4979 | 0.5577 | -0.0075 | 0.0224 | EXP-H177 FUSE/IMGT (0.4979) |
| ESM-2 | 0.5235 | 0.5199 | 0.4945 | 0.5546 | -0.0020 | 0.0234 | EXP-H266 JOINT/REGION (0.4945) |
| Scratch | 0.5255 | 0.5255 | 0.5006 | 0.5495 | 0.0000 | 0.0404 | EXP-H152 XREG/BASE (0.5006) |
| ESM-C | 0.5255 | 0.5255 | 0.5004 | 0.5826 | 0.0001 | 0.0232 | EXP-H288 REG-SEP/BASE (0.5004) |
| ESM-1b | 0.5332 | 0.5316 | 0.5142 | 0.5717 | 0.0077 | 0.0370 | EXP-H248 REG-SEP/BASE (0.5142) |
| AbLang1 | 0.5337 | 0.5355 | 0.5156 | 0.5505 | 0.0082 | 0.0196 | EXP-H183 SEP/FULL (0.5156) |
| CurrAb SEPARATE | 0.5392 | 0.5384 | 0.5232 | 0.5575 | 0.0138 | 0.0216 | EXP-H308 REG-SEP/BASE (0.5232) |
| CurrAb PAIRED | 0.5414 | 0.5410 | 0.5212 | 0.5680 | 0.0160 | 0.0258 | EXP-H329 REG-SEP/IMGT (0.5212) |
| AbLang2 SEPARATE | 0.5427 | 0.5401 | 0.5280 | 0.5768 | 0.0172 | 0.0206 | EXP-H204 JOINT/BASE (0.5280) |
| AbLang2 PAIRED | 0.5447 | 0.5443 | 0.5149 | 0.5815 | 0.0192 | 0.0346 | EXP-H226 JOINT/REGION (0.5149) |

- Spread across Topo×Annot is material (e.g. ESM-C max−min ≈ 0.0822).
- Several PLMs (AbLang2, CurrAb) are **worse on average than Scratch** under this factorial.
- Do **not** claim AbLingua (or any family) is universally best for HIC.

## 5. Topology

Overall mean TEST_mean: SEP=0.5353, JOINT=0.5314, REG-SEP=0.5331, XREG=0.5315, FUSE=0.5325

### Contrasts vs SEP (40 Rep×Annot strata each)

| Contrast | Δmean | Δmedian | both-improve | primary CI (mean of strata) | tier |
|----------|-------|---------|--------------|-----------------------------|------|
| JOINT−SEP | -0.0038 | -0.0017 | 0.275 | [-0.0423, 0.0431] | Tier3 |
| REG-SEP−SEP | -0.0022 | -0.0010 | 0.225 | [-0.0401, 0.0425] | Tier3 |
| XREG−SEP | -0.0038 | 0.0006 | 0.300 | [-0.0411, 0.0394] | Tier3 |
| FUSE−SEP | -0.0027 | -0.0015 | 0.150 | [-0.0397, 0.0396] | Tier3 |

**Conclusion:** small average improvements exist for JOINT/XREG/FUSE vs SEP, but both-scheme rates are low (~0.24) and mean effects are ~0.002–0.004. **No general topology upgrade claim.** Topology effects are representation-dependent (see interaction robustness).

## 6. Annotation

Overall mean TEST_mean: BASE=0.5322, IMGT=0.5361, REGION=0.5320, FULL=0.5308

| Contrast | Δmean | Δmedian | both-improve | primary CI (mean of strata) | tier |
|----------|-------|---------|--------------|-----------------------------|------|
| IMGT−BASE | 0.0039 | 0.0062 | 0.240 | [-0.0364, 0.0484] | Tier3 |
| REGION−BASE | -0.0002 | 0.0017 | 0.340 | [-0.0409, 0.0446] | Tier4 |
| FULL−BASE | -0.0014 | 0.0006 | 0.240 | [-0.0430, 0.0434] | Tier4 |

**Conclusion:** FULL is slightly better than BASE on average; IMGT slightly worse; effects are small and **representation-dependent**. Do not claim FULL is always best.

## 7. Predefined contrasts (PLM − Scratch)

| PLM | Δmean | both-improve | tier |
|-----|-------|--------------|------|
| AbLingua | -0.0075 | 0.150 | Tier3 |
| ESM-2 | -0.0020 | 0.200 | Tier3 |
| ESM-C | 0.0001 | 0.200 | Tier4 |
| ESM-1b | 0.0077 | 0.000 | Tier3 |
| AbLang1 | 0.0082 | 0.100 | Tier3 |
| CurrAb SEPARATE | 0.0138 | 0.050 | Tier3 |
| CurrAb PAIRED | 0.0160 | 0.100 | Tier3 |
| AbLang2 SEPARATE | 0.0172 | 0.000 | Tier3 |
| AbLang2 PAIRED | 0.0192 | 0.050 | Tier3 |

Only AbLingua (and weakly ESM-2) show average improvement vs Scratch; several antibody PLMs are average-worse. This is a factorial-average fact, not mechanism.

## 8. Interactions

### A. Single-cell extremes (not robust patterns)
- Rep×Topo extreme residual: **esmc600m × REG-SEP** (IMGT/EXP-H289), resid=0.0567 — outlier-sensitive.
- Rep×Annot extreme residual: **esmc600m × IMGT** (REG-SEP/EXP-H289), resid=0.0537 — outlier-sensitive.
- `EXP-H289` (esmc600m × REG-SEP × IMGT) is an **extreme high-error cell**, not a replicated interaction template.

### B. Reproducible / robustness-filtered Rep×Topo (vs SEP across annotations)

See `HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv`. Patterns with higher direction consistency and lower outlier_sensitive flags are preferred over max-residual cells.

| Rep | Topo | median Δ | LOO mean | dir cons. | P/S cons. | outlier? | tier |
|-----|------|----------|----------|-----------|-----------|----------|------|
| esm1b | REG-SEP | -0.0205 | -0.0179 | 0.75 | 0.50 | False | Tier2 |
| scratch | XREG | -0.0175 | -0.0169 | 1.00 | 0.50 | False | Tier2 |
| esm1b | FUSE | -0.0160 | -0.0142 | 0.75 | 0.00 | False | Tier3 |
| ablang2_paired | XREG | -0.0154 | -0.0112 | 0.75 | 0.75 | False | Tier2 |
| currab_unpaired | JOINT | -0.0149 | -0.0069 | 0.75 | 0.50 | True | Tier3 |

### C. Rep×Annotation

AbLingua / ESM-2 / ESM-C / Scratch annotation deltas vary by topology; IMGT harm and FULL benefit are **not uniform** across representations. Use the robustness table; do not over-interpret single cells.

### D. Topology × Annotation

Marginal Topo×Annot means are weak relative to representation main effects. Any Topo×Annot preference should be treated as **representation-dependent**.

## 9. Context effects

### ablang2
- n_strata=20; Δmean=0.0020; ΔP=0.0107; ΔS=-0.0067; both-improve=0.100
- Evidence: **Tier4** — Primary/Shadow direction disagreement → no robust context advantage

### currab
- n_strata=20; Δmean=0.0022; ΔP=0.0049; ΔS=-0.0005; both-improve=0.250
- Evidence: **Tier4** — Primary/Shadow direction disagreement → no robust context advantage

**AbLang2 / CurrAb pair context: no robust advantage** under Primary+Shadow agreement rules.

## 10. HIGH-tail

Definition: `HIC > 11.5` (diagnostic only; not used for selection).

- n_high = 6 for every cell
- MAE_high: mean=2.6588, min=1.9814, max=3.0446
- mean signed error (pred−true): -2.6588 (systematic underprediction)
- Best HIGH-tail MAE cell (descriptive): EXP-H272 (esm2/XREG/BASE) MAE_high=1.9814

**SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE**

No Rep/Topo/Annot combination in this design eliminated the failure mode. Causes (scarcity / representation / noise / loss) remain **unresolved** — do not pick one.

## 11. Sequence-only plateau reassessment

Prior observation: V3-tested configurations plateau near MAE ~0.50.
This factorial’s best internal TEST_mean is **0.4945** (EXP-H266).

> Broad Rep×Topo×Annot search modestly improved the best internal sequence-only score but did not fundamentally break the previously observed ~0.50 regime **within the tested design space**.

This is **not** a theoretical lower bound on HIC MAE.

## 12. TmApp vs HIC

See `reports/TMAPP_VS_HIC_FACTORIAL_SCIENTIFIC_COMPARISON.md`. Absolute MAE not compared.

Representation mean ranks (lower rank = better mean):

| Representation | HIC rank | TmApp rank |
|----------------|----------|------------|
| AbLingua | 1 | 4 |
| ESM-2 | 2 | 10 |
| Scratch | 3 | 7 |
| ESM-C | 4 | 3 |
| ESM-1b | 5 | 6 |
| AbLang1 | 6 | 9 |
| CurrAb SEPARATE | 7 | 8 |
| CurrAb PAIRED | 8 | 5 |
| AbLang2 SEPARATE | 9 | 1 |
| AbLang2 PAIRED | 10 | 2 |

**Same VH/VL inputs do not imply the same effective representation / inductive bias for TmApp vs HIC** (descriptive pattern difference).

## 13. Evidence hierarchy

### Tier 1 — Established internal observation
- 200/200 cells completed under frozen prereg; 0 failed/blocked
- Best descriptive cell: EXP-H266 TEST_mean=0.4945
- Representation mean ordering under this factorial (AbLingua best average)
- Persistent HIGH-tail underprediction (n=6, signed error ≈ −2.66) across all cells
- PUBLIC_PRIVATE_NOT_CONSULTED

### Tier 2 — Supported pattern
- Several PLMs average-worse than Scratch (consistent across many Topo×Annot strata)
- Topology/annotation *average* effects are small relative to cell-to-cell spread
- Context contrasts lack Primary+Shadow directional agreement

### Tier 3 — Suggestive
- Modest JOINT/XREG mean edges vs SEP
- Modest FULL edge vs BASE
- Single-cell interaction extremes (e.g. EXP-H289)
- Specific Rep×Topo/Annot robustness rows with mixed P/S consistency

### Tier 4 — Unresolved
- Causal driver of HIGH-tail failure
- Whether any untested sequence-only design could break ~0.50
- Mechanism behind AbLingua vs ESM-2 average-vs-best divergence
- External generalization (Public/Private)

## 14–17. Claim lists

See executive conclusions (§3) and hierarchy (§13). Validated numeric contrasts live in CSVs.

## 18. Non-claims

- AbLingua is not claimed universally best for HIC
- ESM-2 is not claimed to ‘understand’ HIC mechanism
- FULL annotation is not always best
- JOINT/XREG are not universally superior topologies
- HIGH-tail failure is not attributed solely to data scarcity
- No aromatic/surface causality from this factorial
- No Public/Private generalization claimed
- No theoretical MAE floor at 0.49

## 19. Implications for next research stage

Sequence-only design-space headroom under this factorial appears limited (best ≈ 0.4945; HIGH-tail unresolved). This **increases the rationale for re-examining independent surface/physics evidence**, but **does not prove surface superiority from this factorial alone**.

## 20. Reproducibility

| Artifact | Role |
|----------|------|
| `4e175ab4f98864ba7f7c6496f832e06ea9730519` | Formal prereg |
| `379e0751a93c2af8f6fbfeedaad4d72f3556996b` | Evaluation freeze |
| `cca9bd5016ecd14f06727f285bd26bf58fc46647` | Internal results freeze |
| `b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3` | Manifest SHA256 |
| `reports/HIC_FACTORIAL_VALIDATED_CONTRASTS.csv` | Contrast table |
| `reports/HIC_FACTORIAL_INTERACTION_ROBUSTNESS.csv` | Interaction robustness |
| `reports/TMAPP_VS_HIC_FACTORIAL_SCIENTIFIC_COMPARISON.md` | Cross-target |

Bootstrap: antibody-level paired residual; N_BOOT=2000; seed=101; Primary/Shadow separate.

## Required explicit answers

1. **Best average representation:** AbLingua (under this factorial).
2. **Best single cell:** EXP-H266 (esm2 × JOINT × REGION).
3. **Same family?** No (AbLingua vs ESM-2).
4. **General topology improvement?** Not strong; only small average Δ vs SEP.
5. **Topology representation-dependent?** Yes.
6. **General annotation improvement?** Weak; FULL slight average gain, IMGT slight harm.
7. **Annotation representation-dependent?** Yes.
8. **AbLang2 pair context effective?** No robust advantage (P/S disagree).
9. **CurrAb pair context effective?** No robust advantage.
10. **HIGH-tail resolved?** No — SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE.
11. **Broke ~0.50 plateau?** Modestly improved best cell; did not fundamentally break regime in tested space.
12. **Same optimal representation pattern as TmApp?** No (rank patterns differ).
13. **Surface follow-up justified?** Rational to re-examine independent surface evidence due to limited sequence-only headroom; not proven by this factorial alone.

