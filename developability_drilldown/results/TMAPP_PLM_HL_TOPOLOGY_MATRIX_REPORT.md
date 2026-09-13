# TmApp PLM × H/L Topology Matrix Report (T151–T156)

**Question:** Does optimal H/L communication topology depend on the pretrained residue representation?  
**Git at analyze:** `c8714173b3ad966b3bab4e658a144b82a4120cce`  
**Internal freeze:** `results/TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.*`  
**Next unused code:** `EXP-T157` (do not run).

## Central matrix (mean(P,S) with P/S)

| PLM | A | B1 full joint | B2 residue joint | C REG→residue | D REG↔REG |
|-----|--:|--------------:|-----------------:|--------------:|----------:|
| AbLingua | 3.283 (P 3.260/S 3.307) | 3.456 (P 3.431/S 3.480) | 3.304 (P 3.243/S 3.364) | 3.256 (P 3.235/S 3.276) | 3.353 (P 3.373/S 3.334) |
| AbLang2 | 3.242 (P 3.203/S 3.280) | 3.139 (P 2.995/S 3.283) | 3.276 (P 3.241/S 3.312) | 3.138 (P 3.023/S 3.252) | 3.297 (P 3.223/S 3.372) |
| ESM-2 | 3.526 (P 3.482/S 3.569) | 3.396 (P 3.360/S 3.433) | 3.370 (P 3.361/S 3.378) | 3.409 (P 3.283/S 3.536) | 3.332 (P 3.244/S 3.420) |

## Gain vs A (mean; negative = helps)

| PLM | ΔB1 vs A | ΔB2 vs A | ΔC vs A | ΔD vs A |
|-----|---------:|---------:|--------:|--------:|
| AbLingua | +0.172 | +0.020 | -0.028 | +0.070 |
| AbLang2 | -0.103 | +0.035 | -0.104 | +0.056 |
| ESM-2 | -0.130 | -0.156 | -0.116 | -0.194 |

## PLM × topology interaction (DiD)

| Topology | PLM pair | Primary DiD | CI | Shadow DiD | CI | Interpretation |
|----------|----------|-------------|----|------------|----|----------------|
| B1 | ablingua vs ablang2 | +0.379 | [+0.075,+0.713] | +0.170 | [-0.133,+0.458] | ablang2 gains more from B1 than ablingua (CI>0) |
| B1 | ablingua vs esm2 | +0.294 | [+0.004,+0.601] | +0.310 | [+0.018,+0.574] | esm2 gains more from B1 than ablingua (CI>0) |
| B1 | ablang2 vs esm2 | -0.085 | [-0.413,+0.221] | +0.139 | [-0.167,+0.459] | DiD CI includes 0 — no strong interaction claim |
| B2 | ablingua vs ablang2 | -0.055 | [-0.300,+0.187] | +0.026 | [-0.298,+0.331] | DiD CI includes 0 — no strong interaction claim |
| B2 | ablingua vs esm2 | +0.105 | [-0.104,+0.323] | +0.248 | [-0.034,+0.519] | DiD CI includes 0 — no strong interaction claim |
| B2 | ablang2 vs esm2 | +0.160 | [-0.098,+0.416] | +0.222 | [-0.073,+0.519] | DiD CI includes 0 — no strong interaction claim |
| C | ablingua vs ablang2 | +0.155 | [-0.084,+0.407] | -0.002 | [-0.296,+0.295] | DiD CI includes 0 — no strong interaction claim |
| C | ablingua vs esm2 | +0.174 | [-0.070,+0.425] | +0.003 | [-0.263,+0.292] | DiD CI includes 0 — no strong interaction claim |
| C | ablang2 vs esm2 | +0.020 | [-0.281,+0.310] | +0.005 | [-0.310,+0.311] | DiD CI includes 0 — no strong interaction claim |
| D | ablingua vs ablang2 | +0.092 | [-0.210,+0.390] | -0.064 | [-0.325,+0.198] | DiD CI includes 0 — no strong interaction claim |
| D | ablingua vs esm2 | +0.351 | [+0.099,+0.608] | +0.177 | [-0.100,+0.458] | esm2 gains more from D than ablingua (CI>0) |
| D | ablang2 vs esm2 | +0.259 | [-0.079,+0.598] | +0.241 | [-0.031,+0.519] | DiD CI includes 0 — no strong interaction claim |

## Scratch context (not primary matrix)

| Topology | Code | Primary | Shadow | Mean | Worst |
|----------|------|---------|--------|------|-------|
| A | EXP-T091 | 3.3655 | 3.5995 | 3.4825 | 3.5995 |
| B1 | EXP-T094 | 3.1880 | 3.4077 | 3.2979 | 3.4077 |
| B2 | EXP-T096 | 3.2524 | 3.2638 | 3.2581 | 3.2638 |
| C | EXP-T102 | 3.2751 | 3.4288 | 3.3520 | 3.4288 |


## External diagnostic (does not change internal verdict)

| Code | PLM | Topo | Public | Private | Overall |
|------|-----|------|--------|---------|---------|
| EXP-T080 | ablingua | A | 3.5794 | 3.3329 | 3.4562 |
| EXP-T081 | ablingua | B1 | 3.5145 | 3.2373 | 3.3759 |
| EXP-T082 | ablingua | B2 | 3.4994 | 3.2940 | 3.3967 |
| EXP-T087 | ablingua | C | 3.5127 | 3.2646 | 3.3887 |
| EXP-T151 | ablingua | D | 3.5061 | 3.3383 | 3.4222 |
| EXP-T110 | ablang2 | A | 3.2524 | 3.0645 | 3.1585 |
| EXP-T113 | ablang2 | B1 | 3.1054 | 2.9991 | 3.0523 |
| EXP-T115 | ablang2 | B2 | 3.1967 | 2.9867 | 3.0917 |
| EXP-T121 | ablang2 | C | 3.2474 | 3.1936 | 3.2205 |
| EXP-T149 | ablang2 | D | 3.2115 | 3.0732 | 3.1424 |
| EXP-T152 | esm2 | A | 3.6524 | 3.4657 | 3.5590 |
| EXP-T153 | esm2 | B1 | 3.6436 | 3.5003 | 3.5720 |
| EXP-T154 | esm2 | B2 | 3.7595 | 3.4756 | 3.6176 |
| EXP-T155 | esm2 | C | 3.6376 | 3.5310 | 3.5843 |
| EXP-T156 | esm2 | D | 3.6521 | 3.5980 | 3.6251 |

## Answers

1. **Best for AbLingua?** **C** (EXP-T087, mean=3.2559)
2. **Best for AbLang2?** **C** (EXP-T121, mean=3.1377)
3. **Best for ESM-2?** **D** (EXP-T156, mean=3.3317)
4. **Does B1 help every PLM?** No — see gain table / bootstrap
5. **B2 representation dependence?** Gains: AbLingua +0.020, AbLang2 +0.035, ESM-2 -0.156. See DiD B2 rows.
6. **C representation dependence?** Mild. C helps all three PLMs on mean (AbLingua −0.028, AbLang2 −0.104, ESM-2 −0.116); DiD C contrasts are **not** significant (CIs include 0). C is a stable cross-PLM inductive bias more than a strong interaction driver.
7. **Does D help any PLM?** **ESM-2 only** on mean (−0.194 vs A; winner). AbLingua/AbLang2 D *hurt* vs A. Treat ESM-2 D as representation-specific, not a universal pair-compatibility win.
8. **Scratch residue-level preference reproduced by ESM-2?** Partially, not identically. Scratch prefers B2 (3.258) over A (3.483) and B1 (3.298); ESM-2 improves with B1/B2/C/D vs A, but the **best** ESM-2 cell is **D** (3.332), not B2 (3.370). Residue-level joint helps ESM-2, but post-summary D wins the ESM-2 screen.
9. **Do antibody PLMs need less residue-level mixing than ESM-2?** **Suggestive yes.** AbLingua/AbLang2 select **C** (REG-only opposite-chain read); AbLingua B1 *hurts*; AbLang2 B1 helps but ≈C; ESM-2 gains from B1/B2 and especially D. Primary DiD supports AbLang2≫AbLingua on B1 benefit and ESM-2≫AbLingua on D benefit.
10. **Bootstrap-supported PLM×topology interaction?** Primary DiD significant contrasts: **3**. Yes for listed contrasts (B1: AbLingua vs AbLang2; B1: AbLingua vs ESM-2; D: AbLingua vs ESM-2). Most other DiD CIs include 0.
11. **AbLang2 B1≈C unique?** Shared with neither AbLingua (B1 much worse than C) nor ESM-2 (D best; B1/B2/C all help). **B1≈C is AbLang2-characteristic** in this matrix.
12. **Robust across P and S?** Strongest dual-scheme signals: AbLingua C helps both schemes; ESM-2 B1/B2/D help both; AbLang2 B1 helps Primary strongly but Shadow≈0; AbLingua B1 hurts both. Prefer C for antibody PLMs; treat ESM-2 D with Primary+Shadow agreement but capacity/concept caution.
13. **Explainable by param counts alone?** Projection differs (AbLang2 61k vs 164k) but A/B1/B2 share totals within PLM; C/D add interaction params only. Ranking changes are not reducible to capacity alone within a PLM.
14. **Recommended canonical TmApp model?** **EXP-T121** (ablang2 / C) — Lowest AbLang2 mean(P,S) in matrix; antibody-specific PLM preferred when identity known. AbLang2 B1 (T113) is essentially tied.
15. **Topology if PLM unknown?** **C** (REG-only opposite-chain read) — wins both antibody PLMs; competitive on ESM-2 without requiring residue-joint.
16. **Further topology refinement?** Low priority for another C/D micro-variant screen. Next axis: PLM choice / features / data — unless replicating the 3 significant DiD contrasts on a held-out design.

## Non-claims

Do not infer that a preferred topology means a PLM “contains structure,” nor that B2 is the literal biophysical H/L mechanism. These are predictive inductive-bias results about **representation-dependent downstream communication requirements**.

## Artifacts

- Scores: `results/T151_T156_MATRIX_SCORES.csv`
- Gains: `results/T151_T156_GAINS_VS_A.csv`
- Bootstrap: `results/T151_T156_PAIRED_BOOTSTRAP.csv`
- DiD: `results/T151_T156_DID_BOOTSTRAP.csv`
- Freeze V2: `results/TMAPP_HL_TOPOLOGY_FREEZE_V2.*`
- Pre-external freeze: `results/TMAPP_PLM_TOPOLOGY_PRE_EXTERNAL_FREEZE.*`

**STOP.** Do not run EXP-T157.
