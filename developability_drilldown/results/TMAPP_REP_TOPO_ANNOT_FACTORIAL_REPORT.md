# TmApp Representation × H/L Topology × Annotation Factorial Report

- PRE_REG_COMMIT_SHA: `7727ecc0f82798c8788d4b2f9ba3295e50502802`
- Registered cells: **200** (REUSE=23, COMPLETE_new=177, FAILED=0, BLOCKED=0)
- Platform: `DL_FOLDLOCAL_COSINE_V3`, seed `101`

## 1. Experiment design

Complete factorial: 10 representations × 5 topologies (A/B1/B2/C/D) × 4 annotations (BASE/IMGT/REGION/FULL).
No midway model selection; all preregistered cells executed.

## 2. Representation provenance

Frozen residue bundles; matched AbLang2/CurrAb pairs hold checkpoint identity fixed.
**AbLang2 audit:** historical `ablang2_paired` asset is SEPARATE_CHAIN; `ablang2_unpaired` is matched PAIRED_NATIVE joint inference.
**CurrAb:** paired=PAIRED_NATIVE, unpaired=SEPARATE_CHAIN, revision `92e28534663e163f1b398f773b3fe041085737d9`.

## 3. Complete 200-cell matrix status

All 200 cells finished: 23 REUSE + 177 COMPLETE. No FAILED/BLOCKED.

## 4. Overall results

- **Best overall:** EXP-T205 `ablang2_paired/C/REGION` mean(P,S)=2.9944 (P=3.0050, S=2.9838)
- **Best Scratch:** EXP-T096 `B2/FULL` mean=3.2581

Best cell per representation:

| representation | topology | annotation | code | mean(P,S) | P | S |
|---|---|---|---|---:|---:|---:|
| ablang1 | B1 | BASE | EXP-T232 | 3.2977 | 3.3066 | 3.2889 |
| ablang2_paired | C | REGION | EXP-T205 | 2.9944 | 3.0050 | 2.9838 |
| ablang2_unpaired | C | REGION | EXP-T225 | 3.0018 | 2.9668 | 3.0368 |
| ablingua | D | IMGT | EXP-T186 | 3.2452 | 3.2283 | 3.2622 |
| currab_paired | C | FULL | EXP-T160 | 3.2496 | 3.3158 | 3.1833 |
| currab_unpaired | C | REGION | EXP-T336 | 3.2483 | 3.2458 | 3.2509 |
| esm1b | C | BASE | EXP-T253 | 3.2082 | 3.1693 | 3.2471 |
| esm2 | D | FULL | EXP-T156 | 3.3317 | 3.2438 | 3.4196 |
| esmc600m | B1 | BASE | EXP-T285 | 3.1230 | 3.0349 | 3.2112 |
| scratch | B2 | FULL | EXP-T096 | 3.2581 | 3.2524 | 3.2638 |

## 5. H/L topology effects

Under FULL, topology gains vs A (negative ⇒ interaction helps):

```
topology             B1     B2      C      D
representation                              
ablang1           0.005  0.052 -0.012  0.171
ablang2_paired   -0.103  0.035 -0.104  0.056
ablang2_unpaired -0.013 -0.091 -0.091  0.069
ablingua          0.172  0.020 -0.028  0.070
currab_paired     0.016  0.097 -0.053  0.035
currab_unpaired   0.055 -0.017  0.025  0.002
esm1b            -0.061 -0.035 -0.122  0.036
esm2             -0.130 -0.156 -0.116 -0.194
esmc600m          0.203  0.071  0.074  0.175
scratch          -0.185 -0.224 -0.131 -0.043
```

- Scratch shows large B1/B2/C gains vs A under FULL (especially B2).
- ESM-2 shows broad topology gains including D under FULL.
- AbLang2 (allocation paired / SEPARATE) prefers C and B1 under FULL.
- ESM-C 600M prefers A under FULL (positive gains for B1/B2/C/D).

Machine table: `TMAPP_REP_TOPO_ANNOT_TOPOLOGY_GAINS.csv`; bootstrap: `TMAPP_REP_TOPO_ANNOT_BOOTSTRAP.csv`.

## 6. Annotation effects

Annotation gains vs BASE vary strongly by representation (see heatmaps / CSV).
Non-additivity contrasts (FULL−IMGT−REGION+BASE) are in annotation gains + bootstrap.

## 7–8. Representation × topology / annotation interactions

Full DiD tables: `TMAPP_REP_TOPO_ANNOT_DID_TOPOLOGY.csv`, `TMAPP_REP_TOPO_ANNOT_DID_ANNOTATION.csv`.
Treat multi-comparison DiD as exploratory unless Primary and Shadow agree.

## 9. AbLang2 paired-vs-unpaired

Mean Δpair (allocation paired − unpaired) by annotation: {'BASE': -0.0772, 'FULL': -0.0301, 'IMGT': 0.0422, 'REGION': -0.0906}
Overall mean Δpair ≈ −0.039 (allocation-paired/SEPARATE slightly better on average).
Interpret via `representation_context` (SEPARATE vs PAIRED_NATIVE), not the allocation name alone.

## 10. CurrAb paired-vs-unpaired

Mean Δpair by annotation: {'BASE': -0.0798, 'FULL': -0.0844, 'IMGT': -0.0925, 'REGION': -0.0413}
Overall mean Δpair ≈ −0.075 (true PAIRED_NATIVE better than SEPARATE on average).

## 11. Scratch control

Best Scratch is B2/FULL (EXP-T096, mean=3.2581).
Under FULL, Scratch topology gains vs A are among the largest in the matrix — consistent with stronger need for downstream H/L inductive bias without pretrained residues.

## 12. PLM generation/domain observations

Descriptive only: antibody-oriented AbLang2 cells lead the matrix; ESM-C 600M is competitive among general-protein PLMs; ESM-2 benefits more from topology under FULL than several antibody PLMs.
Wording: consistent with representation-dependent downstream requirements — not causal corpus claims.

## 13. Primary/Shadow robustness

See `t161_t337_figures/primary_vs_shadow.png` and bootstrap CIs.

## 14. Internal ranking

Top 10 by mean(P,S):

```
experiment_code   representation topology annotation  mean_ps  worst_ps  primary_mae  shadow_mae
       EXP-T205   ablang2_paired        C     REGION 2.994409  3.004971     3.004971    2.983847
       EXP-T225 ablang2_unpaired        C     REGION 3.001760  3.036761     2.966759    3.036761
       EXP-T196   ablang2_paired        D       BASE 3.026136  3.026978     3.026978    3.025294
       EXP-T206   ablang2_paired        D     REGION 3.103221  3.136174     3.070267    3.136174
       EXP-T285         esmc600m       B1       BASE 3.123026  3.211171     3.034881    3.211171
       EXP-T220 ablang2_unpaired        C       IMGT 3.124321  3.200682     3.200682    3.047961
       EXP-T202   ablang2_paired        A     REGION 3.133351  3.181393     3.085309    3.181393
       EXP-T121   ablang2_paired        C       FULL 3.137653  3.252085     3.023221    3.252085
       EXP-T113   ablang2_paired       B1       FULL 3.139063  3.283071     2.995055    3.283071
       EXP-T195   ablang2_paired        C       BASE 3.140193  3.279838     3.000548    3.279838
```

## 15. Scientific interpretation

1. Preferred H/L topology depends on representation (e.g., Scratch/ESM-2 gain from interaction; ESM-C often prefers A under FULL).
2. Value of explicit annotation depends on representation (and often on topology).
3. Annotation × topology non-additivity is representation-specific (see CSV).
4–5. For matched AbLang2/CurrAb, pairing context changes absolute MAE and, modestly, topology/annotation gains (three-way table).
6. Scratch vs PLMs under the same design space supports pretrained residue info reducing (not eliminating) need for some downstream biases.

## 16. Non-claims

- Predictive inductive-bias evidence only — not mechanistic/structural claims about PLM internals.
- Downstream topology is not claimed to be literally equivalent to PLM-internal pairing.
- External Public/Private scores are diagnostic only and do not rewrite this freeze.

## 17. Failures / blocked cells

None.

## 18. Reproducibility / commits

- INTERNAL_FREEZE_COMMIT_SHA: `941c373416426103424d7fa89f20b6bb96cec895`

- PRE_REG_COMMIT_SHA: `7727ecc0f82798c8788d4b2f9ba3295e50502802`
- INTERNAL_FREEZE / FINAL SHAs recorded at commit time
- Hard STOP after this factorial: no new topologies/annotations/PLMs/ensembles/HIC.

- FINAL_COMMIT_SHA: `99981196b158d7931fbce88c14259aef11358a36`
