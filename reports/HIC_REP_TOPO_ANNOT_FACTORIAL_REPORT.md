# HIC Representation × Topology × Annotation Factorial — INTERNAL REPORT

**STATUS: INTERNAL_ANALYSIS (Public/Private embargo)**

- code_sha: `e4271b245493b03e93160270127fbc4168646a42`
- manifest_sha256: `b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3`
- cells COMPLETE: 200 / 200
- best TEST_mean (descriptive): `EXP-H266` esm2/JOINT/REGION = 0.494523

## 1. Execution completeness
All 200 cells COMPLETE; 0 FAILED; 0 BLOCKED. See `HIC_H140_H339_COMPLETION_AUDIT.md`.

## 2. Representation main patterns
| representation   |     mean |   median |      min |      max |
|:-----------------|---------:|---------:|---------:|---------:|
| ablingua         | 0.518019 | 0.512555 | 0.497948 | 0.557712 |
| esm2             | 0.523453 | 0.519893 | 0.494523 | 0.554628 |
| scratch          | 0.525481 | 0.52546  | 0.500649 | 0.549529 |
| esmc600m         | 0.525545 | 0.525499 | 0.500405 | 0.582601 |
| esm1b            | 0.533199 | 0.53159  | 0.514247 | 0.571706 |
| ablang1          | 0.533726 | 0.535471 | 0.515578 | 0.550492 |
| currab_unpaired  | 0.539248 | 0.538376 | 0.523174 | 0.557543 |
| currab_paired    | 0.541446 | 0.540976 | 0.521191 | 0.567994 |
| ablang2_paired   | 0.542691 | 0.54006  | 0.527988 | 0.576832 |
| ablang2_unpaired | 0.544694 | 0.544324 | 0.514857 | 0.58149  |

## 3. Topology patterns (mean TEST_mean)
| topology   |   TEST_mean |
|:-----------|------------:|
| SEP        |    0.535257 |
| JOINT      |    0.531437 |
| REG-SEP    |    0.533078 |
| XREG       |    0.531463 |
| FUSE       |    0.532517 |

## 4. Annotation patterns (mean TEST_mean)
| annotation   |   TEST_mean |
|:-------------|------------:|
| BASE         |    0.532178 |
| IMGT         |    0.536097 |
| REGION       |    0.531953 |
| FULL         |    0.530773 |

## 5–7. Interactions
- Rep×Topo table: `/workspace_developability_acquisition/developability_drilldown/results/h140_h339_analysis/rep_topo_mean.csv`
- Rep×Annot table: `/workspace_developability_acquisition/developability_drilldown/results/h140_h339_analysis/rep_annot_mean.csv`
- Topo×Annot table: `/workspace_developability_acquisition/developability_drilldown/results/h140_h339_analysis/topo_annot_mean.csv`

## 8–9. Context (PAIRED_NATIVE − SEPARATE_CHAIN)
| family   |   delta_mean |    delta_P |      delta_S |
|:---------|-------------:|-----------:|-------------:|
| ablang2  |   0.00200263 | 0.0107144  | -0.00670913  |
| currab   |   0.00219765 | 0.00487235 | -0.000477049 |

## 10. Scratch controls
Scratch mean TEST_mean = 0.525481

## 11. Primary/Shadow robustness
- mean |P−S| = 0.026865
- topology-vs-SEP both-scheme improve rate = 0.237

## 12. Bootstrap
N_BOOT=2000, seed=101; tables in `HIC_REP_TOPO_ANNOT_BOOTSTRAP.csv`.

## 13. HIGH-tail diagnostic (HIC > 11.5)
mean n_high across cells = 6.00; mean MAE_high = 2.6588; mean signed error = -2.6588
_Diagnostic only — not used for factor selection._

## 14. Best cells (descriptive only)
| experiment_code   | representation   | topology   | annotation   |   TEST_P |   TEST_S |   TEST_mean |
|:------------------|:-----------------|:-----------|:-------------|---------:|---------:|------------:|
| EXP-H266          | esm2             | JOINT      | REGION       | 0.487232 | 0.501814 |    0.494523 |
| EXP-H177          | ablingua         | FUSE       | IMGT         | 0.496334 | 0.499561 |    0.497948 |
| EXP-H288          | esmc600m         | REG-SEP    | BASE         | 0.522661 | 0.478148 |    0.500405 |
| EXP-H152          | scratch          | XREG       | BASE         | 0.513238 | 0.48806  |    0.500649 |
| EXP-H265          | esm2             | JOINT      | IMGT         | 0.520015 | 0.486947 |    0.503481 |

## 15–17. Claims / non-claims
- Established: internal 200-cell matrix completed under frozen prereg.
- Weak/exploratory: cross-target TmApp comparison is descriptive only.
- Non-claims: no Public/Private conclusions; HIGH-tail not used for selection; smoke MAE irrelevant.

## 18. Reproducibility
- prereg: `4e175ab4f98864ba7f7c6496f832e06ea9730519`
- evaluation freeze: `379e0751a93c2af8f6fbfeedaad4d72f3556996b`
- production start SHA: `e4271b245493b03e93160270127fbc4168646a42`
- analysis HEAD: `e4271b245493b03e93160270127fbc4168646a42`

## Cross-target TmApp comparison
See `developability_drilldown/results/HIC_VS_TMAPP_FACTORIAL_COMPARISON.csv` (descriptive; does not alter HIC selection).

