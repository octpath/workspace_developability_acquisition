# HIC H140–H339 internal freeze

**STATUS: INTERNAL_FROZEN**  
**Public / Private / GEN_0001 consulted: NO**

| Field | Value |
|-------|--------|
| Formal prereg | `4e175ab4f98864ba7f7c6496f832e06ea9730519` |
| Evaluation freeze | `379e0751a93c2af8f6fbfeedaad4d72f3556996b` |
| Smoke freeze | `51db0e0cf38487c4de452a6345880a3c125c0fad` |
| Runner harden | `c5000a0a6f3b085ed5db727b5fa803e28b93e853` |
| Venv-detect patch | `d15edb8925fa8b4e07d8bd2dabcfb7ba7e91e9bf` |
| Status-CSV patch | `e4271b245493b03e93160270127fbc4168646a42` |
| Bundle-loader patch | `cc305d4b4e06209fb15281e3ae6431c2664a1318` |
| Manifest SHA256 | `b0790d0609da3beaa22c7aaa63be10835ce0833247575c5ee039467d64e5b6c3` |
| COMPLETE / FAILED / BLOCKED | **200 / 0 / 0** |

## Descriptive best cell (not a selection decision)

`EXP-H266` — esm2 × JOINT × REGION — TEST_mean ≈ 0.4945

## Major internal patterns (summary)

- Representation: AbLingua and ESM-2 strongest on average; AbLang2/CurrAb weaker than Scratch on mean TEST_mean
- Topology: modest gains vs SEP; JOINT / XREG slightly better on average
- Annotation: FULL slightly better than BASE on average; IMGT slightly worse
- Context: PAIRED_NATIVE − SEPARATE_CHAIN mean deltas near zero / mixed Primary vs Shadow (no strong consistent gain)
- HIGH-tail (HIC>11.5): persistent underprediction (diagnostic only)

Machine-readable twin: `reports/HIC_H140_H339_INTERNAL_FREEZE.yaml`  
Full report: `reports/HIC_REP_TOPO_ANNOT_FACTORIAL_REPORT.md`
