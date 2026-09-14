# TmApp vs HIC Factorial — Scientific Comparison (Descriptive)

**STATUS: DESCRIPTIVE_CROSS_TARGET_ONLY**

Based solely on frozen TmApp and HIC internal factorials. Does **not** alter either target’s frozen conclusions. **Absolute MAE values are not compared** (different target scales).

## Representation mean ranks

| Representation | HIC mean-rank | TmApp mean-rank | Pattern note |
|----------------|---------------|-----------------|--------------|
| AbLingua | 1 | 4 | rank shift ≥3 |
| ESM-2 | 2 | 10 | rank shift ≥3 |
| Scratch | 3 | 7 | rank shift ≥3 |
| ESM-C | 4 | 3 |  |
| ESM-1b | 5 | 6 |  |
| AbLang1 | 6 | 9 | rank shift ≥3 |
| CurrAb SEPARATE | 7 | 8 |  |
| CurrAb PAIRED | 8 | 5 | rank shift ≥3 |
| AbLang2 SEPARATE | 9 | 1 | rank shift ≥3 |
| AbLang2 PAIRED | 10 | 2 | rank shift ≥3 |

Notable: AbLingua is strongest on average for **HIC**; AbLang2/CurrAb are weak on HIC relative to Scratch, whereas TmApp factorial ranks differ — evidence that **effective representation bias is target-dependent** for the same VH/VL inputs.

## Topology relative gains (vs SEP/A)

| Topology | HIC mean Δ | HIC median Δ | TmApp mean Δ | TmApp median Δ |
|----------|------------|--------------|--------------|----------------|
| JOINT | -0.0038 | -0.0017 | 0.0165 | 0.0097 |
| REG-SEP | -0.0022 | -0.0010 | 0.0053 | 0.0090 |
| XREG | -0.0038 | 0.0006 | -0.0311 | -0.0328 |
| FUSE | -0.0027 | -0.0015 | 0.0140 | 0.0302 |

Compare **directions/ranks of contrasts**, not absolute MAE.

## Annotation relative gains (vs BASE)

| Annotation | HIC mean Δ | TmApp mean Δ |
|------------|------------|--------------|
| IMGT | 0.0039 | 0.0296 |
| REGION | -0.0002 | -0.0036 |
| FULL | -0.0014 | 0.0041 |

## Inference context (PAIRED − SEPARATE)

- AbLang2: HIC mean Δ=0.0020; TmApp mean Δ=0.0389
- CurrAb: HIC mean Δ=0.0022; TmApp mean Δ=-0.0745

## Scientific takeaway

The same antibody sequence inputs do **not** yield identical relative representation / topology / annotation preferences for TmApp vs HIC. This supports treating developability targets as **distinct prediction problems** even under shared sequence featurization — without claiming mechanism.

## Non-interference

- Does not rewrite TmApp frozen conclusions
- Does not rewrite HIC scientific freeze
- No Public/Private

