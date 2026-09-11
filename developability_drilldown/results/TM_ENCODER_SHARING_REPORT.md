# TmApp encoder-sharing ablation (T130–T141)

Platform: `DL_FOLDLOCAL_COSINE_V3`. **Only** H/L `TransformerEncoder` stacks are untied.
Embeddings / REG / ARCH-7 `cross_attn` / merge / head remain shared. Encoders start identical.

| Rep | Arch | Merge | Shared | Unshared | Params_s | Params_u | Shared TEST_mean | Unshared TEST_mean | Δ | Shared worst | Unshared worst | Pub | Priv | Overall | Shared Overall |
|-----|------|-------|--------|----------|---------:|---------:|-----------------:|-------------------:|---:|-------------:|---------------:|----:|-----:|--------:|---------------:|
| ABLINGUA | ARCH-1 | concat | EXP-T075 | EXP-T130 | 501633 | 766593 | 3.4020 | 3.4836 | +0.0815 | 3.5121 | 3.5184 | 3.4542 | 3.2533 | 3.3537 | 3.3831 |
| ABLINGUA | ARCH-1 | mean | EXP-T080 | EXP-T131 | 476033 | 750209 | 3.2835 | 3.3446 | +0.0612 | 3.3067 | 3.3681 | 3.4552 | 3.1409 | 3.2981 | 3.4562 |
| ABLINGUA | ARCH-7 | concat | EXP-T086 | EXP-T132 | 558465 | 832641 | 3.3558 | 3.3282 | −0.0276 | 3.3735 | 3.3877 | 3.5209 | 3.2532 | 3.3871 | 3.3832 |
| ABLINGUA | ARCH-7 | mean | EXP-T087 | EXP-T133 | 542081 | 816257 | 3.2559 | 3.3246 | +0.0687 | 3.2764 | 3.3607 | 3.4713 | 3.1432 | 3.3072 | 3.3887 |
| SCRATCH | ARCH-1 | concat | EXP-T090 | EXP-T134 | 331265 | 605441 | 3.5504 | 3.3135 | **−0.2369** | 3.6792 | 3.3918 | 3.7141 | 3.4695 | 3.5918 | 3.5830 |
| SCRATCH | ARCH-1 | mean | EXP-T091 | EXP-T135 | 314881 | 589057 | 3.4825 | 3.2854 | **−0.1971** | 3.5995 | 3.3242 | 3.6165 | 3.3106 | 3.4635 | 3.5227 |
| SCRATCH | ARCH-7 | concat | EXP-T101 | EXP-T136 | 397313 | 671489 | 3.3409 | 3.2997 | −0.0412 | 3.4915 | 3.3852 | 3.5238 | 3.4642 | 3.4940 | 3.4149 |
| SCRATCH | ARCH-7 | mean | EXP-T102 | EXP-T137 | 380929 | 655105 | 3.3520 | 3.2919 | −0.0600 | 3.4288 | 3.3423 | 3.5655 | 3.3410 | 3.4533 | 3.5722 |
| ABLANG2 | ARCH-1 | concat | EXP-T109 | EXP-T138 | 390017 | 664193 | 3.2518 | 3.3024 | +0.0507 | 3.3070 | 3.3474 | 3.2012 | 2.9679 | 3.0845 | 3.1616 |
| ABLANG2 | ARCH-1 | mean | EXP-T110 | EXP-T139 | 373633 | 647809 | 3.2416 | 3.3236 | +0.0820 | 3.2803 | 3.3799 | 3.3063 | 2.9784 | 3.1423 | 3.1585 |
| ABLANG2 | ARCH-7 | concat | EXP-T120 | EXP-T140 | 456065 | 730241 | 3.1999 | 3.3516 | +0.1516 | 3.2912 | 3.3823 | 3.2830 | 3.0992 | 3.1911 | 3.1137 |
| ABLANG2 | ARCH-7 | mean | EXP-T121 | EXP-T141 | 439681 | 713857 | **3.1377** | 3.2577 | +0.1200 | 3.2521 | 3.2733 | 3.2790 | 3.1101 | 3.1945 | **3.2205** |

Δ = MAE_unshared − MAE_shared (negative ⇒ unshared better). External columns are unshared Primary-mean unless labeled Shared Overall.

## Scientific verdicts

- Pairs with Δ<0: **5/12**; Δ>0: **7/12**; mean Δ ≈ **+0.004** (near zero overall).
- By representation mean Δ: **Scratch −0.134** (helps) · AbLingua +0.046 · **AbLang2 +0.101** (hurts).
- By architecture mean Δ: ARCH-1 −0.026 · ARCH-7 +0.035.
- By merge mean Δ: CONCAT −0.004 · MEAN +0.012 (no strong merge interaction).
- Mean H/L encoder normalized L2 divergence ≈ **0.20** (init = 0) → specialization occurs in weight space.

### Answers

**A. Is shared encoder beneficial?**  
**Often yes for PLM backbones** (AbLang2 / AbLingua): unsharing usually worsens TEST_mean. Shared encoder acts as useful small-data regularizer when residue features are strong.

**B. Does full H/L specialization improve TmApp?**  
**Only for Scratch**, especially ARCH-1 (Δ ≈ −0.20 to −0.24). Not for the current AbLang2 best cluster (T113/T121 family controls).

**C. Representation-dependent?**  
**Yes — strongly.** Scratch benefits; AbLang2 is hurt most; AbLingua mixed/slightly hurt.

**D. ARCH-1 vs ARCH-7?**  
Scratch gains are larger on ARCH-1 than ARCH-7. For PLMs, ARCH-7 unsharing is not helpful (AbLang2 ARCH-7 Δ ≈ +0.12–0.15).

**E. Merge interaction?**  
Weak. CONCAT vs MEAN does not flip the sharing conclusion within a representation.

**F. Do encoders diverge?**  
**Yes.** Mean normalized ‖θ_H−θ_L‖ ≈ 0.20 across selected checkpoints.

**G. Capacity confound?**  
Unsharing adds ≈264960 encoder params. Prior T124–T129 generic capacity increases did **not** help AbLang2 — so Scratch gains are less likely to be “just more params,” but the confound is **not eliminated**. AbLang2 degradation despite extra capacity further argues against a pure capacity story for PLMs.

## Best / near-best context

- Best shared internal remains **T121** (TEST_mean ≈ 3.138) / **T113** cluster (external).
- Best unshared internal among this batch: Scratch T135/T137 (~3.29) — still behind AbLang2 shared baselines.
- Unshared AbLang2 does **not** displace T113/T121.

## Partial sharing

Scratch is a **robust matched family** with material unsharing gains → **partial sharing remains a justified future option for Scratch-like settings** (e.g. shared lower layers / chain-specific FFN).  
For the current AbLang2 flagship path: **do not pursue partial sharing**; treat full unsharing as **closed for PLM TmApp**.

Do **not** implement partial sharing in this batch.

## Artifacts

- `T130_T141_ENCODER_SHARING_PREREGISTRATION.yaml`
- `T130_T141_PRE_EXTERNAL_FREEZE.yaml`
- `T130_T141_ENCODER_SHARING_BOOTSTRAP.csv`
- `T130_T141_ENCODER_DIVERGENCE.csv`
- `T130_T141_EXTERNAL_FOUR_WAY.csv`
- `T130_T141_ENCODER_SHARING_IMPLEMENTATION_AUDIT.md`
