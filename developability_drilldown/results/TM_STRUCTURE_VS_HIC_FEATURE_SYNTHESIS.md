# Tm structure vs HIC feature synthesis

Working hypothesis under frozen `DL_FOLDLOCAL_COSINE_V3`:

- **TmApp**: gains from residue representation + architecture/capacity
- **HIC**: gains from explicit physicochemical / surface features

## A. Did increasing Transformer capacity help Tm?

**No.** T124–T129 all worse than T113/T121 on TEST_mean (~3.177–3.241 vs ~3.138). Depth and WIDE package do not help; DEEP_WIDE trends harmful. Stop Tm structural expansion.

## B. Did physical feature fusion help HIC?

**Yes.** Late fusion of H047-derived SURFACE (and F4) improves DL HIC on all three backbones, with external Overall moving from ~0.47–0.48 toward/below historical H047 (~0.427). LOCAL_RASA alone does not help; SEQUENCE_TITRATION alone is weak.

## C. Contrast hypothesis

**Partially supported — asymmetric.**

| Target | Structure/capacity | Explicit features |
|--------|--------------------|-------------------|
| TmApp | Capacity refinement failed; prior architecture/representation still the lever | Not tested this batch |
| HIC | Broad architecture search plateaued ~0.50 | Feature fusion is the successful lever |

So: “HIC = feature-information problem” is **supported** in this batch. “Tm = model-structure problem” is **only partially supported** historically (AbLang2 ARCH-3/7 cluster); further capacity structure search is **rejected** here.

Do not over-claim a single best model; report clusters + paired CIs + external concordance.
