# New research launch brief

Classical feature refinement is closed. This brief selects **one** first architecture-phase experiment from CV/ablation evidence only (Public/Private not used for selection).

**Do not execute yet** — review this brief first.

---

## TmApp

### 1. Canonical best singles (eligible only)

| metric | code | family | value |
|---|---|---|---:|
| CV Primary | EXP-T003 | LINEAR (REPRODUCED) | 2.7027 |
| CV Shadow | EXP-T045 | LINEAR (REPRODUCED) | 2.7289 |
| CV worst | EXP-T045 | LINEAR (REPRODUCED) | 2.7289 |
| Public | EXP-T045 | LINEAR | 3.0197 |
| Private | EXP-T058 | LINEAR | 3.0292 |
| Overall | EXP-T048 | LINEAR | 3.0722 |

Best eligible **Transformer** by CV-worst: **EXP-T037** (AbLingua FULL concat fusion + BioEmu/MPNN) W≈**2.7730**.

### 2. Best classical-refinement result

**EXP-T045** — BioEmu+MPNN + AbLingua CDR3 + AbLang2 RASA/CDR + AbLingua CDR_ALL, Ridge  
CV P/S/W ≈ 2.706 / 2.729 / **2.729**

### 3–10. Classical findings (CV-driven)

3. **AbLang2 region-aware pooling**: yes — FR/CDR split / CDR blocks improved Stage A/B when added to strong bases.  
4. **AbLingua region-aware pooling**: yes — CDR3 / CDR_ALL blocks on BioEmu and AbLingua bases.  
5. **CDR weighting**: mixed; useful in combo paths, not universal alone.  
6. **CDR3 weighting**: yes — FB_AL_CDR3 / related blocks accepted in forward selection.  
7. **RASA weighting**: yes — AbLang2 RASA_P1/CDR improved add-to-base; present in best set (FB_AL2_RASA_CDR).  
8. **Fixed AbLang2+AbLingua concat**: no — did not beat top multi-block CV-worst paths.  
9. **BioEmu/MPNN**: yes — remained essential anchor of the winning combo.  
10. **Ablation survivors (top BioEmu path)**: FB_AL_CDR3, FB_AL2_RASA_CDR, FB_AL_CDR_ALL each increase CV-worst when dropped (all contribute).

Estimator on top Tm set: **Ridge** beat SVR/XGB/Lasso/ENet on CV-worst.

---

## HIC

### 1. Canonical best singles (eligible)

| metric | code | family | value |
|---|---|---|---:|
| CV Primary | EXP-H033 | TRANSFORMER | 0.4345 |
| CV Shadow | EXP-H030 | TRANSFORMER | 0.4413 |
| CV worst | EXP-H030 | TRANSFORMER | **0.4422** |
| Public | EXP-H030 | TRANSFORMER | 0.3976 |
| Private | EXP-H042 | LINEAR | 0.4351 |
| Overall | EXP-H047 | LINEAR | 0.4275 |

### 2. Best classical-refinement result

**EXP-H047** — Hydro titration + ESM2 RASA CDR3, **SVR**  
CV P/S/W ≈ 0.453 / 0.444 / **0.453** (still behind Transformer CV-worst ≈0.442)

### 3–10. Classical findings

3. **ESM2 region-aware pooling**: yes vs global alone.  
4. **H-CDR / H-CDR3**: yes — FB_ESM2_CDR3 / RASA_CDR3 helped on hydro/surface bases.  
5. **RASA weighting**: yes — central to best HIC classical path.  
6. **Exposed aromatic/RASA summaries**: secondary; dominated by ESM2 RASA/CDR3.  
7. **Region-specific aromatic exposure**: did not beat ESM2 RASA/CDR3 on CV-worst.  
8. **Continuous surface / hydro**: complementary bases; best used hydro + ESM2 RASA CDR3.  
9. **Ablation**: dropping FB_ESM2_RASA_CDR3 from hydro path reverts toward base CV-worst.  
10. **Estimator**: **SVR** best on top HIC classical set; Ridge/Lasso/ENet/XGB worse on CV-worst.

---

## Recommended FIRST new research experiment

### Scientific question

Does **residue-level RASA weighting of AbLingua token inputs** improve TmApp CV-worst versus the same Transformer without RASA weights?

### Why this (and not a broad sweep)

- Classical Stage A–E showed **RASA-weighted PLM pooling** consistently helped Tm (and HIC).  
- Best classical Tm set includes **FB_AL2_RASA_CDR**; ablation shows it matters.  
- Best eligible Tm Transformer (**EXP-T037**) uses AbLingua FULL residue stream + BioEmu/MPNN fusion but **no RASA input weighting**.  
- Isolates one mechanism already validated in fixed-length features before dual-PLM or distance-bias work.  
- Selection uses **CV-worst / ablation** only (not Public/Private).

### Spec (ablation-style; do not run yet)

| field | value |
|---|---|
| target | **TmApp** |
| closest historical control | **EXP-T037** (AbLingua FULL concat fusion + BioEmu/MPNN) |
| one architectural difference | Apply per-residue **RASA gate/weight** (ESMFold Shrake–Rupley, exposed threshold **0.20**, same as classical) to AbLingua residue embeddings **before** the existing Transformer encoder; leave d_model/layers/heads/fusion/recipe/protocol unchanged |
| expected family | **TRANSFORMER** |
| next EXP code (from registry) | **EXP-T065** (do not issue until execution is approved) |
| primary success criterion | CV-worst MAE **strictly better** than EXP-T037 (≈2.7730) under identical primary/shadow folds |
| secondary diagnostics | CV Primary/Shadow; compare to EXP-T045 classical W≈2.729 as soft ceiling; attention/weight sanity on CDR vs FR; no Public/Private for selection |

### Explicitly deferred (not first)

- Dual AbLang2+AbLingua residue input (classical fixed concat was weak; need RASA lesson first)  
- Joint H/L single-REG  
- Distance / 3D attention bias  
- HIC-first architecture (Transformer already leads HIC CV-worst; Tm has clearer classical→architecture RASA story)

---

## Flags

- CLASSICAL_FEATURE_REFINEMENT_CLOSED = YES  
- NEW_ARCHITECTURE_READY = YES  
- **Execution of EXP-T065: STOP until this brief is reviewed**
