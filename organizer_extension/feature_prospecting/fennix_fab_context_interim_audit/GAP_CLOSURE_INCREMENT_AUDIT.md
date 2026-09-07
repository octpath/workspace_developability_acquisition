# Gap Closure — Increment Evaluation Audit

**Date:** 2026-09-07  
**Scope:** Clarify what prior Gap Closure “no increment / NO_SIGNAL” claims meant.  
**Code audited:** `structure_gap_closure/scripts/eval_gap_closure.py`  
**Scorecard:** `structure_gap_closure/results/GAP_CLOSURE_SCORECARD.csv`

---

## Global procedure (all families)

| Question | Answer |
|----------|--------|
| Exact Primary / Shadow folds? | **Yes** — `virtual_participant/stage0_cv/cv_primary.csv` and `cv_shadow.csv` |
| Incumbent frozen? | **Yes** — OOF from `TmApp__META_performance__ridge_100.0` / `HIC__SIMPLE_blend_seq_surf_adv` (`y_pred` used as frozen reference; not re-fit) |
| Fold-local StandardScaler? | **Yes** — fit on outer-train fold only inside nested Ridge |
| Fold-local alpha selection? | **Yes** — alphas `{0.1,1,10,100}` on inner CV over training folds |
| PCA? | **Not used** for Gap Closure structural families (low-dim Ridge only) |
| Leakage-safe? | **Mostly yes** for incremental modes (see caveats) |

### Modes actually run

1. **`standalone`** — nested Ridge on structural features vs **fold-local median baseline** (not vs incumbent).  
2. **`incr_incumbent` / `incr_ablang2` / `incr_esm2` / `incr_esm2_aro`** — **feature concat**: structural matrix + **one frozen OOF scalar column** `__ref__`, then nested Ridge on *y*. Compared to the frozen OOF alone.  
3. **`residual`** — nested Ridge predicts `(y − frozen_OOF)` from structural features; candidate = `OOF + residual_pred`. Compared to frozen OOF.

This is **not** “standalone MAE vs incumbent MAE as a horse-race only.”  
It **is** leakage-safe **incremental combination** with a frozen OOF (scalar stack / residual), matching the project’s intended Ridge+OOF pattern — **not** a full retrain of the incumbent’s internal feature set.

### Caveats (honesty)

- Global `prep_X` median-impute + constant-column drop was applied **before** CV (mild preprocessing leakage).  
- Incremental concat uses **incumbent OOF scalar**, not the incumbent’s underlying multi-feature representation.  
- Scorecard rows were built from **incremental/residual modes only**; standalone lived in `GAP_CLOSURE_EVAL_RAW.csv`.

---

## Per-family audit

### TmApp / PACKING_CAVITY

| Item | Detail |
|------|--------|
| Standalone model? | Yes (vs median) |
| Simple score vs incumbent only? | **No** |
| Feature concat with incumbent? | **Yes** (`incr_incumbent`: features + frozen OOF) |
| Residual prediction? | **Yes** |
| OOF linear stack (2-col)? | Equivalent to concat with OOF scalar + phys features in one Ridge (not separate phys-OOF then 2-col stack) |
| Leakage-safe incremental? | **Yes** (with prep_X caveat) |
| Prior scorecard label | `NO_SIGNAL` on incr/residual |
| Correct vocabulary | **`NO_INCREMENT`** is allowed for incr/residual (combination was performed). Standalone-only claim must be labeled separately if used. |

### TmApp / BURIED_UNSAT

Same incremental framework as above.  
Standalone vs median showed **Primary+Shadow improvement** (later bootstrapped); **incr/residual vs incumbent did not** → incremental claim = **`NO_INCREMENT`**; standalone = possible **`STANDALONE_WEAK_ADVANTAGE`** (not “NO_INCREMENT”).

### TmApp / FAB_INTERFACE

Same as PACKING_CAVITY → incremental **`NO_INCREMENT`** valid.

### HIC / CONT_SURFACE_FV_* and DELTA_PATCH_CONTEXT

| Item | Detail |
|------|--------|
| Incremental vs ESM2-H OOF? | Yes (`incr_esm2`) |
| Incremental vs ESM2-H + AROMATIC OOF blend? | Yes (`incr_esm2_aro`: 0.5·ESM2 + 0.5·ARO nested OOF) |
| Incremental vs HIC incumbent OOF? | Yes |
| Residual vs incumbent? | Yes |
| Prior label | `NO_SIGNAL` |
| Correct vocabulary | **`NO_INCREMENT`** for those combination modes |

---

## Relabeling rule applied going forward

- **`NO_INCREMENT`**: leakage-safe combination (concat and/or residual) with frozen incumbent/reference OOF did not improve MAE on Primary **and** Shadow.  
- **`STANDALONE_NO_ADVANTAGE`**: only if the claim was based on standalone vs incumbent horse-race **without** combination — **not** what the Gap Closure scorecard incremental rows were.  
- Prior scorecard `NO_SIGNAL` on incremental rows ≡ **`NO_INCREMENT`** under this audit.

---

## Combination closure (Part 2)

See `GAP_CLOSURE_COMBINATION_CLOSURE.csv`: predeclared aggregates `CORE_DEFECT`, `FAB_INTERFACE`, `GAP_ALL`, `HIC_SURFACE_ALL` evaluated with **both** Ridge-concat and residual frameworks; neither is post-hoc selected as “the” result.

---

## Strict nested follow-up (2026-09-07)

Prior claim of leakage-safe incremental **overstated**: global incumbent OOF as a feature inside a second outer CV causes **cross-fold contamination** on outer-train samples. See `STRICT_NESTED_CONTAMINATION_AUDIT.md` and `STRICT_NESTED_INCREMENT_RESULTS.csv`. Final NO_INCREMENT must be read from `STRICT_NESTED_INCREMENT`.
