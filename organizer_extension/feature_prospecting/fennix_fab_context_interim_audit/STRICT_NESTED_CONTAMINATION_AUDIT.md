# Strict Nested Increment — Cross-Fold Contamination Audit

**Date:** 2026-09-07  
**Scope:** Gap Closure combination + Interim FeNNix incremental frameworks  
**Code audited:** `eval_gap_closure.py`, `run_interim_audit.py` (`concat_inc` / `residual_on`)

---

## Finding

**`CROSS_FOLD_CONTAMINATION = YES`** for `OLD_GLOBAL_OOF_INCREMENT`.

Both Gap Closure and Interim FeNNix incremental modes inject a **globally precomputed 5-fold incumbent OOF scalar** (`TmApp__META_performance__ridge_100.0` / `HIC__SIMPLE_blend_seq_surf_adv` `y_pred`) into a **second** outer CV (as a concat column or residual base).

### Why this contaminates

Global OOF construction for sample \(i\) in fold \(j\):

- base/meta model is trained on all folds **except** \(j\).

When the incremental evaluator later holds out outer fold \(k\):

| Role | What it receives | Contamination? |
|------|------------------|----------------|
| Outer-validation (\(k\)) | Global OOF\(_k\) (trained excluding \(k\)) | OK as a held-out prediction *if used alone* |
| Outer-training (\(j \neq k\)) | Global OOF\(_j\) trained excluding \(j\) but **including fold \(k\)** | **YES** — train-side incumbent features / residual targets saw outer-val labels |

Therefore:

1. Fitting a structural Ridge (concat or residual) on outer-train that uses global OOF as a feature/target **leaks outer-validation information into the meta-model**.
2. Comparing against global OOF MAE does **not** cancel this; the candidate path is optimistic/biased relative to a true nested protocol.

A globally precomputed OOF vector is **not** sufficient for leakage-safe incremental claims under a second outer CV.

---

## Required protocol (`STRICT_NESTED_INCREMENT`)

For each outer fold \(k\):

1. Split outer-train / outer-validation using frozen Primary or Shadow folds.
2. Build incumbent predictions for **outer-train** via **inner CV using only outer-train**.
3. Fit structural residual / concat Ridge using only those clean outer-train incumbent predictions.
4. Fit incumbent on **full outer-train** and predict **outer-validation**.
5. Combine (concat or residual) and score outer-validation.
6. Aggregate over folds.

Incumbent reconstruction reuses Stage5 frozen base configs + precomputed feature matrices / PLM embeddings (no new PLM embedding generation; no hyperparameter search beyond frozen choices).

- **TmApp:** performance 4-base nest → Ridge \(\alpha=100\) meta (same as `TmApp__META_performance__ridge_100.0`).
- **HIC:** equal mean of 3 bases (same as `HIC__SIMPLE_blend_seq_surf_adv`).

---

## Deliverables

| Artifact | Role |
|----------|------|
| `STRICT_NESTED_INCREMENT_RESULTS.csv` | Side-by-side `OLD_GLOBAL_OOF_INCREMENT` vs `STRICT_NESTED_INCREMENT` |
| `STRICT_NESTED_INCREMENT_REPORT_JA.md` | Japanese summary + direction labels |
| `scripts/run_strict_nested_increment.py` | Evaluator (FeNNix worker untouched) |
