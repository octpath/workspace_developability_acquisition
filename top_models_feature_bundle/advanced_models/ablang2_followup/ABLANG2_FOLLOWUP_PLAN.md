# AbLang2 position-aware Transformer follow-up — FROZEN PLAN

**Status:** PLAN_FROZEN  
**Frozen at:** 2026-09-09T02:25:38Z  
**Baseline HEAD:** `3bff7a5ce27155feec109a464ed9df6f4a43bd9e`  
**Target:** TmApp ONLY (no new HIC AbLang2 exploration)

## Scientific hypothesis

AbLang2 residue representations, re-aggregated by a small Transformer that explicitly uses IMGT position and FR/CDR region annotations, may recover position-dependent TmApp signal lost in ordinary whole-chain / pooled AbLang2 representations.

Separately evaluate:

1. AbLang2 residue content (vs AbLingua TMF*)
2. Explicit antibody-position annotation (AL2F1 minimal → AL2F2 full)
3. Region-aware pooling (AL2F4 gate weights; descriptive only, not causal)

## Exact AbLang2 identity (do not swap)

- Package: `ablang2` (pip)
- API entry: `ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ...)`
- Same checkpoint family as Stage-2 / bundle AbLang2 seqcoding (`ablang2-paired`)
- Residue API: official `mode="rescoding"` → `AbRep(...).last_hidden_states` then AA-token slice
- Special tokens in formatted string: `<`, `>`, `|` (tokenizer AA vocab)
- Chain extraction: H-only `[[heavy, ""]]`, L-only `[[ "", light]]` (matches Stage-2 H/L caches)
- Frozen backbone: no AbLang2 fine-tuning

## Model variants (pre-registered; do not alter after results)

| ID | content | annotation | merge | pooling |
|----|---------|------------|-------|---------|
| AL2F1_MIN_CONCAT | frozen AbLang2 residue | minimal (seq pos + chain) | concat | [REG] |
| AL2F2_FULL_CONCAT | frozen AbLang2 residue | full (IMGT + FR/CDR) | concat | [REG] |
| AL2F3_FULL_MEAN | frozen AbLang2 residue | full | mean | [REG] |
| AL2F4_FULL_REGION_GATE | frozen AbLang2 residue | full | concat | region gate (FR_ALL, CDR1–3) |

Architecture (match advanced suite): Linear(plm→128) + embeddings → TransformerEncoder 2× / d=128 / nhead=4 / FF=256 / dropout=0.2 / norm_first / batch_first. Shared H/L weights + chain embedding.

Region gate: global learnable logits (not sample-dependent attention); missing regions masked + renormalize.

## Seeds / CV / training

- Seeds: 101, 202, 303
- CV: frozen Primary + Shadow Simple TVT, 5 rotations (TEST=k, VAL=(k+1)%5, TRAIN=rest)
- Protocol: Phase A (TRAIN, VAL early stop) → Phase B (TRAIN+VAL exactly best_epoch) → Phase C TEST
- Units: 2×5×3 = 30 / variant; 4 variants = 120 sequence units
- Hyperparameters: AdamW lr=3e-4, wd=1e-2, bs=16, max_epochs=300, patience=30, grad_clip=1.0, SmoothL1, target std TRAIN-only
- OOM only: bs 16→8→4 (record in `runtime_adaptations.json`)
- No Optuna / no post-hoc architecture or LR changes

## Selection rule (CV only)

- Rank by `cv_worst = max(Primary, Shadow)` (min)
- Tie-break: `cv_mean`; then simpler model
- **Public/Private forbidden** for any selection (POSTMORTEM ONLY after freeze)

## Fusion recipes (after BEST_ABLANG2_TRANSFORMER freeze)

1. BEST ⊕ TM_PARENT_ABLINGUA_CDR3  
2. BEST ⊕ TM_PARENT_ABLINGUA_GLOBAL  
3. BEST ⊕ TM_BASE_BIOEMU_MPNN  

Same fusion architecture and TRAIN-only fixed preprocessing as advanced suite.

## Ensemble

Reuse family-winner → equal-mean all non-empty subsets → cv_worst → cv_mean → fewer members.  
Stacking: SKIP unless safe unified OOF appears (default SKIP).

## Postmortem policy

Read `solution.csv` only after `ABLANG2_FINAL_CV_SELECTION.json` freeze. Never track or copy solution into git.

## Stopping rule

Run the pre-registered matrix once (with resume). No result-driven extra models.

## License

AbLang2 code/weights/derived residue outputs: **REVIEW_MODEL_OUTPUT** (no redistribution claim).
