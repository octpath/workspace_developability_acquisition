# EXP-T073 Pre-implementation Audit

**Code free:** EXP-T073 (next unused = EXP-T073).  
**Protocol id:** `DL_FOLD_LOCAL_LR_CHECKPOINT_ENSEMBLE_V2`

## A. T030 scientific architecture (confirmed in code)

| Item | Value |
|------|-------|
| Content | AbLingua frozen PLM → `Linear(plm→d_model)` |
| Annotation | FULL: seq pos + chain + IMGT + region |
| Encoder | shared `TransformerEncoder`, H and L **separate** calls |
| REG | learned `REG_H`, `REG_L` (+ chain emb) |
| Readout | `concat(REG_H, REG_L)` → `2*d_model` |
| Head | `Linear(256→128)→GELU→Dropout→Linear(128→1)` |
| Joint / cross / RASA / CA | **off** for T030 |

## B. Architecture hyperparameters

| Item | Value |
|------|-------|
| d_model | 128 |
| n_layers | 2 (hard-frozen) |
| n_heads | 4 |
| FFN | 256 |
| dropout | 0.20 |
| norm_first | True |
| activation | gelu |
| batch_first | True |

## C. Historical / REPLAY training configuration

| Item | Old T030 / REPLAY |
|------|-------------------|
| optimizer | AdamW |
| lr | **3e-4** (= `lr_ref`) |
| weight_decay | 1e-2 |
| loss | SmoothL1(β=0.5) on z-scored y |
| batch_size | 16 |
| max_epochs | 300 |
| patience | 30 |
| scheduler | **none** |
| AMP | none |
| grad clip | 1.0 |
| seeds | **[101, 202, 303]** — seed **101** is authoritative |

## D. Split semantics

`tvt_split(fold_map, k, ids)`:

- TEST = fold == k  
- VAL = fold == (k+1) % 5  
- TRAIN = remaining  

Primary and Shadow are independent fold maps (5 rotations each).

## E. Old final prediction protocol

1. **Phase A:** train on TRAIN, early-stop on VAL → keep `best_epoch` only (weights discarded)  
2. **Phase B:** reinit; train TRAIN+VAL for exactly `best_epoch` → predict fold TEST (OOF)  
3. **full-Dev refit:** median Primary `best_epoch` per seed; train all DEV; average 3 seeds for external Test  

**EXP-T073 deliberately abandons Phase B and full-Dev refit.**  
Selected VAL checkpoint from TRAIN-only training is the fold model.

## F. EXP-T073 deltas (protocol only)

- Architecture: identical to T030 scientific model  
- Seed: **101 only**  
- max_epochs **200**, patience **20**  
- CosineAnnealingLR(T_max=200, eta_min=0)  
- Fold-local LR grid: `{0.1, 0.3, 1.0, 3.0} × 3e-4`  
- No nested CV; no full-Dev refit  
- External Test = mean/median of 5 fold checkpoints  

## Verdict

Assumptions hold → proceed to implement protocol V2 baseline as EXP-T073.
