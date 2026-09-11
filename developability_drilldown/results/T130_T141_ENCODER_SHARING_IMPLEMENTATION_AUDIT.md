# T130–T141 encoder-sharing implementation audit

**HEAD at audit:** recorded in preregistration YAML.  
**Platform:** `DL_FOLDLOCAL_COSINE_V3` (frozen).  
**Scientific variable:** Transformer encoder weight sharing between Heavy and Light only.

---

## 1. Control verification (STOP check)

All 12 shared controls match canonical configs (material agreement):

| Family | ARCH-1 CONCAT | ARCH-1 MEAN | ARCH-7 CONCAT | ARCH-7 MEAN |
|--------|---------------|-------------|---------------|-------------|
| AbLingua | EXP-T075 | EXP-T080 | EXP-T086 | EXP-T087 |
| Scratch | EXP-T090 | EXP-T091 | EXP-T101 | EXP-T102 |
| AbLang2 | EXP-T109 | EXP-T110 | EXP-T120 | EXP-T121 |

Common size: d_model=128, layers=2, heads=4, FFN=256, dropout=0.2, GELU, `norm_first=True`.

Note: EXP-T075 lacks explicit `arch_id` in YAML but defaults to ARCH-1 separate dual-REG CONCAT (historical baseline).

**No STOP discrepancy.**

---

## 2. What is shared today (before this batch)

Authoritative: `models/antibody_transformer/model.py` (`AnnotatedTransformer`).

| Module | Shared across H/L? | Notes |
|--------|--------------------|-------|
| `aa_emb` (Scratch) | YES | single Embedding |
| `plm_proj` (frozen PLM) | YES | single Linear |
| `pos_emb` / `imgt_emb` / `region_emb` | YES | single tables |
| `chain_emb` | one module, row 0=H / 1=L | not duplicated |
| `reg_token` | one Parameter `[2,d]` | rows are H/L-specific values, not a second encoder |
| **`TransformerEncoder`** | **YES (historical)** | `self.encoder` applied to both chains |
| `cross_attn` (ARCH-7) | YES | single `nn.MultiheadAttention` for both directions |
| merge / `head` | YES | after dual REG |

ARCH-1 path: `encode_chain` → shared encoder → dual REG merge.  
ARCH-7 path: per-chain encode with shared encoder → REG-only cross-attn → merge.

---

## 3. Unshared definition (this batch)

Flag: `share_hl_encoder=False` (`arch_share_hl_encoder: false` in YAML).

Creates:

- `encoder_h = TransformerEncoder(...)`
- `encoder_l = TransformerEncoder(...)`
- `encoder_l.load_state_dict(deepcopy(encoder_h.state_dict()))` at init

Helper: `_encoder_for_chain(chain_idx)`.

**Not untied:** embeddings, REG tokens, ARCH-7 `cross_attn`, merge, head.

**Forbidden with unsharing:** joint ARCH-2/3/4, mid-bridge ARCH-5/6/8, CA-distance bias.

---

## 4. Initial equivalence QC

Unit tests: `tests/test_encoder_sharing.py`

- init hash(encoder_H) == hash(encoder_L)
- Parameter objects distinct (`data_ptr` differs)
- After aligning non-encoder + encoder weights to a shared twin: max |Δpred| ≤ 1e-6 (ARCH-1 and ARCH-7)
- After asymmetric update on encoder_H only: hashes diverge
- Param count: unshared encoder = 2 × shared encoder; non-encoder unchanged

---

## 5. Code map

| New code | Control | Rep | Arch | Merge |
|----------|---------|-----|------|-------|
| T130 | T075 | AbLingua | ARCH-1 | CONCAT |
| T131 | T080 | AbLingua | ARCH-1 | MEAN |
| T132 | T086 | AbLingua | ARCH-7 | CONCAT |
| T133 | T087 | AbLingua | ARCH-7 | MEAN |
| T134 | T090 | Scratch | ARCH-1 | CONCAT |
| T135 | T091 | Scratch | ARCH-1 | MEAN |
| T136 | T101 | Scratch | ARCH-7 | CONCAT |
| T137 | T102 | Scratch | ARCH-7 | MEAN |
| T138 | T109 | AbLang2 | ARCH-1 | CONCAT |
| T139 | T110 | AbLang2 | ARCH-1 | MEAN |
| T140 | T120 | AbLang2 | ARCH-7 | CONCAT |
| T141 | T121 | AbLang2 | ARCH-7 | MEAN |

---

## 6. Capacity confound

Unsharing doubles encoder parameters (~+264960 for d=128/L=2).  
No parameter-matched control in this batch.  
Prior T124–T129 generic depth/width capacity increases did **not** help TmApp — context only; confound not mathematically eliminated.
