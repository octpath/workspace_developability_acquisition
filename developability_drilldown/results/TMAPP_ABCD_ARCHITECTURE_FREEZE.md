# TmApp A/B/C*/D* Architecture Freeze

**Status:** FROZEN after T142–T150 (AbLang2 screen only).
**Git:** `ea2449715f3b3726524cc6a103ad01bd1245929a`
**Next unused code:** `EXP-T151` (do not run PLM matrix yet).

| Slot | Code | Definition |
|------|------|------------|
| **A** | EXP-T110 | Separate H/L + dual REG, mean merge |
| **B** | EXP-T113 | ARCH-3 joint unrestricted + dual REG, mean |
| **C\*** | EXP-T121 (C0) | No new C showed convincing incremental value over C0/T121. |
| **D\*** | EXP-T149 (D3) | D4 capacity flagged; selected simpler D by mean/worst/params (D4 mean gain vs A not large enough to justify +66k params). |

Common: frozen AbLang2 residue embeddings, FULL annotation, `share_hl_encoder=True`, d_model=128, V3 folds/seed, DL_FOLDLOCAL_COSINE_V3.

Machine-readable: `results/TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml`
