# Implementation integrity

- ID uniqueness: PASS
- role partition 162/81/81: PASS
- hidden↔population id join for TmApp: PASS (exact)
- no positional-only dependency required: PASS
- Clean-core Public→Private ρ (recomputed preds) = -0.433
- Clean-core CV→Private ρ = 0.783
- Clean-core CV→Public ρ = -0.100
- NOTE score drift PLM_ABLANG2_PCA64/SVR_RBF: dPub=0.059 dPriv=-0.025
- NOTE score drift PLM_ABLANG2_PCA32/SVR_RBF: dPub=0.018 dPriv=-0.067
- Recomputed vs stored agreement: PASS (within tol)

## Proxy finalist issue

In `gate_b3/scripts/05_finalists_oneshot.py`, tags starting with `ENSEMBLE` or `RESID_` were fit using a **proxy base** (`PLM_ESM2` for TmApp) instead of true ensemble/residual reconstruction. Affected TmApp rows: `RESID_BIO_SHORTCUT__PLM_ESM2`, `ENSEMBLE_NNLS` (identical Public/Private scores). True Train-OOF ensemble weights were not persisted for exact one-shot rebuild; these rows are **excluded** from CLEAN CORE.

## CLEAN CORE transfer (n=9)

- CV → Public ρ = **-0.100** (Kendall τ=0.000)
- Public → Private ρ = **-0.433** (Kendall τ=-0.278)
- CV → Private ρ = **0.783** (Kendall τ=0.611)

Anomaly **reproduced** on clean independently predicted finalists.
