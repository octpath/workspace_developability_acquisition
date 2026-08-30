# Target comparison — HIC vs TmApp

## Overlap

- N antibodies with both labels: **324**
- Spearman(HIC, TmApp): **0.118**
- Targets are nearly independent on the overlap — dual scoring would not be redundant.

## Modeling headroom (mean over splits)

```
             plm_minus_bio  abb_minus_plm  best_overall
HIC               +0.262         +0.004         0.567
TmApp             +0.144         -0.170         0.499
```

## Modality fingerprint

| | HIC | TmApp |
|---|---|---|
| Best family | Native ESMFold surface / fusion | PLM (AbLang2 / ESM) |
| BIO_SHORTCUT risk | Low (BIO≪SEQ_SIMPLE) | High (BIO≈best PLM) |
| Structure residual on SEQ_SIMPLE | Strong (ESMFN ~0.45) | Weak |
| Leaderboard Pub→Priv fidelity | Moderate (0.59) | Better (0.73) |

## Competition-format implication

Weak target correlation + complementary modalities supports a dual track scientifically.
Default Gate B2 recommendation remains **HIC primary** (clearer non-shortcut physics/structure story) with **TmApp secondary** (or optional equal dual with BIO audits). See `GATE_B2_FINAL.md`.
