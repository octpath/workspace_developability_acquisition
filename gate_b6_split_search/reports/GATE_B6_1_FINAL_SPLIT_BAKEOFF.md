Recommended production split:

SELECT_CAND_12528

Second choice:

CAND_04974

Reason:

Priority-ordered bake-off (TmApp fixed → Public-selection regret both tracks → HIC usefulness → balance → local robustness) favors **CAND_12528**. TmApp Public→Private MAE rank transfer matches 04974/12207 at ρ=0.80, while Public-winner Private regret is lowest on **both** TmApp (0.186 °C) and HIC (0.0155 min). HIC CV→Private MAE ρ improves to 0.60 vs 04974’s 0.26. Trade-off: weaker HIC CV→Public (MAE ρ=0.49; Pearson ρ=−0.10) and slightly worse pre-model/local-stability scores than 04974 — not enough to outweigh lower participant-selection harm.

TmApp:
    Public-winner Private regret:
        04974: 0.2158 °C
        12528: 0.1862 °C
        12207: 0.2006 °C

    Public→Private MAE rank transfer:
        04974: 0.8000
        12528: 0.8000
        12207: 0.8000

    bootstrap robustness:
        Strategy A (Public-only), n=10000:
        04974: median=0.202, p95=0.717, P(regret>0.25°C)=0.427, P(>0.5°C)=0.151
        12528: median=0.162, p95=0.695, P(regret>0.25°C)=0.385, P(>0.5°C)=0.147
        12207: median=0.139, p95=0.598, P(regret>0.25°C)=0.346, P(>0.5°C)=0.098
        (12207 has the best TmApp bootstrap tail; 12528 best among HIC-safe splits.)

HIC:
    Public-winner Private regret:
        04974: 0.0235 min
        12528: 0.0155 min
        12207: 0.0246 min

    Public→Private MAE rank transfer:
        04974: 0.7714
        12528: 0.7714
        12207: 0.4857  (below B6 safety floor 0.50)

    bootstrap robustness:
        Strategy A, n=10000 (thresholds vs Test HIC SD≈0.857):
        04974: median=0.033, p95=0.077
        12528: median=0.022, p95=0.105
        12207: median=0.025, p95=0.090

Pre-model balance winner:

CAND_04974 (mean |SMD|≈0.083; 12528≈0.101; 12207≈0.093; baseline≈0.134)

Local-stability winner:

CAND_04974 (TmApp mae_pp median 0.80, P(ρ<0)=0.017; HIC pub-winner regret median 0.025).
CAND_12528 is essentially tied on TmApp mae_pp and better on HIC pub-winner regret under perturbation (median 0.018).

Any candidate with unacceptable participant harm:

NONE for TmApp. **CAND_12207** is CONCERNING for HIC Public→Private MAE transfer (ρ=0.486 < 0.50 floor) despite acceptable absolute Public-winner regret.

Permanent freeze recommended:

**YES — freeze CAND_12528** as the common HIC/TmApp production Public/Private mask (human approval still required before packaging).

---

# Gate B6.1 — Final Production Split Bake-Off

## Scope

Frozen candidates only (no new search, no retrain):

| ID | Role |
|----|------|
| CAND_04974 | B6 selected / strong Public feedback |
| CAND_12528 | Stronger HIC CV→Private |
| CAND_12207 | Putative compromise (failed B6 HIC floor) |
| CURRENT_BASELINE_SPLIT | Control only |

Config freeze: `config/B6_1_FINAL_CANDIDATE_SET.json`, `config/B6_1_ANALYSIS_PROTOCOL.json`.

## Model set (frozen B6)

**TmApp (5):** CONST_MEDIAN, SEQ_SIMPLE_Ridge, BIO_Ridge, PLM_ABLANG2_PCA32_SVR, NESTED_STACK_MEAN

**HIC (6):** CONST_MEDIAN, SEQ_SIMPLE_Ridge, PLM_ESM2_PCA64_SVR, ESMFN_STRUCTURE_ElasticNet, FUSION_ESM2_ESMFN_ElasticNet, NESTED_STACK_NNLS

Near-duplicates: HIC NESTED↔PLM_ESM2 r≈0.90; NESTED↔FUSION≈0.86; NESTED↔ESMFN≈0.86. TmApp NESTED↔PLM_ABLANG2≈0.80. CONST has undefined Pearson (constant).

## Central finding

Rank-ρ alone does **not** separate 04974 vs 12528 on TmApp (both Public→Private MAE ρ=0.80). **Public-winner Private regret** and bootstrap Strategy-A distributions do: **12528 ≤ 04974** on both targets. 04974’s weak HIC CV→Private ρ (0.26) is mostly small rank swaps (Public-winner regret only 0.0235 min; effect-weighted inversion burden low).

12207 does **not** redeem itself on effect sizes: HIC Public→Private MAE ρ stays below the production floor, and regrets are not better than 12528.

## Decision table (summary)

| Dimension | 04974 | 12528 | 12207 |
|-----------|-------|-------|-------|
| TmApp Public feedback | EXCELLENT | EXCELLENT | EXCELLENT |
| TmApp Public-selection regret | ACCEPTABLE | ACCEPTABLE (best) | ACCEPTABLE |
| TmApp bootstrap | ACCEPTABLE | ACCEPTABLE | GOOD (best tail) |
| HIC Public feedback | EXCELLENT | EXCELLENT | CONCERNING |
| HIC Public-selection regret | GOOD | EXCELLENT | GOOD |
| HIC CV→Private | ACCEPTABLE | GOOD | GOOD |
| Pre-model balance | EXCELLENT (best) | EXCELLENT | EXCELLENT |
| Local stability | EXCELLENT (best) | EXCELLENT | GOOD/EXCELLENT |

## Candidate-specific answers

### CAND_04974
1. Weak HIC CV→Private ρ is **not** practically catastrophic — Public-winner regret 0.0235 min.
2. Driven by **small MAE swaps**, not large score gaps (Public MAE range only ~0.15 min).
3. Trusting HIC Public → Private regret ≈ 0.0235 min (bootstrap median ≈ 0.033).
4. Advanced vs baseline distinctions largely preserved (Public Top-3 ∩ Private Top-3 = 3).
5. Locally robust (300 group-swaps; TmApp mae_pp median 0.80; P(ρ<0)=0.017).

### CAND_12528
1. Stronger HIC CV→Private **does** coincide with **lower** Public-winner regret (0.0155 vs 0.0235).
2. Negative HIC Pearson CV→Public (−0.10) is real but **secondary** under MAE-primary leaderboard.
3. Under MAE-facing Public, harm is limited; Public MAE CV→Public ρ=0.49 still positive.
4. Vs 04974 on score gaps: **yes, better compromise** on regret; slightly worse balance SMD.
5. Locally robust; HIC pub-winner regret under perturbation is the best of the three.

### CAND_12207
1. **Not** the best compromise once HIC floor + regrets are applied.
2. HIC PP ρ≈0.49 is **insufficient** for the pre-registered production floor.
3. Selection regret is middling — not better than 12528.
4. Balance is fine but not superior to 04974.
5. Local TmApp stability OK; HIC mae_pp median drops to ~0.57 under perturbation.

## Recommendation

```text
SELECT_CAND_12528
second_choice: CAND_04974
main_reason: lowest Public-driven participant regret on BOTH tracks; HIC CV→Private repaired without sacrificing TmApp ρ=0.80
main_weakness: HIC CV→Public weaker than 04974 (MAE ρ=0.49; Pearson ρ=−0.10); balance/local-stability slightly behind 04974
```

Do **not** auto-package. Human review should confirm acceptance of the HIC CV→Public trade-off.

---

# Answers to required questions (1–20)

1. Blind Public MAE winners: **04974/12528/12207 TmApp → NESTED_STACK_MEAN**; HIC → NESTED_STACK_NNLS / ESMFN_STRUCTURE / FUSION_ESM2_ESMFN respectively.
2. Private ranks: TmApp all **Private #2**; HIC **#3 / #3 / #4**.
3. Private MAE regret: TmApp **0.216 / 0.186 / 0.201 °C**; HIC **0.0235 / 0.0155 / 0.0246 min**.
4. CV-only: TmApp same as Public (NESTED) on all three; HIC CV picks NESTED → on 12528 that is Private #1 (regret 0).
5. CV-among-Public-Top2: identical to CV-only here (NESTED in Public Top2).
6. Minimizes TmApp Public-selection regret: **CAND_12528**.
7. Minimizes HIC Public-selection regret: **CAND_12528**.
8. 04974 weak HIC CV→Private: **small harmless rank swaps** (regret 0.0235 min).
9. 12528 negative HIC Pearson CV→Public: **limited practical harm** under MAE-primary Public.
10. 12207 better compromise after effect sizes? **No** — HIC PP floor miss + worse regrets than 12528.
11. Best TmApp bootstrap (Strategy A median/tail): **CAND_12207**; among HIC-safe: **CAND_12528**.
12. Best HIC bootstrap: **CAND_12528**.
13. Best pre-model balance: **CAND_04974**.
14. Most locally stable: **CAND_04974** (12528 close; better HIC regret under swaps).
15. Sequence-neighborhood easiness imbalance: **no strong signal** (max \|SMD\| nearest-Train VL ≈ 0.25 on 12528).
16. Single-model dependence: moderate Public-winner concentration on NESTED/ESMFN (P≈0.37–0.76); not unique to one candidate.
17. Least misleading for inexperienced Public-followers: **CAND_12528**.
18. Preferred by experienced CV-driven participant: **CAND_12528**.
19. Novice vs experienced answers differ? **NO**.
20. Permanently freeze: **CAND_12528** (common HIC/TmApp mask).

## Artifacts

- `metrics/per_model_candidate_scores.csv`
- `metrics/model_rank_transfer.csv`
- `metrics/public_selection_regret.csv`
- `metrics/participant_strategy_regret.csv`
- `metrics/pairwise_inversions.csv`
- `metrics/bootstrap_regret.csv`
- `metrics/bootstrap_winner_stability.csv`
- `metrics/target_balance.csv`
- `metrics/feature_balance.csv`
- `metrics/local_mask_sensitivity.csv`
- `metrics/final_decision_table.csv`
- Deep dives under `reports/candidate_*_deep_dive.md`

_Stop: bake-off complete. No new candidates. No packaging._
