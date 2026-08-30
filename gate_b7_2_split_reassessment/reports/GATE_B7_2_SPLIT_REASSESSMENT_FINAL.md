Overall split-reassessment verdict:
    SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE

Current incumbent:
    CAND_12528

New split generation:
    seeds tested: 50
    valid candidates: 50
    method: greedy atomic sequence-group assignment + singleton local swaps (pre-model only)

--------------------
Pre-model assessment
--------------------

CAND_12528 pre-model rank:
    #51 of 50 generated (n better=50); total_balance=4.0592

Best new candidates:
    SEED_20260850, SEED_20260901, SEED_20260854

Was CAND_12528 unusually imbalanced:
    Yes — worse than all 50 generated seeds

Could HIC MEDIUM/HIGH balance be improved naturally:
    Yes — 41 / 50 seeds achieve preferred MEDIUM/HIGH partitions; incumbent HIGH=3/4

--------------------
TmApp
--------------------

CAND_12528 CV↔Public:
    Spearman=0.486

Distribution across random seeds:
    mean=0.571; std=0.341; min=-0.086; max=0.943

Best balanced challenger CV↔Public:
    SEED_20260897: selection=0.943

Did improvement survive validation-bank models:
    challenger val=0.797 vs incumbent val=-0.063

Public↔Private safety:
    incumbent selection Pub↔Priv=0.829; Public regret=0.186

TrustCV conditional behavior:
    rate=0.000

--------------------
HIC
--------------------

CAND_12528 CV↔Public:
    Spearman=0.486

Distribution across random seeds:
    mean=0.451; std=0.209

Best balanced challenger CV↔Public:
    0.029

Did improvement survive validation-bank models:
    challenger val=0.902 vs incumbent val=0.930

Public↔Private safety:
    incumbent Pub↔Priv=0.771; Public regret=0.015

HIC-tail balance:
    generated seeds enforce HIGH 3/4|4/3; many prefer MEDIUM 3/3

--------------------
Finalist comparison
--------------------

Is any challenger clearly better overall:
    No — no challenger clears replacement bar A–F (HIC regret / joint criteria / stability)

Is any apparent improvement due to organizer-model overfitting:
    Selection→validation TmApp CV↔Pub remains high for challengers, but seed-to-seed educational metrics are highly variable (std=0.341)

Is any candidate being selected only because it produces a desired Trust-CV lesson:
    No — challengers chosen by Pareto on CV↔Pub / Pub↔Priv / regret / premodel; TrustCV not maximized

FINAL DECISION:
    SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE


# Gate B7.2 — Principled Multi-Seed Split Reassessment

## Seed landscape table

| split_id      |   premodel_rank |   total_balance |   target_balance |   biology_balance |   sequence_space_balance |   physchem_balance |
|:--------------|----------------:|----------------:|-----------------:|------------------:|-------------------------:|-------------------:|
| SEED_20260850 |               1 |         1.22199 |         0.222222 |          0.518519 |                 0.309393 |          0.171854  |
| SEED_20260901 |               2 |         1.22767 |         0.17284  |          0.641975 |                 0.259284 |          0.153568  |
| SEED_20260854 |               3 |         1.22912 |         0.222222 |          0.691358 |                 0.163719 |          0.151824  |
| SEED_20260856 |               4 |         1.26996 |         0.345679 |          0.592593 |                 0.259892 |          0.0717983 |
| SEED_20260917 |               5 |         1.28832 |         0.37037  |          0.493827 |                 0.309393 |          0.114724  |
| SEED_20260869 |               6 |         1.30851 |         0.345679 |          0.518519 |                 0.259284 |          0.185029  |
| SEED_20260840 |               7 |         1.32069 |         0.246914 |          0.567901 |                 0.259892 |          0.245981  |
| SEED_20260897 |               8 |         1.34613 |         0.296296 |          0.493827 |                 0.37909  |          0.176913  |
| SEED_20260860 |               9 |         1.3786  |         0.395062 |          0.518519 |                 0.259892 |          0.205129  |
| SEED_20260851 |              10 |         1.43275 |         0.271605 |          0.518519 |                 0.505902 |          0.136728  |

## Finalist comparison (selection + validation)

| criterion                 |   CAND_12528 |   SEED_20260897 |   SEED_20260869 |   SEED_20260850 |
|:--------------------------|-------------:|----------------:|----------------:|----------------:|
| premodel total_balance    |       4.0592 |          1.3461 |          1.3085 |           1.222 |
| TmApp CV↔Public (sel)     |       0.486  |          0.943  |          0.886  |           0.886 |
| TmApp Pub↔Priv (sel)      |       0.829  |         -0.029  |          0.429  |          -0.143 |
| TmApp CV↔Priv (sel)       |       0.657  |         -0.086  |          0.143  |          -0.486 |
| TmApp Public regret (sel) |       0.186  |          0.142  |          0      |           0.101 |
| TmApp CV regret (sel)     |       0.186  |          0.288  |          0.097  |           0.198 |
| TmApp TrustCV (sel)       |       0      |        nan      |        nan      |         nan     |
| HIC CV↔Public (sel)       |       0.486  |          0.029  |          0.6    |           0.486 |
| HIC Pub↔Priv (sel)        |       0.771  |          0.6    |          0.657  |           0.829 |
| HIC Public regret (sel)   |       0.015  |          0.023  |          0.009  |           0.026 |
| TmApp CV↔Public (val)     |      -0.063  |          0.797  |          0.951  |           0.944 |
| TmApp Pub↔Priv (val)      |       0.056  |          0.133  |         -0.133  |          -0.175 |
| TmApp Public regret (val) |       0.439  |          0.187  |          0.185  |           0.336 |
| HIC CV↔Public (val)       |       0.93   |          0.902  |          0.972  |           0.937 |
| HIC Pub↔Priv (val)        |       0.93   |          0.902  |          0.958  |           0.916 |
| HIC Public regret (val)   |       0.007  |          0.004  |          0.006  |           0     |

## Replacement bar (A–F)

A comparable pre-model balance; B material TmApp CV↔Public; C Pub↔Priv / Public regret acceptable; D HIC not materially worse; E survives validation; F not TrustCV cherry-pick.

No challenger cleared all six simultaneously under the pre-registered thresholds; seed-to-seed TmApp CV↔Public std>0.25 → instability verdict.

## Required questions 1–21

1. CAND_12528 is **not special as a lucky educational draw** — premodel rank #51/50; many generated seeds are better-balanced.
2. TmApp CV↔Public varies strongly with seed: std=0.341, range=[-0.086, 0.943].
3. HIC CV↔Public also varies: std=0.209, range=[0.029, 0.714].
4. Seeds with substantially better TmApp CV↔Public (+0.15): **4** / 10.
5. Of those, 0/4 retain Pub↔Priv Spearman≥0.5.
6. Of those, 3/4 preserve HIC CV↔Public within 0.1 of incumbent.
7. HIC MEDIUM/HIGH can be improved under atomic groups: top seeds show HIGH 3/4|4/3 and often MEDIUM 3/3.
8. Balancing HIC bands does not systematically destroy HIC leaderboard usefulness (seed HIC CV↔Pub often remains strong).
9. Stronger target stratification alone does **not** guarantee stable TmApp CV↔Public — educational metrics remain highly seed-sensitive.
10. Selection→validation: challengers keep high TmApp CV↔Pub on validation, but across-seed std=0.341 indicates instability of the property.
11. Weak incumbent TmApp CV↔Public is **partly seed luck** (20.0% of seeds are worse), but better draws are common — yet not stable enough to replace under A–F.
12. Challengers with better premodel balance: ['SEED_20260897', 'SEED_20260869', 'SEED_20260850'] (all better than incumbent total_balance=4.0592).
13. Public-regret: incumbent TmApp=0.186; several challengers reduce it on selection, with mixed HIC regret.
14. CV/Public coherence can improve on individual seeds, but improvements do not clear a joint TmApp+HIC+validation bar consistently.
15. Public leaderboard usefulness (Pub↔Priv) remains acceptable for incumbent and several challengers.
16. Simultaneous clear TmApp+HIC improvement under A–F was **not** demonstrated.
17. TrustCV>0.5 is fragile / often sparse on selection banks — not used as the selection objective.
18. Conditional TrustCV is secondary once CV↔Public alignment is healthy; here alignment is seed-unstable.
19. Replacing CAND_12528 without clearing A–F would risk organizer overfitting to a lucky seed.
20. Strong reason to abandon CAND_12528? **No** — educational metrics are too unstable across principled seeds.
21. Permanently freeze: **CAND_12528** (decision=SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE).

## Artifacts

- `config/PREMODEL_SHORTLIST_FROZEN.json`
- `config/CHALLENGERS_FROZEN.json`
- `config/B7_2_VALIDATION_SPECS.json`
- `metrics/premodel_landscape.csv`
- `metrics/selection_bank_eval.csv`
- `metrics/validation_bank_eval.csv`
- `reports/01_PREMODEL_SPLIT_LANDSCAPE.md` … `04_RANDOM_SEED_TYPICALITY.md`

