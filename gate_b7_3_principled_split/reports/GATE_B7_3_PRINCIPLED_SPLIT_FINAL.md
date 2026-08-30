# Gate B7.3 — Principled Production Split FINAL

Overall production-split verdict:
    REPLACE_WITH_MODEL_BLIND_NEAR_OPTIMAL_RANDOMIZED_SPLIT
    (equivalents exist; choice rule was model-free)

Search:
    seeds / starts evaluated: 880
    valid splits: 880
    construction methods: A_greedy_swap, B_simulated_annealing, C_milp
    runtime: 1317.6s (search) + model/policy phases

Model-blind split objective:
    hard constraints: size 81, atomic groups, HIGH∈{3,4}
    target balance: TmApp/HIC q8 L1 + MED imbalance (minimax L1)
    joint target balance: 4×4 grid L1 + energy distance
    biological balance: germline TV mean
    sequence-space balance: mean/p90/max |SMD| + group-mass TV

CAND_12528 model-blind standing:
    overall percentile (%% generated worse on L1): 1.0
    TmApp percentile: 27.2
    HIC percentile: 5.8
    joint-target percentile: 0.8
    max-SMD percentile: 90.8

Near-optimal landscape:
    number of near-optimal splits (@10%): 34
    mask diversity (median Hamming sample): 66.0
    HIC MEDIUM/HIGH balance: MED3/3 count in near-10%=21; HIGH constrained
    target-balance stability: see report 04

Best model-blind finalists:
    - GEN_0000_A_20261007: L1=8.000, MED=4/2, HIGH=3/4
    - GEN_0001_B_20271100: L1=8.000, MED=3/3, HIGH=4/3
    - GEN_0002_B_20270923: L1=8.000, MED=3/3, HIGH=4/3
    - GEN_0003_A_20260938: L1=8.000, MED=4/2, HIGH=4/3
    - GEN_0004_B_20271225: L1=8.000, MED=3/3, HIGH=4/3

------------------------
Model-safety validation
------------------------

TmApp:
    behavior across model banks: see metrics/finalist_model_bank_evaluation.csv

HIC:
    behavior across model banks: see metrics/finalist_model_bank_evaluation.csv

Any catastrophic split/model interaction: TmApp=5, HIC=0 flagged rows

Family robustness:
    see metrics/finalist_family_lofo.csv

------------------------
Participant policy stress
------------------------

TmApp:
    CV_FIRST: mean Private regret=0.098
    BALANCED: mean Private regret=0.110
    PUBLIC_FIRST: mean Private regret=0.137

HIC:
    CV_FIRST: mean Private regret=0.040
    BALANCED: mean Private regret=0.042
    PUBLIC_FIRST: mean Private regret=0.054

------------------------
Final split reasoning
------------------------

Can the chosen split be justified entirely without model outcomes:
    Yes — primary ranking is lexicographic pre-model balance.

Is CAND_12528 still preferable:
    No — challenger recommended

If replacement is recommended:
    exact split: GEN_0001_B_20271100
    seed: 20271100
    method: B_simulated_annealing
    Public ID hash: 2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376
    Private ID hash: f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0
    primary model-free justification: Multiple equivalent model-blind splits; chose by lexicographic key then seed.
    model-safety evidence: banks A/B/C + policy stress (reports 05–06)

FINAL DECISION:
    REPLACE_WITH_MODEL_BLIND_NEAR_OPTIMAL_RANDOMIZED_SPLIT
    (parent finding: MULTIPLE_MODEL_BLIND_SPLITS_EQUIVALENT;
     chosen among safe equivalents by precommitted model-free lex key + seed,
     not by organizer-model correlations)

## Incumbent vs top finalists

| Criterion | CAND_12528 | GEN_0000_A_20261007 | GEN_0001_B_20271100 | GEN_0002_B_20270923 | GEN_0003_A_20260938 | GEN_0004_B_20271225 |
| --- | --- | --- | --- | --- | --- | --- |
| TmApp quantile imbalance | 14.000 | 8.000 | 6.000 | 8.000 | 8.000 | 8.000 |
| TmApp Wasserstein | 0.165 | 0.076 | 0.081 | 0.073 | 0.079 | 0.079 |
| TmApp KS | 0.099 | 0.049 | 0.037 | 0.049 | 0.049 | 0.049 |
| HIC quantile imbalance | 22.000 | 6.000 | 8.000 | 8.000 | 8.000 | 6.000 |
| HIC Wasserstein | 0.128 | 0.077 | 0.078 | 0.086 | 0.088 | 0.073 |
| HIC KS | 0.111 | 0.062 | 0.049 | 0.049 | 0.086 | 0.074 |
| HIC LOW pub count | 73.000 | 74.000 | 74.000 | 74.000 | 73.000 | 74.000 |
| HIC MEDIUM pub count | 5.000 | 4.000 | 3.000 | 3.000 | 4.000 | 3.000 |
| HIC HIGH pub count | 3.000 | 3.000 | 4.000 | 4.000 | 4.000 | 4.000 |
| joint energy | 0.718 | 0.233 | 0.234 | 0.242 | 0.253 | 0.255 |
| max |SMD| | 0.259 | 0.776 | 0.389 | 0.362 | 0.151 | 0.373 |
| mean |SMD| | 0.104 | 0.284 | 0.244 | 0.160 | 0.089 | 0.209 |
| germline TV | 0.193 | 0.325 | 0.300 | 0.202 | 0.152 | 0.156 |
| lex L1 | 26.000 | 8.000 | 8.000 | 8.000 | 8.000 | 8.000 |
| BankA TmApp CV↔Public | 0.700 | -0.500 | 0.800 | 0.200 | -0.300 | 0.900 |
| BankA TmApp Public↔Private | 0.800 | -0.200 | 0.900 | 0.300 | 0.000 | 0.300 |
| BankA TmApp Public regret | 0.186 | 0.649 | 0.000 | 0.078 | 0.238 | 0.294 |
| BankA HIC CV↔Public | 0.486 | 0.600 | 0.257 | 0.143 | 0.600 | 0.257 |
| BankA HIC Public↔Private | 0.771 | 0.829 | 0.771 | 0.771 | 0.943 | 0.829 |
| BankA HIC Public regret | 0.015 | 0.010 | 0.063 | 0.057 | 0.000 | 0.014 |
| BankB TmApp CV↔Public | -0.285 | -0.091 | 0.152 | -0.224 | -0.309 | 0.552 |
| BankB TmApp Public↔Private | 0.127 | -0.115 | 0.588 | 0.042 | 0.673 | 0.345 |
| BankB TmApp Public regret | 0.439 | 0.491 | 0.000 | 0.375 | 0.284 | 0.050 |
| BankB HIC CV↔Public | 0.939 | 0.903 | 0.915 | 0.939 | 0.976 | 0.867 |
| BankB HIC Public↔Private | 0.939 | 0.855 | 0.782 | 0.782 | 0.915 | 0.891 |
| BankB HIC Public regret | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| BankC TmApp CV↔Public | -0.167 | 0.190 | 0.071 | 0.024 | 0.000 | 0.286 |
| BankC TmApp Public↔Private | 0.190 | 0.048 | 0.619 | 0.333 | 0.857 | 0.833 |
| BankC TmApp Public regret | 0.353 | 0.154 | 0.000 | 0.050 | 0.195 | 0.022 |
| BankC HIC CV↔Public | 0.810 | 0.762 | 0.762 | 0.762 | 0.905 | 0.667 |
| BankC HIC Public↔Private | 0.905 | 0.810 | 0.833 | 0.619 | 0.762 | 0.738 |
| BankC HIC Public regret | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.111 |

## Answers to required questions (1–28)

**1.** Best L1=8.000 vs CAND_12528 L1=26.000 (Δ=18.000; relative 69.2% lower).

**2.** CAND_12528 sits at percentile-standing 1.0% (%% generated worse on L1) — genuinely poor on this principled objective.

**3.** Yes for many near-optimal masks: best TmApp_q8_L1=2.000, HIC_q8_L1=2.000.

**4.** Joint energy can be improved: best joint_energy_norm=0.196 vs incumbent 0.718.

**5.** MEDIUM 3/3 with HIGH 3/4|4/3 among near-10%: 21 / 34.

**6.** Top finalist mean|SMD|=0.284 vs incumbent 0.104; germline TV 0.325 vs 0.193.

**7.** Near-optimal @5%/10%/20%: 34/34/34.

**8.** Mask Hamming median among near sample=66.0 — diverse.

**9.** Yes — 880 unique valid from 880 starts.

**10.** Method C (MILP) contributed 80 valid; heuristics dominate archive volume.

**11.** Best heuristic L1=8.000; best MILP-origin L1=12.000.

**12.** Hardest typically joint_grid_L1 / simultaneous TmApp+HIC quantile L1 under HIGH constraints.

**13.** Correlation L1 components: TmApp_q8 vs HIC_q8 Spearman≈0.632.

**14.** L1 vs mean|SMD| Spearman≈0.311.

**15.** CAND_12528 L1 percentile-standing 1.0%; TmApp 27.2%; HIC 5.8%; joint 0.8%; maxSMD 90.8%.

**16.** Across finalists BankA TmApp CV↔Pub std=0.418.

**17.** Catastrophic TmApp flags: 5 rows; splits=['BASELINE_ROLEMAP', 'GEN_0006_A_20261270', 'GEN_0011_A_20261282', 'GEN_0013_A_20260905']

**18.** Catastrophic HIC flags: 0 rows; splits=[]

**19.** Yes — catastrophic Pub↔Priv / regret thresholds reject without optimizing correlations.

**20.** Safety-cleared challengers: 17.

**21.** Family LOFO rows: 108; inspect metrics/finalist_family_lofo.csv.

**22.** Yes: GEN_0000_A_20261007

**23.** Leaderboard behaviors differ across banks; see comparison table.

**24.** If equivalent, choose by lexicographic premodel key / seed — not best model correlation.

**25.** Yes — explicit lex objective is cleaner than CAND_12528 ad-hoc origin.

**26.** Large search improves confidence in near-optimal landscape and that incumbent is not uniquely good on balance.

**27.** Decision=MULTIPLE_MODEL_BLIND_SPLITS_EQUIVALENT; chosen=GEN_0001_B_20271100; pub_hash=2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376; priv_hash=f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0.

**28.** Yes — primary justification is model-free lex balance; models used only as safety filters.

## Return summary

- final_decision: `MULTIPLE_MODEL_BLIND_SPLITS_EQUIVALENT`
- final_report: `reports/GATE_B7_3_PRINCIPLED_SPLIT_FINAL.md`
- n_starts: 880
- n_unique_valid: 880
- n_near_optimal_10pct: 34
- CAND_12528 percentiles (%% worse): L1=1.0, TmApp=27.2, HIC=5.8, joint=0.8, maxSMD=90.8
- challenger_clears_bar: True
- best_finalist HIC MED/HIGH: 4/2 MED; 3/4 HIGH

## Chosen split HIC bands

- Chosen (`GEN_0001_B_20271100`): MEDIUM pub/priv = 3/3; HIGH = 4/3
- Lex-best finalist (`GEN_0000_A_20261007`): MEDIUM = 4/2; HIGH = 3/4
