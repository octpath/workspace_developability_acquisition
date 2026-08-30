Overall educational-split verdict:
    CAND_12528 is a healthy noisy leaderboard (Public useful; CV↔Public usually agree). Conditional Trust-CV on material disagreement is NOT demonstrated (HIC favors Public when they disagree).

Production candidate:
    CAND_12528

Educational goal:
    CV and Public broadly agree;
    when they materially disagree, Trust CV.

----------------
TmApp
----------------

    Unique eligible models:
        38

    CV↔Public agreement:
        directional=0.546; Spearman=0.093; Pearson=0.139

    Material CV/Public disagreements:
        N=113; rate_among_signed_pairs=0.162

    Private follows CV:
        count: 53
        rate: 0.482
        uncertainty: [0.392, 0.577]

    Private follows Public:
        count: 57
        rate: 0.518

    FOLLOW_CV Private regret:
        median: 0.0002
        mean: 0.0626
        p90: 0.2107
        max: 0.5208

    FOLLOW_PUBLIC Private regret:
        median: 0.0000
        mean: 0.0682
        p90: 0.2038
        max: 0.3545

    Model-family robustness:
        ablang2→0.45; bio→0.47; const→0.47; ensemble→0.48; esm2→0.49; fusion→0.76; simple→0.37; structure→0.48

    TmApp Trust-CV verdict:
        TRUST_CV_AMBIGUOUS

----------------
HIC
----------------

    Unique eligible models:
        37

    CV↔Public agreement:
        directional=0.846; Spearman=0.849; Pearson=0.966

    Material CV/Public disagreements:
        N=17; rate_among_signed_pairs=0.026

    Private follows CV:
        count: 2
        rate: 0.133
        uncertainty: [0.000, 0.355]

    Private follows Public:
        count: 13
        rate: 0.867

    FOLLOW_CV Private regret:
        median: 0.0243
        mean: 0.0267
        p90: 0.0552
        max: 0.0754

    FOLLOW_PUBLIC Private regret:
        median: 0.0000
        mean: 0.0018
        p90: 0.0004
        max: 0.0292

    Model-family robustness:
        const→0.07; ensemble→0.14; esm2→0.11; fusion→0.10; simple→0.17; structure→0.25

    HIC Trust-CV verdict:
        PUBLIC_AT_LEAST_AS_RELIABLE

----------------
Conditional action taken
----------------

    Phase 2B + Phase 3 limited split review

    Phase-1: TmApp=PUBLIC_AT_LEAST_AS_RELIABLE, HIC=PUBLIC_AT_LEAST_AS_RELIABLE

----------------
Final interpretation
----------------

Property 1 — CV/Public correlate:
    TmApp Spearman=0.093, agree=0.546;
    HIC Spearman=0.849, agree=0.846.
    Verdict: WEAK/MIXED.

Property 2 — Public is useful for Private:
    Supported as noisy but informative (agreement mostly positive; Public-following not catastrophic on this split).

Property 3 — disagreement favors CV:
    TmApp TrustCV_rate=0.4818181818181818; HIC TrustCV_rate=0.13333333333333333
    (material CLEAR disagreements only; see FOLLOW_* regrets).

Is "Trust CV" educationally justified:
    Only weakly / not strongly demonstrated.

Would a novice who follows Public be punished excessively:
    No.

Would a participant with robust local CV be rewarded:
    Mixed.

Is CAND_12528 a healthy noisy leaderboard:
    YES

Is there a strong reason to change split:
    NO

FINAL DECISION:
    FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED

## Compact summary table

| Target | CV↔Public agree | Material N | Follow CV | Follow Public | Follow-CV mean regret | Follow-Public mean regret | Verdict |
|--------|----------------:|-----------:|----------:|--------------:|----------------------:|--------------------------:|---------|
| TmApp | 0.55 | 113 | 53 | 57 | 0.0625522569038568 | 0.06820564764288274 | TRUST_CV_AMBIGUOUS |
| HIC | 0.85 | 17 | 2 | 13 | 0.02673937821206028 | 0.0017787261184235112 | PUBLIC_AT_LEAST_AS_RELIABLE |

## Random policy summary

target       policy  mean_private  mean_regret  median_regret  p90_regret
   HIC     BALANCED      0.447573     0.016314       0.018122    0.018122
   HIC     CV_FIRST      0.454847     0.023587       0.024599    0.039776
   HIC PUBLIC_FIRST      0.447573     0.016314       0.018122    0.018122
 TmApp     BALANCED      3.388719     0.114469       0.155957    0.155957
 TmApp     CV_FIRST      3.359674     0.085424       0.080397    0.155957
 TmApp PUBLIC_FIRST      3.388719     0.114469       0.155957    0.155957

### Policy win-rates

target policy_A     policy_B  A_beats_B  B_beats_A  tie
 TmApp CV_FIRST PUBLIC_FIRST       0.31       0.09 0.60
 TmApp CV_FIRST     BALANCED       0.31       0.09 0.60
 TmApp BALANCED PUBLIC_FIRST       0.00       0.00 1.00
   HIC CV_FIRST PUBLIC_FIRST       0.00       0.64 0.36
   HIC CV_FIRST     BALANCED       0.00       0.64 0.36
   HIC BALANCED PUBLIC_FIRST       0.00       0.00 1.00

## Required answers (1–26)

1. TmApp CV↔Public after dedup: Spearman=0.093, Pearson=0.139, directional agree=0.546.
2. HIC CV↔Public after dedup: Spearman=0.849, Pearson=0.966, directional agree=0.846.
3. Material disagreement counts: TmApp N=113, HIC N=17.
4. Enough for an educational conclusion? Yes if N≥3 and rates stable under sensitivity; see verdicts.
5. TmApp Private follows CV rate = 0.4818181818181818.
6. HIC Private follows CV rate = 0.13333333333333333.
7. Sensitivity to thresholds: TmApp: SENS_0.75_0.25→rate=0.21875; SENS_0.8_0.2→rate=0.16666666666666666; SENS_0.85_0.15→rate=0.16666666666666666; SENS_0.9_0.1→rate=0.14285714285714285 | HIC: SENS_0.75_0.25→rate=0.5; SENS_0.8_0.2→rate=0.3333333333333333; SENS_0.85_0.15→rate=0.0; SENS_0.9_0.1→rate=0.0.
8. Family-driven? LOFO TmApp: ablang2→0.45; bio→0.47; const→0.47; ensemble→0.48; esm2→0.49; fusion→0.76; simple→0.37; structure→0.48; HIC: const→0.07; ensemble→0.14; esm2→0.11; fusion→0.10; simple→0.17; structure→0.25.
9. Persona/duplicate-driven? Dedup by sha256(round(OOF,6)+round(test,6)); unique B7 transitions drop persona duplicates of same from→to. Results are on unique banks.
10. Safer on disagreement: TmApp: FOLLOW_CV; HIC: FOLLOW_PUBLIC.
11. Regret distributions: see FOLLOW_* blocks and metrics/follow_strategy_private_regret.csv.
12. Are Public mistakes more costly than CV mistakes? HIC: no (FOLLOW_PUBLIC mean regret lower). TmApp: similar means.
13. Are CV mistakes more costly? HIC: yes on mean/p90 when forcing FOLLOW_CV against Public. TmApp: comparable.
14. Do TmApp and HIC differ? Phase-1 PUBLIC_AT_LEAST_AS_RELIABLE vs PUBLIC_AT_LEAST_AS_RELIABLE; final-use TRUST_CV_AMBIGUOUS vs PUBLIC_AT_LEAST_AS_RELIABLE.
15. After dedup, does B7 Public-chasing advantage persist? Not as a reason to change split; unique-bank Trust-CV audit is the educational criterion.
16. Slogan defensible for TmApp? Only weakly / no.
17. Slogan defensible for HIC? No — Public tends to win material disagreements.
18. Expanded stress (if run): action=Phase 2B + Phase 3 limited split review; see metrics/trust_cv_summary_stress.csv and reports/02_*.
19. Random trajectories lowest Private regret policy: TmApp: lowest mean regret = CV_FIRST (0.0854); HIC: lowest mean regret = BALANCED (0.0163).
20. Stable across trajectories? See p90_regret and win-rates; CV_FIRST vs PUBLIC_FIRST win matrix above.
21. Phase 3 alternative dominate CAND_12528? No — KEEP CAND_12528.
22. Validation-bank survival of alternative? N/A — no remapped alternative dominance.
23. Preserve TmApp/HIC balance / prior safety? Yes by keeping CAND_12528.
24. Strong enough to abandon CAND_12528? NO.
25. Permanently freeze CAND_12528? **YES (recommend freeze; Trust-CV slogan not claimed)**.
26. Participant guidance: Public feedback is informative and often aligned with local CV. Clear CV/Public conflicts are uncommon. Do not assume that local CV always wins when it conflicts with Public — especially for HIC, treat large stable Public improvements seriously. Prefer models that improve under both signals when possible, and avoid overreacting to tiny Public MAE changes.

## Participant-facing guidance (no Private leakage)

Public feedback is informative and often aligned with local CV. Clear CV/Public conflicts are uncommon. Do not assume that local CV always wins when it conflicts with Public — especially for HIC, treat large stable Public improvements seriously. Prefer models that improve under both signals when possible, and avoid overreacting to tiny Public MAE changes.
