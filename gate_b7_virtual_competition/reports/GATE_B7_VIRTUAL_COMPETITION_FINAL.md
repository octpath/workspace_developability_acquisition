# Gate B7 — Blind Virtual Competition Final Report

Overall competition-simulation verdict:

**CAND_12528 is suitable for production freeze** as an informative-but-imperfect Public leaderboard: sequential participant iteration shows broad improvement transfer with modest noise; Public-driven overfitting risk is limited under a short submission budget; final-selection Private regret remains small for reasonable personas.

Split:
- **CAND_12528** (protocol sha256 `d2b485131e570b80b03167d47a1aa34d763b7f623923589ae9c3c40ff5b4f168`)


### TmApp
- A_CV_FIRST final Private MAE: 3.3518 (ENSEMBLE_TOP3_CV)
- B_BALANCED final Private MAE: 3.4727 (ENSEMBLE_TOP2_PUBLIC)
- C_PUBLIC_DRIVEN final Private MAE: 3.4921 (ENSEMBLE_PUBLIC_BEST2)
- Best achievable Private among attempted: 3.3518
- A_CV_FIRST: CV→Private agree=0.43, Public→Private agree=0.71, final regret=0.0000, Public FPs=1
- B_BALANCED: CV→Private agree=0.43, Public→Private agree=0.57, final regret=0.0781, Public FPs=1
- C_PUBLIC_DRIVEN: CV→Private agree=0.12, Public→Private agree=0.75, final regret=0.1400, Public FPs=1
- Does Public feedback generally help: YES, imperfectly (mean Public→Private directional agree=0.68)
- Evidence of Public overfitting: LIMITED (early FP rate=0.00, late FP rate=0.00)
- Final-selection regret:
    - A_CV_FIRST: 0.0000
    - B_BALANCED: 0.0781
    - C_PUBLIC_DRIVEN: 0.1400


### HIC
- A_CV_FIRST final Private MAE: 0.4735 (STRUCT_ElasticNet)
- B_BALANCED final Private MAE: 0.4494 (ENSEMBLE_TOP2_PUBLIC)
- C_PUBLIC_DRIVEN final Private MAE: 0.4437 (FUSION_ESM2_STRUCT_ENet)
- Best achievable Private among attempted: 0.4437
- A_CV_FIRST: CV→Private agree=0.83, Public→Private agree=0.83, final regret=0.0298, Public FPs=1
- B_BALANCED: CV→Private agree=0.62, Public→Private agree=0.75, final regret=0.0057, Public FPs=2
- C_PUBLIC_DRIVEN: CV→Private agree=0.88, Public→Private agree=0.88, final regret=0.0000, Public FPs=1
- Does Public feedback generally help: YES, imperfectly (mean Public→Private directional agree=0.82)
- Evidence of Public overfitting: LIMITED (early FP rate=0.33, late FP rate=0.14)
- Final-selection regret:
    - A_CV_FIRST: 0.0298
    - B_BALANCED: 0.0057
    - C_PUBLIC_DRIVEN: 0.0000

- HIGH-tail post-hoc (final picks): A_CV_FIRST: HIGH→LOW=1.00, HIGH MAE=1.974, elev≥10.5=0.00; B_BALANCED: HIGH→LOW=1.00, HIGH MAE=1.977, elev≥10.5=0.00; C_PUBLIC_DRIVEN: HIGH→LOW=0.75, HIGH MAE=1.860, elev≥10.5=0.20

## Cross-target conclusion

- Would a novice trusting Public be materially misled? **Mostly no** under ≤10 submissions — Public FPs occur but final regrets stay modest (TmApp Public→Private agree≈0.68, HIC≈0.82).
- Would an experienced CV-driven participant be rewarded? **Yes** — especially on TmApp (0 regret) and still competitive on HIC; mean CV→Private agree TmApp=0.33, HIC=0.78.
- Does repeated submitting cause severe leaderboard overfitting? **Not strongly** in this budget (LIMITED / LIMITED).
- Is CAND_12528 suitable for production: **YES — permanently freeze CAND_12528**
- Any reason to reopen split selection: **NO**
- Recommended next step: **Permanently freeze CAND_12528 and proceed to packaging/docs**, with participant guidance: trust grouped CV; treat Public as noisy confirmation; for HIC prefer SGKF-style CV and do not chase tiny Public gains. HIGH-tail remains hard regardless of persona.

When participants followed Public improvements, Private moved the same direction in **70%** of those transitions (pooled).

## Required questions (detailed)


### TmApp
1. A_CV_FIRST sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:BIO_Ridge → S4:ABLANG2_PCA32_SVR → S5:ESM2_PCA64_SVR → S6:STRUCT_ElasticNet → S7:BIO_ABLANG2_Ridge → S8:ENSEMBLE_TOP2_CV → S9:ENSEMBLE_TOP3_CV
2. A_CV_FIRST CV MAE path: 3.5444 → 3.7530 → 3.2267 → 3.3743 → 3.3614 → 3.5248 → 3.0948 → 2.7361 → 2.7361
3. A_CV_FIRST Public MAE path: 3.7840 → 3.5763 → 3.7336 → 3.6448 → 3.5249 → 3.7762 → 4.0355 → 3.4649 → 3.4649
4. A_CV_FIRST Private MAE path (post-reveal): 3.7716 → 3.6531 → 3.5789 → 3.3946 → 3.5029 → 3.5033 → 3.5278 → 3.3518 → 3.3518
1. B_BALANCED sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:BIO_Ridge → S4:ABLANG2_PCA32_SVR → S5:ESM2_PCA64_SVR → S6:ENSEMBLE_CV_PUBLIC → S7:ENSEMBLE_TOP2_PUBLIC → S8:STRUCT_ElasticNet
2. B_BALANCED CV MAE path: 3.5444 → 3.7530 → 3.2267 → 3.3743 → 3.3614 → 3.1567 → 3.2322 → 3.5248
3. B_BALANCED Public MAE path: 3.7840 → 3.5763 → 3.7336 → 3.6448 → 3.5249 → 3.5451 → 3.4949 → 3.7762
4. B_BALANCED Private MAE path (post-reveal): 3.7716 → 3.6531 → 3.5789 → 3.3946 → 3.5029 → 3.5026 → 3.4727 → 3.5033
1. C_PUBLIC_DRIVEN sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:BIO_Ridge → S4:ABLANG2_PCA32_SVR → S5:ESM2_PCA64_SVR → S6:ENSEMBLE_PUBLIC_BEST2 → S7:BIO_ABLANG2_Ridge → S8:STRUCT_ElasticNet → S9:FUSION_ESM2_STRUCT_ENet
2. C_PUBLIC_DRIVEN CV MAE path: 3.5444 → 3.7530 → 3.2267 → 3.3743 → 3.3614 → 3.4601 → 3.0948 → 3.5248 → 3.5801
3. C_PUBLIC_DRIVEN Public MAE path: 3.7840 → 3.5763 → 3.7336 → 3.6448 → 3.5249 → 3.4178 → 4.0355 → 3.7762 → 3.5299
4. C_PUBLIC_DRIVEN Private MAE path (post-reveal): 3.7716 → 3.6531 → 3.5789 → 3.3946 → 3.5029 → 3.4921 → 3.5278 → 3.5033 → 3.3521
5. When CV improved, Private improved in 50% of transitions (n=10).
6. When Public improved, Private improved in 73% of transitions (n=15).
7. Public improvements were typically real more often than not (Private co-improve rate=0.73).
8. Public-better-but-Private-worse submissions (FP transitions): 3.
9. FP magnitude (Private worsening): mean ΔPrivate=+0.1083, max=+0.1083.
10. Late Public overfitting evidence: early FP=0.00, late FP=0.00 → not strong.
11. CV-first final: ENSEMBLE_TOP3_CV (Private=3.3518).
12. Public-driven final: ENSEMBLE_PUBLIC_BEST2 (Private=3.4921).
13. Best final Private persona: A_CV_FIRST (3.3518).
14. A_CV_FIRST final Private regret: 0.0000.
14. B_BALANCED final Private regret: 0.0781.
14. C_PUBLIC_DRIVEN final Private regret: 0.1400.
15. A_CV_FIRST best Private at S8 (ENSEMBLE_TOP2_CV).
15. B_BALANCED best Private at S4 (ABLANG2_PCA32_SVR).
15. C_PUBLIC_DRIVEN best Private at S9 (FUSION_ESM2_STRUCT_ENet).
16. Stage vs MAE Spearman: CV=-0.57, Public=-0.43, Private=-0.78 (negative means higher stage → lower MAE).

### HIC
1. A_CV_FIRST sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:ESM2_PCA64_SVR → S4:STRUCT_ElasticNet → S5:ESM2_PHYS_STRUCT_Ridge → S6:FUSION_ESM2_STRUCT_ENet → S7:PHYS_STRUCT_Huber
2. A_CV_FIRST CV MAE path: 0.5189 → 0.5811 → 0.4872 → 0.4780 → 0.5882 → 0.4852 → 1.3751
3. A_CV_FIRST Public MAE path: 0.5408 → 0.6695 → 0.5342 → 0.5260 → 0.5889 → 0.5134 → 1.0606
4. A_CV_FIRST Private MAE path (post-reveal): 0.5041 → 0.5823 → 0.4674 → 0.4735 → 0.5255 → 0.4437 → 1.0545
1. B_BALANCED sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:ESM2_PCA64_SVR → S4:STRUCT_ElasticNet → S5:ESM2_PHYS_STRUCT_Ridge → S6:ENSEMBLE_CV_PUBLIC → S7:FUSION_ESM2_STRUCT_ENet → S8:ENSEMBLE_TOP2_PUBLIC → S9:PHYS_STRUCT_Huber
2. B_BALANCED CV MAE path: 0.5189 → 0.5811 → 0.4872 → 0.4780 → 0.5882 → 0.4733 → 0.4852 → 0.4727 → 1.3751
3. B_BALANCED Public MAE path: 0.5408 → 0.6695 → 0.5342 → 0.5260 → 0.5889 → 0.5159 → 0.5134 → 0.5091 → 1.0606
4. B_BALANCED Private MAE path (post-reveal): 0.5041 → 0.5823 → 0.4674 → 0.4735 → 0.5255 → 0.4583 → 0.4437 → 0.4494 → 1.0545
1. C_PUBLIC_DRIVEN sequence: S1:CONST_MEDIAN → S2:SEQ_SIMPLE_Ridge → S3:ESM2_PCA64_SVR → S4:STRUCT_ElasticNet → S5:ESM2_PHYS_STRUCT_Ridge → S6:ENSEMBLE_PUBLIC_BEST2 → S7:PHYS_STRUCT_Ridge → S8:FUSION_ESM2_STRUCT_ENet → S9:PHYS_STRUCT_Huber
2. C_PUBLIC_DRIVEN CV MAE path: 0.5189 → 0.5811 → 0.4872 → 0.4780 → 0.5882 → 0.4747 → 0.6733 → 0.4852 → 1.3751
3. C_PUBLIC_DRIVEN Public MAE path: 0.5408 → 0.6695 → 0.5342 → 0.5260 → 0.5889 → 0.5158 → 0.6594 → 0.5134 → 1.0606
4. C_PUBLIC_DRIVEN Private MAE path (post-reveal): 0.5041 → 0.5823 → 0.4674 → 0.4735 → 0.5255 → 0.4559 → 0.6132 → 0.4437 → 1.0545
5. When CV improved, Private improved in 64% of transitions (n=11).
6. When Public improved, Private improved in 67% of transitions (n=12).
7. Public improvements were typically real more often than not (Private co-improve rate=0.67).
8. Public-better-but-Private-worse submissions (FP transitions): 4.
9. FP magnitude (Private worsening): mean ΔPrivate=+0.0059, max=+0.0060.
10. Late Public overfitting evidence: early FP=0.33, late FP=0.14 → not strong.
11. CV-first final: STRUCT_ElasticNet (Private=0.4735).
12. Public-driven final: FUSION_ESM2_STRUCT_ENet (Private=0.4437).
13. Best final Private persona: C_PUBLIC_DRIVEN (0.4437).
14. A_CV_FIRST final Private regret: 0.0298.
14. B_BALANCED final Private regret: 0.0057.
14. C_PUBLIC_DRIVEN final Private regret: 0.0000.
15. A_CV_FIRST best Private at S6 (FUSION_ESM2_STRUCT_ENet).
15. B_BALANCED best Private at S7 (FUSION_ESM2_STRUCT_ENet).
15. C_PUBLIC_DRIVEN best Private at S8 (FUSION_ESM2_STRUCT_ENet).
16. Stage vs MAE Spearman: CV=-0.16, Public=-0.29, Private=-0.23 (negative means higher stage → lower MAE).

### HIC-specific
17/18. A_CV_FIRST final HIGH→LOW=1.00, HIGH MAE=1.974; overall MAE gains do not fully fix HIGH-tail.
17/18. B_BALANCED final HIGH→LOW=1.00, HIGH MAE=1.977; overall MAE gains do not fully fix HIGH-tail.
17/18. C_PUBLIC_DRIVEN final HIGH→LOW=0.75, HIGH MAE=1.860; overall MAE gains do not fully fix HIGH-tail.


### Shared wrap-up (Q19–Q24)

19. TmApp improvement path is generally smoother than HIC (fewer catastrophic late models; HIC Huber attempts explode MAE).
20. Public feedback is useful for both; HIC shows higher Public→Private directional agreement (0.82 vs 0.68), while TmApp CV-first selection is more decisive for final Private.
21. CAND_12528 behaves like a **healthy, noisy** competition leaderboard in realistic sequential use.
22. Public is **informative but imperfect**, not actively misleading on average.
23. Neither target shows enough Public overfitting to **require** split redesign.
24. **Yes — permanently freeze CAND_12528** (pending human confirmation).

## Blindness attestation

- Participant decisions used only Train CV + returned Public MAE / personal Public rank.
- Private MAE was vaulted by the organizer and revealed only after `participant/final_submission_choice_*.json` freeze.
- Historical B6 Private-bearing reports were not used for live decisions.
- Append-only timeline: `logs/competition_timeline.jsonl` (Private fields appended only post-reveal).
- Frozen protocol: `config/B7_SIMULATION_PROTOCOL.json`.

## Artifacts

- Metrics: `metrics/all_submissions.csv`, `submission_transitions.csv`, `directional_agreement.csv`, `best_so_far.csv`, `regret_over_time.csv`, `final_selection_quality.csv`, `cross_persona_comparison.csv`, `hic_submission_developability_diagnostics.csv`
- Narratives: `reports/B7_NARRATIVES.md`
- Plots: `plots/*_{persona}_cv_public_private_trajectory.png`, `*_normalized_improvement.png`, `*_best_so_far.png`, `*_private_regret_over_time.png`, plus persona comparison and Public-vs-Private scatters
