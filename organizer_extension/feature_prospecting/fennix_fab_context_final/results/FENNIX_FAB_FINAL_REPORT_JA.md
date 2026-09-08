# FeNNix Fab — 最終レポート（フルコホート）

UTC: 2026-09-08T11:15:29.374936+00:00

## 冒頭 Q&A

1. **323本はきれいに組めたか？** YES（`FINAL_FENNIX_COHORT.csv`、B/C/M SUCCESS、r1/matched OK）。除外 ADI-47265。
2. **fresh-worker 再現性は？** **NUMERICALLY_CLOSE**
3. **usable Dev N？** 161 / 162（欠: ['ADI-47265']）
4. **interim CONSTANT はフルで再現したか？** **DID_NOT_REPLICATE**（interim P/S Δ=0.1160/0.0512 → full -0.0264/0.0222）
5. **BIOEMU_NEW_CONTACT+CONSTANT？** **DID_NOT_REPLICATE** / full sci=SCI_MIXED
6. **Simple TVT で Primary∩Shadow 改善 family？** なし
7. **BASE_FIXED_ALPHA 生存？** なし
8. **strict nested late-fusion？** ほぼ全て NO_INCREMENT（悪化側）。CONSTANT residual Primary のみごく弱い正。
9. **科学的結論** FeNNix Fab 特徴は技術的に組立可能だが、フル Dev では安定した予測増分を示さない。interim N=100 CONSTANT の正信号は **再現しなかった**。
10. **競技結論** 全 family / BIOEMU+CONSTANT → **COMP_DROP**（direct fusion でも Primary/Shadow 同時安定改善なし）。
11. **参加者リリースしてよいか？** 技術 QC=QC_PASS_WITH_LIMITATIONS、再現性=NUMERICALLY_CLOSE。特徴自体は target-blind で科学的に定義済みのため **配布可**。ただし CV ヒントは「安定改善なし」と明記。ライセンスは FeNNol LGPL-3.0 → `PARTICIPANT_ONLY_REVIEW_RECOMMENDED`。
12. **v1.1 パッケージ？** YES（`feature_extension_v1.1_*.zip`）

## Simple TVT（抜粋）

| family                      |   N |   struct_dim |   primary_base_mae |   primary_cand_mae |   primary_delta |   shadow_base_mae |   shadow_cand_mae |   shadow_delta |   fixed_primary_delta |   fixed_shadow_delta | sci_verdict   | comp_verdict   |
|:----------------------------|----:|-------------:|-------------------:|-------------------:|----------------:|------------------:|------------------:|---------------:|----------------------:|---------------------:|:--------------|:---------------|
| DELTA_GEOM                  | 161 |           17 |            2.75827 |            2.80834 |     -0.0500724  |           2.88597 |           3.04657 |     -0.160601  |           -0.0500724  |           -0.0688567 | SCI_NO_SIGNAL | COMP_DROP      |
| DELTA_ENV                   | 161 |           17 |            2.75827 |            2.75468 |      0.00358575 |           2.88597 |           2.93819 |     -0.0522155 |            0.00358575 |           -0.0522155 | SCI_MIXED     | COMP_DROP      |
| CONSTANT                    | 161 |           20 |            2.75827 |            2.78464 |     -0.0263729  |           2.88597 |           2.86379 |      0.0221799 |           -0.0263729  |            0.0221799 | SCI_MIXED     | COMP_DROP      |
| INTERFACE                   | 161 |           18 |            2.75827 |            2.84092 |     -0.0826555  |           2.88597 |           3.03951 |     -0.153538  |           -0.0826555  |           -0.0568581 | SCI_NO_SIGNAL | COMP_DROP      |
| FULL_FAB_NORMALIZED         | 161 |            3 |            2.75827 |            2.77527 |     -0.0170032  |           2.88597 |           2.90587 |     -0.0198926 |           -0.0170032  |           -0.0198926 | SCI_NO_SIGNAL | COMP_DROP      |
| PREP_RELAX_SENSITIVITY      | 161 |           17 |            2.75827 |            2.789   |     -0.0307298  |           2.88597 |           2.8967  |     -0.0107227 |           -0.0307298  |           -0.0107227 | SCI_NO_SIGNAL | COMP_DROP      |
| COMBINED_PREDECLARED        | 161 |           86 |            2.75827 |            2.98462 |     -0.226352   |           2.88597 |           2.98886 |     -0.102888  |           -0.0963579  |           -0.0649123 | SCI_NO_SIGNAL | COMP_DROP      |
| BIOEMU_NEW_CONTACT+CONSTANT | 161 |           40 |            2.75827 |            2.78225 |     -0.0239867  |           2.88597 |           2.83227 |      0.0536992 |           -0.0239867  |            0.0536992 | SCI_MIXED     | COMP_DROP      |

## Interim vs Full

| family                      |   interim_N |   interim_Primary_delta |   interim_Shadow_delta | interim_verdict                  |   full_N |   full_Primary_delta |   full_Shadow_delta | full_verdict   | replication       |
|:----------------------------|------------:|------------------------:|-----------------------:|:---------------------------------|---------:|---------------------:|--------------------:|:---------------|:------------------|
| DELTA_GEOM                  |         100 |              0.0253813  |            -0.0366071  | SIMPLE_CV_MIXED                  |      161 |          -0.0500724  |          -0.160601  | SCI_NO_SIGNAL  | INCONCLUSIVE      |
| DELTA_ENV                   |         100 |              0.0288105  |            -0.239663   | SIMPLE_CV_MIXED                  |      161 |           0.00358575 |          -0.0522155 | SCI_MIXED      | INCONCLUSIVE      |
| CONSTANT                    |         100 |              0.115986   |             0.0511702  | SIMPLE_CV_CONSISTENT_IMPROVEMENT |      161 |          -0.0263729  |           0.0221799 | SCI_MIXED      | DID_NOT_REPLICATE |
| INTERFACE                   |         100 |              0.061986   |            -0.0309325  | SIMPLE_CV_MIXED                  |      161 |          -0.0826555  |          -0.153538  | SCI_NO_SIGNAL  | INCONCLUSIVE      |
| FULL_FAB_NORMALIZED         |         100 |             -0.00826934 |            -0.00695893 | SIMPLE_CV_NO_IMPROVEMENT         |      161 |          -0.0170032  |          -0.0198926 | SCI_NO_SIGNAL  | INCONCLUSIVE      |
| PREP_RELAX_SENSITIVITY      |         100 |              0.0770838  |             0.0582692  | SCI_ROBUST_POSITIVE              |      161 |          -0.0307298  |          -0.0107227 | SCI_NO_SIGNAL  | DID_NOT_REPLICATE |
| COMBINED_PREDECLARED        |         100 |              0.0895688  |            -0.159037   | SIMPLE_CV_MIXED                  |      161 |          -0.226352   |          -0.102888  | SCI_NO_SIGNAL  | INCONCLUSIVE      |
| BIOEMU_NEW_CONTACT+CONSTANT |         100 |              0.16305    |             0.118949   | SCI_ROBUST_POSITIVE              |      161 |          -0.0239867  |           0.0536992 | SCI_MIXED      | DID_NOT_REPLICATE |

## Strict nested (STRICT_NESTED_INCREMENT, delta = incumbent MAE − combined MAE)

正 = 改善。結果は概ね負。

## QC

# FeNNix Fab — Final target-blind QC

UTC: 2026-09-08T10:31:12.741155+00:00

## Coverage
- expected IDs: 323 (exclude ADI-47265)
- feature rows: 323
- duplicate IDs: 0
- missing IDs vs cohort: 0
- missing feature cells: 0
- ±inf cells: 0
- zero-variance features: 6

## B/C/M integrity
- all accepted: True
- r1 npz ok: True

## Technical associations (flags |ρ|>0.85 vs n_atoms/n_sites)
- none above |ρ|=0.85 among sampled features

## Overall QC verdict

**QC_PASS_WITH_LIMITATIONS**

### Conditions A/B/C/M (SPEC)
- **A**: isolated Fv ESMFold (FeNNix-v2)
- **B**: Fab-geom Fv (from Fab) + R1 relax
- **C**: full prepared Fab + R1
- **M**: matched Fv coords from C (constants removed), no re-relax
- **DELTA_GEOM** = B−A; **DELTA_ENV** = C_var−M_var (environment, not absolute energy across systems)

