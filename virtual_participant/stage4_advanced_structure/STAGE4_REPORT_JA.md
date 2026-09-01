# Stage 4 — PDB由来の高度な物理・構造特徴

## 1. このStageで何をしたか

Stage 3で、HICでは SURFACE_CHEM が sequence overall に匹敵し、TmAppでは packing/RASA の小さな追加効果が Shadow 再現した。そこで Stage 4では、HIC向けに静電・プロトン化・高度 surface patch、TmApp向けに interaction / packing / cavity / buried-polar proxy、両target向けに ESM-IF inverse folding を評価した。利用できたのは PROPKA、PDB2PQR、APBS、幾何 interaction、ESM-IF1。FoldX は license 未取得のため未実施、Rosetta/ProteinMPNN はスキップした。advanced-only は sequence incumbent に届かない一方、Stage3 provisional への fusion では TmApp の ADV_INTERACTIONS と HIC の ADV_SURFACE_PATCH が Primary/Shadow 双方でわずかに改善した。改善幅は小さく Stage3 成果物は上書きせず、Stage5 候補として記録した。

## 2. Stage 3からの出発点

| Target | track | Primary | Shadow |
|---|---|---:|---:|
| TmApp | PLM-only (Stage2b) | 2.8634 | 2.9803 |
| TmApp | sequence overall (Stage2) | 2.7756 | 2.8316 |
| TmApp | Stage3 provisional (overall+RASA) | ≈2.754 | ≈2.820 |
| HIC | PLM-only (Stage2) | 0.4552 | 0.4529 |
| HIC | sequence overall (Stage2) | 0.4485 | 0.4510 |
| HIC | Stage3 provisional (overall+SURFACE_ALL) | ≈0.441 | ≈0.438 |

詳細: `artifacts/stage4_incumbents_manifest.md`

## 3. 環境・外部tool監査

| tool | version | status | purpose | failure/fallback |
|---|---|---|---|---|
| PROPKA | 3.5.1 | INSTALLED | pKa / charge | py3.12 venv |
| PDB2PQR | 3.7.1 | INSTALLED | PQR | pH 6.5 固定 |
| APBS | 3.4.1 | INSTALLED | continuum potential | 全162 DX 再パース成功 |
| FoldX | — | BLOCKED_BY_LICENSE | energy | 取得せず |
| Rosetta | — | SKIPPED_OPTIONAL | energy | authorized install なし |
| ESM-IF1 | fair-esm | INSTALLED | inverse folding LL | `.venv_esmif` CPU |
| ProteinMPNN | — | SKIPPED_OPTIONAL | native LL | 依存パス不適で skip |
| cavity tool | — | SKIPPED_OPTIONAL | volume | geometry proxy |

詳細: `STAGE4_ENVIRONMENT_AUDIT_JA.md` / `stage4_tool_status.csv`

## 4. 実験条件とfeature provenance

参加者 README / DATA_DICTIONARY には Shehata HIC の **pH・塩濃度の完全数値は無い**。そのため静電記述子は **assay-matched とは呼ばない**。

固定条件（Optuna 対象外）:

- protonation pH = **6.5**（predeclared generic reference condition）
- APBS ionic strength = **0.15 M**（generic）
- dielectric / grid テンプレートは全抗体統一

詳細: `HIC_CONDITION_PROVENANCE_JA.md`

## 5. HIC — electrostatics / protonation

- PROPKA / PQR: 162/162 成功。exposed net charge 等を ADV_PROPKA / ADV_PQR_CHARGE に集約
- APBS: 162/162 で potential DX を取得し、露出残基近傍の mean/q10/q90、正負 fraction、電位パッチを ADV_ELECTROSTATICS に集約
- advanced-only（Primary）: SURFACE_PATCH（≈0.472）＞ HIC_ALL ≒ electrostatics 単体より patch が強い傾向。静電単独は sequence overall（0.449）には届かない
- fusion では ELECTROSTATICS 追加は Stage3 provisional とほぼ同等〜わず差（選抜外）

→ 今回採用した generic fixed electrostatic condition（pH 6.5, ionic strength 0.15 M）では、APBS / PROPKA 由来特徴による明確な追加予測価値は認められなかった。ただし、assay 条件と完全には整合していないため、この結果だけから HIC に対する静電性一般の寄与を否定することはできない。

## 6. HIC — advanced surface patches

- local hydrophobic/aromatic SASA、spatial hydrophobicity（generic; SAP とは名乗らない）、patch compactness/extent
- **advanced-only 最良**: ADV_SURFACE_PATCH + SVROpt — Primary **0.472** / Shadow **0.478**
- **fusion 最良**: Stage3 incumbent + ADV_SURFACE_PATCH — Primary **0.437** / Shadow **0.433**（`ADV_SHADOW_CONFIRMED`）
- sequence residual と `adv_max_local_aromatic_sasa` の相関 ≈0.34

→ Stage3 の「芳香族・疎水露出」方向を、より空間的な記述子で補強した可能性がある（因果断定なし）。

## 7. TmApp — interactions / packing

- salt bridge、H-bond geometry proxy、clash、contact density、packing degree quantiles、interface proxies
- advanced-only: ADV_INTERACTIONS ≈3.30、ADV_TMAPP_ALL（SVR）≈**3.224**（最良 advanced-only）
- **fusion**: Stage3 incumbent + ADV_INTERACTIONS — Primary **2.743** / Shadow **2.770**（`ADV_SHADOW_CONFIRMED`）

## 8. TmApp — energy-like descriptors

- FoldX / Rosetta: **未実施**（license / authorized install）
- 代替: cavity_proxy、buried_unsatisfied_polar_proxy（heuristic）
- fusion で UNSAT_POLAR も Shadow 確認（Primary 2.745 / Shadow 2.780）だが改善は小さい
- これらは **physical energy そのものではない**

## 9. Inverse folding

- ESM-IF1 で ESMFold_native の A/B 鎖 native LL をスコア（162/162）
- advanced-only: TmApp ≈3.34、HIC ≈0.539（弱い）
- fusion への単独追加は Stage3 provisional を明確には上回らず
- residual と invfold LL に弱い関連（TmApp |r|≈0.17）

**解釈上の注意:** ESM-IF1 score は、sequence から予測された構造に対する structure-conditioned sequence compatibility である。実験構造に対する物理的 folding stability や thermodynamic stability そのものではない。また ESMFold と ESM-IF1 の学習分布・model bias を部分的に共有する可能性もあるため、model-level self-consistency を含む指標として解釈する。

詳細: `INVERSE_FOLDING_CACHE_AUDIT_JA.md`

## 10. Advanced feature-only performance

| target | best family | model | Primary | Shadow |
|---|---|---|---:|---:|
| TmApp | ADV_TMAPP_ALL | SVROpt | 3.224 | 3.215 |
| HIC | ADV_SURFACE_PATCH | SVROpt | 0.472 | 0.478 |

いずれも sequence / Stage3 incumbent より悪い。**advanced descriptor 単体では不足**。

## 11. TmApp incumbentへの追加効果

以下は registry / JSON から取得した **exact value**（6桁）。本文の丸め値との差は判断に用いない。

| 設定 | Primary exact | Shadow exact | ΔPrimary exact | ΔShadow exact |
|---|---:|---:|---:|---:|
| Stage3 provisional | 2.753911 | 2.820234 | — | — |
| + ADV_INTERACTIONS | 2.742609 | 2.770430 | −0.011302 | −0.049804 |
| + ADV_UNSAT_POLAR | 2.744957 | 2.779734 | −0.008954 | −0.040500 |

（参考・3桁丸め）Stage3 provisional 2.754 / 2.820、+ ADV_INTERACTIONS **2.743** / **2.770**、+ ADV_UNSAT_POLAR 2.745 / 2.780。

小さな改善が Shadow 再現。interaction / unsatisfied-polar proxy が packing 仮説と整合しうる。

## 12. HIC incumbentへの追加効果

以下は registry / JSON から取得した **exact value**（6桁）。

| 設定 | Primary exact | Shadow exact | ΔPrimary exact | ΔShadow exact |
|---|---:|---:|---:|---:|
| Stage3 provisional | 0.440524 | 0.438397 | — | — |
| + ADV_SURFACE_PATCH | 0.437272 | 0.433172 | −0.003252 | −0.005225 |
| + ADV_HIC_ALL | 0.438885 | 0.436529 | −0.001639 | −0.001868 |

（参考・3桁丸め）Stage3 provisional 0.441 / 0.438、+ ADV_SURFACE_PATCH **0.437** / **0.433**、+ ADV_HIC_ALL 0.439 / 0.437。

極小〜小さな改善。electrostatics 単独追加はほぼ同等。

## 13. Primary / Shadow整合性

| 候補 | 判定 |
|---|---|
| TmApp fusion + INTERACTIONS | `ADV_SHADOW_CONFIRMED` |
| TmApp fusion + UNSAT_POLAR | `ADV_SHADOW_CONFIRMED` |
| HIC fusion + SURFACE_PATCH | `ADV_SHADOW_CONFIRMED` |
| HIC fusion + HIC_ALL | `ADV_SHADOW_CONFIRMED` |

**incumbent更新**: Stage3 JSON は上書きしない。Stage4 best fusion を `stage4_best_models.json` に provisional 候補として記録。改善幅は小さい（特に HIC Δ≈0.003）。

## 14. HIC high-tail診断

凍結定義: **HIC ≥ 10.5372 min（N=17）**

| model | overall MAE | tail MAE | bias (pred−true) | underpred |
|---|---:|---:|---:|---:|
| PLM-only | 0.455 | 1.836 | −1.836 | 17/17 |
| seq overall | 0.449 | 1.852 | −1.852 | 17/17 |
| Stage3 overall+struct | 0.441 | 1.769 | −1.769 | 17/17 |
| best adv-only | 0.472 | 1.927 | −1.927 | 17/17 |
| best Stage4 fusion | 0.438 | **1.745** | −1.745 | 17/17 |

tail は一貫して underprediction。fusion で tail MAE はわずかに改善するが、選抜指標には未使用。

## 15. 残差の相補性

### モデル間の残差相関

| target | corr(residual_seq, residual_adv) |
|---|---:|
| TmApp | 0.875 |
| HIC | 0.914 |

誤差パターンの共有が大きく、強い model-error complementarity とは言えない。

### sequence残差とstructure特徴

- HIC: local aromatic SASA、spatial hydrophobicity が相対的に大きい
- TmApp: invfold LL、salt bridge、cavity proxy が弱い関連

## 16. 外部tool failure / limitation

- FoldX: BLOCKED_BY_LICENSE
- Rosetta / ProteinMPNN: SKIPPED
- APBS 初回パース失敗 → DX 名 `*-PE0.dx` 対応後に 162/162 再集計（抗体削除なし）
- electrostatics は generic condition（assay-matched ではない）
- ESM-IF は CPU 推論（CUDA device mismatch 回避）

## 17. Stage 4で分かったこと

1. 高度表面 patch は HIC の Stage3 方向をわずかに補強
2. APBS/PROPKA の単独寄与は今回の固定条件では限定的
3. TmApp では interaction / unsatisfied-polar proxy の fusion が小さく再現
4. inverse folding 単独は弱いが残差との弱い関連あり
5. energy 本体（FoldX/Rosetta）は未評価のまま残る
6. HIC では overall MAE は Stage を追うごとに改善した一方、frozen high-tail 17 抗体は主要モデルすべてで一貫して underprediction された。Stage4 fusion でも tail MAE は多少改善したが、prediction shrinkage そのものは解消していない。したがって現在の HIC の主要な残存課題の一つは、平均 MAE だけでなく高 HIC 側への系統的縮小である（high-tail metric を model-selection objective にはしない）。
7. 新規物理 feature の限界利益が小さくなってきたため、Stage5 では model integration / calibration / residual modeling を主テーマとする

## 18. 次に試すべきこと（Stage5方針・未実行）

Stage5 の主探索は model integration / calibration / residual modeling とする。authorized FoldX / Rosetta や assay-condition-matched electrostatics は将来追加できる横枝として保持するが、Stage5 の主探索をそれらの利用可能性に依存させない。

横枝候補（Stage5 主探索外）:

1. HIC: assay 条件が判明した場合の ionic strength 整合静電、または固定パネル感度（score 最適化はしない）
2. TmApp: authorized FoldX/Rosetta energy（取得できる場合のみ）
3. inverse folding の CDR/HCDR3 分割の精緻化
4. PLM fine-tuning / LoRA はなお後段

---

**最終状態:** `STAGE4_FINAL_REPORT_FROZEN`
