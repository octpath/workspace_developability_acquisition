# Stage 3 — 予測立体構造から得られる基本物性特徴

## 1. このStageで何をしたか

Stage 3では、配列だけのモデルに対して、**予測Fv立体構造から計算した基本 geometry / 表面特徴**が追加情報を持つかを検証した。使用した構造は参加者側の **ESMFold_native（primary）** と **ABodyBuilder2（source比較）** で、いずれも Dev配列からラベル非依存に生成されたPDBである。feature familyごとに structure-only を評価し、その後 PLM-only / overall sequence incumbent への fusion を Primary CV（Optuna）で比較、有望候補のみ Shadow で監査した。TmAppでは packing を PLM に足すとわずかな改善が Shadow で再現し、overall + RASA もわずかに改善した。HICでは **SURFACE_CHEM structure-only が Primary で Stage2 sequence overall をわずかに上回り、Shadow でも同程度**となり、表面化学特徴だけでも sequence-only に匹敵する性能が得られた。高度なエネルギー計算や学習型構造表現は Stage 4 に残した。

## 2. Stage 2/2bからの出発点

| Target | track | source | Primary MAE | Shadow MAE |
|---|---|---|---:|---:|
| TmApp | PLM-only | Stage2b AbLang2 HL_paired raw Ridge | 2.8634 | 2.9803 |
| TmApp | overall | Stage2 AbLang2 + SEQ_BASIC | 2.7756 | 2.8316 |
| HIC | PLM-only | Stage2 ESM-2 Heavy SVR | 0.4552 | 0.4529 |
| HIC | overall | Stage2 ESM-2 Heavy + SEQ_ALL | 0.4485 | 0.4510 |

## 3. 使用した予測構造

| source | 役割 | 備考 |
|---|---|---|
| ESMFold_native | primary | `esmfold_native/{id}.pdb`、鎖 A=VH / B=VL |
| ABodyBuilder2 | secondary（ablation） | `gate_b1/.../abodybuilder2` + manifest、H/L IMGT |

B1 Gly-linker ESMFold は非ネイティブ連結のため未使用。SASA/RASA/patch は label-independent な gate_b2 キャッシュを再利用し、packing / Rg / 接触密度は Stage3 で PDB から追加計算した。

詳細: `STRUCTURE_CACHE_AUDIT_JA.md`

## 4. 構造品質・mapping監査

| source | N complete | VH/VL exact match | malformed |
|---|---:|---:|---:|
| ESMFold_native | 162/162 | 162/162 | 0 |
| ABodyBuilder2 | 162/162 | 162/162 | 0 |

- Dev ID ↔ PDB 対応、両鎖存在、配列一致を確認
- CDR/FW 集約は gate_b2 側の IMGT（ABB）/ author region lengths（ESMFold sequential）に依存。信頼度（plddt等）は GLOBAL に含め、物性と混同しない注記付き
- interface 分数特徴に欠損がある抗体はあり、fold内 median impute（黙った0埋めはしない）

出力: `stage3_structure_quality.csv`

## 5. 今回作ったstructure feature

| family | 内容 | 科学的な狙い |
|---|---|---|
| STRUCT_GLOBAL | 残基数、露出/埋没代理、信頼度要約、Rg関連 | 全体サイズ・コンパクトさ |
| STRUCT_SASA | Fv/VH/VL/CDR別 total・mean SASA | 溶媒露出面積そのもの |
| STRUCT_RASA | mean/median/max RASA、露出割合（閾値 0.20/0.25/0.50 事前固定） | 相対露出 |
| STRUCT_SURFACE_CHEM | アミノ酸群別 SASA、charge proxy、疎水/芳香比率 | 「何が表面に出ているか」 |
| STRUCT_PATCH | 疎水/荷電 patch 数・最大面積（RASA≥0.2、CA距離≤8Å） | 表面疎水クラスター |
| STRUCT_INTERFACE | BSA、interface残基数・組成、鎖間接触 | VH–VL 界面 |
| STRUCT_PACKING | 接触密度、NN距離、clash proxy、Rg、compactness | パッキング粗視化 |
| STRUCT_SURFACE_ALL | SASA+RASA+SURFACE_CHEM+PATCH | 表面情報の束 |
| STRUCT_GEOMETRY_ALL | GLOBAL+INTERFACE+PACKING | 幾何情報の束 |
| STRUCT_ALL | 上記すべて | 上限確認用 |

定義は `stage3_feature_manifest.csv`。FoldX / Rosetta / APBS / inverse folding 等は未実施。

## 6. Structure-only結果 — TmApp

主要 Optuna（Primary）:

| family | model | Primary MAE | fold SD |
|---|---|---:|---:|
| RASA | ElasticNetOpt | **3.241** | 0.289 |
| ALL | ElasticNetOpt | 3.246 | 0.278 |
| PACKING | RidgeOpt | 3.342 | 0.205 |
| PATCH | RidgeOpt | 3.402 | 0.221 |
| INTERFACE | ElasticNetOpt | 3.447 | — |
| SASA | ElasticNetOpt | 3.447 | — |

固定 Ridge では PACKING（3.36）が相対的に良いが、いずれの structure-only も PLM-only（2.86）や overall（2.78）には及ばない。**3D記述子単体の信号は弱い〜中程度**。

## 7. Structure-only結果 — HIC

| family | model | Primary MAE | fold SD |
|---|---|---:|---:|
| SURFACE_CHEM | SVROpt | **0.440** | 0.049 |
| SURFACE_CHEM | ElasticNetOpt | 0.466 | 0.047 |
| SASA | ElasticNetOpt | 0.502 | 0.062 |
| INTERFACE | ElasticNetOpt | 0.536 | 0.050 |
| PACKING | ElasticNetOpt | 0.538 | 0.— |
| PATCH | ElasticNetOpt | 0.581 | — |
| RASA | ElasticNetOpt | 0.561 | — |

SURFACE_CHEM + SVROpt は Primary ≈ **0.440**、Shadow ≈ **0.449**。Stage 2 sequence overall は Primary ≈ **0.4485**、Shadow ≈ **0.4510** である。したがって **SURFACE_CHEM structure-only は、Primary では Stage 2 sequence overall をわずかに上回り、Shadow でも同程度**だった。今回の CV では、予測構造から得られる表面化学特徴だけでも sequence-only モデルに匹敵する予測性能が得られた。ただし「構造情報の方が sequence より本質的に優れている」「表面化学が HIC を決定する」とは言わない。

## 8. SASA / RASAは有効だったか

- **HIC**: SASA単独は中程度（≈0.50）。RASA単独は弱い。表面化学（群別SASA・charge）が最も強い → 「総露出」より「何が露出しているか」が効いている可能性
- **TmApp**: Optuna後は RASA が structure-only 最良だが、絶対MAEは sequence に劣る。overall fusion では RASA 追加がわずかに改善（§12）

## 9. Surface hydrophobicity / aromaticity / patchは有効だったか

- HIC 残差と **Fv_sasa_aromatic** の相関が相対的に大きい（overall残差 vs aromatic SASA ≈0.33）
- hydrophobic SASA・patch SASA も正の弱い相関
- PATCH family 単体のMAEは高くないが、SURFACE_ALL / SURFACE_CHEM に含まれると fusion で寄与しうる
- 表面疎水性・芳香族露出仮説と**整合する**が、決定要因とは言えない

## 10. VH–VL interface / packingは有効だったか

- **TmApp**: 固定Ridgeで PACKING が相対最良。PLM + PACKING（SVROpt）が Primary 2.836 / Shadow 2.933 で PLM-only をわずかに上回り Shadow 再現（`STRUCT_SHADOW_CONFIRMED`）
- INTERFACE 単独の structure-only は弱いが、overall + INTERFACE は Primary ではほぼ同等〜微差
- packing / interface と熱安定性の関連仮説と**部分的に整合**（弱い追加信号）

## 11. Structure sourceによる違い

基本的な feature extraction logic（SASA/RASA/patch/interface の集約手順）は共通化したが、**CDR/FW region mapping には structure source 間で実装上の差が残っている**。

- ABodyBuilder2: IMGT region mapping
- ESMFold_native: author region lengths を用いた sequential mapping

したがって **region-specific feature を含む比較は、structure predictor だけを変えた完全な controlled ablation とはみなさない**。Fv 全体の SASA など region boundary に依存しない特徴は、より直接的な source comparison として解釈できる。

共通ロジック・RidgeOpt（20 trials）での Primary MAE 比較（抜粋）:

| target | family | ESMFold | ABodyBuilder2 | region依存 |
|---|---|---:|---:|---|
| HIC | SURFACE_CHEM | 0.473 | 0.482 | 一部あり |
| HIC | SASA（Fv含む） | 0.508 | 0.512 | 部分的 |
| TmApp | SASA | 3.498 | **3.217** | 部分的 |
| TmApp | SURFACE_ALL | **3.480** | 3.697 | あり |
| TmApp | INTERFACE | 3.506 | 3.490 | 低め |

小差が多く、**どちらが絶対に正しい構造か**は結論しない。下流では ESMFold を primary とし、TmAppの一部SASAではABBが良い例もあるため Stage4 で必要なら両源を再確認する。

## 12. TmApp: sequence modelへの追加効果

| 設定 | Primary | Shadow | Δ vs PLM-only (P) | Δ vs overall (P) |
|---|---:|---:|---:|---:|
| Stage2b PLM-only | 2.8634 | 2.9803 | — | — |
| best structure-only (RASA) | 3.241 | 3.215 | +0.38 | +0.47 |
| PLM + PACKING | **2.836** | **2.933** | **-0.027** | +0.061 |
| Stage2 overall | 2.7756 | 2.8316 | — | — |
| overall + RASA | **2.754** | **2.820** | -0.109 | **-0.022** |

解釈: structure-onlyは弱いが、**PLMとpackingがわずかに相補的**（型B）。overallへの追加も小さく Shadow 再現あり。

## 13. HIC: sequence modelへの追加効果

| 設定 | Primary | Shadow | Δ vs PLM-only (P) | Δ vs overall (P) |
|---|---:|---:|---:|---:|
| Stage2 PLM-only | 0.4552 | 0.4529 | — | — |
| best structure-only (SURFACE_CHEM) | **0.440** | 0.449 | **-0.015** | -0.008 |
| PLM + SURFACE_CHEM | **0.444** | **0.442** | **-0.011** | -0.005 |
| Stage2 overall | 0.4485 | 0.4510 | — | — |
| overall + SURFACE_ALL | **0.441** | **0.438** | -0.015 | **-0.008** |

解釈: 表面特徴に予測情報があり（型A寄り）、PLM/overallへの追加も Shadow で再現（型Bも併存）。改善幅は小さい。

## 14. Primary / Shadow整合性

| 候補 | Primary改善 | Shadow | 判定 |
|---|---|---|---|
| TmApp PLM+PACKING | あり（小） | あり | `STRUCT_SHADOW_CONFIRMED` |
| TmApp overall+RASA | あり（小） | あり | `STRUCT_SHADOW_CONFIRMED` |
| HIC PLM+SURFACE_CHEM | あり（小） | あり | `STRUCT_SHADOW_CONFIRMED` |
| HIC overall+SURFACE_ALL | あり（小） | あり | `STRUCT_SHADOW_CONFIRMED` |

Primaryだけの微小改善で落とす候補もあった（例: TmApp overall+PACKING は Shadow良好だが Primary が overall を超えず `STRUCT_NO_GAIN`）。

**incumbent更新**: 両targetとも overall + best structure が Primary/Shadow でわずかに改善したため、`stage3_best_models.json` に **Stage3 provisional incumbent** として記録した。改善幅は小さいため、Stage2 overall を完全破棄せず併記する。

## 15. HIC high-tail診断

Stage 0 凍結定義を使用（既存 OOF の再集計のみ。再学習なし）:

- high-tail: **HIC ≥ 10.5372 min**
- **N = 17**（同一抗体セット）

| model | overall MAE | high-tail MAE | high-tail bias (pred−true) | underpred count | overall pred SD |
|---|---:|---:|---:|---:|---:|
| PLM-only | 0.4554 | **1.836** | −1.836 | 17/17 | 0.303 |
| Stage2 overall | 0.4489 | **1.852** | −1.852 | 17/17 | 0.283 |
| best structure-only (SURFACE_CHEM) | 0.4398 | **1.632** | −1.632 | 17/17 | 0.385 |
| Stage3 overall+SURFACE_ALL | 0.4409 | **1.769** | −1.769 | 17/17 | 0.310 |

全モデルで high-tail は一貫して **underprediction**（bias < 0、17/17）。structure-only の tail MAE は相対的にやや良いが、予測分散も大きい。**high-tail は診断用のみで、model selection には使っていない**。

## 16. 残差の相補性

既存 OOF からの再集計（新規学習なし）。「相補性」は原則として **残差–残差相関** を指す。

### モデル間の残差相関

`residual_seq = y − pred_seq`（Stage2 overall）、`residual_struct = y − pred_struct`（best structure-only）

| target | corr(residual_seq, residual_struct) | 解釈メモ |
|---|---:|---|
| TmApp | **0.876** | 誤差パターンが大きく共有（強い相補性ではない） |
| HIC | **0.891** | 同様に誤差の共通成分が大きい |

### sequence残差とstructure予測の関連

| target | corr(residual_seq, pred_struct) |
|---|---:|
| TmApp | 0.196 |
| HIC | 0.410 |

HIC では structure-only 予測が sequence 残差と中程度に関連しうるが、残差–残差相関が高いため **独立な誤差の打ち消し（強い model-error complementarity）とは言い切れない**。

### sequence残差とstructure特徴の関連

| target | feature | corr(residual_seq, feature) |
|---|---|---:|
| HIC | Fv_sasa_aromatic | 0.327 |
| HIC | H_CDR3_mean_rasa | 0.198 |
| HIC | Fv_rg | 0.169 |
| HIC | Fv_sasa_hydrophobic | 0.113 |
| TmApp | largest_hydrophobic_patch_sasa | 0.153 |
| TmApp | Fv_rg | 0.153 |
| TmApp | H_CDR3_mean_rasa | 0.110 |
| TmApp | Fv_buried_frac | −0.104 |

本Stageでは本格 residual stacking は行っていない。

## 17. Stage 3で分かったこと

1. PDB–配列 mapping は両sourceとも Dev 162で健全
2. structure-only: TmAppは弱い。HICの SURFACE_CHEM は sequence overall に匹敵（Primaryでわずかに上回り、Shadow同程度）
3. SASA「総量」より SURFACE_CHEM（露出組成）が HIC で有用
4. patch単独は弱いが表面束には寄与しうる
5. packingは TmApp の PLM 追加でわずかに有効
6. sequence incumbent への追加効果は**小さいが Shadow再現あり**。残差–残差相関は高く、強い誤差相補性は限定的
7. Stage4の高度PDB物理・学習表現を試す価値はある（特に HIC 静電/詳細patch、TmApp energy/packing）

## 18. Stage 4で試すべきこと（優先順）

**HIC（優先）**
1. 表面静電ポテンシャル / 電荷パッチ（APBS系）— 表面化学が効いたことと整合
2. より詳細な疎水・芳香族 patch 幾何
3. pKa / プロトン化状態の粗い代理（慎重に）
4. inverse-folding / 学習型構造表現

**TmApp（優先）**
1. FoldX / Rosetta 系の粗視化エネルギー・パッキング
2. H-bond / salt bridge / buried unsatisfied polar
3. cavity / 局所packingの精緻化
4. inverse-folding likelihood

Stage 3中にはこれらを実行しない。

---

注: §15–16の数値は既存 OOF からの再集計であり、Stage3学習結果自体は変更していない。

**最終状態:** `STAGE3_FINAL_REPORT_FROZEN`
