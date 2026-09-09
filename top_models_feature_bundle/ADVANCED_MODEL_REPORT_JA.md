# 先進モデルベンチマーク報告（CV 選択優先）

Public / Private は **事後解析専用（POSTMORTEM ONLY）**。モデル選択には未使用。

線形ベースライン（凍結 Top-3 Ridge/Lasso）との比較では

`improvement = linear_MAE − advanced_MAE`（正 = 先進モデルが良い）。

---

## CV 選択サマリ（推奨構成）

| target | family | 選択 | CV Primary / Shadow | worst |
|--------|--------|------|---------------------|-------|
| TmApp | XGBoost | `XGB__TM_BASE_BIOEMU_MPNN__RIDGE` (XGB_A) | 2.926 / 3.044 | 3.044 |
| TmApp | scratch TF | `TMS3` full+mean | 3.248 / 3.268 | 3.268 |
| TmApp | frozen TF | `TMF2` full+concat | 3.318 / 3.215 | 3.318 |
| TmApp | **overall** | `TMF2__FUSION__TM_BASE_BIOEMU_MPNN__RIDGE` | **2.754 / 2.773** | **2.773** |
| HIC | XGBoost | `XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO` (XGB_A) | 0.446 / 0.445 | 0.446 |
| HIC | scratch TF | `HICS2` full | 0.499 / 0.511 | 0.511 |
| HIC | frozen TF | `HICF2` full | 0.510 / 0.516 | 0.516 |
| HIC | **overall** | `HICS2__FUSION__HIC_ESM2_SEQ_AROMATIC__LASSO` | **0.442 / 0.441** | **0.442** |

---

## 科学的な問いへの回答

### 1. XGBoost は Ridge/Lasso を上回ったか？

- **HIC: YES**（Top-3 すべてで Primary Δ ≈ +0.033〜+0.036）。固定長特徴の非線形相互作用が効いた解釈が妥当。
- **TmApp: NO**（全レシピで悪化、Primary Δ ≈ −0.15〜−0.22）。小 N・高次元では浅い木でも正則化線形に勝てず。

### 2. 2層 scratch Transformer は線形ベースを超えたか？

- **NO**（TmApp worst≈3.27 vs 最良線形≈2.73；HIC≈0.51 vs ≈0.48）。
- 162 抗体の DEV だけでは、配列+アノテーションからの学習表現が固定長線形レシピに届かない。

### 3. IMGT + FR/CDR 注釈は scratch に効いたか？

- **弱い YES（両ターゲット）**。
  - TmApp: TMS2/TMS3（full）が TMS1（minimal）より良い（worst 3.32/3.27 vs 3.36）。
  - HIC: HICS2（full）が HICS1 より良い（0.511 vs 0.517）。
- 個体 IMGT 位置の因果ではない。位置・領域 prior の有用性を支持する程度。

### 4. 凍結残基 PLM は scratch より良いか？

- **TmApp: ほぼ同等〜わずかに mixed**（最良 frozen TMF2 worst 3.318 vs scratch TMS3 3.268 → scratch がわずかに勝ち）。
- **HIC: ほぼ同等**（HICF2 0.516 vs HICS2 0.511）。
- 事前学習残基表現の明確な勝ちは、この小規模設定では出ていない。

### 5. TmApp の concat vs mean

- sequence-only: **mean（TMS3）の方が robust（worst が小さい）**。
- frozen: **concat（TMF2）が mean（TMF3）よりわずかに良い**。
- fusion でも scratch 側は mean、frozen 側は concat を CV で保持。

### 6. Top-3 固定長 fusion は Transformer を改善したか？

- **強く YES（特に TmApp）**。sequence-only ~3.3 → fusion ~2.75 付近まで回復し、線形に接近。
- **HIC も YES**。sequence-only ~0.51 → fusion ~0.44 で **線形・XGBoost をも上回る CV**。
- 解釈: 学習系列表現と構造/物理固定特徴の相補。

### 7. HIC の ESM-2 Heavy 残基 Transformer は pooled ESM-2+構造記述子に勝ったか？

- sequence-only frozen（HICF*）は **NO**（線形 ~0.48 に未達）。
- **fusion（HICF2/HICS2 × Top-3）は YES**（worst ~0.44）。残基 Transformer 単体ではなく、固定特徴との結合が鍵。

### 8. Primary / Shadow 一貫性

- HIC XGB / fusion は P/S が近い。
- TmApp XGB・一部 fusion は Shadow がやや悪化しやすい。
- 選択は常に `cv_worst = max(P,S)`。

### 9. シード感度

- Transformer の Primary seed dispersion はおおむね 0.05〜0.25（TmApp scratch concat が大きめ）。
- 3 seed 平均 OOF を正規スコアとしている。

### 10. Public / Private（POSTMORTEM ONLY）

CV 選択 overall:

| target | config | Public | Private |
|--------|--------|--------|---------|
| TmApp | TMF2⊕BIOEMU_MPNN fusion | 3.268 | 3.281 |
| HIC | HICS2⊕ESM2_SEQ_ARO fusion | 0.398 | 0.480 |

- HIC fusion は Public が特に良いが、**Private も線形帯より良い**。
- TmApp fusion は CV では線形近傍だが、Public/Private は線形 Ridge の歴史的帯（~3.1）よりやや悪い場合あり。**CV 推奨を Public 勝者に差し替えない。**

### 11. 競技モデルとしての推奨は？

- **HIC**: CV 上、`HICS2`（または同程度の `HICF2`）× `HIC_ESM2_SEQ_AROMATIC` fusion、もしくは XGB on CONTINUOUS_SURFACE が有力。改善は worst ベースでロバスト。
- **TmApp**: 純粋 Transformer / XGB は非推奨。fusion は線形に接近するが、**凍結 Ridge Top-3 を明確に安定して上回る証拠は弱い**。提出の安全策は依然として線形（または線形寄りの fusion）を CV で確認すること。

---

## 環境

- GPU: NVIDIA GeForce RTX 3090
- torch / xgboost: `.venv_b1`（validate_environment 通過）
- 残基 `.npy` は gitignore；`.part0/.part1` + `assemble_embeddings.sh`（ローダにフォールバックあり）

## 提出

`advanced_outputs/submissions/` に `id,TmApp,HIC`（162行）:

- `submission_xgboost.csv`
- `submission_scratch_transformer.csv`
- `submission_frozen_transformer.csv`
- `submission_best_cv.csv`
