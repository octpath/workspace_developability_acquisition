# Fusion Preprocessing Stability Audit（日本語）

**State:** `ORGANIZER_FUSION_PREPROCESSING_STABILITY_AUDIT_COMPLETE`  
**Scope:** HIC × AROMATIC-TOPO；歴史的 Fusion Closure 結果は未改変  
**閾値凍結:** `STABILITY_AUDIT_THRESHOLDS.json`（HIC スコア解釈前）

実装の詳細は `PREPROCESSING_IMPLEMENTATION_AUDIT.md`。

---

## 冒頭の直接回答

1. **Aromatic StandardScaler は実質不安定か？** → **No.** 全15特徴で不安定フラグなし。median |Δz| ≈ 0.08–0.19、SD の fold CV ≈ 0.01–0.05。
2. **ESM2 PCA32 は fold 依存か？** → **Yes（full 32D 部分空間）.** 凍結閾値では `HIGHLY_FOLD_DEPENDENT`（mean cos²≈0.85 だが min cos²≈0）。末尾 PC の回転が主因。leading 4–8（center-only）は `MODERATELY_VARIABLE`。
3. **StandardScaler-before-PCA は重要な選択か？** → **幾何的には Yes、HIC MAE にはほぼ No.** 同一 fold で scale-before vs center の mean cos²≈0.78；PC1 寄与 22% vs 48%。だが P0 vs P1 の CV MAE 差は ≈0.002。
4. **Nested CV が PCA を ~100 標本にしたか？** → **No.** 実装では PCA/aromatic scaler は **outer train ≈129–130** で適合。nested は連結後の再スケールと α 選択のみ（inner ≈96–98）。
5. **Global-Dev 前処理は大きな診断ゲインか？** → **No.** P5 MAE=0.462 は P0(0.456)/P1(0.454) より **悪い**（漏洩なのに改善せず）。
6. **Raw ESM2 + 強 Ridge は PCA32 が過剰制約か？** → **示唆せず.** P3=0.474 > P0/P1。PCA64 も 0.482 と悪化。
7. **前回 PLM+AROMATIC 融合は前処理律速か？** → **弱い証拠のみ.** 代替前処理で実質的 MAE 改善なし。部分空間は fold 依存だが、それが性能頭打ちの主因とは言えない。

---

## 1. 実際の前処理（実装回収）

| 枝 | 内容 |
|----|------|
| PLM | ESM2-H **1280d** → **StandardScaler → PCA32（whiten=False）** |
| AROMATIC | **15** 特徴 → train-median 欠損埋め → **StandardScaler**（log/clip なし） |
| Fusion | ブロック独立スケール後 concat → **もう一度 StandardScaler** → Ridge |
| α | 内 GroupKFold（残グループ数=4 のため **n_splits=4**）で {0.1,1,10,100} |

---

## 2. 標本サイズ

Dev N=162。Outer val ≈32–33 / outer train ≈129–130。

| 適合対象 | N |
|----------|---|
| Aromatic scaler / ESM2 scaler / PCA（OOF 本体） | **≈129–130** |
| Inner HP 用 concat scaler | **≈96–98** |
| PCA 段階で p≫n | **1280 ≫ 130** → Yes |

**注:** 小さい K は訓練をさらに減らすため、前処理推定の不安定解決にはならない。補足の固定α KFold: 5-fold train≈130 / 10-fold≈146；PCA mean cos² 0.86→0.93。

---

## 3. Aromatic scaler 安定性（ラベル無し）

ESMFold（ABB2/Boltz2 も同様の傾向）:

- 全特徴 `flags=none`
- `sd_max/min` ≤ 1.15、`sd_cv` ≤ 0.05
- median |z variation| < 0.25 閾値未満

**結論: aromatic StandardScaler は実質安定。** RobustScaler 必須の証拠なし。

---

## 4. PCA32 部分空間安定性（ラベル無し）

凍結分類（full k 次元）:

| method | k | mean cos² | min cos² | class |
|--------|---|-----------|----------|-------|
| scale_before | 32 | 0.847 | 0.002 | HIGHLY_FOLD_DEPENDENT |
| center_only | 32 | 0.859 | 0.000 | HIGHLY_FOLD_DEPENDENT |
| scale_before | 64 | 0.814 | ≈0 | HIGHLY_FOLD_DEPENDENT |
| center_only | 64 | 0.818 | ≈0 | HIGHLY_FOLD_DEPENDENT |

Leading-k 補足（center_only）: k=4/8 は mean cos²≈0.93/0.95 → **MODERATELY_VARIABLE**。末尾方向の縮退が full-32 の min cos² を潰している。

累積寄与（outer fold0）: scale-before PCA32 ≈ **88%**；center-only ≈ **93%**。

---

## 5. StandardScaler-before vs center-only

同一 outer train 比較: mean cos²≈**0.78**、min cos²≈**0.10** → 学習部分空間は明確に異なる。  
スコア分布・寄与率プロファイルも異なる（支配的 PC1 の有無）。

---

## 6. 制御付き性能（HIC / AROMATIC / ESMFold / Ridge）

Public/Private は選択に未使用。P5 は **INVALID_AS_CV_PERFORMANCE_EVIDENCE**。

| Recipe | CV MAE | vs PLM_ONLY P0 |
|--------|--------|----------------|
| PLM_ONLY P0-style | 0.4969 | — |
| PLM_ONLY P1-style | 0.4904 | — |
| **P0_CURRENT** | **0.4557** | −0.041 |
| **P1_CENTER_PCA32** | **0.4537** | −0.043 |
| P2_CENTER_PCA64 | 0.4818 | −0.015 |
| P3_RAW_ESM2 (α∈{10..10000}) | 0.4741 | −0.023 |
| P4_AROMATIC_UNSCALED | 0.4537 | P1 と同一* |
| P5_GLOBAL_DEV（漏洩） | 0.4618 | −0.035 |

\* P4=P1: concat 後の StandardScaler が aromatic 生値差を吸収。

固定α=100（P0 の全 outer で選択されたモード）の non-nested outer CV は nested と **完全一致**（Δ=0）。

---

## 7. Nested vs fixed non-nested

- 固定α=100 を P0 から回収して凍結。
- Nested の有無は **MAE に影響せず**（αが常に100のため）。
- PCA を ~100 に落とす実装ではなかったため、「nested が PCA 標本不足で悪化」仮説は **棄却**。

---

## 8. 前回融合は前処理律速か？

| 仮説 | 判定 |
|------|------|
| Aromatic scaler 不安定 | 棄却 |
| Nested が PCA を ~100 に | 棄却（実装事実） |
| Global 前処理で劇的改善 | 棄却（むしろ悪化） |
| PCA32 過剰制約（raw が良い） | 棄却 |
| scale-before vs center で大差 | MAE 上は棄却（幾何差はある） |

→ **前回の PLM+AROMATIC 融合結果は、小 fold 前処理推定の不安定が主ボトルネックだったとは言えない。**  
一方、full PCA32 部分空間の fold 依存は実在するため、次の融合では表現の扱い（固定低 DoF / leading-PC / 正則化）を意識すべき。

---

## Decision table

| Finding | Evidence | Consequence for next fusion |
|---------|----------|----------------------------|
| Aromatic scaler 安定 | flags=none；median Δz<0.2 | **A/B:** RobustScaler 必須ではない |
| Full PCA32 は fold 依存 | min cos²≈0；class HIGHLY_FOLD_DEPENDENT | **C:** 変調前に PCA32 表現を再検討する価値あり（ただし MAE 律速の証拠は弱い） |
| scale-before vs center は幾何差大・MAE差小 | cos²≈0.78；ΔMAE≈0.002 | 次実験はどちらかに **凍結**すれば足りる；切替だけでは劇的改善は期待薄 |
| Nested≠PCA~100；fixed α 一致 | 実装監査；P0 α≡100 | **D:** 次は固定 HP + outer CV で簡略化してよい |
| Global-Dev ゲインなし | P5> P0 MAE | **E 不成立**；漏洩スコアを性能証拠に使わない |
| Raw/PCA64 は PCA32 より悪い | P3/P2 > P0/P1 | PCA32 が過剰制約という主張は支持されない |

**推奨:** 変調 / FiLM へ進んでよい（aromatic 前処理は現状維持可）。ただし PLM 枝は **fold-local PCA の末尾方向が不安定**であることを前提に、固定α・低 DoF・または leading 成分重視の設計を推奨。

---

## STOP

additive/multiplicative modulation、FiLM、MLP fusion、特徴選択は未実装。

## Files

- `preprocessing_stability_audit/PREPROCESSING_IMPLEMENTATION_AUDIT.md`
- `preprocessing_stability_audit/PREPROCESSING_STABILITY_AUDIT_JA.md`
- `preprocessing_stability_audit/STABILITY_AUDIT_THRESHOLDS.json`
- `preprocessing_stability_audit/results/*`
- runner: `preprocessing_stability_audit/scripts/run_preprocessing_stability_audit.py`
