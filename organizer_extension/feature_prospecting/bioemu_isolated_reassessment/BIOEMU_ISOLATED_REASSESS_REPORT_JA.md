# BioEmu 孤立 VH/VL — 物理フレーム再評価レポート

**状態:** `ORGANIZER_TMAPP_BIOEMU_ISOLATED_REASSESS_COMPLETE`  
**最終判定:** **`BIOEMU_ISOLATED_CONFIRMED_NO_INCREMENT`**  
**対象:** TmApp のみ（HIC なし）  
**証拠区分:** ORGANIZER-EXPLORATORY  

情報科学／ML 読者向け。抗体専門知識は最小限でよい。

---

## 1. 一文サマリ

v2 の「N=16 収束」は **生 NPZ 数**に基づく誤表現だった。実測の物理通過フレームは中央値 ~5。ターゲット盲で物理フレーム収束をやり直し **Nphys=8** に凍結し、324 抗体を再サンプリング・再特徴量化したところ、記述子は安定したが **TmApp への単独／増分／残差シグナルはいずれも確認されなかった**（contact 家系も「相対的にマシ」止まりで有意ではない）。

---

## 2. 用語：raw samples vs physical frames

| 用語 | 意味 |
|------|------|
| **raw / NPZ** | BioEmu が吐いた未フィルタ構造サンプル数 |
| **physical frames** | 公式 `filter_samples=True` 通過後の `samples.xtc` フレーム数 |
| **pass rate** | physical / NPZ |

v2 の「648/648 valid」は NPZ 存在確認であり、物理通過ではない。特徴量は当初から XTC（フィルタ後）由来だったが、**使えるアンサンブルサイズの見積りが過大**だった。

---

## 3. なぜ ~5 フレームが問題だったか

**OBSERVATION**

- コホート中央値 pass ≈ 0.31 → 要求 16 NPZ から physical ≈ 5
- contact occupancy は「接触あり／なし」の経験頻度。5 フレームだと占有率は 0, 0.2, 0.4, … の粗い量子化
- Monte Carlo（同一抗体からランダム部分集合を繰り返し）で、抗体間ばらつきに対する推定ノイズ比が高い

**INTERPRETATION**

前回の「contact が相対的に有望」は、推定ノイズが大きい条件での順位であり、**真の予測シグナルとは限らない**。

---

## 4. パイロット（12 抗体）→ 物理収束

- 既存 v2 収束パイロットを再利用（配列のみ・TmApp 非使用）
- 各鎖で **≥64 physical** までチャンク過サンプリング（steering なし）
- Nphys ∈ {4,8,16,32,64}、各 N で **20 回の決定論的ランダム部分集合**（先頭 N のみに依存しない）

### 収束基準（ターゲット盲）

- median Spearman(N vs N64) ≥ 0.95
- median normalized absolute deviation ≤ 10%

### 決定

| 鎖 | 凍結 Nphys | N=8 の major MC_noise_ratio 中央値 |
|----|------------|-----------------------------------|
| VH | **8** | 0.231 |
| VL | **8** | 0.208 |
| コホート | **8** | — |

**Q1:** 主要記述子の安定化に必要な物理フレーム数は **8**（本パイロット基準）。  
**Q2:** 前回中央値 ~5 は **やや不足**（N=4 は NAD／Spearman が境界割れ；N=8 で概ね通過）。極端な高分散（N=32 でも不可）ではない。  
**Q3:** 感度が高いのは **contact occupancy / entropy**（N=8 でも MC_noise_ratio: VH 0.49 / VL 0.30）。pairwise RMSD・Rg は相対的に頑健。RMSF はその中間。

Contact 家系は N=8 でも Spearman 基準は満たすが、MC ノイズは他家系より高い。**TmApp を見て N を上げることは禁止**したため、Nphys=8 のまま凍結し、contact は「残差 MC 不確かさ付き」で報告する（`BIOEMU_ISOLATED_NPHYS_DECISION.md`）。

---

## 5. フィルタ選択バイアス（孤立ドメイン）

パイロットで RAW を再構築し PASS vs FAIL を比較。

**OBSERVATION**

- 本パイロット抗体は pass が高い（中央値 ~0.98；コホート全体の ~0.31 とは異なる）
- FAIL 側に **極端な CA ギャップ**（最大 ~139 Å）が混じる
- 典型的 FAIL の Rg 中央値差はほぼ 0（|ΔRg|/Rg ≈ 0.7%）

**INTERPRETATION**

孤立ドメインでは公式フィルタは主に **明白な物理破綻（鎖切れ・衝突系）**を落としており、PASS 側の大局的 Rg 分布を大きく歪めるタイプの強い選択ではない。ただしコホート低 pass 抗体では棄却率が高く、**通過集団＝サンプラーの物理的尾部**である点は変わらない。未通過フレームをアンサンブルとして使ってはいけない。

---

## 6. フルコホート再サンプリング

- 凍結 Nphys=8、既存 v2 NPZ をコピー再利用、不足分のみ過サンプリング
- **324/324 抗体 × VH/VL = 648/648 SUCCESS**
- 中央値: physical≈11, NPZ≈36, pass≈0.29（VH/VL 同程度）
- 壁時間合計 ≈ 4.7 GPU-h（極端ではないため自動継続）

特徴量は各鎖の **先頭 Nphys=8 physical フレーム**（決定論的部分集合）から再計算 → `BIOEMU_ISOLATED_REASSESS_FEATURES.csv`（324 行）。

---

## 7. TmApp 評価（凍結 Primary / Shadow / AbLang2）

手続きは foundation_stability_v2 と同じ: fold-local StandardScaler + Ridge（ネスト alpha）、AbLang2 PCA32、paired bootstrap B=10000。

### 7A. Standalone（vs median）

| family | CV MAE | ΔMAE vs median | 95% CI | p_improve |
|--------|--------|----------------|--------|-----------|
| NEW_CONTACT | 3.359 | −0.079 | [−0.235, 0.078] | 0.84 |
| NEW_FLEX | 3.427 | −0.012 | [−0.126, 0.100] | 0.58 |
| NEW_COMBINED | 3.463 | +0.025 | [−0.119, 0.170] | 0.37 |
| OLD_BIOEMU | 3.464 | +0.026 | [−0.116, 0.171] | 0.36 |

いずれも CI が 0 をまたぎ、**単独シグナルなし**。

### 7B. Incremental（AbLang2 + BioEmu vs AbLang2）

| family | CV MAE | Δ vs PLM | 95% CI | p_improve |
|--------|--------|----------|--------|-----------|
| NEW_CONTACT | 2.997 | −0.031 | [−0.166, 0.104] | 0.67 |
| NEW_PAIRWISE | 2.987 | −0.042 | [−0.118, 0.035] | 0.86 |
| NEW_COMBINED | 3.026 | −0.003 | [−0.146, 0.139] | 0.52 |
| OLD_BIOEMU | 3.088 | +0.060 | [−0.076, 0.193] | 0.19 |

**増分なし**（contact も有意でない）。

### 7C. Residual（AbLang2 残差への当てはめ）

同様に contact が相対最良（Δ≈−0.047）だが CI は 0 を含む。**残差シグナルなし**。

### 7D. OLD vs NEW

| 比較 | 結果 |
|------|------|
| Standalone MAE | NEW 3.463 vs OLD 3.464（Δ≈−0.001, CI ゼロ横断） |
| 特徴量 Spearman（共通列） | 中央値 ~0.68（RMSD 高相関、RMSF はややドリフト） |

**OBSERVATION:** 物理フレームを ~5→8 に揃えても予測 MAE は実質不変。  
**INTERPRETATION:** 前回ネガティブは「サンプリング不足だけで偽陰性」だったわけではなく、**孤立 VH/VL BioEmu 記述子自体がこの設定で TmApp 増分を持たない**方向の証拠が強まった。

---

## 8. 中央質問への回答

| Q | 回答 |
|---|------|
| Q1 必要物理フレーム数 | 主要家系は **Nphys=8** で目標盲基準を満たす |
| Q2 ~5 は小さすぎたか | **やや小さい**（N=4 不合格寄り；実用上は 8 へ） |
| Q3 ノイズ感度 | **contact ≫ RMSF > RMSD/Rg** |
| Q4 シグナル | standalone / incremental / residual いずれも **なし** |
| Q5 contact 家系 | ノイズ低減後も **相対最良だが null のまま**（強化も明確な弱体化もせず、結論は「確認された増分なし」） |

---

## 9. 解釈ルールとの対応

ユーザー定義のアウトカム:

- ~~A. 再サンプリングで再現可能シグナルが現れた~~ → なし
- **B. 記述子は安定したが予測は null** → **これに該当**
- ~~C. 多数フレームでも収束しない~~ → 該当せず（Nphys=8 で major 収束）

よって最終ラベル:

# `BIOEMU_ISOLATED_CONFIRMED_NO_INCREMENT`

**HYPOTHESIS（拡張しない）:** Fab 全体の熱力学やドメイン間界面が TmApp に効くなら、孤立 VH/VL だけでは足りない可能性がある。本結果は Fab 熱力学の直接反証ではない（孤立ドメイン・BioEmu 限定）。

---

## 10. 先行記録の訂正

`BIOEMU_V2_INTERPRETATION_CORRECTION.md` を参照。

- 「N=16 収束」は **物理アンサンブル収束の主張としては無効**
- v2 の予測結果は「当時のフィルタ後フレームから計算した特徴」としては歴史的に有効
- 推定精度の主張だけを訂正（成果物の上書きはしない）

---

## 11. 成果物一覧

`organizer_extension/feature_prospecting/bioemu_isolated_reassessment/`

- SPEC: `BIOEMU_ISOLATED_REASSESS_SPEC.md` / `.json`
- 収束: `BIOEMU_ISOLATED_PHYSICAL_CONVERGENCE.csv`, `BIOEMU_ISOLATED_MC_NOISE.csv`, `BIOEMU_ISOLATED_FILTER_BIAS.csv`, `BIOEMU_ISOLATED_NPHYS_DECISION.md`
- 特徴: `BIOEMU_ISOLATED_REASSESS_FEATURES.csv`
- 評価: `*_STANDALONE/INCREMENTAL/RESIDUAL/BOOTSTRAP.csv`, `BIOEMU_ISOLATED_OLD_VS_NEW.csv`
- 訂正: `BIOEMU_V2_INTERPRETATION_CORRECTION.md`
- 本レポート

**実施しなかったもの:** VL+CL / VH+CH1 / FeNNix-on-Fab / AbLingua / 物理ステアリング本格採用。

---

## 12. OBSERVATION / INTERPRETATION / HYPOTHESIS（分離）

**OBSERVATION**

1. v2 の usable N は中央値 ~5 physical だった。  
2. Nphys=8 で主要記述子は N64 と高相関・低 NAD。  
3. 324×2 全て Nphys≥8 を確保し特徴を再計算した。  
4. TmApp CV で NEW BioEmu は median / AbLang2 / OLD のいずれに対しても有意改善なし。  

**INTERPRETATION**

孤立 VH/VL BioEmu アンサンブル記述子は、推定ノイズを実用レベルまで下げても、本パイプライン（Ridge + 凍結 CV + AbLang2 ベースライン）では **TmApp 特徴源として増分を与えない**。

**HYPOTHESIS**

界面・定常ドメインを含む文脈サンプリングや別の構造生成器が必要かもしれないが、本タスク範囲外であり未検証。
