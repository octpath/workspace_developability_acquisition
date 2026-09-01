# Stage 0 — CV Design Report

**Status:** `CV_PROTOCOL_FROZEN_READY_FOR_STAGE1`

**Role boundary:** participant-safe Dev labels / sequences / distributed annotations only.
No Test labels, solution.csv, Public/Private IDs, or organizer split/model-selection materials were used.

---

## このCV設計を一言で説明すると

本Stageでは、**Heavy/Light配列が非常に似ている抗体を同一グループとしてFold間に分離しないようにしたうえで、Foldサイズ、TmAppの分布、HICの分布、HIC高値側のサンプル数が5つのFold間でできるだけ均等になるように、抗体グループのFold割当を最適化した共通5-fold CV**を採用した。

この処置には主に2つの理由がある。

1. **類似配列によるCVの過大評価を防ぐため**  
   非常によく似た抗体がTrain側とValidation側に分かれると、モデルは「未知の抗体を予測する」というより、Train中の近縁配列を手掛かりにValidationを予測できてしまう可能性がある。その場合、CVスコアが実際の未知抗体への汎化性能より楽観的になる。そこで、Heavy/Lightの両方が高い配列同一性を持つ抗体は同一sequence groupとして扱い、必ず同じFoldに配置した。

2. **Foldごとの難易度差をできるだけ小さくするため**  
   N=162とデータ数が少なく、特にHICは高値側に少数のサンプルを持つ右裾の長い分布である。そのため、単純なrandom KFoldでは、あるFoldだけTmApp/HICの分布が偏ったり、high-HIC抗体が集中したりして、FoldごとのMAEが偶然大きく変動する可能性がある。そこで、各Foldについて
   - サンプル数
   - TmAppの分布
   - HICの分布
   - HIC高値側（Train q90以上）の件数
   がなるべく揃うようにFold割当を最適化した。

TmAppとHICを別々に分割する方法も検討したが、両targetを同時に十分よくバランスできたため、今後の特徴量・モデル比較を同じValidation抗体上で行える利点を優先し、**TmApp/HIC共通の5-fold**を採用した。

なお、Sturgesの式による等幅binningも候補として検討したが、HICでは高値側binが1〜数件と疎になり、5-foldの層化には不向きだった。そのため、最終的にはSturges binを直接使うのではなく、**quantileベースの分布バランスとHIC high-tailの均等化を目的関数に含める方式**を採用した。

最終的なFold割当はモデルのMAEが良くなるように選んだものではない。Foldサイズ・target分布・high-tail・sequence groupのバランスを基準に決定し、モデルはその安定性を診断する目的にのみ使用した。


## Freeze summary

| Item | Value |
|---|---|
| Primary CV | `opt_joint_group_k5_s42` |
| Shadow CV | `opt_joint_group_k5_s2026` |
| Common vs target-specific | **common (adopted)** |
| n_folds | 5 |
| Primary seed | 42 |
| Shadow seed | 2026 |
| Sequence group rule | min(VH,VL) identity ≥ 0.9; groups atomic |
| HIC high-tail rule | Train-only HIC ≥ q90 (= 10.5372 min); n=17 |

Files:

- `cv_primary.csv` / `cv_shadow.csv` — columns `id,fold`
- `cv_candidate_summary.csv`
- `cv_fold_statistics.csv`
- `cv_stability_results.csv`
- `CV_FREEZE_MANIFEST.json`
- plots under `plots/`

---

## 1. TmApp/HIC共通foldは成立したか？

**はい。共通 Primary / Shadow を採用した。**

共同fold最適化（group-aware + joint target balance）により、

- size imbalance = 0.0148
- TmApp quantile imbalance = 0.0138
- HIC quantile imbalance = 0.0173
- HIC high-tail imbalance = 0.0132

を得た。

対比:

- 最良TmApp単体stratの tm_q_imb ≈ 0.0195（候補 `strat_sturges_tm_k4_s42`）
- 最良HIC単体stratの hic_tail_imb ≈ 0.0086（候補 `strat_hybrid_hic_k4_s42`）

共通foldの不均衡は、単体最適に対して **致命的に悪化していない**（相対閾値チェック `common_ok=True`）。
したがって Priority 1（common）を採用。target-specific foldは作らない（fallback監査用に `artifacts/` へ残す場合あり）。

---

## 2. Sturges binningは実際に安定だったか？

**HICでは不安定。TmAppでは許容範囲だが第一選択ではない。**

N=162 → Sturges k = 9。

HIC equal-width occupancy（疎な上側）:

```
{
  "0": 59,
  "1": 59,
  "2": 15,
  "3": 9,
  "4": 7,
  "5": 6,
  "6": 1,
  "7": 5,
  "8": 1
}
```

上側binが 1〜数件に割れ、StratifiedKFoldの層として機能しにくい。
TmApp occupancy:

```
{
  "0": 4,
  "1": 8,
  "2": 17,
  "3": 45,
  "4": 36,
  "5": 31,
  "6": 15,
  "7": 4,
  "8": 2
}
```

端binは疎だがHICほど極端ではない。

---

## 3. Quantile binningの方が良かったか？

**はい。特にHICで明確。**

HIC qcut8 occupancy:

```
{
  "0": 21,
  "1": 20,
  "2": 20,
  "3": 21,
  "4": 19,
  "5": 20,
  "6": 20,
  "7": 21
}
```

各binのサンプル数が揃い、層化CVの前提を満たす。
最終Primaryは単純な単一target quantile stratではなく、**quantile imbalanceをobjectiveに含む joint最適化**を使った（quantile思想を保持しつつ両targetを同時に扱う）。

---

## 4. HIC high tailをどう扱ったか？

Train分布のみで定義:

- high-tail = HIC ≥ Train q90 = **10.5372 min**
- 該当件数 = **17**

扱った方法:

1. Hybrid binning候補: 中央をquantile、上側tailを独立bin（`strat_hybrid_hic_*`）
2. Joint最適化の loss に `w_tail * HIC high-tail proportion imbalance` を明示加算

Public/PrivateやTestの既知閾値は使っていない。
READMEの解釈帯（>11.5）は参考知識として認識したが、fold設計の定義には **Train分位点のみ** を採用。

Primaryの fold別 high-tail件数:

```
[
  {
    "fold": 0,
    "n": 32,
    "n_high_tail": 3
  },
  {
    "fold": 1,
    "n": 33,
    "n_high_tail": 4
  },
  {
    "fold": 2,
    "n": 33,
    "n_high_tail": 4
  },
  {
    "fold": 3,
    "n": 32,
    "n_high_tail": 3
  },
  {
    "fold": 4,
    "n": 32,
    "n_high_tail": 3
  }
]
```

---

## 5. sequence groupingは必要だったか？

**予防的に必要（実害は小さいが atomic 化はコストが低い）。**

- グループ数: 161
- 複数メンバーgroup: 1（メンバー合計 2）
- 閾値: min(VH,VL) identity ≥ 0.9

Dev内の高類似ペアはごく少数（ほぼ1ペア）。したがって grouping 有無でスコアが大きく変わる可能性は低い。
しかし Stage1以降の特徴量で近傍漏洩が楽観バイアスを生むリスクがあるため、**groupをatomic unitとして必ず守る**規則をfreezeした。

Multi-member groups:

```
[
  {
    "id": "ADI-47091",
    "seq_group": 66,
    "TmApp": 74.5,
    "HIC": 8.959
  },
  {
    "id": "ADI-47096",
    "seq_group": 66,
    "TmApp": 74.5,
    "HIC": 9.027
  }
]
```

---

## 6. random KFoldとの差は？

同一fold数での比較（Primary seed系）:

| Protocol | balance_loss | hic_tail_imb | stab_score (mean fold-MAE SD) |
|---|---:|---:|---:|
| plain KFold | 0.9247 | 0.0282 | 0.2289 |
| shuffled KFold | 0.7529 | 0.0759 | 0.2795 |
| **Primary (opt joint group)** | **0.2636** | **0.0132** | **0.1861** |

random KFoldは実装が単純だが、HIC tailの偏りとjoint target balanceが劣りやすい。
Primaryは **モデルMAEを直接最適化せず**、分布バランス＋安定性診断で選んでいる。

---

## 7. 4/5/6-foldのうち何を選んだか？

**5-fold を選択。**

理由:

- N=162では 5-fold → validation ≈ 32/fold で、HIC high-tail（n≈17）を各foldへ分散しやすい
- 4-foldはvalが大きいが実験回数は減る一方、6-foldはvalが小さくtailカウントが0/1になりやすい
- 今後のPLM / Optuna / 構造特徴の計算コストを考えると **5-foldが現実的**
- near-bestの最適化候補の中で5-foldを優先する規則を事前に置いた

---

## 8. Primary CVをなぜ選んだか？

候補族を比較し、次を満たすものを選んだ:

1. TmApp/HIC **共通** fold
2. sequence group を分割しない
3. joint balance loss（size / quantile / mean-median / Wasserstein / HIC tail）が良好
4. 単純モデルの fold-MAE SD が相対的に安定
5. 5-foldを優先

選ばれた Primary: `opt_joint_group_k5_s42`

- family: `optimized_joint_group`
- balance_loss: 0.2636
- selection_score: 0.2915
- notes: greedy+swap joint balance loss=0.2636

**選んでいない理由の明示:** Ridge/ElasticNetの mean MAE が最小だから、ではない。

---

## 9. Shadow CVはPrimaryとどれくらい異なるか？

Shadow: `opt_joint_group_k5_s2026`（seed=2026）

- pairwise co-membership disagreement = **0.316**
  （「同じvalidation foldに共起するか」がPrimaryと異なるペアの割合）
- Shadow balance_loss = 0.2355
- Shadow hic_tail_imb = 0.0132

用途: Optunaの主目的には使わない。Primaryで見えた改善が Shadowでも再現するかの監査用。

---

## 10. simple baselineのCV score varianceはどの程度か？

### Primary — TmApp
  - median: mean MAE=3.4374, fold SD=0.1996, worst=3.6667, folds=[3.34375, 3.3484848484848486, 3.6666666666666665, 3.625, 3.203125]
  - ridge_seq: mean MAE=3.2778, fold SD=0.4126, worst=3.9475, folds=[3.163663631916337, 2.832838286459697, 3.94748193809805, 3.1344458338866703, 3.31045501068021]
  - elastic_ann: mean MAE=3.3035, fold SD=0.3802, worst=3.7736, folds=[3.3547163876860804, 2.8509903027149295, 3.5406115387672026, 3.773608495903459, 2.9974877968702356]

### Primary — HIC
  - median: mean MAE=0.5180, fold SD=0.0230, worst=0.5355, folds=[0.4938437499999998, 0.5345757575757575, 0.5343030303030303, 0.491625, 0.53546875]
  - ridge_seq: mean MAE=0.5876, fold SD=0.0357, worst=0.6282, folds=[0.6062036981345542, 0.5717175350611763, 0.6281582189483372, 0.5968510621603227, 0.5350264121894919]
  - elastic_ann: mean MAE=0.5399, fold SD=0.0656, worst=0.6130, folds=[0.5472238997872243, 0.5919995037572738, 0.6129575958283805, 0.4607737564702333, 0.4867777125526763]

fold-MAE SD（特にHIC）はゼロではないが、random splitより制御された範囲。
Stage0の目的は性能向上ではないため、この分散は **物差しの安定性診断** として記録する。

---

## 11. 今後のOptuna/model selectionに十分安定した物差しと言えるか？

**条件付きで Yes — Stage1の主評価軸としてfreezeする。**

根拠:

- 共通5-foldで両targetの分布が大きく破綻していない
- HIC high-tailを明示的に分散
- sequence-group漏洩をatomicに遮断
- Shadow CVでPrimary過学習を検知可能
- 単純モデルでseed/assignment感度を確認済み

限界（自覚）:

- N=162 / HIC tail少数のため、どんなCVでもfold噪声は残る
- 本CVはPrivate性能を保証しない（保証不能）
- 今後は Primary ΔMAE と Shadow ΔMAE の一致を見て改善を採否する

---

## Candidate leaderboard (top by selection_score)

```
                    name                    family  n_folds  balance_loss  selection_score  hic_tail_imb  tm_q_imb  size_imb  stab_score_fold_mae_sd  tm_ridge_fold_sd  hic_ridge_fold_sd
  opt_joint_group_k4_s42     optimized_joint_group        4      0.210283         0.235121      0.008567  0.015865  0.012346                0.165589          0.172290           0.108601
opt_joint_group_k4_s2026     optimized_joint_group        4      0.208838         0.246395      0.008567  0.014041  0.012346                0.250379          0.681364           0.052751
 opt_joint_group_k5_s123     optimized_joint_group        5      0.235233         0.255871      0.013223  0.015241  0.014815                0.137582          0.369628           0.027823
opt_joint_group_k5_s2026     optimized_joint_group        5      0.235480         0.260341      0.013223  0.017755  0.014815                0.165739          0.396034           0.073201
   opt_joint_group_k5_s7     optimized_joint_group        5      0.258020         0.290352      0.013223  0.016653  0.014815                0.215545          0.470833           0.064059
  opt_joint_group_k5_s42     optimized_joint_group        5      0.263557         0.291475      0.013223  0.013751  0.014815                0.186120          0.412647           0.035682
  opt_joint_group_k6_s42     optimized_joint_group        6      0.258366         0.297813      0.010288  0.020165  0.000000                0.262981          0.586460           0.076419
opt_joint_group_k6_s2026     optimized_joint_group        6      0.287470         0.318233      0.010288  0.020988  0.000000                0.205087          0.310718           0.043077
     strat_q8_hic_k4_s42 continuous_strat_quantile        4      0.291045         0.318583      0.018445  0.033277  0.012346                0.183590          0.226388           0.043792
 strat_sturges_tm_k4_s42  continuous_strat_sturges        4      0.322265         0.352332      0.017226  0.019502  0.012346                0.200449          0.445640           0.053662
 stratgroup_joint_k4_s42          stratified_group        4      0.333330         0.361075      0.022439  0.019817  0.012346                0.184965          0.388659           0.045404
      strat_q8_tm_k4_s42 continuous_strat_quantile        4      0.349985         0.376300      0.017226  0.026223  0.012346                0.175429          0.399103           0.070950
```

---

## Method notes

### CV families compared

1. plain KFold
2. shuffled KFold
3. continuous-target stratification (Sturges / quantile / HIC hybrid-tail)
4. sequence-group-aware assignment
5. stratified-group and **optimized joint group** (greedy + local swap)

### Optimization objective (not model score)

```
loss = w_size * size_imbalance
     + w_tm  * (TmApp quantile imb + 0.5*mean/median imb + 0.5*Wasserstein imb)
     + w_hic * (HIC quantile imb + 0.5*mean/median imb + 0.5*Wasserstein imb)
     + w_tail * HIC_high_tail_imbalance
```

### Baselines used only for stability diagnosis

1. Median predictor
2. Simple sequence descriptors + Ridge
3. Distributed annotations + ElasticNet

---

## Stop condition

Stage 0完了。PLM / ESMFold / 構造特徴 / Optuna本探索には進まない。

**`CV_PROTOCOL_FROZEN_READY_FOR_STAGE1`**
