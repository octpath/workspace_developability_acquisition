# Gate B0.5 — Low Physical-Pass Diagnostic (BioEmu VL+CL)

**Evidence:** ORGANIZER-EXPLORATORY（TmApp 未使用・target-blind）  
**Scope:** 12 LIGHT pilot Abs（`pilots/BIOEMU_CONTEXT_CONVERGENCE_ANTIBODIES.csv`）  
**Model:** `bioemu-v1.2`（チェックポイント未変更）  
**Filter:** 常時 ON（無効化なし）  
**Full 324 cohort:** **ブロック中**（本ゲート完了まで開始しない）

---

## Decision

### `BIOEMU_EXTENDED_CHAIN_UNRESOLVED`

| Option | Chosen? | Why |
|--------|---------|-----|
| `PROCEED_DEFAULT` | No | 通過率中央値 ≈9%。通過フレームは **狭い構造サブセット**（主にドメイン間非衝突＝より開いた配置）に偏る |
| `PROCEED_STEERED_WITH_CAUTION` | No | 公式 physical steering で通過率は改善（中央値 ≈32%）するが、なお多数棄却。動的記述子の A↔B 一致は弱く、**誘導後分布を「同じ平衡」とみなせない** |
| `BIOEMU_EXTENDED_CHAIN_UNRESOLVED` | **Yes** | 低通過は **多ドメイン VL+CL 特有**（孤立 VL/CL より大幅に悪い）。棄却の主因は **ドメイン間ステリック衝突**。生存 5–10% を unbiased ensemble として全コホートへ進めるのは不当 |

**Implication:** 全 324 LIGHT サンプリングは開始しない。定数ドメイン文脈の BioEmu 一次主張は保留。

---

## 1. Filter failure taxonomy（OBSERVATION）

Source: `BIOEMU_FILTER_FAILURES.csv`（各 Ab の RAW npz→未フィルタ XTC に公式マスク適用）

| Reason | Fraction (all frames, 12 Abs) |
|--------|-------------------------------|
| `steric_clash` only | **85.3%** |
| `PASS` | **9.2%** |
| `both_break_and_clash` | 5.1% |
| `chain_discontinuity` only | 0.4% |

Per-Ab pass rate range: **1.6%–14.1%**（中央値 ≈9.5%）。

**INTERPRETATION:** 主因は鎖切断ではなく **ステリック衝突**。連続性単独の失敗は稀。

---

## 2. Failure localization（OBSERVATION）

Among failures:

| Clash locus | Count |
|-------------|------:|
| `interdomain_clash` | **587**（最多） |
| `VL_internal_clash` | 246 |
| `CL_internal_clash` | 104 |
| `none`（ほぼ純切断） | 4 |

Among `steric_clash` only: interdomain **552 / 884**.

Chain-break locus (when discontinuity present): VL または CL；**VL–CL junction 単独の優勢は見られない**。

**INTERPRETATION:** フィルタが落とす主因は **V–C ドメイン間の原子衝突**。多ドメイン相対配置の不安定／非物理的重なりがボトルネック。

---

## 3. Isolated VL vs CL vs VL+CL（OBSERVATION）

Target-blind subset: 4 Abs（κ2 + λ2）、各 64 raw、同一プロトコル。  
`results/B05_DOMAIN_COMPARE_PASS.csv`

| Construct | Median pass rate | Mean |
|-----------|------------------|------|
| **CL** alone | **0.58** | 0.58 |
| **VL** alone | **0.38** | 0.40 |
| **VL+CL** | **0.10** | 0.11 |

例 ADI-47230: VL 37.5% / CL 85.9% / VLCL **9.4%**。

**Main answer:** 低通過率は **個々のドメイン単独では説明できず、多ドメイン VL+CL サンプリングに特異的**。

---

## 4. Selection bias: RAW vs PASS（OBSERVATION）

`results/B05_RAW_VS_PASS_BIAS.csv`

PASS − FAIL（Ab ごとの中央値）:

| Metric | Median Δ (PASS − FAIL) |
|--------|-------------------------|
| Rg | **+0.06 nm**（通過の方がやや広がる） |
| VL–CL COM distance | **+0.23 nm**（通過の方がドメイン間が離れる） |
| Orientation angle | −1.9°（弱い） |
| max CA gap | −0.05 Å（通過は連続性良好＝定義通り） |

**INTERPRETATION:** 物理フィルタは「衝突しない＝ドメイン間がより開いた配置」を優先的に残す。**狭い構造テールの選択**であり、RAW 生成分布の代表ではない。

**HYPOTHESIS（未検証）:** 通過フレームは Fab 様の密な V–C 配置より、開いた／緩いエルボー側に偏っている可能性。

---

## 5–6. Default (A) vs official physical steering (B)

Config: `bioemu/config/steering/physical_steering.yaml`  
（CaCaDistance umbrella + PairwiseClash umbrella；SMC `num_particles=5`）  
Post-filter: **同一**公式フィルタ。チェックポイント変更なし。

`BIOEMU_DEFAULT_VS_STEERED.csv`

| | A default+filter | B steered+filter |
|--|------------------|------------------|
| Median pass rate | **0.094** | **0.320** |
| Mean pass rate | 0.089 | 0.297 |
| Median runtime / Ab (64 raw) | ~190 s（先行パイロット） | **~350 s** |
| Median PASS n | 6 | 20.5 |

Ensemble stats on **PASS frames only**（n≥3 の Ab）:

| Descriptor | Spearman(A,B) | Note |
|------------|---------------|------|
| ESMFold-ref Q | **0.92** | 相対順位は安定 |
| VL–CL COM mean | 0.48 | 中程度 |
| pairwise RMSD / RMSF / contact persist / orient SD | **0.22–0.39** | **弱い** |

**INTERPRETATION:** Steering は衝突回避を助け通過率を上げるが、**(1) なお多数棄却、(2) 動的アンサンブル統計は A と質的にずれうる、(3) 誘導は平衡サンプリングの保証ではない**。TmApp による A/B 選択はしていない。

---

## 7. What this means for constant-context BioEmu

| Question | Answer |
|----------|--------|
| 通過 5–10% を平衡アンサンブルとみなしてよいか？ | **No** |
| 単純オーバーサンプルで全 324 へ進むべきか？ | **No**（本ゲートで禁止を確認） |
| VL+CL は技術的に「動く」か？ | サンプリング自体は完走するが、**物理的合格率が科学的利用に不足** |
| CH1 重鎖診断との関係 | 本ゲートは LIGHT のみ。HEAVY は別途さらに厳しい可能性 |

---

## 8. Files

| File | Role |
|------|------|
| `BIOEMU_FILTER_FAILURES.csv` | フレーム単位の失敗理由・局在 |
| `BIOEMU_DEFAULT_VS_STEERED.csv` | A vs B 通過率・runtime・アンサンブル指標 |
| `BIOEMU_LOW_PASS_DIAGNOSTIC.md` | 本報告 |
| `results/B05_FILTER_FAILURE_SUMMARY.csv` | Ab 集計 |
| `results/B05_RAW_VS_PASS_BIAS.csv` | 選択バイアス |
| `results/B05_DOMAIN_COMPARE_PASS.csv` | VL / CL / VLCL 通過率 |
| `results/B05_LIGHT_PILOT_SNAPSHOT.csv` | パイロット npz/phys スナップショット |
| `scripts/gate_b05_diagnostic.py` | 再現スクリプト |

---

## 9. Operational note

物理 64 フレーム到達を目指すオーバーサンプルは、本ゲート指示（「数千 raw を生き残りの 5–10% として扱うな」）に従い **停止**。Gate B0 の 12×≥64 **unfiltered** LIGHT パイロットは診断入力として完了扱い。

**Next（自動では進めない）:** 人間判断後にのみ、別プロトコル（例: ドメイン条件付き、別構造法、Fab 非モノマー手法）を検討。AbLingua / FeNNix-on-Fab / 全 324 BioEmu は自動開始しない。
