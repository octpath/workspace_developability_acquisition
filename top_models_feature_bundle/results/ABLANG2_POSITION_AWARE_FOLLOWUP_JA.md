# AbLang2 × antibody-position-aware Transformer follow-up（日本語）

**Status:** COMPLETE  
**Target:** TmApp only  
**Selection:** Primary/Shadow CV only（Public/Private は POSTMORTEM）

## 要旨

AbLang2 residue を annotation-aware Transformer で再集約する仮説を、事前登録の4変種＋Top-3 fusion＋cross-family 再実行で検証した。

**結論ラベル:** `PARTIALLY_SUPPORTED`（annotation 効果は Shadow 側に限定） / `NO_INCREMENT`（sequence-only・旧 TMF2 fusion を下回る） / `CROSS_FAMILY_INCREMENT`（極小） / region gate は `REGION_SIGNAL_ONLY` ではなく **性能非改善**。

全体としての科学的判定は **PARTIALLY_SUPPORTED〜INCONCLUSIVE**（AbLang2 が TmApp で強い理由の「位置 annotation 再集約」だけでは説明しきれない）。

---

## Q1 — AbLang2 residue TF vs AbLingua TMF2（sequence-only）

| model | Primary | Shadow | worst |
|-------|---------|--------|-------|
| AL2F3_FULL_MEAN（best AbLang2） | 2.994 | 3.043 | **3.043** |
| AL2F2_FULL_CONCAT | 3.195 | 3.108 | 3.195 |
| TMF2（AbLingua frozen full concat; Phase-1） | （master CSV 参照） | | より悪い〜同程度帯 |

AbLang2 best（mean merge）は sequence-only として AbLingua TMF 帯より良いが、**linear T1（worst≈2.785）には遠く及ばない**。

## Q2 — annotation（AL2F1 → AL2F2）

| | delta (F1−F2; +は full が良い) |
|--|--|
| Primary | **−0.038**（full が悪化） |
| Shadow | **+0.211**（full が改善） |
| worst | **+0.124**（full の worst が改善） |

Primary/Shadow **両方の改善ではない**。annotation 増分は **Shadow に偏る**。

## Q3 — concat vs mean（AL2F2 vs AL2F3）

mean（AL2F3）が明確に優勢（worst 3.195 → **3.043**）。H/L concat より mean merge が安定。

## Q4 — region-gate（AL2F4）

worst **3.294**（AL2F2 より悪化）。性能改善なし。

## Q5 — learned region weights（descriptive）

概ね **≈0.25 一様**。Heavy で CDR2 がわずかに高い（≈0.255）。**CDR3 が突出して高い事実はない**。seed/fold std は小さく安定。因果的重要度とは解釈しない。

## Q6 — 「AbLingua が CDR を見ていた」仮説

**NOT_SUPPORTED〜INCONCLUSIVE**。region-gate は CDR3 偏重を示さず、性能も改善しない。

## Q7 — sequence-only vs fusion

sequence-only は弱い。fusion（特に BIOEMU+MPNN）で linear 近傍まで戻るが、**旧 TMF2⊕BIOEMU fusion（worst≈2.773）には届かない**（新 best fusion worst≈2.835）。

## Q8 — residual complementarity

AL2F3 vs linear T1 residual Pearson ≈ **0.85 / 0.77**（P/S）。一定の相補性はあるが、単独では弱いため ensemble 寄与は限定的。

## Q9 — cross-family vs 2.674/2.738

Phase-1: `CROSS_FAMILY_EQUAL_MEAN__TmApp__2m` = T1 ⊕ TMF2-fusion → worst **2.7379**。

Follow-up equal-mean 再実行（family winners）: T1 ⊕ XGB ⊕ TMF2-fusion ⊕ AL2F3 → worst **2.7348**。

**極小の CV 改善あり（winner_changed=YES）**だが実用的インパクトは小さい。

## Q10 — Public/Private POSTMORTEM

| model | Public | Private | Overall |
|-------|--------|---------|---------|
| Phase-1 cross-family | 3.170 | 3.261 | 3.215 |
| Follow-up cross-family（同上 members+…） | （finalize 参照） | | |
| Best AbLang2 sequence | 3.522 | 3.252 | 3.387 |
| Best AbLang2 fusion | 3.142 | 3.236 | 3.189 |

CV の極小改善が Test で大きな勝ち越しになる証拠はない。sequence-only は Public が特に悪い。

---

## Verdict labels（使用）

- `ANNOTATION_SIGNAL_ONLY`（Shadow 側）
- `NO_INCREMENT`（sequence-only / vs 旧 fusion）
- `CROSS_FAMILY_INCREMENT`（極小）
- CDR3 region-gate 仮説は **支持しない**

## License

`REVIEW_MODEL_OUTPUT`（AbLang2 由来 residue 出力の再配布は未確定）
