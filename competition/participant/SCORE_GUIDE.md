# How good is my score?

English first; Japanese follows.  
Approximate **local cross-validation** landmarks for this competition only.  
Not universal developability thresholds. Exact organizer model details are not disclosed.

---

## Local-CV landmarks

| Level | TmApp MAE | HIC MAE |
|---|---:|---:|
| Baseline-like | ~3.5 °C | ~0.52 min |
| Clear predictive signal | <~3.3 °C | <~0.50 min |
| Strong benchmark-level | ~2.8–3.0 °C | ~0.45–0.48 min |
| Exceptional vs current benchmark | <~2.5 °C | <~0.40 min |

First practical targets: beat **~3.5 °C** (TmApp) and **~0.52 min** (HIC) in reliable local CV.

## Scientific context (short)

A **strong** score means your sequence model is learning real information about the experimentally measured property.

However, these prediction errors are still **larger** than the technical variation reported for some related DSF/HIC experiments.

Therefore a strong competition score should **not** be read as:

- “the model replaces the experiment,” or  
- “the antibody is developable.”

For very low errors: TmApp around **~1 °C** or HIC around **~0.1 min** would be scientifically remarkable because those values approach repeatability *scales* reported in some related assays. Even then, **direct experimental validation** would still be required.

MAE is an average absolute error. It does **not** mean every pair separated by that same delta can be reliably ranked.

## Public vs local CV

Public leaderboard **N = 81** is small enough that score ordering can fluctuate.  
Use reliable **local CV** rather than chasing tiny Public-score changes.

Do not treat local-CV landmarks as Public-leaderboard targets (transfer can differ, especially for TmApp).

---

# スコアの目安（日本語）

これは本コンペの **local CV** 向けのおよその目安です。  
普遍的な developability 基準ではありません。

## local CV の目安

| レベル | TmApp MAE | HIC MAE |
|---|---:|---:|
| baseline 級 | ~3.5 °C | ~0.52 min |
| 予測 signal あり | <~3.3 °C | <~0.50 min |
| 強い（benchmark 級） | ~2.8–3.0 °C | ~0.45–0.48 min |
| 現行 benchmark 超え | <~2.5 °C | <~0.40 min |

まず目指す: local CV で **~3.5 °C**（TmApp）と **~0.52 min**（HIC）を安定して下回る。

## 科学的な注意（短く）

強いスコアは、「配列モデルが実験値に関する本物の情報を学習している」ことを意味します。

一方で、これらの予測誤差は、関連する DSF / HIC 実験で報告される technical variation より **まだ大きい** ことが一般的です。

したがって、強いコンペスコアを次のように読んではいけません:

- 「モデルが実験の代わりになる」  
- 「その抗体は developable である」

非常に低い誤差（TmApp ~1 °C や HIC ~0.1 min）は、関連アッセイの再現性スケールに近づくため科学的には注目に値します。それでも **実験的検証** は必要です。

MAE は平均絶対誤差であり、その数値差ですべてのペアを区別できる意味ではありません。

## Public と local CV

Public は N=81 と小さく、順位が揺れます。  
小さな Public 変動を追うより、信頼できる **local CV** を使ってください。

local CV の目安を、そのまま Public 目標値として使わないでください。
