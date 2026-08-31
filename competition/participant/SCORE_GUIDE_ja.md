# スコアの目安

English: [SCORE_GUIDE.md](SCORE_GUIDE.md)

本コンペの **local CV** 向けのおおよその目安です。普遍的な developability 基準ではありません。organizer モデルの詳細は非公開です。

---

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

Public は N=81 と小さく、順位が揺れます。小さな Public 変動を追うより、信頼できる **local CV** を使ってください。

local CV の目安を、そのまま Public 目標値として使わないでください。
