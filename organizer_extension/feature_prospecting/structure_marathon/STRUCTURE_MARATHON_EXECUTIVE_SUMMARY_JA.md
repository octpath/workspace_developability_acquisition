# Structure Marathon — エグゼクティブサマリー（日本語）

更新時刻（UTC）: 2026-09-06T01:45 前後（自律キャンペーン途中時点）

## 結論（先に）

- **TmApp**: 成熟 PLM incumbent を明確に上回る構造表現は **未確認**（最良でも CI が 0 を跨ぐ微増）。
- **HIC**: 既存の露出芳香族 + PLM 解を上回る新規構造表現は **未確認**（いずれも incumbent 対比で悪化寄り）。
- **FeNNix full-Fab**: パイロット 12 の B/C/M 実行中（本サマリー時点で 8–9/12）。TmApp スコアリングは未実施。

## 結果表

| family | TmApp (INC vs incumbent ΔMAE) | HIC (INC vs incumbent ΔMAE) | increment? | robustness | recommendation |
|--------|-------------------------------|-----------------------------|------------|------------|----------------|
| S1_PATCH_NEIGHBORS | +0.027 (CI cross 0) | −0.022 | 否（弱い候補） | ESMFold のみ | 弱信号として記録のみ |
| S1_CDR_ALL | +0.004 | −0.023 | 否 | ESMFold | 停止候補 |
| S1_STRONGLY_EXPOSED_AROMATIC | −0.006 | −0.024 | 否 | — | HIC 仮説は冗長 |
| S2_disagreement | −0.009 | −0.029 | 否 | 未複製 | 優先度下げ |
| S3_surface_graph | −0.050 | −0.017 | 否 | — | HIC 単独は弱い |
| S4_contact_graph | −0.041 | −0.020 | 否 | — | 停止候補 |
| M1_ProteinMPNN | −0.017 | −0.030 | 否 | — | 追加設計不要 |
| M2_ESM_IF1 | +0.003 | −0.020 | 否 | — | 低優先 |
| M3_SaProt35 | −0.017〜−0.051 | −0.019〜−0.023 | 否 | — | 停止候補 |
| FUSION (TmApp top3) | +0.027 | — | 否（単体と同程度） | — | 融合メリットなし |
| T1 SPURS | — | — | — | — | DEFERRED_TECHNICAL |
| M4 ProSST | — | — | — | — | DEFERRED_TECHNICAL |
| T2 ThermoMPNN | — | — | — | — | DEFERRED_TECHNICAL |
| G1 GearNet | — | — | — | — | SKIP_DEFERRED |

## 明示回答

- **Best TmApp structural candidate:** `S1_PATCH_NEIGHBORS`（ΔMAE≈+0.027、bootstrap CI は 0 を含む → **WEAK_SIGNAL / NO_SIGNAL 境界**）
- **Best HIC structural candidate:** なし（いずれも incumbent を下回る）。相対的にマシなのは `S1_EXPOSED` / `S3` だが負。
- **Best negative result:** HIC 向け「強く露出した芳香族への ESM2 プーリング」は incumbent（ESM2+AROMATIC-TOPO）を超えず **REDUNDANT / NO_INCREMENT**
- **Most promising follow-up:** FeNNix full-Fab の DELTA_ENV（準備・監査完了後）。構造マラソン側は新規スカラー再発明を止め、Fab 文脈の逆折りたたみ差分に限定。
- **Models technically blocked:** SPURS, ProSST, ThermoMPNN, GearNet
- **FeNNix-Fab status:** パイロット B/C/M 実行中（完了後に技術 QC → ADI-47317 CPU → CPU/CUDA 監査）
- **GPU/CPU runtime:** ProteinMPNN ~1 min; SaProt ~42 s; ESM-IF1 ~10 min (CPU); FeNNix pilot 進行中（抗体あたり数十分）
- **Host instability:** 本セッションでは新規ハングなし。CUDA 連続 OpenMM prep は再開していない。

## 最終状態（暫定）

- Structure Marathon: `ORGANIZER_STRUCTURE_MARATHON_PARTIAL_COMPLETE`（必須 P0/P1 主要は完了、任意モデルは defer）
- FeNNix: パイロット未完了のため `RUNNING_PILOT` / 完了後に更新
