# Inverse Folding Cache Audit

## 調査結果

| artifact | location | participant-safe? | decision |
|---|---|---|---|
| ESM-IF pretrained weights | fair-esm hub download | yes（public pretrained） | **新規計算に使用** |
| ProteinMPNN scripts in `/root/FLAb` | FLAb models | 設計が IgFold 再予測寄り・依存が重い | Stage4 では **SKIP** |
| 既存 Dev ESM-IF OOF / NLL cache | workspace 内に該当なし | — | なし |
| organizer OOF / ranking | — | **使用禁止** | 未参照 |

## 採用経路

- Model: **ESM-IF1** (`esm_if1_gvp4_t16_142M_UR50`)
- Input: Stage3 と同じ `esmfold_native/{id}.pdb`（必要なら ABB も感度比較）
- Mode: **native sequence scoring**（大量 design はしない）
- Chains: A=VH, B=VL を個別スコアし、Fv 平均・鎖別・分散を集約
- Env: `.venv_esmif`（biotite 0.41.2 + torch-scatter）。推論は CPU（device mismatch 回避）

## 再利用ポリシー

Dev PDB と native sequence のみから label-independent に計算した raw LL / NLL のみを特徴に使う。
