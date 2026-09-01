# Stage 3 — Structure Cache Audit

参加者側で利用可能な予測立体構造キャッシュの監査。organizer score / ranking は参照していない。

## 概要表

| structure source | N | H/L chains | sequence match | complete | participant-safe | reusable | notes |
|---|---:|---|---|---|---|---|---|
| ESMFold_native | 162 | A=VH,B=VL (mapped by length/seq) | 162/162 exact VH+VL | 162/162 | yes (Dev seq → ESMFold, label-independent) | yes | path=/workspace_developability_acquisition/esmfold_native/{id}.pdb; also mirrored under gate_b2 hash names |
| ABodyBuilder2 | 162 | H/L (IMGT-numbered) | 162/162 usable (exact or prefix±2) | 162/162 | yes (Dev seq → ABB2, label-independent) | yes | gate_b1/cache/structures/abodybuilder2 + abb2_manifest.csv |

## ESMFold_native
- path: `/workspace_developability_acquisition/esmfold_native/{id}.pdb`
- exact VH+VL match: 162/162
- complete flag: 162/162
- malformed: 0

## ABodyBuilder2
- manifest: `/workspace_developability_acquisition/gate_b1/cache/structures/abb2_manifest.csv`
- complete (exact or prefix±2): 162/162
- exact match: 162/162
- malformed: 0

## 採用方針

- **Primary structure source**: ESMFold_native（Dev 162/162、ADI-named PDB、配列完全一致）
- **Secondary**: ABodyBuilder2（source ablation用）。同一特徴抽出ロジックの結果を比較する
- B1 Gly-linker ESMFold は使用しない（非ネイティブ連結）

label-independent な gate_b2 SASA/RASA/patch 特徴を再利用し、packing / Rg を Stage3 で追加計算した。
