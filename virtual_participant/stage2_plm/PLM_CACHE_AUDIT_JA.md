# Stage 2 — PLMキャッシュ監査

## 目的

Stage 2で再利用可能な、participant-safeなPLM埋め込みキャッシュを棚卸しした。
targetラベルやPublic/Private情報に依存する表現は使わない。

## 監査結果サマリー

| cache | PLM | chain | representation | N (full) | Dev 162 | participant-safe | reusable | reason |
|---|---|---|---|---:|---:|---|---|---|
| `gate_b1/cache/plm/esm1b_t33_650M_UR50S/HL_concat_mean_*.npy` | ESM-1b (650M) | H+L | whole-chain mean → concat (2560=1280+1280) | 400 | 162/162 | はい | **再利用** | 配列のみ・label非依存・raw mean-pool |
| `gate_b1/cache/plm/esm2_t33_650M_UR50D/HL_concat_mean_*.npy` | ESM-2 (650M) | H+L | whole-chain mean → concat (2560) | 400 | 162/162 | はい | **再利用** | 同上 |
| `gate_b1/cache/plm/esm2_t33_650M_UR50D/CDR6_concat_mean_*.npy` | ESM-2 (650M) | CDR×6 | CDR mean concat (7680) | 400 | 162/162 | はい | optional | Stage2主解析はwhole-chain優先のため補助 |
| `gate_b1/cache/plm/ablang2_default/HL_concat_mean_*.npy` | AbLang2 paired | paired | seqcoding (480) | 400 | 162/162 | はい | **再利用** | 配列のみ。paired表現 |
| AbLang2 Heavy-only / Light-only | AbLang2 | H or L | seqcoding (480) | — | 要生成 | はい | **Stage2で生成** | controlled ablation用。空鎖パートナーでseqcoding |
| AbLang (original) | AbLang1 | — | — | 0 | — | — | 不可 | キャッシュ未作成（download hangでskip） |
| `gate_b5_ceiling/cache/tokens/esm2_t30_150M/*.npz` | ESM-2 150M | H/L tokens | residue tokens (L,640) | 324 | 162/162 | はい | 今回不使用 | 主比較は650M系で統一 |
| `gate_b7_*/cache/**/ESM2_*.npz` 等 | — | — | OOF/test予測 | — | — | — | **禁止** | 予測値・supervised成果物 |
| Optuna PCA済み特徴 | — | — | full-data PCAの可能性 | — | — | — | **禁止** | fold内fitできない形は不使用 |

## 再利用方針

1. ESM-1b / ESM-2 の `HL_concat_mean` を主キャッシュとして使う。  
   - Heavy-only = 先頭1280次元  
   - Light-only = 末尾1280次元  
   - H+L = 全2560次元（concat）
2. AbLang2 paired 480次元を抗体特化PLMの主表現として使う。  
   - Heavy/Light ablation用に Stage2 側で H-only / L-only を追加計算し、`stage2_plm/cache/` に保存。  
   - 公平な H+L concat 比較用に `concat(H,L)=960` も用意する。
3. PCA / StandardScaler は必ず Primary/Shadow の各 training fold 内で fit する。
4. organizerのOOF予測・target依存特徴は使わない。

## 環境

- 埋め込み読込・AbLang2追加計算: `.venv_b1`（`fair-esm` / `ablang2` / `torch`）
- 以降のCVモデリングも同環境（sklearn / optuna 利用可）

## 監査結論

Stage 2のコア3 family（ESM-1b / ESM-2 / AbLang2）は既存キャッシュ＋少量の追加計算で実施可能。AbLang1はスキップする。
