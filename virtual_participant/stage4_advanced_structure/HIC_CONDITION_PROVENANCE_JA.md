# HIC 実験条件 — provenance（Stage 4 electrostatics 用）

## 結論（先に）

本コンペの参加者配布物には、**Shehata et al. 2019 の HIC について buffer pH / salt / ionic strength の完全な数値プロトコルは明記されていない**。

したがって Stage 4 の静電計算は **assay-matched electrostatics とは呼ばない**。  
代わりに、関連する公開 HIC 文献で広く使われる条件を参照した **generic fixed-condition electrostatic descriptors** として扱う。

## 参加者側で確認できたこと

| 項目 | 内容 | source |
|---|---|---|
| ターゲット定義 | IgG の HIC retention time（分） | `competition/participant/README.md` §7、`DATA_DICTIONARY.md` |
| プロトコル依存性 | カラム・移動相・塩・グラジエント等に依存 | 同上 |
| 原研究 | Shehata et al., *Cell Reports* (2019) | participant README 引用 |
| 正確な pH / [salt] | **配布 README / DATA_DICTIONARY には数値なし** | 上記 |

## 公開文献からの参考（assay-matched ではない）

Adimab 系 / Jain らの近縁 HIC セットアップで公開されている例:

- Mobile phase A: **1.8 M ammonium sulfate**, **0.1 M sodium phosphate**, **pH 6.5**
- Mobile phase B: **0.1 M sodium phosphate**, **pH 6.5**
- 例: Jain et al. related methods / Estep et al. 系の記述（公開 paper・二次資料）

これは Shehata 2019 の本データセット条件と同一であるとは確認できない。

## Stage 4 で固定した計算条件

| parameter | value | 理由 |
|---|---|---|
| PROPKA / PDB2PQR pH | **6.5** | 上記公開 HIC 系で頻出。事前固定（Optuna 対象外） |
| APBS ionic strength | **0.15 M** (1:1 salt) | generic physiological screening。高塩 HIC と同定しない |
| dielectric | protein 2.0 / solvent 78.54 | APBS 慣例、全抗体統一 |
| grid / mg-auto | 全抗体同一 input テンプレート | reproducible |

pH や I を target score で探索しない。感度解析の大規模スキャンもしない。

## 命名

- 使用する語: **generic fixed-condition electrostatic descriptors (pH 6.5, I=0.15 M)**
- 使用しない語: assay-matched APBS / Shehata-buffer electrostatics
