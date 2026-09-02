# Organizer Extension — Feature Prospecting

**状態:** `ORGANIZER_FEATURE_PROSPECTING_GATE1_1_CONTRACT_FROZEN`

Round1 Participant simulation とは独立した、主催者側の科学的探索拡張ワークスペース。

## Evidence boundary

| 区分 | 説明 |
|------|------|
| **ORGANIZER-EXPLORATORY** | 本ワークスペースの全成果物 |
| **Round1 PRIMARY** | `virtual_participant/` 配下の frozen artifact（**READ-ONLY**） |
| **Participant Round2** | 別系統（本ワークスペースとは独立） |

Public / Private は **post-hoc scientific replication** に利用。unseen-test / prospective evidence とは呼ばない。

## Gate 1 → Gate 1.1

Gate1 基本思想は維持。Gate1.1 で baseline（median）、exact nested Ridge、CANONICAL_RESIDUAL_RIDGE、WEAK/MIXED CI 規則、structure crosswalk、mechanistic/empirical relevance UX を追加。

詳細: [GATE1_1_CORRECTION_REPORT_JA.md](GATE1_1_CORRECTION_REPORT_JA.md)

## 主要成果物

| ファイル | 内容 |
|----------|------|
| [ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md](ORGANIZER_FEATURE_PROSPECTING_CONTRACT.md) | 共通実験契約 |
| [SIGNAL_CLASSIFICATION_SPEC.md](SIGNAL_CLASSIFICATION_SPEC.md) | signal / baseline / residual Ridge / verdict |
| [STRUCTURE_ROBUSTNESS_SPEC.md](STRUCTURE_ROBUSTNESS_SPEC.md) | dual-structure robustness |
| [STRUCTURE_INPUT_CROSSWALK.csv](STRUCTURE_INPUT_CROSSWALK.csv) | N=324 canonical structure paths |
| [FEATURE_FAMILY_REGISTRY.csv](FEATURE_FAMILY_REGISTRY.csv) | family backlog + mechanistic priors |
| [FEATURE_RELEVANCE_SUMMARY.csv](FEATURE_RELEVANCE_SUMMARY.csv) | family×target 「何が効きそうか」表 |
| [FEATURE_PROSPECTING_SCORE_REGISTRY.csv](FEATURE_PROSPECTING_SCORE_REGISTRY.csv) | 実験スコア schema |
| [FAMILY_REPORT_TEMPLATE.md](FAMILY_REPORT_TEMPLATE.md) | Bottom-line 付き REPORT テンプレ |
| [GATE1_FREEZE_MANIFEST.json](GATE1_FREEZE_MANIFEST.json) | SHA-256 / git freeze |

## 次のステップ

Human review 後に first-wave family の `FEATURE_SPEC.json` 凍結 → extraction → 本契約に従った評価。Gate1.1 時点では実験未開始（empirical = NOT_RUN）。
