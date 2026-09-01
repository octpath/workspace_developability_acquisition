# Stage 4 — Environment / External Tool Audit

## Snapshot

- OS: Linux (Ubuntu noble-based container)
- Python (Stage4 feature env): 3.12.13 (`.venv_stage4`)
- Python (ESM-IF): 3.12.13 (`.venv_esmif`)
- uv: 0.11.16
- GPU: RTX 3090 + GTX 1080 Ti（ESM-IF は CPU で安定実行）
- compilers: gcc/g++/make available

詳細: `stage4_environment_snapshot.txt` / `stage4_install_log.txt`

## Tool status

| tool | version | status | purpose | failure/fallback |
|---|---|---|---|---|
| PROPKA | 3.5.1 | INSTALLED | pKa / charge proxy at fixed pH | Python 3.14 では壊れるため Stage4 venv は 3.12 |
| PDB2PQR | 3.7.1 | INSTALLED | protonation-aware PQR | PROPKA 依存 |
| APBS | 3.4.1 | INSTALLED | continuum electrostatic potential | apt 再試行後成功 |
| FoldX | — | BLOCKED_BY_LICENSE | energy terms | 未取得・login/license 回避せず |
| Rosetta / PyRosetta | — | SKIPPED_OPTIONAL | energy score | authorized install なし |
| ESM-IF1 | fair-esm IF | INSTALLED | structure-conditioned LL | 専用 `.venv_esmif`（biotite 0.41 + torch-scatter） |
| ProteinMPNN | — | SKIPPED_OPTIONAL | native scoring | 依存パスが IgFold 設計寄りで Stage4 では未採用 |
| cavity external | — | SKIPPED_OPTIONAL | cavity volume | geometry approximation に fallback |

## Installation notes

1. `.venv_stage4`（Python 3.12）に propka / pdb2pqr を uv で追加
2. `apbs` は apt（初回 mirror 404 → `apt-get update` 後成功）
3. ESM-IF はメイン環境を壊さないよう `.venv_esmif` を分離
4. FoldX / Rosetta は license・新規大規模導入を行わず skip
