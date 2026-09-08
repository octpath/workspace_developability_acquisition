# Organizer Score Integrity Audit — Simple TVT

**Evaluator:** `canonical_simple_tvt_v1`  
**Directory:** `organizer_extension/feature_prospecting/score_integrity/`  
**Scope:** Organizer Simple TVT only（Public/Private は未実施）

---

## 冒頭15問

1. **監査した重要歴史レシピ数?** **26**（registry 行。うち比較可能な歴史値あり 21）
2. **完全再現 (EXACT_MATCH)?** **19**
3. **文書化されたプロトコル差のみ?** **0**（本ラウンド）
4. **ID/index bug の影響?** **2**（`TmApp_SEQ+AbLingua` 歴史値、`TmApp_AbLingua_ADD_HISTORICAL_INVALID`）
5. **変わった結論?**  
   - AbLingua **ADD → ABL_TRY 撤回**（MIXED / ABL_DROP）。権威 PARENT = **2.7466 / 2.8227**  
   - SEQ+AbLang2+AbLingua の「両側正」も撤回（MIXED）  
   - BioEmu / MPNN / HIC aromatic・surface / OpenMM / FeNNix interim / CDR3 の主結論は **維持**
6. **AbLang2 結果は有効?** **YES**（BASE・recipe は ID 整列済み `load_bases` 経路。EXACT_MATCH）
7. **BioEmu 結果は有効?** **YES**（`BIOEMU_NEW_PAIRWISE` + BASE EXACT_MATCH）
8. **ProteinMPNN 結果は有効?** **YES**（`M1_PROTEINMPNN` EXACT_MATCH）
9. **HIC aromatic 結果は有効?** **YES**（`HIC_BASE_ARO` / CONT / HYDRO / TITR EXACT_MATCH）
10. **FeNNix 結果は有効?** **YES（interim CONSTANT N=100）** — EXACT_MATCH。フル Fab 未完了分は別途
11. **OpenMM MD 結果は有効?** **YES**（whole block + GLOBAL_DYNAMICS + DOMAIN_FLEXIBILITY EXACT_MATCH）
12. **AbLingua CDR3 は有効?** **YES**（guided プロトコル `PCA32_per_abl_block` で EXACT_MATCH **2.7321 / 2.7850** → GUIDED_TRY 維持）
13. **Canonical Simple TVT 評価器は一本化?** **YES** — `score_integrity/canonical_simple_tvt.py`
14. **回帰テスト?** **YES** — `test_golden_alignment.py`（A–F 全通過）
15. **Registry は後で Public/Private 拡張可能?** **YES** — `ORGANIZER_SCORE_REGISTRY.csv`（`validity=AUTHORITATIVE` のみ将来表に使用）

---

## 発見したバグ（確認済み）

| 経路 | 問題 | 状態 |
|------|------|------|
| AbLingua Sprint A `load_seq_basic` | `make_xy` の RangeIndex を誤って ADI ID に reindex → **SEQ_BASIC 全 NaN** | **修正済**（`index=ids` + all-NaN hard-fail） |
| 他の organizer Simple TVT（rescreen/cv） | `seq_b.index = ids` | **非該当** |

追加の silent alignment bug は、監査した重要 multi-block では **未検出**（全ブロック finite_frac=1.0、分散>0）。

---

## Compact 比較表（BUG/INVALID 優先）

| method | target | old P/S | canonical P/S | status | conclusion changed? |
|--------|--------|---------|---------------|--------|---------------------|
| AbLingua ADD sprintA | TmApp | 2.811 / 2.991 | 2.747 / 2.823 | BUG_AFFECTED | **YES** ABL_TRY→DROP |
| SEQ+AbLingua hist | TmApp | 3.186 / 3.162 | 2.987 / 3.116 | BUG_AFFECTED | **YES**（歴史=standalone偽） |
| BASE | TmApp | 2.763 / 2.897 | 2.763 / 2.897 | EXACT | NO |
| BASE+BIOEMU_PAIR | TmApp | 2.729 / 2.864 | 2.729 / 2.864 | EXACT | NO |
| BASE+M1 | TmApp | 2.737 / 2.878 | 2.737 / 2.878 | EXACT | NO |
| BASE+BIOEMU+M1 | TmApp | 2.703 / 2.846 | 2.703 / 2.846 | EXACT | NO |
| CURRENT_RECIPE | TmApp | 2.703 / 2.846 | 2.703 / 2.846 | EXACT | NO |
| PARENT+AbLingua GLOBAL | TmApp | 2.747 / 2.823 | 2.747 / 2.823 | EXACT | NO（権威） |
| SEQ+AbLang2+AbLingua corr | TmApp | 2.796 / 2.862 | 2.796 / 2.862 | EXACT | YES vs sprintA |
| PARENT+CDR3 | TmApp | 2.732 / 2.785 | 2.732 / 2.785 | EXACT | NO GUIDED_TRY |
| BASE+INTERIM_CONSTANT | TmApp | 3.052 / 2.875 | 3.052 / 2.875 | EXACT | NO |
| BASE+OPENMM_MD | TmApp | 2.874 / 2.956 | 2.874 / 2.956 | EXACT | NO（悪化維持） |
| OPENMM_GLOBAL_DYNAMICS | TmApp | 2.784 / 2.910 | 2.784 / 2.910 | EXACT | NO |
| OPENMM_DOMAIN_FLEX | TmApp | 2.777 / 2.898 | 2.777 / 2.898 | EXACT | NO |
| HIC_BASE | HIC | 0.535 / 0.559 | 0.535 / 0.559 | EXACT | NO |
| HIC_BASE_ARO | HIC | 0.528 / 0.559 | 0.528 / 0.559 | EXACT | NO |
| HIC_ARO+CONT | HIC | 0.512 / 0.557 | 0.512 / 0.557 | EXACT | NO |
| HIC_ARO+HYDRO | HIC | 0.520 / 0.556 | 0.520 / 0.556 | EXACT | NO |
| HIC_ARO+TITR | HIC | 0.526 / 0.550 | 0.526 / 0.550 | EXACT | NO |
| HIC_ARO+CONT+TITR | HIC | 0.507 / 0.552 | 0.507 / 0.552 | EXACT | NO |

定数・SEQ単独・AbLang2単独・ARO単独など歴史 MAE 未記録の行は canonical のみ登録（`SOURCE_NOT_RECONSTRUCTABLE`）。

---

## SEQ_BASIC 専用監査

- raw dim **78** / matched **162** / finite **1.0** / variance_sum **61.09**
- AbLang2 alone vs BASE（+SEQ）の予測 maxdiff **≫ 0** → SEQ は効いている
- 歴史上の疑わしい noop: Sprint A の `SEQ+AbLingua PCA32 == AbLingua standalone`（SEQ 全 NaN）

詳細: `SEQ_BASIC_AUDIT.json`

---

## Multi-block fingerprints

全重要ブロックについて `BLOCK_FINGERPRINTS.csv` に shape / matched_ids / all_nan / variance / SHA16 を記録。  
**all_nan=False**、**matched_ids>0** を確認。

---

## 成果物

| ファイル | 内容 |
|----------|------|
| `canonical_simple_tvt.py` | align_feature_block + Simple TVT + abl-block PCA |
| `test_golden_alignment.py` | TEST A–F |
| `SCORE_INTEGRITY_CODEPATH_AUDIT.csv` | コードパス監査 |
| `ORGANIZER_SCORE_REGISTRY.csv` | 権威レジストリ |
| `HISTORICAL_VS_CANONICAL.csv` | 比較 |
| `BLOCK_FINGERPRINTS.csv` | ブロック健全性 |
| `SEQ_BASIC_AUDIT.json` | SEQ 監査 |

---

## 更新した stale 報告

- `ablingua600m/ABLINGUA_FINAL_REPORT_JA.md` — 候補表に SUPERSEDED / CORRECTED を明記
- 既存 ERRATUM / PARENT reconciliation / guided 注記は維持
- 参加者向け配布データは変更なし（organizer スコア監査のみ）

---

## 次のステップ（本監査後）

1. 追加の silent alignment bug は重要レシピでは未検出 → **endgame 再開可（organizer レビュー後）**
2. Public / Private は **registry 拡張として後付け**（今回は未計算）
3. 新実験特徴の追加は、レジストリ freeze 後に限定
