# Feature Prospecting Postmortem v1

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_LATE_BATCH3_AND_POSTMORTEM_COMPLETE`  
**性質:** ORGANIZER-EXPLORATORY（competition-valid prospective ではない）

関連:
- [SIGNAL_SOURCE_AUDIT_JA.md](SIGNAL_SOURCE_AUDIT/SIGNAL_SOURCE_AUDIT_JA.md)
- [ADVANCED_BATCH2_REPORT_JA.md](ADVANCED_BATCH2_REPORT_JA.md)
- [PHYSICAL_BATCH1_REPORT_JA.md](PHYSICAL_BATCH1_REPORT_JA.md)
- [LATE_BATCH3_TARGET_BLIND_FREEZE_MANIFEST.json](LATE_BATCH3_TARGET_BLIND_FREEZE_MANIFEST.json)

---

## 結局TmAppには何が関係していそうか

**再現する物理関連（standalone）はあるが、Round1 incumbent を超える incremental はほぼ無い。**

最も信用できるのは:
1. **POLAR-SAT** — 埋没未充足極性（standalone REPRODUCIBLE、ただし FRAGILE・冗長）。配列コントロールでは **UNRESOLVED**（構造寄与をはっきり切れない）。
2. **VHL-ANGLE** — 配向 standalone あり、ただし ABB2 mapping 懸念と冗長。
3. **PKA-SHIFT corrected** — ESMFold 限定の MIXED。
4. **TITRATION-SHAPE** — 全分子電荷曲線は **ROBUST** だが TmApp は WEAK/MIXED で、残基 ΔpKa を明確に超えない。

**効かなさそう / 現状データでは否定的:**
- ANM（global soft modes）
- INTERFACE-ENERGY（固定座標エネルギー；clash アーティファクト）
- OPENMM-STRAIN（緩和応答は clash 支配ではないが、真空最小化の重尾エネルギーで CV 不安定・有用 signal なし）
- 3DI-FROZEN（学習表現は generator 間でそこそこ一致するが target 情報なし）

**言い換え:** TmApp は「Fv 全体の柔らかい力学」より **局所的な極性・界面 packing** の方がまだマシ、だがどれも incumbent を押し上げるほどではない。配列組成と構造の分離は POLAR では未解決。

---

## 結局HICには何が関係していそうか

**露出芳香族トポロジー（AROMATIC-TOPO）が唯一の明瞭な reproducible standalone。**  
Signal-source audit では **`GEOMETRY_ADDS_SIGNAL`** — 配列 Y/W/F 組成だけでは説明しきれず、露出・空間トポロジーが予測情報を足す。

ただし:
- Round1 surface に対しては **incremental なし**（PROMISING_BUT_REDUNDANT）
- 「3D 芳香族が因果的に HIC を決める」までは言わない（探索的診断）

**効かなさそう:**
- STATIC-SAP（静的近似）
- HYDRO-FIELD（一般連続疎水場）
- ELEC-HYDRO-COPATCH（静電は足さない）
- ANM / pKa / titration / 3Di / INTERFACE / STRAIN
- SURFACE-DL は **METHOD_BLOCKED**（評価不能）

**言い換え:** HIC は「広い疎水性」ではなく **芳香族の露出幾何** に寄っている。一般疎水場や静電 co-patch に広げると signal が消える。

---

## TmApp table

| Family | Prior | Robustness | Standalone | Increment | Verdict | Paper | Repo |
|--------|-------|------------|------------|-----------|---------|-------|------|
| ANM-SPECTRUM | 5 | MODERATE | TEST_ONLY | CV_ONLY | NO_EVIDENCE | [DOI](https://doi.org/10.1016/S0006-3495(01)76033-X) | [ProDy](https://github.com/prody/ProDy) |
| VHL-ANGLE | 5 | PROVISIONAL_MAP | REPRODUCIBLE | NO | PROMISING_BUT_REDUNDANT | [DOI](https://doi.org/10.1093/protein/gzt020) | [ABangle](https://github.com/jaredsampson/ABangle) |
| PKA-SHIFT corr. | 5 | FRAGILE | REPRODUCIBLE | WEAK/MIXED | MIXED | [DOI](https://doi.org/10.1021/ct100578z) | [propka](https://github.com/jensengroup/propka) |
| AROMATIC-TOPO | 3 | MODERATE | WEAK | NO | MIXED | [DOI](https://doi.org/10.1080/19420862.2020.1743053) | — |
| STATIC-SAP | 3 | MODERATE | WEAK | NO | MIXED | [DOI](https://doi.org/10.1073/pnas.0904191106) | — |
| VOID-EXPLICIT | 5 | FRAGILE | WEAK | NO | MIXED | [DOI](https://doi.org/10.1186/s12859-021-04519-4) | [pyKVFinder](https://github.com/LBC-LNBio/pyKVFinder) |
| POLAR-SAT | 5 | FRAGILE | REPRODUCIBLE | NO | PROMISING_BUT_REDUNDANT | [DOI](https://doi.org/10.1371/journal.pcbi.1008061) | [pdb2pqr](https://github.com/Electrostatics/pdb2pqr) |
| HYDRO-FIELD | 3 | MODERATE | WEAK | NO | MIXED | [DOI](https://doi.org/10.1016/0223-5234(89)90109-8) | [FreeSASA](https://github.com/mittinatten/freesasa) |
| ELEC-HYDRO-COPATCH | 3 | FRAGILE | TEST_ONLY | NO | NO_EVIDENCE | [DOI](https://doi.org/10.1002/jcc.10344) | [APBS](https://github.com/Electrostatics/apbs) |
| INTERFACE-ENERGY | 5 | FRAGILE | NO | NO | UNLIKELY | [DOI](https://doi.org/10.1371/journal.pcbi.1005659) | [OpenMM](https://github.com/openmm/openmm) |
| 3DI-FROZEN | 4 | MODERATE | TEST_ONLY | NO | NO_EVIDENCE | [DOI](https://doi.org/10.1038/s41587-023-01773-0) | [Foldseek](https://github.com/steineggerlab/foldseek) |
| TITRATION-SHAPE | 5 | **ROBUST** | WEAK | NO | MIXED | [DOI](https://doi.org/10.1021/ct100578z) | [propka](https://github.com/jensengroup/propka) |
| OPENMM-STRAIN | 4 | FRAGILE | WEAK* | NO | MIXED* | [DOI](https://doi.org/10.1371/journal.pcbi.1005659) | [OpenMM](https://github.com/openmm/openmm) |
| SURFACE-DL | 3 | BLOCKED | — | — | METHOD_BLOCKED | [DOI](https://doi.org/10.1007/978-3-030-87199-4_5) | [dMaSIF](https://github.com/FreyrS/dMaSIF) |

\*OPENMM-STRAIN: CV MAE が重尾エネルギーで数値崩壊；Public/Private は baseline 付近。clash 支配ではない。

## HIC table

| Family | Prior | Robustness | Standalone | Increment | Verdict | Paper | Repo |
|--------|-------|------------|------------|-----------|---------|-------|------|
| AROMATIC-TOPO | 5 | MODERATE | REPRODUCIBLE | NO | **PROMISING_BUT_REDUNDANT** | [DOI](https://doi.org/10.1080/19420862.2020.1743053) | — |
| STATIC-SAP | 5 | MODERATE | TEST_ONLY | NO | NO_EVIDENCE | [DOI](https://doi.org/10.1073/pnas.0904191106) | — |
| HYDRO-FIELD | 5 | MODERATE | TEST_ONLY | NO | NO_EVIDENCE | [DOI](https://doi.org/10.1016/0223-5234(89)90109-8) | [FreeSASA](https://github.com/mittinatten/freesasa) |
| ELEC-HYDRO-COPATCH | 4 | FRAGILE | NO | NO | NO_EVIDENCE | [DOI](https://doi.org/10.1002/jcc.10344) | [APBS](https://github.com/Electrostatics/apbs) |
| PKA / ANM / VHL / VOID / POLAR / INTERFACE / 3DI / TITRATION / STRAIN | ≤4 | — | NO / weak | NO | NO_EVIDENCE | — | — |
| SURFACE-DL | 4 | BLOCKED | — | — | METHOD_BLOCKED | [DOI](https://doi.org/10.1007/978-3-030-87199-4_5) | [dMaSIF](https://github.com/FreyrS/dMaSIF) |

---

## Evidence hierarchy（MAE 順位ではない）

1. **Mechanistic + audit:** HIC では AROMATIC が配列を超える幾何寄与（GEOMETRY_ADDS_SIGNAL）。TmApp では POLAR が最有力だが配列分離は UNRESOLVED。
2. **CV/Public/Private:** AROMATIC / POLAR / VHL が standalone 再現。incremental はほぼ全滅。
3. **Generator:** TITRATION のみ ROBUST。多くの物理量は FRAGILE。
4. **Artifact:** INTERFACE は clash 相関強。STRAIN は clash 非支配だが数値重尾。
5. **Blocked:** SURFACE-DL は依存/ライセンスで評価不能（負の結果として記録）。

---

## 重要な負の結果（意味すること / しないこと）

| 結果 | 意味する | 意味しない |
|------|----------|------------|
| ANM no evidence | 低周波 ENM スペクトルはこの規約・データで TmApp を説明しない | 動力学一般が無関係と証明したわけではない |
| HYDRO-FIELD / STATIC-SAP / COPATCH | 一般疎水場・静的 SAP・静電 co-patch は AROMATIC の代替にならない | 疎水性が HIC に無関係なわけではない（芳香族に局在） |
| 3DI no evidence | Foldseek3Di+ProstT5 frozen emb は target 情報を回収しない | すべての学習表現が無駄とは言えない |
| INTERFACE unlikely | 固定座標 FF 界面エネルギーは予測 PDB では危険 | 界面物理そのものが無意味とは言えない |
| OPENMM-STRAIN mixed/unstable | 単純真空最小化ひずみは有用 descriptor になっていない | 明示溶媒・拘束付き緩和が無用とは未検証 |
| SURFACE-DL blocked | 再現可能な pretrained frozen surface DL を本環境で回せなかった | surface DL 仮説の棄却ではない |

---

## Signal-source audit（要約）

- **AROMATIC × HIC:** `GEOMETRY_ADDS_SIGNAL`（seq+AROMATIC が seq-only を CV/Pub/Pri で改善）
- **POLAR-SAT × TmApp:** `UNRESOLVED`（split 間で不安定）
- 歴史的 verdict は不変。

---

## Batch3 family notes

### TITRATION-SHAPE
- PypKa は py3.12 で delphi4py 非対応 → **PROPKA HH independent-site** を target 前に選択
- 側鎖のみ Q(pH)；人工 C 末端除外
- Robustness **ROBUST** · TmApp **MIXED** · HIC no evidence
- PKA-SHIFT を明確に上回らず

### OPENMM-STRAIN
- OpenMM 8.6 / amber14-all / LocalEnergyMinimizer tol=10, maxIter=500 / NoCutoff vacuum
- **GENERATOR_GEOMETRY_ARTIFACT_DOMINATED = False**（INTERFACE との差）
- だが初期力・ΔE が重尾 → CV Ridge 数値不安定 · Verdict **MIXED**（実用 signal なし）

### SURFACE-DL
- **METHOD_BLOCKED_DEPENDENCY_OR_LICENSE**（dMaSIF CC-BY-NC-ND、pretrained 重み不確実）

---

## 追加 prospecting はまだ価値があるか

**diminishing returns に近い。**

| 候補 | 推奨 |
|------|------|
| ENCOM-CHEM | 低 — ANM 系は既に空振り |
| GEARNET-FROZEN | 低〜中 — 3Di 空振り後、別 GNN の情報利得は不確実 |
| MLFF | 低 — 予測 PDB 上のエネルギーは STRAIN/INTERFACE と同型リスク |
| SHORT-ENSEMBLE | 低〜中 — 計算コスト大；STRAIN が既に示唆する通り単純緩和では不足の可能性 |
| SURFACE-DL（適正ライセンス・公式权重み） | 条件付き — HIC の芳香族仮説の拡張としては科学的に最も残っている |

**推奨ストップ方針:** 新規 family の自動連鎖は止め、必要なら HIC 側の **芳香族／表面幾何の限定的深掘り**（適正ライセンスの frozen surface model）のみ検討。

---

**Final state:** `ORGANIZER_FEATURE_PROSPECTING_LATE_BATCH3_AND_POSTMORTEM_COMPLETE`
