#!/usr/bin/env python3
"""One-shot: Draft 0.9 -> FINAL by filling CURSOR_VERIFY from audit. No training."""
from __future__ import annotations

from pathlib import Path

DRAFT = Path(
    "/root/.cursor/projects/workspace-developability-acquisition/attachments/"
    "8e25e77b-0075-49be-b053-65a6d4cea982/TM_HIC_PROMISING_MODEL_REPORT_DRAFT.md"
)
OUT = Path(__file__).resolve().parents[1] / "results" / "TM_HIC_PROMISING_MODEL_REPORT_FINAL.md"
AUDIT = Path(__file__).resolve().parents[1] / "results" / "TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md"
HEAD = "df1da85339a67599018ed0a1a902afc8fe56b2c2"


def cut_replace(s: str, start: str, end: str, mid: str) -> str:
    i = s.find(start)
    if i < 0:
        raise SystemExit(f"missing start {start[:80]!r}")
    j = s.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"missing end {end[:80]!r}")
    return s[:i] + mid + s[j:]


def replace_backtick_block(s: str, start_marker: str, replacement: str) -> str:
    """Replace a Draft marker of the form `[CURSOR_VERIFY: ...]` (closes with ]`)."""
    i = s.find(start_marker)
    if i < 0:
        raise SystemExit(f"missing marker {start_marker[:80]!r}")
    j = s.find("]`", i)
    if j < 0:
        raise SystemExit(f"missing close after {start_marker[:80]!r}")
    return s[:i] + replacement + s[j + 2 :]


def main() -> None:
    text = DRAFT.read_text()

    text = text.replace(
        "**文書状態:** Draft 0.9 — 科学的本文は作成済み。  \n"
        "**目的:** これまでの `developability_drilldown` 実験を、単なる leaderboard 表ではなく、\n"
        "「どの情報を、どのモデル構造で統合すると有効だったか」という観点から整理する。  \n"
        "**重要:** 本稿では、ChatGPT が現時点で repository 実装まで独立確認できていない箇所を\n"
        "`[CURSOR_VERIFY]` と明示した。これらは Cursor に実装監査させた後、最終レビューを行う。",
        f"**文書状態:** FINAL — repository implementation audit 完了（HEAD `{HEAD[:8]}`）。  \n"
        "**目的:** これまでの `developability_drilldown` 実験を、単なる leaderboard 表ではなく、\n"
        "「どの情報を、どのモデル構造で統合すると有効だったか」という観点から整理する。  \n"
        "**監査:** 旧 Draft の `[CURSOR_VERIFY]` は authoritative source と照合済み。"
        "詳細は `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`。",
    )

    text = cut_replace(
        text,
        "## 3.2 AbLang2 residue representation\n",
        "\n---\n\n# 4. TmApp 有力構造 A",
        """## 3.2 AbLang2 residue representation

現在の T113/T121 residue path で使う AbLang2 は次のとおり（historical 480-d `seqcoding` とは**非同義**）。

| 項目 | 実装 |
|---|---|
| package | `ablang2==0.2.1` |
| checkpoint | `ablang2-paired`（`random_init=False`） |
| 抽出 API | `AbLang.AbRep(tokens).last_hidden_states` から AA index のみ slice |
| H/L 呼び出し | **別々に encode**: Heavy=`[heavy,""]`、Light=`["",light]`（相手を空にし pre-contextualize しない） |
| special tokens | `<`,`>`,`|` を index 選択で除外（`EXACT_AA_1TO1_SPECIAL_STRIPPED`） |
| raw dim | **480**（残基ごと） |
| dtype | cache float16 → load float32 |
| cache | `top_models_feature_bundle/residue_level/ablang2/{heavy,light}_embeddings.npy` |
| 学習時 | 埋め込みは凍結キャッシュ；`nn.Linear(480→128)` で投影（trainable） |

したがって downstream Transformer で H/L interaction を入れる実験には意味がある。
PLM 出力時点ですでに H が L を見ているわけではない。

**出典:** `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`；
`residue_level/ablang2/metadata.json`；`models/antibody_transformer/model.py`（`plm_proj`）。
""",
    )

    text = replace_backtick_block(
        text,
        "`[CURSOR_VERIFY: actual token order / PyTorch module class /",
        """
### 実装確定事項（ARCH-3 / EXP-T113）

| 項目 | 値 |
|---|---|
| class / path | `AnnotatedTransformer.encode_joint_hl_dual_reg`（`chain_specific_reg=False`）；`models/antibody_transformer/model.py` |
| token 順 | `[REG_H, H_1..H_n, REG_L, L_1..L_m]` |
| REG init | `nn.Parameter` → `normal_(std=0.02)`；chain embedding のみ（IMGT/region/pos なし） |
| 残基 embedding 加算順 | `content(=plm_proj) + pos + chain` → `+imgt` → `+region` |
| attention | `attn_mask=None`；**`src_key_padding_mask` のみ**（「padding 以外 unrestricted」は正確） |
| Pre-LN | `norm_first=True` |
| d_model / layers / heads / FFN / dropout | 128 / 2 / 4 / 256 / 0.2 |
| regression head | `Linear(128→128)→GELU→Dropout(0.2)→Linear(128→1)` |
| trainable params（実ラン） | 382849（YAML preregistered 373633；語彙長差 +9216） |
""",
    )

    text = text.replace(
        "を回帰 head に渡す、というのが科学的な設計意図である。",
        """を回帰 head に渡す、というのが科学的な設計意図である。

### 実装確定式（ARCH-7 / EXP-T121）— コード通り

1. 共有 `self.encoder` で H/L を別々に full encode（REG は各鎖先頭）。
2. その**後に一度だけ** REG-only cross-attention（`nn.MultiheadAttention`, `batch_first=True`,
   heads=4, dropout=0.2；**H→L と L→H で同一 module**）。
3. K/V は相手鎖 residue のみ（REG は `[:,1:]` で除外）。mask は `key_padding_mask` のみ。
4. cross 後に LayerNorm / FFN / gate **なし**。

Exact residual:

```
delta_h = MHA(reg_h, l_res, l_res; key_padding_mask=~ml)
delta_l = MHA(reg_l, h_res, h_res; key_padding_mask=~mh)
reg_h' = reg_h + delta_h
reg_l' = reg_l + delta_l
z = (reg_h' + reg_l') / 2
```

depth 増加（T127/T129）は encoder self-attention 層のみ。cross stage は増やさない。

**出典:** `encode_reg_only_cross_attention`（`model.py` L779–829）。実ラン params ≈ 448897。""",
    )

    text = replace_backtick_block(
        text,
        "`[CURSOR_VERIFY: exact location of ARCH-7 cross-attention",
        "（上記「実装確定式」で監査完了。）",
    )

    text = replace_backtick_block(
        text,
        "`[CURSOR_VERIFY: 「10 Å 内」の対象集合が",
        """近傍集合は **exposed aromatic のみ**（全 aromatic でも全残基でもない）。
各 exposed aromatic（Cαあり）i について自己を含む 10 Å 内の exposed aromatic SASA 合計 L_i を取り、
feature は max_i L_i（空なら 0）。

**出典:** `aromatic_features` L82–89；`LOCAL_R=10.0`。""",
    )

    text = cut_replace(
        text,
        "## 10.8 19-dimensional output\n",
        "\n---\n\n# 11. HYDRO_FIELD",
        """## 10.8 19-dimensional output

EXP-H047 / late-fusion 列名は `aro_` 接頭辞。15 canonical + 4 QC。すべて **Fv 合算**。
aromatic={F,W,Y}。exposed: RASA≥0.20。strongly: RASA≥0.50。

| # | column name | definition | unit | scope | source |
|--:|---|---|---|---|---|
| 1 | `aro_exposed_TYR_count` | # exposed Y | count | Fv | `aromatic_features` L92 |
| 2 | `aro_exposed_TRP_count` | # exposed W | count | Fv | L93 |
| 3 | `aro_exposed_PHE_count` | # exposed F | count | Fv | L94 |
| 4 | `aro_exposed_aromatic_total_count` | # exposed F∪W∪Y | count | Fv | L95 |
| 5 | `aro_aromatic_exposed_SASA_total` | sum S over exposed arom | Å² | Fv | L63,96 |
| 6 | `aro_aromatic_exposed_SASA_fraction` | (#5) / sum_Fv S | fraction | den=**all Fv residues** | L62–63,97 |
| 7 | `aro_strongly_exposed_aromatic_count` | # aromatic RASA≥0.50 | count | Fv | L98 |
| 8 | `aro_strongly_exposed_aromatic_SASA` | sum S over strongly exposed arom | Å² | Fv | L99 |
| 9 | `aro_CDR_exposed_aromatic_count` | # exposed arom ∩ IMGT CDR | count | H+L CDR | L100 |
| 10 | `aro_CDR_aromatic_SASA` | sum S over CDR∩exposed arom | Å² | H+L CDR | L101 |
| 11 | `aro_CDR_aromatic_fraction` | (#10)/(#5) | fraction | den=**all exposed arom SASA** | L102 |
| 12 | `aro_aromatic_patch_count` | # CC; edge if Cα≤8 Å among exposed arom with CA | count | Fv | L71–73,103 |
| 13 | `aro_largest_aromatic_patch_n_res` | max \|c\|; empty→0; singleton=size-1 patch | count | Fv | L74–76,104 |
| 14 | `aro_largest_aromatic_patch_exposed_SASA` | sum S in largest CC; empty→0 | Å² | Fv | L77,105 |
| 15 | `aro_max_local_aromatic_SASA` | max local 10 Å sum (§10.7); empty→0 | Å² | exposed arom only | L82–89,106 |
| 16 | `aro_sequence_aromatic_count` | # F/W/Y (ignore exposure) | count | Fv | L107 |
| 17 | `aro_sequence_TYR_count` | # Y | count | Fv | L108 |
| 18 | `aro_sequence_TRP_count` | # W | count | Fv | L109 |
| 19 | `aro_sequence_PHE_count` | # F | count | Fv | L110 |

SASA: Bio.PDB `ShrakeRupley(probe_radius=1.4, n_points=100)`。
RASA: S / Tien2013 `MAX_ASA`。parquet 19 列はコードと exact set 一致。

**出典:** `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py`；
`common/structure_utils.py`；`AROMATIC-TOPO/FEATURE_SPEC.json`。
""",
    )

    text = replace_backtick_block(
        text,
        "`[CURSOR_VERIFY: 実装が単純 Euclidean pair-link graph /",
        """実装は `hydro_surface.connected_components`（Union–Find、全ペア Euclidean ≤ LINK=2.0 Å）。
**最大パッチは頂点数最大の CC**（面積最大ではない）。その CC の面積和で fraction を取る。""",
    )

    text = cut_replace(
        text,
        "## 11.8 16-dimensional output\n",
        "\n---\n\n# 12. SURFACE が HIC に効いたという意味",
        """## 11.8 16-dimensional output

14 canonical + 2 QC。EXP-H047 列名（接頭辞なし）:

| # | column name | formula / algorithm | unit | weighting | threshold | source |
|--:|---|---|---|---|---|---|
| 1 | `mean_H_surface` | area-weighted mean of H(s) | field | area | — | `hydro_summaries` L234 |
| 2 | `q75_H_surface` | quantile(H, 0.75) | field | none | 0.75 | L236 |
| 3 | `q90_H_surface` | quantile(H, 0.90) | field | none | 0.90 | L237 |
| 4 | `q95_H_surface` | quantile(H, 0.95) | field | none | 0.95 | L238 |
| 5 | `max_H_surface` | max(H) | field | none | — | L239 |
| 6 | `positive_H_area_fraction` | area(H>0)/total area | fraction | area | H>0 | L240 |
| 7 | `top10_H_mean` | mean of top ceil(0.1 N) points by H | field | none | top 10% count | L224–225,241 |
| 8 | `top10_H_area_fraction` | area of those / total | fraction | area | top 10% | L242 |
| 9 | `high_H_patch_count` | # CC of high-H points | count | none | q≥0.80; link≤2.0 Å | L243 |
| 10 | `largest_high_H_patch_area_fraction` | area(max-\|V\| CC)/total | fraction | area | same | L244 |
| 11 | `largest_high_H_patch_n_vertices` | \|V\| of that CC | count | none | same | L245 |
| 12 | `CDR_mean_H` | mean(H[cdr]) else 0 | field | none | CDR flag | L246 |
| 13 | `CDR_q90_H` | quantile(H[cdr],0.90) else 0 | field | none | 0.90 | L247 |
| 14 | `CDR_high_H_area_fraction` | area(cdr∧high)/area(cdr) else 0 | fraction | area | high∩CDR | L248–250 |
| 15 | `n_surface_points` | # surface points (≤2000) | count | — | MAX_POINTS | L250 |
| 16 | `phi_finite_frac` | mean(isfinite(φ)) on APBS DX samples（**疎水場以外の QC**；copatch 同居） | fraction | none | — | `extract_hydro_copatch_batch2.py` |

Field: FreeSASA Lee–Richards probe 1.4；Fibonacci 0.35/Å²；Fauchère–Pliska π を重原子へ複製；
寄与に SASA 非乗算；cutoff 7 Å；α=1.0。

**出典:** `organizer_extension/feature_prospecting/common/hydro_surface.py`；
`HYDRO-FIELD/FEATURE_SPEC.json`。
""",
    )

    text = cut_replace(
        text,
        "## 13.2 auxiliary branch\n",
        "\n---\n\n# 14. HIC 有望 backbone 1",
        """## 13.2 auxiliary branch

production（`LateFusionAuxMLP`）は preregistration と一致。**モデル内 LayerNorm はない**
（標準化は TRAIN-only `StandardScaler` を外部前処理で適用）:

```
x_aux -> Linear(p,64) -> GELU -> Dropout(0.2) -> Linear(64,32) -> GELU = z_aux ∈ R^32
z = [z_DL ; z_aux]
  -> Linear(dim(z), d_model) -> GELU -> Dropout -> Linear(d_model, 1)
```

F1: p=35、z_DL∈R^128（MEAN）、concat dim=160。gate / FiLM / token injection なし。

前処理（scheme/fold ごと TRAIN のみ fit）:

- F1: median impute → StandardScaler；concat 順 = **AROMATIC_TOPO19 → HYDRO_FIELD16**
- F4: 各 block 独立に impute →（FB のみ）**PCA32** → StandardScaler → concat
  順 = ARO → HYDRO → SEQ → TITR → PCA32（計 200）。**PCA 前の StandardScaler は行わない。**
- グローバル ESM2_H は fusion bundle に含めない。

**出典:** `models/antibody_transformer/late_fusion.py`；`h047_aux_features.py`。
""",
    )

    text = replace_backtick_block(
        text,
        "`[CURSOR_VERIFY: H061/H086 における ARCH-4 attention mask",
        """### ARCH-4 query→key permission matrix（1=allowed, 0=blocked）

実装: `build_chain_specific_dual_reg_attn_mask`。PyTorch では True=blocked。
layout `[REG_H, H…, REG_L, L…]`。

| query \\ key | REG_H | H res | REG_L | L res |
|---|:-:|:-:|:-:|:-:|
| REG_H | 1 | 1 | 0 | 0 |
| H res | 1 | 1 | 0 | 1 |
| REG_L | 0 | 0 | 1 | 1 |
| L res | 0 | 1 | 1 | 1 |

REG は自鎖＋自己のみ；残基同士の H↔L は許可；他鎖 REG への残基 attention は禁止。
""",
    )

    text = cut_replace(
        text,
        "# 20. 実装監査が必要な箇所\n",
        "\n---\n\n# 21. 最終版で採用する推奨表現",
        """# 20. 実装監査ステータス

本 FINAL では実装依存記述を repository と照合済み。
監査ログ: `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`。

| ID | 項目 | 結果 |
|---|---|---|
| A1 | AROMATIC_TOPO 19列 | exact |
| A2 | 10 Å local SASA 近傍 | exposed aromatic only |
| H1 | HYDRO_FIELD 16列 | exact（14 canonical + 2 QC） |
| H2 | high-H CC | Union–Find ≤2.0 Å；最大=max-|V| |
| T1 | ARCH-3 | padding-only unrestricted；exact |
| T2 | ARCH-7 | encode後・共有 MHA・ungated residual・Norm/FFNなし |
| T3 | ARCH-4 4×4 | exact |
| F1–F2 | late-fusion / preprocess | exact（aux に LayerNorm なし） |
| P1 | AbLang2 residue | exact；≠ seqcoding |
""",
    )

    # §22: replace through EOF
    i22 = text.find("# 22. Evidence / provenance notes\n")
    if i22 < 0:
        raise SystemExit("missing §22")
    text = (
        text[:i22]
        + f"""# 22. Evidence / provenance notes

- Git HEAD: `{HEAD}`
- Registry: `developability_drilldown/results/experiments.csv`
- AROMATIC_TOPO: `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py`,
  `common/structure_utils.py`, `AROMATIC-TOPO/FEATURE_SPEC.json`
- HYDRO_FIELD: `common/hydro_surface.py`, `HYDRO-FIELD/FEATURE_SPEC.json`,
  `scripts/extract_hydro_copatch_batch2.py`
- Transformers: `developability_drilldown/models/antibody_transformer/model.py`
- Late fusion: `late_fusion.py`, `h047_aux_features.py`
- AbLang2 residue: `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py`
- Configs: `EXP-T113.yaml`, `EXP-T121.yaml`, `EXP-H061.yaml`, `EXP-H086.yaml`, …
- Score convention: V3 の `cv_*` = OOF TEST（`dl_foldlocal_cosine_v3_oof_test`）

companion: `TM_HIC_PROMISING_MODEL_REPORT_IMPLEMENTATION_AUDIT.md`
"""
    )

    leftovers = [
        line
        for line in text.splitlines()
        if "CURSOR_VERIFY:" in line or "CURSOR_VERIFY_TABLE" in line
    ]
    if leftovers:
        for line in leftovers:
            print("LEFTOVER:", line[:120])
        raise SystemExit(f"leftover CURSOR_VERIFY tags: {len(leftovers)}")

    OUT.write_text(text)
    print("wrote", OUT)

    audit = f"""# TM_HIC_PROMISING_MODEL_REPORT — Implementation Audit

**Status:** COMPLETE (FINAL issued)  
**HEAD:** `{HEAD}`  
**Draft audited:** `TM_HIC_PROMISING_MODEL_REPORT_DRAFT.md` (Draft 0.9)  
**Output:** `results/TM_HIC_PROMISING_MODEL_REPORT_FINAL.md`  
**Scope:** documentation fact-check only (no training / no new EXP / no feature regen)

---

## Checked configs

- `experiments/configs/EXP-T113.yaml`
- `experiments/configs/EXP-T121.yaml`
- `experiments/configs/EXP-H061.yaml`
- `experiments/configs/EXP-H086.yaml`
- `experiments/configs/EXP-H090.yaml`
- `experiments/configs/EXP-H085.yaml`
- `experiments/configs/EXP-H089.yaml`
- `experiments/features/EXP-H047.parquet` (column slices)
- `results/experiments.csv`

## Checked source paths

| Topic | Path | Function / notes |
|---|---|---|
| AROMATIC_TOPO | `organizer_extension/feature_prospecting/scripts/extract_physical_batch1.py` | `aromatic_features` L58–111 |
| SASA/RASA/patch consts | `.../common/structure_utils.py` | PROBE/N_POINTS/RASA/PATCH_R/LOCAL_R/MAX_ASA/AROMATIC |
| ARO SPEC | `.../AROMATIC-TOPO/FEATURE_SPEC.json` | 15 canonical + 4 QC |
| HYDRO_FIELD | `.../common/hydro_surface.py` | `build_surface_and_field`, `hydro_summaries`, `connected_components` |
| HYDRO SPEC | `.../HYDRO-FIELD/FEATURE_SPEC.json` | 14 canonical |
| phi_finite_frac | `.../scripts/extract_hydro_copatch_batch2.py` | APBS QC colocated |
| ARCH-3/4/6/7 | `developability_drilldown/models/antibody_transformer/model.py` | encode_* / mask builders |
| Late fusion | `.../late_fusion.py` | `LateFusionAuxMLP`, `LateFusionModel` |
| H047 aux | `.../h047_aux_features.py` | F1–F4 slices + TRAIN-only preprocess |
| AbLang2 extract | `top_models_feature_bundle/scripts/extract_ablang2_residue_embeddings.py` | paired empty-partner |
| AbLang2 meta | `top_models_feature_bundle/residue_level/ablang2/metadata.json` | ablang2==0.2.1, dim 480 |
| ARCH-4 mask test | `tests/test_chain_specific_dual_reg.py` | True=blocked convention |

---

## AROMATIC_TOPO — 19/19

Exact columns listed in FINAL §10.8. Parquet `aro_*` set matches generator output.

**Local 10 Å neighbor set:** exposed aromatic only (draft was uncertain; now fixed).

**CDR fraction:** numerator = CDR∩exposed-arom SASA; denominator = all exposed-arom SASA.

**Empty / singleton:** empty → 0; singleton is a size-1 patch counted in `aromatic_patch_count`.

## HYDRO_FIELD — 16/16

Exact columns in FINAL §11.8. Draft field equation / FreeSASA / Fibonacci / Fauchère / q0.80 / link 2.0 match code.

**QC boundary:** #1–14 canonical; #15 `n_surface_points`; #16 `phi_finite_frac` (APBS; not hydrophobicity).

**Largest patch:** max vertex-count CC (not max area), then area fraction of that CC.

**SPEC wording mismatch (not draft):** FEATURE_SPEC text about equal area weights vs code using FreeSASA-derived `areas` — **code is authoritative**; FINAL follows code.

## ARCH-3 — exact

Padding-only unrestricted confirmed (`attn_mask=None`, `src_key_padding_mask` only).  
Pre-LN, MEAN, head layers as FINAL §4.2 table.

## ARCH-7 — exact

Draft conceptual residual matched; details filled:

- cross-attn **after** full shared encoder
- **shared** `self.cross_attn` for both directions
- **no** LN / FFN / gate after cross
- REG-only query; opposite-chain residues only as K/V

## ARCH-4 — exact

4×4 matrix in FINAL §14. Mask polarity verified against PyTorch (True=blocked).

## AbLang2 residue — exact

Distinct from historical seqcoding. Empty partner per chain. Raw 480 → Linear 128.

## Late fusion — exact

Aux MLP has **no LayerNorm** (draft prereg suggested possible LN; production has none).  
F1 order ARO→HYDRO. F4: impute→PCA(FB)→scale per block; no global PCA; no ESM2_H.

## Score registry consistency

V3 Transformer `cv_*` = OOF TEST (`cv_protocol=dl_foldlocal_cosine_v3_oof_test`).  
H047 uses classical protocol (`canonical_simple_tvt_primary_shadow`).

| Code | cv_P | cv_S | cv_mean | Public | Private | Overall |
|---|---:|---:|---:|---:|---:|---:|
| EXP-T113 | 2.995055 | 3.283071 | 3.139063 | 3.105442 | 2.999122 | 3.052282 |
| EXP-T121 | 3.023221 | 3.252085 | 3.137653 | 3.247434 | 3.193637 | 3.220536 |
| EXP-T096 | 3.252413 | 3.263832 | 3.258122 | 3.618574 | 3.460298 | 3.539436 |
| EXP-H047 | 0.452878 | 0.443979 | 0.448429 | 0.407578 | 0.447367 | 0.427473 |
| EXP-H054 | 0.502151 | 0.512973 | 0.507562 | 0.483359 | 0.462787 | 0.473073 |
| EXP-H061 | 0.533068 | 0.480523 | 0.506795 | 0.481635 | 0.461520 | 0.471577 |
| EXP-H071 | 0.528430 | 0.474947 | 0.501688 | 0.479571 | 0.480027 | 0.479799 |
| EXP-H085 | 0.472874 | 0.488744 | 0.480809 | 0.404951 | 0.428063 | 0.416507 |
| EXP-H086 | 0.471437 | 0.501216 | 0.486326 | 0.401757 | 0.410931 | 0.406344 |
| EXP-H089 | 0.471707 | 0.467812 | 0.469759 | 0.401818 | 0.439192 | 0.420505 |
| EXP-H090 | 0.457571 | 0.485454 | 0.471512 | 0.405876 | 0.412875 | 0.409375 |

Draft rounded tables match registry within rounding (no material numeric mismatch).

---

## Mismatches corrected in FINAL

| Draft statement | Actual implementation | Evidence | Material? |
|---|---|---|---|
| 10 Å local SASA neighbor set unspecified | **exposed aromatic only** | `aromatic_features` L82–89 | Yes (definition) |
| ARCH-7 LN/FFN/share unspecified | shared MHA; **no** LN/FFN after cross; ungated residual | `model.py` L820–829 | Yes (equation) |
| possible LayerNorm in aux branch | **no** LayerNorm in `LateFusionAuxMLP` | `late_fusion.py` L23–29 | Yes (architecture) |
| AbLang2 API / seqcoding ambiguity | residue AA-only 480-d ≠ seqcoding | metadata + extract script | Yes (provenance) |
| ARCH-4 mask not tabulated | 4×4 matrix filled | `build_chain_specific_dual_reg_attn_mask` | Documentation |
| YAML vs run param counts | run = YAML +9216 (dynamic emb sizes) | T113/T121 summary.json | Minor |

Scientific narrative (Tm vs HIC levers; SURFACE value; capacity failure) left unchanged.

---

## Unresolved items

**None.** FINAL issued (not REVIEW_DRAFT_1).

Optional non-blocking notes (not unresolved for report claims):

- YAML `n_trainable_preregistered` vs run counts differ by embedding vocabulary length (+9216).
- HYDRO FEATURE_SPEC prose on area weighting differs from code; FINAL follows code.
"""
    AUDIT.write_text(audit)
    print("wrote", AUDIT)


if __name__ == "__main__":
    main()
