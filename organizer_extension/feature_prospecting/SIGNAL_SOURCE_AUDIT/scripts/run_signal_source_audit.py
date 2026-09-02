#!/usr/bin/env python3
"""Gate 2L — sequence-control audit for AROMATIC-TOPO / POLAR-SAT (POST-COMPETITION_DIAGNOSTIC)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "SIGNAL_SOURCE_AUDIT"
import sys

sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    HIC_REF,
    OOF_DIR,
    TMAPP_REF,
    classify_standalone,
    mae,
    median_baseline_oof,
    metrics,
    nested_ridge,
)

# Frozen chemical categories (sequence-only; no burial/H-bond)
DONOR_CAPABLE = set("STNQYWHKR")
ACCEPTOR_CAPABLE = set("DENQSTYH")
POLAR_UNCHARGED = set("STNQYW")
CHARGED = set("DEKRH")
POLAR_ALL = set("STNQYWDEKRH")

AROM_SEQ_FEATS = [
    "seq_Y_count",
    "seq_W_count",
    "seq_F_count",
    "seq_YWF_total",
    "seq_YWF_fraction",
    "CDR_seq_Y_count",
    "CDR_seq_W_count",
    "CDR_seq_F_count",
    "CDR_seq_YWF_total",
]
POLAR_SEQ_FEATS = [
    "seq_polar_count",
    "seq_polar_fraction",
    "seq_donor_capable_count",
    "seq_acceptor_capable_count",
    "seq_polar_uncharged_count",
    "seq_charged_count",
    "CDR_seq_polar_count",
    "CDR_seq_polar_fraction",
]


def load_sequences():
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def build_seq_controls():
    seqs = load_sequences()
    cdr = pd.read_csv(FP / "cdr_sequence_index_imgt.csv")
    rows = []
    for aid, r in seqs.iterrows():
        fv = r["heavy"] + r["light"]
        n = len(fv)
        g = cdr[cdr.id == aid]
        cdr_seq = "".join(g.loc[g.is_cdr.astype(str).str.lower().isin(["true", "1"]), "amino_acid"].tolist())
        if not cdr_seq:
            # boolean
            cdr_seq = "".join(g.loc[g.is_cdr == True, "amino_acid"].tolist())  # noqa: E712
        def cnt(s, aa):
            return sum(1 for x in s if x == aa)

        ywf = cnt(fv, "Y") + cnt(fv, "W") + cnt(fv, "F")
        polar = sum(1 for x in fv if x in POLAR_ALL)
        cdr_polar = sum(1 for x in cdr_seq if x in POLAR_ALL)
        rows.append(
            {
                "id": aid,
                "seq_Y_count": cnt(fv, "Y"),
                "seq_W_count": cnt(fv, "W"),
                "seq_F_count": cnt(fv, "F"),
                "seq_YWF_total": ywf,
                "seq_YWF_fraction": ywf / n if n else 0.0,
                "CDR_seq_Y_count": cnt(cdr_seq, "Y"),
                "CDR_seq_W_count": cnt(cdr_seq, "W"),
                "CDR_seq_F_count": cnt(cdr_seq, "F"),
                "CDR_seq_YWF_total": cnt(cdr_seq, "Y") + cnt(cdr_seq, "W") + cnt(cdr_seq, "F"),
                "seq_polar_count": polar,
                "seq_polar_fraction": polar / n if n else 0.0,
                "seq_donor_capable_count": sum(1 for x in fv if x in DONOR_CAPABLE),
                "seq_acceptor_capable_count": sum(1 for x in fv if x in ACCEPTOR_CAPABLE),
                "seq_polar_uncharged_count": sum(1 for x in fv if x in POLAR_UNCHARGED),
                "seq_charged_count": sum(1 for x in fv if x in CHARGED),
                "CDR_seq_polar_count": cdr_polar,
                "CDR_seq_polar_fraction": cdr_polar / len(cdr_seq) if cdr_seq else 0.0,
                "fv_length": n,
            }
        )
    return pd.DataFrame(rows).set_index("id")


def eval_block(X_all, y, y_test, folds, sol, label):
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    common = X_all.index.intersection(y.index)
    X = X_all.loc[common]
    y = y.loc[common]
    oof, model, sc, a_full, alphas = nested_ridge(X, y, folds)
    base_oof, base_med = median_baseline_oof(y, folds)
    te_ids = X_all.index.intersection(y_test.index)
    Xte = X_all.loc[te_ids]
    yte = y_test.loc[te_ids]
    test_pred = pd.Series(model.predict(sc.transform(Xte)), index=Xte.index)
    pub_g = [i for i in pub if i in test_pred.index]
    pri_g = [i for i in pri if i in test_pred.index]
    m = {"label": label, "alpha_full": a_full, "n_dev": int(len(y)), "n_test": int(len(yte))}
    for split, yy, pp, bb in [
        ("CV", y, oof, base_oof),
        ("Public", yte.loc[pub_g], test_pred.loc[pub_g], pd.Series(base_med, index=pub_g)),
        ("Private", yte.loc[pri_g], test_pred.loc[pri_g], pd.Series(base_med, index=pri_g)),
        ("AllTest", yte, test_pred, pd.Series(base_med, index=yte.index)),
    ]:
        for k, v in metrics(yy, pp).items():
            m[f"{k}_{split}"] = v
        m[f"baseline_MAE_{split}"] = mae(yy, bb)
    m["standalone_signal"] = classify_standalone(m)
    return m, oof, test_pred


def classify_aromatic(seq_m, arom_m, combo_m):
    # Pre-specified operational rules (diagnostic only)
    seq_better_or_close = all(
        abs(seq_m[f"MAE_{s}"] - arom_m[f"MAE_{s}"]) <= 0.01 or seq_m[f"MAE_{s}"] <= arom_m[f"MAE_{s}"]
        for s in ("CV", "Public", "Private")
    )
    arom_beats_seq = all(arom_m[f"MAE_{s}"] + 1e-12 < seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private"))
    combo_beats_seq = all(combo_m[f"MAE_{s}"] + 1e-12 < seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private"))
    delta_combo = {s: combo_m[f"MAE_{s}"] - seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private")}
    if (arom_beats_seq or combo_beats_seq) and any(d < -0.005 for d in delta_combo.values()):
        # require reproducible combo improvement OR arom better on all splits
        if arom_beats_seq or all(d < 0 for d in delta_combo.values()):
            return "GEOMETRY_ADDS_SIGNAL", delta_combo
    if seq_better_or_close and not arom_beats_seq and not all(d < -0.005 for d in delta_combo.values()):
        return "MOSTLY_SEQUENCE_COMPOSITION", delta_combo
    return "UNRESOLVED", delta_combo


def classify_polar(seq_m, pol_m, combo_m):
    seq_better_or_close = all(
        abs(seq_m[f"MAE_{s}"] - pol_m[f"MAE_{s}"]) <= 0.05 or seq_m[f"MAE_{s}"] <= pol_m[f"MAE_{s}"]
        for s in ("CV", "Public", "Private")
    )
    pol_beats_seq = all(pol_m[f"MAE_{s}"] + 1e-12 < seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private"))
    combo_beats_seq = all(combo_m[f"MAE_{s}"] + 1e-12 < seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private"))
    delta_combo = {s: combo_m[f"MAE_{s}"] - seq_m[f"MAE_{s}"] for s in ("CV", "Public", "Private")}
    if (pol_beats_seq or combo_beats_seq) and (pol_beats_seq or all(d < 0 for d in delta_combo.values())):
        return "STRUCTURAL_POLAR_SAT_ADDS_SIGNAL", delta_combo
    if seq_better_or_close and not pol_beats_seq and not all(d < -0.02 for d in delta_combo.values()):
        return "MOSTLY_SEQUENCE_COMPOSITION", delta_combo
    return "UNRESOLVED", delta_combo


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    sol = pd.read_csv(ROOT / "competition/data/secret/solution.csv")
    seq = build_seq_controls()
    seq.to_csv(OUT / "sequence_controls.csv")
    (OUT / "SEQUENCE_CONTROL_SPEC.json").write_text(
        json.dumps(
            {
                "aromatic_features": AROM_SEQ_FEATS,
                "polar_features": POLAR_SEQ_FEATS,
                "DONOR_CAPABLE": sorted(DONOR_CAPABLE),
                "ACCEPTOR_CAPABLE": sorted(ACCEPTOR_CAPABLE),
                "POLAR_UNCHARGED": sorted(POLAR_UNCHARGED),
                "CHARGED": sorted(CHARGED),
                "POLAR_ALL": sorted(POLAR_ALL),
                "note": "POST_COMPETITION_DIAGNOSTIC; does not alter historical family verdicts",
            },
            indent=2,
        )
        + "\n"
    )

    arom_canon = json.loads((FP / "AROMATIC-TOPO/FEATURE_SPEC.json").read_text())["canonical_features"]
    pol_canon = json.loads((FP / "POLAR-SAT/FEATURE_SPEC.json").read_text())["canonical_features"]
    arom = (
        pd.read_parquet(FP / "AROMATIC-TOPO/features_esmfold.parquet")
        .query("extraction_status == 'SUCCESS'")
        .set_index("id")[arom_canon]
        .astype(float)
    )
    pol = (
        pd.read_parquet(FP / "POLAR-SAT/features_esmfold.parquet")
        .query("extraction_status == 'SUCCESS'")
        .set_index("id")[pol_canon]
        .astype(float)
    )

    results = {"label": "POST_COMPETITION_DIAGNOSTIC", "generator": "esmfold"}

    # HIC aromatic
    y_hic = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    y_hic_te = sol.set_index("id")["HIC"].astype(float)
    seq_a = seq[AROM_SEQ_FEATS]
    m_seq, _, _ = eval_block(seq_a, y_hic, y_hic_te, folds, sol, "seq_YWF")
    m_arom, _, _ = eval_block(arom, y_hic, y_hic_te, folds, sol, "AROMATIC-TOPO")
    combo_a = seq_a.join(arom, how="inner")
    m_combo, _, _ = eval_block(combo_a, y_hic, y_hic_te, folds, sol, "seq+AROMATIC")
    cls_a, d_a = classify_aromatic(m_seq, m_arom, m_combo)
    results["AROMATIC_HIC"] = {
        "sequence_only": m_seq,
        "AROMATIC_TOPO": m_arom,
        "sequence_plus_AROMATIC": m_combo,
        "classification": cls_a,
        "combo_minus_seq_MAE": d_a,
        "historical_verdict_immutable": "PROMISING_BUT_REDUNDANT",
    }

    # TmApp polar
    y_tm = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)
    y_tm_te = sol.set_index("id")["TmApp"].astype(float)
    seq_p = seq[POLAR_SEQ_FEATS]
    m_seq_p, _, _ = eval_block(seq_p, y_tm, y_tm_te, folds, sol, "seq_polar")
    m_pol, _, _ = eval_block(pol, y_tm, y_tm_te, folds, sol, "POLAR-SAT")
    combo_p = seq_p.join(pol, how="inner")
    m_combo_p, _, _ = eval_block(combo_p, y_tm, y_tm_te, folds, sol, "seq+POLAR")
    cls_p, d_p = classify_polar(m_seq_p, m_pol, m_combo_p)
    results["POLAR_SAT_TmApp"] = {
        "sequence_only": m_seq_p,
        "POLAR_SAT": m_pol,
        "sequence_plus_POLAR_SAT": m_combo_p,
        "classification": cls_p,
        "combo_minus_seq_MAE": d_p,
        "historical_verdict_immutable": "PROMISING_BUT_REDUNDANT",
    }

    (OUT / "SIGNAL_SOURCE_AUDIT_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

    # Japanese report
    def row(m):
        return f"MAE CV/Pub/Pri **{m['MAE_CV']:.4f} / {m['MAE_Public']:.4f} / {m['MAE_Private']:.4f}** · stand={m['standalone_signal']}"

    md = f"""# Signal-Source Audit（Gate 2L）

**性質:** `POST_COMPETITION_DIAGNOSTIC`  
**目的:** AROMATIC-TOPO / POLAR-SAT の明瞭な物理 signal が、構造特異的か配列組成 proxy かを分解する。  
**不変:** 歴史的 family verdict は変更しない。

化学カテゴリ（配列のみ・凍結）:
- donor-capable: `{''.join(sorted(DONOR_CAPABLE))}`
- acceptor-capable: `{''.join(sorted(ACCEPTOR_CAPABLE))}`
- polar uncharged: `{''.join(sorted(POLAR_UNCHARGED))}`
- charged: `{''.join(sorted(CHARGED))}`

Generator: ESMFold（構造 family の primary と整合）

---

## AROMATIC-TOPO × HIC

| Model | Metrics |
|-------|---------|
| sequence-only YWF | {row(m_seq)} |
| AROMATIC-TOPO | {row(m_arom)} |
| sequence + AROMATIC | {row(m_combo)} |

combo − seq ΔMAE CV/Pub/Pri: **{d_a['CV']:.4f} / {d_a['Public']:.4f} / {d_a['Private']:.4f}**

**分類:** `{cls_a}`

解釈: 歴史的 verdict `PROMISING_BUT_REDUNDANT` は不変。本診断は「3D 芳香族幾何が HIC を引き起こす」と断定するためのものではなく、配列組成との分離可能性を測る。

---

## POLAR-SAT × TmApp

| Model | Metrics |
|-------|---------|
| sequence-only polar | {row(m_seq_p)} |
| POLAR-SAT | {row(m_pol)} |
| sequence + POLAR-SAT | {row(m_combo_p)} |

combo − seq ΔMAE CV/Pub/Pri: **{d_p['CV']:.4f} / {d_p['Public']:.4f} / {d_p['Private']:.4f}**

**分類:** `{cls_p}`

解釈: 歴史的 verdict `PROMISING_BUT_REDUNDANT` は不変。埋没未充足極性の構造情報が配列組成を超えるかを評価。

---

## 直答

- AROMATIC は幾何特異的か？ → **{cls_a}**
- POLAR-SAT は構造特異的か？ → **{cls_p}**
"""
    (OUT / "SIGNAL_SOURCE_AUDIT_JA.md").write_text(md)
    print("AROMATIC", cls_a)
    print("POLAR", cls_p)
    print(json.dumps({"arom_d": d_a, "polar_d": d_p}, indent=2))


if __name__ == "__main__":
    main()
