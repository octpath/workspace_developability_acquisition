#!/usr/bin/env python3
"""Freeze finalists from Dev CV; evaluate Public then Private; paired comparisons; shadows already in metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import CONFIG, METRICS, PREDS, REPORTS, TARGET_COLS, ensure_dirs, write_json  # noqa: E402


def load_all_results() -> pd.DataFrame:
    frames = []
    for p in [
        METRICS / "stage_ABC_results.csv",
        METRICS / "stage_PLM_STR_results.csv",
        METRICS / "stage_fusion_xgb_results.csv",
    ]:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No metrics yet")
    return pd.concat(frames, ignore_index=True)


def pick_finalists(df: pd.DataFrame, target: str, split: str = "canonical") -> list[dict]:
    sub = df[(df["target"] == target) & (df["split"] == split)].copy()
    # exclude illegal
    sub = sub[sub["feature_class"] != "ILLEGAL_FOR_PARTICIPANTS"]
    # best per representation family
    sub = sub.sort_values("cv_spearman", ascending=False)

    def best_matching(pred):
        m = sub[sub["representation"].map(pred)]
        if len(m) == 0:
            return None
        return m.iloc[0].to_dict()

    picks = []
    roles = [
        ("P0_simple_physchem", lambda r: r.startswith("A2_") or r == "A2_physchem" or r.startswith("B_cdr")),
        ("P1_BIO_SHORTCUT", lambda r: "BIO_SHORTCUT" in r or r == "C_BIO_SHORTCUT"),
        ("P2_ab_PLM", lambda r: "ablang" in r.lower()),
        ("P3_generic_PLM", lambda r: "esm" in r.lower()),
        ("P4_structure", lambda r: r.startswith("STR_")),
        ("P5_fusion", lambda r: "FUSION" in r or "fusion" in r.lower()),
        ("P6_nonlinear", lambda r: r.get("model") == "XGBoost" if False else False),
    ]
    # Fix: best_matching takes representation string predicate
    used = set()
    # P0
    for name, pred in [
        ("P0_best_simple", lambda r: r in ("A2_physchem", "B_cdr_descriptors", "A1_aa_comp", "B_cdr_hydrophobicity_only")),
        ("P1_BIO_SHORTCUT", lambda r: r == "C_BIO_SHORTCUT"),
        ("P2_ab_PLM", lambda r: "ablang" in r.lower()),
        ("P3_generic_PLM", lambda r: "esm1b" in r.lower() or "esm2" in r.lower()),
        ("P4_structure", lambda r: r.startswith("STR_")),
        ("P5_fusion", lambda r: "FUSION" in r),
    ]:
        cand = sub[sub["representation"].map(pred) & ~sub["representation"].isin(used)]
        if len(cand):
            row = cand.iloc[0].to_dict()
            row["pipeline_role"] = name
            picks.append(row)
            used.add(row["representation"])
    # P6 nonlinear: best XGBoost among remaining
    xgb = sub[sub["model"].astype(str).str.contains("XGB|XGBoost|SVR|LGBM", case=False, na=False)]
    if len(xgb):
        row = xgb.iloc[0].to_dict()
        row["pipeline_role"] = "P6_nonlinear"
        picks.append(row)
    # fill up to 7 with overall best unused
    for _, row in sub.iterrows():
        if len(picks) >= 7:
            break
        if row["representation"] in used:
            continue
        d = row.to_dict()
        d["pipeline_role"] = f"P_extra_{len(picks)}"
        picks.append(d)
        used.add(row["representation"])
    return picks[:7]


def ranking_spearman(df, target, split, col_a, col_b):
    sub = df[(df["target"] == target) & (df["split"] == split)].copy()
    sub = sub[sub["feature_class"] != "ILLEGAL_FOR_PARTICIPANTS"]
    # one row per representation (best model by col_a)
    sub = sub.sort_values(col_a, ascending=False).groupby("representation", as_index=False).first()
    a = sub[col_a].values
    b = sub[col_b].values
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    return float(spearmanr(a[mask], b[mask]).correlation), int(mask.sum())


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLS:
        for split in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
            sub = df[(df["target"] == target) & (df["split"] == split)]
            sub = sub[sub["feature_class"] != "ILLEGAL_FOR_PARTICIPANTS"]

            def best(pred, metric="cv_spearman"):
                m = sub[sub["representation"].map(pred)]
                if len(m) == 0:
                    return np.nan
                return float(m[metric].max())

            simple = best(lambda r: r in ("A2_physchem", "B_cdr_descriptors", "B_cdr_hydrophobicity_only"))
            bio = best(lambda r: r == "C_BIO_SHORTCUT")
            plm = best(lambda r: r.startswith("PLM_"))
            struct = best(lambda r: r.startswith("STR_"))
            sasa = best(lambda r: "SASA" in r and "RASA" not in r and r.startswith("STR_"))
            sasa_rasa = best(lambda r: "SASA_RASA" in r)
            surf = best(lambda r: "SURFACE_PHYS" in r)
            abb = best(lambda r: r.startswith("STR_ABB"))
            esm = best(lambda r: r.startswith("STR_ESMF"))
            fusion = best(lambda r: "FUSION" in r)
            nonlinear = float(
                sub[sub["model"].astype(str).str.contains("XGB|SVR", case=False, na=False)]["cv_spearman"].max()
            ) if sub["model"].astype(str).str.contains("XGB|SVR", case=False, na=False).any() else np.nan
            linear_best = float(
                sub[sub["model"].isin(["Ridge", "Lasso", "ElasticNet"])]["cv_spearman"].max()
            )

            comps = {
                "A_bio_minus_simple": bio - simple if np.isfinite(bio) and np.isfinite(simple) else np.nan,
                "B_plm_minus_bio": plm - bio if np.isfinite(plm) and np.isfinite(bio) else np.nan,
                "C_plm_minus_simple": plm - simple if np.isfinite(plm) and np.isfinite(simple) else np.nan,
                "D_struct_minus_simple": struct - simple if np.isfinite(struct) and np.isfinite(simple) else np.nan,
                "E_rasa_minus_sasa": sasa_rasa - sasa if np.isfinite(sasa_rasa) and np.isfinite(sasa) else np.nan,
                "F_surf_minus_simple": surf - simple if np.isfinite(surf) and np.isfinite(simple) else np.nan,
                "G_abb_minus_esmfold": abb - esm if np.isfinite(abb) and np.isfinite(esm) else np.nan,
                "H_fusion_minus_plm": fusion - plm if np.isfinite(fusion) and np.isfinite(plm) else np.nan,
                "I_nonlinear_minus_linear": nonlinear - linear_best
                if np.isfinite(nonlinear) and np.isfinite(linear_best)
                else np.nan,
                "best_simple": simple,
                "best_bio": bio,
                "best_plm": plm,
                "best_struct": struct,
                "best_overall": float(sub["cv_spearman"].max()) if len(sub) else np.nan,
            }
            for k, v in comps.items():
                rows.append({"target": target, "split": split, "comparison": k, "value": v})
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    df = load_all_results()
    df.to_csv(METRICS / "all_results.csv", index=False)

    registry = {}
    lines = ["# Canonical freeze", ""]
    for target in TARGET_COLS:
        picks = pick_finalists(df, target)
        registry[target] = [
            {
                "role": p["pipeline_role"],
                "representation": p["representation"],
                "model": p["model"],
                "cv_spearman": p["cv_spearman"],
                "public_spearman": p.get("public_spearman"),
                "private_spearman": p.get("private_spearman"),
            }
            for p in picks
        ]
        lines.append(f"## {target}")
        lines.append("| role | representation | model | CV ρ | Pub ρ | Priv ρ |")
        lines.append("|------|----------------|-------|------:|------:|-------:|")
        for p in picks:
            lines.append(
                f"| {p['pipeline_role']} | {p['representation']} | {p['model']} | "
                f"{p['cv_spearman']:.3f} | {p.get('public_spearman', float('nan')):.3f} | "
                f"{p.get('private_spearman', float('nan')):.3f} |"
            )
        lines.append("")
    write_json(CONFIG / "final_pipeline_registry.json", registry)
    (REPORTS / "canonical_freeze.md").write_text("\n".join(lines) + "\n")

    comps = paired_comparisons(df)
    comps.to_csv(METRICS / "paired_comparisons.csv", index=False)

    # Leaderboard reliability
    lb_lines = ["# Leaderboard reliability", ""]
    lb_rows = []
    for target in TARGET_COLS:
        lb_lines.append(f"## {target}")
        for split in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
            cv_pub, n1 = ranking_spearman(df, target, split, "cv_spearman", "public_spearman")
            pub_priv, n2 = ranking_spearman(df, target, split, "public_spearman", "private_spearman")
            cv_priv, n3 = ranking_spearman(df, target, split, "cv_spearman", "private_spearman")
            lb_lines.append(
                f"- {split}: CV→Pub {cv_pub:.3f} (n={n1}); Pub→Priv {pub_priv:.3f} (n={n2}); CV→Priv {cv_priv:.3f} (n={n3})"
            )
            lb_rows.append(
                {
                    "target": target,
                    "split": split,
                    "cv_to_public": cv_pub,
                    "public_to_private": pub_priv,
                    "cv_to_private": cv_priv,
                }
            )
        lb_lines.append("")
    pd.DataFrame(lb_rows).to_csv(METRICS / "leaderboard_reliability.csv", index=False)
    (REPORTS / "leaderboard_reliability.md").write_text("\n".join(lb_lines) + "\n")
    print("FREEZE_OK", {t: len(registry[t]) for t in registry})


if __name__ == "__main__":
    main()
