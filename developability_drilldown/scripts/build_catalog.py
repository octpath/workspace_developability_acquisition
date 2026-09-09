#!/usr/bin/env python3
"""Build CATALOG.md / CATALOG_JA.md and XGB_REPRODUCTION_AUDIT.csv from master registry."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import BUNDLE, ROOT, load_dev_test_folds, load_solution, mae  # noqa: E402


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or pd.isna(x))) or x == "":
        return "NA"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def best_rows(df: pd.DataFrame, col: str) -> pd.Series:
    sub = df[pd.to_numeric(df[col], errors="coerce").notna()].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.loc[pd.to_numeric(sub[col], errors="coerce").idxmin()]


def catalog_md(df: pd.DataFrame, lang: str) -> str:
    ja = lang == "ja"
    lines = []
    if ja:
        lines += [
            "# Developability Drilldown カタログ",
            "",
            "出典: `results/experiments.csv`（単一マスターレジストリ）",
            "",
            "**現行評価モード:** `POSTCOMP_EXPLORATORY`",
            "",
            "Public / Private は post-competition の探索ベンチマークであり、",
            "competition 当時の selection history（`selection_policy_at_creation`）を書き換えません。",
            "Public/Private を independent held-out validation とは呼びません。",
            "",
        ]
    else:
        lines += [
            "# Developability Drilldown Catalog",
            "",
            "Source: `results/experiments.csv` (single master registry).",
            "",
            "**Current evaluation mode:** `POSTCOMP_EXPLORATORY`",
            "",
            "Public/Private are post-competition exploratory benchmarks.",
            "They do not rewrite competition-time `selection_policy_at_creation`.",
            "Do not call Public/Private independent held-out validation.",
            "",
        ]

    for target in ("TmApp", "HIC"):
        sub = df[df["target"] == target].copy()
        lines.append(f"## {target}")
        lines.append("")
        header = (
            "| experiment_id | family | model_type | feature_recipe | CV Primary | CV Shadow | CV worst | Public | Private | Test overall | artifact_status | feature_path | test_prediction_path |"
        )
        lines.append(header)
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        order = sub.copy()
        order["_sort"] = pd.to_numeric(order["cv_worst_mae"], errors="coerce")
        order = order.sort_values(["family", "_sort", "experiment_id"])
        for _, r in order.iterrows():
            lines.append(
                "| {experiment_id} | {family} | {model_type} | {feature_recipe} | {cv_primary_mae} | {cv_shadow_mae} | {cv_worst_mae} | {public_mae} | {private_mae} | {test_overall_mae} | {artifact_status} | {feature_path} | {test_prediction_path} |".format(
                    experiment_id=r["experiment_id"],
                    family=r["family"],
                    model_type=r["model_type"],
                    feature_recipe=r["feature_recipe"],
                    cv_primary_mae=_fmt(r["cv_primary_mae"]),
                    cv_shadow_mae=_fmt(r["cv_shadow_mae"]),
                    cv_worst_mae=_fmt(r["cv_worst_mae"]),
                    public_mae=_fmt(r["public_mae"]),
                    private_mae=_fmt(r["private_mae"]),
                    test_overall_mae=_fmt(r["test_overall_mae"]),
                    artifact_status=r["artifact_status"],
                    feature_path=r["feature_path"] or "",
                    test_prediction_path=r["test_prediction_path"] or "",
                )
            )
        lines.append("")
        lines.append("### BEST (auto)" if not ja else "### BEST（自動抽出）")
        lines.append("")
        for label, col in [
            ("CV Primary", "cv_primary_mae"),
            ("CV Shadow", "cv_shadow_mae"),
            ("CV worst", "cv_worst_mae"),
            ("Public (POSTCOMP_EXPLORATORY)", "public_mae"),
            ("Private (POSTCOMP_EXPLORATORY)", "private_mae"),
            ("Test overall", "test_overall_mae"),
        ]:
            b = best_rows(sub, col)
            if b.empty:
                lines.append(f"- **{label}:** NA")
            else:
                lines.append(f"- **{label}:** `{b['experiment_id']}` = {_fmt(b[col])} ({b['family']}, {b['artifact_status']})")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_xgb_audit(df: pd.DataFrame) -> pd.DataFrame:
    sol = load_solution()
    dev, _, _ = load_dev_test_folds()
    rows = []
    xgb = df[df["family"] == "XGBOOST"]
    for _, r in xgb.iterrows():
        eid = r["experiment_id"]
        target = r["target"]
        pred_dir = ROOT / "experiments" / "predictions" / eid
        oof_p = pd.read_csv(pred_dir / "oof_primary.csv")
        oof_s = pd.read_csv(pred_dir / "oof_shadow.csv")
        y = dev.set_index("id").loc[oof_p["id"].astype(str), target].to_numpy(float)
        prim_r = mae(y, oof_p[target].to_numpy(float))
        # shadow uses same label vector aligned to oof_s ids
        y_s = dev.set_index("id").loc[oof_s["id"].astype(str), target].to_numpy(float)
        shad_r = mae(y_s, oof_s[target].to_numpy(float))
        pub_r = priv_r = float("nan")
        if sol is not None:
            test = pd.read_csv(pred_dir / "test.csv")
            m = sol.merge(test.rename(columns={target: "pred"}), on="id")
            yt = m[target].to_numpy(float)
            yp = m["pred"].to_numpy(float)
            pub = m["is_public"].astype(bool).to_numpy()
            priv = m["is_private"].astype(bool).to_numpy()
            pub_r = mae(yt[pub], yp[pub])
            priv_r = mae(yt[priv], yp[priv])
        auth_p = float(r["cv_primary_mae"])
        auth_s = float(r["cv_shadow_mae"])
        auth_pub = float(r["public_mae"])
        auth_priv = float(r["private_mae"])
        d_p = prim_r - auth_p
        d_s = shad_r - auth_s
        d_pub = pub_r - auth_pub if not math.isnan(pub_r) else float("nan")
        d_priv = priv_r - auth_priv if not math.isnan(priv_r) else float("nan")
        # tolerances
        cv_ok = abs(d_p) <= 1e-6 and abs(d_s) <= 1e-6
        pp_ok = (math.isnan(d_pub) and math.isnan(d_priv)) or (abs(d_pub) <= 1e-5 and abs(d_priv) <= 1e-5)
        status = "PASS" if cv_ok and pp_ok else ("PASS_CV_ONLY" if cv_ok else "INCONSISTENT")
        notes = []
        if not cv_ok:
            notes.append("CV delta exceeds 1e-6")
        if not pp_ok:
            notes.append("Public/Private delta exceeds 1e-5 (library/float)")
        rows.append(
            {
                "experiment_id": eid,
                "primary_authoritative": auth_p,
                "primary_reconstructed": prim_r,
                "primary_delta": d_p,
                "shadow_authoritative": auth_s,
                "shadow_reconstructed": shad_r,
                "shadow_delta": d_s,
                "public_authoritative": auth_pub,
                "public_reconstructed": pub_r,
                "public_delta": d_pub,
                "private_authoritative": auth_priv,
                "private_reconstructed": priv_r,
                "private_delta": d_priv,
                "reproduction_status": status,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "experiments.csv")
    (ROOT / "results" / "CATALOG.md").write_text(catalog_md(df, "en"), encoding="utf-8")
    (ROOT / "results" / "CATALOG_JA.md").write_text(catalog_md(df, "ja"), encoding="utf-8")
    audit = build_xgb_audit(df)
    audit.to_csv(ROOT / "results" / "XGB_REPRODUCTION_AUDIT.csv", index=False)
    # update reproduction_status on XGB rows
    for _, a in audit.iterrows():
        df.loc[df["experiment_id"] == a["experiment_id"], "reproduction_status"] = a["reproduction_status"]
        if a["reproduction_status"] == "INCONSISTENT":
            df.loc[df["experiment_id"] == a["experiment_id"], "artifact_status"] = "INCONSISTENT"
    df.to_csv(ROOT / "results" / "experiments.csv", index=False)
    print("wrote catalogs + XGB_REPRODUCTION_AUDIT.csv")
    print(audit[["experiment_id", "reproduction_status", "primary_delta", "shadow_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
