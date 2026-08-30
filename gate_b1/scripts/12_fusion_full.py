#!/usr/bin/env python3
"""Fusion features + XGBoost on finalist representations; full-target confirmation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    CACHE,
    DATA,
    METRICS,
    PREDS,
    REPORTS,
    SPLITS,
    TARGET_COLS,
    ensure_dirs,
)
import importlib.util

spec = importlib.util.spec_from_file_location("tplm", Path(__file__).resolve().parent / "10_train_plm_structure.py")
tplm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tplm)


def concat_features(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1)


def main():
    ensure_dirs()
    df = pd.read_csv(DATA / "triple_core.csv")

    # Build candidate fusions if components exist
    emb_cands = []
    for mid in ["ablang2_default", "esm2_t33_650M_UR50D", "esm1b_t33_650M_UR50S"]:
        p = CACHE / "plm" / f"manifest_{mid}.csv"
        if p.exists():
            emb_cands.append((mid, p))
    str_path = CACHE / "structure_features" / "abodybuilder2_sasa_rasa.csv"
    has_str = str_path.exists()

    rows = []
    for split_name in ["canonical", "shadow_1", "shadow_2", "shadow_3"]:
        split_df = pd.read_csv(SPLITS / f"triple_{split_name}.csv")
        m = split_df.merge(df, on="antibody_id", how="left")
        roles = m["role"].values
        groups = m["cluster_id"].values
        ids = m["antibody_id"].values
        dev = roles == "Dev"

        for target, col in TARGET_COLS.items():
            y = m[col].values.astype(float)
            # Fusions
            fusion_defs = []
            if emb_cands and has_str:
                mid, man = emb_cands[0]
                fusion_defs.append(
                    (
                        f"FUSION_{mid}_ABB_SURFACE",
                        [
                            ("emb", man),
                            (
                                "str",
                                str_path,
                                {
                                    "include": [
                                        "rasa_weighted",
                                        "exposed_",
                                        "sasa_hydrophobic",
                                        "mean_rasa",
                                        "total_sasa",
                                    ],
                                    "exclude": [],
                                },
                            ),
                        ],
                    )
                )
            # best sequence physchem + structure
            a2 = CACHE / "features" / "stage_A_simple.csv"
            if has_str and a2.exists():
                fusion_defs.append(
                    (
                        "FUSION_A2_ABB_SURFACE",
                        [
                            ("csv", a2, "A2_"),
                            (
                                "str",
                                str_path,
                                {
                                    "include": ["rasa_weighted", "exposed_", "sasa_hydrophobic"],
                                    "exclude": [],
                                },
                            ),
                        ],
                    )
                )
            # bio + plm
            cpath = CACHE / "features" / "stage_C_shortcut.csv"
            if emb_cands and cpath.exists():
                mid, man = emb_cands[0]
                fusion_defs.append(
                    (
                        f"FUSION_BIO_{mid}",
                        [("csv_oh", cpath), ("emb", man)],
                    )
                )

            for fname, parts in fusion_defs:
                try:
                    mats = []
                    for part in parts:
                        if part[0] == "emb":
                            mats.append(tplm.load_embedding_matrix(part[1], m["antibody_id"]))
                        elif part[0] == "str":
                            mats.append(
                                tplm.load_structure_features(part[1], m["antibody_id"], **part[2]).values.astype(
                                    float
                                )
                            )
                        elif part[0] == "csv":
                            feat = pd.read_csv(part[1])
                            feat = m[["antibody_id"]].merge(feat, on="antibody_id", how="left")
                            cols = [c for c in feat.columns if c.startswith(part[2])]
                            mats.append(feat[cols].apply(pd.to_numeric, errors="coerce").values.astype(float))
                        elif part[0] == "csv_oh":
                            feat = pd.read_csv(part[1])
                            feat = m[["antibody_id"]].merge(feat, on="antibody_id", how="left")
                            # numeric only for fusion simplicity (+ hash cats)
                            num = feat.select_dtypes(include=[np.number])
                            mats.append(num.values.astype(float))
                    X = concat_features(mats)
                except Exception as e:
                    print("FUSION_SKIP", fname, e)
                    continue

                for mn in ["Ridge", "ElasticNet"]:
                    print(f"FUSION {split_name} {target} {fname} {mn}", flush=True)
                    cv = tplm.nested_cv_dense(X[dev], y[dev], groups[dev], model_name=mn)
                    model = cv["model"]
                    role_m = {}
                    for role in ["Public", "Private"]:
                        mask = roles == role
                        pred = model.predict(X[mask])
                        role_m[role] = tplm.metrics_dict(y[mask], pred)
                    rows.append(
                        {
                            "representation": fname,
                            "model": mn,
                            "target": target,
                            "split": split_name,
                            "cv_spearman": cv["cv"]["spearman"],
                            "cv_pearson": cv["cv"]["pearson"],
                            "cv_mae": cv["cv"]["mae"],
                            "cv_rmse": cv["cv"]["rmse"],
                            "public_spearman": role_m.get("Public", {}).get("spearman"),
                            "private_spearman": role_m.get("Private", {}).get("spearman"),
                            "public_mae": role_m.get("Public", {}).get("mae"),
                            "private_mae": role_m.get("Private", {}).get("mae"),
                            "feature_class": "PARTICIPANT_LEGAL",
                            "best_params": json.dumps(cv["params"]),
                        }
                    )

                # XGBoost on fusion + best PLM alone
                print(f"XGB {split_name} {target} {fname}", flush=True)
                try:
                    cvx = tplm.run_xgb(X[dev], y[dev], groups[dev])
                    model = cvx["model"]
                    role_m = {}
                    for role in ["Public", "Private"]:
                        mask = roles == role
                        pred = model.predict(X[mask])
                        role_m[role] = tplm.metrics_dict(y[mask], pred)
                    rows.append(
                        {
                            "representation": fname,
                            "model": "XGBoost",
                            "target": target,
                            "split": split_name,
                            "cv_spearman": cvx["cv"]["spearman"],
                            "cv_pearson": cvx["cv"]["pearson"],
                            "cv_mae": cvx["cv"]["mae"],
                            "cv_rmse": cvx["cv"]["rmse"],
                            "public_spearman": role_m.get("Public", {}).get("spearman"),
                            "private_spearman": role_m.get("Private", {}).get("spearman"),
                            "feature_class": "PARTICIPANT_LEGAL",
                            "best_params": json.dumps(cvx["params"]),
                        }
                    )
                except Exception as e:
                    print("XGB_FAIL", e)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(METRICS / "stage_fusion_xgb_results.csv", index=False)
        print("FUSION_OK", len(out))
    else:
        print("FUSION_EMPTY")


def full_target_confirm():
    """Run leading 3–4 families on FULL target sets."""
    ensure_dirs()
    # Use existing train_ABC style for A2, B, C and PLM if available
    import importlib

    train = importlib.import_module("05_train_ABC")
    from b1_common import CONFIG, read_json

    registry = read_json(CONFIG / "representation_registry.json")
    rows = []
    for tname, fname, tcol in [
        ("PSR", "psr_full", "psr_score"),
        ("HIC", "hic_full", "hic_rt_min"),
        ("TmApp", "tmapp_full", "tm_app_C"),
    ]:
        d = pd.read_csv(DATA / f"{tname.lower()}_full.csv")
        # clusters from split file
        sp = pd.read_csv(SPLITS / f"{fname}_canonical.csv")
        d = d.merge(sp, on="antibody_id")
        # fake TARGET_COLS by temporarily using column — run_rep_target expects TARGET_COLS keys
        # simpler: manual loop with nested cv from train module via patching
        for rep in ["A2_physchem", "B_cdr_descriptors", "C_BIO_SHORTCUT"]:
            # Build mini target using existing machinery: write temp with standard col names
            tmp = d.copy()
            # ensure cluster_id
            print(f"FULL {tname} {rep}", flush=True)
            # Use run_rep_target with target name but need matching TARGET_COLS
            # Monkey: set column already named correctly in full files
            try:
                r, _ = train.run_rep_target(rep, tname, f"{fname}_canonical", registry, tmp, sp)
                for rec in r:
                    rec["dataset"] = f"{tname}_FULL"
                    rows.append(rec)
            except Exception as e:
                print("FULL_FAIL", tname, rep, e)

        # PLM if ready
        man = CACHE / "plm" / "manifest_ablang2_default.csv"
        if man.exists():
            try:
                X = tplm.load_embedding_matrix(man, d["antibody_id"])
                y = d[TARGET_COLS[tname]].values.astype(float)
                roles = d["role"].values
                groups = d["cluster_id"].values
                dev = roles == "Dev"
                cv = tplm.nested_cv_dense(X[dev], y[dev], groups[dev], "Ridge")
                model = cv["model"]
                role_m = {}
                for role in ["Public", "Private"]:
                    mask = roles == role
                    role_m[role] = tplm.metrics_dict(y[mask], model.predict(X[mask]))
                rows.append(
                    {
                        "representation": "PLM_ablang2_default",
                        "model": "Ridge",
                        "target": tname,
                        "split": f"{fname}_canonical",
                        "dataset": f"{tname}_FULL",
                        "cv_spearman": cv["cv"]["spearman"],
                        "public_spearman": role_m.get("Public", {}).get("spearman"),
                        "private_spearman": role_m.get("Private", {}).get("spearman"),
                        "feature_class": "PARTICIPANT_LEGAL",
                    }
                )
            except Exception as e:
                print("FULL_PLM_FAIL", tname, e)

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "full_target_confirmation.csv", index=False)
    lines = ["# Full-target confirmation", "", "Leading pipelines on PSR_FULL / HIC_FULL / TMAPP_FULL.", ""]
    if len(out):
        for t in ["PSR", "HIC", "TmApp"]:
            lines.append(f"## {t}_FULL")
            sub = out[out.target == t]
            lines.append("| rep | model | CV ρ | Pub ρ | Priv ρ |")
            lines.append("|-----|-------|------:|------:|-------:|")
            for _, r in sub.sort_values("cv_spearman", ascending=False).iterrows():
                lines.append(
                    f"| {r['representation']} | {r['model']} | {r['cv_spearman']:.3f} | "
                    f"{r.get('public_spearman', float('nan')):.3f} | {r.get('private_spearman', float('nan')):.3f} |"
                )
            lines.append("")
    (REPORTS / "full_target_confirmation.md").write_text("\n".join(lines) + "\n")
    print("FULL_OK", len(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_target_confirm()
    else:
        main()
        full_target_confirm()
