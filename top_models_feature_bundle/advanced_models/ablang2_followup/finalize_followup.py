#!/usr/bin/env python3
"""Post-training AbLang2 follow-up: ensemble, residual, postmortem, benchmark merge."""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BUNDLE = Path(__file__).resolve().parents[2]
ROOT = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(BUNDLE / "scripts"))

from advanced_models.data import load_dev_test, load_folds, load_solution  # noqa: E402
from advanced_models.metrics import mae, score_solution  # noqa: E402
from build_model_benchmark_summary import load_oof_series, load_xgb_oof  # noqa: E402

FOLLOWUP = Path(__file__).resolve().parent
OUT = BUNDLE / "advanced_outputs" / "ablang2_followup"
RES = BUNDLE / "results" / "ablang2_followup"
MASTER = BUNDLE / "results" / "MODEL_BENCHMARK_SUMMARY.csv"
XENS = BUNDLE / "results" / "cross_family_ensemble_ablang2_followup"
PHASE1_XENS = BUNDLE / "results" / "cross_family_ensemble"
CV_PROTO = "canonical_simple_tvt_primary_shadow"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(**kw):
    p = FOLLOWUP / "FOLLOWUP_STATUS.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kw)
    d["timestamp"] = utcnow()
    p.write_text(json.dumps(d, indent=2) + "\n")


def pearson_resid(y, a, b):
    ra = (y - a).to_numpy(float)
    rb = (y - b).to_numpy(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def postmortem_pred(path: Path, sol):
    empty = {
        "public_mae": np.nan,
        "private_mae": np.nan,
        "overall_test_mae": np.nan,
        "bias": np.nan,
        "rmse": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
    }
    if sol is None or not path.exists():
        return empty
    pred = pd.read_csv(path)
    pred["id"] = pred["id"].astype(str)
    col = "prediction" if "prediction" in pred.columns else [c for c in pred.columns if c != "id"][0]
    sc = score_solution(pred.rename(columns={col: "prediction"}), sol, "TmApp")
    m = pred.merge(sol[["id", "TmApp"]], on="id")
    err = m[col].to_numpy(float) - m["TmApp"].to_numpy(float)
    sc["bias"] = float(err.mean())
    sc["rmse"] = float(np.sqrt((err**2).mean()))
    sc["pearson"] = float(stats.pearsonr(m[col], m["TmApp"])[0])
    sc["spearman"] = float(stats.spearmanr(m[col], m["TmApp"])[0])
    return sc


def main():
    RES.mkdir(parents=True, exist_ok=True)
    XENS.mkdir(parents=True, exist_ok=True)
    write_status(stage="ENSEMBLE")

    dev, test = load_dev_test(BUNDLE / "dev.csv", BUNDLE / "test.csv")
    folds = load_folds()
    dev_ids = dev["id"].astype(str).tolist()
    y = dev.set_index(dev["id"].astype(str))["TmApp"].astype(float)

    seq = pd.read_csv(RES / "ABLANG2_SEQUENCE_RESULTS.csv")
    fus = pd.read_csv(RES / "ABLANG2_FUSION_RESULTS.csv")
    sel = json.loads((RES / "ABLANG2_FINAL_CV_SELECTION.json").read_text())
    best_seq = sel["best_ablang2_sequence"]["id"]
    best_fus_id = sel["best_ablang2_fusion"]["id"]

    master = pd.read_csv(MASTER)
    if "experiment_phase" not in master.columns:
        master["experiment_phase"] = "PHASE1_CLOSURE"
    master = master[
        ~master["experiment_phase"].astype(str).eq("ABLANG2_POSITION_AWARE_FOLLOWUP")
    ].copy()

    old_cf = master[
        (master.target == "TmApp") & (master.model_id.str.startswith("CROSS_FAMILY_EQUAL_MEAN__TmApp"))
    ].iloc[0]
    old_tmf2f = master[
        master.model_id.astype(str).str.contains("TMF2__FUSION__TM_BASE_BIOEMU_MPNN")
    ].iloc[0]

    follow_rows = []
    for _, r in seq.iterrows():
        pred = f"advanced_outputs/ablang2_followup/predictions/TmApp__seq__{r.variant_id}__test.csv"
        follow_rows.append(
            dict(
                target="TmApp",
                model_id=r.variant_id,
                family="TRANSFORMER_SEQUENCE",
                base_model="Frozen AbLang2 Transformer",
                feature_recipe="",
                transformer_variant=r.variant_id,
                cv_primary_mae=float(r.primary_mae),
                cv_shadow_mae=float(r.shadow_mae),
                cv_mean_mae=float(r.cv_mean_mae),
                cv_worst_mae=float(r.cv_worst_mae),
                public_mae=np.nan,
                private_mae=np.nan,
                overall_test_mae=np.nan,
                cv_protocol=CV_PROTO,
                cv_comparable="YES",
                cv_selected="NO",
                family_winner="NO",
                recommended="NO",
                public_private_role="POSTMORTEM_ONLY",
                prediction_path=pred if (BUNDLE / pred).exists() else "NA",
                submission_path="NA",
                source_results_path="results/ablang2_followup/ABLANG2_SEQUENCE_RESULTS.csv",
                reproducible="YES",
                notes="AbLang2 position-aware follow-up sequence-only",
                experiment_phase="ABLANG2_POSITION_AWARE_FOLLOWUP",
            )
        )
    for _, r in fus.iterrows():
        pred = (
            "advanced_outputs/ablang2_followup/predictions/"
            f"TmApp__fusion__{r.variant_id}__test.csv"
        )
        follow_rows.append(
            dict(
                target="TmApp",
                model_id=r.variant_id,
                family="TRANSFORMER_FUSION",
                base_model="Frozen AbLang2 Transformer fusion",
                feature_recipe=r.fixed_recipe,
                transformer_variant=r.transformer_variant,
                cv_primary_mae=float(r.primary_mae),
                cv_shadow_mae=float(r.shadow_mae),
                cv_mean_mae=float(r.cv_mean_mae),
                cv_worst_mae=float(r.cv_worst_mae),
                public_mae=np.nan,
                private_mae=np.nan,
                overall_test_mae=np.nan,
                cv_protocol=CV_PROTO,
                cv_comparable="YES",
                cv_selected="NO",
                family_winner="NO",
                recommended="NO",
                public_private_role="POSTMORTEM_ONLY",
                prediction_path=pred if (BUNDLE / pred).exists() else "NA",
                submission_path="NA",
                source_results_path="results/ablang2_followup/ABLANG2_FUSION_RESULTS.csv",
                reproducible="YES",
                notes="AbLang2 position-aware follow-up fusion",
                experiment_phase="ABLANG2_POSITION_AWARE_FOLLOWUP",
            )
        )

    combined = pd.concat([master, pd.DataFrame(follow_rows)], ignore_index=True)
    fams = ["LINEAR", "LINEAR_ENSEMBLE", "XGBOOST", "TRANSFORMER_SEQUENCE", "TRANSFORMER_FUSION"]
    # Family winners: preserve Phase-1 LINEAR_ENSEMBLE flags (TmApp equal-mean was not CV-selected)
    combined["family_winner"] = "NO"
    for target in ("TmApp", "HIC"):
        for fam in fams:
            sub = combined[
                (combined.target == target)
                & (combined.family == fam)
                & (combined.cv_comparable == "YES")
            ]
            if sub.empty:
                continue
            if fam == "LINEAR_ENSEMBLE" and target == "TmApp":
                # Keep Phase-1: TmApp Top-3 equal-mean was NOT the CV-selected linear ensemble
                continue
            idx = sub.sort_values(["cv_worst_mae", "cv_mean_mae", "model_id"]).index[0]
            combined.loc[idx, "family_winner"] = "YES"
        # restore HIC linear ensemble winner from phase1
        if target == "HIC":
            hit = combined[
                (combined.target == "HIC")
                & (combined.family == "LINEAR_ENSEMBLE")
                & (combined.cv_comparable == "YES")
            ]
            if len(hit):
                combined.loc[
                    hit.sort_values(["cv_worst_mae", "cv_mean_mae", "model_id"]).index[0],
                    "family_winner",
                ] = "YES"

    # Cross-family TmApp
    winners = combined[
        (combined.target == "TmApp")
        & (combined.family_winner == "YES")
        & (combined.family.isin(fams))
    ]
    print("[ensemble] winners:", winners.model_id.tolist())

    oof_p, oof_s, test_pred = {}, {}, {}
    skipped = []
    for _, c in winners.iterrows():
        mid = c.model_id
        try:
            if c.family == "XGBOOST":
                path = PHASE1_XENS / f"xgb_oof_TmApp_{c.feature_recipe}.npz"
                if not path.exists():
                    alts = list(PHASE1_XENS.glob("xgb_oof_TmApp_*.npz"))
                    path = alts[0]
                oof_p[mid] = load_xgb_oof(path, "primary", dev_ids)
                oof_s[mid] = load_xgb_oof(path, "shadow", dev_ids)
            else:
                oof_p[mid] = load_oof_series(mid, "TmApp", "primary", dev_ids)
                oof_s[mid] = load_oof_series(mid, "TmApp", "shadow", dev_ids)
            tp_path = BUNDLE / str(c.prediction_path)
            if not tp_path.exists():
                tp_path = ROOT / str(c.prediction_path)
            if not tp_path.exists() or str(c.prediction_path) in ("", "NA"):
                raise FileNotFoundError(f"missing test pred for {mid}")
            tp = pd.read_csv(tp_path)
            tp["id"] = tp["id"].astype(str)
            col = "prediction" if "prediction" in tp.columns else [x for x in tp.columns if x != "id"][0]
            test_pred[mid] = tp.set_index("id")[col].astype(float)
        except Exception as e:
            skipped.append({"model_id": mid, "reason": str(e)})
            oof_p.pop(mid, None)
            oof_s.pop(mid, None)
            test_pred.pop(mid, None)
    (XENS / "ENSEMBLE_SKIPPED_CANDIDATES.json").write_text(
        json.dumps(skipped, indent=2) + "\n"
    )
    print("[ensemble] usable:", list(oof_p.keys()), "skipped:", skipped)

    ids_c = list(oof_p.keys())
    subset_rows = []
    for r in range(1, len(ids_c) + 1):
        for subset in itertools.combinations(ids_c, r):
            subset = list(subset)
            pp = sum(oof_p[m] for m in subset) / len(subset)
            ss = sum(oof_s[m] for m in subset) / len(subset)
            p_mae = mae(y.loc[dev_ids], pp.loc[dev_ids])
            s_mae = mae(y.loc[dev_ids], ss.loc[dev_ids])
            subset_rows.append(
                {
                    "subset": "+".join(subset),
                    "n_models": len(subset),
                    "primary_mae": p_mae,
                    "shadow_mae": s_mae,
                    "cv_mean_mae": 0.5 * (p_mae + s_mae),
                    "cv_worst_mae": max(p_mae, s_mae),
                    "members": subset,
                }
            )
    sdf = pd.DataFrame(subset_rows).sort_values(["cv_worst_mae", "cv_mean_mae", "n_models"])
    sdf.to_csv(XENS / "TmApp__equal_mean_subsets.csv", index=False)
    best = sdf.iloc[0]
    members = list(best.members)
    te_ids = test["id"].astype(str).tolist()
    te = sum(test_pred[m].reindex(te_ids) for m in members) / len(members)
    te_df = pd.DataFrame({"id": te_ids, "prediction": te.to_numpy(float)})
    te_df.to_csv(XENS / "TmApp__cross_family_equal_mean_test.csv", index=False)
    (XENS / "TmApp__STACKING_SKIP.md").write_text(
        "# Stacking SKIP\n\nSame Phase-1 policy: stacking not required for this follow-up.\n"
    )
    (XENS / "HIC__STACKING_SKIP.md").write_text(
        "# HIC unchanged\n\nNo new HIC AbLang2 exploration; Phase-1 HIC cross-family retained.\n"
    )

    # Residuals
    t1_p = load_oof_series("TM_PARENT_ABLINGUA_CDR3__RIDGE", "TmApp", "primary", dev_ids)
    t1_s = load_oof_series("TM_PARENT_ABLINGUA_CDR3__RIDGE", "TmApp", "shadow", dev_ids)
    tmf2_p = load_oof_series(str(old_tmf2f.model_id), "TmApp", "primary", dev_ids)
    tmf2_s = load_oof_series(str(old_tmf2f.model_id), "TmApp", "shadow", dev_ids)
    seq_p = load_oof_series(best_seq, "TmApp", "primary", dev_ids)
    seq_s = load_oof_series(best_seq, "TmApp", "shadow", dev_ids)
    fus_p = load_oof_series(best_fus_id, "TmApp", "primary", dev_ids)
    fus_s = load_oof_series(best_fus_id, "TmApp", "shadow", dev_ids)
    ens_p = sum(oof_p[m] for m in members) / len(members)
    ens_s = sum(oof_s[m] for m in members) / len(members)
    models_oof = {
        "linear_T1": (t1_p, t1_s),
        "old_TMF2_fusion": (tmf2_p, tmf2_s),
        "best_ablang2_seq": (seq_p, seq_s),
        "best_ablang2_fusion": (fus_p, fus_s),
        "followup_cross_family": (ens_p, ens_s),
    }
    resid_rows = []
    names = list(models_oof)
    for scheme, gi in (("primary", 0), ("shadow", 1)):
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                resid_rows.append(
                    {
                        "scheme": scheme,
                        "model_a": a,
                        "model_b": b,
                        "residual_pearson": pearson_resid(
                            y.loc[dev_ids],
                            models_oof[a][gi].loc[dev_ids],
                            models_oof[b][gi].loc[dev_ids],
                        ),
                    }
                )
    pd.DataFrame(resid_rows).to_csv(RES / "ABLANG2_RESIDUAL_CORRELATIONS.csv", index=False)

    write_status(stage="POSTMORTEM")
    sol = load_solution(BUNDLE / "solution.csv") if (BUNDLE / "solution.csv").exists() else None
    pm = {
        "solution_available": sol is not None,
        "phase1_cross_family": {
            **postmortem_pred(
                PHASE1_XENS / "TmApp__cross_family_equal_mean_test.csv", sol
            ),
            "model_id": str(old_cf.model_id),
            "cv_primary": float(old_cf.cv_primary_mae),
            "cv_shadow": float(old_cf.cv_shadow_mae),
            "cv_worst": float(old_cf.cv_worst_mae),
        },
        "best_ablang2_sequence": postmortem_pred(
            OUT / "predictions" / f"TmApp__seq__{best_seq}__test.csv", sol
        ),
        "best_ablang2_fusion": postmortem_pred(
            OUT / "predictions" / f"TmApp__fusion__{best_fus_id}__test.csv", sol
        ),
        "followup_cross_family": postmortem_pred(
            XENS / "TmApp__cross_family_equal_mean_test.csv", sol
        ),
        "hic_phase1_cross_family_ref": {},
    }
    hic_cf = master[master.model_id == "CROSS_FAMILY_EQUAL_MEAN__HIC__2m"]
    if len(hic_cf):
        r = hic_cf.iloc[0]
        pm["hic_phase1_cross_family_ref"] = {
            "model_id": str(r.model_id),
            "public_mae": float(r.public_mae) if pd.notna(r.public_mae) else None,
            "private_mae": float(r.private_mae) if pd.notna(r.private_mae) else None,
            "overall_test_mae": float(r.overall_test_mae) if pd.notna(r.overall_test_mae) else None,
            "cv_worst": float(r.cv_worst_mae),
        }
    (RES / "ABLANG2_POSTMORTEM.json").write_text(json.dumps(pm, indent=2, default=str) + "\n")

    # Fill postmortem into follow rows
    for row in follow_rows:
        p = BUNDLE / row["prediction_path"] if row["prediction_path"] != "NA" else None
        if p and p.exists():
            sc = postmortem_pred(p, sol)
            row["public_mae"] = sc["public_mae"]
            row["private_mae"] = sc["private_mae"]
            row["overall_test_mae"] = sc["overall_test_mae"]

    ens_id = f"CROSS_FAMILY_EQUAL_MEAN__TmApp__{int(best.n_models)}m__FOLLOWUP"
    ens_sc = pm["followup_cross_family"]
    follow_rows.append(
        dict(
            target="TmApp",
            model_id=ens_id,
            family="CROSS_FAMILY_ENSEMBLE",
            base_model="equal_mean",
            feature_recipe="|".join(members),
            transformer_variant="",
            cv_primary_mae=float(best.primary_mae),
            cv_shadow_mae=float(best.shadow_mae),
            cv_mean_mae=float(best.cv_mean_mae),
            cv_worst_mae=float(best.cv_worst_mae),
            public_mae=ens_sc.get("public_mae", np.nan),
            private_mae=ens_sc.get("private_mae", np.nan),
            overall_test_mae=ens_sc.get("overall_test_mae", np.nan),
            cv_protocol=CV_PROTO,
            cv_comparable="YES",
            cv_selected="NO",
            family_winner="NO",
            recommended="NO",
            public_private_role="POSTMORTEM_ONLY",
            prediction_path=str((XENS / "TmApp__cross_family_equal_mean_test.csv").relative_to(BUNDLE)),
            submission_path="NA",
            source_results_path=str((XENS / "TmApp__equal_mean_subsets.csv").relative_to(BUNDLE)),
            reproducible="YES",
            notes=f"follow-up equal_mean members={members}; stacking SKIP",
            experiment_phase="ABLANG2_POSITION_AWARE_FOLLOWUP",
        )
    )

    # Final CV selection update
    fus_row = fus.loc[fus.variant_id == best_fus_id].iloc[0]
    candidates = [
        (
            "PHASE1_CROSS_FAMILY",
            str(old_cf.model_id),
            float(old_cf.cv_primary_mae),
            float(old_cf.cv_shadow_mae),
            float(old_cf.cv_worst_mae),
            float(old_cf.cv_mean_mae),
            str(old_cf.feature_recipe).split("|"),
        ),
        (
            "BEST_ABLANG2_FUSION",
            best_fus_id,
            float(fus_row.primary_mae),
            float(fus_row.shadow_mae),
            float(fus_row.cv_worst_mae),
            float(fus_row.cv_mean_mae),
            [best_fus_id],
        ),
        (
            "FOLLOWUP_CROSS_FAMILY",
            ens_id,
            float(best.primary_mae),
            float(best.shadow_mae),
            float(best.cv_worst_mae),
            float(best.cv_mean_mae),
            members,
        ),
    ]
    candidates.sort(key=lambda x: (x[4], x[5], len(x[6])))
    win = candidates[0]
    final = {
        "best_ablang2_sequence": sel["best_ablang2_sequence"],
        "best_ablang2_fusion": sel["best_ablang2_fusion"],
        "updated_cross_family": {
            "id": ens_id,
            "members": members,
            "primary": float(best.primary_mae),
            "shadow": float(best.shadow_mae),
            "worst": float(best.cv_worst_mae),
            "mean": float(best.cv_mean_mae),
            "weights": "equal_mean",
        },
        "BEST_CV_TmApp": {
            "source": win[0],
            "id": win[1],
            "primary": win[2],
            "shadow": win[3],
            "worst": win[4],
            "mean": win[5],
            "members": win[6],
        },
        "phase1_cross_family": {
            "id": str(old_cf.model_id),
            "primary": float(old_cf.cv_primary_mae),
            "shadow": float(old_cf.cv_shadow_mae),
            "worst": float(old_cf.cv_worst_mae),
            "public": float(old_cf.public_mae) if pd.notna(old_cf.public_mae) else None,
            "private": float(old_cf.private_mae) if pd.notna(old_cf.private_mae) else None,
        },
        "winner_changed_vs_phase1": win[1] != str(old_cf.model_id),
        "selection_policy": "CV only: min cv_worst, then cv_mean, then fewer/simpler; Public/Private unused",
        "timestamp": utcnow(),
    }
    (RES / "ABLANG2_FINAL_CV_SELECTION.json").write_text(json.dumps(final, indent=2) + "\n")

    # Merge master
    master2 = pd.concat([master, pd.DataFrame(follow_rows)], ignore_index=True)
    master2["family_winner"] = "NO"
    master2["cv_selected"] = "NO"
    master2["recommended"] = "NO"
    for target in ("TmApp", "HIC"):
        for fam in master2.family.unique():
            if fam == "HISTORICAL_CONTEXT":
                continue
            if fam == "LINEAR_ENSEMBLE" and target == "TmApp":
                continue
            sub = master2[
                (master2.target == target)
                & (master2.family == fam)
                & (master2.cv_comparable == "YES")
            ]
            if sub.empty:
                continue
            idx = sub.sort_values(["cv_worst_mae", "cv_mean_mae", "model_id"]).index[0]
            master2.loc[idx, "family_winner"] = "YES"
        # HIC linear ensemble
        if target == "HIC":
            hit = master2[
                (master2.target == "HIC")
                & (master2.family == "LINEAR_ENSEMBLE")
                & (master2.cv_comparable == "YES")
            ]
            if len(hit):
                master2.loc[
                    hit.sort_values(["cv_worst_mae", "cv_mean_mae", "model_id"]).index[0],
                    "family_winner",
                ] = "YES"
        sub = master2[
            (master2.target == target)
            & (master2.cv_comparable == "YES")
            & (~master2.family.isin(["HISTORICAL_CONTEXT"]))
        ]
        if not sub.empty:
            idx = sub.sort_values(["cv_worst_mae", "cv_mean_mae", "model_id"]).index[0]
            master2.loc[idx, "cv_selected"] = "YES"
            master2.loc[idx, "recommended"] = "YES"
    master2["public_private_role"] = "POSTMORTEM_ONLY"
    master2.to_csv(MASTER, index=False)
    write_status(stage="BENCHMARK_UPDATE")
    print("BEST_CV", final["BEST_CV_TmApp"])
    print("winner_changed", final["winner_changed_vs_phase1"])
    print("master rows", len(master2))
    (XENS / "CROSS_FAMILY_META.json").write_text(
        json.dumps(
            {
                "TmApp": final["updated_cross_family"],
                "BEST_CV": final["BEST_CV_TmApp"],
                "winner_changed_vs_phase1": final["winner_changed_vs_phase1"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
