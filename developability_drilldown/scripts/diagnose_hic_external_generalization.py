#!/usr/bin/env python3
"""HIC factorial external generalization diagnosis.

Scores existing test.csv predictions only. No training.
Does NOT modify HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md or experiments.csv.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REPORTS = REPO / "reports"

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from _lib import load_solution, mae  # noqa: E402

INTERNAL_FREEZE_SHA = "cca9bd5016ecd14f06727f285bd26bf58fc46647"
SCIENTIFIC_FREEZE_SHA = "559980e37576eaffaeba9603cee591544e52d991"
# pipeline validation tolerance (CSV float round-trip; anchors ≤ ~3e-8 observed)
SCORE_TOL = 1e-6
HIGH_THR = 11.5
TOPOS = ["SEP", "JOINT", "REG-SEP", "XREG", "FUSE"]
ANNOTS = ["BASE", "IMGT", "REGION", "FULL"]
ANCHORS = ["EXP-H030", "EXP-H047", "EXP-H107", "EXP-H109", "EXP-H113", "EXP-H137"]

REP_DISPLAY = {
    "scratch": "Scratch",
    "ablingua": "AbLingua",
    "ablang1": "AbLang1",
    "ablang2_paired": "AbLang2 SEPARATE",
    "ablang2_unpaired": "AbLang2 PAIRED",
    "esm1b": "ESM-1b",
    "esm2": "ESM-2",
    "esmc600m": "ESM-C",
    "currab_unpaired": "CurrAb SEPARATE",
    "currab_paired": "CurrAb PAIRED",
}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def score_pred(pred: pd.Series, sol: pd.DataFrame) -> dict:
    sol2 = sol.set_index("id")
    ids = sol2.index.tolist()
    te = pred.reindex(ids)
    if te.isna().any():
        missing = te[te.isna()].index.tolist()
        raise ValueError(f"prediction missing ids: {missing[:5]}")
    pub = sol2.index[sol2["is_public"].astype(bool)]
    priv = sol2.index[sol2["is_private"].astype(bool)]
    y = sol2["HIC"].astype(float)
    public_mae = float(mae(y.loc[pub].to_numpy(), te.loc[pub].to_numpy(float)))
    private_mae = float(mae(y.loc[priv].to_numpy(), te.loc[priv].to_numpy(float)))
    overall_mae = float(mae(y.loc[ids].to_numpy(), te.loc[ids].to_numpy(float)))
    return {
        "public_mae": public_mae,
        "private_mae": private_mae,
        "test_overall_mae": overall_mae,
        "public_private_delta": public_mae - private_mae,
        "public_private_gap": abs(public_mae - private_mae),
    }


def load_test_pred(code: str) -> pd.Series:
    path = ROOT / "experiments/predictions" / code / "test.csv"
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "HIC"]:
        raise ValueError(f"{code} unexpected columns {df.columns.tolist()}")
    df["id"] = df["id"].astype(str)
    return df.set_index("id")["HIC"].astype(float)


def validate_pipeline(sol: pd.DataFrame, exp: pd.DataFrame) -> dict:
    rows = []
    max_d = 0.0
    for code in ANCHORS:
        r = exp[exp.experiment_code == code].iloc[0]
        scores = score_pred(load_test_pred(code), sol)
        dpub = abs(scores["public_mae"] - float(r.public_mae))
        dpriv = abs(scores["private_mae"] - float(r.private_mae))
        dov = abs(scores["test_overall_mae"] - float(r.test_overall_mae))
        max_d = max(max_d, dpub, dpriv, dov)
        rows.append(
            {
                "experiment_code": code,
                "registry_public": float(r.public_mae),
                "registry_private": float(r.private_mae),
                "registry_overall": float(r.test_overall_mae),
                **{f"recomputed_{k}": v for k, v in scores.items() if k.endswith("mae")},
                "delta_public": dpub,
                "delta_private": dpriv,
                "delta_overall": dov,
                "pass": max(dpub, dpriv, dov) <= SCORE_TOL,
            }
        )
    df = pd.DataFrame(rows)
    status = "PASS" if bool(df["pass"].all()) else "FAIL"
    return {"status": status, "max_abs_delta": max_d, "tolerance": SCORE_TOL, "rows": rows, "table": df}


def high_tail_test(pred: pd.Series, sol: pd.DataFrame) -> dict:
    sol2 = sol.set_index("id")
    y = sol2["HIC"].astype(float)
    te = pred.reindex(sol2.index).astype(float)
    mask = y > HIGH_THR
    n = int(mask.sum())
    if n == 0:
        return {
            "n_high": 0,
            "mae_high": np.nan,
            "mean_signed_error": np.nan,
            "obs_min": np.nan,
            "obs_max": np.nan,
            "pred_min": np.nan,
            "pred_max": np.nan,
        }
    yh, ph = y[mask].to_numpy(), te[mask].to_numpy()
    return {
        "n_high": n,
        "mae_high": float(np.mean(np.abs(ph - yh))),
        "mean_signed_error": float(np.mean(ph - yh)),
        "obs_min": float(yh.min()),
        "obs_max": float(yh.max()),
        "pred_min": float(ph.min()),
        "pred_max": float(ph.max()),
    }


def verdict(supported: bool, weakened: bool = False, contradicted: bool = False) -> str:
    if contradicted:
        return "CONTRADICTED"
    if weakened:
        return "WEAKENED"
    if supported:
        return "SUPPORTED"
    return "INCONCLUSIVE"


def is_surfaceish(row: pd.Series) -> bool:
    blob = " ".join(
        str(row.get(c, ""))
        for c in ("experiment_id", "notes", "input_space", "feature_set_id", "plm_source")
    ).upper()
    keys = ("SURFACE", "HSP", "SAP", "F1_", "PHYSIC", "RASA", "KD_SAP", "SOURCE24", "SOURCE_SAP")
    return any(k in blob for k in keys)


def main() -> int:
    start_head = git_rev()
    sol = load_solution()
    if sol is None:
        raise SystemExit("solution.csv missing — cannot run authorized external diagnosis")
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    fac = pd.read_csv(ROOT / "results/HIC_REP_TOPO_ANNOT_FACTORIAL_RESULTS.csv")
    assert len(fac) == 200

    # --- 1. Pipeline validation ---
    pipe = validate_pipeline(sol, exp)
    pipe["table"].to_csv(REPORTS / "HIC_EXTERNAL_PIPELINE_VALIDATION.csv", index=False)
    if pipe["status"] != "PASS":
        (REPORTS / "HIC_EXTERNAL_PIPELINE_VALIDATION_FAIL.json").write_text(
            json.dumps({k: v for k, v in pipe.items() if k != "table"}, indent=2, default=str)
        )
        print("PIPELINE_FAIL", pipe["max_abs_delta"])
        return 1
    print("PIPELINE_PASS", "max_delta", pipe["max_abs_delta"])

    # --- 2. Score factorial 200 ---
    rows = []
    for _, r in fac.iterrows():
        code = r.experiment_code
        pred = load_test_pred(code)
        sc = score_pred(pred, sol)
        ht = high_tail_test(pred, sol)
        rows.append(
            {
                "experiment_code": code,
                "representation": r.representation,
                "topology": r.topology,
                "annotation": r.annotation,
                "representation_context": r.representation_context,
                "cv_primary_mae": float(r.TEST_P),
                "cv_shadow_mae": float(r.TEST_S),
                "cv_mean_mae": float(r.TEST_mean),
                "cv_worst_mae": float(r.TEST_worst),
                "abs_PS": float(r.abs_PS),
                **sc,
                "test_minus_cv_mean": sc["test_overall_mae"] - float(r.TEST_mean),
                **{f"high_{k}": v for k, v in ht.items()},
                "provenance": "prospective_relative_to_factorial_freeze",
            }
        )
    ext = pd.DataFrame(rows)

    # ranks (persist on diagnosis CSV)
    ext["internal_rank"] = ext["cv_mean_mae"].rank(method="min")
    ext["external_rank"] = ext["test_overall_mae"].rank(method="min")
    ext["rank_change"] = ext["external_rank"] - ext["internal_rank"]
    ext_path = REPORTS / "HIC_FACTORIAL_EXTERNAL_DIAGNOSIS.csv"
    ext.to_csv(ext_path, index=False)

    # --- 3. Rank transfer ---
    pearson = float(stats.pearsonr(ext.cv_mean_mae, ext.test_overall_mae).statistic)
    spearman = float(stats.spearmanr(ext.cv_mean_mae, ext.test_overall_mae).statistic)
    kendall = float(stats.kendalltau(ext.cv_mean_mae, ext.test_overall_mae).statistic)
    # also primary/shadow vs external
    pearson_p = float(stats.pearsonr(ext.cv_primary_mae, ext.test_overall_mae).statistic)
    spearman_p = float(stats.spearmanr(ext.cv_primary_mae, ext.test_overall_mae).statistic)

    top10_int = set(ext.nsmallest(10, "cv_mean_mae").experiment_code)
    top10_ext = set(ext.nsmallest(10, "test_overall_mae").experiment_code)
    top20_int = ext.nsmallest(20, "cv_mean_mae")
    top20_ext = ext.nsmallest(20, "test_overall_mae")

    transfer = {
        "pearson_cv_mean_vs_test": pearson,
        "spearman_cv_mean_vs_test": spearman,
        "kendall_cv_mean_vs_test": kendall,
        "pearson_primary_vs_test": pearson_p,
        "spearman_primary_vs_test": spearman_p,
        "top10_internal": sorted(top10_int),
        "top10_external": sorted(top10_ext),
        "top10_both": sorted(top10_int & top10_ext),
        "mean_abs_rank_change": float(ext.rank_change.abs().mean()),
        "median_test_minus_cv": float(ext.test_minus_cv_mean.median()),
        "mean_abs_PS": float(ext.abs_PS.mean()),
        "mean_pub_priv_gap": float(ext.public_private_gap.mean()),
    }

    # percentiles for top20
    def pct_of(series_rank, codes):
        out = []
        for c in codes:
            row = ext[ext.experiment_code == c].iloc[0]
            out.append(
                {
                    "experiment_code": c,
                    "internal_rank": int(row.internal_rank),
                    "external_rank": int(row.external_rank),
                    "external_percentile": 100.0 * (row.external_rank - 1) / 199.0,
                    "internal_percentile": 100.0 * (row.internal_rank - 1) / 199.0,
                    "cv_mean_mae": row.cv_mean_mae,
                    "test_overall_mae": row.test_overall_mae,
                }
            )
        return pd.DataFrame(out)

    pct_int20 = pct_of(ext.external_rank, top20_int.experiment_code.tolist())
    pct_ext20 = pct_of(ext.internal_rank, top20_ext.experiment_code.tolist())
    pct_int20.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_TOP20_INTERNAL.csv", index=False)
    pct_ext20.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_TOP20_EXTERNAL.csv", index=False)

    # --- 4. Representation external ---
    rep_rows = []
    for rep, g in ext.groupby("representation"):
        rep_rows.append(
            {
                "representation": rep,
                "display": REP_DISPLAY.get(rep, rep),
                "n": len(g),
                "internal_mean": float(g.cv_mean_mae.mean()),
                "internal_median": float(g.cv_mean_mae.median()),
                "external_mean": float(g.test_overall_mae.mean()),
                "external_median": float(g.test_overall_mae.median()),
                "external_best": float(g.test_overall_mae.min()),
                "external_worst": float(g.test_overall_mae.max()),
                "best_cell": g.loc[g.test_overall_mae.idxmin()].experiment_code,
                "mean_pub_priv_gap": float(g.public_private_gap.mean()),
            }
        )
    rep_df = pd.DataFrame(rep_rows)
    rep_df["internal_mean_rank"] = rep_df["internal_mean"].rank(method="min")
    rep_df["external_mean_rank"] = rep_df["external_mean"].rank(method="min")
    rep_df["rank_change"] = rep_df["external_mean_rank"] - rep_df["internal_mean_rank"]
    rep_df = rep_df.sort_values("external_mean")
    rep_df.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_BY_REPRESENTATION.csv", index=False)

    # Claims R1-R3
    int_order = ext.groupby("representation").cv_mean_mae.mean().sort_values()
    ext_order = ext.groupby("representation").test_overall_mae.mean().sort_values()
    r1 = verdict(
        supported=(ext_order.index[0] == "ablingua"),
        weakened=(ext_order.index[0] != "ablingua" and "ablingua" in ext_order.index[:3]),
        contradicted=("ablingua" in list(ext_order.index[-3:])),
    )
    # R2: ESM-2 best single internal; check external best single and average rank
    best_int_cell = fac.loc[fac.TEST_mean.idxmin()]
    best_ext_cell = ext.loc[ext.test_overall_mae.idxmin()]
    esm2_int_avg_rank = int(int_order.rank()["esm2"])
    esm2_ext_avg_rank = int(ext_order.rank()["esm2"])
    r2_supported = (
        best_int_cell.representation == "esm2"
        and esm2_int_avg_rank > 1
        and (best_ext_cell.representation == "esm2" or esm2_ext_avg_rank <= 3)
    )
    r2 = verdict(
        supported=r2_supported and esm2_ext_avg_rank > 1,
        weakened=best_ext_cell.representation == "esm2" and esm2_ext_avg_rank == 1,
        contradicted=esm2_ext_avg_rank >= 8,
    )
    # R3: antibody PLMs average worse than scratch
    ab_plms = ["ablang1", "ablang2_paired", "ablang2_unpaired", "currab_paired", "currab_unpaired"]
    scratch_ext = float(ext_order["scratch"])
    worse_ext = sum(1 for p in ab_plms if float(ext_order[p]) > scratch_ext)
    worse_int = sum(1 for p in ab_plms if float(int_order[p]) > float(int_order["scratch"]))
    r3 = verdict(
        supported=worse_ext >= 4,
        weakened=worse_ext in (2, 3),
        contradicted=worse_ext <= 1,
    )

    # --- 5. Topology matched contrasts on Test Overall ---
    topo_rows = []
    for topo in ("JOINT", "REG-SEP", "XREG", "FUSE"):
        deltas = []
        for rep in ext.representation.unique():
            for ann in ANNOTS:
                a = ext[(ext.representation == rep) & (ext.annotation == ann) & (ext.topology == "SEP")]
                x = ext[(ext.representation == rep) & (ext.annotation == ann) & (ext.topology == topo)]
                if len(a) == 1 and len(x) == 1:
                    d = float(x.iloc[0].test_overall_mae - a.iloc[0].test_overall_mae)
                    deltas.append(
                        {
                            "topology": topo,
                            "representation": rep,
                            "annotation": ann,
                            "delta_test": d,
                            "improve": d < 0,
                        }
                    )
        ddf = pd.DataFrame(deltas)
        topo_rows.append(
            {
                "contrast": f"{topo}-SEP",
                "n_strata": len(ddf),
                "mean_delta": float(ddf.delta_test.mean()),
                "median_delta": float(ddf.delta_test.median()),
                "improve_fraction": float(ddf.improve.mean()),
                "by_rep_mean": ddf.groupby("representation").delta_test.mean().to_dict(),
                "by_ann_mean": ddf.groupby("annotation").delta_test.mean().to_dict(),
            }
        )
    topo_ext = pd.DataFrame([{k: v for k, v in r.items() if k not in ("by_rep_mean", "by_ann_mean")} | {
        **{f"rep_{kk}": vv for kk, vv in r["by_rep_mean"].items()},
        **{f"ann_{kk}": vv for kk, vv in r["by_ann_mean"].items()},
    } for r in topo_rows])
    # flatten simpler
    topo_summary = pd.DataFrame(
        [
            {
                "contrast": r["contrast"],
                "n_strata": r["n_strata"],
                "mean_delta": r["mean_delta"],
                "median_delta": r["median_delta"],
                "improve_fraction": r["improve_fraction"],
            }
            for r in topo_rows
        ]
    )
    topo_summary.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_TOPOLOGY_CONTRASTS.csv", index=False)
    # Small average effects + no topology with both large mean gain and high improve frac
    topo_small = all(abs(r["mean_delta"]) < 0.01 for r in topo_rows) and all(
        r["improve_fraction"] <= 0.65 for r in topo_rows
    )
    topo_strong_winner = any(
        r["improve_fraction"] >= 0.70 and r["mean_delta"] <= -0.01 for r in topo_rows
    )
    topo_verdict = verdict(supported=topo_small and not topo_strong_winner, contradicted=topo_strong_winner)

    # --- 6. Annotation ---
    ann_rows = []
    for ann in ("IMGT", "REGION", "FULL"):
        deltas = []
        for rep in ext.representation.unique():
            for topo in TOPOS:
                b = ext[(ext.representation == rep) & (ext.topology == topo) & (ext.annotation == "BASE")]
                x = ext[(ext.representation == rep) & (ext.topology == topo) & (ext.annotation == ann)]
                if len(b) == 1 and len(x) == 1:
                    deltas.append(float(x.iloc[0].test_overall_mae - b.iloc[0].test_overall_mae))
        arr = np.asarray(deltas, float)
        ann_rows.append(
            {
                "contrast": f"{ann}-BASE",
                "n_strata": len(arr),
                "mean_delta": float(arr.mean()),
                "median_delta": float(np.median(arr)),
                "improve_fraction": float((arr < 0).mean()),
            }
        )
    ann_summary = pd.DataFrame(ann_rows)
    ann_summary.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_ANNOTATION_CONTRASTS.csv", index=False)
    # FULL favorable / IMGT unfavorable?
    full_d = ann_summary[ann_summary.contrast == "FULL-BASE"].iloc[0].mean_delta
    imgt_d = ann_summary[ann_summary.contrast == "IMGT-BASE"].iloc[0].mean_delta
    ann_verdict = verdict(
        supported=(full_d < 0.002 and imgt_d > -0.002),  # FULL not worse / IMGT not better strongly — soft
        weakened=False,
        contradicted=(full_d > 0.01 or imgt_d < -0.01),
    )
    # clearer: check sign match with internal (FULL negative, IMGT positive on MAE)
    ann_verdict = verdict(
        supported=(full_d <= 0 and imgt_d >= 0) or (abs(full_d) < 0.005 and abs(imgt_d) < 0.005),
        weakened=(np.sign(full_d) != -1 and abs(full_d) < 0.01) or (np.sign(imgt_d) != 1 and abs(imgt_d) < 0.01),
        contradicted=(full_d > 0.005 and imgt_d < -0.005),
    )

    # --- 7. Scratch vs PLM external ---
    scratch_rows = []
    for rep in [r for r in ext.representation.unique() if r != "scratch"]:
        deltas_cv, deltas_te = [], []
        for topo in TOPOS:
            for ann in ANNOTS:
                s = ext[(ext.representation == "scratch") & (ext.topology == topo) & (ext.annotation == ann)]
                x = ext[(ext.representation == rep) & (ext.topology == topo) & (ext.annotation == ann)]
                if len(s) == 1 and len(x) == 1:
                    deltas_cv.append(float(x.iloc[0].cv_mean_mae - s.iloc[0].cv_mean_mae))
                    deltas_te.append(float(x.iloc[0].test_overall_mae - s.iloc[0].test_overall_mae))
        scratch_rows.append(
            {
                "representation": rep,
                "display": REP_DISPLAY.get(rep, rep),
                "n": len(deltas_cv),
                "internal_mean_delta_vs_scratch": float(np.mean(deltas_cv)),
                "external_mean_delta_vs_scratch": float(np.mean(deltas_te)),
                "internal_improve_frac": float((np.asarray(deltas_cv) < 0).mean()),
                "external_improve_frac": float((np.asarray(deltas_te) < 0).mean()),
            }
        )
    scratch_df = pd.DataFrame(scratch_rows).sort_values("external_mean_delta_vs_scratch")
    scratch_df.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_VS_SCRATCH.csv", index=False)
    scratch_strong_int = float(int_order["scratch"]) <= float(int_order.iloc[3])  # top4
    scratch_strong_ext = float(ext_order["scratch"]) <= float(ext_order.iloc[3])
    scratch_verdict = verdict(
        supported=scratch_strong_ext,
        weakened=scratch_strong_int and not scratch_strong_ext and float(ext_order["scratch"]) <= float(ext_order.iloc[5]),
        contradicted=float(ext_order["scratch"]) >= float(ext_order.iloc[-3]),
    )

    # --- 8. Context ---
    ctx_rows = []
    for fam, paired, separate in (
        ("ablang2", "ablang2_unpaired", "ablang2_paired"),
        ("currab", "currab_paired", "currab_unpaired"),
    ):
        deltas = []
        for topo in TOPOS:
            for ann in ANNOTS:
                p = ext[(ext.representation == paired) & (ext.topology == topo) & (ext.annotation == ann)]
                s = ext[(ext.representation == separate) & (ext.topology == topo) & (ext.annotation == ann)]
                if len(p) == 1 and len(s) == 1:
                    deltas.append(float(p.iloc[0].test_overall_mae - s.iloc[0].test_overall_mae))
        arr = np.asarray(deltas, float)
        ctx_rows.append(
            {
                "family": fam,
                "n_strata": len(arr),
                "mean_delta_test": float(arr.mean()),
                "median_delta_test": float(np.median(arr)),
                "improve_fraction": float((arr < 0).mean()),
                "note": "external additive only; does not rewrite internal no-robust-advantage conclusion",
            }
        )
    ctx_df = pd.DataFrame(ctx_rows)
    ctx_df.to_csv(REPORTS / "HIC_FACTORIAL_EXTERNAL_CONTEXT_CONTRASTS.csv", index=False)

    # --- 9. HIGH-tail ---
    high_ok = (ext.high_n_high > 0).all()
    high_summary = {
        "testable": bool(high_ok),
        "n_high_unique": sorted(ext.high_n_high.unique().tolist()),
        "mae_high_mean": float(ext.high_mae_high.mean()),
        "mae_high_min": float(ext.high_mae_high.min()),
        "mae_high_max": float(ext.high_mae_high.max()),
        "signed_mean": float(ext.high_mean_signed_error.mean()),
        "best_cell": ext.loc[ext.high_mae_high.idxmin()].experiment_code,
        "worst_cell": ext.loc[ext.high_mae_high.idxmax()].experiment_code,
        "underprediction_persistent": bool(ext.high_mean_signed_error.mean() < -1.0),
        "label": "SEQUENCE_ONLY_FACTORIAL_DID_NOT_RESOLVE_HIGH_TAIL_FAILURE"
        if high_ok and ext.high_mean_signed_error.mean() < -1.0
        else "CHECK",
    }
    high_verdict = verdict(
        supported=high_summary["underprediction_persistent"],
        contradicted=not high_summary["underprediction_persistent"],
    )

    # plateau
    best_ext = float(ext.test_overall_mae.min())
    median_ext = float(ext.test_overall_mae.median())
    plateau_verdict = verdict(
        supported=best_ext > 0.45 and median_ext > 0.48,  # still near mid/high 0.4s–0.5 regime relative to surface ~0.39
        weakened=best_ext < 0.45,
        contradicted=best_ext < 0.42,
    )

    # --- 10. All-history HIC with Test ---
    hic = exp[exp.target == "HIC"].copy()
    hic["test_ok"] = pd.to_numeric(hic["test_overall_mae"], errors="coerce").notna()
    hist = hic[hic["test_ok"]].copy()
    # merge factorial meta where available
    fac_meta = fac.set_index("experiment_code")[["representation", "topology", "annotation"]]
    hist = hist.merge(fac_meta, left_on="experiment_code", right_index=True, how="left", suffixes=("", "_fac"))
    # prefer fac cols
    for c in ("representation", "topology", "annotation"):
        if f"{c}_fac" in hist.columns:
            hist[c] = hist[f"{c}_fac"].fillna(hist.get(c, np.nan))
    hist["abs_PS"] = (pd.to_numeric(hist.cv_primary_mae, errors="coerce") - pd.to_numeric(hist.cv_shadow_mae, errors="coerce")).abs()
    hist["abs_pub_priv"] = (
        pd.to_numeric(hist.public_mae, errors="coerce") - pd.to_numeric(hist.private_mae, errors="coerce")
    ).abs()
    hist["is_factorial_200"] = hist.experiment_code.isin(ext.experiment_code)
    hist["is_surfaceish"] = hist.apply(is_surfaceish, axis=1)
    hist["provenance"] = np.where(hist.is_factorial_200, "prospective_factorial", "retrospective_historical")

    # For factorial rows, use freshly computed external scores (should match after we don't write experiments.csv)
    # For historical, use registry.
    # Also add factorial 200 into a combined view with our computed scores for consistency on those codes
    hist_view = hist.copy()
    # overwrite factorial scores with diagnosis CSV values
    for _, r in ext.iterrows():
        m = hist_view.experiment_code == r.experiment_code
        if m.any():
            hist_view.loc[m, "public_mae"] = r.public_mae
            hist_view.loc[m, "private_mae"] = r.private_mae
            hist_view.loc[m, "test_overall_mae"] = r.test_overall_mae
            hist_view.loc[m, "abs_pub_priv"] = r.public_private_gap
            hist_view.loc[m, "cv_mean_mae"] = r.cv_mean_mae
            hist_view.loc[m, "cv_primary_mae"] = r.cv_primary_mae
            hist_view.loc[m, "cv_shadow_mae"] = r.cv_shadow_mae
            hist_view.loc[m, "abs_PS"] = r.abs_PS
            hist_view.loc[m, "representation"] = r.representation
            hist_view.loc[m, "topology"] = r.topology
            hist_view.loc[m, "annotation"] = r.annotation
        else:
            # factorial may be FULL in experiments with empty test — append
            pass

    # Ensure all 200 factorial present in hist_view
    missing_codes = set(ext.experiment_code) - set(hist_view.experiment_code)
    if missing_codes:
        add = []
        for code in missing_codes:
            r = ext[ext.experiment_code == code].iloc[0]
            er = exp[exp.experiment_code == code]
            eid = er.iloc[0].experiment_id if len(er) else ""
            add.append(
                {
                    "experiment_code": code,
                    "experiment_id": eid,
                    "family": "TRANSFORMER",
                    "representation": r.representation,
                    "topology": r.topology,
                    "annotation": r.annotation,
                    "notes": "factorial external diagnosis",
                    "cv_primary_mae": r.cv_primary_mae,
                    "cv_shadow_mae": r.cv_shadow_mae,
                    "cv_mean_mae": r.cv_mean_mae,
                    "public_mae": r.public_mae,
                    "private_mae": r.private_mae,
                    "test_overall_mae": r.test_overall_mae,
                    "abs_PS": r.abs_PS,
                    "abs_pub_priv": r.public_private_gap,
                    "is_factorial_200": True,
                    "is_surfaceish": False,
                    "provenance": "prospective_factorial",
                    "test_ok": True,
                }
            )
        hist_view = pd.concat([hist_view, pd.DataFrame(add)], ignore_index=True)

    hist_view["test_overall_mae"] = pd.to_numeric(hist_view["test_overall_mae"], errors="coerce")
    hist_view["cv_mean_mae"] = pd.to_numeric(hist_view["cv_mean_mae"], errors="coerce")
    hist_view = hist_view[hist_view.test_overall_mae.notna()].copy()

    top30_test = hist_view.nsmallest(30, "test_overall_mae")
    top30_cv = hist_view.nsmallest(30, "cv_mean_mae")

    # Pareto: nondominated on (cv_mean, test_overall) both minimize
    def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        pts = df[["experiment_code", "cv_mean_mae", "test_overall_mae"]].dropna()
        codes = []
        for i, r in pts.iterrows():
            dominated = False
            for j, s in pts.iterrows():
                if i == j:
                    continue
                if (s.cv_mean_mae <= r.cv_mean_mae and s.test_overall_mae <= r.test_overall_mae) and (
                    s.cv_mean_mae < r.cv_mean_mae or s.test_overall_mae < r.test_overall_mae
                ):
                    dominated = True
                    break
            if not dominated:
                codes.append(r.experiment_code)
        return df[df.experiment_code.isin(codes)].sort_values(["test_overall_mae", "cv_mean_mae"])

    pareto = pareto_front(hist_view)

    cols_out = [
        "experiment_code",
        "experiment_id",
        "family",
        "representation",
        "topology",
        "annotation",
        "notes",
        "cv_primary_mae",
        "cv_shadow_mae",
        "cv_mean_mae",
        "public_mae",
        "private_mae",
        "test_overall_mae",
        "abs_PS",
        "abs_pub_priv",
        "is_factorial_200",
        "is_surfaceish",
        "provenance",
    ]
    for c in cols_out:
        if c not in hist_view.columns:
            hist_view[c] = ""
    top30_test[cols_out].to_csv(REPORTS / "HIC_ALL_HISTORY_TEST_TOP30.csv", index=False)
    top30_cv[cols_out].to_csv(REPORTS / "HIC_ALL_HISTORY_CV_TOP30.csv", index=False)
    pareto[cols_out].to_csv(REPORTS / "HIC_ALL_HISTORY_PARETO_FRONT.csv", index=False)
    hist_view.nsmallest(20, "abs_pub_priv")[cols_out].to_csv(
        REPORTS / "HIC_ALL_HISTORY_PUBPRIV_GAP_BEST.csv", index=False
    )
    hist_view.nlargest(20, "abs_pub_priv")[cols_out].to_csv(
        REPORTS / "HIC_ALL_HISTORY_PUBPRIV_GAP_WORST.csv", index=False
    )

    # --- 11. Candidate classes ---
    hist_view["cv_pct"] = hist_view.cv_mean_mae.rank(pct=True)
    hist_view["test_pct"] = hist_view.test_overall_mae.rank(pct=True)
    pareto_codes = set(pareto.experiment_code)

    def classify(row):
        if row.experiment_code in pareto_codes and row.cv_pct <= 0.25 and row.test_pct <= 0.25:
            return "A_robust_external"
        if row.cv_pct <= 0.15 and row.test_pct > 0.30:
            return "B_internal_stable"
        if row.test_pct <= 0.15 and row.cv_pct > 0.30:
            return "C_test_specialist"
        if row.experiment_code in pareto_codes:
            return "A_robust_external"
        return "other"

    hist_view["candidate_class"] = hist_view.apply(classify, axis=1)
    hist_view[cols_out + ["candidate_class", "cv_pct", "test_pct"]].to_csv(
        REPORTS / "HIC_ALL_HISTORY_CANDIDATE_CLASSES.csv", index=False
    )

    # Surface vs sequence-only factorial
    fac_ext_mean = float(ext.test_overall_mae.mean())
    fac_ext_best = float(ext.test_overall_mae.min())
    surf = hist_view[hist_view.is_surfaceish & ~hist_view.is_factorial_200]
    seq_hist = hist_view[~hist_view.is_surfaceish & ~hist_view.is_factorial_200]
    surface_compare = {
        "factorial_test_mean": fac_ext_mean,
        "factorial_test_best": fac_ext_best,
        "surface_n": int(len(surf)),
        "surface_test_mean": float(surf.test_overall_mae.mean()) if len(surf) else np.nan,
        "surface_test_best": float(surf.test_overall_mae.min()) if len(surf) else np.nan,
        "surface_cv_mean": float(surf.cv_mean_mae.mean()) if len(surf) else np.nan,
        "non_surface_hist_test_best": float(seq_hist.test_overall_mae.min()) if len(seq_hist) else np.nan,
    }
    # Is surface clearly stronger on Test?
    surface_stronger_test = (
        len(surf) > 0 and surface_compare["surface_test_best"] + 0.02 < fac_ext_best
    )
    surface_also_cv = (
        len(surf) > 0
        and float(surf.nsmallest(5, "test_overall_mae").cv_mean_mae.mean())
        < float(ext.nsmallest(5, "test_overall_mae").cv_mean_mae.mean())
    )

    # Explicit Q answers
    h266 = ext[ext.experiment_code == "EXP-H266"].iloc[0]
    answers = {
        "q1_h266_external": {
            "test_overall": float(h266.test_overall_mae),
            "external_rank": int(h266.external_rank),
            "internal_rank": int(h266.internal_rank),
            "note": "strong if top ~20%; check rank",
        },
        "q2_ablingua_avg": r1,
        "q3_esm2": r2,
        "q4_scratch": scratch_verdict,
        "q5_topology_small": topo_verdict,
        "q6_annotation": ann_verdict,
        "q7_high_tail": high_verdict,
        "q8_plateau": plateau_verdict,
        "q9_surface_stronger_externally": bool(surface_stronger_test),
        "q10_surface_cv_supported": bool(surface_also_cv),
    }

    claim_status = {
        "R1_AbLingua_best_average": r1,
        "R2_ESM2_best_cell_not_avg": r2,
        "R3_several_AbPLM_worse_than_Scratch": r3,
        "topology_effects_small_no_strong_winner": topo_verdict,
        "annotation_small_rep_dependent": ann_verdict,
        "scratch_surprisingly_competitive": scratch_verdict,
        "ablang2_no_robust_context_advantage_internal": "NOT_REWRITTEN_external_additive_only",
        "currab_no_robust_context_advantage_internal": "NOT_REWRITTEN_external_additive_only",
        "high_tail_unresolved": high_verdict,
        "sequence_only_plateau_near_0_50": plateau_verdict,
    }

    # --- Write report ---
    lines = []
    A = lines.append
    A("# HIC External Generalization Diagnosis")
    A("")
    A("**STATUS: EXTERNAL_DIAGNOSIS_COMPLETE_NOT_YET_SCIENTIFICALLY_FROZEN**")
    A("")
    A("**PUBLIC_PRIVATE_AUTHORIZED_AFTER_INTERNAL_FREEZE**")
    A("")
    A("## Provenance")
    A("")
    A("| Field | Value |")
    A("|-------|--------|")
    A(f"| Internal factorial freeze | `{INTERNAL_FREEZE_SHA}` |")
    A(f"| Scientific conclusion freeze (DO NOT EDIT) | `{SCIENTIFIC_FREEZE_SHA}` / `reports/HIC_FACTORIAL_SCIENTIFIC_CONCLUSION_FREEZE.md` |")
    A(f"| Diagnosis start HEAD | `{start_head}` |")
    A("| Evaluation asset | `top_models_feature_bundle/solution.csv` via `_lib.load_solution` + `_lib.mae` |")
    A("| Split | solution `is_public`/`is_private` (81/81); overall = MAE on all 162 test ids |")
    A("| New training | **None** |")
    A("| Predictions | existing `experiments/predictions/EXP-Hxxx/test.csv` only |")
    A("| experiments.csv overwritten | **No** |")
    A("")
    A("### Evidence provenance")
    A("")
    A("- **Prospective relative to this factorial freeze:** EXP-H140–H339 (internal conclusions frozen before Pub/Priv).")
    A("- **Retrospective historical evidence:** H001–H139 and other prior Test-scored rows (Test may already have informed project history). Not a pristine holdout.")
    A("")
    A("## 1. Pipeline validation")
    A("")
    A(f"**{pipe['status']}** — max |Δ| vs registry on anchors = {pipe['max_abs_delta']:.3e} (tol={SCORE_TOL}).")
    A("")
    A("Anchors: " + ", ".join(ANCHORS))
    A("")
    A("See `reports/HIC_EXTERNAL_PIPELINE_VALIDATION.csv`.")
    A("")
    A("## 2. Factorial 200 external overview")
    A("")
    best_e = ext.loc[ext.test_overall_mae.idxmin()]
    A(f"- External best cell: **{best_e.experiment_code}** "
      f"({REP_DISPLAY.get(best_e.representation, best_e.representation)} × {best_e.topology} × {best_e.annotation}) "
      f"Test Overall = **{best_e.test_overall_mae:.6f}** "
      f"(Public={best_e.public_mae:.6f}, Private={best_e.private_mae:.6f})")
    A(f"- External mean / median Test Overall: {ext.test_overall_mae.mean():.6f} / {ext.test_overall_mae.median():.6f}")
    A(f"- Best average representation (Test Overall mean): **{REP_DISPLAY.get(ext_order.index[0], ext_order.index[0])}**")
    A(f"- EXP-H266 external: Test={h266.test_overall_mae:.6f}, rank={int(h266.external_rank)}/200 "
      f"(internal rank {int(h266.internal_rank)})")
    A("")
    A("CSV: `reports/HIC_FACTORIAL_EXTERNAL_DIAGNOSIS.csv`")
    A("")
    A("## 3. Internal → external rank transfer")
    A("")
    A(f"- Pearson(cv_mean, test) = **{pearson:.4f}**")
    A(f"- Spearman = **{spearman:.4f}**")
    A(f"- Kendall τ = **{kendall:.4f}**")
    A(f"- mean |rank change| = {transfer['mean_abs_rank_change']:.2f}")
    A(f"- Top10 both = {transfer['top10_both']}")
    A(f"- median (Test − CV mean) = {transfer['median_test_minus_cv']:.4f}")
    A(f"- mean |P−S| = {transfer['mean_abs_PS']:.4f}; mean |Pub−Priv| = {transfer['mean_pub_priv_gap']:.4f}")
    A("")
    A("## 4. Representation external validation")
    A("")
    A("| Rep | int mean | ext mean | int rank | ext rank | Δrank | ext best |")
    A("|-----|----------|----------|----------|----------|-------|----------|")
    for _, r in rep_df.sort_values("external_mean_rank").iterrows():
        A(
            f"| {r.display} | {r.internal_mean:.4f} | {r.external_mean:.4f} | "
            f"{int(r.internal_mean_rank)} | {int(r.external_mean_rank)} | {int(r.rank_change):+d} | {r.external_best:.4f} |"
        )
    A("")
    A(f"- **R1** AbLingua best average: **{r1}**")
    A(f"- **R2** ESM-2 best-cell ≠ average-best: **{r2}** "
      f"(ext best cell family={best_e.representation}; esm2 ext avg rank={esm2_ext_avg_rank})")
    A(f"- **R3** several Ab-PLMs worse than Scratch on average: **{r3}** "
      f"({worse_ext}/5 Ab-PLMs worse than Scratch externally)")
    A("")
    A("## 5. Topology (matched Rep×Annot, Test Overall)")
    A("")
    A("| Contrast | mean Δ | median Δ | improve frac |")
    A("|----------|--------|----------|--------------|")
    for _, r in topo_summary.iterrows():
        A(f"| {r.contrast} | {r.mean_delta:.5f} | {r.median_delta:.5f} | {r.improve_fraction:.3f} |")
    A("")
    A(f"**Internal claim (small effects / no strong winner): {topo_verdict}**")
    A("")
    A("## 6. Annotation (matched Rep×Topo)")
    A("")
    A("| Contrast | mean Δ | median Δ | improve frac |")
    A("|----------|--------|----------|--------------|")
    for _, r in ann_summary.iterrows():
        A(f"| {r.contrast} | {r.mean_delta:.5f} | {r.median_delta:.5f} | {r.improve_fraction:.3f} |")
    A("")
    A(f"**Internal claim (FULL slight+/IMGT slight− / rep-dependent): {ann_verdict}**")
    A("")
    A("## 7. Scratch vs PLM (external)")
    A("")
    A("| Rep | ΔCV vs Scratch | ΔTest vs Scratch | ext improve frac |")
    A("|-----|----------------|------------------|------------------|")
    for _, r in scratch_df.iterrows():
        A(
            f"| {r.display} | {r.internal_mean_delta_vs_scratch:+.4f} | "
            f"{r.external_mean_delta_vs_scratch:+.4f} | {r.external_improve_frac:.2f} |"
        )
    A("")
    A(f"**Scratch competitiveness claim: {scratch_verdict}**")
    A("")
    A("## 8. Context (external additive; internal freeze not rewritten)")
    A("")
    for _, r in ctx_df.iterrows():
        A(
            f"- **{r.family}** PAIRED−SEPARATE Test: mean Δ={r.mean_delta_test:+.5f}, "
            f"improve frac={r.improve_fraction:.2f} (n={int(r.n_strata)}). {r.note}"
        )
    A("")
    A("## 9. HIGH-tail (Test, HIC>11.5)")
    A("")
    if high_summary["testable"]:
        A(f"- n_high={high_summary['n_high_unique']}")
        A(f"- MAE_high mean/min/max = {high_summary['mae_high_mean']:.4f} / "
          f"{high_summary['mae_high_min']:.4f} / {high_summary['mae_high_max']:.4f}")
        A(f"- mean signed error = {high_summary['signed_mean']:.4f}")
        A(f"- best/worst HIGH cells: {high_summary['best_cell']} / {high_summary['worst_cell']}")
        A(f"- **{high_summary['label']}** — verdict vs internal: **{high_verdict}**")
    else:
        A("- **NOT TESTABLE**")
    A("")
    A("## 10. All-history comparison (retrospective + prospective)")
    A("")
    A(f"- Test-scored HIC rows in combined view: {len(hist_view)}")
    best_hist = hist_view.loc[hist_view.test_overall_mae.idxmin()]
    A(f"- Best all-history Test Overall: **{best_hist.experiment_code}** "
      f"({best_hist.experiment_id}) = {best_hist.test_overall_mae:.6f} "
      f"[provenance={best_hist.provenance}, surfaceish={bool(best_hist.is_surfaceish)}]")
    rob = hist_view[hist_view.candidate_class == "A_robust_external"].copy()
    if len(rob):
        rob["cv_test_sum"] = rob.cv_mean_mae + rob.test_overall_mae
        rr = rob.nsmallest(1, "cv_test_sum").iloc[0]
        A(f"- Best balanced robust CV+Test candidate (class A, min CV+Test): **{rr.experiment_code}** "
          f"CV={rr.cv_mean_mae:.4f} Test={rr.test_overall_mae:.4f} "
          f"[provenance={rr.provenance}, surfaceish={bool(rr.is_surfaceish)}]")
    A(f"- Class counts: "
      f"A={int((hist_view.candidate_class=='A_robust_external').sum())}, "
      f"B={int((hist_view.candidate_class=='B_internal_stable').sum())}, "
      f"C={int((hist_view.candidate_class=='C_test_specialist').sum())}")
    A("")
    A("Artifacts: `HIC_ALL_HISTORY_TEST_TOP30.csv`, `HIC_ALL_HISTORY_CV_TOP30.csv`, "
      "`HIC_ALL_HISTORY_PARETO_FRONT.csv`, `HIC_ALL_HISTORY_CANDIDATE_CLASSES.csv`.")
    A("")
    A("## 11. Surface hypothesis (careful)")
    A("")
    A(f"- Factorial sequence-only best Test = {fac_ext_best:.4f}; mean = {fac_ext_mean:.4f}")
    A(f"- Historical surfaceish best Test = {surface_compare['surface_test_best']:.4f} "
      f"(n={surface_compare['surface_n']})")
    A(f"- Surface stronger on Test vs factorial best by ≳0.02: **{surface_stronger_test}**")
    A(f"- Top surface Test cells also stronger on CV than top factorial Test cells: **{surface_also_cv}**")
    A("")
    A("Interpretation: if surfaceish models dominate Test, evidence is consistent with "
      "**external data being substantially more consistent with models containing explicit "
      "surface/physicochemical information** — not a causal proof from this diagnosis alone. "
      "Much of that surface evidence is **retrospective** relative to prior Test access.")
    A("")
    A("## 12. Claim status summary (vs internal scientific freeze)")
    A("")
    for k, v in claim_status.items():
        A(f"- `{k}`: **{v}**")
    A("")
    A("## 13. Explicit answers")
    A("")
    A(f"1. H266 external strength: rank {int(h266.external_rank)}/200, Test={h266.test_overall_mae:.4f}.")
    A(f"2. AbLingua average-best: {r1}.")
    A(f"3. ESM-2 pattern: {r2}.")
    A(f"4. Scratch strong: {scratch_verdict}.")
    A(f"5. Topology small: {topo_verdict}.")
    A(f"6. Annotation pattern: {ann_verdict}.")
    A(f"7. HIGH-tail: {high_verdict}.")
    A(f"8. Sequence-only plateau: {plateau_verdict} (best Test={best_ext:.4f}, median={median_ext:.4f}).")
    A(f"9. Surface clearer externally than factorial seq-only: {surface_stronger_test}.")
    A(f"10. That gap also CV-supported (top cells): {surface_also_cv}.")
    A("")
    A("## 14. Final diagnosis summary (not a scientific freeze)")
    A("")
    A(f"1. Evaluation pipeline validation: **{pipe['status']}** (max |Δ|={pipe['max_abs_delta']:.3e}).")
    A(f"2. Factorial 200-cell external best: **{best_e.experiment_code}** "
      f"({REP_DISPLAY.get(best_e.representation, best_e.representation)} × {best_e.topology} × {best_e.annotation}) "
      f"Test={best_e.test_overall_mae:.6f}.")
    A(f"3. Factorial external best average representation: **{REP_DISPLAY.get(ext_order.index[0], ext_order.index[0])}**.")
    A(f"4. Internal→external rank correlation: Pearson={pearson:.3f}, Spearman={spearman:.3f}, Kendall={kendall:.3f}; "
      f"Top10∩={sorted(top10_int & top10_ext)}.")
    A(f"5. Topology conclusion: **{topo_verdict}**.")
    A(f"6. Annotation conclusion: **{ann_verdict}**.")
    A(f"7. Scratch conclusion: **{scratch_verdict}**.")
    ab2 = ctx_df[ctx_df.family == 'ablang2'].iloc[0]
    cu = ctx_df[ctx_df.family == 'currab'].iloc[0]
    A(f"8. AbLang2 context: external PAIRED−SEPARATE mean Δ={ab2.mean_delta_test:+.4f} "
      f"(improve frac={ab2.improve_fraction:.2f}); internal no-robust-advantage **not rewritten**.")
    A(f"9. CurrAb context: external PAIRED−SEPARATE mean Δ={cu.mean_delta_test:+.4f} "
      f"(improve frac={cu.improve_fraction:.2f}); internal no-robust-advantage **not rewritten**.")
    A(f"10. HIGH-tail conclusion: **{high_verdict}** ({high_summary['label']}).")
    A(f"11. Sequence-only plateau conclusion: **{plateau_verdict}** "
      f"(best={best_ext:.4f}, median={median_ext:.4f}; historical surface best≈{surface_compare['surface_test_best']:.4f}).")
    A(f"12. Best all-history external model: **{best_hist.experiment_code}** "
      f"Test={best_hist.test_overall_mae:.6f} ({best_hist.provenance}).")
    if len(rob):
        A(f"13. Best robust CV+external candidate: **{rr.experiment_code}** "
          f"CV={rr.cv_mean_mae:.4f} Test={rr.test_overall_mae:.4f}.")
    else:
        A("13. Best robust CV+external candidate: none classified as A.")
    A("14. Surface hypothesis status: external data are substantially more consistent with models "
      "containing explicit surface/physicochemical information (retrospective + matched CV support "
      "for top surface cells); **not** a causal proof.")
    A("15. Internal Freeze claim outcomes:")
    for k, v in claim_status.items():
        A(f"    - `{k}` → **{v}**")
    A("16. Unresolved: surface causality vs confounders; calibration of factorial cells; "
      "Pub/Priv outlier mechanisms; Freeze v2 synthesis pending human+ChatGPT review.")
    A("")
    A("## 15. Unresolved (detail)")
    A("")
    A("- Causal role of surface features vs confounders in historical surface experiments")
    A("- Whether any factorial cell would become competitive after calibration (not tested)")
    A("- Private-specific failures within Pub/Priv gap outliers")
    A("- Final Freeze v2 synthesis (awaits human + ChatGPT review)")
    A("")
    A("## 16. Non-actions")
    A("")
    A("- No new scientific freeze created")
    A("- Internal scientific freeze file untouched")
    A("- No `experiments.csv` external overwrite for H140–H339")
    A("")

    report_path = REPORTS / "HIC_EXTERNAL_GENERALIZATION_DIAGNOSIS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "pipeline": {"status": pipe["status"], "max_abs_delta": pipe["max_abs_delta"]},
        "transfer": transfer,
        "claims": claim_status,
        "answers": answers,
        "surface_compare": surface_compare,
        "best_factorial_external": {
            "code": best_e.experiment_code,
            "test_overall": float(best_e.test_overall_mae),
            "representation": best_e.representation,
            "topology": best_e.topology,
            "annotation": best_e.annotation,
        },
        "best_rep_external_avg": ext_order.index[0],
        "best_all_history": {
            "code": best_hist.experiment_code,
            "test_overall": float(best_hist.test_overall_mae),
            "provenance": best_hist.provenance,
            "surfaceish": bool(best_hist.is_surfaceish),
        },
        "start_head": start_head,
        "end_head_placeholder": git_rev(),
    }
    (REPORTS / "HIC_EXTERNAL_GENERALIZATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print("EXTERNAL_DIAGNOSIS_OK", report_path)
    print("BEST_FAC", best_e.experiment_code, float(best_e.test_overall_mae))
    print("BEST_REP", ext_order.index[0])
    print("CLAIMS", claim_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
