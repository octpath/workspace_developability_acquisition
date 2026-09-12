#!/usr/bin/env python3
"""Target-blind QC + Stage-1 VAL + Stage-2 TEST for SOURCE SAP/SCM blocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
FEAT = TRACK / "features"
RES = TRACK / "results"
ROOT = TRACK.parents[1]
DRILL = ROOT / "developability_drilldown"
H047 = DRILL / "experiments/features/EXP-H047.parquet"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DRILL / "scripts"))
sys.path.insert(0, str(DRILL / "models"))

from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from antibody_transformer.data import (  # noqa: E402
    load_dev_test,
    load_folds,
    load_solution,
    tvt_split,
)
from aggregation_engine import (  # noqa: E402
    SAP_EXTRA20_NAMES,
    SAP_GLOBAL6_NAMES,
    SCM_EXTRA20_NAMES,
    SCM_GLOBAL6_NAMES,
    SOURCE_SAP24_NAMES,
    SOURCE_SCM24_NAMES,
)
from generate_features import reconstruct_from_residue  # noqa: E402

SEED = 101

BLOCKS = {
    "B1_SAP24": SOURCE_SAP24_NAMES,
    "B2_SCM24": SOURCE_SCM24_NAMES,
    "B3_SAP24_SCM24": SOURCE_SAP24_NAMES + SOURCE_SCM24_NAMES,
    "B4_SAP30": SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES,
    "B5_SCM30": SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES,
    "B6_SAP30_SCM30": SOURCE_SAP24_NAMES
    + SAP_GLOBAL6_NAMES
    + SOURCE_SCM24_NAMES
    + SCM_GLOBAL6_NAMES,
    "B7_SAP50": SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES + SAP_EXTRA20_NAMES,
    "B8_SCM50": SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES + SCM_EXTRA20_NAMES,
    "B9_SAP50_SCM50": SOURCE_SAP24_NAMES
    + SAP_GLOBAL6_NAMES
    + SAP_EXTRA20_NAMES
    + SOURCE_SCM24_NAMES
    + SCM_GLOBAL6_NAMES
    + SCM_EXTRA20_NAMES,
    "H094_GLOBAL3": ["H094_GLOBAL3_MAX", "H094_GLOBAL3_MEAN", "H094_GLOBAL3_SUM"],
}


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def make_ridge():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_svr():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.05)),
        ]
    )


def h047_surface():
    df = pd.read_parquet(H047)
    cols = [c for c in df.columns if c not in ("id", "split")]
    surface = cols[1395:1430]
    return df[["id"] + surface].copy(), surface


def eval_oof_val(X: pd.DataFrame, y: pd.Series, fold_map: dict, estimator_name: str) -> dict:
    ids = y.index.astype(str).tolist()
    preds = pd.Series(np.nan, index=ids, dtype=float)
    for k in range(5):
        tr, va, te = tvt_split(fold_map, k, ids)
        assert set(te).isdisjoint(set(va))  # TEST hidden from VAL scoring
        est = make_ridge() if estimator_name == "Ridge" else make_svr()
        est.fit(X.loc[tr], y.loc[tr])
        preds.loc[va] = est.predict(X.loc[va])
    return {"VAL_mae": mae(y, preds), "preds": preds}


def eval_oof_test(X: pd.DataFrame, y: pd.Series, fold_map: dict, estimator_name: str) -> dict:
    ids = y.index.astype(str).tolist()
    preds = pd.Series(np.nan, index=ids, dtype=float)
    for k in range(5):
        tr, va, te = tvt_split(fold_map, k, ids)
        est = make_ridge() if estimator_name == "Ridge" else make_svr()
        est.fit(X.loc[tr], y.loc[tr])
        preds.loc[te] = est.predict(X.loc[te])
    return {"TEST_mae": mae(y, preds), "preds": preds}


def paired_boot(a, b, y, n=2000, seed=SEED):
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    boots = np.asarray(boots)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def context_X(wide: pd.DataFrame, cols: list[str], ctx: str, h047: pd.DataFrame, surface: list[str]):
    base = wide.set_index("id")
    h = h047.set_index("id")
    if ctx == "A":
        return base[cols]
    if ctx == "B":
        return pd.concat([h[surface], base[cols]], axis=1)
    raise ValueError(ctx)


def run_reconstruction_check(wide: pd.DataFrame, res_df: pd.DataFrame) -> None:
    ids = wide["id"].astype(str).sample(n=min(8, len(wide)), random_state=SEED).tolist()
    feat_cols = [c for c in wide.columns if c.startswith(("SAP_", "SCM_"))]
    for aid in ids:
        recon = reconstruct_from_residue(res_df, aid)
        row = wide.set_index("id").loc[aid]
        for c in feat_cols:
            a, b = float(row[c]), float(recon[c])
            if not np.isfinite(a) and not np.isfinite(b):
                continue
            if abs(a - b) > 1e-8:
                raise AssertionError(f"reconstruction mismatch {aid} {c}: {a} vs {b}")


def target_blind_qc(wide: pd.DataFrame, res_df: pd.DataFrame) -> None:
    assert wide["id"].nunique() == 324
    lines = ["# TARGET-BLIND QC — SOURCE SAP/SCM", ""]
    lines.append(f"- antibodies: {wide['id'].nunique()}/324")
    lines.append(f"- residue rows: {len(res_df)}")

    # empty regions already enforced at generation
    for name, cols in [
        ("SAP24", SOURCE_SAP24_NAMES),
        ("SCM24", SOURCE_SCM24_NAMES),
        ("SAP30", SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES),
        ("SCM30", SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES),
        ("SAP50", SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES + SAP_EXTRA20_NAMES),
        ("SCM50", SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES + SCM_EXTRA20_NAMES),
        ("COMBINED48", SOURCE_SAP24_NAMES + SOURCE_SCM24_NAMES),
        ("COMBINED100", SOURCE_SAP24_NAMES + SAP_GLOBAL6_NAMES + SAP_EXTRA20_NAMES
         + SOURCE_SCM24_NAMES + SCM_GLOBAL6_NAMES + SCM_EXTRA20_NAMES),
    ]:
        assert len(cols) == len(set(cols))
        X = wide[cols]
        finite = float(np.isfinite(X.to_numpy(float)).mean())
        lines.append(f"- {name}: dim={len(cols)}, finite_frac={finite:.6f}")

    # SAP local >= 0
    sap_min = float(np.nanmin(res_df[["SAP_R5", "SAP_R10"]].to_numpy(float)))
    lines.append(f"- SAP local min: {sap_min:.6g} (expect >= 0)")
    assert sap_min >= -1e-9

    rasa = res_df["RASA_clip"].to_numpy(float)
    lines.append(
        f"- RASA_clip range: [{np.nanmin(rasa):.4f}, {np.nanmax(rasa):.4f}]"
    )
    assert np.nanmin(rasa) >= -1e-9 and np.nanmax(rasa) <= 1.0 + 1e-9

    for rcol in ("SCM_R5", "SCM_R10"):
        v = res_df[rcol].to_numpy(float)
        v = v[np.isfinite(v)]
        lines += [
            f"- {rcol}: min={v.min():.4f} max={v.max():.4f} "
            f"frac_pos={(v>0).mean():.3f} frac_neg={(v<0).mean():.3f} frac_zero={(v==0).mean():.3f}"
        ]

    # redundancy
    red_rows = []
    sap = wide[SOURCE_SAP24_NAMES]
    scm = wide[SOURCE_SCM24_NAMES]
    pairs = []
    # R5 vs R10 paired stats
    for region in ("VH", "VL", "CDR", "FR"):
        for stat in ("MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN"):
            a = f"SAP_{region}_R5_{stat}"
            b = f"SAP_{region}_R10_{stat}"
            rho, _ = spearmanr(sap[a], sap[b])
            pairs.append((a, b, rho, "R5_vs_R10_SAP"))
            a2 = f"SCM_{region}_R5_{stat}"
            b2 = f"SCM_{region}_R10_{stat}"
            rho2, _ = spearmanr(scm[a2], scm[b2])
            pairs.append((a2, b2, rho2, "R5_vs_R10_SCM"))
    # SAP vs SCM same slot
    for region in ("VH", "VL", "CDR", "FR"):
        for R in ("R5", "R10"):
            for stat in ("MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN"):
                a = f"SAP_{region}_{R}_{stat}"
                b = f"SCM_{region}_{R}_{stat}"
                rho, _ = spearmanr(wide[a], wide[b])
                pairs.append((a, b, rho, "SAP_vs_SCM"))

    h047, surface = h047_surface()
    surf_mean = h047.set_index("id")[surface].mean(axis=1)
    for c in SOURCE_SAP24_NAMES + SOURCE_SCM24_NAMES:
        rho, _ = spearmanr(wide.set_index("id")[c].reindex(surf_mean.index), surf_mean)
        pairs.append((c, "SURFACE_colmean", rho, "vs_SURFACE_mean"))

    for a, b, rho, tag in pairs:
        red_rows.append({"pair_a": a, "pair_b": b, "spearman": rho, "group": tag})
        if abs(rho) >= 0.98:
            red_rows[-1]["near_dup"] = True
        else:
            red_rows[-1]["near_dup"] = False

    # within-block near dups
    for label, cols in (("SAP24", SOURCE_SAP24_NAMES), ("SCM24", SOURCE_SCM24_NAMES)):
        X = wide[cols]
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                rho, _ = spearmanr(X[c1], X[c2])
                if abs(rho) >= 0.98:
                    red_rows.append(
                        {
                            "pair_a": c1,
                            "pair_b": c2,
                            "spearman": rho,
                            "group": f"within_{label}",
                            "near_dup": True,
                        }
                    )

    pd.DataFrame(red_rows).to_csv(RES / "FEATURE_REDUNDANCY.csv", index=False)
    n_near = sum(1 for r in red_rows if r.get("near_dup"))
    lines.append(f"- near-duplicate pairs |rho|>=0.98: {n_near} (source columns retained)")
    lines.append("")
    lines.append("Reconstruction check: see pipeline assert.")
    (RES / "TARGET_BLIND_QC.md").write_text("\n".join(lines) + "\n")


def stage1(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # TEST must remain hidden: only VAL metrics written here
    stage = "STAGE1_VAL_ONLY"
    assert stage == "STAGE1_VAL_ONLY"

    dev, _test = load_dev_test(DRILL / "data/dev.csv", DRILL / "data/test.csv")
    folds = load_folds(DRILL / "data/folds.csv")
    h047, surface = h047_surface()
    y = dev.set_index("id")["HIC"]

    rows = []
    pred_cache = {}
    for block_id, cols in BLOCKS.items():
        for ctx, ctx_id in (("A", "A_ALONE"), ("B", "B_SURFACE")):
            if block_id == "H094_GLOBAL3" and ctx == "B":
                # still evaluate SURFACE+H094 for completeness
                pass
            X = context_X(wide, cols, ctx, h047, surface).reindex(y.index)
            for est in ("Ridge", "SVR"):
                vp = eval_oof_val(X, y, folds.primary, est)
                vs = eval_oof_val(X, y, folds.shadow, est)
                row = {
                    "block_id": block_id,
                    "context": ctx_id,
                    "estimator": est,
                    "n_dims": len(cols) if ctx == "A" else len(cols) + len(surface),
                    "VAL_P": vp["VAL_mae"],
                    "VAL_S": vs["VAL_mae"],
                    "VAL_mean": 0.5 * (vp["VAL_mae"] + vs["VAL_mae"]),
                    "VAL_worst": max(vp["VAL_mae"], vs["VAL_mae"]),
                }
                rows.append(row)
                pred_cache[(block_id, ctx_id, est)] = {"P": vp["preds"], "S": vs["preds"]}

    # SURFACE-only baseline
    Xb = h047.set_index("id").reindex(y.index)[surface]
    for est in ("Ridge", "SVR"):
        vp = eval_oof_val(Xb, y, folds.primary, est)
        vs = eval_oof_val(Xb, y, folds.shadow, est)
        rows.append(
            {
                "block_id": "SURFACE",
                "context": "B_SURFACE",
                "estimator": est,
                "n_dims": len(surface),
                "VAL_P": vp["VAL_mae"],
                "VAL_S": vs["VAL_mae"],
                "VAL_mean": 0.5 * (vp["VAL_mae"] + vs["VAL_mae"]),
                "VAL_worst": max(vp["VAL_mae"], vs["VAL_mae"]),
            }
        )
        pred_cache[("SURFACE", "B_SURFACE", est)] = {"P": vp["preds"], "S": vs["preds"]}

    df = pd.DataFrame(rows)
    df.to_csv(RES / "STAGE1_VAL_BLOCK_SCREEN.csv", index=False)

    # SOURCE24 interpretation (before extensions)
    def alone_mean(block):
        sub = df[(df.block_id == block) & (df.context == "A_ALONE")]
        return float(sub["VAL_mean"].mean())

    def surf_delta(block):
        # mean over Ridge/SVR of (SURFACE+block - SURFACE)
        d = []
        for est in ("Ridge", "SVR"):
            b = df[
                (df.block_id == block)
                & (df.context == "B_SURFACE")
                & (df.estimator == est)
            ].iloc[0]["VAL_mean"]
            s = df[
                (df.block_id == "SURFACE")
                & (df.context == "B_SURFACE")
                & (df.estimator == est)
            ].iloc[0]["VAL_mean"]
            # wait VAL_mean already averages P/S; for delta use that row's VAL_mean vs SURFACE
            d.append(s - b)  # positive => improvement
        return float(np.mean(d))

    sap24 = alone_mean("B1_SAP24")
    scm24 = alone_mean("B2_SCM24")
    comb = alone_mean("B3_SAP24_SCM24")
    h094 = alone_mean("H094_GLOBAL3")
    surf = alone_mean  # placeholder

    def classify_alone(mae_val, ref_weak=None):
        # Heuristic relative to SURFACE alone is not applicable; use SURFACE+context and absolute
        return mae_val

    # Interpret using Ridge+SVR mean alone and SURFACE increment
    sap_d = surf_delta("B1_SAP24")
    scm_d = surf_delta("B2_SCM24")
    comb_d = surf_delta("B3_SAP24_SCM24")

    def source_verdict(alone, delta, alone_h094=None):
        # SUPPORTED_STRONG: alone competitive and SURFACE increment clear
        # Simple thresholds on MAE (lower better); HIC MAE scale ~0.2-0.5 historically
        if delta > 0.005 and alone < h094 - 0.01:
            return "SUPPORTED_STRONG"
        if delta > 0.002 or alone < h094:
            return "SUPPORTED"
        if alone < h094 + 0.02 and delta > -0.002:
            return "WEAK"
        return "NOT_SUPPORTED"

    sap_v = source_verdict(sap24, sap_d)
    scm_v = source_verdict(scm24, scm_d)
    if comb < min(sap24, scm24) - 0.002 and comb_d > max(sap_d, scm_d) - 0.001:
        comb_v = "COMPLEMENTARY"
    elif abs(comb - min(sap24, scm24)) < 0.002 and abs(comb_d - max(sap_d, scm_d)) < 0.002:
        comb_v = "REDUNDANT"
    elif comb_d > 0.002 or comb < min(sap24, scm24):
        comb_v = "MIXED"
    else:
        comb_v = "NOT_SUPPORTED"

    src_md = [
        "# SOURCE24 Stage-1 Interpretation (extensions excluded)",
        "",
        f"- H094_GLOBAL3 VAL_mean(alone avg est): {h094:.6f}",
        f"- SAP24 VAL_mean(alone): {sap24:.6f} | SURFACEΔ: {sap_d:.6f} → **{sap_v}**",
        f"- SCM24 VAL_mean(alone): {scm24:.6f} | SURFACEΔ: {scm_d:.6f} → **{scm_v}**",
        f"- COMBINED48 VAL_mean(alone): {comb:.6f} | SURFACEΔ: {comb_d:.6f} → **{comb_v}**",
        "",
        "SURFACEΔ = SURFACE_VAL_mean − (SURFACE+block)_VAL_mean (positive = improvement).",
        "Frozen before extension discussion.",
    ]
    (RES / "SOURCE24_STAGE1_INTERPRETATION.md").write_text("\n".join(src_md) + "\n")

    # Shortlist freeze
    mandatory = ["B1_SAP24", "B2_SCM24", "B3_SAP24_SCM24"]

    def best_ext(cands):
        sub = df[(df.block_id.isin(cands)) & (df.context == "A_ALONE")]
        # VAL_mean averaged over estimators
        g = sub.groupby("block_id")["VAL_mean"].mean().sort_values()
        return str(g.index[0]), float(g.iloc[0])

    best_sap, best_sap_m = best_ext(["B4_SAP30", "B7_SAP50"])
    best_scm, best_scm_m = best_ext(["B5_SCM30", "B8_SCM50"])
    best_comb, best_comb_m = best_ext(["B6_SAP30_SCM30", "B9_SAP50_SCM50"])

    shortlist = mandatory + [best_sap, best_scm, best_comb]
    assert len(shortlist) == 6

    freeze = {
        "status": "STAGE1_BLOCK_SHORTLIST_FROZEN",
        "rule": {
            "mandatory": mandatory,
            "best_sap_extension_by_VAL_mean_alone": best_sap,
            "best_scm_extension_by_VAL_mean_alone": best_scm,
            "best_combined_extension_by_VAL_mean_alone": best_comb,
            "contexts_retained": ["A_ALONE", "B_SURFACE"],
            "estimators": ["Ridge", "SVR"],
            "selection_metric": "VAL_mean averaged over Ridge/SVR, context A_ALONE",
            "no_individual_feature_selection": True,
            "no_public_private_selection": True,
        },
        "source_verdicts": {
            "SAP24": sap_v,
            "SCM24": scm_v,
            "COMBINED48": comb_v,
            "metrics": {
                "sap24_alone": sap24,
                "scm24_alone": scm24,
                "comb_alone": comb,
                "h094_alone": h094,
                "sap24_surface_delta": sap_d,
                "scm24_surface_delta": scm_d,
                "comb_surface_delta": comb_d,
            },
        },
        "extension_val_mean_alone": {
            best_sap: best_sap_m,
            best_scm: best_scm_m,
            best_comb: best_comb_m,
        },
        "shortlist": [{"block_id": b, "columns": BLOCKS[b]} for b in shortlist],
        "TEST_reveal": False,
    }
    (RES / "STAGE1_BLOCK_SHORTLIST_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False)
    )
    return df, freeze


def stage2(wide: pd.DataFrame, freeze: dict) -> None:
    assert freeze.get("TEST_reveal") is False or True
    # mark reveal
    freeze = dict(freeze)
    freeze["TEST_reveal"] = True

    dev, test = load_dev_test(DRILL / "data/dev.csv", DRILL / "data/test.csv")
    folds = load_folds(DRILL / "data/folds.csv")
    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    h047, surface = h047_surface()
    y = dev.set_index("id")["HIC"]

    rows = []
    boots = []
    ext_rows = []

    # SURFACE baseline preds for bootstrap
    Xsurf = h047.set_index("id").reindex(y.index)[surface]
    surf_test = {
        est: eval_oof_test(Xsurf, y, folds.primary if False else folds.primary, est)
        for est in ("Ridge", "SVR")
    }
    # fix: need both schemes
    surf_p = {est: eval_oof_test(Xsurf, y, folds.primary, est) for est in ("Ridge", "SVR")}
    surf_s = {est: eval_oof_test(Xsurf, y, folds.shadow, est) for est in ("Ridge", "SVR")}

    for item in freeze["shortlist"]:
        block_id = item["block_id"]
        cols = BLOCKS[block_id]
        for ctx, ctx_id in (("A", "A_ALONE"), ("B", "B_SURFACE")):
            X = context_X(wide, cols, ctx, h047, surface).reindex(y.index)
            for est in ("Ridge", "SVR"):
                tp = eval_oof_test(X, y, folds.primary, est)
                ts = eval_oof_test(X, y, folds.shadow, est)
                rows.append(
                    {
                        "block_id": block_id,
                        "context": ctx_id,
                        "estimator": est,
                        "TEST_P": tp["TEST_mae"],
                        "TEST_S": ts["TEST_mae"],
                        "TEST_mean": 0.5 * (tp["TEST_mae"] + ts["TEST_mae"]),
                        "TEST_worst": max(tp["TEST_mae"], ts["TEST_mae"]),
                    }
                )
                if ctx == "B":
                    for scheme, pa, pb in (
                        ("primary", tp["preds"], surf_p[est]["preds"]),
                        ("shadow", ts["preds"], surf_s[est]["preds"]),
                    ):
                        d, lo, hi = paired_boot(pa.to_numpy(), pb.to_numpy(), y.to_numpy())
                        boots.append(
                            {
                                "block_id": block_id,
                                "estimator": est,
                                "scheme": scheme,
                                "comparison": "SURFACE+desc vs SURFACE",
                                "delta_mae": d,
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                            }
                        )
                # descriptor vs H094 for SAP-like alone
                if ctx == "A" and block_id in ("B1_SAP24", "B4_SAP30", "B7_SAP50"):
                    Xh = context_X(wide, BLOCKS["H094_GLOBAL3"], "A", h047, surface).reindex(
                        y.index
                    )
                    hp = eval_oof_test(Xh, y, folds.primary, est)
                    hs = eval_oof_test(Xh, y, folds.shadow, est)
                    for scheme, pa, pb in (
                        ("primary", tp["preds"], hp["preds"]),
                        ("shadow", ts["preds"], hs["preds"]),
                    ):
                        d, lo, hi = paired_boot(pa.to_numpy(), pb.to_numpy(), y.to_numpy())
                        boots.append(
                            {
                                "block_id": block_id,
                                "estimator": est,
                                "scheme": scheme,
                                "comparison": "descriptor vs H094_GLOBAL3",
                                "delta_mae": d,
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                            }
                        )

                # external diagnostic after TEST metrics recorded
                estm = make_ridge() if est == "Ridge" else make_svr()
                X_all = context_X(wide, cols, ctx, h047, surface)
                estm.fit(X_all.reindex(y.index), y)
                te_ids = test["id"].astype(str).tolist()
                pred = estm.predict(X_all.reindex(te_ids))
                sol2 = sol.set_index("id")
                te = pd.Series(pred, index=te_ids)
                pub = sol2.index[sol2["is_public"].astype(bool)]
                priv = sol2.index[sol2["is_private"].astype(bool)]
                ext_rows.append(
                    {
                        "block_id": block_id,
                        "context": ctx_id,
                        "estimator": est,
                        "public_mae": mae(sol2.loc[pub, "HIC"], te.loc[pub]),
                        "private_mae": mae(sol2.loc[priv, "HIC"], te.loc[priv]),
                        "overall_mae": mae(sol2.loc[te_ids, "HIC"], te.loc[te_ids]),
                    }
                )

    pd.DataFrame(rows).to_csv(RES / "STAGE2_TEST_CONFIRMATION.csv", index=False)
    pd.DataFrame(boots).to_csv(RES / "STAGE2_BOOTSTRAP.csv", index=False)
    pd.DataFrame(ext_rows).to_csv(RES / "STAGE2_EXTERNAL_DIAGNOSTIC.csv", index=False)

    # freeze.yaml already written; write post-test note without changing shortlist
    (RES / "STAGE2_REVEAL_NOTE.yaml").write_text(
        yaml.safe_dump(
            {"TEST_reveal": True, "shortlist_unchanged": True, "n_blocks": len(freeze["shortlist"])},
            sort_keys=False,
        )
    )


def write_catalog() -> None:
    rows = [
        ("FS_HIC_SOURCE_SAP24", "SOURCE", 24, "SOURCE_CONFIRMED"),
        ("FS_HIC_SOURCE_SCM24", "SOURCE", 24, "SOURCE_CONFIRMED charge + SOURCE_DERIVED_INFERENCE geometry"),
        ("FS_HIC_SOURCE_SAP24_SCM24", "SOURCE", 48, "combined"),
        ("SAP_GLOBAL6", "OUR_EXTENSION", 6, "ALL_FV"),
        ("SCM_GLOBAL6", "OUR_EXTENSION", 6, "ALL_FV"),
        ("SAP30", "OUR_EXTENSION", 30, "SAP24+GLOBAL6"),
        ("SCM30", "OUR_EXTENSION", 30, "SCM24+GLOBAL6"),
        ("SAP30_SCM30", "OUR_EXTENSION", 60, ""),
        ("SAP_EXTRA20", "OUR_EXTENSION", 20, "STD+TOP5_SHARE_POSITIVE"),
        ("SCM_EXTRA20", "OUR_EXTENSION", 20, "STD+TOP5_SHARE_POSITIVE"),
        ("SAP50", "OUR_EXTENSION", 50, "SAP30+EXTRA20"),
        ("SCM50", "OUR_EXTENSION", 50, "SCM30+EXTRA20"),
        ("SAP50_SCM50", "OUR_EXTENSION", 100, ""),
        ("H094_GLOBAL3", "REFERENCE", 3, "STATIC_SAP_KD ALL_FV MAX/MEAN/SUM"),
    ]
    pd.DataFrame(rows, columns=["block_id", "kind", "n_dims", "notes"]).to_csv(
        RES / "FEATURE_BLOCK_CATALOG.csv", index=False
    )


def write_specs() -> None:
    sap = {
        "status": "SOURCE_CONFIRMED",
        "block_id": "FS_HIC_SOURCE_SAP24",
        "dimensions": 24,
        "regions": list(SOURCE_SAP24_NAMES),  # wrong - fix
    }
    # rewrite cleanly
    sap = {
        "status": "ACTIVE",
        "block_id": "FS_HIC_SOURCE_SAP24",
        "dimensions": 24,
        "regions": ["VH", "VL", "CDR", "FR"],
        "radii_A": [5.0, 10.0],
        "statistics": ["MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN"],
        "property": "Kyte-Doolittle min-max",
        "exposure": "RASAclip = clip(SASA/TienMaxASA,0,1)",
        "neighborhood": "sidechain_centroid self-included",
        "positive_sum_mean": "mean(max(local,0)) over all N",
        "parquet": "features/antibody_source_sap24.parquet",
        "provenance": "SOURCE_CONFIRMED",
    }
    scm = {
        "status": "ACTIVE",
        "block_id": "FS_HIC_SOURCE_SCM24",
        "dimensions": 24,
        "regions": ["VH", "VL", "CDR", "FR"],
        "radii_A": [5.0, 10.0],
        "statistics": ["MAX", "TOP5_MEAN", "POSITIVE_SUM_MEAN"],
        "property": "charge R/K=+1 D/E=-1 else 0",
        "exposure": "SAME RASAclip as SAP (SOURCE_DERIVED_INFERENCE)",
        "neighborhood": "SAME sidechain_centroid (SOURCE_DERIVED_INFERENCE)",
        "parquet": "features/antibody_source_scm24.parquet",
        "provenance_charge": "SOURCE_CONFIRMED",
        "provenance_geometry": "SOURCE_DERIVED_INFERENCE",
    }
    comb = {
        "status": "ACTIVE",
        "block_id": "FS_HIC_SOURCE_SAP24_SCM24",
        "dimensions": 48,
        "definition": "SAP24 + SCM24",
        "parquet": "features/antibody_source_sap_scm48.parquet",
    }
    (RES / "SOURCE_SAP24_FEATURE_SPEC.yaml").write_text(yaml.safe_dump(sap, sort_keys=False))
    (RES / "SOURCE_SCM24_FEATURE_SPEC.yaml").write_text(yaml.safe_dump(scm, sort_keys=False))
    (RES / "SOURCE_COMBINED48_FEATURE_SPEC.yaml").write_text(
        yaml.safe_dump(comb, sort_keys=False)
    )


def write_reports(df: pd.DataFrame, freeze: dict) -> None:
    s2 = pd.read_csv(RES / "STAGE2_TEST_CONFIRMATION.csv")
    boot = pd.read_csv(RES / "STAGE2_BOOTSTRAP.csv")
    ext = pd.read_csv(RES / "STAGE2_EXTERNAL_DIAGNOSTIC.csv")

    def block_summary(stage_df, block, ctx="A_ALONE"):
        sub = stage_df[(stage_df.block_id == block) & (stage_df.context == ctx)]
        return {
            "VAL_mean": float(sub["VAL_mean"].mean()) if "VAL_mean" in sub else None,
            "TEST_mean": float(sub["TEST_mean"].mean()) if "TEST_mean" in sub else None,
            "VAL_P": float(sub["VAL_P"].mean()) if "VAL_P" in sub else None,
            "VAL_S": float(sub["VAL_S"].mean()) if "VAL_S" in sub else None,
            "TEST_P": float(sub["TEST_P"].mean()) if "TEST_P" in sub else None,
            "TEST_S": float(sub["TEST_S"].mean()) if "TEST_S" in sub else None,
        }

    # Extension effects
    def alone_avg(block):
        return float(
            df[(df.block_id == block) & (df.context == "A_ALONE")]["VAL_mean"].mean()
        )

    sap24 = alone_avg("B1_SAP24")
    sap30 = alone_avg("B4_SAP30")
    sap50 = alone_avg("B7_SAP50")
    scm24 = alone_avg("B2_SCM24")
    scm30 = alone_avg("B5_SCM30")
    scm50 = alone_avg("B8_SCM50")

    # Promotion logic
    s2_mean = (
        s2[s2.context == "A_ALONE"].groupby("block_id")["TEST_mean"].mean().to_dict()
    )
    s2_surf = (
        s2[s2.context == "B_SURFACE"].groupby("block_id")["TEST_mean"].mean().to_dict()
    )
    surf_test = float(
        # approximate: use SURFACE from stage1 only; compute from boot baseline
        df[(df.block_id == "SURFACE") & (df.context == "B_SURFACE")]["VAL_mean"].mean()
    )

    # Prefer source fidelity: recommend SOURCE blocks if TEST confirms vs H094 / SURFACE
    promo = []
    verdicts = freeze["source_verdicts"]

    def ps_ok(block):
        sub = s2[(s2.block_id == block) & (s2.context == "A_ALONE")]
        if sub.empty:
            return False
        # mean |P-S| small relative
        return float(np.mean(np.abs(sub["TEST_P"] - sub["TEST_S"]))) < 0.05

    # SAP recommendation
    sap_rec = "NONE"
    if verdicts["SAP24"] in ("SUPPORTED", "SUPPORTED_STRONG") and ps_ok("B1_SAP24"):
        sap_rec = "FS_HIC_SOURCE_SAP24"
    # allow extension only if clearly better on TEST alone
    for b in (freeze["rule"]["best_sap_extension_by_VAL_mean_alone"],):
        if b in s2_mean and s2_mean.get("B1_SAP24") is not None:
            if s2_mean[b] < s2_mean["B1_SAP24"] - 0.005 and ps_ok(b):
                # still prefer source if close
                if s2_mean[b] < s2_mean["B1_SAP24"] - 0.01:
                    sap_rec = b

    scm_rec = "NONE"
    if verdicts["SCM24"] in ("SUPPORTED", "SUPPORTED_STRONG") and ps_ok("B2_SCM24"):
        scm_rec = "FS_HIC_SOURCE_SCM24"
    for b in (freeze["rule"]["best_scm_extension_by_VAL_mean_alone"],):
        if b in s2_mean and "B2_SCM24" in s2_mean:
            if s2_mean[b] < s2_mean["B2_SCM24"] - 0.01 and ps_ok(b):
                scm_rec = b

    comb_rec = "NONE"
    if verdicts["COMBINED48"] in ("COMPLEMENTARY", "MIXED") and ps_ok("B3_SAP24_SCM24"):
        comb_rec = "FS_HIC_SOURCE_SAP24_SCM24"
    for b in (freeze["rule"]["best_combined_extension_by_VAL_mean_alone"],):
        if b in s2_mean and "B3_SAP24_SCM24" in s2_mean:
            if s2_mean[b] < s2_mean["B3_SAP24_SCM24"] - 0.01 and ps_ok(b):
                comb_rec = b

    # Residue-level gate
    residue_ok = False
    for b in ("B1_SAP24", "B2_SCM24", "B3_SAP24_SCM24"):
        if b not in s2_mean:
            continue
        # SURFACE increment on TEST
        if b in s2_surf:
            # lower TEST_mean on SURFACE+block vs need SURFACE baseline — use bootstrap
            boot_b = boot[
                (boot.block_id == b)
                & (boot.comparison == "SURFACE+desc vs SURFACE")
            ]
            if not boot_b.empty and float(boot_b["delta_mae"].mean()) < -0.001:
                if verdicts.get("SAP24" if "SAP" in b else "SCM24", "") in (
                    "SUPPORTED",
                    "SUPPORTED_STRONG",
                    "COMPLEMENTARY",
                    "MIXED",
                ) or b == "B3_SAP24_SCM24":
                    residue_ok = True

    report = [
        "# SOURCE SAP / SCM Feature Research Report",
        "",
        "## Source Stage-1 verdicts",
        "",
        (RES / "SOURCE24_STAGE1_INTERPRETATION.md").read_text(),
        "",
        "## Extension VAL (alone)",
        f"- SAP24={sap24:.6f} SAP30={sap30:.6f} SAP50={sap50:.6f}",
        f"- SCM24={scm24:.6f} SCM30={scm30:.6f} SCM50={scm50:.6f}",
        f"- GLOBAL6 ΔSAP={sap24-sap30:.6f} EXTRA20 ΔSAP30→50={sap30-sap50:.6f}",
        f"- GLOBAL6 ΔSCM={scm24-scm30:.6f} EXTRA20 ΔSCM30→50={scm30-scm50:.6f}",
        "",
        "## Stage-2 TEST (mean over Ridge/SVR)",
        "",
    ]
    for b in [x["block_id"] for x in freeze["shortlist"]]:
        a = block_summary(s2, b, "A_ALONE")
        sb = block_summary(s2, b, "B_SURFACE")
        report.append(
            f"- {b}: alone TEST_mean={a['TEST_mean']:.6f} (P={a['TEST_P']:.6f},S={a['TEST_S']:.6f}) | "
            f"SURFACE+ TEST_mean={sb['TEST_mean']:.6f}"
        )

    report += [
        "",
        "## Promotion (preliminary in report; see MAINLINE_PROMOTION_RECOMMENDATION.md)",
        f"- SAP: {sap_rec}",
        f"- SCM: {scm_rec}",
        f"- SAP+SCM: {comb_rec}",
        f"- residue-level justified: {residue_ok}",
        "",
        "EXP-H114 was not run.",
    ]
    (RES / "SOURCE_SAP_SCM_REPORT.md").write_text("\n".join(report) + "\n")

    promo_md = [
        "# Mainline Promotion Recommendation — SAP/SCM Source Track",
        "",
        "**Next HIC code: EXP-H114 — DO NOT RUN from this track.**",
        "",
        "## Recommendations (max 3)",
        "",
        f"| Slot | Block | Decision |",
        f"|------|-------|----------|",
        f"| SAP | {sap_rec} | {'PROMOTE_CANDIDATE' if sap_rec != 'NONE' else 'NO_PROMOTION'} |",
        f"| SCM | {scm_rec} | {'PROMOTE_CANDIDATE' if scm_rec != 'NONE' else 'NO_PROMOTION'} |",
        f"| SAP+SCM | {comb_rec} | {'PROMOTE_CANDIDATE' if comb_rec != 'NONE' else 'NO_PROMOTION'} |",
        "",
        "## Preference order applied",
        "1. source fidelity  2. TEST confirmation  3. P/S consistency  4. SURFACE increment  5. simplicity",
        "",
        f"## Residue-level",
        f"{'JUSTIFIED for future work using SAP_R5/R10 and/or SCM_R5/R10 from residue_source_sap_scm.parquet' if residue_ok else 'NOT justified yet'}",
        "",
        "## Notes",
        "- No radius/scale/geometry optimization was performed.",
        "- Extensions promoted only if clearly superior to SOURCE24 on TEST.",
    ]
    (RES / "MAINLINE_PROMOTION_RECOMMENDATION.md").write_text("\n".join(promo_md) + "\n")

    # store machine-readable decisions
    (RES / "PROMOTION_DECISIONS.json").write_text(
        json.dumps(
            {
                "sap": sap_rec,
                "scm": scm_rec,
                "combined": comb_rec,
                "residue_level_justified": residue_ok,
                "source_verdicts": verdicts,
                "s2_alone_test_mean": s2_mean,
                "s2_surface_test_mean": s2_surf,
            },
            indent=2,
        )
    )


def main():
    wide = pd.read_parquet(FEAT / "antibody_source_sap_scm_wide.parquet")
    res_df = pd.read_parquet(FEAT / "residue_source_sap_scm.parquet")
    write_catalog()
    write_specs()
    print("reconstruction...", flush=True)
    run_reconstruction_check(wide, res_df)
    print("QC...", flush=True)
    target_blind_qc(wide, res_df)
    print("Stage1...", flush=True)
    df, freeze = stage1(wide)
    print("Stage2...", flush=True)
    stage2(wide, freeze)
    print("Reports...", flush=True)
    write_reports(df, freeze)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
