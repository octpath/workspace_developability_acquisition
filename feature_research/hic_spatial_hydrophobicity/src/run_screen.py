#!/usr/bin/env python3
"""Stage-1 VAL screen + Stage-2 TEST confirmation for HSP atlas."""
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
ROOT = HERE.parent
RES = ROOT / "results"
FEAT = ROOT / "features"
DRILL = ROOT.parents[1] / "developability_drilldown"
sys.path.insert(0, str(DRILL / "scripts"))
sys.path.insert(0, str(DRILL / "models"))
sys.path.insert(0, str(HERE))

from antibody_transformer.data import load_dev_test, load_folds, load_solution, tvt_split  # noqa: E402
from antibody_transformer.config import BUNDLE_ROOT  # noqa: E402
from spatial_engine import family_id, feature_name, SCOPES, AGGS  # noqa: E402
from scales import all_scale_ids  # noqa: E402
from spatial_engine import EXPOSURES, NEIGHBORHOODS, RADII, TRANSFORMS  # noqa: E402

H047 = DRILL / "experiments/features/EXP-H047.parquet"
SEED = 101


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


def h047_blocks():
    df = pd.read_parquet(H047)
    cols = [c for c in df.columns if c not in ("id", "split")]
    # ESM2_H 1280 | SEQ 115 | ARO 19 | HYDRO 16 | TITR 18 | FB 1280
    surface = cols[1395:1430]  # 35
    seq = cols[1280:1395]
    titr = cols[1430:1448]
    return df[["id"] + surface + seq + titr].copy(), surface, seq, titr


def bundle_columns(family: str, bundle: str) -> list[str]:
    if bundle == "B3":
        return [feature_name(family, "ALL_FV", a) for a in ("MAX", "MEAN", "SUM")]
    if bundle == "BH":
        return [feature_name(family, "ALL_FV", a) for a in ("MAX", "Q90", "Q95", "TOP3_MEAN", "TOP5_MEAN")]
    if bundle == "BC":
        cols = []
        for scope in ("HEAVY", "LIGHT"):
            for a in ("MAX", "MEAN", "SUM", "Q95", "TOP5_MEAN"):
                cols.append(feature_name(family, scope, a))
        return cols
    if bundle == "BR":
        cols = []
        for scope in ("ALL_CDR", "H_CDR3", "L_CDR3"):
            for a in ("MAX", "MEAN", "SUM", "Q95", "TOP5_MEAN"):
                cols.append(feature_name(family, scope, a))
        return cols
    if bundle == "BALL":
        cols = []
        for scope in SCOPES:
            for a in AGGS:
                cols.append(feature_name(family, scope, a))
        return cols
    raise ValueError(bundle)


def lit_family_columns(ab_cols: list[str], prefix: str) -> list[str]:
    return [c for c in ab_cols if c.startswith(prefix)]


def eval_oof_val(X: pd.DataFrame, y: pd.Series, fold_map: dict, estimator_name: str) -> dict:
    ids = y.index.astype(str).tolist()
    preds = pd.Series(np.nan, index=ids, dtype=float)
    for k in range(5):
        tr, va, te = tvt_split(fold_map, k, ids)
        # Stage1: fit TRAIN only; score VAL; ignore TEST
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
        # Stage2 confirmation: same fixed estimators; fit TRAIN (not VAL) predict TEST
        est.fit(X.loc[tr], y.loc[tr])
        preds.loc[te] = est.predict(X.loc[te])
    return {"TEST_mae": mae(y, preds), "preds": preds}


def context_matrix(ab: pd.DataFrame, cols: list[str], ctx: str, h047: pd.DataFrame, surface, seq, titr):
    base = ab.set_index("id")
    h = h047.set_index("id")
    parts = []
    if ctx == "A":
        parts.append(base[cols])
    elif ctx == "B":
        parts.append(h[surface])
        parts.append(base[cols])
    elif ctx == "C":
        parts.append(h[surface + seq + titr])
        parts.append(base[cols])
    else:
        raise ValueError(ctx)
    return pd.concat(parts, axis=1)


def paired_boot(a, b, y, n=2000, seed=101):
    ea = np.abs(a - y)
    eb = np.abs(b - y)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    boots = np.asarray(boots)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def run_qc_and_univariate(ab: pd.DataFrame, dev: pd.DataFrame):
    feat_cols = [c for c in ab.columns if c != "id"]
    X = ab.set_index("id")[feat_cols]
    # target-blind QC
    rows = []
    for c in feat_cols:
        v = X[c].to_numpy(float)
        rows.append(
            {
                "feature": c,
                "finite_frac": float(np.isfinite(v).mean()),
                "missing_frac": float((~np.isfinite(v)).mean()),
                "std": float(np.nanstd(v)),
                "min": float(np.nanmin(v)),
                "q05": float(np.nanquantile(v[np.isfinite(v)], 0.05)) if np.isfinite(v).any() else np.nan,
                "median": float(np.nanmedian(v)),
                "q95": float(np.nanquantile(v[np.isfinite(v)], 0.95)) if np.isfinite(v).any() else np.nan,
                "max": float(np.nanmax(v)),
                "constant": bool(np.nanstd(v) < 1e-12),
            }
        )
    qc = pd.DataFrame(rows)
    qc.to_csv(RES / "TARGET_BLIND_QC_DETAIL.csv", index=False)

    # confound correlations (antibody-level size proxies)
    geo = pd.read_parquet(FEAT / "cache/residue_geometry_sasa.parquet")
    sh = geo[geo.chain == "H"].groupby("id")["total_SASA"].sum()
    sl = geo[geo.chain == "L"].groupby("id")["total_SASA"].sum()
    size = geo.groupby("id").size().rename("n_res").to_frame()
    size["total_SASA"] = geo.groupby("id")["total_SASA"].sum()
    size["n_H"] = geo[geo.chain == "H"].groupby("id").size()
    size["n_L"] = geo[geo.chain == "L"].groupby("id").size()
    size["SASA_H"] = sh
    size["SASA_L"] = sl

    # redundancy on a reduced set: family-level ALL_FV MEAN columns to keep tractable
    mean_cols = [c for c in feat_cols if c.endswith("__ALL_FV__MEAN") or c.startswith("LIT")]
    # sample if huge
    if len(mean_cols) > 800:
        mean_cols = mean_cols[:800]
    Xm = X[mean_cols].to_numpy(float)
    # pairwise spearman clustering greedy
    cluster = {}
    cid = 0
    assigned = {}
    for i, c in enumerate(mean_cols):
        if c in assigned:
            continue
        assigned[c] = cid
        for j in range(i + 1, len(mean_cols)):
            cj = mean_cols[j]
            if cj in assigned:
                continue
            a, b = X[c].to_numpy(float), X[cj].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            rho = spearmanr(a[m], b[m])[0]
            if np.isfinite(rho) and abs(rho) >= 0.98:
                assigned[cj] = cid
        cid += 1
    red = pd.DataFrame([{"feature": k, "redundancy_cluster_id": v} for k, v in assigned.items()])
    red.to_csv(RES / "FEATURE_REDUNDANCY.csv", index=False)

    # HYDRO_FIELD / ARO max corr for ALL_FV MEAN generic
    h047, surface, seq, titr = h047_blocks()
    hydro_cols = surface[19:]  # HYDRO 16 after ARO 19
    aro_cols = surface[:19]
    hm = h047.set_index("id")
    corr_rows = []
    for c in [x for x in mean_cols if "__ALL_FV__MEAN" in x][:200]:
        for block, bcols in (("HYDRO_FIELD", hydro_cols), ("AROMATIC_TOPO", aro_cols)):
            rhos = []
            for bc in bcols:
                a, b = X[c].to_numpy(float), hm[bc].reindex(X.index).to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 10:
                    rhos.append(abs(spearmanr(a[m], b[m])[0]))
            corr_rows.append(
                {
                    "feature": c,
                    "block": block,
                    "spearman_max_abs": float(np.nanmax(rhos)) if rhos else np.nan,
                }
            )
        for nu in ("n_res", "total_SASA", "SASA_H", "SASA_L"):
            a = X[c].to_numpy(float)
            b = size.reindex(X.index)[nu].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            rho = spearmanr(a[m], b[m])[0] if m.sum() > 10 else np.nan
            corr_rows.append({"feature": c, "block": f"NUISANCE_{nu}", "spearman_max_abs": float(rho) if np.isfinite(rho) else np.nan})
    pd.DataFrame(corr_rows).to_csv(RES / "FEATURE_BLOCK_CORR.csv", index=False)

    # univariate HIC on Dev
    d = X.join(dev.set_index("id")["HIC"], how="inner")
    uni = []
    for c in feat_cols:
        rho, p = spearmanr(d[c], d["HIC"], nan_policy="omit")
        uni.append({"feature": c, "spearman_rho_HIC_dev": float(rho) if np.isfinite(rho) else np.nan, "pvalue": float(p) if np.isfinite(p) else np.nan})
    pd.DataFrame(uni).to_csv(RES / "UNIVARIATE_HIC_ASSOCIATION.csv", index=False)

    (RES / "TARGET_BLIND_QC.md").write_text(
        "\n".join(
            [
                "# Target-blind QC",
                "",
                f"- antibody features: {len(feat_cols)}",
                f"- constant features: {int(qc['constant'].sum())}",
                f"- median finite_frac: {qc['finite_frac'].median():.4f}",
                f"- redundancy clusters (on MEAN/LIT subset): {red['redundancy_cluster_id'].nunique()}",
                "",
                "Near-duplicate threshold: |Spearman|>=0.98 (reported, not deleted).",
                "",
            ]
        )
    )
    return red


def stage1(ab: pd.DataFrame, red: pd.DataFrame):
    dev, test = load_dev_test(DRILL / "data/dev.csv", DRILL / "data/test.csv")
    folds = load_folds(DRILL / "data/folds.csv")
    h047, surface, seq, titr = h047_blocks()
    y = dev.set_index("id")["HIC"]
    ab_i = ab.set_index("id").reindex(y.index)

    # baselines
    baselines = {}
    for ctx_name, cols_fn in (
        ("SURFACE", lambda: surface),
        ("PHYS_LOWDIM", lambda: surface + seq + titr),
        ("NONE", lambda: []),
    ):
        for est in ("Ridge", "SVR"):
            for scheme, fmap in (("primary", folds.primary), ("shadow", folds.shadow)):
                if ctx_name == "NONE":
                    baselines[(ctx_name, est, scheme)] = None
                    continue
                Xb = h047.set_index("id").reindex(y.index)[cols_fn()]
                r = eval_oof_val(Xb, y, fmap, est)
                baselines[(ctx_name, est, scheme)] = r["VAL_mae"]

    # families to screen: all generic + lit prefixes
    families = []
    for scale in all_scale_ids():
        for transform in TRANSFORMS:
            for exposure in EXPOSURES:
                for neigh in NEIGHBORHOODS:
                    for R in RADII:
                        families.append(("GENERIC", family_id(scale, transform, exposure, neigh, R)))
    lit_prefixes = sorted({c.split("__")[0] for c in ab.columns if c.startswith("LIT")})
    for p in lit_prefixes:
        families.append(("LIT", p))

    bundles = ["B3", "BH", "BC", "BR", "BALL"]
    rows = []
    total = len(families) * len(bundles) * 3 * 2
    done = 0
    for kind, fam in families:
        for bundle in bundles:
            if kind == "GENERIC":
                cols = bundle_columns(fam, bundle)
            else:
                cols = lit_family_columns(list(ab.columns), fam + "__")
                if bundle != "B3":
                    # literature anchors evaluated once under B3-equivalent (all their columns)
                    if bundle != "BALL":
                        continue
            cols = [c for c in cols if c in ab.columns]
            if not cols:
                continue
            for ctx, ctx_id, base_key in (
                ("A", "A_ALONE", "NONE"),
                ("B", "B_SURFACE", "SURFACE"),
                ("C", "C_PHYS", "PHYS_LOWDIM"),
            ):
                X = context_matrix(ab, cols, ctx, h047, surface, seq, titr).reindex(y.index)
                for est in ("Ridge", "SVR"):
                    vp = eval_oof_val(X, y, folds.primary, est)["VAL_mae"]
                    vs = eval_oof_val(X, y, folds.shadow, est)["VAL_mae"]
                    base_p = baselines.get((base_key, est, "primary"))
                    base_s = baselines.get((base_key, est, "shadow"))
                    d_p = (vp - base_p) if base_p is not None else np.nan
                    d_s = (vs - base_s) if base_s is not None else np.nan
                    rows.append(
                        {
                            "family_id": fam,
                            "kind": kind,
                            "bundle": bundle,
                            "context": ctx_id,
                            "estimator": est,
                            "VAL_P": vp,
                            "VAL_S": vs,
                            "VAL_mean": 0.5 * (vp + vs),
                            "VAL_worst": max(vp, vs),
                            "delta_VAL_P_vs_baseline": d_p,
                            "delta_VAL_S_vs_baseline": d_s,
                            "delta_VAL_mean_vs_baseline": np.nanmean([d_p, d_s]),
                            "n_features": len(X.columns),
                        }
                    )
                    done += 1
                    if done % 500 == 0:
                        print(f"stage1 {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "STAGE1_VAL_SCREEN.csv", index=False)

    # factor effect summary
    g = df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "A_ALONE")]
    # parse family factors
    def parse_fam(fid: str):
        # HSP_SCALE_TRANSFORM_EXPOSURE_NEIGH_Rr
        parts = fid.split("_")
        # HSP KD MINMAX TOTAL RASA TIEN CENTROID R5p0  — exposure has underscores
        # family_id format: HSP_{scale}_{transform}_{exposure}_{neigh}_R{rtag}
        # exposure in EXPOSURES has underscores
        for exp in EXPOSURES:
            for neigh in NEIGHBORHOODS:
                for R in RADII:
                    rtag = str(R).replace(".", "p")
                    for scale in all_scale_ids():
                        for transform in TRANSFORMS:
                            expect = family_id(scale, transform, exp, neigh, R)
                            if fid == expect:
                                return scale, transform, exp, neigh, R
        return None, None, None, None, None

    parsed = g["family_id"].apply(lambda x: pd.Series(parse_fam(x), index=["scale", "transform", "exposure", "neigh", "radius"]))
    g2 = pd.concat([g.reset_index(drop=True), parsed], axis=1)

    def factor_table(col):
        return (
            g2.groupby(["estimator", col])["VAL_mean"]
            .agg(["median", "mean", "count"])
            .reset_index()
            .sort_values(["estimator", "median"])
        )

    lines = ["# Factor effect summary (Stage-1 VAL; GENERIC B3 ALONE)", ""]
    for col, title in (
        ("scale", "Hydrophobicity scale"),
        ("transform", "RAW vs MINMAX"),
        ("exposure", "Exposure"),
        ("neigh", "Neighborhood"),
        ("radius", "Radius"),
    ):
        lines += [f"## {title}", "", factor_table(col).to_string(index=False), ""]
    # aggregation / region / bundle effects
    lines += ["## Bundle effect (GENERIC, context A, Ridge)", ""]
    lines.append(
        df[(df.kind == "GENERIC") & (df.context == "A_ALONE") & (df.estimator == "Ridge")]
        .groupby("bundle")["VAL_mean"]
        .median()
        .sort_values()
        .to_string()
    )
    lines += ["", "## Context B incremental (GENERIC B3, negative Δ better)", ""]
    lines.append(
        df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "B_SURFACE")]
        .groupby("estimator")["delta_VAL_mean_vs_baseline"]
        .median()
        .to_string()
    )
    (RES / "FACTOR_EFFECT_SUMMARY.md").write_text("\n".join(lines))

    # shortlist
    # map family -> cluster via ALL_FV MEAN feature if present
    mean_map = {}
    for fam in g2["family_id"].unique():
        fn = feature_name(fam, "ALL_FV", "MEAN")
        hit = red[red.feature == fn]
        mean_map[fam] = int(hit.iloc[0].redundancy_cluster_id) if len(hit) else hash(fam) % 10_000

    def pick_track(sub, delta=False, k=2):
        sub = sub.sort_values(["VAL_mean" if not delta else "delta_VAL_mean_vs_baseline", "VAL_worst"])
        if delta:
            sub = sub.sort_values(["delta_VAL_mean_vs_baseline", "VAL_worst"])
        chosen = []
        used = set()
        for _, r in sub.iterrows():
            cl = mean_map.get(r.family_id, -1)
            if cl in used:
                continue
            chosen.append(r.family_id)
            used.add(cl)
            if len(chosen) >= k:
                break
        return chosen

    tracks = {
        "alone_ridge": pick_track(df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "A_ALONE") & (df.estimator == "Ridge")]),
        "alone_svr": pick_track(df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "A_ALONE") & (df.estimator == "SVR")]),
        "inc_surface_ridge": pick_track(
            df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "B_SURFACE") & (df.estimator == "Ridge")],
            delta=True,
        ),
        "inc_surface_svr": pick_track(
            df[(df.kind == "GENERIC") & (df.bundle == "B3") & (df.context == "B_SURFACE") & (df.estimator == "SVR")],
            delta=True,
        ),
    }
    short = []
    for t, fams in tracks.items():
        for f in fams:
            short.append({"track": t, "family_id": f, "source": "data_driven"})

    # mandatory lit
    for pref, label in (
        ("LIT2_STATIC_SAP_BM_R5p0", "mandatory_lit2"),
        ("LIT3_PSH_KD", "mandatory_lit3"),
    ):
        short.append({"track": "mandatory", "family_id": pref, "source": label})
    # best LIT4 WW and MEEK/KD by alone ridge VAL on BALL
    lit4 = df[(df.family_id.str.startswith("LIT4_POS_SASA")) & (df.bundle == "BALL") & (df.context == "A_ALONE") & (df.estimator == "Ridge")]
    if len(lit4):
        for needle, lab in (("WW", "mandatory_lit4_ww"), ("MEEK", "mandatory_lit4_meek"), ("KD", "mandatory_lit4_kd")):
            sub = lit4[lit4.family_id.str.contains(needle)]
            if len(sub):
                best = sub.sort_values("VAL_mean").iloc[0].family_id
                short.append({"track": "mandatory", "family_id": best, "source": lab})

    # unique cap 12
    seen = set()
    final = []
    for r in short:
        if r["family_id"] in seen:
            continue
        seen.add(r["family_id"])
        final.append(r)
        if len(final) >= 12:
            break

    freeze = {
        "status": "STAGE1_SHORTLIST_FROZEN",
        "rule": str(RES / "STAGE1_SHORTLIST_RULE.yaml"),
        "shortlist": final,
        "tracks": tracks,
        "baselines_VAL": {f"{a}|{b}|{c}": v for (a, b, c), v in baselines.items() if v is not None},
        "n_stage1_rows": len(df),
        "mainline_next_hic": "EXP-H102",
        "consumed_EXP_H102": False,
    }
    (RES / "STAGE1_SHORTLIST_FREEZE.yaml").write_text(yaml.safe_dump(freeze, sort_keys=False))
    print("Shortlist:", final, flush=True)
    return df, freeze, baselines


def stage2(ab: pd.DataFrame, freeze: dict, baselines_val):
    dev, test = load_dev_test(DRILL / "data/dev.csv", DRILL / "data/test.csv")
    folds = load_folds(DRILL / "data/folds.csv")
    sol = load_solution(BUNDLE_ROOT / "solution.csv")
    h047, surface, seq, titr = h047_blocks()
    y = dev.set_index("id")["HIC"]

    rows = []
    boots = []
    ext_rows = []
    for item in freeze["shortlist"]:
        fam = item["family_id"]
        if fam.startswith("HSP_"):
            cols = bundle_columns(fam, "B3")
        else:
            cols = lit_family_columns(list(ab.columns), fam + "__")
            if not cols:
                cols = lit_family_columns(list(ab.columns), fam)
        cols = [c for c in cols if c in ab.columns]
        if not cols:
            continue
        for ctx, ctx_id in (("A", "A_ALONE"), ("B", "B_SURFACE")):
            X = context_matrix(ab, cols, ctx, h047, surface, seq, titr).reindex(y.index)
            for est in ("Ridge", "SVR"):
                tp = eval_oof_test(X, y, folds.primary, est)
                ts = eval_oof_test(X, y, folds.shadow, est)
                rows.append(
                    {
                        "family_id": fam,
                        "context": ctx_id,
                        "estimator": est,
                        "TEST_P": tp["TEST_mae"],
                        "TEST_S": ts["TEST_mae"],
                        "TEST_mean": 0.5 * (tp["TEST_mae"] + ts["TEST_mae"]),
                        "TEST_worst": max(tp["TEST_mae"], ts["TEST_mae"]),
                    }
                )
                # bootstrap vs baseline SURFACE for context B; vs empty for A skip
                if ctx == "B":
                    Xb = h047.set_index("id").reindex(y.index)[surface]
                    bp = eval_oof_test(Xb, y, folds.primary, est)
                    bs = eval_oof_test(Xb, y, folds.shadow, est)
                    for scheme, pa, pb in (
                        ("primary", tp["preds"], bp["preds"]),
                        ("shadow", ts["preds"], bs["preds"]),
                    ):
                        d, lo, hi = paired_boot(pa.to_numpy(), pb.to_numpy(), y.to_numpy())
                        boots.append(
                            {
                                "family_id": fam,
                                "estimator": est,
                                "scheme": scheme,
                                "comparison": "SURFACE+desc vs SURFACE",
                                "delta_mae": d,
                                "ci95_lo": lo,
                                "ci95_hi": hi,
                            }
                        )
                # external diagnostic: fit on full Dev with TRAIN-style pipeline (no test leakage from folds:
                # use mean of fold models already — here refit on all Dev for diagnostic only)
                estm = make_ridge() if est == "Ridge" else make_svr()
                X_all = context_matrix(ab, cols, ctx, h047, surface, seq, titr)
                # Dev fit
                estm.fit(X_all.reindex(y.index), y)
                te_ids = test["id"].astype(str).tolist()
                pred = estm.predict(X_all.reindex(te_ids))
                sol2 = sol.set_index("id")
                te = pd.Series(pred, index=te_ids)
                pub = sol2.index[sol2["is_public"].astype(bool)]
                priv = sol2.index[sol2["is_private"].astype(bool)]
                ext_rows.append(
                    {
                        "family_id": fam,
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
    return rows


def main():
    ab = pd.read_parquet(FEAT / "antibody_spatial_hydrophobicity.parquet")
    print("antibody atlas", ab.shape, flush=True)
    dev, _ = load_dev_test(DRILL / "data/dev.csv", DRILL / "data/test.csv")
    red = run_qc_and_univariate(ab, dev)
    df, freeze, baselines = stage1(ab, red)
    print("Stage1 done; freeze written — proceeding Stage2", flush=True)
    stage2(ab, freeze, baselines)
    print("Stage2 done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
