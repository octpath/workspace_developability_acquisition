#!/usr/bin/env python3
"""Evaluate gap-closure families on Primary/Shadow CV (MAE-primary)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "structure_gap_closure"
RESULTS = CTX / "results"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    HIC_REF,
    OOF_DIR,
    TMAPP_REF,
    bootstrap_delta,
    mae,
    median_baseline_oof,
    metrics,
    nested_ridge,
)

PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
ARO = CTX / "cache/aromatic_features_esmfold.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
RNG = np.random.default_rng(42)


def load_y(target: str):
    name = TMAPP_REF if target == "TmApp" else HIC_REF
    s = pd.read_csv(OOF_DIR / f"{name}.csv").set_index("id")["y_true"].astype(float)
    s.index = s.index.astype(str)
    return s


def load_inc(target: str):
    name = TMAPP_REF if target == "TmApp" else HIC_REF
    s = pd.read_csv(OOF_DIR / f"{name}.csv").set_index("id")["y_pred"].astype(float)
    s.index = s.index.astype(str)
    return s


def load_ablang2():
    z = np.load(EMB, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])


def load_esm2_h():
    p2 = OOF_DIR / "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt__primary_nested_base.csv"
    if p2.exists():
        s = pd.read_csv(p2).set_index("id")["y_pred"].astype(float)
        s.index = s.index.astype(str)
        return s
    p = OOF_DIR / "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt.csv"
    s = pd.read_csv(p).set_index("id")["y_pred"].astype(float)
    s.index = s.index.astype(str)
    return s


def select_alpha(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        maes = []
        for f in sorted({fold_map[i] for i in train_ids}):
            te = [i for i in train_ids if fold_map[i] == f]
            tr = [i for i in train_ids if fold_map[i] != f]
            if len(te) == 0 or len(tr) < 2:
                continue
            sc = StandardScaler()
            Xt = np.nan_to_num(sc.fit_transform(X.loc[tr]), nan=0.0, posinf=0.0, neginf=0.0)
            Xe = np.nan_to_num(sc.transform(X.loc[te]), nan=0.0, posinf=0.0, neginf=0.0)
            m = Ridge(alpha=a, random_state=0)
            m.fit(Xt, y.loc[tr])
            maes.append(mae(y.loc[te], m.predict(Xe)))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def nested(X, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        a = select_alpha(X, y, folds, tr)
        sc = StandardScaler()
        Xt = np.nan_to_num(sc.fit_transform(X.loc[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Xe = np.nan_to_num(sc.transform(X.loc[te]), nan=0.0, posinf=0.0, neginf=0.0)
        m = Ridge(alpha=a, random_state=0)
        m.fit(Xt, y.loc[tr])
        oof.loc[te] = m.predict(Xe)
    return oof.astype(float)


def prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.set_index("id").copy()
    X.index = X.index.astype(str)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    # drop constant / near-constant columns (avoid StandardScaler NaNs)
    keep = [c for c in X.columns if float(X[c].std(ddof=0)) > 1e-12]
    X = X[keep]
    return X


def _safe_transform(sc: StandardScaler, Xpart: pd.DataFrame) -> np.ndarray:
    arr = sc.fit_transform(Xpart) if not hasattr(sc, "mean_") or sc.mean_ is None else sc.transform(Xpart)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def eval_family(name, X, y, folds, tag, reference_pred=None, mode="standalone"):
    ids = sorted(set(X.index.astype(str)) & set(y.index) & set(folds.id.astype(str)))
    if reference_pred is not None:
        ids = [i for i in ids if i in set(reference_pred.index.astype(str))]
    X = X.copy()
    X.index = X.index.astype(str)
    X = X.loc[ids]
    y = y.loc[ids]
    folds = folds[folds.id.astype(str).isin(ids)].copy()
    folds["id"] = folds.id.astype(str)
    if X.shape[0] < 30 or X.shape[1] == 0:
        return None
    if mode == "standalone":
        oof = nested(X, y, folds)
        base_oof, _ = median_baseline_oof(y, folds)
        d = mae(y, oof) - mae(y, base_oof)
        ref_mae = mae(y, base_oof)
        pred_mae = mae(y, oof)
        ref_name = "median"
    elif mode.startswith("incr"):
        # stack reference prediction as extra column
        Xp = X.copy()
        Xp["__ref__"] = reference_pred.loc[ids].astype(float)
        oof = nested(Xp, y, folds)
        ref_only = reference_pred.loc[ids].astype(float)
        pred_mae = mae(y, oof)
        ref_mae = mae(y, ref_only)
        d = pred_mae - ref_mae
        ref_name = mode
    elif mode == "residual":
        resid = y - reference_pred.loc[ids]
        oof_r = nested(X, resid, folds)
        # reconstructed
        oof = reference_pred.loc[ids] + oof_r
        pred_mae = mae(y, oof)
        ref_mae = mae(y, reference_pred.loc[ids])
        d = pred_mae - ref_mae
        ref_name = "residual"
    else:
        return None
    boot = {"ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    if mode != "standalone" and d < 0:
        # paired bootstrap on |y-ref| - |y-pred|
        ref_p = reference_pred.loc[ids].astype(float) if mode != "standalone" else None
        if ref_p is not None:
            delta_vec = np.abs(y - ref_p) - np.abs(y - oof)
            n = len(delta_vec)
            boots = []
            for _ in range(min(B_BOOT, 10000)):
                ix = RNG.integers(0, n, n)
                boots.append(float(delta_vec.iloc[ix].mean()))
            boots = np.asarray(boots)
            boot = {
                "ci_lo": float(np.quantile(boots, 0.025)),
                "ci_hi": float(np.quantile(boots, 0.975)),
                "p": float(np.mean(boots <= 0)),  # p that improvement <=0 ? use 1-frac positive
            }
            boot["p_improve"] = float(np.mean(boots > 0))
    return {
        "family": name,
        "cv": tag,
        "mode": mode,
        "N": len(ids),
        "feature_dim": int(X.shape[1]),
        "pred_mae": pred_mae,
        "ref_mae": ref_mae,
        "delta_mae": d,
        "bootstrap_ci_low": boot.get("ci_lo"),
        "bootstrap_ci_high": boot.get("ci_hi"),
        "p_improve": boot.get("p_improve", np.nan),
        "reference": ref_name,
    }


def main():
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    ablang = load_ablang2()
    # AbLang2 OOF via nested on embeddings is expensive; use mean of dims as weak ref —
    # recover exact: use META file's comparison — load PLM ridge OOF if exists
    plm_oof_path = OOF_DIR / "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt__primary_nested_base.csv"
    if plm_oof_path.exists():
        ablang_oof = pd.read_csv(plm_oof_path).set_index("id")["y_pred"].astype(float)
        ablang_oof.index = ablang_oof.index.astype(str)
    else:
        ablang_oof = None

    families = []
    # TmApp families
    pack = RESULTS / "TMAPP_PACKING_CAVITY_FEATURES.csv"
    unsat = RESULTS / "TMAPP_BURIED_UNSAT_FEATURES.csv"
    iface = RESULTS / "TMAPP_INTERFACE_FEATURES.csv"
    surf = RESULTS / "HIC_CONTINUOUS_SURFACE_FEATURES.csv"
    if pack.exists():
        families.append(("TmApp", "PACKING_CAVITY", "raw_fab", prep_X(pd.read_csv(pack))))
    if unsat.exists():
        families.append(("TmApp", "BURIED_UNSAT", "prepared_fab", prep_X(pd.read_csv(unsat))))
    if iface.exists():
        X = prep_X(pd.read_csv(iface).drop(columns=["shape_complementarity_status"], errors="ignore"))
        families.append(("TmApp", "FAB_INTERFACE", "raw_fab", X))
    if surf.exists():
        S = pd.read_csv(surf)
        # Fv ESMFold columns only for primary HIC
        cols = [c for c in S.columns if c.startswith("fv_esmfold__")]
        if cols:
            families.append(("HIC", "CONT_SURFACE_FV_ESMFOLD", "fv_esmfold", prep_X(S[["id"] + cols])))
        cols = [c for c in S.columns if c.startswith("delta_ctx__")]
        if cols:
            families.append(("HIC", "DELTA_PATCH_CONTEXT", "fv_vs_fab", prep_X(S[["id"] + cols])))
        # per-scale subsets on fv_esmfold
        for scale in ("KD", "FP", "BM"):
            cols = [c for c in S.columns if c.startswith(f"fv_esmfold__{scale}__")]
            if cols:
                families.append(("HIC", f"CONT_SURFACE_FV_{scale}", "fv_esmfold", prep_X(S[["id"] + cols])))

    rows = []
    for target, fam, source, X in families:
        y = load_y(target)
        inc = load_inc(target)
        esm2 = load_esm2_h() if target == "HIC" else None
        # aromatic topo features for stack
        aro_pred = None
        if target == "HIC" and ARO.exists() and aro_pred is None:
            A = pd.read_csv(ARO)
            A = prep_X(A)
            ids = sorted(set(A.index) & set(y.index) & set(primary.id.astype(str)))
            Af = A.loc[ids]
            folds_p = primary[primary.id.astype(str).isin(ids)].copy()
            folds_p["id"] = folds_p.id.astype(str)
            aro_pred = nested(Af, y.loc[ids], folds_p)

        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            r = eval_family(fam, X, y, folds, tag, mode="standalone")
            if r:
                r.update({"target": target, "structure_source": source, "subfamily": "all"})
                rows.append(r)
            if target == "TmApp":
                # vs AbLang2 embedding nested
                if ablang_oof is not None:
                    r = eval_family(fam, X, y, folds, tag, reference_pred=ablang_oof, mode="incr_ablang2")
                    if r:
                        r.update({"target": target, "structure_source": source, "subfamily": "incr_ablang2"})
                        rows.append(r)
                r = eval_family(fam, X, y, folds, tag, reference_pred=inc, mode="incr_incumbent")
                if r:
                    r.update({"target": target, "structure_source": source, "subfamily": "incr_incumbent"})
                    rows.append(r)
                r = eval_family(fam, X, y, folds, tag, reference_pred=inc, mode="residual")
                if r:
                    r.update({"target": target, "structure_source": source, "subfamily": "residual"})
                    rows.append(r)
            else:
                r = eval_family(fam, X, y, folds, tag, reference_pred=esm2, mode="incr_esm2")
                if r:
                    r.update({"target": target, "structure_source": source, "subfamily": "incr_esm2"})
                    rows.append(r)
                if esm2 is not None and aro_pred is not None:
                    # stack esm2 + aromatic as reference: use mean of two preds
                    ids = sorted(set(esm2.index) & set(aro_pred.index))
                    ref = 0.5 * (esm2.loc[ids] + aro_pred.loc[ids])
                    r = eval_family(fam, X, y, folds, tag, reference_pred=ref, mode="incr_esm2_aro")
                    if r:
                        r.update({"target": target, "structure_source": source, "subfamily": "incr_esm2_aro"})
                        rows.append(r)
                r = eval_family(fam, X, y, folds, tag, reference_pred=inc, mode="incr_incumbent")
                if r:
                    r.update({"target": target, "structure_source": source, "subfamily": "incr_incumbent"})
                    rows.append(r)
                r = eval_family(fam, X, y, folds, tag, reference_pred=inc, mode="residual")
                if r:
                    r.update({"target": target, "structure_source": source, "subfamily": "residual"})
                    rows.append(r)
        print("scored", target, fam, flush=True)

    out = pd.DataFrame(rows)
    # scorecard reshape
    sc_rows = []
    for (target, fam, source), g in out.groupby(["target", "family", "structure_source"]):
        prim = g[g.cv == "Primary"]
        shad = g[g.cv == "Shadow"]
        stand_p = prim[prim["mode"] == "standalone"]
        stand_s = shad[shad["mode"] == "standalone"]
        for mode in sorted(set(g["mode"]) - {"standalone"}):
            rp = prim[prim["mode"] == mode]
            rs = shad[shad["mode"] == mode]
            if rp.empty or rs.empty or stand_p.empty:
                continue
            dp = float(rp.iloc[0].delta_mae)
            ds = float(rs.iloc[0].delta_mae)
            verdict = "NO_SIGNAL"
            if dp < 0 and ds < 0:
                verdict = "CLEAR_INCREMENT" if abs(dp) > 0.01 else "WEAK_SIGNAL"
            elif dp < 0 or ds < 0:
                verdict = "WEAK_SIGNAL"
            if abs(dp) < 1e-4 and abs(ds) < 1e-4:
                verdict = "REDUNDANT_SIGNAL"
            sc_rows.append(
                {
                    "target": target,
                    "family": fam,
                    "subfamily": mode,
                    "structure_source": source,
                    "scope": source,
                    "N": int(rp.iloc[0].N),
                    "feature_dim": int(rp.iloc[0].feature_dim),
                    "standalone_primary_mae": float(stand_p.iloc[0].pred_mae),
                    "standalone_shadow_mae": float(stand_s.iloc[0].pred_mae) if len(stand_s) else np.nan,
                    "reference": mode,
                    "incremental_primary_mae": float(rp.iloc[0].pred_mae),
                    "incremental_shadow_mae": float(rs.iloc[0].pred_mae),
                    "delta_primary": dp,
                    "delta_shadow": ds,
                    "bootstrap_ci_low": float(rp.iloc[0].bootstrap_ci_low) if pd.notna(rp.iloc[0].bootstrap_ci_low) else np.nan,
                    "bootstrap_ci_high": float(rp.iloc[0].bootstrap_ci_high) if pd.notna(rp.iloc[0].bootstrap_ci_high) else np.nan,
                    "artifact_status": "PENDING",
                    "prep_sensitivity": "PENDING",
                    "generator_robustness": "PENDING",
                    "verdict": verdict,
                    "notes": "",
                }
            )
    pd.DataFrame(sc_rows).to_csv(RESULTS / "GAP_CLOSURE_SCORECARD.csv", index=False)
    out.to_csv(RESULTS / "GAP_CLOSURE_EVAL_RAW.csv", index=False)
    print("wrote scorecard", len(sc_rows), flush=True)


if __name__ == "__main__":
    main()
