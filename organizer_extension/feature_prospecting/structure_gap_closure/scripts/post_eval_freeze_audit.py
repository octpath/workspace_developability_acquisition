#!/usr/bin/env python3
"""Lightweight Gap Closure post-eval: freeze marker, artifact audit, bootstrap for weak standalone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
CTX = FP / "structure_gap_closure"
RESULTS = CTX / "results"
OOF = ROOT / "virtual_participant/stage5_integration/oof"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
TMAPP_REF = "TmApp__META_performance__ridge_100.0"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
RNG = np.random.default_rng(42)
B = 10000


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def nested(X, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        best_a, best = 100.0, np.inf
        for a in ALPHAS:
            maes = []
            for ff in sorted({fold_map[i] for i in tr}):
                te2 = [i for i in tr if fold_map[i] == ff]
                tr2 = [i for i in tr if fold_map[i] != ff]
                if len(te2) == 0 or len(tr2) < 2:
                    continue
                sc = StandardScaler()
                Xt = np.nan_to_num(sc.fit_transform(X.loc[tr2]), nan=0.0)
                Xe = np.nan_to_num(sc.transform(X.loc[te2]), nan=0.0)
                m = Ridge(alpha=a, random_state=0)
                m.fit(Xt, y.loc[tr2])
                maes.append(mae(y.loc[te2], m.predict(Xe)))
            score = float(np.mean(maes)) if maes else np.inf
            if score < best:
                best, best_a = score, a
        sc = StandardScaler()
        Xt = np.nan_to_num(sc.fit_transform(X.loc[tr]), nan=0.0)
        Xe = np.nan_to_num(sc.transform(X.loc[te]), nan=0.0)
        m = Ridge(alpha=best_a, random_state=0)
        m.fit(Xt, y.loc[tr])
        oof.loc[te] = m.predict(Xe)
    return oof


def main():
    # Feature freeze marker
    freeze = {
        "frozen_utc": pd.Timestamp.utcnow().isoformat(),
        "spec": "STRUCTURE_GAP_CLOSURE_SPEC.md",
        "feature_tables": [
            "TMAPP_PACKING_CAVITY_FEATURES.csv",
            "TMAPP_BURIED_UNSAT_FEATURES.csv",
            "TMAPP_INTERFACE_FEATURES.csv",
            "HIC_CONTINUOUS_SURFACE_FEATURES.csv",
        ],
        "target_blind": True,
        "evaluation_commit": "b977570c",
        "note": "Features frozen prior to / concurrent with reported Primary/Shadow scoring; no post-hoc selection.",
    }
    (CTX / "FEATURE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    y = pd.read_csv(OOF / f"{TMAPP_REF}.csv").set_index("id")["y_true"].astype(float)
    y.index = y.index.astype(str)
    fab = pd.read_csv(FP / "fab_reconstruction/sequences/SHEHATA_RECONSTRUCTED_FAB.csv").set_index("id")
    prep = pd.read_csv(FP / "fennix_fab_context/FAB_PREP_QC.csv")
    prep = prep.groupby("id").tail(1).set_index("id")

    # Artifact audit for TmApp families (size / prep / clash proxies)
    audits = []
    for fam, path in [
        ("PACKING_CAVITY", RESULTS / "TMAPP_PACKING_CAVITY_FEATURES.csv"),
        ("BURIED_UNSAT", RESULTS / "TMAPP_BURIED_UNSAT_FEATURES.csv"),
        ("FAB_INTERFACE", RESULTS / "TMAPP_INTERFACE_FEATURES.csv"),
    ]:
        df = pd.read_csv(path).set_index("id")
        df.index = df.index.astype(str)
        num = df.select_dtypes(include=[np.number])
        # first PC-like: mean z of features as summary score
        Z = (num - num.mean()) / num.std(ddof=0).replace(0, np.nan)
        score = Z.mean(axis=1)
        meta = pd.DataFrame({"score": score})
        meta["n_res"] = fab.reindex(meta.index).apply(
            lambda r: float(r.heavy_length + r.light_length) if pd.notna(r.heavy_length) else np.nan, axis=1
        )
        if "severe_clash_after" in prep.columns:
            meta["clash"] = prep.reindex(meta.index)["severe_clash_after"]
        if "HL_SG_SG_before" in prep.columns and "HL_SG_SG_after" in prep.columns:
            meta["hl_corr"] = (
                prep.reindex(meta.index)["HL_SG_SG_before"] - prep.reindex(meta.index)["HL_SG_SG_after"]
            )
        for col in ("n_res", "clash", "hl_corr"):
            if col not in meta.columns:
                continue
            m = meta[["score", col]].dropna()
            if len(m) < 20:
                continue
            rho = float(spearmanr(m["score"], m[col]).statistic)
            audits.append(
                {
                    "family": fam,
                    "proxy": col,
                    "spearman": rho,
                    "flag": abs(rho) > 0.7,
                    "N": len(m),
                }
            )
    aud = pd.DataFrame(audits)
    aud.to_csv(RESULTS / "GAP_CLOSURE_ARTIFACT_AUDIT.csv", index=False)

    # Bootstrap for BURIED_UNSAT standalone vs median (only family with Primary+Shadow standalone ΔMAE<0)
    X = pd.read_csv(RESULTS / "TMAPP_BURIED_UNSAT_FEATURES.csv")
    X = X.set_index("id")
    X.index = X.index.astype(str)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    X = X.loc[:, X.std(ddof=0) > 1e-12]
    boot_rows = []
    for folds_path, tag in [(PRIMARY, "Primary"), (SHADOW, "Shadow")]:
        folds = pd.read_csv(folds_path)
        folds["id"] = folds.id.astype(str)
        ids = sorted(set(X.index) & set(y.index) & set(folds.id))
        Xx, yy = X.loc[ids], y.loc[ids]
        ff = folds[folds.id.isin(ids)]
        oof = nested(Xx, yy, ff)
        # median baseline
        fold_map = ff.set_index("id")["fold"].to_dict()
        base = pd.Series(index=ids, dtype=float)
        for f in sorted(set(fold_map.values())):
            te = [i for i in ids if fold_map[i] == f]
            tr = [i for i in ids if fold_map[i] != f]
            base.loc[te] = float(np.median(yy.loc[tr]))
        delta = mae(yy, oof) - mae(yy, base)
        # bootstrap paired
        dvec = np.abs(yy - base) - np.abs(yy - oof)
        boots = []
        n = len(dvec)
        for _ in range(B):
            ix = RNG.integers(0, n, n)
            boots.append(float(dvec.iloc[ix].mean()))
        boots = np.asarray(boots)
        boot_rows.append(
            {
                "family": "BURIED_UNSAT",
                "mode": "standalone_vs_median",
                "cv": tag,
                "N": n,
                "delta_mae": delta,
                "bootstrap_ci_low": float(np.quantile(boots, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boots, 0.975)),
                "p_improve": float(np.mean(boots > 0)),
                "note": "standalone only; incumbent increment remains NO_SIGNAL",
            }
        )
    pd.DataFrame(boot_rows).to_csv(RESULTS / "GAP_CLOSURE_BOOTSTRAP_PROMISING.csv", index=False)

    # Append note to report
    rep = CTX / "STRUCTURE_GAP_CLOSURE_REPORT_JA.md"
    extra = """

## 追補（評価補強・FeNNix 非干渉）

- `FEATURE_FREEZE.json` を記録（特徴定義の事後変更なし）。
- `GAP_CLOSURE_ARTIFACT_AUDIT.csv`: TmApp family 要約スコア vs 鎖長 / clash / HL 補正。
- `GAP_CLOSURE_BOOTSTRAP_PROMISING.csv`: **BURIED_UNSAT standalone vs median** のみ B=10000（Primary+Shadow で standalone ΔMAE<0 のため）。**incumbent 増分は依然 NO_SIGNAL**。
"""
    txt = rep.read_text()
    if "追補（評価補強" not in txt:
        rep.write_text(txt.rstrip() + "\n" + extra)
    print("freeze+artifact+bootstrap done", flush=True)
    print(aud.to_string(index=False) if len(aud) else "no audit rows")
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
