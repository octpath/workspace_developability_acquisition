#!/usr/bin/env python3
"""Screen structure marathon families (Ridge / PCA32) — Primary+Shadow."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SM = FP / "structure_marathon"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    OOF_DIR,
    TMAPP_REF,
    HIC_REF,
    mae,
    metrics,
    median_baseline_oof,
)

PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_N = 32
RNG = np.random.default_rng(42)


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
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(X.loc[tr]), y.loc[tr])
            maes.append(mae(y.loc[te], m.predict(sc.transform(X.loc[te]))))
        score = float(np.mean(maes)) if maes else np.inf
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def nested_ridge(X, y, folds, use_pca=False):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        med = X.loc[tr].median()
        Xtr = X.loc[tr].fillna(med)
        Xte = X.loc[te].fillna(med)
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(Xtr)
        Zte = sc0.transform(Xte)
        if use_pca and Ztr.shape[1] >= 2:
            pca = PCA(n_components=min(PCA_N, Ztr.shape[1], len(tr) - 1), random_state=42)
            Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        else:
            Ptr, Pte = Ztr, Zte
        a = select_alpha(pd.DataFrame(Ptr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Ptr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Pte))
    return oof


def plm_oof(X_plm, y, folds):
    return nested_ridge(X_plm, y, folds, use_pca=True)


def fuse(X_plm, X_phys, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X_phys.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc0 = StandardScaler()
        Ztr = sc0.fit_transform(X_plm.loc[tr])
        Zte = sc0.transform(X_plm.loc[te])
        pca = PCA(n_components=min(PCA_N, len(tr) - 1, Ztr.shape[1]), random_state=42)
        Ptr, Pte = pca.fit_transform(Ztr), pca.transform(Zte)
        med = X_phys.loc[tr].median()
        scp = StandardScaler()
        Xptr = scp.fit_transform(X_phys.loc[tr].fillna(med))
        Xpte = scp.transform(X_phys.loc[te].fillna(med))
        Xtr = np.hstack([Ptr, Xptr])
        Xte = np.hstack([Pte, Xpte])
        a = select_alpha(pd.DataFrame(Xtr, index=tr), y.loc[tr], folds, tr)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), y.loc[tr])
        oof.loc[te] = m.predict(sc.transform(Xte))
    return oof


def stack_inc(inc, phys_oof, y, folds):
    X = pd.DataFrame({"inc": inc, "phys": phys_oof}).dropna()
    common = X.index.intersection(y.index)
    return nested_ridge(X.loc[common], y.loc[common], folds[folds.id.isin(common)], use_pca=False)


def boot_delta(y, base, cand, B=None):
    import os
    if B is None:
        B=int(os.environ.get("SCREEN_BOOT", B_BOOT))
    idx = y.index.intersection(base.index).intersection(cand.index)
    y, base, cand = y.loc[idx], base.loc[idx], cand.loc[idx]
    d0 = mae(y, base) - mae(y, cand)
    boots = []
    n = len(y)
    for _ in range(B):
        ix = RNG.integers(0, n, n)
        boots.append(mae(y.iloc[ix], base.iloc[ix]) - mae(y.iloc[ix], cand.iloc[ix]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d0), float(lo), float(hi), float(np.mean(np.asarray(boots) > 0))


def load_y(target):
    ref = TMAPP_REF if target == "TmApp" else HIC_REF
    df = pd.read_csv(OOF_DIR / f"{ref}.csv").set_index("id")
    return df["y_true"].astype(float), df["y_pred"].astype(float)


def load_plm(target):
    z = np.load(EMB, allow_pickle=True)
    if target == "TmApp":
        return pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=[str(x) for x in z["ids"]])
    return pd.DataFrame(np.asarray(z["esm2__HL"], float), index=[str(x) for x in z["ids"]])


def eval_family(name, X, target, use_pca=False):
    y, inc = load_y(target)
    plm = load_plm(target)
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    rows = []
    for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
        common = X.index.intersection(y.index).intersection(plm.index).intersection(inc.index)
        folds = folds[folds.id.isin(common)]
        Xc, yc, plmc, incc = X.loc[common], y.loc[common], plm.loc[common], inc.loc[common]
        med, _ = median_baseline_oof(yc, folds)
        phys = nested_ridge(Xc, yc, folds, use_pca=use_pca)
        plm_hat = plm_oof(plmc, yc, folds)
        fused = fuse(plmc, Xc, yc, folds)
        stacked = stack_inc(incc, phys, yc, folds)
        for mode, base, cand in [
            ("STANDALONE_vs_median", med, phys),
            ("INC_vs_PLM", plm_hat, fused),
            ("INC_vs_INCUMBENT", incc, stacked),
        ]:
            d, lo, hi, pimp = boot_delta(yc, base, cand)
            mb, mc = metrics(yc, base), metrics(yc, cand)
            rows.append(
                {
                    "target": target,
                    "family": name,
                    "cv": tag,
                    "mode": mode,
                    "feature_dim": Xc.shape[1],
                    "screen_model": "PCA32_Ridge" if use_pca else "Ridge",
                    "base_MAE": mb["MAE"],
                    "cand_MAE": mc["MAE"],
                    "delta_MAE": d,
                    "boot_lo": lo,
                    "boot_hi": hi,
                    "p_improve": pimp,
                    "cand_Spearman": mc["Spearman"],
                    "n": len(yc),
                    "incumbent_id": TMAPP_REF if target == "TmApp" else HIC_REF,
                }
            )
    return rows


def load_X(path: Path, prefix_keep=None, max_dim=30):
    df = pd.read_csv(path).set_index("id")
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if prefix_keep:
        cols = [c for c in cols if any(c.startswith(p) for p in prefix_keep)]
    # drop pure counts if many dims
    if len(cols) > max_dim:
        cols = [c for c in cols if not c.endswith("_n") and not c.endswith("_n_edges")][:max_dim]
    return df[cols]


def main():
    families = []
    # S2
    p = SM / "generator_disagreement/S2_DISAGREEMENT_FEATURES.csv"
    if p.exists():
        families.append(("S2_disagreement", load_X(p, max_dim=20), False))
    # S3
    p = SM / "surface_patch_graph/S3_FEATURES.csv"
    if p.exists():
        families.append(("S3_surface_graph", load_X(p, max_dim=30), False))
    # S4
    p = SM / "contact_graph/S4_FEATURES.csv"
    if p.exists():
        families.append(("S4_contact_graph", load_X(p, max_dim=30), False))
    # S1 priority masks if available — high dim -> PCA
    p = SM / "structure_guided_pooling/S1_FEATURES.csv"
    if p.exists():
        df = pd.read_csv(p).set_index("id")
        for mask in [
            "STRONGLY_EXPOSED_AROMATIC",
            "EXPOSED_AROMATIC",
            "LARGEST_HYDROPHOBIC_SURFACE_PATCH",
            "PATCH_NEIGHBORS",
            "CDR_ALL",
            "HCDR3",
            "VH_VL_INTERFACE",
            "BURIED_CORE",
            "EXPOSED",
        ]:
            cols = [c for c in df.columns if c.startswith(f"S1_{mask}_") and not c.endswith("_n")
                    and c[len(f"S1_{mask}_"):].split("_")[0].isdigit()]
            if len(cols) >= 8:
                families.append((f"S1_{mask}", df[cols], True))

    # M1 ProteinMPNN
    p = SM / "proteinmpnn/M1_FEATURES.csv"
    if p.exists():
        families.append(("M1_ProteinMPNN_soluble", load_X(p, max_dim=25), False))
    # M2 ESM-IF1
    p = SM / "esm_if1/M2_ESMIF1_FEATURES.csv"
    if p.exists():
        df = pd.read_csv(p)
        ok = df[df.status.astype(str).str.startswith("SUCCESS")].set_index("id")
        cols = [c for c in ok.columns if c.startswith("IF1_")]
        if cols:
            families.append(("M2_ESM_IF1", ok[cols], False))
    # M3 SaProt — high-dim embeddings -> PCA32
    for cand in [SM / "saprot/M3_FEATURES.csv", SM / "saprot/M3_FEATURES_pilot.csv"]:
        if cand.exists():
            df = pd.read_csv(cand)
            ok = df[df.status.astype(str).str.startswith("SUCCESS")].set_index("id")
            for pref, label in [
                ("SaProt35_global_", "M3_SaProt35_global"),
                ("SaProt35_VH_", "M3_SaProt35_VH"),
                ("SaProt35_VL_", "M3_SaProt35_VL"),
            ]:
                cols = [c for c in ok.columns if c.startswith(pref)]
                if len(cols) >= 8:
                    families.append((label, ok[cols], True))
            break

    all_rows = []
    for name, X, use_pca in families:
        print("eval", name, X.shape, flush=True)
        for target in ["TmApp", "HIC"]:
            try:
                all_rows.extend(eval_family(name, X, target, use_pca=use_pca))
            except Exception as e:
                print("FAIL", name, target, e, flush=True)
    out = pd.DataFrame(all_rows)
    out_path = SM / "results/STRUCTURE_FAMILY_SCORECARD.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10:
        prev = pd.read_csv(out_path)
        prev = prev[~prev.family.isin(out.family)] if len(out) else prev
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print("wrote", out_path, len(out), flush=True)


if __name__ == "__main__":
    main()
