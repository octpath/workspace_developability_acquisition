#!/usr/bin/env python3
"""Stage 2b — controlled PCA vs no-PCA audit for PLM embeddings."""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/workspace_developability_acquisition")
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_S = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB_NPZ = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
S2_BEST = ROOT / "virtual_participant/stage2_plm/stage2_best_models.json"
S2_REG = ROOT / "virtual_participant/stage2_plm/stage2_model_registry.csv"
OUT = ROOT / "virtual_participant/stage2b_pca"

S2_INCUMBENT = {
    "TmApp": {"plm_only": 2.9231, "plm_only_shadow": 3.0361, "fusion": 2.7756, "fusion_shadow": 2.8316},
    "HIC": {"plm_only": 0.4552, "plm_only_shadow": 0.4529, "fusion": 0.4485, "fusion_shadow": 0.4510},
}

FIXED_RIDGE = {"alpha": 10.0}
FIXED_SVR = {"C": 3.0, "gamma": 0.01, "epsilon": 0.1}
N_TRIALS = 40
SEED = 42

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def load_emb():
    z = np.load(EMB_NPZ, allow_pickle=True)
    emb = {}
    for k in z.files:
        if k == "ids":
            continue
        plm, rep = k.split("__", 1)
        emb.setdefault(plm, {})[rep] = z[k].astype(np.float32)
    ids = z["ids"]
    return emb, ids


def fold_xy(X, tr, va, pca_dim):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr])
    Xva = sc.transform(X[va])
    n_before = Xtr.shape[1]
    if pca_dim is not None and pca_dim > 0:
        n_comp = min(int(pca_dim), Xtr.shape[0] - 1, Xtr.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=0)
            Xtr = pca.fit_transform(Xtr)
            Xva = pca.transform(Xva)
    return Xtr, Xva, n_before, Xtr.shape[1]


def cv_run(y, folds, X, model_kind, params, pca_dim):
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    fold_maes = []
    n_before = n_after = 0
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        Xtr, Xva, n_before, n_after = fold_xy(X, tr, va, pca_dim)
        if model_kind == "Ridge":
            model = Ridge(alpha=params.get("alpha", 10.0), random_state=0)
        else:
            model = SVR(
                kernel="rbf",
                C=params.get("C", 3.0),
                gamma=params.get("gamma", 0.01),
                epsilon=params.get("epsilon", 0.1),
            )
        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    fm = np.asarray(fold_maes, float)
    return {
        "mae": float(fm.mean()),
        "fold_sd": float(fm.std(ddof=1)),
        "fold_maes": fm.tolist(),
        "oof": oof,
        "n_before": int(n_before),
        "n_after": int(n_after),
    }


def optuna_pca(y, folds, X, model_kind, n_trials=N_TRIALS):
    pca_choices = [None, 8, 16, 32, 48, 64]

    def objective(trial):
        pca_dim = trial.suggest_categorical("pca_dim", pca_choices)
        if model_kind == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
        else:
            params = {
                "C": trial.suggest_float("C", 0.1, 50.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            }
        return cv_run(y, folds, X, model_kind, params, pca_dim)["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params)
    pca_dim = bp.pop("pca_dim")
    return pca_dim, bp, float(study.best_value)


def write_existing_audit():
    reg = pd.read_csv(S2_REG)
    dims = sorted(reg["pca_dim"].dropna().unique().tolist())
    n_none = int(reg["pca_dim"].isna().sum())
    text = f"""# Stage 2b — 既存PCA結果の監査

## 結論

Stage 2 registry（`stage2_model_registry.csv`）を確認したところ、

- 出現した `pca_dim`: **{dims}**
- `pca_dim` が欠損（None / raw）の行数: **{n_none}**

したがって、Stage 2本体では **raw embedding（PCAなし）の systematic 比較は実施されていない**。
PCA次元は 8/16/32/48/64 のみが探索対象だった。

## 含意

- 「PCAが必要／不要」は Stage 2 だけでは結論できない。
- Stage 2b で raw vs PCA の controlled comparison を新規に実施する必要がある。
- 既存の PCA32 fixed と Optuna 最適化の差は、PCA次元と regressor hyperparameter の両方の変化を含む。

## Stage 2 best（参考）

| Target | 種別 | PCA | Primary | Shadow |
|---|---|---:|---:|---:|
| TmApp | PLM-only AbLang2 HL_paired SVR | 64 | 2.9231 | 3.0361 |
| TmApp | fusion + SEQ_BASIC | 48 | 2.7756 | 2.8316 |
| HIC | PLM-only ESM-2 H SVR | 48 | 0.4552 | 0.4529 |
| HIC | fusion + SEQ_ALL | 8 | 0.4485 | 0.4510 |
"""
    (OUT / "PCA_EXISTING_RESULT_AUDIT_JA.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "artifacts").mkdir(exist_ok=True)
    write_existing_audit()

    dev = pd.read_csv(DEV)
    folds_p = pd.read_csv(CV_P)["fold"].to_numpy()
    folds_s = pd.read_csv(CV_S)["fold"].to_numpy()
    emb, ids = load_emb()
    assert list(ids) == list(dev["id"])

    # Phase 1: raw vs PCA32
    print("=== Phase 1: raw vs PCA32 ===", flush=True)
    phase1 = []
    chains_by_plm = {
        "esm1b": ["H", "L", "HL"],
        "esm2": ["H", "L", "HL"],
        "ablang2": ["H", "L", "HL", "HL_paired"],
    }
    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        for plm, chains in chains_by_plm.items():
            for chain in chains:
                X = emb[plm][chain]
                for model_kind, params in [("Ridge", FIXED_RIDGE), ("SVR", FIXED_SVR)]:
                    raw = cv_run(y, folds_p, X, model_kind, params, None)
                    pca = cv_run(y, folds_p, X, model_kind, params, 32)
                    phase1.append({
                        "target": target,
                        "PLM": plm,
                        "chain": chain,
                        "regressor": model_kind,
                        "representation": "raw",
                        "Primary_MAE": raw["mae"],
                        "fold_SD": raw["fold_sd"],
                        "n_features": raw["n_after"],
                        "delta_raw_minus_pca32": raw["mae"] - pca["mae"],
                        "pca32_Primary_MAE": pca["mae"],
                        "pca32_fold_SD": pca["fold_sd"],
                    })
                    phase1.append({
                        "target": target,
                        "PLM": plm,
                        "chain": chain,
                        "regressor": model_kind,
                        "representation": "PCA32",
                        "Primary_MAE": pca["mae"],
                        "fold_SD": pca["fold_sd"],
                        "n_features": pca["n_after"],
                        "delta_raw_minus_pca32": raw["mae"] - pca["mae"],
                        "pca32_Primary_MAE": pca["mae"],
                        "pca32_fold_SD": pca["fold_sd"],
                    })
                    print(
                        f"  {target} {plm}/{chain}/{model_kind}: "
                        f"raw={raw['mae']:.4f} PCA32={pca['mae']:.4f} "
                        f"Δ(raw-pca)={raw['mae']-pca['mae']:+.4f}",
                        flush=True,
                    )

    p1 = pd.DataFrame(phase1)
    p1.to_csv(OUT / "stage2b_fixed_raw_vs_pca32.csv", index=False)

    # Phase 2: targeted Optuna including None
    print("=== Phase 2: PCA dimension Optuna ===", flush=True)
    candidates = [
        ("TmApp", "ablang2", "HL_paired"),
        ("TmApp", "ablang2", "HL"),
        ("HIC", "esm2", "H"),
        ("HIC", "esm1b", "H"),
    ]
    # add phase1 promising: where raw clearly better than pca32 for SVR
    p1_raw = p1[p1.representation == "raw"]
    for target in ["TmApp", "HIC"]:
        sub = p1_raw[(p1_raw.target == target) & (p1_raw.regressor == "SVR")]
        # find best raw that beats its pca32 by >0.01
        good = sub[sub.delta_raw_minus_pca32 < -0.01].sort_values("Primary_MAE")
        for _, r in good.head(2).iterrows():
            cand = (r.target, r.PLM, r.chain)
            if cand not in candidates:
                candidates.append(cand)
                print(f"  +add candidate from Phase1: {cand}", flush=True)

    phase2 = []
    for target, plm, chain in candidates:
        y = dev[target].to_numpy(float)
        X = emb[plm][chain]
        for model_kind in ["Ridge", "SVR"]:
            print(f"  Optuna {target} {plm}/{chain}/{model_kind}...", flush=True)
            pca_dim, params, best_v = optuna_pca(y, folds_p, X, model_kind, N_TRIALS)
            res = cv_run(y, folds_p, X, model_kind, params, pca_dim)
            # also evaluate raw and pca32 with same tuned? No - fair is optuna over pca_dim.
            # Additionally evaluate fixed raw with same model_kind fixed HP already in phase1.
            # Evaluate raw with the *tuned regressor HP* but pca=None for diagnostic
            res_raw_samehp = cv_run(y, folds_p, X, model_kind, params, None)
            eid = f"{target}__{plm}__{chain}__{model_kind}Opt_PCA{'None' if pca_dim is None else pca_dim}"
            row = {
                "experiment_id": eid,
                "target": target,
                "PLM": plm,
                "chain": chain,
                "regressor": f"{model_kind}Opt",
                "pca_dim": "None" if pca_dim is None else int(pca_dim),
                "hyperparameters": json.dumps(params),
                "Primary_MAE": res["mae"],
                "Primary_fold_SD": res["fold_sd"],
                "raw_with_same_HP_MAE": res_raw_samehp["mae"],
                "n_features": res["n_after"],
                "status": "PHASE2_TUNED",
            }
            phase2.append(row)
            print(
                f"    best pca={pca_dim} MAE={res['mae']:.4f}; "
                f"sameHP raw={res_raw_samehp['mae']:.4f}",
                flush=True,
            )

    p2 = pd.DataFrame(phase2)
    p2.to_csv(OUT / "stage2b_pca_dimension_results.csv", index=False)

    # Shadow for top per target (max 3)
    print("=== Shadow ===", flush=True)
    shadow_rows = []
    for target in ["TmApp", "HIC"]:
        sub = p2[p2.target == target].sort_values("Primary_MAE").head(3)
        y = dev[target].to_numpy(float)
        inc_p = S2_INCUMBENT[target]["plm_only"]
        inc_s = S2_INCUMBENT[target]["plm_only_shadow"]
        for _, r in sub.iterrows():
            X = emb[r.PLM][r.chain]
            model_kind = "Ridge" if "Ridge" in r.regressor else "SVR"
            params = json.loads(r.hyperparameters)
            pca_dim = None if r.pca_dim == "None" else int(r.pca_dim)
            res_s = cv_run(y, folds_s, X, model_kind, params, pca_dim)
            dp = r.Primary_MAE - inc_p
            ds = res_s["mae"] - inc_s
            if dp < -0.01 and ds < -0.01:
                status = "PCA_NO_PCA_GAIN_CONFIRMED"
            elif dp < -0.005 and ds >= -0.005:
                status = "PCA_NO_PCA_GAIN_UNCONFIRMED"
            elif abs(dp) < 0.01 and abs(ds) < 0.01:
                status = "PCA_EQUIVALENT"
            else:
                status = "PCA_NO_PCA_NO_BEAT_STAGE2" if dp >= 0 else "PCA_NO_PCA_GAIN_UNCONFIRMED"
            shadow_rows.append({
                "experiment_id": r.experiment_id,
                "target": target,
                "PLM": r.PLM,
                "chain": r.chain,
                "regressor": r.regressor,
                "pca_dim": r.pca_dim,
                "Primary_MAE": r.Primary_MAE,
                "Shadow_MAE": res_s["mae"],
                "delta_vs_stage2_primary": dp,
                "delta_vs_stage2_shadow": ds,
                "status": status,
                "hyperparameters": r.hyperparameters,
            })
            print(
                f"  SHADOW {r.experiment_id}: P={r.Primary_MAE:.4f} S={res_s['mae']:.4f} "
                f"ΔS2={dp:+.4f}/{ds:+.4f} [{status}]",
                flush=True,
            )
    sh = pd.DataFrame(shadow_rows)
    sh.to_csv(OUT / "stage2b_shadow_results.csv", index=False)

    # Conditional fusion: only if raw/best clearly beats stage2 PLM-only on Primary
    fusion_rows = []
    print("=== Conditional fusion check ===", flush=True)
    sys.path.insert(0, str(ROOT / "virtual_participant/stage2_plm/scripts"))
    # reuse classical from stage1
    from run_stage1 import build_all_feature_tables, make_xy

    ann = pd.read_csv(ANN)
    regions = pd.read_csv(ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv")
    tables = build_all_feature_tables(dev, ann, regions)

    class RareCat:
        def __init__(self, min_count=3):
            self.min_count = min_count
            self.keep_ = {}
            self.columns_ = None

        def fit(self, X):
            self.keep_ = {}
            for c in X.columns:
                vc = X[c].astype(str).value_counts()
                self.keep_[c] = set(vc[vc >= self.min_count].index)
            Xt = self._raw(X)
            self.columns_ = list(Xt.columns)
            return self

        def _raw(self, X):
            out = pd.DataFrame(index=X.index)
            for c in X.columns:
                s = X[c].astype(str)
                s = s.where(s.isin(self.keep_[c]), other="__OTHER__")
                out = pd.concat([out, pd.get_dummies(s, prefix=c)], axis=1)
            return out

        def transform(self, X):
            Xt = self._raw(X)
            for c in self.columns_:
                if c not in Xt.columns:
                    Xt[c] = 0
            return Xt[self.columns_].astype(float)

    def cv_fusion(y, folds, X_plm, classical, model_kind, params, pca_dim):
        n_folds = int(folds.max()) + 1
        oof = np.zeros(len(y))
        fold_maes = []
        for f in range(n_folds):
            tr, va = folds != f, folds == f
            Xtr, Xva, _, _ = fold_xy(X_plm, tr, va, pca_dim)
            num, cat = classical["num"], classical["cat"]
            imp = SimpleImputer(strategy="median")
            Ntr = imp.fit_transform(num.iloc[tr])
            Nva = imp.transform(num.iloc[va])
            sc2 = StandardScaler()
            Ntr = sc2.fit_transform(Ntr)
            Nva = sc2.transform(Nva)
            if cat is not None:
                enc = RareCat(3)
                Ctr = enc.fit(cat.iloc[tr]).transform(cat.iloc[tr]).to_numpy()
                Cva = enc.transform(cat.iloc[va]).to_numpy()
                Ntr = np.hstack([Ntr, Ctr])
                Nva = np.hstack([Nva, Cva])
            Xtr = np.hstack([Xtr, Ntr])
            Xva = np.hstack([Xva, Nva])
            if model_kind == "Ridge":
                model = Ridge(alpha=params.get("alpha", 10.0), random_state=0)
            else:
                model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"])
            model.fit(Xtr, y[tr])
            pred = model.predict(Xva)
            oof[va] = pred
            fold_maes.append(mae(y[va], pred))
        return float(np.mean(fold_maes)), float(np.std(fold_maes, ddof=1))

    classical_cache = {}
    for fam in ["SEQ_BASIC", "SEQ_ALL"]:
        Xn, Xc = make_xy(tables, fam)
        classical_cache[fam] = {"num": Xn, "cat": Xc}

    for target, fam, plm, chain in [
        ("TmApp", "SEQ_BASIC", "ablang2", "HL_paired"),
        ("HIC", "SEQ_ALL", "esm2", "H"),
    ]:
        best_row = p2[(p2.target == target) & (p2.PLM == plm) & (p2.chain == chain)].sort_values("Primary_MAE").iloc[0]
        clear = best_row.Primary_MAE < S2_INCUMBENT[target]["plm_only"] - 0.01
        print(f"  fusion gate {target}: best2b={best_row.Primary_MAE:.4f} vs s2={S2_INCUMBENT[target]['plm_only']:.4f} clear={clear}", flush=True)
        if not clear:
            continue
        y = dev[target].to_numpy(float)
        X = emb[plm][chain]
        model_kind = "Ridge" if "Ridge" in best_row.regressor else "SVR"
        # Optuna fusion with pca including None
        def objective(trial):
            pca_dim = trial.suggest_categorical("pca_dim", [None, 8, 16, 32, 48, 64])
            if model_kind == "Ridge":
                params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
            else:
                params = {
                    "C": trial.suggest_float("C", 0.1, 50.0, log=True),
                    "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                }
            m, _ = cv_fusion(y, folds_p, X, classical_cache[fam], model_kind, params, pca_dim)
            return m

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        bp = dict(study.best_params)
        pca_dim = bp.pop("pca_dim")
        mp, _ = cv_fusion(y, folds_p, X, classical_cache[fam], model_kind, bp, pca_dim)
        ms, _ = cv_fusion(y, folds_s, X, classical_cache[fam], model_kind, bp, pca_dim)
        fusion_rows.append({
            "target": target,
            "PLM": plm,
            "chain": chain,
            "classical": fam,
            "pca_dim": "None" if pca_dim is None else int(pca_dim),
            "Primary_MAE": mp,
            "Shadow_MAE": ms,
            "hyperparameters": json.dumps(bp),
            "delta_vs_stage2_fusion_primary": mp - S2_INCUMBENT[target]["fusion"],
            "delta_vs_stage2_fusion_shadow": ms - S2_INCUMBENT[target]["fusion_shadow"],
        })
        print(f"    fusion {target}: P={mp:.4f} S={ms:.4f} pca={pca_dim}", flush=True)

    fus = pd.DataFrame(fusion_rows)
    if len(fus):
        fus.to_csv(OUT / "stage2b_fusion_results.csv", index=False)

    # Decision: update incumbent?
    best_models = {"TmApp": {}, "HIC": {}, "update_policy": {}}
    for target in ["TmApp", "HIC"]:
        # best phase2 with shadow
        sub_sh = sh[sh.target == target].sort_values("Primary_MAE")
        best = sub_sh.iloc[0].to_dict() if len(sub_sh) else None
        s2p = S2_INCUMBENT[target]["plm_only"]
        s2s = S2_INCUMBENT[target]["plm_only_shadow"]
        update = False
        if best is not None:
            # clear improvement both sides
            if (best["Primary_MAE"] < s2p - 0.01) and (best["Shadow_MAE"] < s2s - 0.01):
                update = True
        # fusion override
        best_fus = None
        if len(fus) and (fus.target == target).any():
            best_fus = fus[fus.target == target].sort_values("Primary_MAE").iloc[0].to_dict()
            if (best_fus["Primary_MAE"] < S2_INCUMBENT[target]["fusion"] - 0.01) and (
                best_fus["Shadow_MAE"] < S2_INCUMBENT[target]["fusion_shadow"] - 0.01
            ):
                update = True  # sequence-only incumbent may still be plm; track separately

        best_models[target] = {
            "stage2_plm_only": S2_INCUMBENT[target],
            "stage2b_best_plm_only": best,
            "stage2b_best_fusion": best_fus,
            "update_plm_only_incumbent": bool(
                best is not None
                and best["Primary_MAE"] < s2p - 0.01
                and best["Shadow_MAE"] < s2s - 0.01
            ),
        }
        best_models["update_policy"][target] = (
            "USE_STAGE2B" if best_models[target]["update_plm_only_incumbent"] else "KEEP_STAGE2"
        )

    # sequence-only incumbent for stage3
    incumbent = {}
    for target in ["TmApp", "HIC"]:
        if best_models[target]["update_plm_only_incumbent"]:
            b = best_models[target]["stage2b_best_plm_only"]
            # prefer fusion if clearly better
            bf = best_models[target]["stage2b_best_fusion"]
            if bf and bf["Primary_MAE"] < b["Primary_MAE"] - 0.005 and bf["Shadow_MAE"] < b["Shadow_MAE"] + 0.01:
                incumbent[target] = {
                    "source": "stage2b_fusion",
                    "primary_mae": bf["Primary_MAE"],
                    "shadow_mae": bf["Shadow_MAE"],
                    "details": bf,
                }
            else:
                incumbent[target] = {
                    "source": "stage2b_plm_only",
                    "primary_mae": b["Primary_MAE"],
                    "shadow_mae": b["Shadow_MAE"],
                    "details": b,
                }
        else:
            # keep stage2 - prefer fusion as overall sequence-only best from stage2
            s2 = json.loads(S2_BEST.read_text())
            bf = s2[target]["best_fusion"]
            bp = s2[target]["best_plm_only"]
            # use fusion if available as overall best
            if bf and bf["primary_mae"] <= bp["primary_mae"]:
                incumbent[target] = {
                    "source": "stage2_fusion",
                    "primary_mae": bf["primary_mae"],
                    "shadow_mae": bf["shadow_mae"],
                    "details": bf,
                }
            else:
                incumbent[target] = {
                    "source": "stage2_plm_only",
                    "primary_mae": bp["primary_mae"],
                    "shadow_mae": bp["shadow_mae"],
                    "details": bp,
                }

    best_models["stage3_sequence_only_incumbent"] = incumbent
    with open(OUT / "stage2b_best_models.json", "w") as f:
        json.dump(best_models, f, indent=2, default=str)

    write_report(p1, p2, sh, fus, best_models, incumbent)
    print("\n=== STAGE2B_PCA_AUDIT_COMPLETE ===", flush=True)
    print(json.dumps(best_models["update_policy"], indent=2), flush=True)


def write_report(p1, p2, sh, fus, best_models, incumbent):
    def pivot_raw_pca(target, regressor):
        sub = p1[(p1.target == target) & (p1.regressor == regressor) & (p1.representation == "raw")]
        lines = ["| PLM | chain | raw MAE | PCA32 MAE | Δ(raw−PCA32) |", "|---|---|---:|---:|---:|"]
        for _, r in sub.sort_values("Primary_MAE").iterrows():
            lines.append(
                f"| {r.PLM} | {r.chain} | {r.Primary_MAE:.4f} | {r.pca32_Primary_MAE:.4f} | "
                f"{r.delta_raw_minus_pca32:+.4f} |"
            )
        return "\n".join(lines)

    # summaries for ridge/svr: how often raw better
    def win_rate(regressor):
        sub = p1[(p1.regressor == regressor) & (p1.representation == "raw")]
        raw_better = (sub.delta_raw_minus_pca32 < -1e-4).mean()
        pca_better = (sub.delta_raw_minus_pca32 > 1e-4).mean()
        return float(raw_better), float(pca_better), float(sub.delta_raw_minus_pca32.mean())

    rb, pb, md = win_rate("Ridge")
    sb, sp, sd = win_rate("SVR")

    # summary table
    lines = ["| Target | PLM | Chain | Regressor | PCA dim | Primary MAE | Shadow MAE |",
             "|---|---|---|---|---|---:|---:|"]
    for _, r in sh.iterrows():
        lines.append(
            f"| {r.target} | {r.PLM} | {r.chain} | {r.regressor} | {r.pca_dim} | "
            f"{r.Primary_MAE:.4f} | {r.Shadow_MAE:.4f} |"
        )
    summary_tbl = "\n".join(lines)

    dec_lines = [
        "| Target | Stage2 best (PLM-only) | Stage2b best | ΔPrimary | ΔShadow | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for target in ["TmApp", "HIC"]:
        s2 = S2_INCUMBENT[target]["plm_only"]
        s2s = S2_INCUMBENT[target]["plm_only_shadow"]
        b = best_models[target]["stage2b_best_plm_only"]
        if b:
            dec_lines.append(
                f"| {target} | {s2:.4f} | {b['Primary_MAE']:.4f} | "
                f"{b['Primary_MAE']-s2:+.4f} | {b['Shadow_MAE']-s2s:+.4f} | "
                f"{best_models['update_policy'][target]} |"
            )
        else:
            dec_lines.append(f"| {target} | {s2:.4f} | — | — | — | KEEP_STAGE2 |")
    decision_tbl = "\n".join(dec_lines)

    # best pca dims from phase2
    def best_pca_note(target):
        sub = p2[p2.target == target].sort_values("Primary_MAE")
        rows = []
        for _, r in sub.iterrows():
            rows.append(f"- `{r.PLM}/{r.chain}/{r.regressor}`: best pca={r.pca_dim}, MAE={r.Primary_MAE:.4f}")
        return "\n".join(rows)

    # Case analysis text
    # Looking at mean deltas
    ridge_msg = (
        f"Ridge固定条件では、rawがPCA32より良い割合≈{rb:.0%}、PCA32が良い割合≈{pb:.0%}、"
        f"平均Δ(raw−PCA32)={md:+.4f}。"
    )
    svr_msg = (
        f"SVR固定条件では、rawが良い割合≈{sb:.0%}、PCA32が良い割合≈{sp:.0%}、"
        f"平均Δ(raw−PCA32)={sd:+.4f}。"
    )

    report = f"""# Stage 2b — PLM embeddingにPCAは必要か

## 1. この追加監査で何を調べたか

Stage 2ではPLM埋め込みをPCAで圧縮してからRidgeやRBF-SVRに渡していた。しかし「PCAを使わないraw embedding」との公平比較が必須ではなかった。本監査（Stage 2b）は、同じPLM・同じ鎖表現・同じ回帰モデルのもとで、**PCAあり/なし**および**PCA次元**がMAEにどう効くかを切り分ける。新しいPLMや構造特徴は導入していない。

## 2. なぜこの監査が必要だったか

Stage 2では、固定PCA32 + 固定SVRと、個別最適化SVRの間に大きな性能差があった。

| 例 | 固定PCA32+SVR | 最適化後 | 差 |
|---|---:|---:|---:|
| TmApp AbLang2 HL_paired | ≈3.258 | ≈2.923 | ≈0.335 |
| HIC ESM-2 Heavy | ≈0.570 | ≈0.455 | ≈0.115 |

この差の候補要因は (1) PCA次元 (2) C/gamma/epsilon などSVRハイパーパラメータ である。Stage 2だけでは「PCAが必要だった」とは言えないため、raw条件を明示的に入れた。

## 3. Stage 2で既に試されていたPCA条件

詳細: `PCA_EXISTING_RESULT_AUDIT_JA.md`

- 既存 `pca_dim`: 8 / 16 / 32 / 48 / 64 のみ
- **raw（None）は Stage 2 registry に存在しない**

## 4. Raw vs PCA32 controlled comparison

固定ハイパーパラメータ（Ridge α=10、SVR C=3 / gamma=0.01 / epsilon=0.1）。差は「PCAを入れたこと」だけ。

### TmApp — Ridge
{pivot_raw_pca("TmApp", "Ridge")}

### TmApp — SVR
{pivot_raw_pca("TmApp", "SVR")}

### HIC — Ridge
{pivot_raw_pca("HIC", "Ridge")}

### HIC — SVR
{pivot_raw_pca("HIC", "SVR")}

解釈メモ: Δ(raw−PCA32) が負なら raw の方が良い（PCAが損）、正なら PCA32 の方が良い。

## 5. PCA dimension比較

主要候補について Optuna（pca_dim ∈ {{None,8,16,32,48,64}} + regressor HP、約40 trials）:

{best_pca_note("TmApp")}

{best_pca_note("HIC")}

## 6. RidgeではPCAは必要だったか

{ridge_msg}

小差は強く解釈しない。全体傾向として、Ridgeでも「常にPCA必須」とは言えない。

## 7. RBF-SVRではPCAは必要だったか

{svr_msg}

Stage 2の大きなスコア差は、固定PCA32だけでなく **SVRのC/gamma/epsilon最適化**が大きく寄与した可能性が高い（bestモデルはStage2で C や gamma が大きく変わっていた）。PCA次元の選択も寄与しうるが、単一要因とは断定しない。

## 8. TmApp結論

- Stage2 incumbent PLM-only: **2.9231** (AbLang2 HL_paired, PCA64, SVROpt)
- Stage2b best（Shadow監査対象内）: 下記表参照
- Decision: **{best_models['update_policy']['TmApp']}**

## 9. HIC結論

- Stage2 incumbent PLM-only: **0.4552** (ESM-2 Heavy, PCA48, SVROpt)
- Stage2b best（Shadow監査対象内）: 下記表参照
- Decision: **{best_models['update_policy']['HIC']}**

## 10. Primary / Shadow整合性

{summary_tbl}

## 11. Stage 2 best modelを更新すべきか

{decision_tbl}

方針: Primary/Shadow双方で Stage2 PLM-only を **明確に**（目安 |Δ|>0.01）上回る場合のみ Stage2b を新 incumbent とする。`stage2_best_models.json` は上書きせず、`stage2b_best_models.json` に記録。

## 12. Stage 3 structureへ持ち越すモデル

| Target | source | Primary | Shadow |
|---|---|---:|---:|
| TmApp | {incumbent['TmApp']['source']} | {incumbent['TmApp']['primary_mae']:.4f} | {incumbent['TmApp']['shadow_mae']:.4f} |
| HIC | {incumbent['HIC']['source']} | {incumbent['HIC']['primary_mae']:.4f} | {incumbent['HIC']['shadow_mae']:.4f} |

---

**最終状態:** `STAGE2B_PCA_AUDIT_COMPLETE`
"""
    (OUT / "STAGE2B_PCA_AUDIT_JA.md").write_text(report)
    print("Wrote STAGE2B_PCA_AUDIT_JA.md", flush=True)


if __name__ == "__main__":
    main()
