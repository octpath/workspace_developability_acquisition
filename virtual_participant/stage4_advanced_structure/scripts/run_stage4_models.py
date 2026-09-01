#!/usr/bin/env python3
"""Stage 4 modeling: advanced-only + fusion vs incumbents. Primary Optuna, Shadow audit."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "virtual_participant/stage4_advanced_structure"
FEAT = OUT / "cache/features/stage4_all_features.csv"
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_S = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
S3_FEAT = ROOT / "virtual_participant/stage3_structure/cache/features_ESMFold.csv"
OOF = OUT / "oof"
PLOTS = OUT / "plots"
SEED = 42
N_TRIALS = 30
HIC_TAIL = 10.5372

INC = {
    "TmApp": {
        "plm_only": {"primary": 2.8634, "shadow": 2.9803},
        "seq_overall": {"primary": 2.7756, "shadow": 2.8316},
        "stage3_prov": {"primary": 2.7539, "shadow": 2.8202, "name": "overall+RASA"},
    },
    "HIC": {
        "plm_only": {"primary": 0.4552, "shadow": 0.4529},
        "seq_overall": {"primary": 0.4485, "shadow": 0.4510},
        "stage3_prov": {"primary": 0.4405, "shadow": 0.4384, "name": "overall+SURFACE_ALL"},
    },
}

FAMILIES = {
    "ADV_PROPKA": lambda c: c.startswith("propka"),
    "ADV_PQR_CHARGE": lambda c: c.startswith("pqr"),
    "ADV_ELECTROSTATICS": lambda c: c.startswith("apbs_") and c != "apbs_ok" and "error" not in c,
    "ADV_SURFACE_PATCH": lambda c: c.startswith("adv_"),
    "ADV_INTERACTIONS": lambda c: any(
        x in c
        for x in [
            "salt_bridge",
            "hbond",
            "clash",
            "contact",
            "packing",
            "interface",
            "vh_vl",
            "buried_frac",
            "core_hydrophobic",
            "heavy_contact",
        ]
    ),
    "ADV_CAVITY": lambda c: "cavity" in c,
    "ADV_UNSAT_POLAR": lambda c: "unsatisfied" in c,
    "ADV_INVFOLD": lambda c: c.startswith("invfold_") and c != "invfold_ok" and "error" not in c,
}

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage2_plm/scripts"))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


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


def make_model(kind, params):
    if kind == "Ridge":
        return Ridge(alpha=params.get("alpha", 10.0), random_state=0)
    if kind == "ElasticNet":
        return ElasticNet(
            alpha=params.get("alpha", 0.05),
            l1_ratio=params.get("l1_ratio", 0.3),
            max_iter=8000,
            tol=1e-3,
            random_state=0,
        )
    return SVR(kernel="rbf", C=params.get("C", 1.0), gamma=params.get("gamma", 0.01), epsilon=params.get("epsilon", 0.1))


def fold_prepare(X_adv, X_plm, classical, s3, tr, va, pca_dim):
    parts_tr, parts_va = [], []
    n = 0
    for X in [X_adv, s3]:
        if X is None:
            continue
        imp = SimpleImputer(strategy="median")
        Atr = imp.fit_transform(X[tr])
        Ava = imp.transform(X[va])
        sc = StandardScaler()
        Atr = sc.fit_transform(Atr)
        Ava = sc.transform(Ava)
        parts_tr.append(Atr)
        parts_va.append(Ava)
        n += Atr.shape[1]
    if X_plm is not None:
        sc = StandardScaler()
        Ptr = sc.fit_transform(X_plm[tr])
        Pva = sc.transform(X_plm[va])
        if pca_dim is not None and pca_dim > 0:
            n_comp = min(int(pca_dim), Ptr.shape[0] - 1, Ptr.shape[1])
            if n_comp >= 2:
                pca = PCA(n_components=n_comp, random_state=0)
                Ptr = pca.fit_transform(Ptr)
                Pva = pca.transform(Pva)
        parts_tr.append(Ptr)
        parts_va.append(Pva)
        n += Ptr.shape[1]
    if classical is not None:
        imp = SimpleImputer(strategy="median")
        Ntr = imp.fit_transform(classical["num"].iloc[tr])
        Nva = imp.transform(classical["num"].iloc[va])
        sc = StandardScaler()
        Ntr = sc.fit_transform(Ntr)
        Nva = sc.transform(Nva)
        if classical["cat"] is not None:
            enc = RareCat(3)
            Ctr = enc.fit(classical["cat"].iloc[tr]).transform(classical["cat"].iloc[tr]).to_numpy()
            Cva = enc.transform(classical["cat"].iloc[va]).to_numpy()
            Ntr = np.hstack([Ntr, Ctr])
            Nva = np.hstack([Nva, Cva])
        parts_tr.append(Ntr)
        parts_va.append(Nva)
        n += Ntr.shape[1]
    return np.hstack(parts_tr), np.hstack(parts_va), n


def cv_run(y, folds, X_adv, X_plm, classical, s3, kind, params, pca_dim, save=None, ids=None):
    oof = np.zeros(len(y))
    fold_maes = []
    nfeat = 0
    for f in range(int(folds.max()) + 1):
        tr, va = folds != f, folds == f
        Xtr, Xva, nfeat = fold_prepare(X_adv, X_plm, classical, s3, tr, va, pca_dim)
        m = make_model(kind, params)
        m.fit(Xtr, y[tr])
        pred = m.predict(Xva)
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    fold_maes = np.asarray(fold_maes, float)
    if save is not None:
        pd.DataFrame({"id": ids, "fold": folds, "y_true": y, "y_pred": oof, "residual": y - oof, "experiment_id": Path(save).stem}).to_csv(
            save, index=False
        )
    return {"mae": float(fold_maes.mean()), "fold_sd": float(fold_maes.std(ddof=1)), "oof": oof, "n_features": int(nfeat)}


def optuna_search(y, folds, X_adv, X_plm, classical, s3, kind, pca_choices, n_trials=N_TRIALS):
    def obj(trial):
        pca_dim = trial.suggest_categorical("pca_dim", pca_choices)
        if kind == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
        elif kind == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }
        else:
            params = {
                "C": trial.suggest_float("C", 0.1, 50.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            }
        return cv_run(y, folds, X_adv, X_plm, classical, s3, kind, params, pca_dim)["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params)
    pca_dim = bp.pop("pca_dim")
    return pca_dim, bp


def cols_for(df, fam):
    pred = FAMILIES[fam]
    cols = [c for c in df.columns if c != "antibody_id" and pred(c)]
    # drop non-numeric
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def status_vs(base_p, base_s, p, s, eps=1e-6):
    gain_p = p < base_p - eps
    gain_s = s < base_s - eps
    if gain_p and gain_s:
        return "ADV_SHADOW_CONFIRMED"
    if gain_p and not gain_s:
        return "ADV_PRIMARY_ONLY_GAIN"
    if abs(p - base_p) < 0.005 and abs(s - base_s) < 0.005:
        return "ADV_EQUIVALENT"
    return "ADV_NO_GAIN"


def main():
    OOF.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(DEV)
    folds_p = pd.read_csv(CV_P)["fold"].to_numpy()
    folds_s = pd.read_csv(CV_S)["fold"].to_numpy()
    y_tm = dev["TmApp"].to_numpy(float)
    y_hic = dev["HIC"].to_numpy(float)
    ids = list(dev["id"])
    feat = pd.read_csv(FEAT).set_index("antibody_id").loc[ids].reset_index()

    # Stage3 surface/packing subset for provisional incumbent reconstruction helpers
    s3 = pd.read_csv(S3_FEAT).set_index("antibody_id").loc[ids]
    s3_rasa = s3[[c for c in s3.columns if "rasa" in c.lower()]].to_numpy(float)
    s3_surf = s3[
        [
            c
            for c in s3.columns
            if any(x in c.lower() for x in ["sasa", "rasa", "patch", "exposed", "hydrophobic", "aromatic", "charge"])
        ]
    ].to_numpy(float)

    from run_stage1 import build_all_feature_tables, make_xy

    regions = pd.read_csv(ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv")
    ann = pd.read_csv(ANN)
    tables = build_all_feature_tables(dev, ann, regions)
    classical = {}
    for fam in ["SEQ_BASIC", "SEQ_ALL"]:
        Xn, Xc = make_xy(tables, fam)
        classical[fam] = {"num": Xn, "cat": Xc}

    emb = np.load(EMB, allow_pickle=True)
    plm_tm = emb["ablang2__HL_paired"].astype(np.float32)
    plm_hic = emb["esm2__H"].astype(np.float32)

    # family matrices
    fam_X = {}
    for fam in FAMILIES:
        cols = cols_for(feat, fam)
        fam_X[fam] = feat[cols].to_numpy(float) if cols else None
        print(fam, "n_feat", 0 if fam_X[fam] is None else fam_X[fam].shape[1], flush=True)

    # composites
    def concat_fams(names):
        mats = [fam_X[n] for n in names if fam_X.get(n) is not None and fam_X[n].shape[1] > 0]
        return np.hstack(mats) if mats else None

    fam_X["ADV_HIC_ALL"] = concat_fams(["ADV_PROPKA", "ADV_PQR_CHARGE", "ADV_ELECTROSTATICS", "ADV_SURFACE_PATCH", "ADV_INVFOLD"])
    fam_X["ADV_TMAPP_ALL"] = concat_fams(["ADV_INTERACTIONS", "ADV_CAVITY", "ADV_UNSAT_POLAR", "ADV_INVFOLD"])

    registry, advanced_only, primary_rows, shadow_rows, fusion_rows, residual_rows, tail_rows = [], [], [], [], [], [], []
    best_adv = {"TmApp": None, "HIC": None}
    best_fusion = {"TmApp": None, "HIC": None}
    oofs = {}

    # ---- advanced-only ----
    for target, y, focus in [
        ("TmApp", y_tm, ["ADV_INTERACTIONS", "ADV_CAVITY", "ADV_UNSAT_POLAR", "ADV_INVFOLD", "ADV_TMAPP_ALL", "ADV_SURFACE_PATCH"]),
        ("HIC", y_hic, ["ADV_ELECTROSTATICS", "ADV_SURFACE_PATCH", "ADV_PROPKA", "ADV_PQR_CHARGE", "ADV_INVFOLD", "ADV_HIC_ALL"]),
    ]:
        for fam in focus:
            X = fam_X.get(fam)
            if X is None or X.shape[1] == 0:
                print("skip empty", fam, flush=True)
                continue
            for kind in ["Ridge", "ElasticNet"]:
                pca_dim, bp = optuna_search(y, folds_p, X, None, None, None, kind, [None], n_trials=N_TRIALS)
                eid = f"{target}__{fam}__{kind}Opt"
                res = cv_run(y, folds_p, X, None, None, None, kind, bp, None, save=OOF / f"{eid}.csv", ids=ids)
                oofs[eid] = res["oof"]
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "feature_family": fam,
                    "model": f"{kind}Opt",
                    "hyperparameters": json.dumps(bp),
                    "n_features": res["n_features"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_stage3_prov": res["mae"] - INC[target]["stage3_prov"]["primary"],
                    "delta_vs_seq_overall": res["mae"] - INC[target]["seq_overall"]["primary"],
                    "status": "ADV_ONLY",
                    "notes": "",
                }
                registry.append(row)
                advanced_only.append(row)
                primary_rows.append(row)
                print(f"{eid}: {res['mae']:.4f}", flush=True)
                if best_adv[target] is None or res["mae"] < best_adv[target]["Primary_MAE"]:
                    best_adv[target] = row | {"oof": res["oof"], "X": X, "kind": kind, "params": bp, "fam": fam}
            # SVR on best family later
        # SVR for current best
        if best_adv[target] is not None:
            fam = best_adv[target]["fam"]
            X = fam_X[fam]
            pca_dim, bp = optuna_search(y, folds_p, X, None, None, None, "SVR", [None], n_trials=N_TRIALS)
            eid = f"{target}__{fam}__SVROpt"
            res = cv_run(y, folds_p, X, None, None, None, "SVR", bp, None, save=OOF / f"{eid}.csv", ids=ids)
            oofs[eid] = res["oof"]
            row = {
                "experiment_id": eid,
                "target": target,
                "feature_family": fam,
                "model": "SVROpt",
                "hyperparameters": json.dumps(bp),
                "n_features": res["n_features"],
                "Primary_MAE": res["mae"],
                "Primary_fold_SD": res["fold_sd"],
                "Shadow_MAE": np.nan,
                "delta_vs_stage3_prov": res["mae"] - INC[target]["stage3_prov"]["primary"],
                "delta_vs_seq_overall": res["mae"] - INC[target]["seq_overall"]["primary"],
                "status": "ADV_ONLY",
                "notes": "SVR on best family",
            }
            registry.append(row)
            advanced_only.append(row)
            primary_rows.append(row)
            if res["mae"] < best_adv[target]["Primary_MAE"]:
                best_adv[target] = row | {"oof": res["oof"], "X": X, "kind": "SVR", "params": bp, "fam": fam}

    # ---- fusion ----
    for target, y, plm, clas, s3mat, pca_choices, fus_fams in [
        (
            "TmApp",
            y_tm,
            plm_tm,
            classical["SEQ_BASIC"],
            s3_rasa,
            [None, 32, 48, 64],
            ["ADV_INTERACTIONS", "ADV_CAVITY", "ADV_UNSAT_POLAR", "ADV_INVFOLD", "ADV_TMAPP_ALL"],
        ),
        (
            "HIC",
            y_hic,
            plm_hic,
            classical["SEQ_ALL"],
            s3_surf,
            [None, 8, 32, 48, 64],
            ["ADV_ELECTROSTATICS", "ADV_SURFACE_PATCH", "ADV_PROPKA", "ADV_PQR_CHARGE", "ADV_INVFOLD", "ADV_HIC_ALL"],
        ),
    ]:
        # also include best adv fam
        bf = best_adv[target]["fam"]
        if bf not in fus_fams:
            fus_fams = [bf] + fus_fams
        for fam in fus_fams:
            X = fam_X.get(fam)
            if X is None or X.shape[1] == 0:
                continue
            for kind in ["Ridge", "SVR"]:
                # incumbent = PLM + classical + stage3 subset + advanced
                pca_dim, bp = optuna_search(y, folds_p, X, plm, clas, s3mat, kind, pca_choices, n_trials=N_TRIALS)
                eid = f"{target}__FUSION_S3INC__{fam}__{kind}Opt"
                res = cv_run(y, folds_p, X, plm, clas, s3mat, kind, bp, pca_dim, save=OOF / f"{eid}.csv", ids=ids)
                oofs[eid] = res["oof"]
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "feature_family": fam,
                    "model": f"S3INC+{kind}Opt",
                    "hyperparameters": json.dumps({"pca_dim": pca_dim, **bp}),
                    "n_features": res["n_features"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_stage3_prov": res["mae"] - INC[target]["stage3_prov"]["primary"],
                    "delta_vs_seq_overall": res["mae"] - INC[target]["seq_overall"]["primary"],
                    "status": "FUSION",
                    "notes": "seq overall + stage3 structure subset + advanced family",
                }
                registry.append(row)
                fusion_rows.append(row)
                primary_rows.append(row)
                print(f"{eid}: {res['mae']:.4f}", flush=True)
                if best_fusion[target] is None or res["mae"] < best_fusion[target]["Primary_MAE"]:
                    best_fusion[target] = row | {
                        "oof": res["oof"],
                        "X": X,
                        "kind": kind,
                        "params": bp,
                        "pca_dim": pca_dim,
                        "fam": fam,
                        "plm": plm,
                        "classical": clas,
                        "s3": s3mat,
                    }

    # ---- Shadow ----
    def shadow_one(target, y, meta, label, base_key="stage3_prov"):
        res = cv_run(
            y,
            folds_s,
            meta["X"],
            meta.get("plm"),
            meta.get("classical"),
            meta.get("s3"),
            meta["kind"],
            meta["params"],
            meta.get("pca_dim"),
        )
        st = status_vs(INC[target][base_key]["primary"], INC[target][base_key]["shadow"], meta["Primary_MAE"], res["mae"])
        if label == "advanced_only":
            # vs seq overall for reference
            st = status_vs(INC[target]["seq_overall"]["primary"], INC[target]["seq_overall"]["shadow"], meta["Primary_MAE"], res["mae"])
            # but advanced-only rarely beats; keep ADV_ONLY_SHADOW tag if not gain
            if st == "ADV_NO_GAIN":
                st = "ADV_ONLY_SHADOW_EVAL"
        srow = {
            "experiment_id": meta["experiment_id"],
            "target": target,
            "label": label,
            "feature_family": meta["feature_family"],
            "Primary_MAE": meta["Primary_MAE"],
            "Shadow_MAE": res["mae"],
            "status": st,
        }
        shadow_rows.append(srow)
        for r in registry:
            if r["experiment_id"] == meta["experiment_id"]:
                r["Shadow_MAE"] = res["mae"]
                r["status"] = st
        print(f"SHADOW {meta['experiment_id']}: {res['mae']:.4f} [{st}]", flush=True)
        return srow

    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        # top advanced-only
        sub = pd.DataFrame(advanced_only)
        sub = sub[sub.target == target].sort_values("Primary_MAE").head(3)
        for _, r in sub.iterrows():
            fam = r.feature_family
            kind = r.model.replace("Opt", "")
            meta = {
                "experiment_id": r.experiment_id,
                "feature_family": fam,
                "Primary_MAE": r.Primary_MAE,
                "X": fam_X[fam],
                "kind": kind,
                "params": json.loads(r.hyperparameters),
                "pca_dim": None,
            }
            shadow_one(target, y, meta, "advanced_only")
        # top fusion
        sub = pd.DataFrame(fusion_rows)
        sub = sub[sub.target == target].sort_values("Primary_MAE").head(3)
        for _, r in sub.iterrows():
            fam = r.feature_family
            hp = json.loads(r.hyperparameters)
            pca_dim = hp.pop("pca_dim", None)
            kind = "SVR" if "SVR" in r.model else "Ridge"
            meta = {
                "experiment_id": r.experiment_id,
                "feature_family": fam,
                "Primary_MAE": r.Primary_MAE,
                "X": fam_X[fam],
                "kind": kind,
                "params": hp,
                "pca_dim": pca_dim,
                "plm": plm_tm if target == "TmApp" else plm_hic,
                "classical": classical["SEQ_BASIC"] if target == "TmApp" else classical["SEQ_ALL"],
                "s3": s3_rasa if target == "TmApp" else s3_surf,
            }
            shadow_one(target, y, meta, "fusion", base_key="stage3_prov")

    # ---- residual + hic tail ----
    # load stage3 overall oofs
    s3oof = {
        "TmApp": pd.read_csv(ROOT / "virtual_participant/stage3_structure/oof/REF_TmApp_overall.csv"),
        "HIC": pd.read_csv(ROOT / "virtual_participant/stage3_structure/oof/REF_HIC_overall.csv"),
    }
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        rseq = y - s3oof[target]["y_pred"].to_numpy()
        adv = best_adv[target]
        radv = y - adv["oof"]
        residual_rows.append(
            {
                "target": target,
                "analysis": "residual_residual_corr",
                "feature": "seq_overall_vs_best_adv",
                "value": float(pearsonr(rseq, radv)[0]),
                "notes": adv["experiment_id"],
            }
        )
        residual_rows.append(
            {
                "target": target,
                "analysis": "seq_residual_vs_adv_pred",
                "feature": "best_adv_pred",
                "value": float(pearsonr(rseq, adv["oof"])[0]),
                "notes": adv["experiment_id"],
            }
        )
        # feature correlations
        feat_num = feat.set_index("antibody_id").loc[ids]
        cand = [
            c
            for c in feat_num.columns
            if any(
                x in c
                for x in [
                    "apbs_mean",
                    "propka_exposed",
                    "adv_max_local",
                    "adv_spatial",
                    "salt_bridge",
                    "packing_degree",
                    "cavity_proxy",
                    "unsatisfied",
                    "invfold_fv_mean",
                ]
            )
            and pd.api.types.is_numeric_dtype(feat_num[c])
        ]
        for c in cand:
            x = feat_num[c].to_numpy(float)
            m = np.isfinite(x)
            if m.sum() < 20:
                continue
            residual_rows.append(
                {
                    "target": target,
                    "analysis": "seq_residual_vs_feature",
                    "feature": c,
                    "value": float(pearsonr(rseq[m], x[m])[0]),
                    "notes": "",
                }
            )

    # HIC frozen tail
    tail_mask = y_hic >= HIC_TAIL
    assert int(tail_mask.sum()) == 17
    models_tail = {
        "plm_only": pd.read_csv(ROOT / "virtual_participant/stage3_structure/oof/REF_HIC_PLM_only.csv").y_pred.to_numpy(),
        "seq_overall": s3oof["HIC"].y_pred.to_numpy(),
        "stage3_overall_struct": pd.read_csv(
            ROOT / "virtual_participant/stage3_structure/oof/HIC__FUSION_OVERALL__ESMFold__STRUCT_SURFACE_ALL__SVROpt.csv"
        ).y_pred.to_numpy(),
        "best_adv_only": best_adv["HIC"]["oof"],
        "best_fusion": best_fusion["HIC"]["oof"],
    }
    for name, pred in models_tail.items():
        yt, pt = y_hic[tail_mask], pred[tail_mask]
        tail_rows.append(
            {
                "model": name,
                "n_tail": 17,
                "threshold": HIC_TAIL,
                "overall_MAE": mae(y_hic, pred),
                "tail_MAE": mae(yt, pt),
                "tail_bias_pred_minus_true": float(np.mean(pt - yt)),
                "tail_underprediction_count": int((pt < yt).sum()),
                "tail_pred_sd": float(np.std(pt)),
                "overall_pred_sd": float(np.std(pred)),
            }
        )

    # plots
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(y, best_adv[target]["oof"], s=18, alpha=0.7)
        plt.title(f"{target} best advanced-only")
        plt.savefig(PLOTS / f"{target}_adv_only.png", dpi=130)
        plt.close()
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(y, best_fusion[target]["oof"], s=18, alpha=0.7)
        plt.title(f"{target} best fusion")
        plt.savefig(PLOTS / f"{target}_fusion.png", dpi=130)
        plt.close()

    pd.DataFrame(registry).to_csv(OUT / "stage4_model_registry.csv", index=False)
    pd.DataFrame(advanced_only).to_csv(OUT / "stage4_advanced_only_results.csv", index=False)
    pd.DataFrame(primary_rows).to_csv(OUT / "stage4_primary_results.csv", index=False)
    pd.DataFrame(shadow_rows).to_csv(OUT / "stage4_shadow_results.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(OUT / "stage4_fusion_results.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "stage4_residual_analysis.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(OUT / "stage4_hic_tail_diagnostics.csv", index=False)

    def sh_status(eid):
        for r in shadow_rows:
            if r["experiment_id"] == eid:
                return r["status"], r["Shadow_MAE"]
        return "NOT_SHADOWED", None

    best_models = {}
    for target in ["TmApp", "HIC"]:
        ba, bf = best_adv[target], best_fusion[target]
        sta, sha = sh_status(ba["experiment_id"])
        stf, shf = sh_status(bf["experiment_id"])
        update = stf == "ADV_SHADOW_CONFIRMED" and bf["Primary_MAE"] < INC[target]["stage3_prov"]["primary"] - 1e-4
        # avoid tiny formal threshold; require shadow confirm + not microscopic only narrative
        best_models[target] = {
            "references": INC[target],
            "best_advanced_only": {
                "experiment_id": ba["experiment_id"],
                "family": ba["feature_family"],
                "Primary_MAE": ba["Primary_MAE"],
                "Shadow_MAE": sha,
                "status": sta,
            },
            "best_fusion": {
                "experiment_id": bf["experiment_id"],
                "family": bf["feature_family"],
                "Primary_MAE": bf["Primary_MAE"],
                "Shadow_MAE": shf,
                "delta_vs_stage3_prov": bf["delta_vs_stage3_prov"],
                "status": stf,
            },
            "update_incumbent": update,
            "update_note": "Stage3 provisional kept unless Shadow-confirmed fusion improves with clear direction",
        }
    (OUT / "stage4_best_models.json").write_text(json.dumps(best_models, indent=2), encoding="utf-8")
    print(json.dumps(best_models, indent=2), flush=True)
    print("STAGE4_MODELING_DONE", flush=True)


if __name__ == "__main__":
    main()
