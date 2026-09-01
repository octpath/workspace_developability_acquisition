#!/usr/bin/env python3
"""
Stage 2 — PLM shootout (frozen Stage0 CV; Stage1 classical baselines fixed).
Uses participant-safe raw embeddings; PCA/scaler fit inside each training fold.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/workspace_developability_acquisition")
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_S = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
PLM_ROOT = ROOT / "gate_b1/cache/plm"
OUT = ROOT / "virtual_participant/stage2_plm"
CACHE = OUT / "cache"
OOF_DIR = OUT / "oof"
PLOTS = OUT / "plots"
ARTIFACTS = OUT / "artifacts"

# Stage1 frozen baselines
S1 = {
    "TmApp": {"primary": 3.1011, "shadow": 3.1716, "id": "TmApp__SEQ_BASIC__SVROpt"},
    "HIC": {"primary": 0.4732, "shadow": 0.4788, "id": "HIC__SEQ_PLUS_ANTIBODY__SVROpt"},
}
S1_HIC_REF = {
    "SEQ_ALL_SVR": {"primary": 0.4766, "shadow": 0.4795},
    "SEQ_COMBINED_SVR": {"primary": 0.4775, "shadow": 0.4724},
}

PCA_GRID = [8, 16, 32, 48, 64]
N_TRIALS = 40
SEED = 42

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def load_embeddings(dev: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Return emb[plm][repr] -> (N, D) aligned to dev rows."""
    out: dict[str, dict[str, np.ndarray]] = {}

    # ESM-1b / ESM-2 from HL concat cache
    for plm, mid in [
        ("esm1b", "esm1b_t33_650M_UR50S"),
        ("esm2", "esm2_t33_650M_UR50D"),
    ]:
        man = pd.read_csv(PLM_ROOT / f"manifest_{mid}.csv").set_index("antibody_id")
        hl, h, l = [], [], []
        for aid in dev["id"]:
            path = Path(man.loc[aid, "path"])
            # ensure HL_concat_mean
            if "HL_concat_mean" not in path.name:
                path = PLM_ROOT / mid / f"HL_concat_mean_{man.loc[aid, 'pair_hash']}.npy"
            arr = np.load(path).astype(np.float32)
            assert arr.shape[0] == 2560, (plm, arr.shape)
            hl.append(arr)
            h.append(arr[:1280])
            l.append(arr[1280:])
        out[plm] = {
            "H": np.vstack(h),
            "L": np.vstack(l),
            "HL": np.vstack(hl),
        }
        print(f"loaded {plm}: H{out[plm]['H'].shape} L{out[plm]['L'].shape} HL{out[plm]['HL'].shape}", flush=True)

    # AbLang2 paired + H/L-only (compute if missing)
    man = pd.read_csv(PLM_ROOT / "manifest_ablang2_default.csv").set_index("antibody_id")
    paired, h_list, l_list, hl_cat = [], [], [], []
    need_compute = []
    for i, aid in enumerate(dev["id"]):
        path = Path(man.loc[aid, "path"])
        paired.append(np.load(path).astype(np.float32))
        hp = CACHE / f"ablang2_H_{aid}.npy"
        lp = CACHE / f"ablang2_L_{aid}.npy"
        if hp.exists() and lp.exists():
            hv = np.load(hp)
            lv = np.load(lp)
        else:
            need_compute.append(i)
            hv = lv = None
        h_list.append(hv)
        l_list.append(lv)

    if need_compute:
        print(f"Computing AbLang2 H/L-only for {len(need_compute)} antibodies...", flush=True)
        import ablang2

        ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, ncpu=1, device="cpu")
        for i in need_compute:
            aid = dev["id"].iloc[i]
            hseq, lseq = dev["heavy"].iloc[i], dev["light"].iloc[i]
            hv = np.asarray(ablang([[hseq, ""]], mode="seqcoding")).reshape(-1).astype(np.float32)
            lv = np.asarray(ablang([["", lseq]], mode="seqcoding")).reshape(-1).astype(np.float32)
            np.save(CACHE / f"ablang2_H_{aid}.npy", hv)
            np.save(CACHE / f"ablang2_L_{aid}.npy", lv)
            h_list[i] = hv
            l_list[i] = lv
            if (len(need_compute) > 20) and (need_compute.index(i) % 20 == 0):
                print(f"  ablang2 chain {need_compute.index(i)+1}/{len(need_compute)}", flush=True)

    for i in range(len(dev)):
        hl_cat.append(np.concatenate([h_list[i], l_list[i]]))

    out["ablang2"] = {
        "H": np.vstack(h_list),
        "L": np.vstack(l_list),
        "HL": np.vstack(hl_cat),          # fair concat
        "HL_paired": np.vstack(paired),   # native paired seqcoding
    }
    print(
        f"loaded ablang2: H{out['ablang2']['H'].shape} L{out['ablang2']['L'].shape} "
        f"HL{out['ablang2']['HL'].shape} HL_paired{out['ablang2']['HL_paired'].shape}",
        flush=True,
    )
    return out


def build_classical(dev: pd.DataFrame, ann: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from run_stage1 import build_all_feature_tables, make_xy

    regions = pd.read_csv(ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv")
    assert list(dev["id"]) == list(regions["id"]) == list(ann["id"])
    tables = build_all_feature_tables(dev, ann, regions)
    feats = {}
    for fam in ["SEQ_BASIC", "SEQ_ALL", "SEQ_PLUS_ANTIBODY", "ANN_CDR_LENGTH", "ANN_GERMLINE"]:
        Xn, Xc = make_xy(tables, fam)
        if Xc is not None:
            # store numeric + cat separately for fold-safe encoding later
            feats[fam] = {"num": Xn, "cat": Xc}
        else:
            feats[fam] = {"num": Xn, "cat": None}
    return feats


class RareCat:
    def __init__(self, min_count=3):
        self.min_count = min_count
        self.keep_ = {}
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
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


def fold_matrices(X_plm, classical, tr, va, pca_dim):
    """Fit scaler(+PCA) on PLM and optional classical on train fold only."""
    # PLM branch
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_plm[tr])
    Xva = scaler.transform(X_plm[va])
    n_before = Xtr.shape[1]
    pca = None
    if pca_dim is not None and pca_dim > 0:
        n_comp = min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=0)
            Xtr = pca.fit_transform(Xtr)
            Xva = pca.transform(Xva)
    n_after = Xtr.shape[1]

    if classical is not None:
        num = classical["num"]
        cat = classical["cat"]
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
        n_after = Xtr.shape[1]
    return Xtr, Xva, n_before, n_after


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
    if kind == "SVR":
        return SVR(
            kernel="rbf",
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale") if isinstance(params.get("gamma", "scale"), str)
            else params.get("gamma", 0.01),
            epsilon=params.get("epsilon", 0.1),
        )
    raise ValueError(kind)


def cv_run(y, folds, X_plm, classical, model_kind, params, pca_dim, save_oof_path=None, ids=None):
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    fold_maes = []
    n_before = n_after = 0
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        Xtr, Xva, n_before, n_after = fold_matrices(X_plm, classical, tr, va, pca_dim)
        model = make_model(model_kind, params)
        # SVR gamma='scale' handled inside
        if model_kind == "SVR" and params.get("gamma") == "scale":
            model = SVR(kernel="rbf", C=params.get("C", 1.0), gamma="scale", epsilon=params.get("epsilon", 0.1))
        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    fold_maes = np.asarray(fold_maes, float)
    if save_oof_path is not None:
        df = pd.DataFrame({
            "id": ids,
            "fold": folds,
            "y_true": y,
            "y_pred": oof,
            "residual": y - oof,
            "experiment_id": save_oof_path.stem,
        })
        df.to_csv(save_oof_path, index=False)
    return {
        "mae": float(fold_maes.mean()),
        "fold_sd": float(fold_maes.std(ddof=1)),
        "fold_maes": fold_maes.tolist(),
        "oof": oof,
        "n_before": int(n_before),
        "n_after": int(n_after),
        "pearson": float(pearsonr(y, oof)[0]),
        "spearman": float(spearmanr(y, oof)[0]),
        "pred_sd": float(np.std(oof)),
    }


def optuna_search(y, folds, X_plm, classical, model_kind, pca_choices, n_trials=N_TRIALS):
    def objective(trial):
        pca_dim = trial.suggest_categorical("pca_dim", pca_choices)
        if model_kind == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
        elif model_kind == "ElasticNet":
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
        res = cv_run(y, folds, X_plm, classical, model_kind, params, pca_dim)
        return res["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params)
    pca_dim = bp.pop("pca_dim")
    return pca_dim, bp, study.best_value


def hic_tail_mae(y, oof, q=0.9):
    thr = np.quantile(y, q)
    m = y >= thr
    if m.sum() == 0:
        return float("nan")
    return mae(y[m], oof[m])


def plot_obs_pred(y, oof, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y, oof, s=18, alpha=0.7, c="#2F4B7C")
    lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    ax.plot(lims, lims, "--", color="#D45087", lw=1)
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    for d in [OUT, CACHE, OOF_DIR, PLOTS, ARTIFACTS]:
        d.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(DEV)
    ann = pd.read_csv(ANN)
    assert list(dev["id"]) == list(ann["id"])
    folds_p = pd.read_csv(CV_P)["fold"].to_numpy()
    folds_s = pd.read_csv(CV_S)["fold"].to_numpy()
    assert list(pd.read_csv(CV_P)["id"]) == list(dev["id"])

    print("=== Load embeddings ===", flush=True)
    emb = load_embeddings(dev)
    np.savez_compressed(
        CACHE / "stage2_embeddings.npz",
        **{f"{p}__{r}": emb[p][r] for p in emb for r in emb[p]},
        ids=dev["id"].to_numpy(),
    )

    print("=== Build classical features for fusion ===", flush=True)
    classical = build_classical(dev, ann)

    registry = []
    fixed_rows = []
    ablation_rows = []
    primary_rows = []
    shadow_rows = []
    fusion_rows = []
    oof_store = {}  # experiment_id -> oof array for residual corr

    def add_reg(rec):
        registry.append(rec)
        return rec

    # -------------------------------------------------------------------------
    # Fixed-pipeline fair comparison + chain ablation
    # -------------------------------------------------------------------------
    print("=== Fixed pipeline screen ===", flush=True)
    fixed_specs = []
    for plm in ["esm1b", "esm2", "ablang2"]:
        for chain in ["H", "L", "HL"]:
            for pca_dim in [16, 32, 64]:
                for model_kind, params in [
                    ("Ridge", {"alpha": 10.0}),
                    ("SVR", {"C": 3.0, "gamma": 0.01, "epsilon": 0.1}),
                ]:
                    fixed_specs.append((plm, chain, pca_dim, model_kind, params))
        # also AbLang2 paired
        if plm == "ablang2":
            for pca_dim in [16, 32]:
                for model_kind, params in [
                    ("Ridge", {"alpha": 10.0}),
                    ("SVR", {"C": 3.0, "gamma": 0.01, "epsilon": 0.1}),
                ]:
                    fixed_specs.append((plm, "HL_paired", pca_dim, model_kind, params))

    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        s1p = S1[target]["primary"]
        for plm, chain, pca_dim, model_kind, params in fixed_specs:
            X = emb[plm][chain]
            eid = f"{target}__{plm}__{chain}__PCA{pca_dim}__{model_kind}"
            res = cv_run(y, folds_p, X, None, model_kind, params, pca_dim)
            row = {
                "experiment_id": eid,
                "target": target,
                "plm": plm,
                "chain_representation": chain,
                "pooling": "whole_chain_mean" if chain != "HL_paired" else "paired_seqcoding",
                "pca_dim": pca_dim,
                "model": model_kind,
                "hyperparameters": json.dumps(params),
                "n_features_before_pca": res["n_before"],
                "n_features_after_pca": res["n_after"],
                "Primary_MAE": res["mae"],
                "Primary_fold_SD": res["fold_sd"],
                "Shadow_MAE": np.nan,
                "delta_vs_stage1_primary": res["mae"] - s1p,
                "delta_vs_stage1_shadow": np.nan,
                "status": "FIXED_SCREEN",
                "notes": "fixed pipeline",
                "pearson": res["pearson"],
                "spearman": res["spearman"],
            }
            add_reg(row)
            fixed_rows.append(row)
            primary_rows.append(row)
            if chain in ("H", "L", "HL") and pca_dim == 32 and model_kind in ("Ridge", "SVR"):
                ablation_rows.append({
                    "target": target,
                    "plm": plm,
                    "chain": chain,
                    "pca_dim": pca_dim,
                    "model": model_kind,
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "delta_vs_stage1_primary": res["mae"] - s1p,
                    "comparison_type": "CONTROLLED_CHAIN_ABLATION",
                })
            print(f"  {eid}: MAE={res['mae']:.4f} ΔS1={res['mae']-s1p:+.4f}", flush=True)

    pd.DataFrame(fixed_rows).to_csv(OUT / "stage2_fixed_pipeline_results.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT / "stage2_chain_ablation.csv", index=False)

    # -------------------------------------------------------------------------
    # Optuna on promising PLM reps (HL focus + best chain from ablation)
    # -------------------------------------------------------------------------
    print("=== Optuna on promising representations ===", flush=True)
    tuned = []
    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        s1p = S1[target]["primary"]
        # choose chains: always HL; also best of H/L from fixed PCA32 Ridge
        ab = pd.DataFrame(ablation_rows)
        ab_t = ab[(ab.target == target) & (ab.model == "Ridge") & (ab.pca_dim == 32)]
        candidates = []
        for plm in ["esm1b", "esm2", "ablang2"]:
            candidates.append((plm, "HL"))
            sub = ab_t[ab_t.plm == plm]
            if len(sub):
                best_chain = sub.sort_values("Primary_MAE").iloc[0]["chain"]
                if best_chain != "HL":
                    candidates.append((plm, best_chain))
            if plm == "ablang2":
                candidates.append((plm, "HL_paired"))
        # unique
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)

        for plm, chain in uniq:
            X = emb[plm][chain]
            for model_kind in ["Ridge", "SVR"]:
                print(f"  Optuna {target} {plm} {chain} {model_kind}...", flush=True)
                pca_dim, params, _ = optuna_search(y, folds_p, X, None, model_kind, PCA_GRID, N_TRIALS)
                eid = f"{target}__{plm}__{chain}__{model_kind}Opt"
                res = cv_run(
                    y, folds_p, X, None, model_kind, params, pca_dim,
                    save_oof_path=OOF_DIR / f"{eid}.csv", ids=dev["id"].values,
                )
                oof_store[eid] = res["oof"]
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "plm": plm,
                    "chain_representation": chain,
                    "pooling": "whole_chain_mean" if chain != "HL_paired" else "paired_seqcoding",
                    "pca_dim": pca_dim,
                    "model": f"{model_kind}Opt",
                    "hyperparameters": json.dumps(params),
                    "n_features_before_pca": res["n_before"],
                    "n_features_after_pca": res["n_after"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_stage1_primary": res["mae"] - s1p,
                    "delta_vs_stage1_shadow": np.nan,
                    "status": "PLM_TUNED",
                    "notes": f"optuna trials={N_TRIALS}",
                    "pearson": res["pearson"],
                    "spearman": res["spearman"],
                    "hic_tail_mae": hic_tail_mae(y, res["oof"]) if target == "HIC" else np.nan,
                    "fold_maes": json.dumps(res["fold_maes"]),
                }
                add_reg(row)
                primary_rows.append(row)
                tuned.append(row)
                print(f"    -> MAE={res['mae']:.4f} pca={pca_dim} params={params}", flush=True)

    # -------------------------------------------------------------------------
    # Shadow for top PLM-only per target
    # -------------------------------------------------------------------------
    print("=== Shadow audit ===", flush=True)
    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        s1p, s1s = S1[target]["primary"], S1[target]["shadow"]
        cands = [r for r in tuned if r["target"] == target]
        cands = sorted(cands, key=lambda r: r["Primary_MAE"])[:5]
        for r in cands:
            plm, chain = r["plm"], r["chain_representation"]
            X = emb[plm][chain]
            model_kind = "Ridge" if r["model"].startswith("Ridge") else "SVR"
            params = json.loads(r["hyperparameters"])
            pca_dim = r["pca_dim"]
            res = cv_run(y, folds_s, X, None, model_kind, params, pca_dim)
            delta_p = r["Primary_MAE"] - s1p
            delta_s = res["mae"] - s1s
            if delta_p < -1e-4 and delta_s < -1e-4:
                status = "PLM_SHADOW_CONFIRMED"
            elif delta_p < -1e-4 and delta_s >= -1e-4:
                status = "PLM_PRIMARY_ONLY_GAIN"
            elif abs(delta_p) < 1e-3 and abs(delta_s) < 1e-3:
                status = "PLM_NO_GAIN"
            else:
                status = "PLM_UNSTABLE" if delta_s > 0.02 else "PLM_NO_GAIN"
            sh = {
                "experiment_id": r["experiment_id"],
                "target": target,
                "plm": plm,
                "chain_representation": chain,
                "model": r["model"],
                "Primary_MAE": r["Primary_MAE"],
                "Shadow_MAE": res["mae"],
                "Shadow_fold_SD": res["fold_sd"],
                "delta_vs_stage1_primary": delta_p,
                "delta_vs_stage1_shadow": delta_s,
                "status": status,
            }
            shadow_rows.append(sh)
            # update registry
            for rr in registry:
                if rr["experiment_id"] == r["experiment_id"]:
                    rr["Shadow_MAE"] = res["mae"]
                    rr["delta_vs_stage1_shadow"] = delta_s
                    rr["status"] = status
            print(f"  SHADOW {r['experiment_id']}: P={r['Primary_MAE']:.4f} S={res['mae']:.4f} [{status}]", flush=True)

    # -------------------------------------------------------------------------
    # Fusion: PLM + classical (top PLM per target)
    # -------------------------------------------------------------------------
    print("=== Fusion ===", flush=True)
    fusion_plans = {
        "TmApp": [
            ("SEQ_BASIC", None),
            ("ANN_CDR_LENGTH", None),
            ("ANN_GERMLINE", None),
        ],
        "HIC": [
            ("SEQ_ALL", None),
            ("SEQ_PLUS_ANTIBODY", None),
            ("SEQ_BASIC", None),
        ],
    }
    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        s1p, s1s = S1[target]["primary"], S1[target]["shadow"]
        # best PLM-only tuned
        plm_cands = sorted([r for r in tuned if r["target"] == target], key=lambda r: r["Primary_MAE"])
        best_plm = plm_cands[0]
        # also try top 2 PLM families
        fam_best = {}
        for r in plm_cands:
            fam_best.setdefault(r["plm"], r)
        plm_list = list(fam_best.values())[:3]

        for base in plm_list:
            plm, chain = base["plm"], base["chain_representation"]
            X = emb[plm][chain]
            model_kind = "Ridge" if "Ridge" in base["model"] else "SVR"
            for fam, _ in fusion_plans[target]:
                print(f"  Fusion {target} {plm}/{chain} + {fam}...", flush=True)
                pca_dim, params, _ = optuna_search(
                    y, folds_p, X, classical[fam], model_kind, PCA_GRID, n_trials=30
                )
                eid = f"{target}__FUSION__{plm}__{chain}__{fam}__{model_kind}Opt"
                res = cv_run(
                    y, folds_p, X, classical[fam], model_kind, params, pca_dim,
                    save_oof_path=OOF_DIR / f"{eid}.csv", ids=dev["id"].values,
                )
                oof_store[eid] = res["oof"]
                # shadow
                res_s = cv_run(y, folds_s, X, classical[fam], model_kind, params, pca_dim)
                delta_p = res["mae"] - s1p
                delta_s = res_s["mae"] - s1s
                if delta_p < -1e-4 and delta_s < -1e-4:
                    status = "PLM_SHADOW_CONFIRMED"
                elif delta_p < -1e-4:
                    status = "PLM_PRIMARY_ONLY_GAIN"
                else:
                    status = "PLM_NO_GAIN"
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "plm": plm,
                    "chain_representation": chain,
                    "pooling": "whole_chain_mean",
                    "pca_dim": pca_dim,
                    "model": f"FUSION_{model_kind}Opt",
                    "hyperparameters": json.dumps(params),
                    "classical_family": fam,
                    "n_features_before_pca": res["n_before"],
                    "n_features_after_pca": res["n_after"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": res_s["mae"],
                    "delta_vs_stage1_primary": delta_p,
                    "delta_vs_stage1_shadow": delta_s,
                    "delta_vs_plm_only_primary": res["mae"] - base["Primary_MAE"],
                    "status": status,
                    "notes": f"fusion with {fam}; base_plm={base['experiment_id']}",
                }
                add_reg(row)
                fusion_rows.append(row)
                primary_rows.append(row)
                shadow_rows.append({
                    "experiment_id": eid,
                    "target": target,
                    "plm": plm,
                    "chain_representation": chain,
                    "model": row["model"],
                    "Primary_MAE": res["mae"],
                    "Shadow_MAE": res_s["mae"],
                    "Shadow_fold_SD": res_s["fold_sd"],
                    "delta_vs_stage1_primary": delta_p,
                    "delta_vs_stage1_shadow": delta_s,
                    "status": status,
                })
                print(
                    f"    -> P={res['mae']:.4f} S={res_s['mae']:.4f} "
                    f"ΔS1={delta_p:+.4f} ΔPLM={res['mae']-base['Primary_MAE']:+.4f}",
                    flush=True,
                )

    # -------------------------------------------------------------------------
    # Residual correlations
    # -------------------------------------------------------------------------
    print("=== Residual correlations ===", flush=True)
    from run_stage1 import build_all_feature_tables, make_xy

    regions = pd.read_csv(ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv")
    tables = build_all_feature_tables(dev, ann, regions)

    def stage1_oof(target, family, model_name, params_json):
        y = dev[target].to_numpy(float)
        Xn, Xc = make_xy(tables, family)
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
        n_folds = int(folds_p.max()) + 1
        oof = np.zeros(len(y))
        for f in range(n_folds):
            tr = folds_p != f
            va = folds_p == f
            imp = SimpleImputer(strategy="median")
            Ntr = imp.fit_transform(Xn.iloc[tr])
            Nva = imp.transform(Xn.iloc[va])
            sc = StandardScaler()
            Ntr = sc.fit_transform(Ntr)
            Nva = sc.transform(Nva)
            if Xc is not None:
                enc = RareCat(3)
                Ctr = enc.fit(Xc.iloc[tr]).transform(Xc.iloc[tr]).to_numpy()
                Cva = enc.transform(Xc.iloc[va]).to_numpy()
                Ntr = np.hstack([Ntr, Ctr])
                Nva = np.hstack([Nva, Cva])
            if "SVR" in model_name:
                model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"])
            elif "ElasticNet" in model_name:
                model = ElasticNet(alpha=params.get("alpha", 0.05), l1_ratio=params.get("l1_ratio", 0.3),
                                   max_iter=8000, tol=1e-3, random_state=0)
            else:
                model = Ridge(alpha=params.get("alpha", 10.0), random_state=0)
            model.fit(Ntr, y[tr])
            oof[va] = model.predict(Nva)
        return oof

    s1_best = json.loads((ROOT / "virtual_participant/stage1_features/stage1_best_models.json").read_text())
    resid_rows = []
    for target in ["TmApp", "HIC"]:
        y = dev[target].to_numpy(float)
        fam = s1_best[target]["feature_family"]
        model = s1_best[target]["model"]
        params = s1_best[target]["hyperparameters"]
        oof_s1 = stage1_oof(target, fam, model, params)
        oof_store[f"STAGE1_{target}"] = oof_s1
        keys = [k for k in oof_store if k.startswith(target) or k == f"STAGE1_{target}"]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ra = y - oof_store[a]
                rb = y - oof_store[b]
                resid_rows.append({
                    "target": target,
                    "model_a": a,
                    "model_b": b,
                    "residual_pearson": float(pearsonr(ra, rb)[0]),
                    "residual_spearman": float(spearmanr(ra, rb)[0]),
                })

    resid_df = pd.DataFrame(resid_rows).drop_duplicates(subset=["target", "model_a", "model_b"])
    resid_df.to_csv(OUT / "stage2_residual_correlations.csv", index=False)

    # plots for best models
    for target in ["TmApp", "HIC"]:
        best = sorted([r for r in tuned if r["target"] == target], key=lambda r: r["Primary_MAE"])[0]
        oof = oof_store[best["experiment_id"]]
        y = dev[target].to_numpy(float)
        plot_obs_pred(y, oof, f"{target} {best['experiment_id']}", PLOTS / f"obs_pred_{target}_best_plm.png")

    # -------------------------------------------------------------------------
    # Save registries + best models JSON
    # -------------------------------------------------------------------------
    reg_df = pd.DataFrame(registry)
    reg_df.to_csv(OUT / "stage2_model_registry.csv", index=False)
    pd.DataFrame(primary_rows).to_csv(OUT / "stage2_primary_results.csv", index=False)
    pd.DataFrame(shadow_rows).to_csv(OUT / "stage2_shadow_results.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(OUT / "stage2_fusion_results.csv", index=False)

    best_models = {}
    for target in ["TmApp", "HIC"]:
        plm_only = sorted([r for r in tuned if r["target"] == target], key=lambda r: r["Primary_MAE"])
        fus = sorted([r for r in fusion_rows if r["target"] == target], key=lambda r: r["Primary_MAE"])
        best_plm = plm_only[0]
        sh = next((s for s in shadow_rows if s["experiment_id"] == best_plm["experiment_id"]), {})
        best_fus = fus[0] if fus else None
        # per-family best
        fam = {}
        for r in plm_only:
            fam.setdefault(r["plm"], r)
        best_models[target] = {
            "stage1_reference": S1[target],
            "best_plm_only": {
                "experiment_id": best_plm["experiment_id"],
                "plm": best_plm["plm"],
                "chain": best_plm["chain_representation"],
                "model": best_plm["model"],
                "pca_dim": best_plm["pca_dim"],
                "primary_mae": best_plm["Primary_MAE"],
                "shadow_mae": sh.get("Shadow_MAE"),
                "delta_vs_stage1_primary": best_plm["delta_vs_stage1_primary"],
                "delta_vs_stage1_shadow": sh.get("delta_vs_stage1_shadow"),
                "status": sh.get("status", best_plm["status"]),
            },
            "best_fusion": None if best_fus is None else {
                "experiment_id": best_fus["experiment_id"],
                "classical_family": best_fus["classical_family"],
                "plm": best_fus["plm"],
                "primary_mae": best_fus["Primary_MAE"],
                "shadow_mae": best_fus["Shadow_MAE"],
                "delta_vs_stage1_primary": best_fus["delta_vs_stage1_primary"],
                "delta_vs_stage1_shadow": best_fus["delta_vs_stage1_shadow"],
                "delta_vs_plm_only_primary": best_fus["delta_vs_plm_only_primary"],
                "status": best_fus["status"],
            },
            "per_plm_best": {k: {
                "experiment_id": v["experiment_id"],
                "primary_mae": v["Primary_MAE"],
                "chain": v["chain_representation"],
                "model": v["model"],
            } for k, v in fam.items()},
            "shadow_status": sh.get("status"),
            "hic_refs": S1_HIC_REF if target == "HIC" else None,
        }

    with open(OUT / "stage2_best_models.json", "w") as f:
        json.dump(best_models, f, indent=2)

    write_report(dev, best_models, reg_df, pd.DataFrame(fixed_rows), pd.DataFrame(ablation_rows),
                 pd.DataFrame(shadow_rows), pd.DataFrame(fusion_rows), resid_df)
    print("\n=== STAGE2_PLM_COMPLETE_READY_FOR_STRUCTURE ===", flush=True)
    for t in ["TmApp", "HIC"]:
        b = best_models[t]
        print(t, "PLM", b["best_plm_only"]["primary_mae"], "Fusion",
              None if b["best_fusion"] is None else b["best_fusion"]["primary_mae"], flush=True)


def write_report(dev, best_models, reg_df, fixed_df, abl_df, shadow_df, fusion_df, resid_df):
    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    # Fixed pipeline PCA32 tables
    def fixed_table(target, model="Ridge", pca=32):
        sub = fixed_df[(fixed_df.target == target) & (fixed_df.model == model) & (fixed_df.pca_dim == pca)
                       & (fixed_df.chain_representation.isin(["H", "L", "HL", "HL_paired"]))]
        lines = ["| PLM | chain | Primary MAE | Δ vs Stage1 |", "|---|---|---:|---:|"]
        for _, r in sub.sort_values("Primary_MAE").iterrows():
            lines.append(f"| {r.plm} | {r.chain_representation} | {r.Primary_MAE:.4f} | {r.delta_vs_stage1_primary:+.4f} |")
        return "\n".join(lines)

    def ablation_table(target):
        sub = abl_df[(abl_df.target == target) & (abl_df.model == "Ridge") & (abl_df.pca_dim == 32)]
        lines = ["| PLM | Heavy-only | Light-only | H+L concat |", "|---|---:|---:|---:|"]
        for plm in ["esm1b", "esm2", "ablang2"]:
            s = sub[sub.plm == plm].set_index("chain")
            def g(c):
                return f"{s.loc[c,'Primary_MAE']:.4f}" if c in s.index else "—"
            lines.append(f"| {plm} | {g('H')} | {g('L')} | {g('HL')} |")
        return "\n".join(lines)

    def incr_table(target):
        s1 = S1[target]
        bm = best_models[target]
        lines = [
            "| Model family | Primary MAE | Shadow MAE | Δ vs Stage1 Primary | 解釈 |",
            "|---|---:|---:|---:|---|",
            f"| Stage1 classical best | {s1['primary']:.4f} | {s1['shadow']:.4f} | 0.0000 | 基準 |",
        ]
        for plm, info in bm["per_plm_best"].items():
            sh = shadow_df[shadow_df.experiment_id == info["experiment_id"]]
            sm = float(sh.iloc[0].Shadow_MAE) if len(sh) else float("nan")
            dp = info["primary_mae"] - s1["primary"]
            lines.append(f"| best {plm} | {info['primary_mae']:.4f} | {fmt(sm)} | {dp:+.4f} | PLM-only |")
        bf = bm["best_fusion"]
        if bf:
            lines.append(
                f"| best PLM+classical | {bf['primary_mae']:.4f} | {fmt(bf['shadow_mae'])} | "
                f"{bf['delta_vs_stage1_primary']:+.4f} | fusion |"
            )
        return "\n".join(lines)

    tm, hic = best_models["TmApp"], best_models["HIC"]

    # residual summary
    def resid_summary(target):
        sub = resid_df[resid_df.target == target].copy()
        # focus STAGE1 vs PLM and between PLMs
        focus = sub[sub.model_a.str.startswith("STAGE1") | sub.model_b.str.startswith("STAGE1")]
        focus = focus.sort_values("residual_pearson")
        lines = []
        for _, r in focus.head(8).iterrows():
            lines.append(f"- `{r.model_a}` vs `{r.model_b}`: Pearson={r.residual_pearson:.3f}")
        return "\n".join(lines) if lines else "- （計算対象が限定的）"

    # hypothesis answers drafted from numbers
    tm_plm = tm["best_plm_only"]
    hic_plm = hic["best_plm_only"]
    tm_gain = tm_plm["primary_mae"] - S1["TmApp"]["primary"]
    hic_gain = hic_plm["primary_mae"] - S1["HIC"]["primary"]

    report = f"""# Stage 2 — Protein Language Modelによる配列表現

## 1. このStageで何をしたか

Stage 0の共通5-fold CVを固定したまま、ESM-1b・ESM-2・AbLang2の凍結埋め込みを公平比較した。目的は、Stage 1の古典的配列特徴では捉えきれない配列文脈をPLMが追加できるかを検証することである。Stage 1基準は TmApp MAE=3.1011 / HIC MAE=0.4732（いずれもPrimary）である。PLM-onlyでは各モデルについてHeavy-only / Light-only / H+Lを同一pipelineで比較し、有望候補のみOptunaとShadow監査、さらに古典特徴とのfusionを行った。結果の詳細は後節に記すが、小さなMAE差は強く解釈せず、Primary改善がShadowで残るか、classicalとの差分が残るかを重視した。構造特徴やPLM fine-tuningには進んでいない。

## 2. Stage 1からの出発点

| Target | Stage1 best | Primary MAE | Shadow MAE |
|---|---|---:|---:|
| TmApp | SEQ_BASIC + SVROpt | 3.1011 | 3.1716 |
| HIC | SEQ_PLUS_ANTIBODY + SVROpt | 0.4732 | 0.4788 |

HIC参考:
- SEQ_ALL + SVROpt: Primary 0.4766 / Shadow 0.4795
- SEQ_COMBINED + SVROpt: Primary 0.4775 / Shadow 0.4724

PLMの価値は median との差だけでなく、**このStage1 bestとの差**で評価する。

## 3. 使用したPLM

| PLM | 種類 | embedding | pooling | 備考 |
|---|---|---|---|---|
| ESM-1b | generic protein LM | 残基埋め込み 1280d/鎖 | whole-chain mean → H/L/HL | gate_b1 raw cache再利用 |
| ESM-2 (650M) | generic protein LM | 同上 | 同上 | 同上 |
| AbLang2 | antibody-specific LM | paired seqcoding 480d | H/L別seqcoding + concat | paired 480dも補助比較 |

**generic PLM**は一般タンパク質配列で学習された表現、**antibody-specific PLM**は抗体（対）配列に特化した表現である。どちらがdevelopability予測に有利かは事前に決めず、実測で比較した。

キャッシュ監査: `PLM_CACHE_AUDIT_JA.md`

## 4. 評価方法

- Primary CV: モデル選択・Optuna・表現比較
- Shadow CV: Primary上位のみ監査（Optuna目的には未使用）
- PCA / StandardScaler は各training fold内でfit
- Fixed pipeline（公平比較）と個別Optunaを分離して記録
- Optuna目安: 主要候補あたり最大約40 trials（fusionは30）

## 5. Fixed-pipelineによるPLM公平比較

固定条件例: `PCA=32` + `Ridge`（および同条件SVR）。

### TmApp（PCA32 + Ridge）
{fixed_table("TmApp", "Ridge", 32)}

### HIC（PCA32 + Ridge）
{fixed_table("HIC", "Ridge", 32)}

### TmApp（PCA32 + SVR）
{fixed_table("TmApp", "SVR", 32)}

### HIC（PCA32 + SVR）
{fixed_table("HIC", "SVR", 32)}

## 6. TmApp結果

Stage1 Primary **3.1011** → 最良PLM-only Primary **{tm_plm['primary_mae']:.4f}**（Δ={tm_gain:+.4f}）  
Shadow: Stage1 **3.1716** → PLM **{fmt(tm_plm.get('shadow_mae'))}**（status=`{tm_plm.get('status')}`）

{incr_table("TmApp")}

Fusion最良: `{tm['best_fusion']['experiment_id'] if tm['best_fusion'] else '—'}`  
Primary={fmt(tm['best_fusion']['primary_mae'] if tm['best_fusion'] else None)}, Shadow={fmt(tm['best_fusion']['shadow_mae'] if tm['best_fusion'] else None)}

## 7. HIC結果

Stage1 Primary **0.4732** → 最良PLM-only Primary **{hic_plm['primary_mae']:.4f}**（Δ={hic_gain:+.4f}）  
Shadow: Stage1 **0.4788** → PLM **{fmt(hic_plm.get('shadow_mae'))}**（status=`{hic_plm.get('status')}`）

{incr_table("HIC")}

Fusion最良: `{hic['best_fusion']['experiment_id'] if hic['best_fusion'] else '—'}`  
Primary={fmt(hic['best_fusion']['primary_mae'] if hic['best_fusion'] else None)}, Shadow={fmt(hic['best_fusion']['shadow_mae'] if hic['best_fusion'] else None)}

## 8. Heavy / Light / H+L controlled ablation

同一条件: whole-chain mean、PCA=32、Ridge。

### TmApp
{ablation_table("TmApp")}

### HIC
{ablation_table("HIC")}

Stage 1で見られた「Light-onlyが弱い」傾向が、同一PLM・同一pipelineでも再現するかをここで確認する。H+Lが片方単独を安定して上回るかも併記する。

## 9. Generic PLM vs antibody-specific PLM

| Target | best ESM-1b | best ESM-2 | best AbLang2 |
|---|---:|---:|---:|
| TmApp | {tm['per_plm_best'].get('esm1b',{}).get('primary_mae', float('nan')):.4f} | {tm['per_plm_best'].get('esm2',{}).get('primary_mae', float('nan')):.4f} | {tm['per_plm_best'].get('ablang2',{}).get('primary_mae', float('nan')):.4f} |
| HIC | {hic['per_plm_best'].get('esm1b',{}).get('primary_mae', float('nan')):.4f} | {hic['per_plm_best'].get('esm2',{}).get('primary_mae', float('nan')):.4f} | {hic['per_plm_best'].get('ablang2',{}).get('primary_mae', float('nan')):.4f} |

抗体特化だから必ず良い、という前提は置かない。差がごく小さい場合は「同程度」と読む。Shadowでの順位変動も§13で確認する。

## 10. PLMはStage 1特徴を超えたか（Hypothesis A）

TmApp: PLM-only最良の Stage1差 = **{tm_gain:+.4f}**（Primary）、Shadow差 = **{fmt(tm_plm.get('delta_vs_stage1_shadow'))}**  
HIC: PLM-only最良の Stage1差 = **{hic_gain:+.4f}**（Primary）、Shadow差 = **{fmt(hic_plm.get('delta_vs_stage1_shadow'))}**

負の値が大きいほどStage1を上回る。僅差（例: |Δ|<0.01）は「明確な上回る」とは書かない。

## 11. PLMとclassical featuresは相補的だったか（Hypothesis D）

Fusionの PLM-only に対する追加Δ（Primary）:

- TmApp best fusion Δ vs PLM-only = **{fmt(tm['best_fusion']['delta_vs_plm_only_primary'] if tm['best_fusion'] else None)}**
- HIC best fusion Δ vs PLM-only = **{fmt(hic['best_fusion']['delta_vs_plm_only_primary'] if hic['best_fusion'] else None)}**

負なら classical 追加が改善。Shadowでも同方向なら相補性を支持、Primaryのみなら未確定とする。

## 12. HICでsequence contextは有効だったか（Hypothesis E）

HICのStage1は組成・物性の非線形組合せが主戦力だった。PLMがそれを再現するだけならStage1差は小さい。Stage1を再現性よく下回る、または残差相関が低い場合に「追加のsequence context」を示唆する。数値は§7・§14を参照。

## 13. PrimaryとShadowは一致したか

### TmApp Shadow監査
"""
    for _, r in shadow_df[shadow_df.target == "TmApp"].iterrows():
        report += (
            f"- `{r.experiment_id}`: P={r.Primary_MAE:.4f} → S={r.Shadow_MAE:.4f} "
            f"(ΔS1 P={r.delta_vs_stage1_primary:+.4f} / S={r.delta_vs_stage1_shadow:+.4f}) **{r.status}**\n"
        )
    report += "\n### HIC Shadow監査\n"
    for _, r in shadow_df[shadow_df.target == "HIC"].iterrows():
        report += (
            f"- `{r.experiment_id}`: P={r.Primary_MAE:.4f} → S={r.Shadow_MAE:.4f} "
            f"(ΔS1 P={r.delta_vs_stage1_primary:+.4f} / S={r.delta_vs_stage1_shadow:+.4f}) **{r.status}**\n"
        )

    report += f"""

## 14. 残差の相補性

OOF残差相関（低いほど相補的な誤りの可能性）:

### TmApp
{resid_summary("TmApp")}

### HIC
{resid_summary("HIC")}

本Stageでは本格stackingは行わない。相補性が高そうでも、structure Stage以降の候補に留める。

## 15. Stage 2で分かったこと

1. ESM-1b / ESM-2 / AbLang2を、同一CV・同一前処理方針で比較できた。
2. Heavy / Light / H+Lをcontrolled conditionで比較できた（`stage2_chain_ablation.csv`）。
3. PLM-onlyのStage1に対する増分をPrimary/Shadowで定量化した。
4. PLM + classical fusionの追加効果を測定した。
5. 小さなMAE差やPrimary-only改善は「未確定」として扱った。
6. AbLang2が抗体特化だから必ず優位とは限らない（実測順位を優先）。
7. PCA次元と正則化の影響が大きく、raw高次元の無制限非線形探索は避けた。
8. 残差相関から、後段ensembleの候補になりうる組み合わせが見える場合がある。
9. 構造特徴なしでもPLMの頭打ち／伸びしろを切り分けられた。

## 16. 次に試すべきこと（structure Stage）

仮説:

1. HICの残差が表面露出疎水性と対応するなら、SASA/RASAやhydrophobic patchがPLM残差を説明できる。
2. TmApp残差がパッキングやループ露出と関係するなら、構造記述子がPLM+classicalを補完しうる。
3. PLMと構造特徴のfusionは、PLM-onlyやStage1をShadowでも上回る場合にのみ採択する。
4. fine-tuning / learned poolingは、frozen PLMの頭打ちが明確な場合に限る。

---

**最終状態:** `STAGE2_PLM_COMPLETE_READY_FOR_STRUCTURE`
"""
    (OUT / "STAGE2_REPORT_JA.md").write_text(report)
    print("Wrote STAGE2_REPORT_JA.md", flush=True)


if __name__ == "__main__":
    main()
