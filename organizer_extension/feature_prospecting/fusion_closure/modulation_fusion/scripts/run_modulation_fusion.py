#!/usr/bin/env python3
"""HIC AROMATIC modulation fusion audit (Gates M1–M5)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "fusion_closure/modulation_fusion"
RES = OUT / "results"
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    B_BOOT,
    HIC_REF,
    OOF_DIR,
    bootstrap_delta,
    mae,
    metrics,
)

EMB_PATH = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
PRIMARY_FOLDS = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW_FOLDS = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
SOL_PATH = ROOT / "competition/data/secret/solution.csv"
AROMA_DIR = FP / "AROMATIC-TOPO"
GENS = ["esmfold", "abodybuilder2", "boltz2"]
GEN_LABEL = {"esmfold": "ESMFold", "abodybuilder2": "ABB2", "boltz2": "Boltz2"}
PCA_N = 32
RIDGE_A = 100.0
SEEDS = [11, 23, 42, 77, 101]
LR = 1e-3
WD = 1e-2
EPOCHS = 1000
EN_L1 = [0.1, 0.5, 0.9]
EN_ALPHA = [0.001, 0.01, 0.1, 1.0]
PCA_SEED = 42

FEATURE_META = {
    "exposed_TYR_count": ("Exposed Tyr count (SASA threshold)", "count"),
    "exposed_TRP_count": ("Exposed Trp count", "count"),
    "exposed_PHE_count": ("Exposed Phe count", "count"),
    "exposed_aromatic_total_count": ("Exposed Y+W+F count", "count"),
    "aromatic_exposed_SASA_total": ("Total exposed aromatic SASA", "Angstrom^2"),
    "aromatic_exposed_SASA_fraction": ("Exposed aromatic SASA / total SASA", "fraction"),
    "strongly_exposed_aromatic_count": ("Strongly exposed aromatic residue count", "count"),
    "strongly_exposed_aromatic_SASA": ("Strongly exposed aromatic SASA", "Angstrom^2"),
    "CDR_exposed_aromatic_count": ("CDR exposed aromatic count", "count"),
    "CDR_aromatic_SASA": ("CDR aromatic exposed SASA", "Angstrom^2"),
    "CDR_aromatic_fraction": ("CDR aromatic SASA / aromatic exposed SASA", "fraction"),
    "aromatic_patch_count": ("Number of aromatic surface patches", "count"),
    "largest_aromatic_patch_n_res": ("Residues in largest aromatic patch", "count"),
    "largest_aromatic_patch_exposed_SASA": ("Exposed SASA of largest aromatic patch", "Angstrom^2"),
    "max_local_aromatic_SASA": ("Max local-neighborhood aromatic SASA", "Angstrom^2"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_canonical():
    return list(json.loads((AROMA_DIR / "FEATURE_SPEC.json").read_text())["canonical_features"])


def load_physics(gen: str) -> pd.DataFrame:
    cols = load_canonical()
    df = pd.read_parquet(AROMA_DIR / f"features_{gen}.parquet")
    df = df[df["extraction_status"].astype(str).str.upper().isin(["SUCCESS", "OK"])]
    return df.drop_duplicates("id").set_index("id")[cols].astype(float)


def load_esm2() -> pd.DataFrame:
    z = np.load(EMB_PATH, allow_pickle=True)
    return pd.DataFrame(np.asarray(z["esm2__H"], float), index=[str(x) for x in z["ids"]])


def fill_nan_train_median(Xtr, Xte):
    Xtr = np.asarray(Xtr, float).copy()
    Xte = np.asarray(Xte, float).copy()
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte


def make_zx(X_plm_tr, X_plm_te, Xp_tr, Xp_te):
    """P0-style blocks: z (32 scaled PCs), x (15 scaled aromatic)."""
    sc0 = StandardScaler()
    Z0tr = sc0.fit_transform(X_plm_tr)
    Z0te = sc0.transform(X_plm_te)
    n = min(PCA_N, Z0tr.shape[0] - 1, Z0tr.shape[1])
    pca = PCA(n_components=n, whiten=False, random_state=PCA_SEED)
    Ptr = pca.fit_transform(Z0tr)
    Pte = pca.transform(Z0te)
    scz = StandardScaler()
    z_tr = scz.fit_transform(Ptr)
    z_te = scz.transform(Pte)
    Xp_tr, Xp_te = fill_nan_train_median(Xp_tr, Xp_te)
    scx = StandardScaler()
    x_tr = scx.fit_transform(Xp_tr)
    x_te = scx.transform(Xp_te)
    return z_tr, z_te, x_tr, x_te, Xp_tr  # last is imputed raw train for interpretation


def bilinear_features(z, x):
    n, d = z.shape
    p = x.shape[1]
    inter = (z[:, :, None] * x[:, None, :]).reshape(n, d * p)
    return np.hstack([z, x, inter])


# -------------------- numpy AdamW --------------------
def adamw_step(param, grad, m, v, t, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, wd=WD):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    mhat = m / (1 - beta1**t)
    vhat = v / (1 - beta2**t)
    param = param - lr * mhat / (np.sqrt(vhat) + eps)
    param = param * (1 - lr * wd)  # decoupled weight decay on weights
    return param, m, v


def param_count_m3(r, d=32, p=15):
    # Wp dxr no bias, Wg r->d + bias, head d->1 + bias
    return p * r + (r * d + d) + (d + 1)


def param_count_m4(r, d=32, p=15):
    return p * r + 2 * (r * d + d) + (d + 1)


def param_count_add(r, d=32, p=15):
    return p * r + (r * d + d) + (d + 1)


class ModModel:
    """Linear low-rank modulation; modes: mult, film, add."""

    def __init__(self, mode: str, rank: int, d=32, p=15, seed=42):
        self.mode = mode
        self.rank = rank
        self.d = d
        self.p = p
        rng = np.random.default_rng(seed)
        self.Wp = rng.normal(0, 0.01, size=(p, rank))
        # gamma near zero (unused in add-only)
        self.Wg = rng.normal(0, 1e-3, size=(rank, d)) if mode != "add" else None
        self.bg = np.zeros(d) if mode != "add" else None
        if mode == "film":
            self.Wb = rng.normal(0, 1e-3, size=(rank, d))
            self.bb = np.zeros(d)
        elif mode == "add":
            self.Wb = rng.normal(0, 1e-3, size=(rank, d))
            self.bb = np.zeros(d)
        else:
            self.Wb = None
            self.bb = None
        self.w = rng.normal(0, 0.01, size=(d,))
        self.b = 0.0

    def forward(self, z, x):
        p = x @ self.Wp  # (n,r)
        if self.mode == "mult":
            gamma = p @ self.Wg + self.bg
            h = (1.0 + gamma) * z
        elif self.mode == "film":
            gamma = p @ self.Wg + self.bg
            beta = p @ self.Wb + self.bb
            h = (1.0 + gamma) * z + beta
        else:  # add
            beta = p @ self.Wb + self.bb
            h = z + beta
        return (h @ self.w) + self.b, h, p

    def fit(self, z, x, y, epochs=EPOCHS):
        # AdamW state
        keys = ["Wp", "w", "b"]
        if self.mode != "add":
            keys += ["Wg", "bg"]
        if self.mode in ("film", "add"):
            keys += ["Wb", "bb"]
        state = {}
        for k in keys:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                state[k] = {"m": np.zeros_like(v), "v": np.zeros_like(v)}
            else:
                state[k] = {"m": 0.0, "v": 0.0}

        n = len(y)
        for epoch in range(1, epochs + 1):
            yhat, h, p = self.forward(z, x)
            err = (yhat - y) / n  # dL/dyhat for mean MSE: 2* mean((yhat-y)) but use MSE grad = 2/n * (yhat-y); scale into 2
            # use MSE: L = mean((yhat-y)^2); dL/dyhat = 2/n (yhat-y)
            g_yhat = 2.0 * (yhat - y) / n
            # head
            g_w = h.T @ g_yhat
            g_b = float(np.sum(g_yhat))
            g_h = np.outer(g_yhat, self.w)  # (n,d)
            if self.mode == "mult":
                g_gamma = g_h * z
                g_z_unused = g_h * (1.0 + (p @ self.Wg + self.bg))  # not needed
                g_Wg = p.T @ g_gamma
                g_bg = g_gamma.sum(axis=0)
                g_p = g_gamma @ self.Wg.T
                g_Wp = x.T @ g_p
                grads = {"Wp": g_Wp, "Wg": g_Wg, "bg": g_bg, "w": g_w, "b": g_b}
            elif self.mode == "film":
                gamma = p @ self.Wg + self.bg
                g_gamma = g_h * z
                g_beta = g_h
                g_Wg = p.T @ g_gamma
                g_bg = g_gamma.sum(axis=0)
                g_Wb = p.T @ g_beta
                g_bb = g_beta.sum(axis=0)
                g_p = g_gamma @ self.Wg.T + g_beta @ self.Wb.T
                g_Wp = x.T @ g_p
                grads = {"Wp": g_Wp, "Wg": g_Wg, "bg": g_bg, "Wb": g_Wb, "bb": g_bb, "w": g_w, "b": g_b}
            else:
                g_beta = g_h
                g_Wb = p.T @ g_beta
                g_bb = g_beta.sum(axis=0)
                g_p = g_beta @ self.Wb.T
                g_Wp = x.T @ g_p
                grads = {"Wp": g_Wp, "Wb": g_Wb, "bb": g_bb, "w": g_w, "b": g_b}
                # zero gamma grads placeholders not in keys

            # apply AdamW (no WD on biases)
            for k, g in grads.items():
                cur = getattr(self, k)
                if k in ("bg", "bb", "b"):
                    # bias: Adam without WD
                    st = state[k]
                    if isinstance(cur, np.ndarray):
                        m = 0.9 * st["m"] + 0.1 * g
                        v = 0.999 * st["v"] + 0.001 * (g * g)
                        mhat = m / (1 - 0.9**epoch)
                        vhat = v / (1 - 0.999**epoch)
                        cur = cur - LR * mhat / (np.sqrt(vhat) + 1e-8)
                        state[k] = {"m": m, "v": v}
                        setattr(self, k, cur)
                    else:
                        m = 0.9 * st["m"] + 0.1 * g
                        v = 0.999 * st["v"] + 0.001 * (g * g)
                        mhat = m / (1 - 0.9**epoch)
                        vhat = v / (1 - 0.999**epoch)
                        cur = cur - LR * mhat / (np.sqrt(vhat) + 1e-8)
                        state[k] = {"m": m, "v": v}
                        setattr(self, k, cur)
                else:
                    st = state[k]
                    m, vv = st["m"], st["v"]
                    new, m, vv = adamw_step(cur, g, m, vv, epoch)
                    state[k] = {"m": m, "v": vv}
                    setattr(self, k, new)

        yhat, _, _ = self.forward(z, x)
        return float(mae(y, yhat))


def select_elasticnet(Xtr, ytr, groups):
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    best = None
    best_mae = np.inf
    for l1 in EN_L1:
        for a in EN_ALPHA:
            maes = []
            for tr, va in gkf.split(Xtr, ytr, groups):
                sc = StandardScaler()
                m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000, random_state=0)
                m.fit(sc.fit_transform(Xtr[tr]), ytr[tr])
                maes.append(mae(ytr[va], m.predict(sc.transform(Xtr[va]))))
            score = float(np.mean(maes))
            if score < best_mae:
                best_mae = score
                best = (a, l1)
    return best


def align_Wp_signs(Wp_list):
    """Align factor signs across models to first reference."""
    ref = Wp_list[0].copy()
    out = [ref]
    for W in Wp_list[1:]:
        W2 = W.copy()
        for r in range(W.shape[1]):
            if np.dot(W2[:, r], ref[:, r]) < 0:
                W2[:, r] *= -1
        out.append(W2)
    return out


def run_fold_models(z_tr, z_te, x_tr, x_te, y_tr, y_te, groups_tr, feat_names, fixed_en_hp=None):
    """Fit all models on one outer fold; return dict of test preds and diagnostics."""
    out = {}
    # B0
    m = Ridge(alpha=RIDGE_A, random_state=0)
    m.fit(z_tr, y_tr)
    out["B0_PLM_ONLY"] = {"pred_te": m.predict(z_te), "pred_tr": m.predict(z_tr), "n_params": 32 + 1}

    # B1
    sc = StandardScaler()
    Xtr = sc.fit_transform(np.hstack([z_tr, x_tr]))
    Xte = sc.transform(np.hstack([z_te, x_te]))
    m = Ridge(alpha=RIDGE_A, random_state=0)
    m.fit(Xtr, y_tr)
    out["B1_CURRENT_CONCAT"] = {"pred_te": m.predict(Xte), "pred_tr": m.predict(Xtr), "n_params": 47 + 1, "coef": m.coef_.copy()}

    # M1 single-feature
    m1_single = {}
    for j, name in enumerate(feat_names):
        scj = StandardScaler()
        Xjtr = scj.fit_transform(np.hstack([z_tr, x_tr[:, [j]]]))
        Xjte = scj.transform(np.hstack([z_te, x_te[:, [j]]]))
        m = Ridge(alpha=RIDGE_A, random_state=0)
        m.fit(Xjtr, y_tr)
        m1_single[name] = m.predict(Xjte)
    out["M1_SINGLE"] = m1_single

    # M1 LOO
    m1_loo = {}
    for j, name in enumerate(feat_names):
        keep = [k for k in range(len(feat_names)) if k != j]
        scl = StandardScaler()
        Xltr = scl.fit_transform(np.hstack([z_tr, x_tr[:, keep]]))
        Xlte = scl.transform(np.hstack([z_te, x_te[:, keep]]))
        m = Ridge(alpha=RIDGE_A, random_state=0)
        m.fit(Xltr, y_tr)
        m1_loo[name] = m.predict(Xlte)
    out["M1_LOO"] = m1_loo

    # M2 ElasticNet
    Btr = bilinear_features(z_tr, x_tr)
    Bte = bilinear_features(z_te, x_te)
    if fixed_en_hp is None:
        a, l1 = select_elasticnet(Btr, y_tr, groups_tr)
    else:
        a, l1 = fixed_en_hp
    scb = StandardScaler()
    m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000, random_state=0)
    m.fit(scb.fit_transform(Btr), y_tr)
    coef = m.coef_.copy()
    # interaction block starts at 32+15=47
    C = coef[47:].reshape(32, 15)
    out["M2_FULL_BILINEAR_ELASTICNET"] = {
        "pred_te": m.predict(scb.transform(Bte)),
        "pred_tr": m.predict(scb.transform(Btr)),
        "hp": (a, l1),
        "coef": coef,
        "C": C,
        "n_params": int(np.sum(np.abs(coef) > 1e-12)) ,  # effective nonzero
        "n_params_full": 527 + 1,
        "interaction_l2": np.linalg.norm(C, axis=0),
    }

    # Neural M3/M4
    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std() + 1e-12)
    ytr_s = (y_tr - y_mu) / y_sd

    def run_neural(mode, rank, name):
        preds_te = []
        preds_tr = []
        train_maes = []
        seed_maes = []
        Wps = []
        weight_norms = []
        for seed in SEEDS:
            model = ModModel(mode, rank, seed=seed)
            model.fit(z_tr, x_tr, ytr_s)
            pred_tr_s, _, _ = model.forward(z_tr, x_tr)
            pred_te_s, _, _ = model.forward(z_te, x_te)
            pred_tr = pred_tr_s * y_sd + y_mu
            pred_te = pred_te_s * y_sd + y_mu
            preds_te.append(pred_te)
            preds_tr.append(pred_tr)
            train_maes.append(mae(y_tr, pred_tr))
            seed_maes.append(mae(y_te, pred_te))
            Wps.append(model.Wp.copy())
            wn = float(np.linalg.norm(model.Wp))
            if model.Wg is not None:
                wn += float(np.linalg.norm(model.Wg))
            if model.Wb is not None:
                wn += float(np.linalg.norm(model.Wb))
            weight_norms.append(wn)
        preds_te = np.mean(np.stack(preds_te, 0), 0)
        preds_tr = np.mean(np.stack(preds_tr, 0), 0)
        Wps_aligned = align_Wp_signs(Wps)
        Wp_mean = np.mean(Wps_aligned, axis=0)
        if mode == "film":
            n_params = param_count_m4(rank)
        elif mode == "add":
            n_params = param_count_add(rank)
        else:
            n_params = param_count_m3(rank)
        return {
            "pred_te": preds_te,
            "pred_tr": preds_tr,
            "train_mae": float(np.mean(train_maes)),
            "seed_mae_sd": float(np.std(seed_maes, ddof=0)),
            "seed_maes": seed_maes,
            "Wp_mean": Wp_mean,
            "weight_norm_mean": float(np.mean(weight_norms)),
            "n_params": n_params,
        }

    for r in [1, 2, 4]:
        out[f"M3_MULTIPLICATIVE_R{r}"] = run_neural("mult", r, f"M3_R{r}")
        out[f"M4_FILM_R{r}"] = run_neural("film", r, f"M4_R{r}")
    out["M4_ADD_ONLY_R2"] = run_neural("add", 2, "ADD")
    return out


def oof_collect(X_plm, Xp, y, groups, feat_names, fixed_en_hp=None):
    """Run all models with outer CV; return predictions dict id-aligned arrays."""
    n = len(y)
    model_names = [
        "B0_PLM_ONLY",
        "B1_CURRENT_CONCAT",
        "M2_FULL_BILINEAR_ELASTICNET",
        "M3_MULTIPLICATIVE_R1",
        "M3_MULTIPLICATIVE_R2",
        "M3_MULTIPLICATIVE_R4",
        "M4_FILM_R1",
        "M4_FILM_R2",
        "M4_FILM_R4",
        "M4_ADD_ONLY_R2",
    ]
    preds = {m: np.full(n, np.nan) for m in model_names}
    train_preds = {m: np.full(n, np.nan) for m in model_names}
    fold_diag = []
    m1_single_oof = {f: np.full(n, np.nan) for f in feat_names}
    m1_loo_oof = {f: np.full(n, np.nan) for f in feat_names}
    en_hps = []
    inter_imps = []
    Wp_records = []

    for f in sorted(set(groups.tolist())):
        te = np.where(groups == f)[0]
        tr = np.where(groups != f)[0]
        print(f"  outer fold {f} n_tr={len(tr)} n_te={len(te)}", flush=True)
        z_tr, z_te, x_tr, x_te, _ = make_zx(X_plm[tr], X_plm[te], Xp[tr], Xp[te])
        res = run_fold_models(z_tr, z_te, x_tr, x_te, y[tr], y[te], groups[tr], feat_names, fixed_en_hp=fixed_en_hp)
        for m in model_names:
            preds[m][te] = res[m]["pred_te"]
            if "pred_tr" in res[m]:
                # store train preds only for gap audit on neural / main
                pass
        for name in feat_names:
            m1_single_oof[name][te] = res["M1_SINGLE"][name]
            m1_loo_oof[name][te] = res["M1_LOO"][name]
        en_hps.append({"fold": int(f), "alpha": res["M2_FULL_BILINEAR_ELASTICNET"]["hp"][0], "l1_ratio": res["M2_FULL_BILINEAR_ELASTICNET"]["hp"][1]})
        inter_imps.append(
            {
                "fold": int(f),
                **{f"imp_{feat_names[j]}": float(res["M2_FULL_BILINEAR_ELASTICNET"]["interaction_l2"][j]) for j in range(len(feat_names))},
            }
        )
        for mname in model_names:
            if mname.startswith("M3") or mname.startswith("M4"):
                train_mae = res[mname]["train_mae"]
                val_mae = mae(y[te], res[mname]["pred_te"])
                fold_diag.append(
                    {
                        "fold": int(f),
                        "model": mname,
                        "train_mae": train_mae,
                        "val_mae": val_mae,
                        "gap": train_mae - val_mae,  # negative gap = train better
                        "seed_mae_sd": res[mname]["seed_mae_sd"],
                        "weight_norm_mean": res[mname]["weight_norm_mean"],
                        "n_params": res[mname]["n_params"],
                        "pred_sd_over_true_sd": float(np.std(res[mname]["pred_te"]) / (np.std(y[te]) + 1e-12)),
                    }
                )
                if "Wp_mean" in res[mname]:
                    for j, fname in enumerate(feat_names):
                        for r in range(res[mname]["Wp_mean"].shape[1]):
                            Wp_records.append(
                                {
                                    "fold": int(f),
                                    "model": mname,
                                    "feature": fname,
                                    "rank_factor": r,
                                    "loading": float(res[mname]["Wp_mean"][j, r]),
                                }
                            )
        # train MAE for B0/B1/M2 on outer train
        for mname in ["B0_PLM_ONLY", "B1_CURRENT_CONCAT", "M2_FULL_BILINEAR_ELASTICNET"]:
            # refit already done; use pred_tr
            fold_diag.append(
                {
                    "fold": int(f),
                    "model": mname,
                    "train_mae": mae(y[tr], res[mname]["pred_tr"]),
                    "val_mae": mae(y[te], res[mname]["pred_te"]),
                    "gap": mae(y[tr], res[mname]["pred_tr"]) - mae(y[te], res[mname]["pred_te"]),
                    "seed_mae_sd": np.nan,
                    "weight_norm_mean": np.nan,
                    "n_params": res[mname]["n_params"] if mname != "M2_FULL_BILINEAR_ELASTICNET" else res[mname]["n_params_full"],
                    "pred_sd_over_true_sd": float(np.std(res[mname]["pred_te"]) / (np.std(y[te]) + 1e-12)),
                }
            )

    return preds, m1_single_oof, m1_loo_oof, en_hps, inter_imps, fold_diag, Wp_records


def eval_split(y, pred, ref_plm, ref_concat):
    m = metrics(y, pred)
    d_plm = bootstrap_delta(y, pred, ref_plm, b=B_BOOT)
    d_cat = bootstrap_delta(y, pred, ref_concat, b=B_BOOT)
    return {
        "MAE": m["MAE"],
        "Pearson": m["Pearson"],
        "Spearman": m["Spearman"],
        "delta_vs_PLM": m["MAE"] - mae(y, ref_plm),
        "delta_vs_CONCAT": m["MAE"] - mae(y, ref_concat),
        "boot_vs_PLM_lo": d_plm["ci95_low"],
        "boot_vs_PLM_hi": d_plm["ci95_high"],
        "boot_vs_PLM_p_improve": d_plm["p_improve"],
        "boot_vs_CONCAT_lo": d_cat["ci95_low"],
        "boot_vs_CONCAT_hi": d_cat["ci95_high"],
        "boot_vs_CONCAT_p_improve": d_cat["p_improve"],
    }


def fit_predict_test(X_plm_tr, X_plm_te, Xp_tr, Xp_te, y_tr, groups_tr, feat_names, model_name, fixed_en_hp=None):
    z_tr, z_te, x_tr, x_te, _ = make_zx(X_plm_tr, X_plm_te, Xp_tr, Xp_te)
    # lightweight path for test: reuse run_fold_models but y_te dummy
    y_te_dummy = np.zeros(len(z_te))
    res = run_fold_models(z_tr, z_te, x_tr, x_te, y_tr, y_te_dummy, groups_tr, feat_names, fixed_en_hp=fixed_en_hp)
    return res[model_name]["pred_te"]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / "MODULATION_FUSION_SPEC.json"
    (OUT / "MODULATION_FUSION_SPEC_HASH.json").write_text(
        json.dumps(
            {
                "MODULATION_FUSION_SPEC_FROZEN_BEFORE_SCORING": True,
                "spec_sha256": sha256(spec_path),
                "note": "hashed before target scoring",
            },
            indent=2,
        )
        + "\n"
    )

    feat_names = load_canonical()
    esm = load_esm2()
    y_dev = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_true"].astype(float)
    sol = pd.read_csv(SOL_PATH)
    y_te_all = sol.set_index("id")["HIC"].astype(float)
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    folds_p = pd.read_csv(PRIMARY_FOLDS)
    folds_s = pd.read_csv(SHADOW_FOLDS)

    registry = []
    all_perf = []
    all_m1 = []

    for gen in GENS:
        print(f"==== {GEN_LABEL[gen]} ====", flush=True)
        phys = load_physics(gen)
        ids = [i for i in folds_p["id"].tolist() if i in esm.index and i in phys.index and i in y_dev.index]
        fold_df = folds_p[folds_p["id"].isin(ids)].reset_index(drop=True)
        ids = fold_df["id"].tolist()
        groups = fold_df["fold"].to_numpy()
        X_plm = esm.loc[ids].to_numpy()
        Xp = phys.loc[ids].to_numpy()
        y = y_dev.loc[ids].to_numpy()

        preds, m1s, m1l, en_hps, inter_imps, fold_diag, Wp_records = oof_collect(
            X_plm, Xp, y, groups, feat_names, fixed_en_hp=None
        )
        pd.DataFrame(en_hps).to_csv(RES / f"m2_hps_{gen}.csv", index=False)
        pd.DataFrame(inter_imps).to_csv(RES / f"m2_interaction_importance_{gen}.csv", index=False)
        pd.DataFrame(fold_diag).to_csv(RES / f"fold_diagnostics_{gen}.csv", index=False)
        pd.DataFrame(Wp_records).to_csv(RES / f"wp_loadings_{gen}.csv", index=False)

        # modal EN hp for shadow / test
        hp_df = pd.DataFrame(en_hps)
        # mode of (alpha, l1)
        pairs = list(zip(hp_df["alpha"], hp_df["l1_ratio"]))
        modal = max(set(pairs), key=pairs.count)
        fixed_en = modal

        # M1 summaries
        full_mae = mae(y, preds["B1_CURRENT_CONCAT"])
        for name in feat_names:
            meaning, units = FEATURE_META[name]
            d_single = mae(y, m1s[name]) - mae(y, preds["B0_PLM_ONLY"])
            loo_imp = mae(y, m1l[name]) - full_mae  # positive => feature helps full model
            # public/private for single
            # need test preds for M1 - skip heavy; use CV-focused + optional later
            all_m1.append(
                {
                    "generator": GEN_LABEL[gen],
                    "feature": name,
                    "meaning": meaning,
                    "units": units,
                    "single_delta_vs_PLM_CV": d_single,
                    "loo_MAE_without_minus_full_CV": loo_imp,
                    "single_CV_MAE": mae(y, m1s[name]),
                }
            )

        # Test predictions for main models
        te_ids = [i for i in y_te_all.index if i in esm.index and i in phys.index]
        X_plm_te = esm.loc[te_ids].to_numpy()
        Xp_te = phys.loc[te_ids].to_numpy()
        y_test = y_te_all.loc[te_ids].to_numpy()
        te_index = {a: i for i, a in enumerate(te_ids)}
        pub_g = [i for i in pub if i in te_index]
        pri_g = [i for i in pri if i in te_index]

        model_names = list(preds.keys())
        test_preds = {}
        print("  fitting Dev->Test for main models", flush=True)
        z_tr, z_te, x_tr, x_te, _ = make_zx(X_plm, X_plm_te, Xp, Xp_te)
        res_full = run_fold_models(z_tr, z_te, x_tr, x_te, y, y_test, groups, feat_names, fixed_en_hp=fixed_en)
        for m in model_names:
            test_preds[m] = res_full[m]["pred_te"]
        # M1 test for report
        for name in feat_names:
            j = feat_names.index(name)
            # already in res_full M1
            pass
        m1_test = res_full["M1_SINGLE"]
        m1_loo_test = res_full["M1_LOO"]
        for name in feat_names:
            meaning, units = FEATURE_META[name]
            # update public private for single
            idx_pub = [te_index[i] for i in pub_g]
            idx_pri = [te_index[i] for i in pri_g]
            row = next(r for r in all_m1 if r["generator"] == GEN_LABEL[gen] and r["feature"] == name)
            row["single_delta_vs_PLM_Public"] = mae(y_test[idx_pub], m1_test[name][idx_pub]) - mae(
                y_test[idx_pub], test_preds["B0_PLM_ONLY"][idx_pub]
            )
            row["single_delta_vs_PLM_Private"] = mae(y_test[idx_pri], m1_test[name][idx_pri]) - mae(
                y_test[idx_pri], test_preds["B0_PLM_ONLY"][idx_pri]
            )
            boot = bootstrap_delta(y, m1s[name], preds["B0_PLM_ONLY"], b=B_BOOT)
            row["single_boot_vs_PLM_lo"] = boot["ci95_low"]
            row["single_boot_vs_PLM_hi"] = boot["ci95_high"]

        # Shadow CV
        ids_s = [i for i in folds_s["id"].tolist() if i in esm.index and i in phys.index and i in y_dev.index]
        fold_s = folds_s[folds_s["id"].isin(ids_s)].reset_index(drop=True)
        ids_s = fold_s["id"].tolist()
        groups_s = fold_s["fold"].to_numpy()
        X_plm_s = esm.loc[ids_s].to_numpy()
        Xp_s = phys.loc[ids_s].to_numpy()
        y_s = y_dev.loc[ids_s].to_numpy()
        print("  Shadow CV", flush=True)
        preds_s, _, _, _, _, _, _ = oof_collect(X_plm_s, Xp_s, y_s, groups_s, feat_names, fixed_en_hp=fixed_en)

        for m in model_names:
            registry.append({"model": m, "generator": GEN_LABEL[gen], "status": "scored"})
            ev_cv = eval_split(y, preds[m], preds["B0_PLM_ONLY"], preds["B1_CURRENT_CONCAT"])
            idx_pub = [te_index[i] for i in pub_g]
            idx_pri = [te_index[i] for i in pri_g]
            ev_pub = eval_split(
                y_test[idx_pub],
                test_preds[m][idx_pub],
                test_preds["B0_PLM_ONLY"][idx_pub],
                test_preds["B1_CURRENT_CONCAT"][idx_pub],
            )
            ev_pri = eval_split(
                y_test[idx_pri],
                test_preds[m][idx_pri],
                test_preds["B0_PLM_ONLY"][idx_pri],
                test_preds["B1_CURRENT_CONCAT"][idx_pri],
            )
            shadow_mae = mae(y_s, preds_s[m])
            shadow_d_concat = shadow_mae - mae(y_s, preds_s["B1_CURRENT_CONCAT"])
            pos = ev_cv["delta_vs_CONCAT"] < 0 and ev_pub["delta_vs_CONCAT"] < 0 and ev_pri["delta_vs_CONCAT"] < 0
            all_perf.append(
                {
                    "generator": GEN_LABEL[gen],
                    "model": m,
                    "CV_MAE": ev_cv["MAE"],
                    "delta_vs_PLM_CV": ev_cv["delta_vs_PLM"],
                    "delta_vs_CONCAT_CV": ev_cv["delta_vs_CONCAT"],
                    "Public_MAE": ev_pub["MAE"],
                    "delta_vs_CONCAT_Public": ev_pub["delta_vs_CONCAT"],
                    "Private_MAE": ev_pri["MAE"],
                    "delta_vs_CONCAT_Private": ev_pri["delta_vs_CONCAT"],
                    "Shadow_MAE": shadow_mae,
                    "delta_vs_CONCAT_Shadow": shadow_d_concat,
                    "boot_vs_CONCAT_CV_lo": ev_cv["boot_vs_CONCAT_lo"],
                    "boot_vs_CONCAT_CV_hi": ev_cv["boot_vs_CONCAT_hi"],
                    "boot_vs_CONCAT_CV_p_improve": ev_cv["boot_vs_CONCAT_p_improve"],
                    "boot_vs_PLM_CV_lo": ev_cv["boot_vs_PLM_lo"],
                    "boot_vs_PLM_CV_hi": ev_cv["boot_vs_PLM_hi"],
                    "reproducible_vs_CONCAT": bool(pos),
                    "fixed_en_hp": str(fixed_en),
                }
            )

        # save oof
        np.savez_compressed(
            RES / f"oof_{gen}.npz",
            ids=np.asarray(ids),
            y=y,
            **{m: preds[m] for m in model_names},
        )

    pd.DataFrame(registry).to_csv(OUT / "MODULATION_MODEL_REGISTRY.csv", index=False)
    pd.DataFrame(all_perf).to_csv(RES / "PRIMARY_RESULTS.csv", index=False)
    pd.DataFrame(all_m1).to_csv(RES / "M1_FEATURE_IMPORTANCE.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
