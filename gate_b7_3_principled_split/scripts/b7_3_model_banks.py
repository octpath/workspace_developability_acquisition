"""Gate B7.3 model banks + safety evaluation (post-finalist freeze only)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from b7_3_common import (
    BANK_A,
    CACHE,
    CFG,
    PRED_FINAL,
    PRED_OOF,
    ROOT,
    SEED_BASE,
    log,
    mae,
    spearman_safe,
    write_json,
)

sys.path.insert(0, str(ROOT / "gate_b4_absolute/scripts"))
from b4_common import load_bio, load_representation  # noqa: E402

FEAT_CACHE: dict = {}
BIO_CAT: list | None = None


def remap_b5_predictions(target: str, tag: str, public_ids, private_ids, role_pub, role_priv):
    pub = np.load(PRED_FINAL / f"{target}__{tag}__public.npy").astype(float)
    priv = np.load(PRED_FINAL / f"{target}__{tag}__private.npy").astype(float)
    by_id = dict(zip(role_pub, pub))
    by_id.update(dict(zip(role_priv, priv)))
    return (
        np.array([by_id[i] for i in public_ids], float),
        np.array([by_id[i] for i in private_ids], float),
    )


def build_id_pred_map(target: str, tag: str, role_pub, role_priv) -> dict[str, float]:
    pub = np.load(PRED_FINAL / f"{target}__{tag}__public.npy").astype(float)
    priv = np.load(PRED_FINAL / f"{target}__{tag}__private.npy").astype(float)
    by_id = dict(zip(role_pub, pub.tolist()))
    by_id.update(dict(zip(role_priv, priv.tolist())))
    return by_id


def load_b5_oof(target: str, tag: str) -> np.ndarray:
    return np.load(PRED_OOF / f"{target}__{tag}__meanOOF.npy").astype(float)


def get_X(tag: str, ids: list) -> np.ndarray:
    key = (tag, tuple(ids))
    if key in FEAT_CACHE:
        return FEAT_CACHE[key]
    if tag == "CONST":
        X = np.ones((len(ids), 1))
    elif tag == "BIO":
        X = load_bio(ids, category_ids=BIO_CAT)
    elif tag == "BIO_ABLANG2":
        X = np.hstack([load_bio(ids, category_ids=BIO_CAT), load_representation("PLM_ABLANG2", ids)])
    elif tag == "PHYS_STRUCT":
        X = np.hstack([load_representation("SEQ_SIMPLE", ids), load_representation("ESMFN_STRUCTURE", ids)])
    elif tag == "ESM2_STRUCT":
        X = np.hstack([load_representation("PLM_ESM2", ids), load_representation("ESMFN_STRUCTURE", ids)])
    elif tag == "ESM2":
        X = load_representation("PLM_ESM2", ids)
    elif tag == "ABLANG2":
        X = load_representation("PLM_ABLANG2", ids)
    elif tag == "STRUCT":
        X = load_representation("ESMFN_STRUCTURE", ids)
    elif tag == "SEQ":
        X = load_representation("SEQ_SIMPLE", ids)
    elif tag == "GERMLINE":
        X = load_representation("GERMLINE_REL", ids)
    elif tag == "IMGT":
        X = load_representation("IMGT_POS_HL", ids)
    else:
        X = load_representation(tag, ids)
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = 0.0
    FEAT_CACHE[key] = X
    return X


def fit_predict(kind, Xtr, ytr, Xte, params=None):
    params = params or {}
    if kind == "const":
        return np.full(len(Xte), float(np.median(ytr)))
    if kind == "ridge":
        est = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=params.get("alpha", 10.0)))])
    elif kind == "enet":
        est = Pipeline([
            ("sc", StandardScaler()),
            ("m", ElasticNet(alpha=params.get("alpha", 0.2), l1_ratio=params.get("l1", 0.5), max_iter=20000)),
        ])
    elif kind == "huber":
        est = Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(max_iter=2000))])
    elif kind == "svr_pca":
        n_pca = min(params.get("n_pca", 32), Xtr.shape[0] - 1, Xtr.shape[1])
        est = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=max(1, n_pca), random_state=SEED_BASE)),
            ("m", SVR(C=params.get("C", 1.0), epsilon=0.1, gamma="scale")),
        ])
    else:
        raise ValueError(kind)
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def oof_and_test(kind, feat_tag, train_ids, ytr, folds, test_ids, params=None):
    X = get_X(feat_tag, train_ids) if feat_tag != "CONST" else np.ones((len(train_ids), 1))
    Xt = get_X(feat_tag, test_ids) if feat_tag != "CONST" else np.ones((len(test_ids), 1))
    oof_sum = np.zeros(len(train_ids))
    for fid in folds:
        oof_r = np.zeros(len(train_ids))
        for f in range(5):
            te = fid == f
            tr = ~te
            pred = fit_predict(kind, X[tr], ytr[tr], X[te], params)
            oof_r[te] = pred
        oof_sum += oof_r
    oof = oof_sum / max(len(folds), 1)
    full = fit_predict(kind, X, ytr, Xt, params)
    return oof, full


def bank_b_specs() -> dict:
    """Validation bank — participant-legal heads distinct tilt from Bank A."""
    tmapp = [
        ("VAL_CONST_MEDIAN", "CONST", "const", {}),
        ("VAL_SEQ_Ridge", "SEQ", "ridge", {"alpha": 10.0}),
        ("VAL_SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("VAL_BIO_Ridge", "BIO", "ridge", {"alpha": 10.0}),
        ("VAL_BIO_ENet", "BIO", "enet", {"alpha": 0.15, "l1": 0.7}),
        ("VAL_ABLANG2_PCA32_SVR", "ABLANG2", "svr_pca", {"n_pca": 32, "C": 1.0}),
        ("VAL_ABLANG2_Ridge", "ABLANG2", "ridge", {"alpha": 10.0}),
        ("VAL_BIO_ABLANG2_ENet", "BIO_ABLANG2", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("VAL_GERMLINE_Ridge", "GERMLINE", "ridge", {"alpha": 10.0}),
        ("VAL_IMGT_Ridge", "IMGT", "ridge", {"alpha": 50.0}),
    ]
    hic = [
        ("VAL_CONST_MEDIAN", "CONST", "const", {}),
        ("VAL_SEQ_Ridge", "SEQ", "ridge", {"alpha": 10.0}),
        ("VAL_SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("VAL_ESM2_PCA64_SVR", "ESM2", "svr_pca", {"n_pca": 64, "C": 1.0}),
        ("VAL_ESM2_Ridge", "ESM2", "ridge", {"alpha": 50.0}),
        ("VAL_STRUCT_ENet", "STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("VAL_PHYS_STRUCT_Ridge", "PHYS_STRUCT", "ridge", {"alpha": 10.0}),
        ("VAL_ESM2_STRUCT_ENet", "ESM2_STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("VAL_ESM2_STRUCT_Huber", "ESM2_STRUCT", "huber", {}),
        ("VAL_SEQ_Huber", "SEQ", "huber", {}),
    ]
    return {"TmApp": tmapp, "HIC": hic}


def bank_c_specs() -> dict:
    """Stress bank — additional heads / variants not in A/B."""
    tmapp = [
        ("STRESS_SEQ_Ridge_a1", "SEQ", "ridge", {"alpha": 1.0}),
        ("STRESS_SEQ_Ridge_a100", "SEQ", "ridge", {"alpha": 100.0}),
        ("STRESS_BIO_Huber", "BIO", "huber", {}),
        ("STRESS_ABLANG2_PCA16_SVR", "ABLANG2", "svr_pca", {"n_pca": 16, "C": 3.0}),
        ("STRESS_BIO_ABLANG2_Ridge", "BIO_ABLANG2", "ridge", {"alpha": 5.0}),
        ("STRESS_IMGT_ENet", "IMGT", "enet", {"alpha": 0.3, "l1": 0.3}),
        ("STRESS_GERMLINE_ENet", "GERMLINE", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("STRESS_CONST", "CONST", "const", {}),
    ]
    hic = [
        ("STRESS_SEQ_Ridge_a1", "SEQ", "ridge", {"alpha": 1.0}),
        ("STRESS_SEQ_Ridge_a100", "SEQ", "ridge", {"alpha": 100.0}),
        ("STRESS_ESM2_PCA32_SVR", "ESM2", "svr_pca", {"n_pca": 32, "C": 3.0}),
        ("STRESS_STRUCT_Ridge", "STRUCT", "ridge", {"alpha": 5.0}),
        ("STRESS_PHYS_STRUCT_ENet", "PHYS_STRUCT", "enet", {"alpha": 0.15, "l1": 0.7}),
        ("STRESS_ESM2_STRUCT_Ridge", "ESM2_STRUCT", "ridge", {"alpha": 20.0}),
        ("STRESS_SEQ_Huber", "SEQ", "huber", {}),
        ("STRESS_CONST", "CONST", "const", {}),
    ]
    return {"TmApp": tmapp, "HIC": hic}


def train_spec_bank(data: dict, specs: dict, cache_subdir: str) -> dict:
    global BIO_CAT
    BIO_CAT = list(data["train_ids"])
    test_ids = data["test_ids"]
    store = {}
    cdir = CACHE / cache_subdir
    cdir.mkdir(parents=True, exist_ok=True)
    for target, model_list in specs.items():
        ytr = data["pop"].loc[data["train_ids"], target].astype(float).values
        tdir = cdir / target
        tdir.mkdir(parents=True, exist_ok=True)
        for name, feat_tag, kind, params in model_list:
            npz_path = tdir / f"{name}.npz"
            if npz_path.exists():
                z = np.load(npz_path, allow_pickle=True)
                oof = z["oof"].astype(float)
                test_pred = z["test"].astype(float)
                log(f"  cache hit {cache_subdir}/{target}/{name}")
            else:
                log(f"  train {cache_subdir}/{target}/{name}")
                oof, test_pred = oof_and_test(
                    kind, feat_tag, data["train_ids"], ytr, data["folds"], test_ids, params,
                )
                np.savez(npz_path, oof=oof, test=test_pred, test_ids=np.array(test_ids))
            # id->pred for all 162
            id_pred = {tid: float(test_pred[k]) for k, tid in enumerate(test_ids)}
            store[f"{target}::{name}"] = {
                "oof": oof,
                "test_full": test_pred,
                "id_pred": id_pred,
                "name": name,
                "family": _family(name, feat_tag),
            }
    return store


def _family(name: str, feat_tag: str) -> str:
    n = name.upper()
    if "CONST" in n:
        return "simple"
    if "SEQ" in n or feat_tag == "SEQ":
        return "seq"
    if "BIO" in n and "ABLANG" not in n and feat_tag == "BIO":
        return "bio"
    if "STRUCT" in n or feat_tag in ("STRUCT", "PHYS_STRUCT", "ESM2_STRUCT"):
        return "structure"
    if "ESM" in n or "ABLANG" in n or "PLM" in n:
        return "plm"
    if "FUSION" in n or "BIO_ABLANG" in n:
        return "fusion"
    if "NESTED" in n or "ENSEMBLE" in n:
        return "ensemble"
    if "GERMLINE" in n or "IMGT" in n:
        return "bio"
    return "other"


def load_bank_a(data: dict) -> dict:
    """Load frozen B5 predictions into store format."""
    store = {}
    for target, tags in BANK_A.items():
        for tag in tags:
            oof = load_b5_oof(target, tag)
            id_pred = build_id_pred_map(target, tag, data["role_pub"], data["role_priv"])
            test_full = np.array([id_pred[i] for i in data["test_ids"]], float)
            store[f"{target}::{tag}"] = {
                "oof": oof,
                "test_full": test_full,
                "id_pred": id_pred,
                "name": tag,
                "family": _family(tag, tag),
            }
    return store


def eval_split_bank(entry: dict, data: dict, store: dict, bank_name: str) -> list[dict]:
    """Evaluate one split × all targets in a bank; return flat rows."""
    pub_ids = entry["public_ids"]
    priv_ids = entry["private_ids"]
    split_id = entry.get("split_id") or entry.get("candidate_id")
    id_to_ix = {i: k for k, i in enumerate(data["test_ids"])}
    pub_ix = [id_to_ix[i] for i in pub_ids]
    priv_ix = [id_to_ix[i] for i in priv_ids]
    rows = []
    for target in ("TmApp", "HIC"):
        y_train = data["pop"].loc[data["train_ids"], target].astype(float).values
        y_pub = data["pop"].loc[pub_ids, target].astype(float).values
        y_priv = data["pop"].loc[priv_ids, target].astype(float).values
        keys = [k for k in store if k.startswith(f"{target}::")]
        cv_m, pub_m, priv_m, names, families = [], [], [], [], []
        for k in keys:
            d = store[k]
            cv_m.append(mae(y_train, d["oof"]))
            pub_m.append(mae(y_pub, d["test_full"][pub_ix]))
            priv_m.append(mae(y_priv, d["test_full"][priv_ix]))
            names.append(d["name"])
            families.append(d["family"])
        cv_m, pub_m, priv_m = map(np.asarray, (cv_m, pub_m, priv_m))
        wi_pub = int(np.argmin(pub_m))
        wi_cv = int(np.argmin(cv_m))
        best_priv = float(np.min(priv_m))
        rows.append({
            "split_id": split_id,
            "bank": bank_name,
            "target": target,
            "n_models": len(names),
            "spearman_cv_public": spearman_safe(-cv_m, -pub_m),
            "spearman_public_private": spearman_safe(-pub_m, -priv_m),
            "spearman_cv_private": spearman_safe(-cv_m, -priv_m),
            "public_winner_regret": float(priv_m[wi_pub] - best_priv),
            "cv_winner_regret": float(priv_m[wi_cv] - best_priv),
            "public_winner": names[wi_pub],
            "cv_winner": names[wi_cv],
            "best_private_mae": best_priv,
            "mean_public_mae": float(pub_m.mean()),
            "mean_private_mae": float(priv_m.mean()),
            "catastrophic_pub_priv": bool(
                (np.isfinite(spearman_safe(-pub_m, -priv_m)) and spearman_safe(-pub_m, -priv_m) < -0.2)
                or (priv_m[wi_pub] - best_priv > 1.5)
            ),
        })
    return rows


def family_lofo(entry: dict, data: dict, store: dict, bank_name: str) -> list[dict]:
    """Leave-one-family-out Spearman/regret for a split."""
    pub_ids = entry["public_ids"]
    priv_ids = entry["private_ids"]
    split_id = entry.get("split_id") or entry.get("candidate_id")
    id_to_ix = {i: k for k, i in enumerate(data["test_ids"])}
    pub_ix = [id_to_ix[i] for i in pub_ids]
    priv_ix = [id_to_ix[i] for i in priv_ids]
    rows = []
    for target in ("TmApp", "HIC"):
        y_train = data["pop"].loc[data["train_ids"], target].astype(float).values
        y_pub = data["pop"].loc[pub_ids, target].astype(float).values
        y_priv = data["pop"].loc[priv_ids, target].astype(float).values
        keys = [k for k in store if k.startswith(f"{target}::")]
        families = sorted({store[k]["family"] for k in keys})
        for fam in families:
            keep = [k for k in keys if store[k]["family"] != fam]
            if len(keep) < 3:
                continue
            cv_m, pub_m, priv_m = [], [], []
            for k in keep:
                d = store[k]
                cv_m.append(mae(y_train, d["oof"]))
                pub_m.append(mae(y_pub, d["test_full"][pub_ix]))
                priv_m.append(mae(y_priv, d["test_full"][priv_ix]))
            cv_m, pub_m, priv_m = map(np.asarray, (cv_m, pub_m, priv_m))
            wi = int(np.argmin(pub_m))
            rows.append({
                "split_id": split_id,
                "bank": bank_name,
                "target": target,
                "left_out_family": fam,
                "n_models_kept": len(keep),
                "spearman_cv_public": spearman_safe(-cv_m, -pub_m),
                "spearman_public_private": spearman_safe(-pub_m, -priv_m),
                "public_winner_regret": float(priv_m[wi] - np.min(priv_m)),
            })
    return rows


def hash_and_freeze_bank_assignment(specs_b, specs_c, bank_a_tags) -> str:
    blob = {
        "gate": "B7.3",
        "BANK_A": bank_a_tags,
        "BANK_B": specs_b,
        "BANK_C": specs_c,
        "note": "Hashed BEFORE training Bank B/C. Bank A uses frozen B5 predictions.",
    }
    return write_json(CFG / "B7_3_MODEL_BANK_ASSIGNMENT.json", blob)
