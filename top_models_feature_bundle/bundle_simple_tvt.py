#!/usr/bin/env python3
"""Minimal ID-safe Simple TVT evaluator copied into the participant bundle.

Version: bundle_simple_tvt_v1 (logic aligned with canonical_simple_tvt_v1).
Does NOT import repository-internal packages (organizer_extension, virtual_participant).
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

CANONICAL_VERSION = "bundle_simple_tvt_v1"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
]
LASSO_MAX_ITER = 200_000
LASSO_TOL = 1e-3
PCA_CAP = 32
DIM_PCA_TRIGGER = 200
MIN_TR, MIN_VA, MIN_TE = 20, 5, 5


class FeatureAlignmentError(RuntimeError):
    pass


class LassoConvergenceError(RuntimeError):
    pass


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def sha_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def align_feature_block(
    block: pd.DataFrame,
    ids: list[str],
    block_name: str,
    *,
    allow_intersection: bool = False,
    max_all_nan_col_frac: float = 0.0,
) -> pd.DataFrame:
    if block is None or len(block) == 0:
        raise FeatureAlignmentError(f"{block_name}: empty block")
    X = block.copy()
    if "id" in X.columns:
        X["id"] = X["id"].astype(str)
        if X["id"].duplicated().any():
            raise FeatureAlignmentError(f"{block_name}: duplicate id values in column")
        X = X.set_index("id")
    X.index = X.index.astype(str)
    if X.index.duplicated().any():
        raise FeatureAlignmentError(f"{block_name}: duplicate index IDs")
    X = X.select_dtypes(include=[np.number]).astype(float)
    if X.shape[1] == 0:
        raise FeatureAlignmentError(f"{block_name}: no numeric columns")
    req = [str(i) for i in ids]
    if len(req) != len(set(req)):
        raise FeatureAlignmentError(f"{block_name}: requested ids not unique")
    missing = [i for i in req if i not in X.index]
    if missing and not allow_intersection:
        raise FeatureAlignmentError(
            f"{block_name}: missing {len(missing)} IDs e.g. {missing[:5]}"
        )
    if allow_intersection:
        req = [i for i in req if i in X.index]
        if not req:
            raise FeatureAlignmentError(f"{block_name}: zero matched IDs")
    out = X.reindex(req)
    if list(out.index) != list(req):
        raise FeatureAlignmentError(f"{block_name}: row order != requested ids")
    if out.isna().all().all():
        raise FeatureAlignmentError(f"{block_name}: entirely NaN after alignment")
    # Historical SEQ_BASIC bug guard: RangeIndex + wrong reindex → all-NaN
    if out.shape[0] > 0 and out.isna().all(axis=1).mean() > 0.99:
        raise FeatureAlignmentError(
            f"{block_name}: >=99% rows all-NaN after id alignment "
            "(possible RangeIndex / row-order bug)"
        )
    all_nan_cols = [c for c in out.columns if out[c].isna().all()]
    if all_nan_cols and (len(all_nan_cols) / out.shape[1]) > max_all_nan_col_frac:
        raise FeatureAlignmentError(
            f"{block_name}: {len(all_nan_cols)} all-NaN columns e.g. {all_nan_cols[:5]}"
        )
    finite_frac = float(np.isfinite(out.to_numpy(dtype=float)).mean())
    if finite_frac <= 0:
        raise FeatureAlignmentError(f"{block_name}: zero finite values")
    out.attrs["block_name"] = block_name
    out.attrs["finite_frac"] = finite_frac
    return out


def concat_blocks(blocks: list[pd.DataFrame], ids: list[str], names: list[str]) -> pd.DataFrame:
    parts = []
    for b, name in zip(blocks, names):
        a = align_feature_block(b, ids, name)
        aa = a.copy()
        aa.columns = [f"{name}__{c}" for c in aa.columns]
        parts.append(aa)
    X = pd.concat(parts, axis=1)
    if list(X.index) != list(ids):
        raise FeatureAlignmentError("concat: final index order mismatch")
    return X


def impute_fit(X: pd.DataFrame) -> pd.Series:
    return X.median(numeric_only=True).fillna(0.0)


def impute_apply(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    return X.fillna(med).fillna(0.0)


def select_alpha(Xtr, ytr, Xva, yva) -> float:
    best_a, best = 100.0, np.inf
    Xtr_v = np.asarray(Xtr, float)
    Xva_v = np.asarray(Xva, float)
    ytr_v = np.asarray(ytr, float)
    yva_v = np.asarray(yva, float)
    for a in ALPHAS:
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr_v), ytr_v)
        score = mae(yva_v, m.predict(sc.transform(Xva_v)))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return float(best_a)


def pca_block(Xtr: pd.DataFrame, others: list[pd.DataFrame], n: int = PCA_CAP):
    k = min(n, Xtr.shape[0] - 1, Xtr.shape[1])
    if k < 2:
        return Xtr, others, False
    pca = PCA(n_components=k, random_state=0)
    Ztr = pca.fit_transform(np.asarray(Xtr, float))
    cols = [f"pca_{i}" for i in range(k)]
    outs = [
        pd.DataFrame(pca.transform(np.asarray(X, float)), index=X.index, columns=cols)
        for X in others
    ]
    return pd.DataFrame(Ztr, index=Xtr.index, columns=cols), outs, True


def rotation_splits(folds: pd.DataFrame, ids: list[str]):
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in ids if i in fmap]
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        yield k, tr, va, te


def _pca_named(Xtr: pd.DataFrame, others: list[pd.DataFrame], prefix: str, n: int = PCA_CAP):
    Xtr2, outs, ok = pca_block(Xtr, others, n=n)
    if not ok:
        return Xtr, others
    Xtr2 = Xtr2.rename(columns={c: f"{prefix}{c}" for c in Xtr2.columns})
    outs = [o.rename(columns={c: f"{prefix}{c}" for c in o.columns}) for o in outs]
    return Xtr2, outs


def run_standalone_ridge(X: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, ids: list[str]) -> dict:
    common = [i for i in ids if i in X.index and i in y.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xx = X.loc[common]
    yy = y.loc[common]
    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for _, tr, va, te in rotation_splits(folds, common):
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        a = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        alphas.append(a)
        ids_tv = tr + va
        med2 = impute_fit(Xx.loc[ids_tv])
        Xtv = impute_apply(Xx.loc[ids_tv], med2)
        Xte = impute_apply(Xx.loc[te], med2)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv, float)), np.asarray(yy.loc[ids_tv], float))
        oof.loc[te] = m.predict(sc.transform(np.asarray(Xte, float)))
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "alphas": alphas,
        "raw_dim": int(Xx.shape[1]),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float)),
    }


def run_base_plus_struct_ridge(
    base: pd.DataFrame,
    struct: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
) -> dict:
    """Core recipe + forced PCA32 on struct (AbLingua GLOBAL)."""
    common = [i for i in ids if i in base.index and i in y.index and i in struct.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xb = base.loc[common]
    Xs = struct.loc[common]
    ov = [c for c in Xs.columns if c in Xb.columns]
    if ov:
        Xs = Xs.drop(columns=ov)
    yy = y.loc[common]

    def choose_a(tr, va):
        med_b = impute_fit(Xb.loc[tr])
        med_s = impute_fit(Xs.loc[tr])
        Btr = impute_apply(Xb.loc[tr], med_b)
        Bva = impute_apply(Xb.loc[va], med_b)
        Str = impute_apply(Xs.loc[tr], med_s)
        Sva = impute_apply(Xs.loc[va], med_s)
        Str2, [Sva2], _ = pca_block(Str, [Sva])
        Xtr = pd.concat([Btr, Str2], axis=1)
        Xva = pd.concat([Bva, Sva2], axis=1)
        return select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])

    def predict(tr, va, te, alpha):
        ids_tv = tr + va
        med_b = impute_fit(Xb.loc[ids_tv])
        med_s = impute_fit(Xs.loc[ids_tv])
        Btv = impute_apply(Xb.loc[ids_tv], med_b)
        Bte = impute_apply(Xb.loc[te], med_b)
        Stv = impute_apply(Xs.loc[ids_tv], med_s)
        Ste = impute_apply(Xs.loc[te], med_s)
        Stv2, [Ste2], _ = pca_block(Stv, [Ste])
        Xtv = pd.concat([Btv, Stv2], axis=1)
        Xte = pd.concat([Bte, Ste2], axis=1)
        sc = StandardScaler()
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv, float)), np.asarray(yy.loc[ids_tv], float))
        return pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=te)

    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for _, tr, va, te in rotation_splits(folds, common):
        a = choose_a(tr, va)
        alphas.append(a)
        oof.loc[te] = predict(tr, va, te, a)
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "alphas": alphas,
        "raw_dim": int(Xb.shape[1] + Xs.shape[1]),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float)),
    }


def run_recipe_plus_abl_blocks_ridge(
    recipe: pd.DataFrame,
    always_blocks: list[pd.DataFrame],
    extra_blocks: list[pd.DataFrame],
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
) -> dict:
    """Core + PCA32 per AbLingua block (GLOBAL always; CDR3 as extra)."""
    all_blocks = always_blocks + extra_blocks
    common = [i for i in ids if i in recipe.index and i in y.index]
    for b in all_blocks:
        common = [i for i in common if i in b.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xp = recipe.loc[common]
    always = [b.loc[common].copy() for b in always_blocks]
    extras = [b.loc[common].copy() for b in extra_blocks]
    for lst in (always, extras):
        for i, b in enumerate(lst):
            ov = [c for c in b.columns if c in Xp.columns]
            if ov:
                lst[i] = b.drop(columns=ov)
    yy = y.loc[common]

    def build(tr, va_te_list, blocks):
        med_p = impute_fit(Xp.loc[tr])
        Ptr = impute_apply(Xp.loc[tr], med_p)
        Pothers = [impute_apply(Xp.loc[ix], med_p) for ix in va_te_list]
        if not blocks:
            return Ptr, Pothers
        meds = [impute_fit(b.loc[tr]) for b in blocks]
        Btr = [impute_apply(b.loc[tr], m) for b, m in zip(blocks, meds)]
        Bothers = [
            [impute_apply(b.loc[ix], m) for b, m in zip(blocks, meds)] for ix in va_te_list
        ]
        parts_tr = []
        parts_others = [[] for _ in va_te_list]
        for bi, B in enumerate(Btr):
            others_b = [Bothers[j][bi] for j in range(len(va_te_list))]
            Bt2, ot2 = _pca_named(B, others_b, prefix=f"abl{bi}_")
            parts_tr.append(Bt2)
            for j, o in enumerate(ot2):
                parts_others[j].append(o)
        Xtr = pd.concat([Ptr] + parts_tr, axis=1)
        Xothers = [pd.concat([p] + po, axis=1) for p, po in zip(Pothers, parts_others)]
        return Xtr, Xothers

    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for _, tr, va, te in rotation_splits(folds, common):
        Xtr, [Xva] = build(tr, [va], always + extras)
        a = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        alphas.append(a)
        ids_tv = tr + va
        Xtv, [Xte] = build(ids_tv, [te], always + extras)
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv, float)), np.asarray(yy.loc[ids_tv], float))
        oof.loc[te] = m.predict(sc.transform(np.asarray(Xte, float)))
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "alphas": alphas,
        "raw_dim": int(Xp.shape[1] + sum(b.shape[1] for b in always + extras)),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float)),
    }


def _fit_lasso(X, y, alpha: float) -> Lasso:
    m = Lasso(
        alpha=alpha,
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
        random_state=0,
        selection="cyclic",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(np.asarray(X, float), np.asarray(y, float))
        conv_warn = any(issubclass(x.category, ConvergenceWarning) for x in w)
    n_iter = int(getattr(m, "n_iter_", 0) or 0)
    if conv_warn or n_iter >= LASSO_MAX_ITER:
        raise LassoConvergenceError(
            f"Lasso did not converge alpha={alpha} n_iter={n_iter}"
        )
    return m


def select_lasso_alpha(Xtr, ytr, Xva, yva) -> float:
    best_a, best = None, np.inf
    Xtr_v = np.asarray(Xtr, float)
    Xva_v = np.asarray(Xva, float)
    ytr_v = np.asarray(ytr, float)
    yva_v = np.asarray(yva, float)
    for a in LASSO_ALPHAS:
        sc = StandardScaler()
        Ztr = sc.fit_transform(Xtr_v)
        Zva = sc.transform(Xva_v)
        try:
            m = _fit_lasso(Ztr, ytr_v, a)
        except LassoConvergenceError:
            continue
        score = mae(yva_v, m.predict(Zva))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and (best_a is None or a > best_a)):
            best, best_a = score, a
    if best_a is None:
        raise LassoConvergenceError("No Lasso alpha converged")
    return float(best_a)


def run_standalone_lasso(X: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, ids: list[str]) -> dict:
    common = [i for i in ids if i in X.index and i in y.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xx = X.loc[common]
    yy = y.loc[common]
    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for _, tr, va, te in rotation_splits(folds, common):
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        a = select_lasso_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        alphas.append(a)
        ids_tv = tr + va
        med2 = impute_fit(Xx.loc[ids_tv])
        Xtv = impute_apply(Xx.loc[ids_tv], med2)
        Xte = impute_apply(Xx.loc[te], med2)
        sc = StandardScaler()
        Ztv = sc.fit_transform(np.asarray(Xtv, float))
        Zte = sc.transform(np.asarray(Xte, float))
        m = None
        for a_try in [a] + [x for x in LASSO_ALPHAS if x > a]:
            try:
                m = _fit_lasso(Ztv, yy.loc[ids_tv], a_try)
                a = a_try
                break
            except LassoConvergenceError:
                continue
        if m is None:
            raise LassoConvergenceError("TV refit failed")
        alphas[-1] = a
        oof.loc[te] = m.predict(Zte)
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "alphas": alphas,
        "raw_dim": int(Xx.shape[1]),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float)),
    }


def fit_predict_ridge_raw(X_dev, y_dev, X_te, alpha: float) -> pd.Series:
    med = impute_fit(X_dev)
    Xdv = impute_apply(X_dev, med)
    Xte = impute_apply(X_te, med)
    sc = StandardScaler()
    m = Ridge(alpha=float(alpha), random_state=0)
    m.fit(sc.fit_transform(np.asarray(Xdv, float)), np.asarray(y_dev.loc[X_dev.index], float))
    return pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=X_te.index)


def fit_predict_ridge_base_struct(base_dev, struct_dev, y_dev, base_te, struct_te, alpha: float) -> pd.Series:
    med_b, med_s = impute_fit(base_dev), impute_fit(struct_dev)
    Bdv, Sdv = impute_apply(base_dev, med_b), impute_apply(struct_dev, med_s)
    Bte, Ste = impute_apply(base_te, med_b), impute_apply(struct_te, med_s)
    Sdv2, [Ste2], _ = pca_block(Sdv, [Ste])
    Xdv = pd.concat([Bdv, Sdv2], axis=1)
    Xte = pd.concat([Bte, Ste2], axis=1)
    sc = StandardScaler()
    m = Ridge(alpha=float(alpha), random_state=0)
    m.fit(sc.fit_transform(np.asarray(Xdv, float)), np.asarray(y_dev.loc[base_dev.index], float))
    return pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=Xte.index)


def fit_predict_ridge_abl_blocks(
    recipe_dev, always_dev, extra_dev, y_dev, recipe_te, always_te, extra_te, alpha: float
) -> pd.Series:
    med_r, med_a, med_e = impute_fit(recipe_dev), impute_fit(always_dev), impute_fit(extra_dev)
    Rd, Ad, Ed = impute_apply(recipe_dev, med_r), impute_apply(always_dev, med_a), impute_apply(extra_dev, med_e)
    Rte, Ate, Ete = impute_apply(recipe_te, med_r), impute_apply(always_te, med_a), impute_apply(extra_te, med_e)
    Ad2, [Ate2], _ = pca_block(Ad, [Ate])
    Ed2, [Ete2], _ = pca_block(Ed, [Ete])
    Ad2 = Ad2.rename(columns={c: f"abl0_{c}" for c in Ad2.columns})
    Ate2 = Ate2.rename(columns={c: f"abl0_{c}" for c in Ate2.columns})
    Ed2 = Ed2.rename(columns={c: f"abl1_{c}" for c in Ed2.columns})
    Ete2 = Ete2.rename(columns={c: f"abl1_{c}" for c in Ete2.columns})
    Xdv = pd.concat([Rd, Ad2, Ed2], axis=1)
    Xte = pd.concat([Rte, Ate2, Ete2], axis=1)
    sc = StandardScaler()
    m = Ridge(alpha=float(alpha), random_state=0)
    m.fit(sc.fit_transform(np.asarray(Xdv, float)), np.asarray(y_dev.loc[recipe_dev.index], float))
    return pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=Xte.index)


def fit_predict_lasso(X_dev, y_dev, X_te, alpha: float) -> pd.Series:
    med = impute_fit(X_dev)
    Xdv = impute_apply(X_dev, med)
    Xte = impute_apply(X_te, med)
    sc = StandardScaler()
    Z = sc.fit_transform(np.asarray(Xdv, float))
    Zte = sc.transform(np.asarray(Xte, float))
    m = None
    a = float(alpha)
    for a_try in [a] + [x for x in LASSO_ALPHAS if x > a]:
        try:
            m = _fit_lasso(Z, y_dev.loc[X_dev.index], a_try)
            break
        except LassoConvergenceError:
            continue
    if m is None:
        raise LassoConvergenceError("full-DEV Lasso fit failed")
    return pd.Series(m.predict(Zte), index=Xte.index)
