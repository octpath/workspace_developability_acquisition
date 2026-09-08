#!/usr/bin/env python3
"""Canonical Simple TVT evaluator with hard ID-alignment validation.

Version: canonical_simple_tvt_v1
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
CANONICAL_VERSION = "canonical_simple_tvt_v1"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
# Predeclared Lasso grid (log-scale; no prior authoritative endgame grid)
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
ROOT = Path("/workspace_developability_acquisition")
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"


class FeatureAlignmentError(RuntimeError):
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
    """Return numeric feature rows in EXACT requested ID order. Hard-fail on silent bugs."""
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

    # Drop non-numeric
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
    if len(out) != len(req):
        raise FeatureAlignmentError(f"{block_name}: len mismatch")

    if out.isna().all().all():
        raise FeatureAlignmentError(f"{block_name}: entirely NaN after alignment")

    all_nan_cols = [c for c in out.columns if out[c].isna().all()]
    if all_nan_cols and (len(all_nan_cols) / out.shape[1]) > max_all_nan_col_frac:
        raise FeatureAlignmentError(
            f"{block_name}: {len(all_nan_cols)} all-NaN columns e.g. {all_nan_cols[:5]}"
        )

    finite_frac = float(np.isfinite(out.to_numpy(dtype=float)).mean())
    if finite_frac <= 0:
        raise FeatureAlignmentError(f"{block_name}: zero finite values")

    # Attach metadata as attrs for logging
    out.attrs["block_name"] = block_name
    out.attrs["finite_frac"] = finite_frac
    out.attrs["n_all_nan_cols"] = len(all_nan_cols)
    out.attrs["sha16"] = sha_arr(np.nan_to_num(out.to_numpy(dtype=float), nan=0.0))[:16]
    return out


def concat_blocks(blocks: list[pd.DataFrame], ids: list[str], names: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    aligned = []
    meta = []
    for b, name in zip(blocks, names):
        a = align_feature_block(b, ids, name)
        aligned.append(a)
        meta.append(
            {
                "block": name,
                "shape": list(a.shape),
                "matched_ids": int(a.shape[0]),
                "all_nan": False,
                "finite_frac": a.attrs.get("finite_frac"),
                "variance_sum": float(np.nanvar(a.to_numpy(dtype=float), axis=0).sum()),
                "fingerprint": a.attrs.get("sha16"),
            }
        )
    # Ensure unique colnames across blocks
    parts = []
    for a, name in zip(aligned, names):
        aa = a.copy()
        aa.columns = [f"{name}__{c}" for c in aa.columns]
        parts.append(aa)
    X = pd.concat(parts, axis=1)
    if list(X.index) != list(ids):
        raise FeatureAlignmentError("concat: final index order mismatch")
    return X, meta


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


def run_standalone(
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
    *,
    dim_mode: str = "raw",
) -> dict:
    """Standalone Ridge Simple TVT. dim_mode: raw | PCA32."""
    common = [i for i in ids if i in X.index and i in y.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xx = X.loc[common]
    yy = y.loc[common]
    oof = pd.Series(index=common, dtype=float)
    alphas, nan_before = [], []
    for _, tr, va, te in rotation_splits(folds, common):
        nan_before.append(float(Xx.loc[tr].isna().mean().mean()))
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        if dim_mode == "PCA32":
            Xtr, [Xva], _ = pca_block(Xtr, [Xva])
        a = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        alphas.append(a)
        ids_tv = tr + va
        med2 = impute_fit(Xx.loc[ids_tv])
        Xtv = impute_apply(Xx.loc[ids_tv], med2)
        Xte = impute_apply(Xx.loc[te], med2)
        if dim_mode == "PCA32":
            Xtv, [Xte], _ = pca_block(Xtv, [Xte])
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv, float)), np.asarray(yy.loc[ids_tv], float))
        oof.loc[te] = m.predict(sc.transform(np.asarray(Xte, float)))
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "y": yy,
        "alphas": alphas,
        "nan_frac_train_mean": float(np.mean(nan_before)),
        "raw_dim": int(Xx.shape[1]),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float))[:16],
        "id_hash": sha_ids(common)[:16],
    }


def run_base_plus_struct(
    base: pd.DataFrame,
    struct: pd.DataFrame | None,
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
    *,
    mode: str = "FREE_ALPHA",
    force_pca_struct: bool | None = None,
) -> dict:
    """BASE (+ optional STRUCT with fold-local PCA if dim>trigger or forced)."""
    common = [i for i in ids if i in base.index and i in y.index]
    if struct is not None:
        common = [i for i in common if i in struct.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xb = base.loc[common]
    Xs = None if struct is None else struct.loc[common]
    if Xs is not None:
        ov = [c for c in Xs.columns if c in Xb.columns]
        if ov:
            Xs = Xs.drop(columns=ov)
    yy = y.loc[common]

    def maybe_pca(Str, others):
        do = force_pca_struct
        if do is None:
            do = Str.shape[1] > DIM_PCA_TRIGGER
        if do:
            return pca_block(Str, others)
        return Str, others, False

    def choose_a(tr, va, use_struct):
        med_b = impute_fit(Xb.loc[tr])
        Btr = impute_apply(Xb.loc[tr], med_b)
        Bva = impute_apply(Xb.loc[va], med_b)
        if use_struct and Xs is not None:
            med_s = impute_fit(Xs.loc[tr])
            Str = impute_apply(Xs.loc[tr], med_s)
            Sva = impute_apply(Xs.loc[va], med_s)
            Str2, [Sva2], _ = maybe_pca(Str, [Sva])
            Xtr = pd.concat([Btr, Str2], axis=1)
            Xva = pd.concat([Bva, Sva2], axis=1)
        else:
            Xtr, Xva = Btr, Bva
        return select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])

    def predict(tr, va, te, alpha, use_struct):
        ids_tv = tr + va
        med_b = impute_fit(Xb.loc[ids_tv])
        Btv = impute_apply(Xb.loc[ids_tv], med_b)
        Bte = impute_apply(Xb.loc[te], med_b)
        if use_struct and Xs is not None:
            med_s = impute_fit(Xs.loc[ids_tv])
            Stv = impute_apply(Xs.loc[ids_tv], med_s)
            Ste = impute_apply(Xs.loc[te], med_s)
            Stv2, [Ste2], _ = maybe_pca(Stv, [Ste])
            Xtv = pd.concat([Btv, Stv2], axis=1)
            Xte = pd.concat([Bte, Ste2], axis=1)
        else:
            Xtv, Xte = Btv, Bte
        sc = StandardScaler()
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv, float)), np.asarray(yy.loc[ids_tv], float))
        return pd.Series(m.predict(sc.transform(np.asarray(Xte, float))), index=te)

    use_plus = Xs is not None
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    alphas_b, alphas_p = [], []
    for _, tr, va, te in rotation_splits(folds, common):
        a_base = choose_a(tr, va, False)
        if not use_plus:
            a_plus = a_base
        elif mode == "FREE_ALPHA":
            a_plus = choose_a(tr, va, True)
        else:
            a_plus = a_base
        alphas_b.append(a_base)
        alphas_p.append(a_plus)
        base_oof.loc[te] = predict(tr, va, te, a_base, False)
        plus_oof.loc[te] = predict(tr, va, te, a_plus, use_plus)

    return {
        "N": len(common),
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof) if use_plus else mae(yy, base_oof),
        "delta": (mae(yy, base_oof) - mae(yy, plus_oof)) if use_plus else 0.0,
        "base_oof": base_oof,
        "plus_oof": plus_oof if use_plus else base_oof,
        "y": yy,
        "alphas_base": alphas_b,
        "alphas_plus": alphas_p,
        "base_dim": int(Xb.shape[1]),
        "struct_dim": 0 if Xs is None else int(Xs.shape[1]),
        "prediction_hash": sha_arr((plus_oof if use_plus else base_oof).to_numpy(float))[:16],
        "id_hash": sha_ids(common)[:16],
        "canonical_evaluator_version": CANONICAL_VERSION,
    }


def load_folds():
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)
    return primary, shadow


def _pca_named(Xtr: pd.DataFrame, others: list[pd.DataFrame], prefix: str, n: int = PCA_CAP):
    Xtr2, outs, ok = pca_block(Xtr, others, n=n)
    if not ok:
        return Xtr, others
    Xtr2 = Xtr2.rename(columns={c: f"{prefix}{c}" for c in Xtr2.columns})
    outs = [o.rename(columns={c: f"{prefix}{c}" for c in o.columns}) for o in outs]
    return Xtr2, outs


def run_recipe_plus_abl_blocks(
    recipe: pd.DataFrame,
    always_blocks: list[pd.DataFrame],
    extra_blocks: list[pd.DataFrame],
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
    *,
    mode: str = "FREE_ALPHA",
) -> dict:
    """Match AbLingua guided protocol: fold-local PCA32 per AbLingua block.

    base = recipe + PCA(always_blocks...)
    plus = recipe + PCA(always_blocks...) + PCA(extra_blocks...)
    """
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

    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    alphas_b, alphas_p = [], []
    for _, tr, va, te in rotation_splits(folds, common):
        Xtr_b, [Xva_b] = build(tr, [va], always)
        a_base = select_alpha(Xtr_b, yy.loc[tr], Xva_b, yy.loc[va])
        if mode == "FREE_ALPHA" and extras:
            Xtr_p, [Xva_p] = build(tr, [va], always + extras)
            a_plus = select_alpha(Xtr_p, yy.loc[tr], Xva_p, yy.loc[va])
        else:
            a_plus = a_base
        alphas_b.append(a_base)
        alphas_p.append(a_plus)
        ids_tv = tr + va
        Xtv_b, [Xte_b] = build(ids_tv, [te], always)
        sc = StandardScaler()
        m = Ridge(alpha=a_base, random_state=0)
        m.fit(sc.fit_transform(np.asarray(Xtv_b, float)), np.asarray(yy.loc[ids_tv], float))
        base_oof.loc[te] = m.predict(sc.transform(np.asarray(Xte_b, float)))
        Xtv_p, [Xte_p] = build(ids_tv, [te], always + extras)
        sc2 = StandardScaler()
        m2 = Ridge(alpha=a_plus, random_state=0)
        m2.fit(sc2.fit_transform(np.asarray(Xtv_p, float)), np.asarray(yy.loc[ids_tv], float))
        plus_oof.loc[te] = m2.predict(sc2.transform(np.asarray(Xte_p, float)))

    return {
        "N": len(common),
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof),
        "delta": mae(yy, base_oof) - mae(yy, plus_oof),
        "base_oof": base_oof,
        "plus_oof": plus_oof,
        "y": yy,
        "alphas_base": alphas_b,
        "alphas_plus": alphas_p,
        "prediction_hash": sha_arr(plus_oof.to_numpy(float))[:16],
        "id_hash": sha_ids(common)[:16],
        "canonical_evaluator_version": CANONICAL_VERSION,
    }


class LassoConvergenceError(RuntimeError):
    pass


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
    # sklearn sets n_iter_; if hit max_iter, treat as failure
    n_iter = int(getattr(m, "n_iter_", 0) or 0)
    if conv_warn or n_iter >= LASSO_MAX_ITER:
        raise LassoConvergenceError(
            f"Lasso did not converge alpha={alpha} n_iter={n_iter} max_iter={LASSO_MAX_ITER}"
        )
    return m


def select_lasso_alpha(Xtr, ytr, Xva, yva) -> tuple[float, dict]:
    """Choose alpha on VAL; skip alphas that fail to converge (do not silent-accept)."""
    best_a, best = None, np.inf
    meta = {}
    Xtr_v = np.asarray(Xtr, float)
    Xva_v = np.asarray(Xva, float)
    ytr_v = np.asarray(ytr, float)
    yva_v = np.asarray(yva, float)
    tried, ok = 0, 0
    for a in LASSO_ALPHAS:
        tried += 1
        sc = StandardScaler()
        Ztr = sc.fit_transform(Xtr_v)
        Zva = sc.transform(Xva_v)
        try:
            m = _fit_lasso(Ztr, ytr_v, a)
        except LassoConvergenceError:
            continue
        ok += 1
        score = mae(yva_v, m.predict(Zva))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and (best_a is None or a > best_a)):
            best, best_a = score, a
            nz = int(np.sum(np.abs(m.coef_) > 1e-12))
            meta = {
                "alpha": float(best_a),
                "n_nonzero": nz,
                "n_coef": int(len(m.coef_)),
                "sparsity": 1.0 - nz / max(len(m.coef_), 1),
                "converged": True,
                "alphas_converged": ok,
                "alphas_tried": tried,
            }
    if best_a is None:
        raise LassoConvergenceError(
            f"No Lasso alpha converged among {LASSO_ALPHAS} (tried={tried})"
        )
    return float(best_a), meta


def run_standalone_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    ids: list[str],
) -> dict:
    """StandardScaler -> Lasso Simple TVT. No PCA. Hard-fail on non-convergence."""
    common = [i for i in ids if i in X.index and i in y.index]
    for _, tr, va, te in rotation_splits(folds, common):
        if not (len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE):
            raise FeatureAlignmentError("coverage failure")
    Xx = X.loc[common]
    yy = y.loc[common]
    if Xx.shape[1] == 0:
        raise FeatureAlignmentError("Lasso: zero feature dim")
    oof = pd.Series(index=common, dtype=float)
    alphas, n_nonzero, sparsities, conv = [], [], [], []
    for _, tr, va, te in rotation_splits(folds, common):
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        a, meta = select_lasso_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        alphas.append(a)
        n_nonzero.append(meta["n_nonzero"])
        sparsities.append(meta["sparsity"])
        conv.append(True)
        ids_tv = tr + va
        med2 = impute_fit(Xx.loc[ids_tv])
        Xtv = impute_apply(Xx.loc[ids_tv], med2)
        Xte = impute_apply(Xx.loc[te], med2)
        sc = StandardScaler()
        Ztv = sc.fit_transform(np.asarray(Xtv, float))
        Zte = sc.transform(np.asarray(Xte, float))
        # Refit may need a larger alpha than VAL-selected if TV set differs
        m = None
        for a_try in [a] + [x for x in LASSO_ALPHAS if x > a]:
            try:
                m = _fit_lasso(Ztv, yy.loc[ids_tv], a_try)
                a = a_try
                break
            except LassoConvergenceError:
                continue
        if m is None:
            raise LassoConvergenceError(f"TV refit failed for all alphas >= {alphas[-1] if alphas else a}")
        alphas[-1] = a
        oof.loc[te] = m.predict(Zte)
        # record refit sparsity
        nz = int(np.sum(np.abs(m.coef_) > 1e-12))
        n_nonzero[-1] = nz
        sparsities[-1] = 1.0 - nz / max(len(m.coef_), 1)
    return {
        "N": len(common),
        "mae": mae(yy, oof),
        "oof": oof,
        "y": yy,
        "alphas": alphas,
        "n_nonzero": n_nonzero,
        "sparsities": sparsities,
        "converged": conv,
        "raw_dim": int(Xx.shape[1]),
        "alpha_median": float(np.median(alphas)),
        "n_nonzero_median": float(np.median(n_nonzero)),
        "n_nonzero_min": int(np.min(n_nonzero)),
        "n_nonzero_max": int(np.max(n_nonzero)),
        "sparsity_median": float(np.median(sparsities)),
        "prediction_hash": sha_arr(oof.to_numpy(dtype=float))[:16],
        "id_hash": sha_ids(common)[:16],
        "canonical_evaluator_version": CANONICAL_VERSION,
        "regressor": "LASSO",
        "preprocessing": "impute+StandardScaler+Lasso(no_PCA)",
    }


def fit_full_dev_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    ids: list[str],
    *,
    alpha: float,
    dim_mode: str = "raw",
) -> dict:
    """Fit Ridge on full DEV ids; return model artifacts for Test prediction."""
    common = [i for i in ids if i in X.index and i in y.index]
    Xx = X.loc[common]
    yy = y.loc[common]
    med = impute_fit(Xx)
    Xi = impute_apply(Xx, med)
    pca = None
    if dim_mode == "PCA32":
        Xi, _, ok = pca_block(Xi, [])
        if not ok:
            dim_mode = "raw"
    sc = StandardScaler()
    Z = sc.fit_transform(np.asarray(Xi, float))
    m = Ridge(alpha=alpha, random_state=0)
    m.fit(Z, np.asarray(yy, float))
    return {
        "ids": common,
        "med": med,
        "scaler": sc,
        "model": m,
        "pca": pca,
        "dim_mode": dim_mode,
        "alpha": float(alpha),
        "raw_dim": int(X.loc[common].shape[1]),
    }


def predict_with_ridge_artifact(art: dict, X: pd.DataFrame, ids: list[str]) -> pd.Series:
    use = [i for i in ids if i in X.index]
    Xi = impute_apply(X.loc[use], art["med"])
    if art["dim_mode"] == "PCA32" and art.get("pca") is not None:
        # pca stored not currently; for PCA full-dev we refit path differently
        raise FeatureAlignmentError("PCA artifact path requires pca object")
    Z = art["scaler"].transform(np.asarray(Xi, float))
    return pd.Series(art["model"].predict(Z), index=use)


def fit_full_dev_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    ids: list[str],
    *,
    alpha: float,
) -> dict:
    common = [i for i in ids if i in X.index and i in y.index]
    Xx = X.loc[common]
    yy = y.loc[common]
    med = impute_fit(Xx)
    Xi = impute_apply(Xx, med)
    sc = StandardScaler()
    Z = sc.fit_transform(np.asarray(Xi, float))
    m = _fit_lasso(Z, yy, alpha)
    nz = int(np.sum(np.abs(m.coef_) > 1e-12))
    return {
        "ids": common,
        "med": med,
        "scaler": sc,
        "model": m,
        "alpha": float(alpha),
        "n_nonzero": nz,
        "n_coef": int(len(m.coef_)),
        "sparsity": 1.0 - nz / max(len(m.coef_), 1),
        "raw_dim": int(Xx.shape[1]),
    }


def predict_with_lasso_artifact(art: dict, X: pd.DataFrame, ids: list[str]) -> pd.Series:
    use = [i for i in ids if i in X.index]
    Xi = impute_apply(X.loc[use], art["med"])
    Z = art["scaler"].transform(np.asarray(Xi, float))
    return pd.Series(art["model"].predict(Z), index=use)
