#!/usr/bin/env python3
"""AbLingua-600M TmApp Simple TVT — predeclared candidates only."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "ablingua600m"
RES = OUT / "results"
EMB = OUT / "embeddings"
INTERIM = FP / "fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"
EMB_NPZ = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
REGIONS = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_CAP = 32
DIM_PCA_TRIGGER = 200
B_BOOT = 10000
RNG = np.random.default_rng(42)

RES.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "tvt_rescreen", INTERIM / "scripts/run_simple_tvt_rescreen.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
from run_stage1 import build_all_feature_tables, make_xy  # noqa: E402


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def load_parquet_X(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["id"] = df["id"].astype(str)
    X = df.set_index("id").select_dtypes(include=[np.number]).astype(float)
    return X


def load_ablang2() -> dict[str, pd.DataFrame]:
    z = np.load(EMB_NPZ, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    out = {}
    for key, name in [
        ("ablang2__HL_paired", "AbLang2_HL_paired"),
        ("ablang2__H", "AbLang2_H"),
        ("ablang2__L", "AbLang2_L"),
        ("ablang2__HL", "AbLang2_HL_concat"),
    ]:
        X = pd.DataFrame(np.asarray(z[key], float), index=ids)
        X.columns = [f"{name}_{i}" for i in range(X.shape[1])]
        out[name] = X
    return out


def load_seq_basic(ids: list[str]) -> pd.DataFrame:
    """SEQ_BASIC aligned to competition ids.

    make_xy returns a RangeIndex; must assign ids explicitly (same as load_bases()).
    Never reindex from RangeIndex stringified positions — that yields all-NaN rows.
    """
    dev = pd.read_csv(DEV)
    tables = build_all_feature_tables(dev, pd.read_csv(ANN), pd.read_csv(REGIONS))
    seq_b, _ = make_xy(tables, "SEQ_BASIC")
    if len(seq_b) != len(ids):
        raise RuntimeError(f"SEQ_BASIC rows {len(seq_b)} != n_ids {len(ids)}")
    seq_b = seq_b.copy()
    seq_b.index = list(ids)
    seq_b.columns = [f"seqB_{c}" for c in seq_b.columns]
    if seq_b.isna().all().all():
        raise RuntimeError("SEQ_BASIC is entirely NaN — id alignment bug")
    if float(seq_b.isna().mean().mean()) > 0.5:
        raise RuntimeError("SEQ_BASIC majority-NaN — check id alignment")
    return seq_b.astype(float)


def load_pair_m1() -> tuple[pd.DataFrame, pd.DataFrame]:
    bio = pd.read_csv(BIO)
    bio["id"] = bio["id"].astype(str)
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    X_pair = bio.set_index("id")[pair_cols].apply(pd.to_numeric, errors="coerce")
    m1 = mod.numeric_X(pd.read_csv(M1), drop_cols=["status", "extraction_status"])
    return X_pair, m1


def impute_fit(X):
    return X.median(numeric_only=True).fillna(0.0)


def impute_apply(X, med):
    return X.fillna(med).fillna(0.0)


def select_alpha(Xtr, ytr, Xva, yva):
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), ytr)
        score = mae(yva, m.predict(sc.transform(Xva)))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def apply_pca_fold(Xtr, others, n_comp=PCA_CAP):
    n = min(n_comp, Xtr.shape[0] - 1, Xtr.shape[1])
    if n < 2:
        return Xtr, others, False
    pca = PCA(n_components=n, random_state=0)
    Ztr = pca.fit_transform(Xtr)
    cols = [f"pca_{i}" for i in range(n)]
    outs = [pd.DataFrame(pca.transform(X), index=X.index, columns=cols) for X in others]
    return pd.DataFrame(Ztr, index=Xtr.index, columns=cols), outs, True


def run_standalone(X, y, folds, ids, dim_mode: str, model_kind: str = "Ridge"):
    """Standalone representation as sole features. dim_mode: raw | PCA32."""
    common = sorted(
        set(X.index) & set(y.index) & set(folds.id.astype(str)) & set(ids)
    )
    for k, tr, va, te in mod.rotation_splits(folds, common):
        if not (
            len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        ):
            return None
    if len(common) < 40:
        return None
    Xx = X.loc[common]
    yy = y.loc[common]
    oof = pd.Series(index=common, dtype=float)
    alphas = []
    for k, tr, va, te in mod.rotation_splits(folds, common):
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        Xte = impute_apply(Xx.loc[te], med)
        if dim_mode == "PCA32":
            Xtr, [Xva, Xte], _ = apply_pca_fold(Xtr, [Xva, Xte], PCA_CAP)
        elif dim_mode == "raw":
            pass
        else:
            raise ValueError(dim_mode)
        if model_kind == "Ridge":
            a = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
            alphas.append(a)
            ids_tv = tr + va
            med2 = impute_fit(Xx.loc[ids_tv])
            Xtv = impute_apply(Xx.loc[ids_tv], med2)
            Xte2 = impute_apply(Xx.loc[te], med2)
            if dim_mode == "PCA32":
                Xtv, [Xte2], _ = apply_pca_fold(Xtv, [Xte2], PCA_CAP)
            sc = StandardScaler()
            m = Ridge(alpha=a, random_state=0)
            m.fit(sc.fit_transform(Xtv), yy.loc[ids_tv])
            oof.loc[te] = m.predict(sc.transform(Xte2))
        elif model_kind == "SVR_fixed":
            # Stage2 fixed: PCA32 + SVR(C=3, gamma=0.01, epsilon=0.1)
            ids_tv = tr + va
            med2 = impute_fit(Xx.loc[ids_tv])
            Xtv = impute_apply(Xx.loc[ids_tv], med2)
            Xte2 = impute_apply(Xx.loc[te], med2)
            Xtv, [Xte2], _ = apply_pca_fold(Xtv, [Xte2], PCA_CAP)
            sc = StandardScaler()
            m = SVR(C=3.0, gamma=0.01, epsilon=0.1)
            m.fit(sc.fit_transform(Xtv), yy.loc[ids_tv])
            oof.loc[te] = m.predict(sc.transform(Xte2))
            alphas.append(None)
        else:
            raise ValueError(model_kind)
    return {
        "N": len(common),
        "dim": int(Xx.shape[1]),
        "mae": mae(yy, oof),
        "oof": oof,
        "y": yy,
        "alphas": alphas,
    }


def run_incr(base, struct, y, folds, ids, mode: str, force_pca_struct: bool | None = None):
    """Increment: base vs base+struct. PCA on struct if dim>trigger (or force)."""
    common = sorted(
        set(base.index)
        & set(struct.index)
        & set(y.index)
        & set(folds.id.astype(str))
        & set(ids)
    )
    for k, tr, va, te in mod.rotation_splits(folds, common):
        if not (
            len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE
        ):
            return None
    if len(common) < 40:
        return None
    Xb = base.loc[common]
    Xs = struct.loc[common]
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]

    def maybe_pca(Str, others):
        do = force_pca_struct
        if do is None:
            do = Str.shape[1] > DIM_PCA_TRIGGER
        if do:
            return apply_pca_fold(Str, others, PCA_CAP)
        return Str, others, False

    def choose_a(tr, va, use_struct):
        if use_struct:
            med_b = impute_fit(Xb.loc[tr])
            med_s = impute_fit(Xs.loc[tr])
            Btr = impute_apply(Xb.loc[tr], med_b)
            Bva = impute_apply(Xb.loc[va], med_b)
            Str = impute_apply(Xs.loc[tr], med_s)
            Sva = impute_apply(Xs.loc[va], med_s)
            Str2, [Sva2], _ = maybe_pca(Str, [Sva])
            Xtr = pd.concat([Btr, Str2], axis=1)
            Xva = pd.concat([Bva, Sva2], axis=1)
        else:
            med_b = impute_fit(Xb.loc[tr])
            Xtr = impute_apply(Xb.loc[tr], med_b)
            Xva = impute_apply(Xb.loc[va], med_b)
        return select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])

    def predict(tr, va, te, alpha, use_struct):
        ids_tv = tr + va
        if use_struct:
            med_b = impute_fit(Xb.loc[ids_tv])
            med_s = impute_fit(Xs.loc[ids_tv])
            Btv = impute_apply(Xb.loc[ids_tv], med_b)
            Bte = impute_apply(Xb.loc[te], med_b)
            Stv = impute_apply(Xs.loc[ids_tv], med_s)
            Ste = impute_apply(Xs.loc[te], med_s)
            Stv2, [Ste2], _ = maybe_pca(Stv, [Ste])
            Xtv = pd.concat([Btv, Stv2], axis=1)
            Xte = pd.concat([Bte, Ste2], axis=1)
        else:
            med_b = impute_fit(Xb.loc[ids_tv])
            Xtv = impute_apply(Xb.loc[ids_tv], med_b)
            Xte = impute_apply(Xb.loc[te], med_b)
        sc = StandardScaler()
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(sc.fit_transform(Xtv), yy.loc[ids_tv])
        return pd.Series(m.predict(sc.transform(Xte)), index=te)

    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    for k, tr, va, te in mod.rotation_splits(folds, common):
        a_base = choose_a(tr, va, False)
        a_plus = a_base if mode != "FREE_ALPHA" else choose_a(tr, va, True)
        base_oof.loc[te] = predict(tr, va, te, a_base, False)
        plus_oof.loc[te] = predict(tr, va, te, a_plus, True)
    return {
        "N": len(common),
        "base_dim": int(Xb.shape[1]),
        "struct_dim": int(Xs.shape[1]),
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof),
        "delta": mae(yy, base_oof) - mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
    }


def abl_verdict(dp, ds):
    if dp is None or ds is None:
        return "ABL_DROP"
    if dp > 0.02 and ds > 0.02:
        return "ABL_PRIORITY"
    if dp > 0 and ds > 0:
        return "ABL_TRY"
    if (dp > 0) != (ds > 0):
        return "ABL_MIXED"
    return "ABL_DROP"


def bootstrap_delta(y, b0, p0):
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(b0, float)) - np.abs(y - np.asarray(p0, float))
    boots = np.array(
        [float(d[RNG.integers(0, len(d), len(d))].mean()) for _ in range(B_BOOT)]
    )
    return {
        "mean": float(boots.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "p_le_0": float((boots <= 0).mean()),
    }


def main():
    freeze = json.loads((OUT / "ABLINGUA_FEATURE_FREEZE.json").read_text())
    assert freeze["pooling"]["CLS"] == "CLS_UNSUPPORTED"

    H = load_parquet_X(EMB / "ablingua600m_H_mean.parquet")
    L = load_parquet_X(EMB / "ablingua600m_L_mean.parquet")
    HL = load_parquet_X(EMB / "ablingua600m_HL_mean_concat.parquet")
    ab = load_ablang2()
    ids = [str(x) for x in pd.read_csv(DEV)["id"]]
    assert len(ids) == 162
    seq = load_seq_basic(ids)
    X_pair, m1 = load_pair_m1()
    y = mod.load_y("TmApp")
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    # Current competition recipe components
    ablang_paired = ab["AbLang2_HL_paired"]
    base_ablang_seq = pd.concat([ablang_paired, seq], axis=1, join="inner")
    base_ablang_seq = base_ablang_seq.loc[:, ~base_ablang_seq.columns.duplicated()]
    recipe = pd.concat([base_ablang_seq, X_pair, m1], axis=1, join="inner")
    recipe = recipe.loc[:, ~recipe.columns.duplicated()]

    # Replacement base: AbLingua HL instead of AbLang2
    base_abl_seq = pd.concat([HL.reindex(ids), seq], axis=1, join="inner")
    base_abl_seq = base_abl_seq.loc[:, ~base_abl_seq.columns.duplicated()]
    recipe_replace = pd.concat([base_abl_seq, X_pair, m1], axis=1, join="inner")
    recipe_replace = recipe_replace.loc[:, ~recipe_replace.columns.duplicated()]

    rows = []
    oof_store = {}

    def add_row(**kw):
        rows.append(kw)
        print(
            f"{kw.get('representation')} {kw.get('fusion_recipe')} "
            f"{kw.get('dimensionality_mode')} {kw.get('downstream_model')} "
            f"P={kw.get('primary_mae')} S={kw.get('shadow_mae')} "
            f"dP={kw.get('primary_delta_vs_relevant_baseline')} "
            f"dS={kw.get('shadow_delta_vs_relevant_baseline')}",
            flush=True,
        )

    # ---- Standalone PLM comparisons ----
    stand_specs = [
        ("AbLingua_H_mean", "MASKED_MEAN", "H", H),
        ("AbLingua_L_mean", "MASKED_MEAN", "L", L),
        ("AbLingua_HL_mean_concat", "MASKED_MEAN", "HL", HL),
        ("AbLang2_HL_paired", "AbLang2_default", "HL_paired", ablang_paired),
        ("AbLang2_HL_concat", "AbLang2_default", "HL", ab["AbLang2_HL_concat"]),
        ("AbLang2_H", "AbLang2_default", "H", ab["AbLang2_H"]),
        ("AbLang2_L", "AbLang2_default", "L", ab["AbLang2_L"]),
    ]
    stand_results = {}
    for name, pooling, chains, X in stand_specs:
        for dim_mode in ["raw", "PCA32"]:
            for model_kind in ["Ridge"]:
                rp = run_standalone(X, y, primary, ids, dim_mode, model_kind)
                rs = run_standalone(X, y, shadow, ids, dim_mode, model_kind)
                key = (name, dim_mode, model_kind)
                stand_results[key] = {"P": rp, "S": rs}
                if rp is None or rs is None:
                    continue
                # baseline: AbLang2 same chain family under same dim/model
                if name.startswith("AbLingua"):
                    if chains == "H":
                        bname = "AbLang2_H"
                    elif chains == "L":
                        bname = "AbLang2_L"
                    else:
                        bname = "AbLang2_HL_paired"
                    bp = stand_results.get((bname, dim_mode, model_kind), {}).get("P")
                    bs = stand_results.get((bname, dim_mode, model_kind), {}).get("S")
                    # may not be computed yet if order wrong — compute later pass
                    dP = (bp["mae"] - rp["mae"]) if bp else None
                    dS = (bs["mae"] - rs["mae"]) if bs else None
                else:
                    dP, dS = 0.0, 0.0
                add_row(
                    representation=name,
                    pooling=pooling,
                    chains=chains,
                    fusion_recipe="standalone",
                    downstream_model=model_kind,
                    dimensionality_mode=dim_mode,
                    primary_mae=rp["mae"],
                    shadow_mae=rs["mae"],
                    primary_delta_vs_relevant_baseline=dP,
                    shadow_delta_vs_relevant_baseline=dS,
                    fixed_alpha_primary_delta=None,
                    fixed_alpha_shadow_delta=None,
                    n_dev=rp["N"],
                    notes="standalone Simple TVT",
                )
            # secondary fixed SVR for paired only
            if chains in ("HL", "HL_paired") and name in (
                "AbLingua_HL_mean_concat",
                "AbLang2_HL_paired",
            ):
                rp = run_standalone(X, y, primary, ids, "PCA32", "SVR_fixed")
                rs = run_standalone(X, y, shadow, ids, "PCA32", "SVR_fixed")
                if rp and rs:
                    stand_results[(name, "PCA32", "SVR_fixed")] = {"P": rp, "S": rs}
                    add_row(
                        representation=name,
                        pooling=pooling,
                        chains=chains,
                        fusion_recipe="standalone",
                        downstream_model="SVR_fixed_C3_g0.01_e0.1",
                        dimensionality_mode="PCA32",
                        primary_mae=rp["mae"],
                        shadow_mae=rs["mae"],
                        primary_delta_vs_relevant_baseline=None,
                        shadow_delta_vs_relevant_baseline=None,
                        fixed_alpha_primary_delta=None,
                        fixed_alpha_shadow_delta=None,
                        n_dev=rp["N"],
                        notes="Stage2 fixed SVR replay; no Optuna",
                    )

    # Fix AbLingua standalone deltas now that AbLang2 exists
    for r in rows:
        if r["fusion_recipe"] != "standalone" or not str(r["representation"]).startswith(
            "AbLingua"
        ):
            continue
        if r["downstream_model"] != "Ridge":
            continue
        chains = r["chains"]
        bname = (
            "AbLang2_H"
            if chains == "H"
            else "AbLang2_L"
            if chains == "L"
            else "AbLang2_HL_paired"
        )
        bp = stand_results.get((bname, r["dimensionality_mode"], "Ridge"), {}).get("P")
        bs = stand_results.get((bname, r["dimensionality_mode"], "Ridge"), {}).get("S")
        if bp and bs:
            r["primary_delta_vs_relevant_baseline"] = bp["mae"] - r["primary_mae"]
            r["shadow_delta_vs_relevant_baseline"] = bs["mae"] - r["shadow_mae"]

    # SVR deltas
    for r in rows:
        if r["downstream_model"] != "SVR_fixed_C3_g0.01_e0.1":
            continue
        if r["representation"] != "AbLingua_HL_mean_concat":
            continue
        bp = stand_results.get(("AbLang2_HL_paired", "PCA32", "SVR_fixed"), {}).get("P")
        bs = stand_results.get(("AbLang2_HL_paired", "PCA32", "SVR_fixed"), {}).get("S")
        if bp and bs:
            r["primary_delta_vs_relevant_baseline"] = bp["mae"] - r["primary_mae"]
            r["shadow_delta_vs_relevant_baseline"] = bs["mae"] - r["shadow_mae"]

    # ---- Fusion: AbLingua + SEQ_BASIC ----
    for dim_force, dim_label in [(None, "auto_PCA_if_gt200"), (True, "PCA32_forced"), (False, "raw")]:
        # Only auto + forced PCA32 for high-dim; skip raw if too high for stability? User said raw+PCA32 if protocol tests both.
        if dim_label == "raw":
            force = False
        elif dim_label == "PCA32_forced":
            force = True
        else:
            force = None
        # Use empty-ish: evaluate HL+SEQ as absolute via standalone concat
        X_fuse = pd.concat([HL.reindex(ids), seq], axis=1, join="inner")
        # Also AbLang2+SEQ as baseline absolute
        # Incremental: SEQ_BASIC base + AbLingua struct
        for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
            # parent = SEQ_BASIC alone vs SEQ+AbLingua — better: parent AbLang2+SEQ vs AbLingua+SEQ absolute
            pass

    # Absolute MAE for PLM+SEQ via standalone on concat matrix
    for plm_name, plmX, pooling in [
        ("AbLingua_HL_mean_concat", HL, "MASKED_MEAN"),
        ("AbLang2_HL_paired", ablang_paired, "AbLang2_default"),
    ]:
        for dim_mode in ["raw", "PCA32"]:
            Xc = pd.concat([plmX.reindex(ids), seq], axis=1, join="inner")
            # For PCA32 on whole concat — Stage2 fused differently; here fold-local PCA on full X
            rp = run_standalone(Xc, y, primary, ids, dim_mode, "Ridge")
            rs = run_standalone(Xc, y, shadow, ids, dim_mode, "Ridge")
            if not rp or not rs:
                continue
            key = f"{plm_name}+SEQ_BASIC"
            # delta vs AbLang2+SEQ under same dim — fill in second pass
            add_row(
                representation=plm_name,
                pooling=pooling,
                chains="HL",
                fusion_recipe="PLM+SEQ_BASIC",
                downstream_model="Ridge",
                dimensionality_mode=dim_mode,
                primary_mae=rp["mae"],
                shadow_mae=rs["mae"],
                primary_delta_vs_relevant_baseline=None,
                shadow_delta_vs_relevant_baseline=None,
                fixed_alpha_primary_delta=None,
                fixed_alpha_shadow_delta=None,
                n_dev=rp["N"],
                notes="direct concat PLM+SEQ_BASIC",
            )
            oof_store[(key, dim_mode, "P")] = rp
            oof_store[(key, dim_mode, "S")] = rs

    for r in rows:
        if r["fusion_recipe"] != "PLM+SEQ_BASIC" or r["representation"] != "AbLingua_HL_mean_concat":
            continue
        parent_p = oof_store.get(("AbLang2_HL_paired+SEQ_BASIC", r["dimensionality_mode"], "P"))
        parent_s = oof_store.get(("AbLang2_HL_paired+SEQ_BASIC", r["dimensionality_mode"], "S"))
        if parent_p and parent_s:
            r["primary_delta_vs_relevant_baseline"] = parent_p["mae"] - r["primary_mae"]
            r["shadow_delta_vs_relevant_baseline"] = parent_s["mae"] - r["shadow_mae"]
            r["notes"] = "delta vs AbLang2_HL_paired+SEQ_BASIC same dim"

    # Incremental: SEQ as base, AbLingua as struct (for fixed-alpha)
    for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
        out_p = run_incr(seq, HL, y, primary, ids, mode, force_pca_struct=True)
        out_s = run_incr(seq, HL, y, shadow, ids, mode, force_pca_struct=True)
        if out_p and out_s:
            add_row(
                representation="AbLingua_HL_mean_concat",
                pooling="MASKED_MEAN",
                chains="HL",
                fusion_recipe="SEQ_BASIC+AbLingua_incr",
                downstream_model="Ridge",
                dimensionality_mode="PCA32_struct",
                primary_mae=out_p["plus_mae"],
                shadow_mae=out_s["plus_mae"],
                primary_delta_vs_relevant_baseline=out_p["delta"],
                shadow_delta_vs_relevant_baseline=out_s["delta"],
                fixed_alpha_primary_delta=out_p["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                fixed_alpha_shadow_delta=out_s["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                n_dev=out_p["N"],
                notes=f"incr vs SEQ_BASIC alone; mode={mode}; parent_mae P={out_p['base_mae']:.4f} S={out_s['base_mae']:.4f}",
            )
            if mode == "FREE_ALPHA" and out_p["delta"] > 0 and out_s["delta"] > 0:
                oof_store[("SEQ+ABL_incr", "P")] = out_p
                oof_store[("SEQ+ABL_incr", "S")] = out_s

    # Same incr for AbLang2 for fairness
    for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
        out_p = run_incr(seq, ablang_paired, y, primary, ids, mode, force_pca_struct=True)
        out_s = run_incr(seq, ablang_paired, y, shadow, ids, mode, force_pca_struct=True)
        if out_p and out_s:
            add_row(
                representation="AbLang2_HL_paired",
                pooling="AbLang2_default",
                chains="HL_paired",
                fusion_recipe="SEQ_BASIC+AbLang2_incr",
                downstream_model="Ridge",
                dimensionality_mode="PCA32_struct",
                primary_mae=out_p["plus_mae"],
                shadow_mae=out_s["plus_mae"],
                primary_delta_vs_relevant_baseline=out_p["delta"],
                shadow_delta_vs_relevant_baseline=out_s["delta"],
                fixed_alpha_primary_delta=out_p["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                fixed_alpha_shadow_delta=out_s["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                n_dev=out_p["N"],
                notes=f"incr vs SEQ_BASIC alone; mode={mode}",
            )

    # ---- Competition recipe REPLACEMENT / ADDITION ----
    # Parent recipe absolute MAE via standalone on recipe matrix (auto: treat as raw high-dim with PCA32)
    for name, Xmat, note in [
        ("current_recipe", recipe, "AbLang2+SEQ+BIOEMU_PAIR+M1"),
        ("recipe_replace_AbLingua", recipe_replace, "AbLingua+SEQ+BIOEMU_PAIR+M1"),
    ]:
        for dim_mode in ["PCA32"]:  # high dim → PCA32 fold-local on full matrix
            rp = run_standalone(Xmat, y, primary, ids, dim_mode, "Ridge")
            rs = run_standalone(Xmat, y, shadow, ids, dim_mode, "Ridge")
            if not rp or not rs:
                continue
            oof_store[(name, dim_mode, "P")] = rp
            oof_store[(name, dim_mode, "S")] = rs
            dP = dS = None
            if name == "recipe_replace_AbLingua":
                pp = oof_store.get(("current_recipe", dim_mode, "P"))
                ps = oof_store.get(("current_recipe", dim_mode, "S"))
                if pp and ps:
                    dP = pp["mae"] - rp["mae"]
                    dS = ps["mae"] - rs["mae"]
            add_row(
                representation="AbLingua_HL_mean_concat" if "AbLingua" in name else "AbLang2_HL_paired",
                pooling="MASKED_MEAN" if "AbLingua" in name else "AbLang2_default",
                chains="HL",
                fusion_recipe=name,
                downstream_model="Ridge",
                dimensionality_mode=dim_mode,
                primary_mae=rp["mae"],
                shadow_mae=rs["mae"],
                primary_delta_vs_relevant_baseline=dP if dP is not None else 0.0,
                shadow_delta_vs_relevant_baseline=dS if dS is not None else 0.0,
                fixed_alpha_primary_delta=None,
                fixed_alpha_shadow_delta=None,
                n_dev=rp["N"],
                notes=note,
            )

    # Also evaluate recipe with Simple TVT structure-block PCA convention (BASE+struct)
    # ADDITION: recipe as base, AbLingua as struct
    for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
        out_p = run_incr(recipe, HL, y, primary, ids, mode, force_pca_struct=True)
        out_s = run_incr(recipe, HL, y, shadow, ids, mode, force_pca_struct=True)
        if out_p and out_s:
            add_row(
                representation="AbLingua_HL_mean_concat",
                pooling="MASKED_MEAN",
                chains="HL",
                fusion_recipe="ADD_to_current_recipe",
                downstream_model="Ridge",
                dimensionality_mode="PCA32_struct",
                primary_mae=out_p["plus_mae"],
                shadow_mae=out_s["plus_mae"],
                primary_delta_vs_relevant_baseline=out_p["delta"],
                shadow_delta_vs_relevant_baseline=out_s["delta"],
                fixed_alpha_primary_delta=out_p["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                fixed_alpha_shadow_delta=out_s["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                n_dev=out_p["N"],
                notes=f"mode={mode}; parent current_recipe MAE P={out_p['base_mae']:.4f} S={out_s['base_mae']:.4f}",
            )
            if mode == "FREE_ALPHA":
                oof_store[("ADD_recipe", "P")] = out_p
                oof_store[("ADD_recipe", "S")] = out_s

    # REPLACEMENT via incr convention: compare recipe_replace vs recipe as absolutes already done.
    # Also: BASE(AbLingua+SEQ) + PAIR + M1 vs BASE(AbLang2+SEQ) using structure blocks
    for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
        struct_extra = pd.concat([X_pair, m1], axis=1, join="inner")
        out_p_parent = run_incr(base_ablang_seq, struct_extra, y, primary, ids, mode, force_pca_struct=False)
        out_s_parent = run_incr(base_ablang_seq, struct_extra, y, shadow, ids, mode, force_pca_struct=False)
        out_p_rep = run_incr(base_abl_seq, struct_extra, y, primary, ids, mode, force_pca_struct=False)
        out_s_rep = run_incr(base_abl_seq, struct_extra, y, shadow, ids, mode, force_pca_struct=False)
        if out_p_parent and out_p_rep and out_s_parent and out_s_rep:
            dP = out_p_parent["plus_mae"] - out_p_rep["plus_mae"]
            dS = out_s_parent["plus_mae"] - out_s_rep["plus_mae"]
            add_row(
                representation="AbLingua_HL_mean_concat",
                pooling="MASKED_MEAN",
                chains="HL",
                fusion_recipe="REPLACE_AbLang2_in_recipe",
                downstream_model="Ridge",
                dimensionality_mode="struct_noPCA",
                primary_mae=out_p_rep["plus_mae"],
                shadow_mae=out_s_rep["plus_mae"],
                primary_delta_vs_relevant_baseline=dP,
                shadow_delta_vs_relevant_baseline=dS,
                fixed_alpha_primary_delta=dP if mode == "BASE_FIXED_ALPHA" else None,
                fixed_alpha_shadow_delta=dS if mode == "BASE_FIXED_ALPHA" else None,
                n_dev=out_p_rep["N"],
                notes=f"mode={mode}; parent_plus MAE P={out_p_parent['plus_mae']:.4f} S={out_s_parent['plus_mae']:.4f}",
            )
            if mode == "FREE_ALPHA":
                oof_store[("REPLACE", "P")] = {
                    "y": out_p_rep["y"],
                    "base_oof": out_p_parent["plus_oof"].reindex(out_p_rep["y"].index),
                    "plus_oof": out_p_rep["plus_oof"],
                    "delta": dP,
                }
                oof_store[("REPLACE", "S")] = {
                    "y": out_s_rep["y"],
                    "base_oof": out_s_parent["plus_oof"].reindex(out_s_rep["y"].index),
                    "plus_oof": out_s_rep["plus_oof"],
                    "delta": dS,
                }

    # ---- Direct AbLang2 + AbLingua fusion ----
    for mode in ["FREE_ALPHA", "BASE_FIXED_ALPHA"]:
        # parent: SEQ+AbLang2 ; plus: +AbLingua
        parent_base = pd.concat([seq, ablang_paired], axis=1, join="inner")
        out_p = run_incr(parent_base, HL, y, primary, ids, mode, force_pca_struct=True)
        out_s = run_incr(parent_base, HL, y, shadow, ids, mode, force_pca_struct=True)
        if out_p and out_s:
            add_row(
                representation="AbLingua_HL_mean_concat",
                pooling="MASKED_MEAN",
                chains="HL",
                fusion_recipe="SEQ_BASIC+AbLang2+AbLingua",
                downstream_model="Ridge",
                dimensionality_mode="PCA32_struct",
                primary_mae=out_p["plus_mae"],
                shadow_mae=out_s["plus_mae"],
                primary_delta_vs_relevant_baseline=out_p["delta"],
                shadow_delta_vs_relevant_baseline=out_s["delta"],
                fixed_alpha_primary_delta=out_p["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                fixed_alpha_shadow_delta=out_s["delta"] if mode == "BASE_FIXED_ALPHA" else None,
                n_dev=out_p["N"],
                notes=f"mode={mode}; parent SEQ+AbLang2 MAE P={out_p['base_mae']:.4f} S={out_s['base_mae']:.4f}",
            )
            if mode == "FREE_ALPHA":
                oof_store[("A2+ABL", "P")] = out_p
                oof_store[("A2+ABL", "S")] = out_s

    df = pd.DataFrame(rows)
    # Fill fixed-alpha columns for FREE rows that have FIXED siblings
    df.to_csv(RES / "ABLINGUA_SIMPLE_TVT_RESULTS.csv", index=False)
    df.to_csv(OUT / "ABLINGUA_SIMPLE_TVT_RESULTS.csv", index=False)

    # Bootstraps for paired candidates positive both sides
    boots = {}
    for key in ["ADD_recipe", "A2+ABL", "SEQ+ABL_incr", "REPLACE"]:
        op = oof_store.get((key, "P"))
        os_ = oof_store.get((key, "S"))
        if not op or not os_:
            continue
        dP = op.get("delta")
        dS = os_.get("delta")
        if dP is not None and dS is not None and dP > 0 and dS > 0:
            boots[key] = {
                "Primary": bootstrap_delta(op["y"], op["base_oof"], op["plus_oof"]),
                "Shadow": bootstrap_delta(os_["y"], os_["base_oof"], os_["plus_oof"]),
            }

    # Also bootstrap standalone AbLingua vs AbLang2 if both positive delta
    for dim_mode in ["raw", "PCA32"]:
        abl = stand_results.get(("AbLingua_HL_mean_concat", dim_mode, "Ridge"))
        a2 = stand_results.get(("AbLang2_HL_paired", dim_mode, "Ridge"))
        if not abl or not a2 or not abl["P"] or not a2["P"]:
            continue
        dP = a2["P"]["mae"] - abl["P"]["mae"]
        dS = a2["S"]["mae"] - abl["S"]["mae"]
        if dP > 0 and dS > 0:
            # treat AbLang2 oof as baseline predictions
            boots[f"standalone_{dim_mode}"] = {
                "Primary": bootstrap_delta(abl["P"]["y"], a2["P"]["oof"], abl["P"]["oof"]),
                "Shadow": bootstrap_delta(abl["S"]["y"], a2["S"]["oof"], abl["S"]["oof"]),
                "delta_P": dP,
                "delta_S": dS,
            }

    (RES / "ABLINGUA_BOOTSTRAP.json").write_text(json.dumps(boots, indent=2))
    print("Wrote results", len(df), "rows; boots", list(boots.keys()))


if __name__ == "__main__":
    main()
