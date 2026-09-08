#!/usr/bin/env python3
"""Simple TVT for AbLingua structure-guided pooling (predeclared only)."""
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

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
OUT = FP / "ablingua600m"
RES = OUT / "results"
EMB = OUT / "embeddings"
EMB_G = OUT / "embeddings_guided"
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


def load_X(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["id"] = df["id"].astype(str)
    return df.set_index("id").select_dtypes(include=[np.number]).astype(float)


def impute_fit(X):
    return X.median(numeric_only=True).fillna(0.0)


def impute_apply(X, med):
    return X.fillna(med).fillna(0.0)


def select_alpha(Xtr, ytr, Xva, yva):
    best_a, best = 100.0, np.inf
    Xtr_v = np.asarray(Xtr, dtype=float)
    Xva_v = np.asarray(Xva, dtype=float)
    ytr_v = np.asarray(ytr, dtype=float)
    yva_v = np.asarray(yva, dtype=float)
    for a in ALPHAS:
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr_v), ytr_v)
        score = mae(yva_v, m.predict(sc.transform(Xva_v)))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def fit_predict_ridge(Xtv, ytv, Xte, alpha):
    sc = StandardScaler()
    m = Ridge(alpha=alpha, random_state=0)
    Xtv_v = np.asarray(Xtv, dtype=float)
    Xte_v = np.asarray(Xte, dtype=float)
    m.fit(sc.fit_transform(Xtv_v), np.asarray(ytv, dtype=float))
    return m.predict(sc.transform(Xte_v))


def pca_block(Xtr, others, n=PCA_CAP):
    k = min(n, Xtr.shape[0] - 1, Xtr.shape[1])
    if k < 2:
        return Xtr, others
    pca = PCA(n_components=k, random_state=0)
    Ztr = pca.fit_transform(np.asarray(Xtr, dtype=float))
    cols = [f"pca_{i}" for i in range(k)]
    outs = [
        pd.DataFrame(
            pca.transform(np.asarray(X, dtype=float)), index=X.index, columns=cols
        )
        for X in others
    ]
    return pd.DataFrame(Ztr, index=Xtr.index, columns=cols), outs


def run_standalone_pca32(X, y, folds, ids):
    common = sorted(set(X.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    for _, tr, va, te in mod.rotation_splits(folds, common):
        if not (len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE):
            return None
    if len(common) < 40:
        return None
    Xx, yy = X.loc[common], y.loc[common]
    oof = pd.Series(index=common, dtype=float)
    for _, tr, va, te in mod.rotation_splits(folds, common):
        med = impute_fit(Xx.loc[tr])
        Xtr = impute_apply(Xx.loc[tr], med)
        Xva = impute_apply(Xx.loc[va], med)
        Xtr, [Xva] = pca_block(Xtr, [Xva])
        a = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va])
        ids_tv = tr + va
        med2 = impute_fit(Xx.loc[ids_tv])
        Xtv = impute_apply(Xx.loc[ids_tv], med2)
        Xte = impute_apply(Xx.loc[te], med2)
        Xtv, [Xte] = pca_block(Xtv, [Xte])
        oof.loc[te] = fit_predict_ridge(Xtv, yy.loc[ids_tv], Xte, a)
    return {"N": len(common), "dim": int(Xx.shape[1]), "mae": mae(yy, oof), "oof": oof, "y": yy}


def reduce_abl_blocks(blocks_tr: list[pd.DataFrame], blocks_others: list[list[pd.DataFrame]]):
    """PCA each AbLingua block fold-locally (unique column prefixes per block)."""
    out_tr = []
    out_others = [[] for _ in blocks_others]
    for bi, Btr in enumerate(blocks_tr):
        others_b = [blocks_others[j][bi] for j in range(len(blocks_others))]
        Btr2, others2 = pca_block(Btr, others_b)
        pref = f"abl{bi}_"
        Btr2 = Btr2.rename(columns={c: pref + str(c) for c in Btr2.columns})
        others2 = [o.rename(columns={c: pref + str(c) for c in o.columns}) for o in others2]
        out_tr.append(Btr2)
        for j, o in enumerate(others2):
            out_others[j].append(o)
    return pd.concat(out_tr, axis=1), [pd.concat(o, axis=1) for o in out_others]


def run_parent_plus(
    parent: pd.DataFrame,
    always_blocks: list[pd.DataFrame],
    extra_blocks: list[pd.DataFrame],
    y,
    folds,
    ids,
    mode: str,
    fixed_alphas: list[float] | None = None,
):
    """base = parent + always_blocks; plus = base + extra_blocks.
    PCA32 each AbLingua block fold-locally.
    BASE_FIXED_ALPHA: use alpha from base (or provided fixed_alphas per fold).
    """
    all_blocks = always_blocks + extra_blocks
    common = sorted(
        set(parent.index)
        & set(y.index)
        & set(folds.id.astype(str))
        & set(ids)
        & set.intersection(*[set(b.index) for b in all_blocks])
    )
    for _, tr, va, te in mod.rotation_splits(folds, common):
        if not (len(tr) >= mod.MIN_TR and len(va) >= mod.MIN_VA and len(te) >= mod.MIN_TE):
            return None
    if len(common) < 40:
        return None
    Xp = parent.loc[common]
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
        Btr_c, Bothers_c = reduce_abl_blocks(Btr, Bothers)
        Xtr = pd.concat([Ptr, Btr_c], axis=1)
        Xothers = [pd.concat([p, bb], axis=1) for p, bb in zip(Pothers, Bothers_c)]
        return Xtr, Xothers

    base_blocks = always
    plus_blocks = always + extras
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    alphas_base, alphas_plus = [], []
    fold_i = 0
    for _, tr, va, te in mod.rotation_splits(folds, common):
        Xtr_b, [Xva_b] = build(tr, [va], base_blocks)
        a_base = select_alpha(Xtr_b, yy.loc[tr], Xva_b, yy.loc[va])
        if fixed_alphas is not None:
            a_base = fixed_alphas[fold_i]
        if mode == "FREE_ALPHA":
            Xtr_p, [Xva_p] = build(tr, [va], plus_blocks)
            a_plus = select_alpha(Xtr_p, yy.loc[tr], Xva_p, yy.loc[va])
        else:
            a_plus = a_base
        alphas_base.append(a_base)
        alphas_plus.append(a_plus)
        ids_tv = tr + va
        Xtv_b, [Xte_b] = build(ids_tv, [te], base_blocks)
        base_oof.loc[te] = fit_predict_ridge(Xtv_b, yy.loc[ids_tv], Xte_b, a_base)
        Xtv_p, [Xte_p] = build(ids_tv, [te], plus_blocks)
        plus_oof.loc[te] = fit_predict_ridge(Xtv_p, yy.loc[ids_tv], Xte_p, a_plus)
        fold_i += 1

    return {
        "N": len(common),
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof),
        "delta": mae(yy, base_oof) - mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
        "alphas_base": alphas_base,
        "struct_dim": int(sum(b.shape[1] for b in extras)),
    }


def verdict(dp, ds, fdp, fds):
    if dp is None or ds is None:
        return "GUIDED_DROP"
    if dp > 0 and ds > 0 and fdp is not None and fds is not None and fdp > 0 and fds > 0:
        if min(dp, ds) > 0.02:
            return "GUIDED_PRIORITY"
        return "GUIDED_TRY"
    if dp > 0 and ds > 0:
        return "GUIDED_TRY" if min(dp, ds) <= 0.02 else "GUIDED_PRIORITY"
    if (dp > 0) != (ds > 0):
        return "GUIDED_MIXED"
    return "GUIDED_DROP"


def bootstrap(y, b0, p0):
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(b0, float)) - np.abs(y - np.asarray(p0, float))
    boots = np.array([float(d[RNG.integers(0, len(d), len(d))].mean()) for _ in range(B_BOOT)])
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def load_recipe_and_global(ids):
    z = np.load(EMB_NPZ, allow_pickle=True)
    assert [str(x) for x in z["ids"]] == ids
    ablang = pd.DataFrame(np.asarray(z["ablang2__HL_paired"], float), index=ids)
    ablang.columns = [f"ablang2_{i}" for i in range(ablang.shape[1])]
    tables = build_all_feature_tables(
        pd.read_csv(DEV), pd.read_csv(ANN), pd.read_csv(REGIONS)
    )
    seq_b, _ = make_xy(tables, "SEQ_BASIC")
    seq_b.index = ids
    seq_b.columns = [f"seqB_{c}" for c in seq_b.columns]
    bio = pd.read_csv(BIO)
    bio["id"] = bio["id"].astype(str)
    pair = bio.set_index("id")[[c for c in bio.columns if "ca_rmsd" in c.lower()]].apply(
        pd.to_numeric, errors="coerce"
    )
    m1 = mod.numeric_X(pd.read_csv(M1), drop_cols=["status", "extraction_status"])
    recipe = pd.concat([ablang, seq_b, pair, m1], axis=1, join="inner")
    recipe = recipe.loc[:, ~recipe.columns.duplicated()]
    glob = load_X(EMB / "ablingua600m_HL_mean_concat.parquet")
    # PARENT = recipe + global (raw concat; PCA applied to AbLingua blocks in run_parent_plus)
    # For parent base we need recipe+global as the base model features.
    # Implement parent as: base_features = recipe + PCA(global) inside folds via abl_blocks empty for base...
    # Actually parent MAE should be CURRENT_RECIPE + GLOBAL evaluated as the "base" in increment.
    # So parent matrix = concat(recipe, global) with PCA on global block only when fitting.
    return recipe, glob


def main():
    freeze = json.loads((OUT / "ABLINGUA_GUIDED_POOLING_FREEZE.json").read_text())
    ids = [str(x) for x in pd.read_csv(DEV)["id"]]
    y = mod.load_y("TmApp")
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    recipe, glob = load_recipe_and_global(ids)
    # Build PARENT feature as recipe; GLOBAL added as first AbLingua block always in base via:
    # We define PARENT predictions using run_parent_plus(recipe, [glob], ...) plus_mae as parent.
    # For increment of G: run_parent_plus(recipe, [glob, G]) vs parent = run_parent_plus(recipe, [glob]).

    guided = {
        name: load_X(EMB_G / f"ablingua600m_{name}.parquet")
        for name in [
            "CDR_ALL",
            "CDR3",
            "EXPOSED",
            "BURIED",
            "RASA_WEIGHTED",
            "BURIED_WEIGHTED",
            "EXPOSED_CDR",
            "CDR_FR_SPLIT",
            "H_CDR_ONLY",
            "L_CDR_ONLY",
        ]
    }

    rows = []
    oofs = {}

    # Diagnostic standalone
    stand = {"GLOBAL": glob, **{k: guided[k] for k in guided if k not in ("H_CDR_ONLY", "L_CDR_ONLY")}}
    for name, X in stand.items():
        rp = run_standalone_pca32(X, y, primary, ids)
        rs = run_standalone_pca32(X, y, shadow, ids)
        rows.append(
            {
                "candidate": f"standalone_{name}",
                "pooling_family": name,
                "raw_dim": int(X.shape[1]),
                "pca_mode": "PCA32",
                "parent": "none_standalone",
                "primary_mae": None if not rp else rp["mae"],
                "shadow_mae": None if not rs else rs["mae"],
                "delta_primary": None,
                "delta_shadow": None,
                "fixed_alpha_delta_primary": None,
                "fixed_alpha_delta_shadow": None,
                "bootstrap_ci_primary": "",
                "bootstrap_ci_shadow": "",
                "verdict": "DIAGNOSTIC",
                "notes": "standalone replacement diagnostic",
            }
        )
        print(f"standalone {name} P={rp['mae'] if rp else None} S={rs['mae'] if rs else None}", flush=True)

    # Parent baseline: recipe + GLOBAL
    parent_free = {}
    parent_fixed = {}
    for tag, folds in [("P", primary), ("S", shadow)]:
        # free: base=recipe, plus=recipe+GLOBAL  → plus is PARENT
        parent_free[tag] = run_parent_plus(
            recipe, [glob], [], y, folds, ids, "FREE_ALPHA"
        )
        parent_fixed[tag] = run_parent_plus(
            recipe, [glob], [], y, folds, ids, "BASE_FIXED_ALPHA"
        )
        print(
            f"PARENT {tag} FREE mae={parent_free[tag]['plus_mae']:.6f} "
            f"(recipe_only={parent_free[tag]['base_mae']:.6f})",
            flush=True,
        )

    def add_incr(cand_name, family, blocks, notes=""):
        free, fixed = {}, {}
        for tag, folds in [("P", primary), ("S", shadow)]:
            # base = PARENT (recipe+glob), plus = PARENT+guided
            free[tag] = run_parent_plus(
                recipe, [glob], blocks, y, folds, ids, "FREE_ALPHA"
            )
            fixed[tag] = run_parent_plus(
                recipe,
                [glob],
                blocks,
                y,
                folds,
                ids,
                "BASE_FIXED_ALPHA",
                fixed_alphas=parent_free[tag]["alphas_base"],
            )
        # deltas: improvement of plus over base within same call (= PARENT vs PARENT+G)
        dP = free["P"]["delta"]
        dS = free["S"]["delta"]
        fP = fixed["P"]["delta"]
        fS = fixed["S"]["delta"]
        v = verdict(dP, dS, fP, fS)
        ciP = ciS = ""
        if dP > 0 and dS > 0:
            ciP = str(bootstrap(free["P"]["y"], free["P"]["base_oof"], free["P"]["plus_oof"]))
            ciS = str(bootstrap(free["S"]["y"], free["S"]["base_oof"], free["S"]["plus_oof"]))
            oofs[cand_name] = free
        rows.append(
            {
                "candidate": cand_name,
                "pooling_family": family,
                "raw_dim": int(sum(b.shape[1] for b in blocks)),
                "pca_mode": "PCA32_per_abl_block",
                "parent": "CURRENT_RECIPE+AbLingua_GLOBAL",
                "primary_mae": free["P"]["plus_mae"],
                "shadow_mae": free["S"]["plus_mae"],
                "delta_primary": dP,
                "delta_shadow": dS,
                "fixed_alpha_delta_primary": fP,
                "fixed_alpha_delta_shadow": fS,
                "bootstrap_ci_primary": ciP,
                "bootstrap_ci_shadow": ciS,
                "verdict": v,
                "notes": notes
                + f"; parent_mae P={free['P']['base_mae']:.4f} S={free['S']['base_mae']:.4f}",
            }
        )
        print(
            f"{cand_name} P={free['P']['plus_mae']:.4f} S={free['S']['plus_mae']:.4f} "
            f"dP={dP:+.4f} dS={dS:+.4f} fixed={fP:+.4f}/{fS:+.4f} {v}",
            flush=True,
        )

    # Single guided increments
    for name in [
        "CDR_ALL",
        "CDR3",
        "EXPOSED",
        "BURIED",
        "RASA_WEIGHTED",
        "BURIED_WEIGHTED",
        "EXPOSED_CDR",
        "CDR_FR_SPLIT",
    ]:
        add_incr(f"PARENT+{name}", name, [guided[name]])

    # Combos
    add_incr(
        "COMBO_1_LOCAL_SURFACE",
        "CDR_ALL+RASA_WEIGHTED",
        [guided["CDR_ALL"], guided["RASA_WEIGHTED"]],
        notes="predeclared combo1",
    )
    add_incr(
        "COMBO_2_SURFACE_CORE",
        "RASA_WEIGHTED+BURIED_WEIGHTED",
        [guided["RASA_WEIGHTED"], guided["BURIED_WEIGHTED"]],
        notes="predeclared combo2",
    )
    add_incr(
        "COMBO_3_CDR_SURFACE_CORE",
        "EXPOSED_CDR+BURIED_WEIGHTED",
        [guided["EXPOSED_CDR"], guided["BURIED_WEIGHTED"]],
        notes="predeclared combo3",
    )

    # Chain asymmetry on CDR_ALL (priority order)
    add_incr("PARENT+H_CDR_ONLY", "H_CDR_ONLY", [guided["H_CDR_ONLY"]], notes="chain asymmetry")
    add_incr("PARENT+L_CDR_ONLY", "L_CDR_ONLY", [guided["L_CDR_ONLY"]], notes="chain asymmetry")
    # HL_CDR is PARENT+CDR_ALL already

    # Parent row
    rows.insert(
        0,
        {
            "candidate": "PARENT",
            "pooling_family": "GLOBAL",
            "raw_dim": int(glob.shape[1]),
            "pca_mode": "PCA32_per_abl_block",
            "parent": "CURRENT_RECIPE+AbLingua_GLOBAL",
            "primary_mae": parent_free["P"]["plus_mae"],
            "shadow_mae": parent_free["S"]["plus_mae"],
            "delta_primary": 0.0,
            "delta_shadow": 0.0,
            "fixed_alpha_delta_primary": 0.0,
            "fixed_alpha_delta_shadow": 0.0,
            "bootstrap_ci_primary": "",
            "bootstrap_ci_shadow": "",
            "verdict": "PARENT",
            "notes": f"recipe_only P={parent_free['P']['base_mae']:.4f} S={parent_free['S']['base_mae']:.4f}",
        },
    )

    df = pd.DataFrame(rows)
    df.to_csv(RES / "ABLINGUA_GUIDED_POOLING_SIMPLE_TVT_RESULTS.csv", index=False)
    df.to_csv(OUT / "ABLINGUA_GUIDED_POOLING_SIMPLE_TVT_RESULTS.csv", index=False)
    print("Wrote", len(df), "rows", flush=True)


if __name__ == "__main__":
    main()
