#!/usr/bin/env python3
"""Reconcile AbLingua PARENT scores between sprint A (02) and sprint B (04)."""
from __future__ import annotations

import hashlib
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
OUT = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
RES = OUT / "results" / "parent_reconciliation"
RES.mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context_interim_audit"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_CAP = 32

spec = importlib.util.spec_from_file_location(
    "tvt_rescreen", INTERIM / "scripts/run_simple_tvt_rescreen.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["tvt_rescreen"] = mod
spec.loader.exec_module(mod)

import runpy

evA = runpy.run_path(str(OUT / "scripts/02_simple_tvt_eval.py"))
evB = runpy.run_path(str(OUT / "scripts/04_guided_pooling_tvt.py"))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def sha_ids(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()[:16]


def sha_cols(cols) -> str:
    return hashlib.sha256("\n".join(map(str, cols)).encode()).hexdigest()[:16]


def impute_fit(X):
    return X.median(numeric_only=True).fillna(0.0)


def impute_apply(X, med):
    return X.fillna(med).fillna(0.0)


def select_alpha(Xtr, ytr, Xva, yva, as_numpy=False):
    best_a, best = 100.0, np.inf
    Xtr_in = np.asarray(Xtr, float) if as_numpy else Xtr
    Xva_in = np.asarray(Xva, float) if as_numpy else Xva
    for a in ALPHAS:
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr_in), np.asarray(ytr, float))
        score = mae(yva, m.predict(sc.transform(Xva_in)))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def pca32(Xtr, others, prefix=""):
    n = min(PCA_CAP, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n, random_state=0)
    Ztr = pca.fit_transform(np.asarray(Xtr, float))
    cols = [f"{prefix}pca_{i}" for i in range(n)]
    outs = [
        pd.DataFrame(pca.transform(np.asarray(X, float)), index=X.index, columns=cols)
        for X in others
    ]
    return pd.DataFrame(Ztr, index=Xtr.index, columns=cols), outs, n


def run_recipe_plus_abl(
    recipe, HL, y, folds, ids, *, alpha_on_numpy=False, pca_prefix="", label="X"
):
    """Authoritative PARENT model: recipe (no PCA) + PCA32(AbLingua). FREE_ALPHA on concat."""
    common = sorted(
        set(recipe.index) & set(HL.index) & set(y.index) & set(folds.id.astype(str)) & set(ids)
    )
    Xb, Xs = recipe.loc[common].copy(), HL.loc[common].copy()
    ov = [c for c in Xs.columns if c in Xb.columns]
    if ov:
        Xs = Xs.drop(columns=ov)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    fold_meta = []
    for k, tr, va, te in mod.rotation_splits(folds, common):
        med_b = impute_fit(Xb.loc[tr])
        med_s = impute_fit(Xs.loc[tr])
        Btr = impute_apply(Xb.loc[tr], med_b)
        Bva = impute_apply(Xb.loc[va], med_b)
        Str = impute_apply(Xs.loc[tr], med_s)
        Sva = impute_apply(Xs.loc[va], med_s)
        Str2, [Sva2], n_pca = pca32(Str, [Sva], prefix=pca_prefix)
        # alphas
        a_base = select_alpha(Btr, yy.loc[tr], Bva, yy.loc[va], as_numpy=alpha_on_numpy)
        Xtr = pd.concat([Btr, Str2], axis=1)
        Xva = pd.concat([Bva, Sva2], axis=1)
        a_plus = select_alpha(Xtr, yy.loc[tr], Xva, yy.loc[va], as_numpy=alpha_on_numpy)
        ids_tv = tr + va
        med_b2 = impute_fit(Xb.loc[ids_tv])
        med_s2 = impute_fit(Xs.loc[ids_tv])
        Btv = impute_apply(Xb.loc[ids_tv], med_b2)
        Bte = impute_apply(Xb.loc[te], med_b2)
        Stv = impute_apply(Xs.loc[ids_tv], med_s2)
        Ste = impute_apply(Xs.loc[te], med_s2)
        Stv2, [Ste2], _ = pca32(Stv, [Ste], prefix=pca_prefix)
        Xtv = pd.concat([Btv, Stv2], axis=1)
        Xte = pd.concat([Bte, Ste2], axis=1)
        # base
        sc = StandardScaler()
        m = Ridge(alpha=a_base, random_state=0)
        Xin = np.asarray(Btv, float) if alpha_on_numpy else Btv
        Xte_in = np.asarray(Bte, float) if alpha_on_numpy else Bte
        m.fit(sc.fit_transform(Xin), np.asarray(yy.loc[ids_tv], float))
        base_oof.loc[te] = m.predict(sc.transform(Xte_in))
        # plus
        sc2 = StandardScaler()
        m2 = Ridge(alpha=a_plus, random_state=0)
        Xin2 = np.asarray(Xtv, float) if alpha_on_numpy else Xtv
        Xte2 = np.asarray(Xte, float) if alpha_on_numpy else Xte
        m2.fit(sc2.fit_transform(Xin2), np.asarray(yy.loc[ids_tv], float))
        plus_oof.loc[te] = m2.predict(sc2.transform(Xte2))
        fold_meta.append(
            {
                "fold": k,
                "a_base": a_base,
                "a_plus": a_plus,
                "pca_n": n_pca,
                "base_dim": int(Btv.shape[1]),
                "plus_dim": int(Xtv.shape[1]),
                "tr_hash": sha_ids(tr),
                "va_hash": sha_ids(va),
                "te_hash": sha_ids(te),
                "Xtv_sha": sha_arr(np.asarray(Xtv, float)),
                "n_tr": len(tr),
                "n_va": len(va),
                "n_te": len(te),
            }
        )
    return {
        "label": label,
        "common": common,
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof),
        "y": yy,
        "base_oof": base_oof,
        "plus_oof": plus_oof,
        "fold_meta": fold_meta,
    }


def main():
    ids = [str(x) for x in pd.read_csv(ROOT / "competition/data/distribution/dev.csv")["id"]]
    y = mod.load_y("TmApp")
    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)

    # Feature matrices from each script
    ab = evA["load_ablang2"]()
    seq = evA["load_seq_basic"](ids)
    X_pair, m1 = evA["load_pair_m1"]()
    recipe_A = pd.concat([ab["AbLang2_HL_paired"], seq, X_pair, m1], axis=1, join="inner")
    recipe_A = recipe_A.loc[:, ~recipe_A.columns.duplicated()]
    HL_A = evA["load_parquet_X"](OUT / "embeddings/ablingua600m_HL_mean_concat.parquet")
    recipe_B, HL_B = evB["load_recipe_and_global"](ids)

    comps = []
    for name, XA, XB in [
        ("recipe", recipe_A, recipe_B),
        ("AbLingua_GLOBAL", HL_A.reindex(ids), HL_B.reindex(ids)),
    ]:
        same = (
            XA.shape == XB.shape
            and list(XA.index) == list(XB.index)
            and list(XA.columns) == list(XB.columns)
            and np.allclose(XA.to_numpy(float), XB.to_numpy(float), equal_nan=True)
        )
        comps.append(
            {
                "component": name,
                "A_dim": XA.shape[1],
                "B_dim": XB.shape[1],
                "A_sha": sha_arr(XA.to_numpy(float)),
                "B_sha": sha_arr(XB.to_numpy(float)),
                "same": "YES" if same else "NO",
            }
        )
    # RESIDUE_GLOBAL check
    rg = OUT / "embeddings_guided/ablingua600m_RESIDUE_GLOBAL.parquet"
    if rg.exists():
        RG = evA["load_parquet_X"](rg).reindex(ids)
        comps.append(
            {
                "component": "RESIDUE_GLOBAL_vs_HL_mean",
                "A_dim": HL_A.shape[1],
                "B_dim": RG.shape[1],
                "A_sha": sha_arr(HL_A.reindex(ids).to_numpy(float)),
                "B_sha": sha_arr(RG.to_numpy(float)),
                "same": "YES"
                if np.allclose(HL_A.reindex(ids).to_numpy(float), RG.to_numpy(float), atol=1e-5)
                else "NO",
            }
        )
    pd.DataFrame(comps).to_csv(RES / "COMPONENT_COMPARE.csv", index=False)

    emb = OUT / "embeddings/ablingua600m_HL_mean_concat.parquet"
    meta = {
        "n_dev": len(ids),
        "dev_id_hash": sha_ids(ids),
        "fold_primary": str(PRIMARY),
        "fold_shadow": str(SHADOW),
        "fold_primary_sha256": hashlib.sha256(PRIMARY.read_bytes()).hexdigest(),
        "fold_shadow_sha256": hashlib.sha256(SHADOW.read_bytes()).hexdigest(),
        "ablingua_global_path": str(emb),
        "ablingua_global_sha256": hashlib.sha256(emb.read_bytes()).hexdigest(),
        "rotation": "TEST=k, VAL=(k+1)%5, TRAIN=rest",
    }

    # Call actual functions from each script for ground truth
    outA_p = evA["run_incr"](recipe_A, HL_A, y, primary, ids, "FREE_ALPHA", True)
    outA_s = evA["run_incr"](recipe_A, HL_A, y, shadow, ids, "FREE_ALPHA", True)
    outB_p = evB["run_parent_plus"](recipe_B, [HL_B], [], y, primary, ids, "FREE_ALPHA")
    outB_s = evB["run_parent_plus"](recipe_B, [HL_B], [], y, shadow, ids, "FREE_ALPHA")

    print("A ADD plus Primary", outA_p["plus_mae"], "base", outA_p["base_mae"])
    print("A ADD plus Shadow", outA_s["plus_mae"], "base", outA_s["base_mae"])
    print("B PARENT Primary plus", outB_p["plus_mae"], "base", outB_p["base_mae"])
    print("B PARENT Shadow plus", outB_s["plus_mae"], "base", outB_s["base_mae"])

    # Independent reproduction (A-style DF scaler, B-style numpy + abl0_ prefix)
    indA_p = run_recipe_plus_abl(recipe_A, HL_A, y, primary, ids, alpha_on_numpy=False, label="indA")
    indA_s = run_recipe_plus_abl(recipe_A, HL_A, y, shadow, ids, alpha_on_numpy=False, label="indA")
    indB_p = run_recipe_plus_abl(
        recipe_B, HL_B, y, primary, ids, alpha_on_numpy=True, pca_prefix="abl0_", label="indB"
    )
    indB_s = run_recipe_plus_abl(
        recipe_B, HL_B, y, shadow, ids, alpha_on_numpy=True, pca_prefix="abl0_", label="indB"
    )
    print("indA Primary plus", indA_p["plus_mae"], "indB", indB_p["plus_mae"])
    print("indA Shadow plus", indA_s["plus_mae"], "indB", indB_s["plus_mae"])

    # Pred comparison A plus vs B plus
    rows = []
    for tag, oa, ob, folds in [
        ("Primary", outA_p, outB_p, primary),
        ("Shadow", outA_s, outB_s, shadow),
    ]:
        common = sorted(set(oa["y"].index) & set(ob["y"].index))
        pa, pb = oa["plus_oof"].reindex(common), ob["plus_oof"].reindex(common)
        diff = (pa - pb).abs()
        fmap = folds.set_index("id")["fold"].astype(int).to_dict()
        for i in common:
            rows.append(
                {
                    "id": i,
                    "split": tag,
                    "fold": fmap[i],
                    "y_true": float(oa["y"].loc[i]),
                    "pred_A": float(pa.loc[i]),
                    "pred_B": float(pb.loc[i]),
                    "abs_diff": float(diff.loc[i]),
                }
            )
        print(
            f"{tag} max|A-B|={diff.max():.6e} mean={diff.mean():.6e} corr={np.corrcoef(pa,pb)[0,1]:.6f}"
        )

    preds = pd.DataFrame(rows)
    preds.to_csv(RES / "PARENT_A_vs_B_predictions.csv", index=False)
    preds.rename(columns={"pred_A": "pred"}).drop(columns=["pred_B", "abs_diff"]).to_csv(
        RES / "PARENT_A_predictions.csv", index=False
    )
    preds.rename(columns={"pred_B": "pred"}).drop(columns=["pred_A", "abs_diff"]).to_csv(
        RES / "PARENT_B_predictions.csv", index=False
    )

    # Fold audit
    fold_rows = []
    for tag, folds in [("Primary", primary), ("Shadow", shadow)]:
        for k, tr, va, te in mod.rotation_splits(folds, ids):
            fold_rows.append(
                {
                    "split": tag,
                    "k": k,
                    "n_tr": len(tr),
                    "n_va": len(va),
                    "n_te": len(te),
                    "tr_hash": sha_ids(tr),
                    "va_hash": sha_ids(va),
                    "te_hash": sha_ids(te),
                }
            )
    pd.DataFrame(fold_rows).to_csv(RES / "FOLD_ROTATION_AUDIT.csv", index=False)

    # Alpha comparison from independent runs
    alpha_cmp = {
        "Primary": {
            "A_plus": [m["a_plus"] for m in indA_p["fold_meta"]],
            "B_style": [m["a_plus"] for m in indB_p["fold_meta"]],
            "Xtv_same": [
                indA_p["fold_meta"][i]["Xtv_sha"] == indB_p["fold_meta"][i]["Xtv_sha"]
                for i in range(5)
            ],
        },
        "Shadow": {
            "A_plus": [m["a_plus"] for m in indA_s["fold_meta"]],
            "B_style": [m["a_plus"] for m in indB_s["fold_meta"]],
            "Xtv_same": [
                indA_s["fold_meta"][i]["Xtv_sha"] == indB_s["fold_meta"][i]["Xtv_sha"]
                for i in range(5)
            ],
        },
    }

    # Root cause
    # B PARENT: when always=[glob], base==plus==recipe+PCA(glob). Reported base_mae labeled recipe_only wrongly.
    # If B plus == A plus → same model, historical B number should match A ADD.
    # If not, dig into actual function outputs above.

    hist_A = {"Primary": 2.810733334088188, "Shadow": 2.990836362735377}
    hist_B = {"Primary": 2.7466001170280285, "Shadow": 2.822725206703513}

    reproduced = {
        "A_script_plus": {"Primary": outA_p["plus_mae"], "Shadow": outA_s["plus_mae"]},
        "A_script_base_recipe": {"Primary": outA_p["base_mae"], "Shadow": outA_s["base_mae"]},
        "B_script_plus": {"Primary": outB_p["plus_mae"], "Shadow": outB_s["plus_mae"]},
        "B_script_base": {"Primary": outB_p["base_mae"], "Shadow": outB_s["base_mae"]},
        "hist_A_ADD": hist_A,
        "hist_B_PARENT": hist_B,
        "A_matches_hist": {
            "Primary": abs(outA_p["plus_mae"] - hist_A["Primary"]) < 1e-8,
            "Shadow": abs(outA_s["plus_mae"] - hist_A["Shadow"]) < 1e-8,
        },
        "B_matches_hist": {
            "Primary": abs(outB_p["plus_mae"] - hist_B["Primary"]) < 1e-8,
            "Shadow": abs(outB_s["plus_mae"] - hist_B["Shadow"]) < 1e-8,
        },
        "A_plus_equals_B_plus": {
            "Primary": abs(outA_p["plus_mae"] - outB_p["plus_mae"]) < 1e-8,
            "Shadow": abs(outA_s["plus_mae"] - outB_s["plus_mae"]) < 1e-8,
        },
        "B_base_equals_B_plus": {
            "Primary": abs(outB_p["base_mae"] - outB_p["plus_mae"]) < 1e-8,
            "Shadow": abs(outB_s["base_mae"] - outB_s["plus_mae"]) < 1e-8,
        },
    }

    # Hypothesis
    if reproduced["A_plus_equals_B_plus"]["Primary"] and reproduced["A_plus_equals_B_plus"]["Shadow"]:
        hyp = "SAME_MODEL_HISTORICAL_B_SHOULD_EQUAL_A_IF_CORRECTLY_LABELED"
    elif abs(outB_p["plus_mae"] - outA_p["base_mae"]) < 1e-6:
        hyp = "B_USED_RECIPE_ONLY_AS_PARENT"
    else:
        hyp = "IMPLEMENTATION_DIVERGENCE_SEE_PRED_DIFF"

    out = {
        "meta": meta,
        "components": comps,
        "reproduced": reproduced,
        "alpha_cmp": alpha_cmp,
        "hypothesis": hyp,
        "pred_summary": {
            tag: {
                "max_abs": float(preds.loc[preds.split == tag, "abs_diff"].max()),
                "mean_abs": float(preds.loc[preds.split == tag, "abs_diff"].mean()),
                "corr": float(
                    np.corrcoef(
                        preds.loc[preds.split == tag, "pred_A"],
                        preds.loc[preds.split == tag, "pred_B"],
                    )[0, 1]
                ),
            }
            for tag in ["Primary", "Shadow"]
        },
        "authoritative": {
            "label": "A_ADD_plus_semantics",
            "definition": "CURRENT_RECIPE (no PCA) + fold-local PCA32(AbLingua HL_mean_concat), FREE_ALPHA Ridge",
            "Primary": outA_p["plus_mae"],
            "Shadow": outA_s["plus_mae"],
            "recipe_only_Primary": outA_p["base_mae"],
            "recipe_only_Shadow": outA_s["base_mae"],
        },
    }
    (RES / "RECONCILIATION_RAW.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
