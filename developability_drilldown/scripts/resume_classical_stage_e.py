#!/usr/bin/env python3
"""Resume classical refinement from Stage E (uses existing blocks + candidates)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from classical_features.cv_eval import (  # noqa: E402
    evaluate_primary_shadow,
    fit_full_dev_predict,
    load_folds,
)
from _lib import (  # noqa: E402
    add_split_column,
    feature_content_sha256,
    file_sha256,
    load_block,
    load_dev_test_folds,
)
from experiment_codes import issue_code, load_codes  # noqa: E402

BUNDLE = REPO / "top_models_feature_bundle"
CACHE = ROOT / "experiments" / "classical_cache"
BLOCK_DIR = CACHE / "blocks"
CV_CACHE = CACHE / "cv_cache"
PROGRESS = CACHE / "stage_e_progress.jsonl"
Group = tuple[list[str], bool]


def concat_parts(parts: list[tuple[pd.DataFrame, bool]], ids: list[str]):
    base = pd.DataFrame({"id": ids})
    seen: set[str] = set()
    mats, groups = [], []
    for fr, do_pca in parts:
        fr = fr.copy()
        fr["id"] = fr["id"].astype(str)
        sub = fr.set_index("id").loc[ids]
        rename = {}
        for c in sub.columns:
            name = c
            n = 0
            while name in seen:
                n += 1
                name = f"{c}__d{n}"
            seen.add(name)
            rename[c] = name
        cols = list(rename.values())
        mats.append(sub.rename(columns=rename)[cols].reset_index(drop=True))
        groups.append((cols, bool(do_pca)))
    return pd.concat([base] + mats, axis=1), groups


def is_plm_block(bid: str) -> bool:
    if bid == "FB_ARO_RASA_SUM":
        return False
    return bid.startswith(("FB_AL", "FB_ESM2", "FB_AL2"))


def recipe_parts(recipe_id: str, ids: list[str]):
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    blocks = str(recipes[recipes["recipe_id"] == recipe_id].iloc[0]["feature_blocks"]).split("|")
    return [(load_block(b, ids), b.startswith("AbLingua")) for b in blocks]


BASE_RECIPES = {
    "FS_TM_ABLINGUA_CDR3": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "FS_TM_ABLINGUA_GLOBAL": "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "FS_TM_BIOEMU_MPNN": "TM_BASE_BIOEMU_MPNN__RIDGE",
    "FS_HIC_CONTINUOUS_SURFACE": "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "FS_HIC_HYDRO_TITRATION": "HIC_HYDRO_TITRATION__LASSO",
    "FS_HIC_ESM2_SEQ_AROMATIC": "HIC_ESM2_SEQ_AROMATIC__LASSO",
}


def matrix_key(name: str, target: str, estimator: str, groups, n_features: int) -> str:
    gsig = "|".join(f"{len(c)}:{int(p)}" for c, p in groups)
    return hashlib.sha256(f"{name}|{target}|{estimator}|{n_features}|{gsig}".encode()).hexdigest()[:24]


def build_from_block_ids(block_ids: list[str], all_ids: list[str], blocks: dict, base_parts: dict, scale: float = 1.0):
    parts = []
    for bid in block_ids:
        if bid in BASE_RECIPES:
            parts.extend(base_parts[bid])
        elif bid in blocks:
            df = blocks[bid].copy()
            if scale != 1.0 and is_plm_block(bid):
                cols = [c for c in df.columns if c != "id"]
                df[cols] = df[cols].astype(float) * scale
            parts.append((df, is_plm_block(bid)))
        else:
            raise KeyError(bid)
    return concat_parts(parts, all_ids)


def main() -> int:
    t0 = time.time()
    dev, test, _ = load_dev_test_folds()
    primary, shadow = load_folds(ROOT / "data" / "folds.csv")
    all_ids = dev["id"].astype(str).tolist() + test["id"].astype(str).tolist()
    ymaps = {
        "TmApp": {str(r.id): float(r.TmApp) for _, r in dev.iterrows()},
        "HIC": {str(r.id): float(r.HIC) for _, r in dev.iterrows()},
    }

    blocks = {}
    for p in BLOCK_DIR.glob("*.parquet"):
        df = pd.read_parquet(p)
        df["id"] = df["id"].astype(str)
        blocks[p.stem] = df
    print(f"loaded {len(blocks)} blocks", flush=True)

    base_parts = {fsid: recipe_parts(rid, all_ids) for fsid, rid in BASE_RECIPES.items()}

    cand_df = pd.read_csv(ROOT / "results/CLASSICAL_REFINEMENT_CANDIDATES.csv")
    # unique feature sets
    final_pool: dict[str, list] = {"TmApp": [], "HIC": []}
    seen = set()
    for _, c in cand_df.iterrows():
        key = (c["target"], c["feature_set_name"])
        if key in seen:
            continue
        seen.add(key)
        name = c["feature_set_name"]
        blist = str(c["blocks"]).split("|")
        scale = 1.0
        if "__scale" in name:
            # e.g. ...__scale2.0
            scale = float(name.rsplit("__scale", 1)[1])
            # blist still original blocks
        X, g = build_from_block_ids(blist, all_ids, blocks, base_parts, scale=scale)
        final_pool[c["target"]].append((name, X, g, blist))
        print(f"pool {c['target']} {name} dim={X.shape[1]-1}", flush=True)

    eval_memo = {}

    def eval_set(name, X, groups, target, estimator, keep_oof=False):
        key = matrix_key(name, target, estimator, groups, int(X.shape[1] - 1))
        cache_path = CV_CACHE / f"{key}.json"
        res = None
        if key in eval_memo and (not keep_oof or "oof_primary" in eval_memo[key]):
            res = eval_memo[key]
        elif cache_path.exists():
            res = json.loads(cache_path.read_text())
            if keep_oof:
                op, os_ = CV_CACHE / f"{key}_op.csv", CV_CACHE / f"{key}_os.csv"
                if op.exists() and os_.exists():
                    res["oof_primary"] = pd.read_csv(op).set_index("id")["pred"]
                    res["oof_shadow"] = pd.read_csv(os_).set_index("id")["pred"]
                else:
                    res = None
            if res is not None:
                eval_memo[key] = res
        if res is None or (keep_oof and "oof_primary" not in res):
            Xdev = X[X["id"].isin(dev["id"].astype(str))].copy()
            full = evaluate_primary_shadow(
                Xdev, ymaps[target], primary, shadow, estimator=estimator, groups=groups
            )
            slim = {k: float(full[k]) for k in ("cv_primary_mae", "cv_shadow_mae", "cv_mean_mae", "cv_worst_mae")}
            slim["params_primary"] = full["params_primary"]
            slim["params_shadow"] = full["params_shadow"]
            cache_path.write_text(json.dumps(slim, default=str))
            full["oof_primary"].rename("pred").to_csv(CV_CACHE / f"{key}_op.csv", index_label="id")
            full["oof_shadow"].rename("pred").to_csv(CV_CACHE / f"{key}_os.csv", index_label="id")
            res = {**slim, "oof_primary": full["oof_primary"], "oof_shadow": full["oof_shadow"]}
            eval_memo[key] = res
        print(
            f"  {estimator} {name}: P={res['cv_primary_mae']:.4f} "
            f"S={res['cv_shadow_mae']:.4f} W={res['cv_worst_mae']:.4f}",
            flush=True,
        )
        out = {**res, "groups": groups, "X": X}
        return out

    # load prior progress
    done = set()
    results_store = []
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["target"], row["feature_set_name"], row["estimator"]))
            print(f"resume skip {row['experiment_tag']}", flush=True)

    print("\n=== STAGE E (resume) ===", flush=True)
    with PROGRESS.open("a", encoding="utf-8") as prog:
        for _, c in cand_df.iterrows():
            tgt, name, est = c["target"], c["feature_set_name"], c["estimator"]
            tag = f"{tgt}|{name}|{est}"
            if (tgt, name, est) in done:
                # reload from cache into results_store
                X = g = blist = None
                for n, XX, gg, bb in final_pool[tgt]:
                    if n == name:
                        X, g, blist = XX, gg, bb
                        break
                res = eval_set(f"E__{name}__{est}", X, g, tgt, est, keep_oof=True)
                results_store.append({
                    "candidate_order": int(c["candidate_order"]), "target": tgt,
                    "feature_set_name": name, "blocks": blist, "estimator": est,
                    "X": X, "groups": g, "res": res,
                })
                continue
            X = g = blist = None
            for n, XX, gg, bb in final_pool[tgt]:
                if n == name:
                    X, g, blist = XX, gg, bb
                    break
            try:
                res = eval_set(f"E__{name}__{est}", X, g, tgt, est, keep_oof=True)
            except Exception as e:
                print(f"FAILED {name} {est}: {e}", flush=True)
                prog.write(json.dumps({"target": tgt, "feature_set_name": name, "estimator": est,
                                       "experiment_tag": tag, "status": f"FAILED:{e}"}) + "\n")
                prog.flush()
                continue
            results_store.append({
                "candidate_order": int(c["candidate_order"]), "target": tgt,
                "feature_set_name": name, "blocks": blist, "estimator": est,
                "X": X, "groups": g, "res": res,
            })
            prog.write(json.dumps({
                "target": tgt, "feature_set_name": name, "estimator": est,
                "experiment_tag": tag, "status": "OK",
                "cv_worst_mae": res["cv_worst_mae"],
                "cv_primary_mae": res["cv_primary_mae"],
                "cv_shadow_mae": res["cv_shadow_mae"],
            }) + "\n")
            prog.flush()

    ranked = []
    for tgt in ("TmApp", "HIC"):
        items = sorted(
            [r for r in results_store if r["target"] == tgt],
            key=lambda r: (r["res"]["cv_worst_mae"], r["res"]["cv_mean_mae"]),
        )
        for i, it in enumerate(items):
            ranked.append({
                "candidate_order": it["candidate_order"], "target": tgt,
                "feature_set_name": it["feature_set_name"], "estimator": it["estimator"],
                "rank_cv_worst": i + 1,
                "cv_primary_mae": it["res"]["cv_primary_mae"],
                "cv_shadow_mae": it["res"]["cv_shadow_mae"],
                "cv_worst_mae": it["res"]["cv_worst_mae"],
                "blocks": "|".join(it["blocks"]),
            })
    pd.DataFrame(ranked).to_csv(ROOT / "results/CLASSICAL_REFINEMENT_CV_RANKING.csv", index=False)

    freeze = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_metric": "cv_worst=max(primary,shadow); tie_break=cv_mean",
        "public_private_used_in_selection": False,
        "preprocessing_order": "per-block impute -> PCA32(if AbLingua/new PLM block) -> StandardScaler -> estimator",
        "candidates_path": "results/CLASSICAL_REFINEMENT_CANDIDATES.csv",
        "ranking_path": "results/CLASSICAL_REFINEMENT_CV_RANKING.csv",
        "n_results": len(results_store),
        "top_by_target": {
            tgt: [
                {
                    "feature_set_name": r["feature_set_name"], "estimator": r["estimator"],
                    "cv_worst_mae": float(r["res"]["cv_worst_mae"]),
                    "cv_primary_mae": float(r["res"]["cv_primary_mae"]),
                    "cv_shadow_mae": float(r["res"]["cv_shadow_mae"]),
                }
                for r in sorted(
                    [x for x in results_store if x["target"] == tgt],
                    key=lambda z: z["res"]["cv_worst_mae"],
                )[:5]
            ]
            for tgt in ("TmApp", "HIC")
        },
        "git_head_at_start": "37e47225",
        "resumed_from": "Stage E after crash",
    }
    (ROOT / "results/CLASSICAL_REFINEMENT_FREEZE.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )
    print("\nFROZE candidates", flush=True)

    print("\n=== POST-FREEZE Test ===", flush=True)
    sol = pd.read_csv(BUNDLE / "solution.csv")
    sol["id"] = sol["id"].astype(str)
    pub_ids = set(sol.loc[sol["is_public"].astype(bool), "id"])
    priv_ids = set(sol.loc[sol["is_private"].astype(bool), "id"])
    test_ids = test["id"].astype(str).tolist()

    exp_path = ROOT / "results/experiments.csv"
    exp_df = pd.read_csv(exp_path)
    snap77 = pd.read_csv(ROOT / "results/_preservation_snapshot_77_classical.csv")
    assert len(exp_df) == 77
    fs_df = pd.read_csv(ROOT / "results/FEATURE_SETS.csv")
    new_rows = []
    rasa_cdr_rows = []

    def make_eid(tgt, est, name: str) -> str:
        pref = {"ridge": "LIN", "lasso": "LIN", "enet": "ENET", "svr": "SVR", "xgb": "XGB"}[est]
        tag = name.replace("FS_", "").replace("FB_", "").replace("+", "_").replace("__", "_")
        tag = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in tag)
        tag = "_".join(t for t in tag.split("_") if t)[:70]
        mid = "TM" if tgt == "TmApp" else "HIC"
        est2 = {"ridge": "RIDGE", "lasso": "LASSO", "enet": "ENET", "svr": "SVR", "xgb": "XGB"}[est]
        return f"{pref}_{mid}_{tag}_{est2}"[:120]

    for it in results_store:
        tgt, est, name = it["target"], it["estimator"], it["feature_set_name"]
        X, g, res = it["X"], it["groups"], it["res"]
        eid = make_eid(tgt, est, name)
        existing = set(load_codes()["experiment_id"]) | {r["experiment_id"] for r in new_rows}
        base_eid, n = eid, 2
        while eid in existing:
            eid = f"{base_eid}_{n}"
            n += 1
        code = issue_code(
            eid, tgt, source_model_id=f"CLASSICAL::{name}::{est}",
            phase="CLASSICAL_REFINEMENT", notes="classical feature refinement; no ensemble",
        )
        feat = add_split_column(X.copy(), dev["id"].astype(str), test["id"].astype(str))
        fpath = ROOT / "experiments/features" / f"{code}.parquet"
        feat.to_parquet(fpath, index=False)
        fsha, csha = file_sha256(fpath), feature_content_sha256(feat)
        pred_dir = ROOT / "experiments/predictions" / code
        pred_dir.mkdir(parents=True, exist_ok=True)
        op = res["oof_primary"].reset_index(); op.columns = ["id", tgt]
        os_ = res["oof_shadow"].reset_index(); os_.columns = ["id", tgt]
        op.to_csv(pred_dir / "oof_primary.csv", index=False)
        os_.to_csv(pred_dir / "oof_shadow.csv", index=False)
        te_pred = fit_full_dev_predict(X, ymaps[tgt], primary, test_ids, estimator=est, groups=g)
        te_df = pd.DataFrame({"id": test_ids, tgt: te_pred})
        te_df.to_csv(pred_dir / "test.csv", index=False)
        te_s = te_df.set_index("id")[tgt]
        pub = [i for i in test_ids if i in pub_ids]
        priv = [i for i in test_ids if i in priv_ids]
        pub_mae = float(np.mean(np.abs(sol.set_index("id").loc[pub, tgt] - te_s.loc[pub])))
        priv_mae = float(np.mean(np.abs(sol.set_index("id").loc[priv, tgt] - te_s.loc[priv])))
        overall = float(np.mean(np.abs(sol.set_index("id").loc[test_ids, tgt].to_numpy(float) - te_pred)))

        cfg = {
            "experiment_code": code, "experiment_id": eid, "target": tgt,
            "family": "XGBOOST" if est == "xgb" else "LINEAR", "model_type": est.upper(),
            "feature_set_name": name, "blocks": it["blocks"],
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "ensemble": False,
            "pca_groups": [{"n": len(c), "pca": p} for c, p in g],
        }
        (ROOT / "experiments/configs" / f"{code}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        fsid = f"FS_{code}"
        fs_df = pd.concat([fs_df, pd.DataFrame([{
            "feature_set_id": fsid, "target": tgt, "n_features": feat.shape[1] - 2,
            "feature_content_sha256": csha, "feature_recipe_hash": "",
            "source_recipe_ids": name, "feature_blocks": "|".join(it["blocks"]),
            "notes": "classical refinement",
        }])], ignore_index=True)

        row = {c: "" for c in exp_df.columns}
        row.update({
            "experiment_code": code, "legacy_experiment_code": "", "experiment_id": eid,
            "target": tgt, "family": cfg["family"], "model_type": est.upper(),
            "source_model_id": f"CLASSICAL::{name}::{est}", "feature_set_id": fsid,
            "source_recipe_id": name,
            "cv_primary_mae": res["cv_primary_mae"], "cv_shadow_mae": res["cv_shadow_mae"],
            "cv_mean_mae": res["cv_mean_mae"], "cv_worst_mae": res["cv_worst_mae"],
            "public_mae": pub_mae, "private_mae": priv_mae, "test_overall_mae": overall,
            "public_private_delta": pub_mae - priv_mae, "public_private_gap": abs(pub_mae - priv_mae),
            "cv_protocol": "canonical_simple_tvt_primary_shadow",
            "selection_policy_at_creation": "CV_SELECTED_POSTCOMP_EVALUATED",
            "current_evaluation_mode": "POSTCOMP_EXPLORATORY",
            "artifact_status": "FULL", "source_reproducible": "YES",
            "drilldown_reproducible": "YES", "reproduction_status": "PASS",
            "config_path": f"experiments/configs/{code}.yaml",
            "feature_path": f"experiments/features/{code}.parquet",
            "oof_primary_path": f"experiments/predictions/{code}/oof_primary.csv",
            "oof_shadow_path": f"experiments/predictions/{code}/oof_shadow.csv",
            "test_prediction_path": f"experiments/predictions/{code}/test.csv",
            "n_features": feat.shape[1] - 2, "feature_space": "RAW_PREPROCESS",
            "feature_sha256": fsha, "feature_content_sha256": csha,
            "score_source": "classical_refinement_cv+solution_postfreeze",
            "prediction_source": "classical_refinement", "feature_source": name,
            "license_status": "REVIEW",
            "license_reference": "PLM residue / structure-derived classical features",
            "ensemble_type": "", "member_experiment_codes": "",
            "notes": "CLASSICAL_FEATURE_REFINEMENT; no ensemble; CV-selected then postcomp evaluated",
        })
        new_rows.append(row)
        if any(x in name for x in ("RASA", "CDRW", "CDR3W", "CDR_ALL", "CDR3", "ARO_RASA", "FR_CDR")):
            rasa_cdr_rows.append({
                "target": tgt, "source_embedding": name, "feature_set_id": fsid,
                "estimator": est, "experiment_code": code,
                "cv_primary": res["cv_primary_mae"], "cv_shadow": res["cv_shadow_mae"],
                "cv_worst": res["cv_worst_mae"], "public": pub_mae, "private": priv_mae,
                "overall": overall,
            })
        print(f"REGISTERED {code} {eid} W={res['cv_worst_mae']:.4f} Pub={pub_mae:.4f} Priv={priv_mae:.4f}", flush=True)

    new_df = pd.DataFrame(new_rows)
    for c in exp_df.columns:
        if c not in new_df.columns:
            new_df[c] = ""
    out_exp = pd.concat([exp_df, new_df[exp_df.columns]], ignore_index=True)
    out_exp.to_csv(exp_path, index=False)
    fs_df.to_csv(ROOT / "results/FEATURE_SETS.csv", index=False)
    pd.DataFrame(rasa_cdr_rows).to_csv(ROOT / "results/RASA_CDR_WEIGHTING_RESULTS.csv", index=False)

    cur77 = out_exp[out_exp["experiment_code"].isin(snap77["experiment_code"])]
    m = snap77.merge(cur77, on="experiment_code", suffixes=("_old", "_new"))
    assert len(m) == 77
    for c in ("cv_primary_mae", "private_mae", "public_mae", "cv_shadow_mae", "cv_worst_mae"):
        d = (pd.to_numeric(m[f"{c}_old"], errors="coerce") - pd.to_numeric(m[f"{c}_new"], errors="coerce")).abs().max()
        assert float(d) == 0.0 or pd.isna(d), d

    print(f"\nDone in {(time.time()-t0)/60:.1f} min; added {len(new_rows)}; total={len(out_exp)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
