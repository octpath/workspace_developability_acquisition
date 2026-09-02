#!/usr/bin/env python3
"""Evaluate Advanced Batch2 families after target-blind freeze (incl. 3DI PCA32)."""
from __future__ import annotations

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
sys.path.insert(0, str(FP))
from common.ridge_eval import (  # noqa: E402
    ALPHAS,
    GENS,
    HIC_BASES,
    HIC_REF,
    OOF_DIR,
    TMAPP_BASES,
    TMAPP_REF,
    bootstrap_delta,
    classify_incremental,
    classify_standalone,
    empirical_verdict,
    generator_signal_label,
    hic_tail,
    load_base_oof,
    mae,
    median_baseline_oof,
    metrics,
    quintiles,
    residual_oof,
    run_family_evaluation,
    select_alpha,
)


def nested_ridge_pca(X, y, folds, n_components=32):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    alphas = {}
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        sc = StandardScaler()
        Xs = sc.fit_transform(X.loc[tr])
        pca = PCA(n_components=n_components, random_state=0)
        Xp = pca.fit_transform(Xs)
        # alpha select on PCA space
        Xtr = pd.DataFrame(Xp, index=tr)
        a = select_alpha(Xtr, y.loc[tr], folds, tr)
        alphas[int(f)] = a
        model = Ridge(alpha=a, random_state=0)
        model.fit(Xp, y.loc[tr])
        Xte = pca.transform(sc.transform(X.loc[te]))
        oof.loc[te] = model.predict(Xte)
    # full
    sc = StandardScaler()
    Xs = sc.fit_transform(X.loc[ids])
    pca = PCA(n_components=n_components, random_state=0)
    Xp = pca.fit_transform(Xs)
    Xtr = pd.DataFrame(Xp, index=ids)
    a_full = select_alpha(Xtr, y.loc[ids], folds, ids)
    model = Ridge(alpha=a_full, random_state=0)
    model.fit(Xp, y.loc[ids])
    return oof, model, sc, pca, a_full, alphas


def residual_oof_pca(X, y, folds, frozen_oof, base_mat, target, n_components=32):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    cand = pd.Series(index=ids, dtype=float)
    audit = {"target": target, "outer_folds": [], "pass": True, "failures": [], "pca_n": n_components}
    from common.ridge_eval import ref_crossfit_hic, ref_crossfit_tmapp

    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        if target == "TmApp":
            ref_tr, inner = ref_crossfit_tmapp(base_mat, y, folds, tr)
        else:
            ref_tr = ref_crossfit_hic(base_mat, tr)
            inner = [{"note": "equal_mean_of_nested_base_oof_on_outer_train"}]
        resid = y.loc[tr] - ref_tr.loc[tr]
        sc = StandardScaler()
        Xs = sc.fit_transform(X.loc[tr])
        pca = PCA(n_components=n_components, random_state=0)
        Xp = pca.fit_transform(Xs)
        Xtr = pd.DataFrame(Xp, index=tr)
        a = select_alpha(Xtr, resid, folds, tr)
        model = Ridge(alpha=a, random_state=0)
        model.fit(Xp, resid.loc[tr])
        cand.loc[te] = frozen_oof.loc[te].values + model.predict(pca.transform(sc.transform(X.loc[te])))
        audit["outer_folds"].append({"outer_fold": int(f), "residual_alpha": a, "reference_inner": inner})
    resid_full = y.loc[ids] - frozen_oof.loc[ids]
    sc = StandardScaler()
    Xs = sc.fit_transform(X.loc[ids])
    pca = PCA(n_components=n_components, random_state=0)
    Xp = pca.fit_transform(Xs)
    Xtr = pd.DataFrame(Xp, index=ids)
    a_full = select_alpha(Xtr, resid_full, folds, ids)
    model = Ridge(alpha=a_full, random_state=0)
    model.fit(Xp, resid_full.loc[ids])
    audit["test_residual_alpha"] = a_full
    return cand, model, sc, pca, audit


def eval_3di(family_dir: Path):
    freeze_path = family_dir / "TARGET_BLIND_ARTIFACT_HASHES.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["target_scoring_started_after_feature_freeze"] = True
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")

    folds = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    sol = pd.read_csv(ROOT / "competition/data/secret/solution.csv")
    feat_by_gen = {
        g: pd.read_parquet(family_dir / f"features_{g}.parquet")
        .query("extraction_status == 'SUCCESS'")
        .set_index("id")
        for g in GENS
    }
    tmapp_base = load_base_oof(TMAPP_BASES)
    hic_base = load_base_oof(HIC_BASES)
    tmapp_oof = pd.read_csv(OOF_DIR / f"{TMAPP_REF}.csv").set_index("id")["y_pred"]
    hic_oof = pd.read_csv(OOF_DIR / f"{HIC_REF}.csv").set_index("id")["y_pred"]
    tmapp_test = pd.read_csv(
        ROOT / "virtual_participant/round1_finalization/predictions/TmApp_PRIMARY_predictions.csv"
    ).set_index("id")["prediction"]
    hic_test = pd.read_csv(
        ROOT / "virtual_participant/round1_finalization/predictions/HIC_PRIMARY_predictions.csv"
    ).set_index("id")["prediction"]

    results = {"family_id": "3DI-FROZEN", "version": "v1"}
    audits = {}
    for target, base_mat, frozen_oof, frozen_test in [
        ("TmApp", tmapp_base, tmapp_oof, tmapp_test),
        ("HIC", hic_base, hic_oof, hic_test),
    ]:
        y = pd.read_csv(OOF_DIR / f"{(TMAPP_REF if target=='TmApp' else HIC_REF)}.csv").set_index("id")["y_true"].astype(float)
        y_test = sol.set_index("id")[target].astype(float)
        pub = list(sol.loc[sol.is_public == 1, "id"])
        pri = list(sol.loc[sol.is_private == 1, "id"])
        out = {}
        inc_map = {}
        for gen in GENS:
            print("3DI eval", target, gen, flush=True)
            avail = feat_by_gen[gen].index
            y_g = y.loc[y.index.intersection(avail)]
            y_test_g = y_test.loc[y_test.index.intersection(avail)]
            pub_g = [i for i in pub if i in avail]
            pri_g = [i for i in pri if i in avail]
            cols = [c for c in feat_by_gen[gen].columns if c.startswith("hl_")]
            X = feat_by_gen[gen].loc[y_g.index, cols].astype(float)
            Xte = feat_by_gen[gen].loc[y_test_g.index, cols].astype(float)
            oof, model, sc, pca, a_full, alphas = nested_ridge_pca(X, y_g, folds, 32)
            base_oof, base_med = median_baseline_oof(y_g, folds)
            test_pred = pd.Series(
                model.predict(pca.transform(sc.transform(Xte))), index=y_test_g.index
            )
            m = {"alpha_full": a_full, "alphas_outer": alphas, "pca_n": 32, "n_dev_used": int(len(y_g))}
            for split, yy, pp, bb in [
                ("CV", y_g, oof, base_oof),
                ("Public", y_test_g.loc[pub_g], test_pred.loc[pub_g], pd.Series(base_med, index=pub_g)),
                ("Private", y_test_g.loc[pri_g], test_pred.loc[pri_g], pd.Series(base_med, index=pri_g)),
                ("AllTest", y_test_g, test_pred, pd.Series(base_med, index=y_test_g.index)),
            ]:
                for k, v in metrics(yy, pp).items():
                    m[f"{k}_{split}"] = v
                m[f"baseline_MAE_{split}"] = mae(yy, bb)
            m["standalone_signal"] = classify_standalone(m)
            frozen_oof_g = frozen_oof.loc[y_g.index]
            frozen_test_g = frozen_test.loc[y_test_g.index]
            cand_oof, res_model, res_sc, res_pca, audit = residual_oof_pca(
                X, y_g, folds, frozen_oof_g, base_mat, target, 32
            )
            audits[f"{target}_{gen}"] = audit
            cand_test = frozen_test_g.astype(float) + pd.Series(
                res_model.predict(res_pca.transform(res_sc.transform(Xte))), index=y_test_g.index
            )
            boots = {}
            for split, yy, pc, pr_ in [
                ("CV", y_g, cand_oof, frozen_oof_g),
                ("Public", y_test_g.loc[pub_g], cand_test.loc[pub_g], frozen_test_g.loc[pub_g]),
                ("Private", y_test_g.loc[pri_g], cand_test.loc[pri_g], frozen_test_g.loc[pri_g]),
                ("AllTest", y_test_g, cand_test, frozen_test_g),
            ]:
                m[f"MAE_candidate_{split}"] = mae(yy, pc)
                m[f"MAE_reference_{split}"] = mae(yy, pr_)
                m[f"delta_MAE_{split}"] = m[f"MAE_candidate_{split}"] - m[f"MAE_reference_{split}"]
                if split in ("CV", "Public", "Private"):
                    boots[split] = bootstrap_delta(yy, pc, pr_)
                    m[f"bootstrap_ci95_low_{split}"] = boots[split]["ci95_low"]
                    m[f"bootstrap_ci95_high_{split}"] = boots[split]["ci95_high"]
                    m[f"bootstrap_p_improve_{split}"] = boots[split]["p_improve"]
            m["incremental_signal"] = classify_incremental(
                m["delta_MAE_CV"], m["delta_MAE_Public"], m["delta_MAE_Private"], boots
            )
            inc_map[gen] = m["incremental_signal"]
            m["empirical_verdict"] = empirical_verdict(
                m["standalone_signal"], m["incremental_signal"], target == "TmApp"
            )
            if target == "TmApp":
                m["quintile_MAE_candidate_CV"] = quintiles(y_g, cand_oof)
            else:
                m["hic_tail_candidate_CV"] = hic_tail(y_g, cand_oof)
                m["hic_tail_candidate_AllTest"] = hic_tail(y_test_g, cand_test)
            out[gen] = m
        out["_generator_signal"] = generator_signal_label(inc_map)
        results[target] = out
    (family_dir / "EVALUATION_RESULTS.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (family_dir / "RESIDUAL_RIDGE_IMPLEMENTATION_AUDIT.json").write_text(
        json.dumps({"audits": audits, "overall_pass": all(a.get("pass", False) for a in audits.values())}, indent=2)
        + "\n"
    )
    return results


def main():
    for fam, key in [
        ("HYDRO-FIELD", "HYDRO-FIELD"),
        ("ELEC-HYDRO-COPATCH", "ELEC-HYDRO-COPATCH"),
        ("INTERFACE-ENERGY", "INTERFACE-ENERGY"),
    ]:
        d = FP / fam
        spec = json.loads((d / "FEATURE_SPEC.json").read_text())
        print("eval", fam, flush=True)
        run_family_evaluation(d, fam, "v1", spec["canonical_features"])
    print("eval 3DI", flush=True)
    eval_3di(FP / "3DI-FROZEN")


if __name__ == "__main__":
    main()
