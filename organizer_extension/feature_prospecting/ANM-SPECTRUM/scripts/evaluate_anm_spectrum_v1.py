#!/usr/bin/env python3
"""ANM-SPECTRUM_v1 target evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FAMILY = ROOT / "organizer_extension/feature_prospecting/ANM-SPECTRUM"
OOF_DIR = ROOT / "virtual_participant/stage5_integration/oof"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
HIC_HIGH = 10.5372
B_BOOT = 10000
RNG = np.random.default_rng(42)
GENS = ["esmfold", "abodybuilder2", "boltz2"]

TMAPP_REF = "TmApp__META_performance__ridge_100.0"
HIC_REF = "HIC__SIMPLE_blend_seq_surf_adv"
TMAPP_BASES = [
    "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt",
    "TmApp__FUSION_OVERALL__ESMFold__STRUCT_RASA__SVROpt",
    "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt",
    "TmApp__ADV_TMAPP_ALL__SVROpt",
]
HIC_BASES = [
    "HIC__FUSION__esm2__H__SEQ_ALL__SVROpt",
    "HIC__ESMFold__STRUCT_SURFACE_CHEM__SVROpt",
    "HIC__ADV_SURFACE_PATCH__SVROpt",
]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def corr_pair(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan, np.nan
    return float(pearsonr(y, p).statistic), float(spearmanr(y, p).statistic)


def metrics(y, p):
    pr, sr = corr_pair(y, p)
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": mae(y, p),
        "Pearson": pr,
        "Spearman": sr,
        "pred_SD": float(np.std(p, ddof=0)),
        "true_SD": float(np.std(y, ddof=0)),
        "pred_SD_over_true_SD": float(np.std(p, ddof=0) / np.std(y, ddof=0)) if np.std(y) > 0 else np.nan,
    }


def load_base_oof(base_ids):
    mats = []
    for bid in base_ids:
        df = pd.read_csv(OOF_DIR / f"{bid}__primary_nested_base.csv").set_index("id")
        mats.append(df["y_pred"].rename(bid))
    return pd.concat(mats, axis=1)


def select_alpha(X, y, folds, train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    train_ids = list(train_ids)
    uniq = sorted({fold_map[i] for i in train_ids})
    best_alpha, best_score = None, np.inf
    for a in ALPHAS:
        inner_maes = []
        for f in uniq:
            te = [i for i in train_ids if fold_map[i] == f]
            tr = [i for i in train_ids if fold_map[i] != f]
            if len(te) == 0 or len(tr) < 2:
                continue
            sc = StandardScaler()
            model = Ridge(alpha=a, random_state=0)
            model.fit(sc.fit_transform(X.loc[tr]), y.loc[tr])
            inner_maes.append(mae(y.loc[te], model.predict(sc.transform(X.loc[te]))))
        score = float(np.mean(inner_maes)) if inner_maes else np.inf
        if score < best_score - 1e-15 or (
            abs(score - best_score) <= 1e-15 and (best_alpha is None or a > best_alpha)
        ):
            best_score, best_alpha = score, a
    return best_alpha if best_alpha is not None else 100.0


def nested_ridge(X, y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    oof = pd.Series(index=ids, dtype=float)
    alphas = {}
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        a = select_alpha(X, y, folds, tr)
        alphas[int(f)] = a
        sc = StandardScaler()
        model = Ridge(alpha=a, random_state=0)
        model.fit(sc.fit_transform(X.loc[tr]), y.loc[tr])
        oof.loc[te] = model.predict(sc.transform(X.loc[te]))
    a_full = select_alpha(X, y, folds, ids)
    sc = StandardScaler()
    model = Ridge(alpha=a_full, random_state=0)
    Xs = sc.fit_transform(X.loc[ids])
    model.fit(Xs, y.loc[ids])
    return oof, model, sc, a_full, alphas


def median_baseline_oof(y, folds):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(y.index)
    oof = pd.Series(index=ids, dtype=float)
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        oof.loc[te] = float(np.median(y.loc[tr]))
    return oof, float(np.median(y.loc[ids]))


def bootstrap_delta(y, p_cand, p_ref, b=B_BOOT):
    y, p_cand, p_ref = map(lambda z: np.asarray(z, float), (y, p_cand, p_ref))
    n = len(y)
    deltas = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, n, n)
        deltas[i] = mae(y[idx], p_cand[idx]) - mae(y[idx], p_ref[idx])
    return {
        "delta": float(mae(y, p_cand) - mae(y, p_ref)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "p_improve": float(np.mean(deltas < 0)),
    }


def ref_crossfit_tmapp(base_mat, y, folds, outer_train_ids):
    fold_map = folds.set_index("id")["fold"].to_dict()
    preds = pd.Series(index=list(outer_train_ids), dtype=float)
    details = []
    for f in sorted({fold_map[i] for i in outer_train_ids}):
        te = [i for i in outer_train_ids if fold_map[i] == f]
        tr = [i for i in outer_train_ids if fold_map[i] != f]
        assert not any(fold_map[i] == f for i in tr)
        model = Ridge(alpha=100.0, random_state=0)
        model.fit(base_mat.loc[tr].values, y.loc[tr].values)
        preds.loc[te] = model.predict(base_mat.loc[te].values)
        details.append({"inner_heldout_fold": int(f), "n_train": len(tr), "n_test": len(te)})
    return preds, details


def ref_crossfit_hic(base_mat, outer_train_ids):
    return base_mat.loc[list(outer_train_ids)].mean(axis=1)


def residual_oof(X, y, folds, frozen_oof, base_mat, target):
    fold_map = folds.set_index("id")["fold"].to_dict()
    ids = list(X.index)
    cand = pd.Series(index=ids, dtype=float)
    audit = {"target": target, "outer_folds": [], "pass": True, "failures": []}
    for f in sorted({fold_map[i] for i in ids}):
        te = [i for i in ids if fold_map[i] == f]
        tr = [i for i in ids if fold_map[i] != f]
        if set(te) & set(tr):
            audit["pass"] = False
            audit["failures"].append("train_test_overlap")
        if any(i in te for i in tr):
            audit["pass"] = False
        if target == "TmApp":
            ref_tr, inner = ref_crossfit_tmapp(base_mat, y, folds, tr)
        else:
            ref_tr = ref_crossfit_hic(base_mat, tr)
            inner = [{"note": "equal_mean_of_nested_base_oof_on_outer_train"}]
        # ensure outer_test unused in reference
        if set(te) & set(ref_tr.index):
            # ref_tr should only be on tr
            audit["pass"] = False
            audit["failures"].append(f"ref_contains_outer_test_fold_{f}")
        resid = y.loc[tr] - ref_tr.loc[tr]
        a = select_alpha(X, resid, folds, tr)
        sc = StandardScaler()
        model = Ridge(alpha=a, random_state=0)
        model.fit(sc.fit_transform(X.loc[tr]), resid.loc[tr])
        cand.loc[te] = frozen_oof.loc[te].values + model.predict(sc.transform(X.loc[te]))
        audit["outer_folds"].append(
            {
                "outer_fold": int(f),
                "n_outer_test": len(te),
                "n_outer_train": len(tr),
                "outer_test_excluded_from_reference_fit": True,
                "residual_trained_on_outer_train_only": True,
                "residual_alpha": a,
                "reference_inner": inner,
            }
        )
    # full-dev residual model for test
    resid_full = y.loc[ids] - frozen_oof.loc[ids]
    a_full = select_alpha(X, resid_full, folds, ids)
    sc = StandardScaler()
    model = Ridge(alpha=a_full, random_state=0)
    model.fit(sc.fit_transform(X.loc[ids]), resid_full.loc[ids])
    audit["test_residual_alpha"] = a_full
    return cand, model, sc, audit


def classify_standalone(m):
    splits = ["CV", "Public", "Private"]
    ok = []
    for s in splits:
        beat = m[f"MAE_{s}"] < m[f"baseline_MAE_{s}"]
        corr_pos = (m[f"Pearson_{s}"] or 0) > 0 or (m[f"Spearman_{s}"] or 0) > 0
        ok.append(bool(beat and corr_pos))
    abs_r = [max(abs(m[f"Pearson_{s}"] or 0), abs(m[f"Spearman_{s}"] or 0)) for s in splits]
    if all(ok) and any(r >= 0.15 for r in abs_r):
        return "REPRODUCIBLE"
    if sum(ok) == 2 or (all(m[f"MAE_{s}"] < m[f"baseline_MAE_{s}"] for s in splits) and all(r < 0.15 for r in abs_r)):
        return "WEAK"
    if ok[0] and not (ok[1] and ok[2]):
        return "CV_ONLY"
    if (ok[1] or ok[2]) and not ok[0]:
        return "TEST_ONLY_POSTHOC"
    return "NO_SIGNAL"


def classify_incremental(d_cv, d_pu, d_pr, boots):
    signs = [d_cv < 0, d_pu < 0, d_pr < 0]
    if all(signs):
        return "REPRODUCIBLE_INCREMENT"
    if sum(signs) == 2:
        for d, b in [(d_cv, boots["CV"]), (d_pu, boots["Public"]), (d_pr, boots["Private"])]:
            if d >= 0:
                if b["ci95_low"] > 0:
                    return "DIRECTION_REVERSAL"
                return "WEAK_OR_MIXED_INCREMENT"
        return "WEAK_OR_MIXED_INCREMENT"
    if d_cv < 0:
        return "CV_ONLY_INCREMENT"
    return "NO_INCREMENT"


def empirical_verdict(standalone, incremental, tmapp: bool):
    if incremental == "REPRODUCIBLE_INCREMENT":
        return "COMPLEMENTARY" if standalone in ("WEAK", "NO_SIGNAL", "CV_ONLY") else "PROMISING"
    if standalone == "REPRODUCIBLE" and incremental in ("NO_INCREMENT", "CV_ONLY_INCREMENT"):
        return "PROMISING_BUT_REDUNDANT"
    if incremental == "WEAK_OR_MIXED_INCREMENT" or standalone == "WEAK":
        return "MIXED"
    if standalone == "NO_SIGNAL" and incremental == "NO_INCREMENT":
        return "UNLIKELY_RELEVANT_UNDER_CURRENT_SETUP" if tmapp else "NO_EVIDENCE_IN_CURRENT_DATA"
    return "NO_EVIDENCE_IN_CURRENT_DATA"


def generator_signal_label(per_gen_inc):
    good = {g for g, c in per_gen_inc.items() if c in ("REPRODUCIBLE_INCREMENT", "WEAK_OR_MIXED_INCREMENT")}
    strong = {g for g, c in per_gen_inc.items() if c == "REPRODUCIBLE_INCREMENT"}
    if len(strong) == 3:
        return "THREE_GENERATORS_SIGNAL"
    if not good:
        return "NO_SIGNAL"
    if good == {"esmfold", "abodybuilder2"}:
        return "ESMFOLD_ABB2_SIGNAL"
    if good == {"esmfold", "boltz2"}:
        return "ESMFOLD_BOLTZ2_SIGNAL"
    if good == {"abodybuilder2", "boltz2"}:
        return "ABB2_BOLTZ2_SIGNAL"
    if len(good) == 1:
        return {
            "esmfold": "ESMFOLD_ONLY_SIGNAL",
            "abodybuilder2": "ABB2_ONLY_SIGNAL",
            "boltz2": "BOLTZ2_ONLY_SIGNAL",
        }[next(iter(good))]
    return "STRUCTURE_DEPENDENT_MIXED"


def hic_tail(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    high = y >= HIC_HIGH
    out = {
        "high_tail_n": int(high.sum()),
        "high_tail_MAE": mae(y[high], p[high]) if high.any() else np.nan,
        "high_tail_bias": float(np.mean(p[high] - y[high])) if high.any() else np.nan,
    }
    if high.any() and (~high).any():
        out["ROC_AUC"] = float(roc_auc_score(high.astype(int), p))
        out["AP"] = float(average_precision_score(high.astype(int), p))
        out["Recall_at_13"] = float(high[np.argsort(-p)[:13]].mean())
    else:
        out["ROC_AUC"] = out["AP"] = out["Recall_at_13"] = np.nan
    return out


def quintiles(y, p):
    y, p = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(p, float))
    q = pd.qcut(y, 5, labels=False, duplicates="drop")
    rows = []
    for i in sorted(q.dropna().unique()):
        m = q == i
        rows.append({"quintile": int(i), "n": int(m.sum()), "MAE": mae(y[m], p[m]), "bias": float(np.mean(p[m] - y[m]))})
    return rows


def eval_one_target(target, feat_by_gen, folds, sol, base_mat, frozen_oof, frozen_test):
    ref_name = TMAPP_REF if target == "TmApp" else HIC_REF
    y = pd.read_csv(OOF_DIR / f"{ref_name}.csv").set_index("id")["y_true"].astype(float)
    y_test = sol.set_index("id")[target].astype(float)
    pub = list(sol.loc[sol.is_public == 1, "id"])
    pri = list(sol.loc[sol.is_private == 1, "id"])
    out = {}
    audits = {}
    inc_map = {}
    for gen in GENS:
        print("eval", target, gen, flush=True)
        X = feat_by_gen[gen].loc[y.index, CANON].astype(float)
        Xte = feat_by_gen[gen].loc[y_test.index, CANON].astype(float)
        oof, model, sc, a_full, alphas = nested_ridge(X, y, folds)
        base_oof, base_med = median_baseline_oof(y, folds)
        test_pred = pd.Series(model.predict(sc.transform(Xte)), index=y_test.index)

        m = {"alpha_full": a_full, "alphas_outer": alphas}
        for split, yy, pp, bb in [
            ("CV", y, oof, base_oof),
            ("Public", y_test.loc[pub], test_pred.loc[pub], pd.Series(base_med, index=pub)),
            ("Private", y_test.loc[pri], test_pred.loc[pri], pd.Series(base_med, index=pri)),
            ("AllTest", y_test, test_pred, pd.Series(base_med, index=y_test.index)),
        ]:
            for k, v in metrics(yy, pp).items():
                m[f"{k}_{split}"] = v
            m[f"baseline_MAE_{split}"] = mae(yy, bb)
        m["standalone_signal"] = classify_standalone(m)

        cand_oof, res_model, res_sc, audit = residual_oof(X, y, folds, frozen_oof, base_mat, target)
        audits[gen] = audit
        cand_test = frozen_test.astype(float) + pd.Series(
            res_model.predict(res_sc.transform(Xte)), index=y_test.index
        )
        boots = {}
        for split, yy, pc, pr_ in [
            ("CV", y, cand_oof, frozen_oof.loc[y.index]),
            ("Public", y_test.loc[pub], cand_test.loc[pub], frozen_test.loc[pub]),
            ("Private", y_test.loc[pri], cand_test.loc[pri], frozen_test.loc[pri]),
            ("AllTest", y_test, cand_test, frozen_test),
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
        m["empirical_verdict"] = empirical_verdict(m["standalone_signal"], m["incremental_signal"], target == "TmApp")
        if target == "TmApp":
            m["quintile_MAE_candidate_CV"] = quintiles(y, cand_oof)
            low = y <= np.quantile(y, 0.2)
            high = y >= np.quantile(y, 0.8)
            m["low_Tm_bias_candidate_CV"] = float(np.mean(cand_oof[low] - y[low]))
            m["high_Tm_bias_candidate_CV"] = float(np.mean(cand_oof[high] - y[high]))
        else:
            m["label"] = "SECONDARY_CROSS_ENDPOINT_AUDIT"
            m["hic_tail_candidate_CV"] = hic_tail(y, cand_oof)
            m["hic_tail_candidate_AllTest"] = hic_tail(y_test, cand_test)
        m["prespec_univariate_Dev"] = [
            {"feature": f, "spearman_vs_y": float(spearmanr(X[f], y).statistic)}
            for f in ["log_softness_1_20", "log_lambda_1", "softness_fraction_first5", "spectral_slope_1_20"]
        ]
        out[gen] = m
    out["_generator_signal"] = generator_signal_label(inc_map)
    return out, audits


def main():
    freeze_path = FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["target_scoring_started_after_feature_freeze"] = True
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")

    folds = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")
    sol = pd.read_csv(ROOT / "competition/data/secret/solution.csv")
    feat_by_gen = {
        g: pd.read_parquet(FAMILY / f"features_{g}.parquet")
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

    tmapp, tmapp_audit = eval_one_target("TmApp", feat_by_gen, folds, sol, tmapp_base, tmapp_oof, tmapp_test)
    hic, hic_audit = eval_one_target("HIC", feat_by_gen, folds, sol, hic_base, hic_oof, hic_test)

    (FAMILY / "EVALUATION_RESULTS.json").write_text(
        json.dumps({"family_id": "ANM-SPECTRUM", "version": "v1", "TmApp": tmapp, "HIC": hic}, indent=2, default=str)
        + "\n"
    )
    overall = all(a.get("pass", False) for a in list(tmapp_audit.values()) + list(hic_audit.values()))
    (FAMILY / "RESIDUAL_RIDGE_IMPLEMENTATION_AUDIT.json").write_text(
        json.dumps({"TmApp": tmapp_audit, "HIC": hic_audit, "overall_pass": overall}, indent=2) + "\n"
    )
    print("AUDIT", overall)
    for g in GENS:
        print(
            g,
            "TmApp",
            tmapp[g]["standalone_signal"],
            tmapp[g]["incremental_signal"],
            tmapp[g]["empirical_verdict"],
            {k: round(tmapp[g][k], 4) for k in ["MAE_CV", "MAE_Public", "MAE_Private", "delta_MAE_CV", "delta_MAE_Public", "delta_MAE_Private"]},
        )


if __name__ == "__main__":
    main()
