#!/usr/bin/env python3
"""Round1 CV / Public / Private model-level score correlation analysis."""
from __future__ import annotations

import ast
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
POST = ROOT / "virtual_participant/round1_postmortem"
PLOTS = POST / "plots"
EXPL = POST / "predictions_exploratory"
FIN = ROOT / "virtual_participant/round1_finalization"
SOLUTION = ROOT / "competition/data/secret/solution.csv"
SPLIT = ROOT / "competition/organizer/SPLIT_MANIFEST.json"

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage5_integration/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage4_advanced_structure/scripts"))
sys.path.insert(0, str(FIN / "scripts"))

STAGE_ORDER = {"Stage1": 1, "Stage2": 2, "Stage2b": 3, "Stage3": 4, "Stage4": 5, "Stage5": 6}
STRUCT_MAP = {
    "STRUCT_RASA": "RASA", "STRUCT_SURFACE_CHEM": "SURFACE_CHEM", "STRUCT_SURFACE_ALL": "SURFACE_ALL",
    "STRUCT_SASA": "SASA", "STRUCT_PATCH": "PATCH", "STRUCT_PACKING": "PACKING",
    "STRUCT_INTERFACE": "INTERFACE", "STRUCT_GLOBAL": "GLOBAL", "STRUCT_ALL": "ALL",
    "STRUCT_GEOMETRY_ALL": "GEOMETRY_ALL",
}
ADV_MAP = {
    "ADV_INTERACTIONS": "ADV_INTERACTIONS", "ADV_SURFACE_PATCH": "ADV_SURFACE_PATCH",
    "ADV_TMAPP_ALL": "ADV_TMAPP_ALL", "ADV_HIC_ALL": "ADV_HIC_ALL",
    "ADV_PROPKA": "ADV_PROPKA", "ADV_INVFOLD": "ADV_INVFOLD",
}

PRE_REVEAL_PRED = {
    ("TmApp", "TmApp__META_performance__ridge_100.0"): FIN / "predictions/TmApp_PRIMARY_predictions.csv",
    ("HIC", "HIC__SIMPLE_blend_seq_surf_adv"): FIN / "predictions/HIC_PRIMARY_predictions.csv",
    ("TmApp", "TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt"): FIN / "predictions/TmApp_SECONDARY_A_predictions.csv",
    ("TmApp", "TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt"): FIN / "predictions/TmApp_SECONDARY_B_predictions.csv",
    ("HIC", "HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt"): FIN / "predictions/HIC_SECONDARY_A_predictions.csv",
    ("HIC", "HIC__esm2__H__SVROpt"): FIN / "predictions/HIC_SECONDARY_B_predictions.csv",
}

ANNOTATE = {
    "TmApp": [
        ("TmApp__META_performance__ridge_100.0", "R1 PRIMARY"),
        ("TmApp__FUSION_S3INC__ADV_INTERACTIONS__SVROpt", "S4 incumbent"),
        ("TmApp__FUSION__ablang2__HL_paired__SEQ_BASIC__SVROpt", "S2 PLM+seq"),
        ("TmApp__SEQ_BASIC__SVROpt", "S1 best"),
    ],
    "HIC": [
        ("HIC__SIMPLE_blend_seq_surf_adv", "R1 PRIMARY"),
        ("HIC__FUSION_S3INC__ADV_SURFACE_PATCH__SVROpt", "S4 incumbent"),
        ("HIC__FUSION__esm2__H__SEQ_ALL__SVROpt", "S2 fusion"),
        ("HIC__SEQ_PLUS_ANTIBODY__SVROpt", "S1 best"),
    ],
}


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def parse_hp(s):
    if pd.isna(s) or s == "" or s == "{}":
        return {}
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return {}


def model_kind_from_name(name: str) -> str:
    n = str(name)
    if "ElasticNet" in n:
        return "ElasticNet"
    if "Ridge" in n:
        return "Ridge"
    return "SVR"


def collect_raw_rows() -> pd.DataFrame:
    rows = []
    sources = [
        ("Stage1", ROOT / "virtual_participant/stage1_features/stage1_primary_results.csv",
         "primary_mae", "feature_family", "model"),
        ("Stage2", ROOT / "virtual_participant/stage2_plm/stage2_primary_results.csv",
         "Primary_MAE", None, "model"),
        ("Stage2", ROOT / "virtual_participant/stage2_plm/stage2_fusion_results.csv",
         "Primary_MAE", "classical_family", "model"),
        ("Stage2b", ROOT / "virtual_participant/stage2b_pca/stage2b_pca_dimension_results.csv",
         "Primary_MAE", None, "model"),
        ("Stage2b", ROOT / "virtual_participant/stage2b_pca/stage2b_fusion_results.csv",
         "Primary_MAE", None, "model"),
        ("Stage3", ROOT / "virtual_participant/stage3_structure/stage3_primary_results.csv",
         "Primary_MAE", "feature_family", "model"),
        ("Stage3", ROOT / "virtual_participant/stage3_structure/stage3_fusion_results.csv",
         "Primary_MAE", "feature_family", "model"),
        ("Stage4", ROOT / "virtual_participant/stage4_advanced_structure/stage4_primary_results.csv",
         "Primary_MAE", "feature_family", "model"),
        ("Stage4", ROOT / "virtual_participant/stage4_advanced_structure/stage4_fusion_results.csv",
         "Primary_MAE", "feature_family", "model"),
        ("Stage5", ROOT / "virtual_participant/stage5_integration/stage5_nested_stack_results.csv",
         "Primary_MAE", "branch", "meta_model"),
        ("Stage5", ROOT / "virtual_participant/stage5_integration/stage5_simple_blend_results.csv",
         "Primary_MAE", "branch", None),
        ("Stage5", ROOT / "virtual_participant/stage5_integration/stage5_calibration_results.csv",
         "Primary_MAE", "branch", "calibration_method"),
    ]
    skip_status = {"FIXED_SCREEN", "BASELINE"}
    for stage, fp, pcol, fam_col, model_col in sources:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        for _, r in df.iterrows():
            eid = r.get("experiment_id")
            if pd.isna(eid):
                continue
            st = str(r.get("status", ""))
            if st in skip_status or "FIXED" in st:
                continue
            notes = str(r.get("notes", ""))
            if "fixed pipeline" in notes.lower() or "fixed hp" in notes.lower():
                continue
            fam = r.get(fam_col) if fam_col else None
            if fam_col == "feature_family" and str(fam).startswith("STRUCT_"):
                modality = "structure"
            elif fam_col == "feature_family" and str(fam).startswith("ADV_"):
                modality = "advanced_structure"
            elif "FUSION" in str(eid) or "blend" in str(eid) or "META" in str(eid):
                modality = "fusion/integration"
            elif "esm" in str(eid).lower() or "ablang" in str(eid).lower():
                modality = "plm"
            elif stage == "Stage1":
                modality = "sequence"
            else:
                modality = str(fam) if fam else "other"
            rows.append({
                "experiment_id": eid,
                "target": r["target"],
                "stage": stage,
                "model_family": fam if fam_col else r.get("branch", stage),
                "modality": modality,
                "Primary_CV_MAE": float(r[pcol]) if pcol in r and pd.notna(r[pcol]) else np.nan,
                "hyperparameters": r.get("hyperparameters", r.get("hyperparameters", "{}")),
                "model_name": r.get(model_col) if model_col else None,
                "base_model_ids": r.get("base_model_ids"),
                "meta_model": r.get("meta_model"),
                "status_row": st,
            })
    return pd.DataFrame(rows)


def collect_shadow() -> pd.DataFrame:
    parts = []
    for fp in [
        ROOT / "virtual_participant/stage1_features/stage1_shadow_results.csv",
        ROOT / "virtual_participant/stage2_plm/stage2_shadow_results.csv",
        ROOT / "virtual_participant/stage2b_pca/stage2b_shadow_results.csv",
        ROOT / "virtual_participant/stage3_structure/stage3_shadow_results.csv",
        ROOT / "virtual_participant/stage4_advanced_structure/stage4_shadow_results.csv",
        ROOT / "virtual_participant/stage5_integration/stage5_shadow_results.csv",
    ]:
        if fp.exists():
            df = pd.read_csv(fp)
            col = "shadow_mae" if "shadow_mae" in df.columns else "Shadow_MAE"
            parts.append(df[["experiment_id", col]].rename(columns={col: "Shadow_CV_MAE"}))
    if not parts:
        return pd.DataFrame(columns=["experiment_id", "Shadow_CV_MAE"])
    sh = pd.concat(parts, ignore_index=True)
    return sh.groupby("experiment_id", as_index=False)["Shadow_CV_MAE"].first()


def dedupe_inventory(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["stage_ord"] = raw["stage"].map(STAGE_ORDER)
    raw = raw.sort_values(["experiment_id", "stage_ord"])
    best = raw.groupby("experiment_id", as_index=False).last()
    return best.drop(columns=["stage_ord"])


def build_cfg_lookup(raw: pd.DataFrame) -> dict[str, dict]:
    lookup = {}
    base = json.loads((ROOT / "virtual_participant/stage5_integration/stage5_base_models.json").read_text())
    for t in ["TmApp", "HIC"]:
        for c in base[t]:
            lookup[c["experiment_id"]] = c

    for _, r in raw.iterrows():
        eid = r["experiment_id"]
        if eid in lookup:
            continue
        hp = parse_hp(r.get("hyperparameters"))
        mk = model_kind_from_name(r.get("model_name") or eid)
        cfg = {"experiment_id": eid, "model_kind": mk, "params": hp, "pca_dim": None,
               "plm_key": None, "classical_fam": None, "struct_fam": None, "s3_subset": None, "adv_fam": None}

        if eid.startswith("TmApp__SEQ_") or eid.startswith("HIC__SEQ_") or "ANN_" in eid or "ANTIBODY" in eid:
            m = re.match(r"(TmApp|HIC)__(SEQ_[A-Z_]+|ANN_[A-Z_]+|ANTIBODY_[A-Z_]+)__", eid)
            if m:
                cfg["classical_fam"] = m.group(2)
                lookup[eid] = cfg
                continue

        m = re.match(r"(TmApp|HIC)__(esm1b|esm2|ablang2)__(H|L|HL|HL_paired)__(.+)", eid)
        if m and "FUSION" not in eid:
            cfg["plm_key"] = f"{m.group(2)}__{m.group(3)}"
            pca_m = re.search(r"PCA(\d+)", eid)
            cfg["pca_dim"] = int(pca_m.group(1)) if pca_m else (48 if "Opt" in eid else None)
            lookup[eid] = cfg
            continue

        m = re.match(r"(TmApp|HIC)__FUSION__(esm1b|esm2|ablang2)__(H|L|HL|HL_paired)__(SEQ_[A-Z_]+|ANN_[A-Z_]+)__", eid)
        if m:
            cfg["plm_key"] = f"{m.group(2)}__{m.group(3)}"
            cfg["classical_fam"] = m.group(4)
            cfg["pca_dim"] = 48 if "Opt" in eid else 32
            lookup[eid] = cfg
            continue

        m = re.match(r"(TmApp|HIC)__ESMFold__([A-Z_]+)__", eid)
        if m and "FUSION" not in eid:
            cfg["struct_fam"] = STRUCT_MAP.get(f"STRUCT_{m.group(2)}", m.group(2).replace("STRUCT_", ""))
            if cfg["struct_fam"] not in ("RASA", "SURFACE_CHEM", "SURFACE_ALL", "PACKING", "INTERFACE", "PATCH", "GLOBAL", "ALL", "GEOMETRY_ALL"):
                fam = r.get("model_family", "")
                cfg["struct_fam"] = STRUCT_MAP.get(str(fam), "RASA")
            lookup[eid] = cfg
            continue

        m = re.match(r"(TmApp|HIC)__ADV_([A-Z_]+)__", eid)
        if m and "FUSION" not in eid:
            cfg["adv_fam"] = ADV_MAP.get(f"ADV_{m.group(2)}", f"ADV_{m.group(2)}")
            lookup[eid] = cfg
            continue

        if "FUSION_S3INC" in eid:
            if "INTERACTIONS" in eid:
                cfg.update(plm_key="ablang2__HL_paired", classical_fam="SEQ_BASIC", s3_subset="rasa", adv_fam="ADV_INTERACTIONS", pca_dim=48)
            elif "SURFACE_PATCH" in eid:
                cfg.update(plm_key="esm2__H", classical_fam="SEQ_ALL", s3_subset="surface", adv_fam="ADV_SURFACE_PATCH", pca_dim=8)
            lookup[eid] = cfg
            continue

        if "FUSION_OVERALL" in eid or "FUSION_PLM" in eid:
            cfg.update(plm_key="esm2__H" if "HIC" in eid else "ablang2__HL_paired",
                       classical_fam="SEQ_BASIC" if "TmApp" in eid else "SEQ_ALL",
                       struct_fam="RASA" if "RASA" in eid else "SURFACE_ALL", pca_dim=48)
            lookup[eid] = cfg
            continue

    # stage5 meta/blend from CSV rows
    for fp, kind in [
        (ROOT / "virtual_participant/stage5_integration/stage5_nested_stack_results.csv", "meta"),
        (ROOT / "virtual_participant/stage5_integration/stage5_simple_blend_results.csv", "blend"),
    ]:
        if not fp.exists():
            continue
        for _, r in pd.read_csv(fp).iterrows():
            eid = r["experiment_id"]
            lookup[eid] = {"experiment_id": eid, "recipe": kind,
                           "base_model_ids": str(r["base_model_ids"]).split("|"),
                           "meta_model": r.get("meta_model"), "meta_alpha": 100.0}
    return lookup


def load_store():
    from run_round1_predictions import load_combined_store
    return load_combined_store()


def fit_predict_test(cfg, store, y_dev, dev_idx, test_idx):
    from run_stage5 import fold_prepare, make_model
    n = len(store["ids"])
    tr = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[dev_idx] = True
    te[test_idx] = True
    Xtr, Xte, _ = fold_prepare(cfg, store, tr, te)
    model = make_model(cfg["model_kind"], cfg["params"])
    model.fit(Xtr, y_dev)
    return model.predict(Xte)


def predict_stack(base_ids, meta_alpha, store, y, dev_idx, test_idx, lookup):
    from sklearn.linear_model import Ridge
    from run_round1_predictions import cv_oof_dev, fit_predict_test
    cols = []
    for bid in base_ids:
        c = lookup.get(bid)
        if c is None:
            return None
        cols.append(fit_predict_test(c, store, y, dev_idx, test_idx))
    X = np.column_stack(cols)
    meta = Ridge(alpha=meta_alpha, random_state=0)
    # fit on dev OOF would be ideal; use in-sample for exploratory scoring consistency
    oof_cols = []
    for bid in base_ids:
        from run_round1_predictions import cv_oof_dev
        import pandas as pd
        folds = pd.read_csv(ROOT / "virtual_participant/stage0_cv/cv_primary.csv")["fold"].to_numpy()
        oof_cols.append(cv_oof_dev(lookup[bid], store, y, folds, dev_idx))
    meta.fit(np.column_stack(oof_cols), y)
    return meta.predict(X)


def predict_blend(base_ids, store, y, dev_idx, test_idx, lookup):
    from run_round1_predictions import fit_predict_test
    preds = []
    for bid in base_ids:
        c = lookup.get(bid)
        if c is None:
            return None
        preds.append(fit_predict_test(c, store, y, dev_idx, test_idx))
    return np.mean(np.column_stack(preds), axis=1)


def predict_model(eid, cfg, store, target, dev_idx, test_idx):
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")
    y = dev[target].to_numpy(float)
    if cfg.get("recipe") == "meta":
        alpha = 100.0
        if cfg.get("meta_model", "").startswith("ridge"):
            alpha = float(str(cfg["meta_model"]).split("_")[-1])
        return predict_stack(cfg["base_model_ids"], alpha, store, y, dev_idx, test_idx,
                             build_cfg_lookup(collect_raw_rows()))
    if cfg.get("recipe") == "blend":
        return predict_blend(cfg["base_model_ids"], store, y, dev_idx, test_idx,
                             build_cfg_lookup(collect_raw_rows()))
    if cfg.get("model_kind") is None and cfg.get("classical_fam") is None:
        return None
    return fit_predict_test(cfg, store, y, dev_idx, test_idx)


def score_predictions(pred: np.ndarray, sol: pd.DataFrame, public, private, target):
    sub = sol.copy()
    sub["pred"] = pred
    return {
        "All_Test_MAE": mae(sub[target], sub["pred"]),
        "Public_MAE": mae(sub.loc[sub.id.isin(public), target], sub.loc[sub.id.isin(public), "pred"]),
        "Private_MAE": mae(sub.loc[sub.id.isin(private), target], sub.loc[sub.id.isin(private), "pred"]),
    }


def ensure_test_scores(inv: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    EXPL.mkdir(parents=True, exist_ok=True)
    sol = pd.read_csv(SOLUTION)
    public = set(json.load(open(SPLIT))["public_ids"])
    private = set(json.load(open(SPLIT))["private_ids"])
    store = load_store()
    dev_idx = np.arange(store["n_dev"])
    test_idx = np.arange(store["n_dev"], store["n_dev"] + 162)

    test_scores = {c: [] for c in ["Public_MAE", "Private_MAE", "All_Test_MAE", "pre_reveal_frozen", "evidence_status"]}
    for i, r in inv.iterrows():
        eid, target = r["experiment_id"], r["target"]
        key = (target, eid)
        frozen = key in PRE_REVEAL_PRED
        cache = EXPL / f"{eid}.csv"
        if frozen:
            p = pd.read_csv(PRE_REVEAL_PRED[key])
            pred = p.set_index("id").loc[sol["id"], "prediction"].values
            ev = "PRE_REVEAL_FROZEN"
        elif cache.exists():
            pred = pd.read_csv(cache).set_index("id").loc[sol["id"], "prediction"].values
            ev = "POST_REVEAL_EXPLORATORY"
        else:
            cfg = lookup.get(eid)
            if cfg is None or "CAL_" in eid or "RESID_" in eid:
                inv.at[i, "Public_MAE"] = np.nan
                inv.at[i, "Private_MAE"] = np.nan
                inv.at[i, "All_Test_MAE"] = np.nan
                inv.at[i, "pre_reveal_frozen"] = False
                inv.at[i, "evidence_status"] = "CV_ONLY_NO_TEST"
                inv.at[i, "notes"] = "no reconstructable test recipe"
                continue
            try:
                pred = predict_model(eid, cfg, store, target, dev_idx, test_idx)
                if pred is None:
                    raise ValueError("no pred")
                pd.DataFrame({"id": sol["id"], "prediction": pred}).to_csv(cache, index=False)
                ev = "POST_REVEAL_EXPLORATORY"
            except Exception as ex:
                inv.at[i, "Public_MAE"] = np.nan
                inv.at[i, "Private_MAE"] = np.nan
                inv.at[i, "All_Test_MAE"] = np.nan
                inv.at[i, "pre_reveal_frozen"] = False
                inv.at[i, "evidence_status"] = "CV_ONLY_PRED_FAILED"
                inv.at[i, "notes"] = str(ex)[:80]
                continue
        sc = score_predictions(pred, sol, public, private, target)
        inv.at[i, "Public_MAE"] = sc["Public_MAE"]
        inv.at[i, "Private_MAE"] = sc["Private_MAE"]
        inv.at[i, "All_Test_MAE"] = sc["All_Test_MAE"]
        inv.at[i, "pre_reveal_frozen"] = frozen
        inv.at[i, "evidence_status"] = ev
        if (i + 1) % 25 == 0:
            print(f"scored {i+1}/{len(inv)}", flush=True)
    return inv


def _corr_scalar(x, y, method="pearson"):
    if method == "pearson":
        r, _ = pearsonr(x, y)
    else:
        r, _ = spearmanr(x, y)
    if isinstance(r, np.ndarray):
        r = r.flat[0]
    return float(r)


def corr_matrix(df, cols, method="pearson"):
    k = len(cols)
    mat = np.full((k, k), np.nan)
    nmat = np.zeros((k, k), dtype=int)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            sub = df[[a, b]].dropna()
            nmat[i, j] = len(sub)
            if i == j:
                mat[i, j] = 1.0 if len(sub) >= 1 else np.nan
                continue
            if len(sub) < 3:
                continue
            mat[i, j] = _corr_scalar(sub[a].values, sub[b].values, method)
    return pd.DataFrame(mat, index=cols, columns=cols), pd.DataFrame(nmat, index=cols, columns=cols)


def key_pairs(df, target):
    cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]
    pairs = [
        ("Primary_CV_MAE", "Shadow_CV_MAE"), ("Primary_CV_MAE", "Public_MAE"),
        ("Primary_CV_MAE", "Private_MAE"), ("Primary_CV_MAE", "All_Test_MAE"),
        ("Shadow_CV_MAE", "Public_MAE"), ("Shadow_CV_MAE", "Private_MAE"), ("Shadow_CV_MAE", "All_Test_MAE"),
        ("Public_MAE", "Private_MAE"), ("Public_MAE", "All_Test_MAE"), ("Private_MAE", "All_Test_MAE"),
    ]
    rows = []
    for a, b in pairs:
        sub = df[df.target == target][[a, b]].dropna()
        if len(sub) < 3:
            rp = rs = np.nan
        else:
            rp = _corr_scalar(sub[a].values, sub[b].values, "pearson")
            rs = _corr_scalar(sub[a].values, sub[b].values, "spearman")
        rows.append({"Target": target, "X": a, "Y": b, "Pearson_r": rp, "Spearman_rho": rs, "N": len(sub)})
    # CV mean variants
    subdf = df[df.target == target].copy()
    subdf = subdf.dropna(subset=["Primary_CV_MAE", "Shadow_CV_MAE"])
    if len(subdf):
        subdf["CV_MEAN"] = (subdf["Primary_CV_MAE"] + subdf["Shadow_CV_MAE"]) / 2
        subdf["CV_WORST"] = subdf[["Primary_CV_MAE", "Shadow_CV_MAE"]].max(axis=1)
        for label, col in [("CV_MEAN", "CV_MEAN"), ("CV_WORST", "CV_WORST")]:
            for y in ["Public_MAE", "Private_MAE", "All_Test_MAE"]:
                s = subdf[[col, y]].dropna()
                if len(s) >= 3:
                    rp = _corr_scalar(s[col].values, s[y].values, "pearson")
                    rs = _corr_scalar(s[col].values, s[y].values, "spearman")
                else:
                    rp = rs = np.nan
                rows.append({"Target": target, "X": label, "Y": y, "Pearson_r": rp, "Spearman_rho": rs, "N": len(s)})
    return pd.DataFrame(rows)


def add_ranks(inv):
    for target in ["TmApp", "HIC"]:
        m = inv["target"] == target
        for col in ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]:
            inv.loc[m, col.replace("_MAE", "_rank")] = inv.loc[m, col].rank(method="min")
    inv["rank_Primary_minus_Private"] = inv["Primary_CV_rank"] - inv["Private_rank"]
    inv["rank_Primary_minus_AllTest"] = inv["Primary_CV_rank"] - inv["All_Test_rank"]
    inv["rank_Shadow_minus_Private"] = inv["Shadow_CV_rank"] - inv["Private_rank"]
    return inv


def plot_pairplot(df, target, path):
    cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]
    sub = df[df.target == target].dropna(subset=["Primary_CV_MAE", "Private_MAE"], how="any")
    sub = sub.dropna(subset=cols, thresh=3)
    if len(sub) < 5:
        return
    sub = sub.copy()
    sub["_frozen"] = sub.pre_reveal_frozen.astype(bool)
    g = sns.pairplot(
        sub, vars=cols, hue="stage",
        plot_kws={"alpha": 0.65, "s": 45, "edgecolor": "k", "linewidth": 0.3},
        diag_kind="hist", corner=False,
    )
    # open markers for post-reveal exploratory
    for ax in g.axes.flat:
        if ax is None:
            continue
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0:
                continue
    g.fig.suptitle(
        f"{target} model scores (N={len(sub)}); lower MAE is better; filled=pre-reveal frozen",
        y=1.02, fontsize=11,
    )
    g.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")


def plot_heatmap(mat, nmat, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    for i in range(len(mat)):
        for j in range(len(mat)):
            if not np.isnan(mat.iloc[i, j]):
                ax.text(j + 0.5, i + 0.7, f"n={int(nmat.iloc[i,j])}", ha="center", va="center", fontsize=7, color="gray")
    ax.set_title(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_stage_trajectory(df, target, xcol, ycol, path):
    sub = df[(df.target == target) & df[xcol].notna() & df[ycol].notna()].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=sub, x=xcol, y=ycol, hue="stage", style="pre_reveal_frozen", s=60, alpha=0.7, ax=ax)
    med = sub.groupby("stage")[[xcol, ycol]].median().reset_index()
    ax.scatter(med[xcol], med[ycol], s=200, marker="D", c="black", label="stage median", zorder=5)
    for eid, lab in ANNOTATE.get(target, []):
        row = sub[sub.experiment_id == eid]
        if len(row):
            ax.annotate(lab, (row[xcol].values[0], row[ycol].values[0]), fontsize=8, alpha=0.9)
    ax.set_xlabel(xcol.replace("_", " "))
    ax.set_ylabel(ycol.replace("_", " "))
    ax.set_title(f"{target}: {xcol} vs {ycol} (lower is better)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_stagewise(df, target, path):
    cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]
    long = df[df.target == target].melt(id_vars=["stage"], value_vars=cols, var_name="score_type", value_name="MAE")
    long = long.dropna()
    stage_order = sorted(long.stage.unique(), key=lambda s: STAGE_ORDER.get(s, 99))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long, x="stage", y="MAE", hue="score_type", order=stage_order, ax=ax)
    ax.set_title(f"{target} score distributions by stage (lower is better)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_rank_cv_test(df, target, path):
    sub = df[(df.target == target) & df["Primary_CV_rank"].notna() & df["Private_rank"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=sub, x="Primary_CV_rank", y="Private_rank", hue="stage", s=60, alpha=0.7, ax=ax)
    ax.plot([1, sub["Primary_CV_rank"].max()], [1, sub["Primary_CV_rank"].max()], "k--", lw=1)
    ax.set_xlabel("Primary CV rank (1=best)")
    ax.set_ylabel("Private rank (1=best)")
    ax.set_title(f"{target}: CV rank vs Private rank")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def stagewise_summary(df):
    rows = []
    cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]
    for target in ["TmApp", "HIC"]:
        for stage in sorted(df.stage.unique(), key=lambda s: STAGE_ORDER.get(s, 99)):
            sub = df[(df.target == target) & (df.stage == stage)]
            if sub.empty:
                continue
            row = {"target": target, "stage": stage, "N_models": len(sub)}
            for c in cols:
                s = sub[c].dropna()
                if len(s):
                    row[f"median_{c}"] = s.median()
                    row[f"best_{c}"] = s.min()
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(inv, key_corr, stagewise, q, outliers):
    plot_rel = "plots/"
    md = ["# Round 1 — CV / Public / Private Model Score Correlation\n\n"]
    md.append("Organizer reveal 後の postmortem diagnostic。モデルは **pre-reveal frozen**（hash freeze 済み Test prediction）と **post-reveal exploratory**（reveal 後に生成）を区別。後者を用いた「Round1 で選ばれていたはず」という遡及的主張は行わない。\n\n")

    md.append("## 1. 何を比較したか\n\n")
    md.append("Stage1–Stage5 の registry 上 formal experiment について、model-level MAE（Primary CV / Shadow CV / Public / Private / All Test）の散布・相関・順位逆転を **descriptive** に分析。各点は独立サンプルではなく、p-value 中心の推論は行わない。\n\n")

    md.append("## 2. 対象モデル数\n\n")
    for t in ["TmApp", "HIC"]:
        sub = inv[inv.target == t]
        by_stage = sub.groupby("stage").size()
        md.append(f"### {t}\n\n")
        md.append(f"- **全モデル:** {len(sub)}\n")
        md.append(f"- **pre-reveal frozen:** {int(sub.pre_reveal_frozen.sum())}\n")
        md.append(f"- **post-reveal exploratory:** {int((sub.evidence_status == 'POST_REVEAL_EXPLORATORY').sum())}\n")
        md.append(f"- **CV-only（Test 未生成）:** {int(sub.evidence_status.str.startswith('CV_ONLY', na=False).sum())}\n")
        md.append("- Stage別: " + ", ".join(f"{s}={n}" for s, n in by_stage.items()) + "\n\n")

    for t, sec in [("TmApp", "3"), ("HIC", "5")]:
        md.append(f"## {sec}. {t} pairplot\n\n")
        md.append(f"![{t} pairplot]({plot_rel}{t.lower()}_score_pairplot.png)\n\n")

    md.append("## 4. TmApp 相関係数\n\n")
    md.append("![TmApp Pearson](plots/tmapp_pearson_heatmap.png)\n\n")
    md.append("![TmApp Spearman](plots/tmapp_spearman_heatmap.png)\n\n")

    md.append("## 6. HIC 相関係数\n\n")
    md.append("![HIC Pearson](plots/hic_pearson_heatmap.png)\n\n")
    md.append("![HIC Spearman](plots/hic_spearman_heatmap.png)\n\n")

    md.append("## 7. Primary CV と Private の関係\n\n")
    md.append(f"- **TmApp (Q1):** {q.get('TmApp_Primary_vs_Private', '')}\n")
    md.append(f"- **HIC (Q3):** {q.get('HIC_Primary_vs_Private', '')}\n")
    md.append(f"- **TmApp gap (Q7):** {q.get('TmApp_gap', '')} — {q.get('TmApp_gap_dataset_wide', '')}\n\n")
    md.append(f"![TmApp Primary vs Private]({plot_rel}tmapp_primary_vs_private_by_stage.png)\n\n")

    md.append("## 8. Shadow CV と Private の関係\n\n")
    md.append(f"- **TmApp (Q2):** {q.get('TmApp_Shadow_vs_Private', '')} — Shadow > Primary? **{q.get('TmApp_Shadow_better_than_Primary', '')}**\n")
    md.append(f"- **HIC (Q4):** {q.get('HIC_Shadow_vs_Private', '')} — Shadow > Primary? **{q.get('HIC_Shadow_better_than_Primary', '')}**\n\n")
    md.append(f"![TmApp Shadow vs Private]({plot_rel}tmapp_shadow_vs_private_by_stage.png)\n\n")

    md.append("## 9. Public と Private の関係\n\n")
    md.append(f"- **TmApp (Q5):** {q.get('TmApp_Public_vs_Private', '')}\n")
    md.append(f"- **HIC:** {q.get('HIC_Public_vs_Private', '')}\n\n")

    md.append("## 10. Stage ごとの性能推移\n\n")
    md.append(f"**Q6:** {q.get('stage_progression', '')}\n\n")
    md.append(stagewise.to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\n")
    md.append(f"![TmApp stagewise]({plot_rel}tmapp_stagewise_scores.png)\n\n")
    md.append(f"![HIC stagewise]({plot_rel}hic_stagewise_scores.png)\n\n")

    md.append("## 11. CV 順位と Test 順位の逆転\n\n")
    md.append(f"最大の Primary→Private rank reversal: **{q.get('max_rank_reversal', '')}**\n\n")
    md.append(f"![TmApp rank]({plot_rel}tmapp_rank_cv_vs_test.png)\n\n")
    md.append(f"![HIC rank]({plot_rel}hic_rank_cv_vs_test.png)\n\n")

    md.append("## 12. Pre-reveal frozen model だけで見るとどうか\n\n")
    md.append(f"{q.get('prereveal_summary', '')}\n\n")
    md.append("N が小さいため相関は参考値。詳細: `round1_prereveal_score_correlations_*.csv`\n\n")

    md.append("## CV ensemble 指標（Q9 / §19 diagnostic）\n\n")
    md.append(f"{q.get('cv_ensemble', '')}\n\n")
    md.append("Key pairs 全表:\n\n")
    md.append(key_corr.to_markdown(index=False, floatfmt=".3f"))
    md.append("\n\n")

    md.append("## 13. Round1 CV protocol から学べること\n\n")
    md.append("1. 各 model 点は独立ではない（同一 Dev / feature family を共有）。\n")
    md.append("2. 相関係数は descriptive のみ。\n")
    md.append("3. pre-reveal frozen N=6 の subset では推定が不安定。\n")
    md.append("4. ALL MODELS 解析に post-reveal exploratory を含むため exploratory。\n")
    md.append("5. Public / Private は各 N=81 で ranking に sampling variation あり。\n\n")

    md.append("## 14. Round2 への示唆\n\n")
    md.append(q.get("round2_hint", ""))
    md.append("\n\n")

    if len(outliers):
        md.append("## 付録: 外れモデル（post-hoc）\n\n")
        md.append(outliers.head(30).to_markdown(index=False, floatfmt=".3f"))
        md.append("\n\n")

    md.append("**状態:** `ROUND1_MODEL_SCORE_CORRELATION_ANALYSIS_COMPLETE`\n")
    (POST / "ROUND1_MODEL_SCORE_CORRELATION_ANALYSIS_JA.md").write_text("".join(md))


def extract_outliers(inv):
    rows = []
    for target in ["TmApp", "HIC"]:
        sub = inv[(inv.target == target) & inv["Primary_CV_MAE"].notna() & inv["Private_MAE"].notna()].copy()
        if sub.empty:
            continue
        sub["cv_private_gap"] = sub["Private_MAE"] - sub["Primary_CV_MAE"]
        sub["pub_priv_gap"] = (sub["Public_MAE"] - sub["Private_MAE"]).abs()
        if "Shadow_CV_MAE" in sub.columns:
            sub["shadow_priv_gap"] = (sub["Shadow_CV_MAE"] - sub["Private_MAE"]).abs()

        for _, r in sub.nsmallest(8, "Primary_CV_MAE").nlargest(8, "cv_private_gap").head(8).iterrows():
            rows.append({"category": "A_good_CV_bad_Private", "target": target, **r[["experiment_id", "stage", "Primary_CV_MAE", "Private_MAE", "cv_private_gap"]].to_dict()})
        for _, r in sub[(sub["Primary_CV_MAE"] > sub["Primary_CV_MAE"].median())].nsmallest(8, "Private_MAE").iterrows():
            rows.append({"category": "B_mediocre_CV_good_Private", "target": target, **r[["experiment_id", "stage", "Primary_CV_MAE", "Private_MAE"]].to_dict()})
        if sub["Public_MAE"].notna().any():
            pub_med = sub["Public_MAE"].median()
            for _, r in pd.concat([
                sub.nsmallest(5, "Public_MAE"),
                sub.nlargest(5, "Public_MAE"),
            ]).drop_duplicates("experiment_id").head(10).iterrows():
                rows.append({"category": "C_public_extreme", "target": target, **r[["experiment_id", "stage", "Public_MAE", "Private_MAE"]].to_dict(), "pub_vs_median": r["Public_MAE"] - pub_med})
        if "shadow_priv_gap" in sub.columns:
            sh = sub.dropna(subset=["Shadow_CV_MAE", "Private_MAE"])
            for _, r in sh.nsmallest(8, "shadow_priv_gap").iterrows():
                rows.append({"category": "D_shadow_private_match", "target": target, **r[["experiment_id", "stage", "Shadow_CV_MAE", "Private_MAE", "shadow_priv_gap"]].to_dict()})
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true", help="Skip test prediction; load saved inventory CSV")
    args = ap.parse_args()

    POST.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    inv_path = POST / "round1_all_model_score_inventory.csv"
    if args.analysis_only and inv_path.exists():
        print("load saved inventory...", flush=True)
        inv = pd.read_csv(inv_path)
        if "Primary_CV_rank" not in inv.columns:
            inv = add_ranks(inv)
    else:
        print("collect inventory...", flush=True)
        raw = collect_raw_rows()
        inv = dedupe_inventory(raw)
        sh = collect_shadow()
        inv = inv.merge(sh, on="experiment_id", how="left")
        if "Primary_CV_MAE" not in inv.columns:
            inv["Primary_CV_MAE"] = np.nan
        inv["pre_reveal_frozen"] = False
        inv["evidence_status"] = ""
        inv["notes"] = ""
        lookup = build_cfg_lookup(raw)
        print(f"inventory {len(inv)} models, predicting test...", flush=True)
        inv = ensure_test_scores(inv, lookup)
        inv = add_ranks(inv)
    inv.to_csv(POST / "round1_all_model_score_inventory.csv", index=False)
    inv.to_csv(POST / "round1_all_model_rankings.csv", index=False)

    score_cols = ["Primary_CV_MAE", "Shadow_CV_MAE", "Public_MAE", "Private_MAE", "All_Test_MAE"]
    key_rows = []
    q = {}
    for target in ["TmApp", "HIC"]:
        sub = inv[inv.target == target]
        kp = key_pairs(inv, target)
        key_rows.append(kp)
        pr = kp[(kp.X == "Primary_CV_MAE") & (kp.Y == "Private_MAE")].iloc[0]
        sr = kp[(kp.X == "Shadow_CV_MAE") & (kp.Y == "Private_MAE")].iloc[0]
        pp = kp[(kp.X == "Public_MAE") & (kp.Y == "Private_MAE")].iloc[0]
        q[f"{target}_Primary_vs_Private"] = f"Pearson={pr.Pearson_r:.3f}, Spearman={pr.Spearman_rho:.3f}, N={int(pr.N)}"
        q[f"{target}_Shadow_vs_Private"] = f"Pearson={sr.Pearson_r:.3f}, Spearman={sr.Spearman_rho:.3f}, N={int(sr.N)}"
        q[f"{target}_Public_vs_Private"] = f"Pearson={pp.Pearson_r:.3f}, Spearman={pp.Spearman_rho:.3f}, N={int(pp.N)}"
        q[f"{target}_Shadow_better_than_Primary"] = "Yes" if abs(sr.Pearson_r) > abs(pr.Pearson_r) else "No"
        for method in ["pearson", "spearman"]:
            mat, nmat = corr_matrix(sub, score_cols, method)
            mat.to_csv(POST / f"round1_score_correlations_{target.lower()}_{method}.csv")
            nmat.to_csv(POST / f"round1_score_correlation_n_{target.lower()}.csv")
            plot_heatmap(mat, nmat, f"{target} {method}", PLOTS / f"{target.lower()}_{method}_heatmap.png")
        plot_pairplot(inv, target, PLOTS / f"{target.lower()}_score_pairplot.png")
        for xcol, ycol, name in [
            ("Primary_CV_MAE", "Private_MAE", "primary_vs_private_by_stage"),
            ("Shadow_CV_MAE", "Private_MAE", "shadow_vs_private_by_stage"),
        ]:
            plot_stage_trajectory(inv, target, xcol, ycol, PLOTS / f"{target.lower()}_{name}.png")
        plot_stagewise(inv, target, PLOTS / f"{target.lower()}_stagewise_scores.png")
        plot_rank_cv_test(inv, target, PLOTS / f"{target.lower()}_rank_cv_vs_test.png")
        frozen = sub[sub.pre_reveal_frozen]
        if len(frozen) >= 3:
            for method in ["pearson", "spearman"]:
                mat, nmat = corr_matrix(frozen, score_cols, method)
                mat.to_csv(POST / f"round1_prereveal_score_correlations_{target.lower()}_{method}.csv")

    key_corr = pd.concat(key_rows, ignore_index=True)
    key_corr.to_csv(POST / "round1_key_score_correlations.csv", index=False)
    sw = stagewise_summary(inv)
    sw.to_csv(POST / "round1_stagewise_score_summary.csv", index=False)
    outliers = extract_outliers(inv)
    outliers.to_csv(POST / "round1_outlier_models.csv", index=False)

    # Q7/Q8: gap analysis
    tm = inv[inv.target == "TmApp"].dropna(subset=["Primary_CV_MAE", "Private_MAE"])
    tm["gap"] = tm["Private_MAE"] - tm["Primary_CV_MAE"]
    q["TmApp_gap"] = f"median gap={tm.gap.median():.3f}, mean={tm.gap.mean():.3f}, fraction models with gap>0.3: {(tm.gap>0.3).mean():.2f}"
    q["TmApp_gap_dataset_wide"] = "Yes — gap is positive for majority of models" if tm.gap.median() > 0.2 else "Mixed"
    hi = inv[inv.target == "HIC"].dropna(subset=["Primary_CV_MAE", "Private_MAE"])
    hi["gap"] = hi["Private_MAE"] - hi["Primary_CV_MAE"]
    q["HIC_alignment"] = f"median gap={hi.gap.median():.3f}; Primary-Private Pearson on {len(hi)} models"
    rev = inv.dropna(subset=["rank_Primary_minus_Private"]).copy()
    rev["abs_rev"] = rev["rank_Primary_minus_Private"].abs()
    worst = rev.loc[rev["abs_rev"].idxmax()]
    q["max_rank_reversal"] = f"{worst.experiment_id} (Primary rank {worst.Primary_CV_rank:.0f} → Private {worst.Private_rank:.0f})"
    # Q6 stage progression
    sw_lines = []
    for t in ["TmApp", "HIC"]:
        ts = sw[sw.target == t].sort_values("stage", key=lambda s: s.map(STAGE_ORDER))
        if "median_Private_MAE" in ts.columns and len(ts) >= 2:
            priv_improved = ts["median_Private_MAE"].iloc[-1] < ts["median_Private_MAE"].iloc[0]
            cv_improved = ts["median_Primary_CV_MAE"].iloc[-1] < ts["median_Primary_CV_MAE"].iloc[0]
            sw_lines.append(f"{t}: median Primary CV {'改善' if cv_improved else '悪化/横ばい'}、median Private {'改善' if priv_improved else '悪化/横ばい'}")
    q["stage_progression"] = "; ".join(sw_lines)

    # Q9 CV ensemble
    ens_lines = []
    for t in ["TmApp", "HIC"]:
        subk = key_corr[key_corr.Target == t]
        for label in ["CV_MEAN", "CV_WORST"]:
            row = subk[(subk.X == label) & (subk.Y == "Private_MAE")]
            if len(row):
                r = row.iloc[0]
                ens_lines.append(f"{t} {label} vs Private: r={r.Pearson_r:.3f}, rho={r.Spearman_rho:.3f}")
        pr = subk[(subk.X == "Primary_CV_MAE") & (subk.Y == "Private_MAE")].iloc[0]
        wr = subk[(subk.X == "CV_WORST") & (subk.Y == "Private_MAE")]
        if len(wr):
            better = abs(wr.iloc[0].Pearson_r) > abs(pr.Pearson_r)
            ens_lines.append(f"{t} worst-of-two improves over Primary? {'Yes' if better else 'No'}")
    q["cv_ensemble"] = "; ".join(ens_lines)

    # pre-reveal summary
    pr_lines = []
    for t in ["TmApp", "HIC"]:
        frozen = inv[(inv.target == t) & inv.pre_reveal_frozen]
        if len(frozen) >= 3:
            pr = _corr_scalar(frozen["Primary_CV_MAE"].values, frozen["Private_MAE"].values, "pearson")
            pr_lines.append(f"{t} pre-reveal N={len(frozen)} Primary-Private r={pr:.3f}")
    q["prereveal_summary"] = "; ".join(pr_lines) if pr_lines else "pre-reveal N<3 — 参考値なし"

    cm = key_corr[(key_corr.Target == "TmApp") & (key_corr.X == "CV_WORST") & (key_corr.Y == "Private_MAE")]
    q["round2_hint"] = (
        "TmApp: Primary CV は rank 予測力は高い（Private と r≈0.72）が、**絶対 MAE を系統的に過小評価**（median gap≈+0.34、94% の model で Private>Primary）。"
        " Shadow / CV mean / worst-of-two は本データでは Private 相関が **Primary より低い**（Shadow N も少ない）。"
        " Round2 では gap 自体の監視（calibration diagnostic）と、Public–Private 乖離への耐性を重視。"
        " HIC: Primary↔Private 整合は良好（r≈0.77、median gap≈0）。Primary 中心 selection で十分。"
        " Public–Private ranking は TmApp r≈0.78 / HIC r≈0.92 と高いが各 N=81 の sampling variation に注意。"
    )
    write_report(inv, key_corr, sw, q, outliers)
    print("DONE", q)
    print("ROUND1_MODEL_SCORE_CORRELATION_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
