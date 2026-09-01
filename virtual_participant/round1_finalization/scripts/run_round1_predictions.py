#!/usr/bin/env python3
"""Round 1 — full Dev fit + Test prediction (pre-reveal)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace_developability_acquisition")
FIN = ROOT / "virtual_participant/round1_finalization"
CACHE = FIN / "cache"
PRED = FIN / "predictions"
SUB = FIN / "submissions"
DEV = ROOT / "competition/data/distribution/dev.csv"
TEST = ROOT / "competition/data/distribution/test_features.csv"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
BASE_CFG = ROOT / "virtual_participant/stage5_integration/stage5_base_models.json"
LOCK = FIN / "round1_final_model_specs.json"

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage5_integration/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage4_advanced_structure/scripts"))

from run_stage4_models import FAMILIES, cols_for  # noqa: E402
from run_stage5 import (  # noqa: E402
    classify_struct_col,
    fold_prepare,
    make_model,
    mae,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def load_combined_store():
    from run_stage1 import build_all_feature_tables, make_xy

    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    ann_dev = pd.read_csv(ROOT / "competition/data/distribution/dev_annotations.csv")
    ann_test = pd.read_csv(ROOT / "competition/data/distribution/test_annotations.csv")
    reg_dev = pd.read_csv(ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv")
    reg_test = pd.read_csv(CACHE / "anarci_imgt_regions_test.csv")

    all_df = pd.concat([dev, test], ignore_index=True)
    ann = pd.concat([ann_dev, ann_test], ignore_index=True)
    regions = pd.concat([reg_dev, reg_test], ignore_index=True)
    ids = list(all_df["id"])
    n_dev = len(dev)

    tables = build_all_feature_tables(all_df, ann, regions)
    classical = {}
    for fam in ["SEQ_BASIC", "SEQ_ALL", "SEQ_PLUS_ANTIBODY"]:
        Xn, Xc = make_xy(tables, fam)
        classical[fam] = {"num": Xn, "cat": Xc}

    emb = np.load(CACHE / "round1_embeddings.npz", allow_pickle=True)
    id_index = {x: i for i, x in enumerate(list(emb["ids"]))}
    idx = np.array([id_index[i] for i in ids])
    plm = {k: emb[k][idx].astype(np.float32) for k in emb.files if k != "ids"}

    s3 = pd.read_csv(CACHE / "features_ESMFold_combined.csv").set_index("antibody_id").loc[ids]
    struct_cols = {f: [] for f in ["RASA", "SURFACE_CHEM", "SURFACE_ALL", "PACKING", "INTERFACE", "PATCH", "GLOBAL"]}
    for c in s3.columns:
        if c == "antibody_id":
            continue
        fam = classify_struct_col(c)
        if fam in struct_cols:
            struct_cols[fam].append(c)
    struct_cols["SURFACE_ALL"] = sorted(set(
        struct_cols["RASA"] + struct_cols["SURFACE_CHEM"] + struct_cols.get("PATCH", [])
    ))
    struct = {f: s3[cols].to_numpy(float) if cols else None for f, cols in struct_cols.items() if f != "SASA"}

    s3_rasa = s3[[c for c in s3.columns if "rasa" in c.lower()]].to_numpy(float)
    s3_surf = s3[[c for c in s3.columns if any(
        x in c.lower() for x in ["sasa", "rasa", "patch", "exposed", "hydrophobic", "aromatic", "charge"]
    )]].to_numpy(float)

    s4 = pd.read_csv(CACHE / "stage4_all_features_combined.csv").set_index("antibody_id").loc[ids]
    adv = {}
    for fam in list(FAMILIES) + ["ADV_HIC_ALL", "ADV_TMAPP_ALL"]:
        if fam in ("ADV_HIC_ALL", "ADV_TMAPP_ALL"):
            if fam == "ADV_HIC_ALL":
                names = ["ADV_PROPKA", "ADV_PQR_CHARGE", "ADV_ELECTROSTATICS", "ADV_SURFACE_PATCH", "ADV_INVFOLD"]
            else:
                names = ["ADV_INTERACTIONS", "ADV_CAVITY", "ADV_UNSAT_POLAR", "ADV_INVFOLD"]
            mats = []
            for n in names:
                cols = cols_for(s4.reset_index(), n)
                if cols:
                    mats.append(s4[cols].to_numpy(float))
            adv[fam] = np.hstack(mats) if mats else None
        else:
            cols = cols_for(s4.reset_index(), fam)
            adv[fam] = s4[cols].to_numpy(float) if cols else None

    return {
        "ids": ids,
        "n_dev": n_dev,
        "classical": classical,
        "plm": plm,
        "struct": struct,
        "s3_rasa": s3_rasa,
        "s3_surf": s3_surf,
        "adv": adv,
    }


def cfg_lookup(base_json: dict) -> dict[str, dict]:
    out = {}
    for target in ["TmApp", "HIC"]:
        for c in base_json[target]:
            out[c["experiment_id"]] = c
    return out


def cv_oof_dev(cfg, store, y_dev, folds, dev_idx):
    """Primary CV OOF on Dev rows only; feature store includes Test rows."""
    oof = np.zeros(len(y_dev))
    n = len(store["ids"])
    for f in range(int(folds.max()) + 1):
        tr_local = np.where(folds != f)[0]
        va_local = np.where(folds == f)[0]
        tr_idx = dev_idx[tr_local]
        va_idx = dev_idx[va_local]
        tr = np.zeros(n, dtype=bool)
        va = np.zeros(n, dtype=bool)
        tr[tr_idx] = True
        va[va_idx] = True
        Xtr, Xva, _ = fold_prepare(cfg, store, tr, va)
        model = make_model(cfg["model_kind"], cfg["params"])
        model.fit(Xtr, y_dev[tr_local])
        oof[va_local] = model.predict(Xva)
    return oof


def fit_predict_test(cfg, store, y_dev, dev_idx, test_idx):
    n = len(store["ids"])
    tr = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[dev_idx] = True
    te[test_idx] = True
    Xtr, Xte, _ = fold_prepare(cfg, store, tr, te)
    model = make_model(cfg["model_kind"], cfg["params"])
    model.fit(Xtr, y_dev)
    return model.predict(Xte)


def predict_model(eid, cfg, store, y, folds, dev_idx, test_idx):
    oof = cv_oof_dev(cfg, store, y, folds, dev_idx)
    test_pred = fit_predict_test(cfg, store, y, dev_idx, test_idx)
    return oof, test_pred


def stack_predict(base_ids, meta_alpha, store, y, folds, dev_idx, test_idx, lookup):
    cfgs = [lookup[b] for b in base_ids]
    oof_cols = []
    test_cols = []
    for c in cfgs:
        oof, tp = predict_model(c["experiment_id"], c, store, y, folds, dev_idx, test_idx)
        oof_cols.append(oof)
        test_cols.append(tp)
    X_meta = np.column_stack(oof_cols)
    meta = Ridge(alpha=meta_alpha, random_state=0)
    meta.fit(X_meta, y)
    X_test = np.column_stack(test_cols)
    return meta.predict(X_test), oof_cols, test_cols


def blend_predict(base_ids, store, y, folds, dev_idx, test_idx, lookup):
    preds = []
    for bid in base_ids:
        c = lookup[bid]
        _, tp = predict_model(c["experiment_id"], c, store, y, folds, dev_idx, test_idx)
        preds.append(tp)
    return np.mean(np.column_stack(preds), axis=1)


def main():
    PRED.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    specs = json.loads(LOCK.read_text())
    lookup = cfg_lookup(json.loads(BASE_CFG.read_text()))

    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    y_tm = dev["TmApp"].to_numpy(float)
    y_hic = dev["HIC"].to_numpy(float)
    folds = pd.read_csv(CV_P)["fold"].to_numpy()

    store = load_combined_store()
    n_dev = store["n_dev"]
    dev_idx = np.arange(n_dev)
    test_idx = np.arange(n_dev, n_dev + len(test))

    all_preds = {}

    # PRIMARY TmApp stack
    tm_spec = specs["primary"]["TmApp"]
    tm_pred, _, _ = stack_predict(
        tm_spec["base_model_ids"],
        tm_spec["meta_alpha"],
        store, y_tm, folds, dev_idx, test_idx, lookup,
    )
    all_preds["TmApp_PRIMARY"] = pd.DataFrame({"id": test["id"], "prediction": tm_pred})

    # PRIMARY HIC blend
    hic_spec = specs["primary"]["HIC"]
    hic_pred = blend_predict(
        hic_spec["base_model_ids"],
        store, y_hic, folds, dev_idx, test_idx, lookup,
    )
    all_preds["HIC_PRIMARY"] = pd.DataFrame({"id": test["id"], "prediction": hic_pred})

    # Secondaries
    for sec in specs.get("secondary", []):
        target = sec["target"]
        y = y_tm if target == "TmApp" else y_hic
        if sec.get("recipe") == "stack":
            pred, _, _ = stack_predict(
                sec["base_model_ids"], sec["meta_alpha"],
                store, y, folds, dev_idx, test_idx, lookup,
            )
        elif sec.get("recipe") == "blend":
            pred = blend_predict(sec["base_model_ids"], store, y, folds, dev_idx, test_idx, lookup)
        else:
            c = lookup[sec["experiment_id"]]
            _, pred = predict_model(sec["experiment_id"], c, store, y, folds, dev_idx, test_idx)
        all_preds[sec["output_key"]] = pd.DataFrame({"id": test["id"], "prediction": pred})

    # Save individual prediction files
    for key, df in all_preds.items():
        path = PRED / f"{key}_predictions.csv"
        df.to_csv(path, index=False)

    # PRIMARY submission
    sub = test[["id"]].copy()
    sub["TmApp"] = all_preds["TmApp_PRIMARY"]["prediction"].values
    sub["HIC"] = all_preds["HIC_PRIMARY"]["prediction"].values
    sub_path = SUB / "ROUND1_PRIMARY_submission.csv"
    sub.to_csv(sub_path, index=False)

    # Diagnostic submissions (one target swapped)
    for sec in specs.get("secondary", []):
        if sec.get("diagnostic_submission"):
            dsub = sub.copy()
            col = sec["target"]
            dsub[col] = all_preds[sec["output_key"]]["prediction"].values
            dsub.to_csv(SUB / sec["diagnostic_submission"], index=False)

    print("submission", sub_path)
    print("TmApp range", sub.TmApp.min(), sub.TmApp.max())
    print("HIC range", sub.HIC.min(), sub.HIC.max())
    print("ROUND1_PREDICTIONS_DONE")


if __name__ == "__main__":
    main()
