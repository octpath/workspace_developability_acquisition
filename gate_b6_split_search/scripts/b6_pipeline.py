#!/usr/bin/env python3
"""Gate B6 common Public/Private production split search — executable pipeline."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import kendalltau, spearmanr, wasserstein_distance

ROOT = Path("/workspace_developability_acquisition")
B6 = ROOT / "gate_b6_split_search"
CFG, CAND, MET, REP, FRZ, CACHE = [B6 / x for x in
    ("config", "candidates", "metrics", "reports", "frozen", "cache")]
PRED = ROOT / "gate_b5_ceiling/predictions"
ORG = ROOT / "gate_b3/frozen/organizer"
OUTER_PATH = ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json"

for d in (CFG, CAND, MET, REP, FRZ, CACHE, B6 / "plots", B6 / "logs"):
    d.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def write_json(path: Path, obj) -> str:
    b = json.dumps(obj, indent=2, sort_keys=True, default=float).encode()
    path.write_bytes(b)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_bytes(b) + "\n")
    return sha256_bytes(b)


def load_test_pred(target: str, tag: str) -> np.ndarray:
    return np.concatenate([
        np.load(PRED / "final" / f"{target}__{tag}__public.npy"),
        np.load(PRED / "final" / f"{target}__{tag}__private.npy"),
    ])


def cv_stats(target: str, tag: str, y_train: np.ndarray) -> dict:
    path = PRED / "oof" / f"{target}__{tag}__meanOOF.npy"
    oof = np.load(path)
    mae = float(np.mean(np.abs(y_train - oof)))
    if np.std(oof) < 1e-12 or np.std(y_train) < 1e-12:
        pear = spr = float("nan")
    else:
        pear = float(np.corrcoef(y_train, oof)[0, 1])
        spr = float(spearmanr(y_train, oof).correlation)
    return {"mae": mae, "pearson": pear, "spearman": spr, "oof_sha256": sha256_file(path)}


def pred_corrs(target: str, tags: list[str]) -> dict:
    mats = {t: load_test_pred(target, t) for t in tags}
    out = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            sa, sb = mats[a], mats[b]
            out[f"{a}__vs__{b}"] = (
                None if (np.std(sa) < 1e-12 or np.std(sb) < 1e-12)
                else float(np.corrcoef(sa, sb)[0, 1])
            )
    return out


def enrich_models(specs, target, y_train):
    rows = []
    for m in specs:
        tag = m["tag"]
        pub = PRED / "final" / f"{target}__{tag}__public.npy"
        priv = PRED / "final" / f"{target}__{tag}__private.npy"
        assert pub.exists() and priv.exists(), tag
        cv = cv_stats(target, tag, y_train)
        rows.append({
            **m,
            "target": target,
            "prediction_file_hash": {
                "public": sha256_file(pub),
                "private": sha256_file(priv),
                "oof": cv["oof_sha256"],
            },
            "CV_metrics": {"mae": cv["mae"], "pearson": cv["pearson"], "spearman": cv["spearman"]},
        })
    return rows


def freeze_p0(role: pd.DataFrame, train_ids: list[str]):
    y_tm = role.loc[train_ids, "TmApp"].values.astype(float)
    y_hic = role.loc[train_ids, "HIC"].values.astype(float)
    tm_specs = [
        {"model_id": "TmApp__CONST_MEDIAN", "tag": "CONST_MEDIAN", "model_family": "CONST"},
        {"model_id": "TmApp__SEQ_SIMPLE_Ridge", "tag": "SEQ_SIMPLE_Ridge", "model_family": "SEQ_SIMPLE"},
        {"model_id": "TmApp__BIO_Ridge", "tag": "BIO_Ridge", "model_family": "BIO"},
        {"model_id": "TmApp__PLM_ABLANG2_PCA32_SVR", "tag": "PLM_ABLANG2_PCA32_SVR", "model_family": "ABLANG2_NONLINEAR"},
        {"model_id": "TmApp__NESTED_STACK_MEAN", "tag": "NESTED_STACK_MEAN", "model_family": "NESTED_ENSEMBLE"},
    ]
    hic_specs = [
        {"model_id": "HIC__CONST_MEDIAN", "tag": "CONST_MEDIAN", "model_family": "CONST"},
        {"model_id": "HIC__SEQ_SIMPLE_Ridge", "tag": "SEQ_SIMPLE_Ridge", "model_family": "SEQ_SIMPLE"},
        {"model_id": "HIC__PLM_ESM2_PCA64_SVR", "tag": "PLM_ESM2_PCA64_SVR", "model_family": "ESM2_PLM"},
        {"model_id": "HIC__ESMFN_STRUCTURE_ElasticNet", "tag": "ESMFN_STRUCTURE_ElasticNet", "model_family": "ESMFOLD_STRUCTURE"},
        {"model_id": "HIC__FUSION_ESM2_ESMFN_ElasticNet", "tag": "FUSION_ESM2_ESMFN_ElasticNet", "model_family": "PLM_STRUCTURE_FUSION"},
        {"model_id": "HIC__NESTED_STACK_NNLS", "tag": "NESTED_STACK_NNLS", "model_family": "NESTED_ENSEMBLE"},
    ]
    model_set = {
        "gate": "B6",
        "purpose": "Frozen organizer models for common Public/Private split evaluation",
        "source": "gate_b5_ceiling/predictions",
        "TmApp": enrich_models(tm_specs, "TmApp", y_tm),
        "HIC": enrich_models(hic_specs, "HIC", y_hic),
        "excluded": [
            {"model_id": "TmApp__FUSION_ABLANG2_BIO_ElasticNet", "reason": "Test-pred Pearson~0.979 with NESTED_STACK_MEAN"},
            {"model_id": "POOL/FT variants", "reason": "B5 not valid / unstable"},
            {"model_id": "TmApp germline/IMGT/linear AbLang2", "reason": "no frozen final Test preds"},
        ],
        "prediction_correlations_Test": {
            "TmApp": pred_corrs("TmApp", [m["tag"] for m in tm_specs]),
            "HIC": pred_corrs("HIC", [m["tag"] for m in hic_specs]),
        },
        "diversity_rule": "Near-duplicate Test predictions excluded from primary eval set",
    }
    model_hash = write_json(CFG / "FROZEN_SPLIT_EVALUATION_MODEL_SET.json", model_set)
    protocol = {
        "gate": "B6",
        "title": "Common Public/Private production split search — pre-registered protocol",
        "frozen_before_model_scoring": True,
        "candidate_generation": {
            "seed": 20260830,
            "n_candidates": 20000,
            "public_size": 81,
            "private_size": 81,
            "atomic_sequence_groups": True,
            "method": "Random multi-group assignment + exact singleton fill to Public=81; dedupe by Public hash",
            "include_current_baseline": True,
            "baseline_id": "CURRENT_BASELINE_SPLIT",
        },
        "pre_model_balance": {
            "top_fraction": 0.10,
            "max_survivors": 2000,
            "filter_rule": "Keep top 10% by ascending balance_score; always retain CURRENT_BASELINE_SPLIT",
        },
        "TmApp_evaluation": {
            "model_set_sha256": model_hash,
            "require_positive_mae_transfers": True,
            "shortlist_size": 8,
            "shortlist_jaccard_cap": 0.85,
            "bootstrap_screen_replicates": 1000,
            "bootstrap_screen_top_n": 80,
            "bootstrap_final_replicates": 5000,
            "selection_hierarchy": [
                "pre_model_balance_filter",
                "positive_MAE_CV_Public_and_Public_Private_and_CV_Private",
                "prefer_higher_min_then_mean_MAE_triple",
                "prefer_Pearson_transfers",
                "prefer_worst_LOFO_family_and_bootstrap_stability",
                "Top3_overlap_tiebreak",
            ],
        },
        "HIC_safety": {
            "evaluate_only_on": "frozen TmApp shortlist",
            "floors": {
                "mae_model_rank_Public_to_Private_spearman": 0.50,
                "pearson_model_rank_Public_to_Private_spearman": 0.50,
                "mae_model_rank_CV_to_Private_spearman": 0.0,
            },
            "classification": {
                "HIC_EXCELLENT": ">=0.75",
                "HIC_GOOD": "[0.65,0.75)",
                "HIC_ACCEPTABLE": "[0.50,0.65)",
                "HIC_CONCERNING": "<0.50 or fails floors",
            },
            "role": "safety constraint / tiebreak only",
        },
        "final_decision_rule": [
            "Discard shortlist failing HIC floors",
            "Among remaining choose best robust TmApp",
            "HIC tiebreak only if nearly equivalent",
            "If none: NO_ACCEPTABLE_COMMON_SPLIT_FOUND",
            "Do not regenerate after HIC",
        ],
        "anti_overfit": {"lofo_selection_replay": True, "label_if_collapses": "MODEL_SET_OVERFIT_RISK"},
        "model_set_sha256": model_hash,
        "common_mask_requirement": "HIC and TmApp share identical Public/Private IDs",
        "train_test_boundary": "FROZEN Train=162 / Test=162",
    }
    proto_hash = write_json(CFG / "SPLIT_SELECTION_PROTOCOL.json", protocol)
    print("P0 frozen model_set", model_hash[:16], "protocol", proto_hash[:16])
    return model_set, protocol, model_hash, proto_hash
