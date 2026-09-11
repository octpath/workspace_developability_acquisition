#!/usr/bin/env python3
"""Shared helpers for developability_drilldown Phase 1."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BUNDLE = REPO / "top_models_feature_bundle"
ORG = REPO / "organizer_extension"

BLOCK_FILE = {
    "AROMATIC_TOPO": "data/aromatic_topo.parquet",
    "AbLang2_HL_paired": "data/ablang2.parquet",
    "AbLingua_CDR3": "data/ablingua_cdr3.parquet",
    "AbLingua_HL_mean": "data/ablingua_global.parquet",
    "BIOEMU_NEW_PAIRWISE": "data/bioemu_new_pairwise.parquet",
    "CONTINUOUS_SURFACE": "data/continuous_surface.parquet",
    "ESM2_H": "data/esm2_heavy.parquet",
    "HYDRO_FIELD": "data/hydro_field.parquet",
    "M1_PROTEINMPNN": "data/proteinmpnn.parquet",
    "SEQ_ALL": "data/seq_all.parquet",
    "SEQ_BASIC": "data/seq_basic.parquet",
    "TITRATION_SHAPE": "data/titration_shape.parquet",
}

# Canonical friendly IDs for FULL Linear Top-6
LIN_TOP6_MAP: dict[str, str] = {
    "TM_PARENT_ABLINGUA_CDR3__RIDGE": "LIN_TM_ABLINGUA_CDR3_RIDGE",
    "TM_PARENT_ABLINGUA_GLOBAL__RIDGE": "LIN_TM_ABLINGUA_GLOBAL_RIDGE",
    "TM_BASE_BIOEMU_MPNN__RIDGE": "LIN_TM_BIOEMU_MPNN_RIDGE",
    "HIC_HYDRO_TITRATION__LASSO": "LIN_HIC_HYDRO_TITRATION_LASSO",
    "HIC_ARO_CONTINUOUS_SURFACE__LASSO": "LIN_HIC_CONTINUOUS_SURFACE_LASSO",
    "HIC_ESM2_SEQ_AROMATIC__LASSO": "LIN_HIC_ESM2_SEQ_AROMATIC_LASSO",
}

XGB_MAP: dict[str, str] = {
    "XGB__TM_BASE_BIOEMU_MPNN__RIDGE": "XGB_TM_BIOEMU_MPNN",
    "XGB__TM_PARENT_ABLINGUA_GLOBAL__RIDGE": "XGB_TM_ABLINGUA_GLOBAL",
    "XGB__TM_PARENT_ABLINGUA_CDR3__RIDGE": "XGB_TM_ABLINGUA_CDR3",
    "XGB__HIC_ARO_CONTINUOUS_SURFACE__LASSO": "XGB_HIC_CONTINUOUS_SURFACE",
    "XGB__HIC_HYDRO_TITRATION__LASSO": "XGB_HIC_HYDRO_TITRATION",
    "XGB__HIC_ESM2_SEQ_AROMATIC__LASSO": "XGB_HIC_ESM2_SEQ_AROMATIC",
}

# experiment_id -> source recipe_id for XGB (same freeze recipes as Linear)
XGB_RECIPE: dict[str, str] = {
    "XGB_TM_BIOEMU_MPNN": "TM_BASE_BIOEMU_MPNN__RIDGE",
    "XGB_TM_ABLINGUA_GLOBAL": "TM_PARENT_ABLINGUA_GLOBAL__RIDGE",
    "XGB_TM_ABLINGUA_CDR3": "TM_PARENT_ABLINGUA_CDR3__RIDGE",
    "XGB_HIC_CONTINUOUS_SURFACE": "HIC_ARO_CONTINUOUS_SURFACE__LASSO",
    "XGB_HIC_HYDRO_TITRATION": "HIC_HYDRO_TITRATION__LASSO",
    "XGB_HIC_ESM2_SEQ_AROMATIC": "HIC_ESM2_SEQ_AROMATIC__LASSO",
}

XGB_SOURCE_BY_EXP = {v: k for k, v in XGB_MAP.items()}

ARTIFACT_STATUSES = {"FULL", "SCORE_ONLY", "PARTIAL", "RECONSTRUCTABLE", "INCONSISTENT"}
FEATURE_SPACE = "RAW_PREPROCESS"
CV_PROTOCOL = "canonical_simple_tvt_primary_shadow"
CV_PROTOCOL_REGISTRY = "canonical_simple_tvt_v1"


def lin_experiment_id(source_model_id: str) -> str:
    if source_model_id in LIN_TOP6_MAP:
        return LIN_TOP6_MAP[source_model_id]
    eid = "LIN_" + source_model_id.replace("__", "_")
    if not re.fullmatch(r"[A-Za-z0-9_]+", eid):
        raise ValueError(f"unsafe experiment_id from {source_model_id}: {eid}")
    return eid


def xgb_experiment_id(source_model_id: str) -> str:
    if source_model_id not in XGB_MAP:
        raise KeyError(source_model_id)
    return XGB_MAP[source_model_id]


def load_dev_test_folds() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(ROOT / "data" / "dev.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    folds = pd.read_csv(ROOT / "data" / "folds.csv")
    for df in (dev, test, folds):
        df["id"] = df["id"].astype(str)
    return dev, test, folds


def load_solution() -> Optional[pd.DataFrame]:
    path = BUNDLE / "solution.csv"
    if not path.exists():
        return None
    sol = pd.read_csv(path)
    sol["id"] = sol["id"].astype(str)
    return sol


def recipe_blocks_from_recipes_csv(recipe_id: str) -> list[str]:
    recipes = pd.read_csv(BUNDLE / "recipes.csv")
    row = recipes[recipes["recipe_id"] == recipe_id]
    if len(row) != 1:
        raise KeyError(recipe_id)
    return str(row.iloc[0]["feature_blocks"]).split("|")


def load_block(name: str, ids: list[str]) -> pd.DataFrame:
    path = BUNDLE / BLOCK_FILE[name]
    raw = pd.read_parquet(path)
    raw["id"] = raw["id"].astype(str)
    idx = raw.set_index("id")
    missing = [i for i in ids if i not in idx.index]
    if missing:
        raise KeyError(f"{name} missing ids: {missing[:5]}")
    out = idx.loc[ids].copy()
    out.index.name = "id"
    return out.reset_index()


def concat_recipe_features(blocks: list[str], ids: list[str]) -> pd.DataFrame:
    parts = []
    base = None
    for b in blocks:
        df = load_block(b, ids)
        if base is None:
            base = df[["id"]].copy()
        feat = df.drop(columns=["id"])
        # keep column names as-is; fail on collision across blocks
        overlap = set(feat.columns) & set(c for p in parts for c in p.columns)
        if overlap:
            raise ValueError(f"feature column collision involving {b}: {sorted(overlap)[:10]}")
        parts.append(feat)
    out = pd.concat([base] + parts, axis=1)
    return out


def add_split_column(df: pd.DataFrame, dev_ids: Iterable[str], test_ids: Iterable[str]) -> pd.DataFrame:
    dev_set, test_set = set(dev_ids), set(test_ids)
    splits = []
    for i in df["id"].astype(str):
        if i in dev_set:
            splits.append("dev")
        elif i in test_set:
            splits.append("test")
        else:
            raise ValueError(f"id not in dev/test: {i}")
    out = df.copy()
    # place split after id
    cols = [c for c in out.columns if c != "id"]
    out.insert(1, "split", splits)
    return out[["id", "split"] + cols]


def feature_column_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("id", "split")]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_content_sha256(df: pd.DataFrame) -> str:
    """Canonical content hash independent of parquet container metadata."""
    ids = df["id"].astype(str).tolist()
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    feat_cols = feature_column_names(df)
    # stable column order as stored
    h = hashlib.sha256()
    h.update(("|".join(feat_cols)).encode())
    h.update(b"\n")
    for i in order:
        h.update(ids[i].encode())
        h.update(b"\0")
        row = df.iloc[i]
        for c in feat_cols:
            v = row[c]
            if pd.isna(v):
                h.update(b"NaN")
            else:
                h.update(np.format_float_scientific(float(v), unique=True, trim="k").encode())
            h.update(b",")
        h.update(b"\n")
    return h.hexdigest()


def feature_recipe_hash(blocks: list[str], feature_cols: list[str]) -> str:
    payload = {
        "blocks": list(blocks),
        "n_features": len(feature_cols),
        "columns": list(feature_cols),
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def normalize_prediction_csv(
    src: Path,
    target: str,
    expected_ids: list[str],
    dest: Path,
) -> pd.DataFrame:
    df = pd.read_csv(src)
    df["id"] = df["id"].astype(str)
    if "prediction" in df.columns:
        pred = df["prediction"]
    elif target in df.columns:
        pred = df[target]
    else:
        cols = [c for c in df.columns if c != "id"]
        if len(cols) != 1:
            raise ValueError(f"cannot infer prediction column in {src}: {df.columns.tolist()}")
        pred = df[cols[0]]
    out = pd.DataFrame({"id": df["id"], target: pred.astype(float)})
    # align to expected order
    out = out.set_index("id").reindex(expected_ids).reset_index()
    if out[target].isna().any():
        raise ValueError(f"missing predictions after align for {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def derived_scores(primary: float, shadow: float, public: Optional[float], private: Optional[float]):
    cv_mean = (primary + shadow) / 2.0
    cv_worst = max(primary, shadow)
    if public is not None and private is not None and not (pd.isna(public) or pd.isna(private)):
        delta = float(public) - float(private)
        gap = abs(delta)
        overall = (float(public) + float(private)) / 2.0
    else:
        delta = gap = overall = float("nan")
    return cv_mean, cv_worst, overall, delta, gap


EXPERIMENTS_COLUMNS = [
    "experiment_code",
    "legacy_experiment_code",
    "experiment_id",
    "target",
    "family",
    "model_type",
    "source_model_id",
    "feature_set_id",
    "source_recipe_id",
    "cv_primary_mae",
    "cv_shadow_mae",
    "cv_mean_mae",
    "cv_worst_mae",
    "public_mae",
    "private_mae",
    "test_overall_mae",
    "public_private_delta",
    "public_private_gap",
    "cv_protocol",
    "selection_policy_at_creation",
    "current_evaluation_mode",
    "artifact_status",
    "source_reproducible",
    "drilldown_reproducible",
    "reproduction_status",
    "config_path",
    "feature_path",
    "oof_primary_path",
    "oof_shadow_path",
    "test_prediction_path",
    "n_features",
    "feature_space",
    "feature_sha256",
    "feature_content_sha256",
    "feature_recipe_hash",
    "linear_alpha",
    "xgb_preset",
    "xgb_final_n_estimators",
    "score_source",
    "prediction_source",
    "feature_source",
    "license_status",
    "license_reference",
    "ensemble_type",
    "member_experiment_codes",
    "notes",
    # Phase 2A Transformer columns (empty for Linear/XGB)
    "transformer_type",
    "plm_source",
    "annotation_mode",
    "chain_mode",
    "pooling_mode",
    "representation_status",
    "input_space",
    "input_asset_ref",
]

LICENSE_STATUSES = {"OK", "REVIEW", "RESTRICTED", "UNKNOWN"}
SELECTION_POLICIES = {
    "CV_ONLY",
    "CV_SELECTED_POSTCOMP_EVALUATED",
    "POSTCOMP_EXPLORATORY",
    "UNKNOWN",
}
SOURCE_REPRO = {"YES", "NO", "UNKNOWN"}
DRILLDOWN_REPRO = {"YES", "PARTIAL", "NO"}
CODE_RE = re.compile(r"^EXP-[THM][0-9]{3,}$")
LEGACY_CODE_RE = re.compile(r"^EXP[0-9]{3,}$")

# Expected registry sizes after T130–T141 encoder-sharing ablation (+12 Tm)
N_EXPERIMENTS_TOTAL = 242  # 234 prior + 8 HIC STATIC_SAP_KD
N_LEGACY_MAP = 48  # Linear/XGB only; Transformer + classical-refinement rows have empty legacy
N_FULL_LINEAR_XGB = 86  # 12 historical FULL + 34 reconstructed + 40 classical refinement
N_TRANSFORMER = 154  # 146 prior + 8 HIC STATIC_SAP_KD
N_HISTORICAL_TRANSFORMER = 29  # Phase2A backfill only (TRANSFORMER_BACKFILL_AUDIT)
N_CLASSICAL_REFINEMENT = 40
N_LINEAR = 74
N_XGBOOST = 14
PRESERVATION_SNAPSHOT_77 = ROOT / "results" / "_preservation_snapshot_77_classical.csv"


def is_classical_refinement_code(code: str) -> bool:
    s = str(code)
    if s.startswith("EXP-T"):
        return 45 <= int(s.split("-T")[1]) <= 64
    if s.startswith("EXP-H"):
        return 34 <= int(s.split("-H")[1]) <= 53
    return False


def is_historical_transformer_code(code: str) -> bool:
    """Phase2A backfill Transformers (no shareable fixed-branch parquet contract)."""
    s = str(code)
    if s.startswith("EXP-T"):
        return 26 <= int(s.split("-T")[1]) <= 44
    if s.startswith("EXP-H"):
        return 24 <= int(s.split("-H")[1]) <= 33
    return False

REPRODUCIBILITY_STATUSES = {"REPRODUCED", "RESULT_VERIFIED", "UNVERIFIED_HISTORICAL"}
SHAREABILITY_STATUSES = {"SHAREABLE_COMPLETE", "SHAREABLE_PARTIAL", "HISTORICAL_ONLY"}
CANONICAL_ELIGIBLE = {"YES", "NO"}

# Ensure experiment_code is first identity columns when rewriting tables
IDENTITY_PREFIX = [
    "experiment_code",
    "legacy_experiment_code",
    "experiment_id",
]
