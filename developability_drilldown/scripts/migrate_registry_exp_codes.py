#!/usr/bin/env python3
"""One-shot registry migration: permanent EXP codes + path/schema cleanup.

Authority for EXP001–EXP048: committed experiments.csv row order at migration time.
Never renumber existing codes after this script issues EXPERIMENT_CODES.csv.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    BUNDLE,
    BLOCK_FILE,
    ROOT,
    feature_column_names,
    feature_content_sha256,
    file_sha256,
    recipe_blocks_from_recipes_csv,
)

SCORE_COLS = [
    "cv_primary_mae",
    "cv_shadow_mae",
    "cv_mean_mae",
    "cv_worst_mae",
    "public_mae",
    "private_mae",
    "test_overall_mae",
]

# Verified FULL shared raw matrices → permanent feature_set_id
# Keys are feature_content_sha256 from Phase 1 FULL exports.
FEATURE_SET_BY_HASH: dict[str, str] = {
    "cf1f16b3788022950073bc7e66f5dc4fea018a8e65915a29a229247690124b37": "FS_TM_ABLINGUA_CDR3",
    "04d2022102f5d7045ff3939fc45a15b56f9c26fc55b5ddb71d1af5ff83889c59": "FS_TM_ABLINGUA_GLOBAL",
    "8e92cba7d639421e2952570fd24462acbdb576950678b5f4879c9f18e3ee25cc": "FS_TM_BIOEMU_MPNN",
    "551237ca9e3a94713d3ecb205fa2b57b2372b54a42bb1769b6d191f942ba6f83": "FS_HIC_HYDRO_TITRATION",
    "1807d98170a4e31f4760735417d7009742b5de75955f8c642ec86cafe2f23797": "FS_HIC_CONTINUOUS_SURFACE",
    "dcda2f3907583d285321d67a48536a9d47cbbf894a689e9508e7ff5fdaef73ba": "FS_HIC_ESM2_SEQ_AROMATIC",
}

# Descriptive FS ids for SCORE_ONLY/RECONSTRUCTABLE when blocks known from registry
# (not used for FULL — hash authority wins)
FS_FROM_SOURCE_BASE: dict[str, str] = {
    "TM_PARENT_ABLINGUA_CDR3": "FS_TM_ABLINGUA_CDR3",
    "TM_PARENT_ABLINGUA_GLOBAL": "FS_TM_ABLINGUA_GLOBAL",
    "TM_BASE_BIOEMU_MPNN": "FS_TM_BIOEMU_MPNN",
    "HIC_HYDRO_TITRATION": "FS_HIC_HYDRO_TITRATION",
    "HIC_ARO_CONTINUOUS_SURFACE": "FS_HIC_CONTINUOUS_SURFACE",
    "HIC_ESM2_SEQ_AROMATIC": "FS_HIC_ESM2_SEQ_AROMATIC",
}

LICENSE_MAP = {
    "OK_COMPETITION_DERIVED": "OK",
    "REVIEW_MODEL_OUTPUT": "REVIEW",
    "SEE_FEATURE_EXTENSION": "REVIEW",
}

NEW_COLUMNS = [
    "experiment_code",
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
]


def source_base(recipe_id: str) -> str:
    if recipe_id.endswith("__RIDGE") or recipe_id.endswith("__LASSO"):
        return recipe_id.rsplit("__", 1)[0]
    return recipe_id


def load_manifest_license() -> dict[str, str]:
    man = pd.read_csv(BUNDLE / "feature_manifest.csv")
    out = {}
    for _, r in man.iterrows():
        out[str(r["block_name"])] = LICENSE_MAP.get(str(r["license_status"]), "UNKNOWN")
    return out


def license_for_blocks(blocks: list[str], man: dict[str, str]) -> tuple[str, str]:
    statuses = []
    for b in blocks:
        statuses.append(man.get(b, "UNKNOWN"))
    if not statuses:
        return "UNKNOWN", "top_models_feature_bundle/feature_manifest.csv"
    if "RESTRICTED" in statuses:
        st = "RESTRICTED"
    elif "REVIEW" in statuses:
        st = "REVIEW"
    elif all(s == "OK" for s in statuses):
        st = "OK"
    else:
        st = "UNKNOWN"
    return st, "top_models_feature_bundle/feature_manifest.csv"


def snapshot_predictions(codes_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Hash prediction contents keyed by experiment_id before rename."""
    snap = {}
    pred_root = ROOT / "experiments" / "predictions"
    for eid, code in codes_map.items():
        d = pred_root / eid
        if not d.exists():
            continue
        entry = {}
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            p = d / name
            if p.exists():
                df = pd.read_csv(p)
                entry[name] = {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "ids": df["id"].astype(str).tolist(),
                    "cols": list(df.columns),
                    "values": df.iloc[:, 1].astype(float).tolist(),
                }
        snap[eid] = entry
    return snap


def main() -> None:
    exp_path = ROOT / "results" / "experiments.csv"
    df = pd.read_csv(exp_path)
    assert len(df) == 48, len(df)
    assert df["experiment_id"].is_unique

    # Freeze score snapshot
    score_snap = df[["experiment_id"] + SCORE_COLS].copy()

    # Issue codes from current row order
    codes_rows = []
    id_to_code: dict[str, str] = {}
    for i, row in df.iterrows():
        code = f"EXP{i+1:03d}"
        eid = str(row["experiment_id"])
        id_to_code[eid] = code
        codes_rows.append(
            {
                "experiment_code": code,
                "experiment_id": eid,
                "source_model_id": row["source_model_id"],
                "issued_at_phase": "PHASE1_BACKFILL",
                "status": "ACTIVE",
                "notes": "Frozen from experiments.csv row order at registry migration",
            }
        )
    codes_df = pd.DataFrame(codes_rows)
    codes_path = ROOT / "results" / "EXPERIMENT_CODES.csv"
    codes_df.to_csv(codes_path, index=False)
    print(f"issued {len(codes_df)} codes -> {codes_path}")

    pred_snap = snapshot_predictions(id_to_code)

    man_lic = load_manifest_license()

    # Migrate FULL artifact paths: rename configs, features, predictions
    cfg_dir = ROOT / "experiments" / "configs"
    feat_dir = ROOT / "experiments" / "features"
    pred_dir = ROOT / "experiments" / "predictions"

    for eid, code in id_to_code.items():
        old_cfg = cfg_dir / f"{eid}.yaml"
        new_cfg = cfg_dir / f"{code}.yaml"
        if old_cfg.exists():
            data = yaml.safe_load(old_cfg.read_text()) or {}
            data["experiment_code"] = code
            data["experiment_id"] = eid
            # rename feature_recipe semantics in config
            if "feature_recipe" in data and "source_recipe_id" not in data:
                data["source_recipe_id"] = data["feature_recipe"]
            old_cfg.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
            old_cfg.rename(new_cfg)

        old_feat = feat_dir / f"{eid}.parquet"
        new_feat = feat_dir / f"{code}.parquet"
        if old_feat.exists():
            old_feat.rename(new_feat)

        old_pred = pred_dir / eid
        new_pred = pred_dir / code
        if old_pred.exists():
            old_pred.rename(new_pred)

    # Build feature sets from FULL hashes
    fs_rows = []
    seen_fs: dict[str, dict] = {}
    for _, row in df[df["artifact_status"] == "FULL"].iterrows():
        h = str(row["feature_content_sha256"])
        fsid = FEATURE_SET_BY_HASH[h]
        recipe = str(row["feature_recipe"])
        code = id_to_code[str(row["experiment_id"])]
        feat_path = feat_dir / f"{code}.parquet"
        blocks = recipe_blocks_from_recipes_csv(recipe)
        if fsid not in seen_fs:
            seen_fs[fsid] = {
                "feature_set_id": fsid,
                "target": row["target"],
                "n_features": int(row["n_features"]),
                "feature_content_sha256": h,
                "feature_recipe_hash": row["feature_recipe_hash"],
                "source_recipe_ids": [recipe],
                "feature_blocks": "|".join(blocks),
                "notes": "Verified via shared RAW_PREPROCESS feature_content_sha256",
            }
        else:
            if recipe not in seen_fs[fsid]["source_recipe_ids"]:
                seen_fs[fsid]["source_recipe_ids"].append(recipe)
            if seen_fs[fsid]["feature_content_sha256"] != h:
                raise SystemExit(f"hash mismatch for {fsid}")

    for fs in seen_fs.values():
        fs["source_recipe_ids"] = "|".join(sorted(fs["source_recipe_ids"]))
        fs_rows.append(fs)
    fs_df = pd.DataFrame(fs_rows).sort_values("feature_set_id")
    fs_path = ROOT / "results" / "FEATURE_SETS.csv"
    fs_df.to_csv(fs_path, index=False)
    print(f"feature sets {len(fs_df)} -> {fs_path}")

    # Rebuild experiments.csv
    new_rows = []
    for _, row in df.iterrows():
        eid = str(row["experiment_id"])
        code = id_to_code[eid]
        recipe = str(row["feature_recipe"])
        status = str(row["artifact_status"])
        h = row.get("feature_content_sha256")
        if status == "FULL" and isinstance(h, str) and h in FEATURE_SET_BY_HASH:
            fsid = FEATURE_SET_BY_HASH[h]
        else:
            base = source_base(recipe)
            fsid = FS_FROM_SOURCE_BASE.get(base, f"FS_{base}")

        # paths
        if status == "FULL":
            cfg_p = f"experiments/configs/{code}.yaml"
            feat_p = f"experiments/features/{code}.parquet"
            oof_p = f"experiments/predictions/{code}/oof_primary.csv"
            oof_s = f"experiments/predictions/{code}/oof_shadow.csv"
            test_p = f"experiments/predictions/{code}/test.csv"
            feat_sha = file_sha256(ROOT / feat_p)
            # content hash must be preserved
            content_sha = str(row["feature_content_sha256"])
            # verify
            content_now = feature_content_sha256(pd.read_parquet(ROOT / feat_p))
            if content_now != content_sha:
                raise SystemExit(f"content hash changed for {code}")
            try:
                blocks = recipe_blocks_from_recipes_csv(recipe)
            except KeyError:
                blocks = []
            lic_st, lic_ref = license_for_blocks(blocks, man_lic)
            source_rep = "YES"
            drill_rep = "YES"
            n_feat = int(row["n_features"])
            fspace = row["feature_space"]
            frhash = row["feature_recipe_hash"]
        else:
            cfg_p = feat_p = oof_p = oof_s = test_p = ""
            feat_sha = content_sha = frhash = ""
            n_feat = row.get("n_features") if pd.notna(row.get("n_features")) else ""
            fspace = row.get("feature_space") if pd.notna(row.get("feature_space")) else ""
            # try blocks from recipes if known
            try:
                blocks = recipe_blocks_from_recipes_csv(recipe)
                lic_st, lic_ref = license_for_blocks(blocks, man_lic)
            except Exception:
                blocks = []
                lic_st, lic_ref = "UNKNOWN", "top_models_feature_bundle/feature_manifest.csv"
            source_rep = "YES" if str(row.get("reproducible")) == "YES" else (
                "YES" if str(row.get("reproduction_status", "")).startswith("REPLAYED") else "UNKNOWN"
            )
            if str(row.get("reproducible")) == "YES" or "REPLAYED" in str(row.get("reproduction_status", "")):
                source_rep = "YES"
            elif str(row.get("reproducible")) == "PARTIAL":
                source_rep = "YES"
            else:
                source_rep = "UNKNOWN"
            drill_rep = "PARTIAL" if status == "RECONSTRUCTABLE" else "NO"

        # Update FULL configs with feature_set_id
        if status == "FULL":
            cfg_file = ROOT / cfg_p
            data = yaml.safe_load(cfg_file.read_text()) or {}
            data["experiment_code"] = code
            data["experiment_id"] = eid
            data["feature_set_id"] = fsid
            data["source_recipe_id"] = recipe
            data.pop("feature_recipe", None)
            cfg_file.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

        nr = {c: "" for c in NEW_COLUMNS}
        nr.update(
            {
                "experiment_code": code,
                "experiment_id": eid,
                "target": row["target"],
                "family": row["family"],
                "model_type": row["model_type"],
                "source_model_id": row["source_model_id"],
                "feature_set_id": fsid,
                "source_recipe_id": recipe,
                "cv_primary_mae": row["cv_primary_mae"],
                "cv_shadow_mae": row["cv_shadow_mae"],
                "cv_mean_mae": row["cv_mean_mae"],
                "cv_worst_mae": row["cv_worst_mae"],
                "public_mae": row["public_mae"] if pd.notna(row["public_mae"]) else "",
                "private_mae": row["private_mae"] if pd.notna(row["private_mae"]) else "",
                "test_overall_mae": row["test_overall_mae"] if pd.notna(row["test_overall_mae"]) else "",
                "public_private_delta": row["public_private_delta"] if pd.notna(row["public_private_delta"]) else "",
                "public_private_gap": row["public_private_gap"] if pd.notna(row["public_private_gap"]) else "",
                "cv_protocol": row["cv_protocol"],
                "selection_policy_at_creation": row["selection_policy_at_creation"],
                "current_evaluation_mode": row["current_evaluation_mode"],
                "artifact_status": status,
                "source_reproducible": source_rep,
                "drilldown_reproducible": drill_rep,
                "reproduction_status": row.get("reproduction_status", ""),
                "config_path": cfg_p,
                "feature_path": feat_p,
                "oof_primary_path": oof_p,
                "oof_shadow_path": oof_s,
                "test_prediction_path": test_p,
                "n_features": n_feat,
                "feature_space": fspace,
                "feature_sha256": feat_sha,
                "feature_content_sha256": content_sha,
                "feature_recipe_hash": frhash,
                "linear_alpha": row["linear_alpha"] if pd.notna(row["linear_alpha"]) else "",
                "xgb_preset": row["xgb_preset"] if pd.notna(row["xgb_preset"]) else "",
                "xgb_final_n_estimators": row["xgb_final_n_estimators"]
                if pd.notna(row["xgb_final_n_estimators"])
                else "",
                "score_source": row["score_source"],
                "prediction_source": row["prediction_source"] if pd.notna(row["prediction_source"]) else "",
                "feature_source": row["feature_source"] if pd.notna(row["feature_source"]) else "",
                "license_status": lic_st,
                "license_reference": lic_ref,
                "ensemble_type": "",
                "member_experiment_codes": "",
                "notes": row["notes"] if pd.notna(row["notes"]) else "",
            }
        )
        new_rows.append(nr)

    out = pd.DataFrame(new_rows, columns=NEW_COLUMNS)
    out.to_csv(exp_path, index=False)

    # Score preservation check
    merged = score_snap.merge(out[["experiment_id"] + SCORE_COLS], on="experiment_id", suffixes=("_old", "_new"))
    max_delta = 0.0
    for c in SCORE_COLS:
        a = pd.to_numeric(merged[f"{c}_old"], errors="coerce")
        b = pd.to_numeric(merged[f"{c}_new"], errors="coerce")
        # both NA ok
        mask = a.notna() | b.notna()
        if mask.any():
            d = (a - b).abs()
            d = d[a.notna() & b.notna()]
            if len(d):
                max_delta = max(max_delta, float(d.max()))
                if float(d.max()) > 1e-12:
                    raise SystemExit(f"score changed for {c}: max delta {d.max()}")
    print(f"score preservation max_delta={max_delta}")

    # Prediction preservation
    max_pred_delta = 0.0
    for eid, entry in pred_snap.items():
        code = id_to_code[eid]
        for name, meta in entry.items():
            p = pred_dir / code / name
            dfp = pd.read_csv(p)
            assert list(dfp.columns) == meta["cols"]
            assert dfp["id"].astype(str).tolist() == meta["ids"]
            vals = dfp.iloc[:, 1].astype(float).tolist()
            for a, b in zip(meta["values"], vals):
                max_pred_delta = max(max_pred_delta, abs(a - b))
            if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
                # float formatting might change? we used rename so bytes should match
                raise SystemExit(f"prediction file bytes changed: {p}")
    print(f"prediction preservation max_delta={max_pred_delta}")

    # Save migration snapshot for report
    snap_path = ROOT / "results" / "_migration_snapshot.json"
    snap_path.write_text(
        json.dumps(
            {
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "n_codes": 48,
                "score_max_delta": max_delta,
                "prediction_max_delta": max_pred_delta,
                "first5": codes_rows[:5],
                "last5": codes_rows[-5:],
            },
            indent=2,
        )
        + "\n"
    )
    print("migration core done")


if __name__ == "__main__":
    main()
