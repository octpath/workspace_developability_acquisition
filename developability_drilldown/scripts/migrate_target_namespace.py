#!/usr/bin/env python3
"""Migrate EXP001–EXP048 → EXP-Txxx / EXP-Hxxx (one-shot).

Preserves relative legacy numeric order within each target.
Does not create EXP-M rows. Does not change scores/predictions/feature bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import ROOT, feature_content_sha256, file_sha256  # noqa: E402

SCORE_COLS = [
    "cv_primary_mae",
    "cv_shadow_mae",
    "cv_mean_mae",
    "cv_worst_mae",
    "public_mae",
    "private_mae",
    "test_overall_mae",
]

LEGACY_RE = re.compile(r"^EXP([0-9]{3,})$")
NEW_RE = re.compile(r"^EXP-[THM][0-9]{3,}$")


def legacy_num(code: str) -> int:
    m = LEGACY_RE.match(code)
    if not m:
        raise ValueError(code)
    return int(m.group(1))


def main() -> None:
    codes_path = ROOT / "results" / "EXPERIMENT_CODES.csv"
    exp_path = ROOT / "results" / "experiments.csv"
    codes = pd.read_csv(codes_path)
    exp = pd.read_csv(exp_path)
    assert len(codes) == 48 and len(exp) == 48

    # Guard: already migrated?
    if codes["experiment_code"].astype(str).str.match(r"^EXP-[THM]").any():
        raise SystemExit("already looks target-namespaced; abort")

    merged = codes.merge(exp[["experiment_id", "target"]], on="experiment_id", how="inner")
    assert len(merged) == 48

    score_snap = exp[["experiment_id"] + SCORE_COLS].copy()

    # Build mapping: legacy -> new, preserving legacy numeric order within target
    mapping_rows = []
    id_to_new: dict[str, str] = {}
    legacy_to_new: dict[str, str] = {}

    for target, prefix in (("TmApp", "T"), ("HIC", "H")):
        sub = merged[merged["target"] == target].copy()
        sub["_n"] = sub["experiment_code"].map(legacy_num)
        sub = sub.sort_values("_n")
        for i, (_, r) in enumerate(sub.iterrows(), start=1):
            new_code = f"EXP-{prefix}{i:03d}"
            legacy = str(r["experiment_code"])
            eid = str(r["experiment_id"])
            id_to_new[eid] = new_code
            legacy_to_new[legacy] = new_code
            mapping_rows.append(
                {
                    "legacy_experiment_code": legacy,
                    "experiment_code": new_code,
                    "experiment_id": eid,
                    "target": target,
                    "migration_reason": "target_namespace_from_legacy_EXP_order",
                    "migration_commit": "pending",
                }
            )

    map_df = pd.DataFrame(mapping_rows)
    assert len(map_df) == 48
    assert map_df["experiment_code"].is_unique
    assert map_df["legacy_experiment_code"].is_unique
    t_count = int((map_df["target"] == "TmApp").sum())
    h_count = int((map_df["target"] == "HIC").sum())
    print(f"TmApp -> EXP-T001..EXP-T{t_count:03d} ({t_count})")
    print(f"HIC   -> EXP-H001..EXP-H{h_count:03d} ({h_count})")

    # Snapshot prediction/feature hashes keyed by experiment_id
    pred_snap = {}
    feat_snap = {}
    for _, r in exp[exp["artifact_status"] == "FULL"].iterrows():
        eid = str(r["experiment_id"])
        old = str(r["experiment_code"])
        pred_dir = ROOT / "experiments" / "predictions" / old
        entry = {}
        for name in ("oof_primary.csv", "oof_shadow.csv", "test.csv"):
            p = pred_dir / name
            dfp = pd.read_csv(p)
            entry[name] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "ids": dfp["id"].astype(str).tolist(),
                "cols": list(dfp.columns),
                "values": dfp.iloc[:, 1].astype(float).tolist(),
            }
        pred_snap[eid] = entry
        fp = ROOT / "experiments" / "features" / f"{old}.parquet"
        feat_snap[eid] = {
            "file_sha256": file_sha256(fp),
            "content_sha256": feature_content_sha256(pd.read_parquet(fp)),
        }

    # Rename FULL artifacts
    cfg_dir = ROOT / "experiments" / "configs"
    feat_dir = ROOT / "experiments" / "features"
    pred_dir = ROOT / "experiments" / "predictions"

    for _, r in exp[exp["artifact_status"] == "FULL"].iterrows():
        eid = str(r["experiment_id"])
        old = str(r["experiment_code"])
        new = id_to_new[eid]

        old_cfg = cfg_dir / f"{old}.yaml"
        new_cfg = cfg_dir / f"{new}.yaml"
        data = yaml.safe_load(old_cfg.read_text()) or {}
        data["experiment_code"] = new
        data["legacy_experiment_code"] = old
        data["experiment_id"] = eid
        old_cfg.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        old_cfg.rename(new_cfg)

        old_feat = feat_dir / f"{old}.parquet"
        new_feat = feat_dir / f"{new}.parquet"
        old_feat.rename(new_feat)

        old_pred = pred_dir / old
        new_pred = pred_dir / new
        old_pred.rename(new_pred)

    # Update EXPERIMENT_CODES.csv
    new_codes = []
    for _, r in map_df.iterrows():
        src = codes[codes["experiment_id"] == r["experiment_id"]].iloc[0]
        new_codes.append(
            {
                "experiment_code": r["experiment_code"],
                "experiment_id": r["experiment_id"],
                "target": r["target"],
                "source_model_id": src["source_model_id"],
                "issued_at_phase": "TARGET_NAMESPACE_MIGRATION",
                "status": "ACTIVE",
                "legacy_experiment_code": r["legacy_experiment_code"],
                "notes": "Migrated from flat EXP001–EXP048; order preserved within target",
            }
        )
    # Sort codes: T then H by number for readability
    codes_out = pd.DataFrame(new_codes)

    def sort_key(code: str):
        body = code.split("-", 1)[1]
        return ({"T": 0, "H": 1, "M": 2}[body[0]], int(body[1:]))

    codes_out = codes_out.iloc[
        sorted(range(len(codes_out)), key=lambda i: sort_key(str(codes_out.iloc[i]["experiment_code"])))
    ].reset_index(drop=True)
    codes_out.to_csv(codes_path, index=False)

    map_df.to_csv(ROOT / "results" / "LEGACY_EXPERIMENT_CODE_MAP.csv", index=False)

    # Update experiments.csv
    cols = list(exp.columns)
    # insert legacy_experiment_code after experiment_code if needed
    if "legacy_experiment_code" not in cols:
        cols.insert(cols.index("experiment_code") + 1, "legacy_experiment_code")

    rows = []
    for _, r in exp.iterrows():
        d = r.to_dict()
        eid = str(d["experiment_id"])
        old = str(d["experiment_code"])
        new = id_to_new[eid]
        d["experiment_code"] = new
        d["legacy_experiment_code"] = old
        if d.get("artifact_status") == "FULL":
            d["config_path"] = f"experiments/configs/{new}.yaml"
            d["feature_path"] = f"experiments/features/{new}.parquet"
            d["oof_primary_path"] = f"experiments/predictions/{new}/oof_primary.csv"
            d["oof_shadow_path"] = f"experiments/predictions/{new}/oof_shadow.csv"
            d["test_prediction_path"] = f"experiments/predictions/{new}/test.csv"
            # verify hashes unchanged
            fp = ROOT / d["feature_path"]
            assert file_sha256(fp) == feat_snap[eid]["file_sha256"]
            assert feature_content_sha256(pd.read_parquet(fp)) == feat_snap[eid]["content_sha256"]
            d["feature_sha256"] = feat_snap[eid]["file_sha256"]
            d["feature_content_sha256"] = feat_snap[eid]["content_sha256"]
        rows.append(d)

    out = pd.DataFrame(rows)
    # reorder columns: experiment_code, legacy_experiment_code, experiment_id, ...
    preferred = ["experiment_code", "legacy_experiment_code", "experiment_id"]
    rest = [c for c in out.columns if c not in preferred]
    out = out[preferred + rest]
    out.to_csv(exp_path, index=False)

    # Score preservation
    merged_s = score_snap.merge(out[["experiment_id"] + SCORE_COLS], on="experiment_id", suffixes=("_old", "_new"))
    max_delta = 0.0
    for c in SCORE_COLS:
        a = pd.to_numeric(merged_s[f"{c}_old"], errors="coerce")
        b = pd.to_numeric(merged_s[f"{c}_new"], errors="coerce")
        both = a.notna() & b.notna()
        if both.any():
            d = (a[both] - b[both]).abs().max()
            max_delta = max(max_delta, float(d))
            if float(d) > 1e-12:
                raise SystemExit(f"score changed {c}: {d}")
    print(f"score max_delta={max_delta}")

    # Prediction preservation
    max_pred = 0.0
    for eid, entry in pred_snap.items():
        new = id_to_new[eid]
        for name, meta in entry.items():
            p = pred_dir / new / name
            dfp = pd.read_csv(p)
            assert list(dfp.columns) == meta["cols"]
            assert dfp["id"].astype(str).tolist() == meta["ids"]
            for a, b in zip(meta["values"], dfp.iloc[:, 1].astype(float).tolist()):
                max_pred = max(max_pred, abs(a - b))
            if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
                raise SystemExit(f"prediction bytes changed {p}")
    print(f"prediction max_delta={max_pred}")

    snap = {
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "t_count": t_count,
        "h_count": h_count,
        "score_max_delta": max_delta,
        "prediction_max_delta": max_pred,
        "first5": mapping_rows[:5],
        "last5": mapping_rows[-5:],
    }
    (ROOT / "results" / "_target_namespace_snapshot.json").write_text(json.dumps(snap, indent=2) + "\n")
    print("target-namespace migration core done")


if __name__ == "__main__":
    main()
