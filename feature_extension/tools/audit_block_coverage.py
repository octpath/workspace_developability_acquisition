#!/usr/bin/env python3
"""Generate BLOCK_COVERAGE.csv + refresh MANIFEST coverage fields (metadata only)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FE = Path(__file__).resolve().parents[1]
ROOT = FE.parent
CROSSWALK = ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_ids(ids: list[str], expected: set[str]) -> dict:
    s = pd.Series(ids).astype(str)
    available = set(s)
    return {
        "expected_ids": len(expected),
        "available_ids": len(available),
        "missing_ids": len(expected - available),
        "extra_ids": len(available - expected),
        "duplicate_ids": int(s.duplicated().sum()),
        "missing_id_list": ";".join(sorted(expected - available)[:20]),
    }


def audit_table(path: Path, expected: set[str]) -> dict:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(path)
    info = audit_ids(df["id"].astype(str).tolist(), expected)
    feat = [c for c in df.columns if c != "id"]
    miss_cols = []
    total_miss = 0
    has_inf = False
    nonnum = []
    for c in feat:
        if not pd.api.types.is_numeric_dtype(df[c]):
            nonnum.append(c)
            continue
        arr = df[c].to_numpy(dtype=float)
        nmiss = int(np.isnan(arr).sum())
        if nmiss:
            miss_cols.append(c)
            total_miss += nmiss
        if np.isinf(arr).any():
            has_inf = True
    status = "COMPLETE"
    notes = []
    if info["missing_ids"]:
        status = "INCOMPLETE_IDS"
        notes.append(f"missing_ids={info['missing_id_list']}")
    if total_miss:
        notes.append(f"cell_missing={total_miss} across {len(miss_cols)} cols")
        if status == "COMPLETE":
            status = "COMPLETE_IDS_PARTIAL_VALUES"
    if has_inf:
        status = "HAS_INF"
        notes.append("contains_inf")
    if nonnum:
        notes.append(f"nonnumeric={nonnum}")
    return {
        "block": path.stem if path.parent.name != "bioemu_isolated" else "bioemu_isolated_features",
        "path": str(path.relative_to(FE)),
        **{k: info[k] for k in ("expected_ids", "available_ids", "missing_ids", "duplicate_ids")},
        "extra_ids": info["extra_ids"],
        "feature_dim": len(feat),
        "columns_with_missing": len(miss_cols),
        "total_missing_values": total_miss,
        "status": status,
        "notes": "; ".join(notes) if notes else "",
        "missing_id_list": info["missing_id_list"],
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def main() -> None:
    cw = pd.read_csv(CROSSWALK)
    expected = set(cw["id"].astype(str))
    rows = []

    for sub, label in [("esmfold_fv", "esmfold_fv_pdbs"), ("esmfold_fab", "esmfold_fab_pdbs")]:
        ids = [p.stem for p in (FE / "data" / sub).glob("*.pdb")]
        info = audit_ids(ids, expected)
        status = "COMPLETE" if info["missing_ids"] == 0 and info["duplicate_ids"] == 0 else "INCOMPLETE_IDS"
        rows.append(
            {
                "block": label,
                "path": f"data/{sub}",
                "expected_ids": info["expected_ids"],
                "available_ids": info["available_ids"],
                "missing_ids": info["missing_ids"],
                "duplicate_ids": info["duplicate_ids"],
                "extra_ids": info["extra_ids"],
                "feature_dim": "",
                "columns_with_missing": "",
                "total_missing_values": "",
                "status": status,
                "notes": info["missing_id_list"],
            }
        )

    rows.append(audit_table(FE / "data/bioemu_isolated/features.parquet", expected))
    for p in sorted((FE / "data/precomputed_features").glob("*.parquet")):
        rows.append(audit_table(p, expected))

    cov = pd.DataFrame(rows)
    # drop helper cols not in suggested schema for public csv
    out_cols = [
        "block",
        "path",
        "expected_ids",
        "available_ids",
        "missing_ids",
        "duplicate_ids",
        "feature_dim",
        "columns_with_missing",
        "total_missing_values",
        "status",
        "notes",
    ]
    # keep extra_ids in notes if any
    for i, r in cov.iterrows():
        if int(r.get("extra_ids") or 0) > 0:
            cov.at[i, "notes"] = (str(r["notes"]) + f"; extra_ids={r['extra_ids']}").strip("; ")
    cov[out_cols].to_csv(FE / "BLOCK_COVERAGE.csv", index=False)
    print(cov[out_cols].to_string(index=False))

    # refresh MANIFEST coverage fields (cast to plain Python types for Arrow/string cols)
    man = pd.read_csv(FE / "MANIFEST.csv")
    for col in ("expected_n_ids", "missing_n_ids"):
        if col not in man.columns:
            man[col] = pd.NA
    man["expected_n_ids"] = man["expected_n_ids"].astype(object)
    man["missing_n_ids"] = man["missing_n_ids"].astype(object)
    man["n_ids"] = man["n_ids"].astype(object)
    man["feature_dim"] = man["feature_dim"].astype(object)
    man["qc_status"] = man["qc_status"].astype(object)
    man["notes"] = man["notes"].astype(object)

    by_path = {str(r["path"]): r for _, r in cov.iterrows()}
    for i, r in man.iterrows():
        rel = str(r["relative_path"])
        hit = by_path.get(rel)
        if hit is None:
            for path, c in by_path.items():
                if path == rel or rel.startswith(path.rstrip("/") + "/") or path.startswith(rel.rstrip("/") + "/"):
                    # prefer exact; structure dirs match prefix
                    if path == rel or (r.get("artifact_type") == "structure_dir" and path == rel):
                        hit = c
                        break
            if hit is None and rel in ("data/esmfold_fv", "data/esmfold_fab"):
                hit = by_path.get(rel)
        if hit is None:
            continue
        man.at[i, "expected_n_ids"] = int(hit["expected_ids"])
        man.at[i, "missing_n_ids"] = int(hit["missing_ids"])
        man.at[i, "n_ids"] = int(hit["available_ids"])
        if hit.get("feature_dim") != "" and pd.notna(hit.get("feature_dim")):
            man.at[i, "feature_dim"] = int(hit["feature_dim"])
        man.at[i, "qc_status"] = str(hit["status"])
        if hit["notes"]:
            prev = "" if pd.isna(r.get("notes")) else str(r.get("notes"))
            if str(hit["notes"]) not in prev:
                man.at[i, "notes"] = (prev + "; " + str(hit["notes"])).strip("; ")
    man.to_csv(FE / "MANIFEST.csv", index=False)
    print("updated MANIFEST.csv")


if __name__ == "__main__":
    main()
