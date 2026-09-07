#!/usr/bin/env python3
"""Validate participant feature_extension release for label leakage and integrity."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FE = Path(__file__).resolve().parents[1]

FORBIDDEN_TOKENS = (
    "tmap",
    "hic",
    "public",
    "private",
    "y_true",
    "y_pred",
    "residual",
    "incumbent",
    "oof",
    "leaderboard",
)

ABS_PATH_RE = re.compile(r"/(?:workspace|home|Users|data|mnt)/[^\s\"']+")


def check_forbidden_columns(columns: list[str], where: str, errors: list[str]) -> None:
    for c in columns:
        cl = c.lower()
        for tok in FORBIDDEN_TOKENS:
            if tok in cl:
                errors.append(f"forbidden column '{c}' in {where} (token={tok})")


def check_tabular(path: Path, errors: list[str], warnings: list[str]) -> None:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    check_forbidden_columns(list(df.columns), str(path.relative_to(FE)), errors)
    if "id" in df.columns:
        if df["id"].duplicated().any():
            errors.append(f"duplicate ids in {path}")
        if df["id"].isna().any():
            errors.append(f"null ids in {path}")
    for c in df.columns:
        if c == "id":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if np.isinf(df[c].to_numpy(dtype=float)).any():
                errors.append(f"±inf in {path}:{c}")
            n_nan = int(df[c].isna().sum())
            if n_nan:
                warnings.append(f"missing {n_nan} in {path.name}:{c}")


def scan_text_for_abs_paths(path: Path, errors: list[str]) -> None:
    # skip large binary-ish; only text-like
    if path.suffix.lower() in {".pdb", ".parquet", ".zip", ".pyc"}:
        return
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    # allow mentions inside RELEASE docs of organizer paths as provenance? User said no absolute organizer paths in participant files.
    # Be strict for code/csv/json; allow RELEASE_NOTES/AUDIT to mention repo-relative paths only.
    if path.name in {"RELEASE_AUDIT.md", "RELEASE_NOTES.md", "_build_meta.json"}:
        # still flag absolute /workspace paths
        pass
    for m in ABS_PATH_RE.finditer(text):
        errors.append(f"absolute path in {path.relative_to(FE)}: {m.group(0)[:80]}")


def validate_folds(errors: list[str]) -> None:
    folds = pd.read_csv(FE / "folds.csv")
    need = {"id", "fold_primary", "fold_shadow"}
    if set(folds.columns) != need and not need.issubset(set(folds.columns)):
        errors.append(f"folds.csv columns expected {need}, got {list(folds.columns)}")
    if folds["id"].duplicated().any():
        errors.append("duplicate fold ids")
    for col in ("fold_primary", "fold_shadow"):
        bad = ~folds[col].astype(int).isin(range(5))
        if bad.any():
            errors.append(f"invalid {col} values")
    # no public/private columns
    check_forbidden_columns(list(folds.columns), "folds.csv", errors)


def validate_pdbs(errors: list[str], warnings: list[str]) -> None:
    for sub in ("data/esmfold_fv", "data/esmfold_fab"):
        d = FE / sub
        if not d.is_dir():
            errors.append(f"missing {sub}")
            continue
        pdbs = list(d.glob("*.pdb"))
        if len(pdbs) < 100:
            warnings.append(f"few PDBs in {sub}: {len(pdbs)}")
        for p in pdbs:
            if not re.fullmatch(r"ADI-\d+\.pdb", p.name):
                # allow broader id patterns
                if not re.fullmatch(r"[A-Za-z0-9_.-]+\.pdb", p.name):
                    errors.append(f"malformed PDB name {p.name}")


def validate_symlinks(errors: list[str]) -> None:
    for p in FE.rglob("*"):
        if p.is_symlink():
            target = p.resolve()
            try:
                target.relative_to(FE.resolve())
            except ValueError:
                errors.append(f"symlink escapes package: {p} -> {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-paths", action="store_true", default=True)
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if not (FE / "folds.csv").is_file():
        errors.append("missing folds.csv")
    else:
        validate_folds(errors)

    validate_pdbs(errors, warnings)
    validate_symlinks(errors)

    for path in list((FE / "data").rglob("*.parquet")) + list((FE / "data").rglob("*.csv")):
        if path.name == "MANIFEST.csv":
            continue
        try:
            check_tabular(path, errors, warnings)
        except Exception as e:
            errors.append(f"unreadable {path}: {e}")

    # MANIFEST
    man = FE / "MANIFEST.csv"
    if man.is_file():
        m = pd.read_csv(man)
        if "target_used" in m.columns:
            bad = m[m["target_used"].astype(str).str.upper() != "NO"]
            # folds / N/A rows: allow only NO
            if len(bad):
                # allow empty
                for _, r in bad.iterrows():
                    if str(r.get("artifact_type")) == "cv_folds":
                        continue
                    if str(r.get("target_used")).upper() not in {"NO", "NAN"}:
                        errors.append(f"MANIFEST target_used not NO: {r.get('artifact')}")

    # absolute paths in participant-facing text/code (exclude tools build script which may hardcode ROOT for organizers)
    skip_names = {"build_v1_data.py", "_build_meta.json"}
    for path in FE.rglob("*"):
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        if "dist/" in str(path.relative_to(FE)):
            continue
        if args.strict_paths:
            scan_text_for_abs_paths(path, errors)

    print("=== validate_release ===")
    for w in warnings[:50]:
        print("WARN:", w)
    if len(warnings) > 50:
        print(f"WARN: ... {len(warnings) - 50} more")
    if errors:
        print(f"FAIL ({len(errors)} errors)")
        for e in errors[:80]:
            print("ERR:", e)
        if len(errors) > 80:
            print(f"ERR: ... {len(errors) - 80} more")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
