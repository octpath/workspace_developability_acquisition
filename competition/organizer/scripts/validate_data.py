#!/usr/bin/env python3
"""Validate competition distribution and secret solution files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMP = ROOT / "competition"
DIST = COMP / "data" / "distribution"
SECRET = COMP / "data" / "secret"
SPLIT_MANIFEST = COMP / "organizer" / "SPLIT_MANIFEST.json"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HIC_LOW, HIC_HIGH = 10.5, 11.5


def hic_band(v: float) -> str:
    if v < HIC_LOW:
        return "LOW"
    if v <= HIC_HIGH:
        return "MEDIUM"
    return "HIGH"


def check(cond: bool, msg: str, errors: list[str], ok: list[str]):
    if cond:
        ok.append(msg)
    else:
        errors.append(msg)


def validate(dist: Path = DIST, secret: Path = SECRET) -> dict:
    errors: list[str] = []
    ok: list[str] = []

    man = json.loads(SPLIT_MANIFEST.read_text())
    pub = set(man["public_ids"])
    priv = set(man["private_ids"])
    check(man["split_id"] == "GEN_0001_B_20271100", "split_id == GEN_0001_B_20271100", errors, ok)
    check(len(pub) == 81 and len(priv) == 81, "Public/Private N = 81/81", errors, ok)
    check(not pub & priv, "Public ∩ Private empty", errors, ok)

    dev = pd.read_csv(dist / "dev.csv")
    test = pd.read_csv(dist / "test_features.csv")
    sample = pd.read_csv(dist / "sample_submission.csv")
    sol = pd.read_csv(secret / "solution.csv")

    check(list(dev.columns) == ["id", "heavy", "light", "TmApp", "HIC"], "dev.csv columns", errors, ok)
    check(list(test.columns) == ["id", "heavy", "light"], "test_features.csv columns", errors, ok)
    check(list(sample.columns) == ["id", "TmApp", "HIC"], "sample_submission.csv columns", errors, ok)
    check(
        list(sol.columns) == ["id", "TmApp", "HIC", "is_public", "is_private"],
        "solution.csv columns",
        errors,
        ok,
    )

    check(len(dev) == 162, f"dev N=162 (got {len(dev)})", errors, ok)
    check(len(test) == 162, f"test N=162 (got {len(test)})", errors, ok)
    check(len(sol) == 162, f"solution N=162 (got {len(sol)})", errors, ok)

    check(dev.id.is_unique, "dev IDs unique", errors, ok)
    check(test.id.is_unique, "test IDs unique", errors, ok)
    check(sol.id.is_unique, "solution IDs unique", errors, ok)
    check(not set(dev.id) & set(test.id), "dev ∩ test empty", errors, ok)
    check(set(test.id) == set(sol.id), "test IDs == solution IDs", errors, ok)
    check(list(test.id) == list(sample.id), "sample ID order matches test_features", errors, ok)
    check(set(test.id) == pub | priv, "test IDs == Public ∪ Private", errors, ok)

    # no accidental index column
    for name, df in [("dev", dev), ("test", test), ("sample", sample), ("solution", sol)]:
        check("Unnamed: 0" not in df.columns, f"{name}: no Unnamed index column", errors, ok)

    # sequences
    for name, df in [("dev", dev), ("test", test)]:
        for col in ("heavy", "light"):
            check(df[col].notna().all(), f"{name}.{col} non-null", errors, ok)
            bad_ws = ~df[col].map(lambda s: isinstance(s, str) and s == s.strip())
            bad_case = ~df[col].map(lambda s: isinstance(s, str) and s == s.upper())
            bad_aa = ~df[col].map(lambda s: isinstance(s, str) and set(s) <= AA20)
            check(not bad_ws.any(), f"{name}.{col} no whitespace corruption", errors, ok)
            check(not bad_case.any(), f"{name}.{col} uppercase AA only", errors, ok)
            check(not bad_aa.any(), f"{name}.{col} alphabet ⊆ AA20", errors, ok)

    # no exact HL pair across splits
    hl_dev = set(zip(dev.heavy, dev.light))
    hl_te = set(zip(test.heavy, test.light))
    check(not hl_dev & hl_te, "no exact heavy/light pair across dev/test", errors, ok)

    # labels finite
    for name, df, cols in [
        ("dev", dev, ["TmApp", "HIC"]),
        ("solution", sol, ["TmApp", "HIC"]),
        ("sample", sample, ["TmApp", "HIC"]),
    ]:
        for c in cols:
            check(np.isfinite(df[c].astype(float)).all(), f"{name}.{c} finite", errors, ok)

    # secret flags
    # accept True/False or 1/0
    ip = sol["is_public"].map(lambda x: str(x).strip().lower() in ("true", "1"))
    ipv = sol["is_private"].map(lambda x: str(x).strip().lower() in ("true", "1"))
    check(bool((ip ^ ipv).all()), "is_public XOR is_private", errors, ok)
    check(int(ip.sum()) == 81, "sum(is_public)=81", errors, ok)
    check(int(ipv.sum()) == 81, "sum(is_private)=81", errors, ok)
    check(set(sol.loc[ip, "id"]) == pub, "solution public IDs match manifest", errors, ok)
    check(set(sol.loc[ipv, "id"]) == priv, "solution private IDs match manifest", errors, ok)

    # no targets in test_features
    check(
        not any(c in test.columns for c in ("TmApp", "HIC", "is_public", "is_private")),
        "test_features has no labels/split flags",
        errors,
        ok,
    )

    # Sequence-derived annotations (optional participant resources)
    ann_cols = [
        "id",
        "heavy_v_family",
        "heavy_j_family",
        "light_v_family",
        "light_j_family",
        "light_chain_type",
        "h_cdr1_length",
        "h_cdr2_length",
        "h_cdr3_length",
        "l_cdr1_length",
        "l_cdr2_length",
        "l_cdr3_length",
        "heavy_germline_identity",
        "light_germline_identity",
    ]
    forbidden_ann = {
        "TmApp",
        "HIC",
        "heavy",
        "light",
        "is_public",
        "is_private",
        "donor",
        "b_cell_subset",
        "b_cell_origin",
        "naive",
        "memory",
        "LLPC",
        "role",
    }
    for ann_name, base_df in [
        ("dev_annotations.csv", dev),
        ("test_annotations.csv", test),
    ]:
        path = dist / ann_name
        check(path.exists(), f"{ann_name} exists", errors, ok)
        if not path.exists():
            continue
        ann = pd.read_csv(path)
        check(list(ann.columns) == ann_cols, f"{ann_name} columns", errors, ok)
        check(len(ann) == 162, f"{ann_name} N=162 (got {len(ann)})", errors, ok)
        check(ann.id.is_unique, f"{ann_name} IDs unique", errors, ok)
        check(set(ann.id) == set(base_df.id), f"{ann_name} ID set matches base CSV", errors, ok)
        check(not forbidden_ann & set(ann.columns), f"{ann_name} no forbidden columns", errors, ok)
        check(
            not any(c in ann.columns for c in ("TmApp", "HIC", "is_public", "is_private")),
            f"{ann_name} no target/split columns",
            errors,
            ok,
        )
        if "light_chain_type" in ann.columns:
            check(
                set(ann.light_chain_type.dropna().astype(str)) <= {"kappa", "lambda"},
                f"{ann_name} light_chain_type in {{kappa,lambda}}",
                errors,
                ok,
            )
        for c in (
            "h_cdr1_length",
            "h_cdr2_length",
            "h_cdr3_length",
            "l_cdr1_length",
            "l_cdr2_length",
            "l_cdr3_length",
        ):
            if c in ann.columns:
                vals = ann[c].astype(float)
                check(vals.notna().all(), f"{ann_name}.{c} non-null", errors, ok)
                check((vals >= 1).all() and (vals <= 40).all(), f"{ann_name}.{c} in [1,40]", errors, ok)
        for c in ("heavy_germline_identity", "light_germline_identity"):
            if c in ann.columns:
                vals = ann[c].astype(float)
                check(vals.notna().all(), f"{ann_name}.{c} non-null", errors, ok)
                check((vals >= 0).all() and (vals <= 1).all(), f"{ann_name}.{c} in [0,1]", errors, ok)
        for c in ("heavy_v_family", "heavy_j_family", "light_v_family", "light_j_family"):
            if c in ann.columns:
                check(ann[c].notna().all(), f"{ann_name}.{c} non-null", errors, ok)
                check(
                    ann[c].astype(str).str.len().gt(0).all(),
                    f"{ann_name}.{c} non-empty",
                    errors,
                    ok,
                )

    # HIC bands
    sol2 = sol.copy()
    sol2["band"] = sol2.HIC.map(hic_band)
    med_pub = int(((sol2.band == "MEDIUM") & ip).sum())
    med_priv = int(((sol2.band == "MEDIUM") & ipv).sum())
    hi_pub = int(((sol2.band == "HIGH") & ip).sum())
    hi_priv = int(((sol2.band == "HIGH") & ipv).sum())
    check((med_pub, med_priv) == (3, 3), f"HIC MEDIUM 3/3 (got {med_pub}/{med_priv})", errors, ok)
    check((hi_pub, hi_priv) == (4, 3), f"HIC HIGH 4/3 (got {hi_pub}/{hi_priv})", errors, ok)

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "n_ok": len(ok),
        "n_errors": len(errors),
        "ok": ok,
        "errors": errors,
        "counts": {
            "n_dev": len(dev),
            "n_test": len(test),
            "tmapp_dev": [float(dev.TmApp.min()), float(dev.TmApp.max())],
            "hic_dev": [float(dev.HIC.min()), float(dev.HIC.max())],
            "tmapp_test": [float(sol.TmApp.min()), float(sol.TmApp.max())],
            "hic_test": [float(sol.HIC.min()), float(sol.HIC.max())],
            "hic_medium_pub_priv": [med_pub, med_priv],
            "hic_high_pub_priv": [hi_pub, hi_priv],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=DIST)
    ap.add_argument("--secret", type=Path, default=SECRET)
    args = ap.parse_args()
    report = validate(args.dist, args.secret)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
