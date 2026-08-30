#!/usr/bin/env python3
"""Build competition distribution + secret CSVs from frozen artifacts.

Does NOT regenerate the Public/Private split. Loads SPLIT_MANIFEST.json
(or creates it from the B7.3 recommended split on first run) and verifies hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMP = ROOT / "competition"
ORG_OUT = COMP / "organizer"
DIST = COMP / "data" / "distribution"
SECRET = COMP / "data" / "secret"

FROZEN_POP = ROOT / "gate_b3" / "frozen" / "organizer" / "final_population.csv"
FROZEN_ROLE = ROOT / "gate_b3" / "frozen" / "organizer" / "role_map.csv"
B73_SPLIT = ROOT / "gate_b7_3_principled_split" / "config" / "B7_3_RECOMMENDED_SPLIT.json"
SPLIT_MANIFEST = ORG_OUT / "SPLIT_MANIFEST.json"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
HIC_LOW, HIC_HIGH = 10.5, 11.5


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def id_list_hash(ids: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def hic_band(v: float) -> str:
    if v < HIC_LOW:
        return "LOW"
    if v <= HIC_HIGH:
        return "MEDIUM"
    return "HIGH"


def load_or_create_split_manifest() -> dict:
    src = json.loads(B73_SPLIT.read_text())
    assert src["split_id"] == "GEN_0001_B_20271100"
    pub = sorted(map(str, src["public_ids"]))
    priv = sorted(map(str, src["private_ids"]))
    pub_h = id_list_hash(pub)
    priv_h = id_list_hash(priv)
    expected_pub = "2a267694613f9aec37613a8c7e532c04b4cbb892ba4a5416d335f20b1ea27376"
    expected_priv = "f9d26ae7ebcc4ed9b06c816a72061c7ee5a7cfb748000cb3534b4fd509725cf0"
    assert pub_h == expected_pub == src.get("public_sha256", expected_pub)
    assert priv_h == expected_priv == src.get("private_sha256", expected_priv)
    assert len(pub) == 81 and len(priv) == 81
    assert not set(pub) & set(priv)

    manifest = {
        "split_id": "GEN_0001_B_20271100",
        "seed": int(src["seed"]),
        "construction_method": src.get("method", "B_simulated_annealing"),
        "public_ids": pub,
        "private_ids": priv,
        "public_id_hash": pub_h,
        "private_id_hash": priv_h,
        "public_n": 81,
        "private_n": 81,
        "source_artifact": str(B73_SPLIT.relative_to(ROOT)),
        "source_artifact_sha256": sha256_file(B73_SPLIT),
        "created_package_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": ["TmApp", "HIC"],
        "primary_metric": "MAE",
        "note": "Explicit public_ids/private_ids are authoritative; seed is metadata only.",
    }

    if SPLIT_MANIFEST.exists():
        existing = json.loads(SPLIT_MANIFEST.read_text())
        # Keep frozen ID assignment; refresh only non-identity metadata if needed
        assert existing["split_id"] == manifest["split_id"]
        assert existing["public_ids"] == pub
        assert existing["private_ids"] == priv
        assert existing["public_id_hash"] == pub_h
        assert existing["private_id_hash"] == priv_h
        # Preserve original created timestamp if already frozen
        manifest["created_package_timestamp"] = existing.get(
            "created_package_timestamp", manifest["created_package_timestamp"]
        )
        manifest["source_artifact_sha256"] = existing.get(
            "source_artifact_sha256", manifest["source_artifact_sha256"]
        )
    return manifest


def write_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic CSV: unix newlines, no index
    text = df.to_csv(index=False, lineterminator="\n")
    path.write_text(text)
    return sha256_bytes(text.encode())


def build(out_dist: Path | None = None, out_secret: Path | None = None) -> dict:
    out_dist = out_dist or DIST
    out_secret = out_secret or SECRET
    out_dist.mkdir(parents=True, exist_ok=True)
    out_secret.mkdir(parents=True, exist_ok=True)

    ORG_OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_split_manifest()
    # Write manifest without varying timestamp on rebuild if already present handled above
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    pop = pd.read_csv(FROZEN_POP)
    role = pd.read_csv(FROZEN_ROLE)
    pop["id"] = pop["id"].astype(str)
    role["id"] = role["id"].astype(str)
    m = pop.merge(role[["id", "role"]], on="id", how="inner", validate="one_to_one")
    assert len(m) == 324

    train = m[m.role == "Train"].copy()
    test = m[m.role != "Train"].copy()
    assert len(train) == 162 and len(test) == 162

    pub_set = set(manifest["public_ids"])
    priv_set = set(manifest["private_ids"])
    test_ids = set(test["id"])
    assert pub_set | priv_set == test_ids
    assert not pub_set & priv_set

    # HIC band sanity under production split
    bands = test.assign(is_public=test.id.isin(pub_set), band=test.HIC.map(hic_band))
    med = bands[bands.band == "MEDIUM"]
    high = bands[bands.band == "HIGH"]
    assert int(med.is_public.sum()) == 3 and int((~med.is_public).sum()) == 3
    assert int(high.is_public.sum()) == 4 and int((~high.is_public).sum()) == 3

    # Order: stable sort by id for train; test order matches sample_submission = sorted id
    # (join-by-id scoring; stable deterministic order)
    train = train.sort_values("id").reset_index(drop=True)
    test = test.sort_values("id").reset_index(drop=True)

    for col in ("heavy", "light"):
        assert train[col].notna().all() and test[col].notna().all()
        for s in pd.concat([train[col], test[col]]):
            assert isinstance(s, str) and s == s.strip() and s == s.upper()
            assert set(s) <= AA20

    assert train.id.is_unique and test.id.is_unique
    assert not set(train.id) & set(test.id)
    assert len(set(zip(train.heavy, train.light)) & set(zip(test.heavy, test.light))) == 0

    for col in ("TmApp", "HIC"):
        assert np.isfinite(train[col].astype(float)).all()
        assert np.isfinite(test[col].astype(float)).all()

    dev = train[["id", "heavy", "light", "TmApp", "HIC"]].copy()
    test_features = test[["id", "heavy", "light"]].copy()

    # Sample submission: Train medians only (no Private labels)
    med_tm = float(np.median(dev["TmApp"].values))
    med_hic = float(np.median(dev["HIC"].values))
    sample = test_features[["id"]].copy()
    sample["TmApp"] = med_tm
    sample["HIC"] = med_hic

    solution = test[["id", "TmApp", "HIC"]].copy()
    solution["is_public"] = solution["id"].isin(pub_set)
    solution["is_private"] = solution["id"].isin(priv_set)
    assert bool((solution.is_public ^ solution.is_private).all())
    assert int(solution.is_public.sum()) == 81
    assert int(solution.is_private.sum()) == 81

    hashes = {
        "dev.csv": write_csv(out_dist / "dev.csv", dev),
        "test_features.csv": write_csv(out_dist / "test_features.csv", test_features),
        "sample_submission.csv": write_csv(out_dist / "sample_submission.csv", sample),
        "solution.csv": write_csv(out_secret / "solution.csv", solution),
    }

    meta = {
        "n_dev": int(len(dev)),
        "n_test": int(len(test_features)),
        "tmapp_dev_range": [float(dev.TmApp.min()), float(dev.TmApp.max())],
        "hic_dev_range": [float(dev.HIC.min()), float(dev.HIC.max())],
        "tmapp_test_range": [float(solution.TmApp.min()), float(solution.TmApp.max())],
        "hic_test_range": [float(solution.HIC.min()), float(solution.HIC.max())],
        "sample_tmapp_median": med_tm,
        "sample_hic_median": med_hic,
        "hic_medium_pub_priv": [3, 3],
        "hic_high_pub_priv": [4, 3],
        "file_sha256": hashes,
        "split_id": manifest["split_id"],
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=None)
    ap.add_argument("--secret", type=Path, default=None)
    args = ap.parse_args()
    meta = build(args.dist, args.secret)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
