"""Tests for classical feature refinement phase."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys_path = str(ROOT)
import sys

sys.path.insert(0, sys_path)
sys.path.insert(0, str(ROOT / "scripts"))

from classical_features.pooling import (  # noqa: E402
    build_pooled_block,
    content_sha_df,
    load_annotations,
    load_plm_pack,
)
from classical_features.cv_eval import evaluate_primary_shadow, load_folds  # noqa: E402
from _lib import (  # noqa: E402
    N_CLASSICAL_REFINEMENT,
    N_EXPERIMENTS_TOTAL,
    PRESERVATION_SNAPSHOT_77,
    feature_content_sha256,
    load_dev_test_folds,
)


@pytest.fixture(scope="module")
def dev_test_ids():
    dev, test, _ = load_dev_test_folds()
    return dev, test, dev["id"].astype(str).tolist() + test["id"].astype(str).tolist()


def test_feature_blocks_registry():
    fb = pd.read_csv(ROOT / "results" / "FEATURE_BLOCKS.csv")
    assert len(fb) >= 30
    assert fb["feature_block_id"].is_unique
    assert (fb["n_features"] > 0).all()


def test_rasa_cache_exists():
    cache = ROOT / "experiments" / "classical_cache"
    assert (cache / "rasa_heavy.npy").exists()
    meta = json.loads((cache / "rasa_meta.json").read_text())
    assert meta["n_ids"] == 324
    assert meta["exposed_threshold"] == 0.2


def test_region_pooling_ablang2(dev_test_ids):
    dev, test, ids = dev_test_ids
    ann = load_annotations()
    pack = load_plm_pack("ablang2")
    df = build_pooled_block(
        pack, ann, ids[:5], chains=["H", "L"], region="CDR3",
        prefix="TEST_CDR3",
    )
    assert len(df) == 5
    assert df.shape[1] == 961  # 960 + id


def test_cdr_weighting_changes_hash(dev_test_ids):
    _, _, ids = dev_test_ids
    ann = load_annotations()
    pack = load_plm_pack("ablang2")
    g1 = build_pooled_block(pack, ann, ids[:3], chains=["H"], region="ALL", cdr_gamma=1.0, prefix="G1")
    g2 = build_pooled_block(pack, ann, ids[:3], chains=["H"], region="ALL", cdr_gamma=2.0, prefix="G2")
    assert content_sha_df(g1) != content_sha_df(g2)


def test_rasa_weighting(dev_test_ids):
    dev, test, ids = dev_test_ids
    ann = load_annotations()
    pack = load_plm_pack("esm2")
    rasa = {
        "ids": ids,
        "H": np.clip(np.random.rand(len(ids), 120), 0, 1).astype(np.float32),
        "L": np.zeros((len(ids), 1)),
    }
    df = build_pooled_block(
        pack, ann, ids[:3], chains=["H"], region="ALL",
        rasa=rasa, rasa_power=1.0, prefix="RASA",
    )
    assert df.shape[0] == 3


def test_content_hash_stable(dev_test_ids):
    _, _, ids = dev_test_ids
    ann = load_annotations()
    pack = load_plm_pack("ablang2")
    df = build_pooled_block(pack, ann, ids[:4], chains=["H"], region="FR", prefix="FR")
    h1 = content_sha_df(df)
    h2 = content_sha_df(df.copy())
    assert h1 == h2


def test_fold_local_cv_ridge(dev_test_ids):
    dev, _, ids = dev_test_ids
    primary, shadow = load_folds(ROOT / "data" / "folds.csv")
    y = {str(r.id): float(r.TmApp) for _, r in dev.iterrows()}
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"id": dev["id"].astype(str)})
    for j in range(8):
        X[f"f{j}"] = rng.normal(size=len(X))
    res = evaluate_primary_shadow(X, y, primary, shadow, estimator="ridge")
    assert res["cv_worst_mae"] > 0
    assert len(res["oof_primary"]) == len(dev)


def test_freeze_firewall():
    freeze = yaml.safe_load((ROOT / "results" / "CLASSICAL_REFINEMENT_FREEZE.yaml").read_text())
    assert freeze["public_private_used_in_selection"] is False
    assert (ROOT / freeze["candidates_path"]).exists()


def test_no_ensemble_new_experiments():
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    snap = set(pd.read_csv(PRESERVATION_SNAPSHOT_77)["experiment_code"])
    new = exp[~exp["experiment_code"].isin(snap)]
    assert len(new) == N_CLASSICAL_REFINEMENT
    for col in ("ensemble_type", "member_experiment_codes"):
        assert new[col].fillna("").astype(str).str.len().eq(0).all()


def test_existing_77_preservation():
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    snap = pd.read_csv(PRESERVATION_SNAPSHOT_77)
    m = snap.merge(exp, on="experiment_code", suffixes=("_old", "_new"))
    assert len(m) == 77
    for c in ("cv_primary_mae", "private_mae"):
        d = (pd.to_numeric(m[f"{c}_old"]) - pd.to_numeric(m[f"{c}_new"])).abs().max()
        assert float(d) == 0.0


def test_registry_total():
    exp = pd.read_csv(ROOT / "results/experiments.csv")
    assert len(exp) == N_EXPERIMENTS_TOTAL
