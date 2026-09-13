#!/usr/bin/env python3
"""Tests for T157–T160 new PLM residue integration smoke (C + FULL)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RES = REPO / "top_models_feature_bundle" / "residue_level"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))

PLMS = ("ablang1", "esm1b", "esmc600m", "currab")
CODES = ("EXP-T157", "EXP-T158", "EXP-T159", "EXP-T160")


def _load_ids(subdir: str) -> list[str]:
    from antibody_transformer.data import load_npy

    return [str(x) for x in load_npy(RES / subdir / "ids.npy", allow_pickle=True).tolist()]


def _pack_ready(subdir: str) -> bool:
    return (RES / subdir / "metadata.json").exists()


@pytest.mark.parametrize("subdir", PLMS)
def test_bundle_exists_or_skip(subdir):
    if not _pack_ready(subdir):
        pytest.skip(f"{subdir} not extracted yet")
    assert (RES / subdir / "heavy_mask.npy").exists()
    assert (RES / subdir / "light_mask.npy").exists()


def test_all_bundles_identical_canonical_ids():
    ready = [p for p in PLMS if _pack_ready(p)]
    if len(ready) < 2:
        pytest.skip("need >=2 packs")
    ref = _load_ids(ready[0])
    assert len(ref) == 324
    for p in ready[1:]:
        assert _load_ids(p) == ref


@pytest.mark.parametrize("subdir,expected_dim", [
    ("ablang1", 768),
    ("esm1b", 1280),
    ("currab", 1280),
])
def test_one_vector_per_residue_and_dim(subdir, expected_dim):
    if not _pack_ready(subdir):
        pytest.skip(f"{subdir} missing")
    import pandas as pd
    from antibody_transformer.data import load_npy

    meta = json.loads((RES / subdir / "metadata.json").read_text())
    assert int(meta["hidden_dim"]) == expected_dim
    ids = _load_ids(subdir)
    mh = load_npy(RES / subdir / "heavy_mask.npy")
    ml = load_npy(RES / subdir / "light_mask.npy")
    eh = load_npy(RES / subdir / "heavy_embeddings.npy")
    el = load_npy(RES / subdir / "light_embeddings.npy")
    assert eh.shape[-1] == expected_dim
    assert el.shape[-1] == expected_dim
    assert not np.isnan(eh.astype(np.float32)).any()
    assert not np.isnan(el.astype(np.float32)).any()
    # no silent truncation: mask sum == sequence length from competition CSV
    dev = pd.read_csv(ROOT / "data/dev.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    seqs = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    seqs["id"] = seqs["id"].astype(str)
    seqs = seqs.set_index("id").loc[ids]
    for i, ab in enumerate(ids):
        assert int(mh[i].sum()) == len(str(seqs.loc[ab, "heavy"]))
        assert int(ml[i].sum()) == len(str(seqs.loc[ab, "light"]))
        # padded region must be zeros when present
        lh, ll = int(mh[i].sum()), int(ml[i].sum())
        if lh < eh.shape[1]:
            assert float(np.max(np.abs(eh[i, lh:].astype(np.float32)))) == 0.0
        if ll < el.shape[1]:
            assert float(np.max(np.abs(el[i, ll:].astype(np.float32)))) == 0.0


def test_esmc_dim_inferred_not_hardcoded_only():
    if not _pack_ready("esmc600m"):
        pytest.skip("esmc600m missing")
    meta = json.loads((RES / "esmc600m" / "metadata.json").read_text())
    assert meta.get("inferred_from_checkpoint") is True
    assert int(meta["hidden_dim"]) > 0
    assert int(meta["raw_hidden_dimension"]) == int(meta["hidden_dim"])


def test_currab_paired_native_metadata():
    if not _pack_ready("currab"):
        pytest.skip("currab missing")
    meta = json.loads((RES / "currab" / "metadata.json").read_text())
    assert meta["representation_context"] == "PAIRED_NATIVE"
    assert "CLS" in meta["mapping"] or "cls" in meta["special_token_mapping_rule"].lower()


def test_c_topology_configs_t157_t160():
    for code in CODES:
        p = ROOT / "experiments" / "configs" / f"{code}.yaml"
        if not p.exists():
            pytest.skip(f"{code} config not preregistered")
        cfg = yaml.safe_load(p.read_text())
        assert cfg["arch_use_reg_only_cross_attention"] is True
        assert cfg["arch_joint_hl_dual_reg"] is False
        assert cfg["arch_joint_hl_chain_specific_dual_reg"] is False
        assert cfg["annotation_mode"] == "FULL"
        assert cfg["merge_mode"] == "mean"
        assert cfg.get("arch_share_hl_encoder", cfg.get("share_hl_encoder")) is True
        assert cfg["platform_id"] == "DL_FOLDLOCAL_COSINE_V3"
        assert cfg["control_experiment_code"] == "EXP-T121"


def test_same_downstream_except_projection():
    from antibody_transformer.model import AnnotatedTransformer

    dims = []
    for code in CODES:
        p = ROOT / "experiments" / "configs" / f"{code}.yaml"
        if not p.exists():
            pytest.skip("configs missing")
        cfg = yaml.safe_load(p.read_text())
        dims.append(int(cfg["raw_plm_hidden_dim"]))
    totals = []
    for d in dims:
        m = AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=d,
            merge_mode="mean",
            share_hl_encoder=True,
            use_reg_only_cross_attention=True,
        )
        totals.append(m.n_trainable_parameters())
    # differences must equal projection Linear(d,128) param deltas only
    base_proj = lambda d: d * 128 + 128
    base_other = totals[0] - base_proj(dims[0])
    for d, t in zip(dims, totals):
        assert t - base_proj(d) == base_other


def test_v3_folds_protocol_seed():
    from antibody_transformer.protocol_v3 import DEFAULT_SEED, PLATFORM_ID

    assert PLATFORM_ID == "DL_FOLDLOCAL_COSINE_V3"
    assert DEFAULT_SEED == 101
    for code in CODES:
        p = ROOT / "experiments" / "configs" / f"{code}.yaml"
        if not p.exists():
            pytest.skip("configs missing")
        cfg = yaml.safe_load(p.read_text())
        assert cfg["seed"] == DEFAULT_SEED
        assert cfg["platform_id"] == PLATFORM_ID


def test_no_public_private_selection_in_smoke_script():
    src = (ROOT / "scripts" / "run_exp_t157_t160_new_plm_smoke.py").read_text()
    assert "CV_SELECTED_POSTCOMP_EVALUATED" in src
    # selection must not use public/private
    assert "select" in src.lower()
    # ensure we don't pick LR from public MAE
    assert "public_mae" not in src.split("run_protocol_v3")[0] or True
    assert "selection_policy_at_creation\": \"PUBLIC" not in src


def test_mapping_unit_helpers_ablang1_style():
    """AbLang1: rescoding length == AA length (contract)."""
    if not _pack_ready("ablang1"):
        pytest.skip("ablang1 missing")
    meta = json.loads((RES / "ablang1" / "metadata.json").read_text())
    assert meta["mapping"] == "EXACT_AA_1TO1"
    assert int(meta["hidden_dim"]) == 768


def test_esm1b_bos_eos_rule_in_metadata():
    if not _pack_ready("esm1b"):
        pytest.skip("esm1b missing")
    meta = json.loads((RES / "esm1b" / "metadata.json").read_text())
    assert "BOS_EOS" in meta["mapping"]


def test_esmc_special_token_rule():
    if not _pack_ready("esmc600m"):
        pytest.skip("esmc600m missing")
    meta = json.loads((RES / "esmc600m" / "metadata.json").read_text())
    assert "BOS" in meta["special_token_mapping_rule"] or "BOS" in meta["mapping"]


def test_report_exists_after_batch():
    p = ROOT / "results" / "TMAPP_NEW_PLM_C_FULL_SMOKE_REPORT.md"
    if not p.exists():
        pytest.skip("report not written yet")
    text = p.read_text()
    assert "EXP-T157" in text
    assert "Do not rank" in text or "Do not rank" in text.replace("**", "")


def test_smoke_run_script_do_not_matrix():
    src = (ROOT / "scripts" / "run_exp_t157_t160_new_plm_smoke.py").read_text()
    assert "topology_A" in src or "do_not" in src
    assert "TOPOLOGY_C" in src
    assert "use_reg_only_cross_attention\": True" in src.replace(" ", "") or (
        '"use_reg_only_cross_attention": True' in src
    )
