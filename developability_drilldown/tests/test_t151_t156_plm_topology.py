#!/usr/bin/env python3
"""Tests for T151–T156 PLM × topology matrix freeze / audit / no-retrain."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


def test_v2_freeze_five_topologies():
    doc = yaml.safe_load((ROOT / "results/TMAPP_HL_TOPOLOGY_FREEZE_V2.yaml").read_text())
    assert set(doc["topologies"]) == {"A", "B1", "B2", "C", "D"}
    assert doc["topologies"]["C"]["flags"]["use_reg_only_cross_attention"] is True
    assert doc["topologies"]["C"]["flags"].get("reg_cross_variant") in (None, "null")
    assert doc["topologies"]["D"]["flags"]["pair_interaction_mode"] == "d3_symmetric_mlp"
    assert doc["topologies"]["B1"]["flags"]["joint_hl_dual_reg"] is True
    assert doc["topologies"]["B2"]["flags"]["joint_hl_chain_specific_dual_reg"] is True
    assert doc["historical_abcd_freeze_unchanged"].endswith("TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml")


def test_historical_abcd_freeze_untouched():
    p = ROOT / "results/TMAPP_ABCD_ARCHITECTURE_FREEZE.yaml"
    assert p.exists()
    doc = yaml.safe_load(p.read_text())
    assert doc["selection"]["C_star"]["code"] == "EXP-T121"
    assert doc["selection"]["D_star"]["code"] == "EXP-T149"


def test_historical_ids_map():
    from run_exp_t151_t156_plm_topology import MATRIX

    reuse = {(c["plm"], c["topology"]): c["code"] for c in MATRIX if c["reuse"]}
    assert reuse[("ablingua", "A")] == "EXP-T080"
    assert reuse[("ablingua", "B1")] == "EXP-T081"
    assert reuse[("ablingua", "B2")] == "EXP-T082"
    assert reuse[("ablingua", "C")] == "EXP-T087"
    assert reuse[("ablang2", "A")] == "EXP-T110"
    assert reuse[("ablang2", "B1")] == "EXP-T113"
    assert reuse[("ablang2", "B2")] == "EXP-T115"
    assert reuse[("ablang2", "C")] == "EXP-T121"
    assert reuse[("ablang2", "D")] == "EXP-T149"


def test_no_silent_retrain_new_only():
    from run_exp_t151_t156_plm_topology import MATRIX

    new = [c for c in MATRIX if not c["reuse"]]
    assert [c["code"] for c in new] == [
        "EXP-T151",
        "EXP-T152",
        "EXP-T153",
        "EXP-T154",
        "EXP-T155",
        "EXP-T156",
    ]


def test_audit_pass():
    text = (ROOT / "results/TMAPP_PLM_TOPOLOGY_MATRIX_AUDIT.md").read_text()
    assert "PASS" in text
    assert "0 mismatches" in text or "mask alignment" in text.lower()


def test_prereg_exists():
    doc = yaml.safe_load((ROOT / "results/TMAPP_PLM_TOPOLOGY_MATRIX_PREREGISTRATION.yaml").read_text())
    assert doc["status"] == "PREREGISTERED_FROZEN"
    assert len(doc["cells"]) == 15
    new = [c for c in doc["cells"] if not c["reuse"]]
    assert len(new) == 6


def test_topology_semantics_flags():
    from antibody_transformer.model import AnnotatedTransformer
    from run_exp_t151_t156_plm_topology import TOPOLOGY_FLAGS

    # A: no cross
    m = AnnotatedTransformer(content_mode="scratch", merge_mode="mean", **{
        k: TOPOLOGY_FLAGS["A"][k]
        for k in (
            "joint_hl_dual_reg",
            "joint_hl_chain_specific_dual_reg",
            "use_reg_only_cross_attention",
            "pair_interaction_mode",
        )
    })
    assert m.cross_attn is None
    assert m.pair_module is None
    # B1 joint
    m = AnnotatedTransformer(
        content_mode="scratch",
        merge_mode="mean",
        joint_hl_dual_reg=True,
    )
    assert m.joint_hl_dual_reg
    # B2
    m = AnnotatedTransformer(
        content_mode="scratch",
        merge_mode="mean",
        joint_hl_chain_specific_dual_reg=True,
    )
    assert m.joint_hl_chain_specific_dual_reg
    # C
    m = AnnotatedTransformer(
        content_mode="scratch",
        merge_mode="mean",
        use_reg_only_cross_attention=True,
    )
    assert m.cross_attn is not None
    assert m.reg_cross_variant is None
    # D
    m = AnnotatedTransformer(
        content_mode="scratch",
        merge_mode="mean",
        pair_interaction_mode="d3_symmetric_mlp",
    )
    assert m.pair_module is not None
    zh = torch.randn(2, 128)
    zl = torch.randn(2, 128)
    assert torch.allclose(m.pair_module(zh, zl), m.pair_module(zl, zh), atol=1e-5)


def test_esm2_alignment_quick():
    from antibody_transformer.data import load_dev_test, load_residue_bundle

    dev, test = load_dev_test(ROOT / "data/dev.csv", ROOT / "data/test.csv")
    rb = load_residue_bundle(dev, test, need_esm2=True)
    assert rb.esm2_h is not None and rb.esm2_l is not None
    assert rb.esm2_hidden == 1280
    for i in range(len(rb.ids)):
        assert int(rb.esm2_h_mask[i].sum()) == int(rb.heavy_mask[i].sum())
        assert int(rb.esm2_l_mask[i].sum()) == int(rb.light_mask[i].sum())


def test_next_code_gate():
    from experiment_codes import load_codes, next_code

    codes = set(load_codes()["experiment_code"].astype(str))
    if "EXP-T156" in codes:
        assert next_code("TmApp") == "EXP-T161"
    else:
        assert next_code("TmApp") == "EXP-T151"
