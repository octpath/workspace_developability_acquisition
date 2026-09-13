#!/usr/bin/env python3
"""Tests for annotation modes and T161–T337 factorial prereg invariants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "models"))


def test_annotation_mode_flags():
    from antibody_transformer.model import AnnotatedTransformer

    for mode, imgt, region in [
        ("minimal", False, False),
        ("base", False, False),
        ("imgt", True, False),
        ("region", False, True),
        ("full", True, True),
    ]:
        m = AnnotatedTransformer(content_mode="scratch", annotation_mode=mode, merge_mode="mean")
        assert m.use_imgt is imgt
        assert m.use_region is region
        assert (getattr(m, "imgt_emb", None) is not None) is imgt
        assert (getattr(m, "region_emb", None) is not None) is region


def test_historical_full_unchanged_vs_explicit():
    from antibody_transformer.model import AnnotatedTransformer

    torch.manual_seed(0)
    a = AnnotatedTransformer(content_mode="scratch", annotation_mode="full", merge_mode="mean")
    torch.manual_seed(0)
    b = AnnotatedTransformer(content_mode="scratch", annotation_mode="full", merge_mode="mean")
    # identical init
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)
    assert a.use_imgt and a.use_region


def test_minimal_equals_base_param_structure():
    from antibody_transformer.model import AnnotatedTransformer

    m0 = AnnotatedTransformer(content_mode="scratch", annotation_mode="minimal", merge_mode="mean")
    m1 = AnnotatedTransformer(content_mode="scratch", annotation_mode="base", merge_mode="mean")
    assert m0.n_trainable_parameters() == m1.n_trainable_parameters()
    assert not m0.use_imgt and not m0.use_region


def test_old_full_config_loads():
    cfg = yaml.safe_load((ROOT / "experiments/configs/EXP-T121.yaml").read_text())
    assert cfg["annotation_mode"] == "FULL" or str(cfg["annotation_mode"]).lower() == "full"
    # model accepts lower-case full
    from antibody_transformer.model import AnnotatedTransformer

    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=480,
        annotation_mode="full",
        merge_mode="mean",
        use_reg_only_cross_attention=True,
    )
    assert m.use_imgt and m.use_region


def test_topology_switches_canonical():
    from preregister_t161_t337_factorial import TOPOLOGY_FLAGS

    assert TOPOLOGY_FLAGS["A"]["use_reg_only_cross_attention"] is False
    assert TOPOLOGY_FLAGS["A"]["joint_hl_dual_reg"] is False
    assert TOPOLOGY_FLAGS["B1"]["joint_hl_dual_reg"] is True
    assert TOPOLOGY_FLAGS["B2"]["joint_hl_chain_specific_dual_reg"] is True
    assert TOPOLOGY_FLAGS["C"]["use_reg_only_cross_attention"] is True
    assert TOPOLOGY_FLAGS["C"].get("reg_cross_variant") in (None, "null")
    assert TOPOLOGY_FLAGS["D"]["pair_interaction_mode"] == "d3_symmetric_mlp"
    # no C1-C5 / D1/D2/D4 leakage in frozen flags
    for t, f in TOPOLOGY_FLAGS.items():
        assert f.get("reg_cross_variant") in (None, "null")
        if t != "D":
            assert f.get("pair_interaction_mode") in (None, "null")


def test_factorial_plan_if_present():
    p = ROOT / "results/TMAPP_REP_TOPO_ANNOT_FACTORIAL_PLAN.csv"
    if not p.exists():
        return
    import pandas as pd

    df = pd.read_csv(p)
    assert len(df) == 200
    assert (df.execution_status == "REUSE").sum() == 23
    assert set(df.execution_status.unique()) <= {"REUSE", "PLANNED", "COMPLETE", "FAILED", "BLOCKED", "BLOCKED_MAPPING"}
    assert (df.execution_status.isin(["REUSE", "COMPLETE", "FAILED", "BLOCKED", "BLOCKED_MAPPING", "PLANNED"])).all()
    n_new = (df.execution_status != "REUSE").sum()
    assert n_new == 177
    assert df.duplicated(["representation", "topology", "annotation"]).sum() == 0
    assert df.duplicated("experiment_code").sum() == 0
    t210 = df[df.experiment_code == "EXP-T210"].iloc[0]
    assert t210.representation == "ablang2_unpaired" and t210.topology == "C" and t210.annotation == "FULL"
    t321 = df[df.experiment_code == "EXP-T321"].iloc[0]
    assert t321.representation == "currab_unpaired" and t321.topology == "C" and t321.annotation == "FULL"
    planned = df[df.execution_status == "PLANNED"]["experiment_code"].tolist()
    if planned:
        nums = sorted(int(c.split("-T")[1]) for c in planned)
        # when pristine: 161..337
        if len(nums) == 177:
            assert nums == list(range(161, 338))
    complete_new = df[df.execution_status == "COMPLETE"]
    if len(complete_new) == 177:
        assert (df.execution_status.isin(["REUSE", "COMPLETE"])).all()
        # spot-check V3 on a new cell config
        cfg = yaml.safe_load((ROOT / "experiments/configs/EXP-T210.yaml").read_text())
        assert cfg.get("platform_id") == "DL_FOLDLOCAL_COSINE_V3" or "V3" in str(cfg.get("platform_id", ""))
        assert int(cfg.get("seed", 101)) == 101


def test_matched_assets_if_present():
    for sub, ctx in (("ablang2_unpaired", "PAIRED_NATIVE"), ("currab_unpaired", "SEPARATE_CHAIN")):
        meta_p = REPO / "top_models_feature_bundle/residue_level" / sub / "metadata.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        assert meta["representation_context"] == ctx
        assert meta.get("matched_to")
        chk = json.loads((meta_p.parent / "REEXTRACT_CHECK.json").read_text())
        assert chk["pass"] is True


def test_currab_same_revision():
    paired = REPO / "top_models_feature_bundle/residue_level/currab/metadata.json"
    unp = REPO / "top_models_feature_bundle/residue_level/currab_unpaired/metadata.json"
    if not (paired.exists() and unp.exists()):
        return
    a = json.loads(paired.read_text())
    b = json.loads(unp.read_text())
    assert a["model_revision"] == b["model_revision"] == "92e28534663e163f1b398f773b3fe041085737d9"


def test_ablang2_same_checkpoint_identity():
    paired = REPO / "top_models_feature_bundle/residue_level/ablang2/metadata.json"
    unp = REPO / "top_models_feature_bundle/residue_level/ablang2_unpaired/metadata.json"
    if not (paired.exists() and unp.exists()):
        return
    a = json.loads(paired.read_text())
    b = json.loads(unp.read_text())
    assert a.get("model_to_use") == "ablang2-paired"
    assert b.get("model_to_use") == "ablang2-paired"
