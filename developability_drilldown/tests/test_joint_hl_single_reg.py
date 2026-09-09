"""Tests for joint H/L single-REG (EXP-T068 / EXP-T069)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from antibody_transformer.equivalence import synthetic_batch  # noqa: E402
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_joint_token_layout_reg_h_l():
    """encode_joint_hl builds [REG, H..., L...] length 1+Lh+Ll."""
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=64,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        d_model=32,
        n_heads=4,
        dim_feedforward=64,
    )
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=64,
        Lh=10,
        Ll=8,
        B=2,
    )
    seen = {}

    class _Cap(torch.nn.Module):
        def forward(self, x, src_key_padding_mask=None, **kwargs):
            seen["T"] = x.shape[1]
            seen["pad"] = src_key_padding_mask
            return x

    m.encoder = _Cap()
    m.eval()
    with torch.no_grad():
        out = m.encode_joint_hl(batch)
    assert seen["T"] == 1 + 10 + 8
    assert out.shape == (2, 32)
    pad = seen["pad"]
    assert pad.shape == (2, 19)
    assert bool((~pad[:, 0]).all())
    assert torch.equal(pad[:, 1:11], ~batch["heavy_mask"])
    assert torch.equal(pad[:, 11:], ~batch["light_mask"])


def test_single_reg_no_chain_emb_on_reg():
    m = AnnotatedTransformer(
        content_mode="scratch",
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        d_model=16,
        n_heads=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    assert m.single_reg_token is not None
    assert m.reg_token is None
    B = 2
    reg = m.single_reg_token.view(1, 1, -1).expand(B, 1, -1)
    assert torch.allclose(reg[0, 0], m.single_reg_token)
    assert m.head[0].in_features == 16


def test_per_chain_pos_restart_independent():
    """H and L both use pos starting at 1 (independent restart)."""
    batch = synthetic_batch(
        content_mode="scratch",
        chain_mode="HL",
        annotation_mode="full",
        Lh=6,
        Ll=5,
        B=1,
    )
    assert int(batch["heavy_pos"][0, 0]) == 1
    assert int(batch["light_pos"][0, 0]) == 1
    assert int(batch["heavy_pos"][0, 5]) == 6
    assert int(batch["light_pos"][0, 4]) == 5

    m = AnnotatedTransformer(
        content_mode="scratch",
        annotation_mode="minimal",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        d_model=16,
        n_heads=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    p1 = m.pos_emb(torch.tensor([1]))
    assert torch.allclose(m.pos_emb(batch["heavy_pos"][0, :1]), p1)
    assert torch.allclose(m.pos_emb(batch["light_pos"][0, :1]), p1)


def test_padding_masks_joint():
    m = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=32,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        d_model=16,
        n_heads=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=32,
        Lh=8,
        Ll=6,
        B=3,
    )
    batch["heavy_mask"] = torch.ones(3, 8, dtype=torch.bool)
    batch["light_mask"] = torch.ones(3, 6, dtype=torch.bool)
    batch["heavy_mask"][:, -2:] = False
    batch["light_mask"][:, -1:] = False
    captured = {}

    class _Cap(torch.nn.Module):
        def forward(self, x, src_key_padding_mask=None, **kwargs):
            captured["pad"] = src_key_padding_mask.clone()
            return torch.zeros_like(x)

    m.encoder = _Cap()
    m.eval()
    with torch.no_grad():
        m.encode_joint_hl(batch)
    pad = captured["pad"]
    assert pad[:, 0].sum() == 0
    assert int(pad[:, 1:9].sum()) == 3 * 2
    assert int(pad[:, 9:].sum()) == 3 * 1
    assert torch.equal(pad[:, 1:9], ~batch["heavy_mask"])
    assert torch.equal(pad[:, 9:], ~batch["light_mask"])


def test_zero_bias_equivalence_joint_ca():
    torch.manual_seed(1)
    base = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=64,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        use_ca_distance_bias=False,
        d_model=32,
        n_heads=4,
        dim_feedforward=64,
        dropout=0.0,
    )
    torch.manual_seed(1)
    dist = AnnotatedTransformer(
        content_mode="frozen",
        plm_hidden=64,
        annotation_mode="full",
        merge_mode="concat",
        chain_mode="HL",
        pooling_mode="reg",
        joint_hl_single_reg=True,
        use_ca_distance_bias=True,
        initial_ell_angstrom=9.0,
        d_model=32,
        n_heads=4,
        dim_feedforward=64,
        dropout=0.0,
    )
    sd = base.state_dict()
    mapped = {
        k: v
        for k, v in sd.items()
        if k in dist.state_dict() and dist.state_dict()[k].shape == v.shape
    }
    dist.load_state_dict(mapped, strict=False)
    assert torch.all(dist.encoder.a == 0)

    batch = synthetic_batch(
        content_mode="frozen",
        chain_mode="HL",
        annotation_mode="full",
        plm_hidden=64,
        Lh=7,
        Ll=5,
        B=2,
    )
    B, Lh = batch["heavy_plm"].shape[:2]
    Ll = batch["light_plm"].shape[1]
    batch["heavy_ca"] = torch.randn(B, Lh, 3)
    batch["light_ca"] = torch.randn(B, Ll, 3)
    base.eval()
    dist.eval()
    with torch.no_grad():
        r0 = base.forward_repr(batch)
        r1 = dist.forward_repr(batch)
        y0 = base(batch)
        y1 = dist(batch)
    assert float((r0 - r1).abs().max()) < 1e-5
    assert float((y0 - y1).abs().max()) < 1e-5


def test_no_fusion_recipe_in_joint_configs():
    for code, space in (
        ("EXP-T068", "JOINT_HL_SINGLE_REG_FROZEN_RESIDUE"),
        ("EXP-T069", "JOINT_HL_SINGLE_REG_FROZEN_RESIDUE_PLUS_CA_DISTANCE"),
    ):
        cfg_path = ROOT / "experiments" / "configs" / f"{code}.yaml"
        if not cfg_path.exists():
            continue
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg.get("fusion_feature_set_id") in (None, "", False) or "fusion_feature_set_id" not in cfg
        assert cfg.get("recipe_id") in (None, "", False)
        assert cfg.get("joint_hl_single_reg") is True
        assert space in str(cfg.get("input_space") or "")
        feat = ROOT / "experiments" / "features" / f"{code}.parquet"
        assert not feat.exists(), f"joint HL must not create fusion parquet: {feat}"


def test_t068_t069_artifacts_if_present():
    for code, freeze_name in (
        ("EXP-T068", "EXP-T068_CV_FREEZE.yaml"),
        ("EXP-T069", "EXP-T069_CV_FREEZE.yaml"),
    ):
        freeze = ROOT / "results" / freeze_name
        pred_p = ROOT / "experiments" / "predictions" / code / "oof_primary.csv"
        pred_s = ROOT / "experiments" / "predictions" / code / "oof_shadow.csv"
        if not (freeze.exists() and pred_p.exists() and pred_s.exists()):
            continue
        import yaml

        fr = yaml.safe_load(freeze.read_text())
        assert code in fr
        dev = pd.read_csv(ROOT / "data" / "dev.csv")
        y = dev.set_index("id")["TmApp"]
        for path, score_key in (
            (pred_p, "cv_primary_mae"),
            (pred_s, "cv_shadow_mae"),
        ):
            p = pd.read_csv(path)
            p["id"] = p["id"].astype(str)
            col = "TmApp" if "TmApp" in p.columns else [c for c in p.columns if c != "id"][0]
            ids = [str(i) for i in dev["id"]]
            mae = float((y.loc[ids] - p.set_index("id").loc[ids, col]).abs().mean())
            assert abs(mae - float(fr[code][score_key])) < 1e-10


def test_experiment_runs_has_t030_replay():
    runs = ROOT / "results" / "experiment_runs.csv"
    assert runs.exists()
    df = pd.read_csv(runs)
    assert "EXP-T030-REPLAY-001" in set(df["run_id"].astype(str))
    row = df[df["run_id"] == "EXP-T030-REPLAY-001"].iloc[0]
    assert str(row["experiment_code"]) == "EXP-T030"
    assert (
        ROOT / "experiments" / "replays" / "EXP-T030" / "EXP-T030-REPLAY-001" / "oof_primary.csv"
    ).exists()


def test_historical_t030_scores_unchanged():
    """Pinned historical EXP-T030 registry scores (must not be overwritten by T068/T069)."""
    exp = pd.read_csv(ROOT / "results" / "experiments.csv")
    r = exp[exp["experiment_code"] == "EXP-T030"].iloc[0]
    assert abs(float(r["cv_primary_mae"]) - 3.3177692899978704) < 1e-12
    assert abs(float(r["cv_shadow_mae"]) - 3.214610237152979) < 1e-12
    assert abs(float(r["public_mae"]) - 3.5955287792064525) < 1e-12
    assert abs(float(r["private_mae"]) - 3.255966657473717) < 1e-12
    assert str(r["oof_primary_path"]) == "experiments/predictions/EXP-T030/oof_primary.csv"
    hist = ROOT / "experiments" / "predictions" / "EXP-T030" / "oof_primary.csv"
    assert hist.exists()
    assert len(_file_sha256(hist)) == 64


def test_joint_rejects_rasa_and_non_reg():
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=32,
            joint_hl_single_reg=True,
            use_continuous_rasa=True,
            pooling_mode="reg",
            chain_mode="HL",
        )
    with pytest.raises(ValueError):
        AnnotatedTransformer(
            content_mode="frozen",
            plm_hidden=32,
            joint_hl_single_reg=True,
            pooling_mode="region_gate",
            chain_mode="HL",
        )
