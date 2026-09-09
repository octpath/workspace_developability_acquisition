#!/usr/bin/env python3
"""Smoke runner for a Transformer experiment config (no historical full retraining).

Usage:
  python scripts/run_transformer_experiment.py --config experiments/configs/EXP-T026.yaml --smoke

Does not launch the 29 historical training jobs. Future representation aggregation
across seeds is UNDECIDED — this runner does not export representations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from antibody_transformer.config import load_presets  # noqa: E402
from antibody_transformer.equivalence import (  # noqa: E402
    compare_forward,
    make_pair,
    synthetic_batch,
)
from antibody_transformer.model import AnnotatedTransformer  # noqa: E402


def smoke_from_config(cfg: dict) -> None:
    content = "scratch" if cfg.get("plm_source") == "NONE" else "frozen"
    plm_h = {"ABLINGUA": 1280, "ESM2": 1280, "ABLANG2": 480, "NONE": 0}[cfg["plm_source"]]
    merge = cfg.get("merge_mode", "concat")
    pooling = "region_gate" if str(cfg.get("pooling_mode", "")).upper() == "REGION_GATE" else "reg"
    chain = cfg.get("chain_mode", "HL")
    ann = "minimal" if str(cfg.get("annotation_mode", "")).upper().startswith("MIN") else "full"
    old, new = make_pair(
        content_mode=content,
        annotation_mode=ann,
        merge_mode=merge,
        chain_mode=chain,
        pooling_mode=pooling,
        plm_hidden=plm_h or 1280,
        fusion=False,
        seed=0,
    )
    batch = synthetic_batch(
        content_mode=content,
        chain_mode=chain,
        annotation_mode=ann,
        plm_hidden=plm_h or 1280,
    )
    stats = compare_forward(old, new, batch)
    print("smoke forward equivalence:", stats)
    if max(stats.values()) > 1e-6:
        raise SystemExit("smoke equivalence failed")
    # instantiate config-shaped model once
    ncfg = load_presets()["neural"]
    _ = AnnotatedTransformer(
        content_mode=content,
        plm_hidden=plm_h if content == "frozen" else 0,
        annotation_mode=ann,
        merge_mode=merge if chain != "H_ONLY" else "h_only",
        chain_mode=chain,
        pooling_mode=pooling,
        d_model=ncfg["d_model"],
        n_heads=ncfg["n_heads"],
        n_layers=ncfg["n_layers"],
        dim_feedforward=ncfg["dim_feedforward"],
        dropout=ncfg["dropout"],
        norm_first=ncfg["norm_first"],
    )
    print("OK config", cfg["experiment_code"], cfg["experiment_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="unit smoke only; no training")
    args = ap.parse_args()
    if not args.smoke:
        print("Refusing full training in Phase 2A. Pass --smoke.", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(Path(args.config).read_text())
    smoke_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
