#!/usr/bin/env python3
"""Validate advanced-model environment and residue assets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import numpy as np
    import pandas as pd

    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"torch: MISSING ({e})")
        return 1
    try:
        import xgboost as xgb

        print(f"xgboost: {xgb.__version__}")
    except Exception as e:
        print(f"xgboost: MISSING ({e})")
        return 1

    dev = pd.read_csv(ROOT / "dev.csv")
    test = pd.read_csv(ROOT / "test.csv")
    print(f"DEV N={len(dev)} TEST N={len(test)}")

    residue = ROOT / "residue_level"
    emb_paths = (
        residue / "ablingua600m/heavy_embeddings.npy",
        residue / "ablingua600m/light_embeddings.npy",
        residue / "esm2/heavy_embeddings.npy",
    )
    if any(not p.exists() for p in emb_paths):
        print(
            "residue embeddings missing; reassemble with:\n"
            "  bash top_models_feature_bundle/residue_level/assemble_embeddings.sh"
        )
    for p in (residue / "annotations.parquet", *emb_paths):
        print(f"residue asset: {p.relative_to(ROOT)} {'FOUND' if p.exists() else 'MISSING'}")
    if (residue / "ablingua600m/ids.npy").exists():
        ids = np.load(residue / "ablingua600m/ids.npy", allow_pickle=True)
        print(f"AbLingua residue IDs: {len(ids)}")
    if (residue / "esm2/ids.npy").exists():
        ids = np.load(residue / "esm2/ids.npy", allow_pickle=True)
        print(f"ESM-2 residue IDs: {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
