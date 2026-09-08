#!/usr/bin/env python3
"""Load and merge feature blocks by id."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_block(name: str) -> pd.DataFrame:
    man = pd.read_csv(ROOT / "feature_manifest.csv")
    row = man.loc[man.block_name == name].iloc[0]
    if not row.file:
        raise FileNotFoundError(f"{name} not included: {row.license_status}")
    return pd.read_parquet(ROOT / row.file)

def merge_blocks(base: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    out = base.copy()
    out["id"] = out["id"].astype(str)
    for b in blocks:
        feat = load_block(b)
        feat["id"] = feat["id"].astype(str)
        out = out.merge(feat, on="id", how="left")
    return out

if __name__ == "__main__":
    seq = pd.read_parquet(ROOT / "data/base_sequences.parquet")
    print(merge_blocks(seq[["id"]], ["SEQ_BASIC"]).shape)
