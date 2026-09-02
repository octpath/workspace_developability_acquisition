#!/usr/bin/env python3
"""Generate Boltz-2 YAML inputs for competition N=324 (VH+VL only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/structure_sources/boltz2_fv_standard_v1"
INP = OUT / "inputs"

DEV = ROOT / "competition/data/distribution/dev.csv"
TEST = ROOT / "competition/data/distribution/test_features.csv"


def main() -> None:
    INP.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(DEV)[["id", "heavy", "light"]], pd.read_csv(TEST)[["id", "heavy", "light"]]]
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 324
    for _, r in df.iterrows():
        yaml = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: H\n"
            f"      sequence: {r['heavy']}\n"
            "  - protein:\n"
            "      id: L\n"
            f"      sequence: {r['light']}\n"
        )
        (INP / f"{r['id']}.yaml").write_text(yaml)
    print(f"wrote {len(df)} yaml files to {INP}")


if __name__ == "__main__":
    main()
