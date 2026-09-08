#!/usr/bin/env python3
"""Run thermal microprobe for exactly one antibody in a fresh process."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

OUT = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/fennix_thermal_microprobe")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: _worker_one_ab.py ADI-xxxxx")
    aid = sys.argv[1]
    spec = importlib.util.spec_from_file_location("run", OUT / "scripts/01_run_dev_microprobe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.setup_cuda_env()
    import fennol
    import pandas as pd

    model = fennol.load(str(mod.MODEL))
    seq = pd.read_csv(mod.SEQ).set_index("id")
    feat = mod.process_one(aid, seq.loc[aid], model)
    print(json.dumps({"id": aid, "status": feat.get("status"), "wall_s": feat.get("wall_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
