#!/usr/bin/env python3
"""One-Ab worker for OpenMM Fab MD (fresh process)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

OUT = Path("/workspace_developability_acquisition/organizer_extension/feature_prospecting/openmm_fab_md")


def main():
    aid = sys.argv[1]
    n_prod = int(sys.argv[2])
    spec = importlib.util.spec_from_file_location("run", OUT / "scripts/01_run_dev_openmm_md.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import pandas as pd

    seq = pd.read_csv(mod.SEQ).set_index("id")
    feat = mod.process_one(aid, seq.loc[aid], n_prod)
    print(json.dumps({"id": aid, "ok": True, "keys": len(feat)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
