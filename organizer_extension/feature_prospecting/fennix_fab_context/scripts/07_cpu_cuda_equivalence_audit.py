#!/usr/bin/env python3
"""CPU/CUDA Fab-prep equivalence audit (target-blind).

Selects a frozen subset of CUDA-prepared Abs, reprepares on CPU, compares
prep QC metrics and a frozen FeNNix torsion-curvature subset.

Acceptance (frozen in SPEC):
  Spearman(CPU, CUDA) >= 0.95 on matched curvature features
  AND median normalized abs diff <= 0.10
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
CTX = ROOT / "organizer_extension/feature_prospecting/fennix_fab_context"
FAB = ROOT / "organizer_extension/feature_prospecting/fab_reconstruction"
PILOT = set(json.loads((FAB / "structures/pilot_ids.json").read_text()))
PREP_QC = CTX / "FAB_PREP_QC.csv"
OUT = CTX / "cache/cpu_cuda_audit"
OUT.mkdir(parents=True, exist_ok=True)
SPEARMAN_MIN = 0.95
NAD_MAX = 0.10
N_AUDIT = 8


def select_ids():
    qc = pd.read_csv(PREP_QC)
    ok = qc[qc.get("status", qc.columns[0]).astype(str).str.contains("SUCCESS|OK|success", case=False, na=False)]
    if "status" not in qc.columns:
        # fallback: rows with prepared path
        ok = qc.copy()
    # Prefer non-pilot CUDA successes
    if "prep_platform" in ok.columns:
        cuda = ok[ok.prep_platform.astype(str).str.upper().str.contains("CUDA", na=True)]
    else:
        cuda = ok
    ids = [str(x) for x in cuda["id"].tolist() if str(x) not in PILOT]
    # deterministic freeze
    ids = sorted(set(ids))[:N_AUDIT]
    (OUT / "audit_ids.json").write_text(json.dumps(ids, indent=2))
    return ids


def main():
    ap_ids = select_ids()
    print("audit_ids", ap_ids, flush=True)
    if not ap_ids:
        print("NO_IDS", flush=True)
        return 1
    # CPU reprep into quarantine audit dir via env override if supported; else default prep with attempt=audit
    py = ROOT / ".venv_b1/bin/python"
    batch = CTX / "scripts/06_cpu_batch_prep.py"
    cmd = [str(py), str(batch), "--ids", *ap_ids, "--batch-tag", "cpu_cuda_audit", "--attempt", "1"]
    print("RUN", cmd, flush=True)
    r = subprocess.run(cmd)
    (OUT / "cpu_reprep_exit.json").write_text(json.dumps({"exit": r.returncode, "ids": ap_ids}))
    print("cpu_reprep_exit", r.returncode, flush=True)
    # Comparison requires FeNNix curvature on both platforms — deferred to after curvature script
    # For now write protocol freeze reminder
    (OUT / "ACCEPTANCE_CRITERIA.json").write_text(
        json.dumps(
            {
                "spearman_min": SPEARMAN_MIN,
                "median_nad_max": NAD_MAX,
                "n_audit": N_AUDIT,
                "ids": ap_ids,
                "note": "Curvature equivalence to be computed after CPU-prepared structures exist and FeNNix torsion subset is scored.",
            },
            indent=2,
        )
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
