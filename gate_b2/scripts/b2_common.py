#!/usr/bin/env python3
"""Shared constants/paths for Gate B2."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b2"
B1 = ROOT / "gate_b1"

DATA = GATE / "data"
SPLITS = GATE / "splits"
CACHE = GATE / "cache"
METRICS = GATE / "metrics"
CONFIG = GATE / "config"
REPORTS = GATE / "reports"
LOGS = GATE / "logs"
PREDS = GATE / "predictions"
PLOTS = REPORTS / "plots"

# Reuse B1 frozen tables
B1_DATA = B1 / "data"
B1_CACHE = B1 / "cache"
B1_SCRIPTS = B1 / "scripts"

sys.path.insert(0, str(B1_SCRIPTS))
from b1_common import (  # noqa: E402
    MAX_ASA_TIEN2013,
    AA20,
    HYDROPHOBIC,
    AROMATIC,
    POSITIVE,
    NEGATIVE,
    POLAR,
    KD,
    TARGET_COLS,
    clean_aa,
    pair_hash,
    seq_hash,
    sha256_text,
    sha256_file,
    write_json,
    read_json,
)

# B2 focuses on HIC + TmApp only
B2_TARGETS = {
    "HIC": "hic_rt_min",
    "TmApp": "tm_app_C",
}

# Frozen SASA params (match B1)
PROBE_RADIUS = 1.4
N_POINTS = 100
RASA_SURFACE_THRESHOLD = 0.20  # frozen before scoring
PATCH_CONTACT_A = 8.0  # CA-CA adjacency Å


def ensure_dirs() -> None:
    for p in [
        DATA,
        SPLITS,
        CACHE / "plm",
        CACHE / "structures" / "esmfold_native",
        CACHE / "structures" / "abodybuilder2",
        CACHE / "structures" / "b1_hf_linker",
        CACHE / "structure_features",
        CACHE / "features",
        METRICS,
        CONFIG,
        REPORTS,
        PLOTS,
        LOGS,
        PREDS,
        GATE / "scripts",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_gpu0() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
