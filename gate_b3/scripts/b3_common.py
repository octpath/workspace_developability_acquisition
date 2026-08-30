#!/usr/bin/env python3
"""Shared paths/constants for Gate B3."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b3"
B1 = ROOT / "gate_b1"
B2 = ROOT / "gate_b2"

CONFIG = GATE / "config"
FROZEN = GATE / "frozen"
ORG = FROZEN / "organizer"
PART = FROZEN / "participant_staging"
CACHE = GATE / "cache"
FEATURES = GATE / "features"
MODELS = GATE / "models"
PREDS = GATE / "predictions"
METRICS = GATE / "metrics"
SCRIPTS = GATE / "scripts"
LOGS = GATE / "logs"
REPORTS = GATE / "reports"

B1_DATA = B1 / "data"
B1_CACHE = B1 / "cache"
B2_CACHE = B2 / "cache"

TARGET_HIC = "HIC"
TARGET_TMAPP = "TmApp"
COL_HIC = "HIC"
COL_TMAPP = "TmApp"

# Frozen SASA (match B1/B2)
PROBE_RADIUS = 1.4
N_POINTS = 100
RASA_SURFACE_THRESHOLD = 0.20

sys.path.insert(0, str(B1 / "scripts"))
from b1_common import (  # noqa: E402
    MAX_ASA_TIEN2013,
    HYDROPHOBIC,
    AROMATIC,
    POSITIVE,
    NEGATIVE,
    POLAR,
    KD,
    AA20,
    pair_hash,
    sha256_text,
    sha256_file,
    write_json,
    read_json,
    clean_aa,
)


def ensure_dirs():
    for p in [
        CONFIG, ORG, PART, CACHE, FEATURES, MODELS, PREDS, METRICS, SCRIPTS, LOGS, REPORTS,
        CACHE / "plm", CACHE / "structures", FEATURES / "imgt", FEATURES / "germline",
        FEATURES / "ngram", FEATURES / "structure_ext", PREDS / "oof", PREDS / "final",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def set_gpu0():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sha256_lines(ids) -> str:
    blob = "\n".join(sorted(map(str, ids))) + "\n"
    return hashlib.sha256(blob.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
