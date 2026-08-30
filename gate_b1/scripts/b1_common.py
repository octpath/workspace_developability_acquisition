#!/usr/bin/env python3
"""Shared paths, hashes, constants for Gate B1."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b1"
DATA = GATE / "data"
SPLITS = GATE / "splits"
CACHE = GATE / "cache"
MODELS = GATE / "models"
PREDS = GATE / "predictions"
METRICS = GATE / "metrics"
CONFIG = GATE / "config"
REPORTS = GATE / "reports"
LOGS = GATE / "logs"
PLOTS = REPORTS / "plots"

SRC_JOINED = ROOT / "interim" / "shehata_full_joined.csv"
SRC_XLSX = ROOT / "raw" / "shehata" / "a05" / "mmc2.xlsx"

TARGET_COLS = {
    "PSR": "psr_score",
    "HIC": "hic_rt_min",
    "TmApp": "tm_app_C",
}

PARTICIPANT_LEGAL_ID_COLS = ["antibody_id", "heavy", "light"]
ORGANIZER_ONLY_COLS = [
    "b_cell_subset",
    "vh_germline",
    "vl_germline",
    "label_source",
    "sequence_source",
    "join_confidence",
]

# Tien et al. 2013 / Wilke MaxASA (Å^2) — empirical
# Source: Tien MZ et al., PLoS ONE 2013; commonly cited as Wilke/Tien scale
MAX_ASA_TIEN2013 = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFVW")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQ")

# KD hydrophobicity scale
KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

CHARGE_PH7 = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}


def ensure_dirs() -> None:
    for p in [
        DATA,
        SPLITS,
        CACHE / "plm",
        CACHE / "structures",
        CACHE / "structure_features",
        CACHE / "numbering",
        CACHE / "features",
        MODELS,
        PREDS,
        METRICS,
        CONFIG,
        REPORTS,
        PLOTS,
        LOGS,
        GATE / "scripts",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_hash(seq: str) -> str:
    return sha256_text(seq)[:32]


def pair_hash(heavy: str, light: str) -> str:
    return sha256_text(f"{heavy}|{light}")[:32]


def clean_aa(seq: str) -> str:
    """Remove gap characters and non-AA; uppercase. Documented filtering."""
    if seq is None or (isinstance(seq, float)):
        return ""
    s = str(seq).upper().replace("-", "").replace(".", "").replace("*", "").replace(" ", "")
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)
    return s


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def set_gpu0() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


AA_COMP_FEATURES = [f"aa_{a}" for a in AA20]
