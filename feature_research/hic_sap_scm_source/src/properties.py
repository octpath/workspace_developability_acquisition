"""SOURCE_CONFIRMED property tables for SAP (KD) and SCM (formal charge)."""
from __future__ import annotations

import hashlib
import json

AA20 = list("ARNDCQEGHILKMFPSTWYV")

# Kyte & Doolittle 1982 — SOURCE_CONFIRMED / EXISTING_REPO_CONFIRMED
KD_RAW: dict[str, float] = {
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

_KD_MIN = min(KD_RAW[a] for a in AA20)
_KD_MAX = max(KD_RAW[a] for a in AA20)
KD_NORM: dict[str, float] = {
    a: (KD_RAW[a] - _KD_MIN) / (_KD_MAX - _KD_MIN) for a in AA20
}

# Formal residue charge — SOURCE_CONFIRMED
CHARGE: dict[str, float] = {a: 0.0 for a in AA20}
CHARGE["R"] = 1.0
CHARGE["K"] = 1.0
CHARGE["D"] = -1.0
CHARGE["E"] = -1.0


def table_hash(values: dict[str, float]) -> str:
    blob = json.dumps({a: float(values[a]) for a in AA20}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


KD_RAW_HASH = table_hash(KD_RAW)
KD_NORM_HASH = table_hash(KD_NORM)
CHARGE_HASH = table_hash(CHARGE)

# Tien MaxASA (repo structure_utils.MAX_ASA) — EXISTING_REPO_CONFIRMED
TIEN_MAXASA: dict[str, float] = {
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
TIEN_MAXASA_HASH = table_hash(TIEN_MAXASA)
