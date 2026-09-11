#!/usr/bin/env python3
"""Verified hydrophobicity scales for HSP atlas (target-blind).

A scale is accepted only with exact 20-AA values from Biopython ProtParamData,
AAIndex1, or a repository-validated artifact. UNAVAILABLE scales are recorded,
not invented.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

AA20 = list("ARNDCQEGHILKMFPSTWYV")


@dataclass(frozen=True)
class HydroScale:
    scale_id: str
    name: str
    source: str
    values: dict[str, float]  # oriented: larger = more hydrophobic
    direction_note: str
    units: str
    zero_meaning: str
    status: str = "VERIFIED"  # or UNAVAILABLE


def _orient(values: dict[str, float], *, larger_is_hydrophobic: bool) -> dict[str, float]:
    if larger_is_hydrophobic:
        return {a: float(values[a]) for a in AA20}
    return {a: float(-values[a]) for a in AA20}


def _hash_table(values: dict[str, float]) -> str:
    blob = json.dumps({a: values[a] for a in AA20}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --- Verified tables ---

# Kyte & Doolittle 1982 (Biopython kd; matches developability STATIC_SAP_KD)
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Black & Mould 1991 0–1 (Biopython bm; matches structure_utils.BLACK_MOULD_01)
_BM = {
    "A": 0.616, "R": 0.0, "N": 0.236, "D": 0.028, "C": 0.68, "Q": 0.251, "E": 0.043,
    "G": 0.501, "H": 0.165, "I": 0.943, "L": 0.943, "K": 0.283, "M": 0.738, "F": 1.0,
    "P": 0.711, "S": 0.359, "T": 0.45, "W": 0.878, "Y": 0.88, "V": 0.825,
}

# Eisenberg normalized consensus 1984 (Biopython es)
_EIS = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.9, "C": 0.29, "Q": -0.85, "E": -0.74,
    "G": 0.48, "H": -0.4, "I": 1.38, "L": 1.06, "K": -1.5, "M": 0.64, "F": 1.19,
    "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Miyazawa contact-energy hydrophobicity 1985 (Biopython mi)
_MIY = {
    "A": 5.33, "R": 4.18, "N": 3.71, "D": 3.59, "C": 7.93, "Q": 3.87, "E": 3.65,
    "G": 4.48, "H": 5.1, "I": 8.83, "L": 8.47, "K": 2.95, "M": 8.95, "F": 9.03,
    "P": 3.87, "S": 4.09, "T": 4.49, "W": 7.66, "Y": 5.89, "V": 7.63,
}

# Meek 1980 HPLC retention pH7.4 (AAIndex MEEJ800101)
_MEEK = {
    "A": 0.5, "R": 0.8, "N": 0.8, "D": -8.2, "C": -6.8, "Q": -4.8, "E": -16.9,
    "G": 0.0, "H": -3.5, "I": 13.9, "L": 8.8, "K": 0.1, "M": 4.8, "F": 13.2,
    "P": 6.1, "S": 1.2, "T": 2.7, "W": 14.9, "Y": 6.1, "V": 2.7,
}

# Wimley–White 1996 interface→water ΔG (AAIndex WIMW960101); larger = more hydrophobic
_WW = {
    "A": 4.08, "R": 3.91, "N": 3.83, "D": 3.02, "C": 4.49, "Q": 3.67, "E": 2.23,
    "G": 4.24, "H": 4.08, "I": 4.52, "L": 4.81, "K": 3.77, "M": 4.48, "F": 5.38,
    "P": 3.80, "S": 4.12, "T": 4.11, "W": 6.10, "Y": 5.19, "V": 4.18,
}

# Fauchère–Pliska π (Biopython fc; HYDRO_FIELD repo artifact) — extra verified scale
_FP = {
    "A": 0.31, "R": -1.01, "N": -0.6, "D": -0.77, "C": 1.54, "Q": -0.22, "E": -0.64,
    "G": 0.0, "H": 0.13, "I": 1.8, "L": 1.7, "K": -0.99, "M": 1.23, "F": 1.79,
    "P": 0.72, "S": -0.04, "T": 0.26, "W": 2.25, "Y": 0.96, "V": 1.22,
}


SCALES: dict[str, HydroScale] = {
    "KD": HydroScale(
        "KD",
        "Kyte-Doolittle",
        "Kyte & Doolittle JMB 1982; Bio.SeqUtils.ProtParamData.kd; STATIC_SAP_KD",
        _orient(_KD, larger_is_hydrophobic=True),
        "native: larger = more hydrophobic",
        "dimensionless GRAVY index",
        "relative to arbitrary zero near Gly",
    ),
    "BM": HydroScale(
        "BM",
        "Black-Mould",
        "Black & Mould Anal Biochem 1991; Bio.SeqUtils.ProtParamData.bm; structure_utils.BLACK_MOULD_01",
        _orient(_BM, larger_is_hydrophobic=True),
        "native 0–1: larger = more hydrophobic",
        "normalized 0–1",
        "Arg=0 most hydrophilic; Phe=1 most hydrophobic",
    ),
    "WW": HydroScale(
        "WW",
        "Wimley-White INTERFACE",
        "Wimley & White Nat Struct Biol 1996; AAIndex WIMW960101 (interface→water ΔG)",
        _orient(_WW, larger_is_hydrophobic=True),
        "ΔG interface→water: larger = more hydrophobic",
        "kcal/mol",
        "relative transfer free energy",
    ),
    "EIS": HydroScale(
        "EIS",
        "Eisenberg consensus",
        "Eisenberg et al. JMB 1984; Bio.SeqUtils.ProtParamData.es",
        _orient(_EIS, larger_is_hydrophobic=True),
        "native: larger = more hydrophobic",
        "normalized consensus",
        "consensus zero",
    ),
    "MEEK": HydroScale(
        "MEEK",
        "Meek HPLC pH7.4",
        "Meek PNAS 1980; AAIndex MEEJ800101",
        _orient(_MEEK, larger_is_hydrophobic=True),
        "HPLC retention: larger = more hydrophobic",
        "retention coefficient",
        "Gly≈0 reference-ish",
    ),
    "MIY": HydroScale(
        "MIY",
        "Miyazawa",
        "Miyazawa & Jernigan Macromolecules 1985; Bio.SeqUtils.ProtParamData.mi",
        _orient(_MIY, larger_is_hydrophobic=True),
        "contact energy: larger = more hydrophobic",
        "contact energy units",
        "scale-specific",
    ),
    "FP": HydroScale(
        "FP",
        "Fauchere-Pliska",
        "Fauchère & Pliska 1983; Bio.SeqUtils.ProtParamData.fc; hydro_surface.FAUCHERE_PI",
        _orient(_FP, larger_is_hydrophobic=True),
        "π: larger = more hydrophobic",
        "π (octanol)",
        "Gly=0",
    ),
}

UNAVAILABLE = {
    "JAIN_HIC": {
        "reason": "No verified 20-AA hydrophobicity scale named 'Jain HIC' found; Jain 2017 is assay data, not a residue hydrophobicity table. Janin burial scale is a different quantity and was not substituted.",
        "status": "UNAVAILABLE",
    },
}

# Black–Mould Gly-centered (SAP literature semantics)
BM_SAP = {a: float(SCALES["BM"].values[a] - SCALES["BM"].values["G"]) for a in AA20}


def property_raw(scale_id: str) -> dict[str, float]:
    return dict(SCALES[scale_id].values)


def property_minmax(scale_id: str) -> dict[str, float]:
    v = SCALES[scale_id].values
    lo = min(v.values())
    hi = max(v.values())
    return {a: (v[a] - lo) / (hi - lo) for a in AA20}


def scale_hash(scale_id: str, transform: str) -> str:
    tab = property_raw(scale_id) if transform == "RAW" else property_minmax(scale_id)
    return _hash_table(tab)


def all_scale_ids() -> list[str]:
    return list(SCALES.keys())
