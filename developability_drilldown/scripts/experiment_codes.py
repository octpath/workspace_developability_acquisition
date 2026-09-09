#!/usr/bin/env python3
"""Append-only experiment_code issuance utilities.

Authority: results/EXPERIMENT_CODES.csv
Existing codes must NEVER be renumbered or reused.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODES_PATH = ROOT / "results" / "EXPERIMENT_CODES.csv"
CODE_RE = re.compile(r"^EXP[0-9]{3,}$")


def load_codes() -> pd.DataFrame:
    if not CODES_PATH.exists():
        raise FileNotFoundError(CODES_PATH)
    df = pd.read_csv(CODES_PATH)
    if df["experiment_code"].duplicated().any() or df["experiment_id"].duplicated().any():
        raise ValueError("duplicate codes or experiment_ids in EXPERIMENT_CODES.csv")
    for c in df["experiment_code"]:
        if not CODE_RE.match(str(c)):
            raise ValueError(f"bad experiment_code: {c}")
    return df


def code_to_id(code: str) -> str:
    df = load_codes()
    m = df[df["experiment_code"] == code]
    if len(m) != 1:
        raise KeyError(code)
    return str(m.iloc[0]["experiment_id"])


def id_to_code(experiment_id: str) -> str:
    df = load_codes()
    m = df[df["experiment_id"] == experiment_id]
    if len(m) != 1:
        raise KeyError(experiment_id)
    return str(m.iloc[0]["experiment_code"])


def resolve_experiment_ref(ref: str) -> tuple[str, str]:
    """Resolve EXP code or descriptive experiment_id -> (code, experiment_id)."""
    df = load_codes()
    if CODE_RE.match(ref):
        m = df[df["experiment_code"] == ref]
        if len(m) != 1:
            raise KeyError(ref)
        return str(m.iloc[0]["experiment_code"]), str(m.iloc[0]["experiment_id"])
    m = df[df["experiment_id"] == ref]
    if len(m) != 1:
        raise KeyError(ref)
    return str(m.iloc[0]["experiment_code"]), str(m.iloc[0]["experiment_id"])


def next_code() -> str:
    df = load_codes()
    nums = [int(str(c)[3:]) for c in df["experiment_code"]]
    n = max(nums) + 1 if nums else 1
    return f"EXP{n:03d}" if n < 1000 else f"EXP{n}"


def issue_code(experiment_id: str, source_model_id: str = "", phase: str = "APPEND", notes: str = "") -> str:
    """Issue a new code for a NEW experiment_id. Refuses if id already mapped."""
    df = load_codes()
    if (df["experiment_id"] == experiment_id).any():
        raise ValueError(f"experiment_id already has a code: {experiment_id}")
    code = next_code()
    row = {
        "experiment_code": code,
        "experiment_id": experiment_id,
        "source_model_id": source_model_id,
        "issued_at_phase": phase,
        "status": "ACTIVE",
        "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CODES_PATH, index=False)
    return code
