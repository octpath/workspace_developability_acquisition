#!/usr/bin/env python3
"""Append-only experiment_code issuance (target-namespaced).

Namespaces:
  TmApp -> EXP-Txxx
  HIC   -> EXP-Hxxx
  MULTI -> EXP-Mxxx (reserved for jointly trained multi-target experiments)

Authority: results/EXPERIMENT_CODES.csv
Legacy flat EXP001–EXP048: results/LEGACY_EXPERIMENT_CODE_MAP.csv
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODES_PATH = ROOT / "results" / "EXPERIMENT_CODES.csv"
LEGACY_PATH = ROOT / "results" / "LEGACY_EXPERIMENT_CODE_MAP.csv"

CODE_RE = re.compile(r"^EXP-[THM][0-9]{3,}$")
LEGACY_RE = re.compile(r"^EXP[0-9]{3,}$")

Namespace = Literal["T", "H", "M"]

TARGET_TO_NS = {
    "TmApp": "T",
    "HIC": "H",
    "MULTI": "M",
    "T": "T",
    "H": "H",
    "M": "M",
}


def load_codes() -> pd.DataFrame:
    if not CODES_PATH.exists():
        raise FileNotFoundError(CODES_PATH)
    df = pd.read_csv(CODES_PATH)
    if df["experiment_code"].duplicated().any() or df["experiment_id"].duplicated().any():
        raise ValueError("duplicate codes or experiment_ids")
    for c in df["experiment_code"]:
        if not CODE_RE.match(str(c)):
            raise ValueError(f"bad experiment_code: {c}")
    return df


def load_legacy_map() -> pd.DataFrame:
    if not LEGACY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "legacy_experiment_code",
                "experiment_code",
                "experiment_id",
                "target",
                "migration_reason",
                "migration_commit",
            ]
        )
    return pd.read_csv(LEGACY_PATH)


def _ns_max(df: pd.DataFrame, ns: Namespace) -> int:
    nums = []
    for c in df["experiment_code"].astype(str):
        body = c.split("-", 1)[1]
        if body[0] == ns:
            nums.append(int(body[1:]))
    return max(nums) if nums else 0


def next_code(target_or_ns: str) -> str:
    ns = TARGET_TO_NS.get(target_or_ns)
    if ns is None:
        raise KeyError(f"unknown namespace/target: {target_or_ns}")
    df = load_codes()
    n = _ns_max(df, ns) + 1  # type: ignore[arg-type]
    return f"EXP-{ns}{n:03d}" if n < 1000 else f"EXP-{ns}{n}"


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
    """Resolve canonical code, descriptive id, or legacy EXPxxx -> (code, experiment_id)."""
    df = load_codes()
    if CODE_RE.match(ref):
        m = df[df["experiment_code"] == ref]
        if len(m) != 1:
            raise KeyError(ref)
        return str(m.iloc[0]["experiment_code"]), str(m.iloc[0]["experiment_id"])

    m = df[df["experiment_id"] == ref]
    if len(m) == 1:
        return str(m.iloc[0]["experiment_code"]), str(m.iloc[0]["experiment_id"])

    if LEGACY_RE.match(ref):
        leg = load_legacy_map()
        m = leg[leg["legacy_experiment_code"] == ref]
        if len(m) != 1:
            # try codes table legacy column
            if "legacy_experiment_code" in df.columns:
                m2 = df[df["legacy_experiment_code"] == ref]
                if len(m2) == 1:
                    warnings.warn(
                        f"legacy code {ref} is deprecated; canonical code is {m2.iloc[0]['experiment_code']}",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return str(m2.iloc[0]["experiment_code"]), str(m2.iloc[0]["experiment_id"])
            raise KeyError(ref)
        code = str(m.iloc[0]["experiment_code"])
        eid = str(m.iloc[0]["experiment_id"])
        warnings.warn(
            f"legacy code {ref} is deprecated; canonical code is {code}",
            DeprecationWarning,
            stacklevel=2,
        )
        return code, eid

    raise KeyError(ref)


def issue_code(
    experiment_id: str,
    target: str,
    source_model_id: str = "",
    phase: str = "APPEND",
    notes: str = "",
) -> str:
    df = load_codes()
    if (df["experiment_id"] == experiment_id).any():
        raise ValueError(f"experiment_id already has a code: {experiment_id}")
    if target not in ("TmApp", "HIC", "MULTI"):
        raise ValueError(target)
    code = next_code(target)
    row = {
        "experiment_code": code,
        "experiment_id": experiment_id,
        "target": target if target != "MULTI" else "MULTI",
        "source_model_id": source_model_id,
        "issued_at_phase": phase,
        "status": "ACTIVE",
        "legacy_experiment_code": "",
        "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CODES_PATH, index=False)
    return code
