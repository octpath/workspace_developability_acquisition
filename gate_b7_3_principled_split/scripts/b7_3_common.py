"""Gate B7.3 shared utilities — paths, IO, data load, hard constraints."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b7_3_principled_split"
CFG = GATE / "config"
SPL = GATE / "splits"
MET = GATE / "metrics"
PLT = GATE / "plots"
REP = GATE / "reports"
LOG = GATE / "logs"
CACHE = GATE / "cache"
P1 = GATE / "phase1_premodel"
P2 = GATE / "phase2_model_safety"
P3 = GATE / "phase3_policy"

FEAT_PATH = ROOT / "gate_b6_split_search/cache/test_balance_features.csv"
CAND_PATH = ROOT / "gate_b6_split_search/config/B6_1_FINAL_CANDIDATE_SET.json"
OUTER_PATH = ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json"
POP_PATH = ROOT / "gate_b3/frozen/organizer/final_population.csv"
ROLE_PATH = ROOT / "gate_b3/frozen/organizer/role_map.csv"
PRED_FINAL = ROOT / "gate_b5_ceiling/predictions/final"
PRED_OOF = ROOT / "gate_b5_ceiling/predictions/oof"

SEED_BASE = 20260901
PUBLIC_N = 81
PRIVATE_N = 81
HIC_LOW, HIC_HIGH = 10.5, 11.5
N_Q_PRIMARY = 8

CONT_COLS = [
    "vh_len",
    "vl_len",
    "H_CDR3_len",
    "A2_HL_pI",
    "A2_HL_charge_ph7",
    "A2_HL_gravy",
    "nearest_train_VH_identity",
    "nearest_train_VL_identity",
    "nearest_train_paired_identity",
]
CAT_COLS = ["C_vh_family", "C_vl_family", "C_kappa_lambda"]

BANK_A = {
    "TmApp": [
        "CONST_MEDIAN",
        "SEQ_SIMPLE_Ridge",
        "BIO_Ridge",
        "PLM_ABLANG2_PCA32_SVR",
        "NESTED_STACK_MEAN",
    ],
    "HIC": [
        "CONST_MEDIAN",
        "SEQ_SIMPLE_Ridge",
        "PLM_ESM2_PCA64_SVR",
        "ESMFN_STRUCTURE_ElasticNet",
        "FUSION_ESM2_ESMFN_ElasticNet",
        "NESTED_STACK_NNLS",
    ],
}

FINAL_DECISIONS = [
    "KEEP_CAND_12528_AFTER_LARGE_PRINCIPLED_SEARCH",
    "REPLACE_WITH_MODEL_BLIND_OPTIMAL_SPLIT",
    "REPLACE_WITH_MODEL_BLIND_NEAR_OPTIMAL_RANDOMIZED_SPLIT",
    "MULTIPLE_MODEL_BLIND_SPLITS_EQUIVALENT",
    "NO_STABLE_PRODUCTION_SPLIT_FOUND",
    "HUMAN_DECISION_REQUIRED",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / "b7_3_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def mask_hash(public_ids) -> str:
    return sha256_bytes(repr(frozenset(map(str, public_ids))).encode())


def write_json(path: Path, obj: Any) -> str:
    b = json.dumps(obj, indent=2, sort_keys=True, default=float).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_bytes(b) + "\n")
    return sha256_bytes(b)


def hic_band(v: float) -> str:
    if v < HIC_LOW:
        return "LOW"
    if v <= HIC_HIGH:
        return "MEDIUM"
    return "HIGH"


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=int)
    n = len(values)
    for rank, ix in enumerate(order):
        bins[ix] = min(n_bins - 1, int(rank * n_bins / n))
    return bins


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def spearman_safe(a, b) -> float:
    from scipy.stats import spearmanr

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(a[m], b[m]).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def fnum(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_base_data() -> dict:
    feat = pd.read_csv(FEAT_PATH)
    feat["id"] = feat["id"].astype(str)
    feat["nearest_train_paired_identity"] = 0.5 * (
        feat["nearest_train_VH_identity"].astype(float)
        + feat["nearest_train_VL_identity"].astype(float)
    )
    pop = pd.read_csv(POP_PATH).set_index("id")
    pop.index = pop.index.astype(str)
    role = pd.read_csv(ROLE_PATH)
    role["id"] = role["id"].astype(str)
    outer = json.loads(OUTER_PATH.read_text())
    train_ids = list(map(str, outer["train_ids"]))
    folds = [np.asarray(r["fold_id"], int) for r in outer["folds"]]
    groups_train = pop.loc[train_ids, "sequence_group"].astype(int).values

    role_pub = role.loc[role.role == "Public", "id"].astype(str).tolist()
    role_priv = role.loc[role.role == "Private", "id"].astype(str).tolist()
    test_ids = sorted(role_pub + role_priv)

    cand_blob = json.loads(CAND_PATH.read_text())
    controls = {}
    for c in cand_blob["candidates"]:
        cid = c["candidate_id"]
        if cid not in ("CAND_12528", "CAND_04974", "CAND_12207"):
            continue
        controls[cid] = {
            "candidate_id": cid,
            "split_id": cid,
            "public_ids": sorted(map(str, c["public_ids"])),
            "private_ids": sorted(map(str, c["private_ids"])),
            "public_hash": c.get("public_hash", sha256_ids(c["public_ids"])),
            "private_hash": c.get("private_hash", sha256_ids(c["private_ids"])),
            "is_control": True,
            "method": "incumbent_control",
            "seed": None,
        }
    controls["BASELINE_ROLEMAP"] = {
        "candidate_id": "BASELINE_ROLEMAP",
        "split_id": "BASELINE_ROLEMAP",
        "public_ids": sorted(role_pub),
        "private_ids": sorted(role_priv),
        "public_hash": sha256_ids(role_pub),
        "private_hash": sha256_ids(role_priv),
        "is_control": True,
        "method": "role_map",
        "seed": None,
    }

    feat = feat[feat.id.isin(test_ids)].copy().reset_index(drop=True)
    assert len(feat) == 162
    feat["hic_band"] = feat.HIC.map(hic_band)
    band_counts = feat["hic_band"].value_counts().to_dict()
    assert band_counts.get("LOW") == 149
    assert band_counts.get("MEDIUM") == 6
    assert band_counts.get("HIGH") == 7

    group_map: dict[int, list[str]] = {}
    for g, sub in feat.groupby("sequence_group"):
        group_map[int(g)] = sorted(sub.id.tolist())

    group_ids = sorted(group_map)
    group_sizes = np.array([len(group_map[g]) for g in group_ids], dtype=int)
    id_to_row = {i: k for k, i in enumerate(feat.id.tolist())}
    group_index = {g: k for k, g in enumerate(group_ids)}
    id_to_group = {}
    for g, ids in group_map.items():
        for i in ids:
            id_to_group[i] = g

    high_ids = set(feat.loc[feat.hic_band == "HIGH", "id"])
    med_ids = set(feat.loc[feat.hic_band == "MEDIUM", "id"])
    group_high = np.array(
        [sum(1 for i in group_map[g] if i in high_ids) for g in group_ids], dtype=int
    )
    group_med = np.array(
        [sum(1 for i in group_map[g] if i in med_ids) for g in group_ids], dtype=int
    )

    tmapp = feat.TmApp.values.astype(float)
    hic = feat.HIC.values.astype(float)
    z_tm = (tmapp - tmapp.mean()) / (tmapp.std() + 1e-12)
    z_hic = (hic - hic.mean()) / (hic.std() + 1e-12)
    z2 = np.column_stack([z_tm, z_hic])

    return {
        "feat": feat,
        "pop": pop,
        "role": role,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "folds": folds,
        "groups_train": groups_train,
        "role_pub": role_pub,
        "role_priv": role_priv,
        "controls": controls,
        "group_map": group_map,
        "group_ids": group_ids,
        "group_sizes": group_sizes,
        "group_index": group_index,
        "id_to_group": id_to_group,
        "id_to_row": id_to_row,
        "group_high": group_high,
        "group_med": group_med,
        "tmapp": tmapp,
        "hic": hic,
        "z2": z2,
        "tmapp_sd": float(tmapp.std()),
        "hic_sd": float(hic.std()),
        "tmapp_iqr": float(np.subtract(*np.percentile(tmapp, [75, 25]))),
        "hic_iqr": float(np.subtract(*np.percentile(hic, [75, 25]))),
        "band_counts": band_counts,
        "qbins": {
            6: {
                "TmApp": quantile_bins(tmapp, 6),
                "HIC": quantile_bins(hic, 6),
            },
            8: {
                "TmApp": quantile_bins(tmapp, 8),
                "HIC": quantile_bins(hic, 8),
            },
            10: {
                "TmApp": quantile_bins(tmapp, 10),
                "HIC": quantile_bins(hic, 10),
            },
        },
        "tm_quart": quantile_bins(tmapp, 4),
        "hic_quart": quantile_bins(hic, 4),
    }


def pub_from_group_mask(mask: np.ndarray, data: dict) -> tuple[list[str], list[str]]:
    """mask[g]=1 => group assigned to Public."""
    pub, priv = [], []
    for k, g in enumerate(data["group_ids"]):
        ids = data["group_map"][g]
        if mask[k]:
            pub.extend(ids)
        else:
            priv.extend(ids)
    return sorted(pub), sorted(priv)


def group_mask_from_pub(pub_ids, data: dict) -> np.ndarray:
    pub = set(map(str, pub_ids))
    mask = np.zeros(len(data["group_ids"]), dtype=int)
    for k, g in enumerate(data["group_ids"]):
        ids = data["group_map"][g]
        in_pub = sum(1 for i in ids if i in pub)
        if 0 < in_pub < len(ids):
            raise ValueError(f"group {g} split across Public/Private")
        mask[k] = 1 if in_pub == len(ids) else 0
    return mask


def check_hard(mask: np.ndarray, data: dict) -> bool:
    if int(mask @ data["group_sizes"]) != PUBLIC_N:
        return False
    high_pub = int(mask @ data["group_high"])
    if high_pub not in (3, 4):
        return False
    return True


def hamming_id(pub_a, pub_b) -> int:
    return len(set(pub_a).symmetric_difference(set(pub_b)))


def lex_better(a: tuple, b: tuple) -> bool:
    return a < b


def lex_key_from_metrics(m: dict) -> tuple:
    return (
        0 if m.get("feasible", False) else 1,
        float(m["L1"]),
        float(m["L2"]),
        float(m["L3"]),
        float(m["L4"]),
    )
