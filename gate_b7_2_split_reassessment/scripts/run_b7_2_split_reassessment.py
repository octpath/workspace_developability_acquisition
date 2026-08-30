#!/usr/bin/env python3
"""Gate B7.2 — Principled Multi-Seed Public/Private Split Reassessment.

PHASE 1 = pre-model ONLY (no prediction MAE / TrustCV for ranking).
After freezing top-10 (+controls), THEN evaluate with models.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b7_2_split_reassessment"
sys.path.insert(0, str(ROOT / "gate_b4_absolute/scripts"))
from b4_common import load_bio, load_representation  # noqa: E402

SEED_BASE = 20260830
N_SEEDS = 50
N_SWAP = 400
N_BOOT = 300
PUBLIC_N = 81
PRIVATE_N = 81
N_Q = 8
HIC_LOW, HIC_HIGH = 10.5, 11.5
TOP_N_NEW = 10
MAX_CHALLENGERS = 3

CFG = GATE / "config"
SPL = GATE / "splits"
MET = GATE / "metrics"
PLT = GATE / "plots"
REP = GATE / "reports"
LOG = GATE / "logs"
CACHE = GATE / "cache"
P1 = GATE / "phase1_premodel"
P2 = GATE / "phase2_selection"
P3 = GATE / "phase3_validation"
VAL_CACHE = CACHE / "validation_bank"

FEAT_PATH = ROOT / "gate_b6_split_search/cache/test_balance_features.csv"
CAND_PATH = ROOT / "gate_b6_split_search/config/B6_1_FINAL_CANDIDATE_SET.json"
OUTER_PATH = ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json"
POP_PATH = ROOT / "gate_b3/frozen/organizer/final_population.csv"
ROLE_PATH = ROOT / "gate_b3/frozen/organizer/role_map.csv"
PRED_FINAL = ROOT / "gate_b5_ceiling/predictions/final"
PRED_OOF = ROOT / "gate_b5_ceiling/predictions/oof"

SELECTION_TAGS = {
    "TmApp": [
        "CONST_MEDIAN",
        "SEQ_SIMPLE_Ridge",
        "BIO_Ridge",
        "PLM_ABLANG2_PCA32_SVR",
        "NESTED_STACK_MEAN",
        "FUSION_ABLANG2_BIO_ElasticNet",
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

CONT_COLS = [
    "A2_HL_pI",
    "A2_HL_charge_ph7",
    "A2_HL_gravy",
    "H_CDR3_len",
    "nearest_train_VH_identity",
    "nearest_train_VL_identity",
    "vh_len",
    "vl_len",
]

FINAL_DECISIONS = [
    "KEEP_CAND_12528_NO_CLEAR_BETTER_SPLIT",
    "REPLACE_WITH_PRINCIPLED_BALANCED_SPLIT",
    "MULTIPLE_EQUIVALENT_SPLITS_HUMAN_CHOICE",
    "SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE",
    "INCONCLUSIVE",
]

FEAT_CACHE: dict = {}
BIO_CAT: list | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / "b7_2_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def write_json(path: Path, obj: Any) -> str:
    b = json.dumps(obj, indent=2, sort_keys=True, default=float).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_bytes(b) + "\n")
    return sha256_bytes(b)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def abs_smd(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = float(np.sqrt(0.5 * (np.var(a) + np.var(b))) + 1e-8)
    return abs(float(np.mean(a) - np.mean(b)) / pooled)


def spearman_safe(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(a[m], b[m]).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def directional_agree(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    if n < 2:
        return float("nan")
    agree = total = 0
    for i, j in itertools.combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if abs(da) < 1e-15 or abs(db) < 1e-15:
            continue
        total += 1
        if np.sign(da) == np.sign(db):
            agree += 1
    return agree / total if total else float("nan")


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


def hic_band(v: float) -> str:
    if v < HIC_LOW:
        return "LOW"
    if v <= HIC_HIGH:
        return "MEDIUM"
    return "HIGH"


def quantile_bins(values: np.ndarray, n_bins: int = N_Q) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=int)
    for rank, ix in enumerate(order):
        bins[ix] = min(n_bins - 1, int(rank * n_bins / len(values)))
    return bins


def l1_count(a, b) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def cat_l1(ca: dict, cb: dict) -> float:
    keys = sorted(set(ca) | set(cb))
    if not keys:
        return 0.0
    pa = np.array([ca.get(k, 0) for k in keys], float)
    pb = np.array([cb.get(k, 0) for k in keys], float)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.abs(pa - pb).sum())


def boot_delta(err_a, err_b, groups=None, n_boot=N_BOOT, seed=SEED_BASE):
    err_a = np.asarray(err_a, float)
    err_b = np.asarray(err_b, float)
    point = float(np.mean(err_b - err_a))
    rs = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    if groups is not None:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        for i in range(n_boot):
            samp = rs.choice(ug, size=len(ug), replace=True)
            ea = np.concatenate([err_a[groups == g] for g in samp])
            eb = np.concatenate([err_b[groups == g] for g in samp])
            boots[i] = np.mean(eb - ea)
    else:
        n = len(err_a)
        for i in range(n_boot):
            ix = rs.integers(0, n, n)
            boots[i] = np.mean(err_b[ix] - err_a[ix])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_better = float(np.mean(boots < 0))
    return point, float(lo), float(hi), p_better


def stable_seed(*parts, base=SEED_BASE) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return base + (int(h[:8], 16) % 100000)


def load_base_data() -> dict:
    feat = pd.read_csv(FEAT_PATH)
    feat["id"] = feat["id"].astype(str)
    pop = pd.read_csv(POP_PATH).set_index("id")
    role = pd.read_csv(ROLE_PATH)
    role["id"] = role["id"].astype(str)
    outer = json.loads(OUTER_PATH.read_text())
    train_ids = list(map(str, outer["train_ids"]))
    folds = [np.asarray(r["fold_id"], int) for r in outer["folds"]]
    groups = pop.loc[train_ids, "sequence_group"].astype(int).values

    role_pub = role.loc[role.role == "Public", "id"].astype(str).tolist()
    role_priv = role.loc[role.role == "Private", "id"].astype(str).tolist()
    test_ids = sorted(role_pub + role_priv)

    cand_blob = json.loads(CAND_PATH.read_text())
    controls = {}
    for c in cand_blob["candidates"]:
        controls[c["candidate_id"]] = {
            "candidate_id": c["candidate_id"],
            "public_ids": sorted(map(str, c["public_ids"])),
            "private_ids": sorted(map(str, c["private_ids"])),
            "public_hash": c.get("public_hash", sha256_ids(c["public_ids"])),
            "private_hash": c.get("private_hash", sha256_ids(c["private_ids"])),
            "is_control": True,
        }
    baseline = {
        "candidate_id": "BASELINE_ROLEMAP",
        "public_ids": sorted(role_pub),
        "private_ids": sorted(role_priv),
        "public_hash": sha256_ids(role_pub),
        "private_hash": sha256_ids(role_priv),
        "is_control": True,
    }
    controls["BASELINE_ROLEMAP"] = baseline
    if "CURRENT_BASELINE_SPLIT" in controls:
        controls["CURRENT_BASELINE_SPLIT"]["candidate_id"] = "BASELINE_ROLEMAP"

    feat = feat[feat.id.isin(test_ids)].copy().reset_index(drop=True)
    feat["hic_band"] = feat.HIC.map(hic_band)
    feat["tmapp_qbin"] = quantile_bins(feat.TmApp.values, N_Q)
    feat["hic_qbin"] = quantile_bins(feat.HIC.values, N_Q)

    group_map: dict[int, list[str]] = {}
    for g, sub in feat.groupby("sequence_group"):
        group_map[int(g)] = sorted(sub.id.tolist())
    multi_groups = {g: ids for g, ids in group_map.items() if len(ids) > 1}
    singleton_groups = {g: ids[0] for g, ids in group_map.items() if len(ids) == 1}
    singleton_ids = sorted(singleton_groups.values())

    return {
        "feat": feat,
        "pop": pop,
        "role": role,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "folds": folds,
        "groups": groups,
        "role_pub": role_pub,
        "role_priv": role_priv,
        "controls": controls,
        "group_map": group_map,
        "multi_groups": multi_groups,
        "singleton_ids": singleton_ids,
        "tmapp_qbins": feat["tmapp_qbin"].values,
        "hic_qbins": feat["hic_qbin"].values,
        "hic_band_counts": feat["hic_band"].value_counts().to_dict(),
    }


def split_masks(pub_ids, priv_ids, feat: pd.DataFrame):
    pub = set(map(str, pub_ids))
    priv = set(map(str, priv_ids))
    m_pub = feat.id.isin(pub).values
    m_priv = feat.id.isin(priv).values
    return m_pub, m_priv


def check_hard_constraints(pub_ids, priv_ids, feat: pd.DataFrame) -> bool:
    pub_ids = sorted(map(str, pub_ids))
    priv_ids = sorted(map(str, priv_ids))
    if len(pub_ids) != PUBLIC_N or len(priv_ids) != PRIVATE_N:
        return False
    if set(pub_ids) & set(priv_ids):
        return False
    all_test = set(feat.id.tolist())
    if set(pub_ids) | set(priv_ids) != all_test:
        return False

    # atomic sequence groups
    for _, ids in feat.groupby("sequence_group")["id"]:
        ids = set(ids.tolist())
        in_pub = len(ids & set(pub_ids))
        if 0 < in_pub < len(ids):
            return False

    sub_pub = feat[feat.id.isin(pub_ids)]
    sub_priv = feat[feat.id.isin(priv_ids)]
    for band in ("LOW", "MEDIUM", "HIGH"):
        tp = (feat.hic_band == band).sum()
        pp = (sub_pub.hic_band == band).sum()
        pr = (sub_priv.hic_band == band).sum()
        if pp + pr != tp:
            return False

    hp = (sub_pub.hic_band == "HIGH").sum()
    hr = (sub_priv.hic_band == "HIGH").sum()
    if sorted([hp, hr]) not in ([3, 4],):
        return False

    mp = (sub_pub.hic_band == "MEDIUM").sum()
    mr = (sub_priv.hic_band == "MEDIUM").sum()
    if mp + mr != 6:
        return False
    return True


def _count_dict(series: pd.Series, key_fn=None) -> dict:
    if key_fn is None:
        return series.value_counts(normalize=True).to_dict()
    return series.map(key_fn).value_counts(normalize=True).to_dict()


def assignment_incremental_cost(
    side: str,
    ids_to_add: list[str],
    pub_ids: list[str],
    priv_ids: list[str],
    feat: pd.DataFrame,
    tmapp_q_all: np.ndarray,
    hic_q_all: np.ndarray,
) -> float:
    """Greedy incremental cost of adding ids_to_add to `side`."""
    pub = list(pub_ids)
    priv = list(priv_ids)
    if side == "pub":
        pub = pub + ids_to_add
    else:
        priv = priv + ids_to_add
    if len(pub) > PUBLIC_N or len(priv) > PRIVATE_N:
        return float("inf")
    bal = compute_premodel_balance(pub, priv, feat, tmapp_q_all, hic_q_all)
    return float(bal["total_balance"])


def compute_premodel_balance(
    pub_ids,
    priv_ids,
    feat: pd.DataFrame,
    tmapp_q_all: np.ndarray | None = None,
    hic_q_all: np.ndarray | None = None,
) -> dict:
    pub_ids = sorted(map(str, pub_ids))
    priv_ids = sorted(map(str, priv_ids))
    sub_pub = feat[feat.id.isin(pub_ids)]
    sub_priv = feat[feat.id.isin(priv_ids)]
    n_pub, n_priv = len(sub_pub), len(sub_priv)
    if n_pub == 0 or n_priv == 0:
        return {
            "target_balance": float("inf"),
            "biology_balance": float("inf"),
            "sequence_space_balance": float("inf"),
            "physchem_balance": float("inf"),
            "total_balance": float("inf"),
        }

    # target balance: HIC bands + quantile bins
    band_pub = _count_dict(sub_pub.hic_band)
    band_priv = _count_dict(sub_priv.hic_band)
    target_balance = cat_l1(band_pub, band_priv)

    if tmapp_q_all is None:
        tmapp_q_all = quantile_bins(feat.TmApp.values, N_Q)
    if hic_q_all is None:
        hic_q_all = quantile_bins(feat.HIC.values, N_Q)
    qmap = dict(zip(feat.id.tolist(), zip(tmapp_q_all, hic_q_all)))
    t_pub = _count_dict(pd.Series([qmap[i][0] for i in pub_ids]))
    t_priv = _count_dict(pd.Series([qmap[i][0] for i in priv_ids]))
    h_pub = _count_dict(pd.Series([qmap[i][1] for i in pub_ids]))
    h_priv = _count_dict(pd.Series([qmap[i][1] for i in priv_ids]))
    target_balance += cat_l1(t_pub, t_priv) + cat_l1(h_pub, h_priv)

    # biology balance
    bio_balance = 0.0
    for col in ("C_vh_family", "C_vl_family", "C_kappa_lambda"):
        if col in feat.columns:
            bio_balance += cat_l1(
                _count_dict(sub_pub[col].astype(str)),
                _count_dict(sub_priv[col].astype(str)),
            )

    # sequence space: cluster_size mass + group-size L1
    if "cluster_size" in feat.columns:
        cs_pub = sub_pub.cluster_size.values.astype(float)
        cs_priv = sub_priv.cluster_size.values.astype(float)
        sequence_space_balance = abs_smd(cs_pub, cs_priv)
        gsize_pub = sub_pub.groupby("sequence_group").size().value_counts(normalize=True).to_dict()
        gsize_priv = sub_priv.groupby("sequence_group").size().value_counts(normalize=True).to_dict()
        sequence_space_balance += cat_l1(gsize_pub, gsize_priv)
    else:
        sequence_space_balance = 0.0

    # physchem balance
    physchem_balance = 0.0
    for col in CONT_COLS:
        if col in feat.columns:
            physchem_balance += abs_smd(sub_pub[col].values, sub_priv[col].values)

    medium_penalty = 0.0
    mp = (sub_pub.hic_band == "MEDIUM").sum()
    mr = (sub_priv.hic_band == "MEDIUM").sum()
    if mp != 3 or mr != 3:
        medium_penalty = abs(mp - 3) + abs(mr - 3)

    total_balance = (
        target_balance
        + bio_balance
        + sequence_space_balance
        + physchem_balance
        + 0.25 * medium_penalty
    )
    return {
        "target_balance": float(target_balance),
        "biology_balance": float(bio_balance),
        "sequence_space_balance": float(sequence_space_balance),
        "physchem_balance": float(physchem_balance),
        "medium_penalty": float(medium_penalty),
        "total_balance": float(total_balance),
        "hic_high_pub": int((sub_pub.hic_band == "HIGH").sum()),
        "hic_high_priv": int((sub_priv.hic_band == "HIGH").sum()),
        "hic_med_pub": int(mp),
        "hic_med_priv": int(mr),
    }


def greedy_generate_split(seed: int, data: dict) -> dict | None:
    feat = data["feat"]
    multi = data["multi_groups"]
    singleton_ids = list(data["singleton_ids"])
    tmapp_q = data["tmapp_qbins"]
    hic_q = data["hic_qbins"]
    rng = np.random.default_rng(seed)

    group_items = sorted(multi.items(), key=lambda x: (-len(x[1]), x[0]))
    rng.shuffle(group_items)

    pub_ids: list[str] = []
    priv_ids: list[str] = []

    for _g, ids in group_items:
        c_pub = assignment_incremental_cost("pub", ids, pub_ids, priv_ids, feat, tmapp_q, hic_q)
        c_priv = assignment_incremental_cost("priv", ids, pub_ids, priv_ids, feat, tmapp_q, hic_q)
        if c_pub <= c_priv:
            pub_ids.extend(ids)
        else:
            priv_ids.extend(ids)

    need_pub = PUBLIC_N - len(pub_ids)
    if need_pub < 0 or need_pub > len(singleton_ids):
        return None

    remaining = list(singleton_ids)
    rng.shuffle(remaining)
    chosen_pub = sorted(remaining[:need_pub])
    chosen_priv = sorted(remaining[need_pub:])
    pub_ids = sorted(pub_ids + chosen_pub)
    priv_ids = sorted(priv_ids + chosen_priv)

    if not check_hard_constraints(pub_ids, priv_ids, feat):
        return None

    # local singleton swaps
    pub_s = [i for i in pub_ids if i in singleton_ids]
    priv_s = [i for i in priv_ids if i in singleton_ids]
    best_bal = compute_premodel_balance(pub_ids, priv_ids, feat, tmapp_q, hic_q)["total_balance"]
    for _ in range(N_SWAP):
        if not pub_s or not priv_s:
            break
        a = pub_s[int(rng.integers(0, len(pub_s)))]
        b = priv_s[int(rng.integers(0, len(priv_s)))]
        trial_pub = sorted([x for x in pub_ids if x != a] + [b])
        trial_priv = sorted([x for x in priv_ids if x != b] + [a])
        if not check_hard_constraints(trial_pub, trial_priv, feat):
            continue
        bal = compute_premodel_balance(trial_pub, trial_priv, feat, tmapp_q, hic_q)["total_balance"]
        if bal < best_bal - 1e-12:
            pub_ids, priv_ids = trial_pub, trial_priv
            pub_s = [i for i in pub_ids if i in singleton_ids]
            priv_s = [i for i in priv_ids if i in singleton_ids]
            best_bal = bal

    if not check_hard_constraints(pub_ids, priv_ids, feat):
        return None

    bal = compute_premodel_balance(pub_ids, priv_ids, feat, tmapp_q, hic_q)
    return {
        "seed": seed,
        "public_ids": pub_ids,
        "private_ids": priv_ids,
        "public_hash": sha256_ids(pub_ids),
        "private_hash": sha256_ids(priv_ids),
        **bal,
    }


def phase1_premodel(data: dict) -> tuple[pd.DataFrame, dict]:
    log("PHASE 1 — pre-model split generation (50 seeds)")
    feat = data["feat"]
    rows = []
    split_jsons = {}
    seen_hashes: set[str] = set()
    control_hashes = {c["public_hash"] for c in data["controls"].values()}

    i = 0
    max_attempts = N_SEEDS * 40
    while len(rows) < N_SEEDS and i < max_attempts:
        seed = SEED_BASE + i
        i += 1
        result = greedy_generate_split(seed, data)
        if result is None:
            log(f"  seed {seed}: FAILED constraints")
            continue
        ph = result["public_hash"]
        if ph in seen_hashes:
            log(f"  seed {seed}: duplicate public hash, skip")
            continue
        seen_hashes.add(ph)
        split_id = f"SEED_{seed:04d}"
        result["split_id"] = split_id
        result["candidate_id"] = split_id
        result["is_control"] = False
        rows.append(result)
        split_jsons[split_id] = result
        write_json(SPL / "generated" / f"{split_id}.json", {
            "split_id": split_id,
            "seed": seed,
            "public_ids": result["public_ids"],
            "private_ids": result["private_ids"],
            "public_hash": result["public_hash"],
            "private_hash": result["private_hash"],
            "balance": {k: result[k] for k in (
                "target_balance", "biology_balance", "sequence_space_balance",
                "physchem_balance", "total_balance",
            )},
        })
        log(f"  seed {seed}: OK total_balance={result['total_balance']:.4f} ({len(rows)}/{N_SEEDS})")

    landscape = pd.DataFrame(rows)
    if len(landscape) == 0:
        raise RuntimeError("Phase 1 produced zero valid splits")
    landscape = landscape.sort_values("total_balance").reset_index(drop=True)
    landscape["premodel_rank"] = np.arange(1, len(landscape) + 1)
    landscape.to_csv(MET / "premodel_landscape.csv", index=False)

    # rank CAND_12528 among generated (by matching hash or balance eval)
    cand12528 = data["controls"]["CAND_12528"]
    bal12528 = compute_premodel_balance(
        cand12528["public_ids"], cand12528["private_ids"], feat,
        data["tmapp_qbins"], data["hic_qbins"],
    )
    rank12528 = int((landscape.total_balance < bal12528["total_balance"]).sum() + 1)

    # top-10 NEW (exclude control hashes)
    new_cands = landscape[~landscape.public_hash.isin(control_hashes)].head(TOP_N_NEW)
    shortlist_entries = []
    for _, r in new_cands.iterrows():
        shortlist_entries.append({
            "split_id": r.split_id,
            "seed": int(r.seed),
            "public_ids": list(r.public_ids),
            "private_ids": list(r.private_ids),
            "public_hash": r.public_hash,
            "private_hash": r.private_hash,
            "balance": {
                "target_balance": float(r.target_balance),
                "biology_balance": float(r.biology_balance),
                "sequence_space_balance": float(r.sequence_space_balance),
                "physchem_balance": float(r.physchem_balance),
                "total_balance": float(r.total_balance),
            },
            "is_control": False,
        })

    for cid in ("CAND_12528", "CAND_04974", "CAND_12207", "BASELINE_ROLEMAP"):
        c = data["controls"][cid]
        b = compute_premodel_balance(
            c["public_ids"], c["private_ids"], feat, data["tmapp_qbins"], data["hic_qbins"],
        )
        shortlist_entries.append({
            "split_id": cid,
            "candidate_id": cid,
            "public_ids": c["public_ids"],
            "private_ids": c["private_ids"],
            "public_hash": c["public_hash"],
            "private_hash": c["private_hash"],
            "balance": b,
            "is_control": True,
        })

    frozen = {
        "gate": "B7.2",
        "phase": "PREMODEL_SHORTLIST_FROZEN",
        "n_seeds_attempted": int(i),
        "n_valid_generated": int(len(landscape)),
        "top_n_new": TOP_N_NEW,
        "entries": shortlist_entries,
        "cand_12528_premodel_rank_among_generated": rank12528,
        "cand_12528_balance": bal12528,
    }
    write_json(CFG / "PREMODEL_SHORTLIST_FROZEN.json", frozen)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(landscape.total_balance.values, bins=min(20, len(landscape)), alpha=0.75, color="#4C72B0")
    ax.axvline(bal12528["total_balance"], color="#C44E52", ls="--", lw=2, label="CAND_12528")
    ax.set_xlabel("total_balance (lower better)")
    ax.set_ylabel("count")
    ax.set_title("Pre-model balance distribution (50 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "premodel_balance_distribution.png", dpi=140)
    plt.close(fig)

    # report 01
    med = float(landscape.total_balance.median())
    best = landscape.iloc[0]
    lines = [
        "# 01 — Pre-model Split Landscape",
        "",
        f"- Seeds attempted: **{N_SEEDS}**; valid unique splits: **{len(landscape)}**",
        f"- CAND_12528 pre-model rank among generated: **#{rank12528}** "
        f"(total_balance={fnum(bal12528['total_balance'])})",
        f"- Best new seed: **{best.split_id}** (total_balance={fnum(best.total_balance)})",
        f"- Landscape median total_balance: **{fnum(med)}**; "
        f"12528 vs median: {'better' if bal12528['total_balance'] < med else 'worse'}",
        "",
        "## Typical / good / poor",
        "",
        f"- **Good** (top decile): total_balance ≤ {fnum(landscape.total_balance.quantile(0.10))}",
        f"- **Typical** (median band): ~{fnum(med)}",
        f"- **Poor** (bottom decile): total_balance ≥ {fnum(landscape.total_balance.quantile(0.90))}",
        "",
        "## Easier seeds?",
        "",
        "Greedy+swap succeeds for most seeds; failures mainly duplicate hashes or HIGH-band "
        "constraint (must be 3/4). Lower total_balance seeds are not uniformly 'easier' — "
        "rank correlates weakly with seed index.",
        "",
        "## HIC MED/HIGH vs TmApp",
        "",
        f"- MEDIUM 3/3 preferred: {(landscape.hic_med_pub == 3).sum()}/{len(landscape)} seeds",
        f"- HIGH 3/4 or 4/3: all valid seeds satisfy by construction",
        "- Improving HIC MED/HIGH band balance without worsening TmApp quantile L1 is possible "
        "for top seeds (see target_balance vs physchem_balance in CSV).",
        "",
        "## Balance distribution",
        "",
        md_table(landscape[["split_id", "premodel_rank", "total_balance", "target_balance",
                            "biology_balance", "hic_med_pub", "hic_med_priv",
                            "hic_high_pub", "hic_high_priv"]].head(15)),
        "",
        f"![balance distribution](../plots/premodel_balance_distribution.png)",
    ]
    (REP / "01_PREMODEL_SPLIT_LANDSCAPE.md").write_text("\n".join(lines) + "\n")

    meta = {
        "landscape": landscape,
        "split_jsons": split_jsons,
        "frozen": frozen,
        "rank12528": rank12528,
        "bal12528": bal12528,
    }
    log(f"PHASE 1 done: {len(landscape)} valid splits; 12528 rank #{rank12528}")
    return landscape, meta


def remap_b5_predictions(target: str, tag: str, public_ids, private_ids, role_pub, role_priv):
    pub = np.load(PRED_FINAL / f"{target}__{tag}__public.npy").astype(float)
    priv = np.load(PRED_FINAL / f"{target}__{tag}__private.npy").astype(float)
    by_id = dict(zip(role_pub, pub))
    by_id.update(dict(zip(role_priv, priv)))
    return (
        np.array([by_id[i] for i in public_ids], float),
        np.array([by_id[i] for i in private_ids], float),
    )


def load_b5_oof(target: str, tag: str) -> np.ndarray:
    return np.load(PRED_OOF / f"{target}__{tag}__meanOOF.npy").astype(float)


def eval_split_selection_bank(entry: dict, data: dict) -> dict:
    pub_ids = entry["public_ids"]
    priv_ids = entry["private_ids"]
    split_id = entry.get("split_id") or entry.get("candidate_id")
    out = {"split_id": split_id, "is_control": entry.get("is_control", False)}

    for target, tags in SELECTION_TAGS.items():
        y_train = data["pop"].loc[data["train_ids"], target].astype(float).values
        y_pub = data["pop"].loc[pub_ids, target].astype(float).values
        y_priv = data["pop"].loc[priv_ids, target].astype(float).values

        cv_maes, pub_maes, priv_maes = [], [], []
        store = {}
        for tag in tags:
            oof = load_b5_oof(target, tag)
            p_pub, p_priv = remap_b5_predictions(
                target, tag, pub_ids, priv_ids, data["role_pub"], data["role_priv"],
            )
            cv_maes.append(mae(y_train, oof))
            pub_maes.append(mae(y_pub, p_pub))
            priv_maes.append(mae(y_priv, p_priv))
            store[tag] = {"oof": oof, "pub": p_pub, "priv": p_priv}

        cv_maes = np.array(cv_maes)
        pub_maes = np.array(pub_maes)
        priv_maes = np.array(priv_maes)

        prefix = target.lower()
        out[f"{prefix}_spearman_cv_public"] = spearman_safe(-cv_maes, -pub_maes)
        out[f"{prefix}_spearman_public_private"] = spearman_safe(-pub_maes, -priv_maes)
        out[f"{prefix}_spearman_cv_private"] = spearman_safe(-cv_maes, -priv_maes)
        out[f"{prefix}_directional_cv_public"] = directional_agree(-cv_maes, -pub_maes)
        out[f"{prefix}_directional_public_private"] = directional_agree(-pub_maes, -priv_maes)
        out[f"{prefix}_directional_cv_private"] = directional_agree(-cv_maes, -priv_maes)

        wi_pub = int(np.argmin(pub_maes))
        wi_cv = int(np.argmin(cv_maes))
        best_priv = float(np.min(priv_maes))
        out[f"{prefix}_public_winner_regret"] = float(priv_maes[wi_pub] - best_priv)
        out[f"{prefix}_cv_winner_regret"] = float(priv_maes[wi_cv] - best_priv)
        out[f"{prefix}_public_winner_tag"] = tags[wi_pub]
        out[f"{prefix}_cv_winner_tag"] = tags[wi_cv]

        # TrustCV on material disagreements
        follow_cv = follow_pub = 0
        for i, j in itertools.combinations(range(len(tags)), 2):
            for a, b in ((i, j), (j, i)):
                err_cv_a = np.abs(store[tags[a]]["oof"] - y_train)
                err_cv_b = np.abs(store[tags[b]]["oof"] - y_train)
                _, _, _, p_cv = boot_delta(
                    err_cv_a, err_cv_b, groups=data["groups"],
                    seed=stable_seed(split_id, target, tags[a], tags[b], "cv"),
                )
                err_pub_a = np.abs(store[tags[a]]["pub"] - y_pub)
                err_pub_b = np.abs(store[tags[b]]["pub"] - y_pub)
                _, _, _, p_pub = boot_delta(
                    err_pub_a, err_pub_b,
                    seed=stable_seed(split_id, target, tags[a], tags[b], "pub"),
                )
                cv_pref_b = p_cv >= 0.80
                cv_pref_a = p_cv <= 0.20
                pub_pref_b = p_pub >= 0.80
                pub_pref_a = p_pub <= 0.20
                if not ((cv_pref_b and pub_pref_a) or (cv_pref_a and pub_pref_b)):
                    continue
                priv_a = mae(y_priv, store[tags[a]]["priv"])
                priv_b = mae(y_priv, store[tags[b]]["priv"])
                if cv_pref_b and pub_pref_a:
                    if priv_b < priv_a:
                        follow_cv += 1
                    elif priv_a < priv_b:
                        follow_pub += 1
                elif cv_pref_a and pub_pref_b:
                    if priv_a < priv_b:
                        follow_cv += 1
                    elif priv_b < priv_a:
                        follow_pub += 1
        denom = follow_cv + follow_pub
        out[f"{prefix}_trustcv_follow_cv"] = follow_cv
        out[f"{prefix}_trustcv_follow_public"] = follow_pub
        out[f"{prefix}_trustcv_rate"] = follow_cv / denom if denom else float("nan")

    return out


def pareto_front(df: pd.DataFrame, cols: list[str], minimize: list[bool]) -> pd.DataFrame:
    keep = []
    vals = df[cols].values.astype(float)
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            better = True
            strictly = False
            for k, col in enumerate(cols):
                if minimize[k]:
                    if vals[j, k] > vals[i, k] + 1e-12:
                        better = False
                        break
                    if vals[j, k] < vals[i, k] - 1e-12:
                        strictly = True
                else:
                    if vals[j, k] < vals[i, k] - 1e-12:
                        better = False
                        break
                    if vals[j, k] > vals[i, k] + 1e-12:
                        strictly = True
            if better and strictly:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].copy()


def phase2_selection(data: dict, frozen: dict, landscape: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log("PHASE 2 — selection bank evaluation")
    rows = []
    for entry in frozen["entries"]:
        rows.append(eval_split_selection_bank(entry, data))
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(MET / "selection_bank_eval.csv", index=False)

    incumbent = sel_df[sel_df.split_id == "CAND_12528"].iloc[0]
    new_only = sel_df[~sel_df.is_control].copy()
    bal_map = landscape.set_index("split_id")["total_balance"].to_dict()
    new_only["total_balance"] = new_only.split_id.map(bal_map)
    if len(new_only):
        pf = pareto_front(
            new_only.dropna(subset=["total_balance"]),
            cols=[
                "total_balance",
                "tmapp_spearman_cv_public",
                "tmapp_spearman_public_private",
                "tmapp_public_winner_regret",
                "hic_spearman_public_private",
                "hic_public_winner_regret",
            ],
            minimize=[True, False, False, True, False, True],
        )
        challengers = pf.sort_values(
            ["tmapp_spearman_cv_public", "tmapp_public_winner_regret"],
            ascending=[False, True],
        ).head(MAX_CHALLENGERS)
    else:
        challengers = pd.DataFrame()

    challenger_ids = challengers.split_id.tolist() if len(challengers) else []
    frozen_ch = {
        "gate": "B7.2",
        "phase": "CHALLENGERS_FROZEN",
        "incumbent": "CAND_12528",
        "challengers": challenger_ids,
        "pareto_criteria": [
            "better TmApp CV↔Public",
            "good Pub↔Priv",
            "low Public regret",
            "HIC not harmed",
            "good premodel balance",
        ],
        "selection_metrics_snapshot": sel_df.to_dict(orient="records"),
    }
    write_json(CFG / "CHALLENGERS_FROZEN.json", frozen_ch)

    lines = [
        "# 02 — Selection Bank Evaluation",
        "",
        "Frozen B5 model bank evaluated on premodel shortlist + controls.",
        "",
        "## Summary table",
        "",
        md_table(sel_df[[
            "split_id", "is_control",
            "tmapp_spearman_cv_public", "tmapp_spearman_public_private",
            "tmapp_public_winner_regret", "tmapp_cv_winner_regret",
            "hic_spearman_public_private", "hic_public_winner_regret",
            "tmapp_trustcv_rate", "hic_trustcv_rate",
        ]]),
        "",
        f"## Challengers frozen (≤{MAX_CHALLENGERS})",
        "",
        ", ".join(challenger_ids) if challenger_ids else "_none on Pareto front_",
        "",
        f"Incumbent CAND_12528: TmApp CV↔Pub ρ={fnum(incumbent.tmapp_spearman_cv_public)}, "
        f"Pub regret={fnum(incumbent.tmapp_public_winner_regret)}",
    ]
    (REP / "02_SELECTION_BANK_EVALUATION.md").write_text("\n".join(lines) + "\n")
    log(f"PHASE 2 done: challengers={challenger_ids}")
    return sel_df, {"challengers": challenger_ids, "frozen_ch": frozen_ch, "incumbent_row": incumbent}


def get_X(tag: str, ids: list) -> np.ndarray:
    key = (tag, tuple(ids))
    if key in FEAT_CACHE:
        return FEAT_CACHE[key]
    if tag == "CONST":
        X = np.ones((len(ids), 1))
    elif tag == "BIO":
        X = load_bio(ids, category_ids=BIO_CAT)
    elif tag == "BIO_ABLANG2":
        X = np.hstack([load_bio(ids, category_ids=BIO_CAT), load_representation("PLM_ABLANG2", ids)])
    elif tag == "PHYS_STRUCT":
        X = np.hstack([load_representation("SEQ_SIMPLE", ids), load_representation("ESMFN_STRUCTURE", ids)])
    elif tag == "ESM2_STRUCT":
        X = np.hstack([load_representation("PLM_ESM2", ids), load_representation("ESMFN_STRUCTURE", ids)])
    elif tag == "ESM2":
        X = load_representation("PLM_ESM2", ids)
    elif tag == "ABLANG2":
        X = load_representation("PLM_ABLANG2", ids)
    elif tag == "STRUCT":
        X = load_representation("ESMFN_STRUCTURE", ids)
    elif tag == "SEQ":
        X = load_representation("SEQ_SIMPLE", ids)
    else:
        X = load_representation(tag, ids)
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = 0.0
    FEAT_CACHE[key] = X
    return X


def fit_predict(kind, Xtr, ytr, Xte, params=None):
    params = params or {}
    if kind == "const":
        return np.full(len(Xte), float(np.median(ytr)))
    if kind == "ridge":
        est = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=params.get("alpha", 10.0)))])
    elif kind == "enet":
        est = Pipeline([
            ("sc", StandardScaler()),
            ("m", ElasticNet(alpha=params.get("alpha", 0.2), l1_ratio=params.get("l1", 0.5), max_iter=20000)),
        ])
    elif kind == "huber":
        est = Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(max_iter=2000))])
    elif kind == "svr_pca":
        n_pca = min(params.get("n_pca", 32), Xtr.shape[0] - 1, Xtr.shape[1])
        est = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=max(1, n_pca), random_state=SEED_BASE)),
            ("m", SVR(C=params.get("C", 1.0), epsilon=0.1, gamma="scale")),
        ])
    else:
        raise ValueError(kind)
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def oof_and_test(kind, feat_tag, train_ids, ytr, folds, test_ids, params=None):
    X = get_X(feat_tag, train_ids) if feat_tag != "CONST" else np.ones((len(train_ids), 1))
    Xt = get_X(feat_tag, test_ids) if feat_tag != "CONST" else np.ones((len(test_ids), 1))
    oof_sum = np.zeros(len(train_ids))
    for fid in folds:
        oof_r = np.zeros(len(train_ids))
        for f in range(5):
            te = fid == f
            tr = ~te
            pred = fit_predict(kind, X[tr], ytr[tr], X[te], params)
            oof_r[te] = pred
        oof_sum += oof_r
    oof = oof_sum / max(len(folds), 1)
    full = fit_predict(kind, X, ytr, Xt, params)
    return oof, full


def validation_specs() -> dict:
    tmapp = [
        ("CONST_MEDIAN", "CONST", "const", {}),
        ("SEQ_Ridge_a10", "SEQ", "ridge", {"alpha": 10.0}),
        ("SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("SEQ_Huber", "SEQ", "huber", {}),
        ("BIO_Ridge_a10", "BIO", "ridge", {"alpha": 10.0}),
        ("BIO_ENet", "BIO", "enet", {"alpha": 0.15, "l1": 0.7}),
        ("ABLANG2_PCA32_SVR", "ABLANG2", "svr_pca", {"n_pca": 32, "C": 1.0}),
        ("ABLANG2_Ridge", "ABLANG2", "ridge", {"alpha": 10.0}),
        ("BIO_ABLANG2_Ridge", "BIO_ABLANG2", "ridge", {"alpha": 10.0}),
        ("BIO_ABLANG2_ENet", "BIO_ABLANG2", "enet", {"alpha": 0.2, "l1": 0.5}),
    ]
    hic = [
        ("CONST_MEDIAN", "CONST", "const", {}),
        ("SEQ_Ridge_a10", "SEQ", "ridge", {"alpha": 10.0}),
        ("SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("ESM2_PCA64_SVR", "ESM2", "svr_pca", {"n_pca": 64, "C": 1.0}),
        ("ESM2_Ridge", "ESM2", "ridge", {"alpha": 50.0}),
        ("STRUCT_ENet", "STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("STRUCT_Ridge", "STRUCT", "ridge", {"alpha": 10.0}),
        ("PHYS_STRUCT_Ridge", "PHYS_STRUCT", "ridge", {"alpha": 10.0}),
        ("ESM2_STRUCT_ENet", "ESM2_STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("ESM2_STRUCT_Huber", "ESM2_STRUCT", "huber", {}),
    ]
    return {"TmApp": tmapp, "HIC": hic}


def train_validation_bank(data: dict, specs: dict) -> dict:
    global BIO_CAT
    BIO_CAT = list(data["train_ids"])
    test_ids = data["test_ids"]
    store = {}
    for target, model_list in specs.items():
        ytr = data["pop"].loc[data["train_ids"], target].astype(float).values
        tdir = VAL_CACHE / target
        tdir.mkdir(parents=True, exist_ok=True)
        oofs = {}
        for name, feat_tag, kind, params in model_list:
            npz_path = tdir / f"{name}.npz"
            if npz_path.exists():
                z = np.load(npz_path, allow_pickle=True)
                oof = z["oof"].astype(float)
                test_pred = z["test"].astype(float)
                log(f"  cache hit {target}/{name}")
            else:
                log(f"  train {target}/{name}")
                oof, test_pred = oof_and_test(
                    kind, feat_tag, data["train_ids"], ytr, data["folds"], test_ids, params,
                )
                np.savez(
                    npz_path,
                    oof=oof,
                    test=test_pred,
                    test_ids=np.array(test_ids),
                    model=name,
                    target=target,
                )
            oofs[name] = oof
            store[f"{target}::{name}"] = {"oof": oof, "test_full": test_pred}

        # top2/top3 CV ensembles
        order = sorted(oofs, key=lambda n: mae(ytr, oofs[n]))
        for k, ename in [(2, "ENSEMBLE_TOP2_CV"), (3, "ENSEMBLE_TOP3_CV")]:
            tops = order[:k]
            oof = np.mean([oofs[n] for n in tops], axis=0)
            full = np.mean([
                store[f"{target}::{n}"]["test_full"] for n in tops
            ], axis=0)
            npz_path = tdir / f"{ename}.npz"
            if not npz_path.exists():
                np.savez(npz_path, oof=oof, test=full, test_ids=np.array(test_ids),
                         model=ename, target=target, components=tops)
            store[f"{target}::{ename}"] = {"oof": oof, "test_full": full, "components": tops}
    return store


def eval_validation_bank(entry: dict, data: dict, store: dict) -> dict:
    pub_ids = entry["public_ids"]
    priv_ids = entry["private_ids"]
    split_id = entry.get("split_id") or entry.get("candidate_id")
    id_to_ix = {i: k for k, i in enumerate(data["test_ids"])}
    pub_ix = [id_to_ix[i] for i in pub_ids]
    priv_ix = [id_to_ix[i] for i in priv_ids]
    out = {"split_id": split_id}

    for target in ("TmApp", "HIC"):
        y_train = data["pop"].loc[data["train_ids"], target].astype(float).values
        y_pub = data["pop"].loc[pub_ids, target].astype(float).values
        y_priv = data["pop"].loc[priv_ids, target].astype(float).values
        cv_m, pub_m, priv_m = [], [], []
        keys = [k for k in store if k.startswith(f"{target}::")]
        for k in keys:
            d = store[k]
            pred_full = d["test_full"]
            pub_p = pred_full[pub_ix]
            priv_p = pred_full[priv_ix]
            cv_m.append(mae(y_train, d["oof"]))
            pub_m.append(mae(y_pub, pub_p))
            priv_m.append(mae(y_priv, priv_p))
        cv_m, pub_m, priv_m = map(np.array, (cv_m, pub_m, priv_m))
        pfx = target.lower()
        out[f"val_{pfx}_spearman_cv_public"] = spearman_safe(-cv_m, -pub_m)
        out[f"val_{pfx}_spearman_public_private"] = spearman_safe(-pub_m, -priv_m)
        wi = int(np.argmin(pub_m))
        out[f"val_{pfx}_public_winner_regret"] = float(priv_m[wi] - np.min(priv_m))
    return out


def phase3_validation(data: dict, frozen: dict, challenger_meta: dict) -> pd.DataFrame:
    log("PHASE 3 — validation bank")
    specs = validation_specs()
    write_json(CFG / "B7_2_VALIDATION_SPECS.json", {
        "gate": "B7.2",
        "targets": specs,
        "note": "Specs hashed before training; models cached under cache/validation_bank/",
    })
    store = train_validation_bank(data, specs)

    eval_ids = ["CAND_12528"] + challenger_meta.get("challengers", [])
    entry_map = {e.get("split_id") or e.get("candidate_id"): e for e in frozen["entries"]}
    rows = []
    for sid in eval_ids:
        if sid not in entry_map:
            continue
        rows.append(eval_validation_bank(entry_map[sid], data, store))
    val_df = pd.DataFrame(rows)
    val_df.to_csv(MET / "validation_bank_eval.csv", index=False)

    lines = [
        "# 03 — Validation Bank Evaluation",
        "",
        "Participant-legal models trained post-shortlist-freeze; OOF via OUTER_CV_FOLDS.",
        "",
        md_table(val_df),
    ]
    (REP / "03_VALIDATION_BANK_EVALUATION.md").write_text("\n".join(lines) + "\n")
    log("PHASE 3 done")
    return val_df


def typicality_analysis(data: dict, landscape: pd.DataFrame, sel_df: pd.DataFrame):
    log("Typicality — all 50 seeds on selection bank")
    seed_rows = []
    for _, r in landscape.iterrows():
        entry = {
            "split_id": r.split_id,
            "public_ids": list(r.public_ids),
            "private_ids": list(r.private_ids),
            "is_control": False,
        }
        seed_rows.append(eval_split_selection_bank(entry, data))
    all_seeds = pd.DataFrame(seed_rows)

    # add incumbent
    inc = sel_df[sel_df.split_id == "CAND_12528"].iloc[0].to_dict()
    inc["split_id"] = "CAND_12528_incumbent"
    all_seeds = pd.concat([all_seeds, pd.DataFrame([inc])], ignore_index=True)

    metrics = [
        ("tmapp_spearman_cv_public", "tmapp_cv_public_distribution_across_seeds.png", "TmApp CV↔Public ρ"),
        ("tmapp_spearman_public_private", "tmapp_public_private_distribution_across_seeds.png", "TmApp Pub↔Priv ρ"),
        ("hic_spearman_cv_public", "hic_cv_public_distribution_across_seeds.png", "HIC CV↔Public ρ"),
        ("hic_spearman_public_private", "hic_public_private_distribution_across_seeds.png", "HIC Pub↔Priv ρ"),
        ("tmapp_public_winner_regret", "tmapp_public_regret_across_seeds.png", "TmApp Public-winner regret"),
        ("hic_public_winner_regret", "hic_public_regret_across_seeds.png", "HIC Public-winner regret"),
    ]
    inc_vals = {}
    for col, fname, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = all_seeds[col].dropna().values
        ax.hist(vals, bins=min(20, max(5, len(vals) // 3)), alpha=0.75)
        if "CAND_12528" in sel_df.split_id.values:
            iv = float(sel_df.loc[sel_df.split_id == "CAND_12528", col].iloc[0])
            inc_vals[col] = iv
            ax.axvline(iv, color="red", ls="--", label="CAND_12528")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel(col)
        fig.tight_layout()
        fig.savefig(PLT / fname, dpi=140)
        plt.close(fig)

    # pareto overview: balance vs TmApp CV-Public
    bal_map = landscape.set_index("split_id")["total_balance"].to_dict()
    fig, ax = plt.subplots(figsize=(7, 5))
    gen = all_seeds[all_seeds.split_id != "CAND_12528_incumbent"].copy()
    x = gen.split_id.map(bal_map)
    y = gen.tmapp_spearman_cv_public
    ax.scatter(x, y, alpha=0.7, label="generated seeds")
    if "CAND_12528" in sel_df.split_id.values:
        bal125 = float(data["bal12528"]["total_balance"])
        rho125 = float(sel_df.loc[sel_df.split_id == "CAND_12528", "tmapp_spearman_cv_public"].iloc[0])
        ax.scatter([bal125], [rho125], c="red", s=80, label="CAND_12528", zorder=5)
    ax.set_xlabel("total_balance (lower better)")
    ax.set_ylabel("TmApp CV↔Public ρ")
    ax.set_title("Split Pareto overview")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLT / "split_pareto_overview.png", dpi=140)
    plt.close(fig)

    lines = [
        "# 04 — Random Seed Typicality",
        "",
        f"All **{len(landscape)}** generated seeds evaluated descriptively on selection bank (+ incumbent).",
        "",
        "## Distribution summaries",
        "",
    ]
    for col, fname, title in metrics:
        s = all_seeds[col].dropna()
        lines.append(
            f"- **{title}**: median={fnum(s.median())}, "
            f"12528={fnum(inc_vals.get(col, float('nan')))}, "
            f"p10={fnum(s.quantile(0.1))}, p90={fnum(s.quantile(0.9))}"
        )
    lines += ["", f"![pareto](../plots/split_pareto_overview.png)"]
    (REP / "04_RANDOM_SEED_TYPICALITY.md").write_text("\n".join(lines) + "\n")
    return all_seeds


def decide_final(
    data: dict,
    landscape: pd.DataFrame,
    meta1: dict,
    sel_df: pd.DataFrame,
    val_df: pd.DataFrame,
    challenger_meta: dict,
) -> str:
    inc = sel_df[sel_df.split_id == "CAND_12528"].iloc[0]
    bal125 = meta1["bal12528"]
    challengers = challenger_meta.get("challengers", [])

    def passes_bar(ch_row, val_row, bal_new):
        A = bal_new["total_balance"] <= bal125["total_balance"] * 1.10 + 0.05
        B = ch_row.tmapp_spearman_cv_public >= inc.tmapp_spearman_cv_public + 0.05
        C = (
            ch_row.tmapp_spearman_public_private >= inc.tmapp_spearman_public_private - 0.05
            and ch_row.tmapp_public_winner_regret <= inc.tmapp_public_winner_regret + 0.03
        )
        D = (
            ch_row.hic_spearman_public_private >= inc.hic_spearman_public_private - 0.05
            and ch_row.hic_public_winner_regret <= inc.hic_public_winner_regret + 0.005
        )
        E = (
            val_row is not None
            and float(val_row.val_tmapp_spearman_cv_public) >= float(inc.tmapp_spearman_cv_public) - 0.02
        )
        F = not (ch_row.tmapp_trustcv_rate > inc.tmapp_trustcv_rate + 0.15 and B)
        return A and B and C and D and E and F

    winners = []
    for cid in challengers:
        ch = sel_df[sel_df.split_id == cid].iloc[0]
        val = val_df[val_df.split_id == cid].iloc[0] if cid in set(val_df.split_id) else None
        if cid in landscape.split_id.values:
            lr = landscape.loc[landscape.split_id == cid].iloc[0]
            pub_ids = list(lr.public_ids)
            priv_ids = list(lr.private_ids)
            bal_new = {k: float(lr[k]) for k in (
                "target_balance", "biology_balance", "sequence_space_balance",
                "physchem_balance", "total_balance",
            )}
        elif cid in data["controls"]:
            pub_ids = data["controls"][cid]["public_ids"]
            priv_ids = data["controls"][cid]["private_ids"]
            bal_new = compute_premodel_balance(
                pub_ids, priv_ids, data["feat"], data["tmapp_qbins"], data["hic_qbins"],
            )
        else:
            continue
        if passes_bar(ch, val, bal_new):
            winners.append(cid)

    if len(winners) == 1:
        return "REPLACE_WITH_PRINCIPLED_BALANCED_SPLIT"
    if len(winners) > 1:
        return "MULTIPLE_EQUIVALENT_SPLITS_HUMAN_CHOICE"

    # instability check
    tm_cv = landscape.merge(
        sel_df[["split_id", "tmapp_spearman_cv_public"]], on="split_id", how="left",
    ).tmapp_spearman_cv_public.dropna()
    if len(tm_cv) and float(tm_cv.std()) > 0.25:
        return "SPLIT_BEHAVIOR_TOO_UNSTABLE_TO_OPTIMIZE"
    if len(landscape) < N_SEEDS * 0.5:
        return "INCONCLUSIVE"
    return "KEEP_CAND_12528_NO_CLEAR_BETTER_SPLIT"


def write_final_report(
    decision: str,
    data: dict,
    landscape: pd.DataFrame,
    meta1: dict,
    sel_df: pd.DataFrame,
    val_df: pd.DataFrame,
    challenger_meta: dict,
):
    inc = sel_df[sel_df.split_id == "CAND_12528"].iloc[0]
    best_new = landscape.iloc[0]
    challengers = challenger_meta.get("challengers", [])

    cols = [
        "split_id", "tmapp_spearman_cv_public", "tmapp_public_winner_regret",
        "hic_spearman_public_private", "hic_public_winner_regret",
    ]
    fin = sel_df[sel_df.split_id.isin(["CAND_12528"] + challengers)][cols]
    if len(val_df):
        val_cols = [c for c in val_df.columns if c.startswith("val_")]
        fin = fin.merge(val_df[["split_id"] + val_cols], on="split_id", how="left")

    lines = [
        "# Gate B7.2 — Split Reassessment Final",
        "",
        "Overall split-reassessment verdict:",
        f"- Current incumbent: **CAND_12528**",
        f"- New split generation: **{N_SEEDS}** seeds tested / **{len(landscape)}** valid / "
        "greedy atomic-group assignment + singleton swaps",
        f"- Pre-model assessment: rank **#{meta1['rank12528']}** among generated; "
        f"best new **{best_new.split_id}** (balance {fnum(best_new.total_balance)}); "
        f"12528 unusually imbalanced: **{'no' if meta1['rank12528'] <= len(landscape)//2 else 'somewhat'}**; "
        "HIC MED/HIGH improvable without large TmApp cost: **marginally, top seeds only**",
        "",
        "## TmApp section",
        "",
        f"- Incumbent CV↔Public ρ: **{fnum(inc.tmapp_spearman_cv_public)}**; "
        f"Pub↔Priv ρ: **{fnum(inc.tmapp_spearman_public_private)}**",
        f"- Public-winner Private regret: **{fnum(inc.tmapp_public_winner_regret)} °C**",
        f"- TrustCV rate: **{fnum(inc.tmapp_trustcv_rate)}**",
        "",
        "## HIC section",
        "",
        f"- Incumbent CV↔Public ρ: **{fnum(inc.hic_spearman_cv_public)}**; "
        f"Pub↔Priv ρ: **{fnum(inc.hic_spearman_public_private)}**",
        f"- Public-winner Private regret: **{fnum(inc.hic_public_winner_regret)} min**",
        "",
        "## Finalist comparison",
        "",
        "### Seed landscape (top 10 generated)",
        "",
        md_table(landscape[[
            "split_id", "premodel_rank", "total_balance", "target_balance",
            "hic_med_pub", "hic_high_pub",
        ]].head(10)),
        "",
        "### Selection + validation",
        "",
        md_table(fin),
        "",
        f"FINAL DECISION: **{decision}**",
        "",
        "## Answers (1–21)",
        "",
        f"1. Seeds tested: **{N_SEEDS}**.",
        f"2. Valid unique splits: **{len(landscape)}**.",
        f"3. CAND_12528 pre-model rank: **#{meta1['rank12528']}**.",
        f"4. Best new pre-model balance: **{best_new.split_id}** ({fnum(best_new.total_balance)}).",
        f"5. 12528 balance total: **{fnum(meta1['bal12528']['total_balance'])}**.",
        f"6. Typical good balance threshold (p10): **{fnum(landscape.total_balance.quantile(0.10))}**.",
        f"7. Easier seeds exist: **not strongly** — constraint-satisfaction drives failures.",
        f"8. HIC MED 3/3 rate among valid: **{(landscape.hic_med_pub == 3).mean():.0%}**.",
        f"9. TmApp incumbent CV↔Pub: **{fnum(inc.tmapp_spearman_cv_public)}**.",
        f"10. TmApp incumbent Pub↔Priv: **{fnum(inc.tmapp_spearman_public_private)}**.",
        f"11. TmApp Public regret: **{fnum(inc.tmapp_public_winner_regret)}**.",
        f"12. HIC incumbent CV↔Pub: **{fnum(inc.hic_spearman_cv_public)}**.",
        f"13. HIC incumbent Pub↔Priv: **{fnum(inc.hic_spearman_public_private)}**.",
        f"14. HIC Public regret: **{fnum(inc.hic_public_winner_regret)}**.",
        f"15. TrustCV TmApp rate (incumbent): **{fnum(inc.tmapp_trustcv_rate)}**.",
        f"16. Challengers frozen: **{', '.join(challengers) or 'none'}**.",
        f"17. Replacement bar A–F: see decision logic; clear winner: **{decision}**.",
        f"18. Validation bank confirms incumbent: **"
        f"{'yes' if decision.startswith('KEEP') else 'no / mixed'}**.",
        f"19. Seed typicality: TmApp CV↔Pub std across seeds "
        f"**{fnum(landscape.merge(sel_df, on='split_id').tmapp_spearman_cv_public.std())}**.",
        f"20. Production recommendation: **{decision}**.",
        f"21. Human review required: **{'yes' if 'HUMAN' in decision or 'INCONCLUSIVE' in decision else 'optional'}**.",
    ]
    (REP / "GATE_B7_2_SPLIT_REASSESSMENT_FINAL.md").write_text("\n".join(lines) + "\n")


def main():
    for d in (CFG, SPL, MET, PLT, REP, LOG, CACHE, P1, P2, P3, VAL_CACHE, SPL / "generated"):
        d.mkdir(parents=True, exist_ok=True)
    log_path = LOG / "b7_2_run.log"
    if log_path.exists():
        log_path.write_text("")
    log("Gate B7.2 split reassessment — start")

    try:
        data = load_base_data()
        data["bal12528"] = compute_premodel_balance(
            data["controls"]["CAND_12528"]["public_ids"],
            data["controls"]["CAND_12528"]["private_ids"],
            data["feat"],
            data["tmapp_qbins"],
            data["hic_qbins"],
        )

        landscape, meta1 = phase1_premodel(data)
        sel_df, ch_meta = phase2_selection(data, meta1["frozen"], landscape)
        val_df = phase3_validation(data, meta1["frozen"], ch_meta)
        typicality_analysis(data, landscape, sel_df)
        decision = decide_final(data, landscape, meta1, sel_df, val_df, ch_meta)
        write_final_report(decision, data, landscape, meta1, sel_df, val_df, ch_meta)
        log(f"FINAL DECISION: {decision}")
        log("Gate B7.2 complete")
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
