#!/usr/bin/env python3
"""Gate B6.1 — Final production split bake-off (no new search / no retrain)."""
from __future__ import annotations

import ast
import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, spearmanr, wasserstein_distance

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/workspace_developability_acquisition")
B6 = ROOT / "gate_b6_split_search"
CFG = B6 / "config"
MET = B6 / "metrics"
REP = B6 / "reports"
PLOTS = B6 / "plots"
ORG = ROOT / "gate_b3/frozen/organizer"
PRED = ROOT / "gate_b5_ceiling/predictions"
OUTER_PATH = ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json"
FEAT_PATH = B6 / "cache/test_balance_features.csv"
MODEL_SET_PATH = CFG / "FROZEN_SPLIT_EVALUATION_MODEL_SET.json"
SHORTLIST_PATH = CFG / "TMAPP_SHORTLIST.json"

SEED = 20260830
N_BOOT = 10000
N_PERTURB = 300
HIC_SD = 0.857
HIC_IQR = 0.694

CAND_IDS = ["CAND_04974", "CAND_12528", "CAND_12207"]
BASELINE_ID = "CURRENT_BASELINE_SPLIT"
ALL_SPLIT_IDS = CAND_IDS + [BASELINE_ID]

TMAPP_TAGS = [
    ("CONST_MEDIAN", "CONST"),
    ("SEQ_SIMPLE_Ridge", "SEQ_SIMPLE"),
    ("BIO_Ridge", "BIO"),
    ("PLM_ABLANG2_PCA32_SVR", "ABLANG2_NONLINEAR"),
    ("NESTED_STACK_MEAN", "NESTED_ENSEMBLE"),
]
HIC_TAGS = [
    ("CONST_MEDIAN", "CONST"),
    ("SEQ_SIMPLE_Ridge", "SEQ_SIMPLE"),
    ("PLM_ESM2_PCA64_SVR", "ESM2_PLM"),
    ("ESMFN_STRUCTURE_ElasticNet", "ESMFOLD_STRUCTURE"),
    ("FUSION_ESM2_ESMFN_ElasticNet", "PLM_STRUCTURE_FUSION"),
    ("NESTED_STACK_NNLS", "NESTED_ENSEMBLE"),
]
STRATEGIES = ["A_public_only", "B_cv_only", "C_cv_among_public_top2", "D_consensus"]

HIC_REGRET_THRESH = {
    "0.05*SD": 0.05 * HIC_SD,
    "0.1*SD": 0.1 * HIC_SD,
    "0.25*SD": 0.25 * HIC_SD,
    "0.5*SD": 0.5 * HIC_SD,
}
TMAPP_REGRET_THRESH = {"0.1C": 0.1, "0.25C": 0.25, "0.5C": 0.5, "1.0C": 1.0}

CONT_FEATS = [
    "TmApp", "HIC", "cluster_size", "A2_HL_charge_ph7", "A2_HL_pI", "A2_HL_gravy",
    "C_vh_germline_distance", "C_vl_germline_distance", "C_combined_germline_distance",
    "vh_len", "vl_len", "H_CDR3_len", "L_CDR3_len",
    "nearest_train_VH_identity", "nearest_train_VL_identity",
]
CAT_FEATS = ["vh_family", "vl_family", "C_kappa_lambda"]
DIAG_CAT = ["donor", "b_cell_subset"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    for d in (CFG, MET, REP, PLOTS, B6 / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def write_json(path: Path, obj) -> str:
    b = json.dumps(obj, indent=2, sort_keys=True, default=float).encode()
    path.write_bytes(b)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_bytes(b) + "\n")
    return sha256_bytes(b)


def fnum(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


# ---------------------------------------------------------------------------
# Metrics / ranking
# ---------------------------------------------------------------------------
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def pearson(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def rank_corr(a, b, method="spearman") -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    if method == "spearman":
        r = spearmanr(a[m], b[m]).correlation
    else:
        r = kendalltau(a[m], b[m]).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def nan_rank(values, higher_better: bool) -> np.ndarray:
    """Average ranks; NaN treated as worst."""
    a = np.asarray(values, float)
    out = np.empty(len(a), float)
    finite = np.isfinite(a)
    n_ok = int(finite.sum())
    if n_ok == 0:
        out[:] = np.arange(1, len(a) + 1)
        return out
    s = pd.Series(a[finite])
    r = s.rank(ascending=not higher_better, method="average").values
    out[finite] = r
    out[~finite] = float(n_ok + 1)  # worst
    return out


def pick_with_ties(indices, cv_mae, model_ids) -> int:
    """Among candidate indices, lower CV MAE then lex model_id."""
    best = None
    for i in indices:
        key = (float(cv_mae[i]), str(model_ids[i]), int(i))
        if best is None or key < best[0]:
            best = (key, i)
    return int(best[1])


def strategy_choose(pub_mae, cv_mae, model_ids, name: str) -> int:
    pub_mae = np.asarray(pub_mae, float)
    cv_mae = np.asarray(cv_mae, float)
    n = len(model_ids)
    if name == "A_public_only":
        best = np.nanmin(pub_mae)
        cands = [i for i in range(n) if np.isfinite(pub_mae[i]) and abs(pub_mae[i] - best) < 1e-15]
        if not cands:
            cands = list(range(n))
        return pick_with_ties(cands, cv_mae, model_ids)
    if name == "B_cv_only":
        best = np.nanmin(cv_mae)
        cands = [i for i in range(n) if np.isfinite(cv_mae[i]) and abs(cv_mae[i] - best) < 1e-15]
        return pick_with_ties(cands, cv_mae, model_ids)
    if name == "C_cv_among_public_top2":
        order = np.argsort(np.where(np.isfinite(pub_mae), pub_mae, np.inf))
        top2 = order[: min(2, n)].tolist()
        return pick_with_ties(top2, cv_mae, model_ids)
    if name == "D_consensus":
        cv_r = nan_rank(cv_mae, higher_better=False)
        pub_r = nan_rank(pub_mae, higher_better=False)
        cons = cv_r + pub_r
        best = np.nanmin(cons)
        cands = [i for i in range(n) if abs(cons[i] - best) < 1e-12]
        return pick_with_ties(cands, cv_mae, model_ids)
    raise ValueError(name)


def smd(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)))
    return 0.0 if pooled < 1e-12 else float((a.mean() - b.mean()) / pooled)


def js_div(p, q, eps=1e-12) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log((x[mask] + eps) / (y[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cat_js(pub_s, priv_s) -> float:
    levels = sorted(set(pub_s.astype(str)) | set(priv_s.astype(str)))
    pa = np.array([(pub_s.astype(str) == lv).mean() for lv in levels])
    pb = np.array([(priv_s.astype(str) == lv).mean() for lv in levels])
    return js_div(pa, pb)


def rate(v, excellent, good, acceptable, higher_better=True) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "CONCERNING"
    if not np.isfinite(v):
        return "CONCERNING"
    if higher_better:
        if v >= excellent:
            return "EXCELLENT"
        if v >= good:
            return "GOOD"
        if v >= acceptable:
            return "ACCEPTABLE"
        return "CONCERNING"
    if v <= excellent:
        return "EXCELLENT"
    if v <= good:
        return "GOOD"
    if v <= acceptable:
        return "ACCEPTABLE"
    return "CONCERNING"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_base():
    role = pd.read_csv(ORG / "role_map.csv")
    pub_order = role.loc[role.role == "Public", "id"].tolist()
    priv_order = role.loc[role.role == "Private", "id"].tolist()
    outer = json.loads(OUTER_PATH.read_text())
    train_ids = list(outer["train_ids"])
    assert len(train_ids) == 162
    lab = pd.read_csv(ORG / "test_labels_hidden.csv").set_index("id")
    feat = pd.read_csv(FEAT_PATH)
    model_set = json.loads(MODEL_SET_PATH.read_text())
    short = json.loads(SHORTLIST_PATH.read_text())
    cand_map = {c["candidate_id"]: c for c in short["candidates"]}
    for cid in CAND_IDS:
        assert cid in cand_map, cid
    base_pub = sorted(pub_order)
    base_priv = sorted(priv_order)
    splits = {}
    for cid in CAND_IDS:
        c = cand_map[cid]
        splits[cid] = {
            "candidate_id": cid,
            "public_ids": list(c["public_ids"]),
            "private_ids": list(c["private_ids"]),
            "public_hash": c["public_hash"],
            "private_hash": c["private_hash"],
            "is_baseline": False,
        }
        assert len(splits[cid]["public_ids"]) == 81
        assert len(splits[cid]["private_ids"]) == 81
    splits[BASELINE_ID] = {
        "candidate_id": BASELINE_ID,
        "public_ids": base_pub,
        "private_ids": base_priv,
        "public_hash": sha256_ids(base_pub),
        "private_hash": sha256_ids(base_priv),
        "is_baseline": True,
    }
    # id -> sequence_group for test
    id_group = dict(zip(feat["id"].astype(str), feat["sequence_group"].astype(int)))
    return role, pub_order, priv_order, train_ids, lab, feat, model_set, splits, id_group


def prepare_target_pack(target, model_set, pub_order, priv_order, train_ids, role, lab):
    tags = TMAPP_TAGS if target == "TmApp" else HIC_TAGS
    test_ids = list(pub_order) + list(priv_order)
    y_train = role.set_index("id").loc[train_ids, target].values.astype(float)
    models = []
    for tag, fam in tags:
        pub = np.load(PRED / "final" / f"{target}__{tag}__public.npy")
        priv = np.load(PRED / "final" / f"{target}__{tag}__private.npy")
        oof = np.load(PRED / "oof" / f"{target}__{tag}__meanOOF.npy")
        assert len(pub) == 81 and len(priv) == 81 and len(oof) == 162
        pred_by_id = dict(zip(test_ids, np.concatenate([pub, priv])))
        cv_mae_v = mae(y_train, oof)
        cv_pear_v = pearson(y_train, oof)
        mid = f"{target}__{tag}"
        for m in model_set.get(target, []):
            if m["tag"] == tag:
                cv_mae_v = float(m["CV_metrics"]["mae"])
                cp = m["CV_metrics"]["pearson"]
                cv_pear_v = float(cp) if cp is not None else float("nan")
                mid = m["model_id"]
                fam = m["model_family"]
                break
        models.append(
            {
                "model_id": mid,
                "tag": tag,
                "family": fam,
                "pred_by_id": pred_by_id,
                "cv_mae": cv_mae_v,
                "cv_pearson": cv_pear_v,
            }
        )
    y_by_id = {i: float(lab.loc[i, target]) for i in test_ids}
    # also include any candidate ids that might differ — labels cover all test
    for i in lab.index:
        y_by_id[str(i)] = float(lab.loc[i, target])
    return models, y_by_id


def scores_on_ids(models, y_by_id, ids):
    y = np.array([y_by_id[i] for i in ids], float)
    mae_a, pear_a = [], []
    for m in models:
        p = np.array([m["pred_by_id"][i] for i in ids], float)
        mae_a.append(mae(y, p))
        pear_a.append(pearson(y, p))
    return np.array(mae_a, float), np.array(pear_a, float)


def transfer_block(cv_mae, cv_pear, pub_mae, priv_mae, pub_pear, priv_pear) -> dict:
    return {
        "mae_cv_public_spearman": rank_corr(-cv_mae, -pub_mae),
        "mae_public_private_spearman": rank_corr(-pub_mae, -priv_mae),
        "mae_cv_private_spearman": rank_corr(-cv_mae, -priv_mae),
        "mae_cv_public_kendall": rank_corr(-cv_mae, -pub_mae, "kendall"),
        "mae_public_private_kendall": rank_corr(-pub_mae, -priv_mae, "kendall"),
        "mae_cv_private_kendall": rank_corr(-cv_mae, -priv_mae, "kendall"),
        "pear_cv_public_spearman": rank_corr(cv_pear, pub_pear),
        "pear_public_private_spearman": rank_corr(pub_pear, priv_pear),
        "pear_cv_private_spearman": rank_corr(cv_pear, priv_pear),
        "pear_cv_public_kendall": rank_corr(cv_pear, pub_pear, "kendall"),
        "pear_public_private_kendall": rank_corr(pub_pear, priv_pear, "kendall"),
        "pear_cv_private_kendall": rank_corr(cv_pear, priv_pear, "kendall"),
    }


def pairwise_inversions(pub_mae, priv_mae):
    n = len(pub_mae)
    inv = 0
    pairs = 0
    burden = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            pub_ord = np.sign(pub_mae[i] - pub_mae[j])
            priv_ord = np.sign(priv_mae[i] - priv_mae[j])
            if pub_ord == 0 or priv_ord == 0:
                continue
            if pub_ord != priv_ord:
                inv += 1
                burden += abs(priv_mae[i] - priv_mae[j])
    return inv, pairs, burden, (inv / pairs if pairs else float("nan"))


def topk_private_ranks(pub_mae, priv_mae, k=3):
    order = np.argsort(pub_mae)
    priv_r = nan_rank(priv_mae, higher_better=False)
    ranks = [int(priv_r[i]) for i in order[:k]]
    return ranks


def freeze_configs(splits, model_set):
    cand_payload = {
        "gate": "B6.1",
        "purpose": "Final production split bake-off candidate freeze",
        "candidates": [
            {
                "candidate_id": cid,
                "public_ids": splits[cid]["public_ids"],
                "private_ids": splits[cid]["private_ids"],
                "public_hash": splits[cid]["public_hash"],
                "private_hash": splits[cid]["private_hash"],
            }
            for cid in CAND_IDS
        ],
        "control": {
            "candidate_id": BASELINE_ID,
            "public_ids": splits[BASELINE_ID]["public_ids"],
            "private_ids": splits[BASELINE_ID]["private_ids"],
            "public_hash": splits[BASELINE_ID]["public_hash"],
            "private_hash": splits[BASELINE_ID]["private_hash"],
        },
        "rules": [
            "DO NOT add another split",
            "DO NOT remove a candidate because an early result looks poor",
            "NO new split search",
            "NO model retraining",
        ],
        "seed": SEED,
    }
    h1 = write_json(CFG / "B6_1_FINAL_CANDIDATE_SET.json", cand_payload)
    protocol = {
        "gate": "B6.1",
        "title": "Final production split bake-off analysis protocol",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "local_perturbations_per_candidate": N_PERTURB,
        "perturbation_k": [1, 2, 3],
        "strategies": {
            "A_public_only": "min Public MAE",
            "B_cv_only": "min CV MAE",
            "C_cv_among_public_top2": "among 2 lowest Public MAE, pick lower CV MAE",
            "D_consensus": "min(CV_rank + Public_rank); ties → lower CV MAE → lex model_id",
        },
        "tie_break": ["lower CV MAE", "lexicographic model_id"],
        "regret_thresholds": {
            "TmApp_C": TMAPP_REGRET_THRESH,
            "HIC": {
                **{k: float(v) for k, v in HIC_REGRET_THRESH.items()},
                "reference_Test_HIC_SD": HIC_SD,
                "reference_Test_HIC_IQR": HIC_IQR,
                "note": "thresholds are fractions of Test HIC SD (0.857); IQR (0.694) documented for context",
            },
        },
        "decision_priority": [
            "TmApp must be fixed (Public not misleading)",
            "low Public-selection regret for BOTH targets",
            "HIC remains useful feedback",
            "statistical/biological balance",
            "local/bootstrap robustness",
        ],
        "allowed_decisions": [
            "SELECT_CAND_04974",
            "SELECT_CAND_12528",
            "SELECT_CAND_12207",
            "NO_CLEAR_WINNER_REQUIRE_HUMAN_DECISION",
        ],
        "model_set_path": str(MODEL_SET_PATH),
        "TmApp_n_models": len(model_set.get("TmApp", [])),
        "HIC_n_models": len(model_set.get("HIC", [])),
        "prediction_correlations_Test": model_set.get("prediction_correlations_Test", {}),
    }
    h2 = write_json(CFG / "B6_1_ANALYSIS_PROTOCOL.json", protocol)
    print(f"Froze configs: candidate_set={h1[:12]}… protocol={h2[:12]}…")
    return h1, h2


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def score_split(cid, split, packs):
    rows_pm = []
    rows_tr = []
    rows_reg = []
    rows_strat = []
    rows_inv = []
    detail = {}
    for target, (models, y_by_id) in packs.items():
        pub_ids = split["public_ids"]
        priv_ids = split["private_ids"]
        cv_mae = np.array([m["cv_mae"] for m in models], float)
        cv_pear = np.array([m["cv_pearson"] for m in models], float)
        mids = [m["model_id"] for m in models]
        pub_mae, pub_pear = scores_on_ids(models, y_by_id, pub_ids)
        priv_mae, priv_pear = scores_on_ids(models, y_by_id, priv_ids)
        pub_mae_r = nan_rank(pub_mae, False)
        priv_mae_r = nan_rank(priv_mae, False)
        pub_pear_r = nan_rank(pub_pear, True)
        priv_pear_r = nan_rank(priv_pear, True)
        for i, m in enumerate(models):
            rows_pm.append(
                {
                    "candidate_id": cid,
                    "target": target,
                    "model_id": m["model_id"],
                    "tag": m["tag"],
                    "family": m["family"],
                    "cv_mae": cv_mae[i],
                    "public_mae": pub_mae[i],
                    "private_mae": priv_mae[i],
                    "cv_pearson": cv_pear[i],
                    "public_pearson": pub_pear[i],
                    "private_pearson": priv_pear[i],
                    "public_mae_rank": pub_mae_r[i],
                    "private_mae_rank": priv_mae_r[i],
                    "public_pearson_rank": pub_pear_r[i],
                    "private_pearson_rank": priv_pear_r[i],
                }
            )
        tr = transfer_block(cv_mae, cv_pear, pub_mae, priv_mae, pub_pear, priv_pear)
        tr.update({"candidate_id": cid, "target": target})
        # separation
        tr["public_mae_best"] = float(np.nanmin(pub_mae))
        tr["public_mae_worst"] = float(np.nanmax(pub_mae))
        tr["public_mae_range"] = tr["public_mae_worst"] - tr["public_mae_best"]
        tr["private_mae_best"] = float(np.nanmin(priv_mae))
        tr["private_mae_worst"] = float(np.nanmax(priv_mae))
        tr["private_mae_range"] = tr["private_mae_worst"] - tr["private_mae_best"]
        diffs = [abs(pub_mae[i] - pub_mae[j]) for i in range(len(pub_mae)) for j in range(i + 1, len(pub_mae))]
        tr["public_mae_median_pairwise_absdiff"] = float(np.median(diffs)) if diffs else float("nan")
        diffs_p = [abs(priv_mae[i] - priv_mae[j]) for i in range(len(priv_mae)) for j in range(i + 1, len(priv_mae))]
        tr["private_mae_median_pairwise_absdiff"] = float(np.median(diffs_p)) if diffs_p else float("nan")
        # top-k
        top3 = topk_private_ranks(pub_mae, priv_mae, 3)
        tr["public_top1_private_rank"] = top3[0] if top3 else float("nan")
        tr["public_top2_private_ranks"] = ",".join(map(str, top3[:2]))
        tr["public_top3_private_ranks"] = ",".join(map(str, top3))
        tr["worst_private_rank_among_public_top3"] = max(top3) if top3 else float("nan")
        pub_top2 = set(np.argsort(pub_mae)[:2])
        priv_top2 = set(np.argsort(priv_mae)[:2])
        pub_top3s = set(np.argsort(pub_mae)[:3])
        priv_top3s = set(np.argsort(priv_mae)[:3])
        tr["public_top2_intersect_private_top2"] = len(pub_top2 & priv_top2)
        tr["public_top3_intersect_private_top3"] = len(pub_top3s & priv_top3s)
        rows_tr.append(tr)

        # Public MAE winner regret
        wi = int(np.nanargmin(pub_mae))
        best_priv = float(np.nanmin(priv_mae))
        regret = float(priv_mae[wi] - best_priv)
        # Pearson winner regret
        valid_pear = np.where(np.isfinite(pub_pear))[0]
        if len(valid_pear):
            wi_p = int(valid_pear[np.nanargmax(pub_pear[valid_pear])])
            best_priv_p = float(np.nanmax(priv_pear[np.isfinite(priv_pear)])) if np.any(np.isfinite(priv_pear)) else float("nan")
            pear_regret = float(best_priv_p - priv_pear[wi_p]) if np.isfinite(best_priv_p) else float("nan")
        else:
            wi_p = -1
            pear_regret = float("nan")
        # CV-only
        wi_cv = int(np.nanargmin(cv_mae))
        rows_reg.append(
            {
                "candidate_id": cid,
                "target": target,
                "public_mae_winner": mids[wi],
                "public_winner_private_rank": float(priv_mae_r[wi]),
                "public_winner_private_mae": float(priv_mae[wi]),
                "best_private_mae": best_priv,
                "public_winner_regret": regret,
                "public_pearson_winner": mids[wi_p] if wi_p >= 0 else None,
                "public_pearson_winner_private_rank": float(priv_pear_r[wi_p]) if wi_p >= 0 else float("nan"),
                "public_pearson_winner_private_pearson": float(priv_pear[wi_p]) if wi_p >= 0 else float("nan"),
                "best_private_pearson": float(np.nanmax(priv_pear)) if np.any(np.isfinite(priv_pear)) else float("nan"),
                "public_pearson_regret": pear_regret,
                "cv_winner": mids[wi_cv],
                "cv_winner_public_rank": float(pub_mae_r[wi_cv]),
                "cv_winner_private_rank": float(priv_mae_r[wi_cv]),
                "cv_winner_private_mae": float(priv_mae[wi_cv]),
                "cv_winner_regret": float(priv_mae[wi_cv] - best_priv),
            }
        )
        for sname in STRATEGIES:
            si = strategy_choose(pub_mae, cv_mae, mids, sname)
            rows_strat.append(
                {
                    "candidate_id": cid,
                    "target": target,
                    "strategy": sname,
                    "chosen_model": mids[si],
                    "private_rank": float(priv_mae_r[si]),
                    "private_mae": float(priv_mae[si]),
                    "regret": float(priv_mae[si] - best_priv),
                }
            )
        inv, pairs, burden, frac = pairwise_inversions(pub_mae, priv_mae)
        rows_inv.append(
            {
                "candidate_id": cid,
                "target": target,
                "metric": "MAE",
                "n_inversions": inv,
                "n_pairs": pairs,
                "fraction_inverted": frac,
                "effect_size_weighted_burden": burden,
            }
        )
        # Pearson inversions (higher better → invert sign)
        invp, pairsp, burdenp, fracp = pairwise_inversions(-pub_pear, -priv_pear)
        rows_inv.append(
            {
                "candidate_id": cid,
                "target": target,
                "metric": "Pearson",
                "n_inversions": invp,
                "n_pairs": pairsp,
                "fraction_inverted": fracp,
                "effect_size_weighted_burden": burdenp,
            }
        )
        detail[target] = {
            "models": models,
            "y_by_id": y_by_id,
            "pub_mae": pub_mae,
            "priv_mae": priv_mae,
            "pub_pear": pub_pear,
            "priv_pear": priv_pear,
            "cv_mae": cv_mae,
            "cv_pear": cv_pear,
            "mids": mids,
            "transfer": tr,
            "public_winner_regret": regret,
            "public_winner": mids[wi],
            "public_winner_private_rank": float(priv_mae_r[wi]),
        }
    return rows_pm, rows_tr, rows_reg, rows_strat, rows_inv, detail


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def run_bootstrap(cid, split, packs, rng):
    boot_rows = []
    stab_rows = []
    regret_hist = {}
    for target, (models, y_by_id) in packs.items():
        pub = list(split["public_ids"])
        priv = list(split["private_ids"])
        n_pub, n_priv = len(pub), len(priv)
        cv_mae = np.array([m["cv_mae"] for m in models], float)
        mids = [m["model_id"] for m in models]
        n_m = len(models)
        # pre-extract pred matrices
        pred_mat = np.zeros((n_m, n_pub + n_priv), float)
        y_pub0 = np.array([y_by_id[i] for i in pub], float)
        y_priv0 = np.array([y_by_id[i] for i in priv], float)
        for i, m in enumerate(models):
            pred_mat[i, :n_pub] = [m["pred_by_id"][x] for x in pub]
            pred_mat[i, n_pub:] = [m["pred_by_id"][x] for x in priv]

        strat_regrets = {s: [] for s in STRATEGIES}
        pub_win_counts = np.zeros(n_m)
        priv_win_counts = np.zeros(n_m)
        topk_hits = {1: 0, 2: 0, 3: 0}

        for _b in range(N_BOOT):
            ip = rng.integers(0, n_pub, n_pub)
            ir = rng.integers(0, n_priv, n_priv)
            yp = y_pub0[ip]
            yr = y_priv0[ir]
            pp = pred_mat[:, ip]
            pr = pred_mat[:, n_pub:][:, ir]
            pub_mae = np.mean(np.abs(pp - yp[None, :]), axis=1)
            priv_mae = np.mean(np.abs(pr - yr[None, :]), axis=1)
            best_priv = float(np.min(priv_mae))
            wi = int(np.argmin(pub_mae))
            priv_order = np.argsort(priv_mae)
            pub_win_counts[wi] += 1
            priv_win_counts[int(priv_order[0])] += 1
            prank = int(np.where(priv_order == wi)[0][0]) + 1
            for k in (1, 2, 3):
                if prank <= k:
                    topk_hits[k] += 1
            for sname in STRATEGIES:
                si = strategy_choose(pub_mae, cv_mae, mids, sname)
                strat_regrets[sname].append(float(priv_mae[si] - best_priv))

        thresh = TMAPP_REGRET_THRESH if target == "TmApp" else HIC_REGRET_THRESH
        for sname, vals in strat_regrets.items():
            a = np.asarray(vals, float)
            row = {
                "candidate_id": cid,
                "target": target,
                "strategy": sname,
                "n_boot": N_BOOT,
                "median_regret": float(np.median(a)),
                "mean_regret": float(np.mean(a)),
                "p75_regret": float(np.quantile(a, 0.75)),
                "p90_regret": float(np.quantile(a, 0.90)),
                "p95_regret": float(np.quantile(a, 0.95)),
                "p99_regret": float(np.quantile(a, 0.99)),
                "max_regret": float(np.max(a)),
            }
            for tk, tv in thresh.items():
                row[f"P_regret_gt_{tk}"] = float(np.mean(a > tv))
            boot_rows.append(row)
            if sname == "A_public_only":
                regret_hist[(cid, target)] = a

        for i, mid in enumerate(mids):
            stab_rows.append(
                {
                    "candidate_id": cid,
                    "target": target,
                    "model_id": mid,
                    "P_public_winner": float(pub_win_counts[i] / N_BOOT),
                    "P_private_winner": float(priv_win_counts[i] / N_BOOT),
                }
            )
        stab_rows.append(
            {
                "candidate_id": cid,
                "target": target,
                "model_id": "__SUMMARY__",
                "P_public_winner": float("nan"),
                "P_private_winner": float("nan"),
                "P_pub_winner_priv_top1": float(topk_hits[1] / N_BOOT),
                "P_pub_winner_priv_top2": float(topk_hits[2] / N_BOOT),
                "P_pub_winner_priv_top3": float(topk_hits[3] / N_BOOT),
            }
        )
    return boot_rows, stab_rows, regret_hist


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------
def dist_stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {}
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    out = {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
        "iqr": float(np.quantile(a, 0.75) - np.quantile(a, 0.25)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(a, q))
    return out


def balance_target(cid, split, feat):
    rows = []
    pub = feat[feat.id.isin(split["public_ids"])]
    priv = feat[feat.id.isin(split["private_ids"])]
    for col in ("TmApp", "HIC"):
        a = pub[col].values.astype(float)
        b = priv[col].values.astype(float)
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        sa, sb = dist_stats(aa), dist_stats(bb)
        w = float(wasserstein_distance(aa, bb)) if len(aa) and len(bb) else float("nan")
        ks = float(ks_2samp(aa, bb).statistic) if len(aa) and len(bb) else float("nan")
        row = {"candidate_id": cid, "feature": col, "wasserstein": w, "ks": ks, "smd": smd(aa, bb)}
        for k, v in sa.items():
            row[f"public_{k}"] = v
        for k, v in sb.items():
            row[f"private_{k}"] = v
        rows.append(row)
    return rows


def balance_features(cid, split, feat):
    rows = []
    pub = feat[feat.id.isin(split["public_ids"])]
    priv = feat[feat.id.isin(split["private_ids"])]
    for col in CONT_FEATS:
        if col not in feat.columns:
            continue
        a = pub[col].values.astype(float)
        b = priv[col].values.astype(float)
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(aa) < 5 or len(bb) < 5:
            continue
        pooled = float(np.std(np.concatenate([aa, bb])) + 1e-8)
        rows.append(
            {
                "candidate_id": cid,
                "feature": col,
                "type": "continuous",
                "diagnostic_only": False,
                "wasserstein_norm": float(wasserstein_distance(aa, bb)) / pooled,
                "ks": float(ks_2samp(aa, bb).statistic),
                "smd": abs(smd(aa, bb)),
                "js": float("nan"),
                "imbalance": abs(smd(aa, bb)),
            }
        )
    for col in CAT_FEATS + DIAG_CAT:
        if col not in feat.columns:
            continue
        j = cat_js(pub[col], priv[col])
        rows.append(
            {
                "candidate_id": cid,
                "feature": col,
                "type": "categorical",
                "diagnostic_only": col in DIAG_CAT,
                "wasserstein_norm": float("nan"),
                "ks": float("nan"),
                "smd": float("nan"),
                "js": j,
                "imbalance": j,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Local sensitivity
# ---------------------------------------------------------------------------
def perturb_once(pub_ids, priv_ids, id_group, k, rng):
    pub_set, priv_set = set(pub_ids), set(priv_ids)
    all_ids = list(pub_set | priv_set)
    g_all = defaultdict(list)
    for i in all_ids:
        g_all[int(id_group[i])].append(i)
    pub_groups = [g for g, mem in g_all.items() if set(mem).issubset(pub_set)]
    priv_groups = [g for g, mem in g_all.items() if set(mem).issubset(priv_set)]
    if not pub_groups or not priv_groups:
        return None
    ks = min(int(k), len(pub_groups))
    for _try in range(80):
        chosen_pub = list(rng.choice(pub_groups, size=ks, replace=False))
        need = sum(len(g_all[g]) for g in chosen_pub)
        priv_shuf = list(rng.permutation(priv_groups))
        chosen_priv, acc = [], 0
        for g in priv_shuf:
            sz = len(g_all[g])
            if acc + sz <= need:
                chosen_priv.append(g)
                acc += sz
                if acc == need:
                    break
        if acc != need:
            continue
        new_pub, new_priv = set(pub_set), set(priv_set)
        for g in chosen_pub:
            for i in g_all[g]:
                new_pub.discard(i)
                new_priv.add(i)
        for g in chosen_priv:
            for i in g_all[g]:
                new_priv.discard(i)
                new_pub.add(i)
        if len(new_pub) == 81 and len(new_priv) == 81 and not (new_pub & new_priv):
            return sorted(new_pub), sorted(new_priv)
    return None


def local_sensitivity(cid, split, packs, id_group, rng):
    rows = []
    store = defaultdict(list)
    n_ok = 0
    attempts = 0
    while n_ok < N_PERTURB and attempts < N_PERTURB * 40:
        attempts += 1
        k = int(rng.choice([1, 2, 3]))
        out = perturb_once(split["public_ids"], split["private_ids"], id_group, k, rng)
        if out is None:
            continue
        pub_ids, priv_ids = out
        n_ok += 1
        for target, (models, y_by_id) in packs.items():
            cv_mae = np.array([m["cv_mae"] for m in models], float)
            mids = [m["model_id"] for m in models]
            pub_mae, pub_pear = scores_on_ids(models, y_by_id, pub_ids)
            priv_mae, priv_pear = scores_on_ids(models, y_by_id, priv_ids)
            mae_pp = rank_corr(-pub_mae, -priv_mae)
            pear_pp = rank_corr(pub_pear, priv_pear)
            wi = int(np.nanargmin(pub_mae))
            regret = float(priv_mae[wi] - np.nanmin(priv_mae))
            store[(target, "mae_pp")].append(mae_pp)
            store[(target, "pear_pp")].append(pear_pp)
            store[(target, "pub_winner_regret")].append(regret)
            store[(target, "k")].append(k)

    for target in ("TmApp", "HIC"):
        for metric in ("mae_pp", "pear_pp", "pub_winner_regret"):
            a = np.asarray(store[(target, metric)], float)
            a = a[np.isfinite(a)]
            rows.append(
                {
                    "candidate_id": cid,
                    "target": target,
                    "metric": metric,
                    "n_perturbations": int(len(a)),
                    "median": float(np.median(a)) if len(a) else float("nan"),
                    "iqr": float(np.quantile(a, 0.75) - np.quantile(a, 0.25)) if len(a) else float("nan"),
                    "p05": float(np.quantile(a, 0.05)) if len(a) else float("nan"),
                    "p_lt0": float(np.mean(a < 0)) if len(a) and metric != "pub_winner_regret" else (
                        float(np.mean(a < 0)) if len(a) else float("nan")
                    ),
                }
            )
    return rows, store


# ---------------------------------------------------------------------------
# Decision + reports
# ---------------------------------------------------------------------------
def decide(xfer_df, reg_df, boot_df, feat_bal_df, loc_df, strat_df):
    scores = {}
    for cid in CAND_IDS:
        tm = xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == "TmApp")].iloc[0]
        hic = xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == "HIC")].iloc[0]
        rtm = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "TmApp")].iloc[0]
        rhic = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "HIC")].iloc[0]
        btm = boot_df[
            (boot_df.candidate_id == cid)
            & (boot_df.target == "TmApp")
            & (boot_df.strategy == "A_public_only")
        ].iloc[0]
        bhic = boot_df[
            (boot_df.candidate_id == cid)
            & (boot_df.target == "HIC")
            & (boot_df.strategy == "A_public_only")
        ].iloc[0]
        bal = feat_bal_df[(feat_bal_df.candidate_id == cid) & (~feat_bal_df.diagnostic_only)]
        bal_score = float(bal["imbalance"].sum()) if len(bal) else float("nan")
        loc_tm = loc_df[
            (loc_df.candidate_id == cid) & (loc_df.target == "TmApp") & (loc_df.metric == "mae_pp")
        ].iloc[0]
        loc_hic = loc_df[
            (loc_df.candidate_id == cid) & (loc_df.target == "HIC") & (loc_df.metric == "mae_pp")
        ].iloc[0]

        rates = {
            "TmApp Public feedback quality": rate(tm.mae_public_private_spearman, 0.75, 0.65, 0.5),
            "TmApp Public-selection regret": rate(rtm.public_winner_regret, 0.05, 0.15, 0.4, higher_better=False),
            "TmApp bootstrap robustness": rate(btm.median_regret, 0.05, 0.15, 0.4, higher_better=False),
            "HIC Public feedback quality": rate(hic.mae_public_private_spearman, 0.75, 0.65, 0.5),
            "HIC Public-selection regret": rate(rhic.public_winner_regret, 0.02, 0.05, 0.1, higher_better=False),
            "HIC bootstrap robustness": rate(bhic.median_regret, 0.02, 0.05, 0.1, higher_better=False),
            "CV→Private consistency (TmApp MAE)": rate(tm.mae_cv_private_spearman, 0.7, 0.5, 0.2),
            "CV→Private consistency (HIC MAE)": rate(hic.mae_cv_private_spearman, 0.7, 0.5, 0.2),
            "pre-model statistical balance": rate(bal_score, 3.5, 4.5, 6.0, higher_better=False),
            "local mask stability (TmApp)": rate(float(loc_tm["median"]), 0.7, 0.5, 0.2),
            "local mask stability (HIC)": rate(float(loc_hic["median"]), 0.7, 0.5, 0.2),
        }
        # numeric priority score (lower better for regret/balance)
        prio = 0.0
        # Priority 1: TmApp PP must be positive / useful
        if tm.mae_public_private_spearman < 0.3:
            prio += 100
        prio -= 10 * float(tm.mae_public_private_spearman)
        # Priority 2: regret both targets
        prio += 25 * float(rtm.public_winner_regret)
        prio += 25 * float(rhic.public_winner_regret)
        prio += 15 * float(btm.median_regret)
        prio += 15 * float(bhic.median_regret)
        # Priority 3: HIC useful
        if hic.mae_public_private_spearman < 0.5:
            prio += 8  # concerning but not fatal
        prio -= 5 * float(hic.mae_public_private_spearman)
        # Priority 4: balance
        prio += 0.5 * bal_score
        # Priority 5: local stability
        prio -= 3 * float(loc_tm["median"])
        prio -= 2 * float(loc_hic["median"])

        scores[cid] = {
            "rates": rates,
            "prio": prio,
            "tm_pp": float(tm.mae_public_private_spearman),
            "hic_pp": float(hic.mae_public_private_spearman),
            "tm_regret": float(rtm.public_winner_regret),
            "hic_regret": float(rhic.public_winner_regret),
            "tm_boot_med": float(btm.median_regret),
            "hic_boot_med": float(bhic.median_regret),
            "bal_score": bal_score,
            "loc_tm_med": float(loc_tm["median"]),
            "loc_hic_med": float(loc_hic["median"]),
            "loc_tm_p_lt0": float(loc_tm["p_lt0"]),
            "btm_p05": float(btm.p95_regret),
            "bhic_p95": float(bhic.p95_regret),
        }

    ranked = sorted(scores.keys(), key=lambda c: scores[c]["prio"])
    best, second = ranked[0], ranked[1]
    # guard: if TmApp all similar and regrets nearly tied within tiny eps → human
    regrets = [scores[c]["tm_regret"] + scores[c]["hic_regret"] for c in CAND_IDS]
    if max(regrets) - min(regrets) < 1e-9 and abs(scores[best]["prio"] - scores[second]["prio"]) < 0.05:
        decision = "NO_CLEAR_WINNER_REQUIRE_HUMAN_DECISION"
    else:
        decision = f"SELECT_{best}"

    # balance / local winners
    bal_winner = min(CAND_IDS, key=lambda c: scores[c]["bal_score"])
    loc_winner = max(CAND_IDS, key=lambda c: scores[c]["loc_tm_med"] + scores[c]["loc_hic_med"])
    harm = [
        c
        for c in CAND_IDS
        if scores[c]["tm_regret"] > 0.5 or scores[c]["hic_regret"] > 0.15 or scores[c]["tm_pp"] < 0
    ]
    return decision, best, second, scores, bal_winner, loc_winner, harm


def write_decision_table(scores, path):
    rows = []
    for cid, s in scores.items():
        for dim, cat in s["rates"].items():
            rows.append({"candidate_id": cid, "dimension": dim, "rating": cat, "prio_score": s["prio"]})
        rows.append(
            {
                "candidate_id": cid,
                "dimension": "RAW_tm_public_winner_regret",
                "rating": fnum(s["tm_regret"]),
                "prio_score": s["prio"],
            }
        )
        rows.append(
            {
                "candidate_id": cid,
                "dimension": "RAW_hic_public_winner_regret",
                "rating": fnum(s["hic_regret"]),
                "prio_score": s["prio"],
            }
        )
        rows.append(
            {
                "candidate_id": cid,
                "dimension": "RAW_balance_score",
                "rating": fnum(s["bal_score"]),
                "prio_score": s["prio"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "```"


def deep_dive(cid, scores, xfer_df, reg_df, strat_df, inv_df, loc_df, feat_bal_df):
    s = scores[cid]
    tm = xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == "TmApp")].iloc[0]
    hic = xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == "HIC")].iloc[0]
    rtm = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "TmApp")].iloc[0]
    rhic = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "HIC")].iloc[0]
    st = strat_df[strat_df.candidate_id == cid]
    inv = inv_df[(inv_df.candidate_id == cid) & (inv_df.metric == "MAE")]
    loc = loc_df[loc_df.candidate_id == cid]
    lines = [f"# Deep dive — {cid}", ""]
    lines.append(f"Priority score (lower better): **{fnum(s['prio'], 3)}**")
    lines.append("")
    lines.append("## Transfer (recomputed)")
    lines.append(
        f"- TmApp MAE PP ρ={fnum(tm.mae_public_private_spearman)}; "
        f"CV→Pub={fnum(tm.mae_cv_public_spearman)}; CV→Priv={fnum(tm.mae_cv_private_spearman)}"
    )
    lines.append(
        f"- HIC MAE PP ρ={fnum(hic.mae_public_private_spearman)}; "
        f"CV→Pub={fnum(hic.mae_cv_public_spearman)}; CV→Priv={fnum(hic.mae_cv_private_spearman)}"
    )
    lines.append(
        f"- HIC Pearson PP ρ={fnum(hic.pear_public_private_spearman)}; "
        f"CV→Pub={fnum(hic.pear_cv_public_spearman)}"
    )
    lines.append("")
    lines.append("## Public-selection regret")
    lines.append(
        f"- TmApp: winner={rtm.public_mae_winner}, Private rank={fnum(rtm.public_winner_private_rank,1)}, "
        f"regret={fnum(rtm.public_winner_regret)} °C"
    )
    lines.append(
        f"- HIC: winner={rhic.public_mae_winner}, Private rank={fnum(rhic.public_winner_private_rank,1)}, "
        f"regret={fnum(rhic.public_winner_regret)}"
    )
    lines.append("")
    lines.append("## Strategies")
    lines.append(md_table(st[["target", "strategy", "chosen_model", "private_rank", "private_mae", "regret"]]))
    lines.append("")
    lines.append("## Pairwise MAE inversions")
    lines.append(md_table(inv))
    lines.append("")
    lines.append("## Local sensitivity")
    lines.append(md_table(loc))
    lines.append("")
    lines.append("## Ratings")
    for k, v in s["rates"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    # candidate-specific Qs
    lines.append("## Candidate-specific answers")
    if cid == "CAND_04974":
        inv_h = inv[inv.target == "HIC"].iloc[0]
        lines.append(
            f"1. Weak HIC CV→Private (ρ={fnum(hic.mae_cv_private_spearman)}) — "
            f"practical Public-winner regret is {fnum(rhic.public_winner_regret)}; "
            f"{'small/harmless' if rhic.public_winner_regret < 0.05 else 'material'}."
        )
        lines.append(
            f"2. Effect-size-weighted inversion burden (HIC MAE)={fnum(inv_h.effect_size_weighted_burden)} "
            f"over {int(inv_h.n_inversions)}/{int(inv_h.n_pairs)} inversions."
        )
        lines.append(f"3. Trusting HIC Public → regret={fnum(rhic.public_winner_regret)}.")
        lines.append(
            f"4. Public top2∩Private top2={int(hic.public_top2_intersect_private_top2)}; "
            f"top3∩top3={int(hic.public_top3_intersect_private_top3)}."
        )
        lm = loc[(loc.target == "TmApp") & (loc.metric == "mae_pp")].iloc[0]
        lines.append(f"5. Local TmApp MAE-PP median={fnum(lm['median'])}, P(ρ<0)={fnum(lm['p_lt0'])}.")
    elif cid == "CAND_12528":
        lines.append(
            f"1. Stronger HIC CV→Private (ρ={fnum(hic.mae_cv_private_spearman)}) vs Public-winner regret={fnum(rhic.public_winner_regret)}."
        )
        lines.append(
            f"2. Negative HIC Pearson CV→Public (ρ={fnum(hic.pear_cv_public_spearman)}) — "
            f"MAE CV→Public remains {fnum(hic.mae_cv_public_spearman)}."
        )
        lines.append("3. Primary participant metric is MAE; negative Pearson transfer is secondary.")
        lines.append(
            f"4. TmApp regret={fnum(rtm.public_winner_regret)}; HIC regret={fnum(rhic.public_winner_regret)}."
        )
        lm = loc[(loc.target == "TmApp") & (loc.metric == "mae_pp")].iloc[0]
        lines.append(f"5. Local TmApp MAE-PP median={fnum(lm['median'])}, P(ρ<0)={fnum(lm['p_lt0'])}.")
    else:
        lines.append(
            f"1. Compromise check: TmApp PP={fnum(tm.mae_public_private_spearman)}, "
            f"HIC PP={fnum(hic.mae_public_private_spearman)} (B6 floor fail if <0.50)."
        )
        lines.append(
            f"2. HIC PP≈{fnum(hic.mae_public_private_spearman)} with Public-winner regret={fnum(rhic.public_winner_regret)}."
        )
        lines.append(
            f"3. Regret vs rank: TmApp regret={fnum(rtm.public_winner_regret)}, HIC regret={fnum(rhic.public_winner_regret)}."
        )
        bal = float(feat_bal_df[(feat_bal_df.candidate_id == cid) & (~feat_bal_df.diagnostic_only)]["imbalance"].sum())
        lines.append(f"4. Pre-model imbalance sum={fnum(bal)}.")
        lm = loc[(loc.target == "TmApp") & (loc.metric == "mae_pp")].iloc[0]
        lines.append(f"5. Local TmApp MAE-PP median={fnum(lm['median'])}, P(ρ<0)={fnum(lm['p_lt0'])}.")
    (REP / f"candidate_{cid.split('_')[1]}_deep_dive.md").write_text("\n".join(lines) + "\n")


def write_reports(
    decision, best, second, scores, bal_winner, loc_winner, harm,
    xfer_df, reg_df, strat_df, boot_df, stab_df, inv_df, tgt_bal_df, feat_bal_df, loc_df,
    model_set, elapsed,
):
    # participant regret
    lines = ["# Participant selection regret", "", md_table(reg_df), "", "## Strategies", md_table(strat_df)]
    (REP / "participant_selection_regret.md").write_text("\n".join(lines) + "\n")

    lines = ["# Bootstrap regret analysis", "", md_table(boot_df), "", "## Winner stability", md_table(stab_df)]
    (REP / "bootstrap_regret_analysis.md").write_text("\n".join(lines) + "\n")

    # balance radar table
    feats = sorted(feat_bal_df[~feat_bal_df.diagnostic_only].feature.unique())
    radar = []
    for f in feats:
        row = {"feature": f}
        vals = {}
        for cid in ALL_SPLIT_IDS:
            sub = feat_bal_df[(feat_bal_df.candidate_id == cid) & (feat_bal_df.feature == f)]
            v = float(sub.iloc[0].imbalance) if len(sub) else float("nan")
            row[cid] = v
            if cid in CAND_IDS:
                vals[cid] = v
        row["best_candidate"] = min(vals, key=vals.get) if vals else None
        radar.append(row)
    radar_df = pd.DataFrame(radar)
    lines = [
        "# Balance deep dive",
        "",
        "## Target distributions",
        md_table(tgt_bal_df),
        "",
        "## Feature imbalance comparison",
        md_table(radar_df),
        "",
        "## Sequence-neighborhood (Train similarity)",
    ]
    for col in ("nearest_train_VH_identity", "nearest_train_VL_identity", "cluster_size"):
        sub = tgt_bal_df if False else feat_bal_df[feat_bal_df.feature == col]
        if len(sub):
            lines.append(f"### {col}")
            lines.append(md_table(sub[["candidate_id", "smd", "wasserstein_norm", "ks", "imbalance"]]))
    (REP / "balance_deep_dive.md").write_text("\n".join(lines) + "\n")

    (REP / "local_split_sensitivity.md").write_text(
        "# Local split sensitivity\n\n" + md_table(loc_df) + "\n"
    )

    # comparison
    cmp_rows = []
    for cid in ALL_SPLIT_IDS:
        for target in ("TmApp", "HIC"):
            tr = xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == target)].iloc[0]
            rg = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == target)].iloc[0]
            cmp_rows.append(
                {
                    "candidate_id": cid,
                    "target": target,
                    "mae_pp": tr.mae_public_private_spearman,
                    "mae_cv_pub": tr.mae_cv_public_spearman,
                    "mae_cv_priv": tr.mae_cv_private_spearman,
                    "pub_winner_regret": rg.public_winner_regret,
                }
            )
    (REP / "candidate_comparison.md").write_text(
        "# Candidate comparison\n\n" + md_table(pd.DataFrame(cmp_rows)) + "\n\n"
        + "## Decision ratings\n\n"
        + "\n".join(f"- {c}: prio={fnum(scores[c]['prio'],3)}" for c in CAND_IDS)
        + "\n"
    )

    for cid in CAND_IDS:
        deep_dive(cid, scores, xfer_df, reg_df, strat_df, inv_df, loc_df, feat_bal_df)

    # Final bakeoff report
    s_best = scores[best]
    s_sec = scores[second]
    bootA = boot_df[boot_df.strategy == "A_public_only"]

    def boot_note(target):
        bits = []
        for cid in CAND_IDS:
            r = bootA[(bootA.candidate_id == cid) & (bootA.target == target)].iloc[0]
            bits.append(f"{cid} med={fnum(r.median_regret)} p95={fnum(r.p95_regret)}")
        return "; ".join(bits)

    def get_reg(cid, target):
        return float(reg_df[(reg_df.candidate_id == cid) & (reg_df.target == target)].iloc[0].public_winner_regret)

    def get_pp(cid, target):
        return float(xfer_df[(xfer_df.candidate_id == cid) & (xfer_df.target == target)].iloc[0].mae_public_private_spearman)

    opening = f"""Recommended production split: {decision}

Second choice: {second}

Reason: Priority-ordered bake-off favors {best} (prio={fnum(s_best['prio'],3)}) over {second} (prio={fnum(s_sec['prio'],3)}) on Public-selection regret, TmApp transfer, HIC usefulness, balance, and local stability.

TmApp:
    Public-winner Private regret:
        04974: {fnum(get_reg('CAND_04974','TmApp'))}
        12528: {fnum(get_reg('CAND_12528','TmApp'))}
        12207: {fnum(get_reg('CAND_12207','TmApp'))}

    Public→Private MAE rank transfer:
        04974: {fnum(get_pp('CAND_04974','TmApp'))}
        12528: {fnum(get_pp('CAND_12528','TmApp'))}
        12207: {fnum(get_pp('CAND_12207','TmApp'))}

    bootstrap robustness: {boot_note('TmApp')}

HIC:
    Public-winner Private regret:
        04974: {fnum(get_reg('CAND_04974','HIC'))}
        12528: {fnum(get_reg('CAND_12528','HIC'))}
        12207: {fnum(get_reg('CAND_12207','HIC'))}

    Public→Private MAE rank transfer:
        04974: {fnum(get_pp('CAND_04974','HIC'))}
        12528: {fnum(get_pp('CAND_12528','HIC'))}
        12207: {fnum(get_pp('CAND_12207','HIC'))}

    bootstrap robustness: {boot_note('HIC')}

Pre-model balance winner: {bal_winner}

Local-stability winner: {loc_winner}

Any candidate with unacceptable participant harm: {(', '.join(harm) if harm else 'NONE')}

Permanent freeze recommended: {'YES' if decision.startswith('SELECT_') else 'NO — human decision required'}
"""
    # Q1–20
    q = ["", "# Answers to required questions", ""]
    for cid in CAND_IDS:
        rtm = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "TmApp")].iloc[0]
        rhic = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == "HIC")].iloc[0]
        q.append(f"### {cid}")
        q.append(f"1/2/3 TmApp Public winner={rtm.public_mae_winner}, Private rank={fnum(rtm.public_winner_private_rank,1)}, regret={fnum(rtm.public_winner_regret)} °C")
        q.append(f"   HIC Public winner={rhic.public_mae_winner}, Private rank={fnum(rhic.public_winner_private_rank,1)}, regret={fnum(rhic.public_winner_regret)}")
    # Q4 CV-only
    q.append("")
    q.append("4. CV-only participant:")
    for cid in CAND_IDS:
        for target in ("TmApp", "HIC"):
            r = reg_df[(reg_df.candidate_id == cid) & (reg_df.target == target)].iloc[0]
            q.append(f"   {cid}/{target}: {r.cv_winner}, PubRank={fnum(r.cv_winner_public_rank,1)}, PrivRank={fnum(r.cv_winner_private_rank,1)}, regret={fnum(r.cv_winner_regret)}")
    q.append("")
    q.append("5. CV-among-Public-Top2:")
    for cid in CAND_IDS:
        for target in ("TmApp", "HIC"):
            r = strat_df[
                (strat_df.candidate_id == cid)
                & (strat_df.target == target)
                & (strat_df.strategy == "C_cv_among_public_top2")
            ].iloc[0]
            q.append(f"   {cid}/{target}: {r.chosen_model}, PrivRank={fnum(r.private_rank,1)}, regret={fnum(r.regret)}")

    tm_best_reg = min(CAND_IDS, key=lambda c: get_reg(c, "TmApp"))
    hic_best_reg = min(CAND_IDS, key=lambda c: get_reg(c, "HIC"))
    tm_boot = min(
        CAND_IDS,
        key=lambda c: float(
            bootA[(bootA.candidate_id == c) & (bootA.target == "TmApp")].iloc[0].median_regret
        ),
    )
    hic_boot = min(
        CAND_IDS,
        key=lambda c: float(
            bootA[(bootA.candidate_id == c) & (bootA.target == "HIC")].iloc[0].median_regret
        ),
    )
    hic049 = xfer_df[(xfer_df.candidate_id == "CAND_04974") & (xfer_df.target == "HIC")].iloc[0]
    inv049 = inv_df[(inv_df.candidate_id == "CAND_04974") & (inv_df.target == "HIC") & (inv_df.metric == "MAE")].iloc[0]
    hic125 = xfer_df[(xfer_df.candidate_id == "CAND_12528") & (xfer_df.target == "HIC")].iloc[0]

    # neighborhood easiness
    neigh = []
    for cid in CAND_IDS:
        for col in ("nearest_train_VH_identity", "nearest_train_VL_identity"):
            sub = feat_bal_df[(feat_bal_df.candidate_id == cid) & (feat_bal_df.feature == col)]
            if len(sub):
                neigh.append((cid, col, float(sub.iloc[0].smd)))
    max_neigh = max(neigh, key=lambda x: abs(x[2])) if neigh else ("NONE", "", 0.0)

    # model dependence via public winner concentration in bootstrap
    dep_notes = []
    for cid in CAND_IDS:
        for target in ("TmApp", "HIC"):
            sub = stab_df[(stab_df.candidate_id == cid) & (stab_df.target == target) & (stab_df.model_id != "__SUMMARY__")]
            if len(sub):
                mx = sub.P_public_winner.max()
                mid = sub.loc[sub.P_public_winner.idxmax(), "model_id"]
                dep_notes.append(f"{cid}/{target}: max P(Pub winner)={fnum(mx)} ({mid})")

    novice = min(CAND_IDS, key=lambda c: get_reg(c, "TmApp") + get_reg(c, "HIC"))
    # experienced: min CV regret sum
    def cv_reg_sum(c):
        return sum(
            float(reg_df[(reg_df.candidate_id == c) & (reg_df.target == t)].iloc[0].cv_winner_regret)
            for t in ("TmApp", "HIC")
        )

    experienced = min(CAND_IDS, key=cv_reg_sum)

    q += [
        "",
        f"6. Minimizes TmApp Public-selection regret: **{tm_best_reg}**",
        f"7. Minimizes HIC Public-selection regret: **{hic_best_reg}**",
        f"8. 04974 weak HIC CV→Private (ρ={fnum(hic049.mae_cv_private_spearman)}): "
        f"inversion burden={fnum(inv049.effect_size_weighted_burden)}; "
        f"Public-winner regret={fnum(get_reg('CAND_04974','HIC'))} → "
        f"{'small harmless swaps' if get_reg('CAND_04974','HIC') < 0.05 else 'consequential'}.",
        f"9. 12528 HIC Pearson CV→Public={fnum(hic125.pear_cv_public_spearman)}; "
        f"MAE CV→Public={fnum(hic125.mae_cv_public_spearman)} — "
        f"{'limited practical harm under MAE-primary selection' if hic125.mae_cv_public_spearman > 0 else 'MAE also weak'}.",
        f"10. 12207 as compromise: HIC PP={fnum(get_pp('CAND_12207','HIC'))} "
        f"(CONCERNING vs B6 floor 0.50); regrets TmApp={fnum(get_reg('CAND_12207','TmApp'))}, "
        f"HIC={fnum(get_reg('CAND_12207','HIC'))}; balance={fnum(scores['CAND_12207']['bal_score'])}. "
        f"{'Effect sizes do not clearly redeem the HIC floor miss' if get_pp('CAND_12207','HIC') < 0.5 else 'Still competitive'}.",
        f"11. Best TmApp bootstrap (Strategy A median regret): **{tm_boot}**",
        f"12. Best HIC bootstrap (Strategy A median regret): **{hic_boot}**",
        f"13. Best pre-model balance: **{bal_winner}**",
        f"14. Most locally stable: **{loc_winner}**",
        f"15. Sequence-neighborhood easiness: largest |SMD| = {max_neigh[0]} / {max_neigh[1]} = {fnum(max_neigh[2])} "
        f"({'notable' if abs(max_neigh[2]) > 0.3 else 'no strong easiness imbalance'}).",
        f"16. Single-model dependence (bootstrap Public-winner concentration): " + "; ".join(dep_notes[:6]),
        f"17. Least misleading for Kaggle-inexperienced (min Public-winner regret sum): **{novice}**",
        f"18. Preferred by experienced CV-driven participant (min CV-winner regret sum): **{experienced}**",
        f"19. Novice vs experienced answers differ? **{'YES' if novice != experienced else 'NO'}**",
        f"20. Permanent freeze recommendation: **{decision}**",
        "",
        "## Model set",
        f"- TmApp models ({len(model_set.get('TmApp', []))}): "
        + ", ".join(m["model_id"] for m in model_set.get("TmApp", [])),
        f"- HIC models ({len(model_set.get('HIC', []))}): "
        + ", ".join(m["model_id"] for m in model_set.get("HIC", [])),
        "",
        "## Pairwise Test prediction correlations",
        "```json",
        json.dumps(model_set.get("prediction_correlations_Test", {}), indent=2),
        "```",
        "",
        f"_Elapsed: {elapsed:.1f}s_",
    ]

    (REP / "GATE_B6_1_FINAL_SPLIT_BAKEOFF.md").write_text(opening + "\n".join(q) + "\n")
    return REP / "GATE_B6_1_FINAL_SPLIT_BAKEOFF.md"


def make_plots(feat, splits, regret_hist):
    try:
        # ECDFs for targets
        for target in ("TmApp", "HIC"):
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
            for ax, cid in zip(axes, ALL_SPLIT_IDS):
                pub = feat[feat.id.isin(splits[cid]["public_ids"])][target].dropna().sort_values()
                priv = feat[feat.id.isin(splits[cid]["private_ids"])][target].dropna().sort_values()
                ax.plot(pub, np.linspace(0, 1, len(pub)), label="Public")
                ax.plot(priv, np.linspace(0, 1, len(priv)), label="Private")
                ax.set_title(cid.replace("CAND_", "").replace("CURRENT_BASELINE_SPLIT", "BASE"))
                ax.set_xlabel(target)
            axes[0].set_ylabel("ECDF")
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS / f"ecdf_{target}.png", dpi=100)
            plt.close(fig)
        # regret histograms
        for (cid, target), a in regret_hist.items():
            if cid == BASELINE_ID:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(a, bins=40, color="#4a6fa5", alpha=0.85)
            ax.set_title(f"{cid} {target} Strategy-A regret")
            ax.set_xlabel("Private regret")
            fig.tight_layout()
            fig.savefig(PLOTS / f"regret_hist_{cid}_{target}.png", dpi=100)
            plt.close(fig)
        print(f"Wrote plots under {PLOTS}")
    except Exception as e:
        print(f"Plotting skipped/partial: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    ensure_dirs()
    print("=== Gate B6.1 Final Production Split Bake-Off ===")
    # self-check syntax of this file
    src = Path(__file__).read_text()
    ast.parse(src)
    print("ast.parse: OK")

    role, pub_order, priv_order, train_ids, lab, feat, model_set, splits, id_group = load_base()
    print(f"Loaded {len(splits)} splits; feat={feat.shape}; models TmApp={len(model_set.get('TmApp',[]))} HIC={len(model_set.get('HIC',[]))}")

    # Freeze configs FIRST
    freeze_configs(splits, model_set)

    packs = {
        "TmApp": prepare_target_pack("TmApp", model_set, pub_order, priv_order, train_ids, role, lab),
        "HIC": prepare_target_pack("HIC", model_set, pub_order, priv_order, train_ids, role, lab),
    }
    # ensure all candidate ids have labels/preds
    all_test = set(lab.index.astype(str))
    for cid, sp in splits.items():
        miss = (set(sp["public_ids"]) | set(sp["private_ids"])) - all_test
        assert not miss, (cid, miss)

    pm, tr, rg, st, inv = [], [], [], [], []
    details = {}
    print("Scoring splits…")
    for cid in ALL_SPLIT_IDS:
        print(f"  score {cid}")
        a, b, c, d, e, det = score_split(cid, splits[cid], packs)
        pm += a
        tr += b
        rg += c
        st += d
        inv += e
        details[cid] = det

    pm_df = pd.DataFrame(pm)
    tr_df = pd.DataFrame(tr)
    rg_df = pd.DataFrame(rg)
    st_df = pd.DataFrame(st)
    inv_df = pd.DataFrame(inv)
    pm_df.to_csv(MET / "per_model_candidate_scores.csv", index=False)
    tr_df.to_csv(MET / "model_rank_transfer.csv", index=False)
    rg_df.to_csv(MET / "public_selection_regret.csv", index=False)
    st_df.to_csv(MET / "participant_strategy_regret.csv", index=False)
    inv_df.to_csv(MET / "pairwise_inversions.csv", index=False)
    print("Wrote core metrics CSVs")

    print(f"Bootstrap ({N_BOOT} × {len(ALL_SPLIT_IDS)} splits)…")
    rng_boot = np.random.default_rng(SEED)
    boot_rows, stab_rows = [], []
    regret_hist = {}
    for cid in ALL_SPLIT_IDS:
        print(f"  bootstrap {cid}")
        br, sr, rh = run_bootstrap(cid, splits[cid], packs, rng_boot)
        boot_rows += br
        stab_rows += sr
        regret_hist.update(rh)
    boot_df = pd.DataFrame(boot_rows)
    stab_df = pd.DataFrame(stab_rows)
    boot_df.to_csv(MET / "bootstrap_regret.csv", index=False)
    stab_df.to_csv(MET / "bootstrap_winner_stability.csv", index=False)

    print("Balance…")
    tgt_rows, feat_rows = [], []
    for cid in ALL_SPLIT_IDS:
        tgt_rows += balance_target(cid, splits[cid], feat)
        feat_rows += balance_features(cid, splits[cid], feat)
    tgt_bal_df = pd.DataFrame(tgt_rows)
    feat_bal_df = pd.DataFrame(feat_rows)
    tgt_bal_df.to_csv(MET / "target_balance.csv", index=False)
    feat_bal_df.to_csv(MET / "feature_balance.csv", index=False)

    print(f"Local sensitivity ({N_PERTURB}/candidate)…")
    rng_pert = np.random.default_rng(SEED + 1)
    loc_rows = []
    for cid in CAND_IDS:
        print(f"  perturb {cid}")
        rows, _ = local_sensitivity(cid, splits[cid], packs, id_group, rng_pert)
        loc_rows += rows
    loc_df = pd.DataFrame(loc_rows)
    loc_df.to_csv(MET / "local_mask_sensitivity.csv", index=False)

    print("Decision…")
    decision, best, second, scores, bal_winner, loc_winner, harm = decide(
        tr_df, rg_df, boot_df, feat_bal_df, loc_df, st_df
    )
    write_decision_table(scores, MET / "final_decision_table.csv")

    make_plots(feat, splits, regret_hist)

    elapsed = time.time() - t0
    out = write_reports(
        decision, best, second, scores, bal_winner, loc_winner, harm,
        tr_df, rg_df, st_df, boot_df, stab_df, inv_df, tgt_bal_df, feat_bal_df, loc_df,
        model_set, elapsed,
    )
    print("=" * 60)
    print(f"RECOMMENDATION: {decision}")
    print(f"Preferred: {best} | Second: {second}")
    print(f"Report: {out}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
