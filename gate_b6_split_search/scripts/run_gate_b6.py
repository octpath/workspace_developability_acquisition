#!/usr/bin/env python3
"""Gate B6 — Common Public/Private production split search (self-contained)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import kendalltau, spearmanr, wasserstein_distance

ROOT = Path("/workspace_developability_acquisition")
B6 = ROOT / "gate_b6_split_search"
CFG = B6 / "config"
CAND = B6 / "candidates"
MET = B6 / "metrics"
REP = B6 / "reports"
FRZ = B6 / "frozen"
CACHE = B6 / "cache"
PRED = ROOT / "gate_b5_ceiling/predictions"
ORG = ROOT / "gate_b3/frozen/organizer"
OUTER_PATH = ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json"
FEAT_C = ROOT / "gate_b1/cache/features/stage_C_shortcut.csv"
NUM_PATH = ROOT / "gate_b1/data/numbering_germline.csv"
SEED = 20260830

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


def ensure_dirs() -> None:
    for d in (CFG, CAND, MET, REP, FRZ, CACHE, B6 / "plots", B6 / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def write_json(path: Path, obj) -> str:
    b = json.dumps(obj, indent=2, sort_keys=True, default=float).encode()
    path.write_bytes(b)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_bytes(b) + "\n")
    return sha256_bytes(b)


def assert_key_paths() -> None:
    print("=== Gate B6 path check ===")
    print(f"ROOT={ROOT}")
    print(f"ORG={ORG}")
    print(f"PRED={PRED}")
    print(f"OUTER={OUTER_PATH}")
    print(f"FEAT_C={FEAT_C}")
    print(f"NUM={NUM_PATH}")
    print(f"B6={B6}")
    required = [
        ORG / "role_map.csv",
        ORG / "final_population.csv",
        ORG / "public_ids.csv",
        ORG / "private_ids.csv",
        ORG / "train_ids.csv",
        ORG / "test_labels_hidden.csv",
        OUTER_PATH,
        FEAT_C,
    ]
    for tag, _ in TMAPP_TAGS + HIC_TAGS:
        target = "TmApp" if (tag, _) in TMAPP_TAGS else "HIC"
        # resolve target properly
    for target, tags in (("TmApp", TMAPP_TAGS), ("HIC", HIC_TAGS)):
        for tag, _fam in tags:
            required += [
                PRED / "final" / f"{target}__{tag}__public.npy",
                PRED / "final" / f"{target}__{tag}__private.npy",
                PRED / "oof" / f"{target}__{tag}__meanOOF.npy",
            ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required files:\n" + "\n".join(missing)
    print(f"OK: {len(required)} key files present")


def load_test_pred(target: str, tag: str) -> np.ndarray:
    return np.concatenate(
        [
            np.load(PRED / "final" / f"{target}__{tag}__public.npy"),
            np.load(PRED / "final" / f"{target}__{tag}__private.npy"),
        ]
    )


def cv_stats(target: str, tag: str, y_train: np.ndarray) -> dict:
    path = PRED / "oof" / f"{target}__{tag}__meanOOF.npy"
    oof = np.load(path)
    assert len(oof) == len(y_train), (target, tag, len(oof), len(y_train))
    mae = float(np.mean(np.abs(y_train - oof)))
    if np.std(oof) < 1e-12 or np.std(y_train) < 1e-12:
        pear = spr = float("nan")
    else:
        pear = float(np.corrcoef(y_train, oof)[0, 1])
        spr = float(spearmanr(y_train, oof).correlation)
    return {"mae": mae, "pearson": pear, "spearman": spr, "oof_sha256": sha256_file(path)}


def pred_corrs(target: str, tags: list[str]) -> dict:
    mats = {t: load_test_pred(target, t) for t in tags}
    out = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            sa, sb = mats[a], mats[b]
            out[f"{a}__vs__{b}"] = (
                None
                if (np.std(sa) < 1e-12 or np.std(sb) < 1e-12)
                else float(np.corrcoef(sa, sb)[0, 1])
            )
    return out


def enrich_models(specs, target, y_train):
    rows = []
    for m in specs:
        tag = m["tag"]
        pub = PRED / "final" / f"{target}__{tag}__public.npy"
        priv = PRED / "final" / f"{target}__{tag}__private.npy"
        assert pub.exists() and priv.exists(), tag
        cv = cv_stats(target, tag, y_train)
        rows.append(
            {
                **m,
                "target": target,
                "prediction_file_hash": {
                    "public": sha256_file(pub),
                    "private": sha256_file(priv),
                    "oof": cv["oof_sha256"],
                },
                "CV_metrics": {"mae": cv["mae"], "pearson": cv["pearson"], "spearman": cv["spearman"]},
            }
        )
    return rows


def freeze_p0(role_ix: pd.DataFrame, train_ids: list[str]):
    y_tm = role_ix.loc[train_ids, "TmApp"].values.astype(float)
    y_hic = role_ix.loc[train_ids, "HIC"].values.astype(float)
    tm_specs = [
        {"model_id": f"TmApp__{tag}", "tag": tag, "model_family": fam} for tag, fam in TMAPP_TAGS
    ]
    hic_specs = [
        {"model_id": f"HIC__{tag}", "tag": tag, "model_family": fam} for tag, fam in HIC_TAGS
    ]
    model_set = {
        "gate": "B6",
        "purpose": "Frozen organizer models for common Public/Private split evaluation",
        "source": "gate_b5_ceiling/predictions",
        "TmApp": enrich_models(tm_specs, "TmApp", y_tm),
        "HIC": enrich_models(hic_specs, "HIC", y_hic),
        "excluded": [
            {
                "model_id": "TmApp__FUSION_ABLANG2_BIO_ElasticNet",
                "reason": "Near-duplicate of NESTED_STACK_MEAN (Test-pred Pearson~0.979)",
            },
            {"model_id": "POOL/FT variants", "reason": "B5 not valid / unstable"},
        ],
        "prediction_correlations_Test": {
            "TmApp": pred_corrs("TmApp", [t for t, _ in TMAPP_TAGS]),
            "HIC": pred_corrs("HIC", [t for t, _ in HIC_TAGS]),
        },
        "diversity_rule": "Near-duplicate Test predictions excluded from primary eval set",
    }
    model_hash = write_json(CFG / "FROZEN_SPLIT_EVALUATION_MODEL_SET.json", model_set)
    # Required exact name; keep alias for older drafts if present
    alias = CFG / "FROZEN_SPLIT_EVAL_MODEL_SET.json"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to("FROZEN_SPLIT_EVALUATION_MODEL_SET.json")

    protocol = {
        "gate": "B6",
        "title": "Common Public/Private production split search — pre-registered protocol",
        "frozen_before_model_scoring": True,
        "candidate_generation": {
            "seed": SEED,
            "n_candidates": int(globals().get("_OVERRIDE_N_CAND", 20000)),
            "public_size": 81,
            "private_size": 81,
            "atomic_sequence_groups": True,
            "method": "Random multi-group assignment + exact singleton fill to Public=81; dedupe by Public hash",
            "include_current_baseline": True,
            "baseline_id": "CURRENT_BASELINE_SPLIT",
        },
        "pre_model_balance": {
            "top_fraction": 0.10,
            "max_survivors": 2000,
            "filter_rule": "Keep top 10% by ascending balance_score; always retain CURRENT_BASELINE_SPLIT",
        },
        "TmApp_evaluation": {
            "model_set_sha256": model_hash,
            "require_positive_mae_transfers": True,
            "shortlist_size": 8,
            "shortlist_jaccard_cap": 0.85,
            "bootstrap_screen_replicates": int(globals().get("_OVERRIDE_BOOT_SCREEN", 1000)),
            "bootstrap_screen_top_n": 80,
            "bootstrap_final_replicates": int(globals().get("_OVERRIDE_BOOT_FINAL", 5000)),
            "selection_hierarchy": [
                "pre_model_balance_filter",
                "positive_MAE_CV_Public_and_Public_Private_and_CV_Private",
                "prefer_higher_min_then_mean_MAE_triple",
                "prefer_Pearson_transfers",
                "prefer_worst_LOFO_family_and_bootstrap_stability",
                "Top3_overlap_tiebreak",
            ],
        },
        "HIC_safety": {
            "evaluate_only_on": "frozen TmApp shortlist",
            "floors": {
                "mae_model_rank_Public_to_Private_spearman": 0.50,
                "pearson_model_rank_Public_to_Private_spearman": 0.50,
                "mae_model_rank_CV_to_Private_spearman": 0.0,
            },
            "classification": {
                "HIC_EXCELLENT": ">=0.75",
                "HIC_GOOD": "[0.65,0.75)",
                "HIC_ACCEPTABLE": "[0.50,0.65)",
                "HIC_CONCERNING": "<0.50 or fails floors",
            },
            "role": "safety constraint / tiebreak only",
        },
        "final_decision_rule": [
            "Discard shortlist failing HIC floors",
            "Among remaining choose best robust TmApp",
            "HIC tiebreak only if nearly equivalent",
            "If none: NO_ACCEPTABLE_COMMON_SPLIT_FOUND",
            "Do not regenerate after HIC",
        ],
        "anti_overfit": {"lofo_selection_replay": True, "label_if_collapses": "MODEL_SET_OVERFIT_RISK"},
        "model_set_sha256": model_hash,
        "common_mask_requirement": "HIC and TmApp share identical Public/Private IDs",
        "train_test_boundary": "FROZEN Train=162 / Test=162",
    }
    proto_hash = write_json(CFG / "SPLIT_SELECTION_PROTOCOL.json", protocol)
    print("P0 frozen", model_hash[:16], proto_hash[:16])
    return model_set, protocol, model_hash, proto_hash


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


def quant_mae(a, b, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    return float(np.mean([abs(np.quantile(a, q) - np.quantile(b, q)) for q in qs]))


def _best_identity(seq: str, refs: list[str]) -> float:
    best = 0.0
    ls = len(seq)
    for t in refs:
        n = min(ls, len(t))
        if n == 0:
            continue
        m = sum(1 for a, b in zip(seq[:n], t[:n]) if a == b)
        best = max(best, m / max(ls, len(t)))
    return best


def build_features(role: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    test = role[role.role.isin(["Public", "Private"])].copy()
    df = test[["id", "role", "sequence_group", "HIC", "TmApp"]].merge(
        pop.drop(columns=["HIC", "TmApp", "sequence_group"], errors="ignore"), on="id", how="left"
    )
    gsize = pop.groupby("sequence_group").size().to_dict()
    df["cluster_size"] = df["sequence_group"].map(gsize).astype(float)

    feat_c = pd.read_csv(FEAT_C)
    df = df.merge(feat_c, left_on="id", right_on="antibody_id", how="left")
    if "antibody_id" in df.columns:
        df = df.drop(columns=["antibody_id"])

    if NUM_PATH.exists():
        num = pd.read_csv(NUM_PATH).rename(columns={"antibody_id": "id"})
        keep = [c for c in ["id", "H_CDR3_len", "L_CDR3_len"] if c in num.columns]
        df = df.merge(num[keep], on="id", how="left")

    charges, pis, gravys = [], [], []
    for _, r in df.iterrows():
        seq = (str(r.heavy) + str(r.light)).replace("X", "A").replace("*", "")
        try:
            pa = ProteinAnalysis(seq)
            charges.append(float(pa.charge_at_pH(7.0)))
            pis.append(float(pa.isoelectric_point()))
            gravys.append(float(pa.gravy()))
        except Exception:
            charges.append(np.nan)
            pis.append(np.nan)
            gravys.append(np.nan)
    df["A2_HL_charge_ph7"] = charges
    df["A2_HL_pI"] = pis
    df["A2_HL_gravy"] = gravys

    train = role[role.role == "Train"].merge(pop[["id", "heavy", "light"]], on="id", how="left")
    tr_h, tr_l = train.heavy.tolist(), train.light.tolist()
    df["nearest_train_VH_identity"] = [_best_identity(h, tr_h) for h in df.heavy]
    df["nearest_train_VL_identity"] = [_best_identity(l, tr_l) for l in df.light]
    return df


def generate_candidates(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    groups = {int(g): sub["id"].tolist() for g, sub in df.groupby("sequence_group")}
    multi = [(g, ids) for g, ids in groups.items() if len(ids) > 1]
    singles = [ids[0] for g, ids in groups.items() if len(ids) == 1]
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[dict] = []

    base_pub = sorted(df.loc[df.role == "Public", "id"].tolist())
    base_priv = sorted(df.loc[df.role == "Private", "id"].tolist())
    bh = sha256_ids(base_pub)
    seen.add(bh)
    out.append(
        {
            "candidate_id": "CURRENT_BASELINE_SPLIT",
            "public_ids": base_pub,
            "private_ids": base_priv,
            "public_hash": bh,
            "private_hash": sha256_ids(base_priv),
            "is_baseline": True,
        }
    )

    attempts = 0
    max_attempts = n * 200
    while len(out) < n + 1 and attempts < max_attempts:
        attempts += 1
        pub_ids: list[str] = []
        priv_ids: list[str] = []
        for _g, ids in multi:
            if rng.random() < 0.5:
                pub_ids.extend(ids)
            else:
                priv_ids.extend(ids)
        need_pub = 81 - len(pub_ids)
        if need_pub < 0 or need_pub > len(singles):
            continue
        perm = rng.permutation(len(singles))
        chosen = [singles[i] for i in perm[:need_pub]]
        rest = [singles[i] for i in perm[need_pub:]]
        pub_ids = sorted(pub_ids + chosen)
        priv_ids = sorted(priv_ids + rest)
        if len(pub_ids) != 81 or len(priv_ids) != 81:
            continue
        if set(pub_ids) & set(priv_ids):
            continue
        h = sha256_ids(pub_ids)
        if h in seen:
            continue
        seen.add(h)
        out.append(
            {
                "candidate_id": f"CAND_{len(out):05d}",
                "public_ids": pub_ids,
                "private_ids": priv_ids,
                "public_hash": h,
                "private_hash": sha256_ids(priv_ids),
                "is_baseline": False,
            }
        )
    return out


def balance_one(df, pub_ids, priv_ids, train_tm_q90, train_hic_q90) -> dict:
    pub = df[df.id.isin(pub_ids)]
    priv = df[df.id.isin(priv_ids)]
    total = 0.0
    terms: dict = {}
    cont_w = {
        "TmApp": 1.5,
        "HIC": 1.5,
        "cluster_size": 1.25,
        "A2_HL_charge_ph7": 1.0,
        "A2_HL_pI": 1.0,
        "A2_HL_gravy": 0.75,
        "C_vh_germline_distance": 0.25,
        "C_vl_germline_distance": 0.25,
        "C_combined_germline_distance": 0.25,
        "vh_len": 0.125,
        "vl_len": 0.125,
        "H_CDR3_len": 0.125,
        "L_CDR3_len": 0.125,
        "nearest_train_VH_identity": 0.375,
        "nearest_train_VL_identity": 0.375,
    }
    for col, wt in cont_w.items():
        if col not in pub.columns:
            continue
        a = pub[col].values.astype(float)
        b = priv[col].values.astype(float)
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(aa) < 5 or len(bb) < 5:
            continue
        pooled = float(np.std(np.concatenate([aa, bb])) + 1e-8)
        w_norm = float(wasserstein_distance(aa, bb)) / pooled
        s = abs(smd(aa, bb))
        q = quant_mae(aa, bb) / pooled
        sd_pen = abs(np.log((np.std(aa) + 1e-8) / (np.std(bb) + 1e-8)))
        if col == "TmApp":
            tail = abs(float(np.mean(aa >= train_tm_q90)) - float(np.mean(bb >= train_tm_q90)))
        elif col == "HIC":
            tail = abs(float(np.mean(aa >= train_hic_q90)) - float(np.mean(bb >= train_hic_q90)))
        else:
            tail = 0.0
        score = w_norm + s + q + 0.5 * sd_pen + tail
        terms[f"cont::{col}"] = score
        total += wt * score

    def cat_term(col, wt):
        if col not in pub.columns:
            return 0.0
        levels = sorted(set(pub[col].astype(str)) | set(priv[col].astype(str)))
        pa = np.array([(pub[col].astype(str) == lv).mean() for lv in levels])
        pb = np.array([(priv[col].astype(str) == lv).mean() for lv in levels])
        j = js_div(pa, pb)
        terms[f"cat::{col}"] = j
        return wt * j

    total += cat_term("vh_family", 0.75)
    total += cat_term("vl_family", 1.0)
    total += cat_term("C_kappa_lambda", 0.5)
    total += cat_term("donor", 0.15)
    total += cat_term("b_cell_subset", 0.15)
    terms["balance_score"] = float(total)
    return terms


def rank_corr(a, b, method="spearman") -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if method == "spearman":
        r = spearmanr(a[mask], b[mask]).correlation
    else:
        r = kendalltau(a[mask], b[mask]).correlation
    return float(r) if r is not None and np.isfinite(r) else float("nan")


def subset_metrics(models, y_by_id, ids):
    y = np.array([y_by_id[i] for i in ids], float)
    mae, pear, spr = [], [], []
    for m in models:
        p = np.array([m["pred_by_id"][i] for i in ids], float)
        mae.append(float(np.mean(np.abs(y - p))))
        if np.std(p) < 1e-12 or np.std(y) < 1e-12:
            pear.append(np.nan)
            spr.append(np.nan)
        else:
            pear.append(float(np.corrcoef(y, p)[0, 1]))
            spr.append(float(spearmanr(y, p).correlation))
    return np.array(mae), np.array(pear), np.array(spr)


def topk_overlap(scores_a, scores_b, k, higher_better=False):
    a = np.asarray(scores_a, float)
    b = np.asarray(scores_b, float)
    if higher_better:
        ia, ib = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    else:
        ia, ib = set(np.argsort(a)[:k]), set(np.argsort(b)[:k])
    return len(ia & ib) / float(k)


def winner_rank(src, dst, higher_better=False):
    a = np.asarray(src, float)
    b = np.asarray(dst, float)
    if higher_better:
        w = int(np.nanargmax(a))
        order = np.argsort(-b)
    else:
        w = int(np.nanargmin(a))
        order = np.argsort(b)
    return int(np.where(order == w)[0][0]) + 1


def eval_split(models, y_by_id, pub_ids, priv_ids) -> dict:
    mae_p, pear_p, spr_p = subset_metrics(models, y_by_id, pub_ids)
    mae_r, pear_r, spr_r = subset_metrics(models, y_by_id, priv_ids)
    cv_mae = np.array([m["cv_mae"] for m in models], float)
    cv_pear = np.array([m["cv_pearson"] for m in models], float)
    cv_spr = np.array([m["cv_spearman"] for m in models], float)
    # MAE model-rank: higher-better via -MAE
    out = {
        "mae_cv_public_spearman": rank_corr(-cv_mae, -mae_p),
        "mae_public_private_spearman": rank_corr(-mae_p, -mae_r),
        "mae_cv_private_spearman": rank_corr(-cv_mae, -mae_r),
        "mae_cv_public_kendall": rank_corr(-cv_mae, -mae_p, "kendall"),
        "mae_public_private_kendall": rank_corr(-mae_p, -mae_r, "kendall"),
        "mae_cv_private_kendall": rank_corr(-cv_mae, -mae_r, "kendall"),
        "pear_cv_public_spearman": rank_corr(cv_pear, pear_p),
        "pear_public_private_spearman": rank_corr(pear_p, pear_r),
        "pear_cv_private_spearman": rank_corr(cv_pear, pear_r),
        "spr_cv_public_spearman": rank_corr(cv_spr, spr_p),
        "spr_public_private_spearman": rank_corr(spr_p, spr_r),
        "spr_cv_private_spearman": rank_corr(cv_spr, spr_r),
        "top2_cv_public": topk_overlap(cv_mae, mae_p, 2),
        "top2_public_private": topk_overlap(mae_p, mae_r, 2),
        "top2_cv_private": topk_overlap(cv_mae, mae_r, 2),
        "top3_cv_public": topk_overlap(cv_mae, mae_p, 3),
        "top3_public_private": topk_overlap(mae_p, mae_r, 3),
        "top3_cv_private": topk_overlap(cv_mae, mae_r, 3),
        "cv_winner_public_rank": winner_rank(cv_mae, mae_p),
        "cv_winner_private_rank": winner_rank(cv_mae, mae_r),
        "public_winner_private_rank": winner_rank(mae_p, mae_r),
    }
    trip = np.array(
        [out["mae_cv_public_spearman"], out["mae_public_private_spearman"], out["mae_cv_private_spearman"]]
    )
    out["mae_triple_min"] = float(np.nanmin(trip))
    out["mae_triple_mean"] = float(np.nanmean(trip))
    return out


def lofo_family(models, y_by_id, pub_ids, priv_ids) -> dict:
    families = sorted({m["family"] for m in models})
    detail = {}
    for fam in families:
        sub = [m for m in models if m["family"] != fam]
        if len(sub) < 3:
            continue
        detail[fam] = eval_split(sub, y_by_id, pub_ids, priv_ids)["mae_public_private_spearman"]
    vals = np.array(list(detail.values()), float) if detail else np.array([np.nan])
    return {
        "worst_lofo_family_mae_pp": float(np.nanmin(vals)),
        "median_lofo_family_mae_pp": float(np.nanmedian(vals)),
        "lofo_family_mae_pp_sd": float(np.nanstd(vals)),
        "lofo_detail": detail,
    }


def lomo(models, y_by_id, pub_ids, priv_ids) -> dict:
    vals = [
        eval_split([m for j, m in enumerate(models) if j != i], y_by_id, pub_ids, priv_ids)[
            "mae_public_private_spearman"
        ]
        for i in range(len(models))
    ]
    vals = np.array(vals, float)
    return {
        "worst_lomo_mae_pp": float(np.nanmin(vals)),
        "median_lomo_mae_pp": float(np.nanmedian(vals)),
        "lomo_mae_pp_sd": float(np.nanstd(vals)),
    }


def subset_robust(models, y_by_id, pub_ids, priv_ids) -> dict:
    simple = [m for m in models if m["family"] in {"CONST", "SEQ_SIMPLE", "BIO"}]
    plm = [
        m
        for m in models
        if m["family"]
        in {"ABLANG2_NONLINEAR", "NESTED_ENSEMBLE", "ESM2_PLM", "ESMFOLD_STRUCTURE", "PLM_STRUCTURE_FUSION"}
    ]
    combo = list({m["model_id"]: m for m in simple + plm}.values())
    out, vals = {}, []
    for name, sub in [("simple_domain", simple), ("plm_ensemble", plm), ("simple_plm_ensemble", combo)]:
        if len(sub) < 3:
            out[f"subset_{name}_mae_pp"] = np.nan
            continue
        v = eval_split(sub, y_by_id, pub_ids, priv_ids)["mae_public_private_spearman"]
        out[f"subset_{name}_mae_pp"] = v
        vals.append(v)
    out["subset_mae_pp_median"] = float(np.nanmedian(vals)) if vals else np.nan
    out["subset_mae_pp_sd"] = float(np.nanstd(vals)) if vals else np.nan
    return out


def bootstrap_transfer(models, y_by_id, pub_ids, priv_ids, n_boot, seed) -> dict:
    rng = np.random.default_rng(seed)
    pub, priv = list(pub_ids), list(priv_ids)
    cv_mae = np.array([m["cv_mae"] for m in models], float)
    cv_pear = np.array([m["cv_pearson"] for m in models], float)
    keys = [
        "mae_cv_public",
        "mae_public_private",
        "mae_cv_private",
        "pear_cv_public",
        "pear_public_private",
        "pear_cv_private",
    ]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        pb = [pub[i] for i in rng.integers(0, len(pub), len(pub))]
        pr = [priv[i] for i in rng.integers(0, len(priv), len(priv))]
        mae_p, pear_p, _ = subset_metrics(models, y_by_id, pb)
        mae_r, pear_r, _ = subset_metrics(models, y_by_id, pr)
        store["mae_cv_public"].append(rank_corr(-cv_mae, -mae_p))
        store["mae_public_private"].append(rank_corr(-mae_p, -mae_r))
        store["mae_cv_private"].append(rank_corr(-cv_mae, -mae_r))
        store["pear_cv_public"].append(rank_corr(cv_pear, pear_p))
        store["pear_public_private"].append(rank_corr(pear_p, pear_r))
        store["pear_cv_private"].append(rank_corr(cv_pear, pear_r))
    summary = {}
    for k, vals in store.items():
        a = np.array(vals, float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            summary[f"{k}_median"] = summary[f"{k}_p05"] = summary[f"{k}_p95"] = summary[f"{k}_p_lt0"] = np.nan
            continue
        summary[f"{k}_median"] = float(np.median(a))
        summary[f"{k}_p05"] = float(np.quantile(a, 0.05))
        summary[f"{k}_p95"] = float(np.quantile(a, 0.95))
        summary[f"{k}_p_lt0"] = float(np.mean(a < 0))
    return summary


def load_model_pack(model_set, target, pub_order, priv_order):
    test_ids = list(pub_order) + list(priv_order)
    lab = pd.read_csv(ORG / "test_labels_hidden.csv").set_index("id")
    models = []
    for m in model_set[target]:
        tag = m["tag"]
        pred = load_test_pred(target, tag)
        assert len(pred) == len(test_ids), (target, tag, len(pred), len(test_ids))
        models.append(
            {
                "model_id": m["model_id"],
                "tag": tag,
                "family": m["model_family"],
                "pred_by_id": dict(zip(test_ids, pred)),
                "cv_mae": m["CV_metrics"]["mae"],
                "cv_pearson": m["CV_metrics"]["pearson"],
                "cv_spearman": m["CV_metrics"]["spearman"],
            }
        )
    y_by_id = {i: float(lab.loc[i, target]) for i in test_ids}
    return models, y_by_id


def hic_class(rho: float) -> str:
    if not np.isfinite(rho) or rho < 0.50:
        return "HIC_CONCERNING"
    if rho >= 0.75:
        return "HIC_EXCELLENT"
    if rho >= 0.65:
        return "HIC_GOOD"
    return "HIC_ACCEPTABLE"


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)


def select_shortlist(tm_df: pd.DataFrame, cand_map: dict, size=8, jaccard_cap=0.85):
    d = tm_df[~tm_df.is_baseline].copy()
    d["pass_pos"] = (
        (d.mae_cv_public_spearman > 0)
        & (d.mae_public_private_spearman > 0)
        & (d.mae_cv_private_spearman > 0)
    )
    d = d.sort_values(
        by=[
            "pass_pos",
            "mae_triple_min",
            "mae_triple_mean",
            "pear_public_private_spearman",
            "worst_lofo_family_mae_pp",
            "top3_public_private",
            "balance_score",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    chosen = []
    for _, row in d.iterrows():
        if len(chosen) >= size:
            break
        cid = row.candidate_id
        pubs = cand_map[cid]["public_ids"]
        if any(jaccard(pubs, cand_map[c]["public_ids"]) >= jaccard_cap for c in chosen):
            continue
        chosen.append(cid)
    bal_best = d.sort_values("balance_score").iloc[0].candidate_id
    if bal_best not in chosen and len(chosen) < size:
        pubs = cand_map[bal_best]["public_ids"]
        if not any(jaccard(pubs, cand_map[c]["public_ids"]) >= jaccard_cap for c in chosen):
            chosen.append(bal_best)
    if len(chosen) < size:
        for _, row in d.iterrows():
            if len(chosen) >= size:
                break
            if row.candidate_id not in chosen:
                chosen.append(row.candidate_id)
    return chosen[:size]


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "```"


def main():
    t0 = time.time()
    ensure_dirs()
    assert_key_paths()

    role = pd.read_csv(ORG / "role_map.csv")
    pop = pd.read_csv(ORG / "final_population.csv")
    outer = json.loads(OUTER_PATH.read_text())
    train_ids = list(outer["train_ids"])
    role_ix = role.set_index("id")
    assert len(train_ids) == 162

    print("=== P0 freeze (before any model-based candidate scoring) ===")
    model_set, protocol, model_hash, proto_hash = freeze_p0(role_ix, train_ids)

    print("=== Features ===")
    feat = build_features(role, pop)
    feat.to_csv(CACHE / "test_balance_features.csv", index=False)
    train_tm_q90 = float(role.loc[role.role == "Train", "TmApp"].quantile(0.9))
    train_hic_q90 = float(role.loc[role.role == "Train", "HIC"].quantile(0.9))

    print("=== Candidates ===")
    n_cand = int(protocol["candidate_generation"]["n_candidates"])
    seed = int(protocol["candidate_generation"]["seed"])
    candidates = generate_candidates(feat, n_cand, seed)
    with open(CAND / "candidate_pool.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"generated {len(candidates)} incl baseline")

    print("=== Pre-model balance ===")
    bal_rows = []
    for c in candidates:
        terms = balance_one(feat, c["public_ids"], c["private_ids"], train_tm_q90, train_hic_q90)
        bal_rows.append(
            {
                "candidate_id": c["candidate_id"],
                "public_hash": c["public_hash"],
                "is_baseline": c["is_baseline"],
                **{k: v for k, v in terms.items() if k == "balance_score" or k.startswith(("cont::", "cat::"))},
            }
        )
    bal_df = pd.DataFrame(bal_rows)
    bal_df.to_csv(MET / "pre_model_candidate_balance.csv", index=False)

    nonbase = bal_df[~bal_df.is_baseline].sort_values("balance_score")
    n_keep = min(
        int(protocol["pre_model_balance"]["max_survivors"]),
        max(1, int(np.ceil(len(nonbase) * protocol["pre_model_balance"]["top_fraction"]))),
    )
    keep_ids = set(nonbase.head(n_keep).candidate_id) | set(
        bal_df.loc[bal_df.is_baseline, "candidate_id"]
    )
    survivors = [c for c in candidates if c["candidate_id"] in keep_ids]
    print(f"balance survivors {len(survivors)} (keep={n_keep}+baseline)")

    print("=== Load models (TmApp scoring starts) ===")
    # B5 final/*.npy are aligned to role_map Public then Private order (NOT public_ids.csv order)
    pub_order = role.loc[role.role == "Public", "id"].tolist()
    priv_order = role.loc[role.role == "Private", "id"].tolist()
    assert set(pub_order) == set(pd.read_csv(ORG / "public_ids.csv")["id"])
    assert set(priv_order) == set(pd.read_csv(ORG / "private_ids.csv")["id"])
    tm_models, y_tm = load_model_pack(model_set, "TmApp", pub_order, priv_order)
    bal_map = bal_df.set_index("candidate_id")["balance_score"].to_dict()
    cand_map = {c["candidate_id"]: c for c in candidates}

    print("=== TmApp scoring ===")
    tm_rows = []
    for i, c in enumerate(survivors):
        e = eval_split(tm_models, y_tm, c["public_ids"], c["private_ids"])
        lf = lofo_family(tm_models, y_tm, c["public_ids"], c["private_ids"])
        lm = lomo(tm_models, y_tm, c["public_ids"], c["private_ids"])
        sr = subset_robust(tm_models, y_tm, c["public_ids"], c["private_ids"])
        tm_rows.append(
            {
                "candidate_id": c["candidate_id"],
                "public_hash": c["public_hash"],
                "private_hash": c["private_hash"],
                "is_baseline": c["is_baseline"],
                "balance_score": bal_map[c["candidate_id"]],
                **e,
                **{k: v for k, v in lf.items() if k != "lofo_detail"},
                **lm,
                **sr,
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  scored {i+1}/{len(survivors)}")
    tm_df = pd.DataFrame(tm_rows)
    tm_df.to_csv(MET / "tmapp_candidate_metrics.csv", index=False)

    print("=== Bootstrap screen ===")
    prom = tm_df[~tm_df.is_baseline].copy()
    prom["pass_pos"] = (
        (prom.mae_cv_public_spearman > 0)
        & (prom.mae_public_private_spearman > 0)
        & (prom.mae_cv_private_spearman > 0)
    )
    prom = prom.sort_values(
        by=["pass_pos", "mae_triple_min", "mae_triple_mean", "worst_lofo_family_mae_pp"],
        ascending=[False, False, False, False],
    )
    top_n = int(protocol["TmApp_evaluation"]["bootstrap_screen_top_n"])
    screen_ids = list(dict.fromkeys(["CURRENT_BASELINE_SPLIT"] + prom.head(top_n).candidate_id.tolist()))
    n_boot_s = int(protocol["TmApp_evaluation"]["bootstrap_screen_replicates"])
    boot_rows = []
    for cid in screen_ids:
        c = cand_map[cid]
        s = bootstrap_transfer(tm_models, y_tm, c["public_ids"], c["private_ids"], n_boot_s, seed=seed + 17)
        boot_rows.append({"candidate_id": cid, "stage": "screen_1000", **s})
    boot_screen = pd.DataFrame(boot_rows)
    for col in ["mae_public_private_median", "mae_public_private_p05", "mae_public_private_p_lt0"]:
        tm_df[col] = tm_df.candidate_id.map(boot_screen.set_index("candidate_id")[col].to_dict())

    print("=== Freeze TmApp shortlist (BEFORE HIC) ===")
    short_ids = select_shortlist(
        tm_df,
        cand_map,
        size=int(protocol["TmApp_evaluation"]["shortlist_size"]),
        jaccard_cap=float(protocol["TmApp_evaluation"]["shortlist_jaccard_cap"]),
    )
    shortlist = [
        {
            "candidate_id": cid,
            "public_hash": cand_map[cid]["public_hash"],
            "private_hash": cand_map[cid]["private_hash"],
            "public_ids": cand_map[cid]["public_ids"],
            "private_ids": cand_map[cid]["private_ids"],
            "balance_score": float(bal_map[cid]),
        }
        for cid in short_ids
    ]
    short_obj = {
        "frozen_note": "TmApp-only shortlist frozen BEFORE any HIC model-performance inspection",
        "protocol_sha256": proto_hash,
        "n": len(shortlist),
        "candidates": shortlist,
    }
    short_hash = write_json(CFG / "TMAPP_SHORTLIST.json", short_obj)
    print(f"shortlist n={len(shortlist)} hash={short_hash[:12]}")

    print("=== Bootstrap final 5000 ===")
    n_boot_f = int(protocol["TmApp_evaluation"]["bootstrap_final_replicates"])
    boot_final_rows = []
    for cid in ["CURRENT_BASELINE_SPLIT"] + short_ids:
        c = cand_map[cid]
        s = bootstrap_transfer(tm_models, y_tm, c["public_ids"], c["private_ids"], n_boot_f, seed=seed + 99)
        boot_final_rows.append({"candidate_id": cid, "stage": "final_5000", **s})
    boot_all = pd.concat([boot_screen, pd.DataFrame(boot_final_rows)], ignore_index=True)
    boot_all.to_csv(MET / "tmapp_bootstrap_robustness.csv", index=False)

    short_met = tm_df[tm_df.candidate_id.isin(short_ids + ["CURRENT_BASELINE_SPLIT"])].copy()
    bf = pd.DataFrame(boot_final_rows).set_index("candidate_id")
    for col in bf.columns:
        if col != "stage":
            short_met[f"final_{col}"] = short_met.candidate_id.map(bf[col].to_dict())
    short_met.to_csv(MET / "tmapp_shortlist_metrics.csv", index=False)

    fam_rows = []
    tm_ix = tm_df.set_index("candidate_id")
    for cid in short_ids + ["CURRENT_BASELINE_SPLIT"]:
        c = cand_map[cid]
        lf = lofo_family(tm_models, y_tm, c["public_ids"], c["private_ids"])
        for fam, v in lf["lofo_detail"].items():
            fam_rows.append({"candidate_id": cid, "left_out_family": fam, "mae_public_private_spearman": v})
        fam_rows.append(
            {
                "candidate_id": cid,
                "left_out_family": "NONE_FULL",
                "mae_public_private_spearman": float(tm_ix.loc[cid, "mae_public_private_spearman"]),
            }
        )
    pd.DataFrame(fam_rows).to_csv(MET / "tmapp_model_family_sensitivity.csv", index=False)

    jac_rows = []
    for i, a in enumerate(short_ids):
        for b in short_ids[i + 1 :]:
            A, B = set(cand_map[a]["public_ids"]), set(cand_map[b]["public_ids"])
            jac_rows.append(
                {"a": a, "b": b, "jaccard_public": len(A & B) / len(A | B), "overlap_n": len(A & B)}
            )
    pd.DataFrame(jac_rows).to_csv(MET / "shortlist_jaccard.csv", index=False)

    print("=== HIC safety on shortlist only ===")
    hic_models, y_hic = load_model_pack(model_set, "HIC", pub_order, priv_order)
    base = cand_map["CURRENT_BASELINE_SPLIT"]
    base_hic_e = eval_split(hic_models, y_hic, base["public_ids"], base["private_ids"])
    floors = protocol["HIC_safety"]["floors"]
    hic_rows = []
    for cid in short_ids + ["CURRENT_BASELINE_SPLIT"]:
        c = cand_map[cid]
        e = eval_split(hic_models, y_hic, c["public_ids"], c["private_ids"])
        pass_floor = (
            e["mae_public_private_spearman"] >= floors["mae_model_rank_Public_to_Private_spearman"]
            and e["pear_public_private_spearman"] >= floors["pearson_model_rank_Public_to_Private_spearman"]
            and e["mae_cv_private_spearman"] > floors["mae_model_rank_CV_to_Private_spearman"]
            and e["mae_public_private_spearman"] >= 0
        )
        hic_rows.append(
            {
                "candidate_id": cid,
                "is_baseline": cid == "CURRENT_BASELINE_SPLIT",
                **e,
                "delta_mae_cv_public": e["mae_cv_public_spearman"] - base_hic_e["mae_cv_public_spearman"],
                "delta_mae_public_private": e["mae_public_private_spearman"]
                - base_hic_e["mae_public_private_spearman"],
                "delta_mae_cv_private": e["mae_cv_private_spearman"] - base_hic_e["mae_cv_private_spearman"],
                "delta_pear_cv_public": e["pear_cv_public_spearman"] - base_hic_e["pear_cv_public_spearman"],
                "delta_pear_public_private": e["pear_public_private_spearman"]
                - base_hic_e["pear_public_private_spearman"],
                "delta_pear_cv_private": e["pear_cv_private_spearman"] - base_hic_e["pear_cv_private_spearman"],
                "hic_class": hic_class(e["mae_public_private_spearman"]),
                "pass_hic_safety": True if cid == "CURRENT_BASELINE_SPLIT" else bool(pass_floor),
            }
        )
    hic_df = pd.DataFrame(hic_rows)
    hic_df.to_csv(MET / "hic_shortlist_metrics.csv", index=False)

    print("=== Final selection ===")
    hic_s = hic_df.set_index("candidate_id")
    passing = [cid for cid in short_ids if bool(hic_s.loc[cid, "pass_hic_safety"])]
    selected = None
    decision = "NO_ACCEPTABLE_COMMON_SPLIT_FOUND"
    if passing:
        pass_tm = tm_df[tm_df.candidate_id.isin(passing)].copy()
        pass_tm["pass_pos"] = (
            (pass_tm.mae_cv_public_spearman > 0)
            & (pass_tm.mae_public_private_spearman > 0)
            & (pass_tm.mae_cv_private_spearman > 0)
        )
        pass_tm = pass_tm.sort_values(
            by=[
                "pass_pos",
                "mae_triple_min",
                "mae_triple_mean",
                "pear_public_private_spearman",
                "worst_lofo_family_mae_pp",
                "top3_public_private",
            ],
            ascending=[False, False, False, False, False, False],
        )
        top = pass_tm.iloc[0]
        if len(pass_tm) > 1:
            second = pass_tm.iloc[1]
            near = abs(top.mae_triple_mean - second.mae_triple_mean) < 0.03 and abs(
                top.mae_triple_min - second.mae_triple_min
            ) < 0.03
            if near and (
                hic_s.loc[second.candidate_id, "mae_public_private_spearman"]
                > hic_s.loc[top.candidate_id, "mae_public_private_spearman"] + 0.02
            ):
                top = second
        selected = top.candidate_id
        decision = "SELECT_COMMON_PRODUCTION_SPLIT"

    overfit_label = None
    replay_rows = []
    if selected is not None:
        for fam in sorted({m["family"] for m in tm_models}):
            sub = [m for m in tm_models if m["family"] != fam]
            ranks = []
            for cid in short_ids:
                c = cand_map[cid]
                e = eval_split(sub, y_tm, c["public_ids"], c["private_ids"])
                ranks.append((cid, e["mae_triple_min"], e["mae_triple_mean"], e["mae_public_private_spearman"]))
            ranks.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            rank_pos = [i for i, r in enumerate(ranks) if r[0] == selected][0] + 1
            replay_rows.append(
                {
                    "left_out_family": fam,
                    "selected_rank_among_shortlist": rank_pos,
                    "shortlist_n": len(short_ids),
                    "selected_still_topk3": rank_pos <= 3,
                }
            )
        if any(r["selected_rank_among_shortlist"] > max(3, len(short_ids) // 2) for r in replay_rows):
            overfit_label = "MODEL_SET_OVERFIT_RISK"
        pd.DataFrame(replay_rows).to_csv(MET / "tmapp_lofo_selection_replay.csv", index=False)

    cmp_rows = []
    for cid in ["CURRENT_BASELINE_SPLIT"] + short_ids:
        tr = tm_ix.loc[cid]
        hr = hic_s.loc[cid]
        cmp_rows.append(
            {
                "candidate_id": cid,
                "is_selected": cid == selected,
                "tm_mae_cv_public": tr.mae_cv_public_spearman,
                "tm_mae_public_private": tr.mae_public_private_spearman,
                "tm_mae_cv_private": tr.mae_cv_private_spearman,
                "tm_pear_cv_public": tr.pear_cv_public_spearman,
                "tm_pear_public_private": tr.pear_public_private_spearman,
                "tm_pear_cv_private": tr.pear_cv_private_spearman,
                "hic_mae_cv_public": hr.mae_cv_public_spearman,
                "hic_mae_public_private": hr.mae_public_private_spearman,
                "hic_mae_cv_private": hr.mae_cv_private_spearman,
                "hic_pear_cv_public": hr.pear_cv_public_spearman,
                "hic_pear_public_private": hr.pear_public_private_spearman,
                "hic_pear_cv_private": hr.pear_cv_private_spearman,
                "balance_score": tr.balance_score,
            }
        )
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(MET / "current_vs_final_split.csv", index=False)

    if selected is not None:
        c = cand_map[selected]
        pd.DataFrame({"id": c["public_ids"]}).to_csv(FRZ / "public_ids.csv", index=False)
        pd.DataFrame({"id": c["private_ids"]}).to_csv(FRZ / "private_ids.csv", index=False)
        train_ids_list = pd.read_csv(ORG / "train_ids.csv")["id"].tolist()
        manifest = {
            "title": "FINAL PRODUCTION PUBLIC/PRIVATE MASK",
            "warning": "FINAL PRODUCTION PUBLIC/PRIVATE MASK — DO NOT CHANGE BASED ON ANY FURTHER MODEL RESULTS",
            "role": "production leaderboard partition — NOT an untouched scientific holdout",
            "scientific_model_development_evidence": "Train grouped CV remains primary",
            "competition_final_ranking_set": "Private",
            "decision": decision,
            "selected_candidate_id": selected,
            "Train_ids_sha256": sha256_ids(train_ids_list),
            "Test_ids_sha256": sha256_ids(c["public_ids"] + c["private_ids"]),
            "Public_ids_sha256": c["public_hash"],
            "Private_ids_sha256": c["private_hash"],
            "protocol_sha256": proto_hash,
            "tmapp_shortlist_sha256": short_hash,
            "model_set_sha256": model_hash,
            "common_mask": True,
            "n_public": 81,
            "n_private": 81,
            "overfit_label": overfit_label,
            "hic_safety_pass_count": len(passing),
            "lofo_selection_replay": replay_rows,
        }
        write_json(FRZ / "FINAL_PRODUCTION_SPLIT_MANIFEST.json", manifest)

    # --- Reports ---
    base_tm = tm_ix.loc["CURRENT_BASELINE_SPLIT"]
    base_hic = hic_s.loc["CURRENT_BASELINE_SPLIT"]
    sel_tm = tm_ix.loc[selected] if selected else None
    sel_hic = hic_s.loc[selected] if selected else None

    cols_short = [
        "candidate_id",
        "balance_score",
        "mae_cv_public_spearman",
        "mae_public_private_spearman",
        "mae_cv_private_spearman",
        "pear_cv_public_spearman",
        "pear_public_private_spearman",
        "pear_cv_private_spearman",
        "top3_public_private",
        "worst_lofo_family_mae_pp",
    ]
    (REP / "tmapp_split_search.md").write_text(
        "\n".join(
            [
                "# Gate B6 — TmApp split search",
                f"- Raw candidates (incl. baseline): **{len(candidates)}**",
                f"- Pre-model balance survivors scored: **{len(survivors)}**",
                f"- Shortlist size: **{len(short_ids)}**",
                f"- Protocol sha256: `{proto_hash}`",
                "",
                "## Shortlist (+ baseline)",
                md_table(short_met[cols_short]),
                "",
                "## Bootstrap final (5000)",
                md_table(
                    pd.DataFrame(boot_final_rows)[
                        [
                            "candidate_id",
                            "mae_public_private_median",
                            "mae_public_private_p05",
                            "mae_public_private_p95",
                            "mae_public_private_p_lt0",
                            "mae_cv_public_median",
                            "mae_cv_private_median",
                        ]
                    ]
                ),
                "",
            ]
        )
    )

    (REP / "hic_shortlist_safety_check.md").write_text(
        "\n".join(
            [
                "# Gate B6 — HIC shortlist safety check",
                "Evaluated **only** on frozen TmApp shortlist (+ baseline).",
                "",
                md_table(
                    hic_df[
                        [
                            "candidate_id",
                            "mae_cv_public_spearman",
                            "mae_public_private_spearman",
                            "mae_cv_private_spearman",
                            "pear_public_private_spearman",
                            "top3_public_private",
                            "public_winner_private_rank",
                            "delta_mae_public_private",
                            "delta_pear_public_private",
                            "hic_class",
                            "pass_hic_safety",
                        ]
                    ]
                ),
                "",
                f"HIC safety pass count (excl. baseline): **{len(passing)} / {len(short_ids)}**",
                "",
            ]
        )
    )

    cov_note = "N/A"
    if selected is not None:

        def abs_imb(ids_pub, ids_priv):
            pu = feat[feat.id.isin(ids_pub)]
            pr = feat[feat.id.isin(ids_priv)]
            out = {}
            for fcol in [
                "cluster_size",
                "A2_HL_charge_ph7",
                "A2_HL_pI",
                "A2_HL_gravy",
                "TmApp",
                "HIC",
                "C_combined_germline_distance",
                "nearest_train_VH_identity",
            ]:
                if fcol in pu.columns:
                    out[fcol] = abs(smd(pu[fcol].values, pr[fcol].values))
            for fcol in ["vl_family", "vh_family"]:
                levels = sorted(set(pu[fcol].astype(str)) | set(pr[fcol].astype(str)))
                pa = np.array([(pu[fcol].astype(str) == lv).mean() for lv in levels])
                pb = np.array([(pr[fcol].astype(str) == lv).mean() for lv in levels])
                out[fcol] = js_div(pa, pb)
            return out

        b = abs_imb(base["public_ids"], base["private_ids"])
        s = abs_imb(cand_map[selected]["public_ids"], cand_map[selected]["private_ids"])
        improved = sorted({k: b[k] - s[k] for k in b}.items(), key=lambda x: -x[1])
        cov_note = ", ".join(f"{k}: Δ|imb|={v:+.3f}" for k, v in improved[:6])

    mean_jac = float(np.mean([r["jaccard_public"] for r in jac_rows])) if jac_rows else float("nan")
    final_path = REP / "GATE_B6_COMMON_SPLIT_FINAL.md"

    flines = [
        "# Gate B6 — Common Public/Private Production Split Search FINAL",
        "",
        "Overall decision:",
        "",
        f"**{decision}**" + (f" (`{selected}`)" if selected else ""),
    ]
    if overfit_label:
        flines.append(f"\nSensitivity flag: `{overfit_label}`")
    flines += [
        "",
        "Current split:",
        f"    TmApp MAE Public→Private: {base_tm.mae_public_private_spearman:.4f}",
        f"    TmApp Pearson Public→Private: {base_tm.pear_public_private_spearman:.4f}",
        f"    HIC MAE Public→Private: {base_hic.mae_public_private_spearman:.4f}",
        f"    HIC Pearson Public→Private: {base_hic.pear_public_private_spearman:.4f}",
        "",
        "Selected production split:",
    ]
    if selected is not None:
        flines += [
            f"    TmApp MAE CV→Public: {sel_tm.mae_cv_public_spearman:.4f}",
            f"    TmApp MAE Public→Private: {sel_tm.mae_public_private_spearman:.4f}",
            f"    TmApp MAE CV→Private: {sel_tm.mae_cv_private_spearman:.4f}",
            "",
            f"    TmApp Pearson CV→Public: {sel_tm.pear_cv_public_spearman:.4f}",
            f"    TmApp Pearson Public→Private: {sel_tm.pear_public_private_spearman:.4f}",
            f"    TmApp Pearson CV→Private: {sel_tm.pear_cv_private_spearman:.4f}",
            "",
            f"    HIC MAE CV→Public: {sel_hic.mae_cv_public_spearman:.4f}",
            f"    HIC MAE Public→Private: {sel_hic.mae_public_private_spearman:.4f}",
            f"    HIC MAE CV→Private: {sel_hic.mae_cv_private_spearman:.4f}",
            "",
            f"    HIC Pearson CV→Public: {sel_hic.pear_cv_public_spearman:.4f}",
            f"    HIC Pearson Public→Private: {sel_hic.pear_public_private_spearman:.4f}",
            f"    HIC Pearson CV→Private: {sel_hic.pear_cv_private_spearman:.4f}",
        ]
    else:
        flines.append("    None — NO_ACCEPTABLE_COMMON_SPLIT_FOUND")
    flines += [
        "",
        f"TmApp shortlist size: {len(short_ids)}",
        f"HIC safety pass count: {len(passing)}",
        "Model-set robustness: " + (overfit_label if overfit_label else "LOFO replay retained near-top ranks"),
        f"Pre-model balance: survivors={len(survivors)}/{len(candidates)}; "
        f"selected balance={sel_tm.balance_score if selected is not None else 'N/A'}",
        "Common split requirement satisfied: YES",
        "Production split freeze recommendation: "
        + ("YES — freeze under frozen/" if selected else "NO — do not replace"),
        "",
        "## Comparison table",
        "",
        md_table(cmp_df),
        "",
        "## Interpretation boundary",
        "",
        "This mask is a **production leaderboard partition** selected with frozen organizer models. "
        "It is **not** an untouched scientific holdout. For scientific organizer conclusions, "
        "**Train grouped CV** remains the primary model-development evidence; **Private** remains "
        "the competition final ranking set.",
        "",
        "## Required questions",
        "",
    ]
    q = [
        f"1. Raw candidate 81/81 partitions generated: **{len(candidates)}** (including CURRENT_BASELINE_SPLIT).",
        f"2. Survived pre-model balance filtering: **{len(survivors)}** (incl. baseline).",
        f"3. Entered TmApp leaderboard scoring: **{len(survivors)}**.",
        f"4. TmApp shortlist size: **{len(short_ids)}**.",
    ]
    if selected is not None:
        d_mae = sel_tm.mae_public_private_spearman - base_tm.mae_public_private_spearman
        d_pear = sel_tm.pear_public_private_spearman - base_tm.pear_public_private_spearman
        bf_sel = pd.DataFrame(boot_final_rows).set_index("candidate_id").loc[selected]
        q += [
            f"5. TmApp Public→Private MAE model-rank Δ: **{d_mae:+.4f}** "
            f"({base_tm.mae_public_private_spearman:.3f} → {sel_tm.mae_public_private_spearman:.3f}).",
            f"6. Pearson-based Public→Private Δ: **{d_pear:+.4f}** "
            f"({base_tm.pear_public_private_spearman:.3f} → {sel_tm.pear_public_private_spearman:.3f}).",
            f"7. CV→Public MAE meaningfully positive? **"
            f"{'YES' if sel_tm.mae_cv_public_spearman > 0.3 else ('positive but modest' if sel_tm.mae_cv_public_spearman > 0 else 'NO')}** "
            f"(ρ={sel_tm.mae_cv_public_spearman:.3f}; baseline {base_tm.mae_cv_public_spearman:.3f}).",
            f"8. CV→Private remained strong? **{'YES' if sel_tm.mae_cv_private_spearman >= 0.7 else 'MIXED'}** "
            f"(ρ={sel_tm.mae_cv_private_spearman:.3f}).",
            f"9. Bootstrap robustness: median Public→Private MAE ρ={bf_sel.mae_public_private_median:.3f}, "
            f"p05={bf_sel.mae_public_private_p05:.3f}, P(ρ<0)={bf_sel.mae_public_private_p_lt0:.3f}.",
            f"10. Family removal robustness: worst LOFO Public→Private={sel_tm.worst_lofo_family_mae_pp:.3f}; "
            f"replay={replay_rows}.",
            f"11. Biologically/statistically balanced? balance_score={sel_tm.balance_score:.3f} "
            f"(baseline {base_tm.balance_score:.3f}; lower better).",
            f"12. Variables improved most vs old Pub/Priv: {cov_note}.",
            f"13. Shortlisted candidates passing HIC safety: **{len(passing)} / {len(short_ids)}**.",
            f"14. HIC degradation vs current: MAE Public→Private Δ={sel_hic.delta_mae_public_private:+.4f}; "
            f"Pearson Δ={sel_hic.delta_pear_public_private:+.4f}; class={sel_hic.hic_class}.",
            f"15. HIC still acceptable participant feedback? **{'YES' if sel_hic.pass_hic_safety else 'NO'}** "
            f"({sel_hic.hic_class}).",
            f"16. Final common split satisfies both tracks? **{'YES' if decision.startswith('SELECT') else 'NO'}**.",
            f"17. Dependent on one organizer family? **"
            f"{'RISK FLAG' if overfit_label else 'No strong evidence of single-family dependence'}**.",
            f"18. Robust region vs lucky isolates? mean pairwise Public Jaccard among shortlist={mean_jac:.3f} "
            f"({'broad region' if mean_jac < 0.7 else 'relatively concentrated'}).",
            "19. Should new mask replace old Public/Private? **YES — recommended for human approval**.",
            "20. After replacement, permanently freeze? **YES** — do not change based on further model results.",
        ]
    else:
        q += [f"{i}. N/A — no acceptable common split found." for i in range(5, 21)]
    flines.extend(q)
    flines += ["", f"_Elapsed: {time.time() - t0:.1f}s_", ""]
    final_path.write_text("\n".join(flines))

    print("=== DECISION ===")
    print(decision, selected)
    print(f"FINAL_REPORT={final_path}")
    print(f"elapsed_s={time.time() - t0:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=None, help="override protocol n_candidates")
    ap.add_argument("--bootstrap-final", type=int, default=None)
    ap.add_argument("--bootstrap-screen", type=int, default=None)
    args = ap.parse_args()
    # stash overrides for freeze_p0 / main via env-like globals
    if args.n_candidates is not None:
        globals()["_OVERRIDE_N_CAND"] = args.n_candidates
    if args.bootstrap_final is not None:
        globals()["_OVERRIDE_BOOT_FINAL"] = args.bootstrap_final
    if args.bootstrap_screen is not None:
        globals()["_OVERRIDE_BOOT_SCREEN"] = args.bootstrap_screen
    main()
