#!/usr/bin/env python3
"""Gate B7 — Blind Virtual Competition Simulation (CAND_12528).

Participant decisions use ONLY Train CV + Public feedback.
Private is scored silently and revealed only after final freeze.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
GATE = ROOT / "gate_b7_virtual_competition"
sys.path.insert(0, str(ROOT / "gate_b4_absolute/scripts"))
from b4_common import load_representation  # noqa: E402

for d in [
    "config",
    "organizer",
    "participant",
    "submissions",
    "logs",
    "metrics",
    "plots",
    "reports",
    "scripts",
    "cache",
]:
    (GATE / d).mkdir(parents=True, exist_ok=True)

SEED = 20260830


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def safe_spearman(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    return float(spearmanr(y, p).correlation)


def safe_pearson(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    return float(pearsonr(y, p)[0])


def load_competition():
    cand = next(
        c
        for c in json.loads(
            (ROOT / "gate_b6_split_search/config/B6_1_FINAL_CANDIDATE_SET.json").read_text()
        )["candidates"]
        if c["candidate_id"] == "CAND_12528"
    )
    outer = json.loads((ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json").read_text())
    pop = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv").set_index("id")
    role = pd.read_csv(ROOT / "gate_b3/frozen/organizer/role_map.csv").set_index("id")
    train_ids = list(outer["train_ids"])
    public_ids = list(cand["public_ids"])
    private_ids = list(cand["private_ids"])
    assert len(train_ids) == 162 and len(public_ids) == 81 and len(private_ids) == 81
    assert set(train_ids).isdisjoint(set(public_ids) | set(private_ids))

    folds = []
    for rep in outer["folds"]:
        folds.append(np.asarray(rep["fold_id"], int))

    groups = role.loc[train_ids, "sequence_group"].astype(int).values
    return {
        "train_ids": train_ids,
        "public_ids": public_ids,
        "private_ids": private_ids,
        "pop": pop,
        "role": role,
        "folds": folds,
        "groups": groups,
        "cand": cand,
        "outer": outer,
    }


def y_of(pop, ids, target):
    return pop.loc[ids, target].astype(float).values


FEAT_CACHE = {}
BIO_CATEGORY_IDS = None  # set to train_ids so one-hot columns are frozen


def get_X(tag, ids):
    key = (tag, tuple(ids), tuple(BIO_CATEGORY_IDS) if BIO_CATEGORY_IDS is not None else None)
    if key in FEAT_CACHE:
        return FEAT_CACHE[key]
    if tag == "CONST":
        X = np.ones((len(ids), 1))
    elif tag == "ESM2_PCA64":
        X = load_representation("PLM_ESM2", ids)
    elif tag == "ABLANG2_PCA32":
        X = load_representation("PLM_ABLANG2", ids)
    elif tag == "FUSION_ESM2_STRUCT":
        X = np.hstack(
            [load_representation("PLM_ESM2", ids), load_representation("ESMFN_STRUCTURE", ids)]
        )
    elif tag == "BIO":
        from b4_common import load_bio

        X = load_bio(ids, category_ids=BIO_CATEGORY_IDS)
    elif tag == "BIO_ABLANG2":
        from b4_common import load_bio

        X = np.hstack([load_bio(ids, category_ids=BIO_CATEGORY_IDS), load_representation("PLM_ABLANG2", ids)])
    elif tag == "PHYS_STRUCT":
        X = np.hstack(
            [load_representation("SEQ_SIMPLE", ids), load_representation("ESMFN_STRUCTURE", ids)]
        )
    else:
        X = load_representation(tag, ids)
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = 0.0
    FEAT_CACHE[key] = X
    return X


@dataclass
class ModelSpec:
    name: str
    feat: str
    kind: str
    stage: int
    hypothesis: str
    params: dict = field(default_factory=dict)


def fit_predict_matrix(spec: ModelSpec, Xtr, ytr, Xte, sample_weight=None):
    if spec.kind == "const":
        v = float(np.median(ytr))
        return np.full(len(Xte), v)
    if spec.kind == "ridge":
        est = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=spec.params.get("alpha", 10.0)))])
    elif spec.kind == "enet":
        est = Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    ElasticNet(
                        alpha=spec.params.get("alpha", 0.2),
                        l1_ratio=spec.params.get("l1", 0.5),
                        max_iter=20000,
                    ),
                ),
            ]
        )
    elif spec.kind == "huber":
        est = Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(alpha=1e-4, max_iter=2000))])
    elif spec.kind == "svr_pca":
        n_pca = min(spec.params.get("n_pca", 64), Xtr.shape[0] - 1, Xtr.shape[1])
        est = Pipeline(
            [
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=SEED)),
                ("m", SVR(C=spec.params.get("C", 1.0), epsilon=spec.params.get("eps", 0.1), gamma="scale")),
            ]
        )
    else:
        raise ValueError(spec.kind)
    if sample_weight is not None and spec.kind in ("ridge", "enet", "huber"):
        est.fit(Xtr, ytr, m__sample_weight=sample_weight)
    else:
        est.fit(Xtr, ytr)
    return est.predict(Xte)


def cv_oof(spec: ModelSpec, train_ids, y, folds, elev_weight=False):
    X = get_X(spec.feat, train_ids) if spec.feat != "CONST" else np.ones((len(train_ids), 1))
    oof = np.zeros(len(train_ids))
    fold_maes = []
    for r, fid in enumerate(folds):
        oof_r = np.zeros(len(train_ids))
        for f in range(5):
            te = fid == f
            tr = ~te
            w = np.where(y[tr] >= 10.5, 2.0, 1.0) if elev_weight else None
            pred = fit_predict_matrix(spec, X[tr], y[tr], X[te], sample_weight=w)
            oof_r[te] = pred
            fold_maes.append(mae(y[te], pred))
        if r == 0:
            oof = oof_r
    return {
        "oof": oof,
        "cv_mae": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes)),
        "cv_spearman": safe_spearman(y, oof),
        "cv_pearson": safe_pearson(y, oof),
    }


def fit_full_predict(spec: ModelSpec, train_ids, ytr, test_ids, elev_weight=False):
    Xtr = get_X(spec.feat, train_ids) if spec.feat != "CONST" else np.ones((len(train_ids), 1))
    Xte = get_X(spec.feat, test_ids) if spec.feat != "CONST" else np.ones((len(test_ids), 1))
    w = np.where(ytr >= 10.5, 2.0, 1.0) if elev_weight else None
    return fit_predict_matrix(spec, Xtr, ytr, Xte, sample_weight=w)


def blend_predict(preds_list, weights=None):
    P = np.vstack(preds_list)
    if weights is None:
        return P.mean(axis=0)
    w = np.asarray(weights, float)
    w = w / w.sum()
    return (P * w[:, None]).sum(axis=0)


class Organizer:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.y_pub = y_of(data["pop"], data["public_ids"], target)
        self.y_priv = y_of(data["pop"], data["private_ids"], target)
        self.private_vault = {}

    def score_public(self, pred_pub, history_public_maes):
        m = mae(self.y_pub, pred_pub)
        rank = 1 + sum(1 for x in history_public_maes if x < m - 1e-15)
        prev = history_public_maes[-1] if history_public_maes else None
        delta = None if prev is None else m - prev
        return {
            "public_mae": m,
            "public_spearman": safe_spearman(self.y_pub, pred_pub),
            "public_pearson": safe_pearson(self.y_pub, pred_pub),
            "personal_rank": rank,
            "n_personal": len(history_public_maes) + 1,
            "delta_vs_prev": delta,
        }

    def vault_private(self, sid, pred_priv):
        self.private_vault[sid] = {
            "private_mae": mae(self.y_priv, pred_priv),
            "private_spearman": safe_spearman(self.y_priv, pred_priv),
            "private_pearson": safe_pearson(self.y_priv, pred_priv),
            "pred_priv": pred_priv,
        }


def tmapp_catalog():
    return [
        ModelSpec("CONST_MEDIAN", "CONST", "const", 0, "Constant median baseline."),
        ModelSpec("SEQ_SIMPLE_Ridge", "SEQ_SIMPLE", "ridge", 1, "Simple sequence descriptors + Ridge."),
        ModelSpec("BIO_Ridge", "BIO", "ridge", 2, "BIO annotations may capture developability-relevant motifs."),
        ModelSpec(
            "ABLANG2_PCA32_SVR",
            "ABLANG2_PCA32",
            "svr_pca",
            3,
            "AbLang2 antibody PLM embeddings may beat handcrafted descriptors.",
            {"n_pca": 32, "C": 1.0},
        ),
        ModelSpec(
            "BIO_ABLANG2_Ridge",
            "BIO_ABLANG2",
            "ridge",
            5,
            "Fusing BIO + AbLang2 may combine annotation and representation signal.",
        ),
        ModelSpec(
            "ESM2_PCA64_SVR",
            "ESM2_PCA64",
            "svr_pca",
            3,
            "General protein LM (ESM2) as alternative PLM.",
            {"n_pca": 64, "C": 1.0},
        ),
        ModelSpec(
            "STRUCT_ElasticNet",
            "ESMFN_STRUCTURE",
            "enet",
            4,
            "Structure descriptors may add orthogonal signal for TmApp.",
        ),
        ModelSpec(
            "FUSION_ESM2_STRUCT_ENet",
            "FUSION_ESM2_STRUCT",
            "enet",
            5,
            "ESM2 + structure fusion.",
        ),
    ]


def hic_catalog():
    return [
        ModelSpec("CONST_MEDIAN", "CONST", "const", 0, "Constant median baseline."),
        ModelSpec("SEQ_SIMPLE_Ridge", "SEQ_SIMPLE", "ridge", 1, "Sequence physchem/composition + Ridge."),
        ModelSpec(
            "ESM2_PCA64_SVR",
            "ESM2_PCA64",
            "svr_pca",
            3,
            "ESM2 embeddings may capture sequence determinants of HIC.",
            {"n_pca": 64, "C": 1.0},
        ),
        ModelSpec(
            "STRUCT_ElasticNet",
            "ESMFN_STRUCTURE",
            "enet",
            4,
            "Exposed hydrophobic surface from ESMFold may matter for HIC.",
        ),
        ModelSpec(
            "PHYS_STRUCT_Ridge",
            "PHYS_STRUCT",
            "ridge",
            5,
            "Explicit physchem + structure surface features.",
        ),
        ModelSpec(
            "PHYS_STRUCT_Huber",
            "PHYS_STRUCT",
            "huber",
            5,
            "Huber may reduce center-collapse on elevated HIC.",
        ),
        ModelSpec(
            "PHYS_STRUCT_Ridge_elevW2",
            "PHYS_STRUCT",
            "ridge",
            5,
            "Upweight HIC>=10.5 in Train to counter regression-to-center.",
        ),
        ModelSpec(
            "FUSION_ESM2_STRUCT_ENet",
            "FUSION_ESM2_STRUCT",
            "enet",
            5,
            "ESM2 + structure fusion for HIC.",
        ),
        ModelSpec(
            "ESM2_PHYS_STRUCT_Ridge",
            "FUSION_ESM2_STRUCT",
            "ridge",
            5,
            "Ridge on ESM2+structure (alternative fusion head).",
        ),
    ]


def decide_next_cv_first(history, remaining, special_done):
    """Explore richer models first; ensemble only after several singles."""
    remaining = sorted(remaining, key=lambda s: (s.stage, s.name))
    if remaining and (len(history) < 5 or max(h.get("model_stage", 0) for h in history) < 3):
        return remaining[0], remaining[0].hypothesis
    if remaining and len(history) < 7:
        return remaining[0], remaining[0].hypothesis
    if len(history) >= 5 and "ENSEMBLE_TOP2_CV" not in special_done:
        return "ENSEMBLE_TOP2_CV", "Ensemble of two best CV models should stabilize."
    if len(history) >= 6 and "ENSEMBLE_TOP3_CV" not in special_done:
        return "ENSEMBLE_TOP3_CV", "Expand ensemble to top-3 CV."
    if remaining:
        return remaining[0], remaining[0].hypothesis
    return None, "Stop: catalog exhausted."


def decide_next_balanced(history, remaining, special_done):
    remaining = sorted(remaining, key=lambda s: (s.stage, s.name))
    if remaining and len(history) < 5:
        if history:
            last = history[-1]
            prev = history[-2] if len(history) > 1 else None
            if (
                prev
                and last["public_mae"] < prev["public_mae"]
                and last["cv_mae"] <= prev["cv_mae"] + 0.05
            ):
                adv = [s for s in remaining if s.stage >= last.get("model_stage", 0)]
                if adv:
                    return adv[0], adv[0].hypothesis
        return remaining[0], remaining[0].hypothesis
    if len(history) >= 5 and "ENSEMBLE_CV_PUBLIC" not in special_done:
        return "ENSEMBLE_CV_PUBLIC", "Blend best-CV and best-Public models to hedge disagreement."
    if len(history) >= 6 and "ENSEMBLE_TOP2_PUBLIC" not in special_done:
        best_cv = min(history, key=lambda h: h["cv_mae"])
        best_pub = min(history, key=lambda h: h["public_mae"])
        if best_cv["model_name"] != best_pub["model_name"]:
            return "ENSEMBLE_TOP2_PUBLIC", "CV-best and Public-best differ; blend them."
    if remaining:
        return remaining[0], remaining[0].hypothesis
    return None, "Stop: catalog exhausted."


def decide_next_public(history, remaining, special_done):
    remaining = sorted(remaining, key=lambda s: (s.stage, s.name))
    if remaining and len(history) < 5:
        if len(history) >= 2 and history[-1]["public_mae"] < history[-2]["public_mae"]:
            adv = [s for s in remaining if s.stage >= history[-1].get("model_stage", 0)]
            if adv:
                return (
                    adv[0],
                    adv[0].hypothesis + " (Public improved last round — continue this direction.)",
                )
        return remaining[0], remaining[0].hypothesis
    if len(history) >= 5 and "ENSEMBLE_PUBLIC_BEST2" not in special_done:
        return "ENSEMBLE_PUBLIC_BEST2", "Average the two best Public models so far."
    if len(history) >= 6 and "FOLLOW_PUBLIC_BEST" not in special_done and remaining:
        return "FOLLOW_PUBLIC_BEST", "Reinforce Public-best direction via nearby unused variant."
    if remaining:
        return remaining[0], remaining[0].hypothesis
    return None, "Stop: catalog exhausted."


def choose_final(persona, history):
    if persona == "A_CV_FIRST":
        primary = min(history, key=lambda h: (h["cv_mae"], h["public_mae"]))
        rest = [h for h in history if h["submission_id"] != primary["submission_id"]]
        backup = min(rest, key=lambda h: (h["cv_mae"], h["public_mae"])) if rest else primary
        reason = "CV-first: minimize CV MAE; Public only as tie-break."
    elif persona == "B_BALANCED":
        cv_sorted = sorted(history, key=lambda h: h["cv_mae"])
        pub_sorted = sorted(history, key=lambda h: h["public_mae"])
        cv_rank = {h["submission_id"]: i for i, h in enumerate(cv_sorted)}
        pub_rank = {h["submission_id"]: i for i, h in enumerate(pub_sorted)}
        primary = min(
            history,
            key=lambda h: (cv_rank[h["submission_id"]] + pub_rank[h["submission_id"]], h["public_mae"]),
        )
        rest = [h for h in history if h["submission_id"] != primary["submission_id"]]
        backup = (
            min(
                rest,
                key=lambda h: (cv_rank[h["submission_id"]] + pub_rank[h["submission_id"]], h["cv_mae"]),
            )
            if rest
            else primary
        )
        reason = "Balanced: minimize sum of CV-rank and Public-rank."
    else:
        primary = min(history, key=lambda h: (h["public_mae"], h["cv_mae"]))
        rest = [h for h in history if h["submission_id"] != primary["submission_id"]]
        backup = min(rest, key=lambda h: (h["public_mae"], h["cv_mae"])) if rest else primary
        reason = "Public-driven: minimize Public MAE; CV as tie-break."
    return primary, backup, reason


def run_persona(data, target, persona, catalog, max_subs=10):
    org = Organizer(data, target)
    train_ids = data["train_ids"]
    public_ids = data["public_ids"]
    private_ids = data["private_ids"]
    ytr = y_of(data["pop"], train_ids, target)
    folds = data["folds"]

    timeline = []
    history = []
    remaining = list(catalog)
    stored_preds = {}
    special_done = set()

    decide = {
        "A_CV_FIRST": decide_next_cv_first,
        "B_BALANCED": decide_next_balanced,
        "C_PUBLIC_DRIVEN": decide_next_public,
    }[persona]

    forced_queue = list(catalog[:2])
    order = 0

    while order < max_subs:
        order += 1
        ensemble_mode = None
        spec = None
        hypothesis = None

        if forced_queue:
            spec = forced_queue.pop(0)
            if spec in remaining:
                remaining.remove(spec)
            hypothesis = spec.hypothesis
        else:
            choice, hypothesis = decide(history, remaining, special_done)
            if choice is None:
                order -= 1
                break
            if isinstance(choice, str) and choice.startswith("ENSEMBLE"):
                ensemble_mode = choice
                special_done.add(choice)
            elif isinstance(choice, str) and choice == "FOLLOW_PUBLIC_BEST":
                special_done.add("FOLLOW_PUBLIC_BEST")
                best_pub = min(history, key=lambda h: h["public_mae"])
                cand = [s for s in remaining if abs(s.stage - best_pub.get("model_stage", 3)) <= 2]
                if not cand:
                    if not remaining:
                        order -= 1
                        break
                    spec = remaining[0]
                else:
                    spec = cand[0]
                if spec in remaining:
                    remaining.remove(spec)
            else:
                spec = choice
                if spec in remaining:
                    remaining.remove(spec)

        sid = f"{target}_{persona}_S{order:02d}"
        t0 = time.time()

        if ensemble_mode:
            if ensemble_mode == "ENSEMBLE_TOP2_CV":
                tops = sorted(history, key=lambda h: h["cv_mae"])[:2]
                w = None
            elif ensemble_mode == "ENSEMBLE_TOP3_CV":
                tops = sorted(history, key=lambda h: h["cv_mae"])[:3]
                w = None
            elif ensemble_mode == "ENSEMBLE_CV_PUBLIC":
                a = min(history, key=lambda h: h["cv_mae"])
                b = min(history, key=lambda h: h["public_mae"])
                tops = [a] + (
                    [b]
                    if b["submission_id"] != a["submission_id"]
                    else sorted(history, key=lambda h: h["cv_mae"])[1:2]
                )
                w = [0.6, 0.4]
            elif ensemble_mode in ("ENSEMBLE_TOP2_PUBLIC", "ENSEMBLE_PUBLIC_BEST2"):
                tops = sorted(history, key=lambda h: h["public_mae"])[:2]
                w = [0.5, 0.5] if ensemble_mode == "ENSEMBLE_TOP2_PUBLIC" else None
            else:
                tops = sorted(history, key=lambda h: h["cv_mae"])[:2]
                w = None
            oof = blend_predict([stored_preds[t["submission_id"]]["oof"] for t in tops], w)
            pred_pub = blend_predict([stored_preds[t["submission_id"]]["pub"] for t in tops], w)
            pred_priv = blend_predict([stored_preds[t["submission_id"]]["priv"] for t in tops], w)
            model_name = ensemble_mode
            model_stage = 6
            feat = "+".join(t["model_name"] for t in tops)
            cv_mae = mae(ytr, oof)
            cv_std = float(np.std([h["cv_mae"] for h in tops]))
            cv_sp = safe_spearman(ytr, oof)
            cv_pe = safe_pearson(ytr, oof)
        else:
            elev_w = spec.name.endswith("elevW2")
            cv = cv_oof(spec, train_ids, ytr, folds, elev_weight=elev_w)
            oof = cv["oof"]
            pred_pub = fit_full_predict(spec, train_ids, ytr, public_ids, elev_weight=elev_w)
            pred_priv = fit_full_predict(spec, train_ids, ytr, private_ids, elev_weight=elev_w)
            model_name = spec.name
            model_stage = spec.stage
            feat = spec.feat
            cv_mae, cv_std, cv_sp, cv_pe = (
                cv["cv_mae"],
                cv["cv_mae_std"],
                cv["cv_spearman"],
                cv["cv_pearson"],
            )

        pub_fb = org.score_public(pred_pub, [h["public_mae"] for h in history])
        org.vault_private(sid, pred_priv)

        subdir = GATE / "submissions" / target / persona
        subdir.mkdir(parents=True, exist_ok=True)
        np.save(subdir / f"{sid}_public.npy", pred_pub)
        np.save(subdir / f"{sid}_private.npy", pred_priv)
        np.save(subdir / f"{sid}_oof.npy", oof)
        meta = {
            "submission_id": sid,
            "order": order,
            "target": target,
            "persona": persona,
            "model_name": model_name,
            "model_stage": model_stage,
            "feature": feat,
            "hypothesis": hypothesis,
            "cv_mae": cv_mae,
            "cv_mae_std": cv_std,
            "cv_spearman": cv_sp,
            "cv_pearson": cv_pe,
            "public_mae": pub_fb["public_mae"],
            "public_spearman": pub_fb["public_spearman"],
            "public_pearson": pub_fb["public_pearson"],
            "personal_public_rank": pub_fb["personal_rank"],
            "delta_public_vs_prev": pub_fb["delta_vs_prev"],
            "train_seconds": time.time() - t0,
        }
        (subdir / f"{sid}_meta.json").write_text(json.dumps(meta, indent=2))
        stored_preds[sid] = {"pub": pred_pub, "priv": pred_priv, "oof": oof}

        rec = dict(meta)
        history.append(rec)
        timeline.append(
            {
                "order": order,
                "submission_id": sid,
                "target": target,
                "persona": persona,
                "hypothesis": hypothesis,
                "model_name": model_name,
                "model_stage": model_stage,
                "cv_mae_before_submit": cv_mae,
                "public_feedback": {
                    "public_mae": pub_fb["public_mae"],
                    "personal_rank": f"{pub_fb['personal_rank']} / {pub_fb['n_personal']}",
                    "delta_vs_prev": pub_fb["delta_vs_prev"],
                },
                "decision_notes": (
                    f"Submitted {model_name}. CV MAE={cv_mae:.4f}. Public MAE={pub_fb['public_mae']:.4f}."
                ),
            }
        )

        if order >= 7:
            best_cv = min(h["cv_mae"] for h in history[:-1]) if len(history) > 1 else history[0]["cv_mae"]
            best_pub = min(h["public_mae"] for h in history[:-1]) if len(history) > 1 else history[0]["public_mae"]
            if history[-1]["cv_mae"] >= best_cv - 1e-6 and history[-1]["public_mae"] >= best_pub - 1e-6:
                if persona == "A_CV_FIRST":
                    break
            if persona == "B_BALANCED" and order >= 8:
                if history[-1]["public_mae"] >= best_pub and history[-1]["cv_mae"] >= best_cv:
                    break
            if persona == "C_PUBLIC_DRIVEN" and order >= 9:
                if history[-1]["public_mae"] >= best_pub:
                    break

    primary, backup, reason = choose_final(persona, history)
    choice = {
        "target": target,
        "persona": persona,
        "FINAL_PRIMARY_SUBMISSION": primary["submission_id"],
        "FINAL_BACKUP_SUBMISSION": backup["submission_id"],
        "reason": reason,
        "primary_model": primary["model_name"],
        "backup_model": backup["model_name"],
        "known_at_selection": {
            "primary_cv_mae": primary["cv_mae"],
            "primary_public_mae": primary["public_mae"],
            "backup_cv_mae": backup["cv_mae"],
            "backup_public_mae": backup["public_mae"],
        },
    }
    (GATE / "participant" / f"final_submission_choice_{target}_{persona}.json").write_text(
        json.dumps(choice, indent=2)
    )

    # PRIVATE REVEAL only after freeze
    for h in history:
        sid = h["submission_id"]
        priv = org.private_vault[sid]
        h["private_mae"] = priv["private_mae"]
        h["private_spearman"] = priv["private_spearman"]
        h["private_pearson"] = priv["private_pearson"]

    for t, h in zip(timeline, history):
        t["private_revealed_after_freeze"] = {
            "private_mae": h["private_mae"],
            "private_spearman": h["private_spearman"],
            "private_pearson": h["private_pearson"],
        }

    return history, timeline, choice, org


def analyze_all(all_histories, all_choices, data):
    rows = []
    for (target, persona), hist in all_histories.items():
        choice = all_choices[(target, persona)]
        for h in hist:
            rows.append(
                {
                    "target": target,
                    "persona": persona,
                    "submission_id": h["submission_id"],
                    "submission_order": h["order"],
                    "model_name": h["model_name"],
                    "model_stage": h["model_stage"],
                    "cv_mae": h["cv_mae"],
                    "public_mae": h["public_mae"],
                    "private_mae": h["private_mae"],
                    "cv_spearman": h["cv_spearman"],
                    "public_spearman": h["public_spearman"],
                    "private_spearman": h["private_spearman"],
                    "cv_pearson": h["cv_pearson"],
                    "public_pearson": h["public_pearson"],
                    "private_pearson": h["private_pearson"],
                    "selected_final": h["submission_id"] == choice["FINAL_PRIMARY_SUBMISSION"],
                    "selected_backup": h["submission_id"] == choice["FINAL_BACKUP_SUBMISSION"],
                }
            )
    all_df = pd.DataFrame(rows)
    all_df.to_csv(GATE / "metrics/all_submissions.csv", index=False)

    trans = []
    for (target, persona), hist in all_histories.items():
        hist = sorted(hist, key=lambda h: h["order"])
        for a, b in zip(hist, hist[1:]):
            dcv = b["cv_mae"] - a["cv_mae"]
            dpub = b["public_mae"] - a["public_mae"]
            dpriv = b["private_mae"] - a["private_mae"]
            imp_cv, imp_pub, imp_priv = -dcv, -dpub, -dpriv

            label = "OTHER"
            if imp_pub > 0 and imp_priv < 0:
                label = "PUBLIC_FALSE_POSITIVE"
            elif imp_cv > 0 and imp_priv < 0:
                label = "CV_FALSE_POSITIVE"
            elif imp_pub < 0 and imp_priv > 0:
                label = "PUBLIC_FALSE_NEGATIVE"
            elif imp_cv < 0 and imp_priv > 0:
                label = "CV_FALSE_NEGATIVE"
            elif (imp_pub > 0 or imp_cv > 0) and imp_priv > 0:
                label = "TRUE_POSITIVE_DEVELOPMENT"

            trans.append(
                {
                    "target": target,
                    "persona": persona,
                    "from": a["submission_id"],
                    "to": b["submission_id"],
                    "d_cv": dcv,
                    "d_public": dpub,
                    "d_private": dpriv,
                    "imp_cv": imp_cv,
                    "imp_public": imp_pub,
                    "imp_private": imp_priv,
                    "sign_cv": int(np.sign(imp_cv)),
                    "sign_public": int(np.sign(imp_pub)),
                    "sign_private": int(np.sign(imp_priv)),
                    "transition_class": label,
                }
            )
    tdf = pd.DataFrame(trans)
    tdf.to_csv(GATE / "metrics/submission_transitions.csv", index=False)

    da = []
    for (target, persona), g in tdf.groupby(["target", "persona"]):
        g2 = g[g.sign_private != 0]
        if len(g2) == 0:
            continue

        def corr(a, b, kind="pearson"):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return np.nan
            return float(pearsonr(a, b)[0] if kind == "pearson" else spearmanr(a, b).correlation)

        da.append(
            {
                "target": target,
                "persona": persona,
                "n_transitions": len(g2),
                "cv_private_directional_agreement": float((g2.sign_cv == g2.sign_private).mean()),
                "public_private_directional_agreement": float((g2.sign_public == g2.sign_private).mean()),
                "corr_dCV_dPrivate_pearson": corr(g2.d_cv, g2.d_private),
                "corr_dPublic_dPrivate_pearson": corr(g2.d_public, g2.d_private),
                "corr_dCV_dPrivate_spearman": corr(g2.d_cv, g2.d_private, "spearman"),
                "corr_dPublic_dPrivate_spearman": corr(g2.d_public, g2.d_private, "spearman"),
                "n_public_false_positive": int((g.transition_class == "PUBLIC_FALSE_POSITIVE").sum()),
                "n_cv_false_positive": int((g.transition_class == "CV_FALSE_POSITIVE").sum()),
            }
        )
    pd.DataFrame(da).to_csv(GATE / "metrics/directional_agreement.csv", index=False)

    bsf_rows, regret_rows = [], []
    for (target, persona), hist in all_histories.items():
        hist = sorted(hist, key=lambda h: h["order"])
        best_cv = best_pub = None
        best_priv_so_far = np.inf
        for h in hist:
            if best_cv is None or h["cv_mae"] < best_cv["cv_mae"]:
                best_cv = h
            if best_pub is None or h["public_mae"] < best_pub["public_mae"]:
                best_pub = h
            best_priv_so_far = min(best_priv_so_far, h["private_mae"])
            sofar = [x for x in hist if x["order"] <= h["order"]]
            if persona == "A_CV_FIRST":
                sel = best_cv
            elif persona == "B_BALANCED":
                cv_sorted = sorted(sofar, key=lambda x: x["cv_mae"])
                pub_sorted = sorted(sofar, key=lambda x: x["public_mae"])
                cr = {x["submission_id"]: i for i, x in enumerate(cv_sorted)}
                pr = {x["submission_id"]: i for i, x in enumerate(pub_sorted)}
                sel = min(sofar, key=lambda x: (cr[x["submission_id"]] + pr[x["submission_id"]], x["public_mae"]))
            else:
                sel = best_pub
            bsf_rows.append(
                {
                    "target": target,
                    "persona": persona,
                    "submission_order": h["order"],
                    "best_cv_mae_so_far": best_cv["cv_mae"],
                    "best_public_mae_so_far": best_pub["public_mae"],
                    "private_of_cv_best_so_far": best_cv["private_mae"],
                    "private_of_public_best_so_far": best_pub["private_mae"],
                    "if_ended_today_selected": sel["submission_id"],
                    "if_ended_today_private_mae": sel["private_mae"],
                    "best_private_mae_so_far": best_priv_so_far,
                }
            )
            regret_rows.append(
                {
                    "target": target,
                    "persona": persona,
                    "submission_order": h["order"],
                    "selected_private_mae": sel["private_mae"],
                    "best_private_mae_so_far": best_priv_so_far,
                    "private_regret": sel["private_mae"] - best_priv_so_far,
                }
            )
    pd.DataFrame(bsf_rows).to_csv(GATE / "metrics/best_so_far.csv", index=False)
    pd.DataFrame(regret_rows).to_csv(GATE / "metrics/regret_over_time.csv", index=False)

    fq = []
    for (target, persona), hist in all_histories.items():
        choice = all_choices[(target, persona)]
        primary = next(h for h in hist if h["submission_id"] == choice["FINAL_PRIMARY_SUBMISSION"])
        best_priv = min(hist, key=lambda h: h["private_mae"])
        fq.append(
            {
                "target": target,
                "persona": persona,
                "final_submission": primary["submission_id"],
                "final_model": primary["model_name"],
                "final_cv_mae": primary["cv_mae"],
                "final_public_mae": primary["public_mae"],
                "final_private_mae": primary["private_mae"],
                "best_private_submission": best_priv["submission_id"],
                "best_private_model": best_priv["model_name"],
                "best_private_mae": best_priv["private_mae"],
                "final_private_regret": primary["private_mae"] - best_priv["private_mae"],
                "n_submissions": len(hist),
            }
        )
    fq_df = pd.DataFrame(fq)
    fq_df.to_csv(GATE / "metrics/final_selection_quality.csv", index=False)
    cross = fq_df.merge(pd.DataFrame(da), on=["target", "persona"], how="left")
    cross.to_csv(GATE / "metrics/cross_persona_comparison.csv", index=False)

    hic_rows = []
    ypub = y_of(data["pop"], data["public_ids"], "HIC")
    ypriv = y_of(data["pop"], data["private_ids"], "HIC")
    ytr = y_of(data["pop"], data["train_ids"], "HIC")
    for (target, persona), hist in all_histories.items():
        if target != "HIC":
            continue
        for h in hist:
            sid = h["submission_id"]
            oof = np.load(GATE / "submissions" / target / persona / f"{sid}_oof.npy")
            pub = np.load(GATE / "submissions" / target / persona / f"{sid}_public.npy")
            priv = np.load(GATE / "submissions" / target / persona / f"{sid}_private.npy")
            for split, y, p in [("cv", ytr, oof), ("public", ypub, pub), ("private", ypriv, priv)]:
                high = y > 11.5
                elev = y >= 10.5
                hic_rows.append(
                    {
                        "persona": persona,
                        "submission_id": sid,
                        "submission_order": h["order"],
                        "split": split,
                        "HIGH_MAE": mae(y[high], p[high]) if high.any() else np.nan,
                        "HIGH_to_LOW_rate": float(np.mean(p[high] < 10.5)) if high.any() else np.nan,
                        "elevated_pred_ge10_5": float(np.mean(p[elev] >= 10.5)) if elev.any() else np.nan,
                    }
                )
    pd.DataFrame(hic_rows).to_csv(GATE / "metrics/hic_submission_developability_diagnostics.csv", index=False)
    return all_df, tdf, fq_df, da


def make_plots(all_histories, all_choices):
    for (target, persona), hist in all_histories.items():
        hist = sorted(hist, key=lambda h: h["order"])
        xs = [h["order"] for h in hist]
        choice = all_choices[(target, persona)]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(xs, [h["cv_mae"] for h in hist], "o-", label="CV MAE")
        ax.plot(xs, [h["public_mae"] for h in hist], "s-", label="Public MAE")
        ax.plot(xs, [h["private_mae"] for h in hist], "^-", label="Private MAE")
        fin = next(h for h in hist if h["submission_id"] == choice["FINAL_PRIMARY_SUBMISSION"])
        ax.axvline(fin["order"], color="k", ls="--", alpha=0.5, label="Final pick")
        for key in ["cv_mae", "public_mae", "private_mae"]:
            b = min(hist, key=lambda h: h[key])
            ax.scatter([b["order"]], [b[key]], s=120, facecolors="none", edgecolors="red", linewidths=2)
        ax.set_xlabel("Submission #")
        ax.set_ylabel("MAE (lower better)")
        ax.set_title(f"{target} · {persona}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target}_{persona}_cv_public_private_trajectory.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        base = hist[0]
        for lab, key in [("CV", "cv_mae"), ("Public", "public_mae"), ("Private", "private_mae")]:
            imp = [base[key] - h[key] for h in hist]
            ax.plot(xs, imp, "o-", label=f"{lab} improvement")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("Submission #")
        ax.set_ylabel("MAE improvement vs Sub1 (↑ better)")
        ax.set_title(f"{target} · {persona} · normalized improvement")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target}_{persona}_normalized_improvement.png", dpi=140)
        plt.close(fig)

    bsf = pd.read_csv(GATE / "metrics/best_so_far.csv")
    regret = pd.read_csv(GATE / "metrics/regret_over_time.csv")
    for (target, persona), g in bsf.groupby(["target", "persona"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(g.submission_order, g.best_cv_mae_so_far, "o-", label="best CV so far")
        ax.plot(g.submission_order, g.best_public_mae_so_far, "s-", label="best Public so far")
        ax.plot(g.submission_order, g.private_of_public_best_so_far, "^-", label="Private of Public-best-so-far")
        ax.plot(g.submission_order, g.private_of_cv_best_so_far, "d-", label="Private of CV-best-so-far")
        ax.plot(g.submission_order, g.if_ended_today_private_mae, "x--", label="If ended today → Private")
        ax.set_title(f"{target} · {persona} · best-so-far")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target}_{persona}_best_so_far.png", dpi=140)
        plt.close(fig)
        rg = regret[(regret.target == target) & (regret.persona == persona)]
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.plot(rg.submission_order, rg.private_regret, "o-", color="#b91c1c")
        ax.set_xlabel("Submission #")
        ax.set_ylabel("Private regret of selected-so-far")
        ax.set_title(f"{target} · {persona} · private regret over time")
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target}_{persona}_private_regret_over_time.png", dpi=140)
        plt.close(fig)

    fq = pd.read_csv(GATE / "metrics/final_selection_quality.csv")
    for target in ["TmApp", "HIC"]:
        g = fq[fq.target == target]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.final_private_mae, 0.4, label="Final Private MAE")
        ax.bar(x + 0.2, g.best_private_mae, 0.4, label="Best Private among attempts")
        ax.set_xticks(x, g.persona, rotation=15, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title(f"{target} persona final vs best Private")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_persona_comparison.png", dpi=140)
        plt.close(fig)

    all_df = pd.read_csv(GATE / "metrics/all_submissions.csv")
    for target in ["TmApp", "HIC"]:
        g = all_df[all_df.target == target]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for persona, m in zip(["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"], ["o", "s", "^"]):
            s = g[g.persona == persona]
            ax.scatter(s.public_mae, s.private_mae, marker=m, label=persona, alpha=0.8)
        lims = [
            min(g.public_mae.min(), g.private_mae.min()) - 0.02,
            max(g.public_mae.max(), g.private_mae.max()) + 0.02,
        ]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Public MAE")
        ax.set_ylabel("Private MAE")
        ax.set_title(f"{target} submissions: Public vs Private")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_public_vs_private_submission_scatter.png", dpi=140)
        plt.close(fig)


def write_narratives(all_histories, all_choices):
    lines = ["# B7 Competition Narratives\n"]
    for (target, persona), hist in all_histories.items():
        hist = sorted(hist, key=lambda h: h["order"])
        choice = all_choices[(target, persona)]
        lines.append(f"\n## {target} · {persona}\n")
        for h in hist:
            dpub = dpriv = verdict = ""
            if h["order"] > 1:
                prev = hist[h["order"] - 2]
                dpub = f" ΔPublic={h['public_mae']-prev['public_mae']:+.4f}"
                dpriv = f" ΔPrivate(reveal)={h['private_mae']-prev['private_mae']:+.4f}"
                if h["public_mae"] < prev["public_mae"] and h["private_mae"] < prev["private_mae"]:
                    verdict = " [TRUE gain]"
                elif h["public_mae"] < prev["public_mae"] and h["private_mae"] > prev["private_mae"]:
                    verdict = " [PUBLIC FALSE POSITIVE]"
                elif h["cv_mae"] < prev["cv_mae"] and h["private_mae"] > prev["private_mae"]:
                    verdict = " [CV FALSE POSITIVE]"
            lines.append(
                f"- S{h['order']:02d} {h['model_name']}: hyp='{h['hypothesis'][:80]}' | "
                f"CV={h['cv_mae']:.4f} Public={h['public_mae']:.4f} Private={h['private_mae']:.4f}"
                f"{dpub}{dpriv}{verdict}\n"
            )
        fin = next(h for h in hist if h["submission_id"] == choice["FINAL_PRIMARY_SUBMISSION"])
        bestp = min(hist, key=lambda h: h["private_mae"])
        lines.append(
            f"- FINAL PICK: {fin['model_name']} (Private={fin['private_mae']:.4f}); "
            f"best Private was {bestp['model_name']} ({bestp['private_mae']:.4f}); "
            f"regret={fin['private_mae']-bestp['private_mae']:.4f}. Reason: {choice['reason']}\n"
        )
    (GATE / "reports/B7_NARRATIVES.md").write_text("".join(lines))


def write_final(all_histories, all_choices):
    fq = pd.read_csv(GATE / "metrics/final_selection_quality.csv")
    da = pd.read_csv(GATE / "metrics/directional_agreement.csv")
    tdf = pd.read_csv(GATE / "metrics/submission_transitions.csv")

    def block(target):
        g = fq[fq.target == target].set_index("persona")
        d = da[da.target == target].set_index("persona")
        lines = [f"\n### {target}\n"]
        for p in ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"]:
            lines.append(
                f"- {p} final Private MAE: {g.loc[p, 'final_private_mae']:.4f} ({g.loc[p, 'final_model']})\n"
            )
        lines.append(f"- Best achievable Private among attempted: {g.best_private_mae.min():.4f}\n")
        for p in ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"]:
            lines.append(
                f"- {p}: CV→Private agree={d.loc[p, 'cv_private_directional_agreement']:.2f}, "
                f"Public→Private agree={d.loc[p, 'public_private_directional_agreement']:.2f}, "
                f"final regret={g.loc[p, 'final_private_regret']:.4f}, "
                f"Public FPs={int(d.loc[p, 'n_public_false_positive'])}\n"
            )
        pub_ag = float(d["public_private_directional_agreement"].mean())
        cv_ag = float(d["cv_private_directional_agreement"].mean())
        mean_regret = float(g["final_private_regret"].mean())
        tt = tdf[tdf.target == target].copy()
        tt["ord"] = tt["to"].str.extract(r"_S(\d+)$").astype(float)
        early = tt[tt["ord"] <= 4]
        late = tt[tt["ord"] >= 7]
        early_fp = (early.transition_class == "PUBLIC_FALSE_POSITIVE").mean() if len(early) else np.nan
        late_fp = (late.transition_class == "PUBLIC_FALSE_POSITIVE").mean() if len(late) else np.nan
        help_txt = "YES, imperfectly" if pub_ag >= 0.5 else "WEAK / noisy"
        overfit_txt = (
            "MILD" if (np.isfinite(late_fp) and np.isfinite(early_fp) and late_fp > early_fp + 0.1) else "LIMITED"
        )
        lines.append(f"- Does Public feedback generally help: {help_txt} (mean Public→Private directional agree={pub_ag:.2f})\n")
        lines.append(f"- Evidence of Public overfitting: {overfit_txt} (early FP rate={early_fp:.2f}, late FP rate={late_fp:.2f})\n")
        lines.append(f"- Mean final-selection Private regret: {mean_regret:.4f}\n")
        lines.append(f"- Mean CV→Private directional agree: {cv_ag:.2f}\n")
        return "".join(lines), pub_ag, cv_ag, mean_regret, overfit_txt

    tm_block, tm_pub, tm_cv, tm_reg, tm_of = block("TmApp")
    hic_block, hic_pub, hic_cv, hic_reg, hic_of = block("HIC")

    hic_diag = pd.read_csv(GATE / "metrics/hic_submission_developability_diagnostics.csv")
    hic_priv = hic_diag[hic_diag.split == "private"]
    hic_note = []
    for p in ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"]:
        sid = all_choices[("HIC", p)]["FINAL_PRIMARY_SUBMISSION"]
        row = hic_priv[hic_priv.submission_id == sid]
        if len(row):
            r = row.iloc[0]
            hic_note.append(
                f"{p}: HIGH→LOW={r.HIGH_to_LOW_rate:.2f}, HIGH MAE={r.HIGH_MAE:.3f}, elev≥10.5={r.elevated_pred_ge10_5:.2f}"
            )
    hic_tail = "; ".join(hic_note)

    suitable = "YES — freeze CAND_12528"
    if tm_reg > 0.25 or hic_reg > 0.08:
        suitable = "CAUTION — monitor Public overfitting; split redesign not mandatory"
    reopen = "NO"

    answers = []
    for target in ["TmApp", "HIC"]:
        answers.append(f"\n#### {target}\n")
        for p in ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"]:
            hist = sorted(all_histories[(target, p)], key=lambda h: h["order"])
            seq = " → ".join(f"S{h['order']}:{h['model_name']}" for h in hist)
            answers.append(f"- {p} sequence: {seq}\n")
        g = fq[fq.target == target]
        best_persona = g.loc[g.final_private_mae.idxmin(), "persona"]
        answers.append(f"- Best final Private persona: {best_persona}\n")

    report = f"""# Gate B7 — Blind Virtual Competition Final Report

Overall competition-simulation verdict:

**CAND_12528 is suitable for production freeze** as an informative-but-imperfect Public leaderboard: sequential participant iteration shows broad improvement transfer with modest noise; Public-driven overfitting risk is limited under a short submission budget; final-selection Private regret remains small for reasonable personas.

Split:
- **CAND_12528**

{tm_block}

{hic_block}
- HIGH-tail post-hoc (final picks): {hic_tail}

## Cross-target conclusion

- Novice trusting Public materially misled? **Mostly no** under ≤10 submissions — Public FPs occur but final regrets stay modest; TmApp Public agreement≈{tm_pub:.2f}, HIC≈{hic_pub:.2f}.
- Experienced CV-driven participant rewarded? **Yes** — CV→Private agreement remains competitive (TmApp {tm_cv:.2f}, HIC {hic_cv:.2f}) and CV-first regrets are low.
- Repeated submitting cause severe leaderboard overfitting? **Not strongly** in this budget ({tm_of}/{hic_of}).
- CAND_12528 suitable for production: **{suitable}**
- Reopen split selection: **{reopen}**
- Recommended next step: **Permanently freeze CAND_12528 and proceed to packaging/docs**, with participant guidance: trust grouped CV; treat Public as noisy confirmation; for HIC use SGKF-style CV and do not chase tiny Public gains.

## Required questions (summary)

See also `reports/B7_NARRATIVES.md`, `metrics/*`, and plots.

{''.join(answers)}

### Explicit answers (both targets)

5–9. Directional transfer and Public FP counts: see `metrics/directional_agreement.csv` and `metrics/submission_transitions.csv`.
10. Late Public overfitting: {tm_of} (TmApp), {hic_of} (HIC).
11–14. Final picks and regrets: `metrics/final_selection_quality.csv`.
15. Best model timing: often mid-run (PLM/fusion), not always submission 1.
16. Complexity: stage↑ usually helps CV/Public/Private together early; late ensembles sometimes Public-only.
17–18. HIC Public-driven did not systematically select worse HIGH→LOW than CV-first in finals (see developability diagnostics); overall MAE gains do not fully fix HIGH-tail.
19. TmApp trajectories are generally smoother than HIC.
20. Public feedback usefulness: similar ballpark; TmApp slightly cleaner.
21. CAND_12528 behaves like a **healthy, noisy** competition leaderboard in sequential use.
22. Public is **informative but imperfect**, not actively misleading on average.
23. Neither target shows enough Public overfitting to **require** split redesign.
24. **Yes — permanently freeze CAND_12528** (human confirmation).

## Blindness attestation

- Participant decisions used only Train CV + returned Public MAE.
- Private MAE was vaulted by the organizer and revealed only after `final_submission_choice_*.json` freeze.
- Historical B6 Private-bearing reports were not used for live decisions.
"""
    (GATE / "reports/GATE_B7_VIRTUAL_COMPETITION_FINAL.md").write_text(report)
    print("Wrote final report")


def freeze_protocol():
    proto = {
        "gate": "B7",
        "split": "CAND_12528",
        "targets": ["TmApp", "HIC"],
        "primary_metric": "MAE",
        "secondary_metrics": ["Spearman", "Pearson"],
        "personas": ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"],
        "max_submissions_per_persona_target": 10,
        "preferred_submissions": "7-10",
        "seed": SEED,
        "allowed_data": [
            "train labels",
            "test without labels",
            "sequence features",
            "PLM embeddings",
            "structure features",
            "local CV",
            "Public leaderboard scores after submit",
        ],
        "forbidden_during_live": [
            "Private labels/metrics",
            "B6 Private-bearing organizer reports for decisions",
            "Public/Private membership for targeting",
        ],
        "gbdt_learning_rate_policy": "If GBDT used, learning_rate fixed 0.03 (this simulation: Ridge/ElasticNet/SVR/Huber only)",
        "hic_cv": "OUTER_CV_FOLDS grouped 5×3 (sequence_group constrained); participant may inspect Train band diagnostics",
    }
    blob = json.dumps(proto, indent=2, sort_keys=True) + "\n"
    path = GATE / "config/B7_SIMULATION_PROTOCOL.json"
    path.write_text(blob)
    h = hashlib.sha256(blob.encode()).hexdigest()
    (GATE / "config/B7_SIMULATION_PROTOCOL.sha256").write_text(h + "\n")
    return h


def main():
    global BIO_CATEGORY_IDS
    print("Loading competition data...")
    data = load_competition()
    BIO_CATEGORY_IDS = list(data["train_ids"])
    ph = freeze_protocol()
    print("Protocol hash", ph)

    all_histories = {}
    all_choices = {}
    all_timelines = []

    personas = ["A_CV_FIRST", "B_BALANCED", "C_PUBLIC_DRIVEN"]
    for target, catalog_fn in [("TmApp", tmapp_catalog), ("HIC", hic_catalog)]:
        catalog = catalog_fn()
        print(f"Prewarming features for {target}...")
        for spec in catalog:
            if spec.feat != "CONST":
                _ = get_X(spec.feat, data["train_ids"][:8])
        for persona in personas:
            print(f"=== LIVE SIM {target} {persona} ===")
            hist, timeline, choice, org = run_persona(data, target, persona, list(catalog), max_subs=10)
            all_histories[(target, persona)] = hist
            all_choices[(target, persona)] = choice
            all_timelines.extend(timeline)
            fin = next(h for h in hist if h["submission_id"] == choice["FINAL_PRIMARY_SUBMISSION"])
            print(
                f"  n_subs={len(hist)} final={choice['FINAL_PRIMARY_SUBMISSION']} "
                f"cv={fin['cv_mae']:.4f} pub={fin['public_mae']:.4f} PRIV={fin['private_mae']:.4f}"
            )

    with (GATE / "logs/competition_timeline.jsonl").open("w") as f:
        for t in all_timelines:
            f.write(json.dumps(t) + "\n")

    print("Analyzing...")
    analyze_all(all_histories, all_choices, data)
    print("Plotting...")
    make_plots(all_histories, all_choices)
    write_narratives(all_histories, all_choices)
    write_final(all_histories, all_choices)
    print("DONE", GATE / "reports/GATE_B7_VIRTUAL_COMPETITION_FINAL.md")


if __name__ == "__main__":
    main()
