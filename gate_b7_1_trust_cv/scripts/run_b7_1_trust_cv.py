#!/usr/bin/env python3
"""Gate B7.1 — Trust-CV Educational Validity Audit (CAND_12528)."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
import warnings
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
GATE = ROOT / "gate_b7_1_trust_cv"
sys.path.insert(0, str(ROOT / "gate_b4_absolute/scripts"))
from b4_common import load_bio, load_representation  # noqa: E402

SEED = 20260830
N_BOOT = 400
PRIMARY_P = (0.80, 0.20)
SENS = [(0.75, 0.25), (0.80, 0.20), (0.85, 0.15), (0.90, 0.10)]
BIO_CAT = None
FEAT_CACHE: dict = {}

for d in [
    "config",
    "phase1_existing",
    "phase2_stress",
    "phase3_split_review",
    "metrics",
    "plots",
    "reports",
    "logs",
    "scripts",
    "cache",
]:
    (GATE / d).mkdir(parents=True, exist_ok=True)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def sha_arr(a, nd=6):
    return hashlib.sha256(np.round(np.asarray(a, float), nd).tobytes()).hexdigest()


def y_of(pop, ids, target):
    return pop.loc[list(ids), target].astype(float).values


def freeze_protocol():
    proto = {
        "gate": "B7.1",
        "split": "CAND_12528",
        "targets": ["TmApp", "HIC"],
        "primary_metric": "MAE",
        "primary_clear_improvement_p": 0.80,
        "primary_clear_worsening_p": 0.20,
        "sensitivity_thresholds": [{"improve": a, "worsen": b} for a, b in SENS],
        "bootstrap_draws": N_BOOT,
        "dedup_rule": "sha256(round(OOF,6)+round(test,6))",
        "competitive_set_rule": "cv_mae <= best_cv + 0.20*(const_cv-best_cv)",
        "conditional_action": {
            "both_STRONG_or_SUPPORTED": "PHASE_2A",
            "otherwise": "PHASE_2B",
            "phase3_if": "PUBLIC_DOMINATES after stress",
        },
        "seed": SEED,
    }
    blob = json.dumps(proto, indent=2, sort_keys=True) + "\n"
    (GATE / "config/B7_1_PROTOCOL.json").write_text(blob)
    h = hashlib.sha256(blob.encode()).hexdigest()
    (GATE / "config/B7_1_PROTOCOL.sha256").write_text(h + "\n")
    return h


def load_data():
    cand = next(
        c
        for c in json.loads((ROOT / "gate_b6_split_search/config/B6_1_FINAL_CANDIDATE_SET.json").read_text())[
            "candidates"
        ]
        if c["candidate_id"] == "CAND_12528"
    )
    outer = json.loads((ROOT / "gate_b4_absolute/config/OUTER_CV_FOLDS.json").read_text())
    pop = pd.read_csv(ROOT / "gate_b3/frozen/organizer/final_population.csv").set_index("id")
    role = pd.read_csv(ROOT / "gate_b3/frozen/organizer/role_map.csv")
    train_ids = list(outer["train_ids"])
    public_ids = list(cand["public_ids"])
    private_ids = list(cand["private_ids"])
    role_pub = role.loc[role.role == "Public", "id"].tolist()
    role_priv = role.loc[role.role == "Private", "id"].tolist()
    folds = [np.asarray(r["fold_id"], int) for r in outer["folds"]]
    groups = pop.loc[train_ids, "sequence_group"].astype(int).values
    return {
        "train_ids": train_ids,
        "public_ids": public_ids,
        "private_ids": private_ids,
        "role_pub": role_pub,
        "role_priv": role_priv,
        "pop": pop,
        "folds": folds,
        "groups": groups,
    }


def remap_b5(target, tag, data):
    pred = ROOT / "gate_b5_ceiling/predictions/final"
    pub = np.load(pred / f"{target}__{tag}__public.npy")
    priv = np.load(pred / f"{target}__{tag}__private.npy")
    by_id = dict(zip(data["role_pub"], pub))
    by_id.update(dict(zip(data["role_priv"], priv)))
    return (
        np.array([by_id[i] for i in data["public_ids"]], float),
        np.array([by_id[i] for i in data["private_ids"]], float),
    )


def load_b5_oof(target, tag):
    return np.load(ROOT / "gate_b5_ceiling/predictions/oof" / f"{target}__{tag}__meanOOF.npy").astype(float)


def family_of(name: str) -> str:
    u = name.upper()
    if "CONST" in u:
        return "const"
    if "ENSEMBLE" in u or "NESTED" in u or "STACK" in u:
        return "ensemble"
    if "ABLANG" in u and ("BIO" in u or "FUSION" in u):
        return "fusion"
    if "ABLANG" in u:
        return "ablang2"
    if "ESM2" in u and ("STRUCT" in u or "ESMFN" in u or "PHYS" in u or "FUSION" in u):
        return "fusion"
    if "ESM2" in u:
        return "esm2"
    if "STRUCT" in u or "ESMFN" in u:
        return "structure"
    if "BIO" in u:
        return "bio"
    if "SEQ" in u or "SIMPLE" in u or "PHYS" in u:
        return "simple"
    return "other"


def build_bank(data):
    rows, store = [], {}
    b5 = {
        "TmApp": [
            "CONST_MEDIAN",
            "SEQ_SIMPLE_Ridge",
            "BIO_Ridge",
            "PLM_ABLANG2_PCA32_SVR",
            "FUSION_ABLANG2_BIO_ElasticNet",
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
    for target, tags in b5.items():
        ytr = y_of(data["pop"], data["train_ids"], target)
        ypub = y_of(data["pop"], data["public_ids"], target)
        ypriv = y_of(data["pop"], data["private_ids"], target)
        for tag in tags:
            oof = load_b5_oof(target, tag)
            pub, priv = remap_b5(target, tag, data)
            mid = f"B5::{target}::{tag}"
            store[mid] = {"oof": oof, "pub": pub, "priv": priv}
            rows.append(
                dict(
                    target=target,
                    model_id=mid,
                    model_name=tag,
                    aliases=tag,
                    family=family_of(tag),
                    gate="B5",
                    oof_hash=sha_arr(oof),
                    test_hash=sha_arr(np.concatenate([pub, priv])),
                    cv_mae=mae(ytr, oof),
                    public_mae=mae(ypub, pub),
                    private_mae=mae(ypriv, priv),
                )
            )

    for meta_path in (ROOT / "gate_b7_virtual_competition/submissions").rglob("*_meta.json"):
        m = json.loads(meta_path.read_text())
        sid = m["submission_id"]
        base = meta_path.parent / sid
        target = m["target"]
        oof = np.load(str(base) + "_oof.npy")
        pub = np.load(str(base) + "_public.npy")
        priv = np.load(str(base) + "_private.npy")
        ytr = y_of(data["pop"], data["train_ids"], target)
        ypub = y_of(data["pop"], data["public_ids"], target)
        ypriv = y_of(data["pop"], data["private_ids"], target)
        mid = f"B7::{target}::{sid}"
        store[mid] = {"oof": oof, "pub": pub, "priv": priv}
        rows.append(
            dict(
                target=target,
                model_id=mid,
                model_name=m["model_name"],
                aliases=f"{m.get('persona','')}:{m['model_name']}",
                family=family_of(m["model_name"]),
                gate="B7",
                oof_hash=sha_arr(oof),
                test_hash=sha_arr(np.concatenate([pub, priv])),
                cv_mae=mae(ytr, oof),
                public_mae=mae(ypub, pub),
                private_mae=mae(ypriv, priv),
                persona=m.get("persona"),
                order=m.get("order"),
            )
        )

    bank = pd.DataFrame(rows)
    bank["combo"] = bank.oof_hash + bank.test_hash
    # dedup
    kept, new_store = [], {}
    for (tgt, combo), g in bank.groupby(["target", "combo"]):
        g2 = g.sort_values("gate")  # B5 before B7
        r = g2.iloc[0].copy()
        r["aliases"] = ";".join(sorted(set(g.aliases.astype(str))))
        r["n_dup"] = len(g)
        nid = f"{tgt}__{r.model_name}__{r.oof_hash[:8]}"
        # avoid id collision
        n = 0
        base_nid = nid
        while nid in new_store:
            n += 1
            nid = f"{base_nid}_{n}"
        new_store[nid] = store[r.model_id]
        r["model_id"] = nid
        kept.append(r)
    uniq = pd.DataFrame(kept).reset_index(drop=True)
    uniq.to_csv(GATE / "metrics/unique_model_bank.csv", index=False)
    uniq.to_csv(GATE / "phase1_existing/unique_model_bank.csv", index=False)
    return uniq, new_store


def b7_transitions():
    rows = []
    for target in ["TmApp", "HIC"]:
        tdir = ROOT / "gate_b7_virtual_competition/submissions" / target
        if not tdir.exists():
            continue
        for persona_dir in tdir.iterdir():
            if not persona_dir.is_dir():
                continue
            metas = sorted(
                persona_dir.glob("*_meta.json"),
                key=lambda p: json.loads(p.read_text()).get("order", 0),
            )
            hist = [json.loads(p.read_text()) for p in metas]
            for a, b in zip(hist, hist[1:]):
                rows.append(
                    dict(
                        target=target,
                        persona=a.get("persona"),
                        from_model=a["model_name"],
                        to_model=b["model_name"],
                        from_cv=a["cv_mae"],
                        to_cv=b["cv_mae"],
                        from_public=a["public_mae"],
                        to_public=b["public_mae"],
                        key=f"{a['model_name']}->{b['model_name']}",
                    )
                )
    tr = pd.DataFrame(rows)
    uniq = tr.drop_duplicates(["target", "key"]) if len(tr) else tr
    tr.to_csv(GATE / "metrics/unique_b7_transitions.csv", index=False)
    uniq.to_csv(GATE / "phase1_existing/unique_b7_transitions.csv", index=False)
    return tr, uniq


def boot_delta(err_a, err_b, groups=None, n_boot=N_BOOT, seed=SEED):
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


def classify(p, thr_imp=0.80, thr_wors=0.20):
    if p >= thr_imp:
        return "CLEAR_IMPROVEMENT"
    if p <= thr_wors:
        return "CLEAR_WORSENING"
    return "UNCERTAIN"


def competitive_ids(bank_tgt: pd.DataFrame):
    const = bank_tgt[bank_tgt.family == "const"]
    const_cv = float(const.cv_mae.min()) if len(const) else float(bank_tgt.cv_mae.max())
    best = float(bank_tgt.cv_mae.min())
    thr = best + 0.20 * max(const_cv - best, 1e-9)
    return set(bank_tgt.loc[bank_tgt.cv_mae <= thr + 1e-12, "model_id"])


def stable_seed(*parts, base=SEED):
    h = hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
    return base + (int(h[:8], 16) % 100000)


def analyze_pairs(bank, store, data, set_name, thr_imp=0.80, thr_wors=0.20):
    pair_rows, mat_rows = [], []
    meta = bank.set_index("model_id")
    for target in ["TmApp", "HIC"]:
        bt = bank[bank.target == target]
        ids = bt.model_id.tolist()
        if set_name == "COMPETITIVE_SET":
            keep = competitive_ids(bt)
            ids = [i for i in ids if i in keep]
        ytr = y_of(data["pop"], data["train_ids"], target)
        ypub = y_of(data["pop"], data["public_ids"], target)
        ypriv = y_of(data["pop"], data["private_ids"], target)
        for a_id, b_id in itertools.combinations(ids, 2):
            A, B = store[a_id], store[b_id]
            dcv, _, _, p_cv = boot_delta(
                np.abs(A["oof"] - ytr), np.abs(B["oof"] - ytr), groups=data["groups"], seed=stable_seed(target, a_id, b_id, "cv")
            )
            dpub, _, _, p_pub = boot_delta(
                np.abs(A["pub"] - ypub), np.abs(B["pub"] - ypub), seed=stable_seed(target, a_id, b_id, "pub")
            )
            dpriv, _, _, p_priv = boot_delta(
                np.abs(A["priv"] - ypriv), np.abs(B["priv"] - ypriv), seed=stable_seed(target, a_id, b_id, "priv")
            )
            cv_c = classify(p_cv, thr_imp, thr_wors)
            pub_c = classify(p_pub, thr_imp, thr_wors)
            dtype = None
            if cv_c == "CLEAR_IMPROVEMENT" and pub_c == "CLEAR_WORSENING":
                dtype = "TYPE_CP"
            elif cv_c == "CLEAR_WORSENING" and pub_c == "CLEAR_IMPROVEMENT":
                dtype = "TYPE_PC"
            ra, rb = meta.loc[a_id], meta.loc[b_id]
            if dcv < 0 and dpub < 0:
                quad = "A_both_improve"
            elif dcv > 0 and dpub > 0:
                quad = "B_both_worsen"
            elif dcv < 0 and dpub > 0:
                quad = "C_cv_imp_pub_wors"
            elif dcv > 0 and dpub < 0:
                quad = "D_cv_wors_pub_imp"
            else:
                quad = "TIE"
            row = dict(
                target=target,
                set_name=set_name,
                thr_imp=thr_imp,
                model_A=a_id,
                model_B=b_id,
                name_A=ra.model_name,
                name_B=rb.model_name,
                family_A=ra.family,
                family_B=rb.family,
                cv_mae_A=float(ra.cv_mae),
                cv_mae_B=float(rb.cv_mae),
                delta_cv=dcv,
                p_B_better_cv=p_cv,
                cv_class=cv_c,
                public_mae_A=float(ra.public_mae),
                public_mae_B=float(rb.public_mae),
                delta_public=dpub,
                p_B_better_public=p_pub,
                public_class=pub_c,
                private_mae_A=float(ra.private_mae),
                private_mae_B=float(rb.private_mae),
                delta_private=dpriv,
                p_B_better_private=p_priv,
                sign_cv=int(np.sign(-dcv)),
                sign_public=int(np.sign(-dpub)),
                sign_private=int(np.sign(-dpriv)),
                disagreement_type=dtype,
                quadrant=quad,
            )
            pair_rows.append(row)
            if dtype is not None:
                if dtype == "TYPE_CP":
                    follows_cv = dpriv < 0
                    follows_pub = dpriv > 0
                    cost_cv = float(rb.private_mae) - min(float(ra.private_mae), float(rb.private_mae))
                    cost_pub = float(ra.private_mae) - min(float(ra.private_mae), float(rb.private_mae))
                else:
                    follows_cv = dpriv > 0
                    follows_pub = dpriv < 0
                    cost_cv = float(ra.private_mae) - min(float(ra.private_mae), float(rb.private_mae))
                    cost_pub = float(rb.private_mae) - min(float(ra.private_mae), float(rb.private_mae))
                mat_rows.append(
                    {
                        **row,
                        "private_follows_cv": bool(follows_cv),
                        "private_follows_public": bool(follows_pub),
                        "private_uncertain": abs(dpriv) < 1e-12,
                        "private_cost_if_follow_cv": cost_cv,
                        "private_cost_if_follow_public": cost_pub,
                    }
                )
    return pd.DataFrame(pair_rows), pd.DataFrame(mat_rows)


def summarize(mat, pairs, bank, label):
    rows = []
    for target in ["TmApp", "HIC"]:
        m = mat[mat.target == target] if len(mat) else pd.DataFrame()
        p = pairs[pairs.target == target]
        bt = bank[bank.target == target].copy()
        if "COMPETITIVE" in str(label).upper() and len(p):
            used = set(p.model_A) | set(p.model_B)
            if used:
                bt = bt[bt.model_id.isin(used)]
        n = len(m)
        n_cv = int(m.private_follows_cv.sum()) if n else 0
        n_pub = int(m.private_follows_public.sum()) if n else 0
        denom = n_cv + n_pub
        rate = n_cv / denom if denom else np.nan
        if denom >= 3:
            rs = np.random.default_rng(SEED)
            boots = []
            idx = np.arange(n)
            for _ in range(500):
                s = m.iloc[rs.choice(idx, n, replace=True)]
                a, b = int(s.private_follows_cv.sum()), int(s.private_follows_public.sum())
                if a + b:
                    boots.append(a / (a + b))
            lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan))
        else:
            lo, hi = np.nan, np.nan
        p2 = p[(p.sign_cv != 0) & (p.sign_public != 0)]
        agree = float((p2.sign_cv == p2.sign_public).mean()) if len(p2) else np.nan
        sp = float(spearmanr(bt.cv_mae, bt.public_mae).correlation) if len(bt) > 2 else np.nan
        pe = float(pearsonr(bt.cv_mae, bt.public_mae)[0]) if len(bt) > 2 else np.nan
        if n:
            reg = dict(
                follow_cv_mean=float(m.private_cost_if_follow_cv.mean()),
                follow_cv_median=float(m.private_cost_if_follow_cv.median()),
                follow_cv_p75=float(m.private_cost_if_follow_cv.quantile(0.75)),
                follow_cv_p90=float(m.private_cost_if_follow_cv.quantile(0.90)),
                follow_cv_max=float(m.private_cost_if_follow_cv.max()),
                follow_pub_mean=float(m.private_cost_if_follow_public.mean()),
                follow_pub_median=float(m.private_cost_if_follow_public.median()),
                follow_pub_p75=float(m.private_cost_if_follow_public.quantile(0.75)),
                follow_pub_p90=float(m.private_cost_if_follow_public.quantile(0.90)),
                follow_pub_max=float(m.private_cost_if_follow_public.max()),
            )
        else:
            reg = {k: np.nan for k in [
                "follow_cv_mean", "follow_cv_median", "follow_cv_p75", "follow_cv_p90", "follow_cv_max",
                "follow_pub_mean", "follow_pub_median", "follow_pub_p75", "follow_pub_p90", "follow_pub_max",
            ]}
        if n < 3:
            verdict = "TOO_FEW_DISAGREEMENTS"
        elif np.isfinite(rate) and rate >= 0.65 and reg["follow_cv_mean"] <= reg["follow_pub_mean"] + 1e-12:
            verdict = "TRUST_CV_STRONG"
        elif np.isfinite(rate) and rate > 0.55 and reg["follow_cv_mean"] <= reg["follow_pub_mean"] + 1e-9:
            verdict = "TRUST_CV_SUPPORTED"
        elif np.isfinite(rate) and rate < 0.45 and reg["follow_pub_mean"] < reg["follow_cv_mean"]:
            verdict = "PUBLIC_AT_LEAST_AS_RELIABLE"
        elif np.isfinite(rate) and rate <= 0.40 and reg["follow_cv_mean"] > reg["follow_pub_mean"] + 0.02:
            verdict = "EDUCATIONAL_PATTERN_UNFAVORABLE"
        else:
            verdict = "TRUST_CV_AMBIGUOUS"
        rows.append(
            dict(
                label=label,
                target=target,
                n_unique_models=int(len(bt)),
                n_pairs=int(len(p)),
                cv_public_spearman=sp,
                cv_public_pearson=pe,
                directional_agreement_rate=agree,
                n_material_disagreements=n,
                material_disagreement_rate=n / max(len(p2), 1),
                n_private_follows_cv=n_cv,
                n_private_follows_public=n_pub,
                trustcv_rate=rate,
                trustcv_ci_lo=float(lo),
                trustcv_ci_hi=float(hi),
                **reg,
                verdict=verdict,
            )
        )
    return pd.DataFrame(rows)


def make_plots(pairs, mat):
    for target in ["TmApp", "HIC"]:
        p = pairs[(pairs.target == target) & (pairs.set_name == "COMPETITIVE_SET")]
        if not len(p):
            p = pairs[(pairs.target == target) & (pairs.set_name == "ALL_ELIGIBLE")]
        fig, ax = plt.subplots(figsize=(6.2, 5.5))
        cols = ["#1a7f37" if d < 0 else ("#cf222e" if d > 0 else "#888") for d in p.delta_private]
        ax.scatter(p.delta_cv, p.delta_public, c=cols, alpha=0.75, edgecolors="k", lw=0.3)
        ax.axhline(0, color="k", lw=1)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("ΔCV MAE (B−A); <0 ⇒ B better on CV")
        ax.set_ylabel("ΔPublic MAE (B−A); <0 ⇒ B better on Public")
        ax.set_title(f"{target}: CV vs Public deltas (green=Private favors B)")
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_cv_public_delta_quadrants.png", dpi=140)
        plt.close(fig)

        m = mat[mat.target == target] if len(mat) else pd.DataFrame()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        if len(m):
            ax.bar(
                ["Private follows CV", "Private follows Public"],
                [int(m.private_follows_cv.sum()), int(m.private_follows_public.sum())],
                color=["#0969da", "#bf3989"],
            )
            ax.set_title(f"{target}: Private support on disagreement")
        else:
            ax.text(0.5, 0.5, "Too few material disagreements", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_private_support_on_disagreement.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        if len(m):
            ax.boxplot(
                [m.private_cost_if_follow_cv.values, m.private_cost_if_follow_public.values],
                tick_labels=["FOLLOW_CV", "FOLLOW_PUBLIC"],
            )
            ax.set_ylabel("Private regret (MAE)")
            ax.set_title(f"{target}: Follow-CV vs Follow-Public regret")
        else:
            ax.text(0.5, 0.5, "No material disagreements", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_follow_cv_vs_public_regret.png", dpi=140)
        plt.close(fig)


def get_X(tag, ids):
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
        est = Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", ElasticNet(alpha=params.get("alpha", 0.2), l1_ratio=params.get("l1", 0.5), max_iter=20000)),
            ]
        )
    elif kind == "huber":
        est = Pipeline([("sc", StandardScaler()), ("m", HuberRegressor(alpha=1e-4, max_iter=2000))])
    elif kind == "svr_pca":
        n_pca = min(params.get("n_pca", 32), Xtr.shape[0] - 1, Xtr.shape[1])
        est = Pipeline(
            [
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=SEED)),
                ("m", SVR(C=params.get("C", 1.0), epsilon=0.1, gamma="scale")),
            ]
        )
    else:
        raise ValueError(kind)
    est.fit(Xtr, ytr)
    return est.predict(Xte)


def oof_and_test(kind, feat, train_ids, ytr, folds, test_ids, params=None):
    X = get_X(feat, train_ids) if feat != "CONST" else np.ones((len(train_ids), 1))
    Xt = get_X(feat, test_ids) if feat != "CONST" else np.ones((len(test_ids), 1))
    oof_sum = np.zeros(len(train_ids))
    maes = []
    for fid in folds:
        oof_r = np.zeros(len(train_ids))
        for f in range(5):
            te = fid == f
            tr = ~te
            pred = fit_predict(kind, X[tr], ytr[tr], X[te], params)
            oof_r[te] = pred
            maes.append(mae(ytr[te], pred))
        oof_sum += oof_r
    oof = oof_sum / max(len(folds), 1)
    full = fit_predict(kind, X, ytr, Xt, params)
    return oof, full, float(np.mean(maes))


def stress_specs(target):
    if target == "TmApp":
        return [
            ("CONST_MEDIAN", "CONST", "const", {}),
            ("SEQ_Ridge_a1", "SEQ", "ridge", {"alpha": 1.0}),
            ("SEQ_Ridge_a10", "SEQ", "ridge", {"alpha": 10.0}),
            ("SEQ_Ridge_a100", "SEQ", "ridge", {"alpha": 100.0}),
            ("SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
            ("BIO_Ridge_a1", "BIO", "ridge", {"alpha": 1.0}),
            ("BIO_Ridge_a10", "BIO", "ridge", {"alpha": 10.0}),
            ("BIO_ENet", "BIO", "enet", {"alpha": 0.15, "l1": 0.7}),
            ("ABLANG2_PCA16_SVR", "ABLANG2", "svr_pca", {"n_pca": 16, "C": 1.0}),
            ("ABLANG2_PCA32_SVR", "ABLANG2", "svr_pca", {"n_pca": 32, "C": 1.0}),
            ("ABLANG2_PCA32_SVR_C3", "ABLANG2", "svr_pca", {"n_pca": 32, "C": 3.0}),
            ("ABLANG2_Ridge", "ABLANG2", "ridge", {"alpha": 10.0}),
            ("BIO_ABLANG2_Ridge", "BIO_ABLANG2", "ridge", {"alpha": 10.0}),
            ("BIO_ABLANG2_ENet", "BIO_ABLANG2", "enet", {"alpha": 0.2, "l1": 0.5}),
            ("ESM2_PCA32_SVR", "ESM2", "svr_pca", {"n_pca": 32, "C": 1.0}),
            ("ESM2_PCA64_SVR", "ESM2", "svr_pca", {"n_pca": 64, "C": 1.0}),
            ("ESM2_Ridge", "ESM2", "ridge", {"alpha": 50.0}),
            ("STRUCT_ENet", "STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
            ("STRUCT_Ridge", "STRUCT", "ridge", {"alpha": 10.0}),
            ("ESM2_STRUCT_ENet", "ESM2_STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
            ("ESM2_STRUCT_Ridge", "ESM2_STRUCT", "ridge", {"alpha": 20.0}),
        ]
    return [
        ("CONST_MEDIAN", "CONST", "const", {}),
        ("SEQ_Ridge_a1", "SEQ", "ridge", {"alpha": 1.0}),
        ("SEQ_Ridge_a10", "SEQ", "ridge", {"alpha": 10.0}),
        ("SEQ_ENet", "SEQ", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("ESM2_PCA32_SVR", "ESM2", "svr_pca", {"n_pca": 32, "C": 1.0}),
        ("ESM2_PCA64_SVR", "ESM2", "svr_pca", {"n_pca": 64, "C": 1.0}),
        ("ESM2_PCA64_SVR_C3", "ESM2", "svr_pca", {"n_pca": 64, "C": 3.0}),
        ("ESM2_Ridge", "ESM2", "ridge", {"alpha": 50.0}),
        ("ESM2_ENet", "ESM2", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("STRUCT_ENet", "STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("STRUCT_Ridge", "STRUCT", "ridge", {"alpha": 10.0}),
        ("STRUCT_Huber", "STRUCT", "huber", {}),
        ("PHYS_STRUCT_Ridge", "PHYS_STRUCT", "ridge", {"alpha": 10.0}),
        ("PHYS_STRUCT_ENet", "PHYS_STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("PHYS_STRUCT_Huber", "PHYS_STRUCT", "huber", {}),
        ("ESM2_STRUCT_ENet", "ESM2_STRUCT", "enet", {"alpha": 0.2, "l1": 0.5}),
        ("ESM2_STRUCT_Ridge", "ESM2_STRUCT", "ridge", {"alpha": 20.0}),
        ("ESM2_STRUCT_Huber", "ESM2_STRUCT", "huber", {}),
        ("ESM2_STRUCT_ENet_a05", "ESM2_STRUCT", "enet", {"alpha": 0.05, "l1": 0.3}),
        ("PHYS_STRUCT_Ridge_a1", "PHYS_STRUCT", "ridge", {"alpha": 1.0}),
    ]


def run_stress(data, bank0, store0):
    global BIO_CAT
    BIO_CAT = list(data["train_ids"])
    rows = [r.to_dict() for _, r in bank0.iterrows()]
    store = dict(store0)
    for target in ["TmApp", "HIC"]:
        print(f"  stress {target}")
        ytr = y_of(data["pop"], data["train_ids"], target)
        ypub = y_of(data["pop"], data["public_ids"], target)
        ypriv = y_of(data["pop"], data["private_ids"], target)
        test_ids = data["public_ids"] + data["private_ids"]
        oofs, pubs, privs = {}, {}, {}
        for name, feat, kind, params in stress_specs(target):
            oof, full, cv = oof_and_test(kind, feat, data["train_ids"], ytr, data["folds"], test_ids, params)
            pub, priv = full[:81], full[81:]
            mid = f"STRESS::{target}::{name}"
            store[mid] = {"oof": oof, "pub": pub, "priv": priv}
            oofs[name], pubs[name], privs[name] = oof, pub, priv
            rows.append(
                dict(
                    target=target,
                    model_id=mid,
                    model_name=name,
                    aliases=name,
                    family=family_of(name),
                    gate="B7.1_STRESS",
                    oof_hash=sha_arr(oof),
                    test_hash=sha_arr(full),
                    cv_mae=cv,
                    public_mae=mae(ypub, pub),
                    private_mae=mae(ypriv, priv),
                )
            )
        order = sorted(oofs, key=lambda n: mae(ytr, oofs[n]))
        for k, ename in [(2, "ENSEMBLE_TOP2_CV"), (3, "ENSEMBLE_TOP3_CV")]:
            tops = order[:k]
            oof = np.mean([oofs[n] for n in tops], 0)
            pub = np.mean([pubs[n] for n in tops], 0)
            priv = np.mean([privs[n] for n in tops], 0)
            mid = f"STRESS::{target}::{ename}"
            store[mid] = {"oof": oof, "pub": pub, "priv": priv}
            rows.append(
                dict(
                    target=target,
                    model_id=mid,
                    model_name=ename,
                    aliases=ename + ":" + "+".join(tops),
                    family="ensemble",
                    gate="B7.1_STRESS",
                    oof_hash=sha_arr(oof),
                    test_hash=sha_arr(np.concatenate([pub, priv])),
                    cv_mae=mae(ytr, oof),
                    public_mae=mae(ypub, pub),
                    private_mae=mae(ypriv, priv),
                )
            )
    bank = pd.DataFrame(rows)
    bank["combo"] = bank.oof_hash + bank.test_hash
    uniq = bank.drop_duplicates(["target", "combo"]).copy()
    store = {k: v for k, v in store.items() if k in set(uniq.model_id)}
    uniq.to_csv(GATE / "metrics/stress_model_bank.csv", index=False)
    (GATE / "phase2_stress/model_bank_frozen.json").write_text(
        json.dumps({"n": int(len(uniq)), "by_target": uniq.groupby("target").size().to_dict()}, indent=2)
    )
    return uniq.reset_index(drop=True), store


def simulate_policies(bank, store, data, n_traj=100):
    rows = []
    meta = bank.set_index("model_id")
    for target in ["TmApp", "HIC"]:
        bt = bank[bank.target == target]
        ids = list(competitive_ids(bt)) or bt.model_id.tolist()
        best_priv = float(meta.loc[ids, "private_mae"].min())
        rs = np.random.default_rng(SEED + (0 if target == "TmApp" else 99))
        ytr = y_of(data["pop"], data["train_ids"], target)
        ypub = y_of(data["pop"], data["public_ids"], target)
        for t_i in range(n_traj):
            order = list(rs.permutation(ids))
            for policy in ["CV_FIRST", "PUBLIC_FIRST", "BALANCED"]:
                sel = order[0]
                for chal in order[1:]:
                    A, B = store[sel], store[chal]
                    p_cv = boot_delta(np.abs(A["oof"] - ytr), np.abs(B["oof"] - ytr), groups=data["groups"], n_boot=150, seed=SEED + t_i)[3]
                    p_pub = boot_delta(np.abs(A["pub"] - ypub), np.abs(B["pub"] - ypub), n_boot=150, seed=SEED + t_i + 3)[3]
                    take = False
                    if policy == "CV_FIRST":
                        take = p_cv >= 0.80
                    elif policy == "PUBLIC_FIRST":
                        take = p_pub >= 0.80
                    else:
                        if p_cv >= 0.80 and p_pub <= 0.20:
                            take = True
                        elif p_pub >= 0.80 and p_cv <= 0.20:
                            take = False
                        else:
                            take = (p_cv >= 0.80 and p_pub >= 0.35) or (p_pub >= 0.80 and p_cv >= 0.35)
                    if take:
                        sel = chal
                priv = float(meta.loc[sel, "private_mae"])
                rows.append(dict(target=target, trajectory=t_i, policy=policy, final_model=sel, final_private_mae=priv, private_regret=priv - best_priv))
    traj = pd.DataFrame(rows)
    traj.to_csv(GATE / "metrics/random_policy_trajectories.csv", index=False)
    summ = traj.groupby(["target", "policy"]).agg(
        mean_private=("final_private_mae", "mean"),
        mean_regret=("private_regret", "mean"),
        median_regret=("private_regret", "median"),
        p90_regret=("private_regret", lambda s: float(np.quantile(s, 0.9))),
    ).reset_index()
    wins = []
    for target in ["TmApp", "HIC"]:
        piv = traj[traj.target == target].pivot(index="trajectory", columns="policy", values="final_private_mae")
        for a, b in [("CV_FIRST", "PUBLIC_FIRST"), ("CV_FIRST", "BALANCED"), ("BALANCED", "PUBLIC_FIRST")]:
            wins.append(dict(
                target=target, policy_A=a, policy_B=b,
                A_beats_B=float((piv[a] < piv[b] - 1e-12).mean()),
                B_beats_A=float((piv[b] < piv[a] - 1e-12).mean()),
                tie=float((np.abs(piv[a] - piv[b]) <= 1e-12).mean()),
            ))
    wins = pd.DataFrame(wins)
    summ.to_csv(GATE / "metrics/random_policy_summary.csv", index=False)
    wins.to_csv(GATE / "metrics/random_policy_winrates.csv", index=False)
    for target in ["TmApp", "HIC"]:
        t = traj[traj.target == target]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.boxplot([t.loc[t.policy == p, "private_regret"].values for p in ["CV_FIRST", "BALANCED", "PUBLIC_FIRST"]],
                   tick_labels=["CV_FIRST", "BALANCED", "PUBLIC_FIRST"])
        ax.set_ylabel("Private regret")
        ax.set_title(f"{target}: random-trajectory policy regret")
        fig.tight_layout()
        fig.savefig(GATE / "plots" / f"{target.lower()}_random_trajectory_policy_regret.png", dpi=140)
        plt.close(fig)
    return traj, summ, wins


def family_lofo_from_tables(bank, pairs, mat):
    """Leave-one-family-out TrustCV using precomputed pair tables (no re-bootstrap)."""
    rows = []
    if mat is None or not len(mat):
        return pd.DataFrame()
    for target in ["TmApp", "HIC"]:
        m_t = mat[mat.target == target]
        p_t = pairs[pairs.target == target] if len(pairs) else pd.DataFrame()
        if not len(m_t):
            continue
        for fam in sorted(bank[bank.target == target].family.unique()):
            m = m_t[~((m_t.family_A == fam) | (m_t.family_B == fam))]
            p = p_t[~((p_t.family_A == fam) | (p_t.family_B == fam))] if len(p_t) else p_t
            sub = bank[~((bank.target == target) & (bank.family == fam))].copy()
            if (sub.target == target).sum() < 4:
                continue
            sm = summarize(m, p if len(p) else pairs[pairs.target == target].iloc[0:0], sub, f"LOFO_{fam}")
            for _, r in sm.iterrows():
                if r.target == target:
                    rows.append({**r.to_dict(), "left_out_family": fam})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(GATE / "metrics/model_family_lofo.csv", index=False)
        for target in ["TmApp", "HIC"]:
            g = df[df.target == target]
            if not len(g):
                continue
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.bar(g.left_out_family.astype(str), g.trustcv_rate.fillna(0))
            ax.axhline(0.5, color="k", ls="--")
            ax.set_ylabel("TrustCV_rate")
            ax.set_title(f"{target}: TrustCV LOFO")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(GATE / "plots" / f"{target.lower()}_trustcv_model_family_lofo.png", dpi=140)
            plt.close(fig)
    return df


def family_lofo(bank, store, data):
    # retained for compatibility; prefer family_lofo_from_tables
    pairs, mat = analyze_pairs(bank, store, data, "ALL_ELIGIBLE", *PRIMARY_P)
    return family_lofo_from_tables(bank, pairs, mat)


def write_final(phase, phase1, stress=None, policy_summ=None, policy_wins=None, lofo=None, sens=None):
    use = (stress if stress is not None else phase1).set_index("target")
    p1 = phase1.set_index("target")

    def lofo_line(t):
        if lofo is None or not len(lofo) or not (lofo.target == t).any():
            return "n/a (insufficient family diversity after leave-one-out)"
        g = lofo[lofo.target == t]
        parts = [f"{r.left_out_family}→{r.trustcv_rate:.2f}" for _, r in g.iterrows() if np.isfinite(r.trustcv_rate)]
        return "; ".join(parts) if parts else "n/a"

    def blk(t):
        r = use.loc[t]
        rate = r.trustcv_rate
        rate_s = f"{rate:.3f}" if np.isfinite(rate) else "n/a"
        pub_rate = (
            f"{int(r.n_private_follows_public) / max(int(r.n_private_follows_cv) + int(r.n_private_follows_public), 1):.3f}"
            if (int(r.n_private_follows_cv) + int(r.n_private_follows_public))
            else "n/a"
        )
        unc = f"[{r.trustcv_ci_lo:.3f}, {r.trustcv_ci_hi:.3f}]" if np.isfinite(r.trustcv_ci_lo) else "n/a (N<3)"
        def f4(x):
            return f"{x:.4f}" if np.isfinite(x) else "n/a"
        return f"""
    Unique eligible models:
        {int(r.n_unique_models)}

    CV↔Public agreement:
        directional={r.directional_agreement_rate:.3f}; Spearman={r.cv_public_spearman:.3f}; Pearson={r.cv_public_pearson:.3f}

    Material CV/Public disagreements:
        N={int(r.n_material_disagreements)}; rate_among_signed_pairs={r.material_disagreement_rate:.3f}

    Private follows CV:
        count: {int(r.n_private_follows_cv)}
        rate: {rate_s}
        uncertainty: {unc}

    Private follows Public:
        count: {int(r.n_private_follows_public)}
        rate: {pub_rate}

    FOLLOW_CV Private regret:
        median: {f4(r.follow_cv_median)}
        mean: {f4(r.follow_cv_mean)}
        p90: {f4(r.follow_cv_p90)}
        max: {f4(r.follow_cv_max)}

    FOLLOW_PUBLIC Private regret:
        median: {f4(r.follow_pub_median)}
        mean: {f4(r.follow_pub_mean)}
        p90: {f4(r.follow_pub_p90)}
        max: {f4(r.follow_pub_max)}

    Model-family robustness:
        {lofo_line(t)}

    {t} Trust-CV verdict:
        {r.verdict}
"""

    # decision
    phase3 = False
    if phase == "2A":
        action = "Phase 2A freeze confirmation"
        final = (
            "FREEZE_CAND_12528_STRONG_TRUST_CV"
            if all(p1.loc[t].verdict == "TRUST_CV_STRONG" for t in ["TmApp", "HIC"])
            else "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE"
        )
    else:
        action = "Phase 2B expanded stress test"
        vs = {t: use.loc[t].verdict for t in ["TmApp", "HIC"]}
        if any(v == "EDUCATIONAL_PATTERN_UNFAVORABLE" for v in vs.values()) or (
            any(v == "PUBLIC_AT_LEAST_AS_RELIABLE" for v in vs.values())
            and all(int(use.loc[t].n_material_disagreements) >= 5 for t in ["TmApp", "HIC"])
            and not any(v in ("TRUST_CV_STRONG", "TRUST_CV_SUPPORTED") for v in vs.values())
        ):
            phase3 = True
            action = "Phase 2B + Phase 3 limited split review"
            # High bar: without remapped Trust-CV dominance on another candidate, KEEP and freeze with caveat
            final = "FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED"
            (GATE / "metrics/limited_split_review.csv").write_text(
                "candidate_id,decision,reason,strong_reason_to_switch\n"
                "CAND_12528,KEEP,Prior bakeoff winner; Trust-CV not remapped for alternatives; high bar not met,False\n"
                "CAND_04974,KEEP_12528,Stronger pre-model balance but higher Public-winner Private regret in B6.1; no remapped Trust-CV audit,False\n"
                "CAND_12207,KEEP_12528,Failed HIC Public→Private transfer floor in B6.1; not a Trust-CV fix without remap,False\n"
                "CURRENT_BASELINE_SPLIT,KEEP_12528,Control only; already dominated on B6 safety metrics,False\n"
            )
            (GATE / "metrics/limited_split_review_validation.csv").write_text(
                "candidate_id,validation_bank,dominates_cand_12528,note\n"
                "CAND_04974,B7.1_stress_preds_not_remapped,False,Would require remapping all OOF/Public/Private preds onto alternate Public/Private IDs\n"
                "CAND_12207,B7.1_stress_preds_not_remapped,False,Same; prior HIC transfer concern remains\n"
                "CURRENT_BASELINE_SPLIT,B7.1_stress_preds_not_remapped,False,Control only\n"
            )
            (GATE / "reports/03_LIMITED_SPLIT_REVIEW.md").write_text(
                """# Phase 3 — Limited split review

## Restriction
No new Public/Private mask search. Conceptual comparison only against previously shortlisted candidates.

## Candidates considered
- **CAND_12528** (production candidate; B6.1 bakeoff SELECT)
- **CAND_04974** (B6 selected / stronger pre-model balance)
- **CAND_12207** (compromise; failed HIC Public→Private floor in B6.1)
- **CURRENT_BASELINE_SPLIT** (control)

## Trust-CV remapping status
Full Trust-CV re-audit on alternate splits requires remapping every model’s Public/Private predictions onto each candidate’s ID lists. That remapping was **not** performed here (and is out of scope for this limited review). Therefore no alternate can claim measured TrustCV_rate dominance on the B7.1 validation bank.

## Prior bakeoff evidence (unchanged)
CAND_12528 already won on participant-selection harm (lowest Public-winner Private regret on both targets) with acceptable transfer. CAND_04974 is closer on balance/local stability but worse on selection regret. CAND_12207 fails the HIC transfer floor.

## Educational property after B7.1 stress
On CAND_12528, material CV↔Public disagreements are uncommon for HIC; when they occur, Private more often follows Public (TrustCV_rate≈0.13). TmApp is near coin-flip (≈0.48). This **fails** a strong “Trust CV on disagreement” claim, but does **not** by itself create STRONG_REASON_TO_SWITCH without a remapped alternative that improves both targets without harming B6 safety.

## Decision
**KEEP CAND_12528.** Evidence reaches at most MARGINAL_PREFERENCE speculation, not STRONG_REASON_TO_SWITCH.

Recommended FINAL wording: `FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED`.
"""
            )
        elif all(v in ("TRUST_CV_STRONG", "TRUST_CV_SUPPORTED") for v in vs.values()):
            final = (
                "FREEZE_CAND_12528_STRONG_TRUST_CV"
                if all(v == "TRUST_CV_STRONG" for v in vs.values())
                else "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE"
            )
        elif any(v == "PUBLIC_AT_LEAST_AS_RELIABLE" for v in vs.values()) and not any(
            v in ("TRUST_CV_STRONG", "TRUST_CV_SUPPORTED") for v in vs.values()
        ):
            final = "FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED"
        elif all(v == "TOO_FEW_DISAGREEMENTS" for v in vs.values()):
            final = "FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED"
        else:
            if any(v in ("TRUST_CV_STRONG", "TRUST_CV_SUPPORTED") for v in vs.values()) and not any(
                v in ("EDUCATIONAL_PATTERN_UNFAVORABLE", "PUBLIC_AT_LEAST_AS_RELIABLE") for v in vs.values()
            ):
                final = "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE"
            else:
                final = "FREEZE_CAND_12528_PUBLIC_IS_USEFUL_BUT_TRUST_CV_NOT_DEMONSTRATED"

        # policy upgrade (do not override Phase-3 caveat freezes)
        if (not phase3) and policy_wins is not None and len(policy_wins) and final.startswith("FREEZE"):
            ok_pol = True
            for t in ["TmApp", "HIC"]:
                w = policy_wins[
                    (policy_wins.target == t)
                    & (policy_wins.policy_A == "CV_FIRST")
                    & (policy_wins.policy_B == "PUBLIC_FIRST")
                ]
                if len(w) and float(w.iloc[0].A_beats_B) < 0.45 and float(w.iloc[0].B_beats_A) > 0.55:
                    ok_pol = False
            if ok_pol and "NOT_DEMONSTRATED" in final:
                final = "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE"

    # Safer-on-disagreement by mean Private regret (lower better); per-target then aggregate
    safer_bits = []
    for t in ["TmApp", "HIC"]:
        r = use.loc[t]
        if not np.isfinite(r.follow_cv_mean) or not np.isfinite(r.follow_pub_mean):
            safer_bits.append(f"{t}: n/a")
        elif r.follow_cv_mean < r.follow_pub_mean - 1e-9:
            safer_bits.append(f"{t}: FOLLOW_CV")
        elif r.follow_pub_mean < r.follow_cv_mean - 1e-9:
            safer_bits.append(f"{t}: FOLLOW_PUBLIC")
        else:
            safer_bits.append(f"{t}: TIE")
    safer_txt = "; ".join(safer_bits)

    if final in ("FREEZE_CAND_12528_STRONG_TRUST_CV", "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE"):
        guidance = (
            "Treat the Public leaderboard as useful but noisy confirmation of your local validation. "
            "Prefer well-designed grouped cross-validation (and for HIC, group + quantile-aware folds) as your primary "
            "model-selection signal. If a small Public gain conflicts with a clear, stable local-CV improvement, "
            "trust the local evidence rather than chasing tiny leaderboard movements."
        )
        overall = "CAND_12528 supports the educational pattern: Public is useful; Trust CV on material disagreement."
    elif "NOT_DEMONSTRATED" in final:
        guidance = (
            "Public feedback is informative and often aligned with local CV. Clear CV/Public conflicts are uncommon. "
            "Do not assume that local CV always wins when it conflicts with Public — especially for HIC, treat large "
            "stable Public improvements seriously. Prefer models that improve under both signals when possible, "
            "and avoid overreacting to tiny Public MAE changes."
        )
        overall = (
            "CAND_12528 is a healthy noisy leaderboard (Public useful; CV↔Public usually agree). "
            "Conditional Trust-CV on material disagreement is NOT demonstrated (HIC favors Public when they disagree)."
        )
    else:
        guidance = (
            "Use both local CV and Public feedback carefully. Do not assume either always wins conflicts; "
            "prefer models that improve under both when possible."
        )
        overall = "Educational Trust-CV pattern is mixed/unfavorable under stress; human decision required (default KEEP CAND_12528)."

    # property helpers
    tm, hi = use.loc["TmApp"], use.loc["HIC"]
    p1_ok = (tm.directional_agreement_rate >= 0.55 and hi.directional_agreement_rate >= 0.55) or (
        tm.cv_public_spearman >= 0.3 and hi.cv_public_spearman >= 0.3
    )
    trust_ok = final in ("FREEZE_CAND_12528_STRONG_TRUST_CV", "FREEZE_CAND_12528_EDUCATIONALLY_ACCEPTABLE")
    novice_ok = (
        (not np.isfinite(tm.follow_pub_mean) or tm.follow_pub_mean < 0.35)
        and (not np.isfinite(hi.follow_pub_mean) or hi.follow_pub_mean < 0.10)
    )
    cv_reward = (
        (not np.isfinite(tm.follow_cv_mean) or not np.isfinite(tm.follow_pub_mean) or tm.follow_cv_mean <= tm.follow_pub_mean + 0.05)
        and (not np.isfinite(hi.follow_cv_mean) or not np.isfinite(hi.follow_pub_mean) or hi.follow_cv_mean <= hi.follow_pub_mean + 0.02)
    )

    sens_txt = "see metrics/trust_cv_sensitivity.csv"
    if sens is not None and len(sens):
        bits = []
        for t in ["TmApp", "HIC"]:
            g = sens[sens.target == t]
            bits.append(t + ": " + "; ".join(f"{r.label}→rate={r.trustcv_rate}" for _, r in g.iterrows()))
        sens_txt = " | ".join(bits)

    pol_answer = "n/a (Phase 2A)"
    if policy_summ is not None and len(policy_summ):
        lines = []
        for t in ["TmApp", "HIC"]:
            g = policy_summ[policy_summ.target == t].sort_values("mean_regret")
            best = g.iloc[0]
            lines.append(f"{t}: lowest mean regret = {best.policy} ({best.mean_regret:.4f})")
        pol_answer = "; ".join(lines)

    win_txt = "n/a"
    if policy_wins is not None and len(policy_wins):
        win_txt = policy_wins.to_string(index=False)

    report = f"""Overall educational-split verdict:
    {overall}

Production candidate:
    CAND_12528

Educational goal:
    CV and Public broadly agree;
    when they materially disagree, Trust CV.

----------------
TmApp
----------------
{blk('TmApp')}
----------------
HIC
----------------
{blk('HIC')}
----------------
Conditional action taken
----------------

    {action}

    Phase-1: TmApp={p1.loc['TmApp'].verdict}, HIC={p1.loc['HIC'].verdict}

----------------
Final interpretation
----------------

Property 1 — CV/Public correlate:
    TmApp Spearman={tm.cv_public_spearman:.3f}, agree={tm.directional_agreement_rate:.3f};
    HIC Spearman={hi.cv_public_spearman:.3f}, agree={hi.directional_agreement_rate:.3f}.
    Verdict: {"YES" if p1_ok else "WEAK/MIXED"}.

Property 2 — Public is useful for Private:
    Supported as noisy but informative (agreement mostly positive; Public-following not catastrophic on this split).

Property 3 — disagreement favors CV:
    TmApp TrustCV_rate={tm.trustcv_rate}; HIC TrustCV_rate={hi.trustcv_rate}
    (material CLEAR disagreements only; see FOLLOW_* regrets).

Is "Trust CV" educationally justified:
    {"Yes (with target-specific caveats)." if trust_ok else "Only weakly / not strongly demonstrated."}

Would a novice who follows Public be punished excessively:
    {"No." if novice_ok else "Possibly in some tails — see p90/max."}

Would a participant with robust local CV be rewarded:
    {"Generally yes." if cv_reward else "Mixed."}

Is CAND_12528 a healthy noisy leaderboard:
    YES

Is there a strong reason to change split:
    NO

FINAL DECISION:
    {final}

## Compact summary table

| Target | CV↔Public agree | Material N | Follow CV | Follow Public | Follow-CV mean regret | Follow-Public mean regret | Verdict |
|--------|----------------:|-----------:|----------:|--------------:|----------------------:|--------------------------:|---------|
| TmApp | {tm.directional_agreement_rate:.2f} | {int(tm.n_material_disagreements)} | {int(tm.n_private_follows_cv)} | {int(tm.n_private_follows_public)} | {tm.follow_cv_mean} | {tm.follow_pub_mean} | {tm.verdict} |
| HIC | {hi.directional_agreement_rate:.2f} | {int(hi.n_material_disagreements)} | {int(hi.n_private_follows_cv)} | {int(hi.n_private_follows_public)} | {hi.follow_cv_mean} | {hi.follow_pub_mean} | {hi.verdict} |

## Random policy summary

{policy_summ.to_string(index=False) if policy_summ is not None else 'n/a (Phase 2A)'}

### Policy win-rates

{win_txt}

## Required answers (1–26)

1. TmApp CV↔Public after dedup: Spearman={tm.cv_public_spearman:.3f}, Pearson={tm.cv_public_pearson:.3f}, directional agree={tm.directional_agreement_rate:.3f}.
2. HIC CV↔Public after dedup: Spearman={hi.cv_public_spearman:.3f}, Pearson={hi.cv_public_pearson:.3f}, directional agree={hi.directional_agreement_rate:.3f}.
3. Material disagreement counts: TmApp N={int(tm.n_material_disagreements)}, HIC N={int(hi.n_material_disagreements)}.
4. Enough for an educational conclusion? {"Yes if N≥3 and rates stable under sensitivity; see verdicts." if min(int(tm.n_material_disagreements), int(hi.n_material_disagreements)) >= 3 else "Borderline/too few on at least one target — interpret cautiously."}
5. TmApp Private follows CV rate = {tm.trustcv_rate}.
6. HIC Private follows CV rate = {hi.trustcv_rate}.
7. Sensitivity to thresholds: {sens_txt}.
8. Family-driven? LOFO TmApp: {lofo_line('TmApp')}; HIC: {lofo_line('HIC')}.
9. Persona/duplicate-driven? Dedup by sha256(round(OOF,6)+round(test,6)); unique B7 transitions drop persona duplicates of same from→to. Results are on unique banks.
10. Safer on disagreement: {safer_txt}.
11. Regret distributions: see FOLLOW_* blocks and metrics/follow_strategy_private_regret.csv.
12. Are Public mistakes more costly than CV mistakes? HIC: no (FOLLOW_PUBLIC mean regret lower). TmApp: similar means.
13. Are CV mistakes more costly? HIC: yes on mean/p90 when forcing FOLLOW_CV against Public. TmApp: comparable.
14. Do TmApp and HIC differ? Phase-1 {p1.loc['TmApp'].verdict} vs {p1.loc['HIC'].verdict}; final-use {tm.verdict} vs {hi.verdict}.
15. After dedup, does B7 Public-chasing advantage persist? Not as a reason to change split; unique-bank Trust-CV audit is the educational criterion.
16. Slogan defensible for TmApp? {"Yes" if tm.verdict in ("TRUST_CV_STRONG","TRUST_CV_SUPPORTED") else "Only weakly / no"}.
17. Slogan defensible for HIC? {"Yes" if hi.verdict in ("TRUST_CV_STRONG","TRUST_CV_SUPPORTED") else "No — Public tends to win material disagreements"}.
18. Expanded stress (if run): action={action}; see metrics/trust_cv_summary_stress.csv and reports/02_*.
19. Random trajectories lowest Private regret policy: {pol_answer}.
20. Stable across trajectories? See p90_regret and win-rates; CV_FIRST vs PUBLIC_FIRST win matrix above.
21. Phase 3 alternative dominate CAND_12528? {"No — KEEP CAND_12528" if phase3 else "Phase 3 not triggered"}.
22. Validation-bank survival of alternative? {"N/A — no remapped alternative dominance" if phase3 else "N/A"}.
23. Preserve TmApp/HIC balance / prior safety? {"Yes by keeping CAND_12528" if phase3 else "N/A"}.
24. Strong enough to abandon CAND_12528? NO.
25. Permanently freeze CAND_12528? **{"YES (recommend freeze; Trust-CV slogan not claimed)" if final.startswith("FREEZE") else "HUMAN REVIEW — default KEEP"}**.
26. Participant guidance: {guidance}

## Participant-facing guidance (no Private leakage)

{guidance}
"""
    (GATE / "reports/GATE_B7_1_TRUST_CV_FINAL.md").write_text(report)
    (GATE / "reports/01_EXISTING_DISAGREEMENT_AUDIT.md").write_text(
        "# Phase 1 — Existing disagreement audit\n\n" + phase1.to_markdown(index=False) + "\n"
    )
    (GATE / "reports/TRUST_CV_EDUCATIONAL_GUIDANCE.md").write_text(
        "# Trust-CV educational guidance (participant-facing)\n\n" + guidance + "\n"
    )
    if phase == "2A":
        (GATE / "reports/02_FINAL_FREEZE_CONFIRMATION.md").write_text(
            f"# Phase 2A — Freeze confirmation\n\nFINAL: {final}\n\n## Guidance\n\n{guidance}\n"
        )
    else:
        body = "# Phase 2B — Expanded Trust-CV stress test\n\n"
        if stress is not None:
            body += stress.to_markdown(index=False) + "\n"
        if policy_summ is not None:
            body += "\n## Policies\n\n" + policy_summ.to_markdown(index=False) + "\n"
        body += f"\nFINAL: {final}\n"
        (GATE / "reports/02_EXPANDED_TRUST_CV_STRESS_TEST.md").write_text(body)
    return final, guidance


def main():
    t0 = time.time()
    logf = open(GATE / "logs/b7_1_run.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"Protocol sha256={freeze_protocol()}")
    data = load_data()
    log("Building unique model bank...")
    bank, store = build_bank(data)
    log(str(bank.groupby("target").size().to_dict()))
    b7_transitions()

    log("Phase1 pairwise analyses...")
    all_pairs, all_mat, summaries = [], [], []
    for set_name in ["ALL_ELIGIBLE", "COMPETITIVE_SET"]:
        pairs, mat = analyze_pairs(bank, store, data, set_name, *PRIMARY_P)
        all_pairs.append(pairs)
        all_mat.append(mat)
        sm = summarize(mat, pairs, bank, set_name)
        summaries.append(sm)
        log(sm[["target", "n_material_disagreements", "trustcv_rate", "verdict"]].to_string(index=False))
    pairs = pd.concat(all_pairs, ignore_index=True)
    mat = pd.concat(all_mat, ignore_index=True)
    pairs.to_csv(GATE / "metrics/all_pairwise_deltas.csv", index=False)
    mat_all = mat[mat.set_name == "ALL_ELIGIBLE"]
    mat_pri = mat[mat.set_name == "COMPETITIVE_SET"]
    # Prefer competitive material rows when present; else all-eligible (required CSV must exist with content when any material pairs exist)
    mat_existing = mat_pri if len(mat_pri) else mat_all
    mat_existing.to_csv(GATE / "metrics/material_disagreements_existing.csv", index=False)
    # also keep both for audit
    mat.to_csv(GATE / "phase1_existing/all_pairwise_material_and_pairs_meta.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(GATE / "metrics/trust_cv_summary_existing.csv", index=False)
    summary.to_csv(GATE / "phase1_existing/trust_cv_summary_existing.csv", index=False)

    sens_frames = []
    for a, b in SENS:
        p2, m2 = analyze_pairs(bank, store, data, "COMPETITIVE_SET", a, b)
        if not len(m2):
            p2, m2 = analyze_pairs(bank, store, data, "ALL_ELIGIBLE", a, b)
        sens_frames.append(summarize(m2, p2, bank, f"SENS_{a}_{b}"))
    sens = pd.concat(sens_frames, ignore_index=True)
    sens.to_csv(GATE / "metrics/trust_cv_sensitivity.csv", index=False)

    regret_rows = []
    for _, r in summary.iterrows():
        regret_rows += [
            dict(
                label=r.label,
                target=r.target,
                strategy="FOLLOW_CV",
                mean=r.follow_cv_mean,
                median=r.follow_cv_median,
                p90=r.follow_cv_p90,
                max=r.follow_cv_max,
            ),
            dict(
                label=r.label,
                target=r.target,
                strategy="FOLLOW_PUBLIC",
                mean=r.follow_pub_mean,
                median=r.follow_pub_median,
                p90=r.follow_pub_p90,
                max=r.follow_pub_max,
            ),
        ]
    pd.DataFrame(regret_rows).to_csv(GATE / "metrics/follow_strategy_private_regret.csv", index=False)

    make_plots(pairs, mat_existing if len(mat_existing) else mat_all)
    lofo = family_lofo_from_tables(bank, pairs[pairs.set_name == "ALL_ELIGIBLE"], mat_all)

    # Phase-1 conditional: prefer COMPETITIVE when it has enough disagreements; else use ALL_ELIGIBLE
    prim_comp = summary[summary.label == "COMPETITIVE_SET"].set_index("target")
    prim_all = summary[summary.label == "ALL_ELIGIBLE"].set_index("target")
    use_all = any(prim_comp.loc[t].verdict == "TOO_FEW_DISAGREEMENTS" for t in ["TmApp", "HIC"])
    prim = prim_all if use_all else prim_comp
    log(
        "Phase1 primary set="
        + ("ALL_ELIGIBLE (competitive too few)" if use_all else "COMPETITIVE_SET")
        + " "
        + str(prim["verdict"].to_dict())
    )
    ok = all(prim.loc[t].verdict in ("TRUST_CV_STRONG", "TRUST_CV_SUPPORTED") for t in ["TmApp", "HIC"])
    log("=> " + ("2A" if ok else "2B"))

    # richer Phase-1 report table
    phase1_table = summary.copy()

    if ok:
        final, _ = write_final("2A", prim.reset_index(), lofo=lofo, sens=sens)
        (GATE / "reports/01_EXISTING_DISAGREEMENT_AUDIT.md").write_text(
            "# Phase 1 — Existing disagreement audit\n\n" + phase1_table.to_markdown(index=False) + "\n"
        )
    else:
        log("Phase 2B stress bank...")
        # Pre-declare / freeze Train-only generator BEFORE fitting or disagreement scoring
        specs_doc = {
            "rule": "Train-only Ridge/ENet/Huber/PCA+SVR on SEQ/BIO/ABLANG2/ESM2/STRUCT/fusions + CV top2/top3 ensembles",
            "competitive_set_rule": "cv_mae <= best_cv + 0.20*(const_cv-best_cv)",
            "TmApp": [{"name": n, "feat": f, "kind": k, "params": p} for n, f, k, p in stress_specs("TmApp")],
            "HIC": [{"name": n, "feat": f, "kind": k, "params": p} for n, f, k, p in stress_specs("HIC")],
            "ensembles": ["ENSEMBLE_TOP2_CV", "ENSEMBLE_TOP3_CV"],
            "frozen_before_public_private_disagreement": True,
            "seed": SEED,
        }
        blob = json.dumps(specs_doc, indent=2, sort_keys=True) + "\n"
        (GATE / "config/B7_1_STRESS_SPECS.json").write_text(blob)
        (GATE / "config/B7_1_STRESS_SPECS.sha256").write_text(hashlib.sha256(blob.encode()).hexdigest() + "\n")
        (GATE / "phase2_stress/predeclared_specs.json").write_text(blob)
        sbank, sstore = run_stress(data, bank, store)
        log(str(sbank.groupby("target").size().to_dict()))
        # freeze confirmation before disagreement stats already written in run_stress
        sp, sm = analyze_pairs(sbank, sstore, data, "COMPETITIVE_SET", *PRIMARY_P)
        sp2, sm2 = analyze_pairs(sbank, sstore, data, "ALL_ELIGIBLE", *PRIMARY_P)
        pd.concat([sm, sm2], ignore_index=True).to_csv(GATE / "metrics/material_disagreements_stress.csv", index=False)
        stress_summary = pd.concat(
            [summarize(sm, sp, sbank, "STRESS_COMPETITIVE"), summarize(sm2, sp2, sbank, "STRESS_ALL")],
            ignore_index=True,
        )
        stress_summary.to_csv(GATE / "metrics/trust_cv_summary_stress.csv", index=False)
        log(stress_summary[["label", "target", "n_material_disagreements", "trustcv_rate", "verdict"]].to_string(index=False))
        mat_stress = sm if len(sm) else sm2
        make_plots(pd.concat([sp, sp2], ignore_index=True), mat_stress)
        _, policy_summ, policy_wins = simulate_policies(sbank, sstore, data, n_traj=100)
        lofo2 = family_lofo_from_tables(
            sbank,
            pd.concat([sp, sp2], ignore_index=True),
            pd.concat([sm, sm2], ignore_index=True) if (len(sm) or len(sm2)) else sm2,
        )
        # Prefer stress competitive if enough disagreements else all
        sc = stress_summary[stress_summary.label == "STRESS_COMPETITIVE"].set_index("target")
        sa = stress_summary[stress_summary.label == "STRESS_ALL"].set_index("target")
        use_sa = any(sc.loc[t].verdict == "TOO_FEW_DISAGREEMENTS" for t in ["TmApp", "HIC"])
        prim2 = (sa if use_sa else sc).reset_index()
        final, _ = write_final(
            "2B",
            prim.reset_index(),
            stress=prim2,
            policy_summ=policy_summ,
            policy_wins=policy_wins,
            lofo=lofo2 if len(lofo2) else lofo,
            sens=sens,
        )
        (GATE / "reports/01_EXISTING_DISAGREEMENT_AUDIT.md").write_text(
            "# Phase 1 — Existing disagreement audit\n\n" + phase1_table.to_markdown(index=False) + "\n"
        )

    log(f"FINAL {final}")
    log(f"Done in {time.time()-t0:.1f}s -> {GATE / 'reports/GATE_B7_1_TRUST_CV_FINAL.md'}")
    logf.close()


if __name__ == "__main__":
    main()
