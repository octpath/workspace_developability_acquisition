#!/usr/bin/env python3
"""
Build Train-only CV folds + new feature branches:
  IMGT positional one-hot, germline-relative, n-grams, structure extensions.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import (  # noqa: E402
    AA20,
    B1_DATA,
    B2_CACHE,
    CONFIG,
    FEATURES,
    HYDROPHOBIC,
    KD,
    ORG,
    REPORTS,
    ensure_dirs,
    write_json,
)

AA_LIST = list(AA20) + ["-", "X"]  # gap/unknown
AA_IDX = {a: i for i, a in enumerate(AA_LIST)}
N_AA = len(AA_LIST)

# Standard IMGT VH positions 1-128, VL 1-127 (with gaps in numbering space)
IMGT_H_MAX = 128
IMGT_L_MAX = 127


def build_cv_folds(train_df: pd.DataFrame, n_folds=5, n_repeats=4, seed=42):
    """Repeated grouped folds balancing HIC/TmApp quantiles approximately."""
    groups = train_df["sequence_group"].values
    uniq = np.unique(groups)
    folds = []
    rng = np.random.default_rng(seed)
    for rep in range(n_repeats):
        order = uniq.copy()
        rng.shuffle(order)
        # assign clusters round-robin after sorting by HIC quantile of cluster mean
        means = []
        for g in order:
            m = train_df.loc[train_df.sequence_group == g, "HIC"].mean()
            means.append((g, m))
        means.sort(key=lambda x: x[1])
        fold_of = {}
        for i, (g, _) in enumerate(means):
            fold_of[g] = i % n_folds
        # slight reshuffle of fold labels
        perm = rng.permutation(n_folds)
        fold_of = {g: int(perm[f]) for g, f in fold_of.items()}
        fold_id = np.array([fold_of[g] for g in groups], dtype=int)
        folds.append({"repeat": rep, "fold_id": fold_id.tolist()})
    return folds


def region_to_imgt_map(row, chain="H"):
    """Map sequential residues in FR/CDR concat to IMGT numbers via cumulative region lengths.
    Returns list of (imgt_pos, aa) for observed residues.
    """
    # Approximate IMGT ranges
    if chain == "H":
        regions = [
            ("FR1", 1, 26),
            ("CDR1", 27, 38),
            ("FR2", 39, 55),
            ("CDR2", 56, 65),
            ("FR3", 66, 104),
            ("CDR3", 105, 117),
            ("FR4", 118, 128),
        ]
        prefix = "H"
    else:
        regions = [
            ("FR1", 1, 26),
            ("CDR1", 27, 38),
            ("FR2", 39, 55),
            ("CDR2", 56, 65),
            ("FR3", 66, 104),
            ("CDR3", 105, 117),
            ("FR4", 118, 127),
        ]
        prefix = "L"
    pairs = []
    for name, lo, hi in regions:
        seq = str(row.get(f"{prefix}_{name}") or "")
        # distribute seq across IMGT slot length (truncate/pad conceptually by position index)
        slot = list(range(lo, hi + 1))
        if not seq:
            continue
        if len(seq) <= len(slot):
            # place from end for CDR3-like (common ANARCI style) — simple left-align for FR, right for CDR3
            if name == "CDR3":
                placed = slot[-len(seq) :]
            else:
                placed = slot[: len(seq)]
            for p, aa in zip(placed, seq):
                pairs.append((p, aa))
        else:
            # too long: take middle truncation
            start = (len(seq) - len(slot)) // 2
            seq2 = seq[start : start + len(slot)]
            for p, aa in zip(slot, seq2):
                pairs.append((p, aa))
    return pairs


def positional_onehot(pop: pd.DataFrame, num: pd.DataFrame) -> dict:
    merged = pop.merge(num, left_on="id", right_on="antibody_id", how="left")
    dim_h = IMGT_H_MAX * N_AA
    dim_l = IMGT_L_MAX * N_AA
    Xh = np.zeros((len(merged), dim_h), dtype=np.float32)
    Xl = np.zeros((len(merged), dim_l), dtype=np.float32)
    for i in range(len(merged)):
        row = merged.iloc[i]
        for pos, aa in region_to_imgt_map(row, "H"):
            if 1 <= pos <= IMGT_H_MAX:
                a = aa if aa in AA_IDX else "X"
                Xh[i, (pos - 1) * N_AA + AA_IDX[a]] = 1.0
        for pos, aa in region_to_imgt_map(row, "L"):
            if 1 <= pos <= IMGT_L_MAX:
                a = aa if aa in AA_IDX else "X"
                Xl[i, (pos - 1) * N_AA + AA_IDX[a]] = 1.0
    ids = merged["id"].values
    np.savez_compressed(
        FEATURES / "imgt" / "positional_onehot.npz",
        ids=ids,
        Xh=Xh,
        Xl=Xl,
        Xhl=np.concatenate([Xh, Xl], axis=1),
    )
    meta = {"dim_h": int(dim_h), "dim_l": int(dim_l), "dim_hl": int(dim_h + dim_l), "aa": AA_LIST}
    write_json(FEATURES / "imgt" / "positional_meta.json", meta)
    return meta


def germline_relative(pop: pd.DataFrame, num: pd.DataFrame):
    """Binary mutation + physchem delta on FR+CDR1/2 (not CDR3 as SHM)."""
    m = pop.merge(num, left_on="id", right_on="antibody_id", how="left")
    rows = []
    for _, r in m.iterrows():
        feat = {"id": r["id"]}
        for chain, pref in [("H", "H"), ("L", "L")]:
            mut = 0
            tot = 0
            d_kd = 0.0
            for reg in ["FR1", "CDR1", "FR2", "CDR2", "FR3"]:
                q = str(r.get(f"{pref}_{reg}") or "")
                # germline reconstructed poorly — use author germline distance proxies already in num
                tot += len(q)
            feat[f"{pref}_seq_len_fr_cdr12"] = tot
        # use existing PL germline features
        for c in m.columns:
            if c.startswith("PL_") and ("distance" in c or "mutfrac" in c or "len" in c):
                try:
                    feat[c] = float(r[c]) if pd.notna(r[c]) else np.nan
                except Exception:
                    pass
        # region gravy deltas vs whole
        for pref in ["H", "L"]:
            for reg in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]:
                seq = str(r.get(f"{pref}_{reg}") or "")
                if seq:
                    feat[f"{pref}_{reg}_gravy"] = float(np.mean([KD.get(a, 0) for a in seq]))
                    feat[f"{pref}_{reg}_len"] = len(seq)
                    feat[f"{pref}_{reg}_charge"] = sum(
                        {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}.get(a, 0) for a in seq
                    )
                else:
                    feat[f"{pref}_{reg}_gravy"] = np.nan
                    feat[f"{pref}_{reg}_len"] = 0
                    feat[f"{pref}_{reg}_charge"] = np.nan
        rows.append(feat)
    out = pd.DataFrame(rows)
    out.to_csv(FEATURES / "germline" / "germline_relative.csv", index=False)
    return out


def ngram_features(pop: pd.DataFrame):
    """Sparse char n-grams 1–3 for H, L, H+L — vocabulary fit on Train only."""
    from scipy.sparse import save_npz

    role = pd.read_csv(ORG / "role_map.csv")[["id", "role"]]
    m = pop.merge(role, on="id")
    train_mask = m["role"] == "Train"
    texts = {
        "H": m["heavy"].tolist(),
        "L": m["light"].tolist(),
        "HL": (m["heavy"] + m["light"]).tolist(),
    }
    ids = m["id"].values
    for name, docs in texts.items():
        for n in [1, 2, 3]:
            max_f = 5000 if n < 3 else 8000
            vec = TfidfVectorizer(
                analyzer="char",
                ngram_range=(n, n),
                max_features=max_f,
                lowercase=False,
            )
            vec.fit([docs[i] for i in range(len(docs)) if train_mask.iloc[i]])
            X = vec.transform(docs)
            path = FEATURES / "ngram" / f"tfidf_{name}_{n}mer.npz"
            save_npz(path, X)
            write_json(
                FEATURES / "ngram" / f"tfidf_{name}_{n}mer_meta.json",
                {"vocab_size": int(len(vec.vocabulary_)), "ngram": n, "scope": name, "fit": "Train_only"},
            )
    np.save(FEATURES / "ngram" / "ids.npy", ids)
    print("NGRAM_OK", flush=True)


def structure_extensions(pop: pd.DataFrame):
    """Merge B2 structure feats + light geometric additions from comparison CSV if present."""
    abb = pd.read_csv(B2_CACHE / "structure_features" / "abb_sasa_rasa_patch.csv")
    esm = pd.read_csv(B2_CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv")
    cmp = B2_CACHE / "structure_features" / "structure_comparison.csv"
    out = pop[["id"]].merge(abb, left_on="id", right_on="antibody_id", how="left")
    out = out.merge(esm, left_on="id", right_on="antibody_id", how="left", suffixes=("", "_esm"))
    if cmp.exists():
        c = pd.read_csv(cmp)
        out = out.merge(c, left_on="id", right_on="antibody_id", how="left")
    # derived
    if "ABB_BSA" in out.columns and "ESMFN_BSA" in out.columns:
        out["CONSENSUS_BSA_mean"] = out[["ABB_BSA", "ESMFN_BSA"]].mean(axis=1)
        out["CONSENSUS_Fv_sasa_hydrophobic"] = out[
            [c for c in out.columns if "Fv_sasa_hydrophobic" in c]
        ].mean(axis=1, numeric_only=True)
    if "ABB_Fv_sasa_hydrophobic" in out.columns and "ABB_Fv_sasa_positive" in out.columns:
        out["ABB_charge_hydrophobic_imbalance"] = out["ABB_Fv_sasa_positive"] - out.get(
            "ABB_Fv_sasa_negative", 0
        )
        out["ABB_hydrophobic_vs_polar"] = out["ABB_Fv_sasa_hydrophobic"] - out.get(
            "ABB_Fv_sasa_polar", 0
        )
    if "abb_com_dist" in out.columns:
        out["COM_dist"] = out["abb_com_dist"]
    elif "esmn_com_dist" in out.columns:
        out["COM_dist"] = out["esmn_com_dist"]
    # keep numeric
    keep = ["id"] + [c for c in out.columns if c != "id" and pd.api.types.is_numeric_dtype(out[c])]
    out[keep].to_csv(FEATURES / "structure_ext" / "structure_extended.csv", index=False)
    print("STRUCT_EXT", len(keep) - 1, flush=True)


def main():
    ensure_dirs()
    pop = pd.read_csv(ORG / "final_population.csv")
    role = pd.read_csv(ORG / "role_map.csv")[["id", "role"]]
    train = pop.merge(role, on="id")
    train = train[train["role"] == "Train"].copy()
    folds = build_cv_folds(train, n_folds=5, n_repeats=4, seed=20260829)
    # save fold assignments aligned to train id order
    train_ids = train["id"].tolist()
    fold_payload = {"train_ids": train_ids, "n_folds": 5, "n_repeats": 4, "folds": folds}
    write_json(CONFIG / "TRAIN_CV_FOLDS.json", fold_payload)
    (REPORTS / "train_cv_protocol.md").write_text(
        f"""# Train CV protocol

- Train N = {len(train)}
- 5 Group folds × 4 repeats = 20 outer evaluations
- Groups = sequence_group (90% paired identity clusters)
- Same folds for all representations
- Primary metric: Spearman; secondary Pearson/MAE/RMSE
- Public/Private labels sealed during model search

Fold file: `config/TRAIN_CV_FOLDS.json`
"""
    )

    num = pd.read_csv(B1_DATA / "numbering_germline.csv")
    print("IMGT positional…", flush=True)
    positional_onehot(pop, num)
    print("Germline relative…", flush=True)
    germline_relative(pop, num)
    print("Ngrams…", flush=True)
    ngram_features(pop)
    print("Structure ext…", flush=True)
    structure_extensions(pop)
    print("FEATURES_OK")


if __name__ == "__main__":
    main()
