#!/usr/bin/env python3
"""Region / CDR / RASA weighted pooling over residue embeddings."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "top_models_feature_bundle"
ANN_PATH = BUNDLE / "residue_level" / "annotations.parquet"
RES = BUNDLE / "residue_level"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha_df(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "id"]
    ids = df["id"].astype(str).tolist()
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    h = hashlib.sha256()
    h.update("|".join(cols).encode())
    for i in order:
        h.update(ids[i].encode())
        row = df.iloc[i]
        for c in cols:
            v = row[c]
            h.update(b"NaN" if pd.isna(v) else np.format_float_scientific(float(v), unique=True, trim="k").encode())
            h.update(b",")
        h.update(b"\n")
    return h.hexdigest()


def load_annotations() -> pd.DataFrame:
    ann = pd.read_parquet(ANN_PATH)
    ann["id"] = ann["id"].astype(str)
    return ann


def load_plm_pack(name: str) -> dict:
    """name in {ablang2, ablingua600m, esm2}."""
    d = RES / name
    ids = [str(x) for x in np.load(d / "ids.npy", allow_pickle=True)]
    out = {"ids": ids, "id_to_i": {a: i for i, a in enumerate(ids)}}
    if name == "esm2":
        out["H"] = np.load(d / "heavy_embeddings.npy")
        out["H_mask"] = np.load(d / "heavy_mask.npy").astype(bool)
        out["L"] = None
        out["L_mask"] = None
        out["hidden"] = out["H"].shape[-1]
    else:
        out["H"] = np.load(d / "heavy_embeddings.npy")
        out["L"] = np.load(d / "light_embeddings.npy")
        out["H_mask"] = np.load(d / "heavy_mask.npy").astype(bool)
        out["L_mask"] = np.load(d / "light_mask.npy").astype(bool)
        out["hidden"] = out["H"].shape[-1]
    return out


def _region_masks(ann: pd.DataFrame, ab_id: str, chain: str, L: int) -> dict[str, np.ndarray]:
    sub = ann[(ann["id"] == ab_id) & (ann["chain"] == chain)].sort_values("seq_index")
    reg = np.array(["UNK"] * L, dtype=object)
    for _, r in sub.iterrows():
        si = int(r["seq_index"])
        if 0 <= si < L:
            reg[si] = str(r["region"])
    masks = {
        "ALL": np.ones(L, bool),
        "FR": np.isin(reg, ["FR1", "FR2", "FR3", "FR4"]),
        "CDR": np.isin(reg, ["CDR1", "CDR2", "CDR3"]),
        "CDR1": reg == "CDR1",
        "CDR2": reg == "CDR2",
        "CDR3": reg == "CDR3",
    }
    return masks


def pool_weighted(emb: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """emb [L,D], mask/weights [L]."""
    w = np.where(mask, weights, 0.0).astype(np.float64)
    s = w.sum()
    if s <= 0:
        return np.zeros(emb.shape[-1], np.float32)
    return ((emb.astype(np.float64) * w[:, None]).sum(axis=0) / s).astype(np.float32)


def build_pooled_block(
    pack: dict,
    ann: pd.DataFrame,
    ids: list[str],
    *,
    chains: list[str],
    region: str,
    cdr_gamma: float = 1.0,
    cdr3_gamma: float | None = None,
    rasa: dict | None = None,
    rasa_power: float | None = None,
    prefix: str,
) -> pd.DataFrame:
    """Build fixed-length pooled features for ordered ids."""
    rows = []
    for ab in ids:
        i = pack["id_to_i"][ab]
        parts = []
        for ch in chains:
            if ch == "L" and pack["L"] is None:
                continue
            emb = pack["H"][i] if ch == "H" else pack["L"][i]
            m = pack["H_mask"][i] if ch == "H" else pack["L_mask"][i]
            L = int(m.sum()) if m.dtype == bool else int(np.sum(m > 0))
            # truncate to valid length
            emb_v = emb[:L]
            m_v = np.ones(L, bool)
            rmasks = _region_masks(ann, ab, ch, L)
            if region == "SPLIT_FR_CDR":
                # concat FR mean + CDR mean
                for key in ("FR", "CDR"):
                    w = np.ones(L, float)
                    vec = pool_weighted(emb_v, rmasks[key] & m_v, w)
                    parts.append(vec)
                continue
            if region == "SPLIT_CDR123":
                for key in ("CDR1", "CDR2", "CDR3"):
                    w = np.ones(L, float)
                    parts.append(pool_weighted(emb_v, rmasks[key] & m_v, w))
                continue
            sel = rmasks.get(region, rmasks["ALL"]) & m_v
            w = np.ones(L, float)
            # CDR weighting relative to FR (on ALL residues)
            if region == "ALL" and (cdr_gamma != 1.0 or cdr3_gamma is not None):
                w = np.ones(L, float)
                w[rmasks["CDR"]] = cdr_gamma
                if cdr3_gamma is not None:
                    w = np.ones(L, float)
                    w[rmasks["CDR3"]] = cdr3_gamma
            if rasa is not None and rasa_power is not None:
                idx = rasa["ids"].index(ab) if ab in rasa["ids"] else None
                if idx is not None:
                    rr = rasa["H"][idx] if ch == "H" else rasa["L"][idx]
                    rr = rr[:L]
                    clipped = np.clip(np.nan_to_num(rr, nan=0.0), 0.0, 1.0)
                    resolved = np.isfinite(rr)
                    rw = np.where(resolved, np.power(clipped, rasa_power), 0.0)
                    w = w * rw
                    sel = sel & resolved
            parts.append(pool_weighted(emb_v, sel, w))
        if not parts:
            vec = np.zeros(pack["hidden"], np.float32)
        else:
            vec = np.concatenate(parts, axis=0)
        rows.append(vec)
    mat = np.stack(rows, axis=0)
    cols = [f"{prefix}_{j}" for j in range(mat.shape[1])]
    out = pd.DataFrame(mat, columns=cols)
    out.insert(0, "id", ids)
    return out


def aromatic_rasa_summaries(dev: pd.DataFrame, test: pd.DataFrame, ann: pd.DataFrame, rasa: dict) -> pd.DataFrame:
    """Compact Y/F/W RASA summaries by region (H+L where available)."""
    all_df = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    all_df["id"] = all_df["id"].astype(str)
    arom = set("YFW")
    rows = []
    for _, r in all_df.iterrows():
        ab = r.id
        idx = rasa["ids"].index(ab)
        feats = {"id": ab}
        for ch, seq in (("H", str(r.heavy)), ("L", str(r.light))):
            rr = rasa[ch][idx][: len(seq)]
            L = len(seq)
            rmasks = _region_masks(ann, ab, ch, L)
            for reg_name, mask in (("ALL", rmasks["ALL"]), ("CDR", rmasks["CDR"]), ("CDR3", rmasks["CDR3"])):
                for aa in ("Y", "F", "W", "ARO"):
                    if aa == "ARO":
                        sel = mask & np.array([c in arom for c in seq])
                    else:
                        sel = mask & np.array([c == aa for c in seq])
                    vals = rr[sel]
                    vals = vals[np.isfinite(vals)]
                    key = f"{ch}_{reg_name}_{aa}"
                    feats[f"{key}_count"] = float(sel.sum())
                    feats[f"{key}_rasa_sum"] = float(np.clip(vals, 0, 1).sum()) if len(vals) else 0.0
                    feats[f"{key}_rasa_mean"] = float(np.clip(vals, 0, 1).mean()) if len(vals) else 0.0
                    feats[f"{key}_rasa_max"] = float(np.clip(vals, 0, 1).max()) if len(vals) else 0.0
        rows.append(feats)
    return pd.DataFrame(rows)
