#!/usr/bin/env python3
"""Bundle-local data loading for advanced models (ID-aligned)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    AA_TO_IDX,
    BUNDLE_ROOT,
    REGION_TO_IDX,
    RESIDUE_ROOT,
    UNK_AA,
)


class DataIntegrityError(RuntimeError):
    pass


@dataclass
class FoldMaps:
    primary: dict[str, int]
    shadow: dict[str, int]


def load_dev_test(dev_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .config import BUNDLE_ROOT  # local

    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    for name, df, cols in (
        ("dev", dev, ["id", "heavy", "light", "TmApp", "HIC"]),
        ("test", test, ["id", "heavy", "light"]),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataIntegrityError(f"{name} missing columns {missing}")
        df["id"] = df["id"].astype(str)
        if df["id"].duplicated().any():
            raise DataIntegrityError(f"{name} has duplicate ids")
        if len(df) != 162:
            raise DataIntegrityError(f"{name} N={len(df)} expected 162")
    if set(dev["id"]) & set(test["id"]):
        raise DataIntegrityError("dev/test ids not disjoint")
    return dev, test


def load_folds(path: Optional[Path] = None) -> FoldMaps:
    from .config import BUNDLE_ROOT

    path = path or (BUNDLE_ROOT / "folds.csv")
    folds = pd.read_csv(path)
    folds["id"] = folds["id"].astype(str)
    primary = (
        folds[folds["scheme"] == "primary"].set_index("id")["fold"].astype(int).to_dict()
    )
    shadow = (
        folds[folds["scheme"] == "shadow"].set_index("id")["fold"].astype(int).to_dict()
    )
    if len(primary) != 162 or len(shadow) != 162:
        raise DataIntegrityError("folds must cover 162 DEV ids each scheme")
    return FoldMaps(primary=primary, shadow=shadow)


def load_solution(path: Path) -> pd.DataFrame:
    sol = pd.read_csv(path)
    need = ["id", "TmApp", "HIC", "is_public", "is_private"]
    missing = [c for c in need if c not in sol.columns]
    if missing:
        raise DataIntegrityError(f"solution missing {missing}")
    sol["id"] = sol["id"].astype(str)
    if sol["id"].duplicated().any():
        raise DataIntegrityError("solution duplicate ids")
    if len(sol) != 162:
        raise DataIntegrityError("solution N != 162")
    pub = sol["is_public"].astype(bool)
    priv = sol["is_private"].astype(bool)
    if not (pub ^ priv).all():
        raise DataIntegrityError("is_public XOR is_private failed")
    if int(pub.sum()) != 81 or int(priv.sum()) != 81:
        raise DataIntegrityError("Public/Private must be 81/81")
    return sol


def load_annotations(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (RESIDUE_ROOT / "annotations.parquet")
    ann = pd.read_parquet(path)
    ann["id"] = ann["id"].astype(str)
    return ann


def build_imgt_vocab(ann: pd.DataFrame) -> dict[str, int]:
    positions = sorted(set(ann["imgt_position"].astype(str).tolist()) | {"PAD", "UNKNOWN"})
    vocab = {"PAD": 0}
    for p in positions:
        if p == "PAD":
            continue
        if p not in vocab:
            vocab[p] = len(vocab)
    if "UNKNOWN" not in vocab:
        vocab["UNKNOWN"] = len(vocab)
    return vocab


@dataclass
class ResidueBundle:
    ids: list[str]
    id_to_idx: dict[str, int]
    heavy_aa: np.ndarray
    light_aa: np.ndarray
    heavy_mask: np.ndarray
    light_mask: np.ndarray
    heavy_pos: np.ndarray
    light_pos: np.ndarray
    heavy_imgt: np.ndarray
    light_imgt: np.ndarray
    heavy_region: np.ndarray
    light_region: np.ndarray
    imgt_vocab: dict[str, int]
    ablingua_h: Optional[np.ndarray] = None
    ablingua_l: Optional[np.ndarray] = None
    ablingua_h_mask: Optional[np.ndarray] = None
    ablingua_l_mask: Optional[np.ndarray] = None
    ablang2_h: Optional[np.ndarray] = None
    ablang2_l: Optional[np.ndarray] = None
    ablang2_h_mask: Optional[np.ndarray] = None
    ablang2_l_mask: Optional[np.ndarray] = None
    esm2_h: Optional[np.ndarray] = None
    esm2_h_mask: Optional[np.ndarray] = None
    ablingua_hidden: int = 0
    ablang2_hidden: int = 0
    esm2_hidden: int = 0
    heavy_rasa: Optional[np.ndarray] = None  # [N, Lh] continuous RASA; pad/missing = NaN
    light_rasa: Optional[np.ndarray] = None


def _encode_chain(
    ann_ab: pd.DataFrame,
    seq: str,
    imgt_vocab: dict[str, int],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L = len(seq)
    aa = np.zeros(max_len, dtype=np.int64)
    mask = np.zeros(max_len, dtype=np.bool_)
    pos = np.zeros(max_len, dtype=np.int64)
    imgt = np.zeros(max_len, dtype=np.int64)
    region = np.zeros(max_len, dtype=np.int64)
    sub = ann_ab.sort_values("seq_index")
    if len(sub) != L:
        raise DataIntegrityError(f"annotation length {len(sub)} != seq {L}")
    got = "".join(sub["aa"].astype(str).tolist())
    if got != seq:
        raise DataIntegrityError("annotation AA != sequence")
    for i, r in enumerate(sub.itertuples(index=False)):
        aa[i] = AA_TO_IDX.get(str(r.aa), UNK_AA)
        mask[i] = True
        pos[i] = i + 1
        imgt[i] = imgt_vocab.get(str(r.imgt_position), imgt_vocab["UNKNOWN"])
        region[i] = REGION_TO_IDX.get(str(r.region), REGION_TO_IDX["UNKNOWN"])
    return aa, mask, pos, imgt, region


def load_npy(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    """Load .npy, or concatenate .npy.part0+.part1 if a size-split checkout is present."""
    path = Path(path)
    if path.exists():
        return np.load(path, allow_pickle=allow_pickle)
    p0, p1 = Path(str(path) + ".part0"), Path(str(path) + ".part1")
    if p0.exists() and p1.exists():
        data = p0.read_bytes() + p1.read_bytes()
        path.write_bytes(data)
        return np.load(path, allow_pickle=allow_pickle)
    raise FileNotFoundError(path)


def load_plm_pack(subdir: str) -> tuple[list[str], dict]:
    d = RESIDUE_ROOT / subdir
    ids = [str(x) for x in load_npy(d / "ids.npy", allow_pickle=True).tolist()]
    meta = json.loads((d / "metadata.json").read_text())
    return ids, meta


def load_residue_bundle(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    *,
    need_ablingua: bool = False,
    need_ablang2: bool = False,
    need_esm2: bool = False,
) -> ResidueBundle:
    seqs = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    seqs["id"] = seqs["id"].astype(str)
    ids = seqs["id"].tolist()
    ann = load_annotations()
    imgt_vocab = build_imgt_vocab(ann)
    max_h = int(seqs["heavy"].str.len().max())
    max_l = int(seqs["light"].str.len().max())
    N = len(ids)
    heavy_aa = np.zeros((N, max_h), dtype=np.int64)
    light_aa = np.zeros((N, max_l), dtype=np.int64)
    heavy_mask = np.zeros((N, max_h), dtype=np.bool_)
    light_mask = np.zeros((N, max_l), dtype=np.bool_)
    heavy_pos = np.zeros((N, max_h), dtype=np.int64)
    light_pos = np.zeros((N, max_l), dtype=np.int64)
    heavy_imgt = np.zeros((N, max_h), dtype=np.int64)
    light_imgt = np.zeros((N, max_l), dtype=np.int64)
    heavy_region = np.zeros((N, max_h), dtype=np.int64)
    light_region = np.zeros((N, max_l), dtype=np.int64)

    for i, row in enumerate(seqs.itertuples(index=False)):
        ab = str(row.id)
        sub = ann[ann.id == ab]
        ha, hm, hp, hi, hr = _encode_chain(
            sub[sub.chain == "H"], str(row.heavy), imgt_vocab, max_h
        )
        la, lm, lp, li, lr = _encode_chain(
            sub[sub.chain == "L"], str(row.light), imgt_vocab, max_l
        )
        heavy_aa[i], heavy_mask[i], heavy_pos[i], heavy_imgt[i], heavy_region[i] = (
            ha,
            hm,
            hp,
            hi,
            hr,
        )
        light_aa[i], light_mask[i], light_pos[i], light_imgt[i], light_region[i] = (
            la,
            lm,
            lp,
            li,
            lr,
        )

    rb = ResidueBundle(
        ids=ids,
        id_to_idx={a: i for i, a in enumerate(ids)},
        heavy_aa=heavy_aa,
        light_aa=light_aa,
        heavy_mask=heavy_mask,
        light_mask=light_mask,
        heavy_pos=heavy_pos,
        light_pos=light_pos,
        heavy_imgt=heavy_imgt,
        light_imgt=light_imgt,
        heavy_region=heavy_region,
        light_region=light_region,
        imgt_vocab=imgt_vocab,
    )

    if need_ablingua:
        a_ids, meta = load_plm_pack("ablingua600m")
        order = [a_ids.index(a) for a in ids]
        eh = load_npy(RESIDUE_ROOT / "ablingua600m/heavy_embeddings.npy")[order].astype(
            np.float32
        )
        el = load_npy(RESIDUE_ROOT / "ablingua600m/light_embeddings.npy")[order].astype(
            np.float32
        )
        mh = load_npy(RESIDUE_ROOT / "ablingua600m/heavy_mask.npy")[order]
        ml = load_npy(RESIDUE_ROOT / "ablingua600m/light_mask.npy")[order]
        rb.ablingua_h = eh[:, :max_h]
        rb.ablingua_l = el[:, :max_l]
        rb.ablingua_h_mask = mh[:, :max_h]
        rb.ablingua_l_mask = ml[:, :max_l]
        for i in range(N):
            if int(rb.ablingua_h_mask[i].sum()) != int(heavy_mask[i].sum()):
                raise DataIntegrityError(f"AbLingua H mask mismatch {ids[i]}")
            if int(rb.ablingua_l_mask[i].sum()) != int(light_mask[i].sum()):
                raise DataIntegrityError(f"AbLingua L mask mismatch {ids[i]}")
        rb.ablingua_hidden = int(meta["hidden_dim"])

    if need_ablang2:
        a_ids, meta = load_plm_pack("ablang2")
        order = [a_ids.index(a) for a in ids]
        eh = load_npy(RESIDUE_ROOT / "ablang2/heavy_embeddings.npy")[order].astype(
            np.float32
        )
        el = load_npy(RESIDUE_ROOT / "ablang2/light_embeddings.npy")[order].astype(
            np.float32
        )
        mh = load_npy(RESIDUE_ROOT / "ablang2/heavy_mask.npy")[order]
        ml = load_npy(RESIDUE_ROOT / "ablang2/light_mask.npy")[order]
        rb.ablang2_h = eh[:, :max_h]
        rb.ablang2_l = el[:, :max_l]
        rb.ablang2_h_mask = mh[:, :max_h]
        rb.ablang2_l_mask = ml[:, :max_l]
        for i in range(N):
            if int(rb.ablang2_h_mask[i].sum()) != int(heavy_mask[i].sum()):
                raise DataIntegrityError(f"AbLang2 H mask mismatch {ids[i]}")
            if int(rb.ablang2_l_mask[i].sum()) != int(light_mask[i].sum()):
                raise DataIntegrityError(f"AbLang2 L mask mismatch {ids[i]}")
        rb.ablang2_hidden = int(meta["hidden_dim"])

    if need_esm2:
        e_ids, meta = load_plm_pack("esm2")
        order = [e_ids.index(a) for a in ids]
        eh = load_npy(RESIDUE_ROOT / "esm2/heavy_embeddings.npy")[order].astype(np.float32)
        mh = load_npy(RESIDUE_ROOT / "esm2/heavy_mask.npy")[order]
        rb.esm2_h = eh[:, :max_h]
        rb.esm2_h_mask = mh[:, :max_h]
        for i in range(N):
            if int(rb.esm2_h_mask[i].sum()) != int(heavy_mask[i].sum()):
                raise DataIntegrityError(f"ESM2 H mask mismatch {ids[i]}")
        rb.esm2_hidden = int(meta["hidden_dim"])

    return rb


def attach_continuous_rasa(
    rb: ResidueBundle,
    *,
    rasa_heavy: np.ndarray,
    rasa_light: np.ndarray,
    rasa_ids: list[str],
    seqs: Optional[pd.DataFrame] = None,
) -> dict:
    """Align classical RASA cache to ResidueBundle; pad NaN outside real residues.

    Returns QC dict. Missing RASA on a real residue -> NaN (model treats as 0 contrib).
    If ``seqs`` provided (id,heavy,light), verify residue-token AA equals competition seq
    and that RASA array lengths cover those sequences (classical cache contract).
    """
    id_to_rasa = {str(a): i for i, a in enumerate(rasa_ids)}
    idx_to_aa = {v: k for k, v in AA_TO_IDX.items()}
    N, Lh = rb.heavy_mask.shape
    Ll = rb.light_mask.shape[1]
    heavy = np.full((N, Lh), np.nan, dtype=np.float32)
    light = np.full((N, Ll), np.nan, dtype=np.float32)
    qc = {
        "n_antibodies": N,
        "n_expected_residues": 0,
        "n_mapped_residues": 0,
        "n_missing_rasa_residues": 0,
        "n_aa_mismatch_flags": 0,
        "missing_by_id": {},
        "aa_mismatch_ids": [],
    }
    seq_map = None
    if seqs is not None:
        s = seqs.copy()
        s["id"] = s["id"].astype(str)
        seq_map = s.set_index("id")[["heavy", "light"]].to_dict("index")

    for i, ab in enumerate(rb.ids):
        if ab not in id_to_rasa:
            raise DataIntegrityError(f"RASA cache missing id {ab}")
        j = id_to_rasa[ab]
        rh = np.asarray(rasa_heavy[j], dtype=np.float32)
        rl = np.asarray(rasa_light[j], dtype=np.float32)
        nh = int(rb.heavy_mask[i].sum())
        nl = int(rb.light_mask[i].sum())
        if rh.shape[0] < nh or rl.shape[0] < nl:
            raise DataIntegrityError(f"RASA length short for {ab}: H {rh.shape[0]}<{nh}")

        if seq_map is not None:
            if ab not in seq_map:
                raise DataIntegrityError(f"sequence table missing id {ab}")
            heavy_seq = str(seq_map[ab]["heavy"])
            light_seq = str(seq_map[ab]["light"])
            if len(heavy_seq) != nh or len(light_seq) != nl:
                raise DataIntegrityError(
                    f"seq/mask length mismatch {ab}: H {len(heavy_seq)}!={nh} L {len(light_seq)}!={nl}"
                )
            dec_h = "".join(idx_to_aa.get(int(x), "?") for x in rb.heavy_aa[i, :nh])
            dec_l = "".join(idx_to_aa.get(int(x), "?") for x in rb.light_aa[i, :nl])
            if dec_h != heavy_seq or dec_l != light_seq:
                qc["n_aa_mismatch_flags"] += 1
                qc["aa_mismatch_ids"].append(ab)
                raise DataIntegrityError(
                    f"AA alignment ambiguity for {ab}: residue tokens != competition sequences"
                )

        h_slice = rh[:nh]
        l_slice = rl[:nl]
        miss_h = int(np.sum(~np.isfinite(h_slice)))
        miss_l = int(np.sum(~np.isfinite(l_slice)))
        qc["n_expected_residues"] += nh + nl
        qc["n_mapped_residues"] += (nh - miss_h) + (nl - miss_l)
        qc["n_missing_rasa_residues"] += miss_h + miss_l
        if miss_h or miss_l:
            qc["missing_by_id"][ab] = {"H": miss_h, "L": miss_l}
        heavy[i, :nh] = h_slice
        light[i, :nl] = l_slice
    rb.heavy_rasa = heavy
    rb.light_rasa = light
    return qc


def tvt_split(
    fold_map: dict[str, int], k: int, ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    test, val, train = [], [], []
    for ab in ids:
        f = int(fold_map[ab])
        if f == k:
            test.append(ab)
        elif f == (k + 1) % 5:
            val.append(ab)
        else:
            train.append(ab)
    if not test or not val or not train:
        raise DataIntegrityError(f"empty TVT split k={k}")
    return train, val, test
