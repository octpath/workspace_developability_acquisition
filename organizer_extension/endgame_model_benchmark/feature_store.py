#!/usr/bin/env python3
"""Load frozen endgame feature blocks with canonical ID alignment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
SI = FP / "score_integrity"
sys.path.insert(0, str(SI))
from canonical_simple_tvt import FeatureAlignmentError, align_feature_block, concat_blocks  # noqa: E402

EMB_ROUND1 = ROOT / "virtual_participant/round1_finalization/cache/round1_embeddings.npz"
BIO = FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
M1 = FP / "structure_marathon/proteinmpnn/M1_FEATURES.csv"
ARO = FP / "structure_gap_closure/cache/aromatic_features_esmfold.csv"
CONT = FP / "structure_gap_closure/results/HIC_CONTINUOUS_SURFACE_FEATURES.csv"
TITR = FP / "TITRATION-SHAPE/features_esmfold.parquet"
HYDRO = FP / "HYDRO-FIELD/features_esmfold.parquet"
ABL_HL = FP / "ablingua600m/embeddings/ablingua600m_HL_mean_concat.parquet"
ABL_CDR3 = FP / "ablingua600m/embeddings_guided/ablingua600m_CDR3.parquet"
OPENMM = FP / "openmm_fab_md/results/OPENMM_FAB_MD_FEATURES_DEV.parquet"
OPENMM_MAP = FP / "openmm_fab_md/OPENMM_MD_FAMILY_MAP.csv"
FENNIX = FP / "fennix_fab_context_final/results/FENNIX_COMBINED_PREDECLARED.parquet"
DEV = ROOT / "competition/data/distribution/dev.csv"
TEST = ROOT / "competition/data/distribution/test_features.csv"
ANN_DEV = ROOT / "competition/data/distribution/dev_annotations.csv"
ANN_TEST = ROOT / "competition/data/distribution/test_annotations.csv"
REG_DEV = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"
REG_TEST = ROOT / "virtual_participant/round1_finalization/cache/anarci_imgt_regions_test.csv"

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
from run_stage1 import build_all_feature_tables, make_xy  # noqa: E402


def _emb_df(key: str, prefix: str) -> pd.DataFrame:
    z = np.load(EMB_ROUND1, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    X = pd.DataFrame(np.asarray(z[key], float), index=ids)
    X.columns = [f"{prefix}_{i:04d}" for i in range(X.shape[1])]
    return X


def _seq_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build SEQ_BASIC and SEQ_ALL for all 324 Abs with explicit ID index."""
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    # unify columns
    for df in (dev, test):
        if "heavy" not in df.columns and "vh_seq" in df.columns:
            df.rename(columns={"vh_seq": "heavy", "vl_seq": "light"}, inplace=True)
    all_seq = pd.concat([dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True)
    all_seq["id"] = all_seq["id"].astype(str)
    ann = pd.concat(
        [pd.read_csv(ANN_DEV), pd.read_csv(ANN_TEST)],
        ignore_index=True,
    )
    ann["id"] = ann["id"].astype(str)
    reg = pd.concat(
        [pd.read_csv(REG_DEV), pd.read_csv(REG_TEST)],
        ignore_index=True,
    )
    reg["id"] = reg["id"].astype(str)
    # align ann/reg to all_seq order
    ids = all_seq["id"].tolist()
    ann = ann.drop_duplicates("id").set_index("id").loc[ids].reset_index()
    reg = reg.drop_duplicates("id").set_index("id").loc[ids].reset_index()
    tables = build_all_feature_tables(all_seq, ann, reg)
    seq_b, _ = make_xy(tables, "SEQ_BASIC")
    seq_a, _ = make_xy(tables, "SEQ_ALL")
    seq_b = seq_b.copy()
    seq_a = seq_a.copy()
    seq_b.index = ids
    seq_a.index = ids
    seq_b.columns = [f"seqB_{c}" for c in seq_b.columns]
    seq_a.columns = [f"seqA_{c}" for c in seq_a.columns]
    if seq_b.isna().all().all() or seq_a.isna().all().all():
        raise FeatureAlignmentError("SEQ tables entirely NaN — ID alignment bug")
    return seq_b.astype(float), seq_a.astype(float)


class FeatureStore:
    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str) -> pd.DataFrame:
        if name == "SEQ_BASIC":
            b, _ = _seq_tables()
            return b
        if name == "SEQ_ALL":
            _, a = _seq_tables()
            return a
        if name == "AbLang2_HL_paired":
            return _emb_df("ablang2__HL_paired", "ablang2")
        if name == "ESM2_H":
            return _emb_df("esm2__H", "esm2h")
        if name == "BIOEMU_NEW_PAIRWISE":
            bio = pd.read_csv(BIO)
            bio["id"] = bio["id"].astype(str)
            cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
            return bio.set_index("id")[cols].apply(pd.to_numeric, errors="coerce")
        if name == "M1_PROTEINMPNN":
            m1 = pd.read_csv(M1)
            m1["id"] = m1["id"].astype(str)
            return m1.set_index("id")[["MPNN_native_score"]].apply(pd.to_numeric, errors="coerce")
        if name == "AbLingua_HL_mean":
            df = pd.read_parquet(ABL_HL)
            df["id"] = df["id"].astype(str)
            X = df.set_index("id").select_dtypes(include=[np.number]).astype(float)
            X.columns = [f"ablingua_global_{i:04d}" for i in range(X.shape[1])]
            return X
        if name == "AbLingua_CDR3":
            df = pd.read_parquet(ABL_CDR3)
            df["id"] = df["id"].astype(str)
            X = df.set_index("id").select_dtypes(include=[np.number]).astype(float)
            X.columns = [f"ablingua_cdr3_{i:04d}" for i in range(X.shape[1])]
            return X
        if name == "AROMATIC_TOPO":
            df = pd.read_csv(ARO)
            df["id"] = df["id"].astype(str)
            X = df.set_index("id").select_dtypes(include=[np.number]).astype(float)
            # drop diagnostic/error columns that are entirely NaN
            drop = [c for c in X.columns if c.lower() in ("error", "aro_error") or c.endswith("_error")]
            X = X.drop(columns=drop, errors="ignore")
            X.columns = [f"aro_{c}" for c in X.columns]
            return X
        if name == "CONTINUOUS_SURFACE":
            df = pd.read_csv(CONT)
            df["id"] = df["id"].astype(str)
            cols = [c for c in df.columns if c.startswith("fv_esmfold__")]
            return df.set_index("id")[cols].apply(pd.to_numeric, errors="coerce")
        if name == "TITRATION_SHAPE":
            df = pd.read_parquet(TITR)
            if "id" in df.columns:
                df["id"] = df["id"].astype(str)
                X = df.set_index("id")
            else:
                X = df
            drop = [c for c in X.columns if c in ("extraction_status", "status")]
            return X.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number]).astype(float)
        if name == "HYDRO_FIELD":
            df = pd.read_parquet(HYDRO)
            if "id" in df.columns:
                df["id"] = df["id"].astype(str)
                X = df.set_index("id")
            else:
                X = df
            drop = [c for c in X.columns if c in ("extraction_status", "status")]
            return X.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number]).astype(float)
        if name == "OPENMM_DOMAIN_FLEXIBILITY":
            ox = pd.read_parquet(OPENMM)
            ox["id"] = ox["id"].astype(str)
            X = ox.set_index("id").select_dtypes(include=[np.number]).astype(float)
            fm = pd.read_csv(OPENMM_MAP)
            cols = fm.loc[fm["ablation_family"].astype(str) == "DOMAIN_FLEXIBILITY", "feature"].astype(str)
            cols = [c for c in cols if c in X.columns]
            return X[cols]
        if name == "FENNIX_COMBINED_PREDECLARED":
            df = pd.read_parquet(FENNIX)
            df["id"] = df["id"].astype(str)
            return df.set_index("id").select_dtypes(include=[np.number]).astype(float)
        raise KeyError(name)

    def matrix_for_recipe(
        self,
        blocks: list[str],
        ids: list[str],
        *,
        allow_intersection: bool = False,
    ) -> tuple[pd.DataFrame, list[dict]]:
        if not blocks:
            return pd.DataFrame(index=ids), []
        parts = []
        metas = []
        use_ids = list(ids)
        for b in blocks:
            raw = self.get(b)
            if allow_intersection:
                a = align_feature_block(raw, use_ids, b, allow_intersection=True)
                use_ids = list(a.index)
                # realign previous parts
                parts = [p.loc[use_ids] for p in parts]
            else:
                a = align_feature_block(raw, use_ids, b, allow_intersection=False)
            aa = a.copy()
            aa.columns = [f"{b}__{c}" for c in aa.columns]
            parts.append(aa)
            metas.append(
                {
                    "block": b,
                    "shape": list(a.shape),
                    "finite_frac": a.attrs.get("finite_frac"),
                    "fingerprint": a.attrs.get("sha16"),
                }
            )
        X = pd.concat(parts, axis=1)
        if list(X.index) != list(use_ids):
            raise FeatureAlignmentError("recipe matrix index mismatch")
        return X, metas


def load_y(target: str) -> pd.Series:
    """DEV labels from official distribution/dev.csv."""
    dev = pd.read_csv(DEV)
    dev["id"] = dev["id"].astype(str)
    return dev.set_index("id")[target].astype(float)


def load_test_y(target: str) -> pd.Series:
    sol = pd.read_csv(ROOT / "competition/data/secret/solution.csv")
    sol["id"] = sol["id"].astype(str)
    return sol.set_index("id")[target].astype(float)


def load_test_mask() -> pd.DataFrame:
    sol = pd.read_csv(ROOT / "competition/data/secret/solution.csv")
    sol["id"] = sol["id"].astype(str)
    return sol[["id", "is_public", "is_private"]]
