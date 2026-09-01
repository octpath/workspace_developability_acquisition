#!/usr/bin/env python3
"""
Stage 3 — basic structure / surface features (participant-safe).
Frozen Stage0 CV; no Test/Public/Private; no FoldX/Rosetta/APBS.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path("/workspace_developability_acquisition")
DEV_PATH = ROOT / "competition/data/distribution/dev.csv"
ANN_PATH = ROOT / "competition/data/distribution/dev_annotations.csv"
CV_P = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_S = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
OUT = ROOT / "virtual_participant/stage3_structure"
CACHE = OUT / "cache"
OOF = OUT / "oof"
PLOTS = OUT / "plots"

ESMFOLD_DIR = ROOT / "esmfold_native"
G2_FEAT = ROOT / "gate_b2/cache/structure_features"
ABB_MAN = ROOT / "gate_b1/cache/structures/abb2_manifest.csv"
EMB_NPZ = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
REGIONS = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"

SEED = 42
N_TRIALS = 30
CONTACT_A = 8.0
CLASH_A = 2.5
RASA_BURY = 0.20

# Frozen sequence-only incumbents (Stage 3 start)
INC = {
    "TmApp": {
        "plm_only": {"primary": 2.8634, "shadow": 2.9803, "name": "Stage2b AbLang2 HL_paired raw Ridge"},
        "overall": {"primary": 2.7756, "shadow": 2.8316, "name": "Stage2 AbLang2+SEQ_BASIC"},
    },
    "HIC": {
        "plm_only": {"primary": 0.4552, "shadow": 0.4529, "name": "Stage2 ESM-2 Heavy SVR"},
        "overall": {"primary": 0.4485, "shadow": 0.4510, "name": "Stage2 ESM-2 Heavy+SEQ_ALL"},
    },
}

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
sys.path.insert(0, str(ROOT / "virtual_participant/stage2_plm/scripts"))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def aa1(resname):
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def pdb_seq_and_coords(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(pdb_path))
    chains = {}
    for model in st:
        for chain in model:
            seq, cas, resseqs = [], [], []
            for res in chain:
                if res.id[0] != " ":
                    continue
                seq.append(aa1(res.get_resname()))
                resseqs.append(res.id[1])
                cas.append(res["CA"].coord.copy() if "CA" in res else None)
            chains[chain.id] = {
                "seq": "".join(seq),
                "ca": cas,
                "resseq": resseqs,
                "n": len(seq),
            }
        break
    return chains


def map_hl(chains, vh_len, vl_len):
    """Return (H_id, L_id) mapping from chain IDs."""
    ids = list(chains.keys())
    if "H" in chains and "L" in chains:
        return "H", "L"
    if len(ids) < 2:
        return ids[0] if ids else None, None
    # prefer length match to VH/VL
    best = None
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if chains[a]["n"] == vh_len and chains[b]["n"] == vl_len:
                return a, b
            score = abs(chains[a]["n"] - vh_len) + abs(chains[b]["n"] - vl_len)
            if best is None or score < best[0]:
                best = (score, a, b)
    return best[1], best[2]


def rg_from_ca(cas):
    pts = np.array([c for c in cas if c is not None], float)
    if len(pts) < 2:
        return np.nan
    center = pts.mean(axis=0)
    return float(np.sqrt(((pts - center) ** 2).sum(axis=1).mean()))


def packing_from_cas(cas, contact_a=CONTACT_A, clash_a=CLASH_A):
    pts = np.array([c for c in cas if c is not None], float)
    n = len(pts)
    if n < 2:
        return {
            "n_ca": n,
            "n_contacts": 0,
            "contact_density": 0.0,
            "mean_nn_dist": np.nan,
            "clash_proxy": 0,
            "rg": np.nan,
            "compactness": np.nan,
        }
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    du = d[iu]
    n_contacts = int((du < contact_a).sum())
    clash = int((du < clash_a).sum())
    # nearest neighbor (exclude self)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    rg = rg_from_ca(cas)
    return {
        "n_ca": n,
        "n_contacts": n_contacts,
        "contact_density": float(n_contacts / max(n, 1)),
        "mean_nn_dist": float(nn.mean()),
        "clash_proxy": clash,
        "rg": rg,
        "compactness": float(rg / (n ** (1.0 / 3.0))) if np.isfinite(rg) else np.nan,
    }


def center_dist(cas_a, cas_b):
    a = np.array([c for c in cas_a if c is not None], float)
    b = np.array([c for c in cas_b if c is not None], float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def audit_structures(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    abb = pd.read_csv(ABB_MAN).set_index("antibody_id")
    rows = []
    cache_summary = []

    # ESMFold native
    esm_ok = 0
    for _, r in dev.iterrows():
        aid = r["id"]
        path = ESMFOLD_DIR / f"{aid}.pdb"
        rec = {
            "antibody_id": aid,
            "structure_source": "ESMFold_native",
            "pdb_path": str(path) if path.exists() else "",
            "exists": path.exists(),
            "heavy_match": False,
            "light_match": False,
            "vh_len_pdb": None,
            "vl_len_pdb": None,
            "chain_H": None,
            "chain_L": None,
            "n_chains": 0,
            "complete": False,
            "malformed": False,
            "notes": "",
        }
        if not path.exists():
            rec["notes"] = "missing_pdb"
            rows.append(rec)
            continue
        try:
            chains = pdb_seq_and_coords(path)
            rec["n_chains"] = len(chains)
            hid, lid = map_hl(chains, len(r["heavy"]), len(r["light"]))
            rec["chain_H"], rec["chain_L"] = hid, lid
            if hid is None or lid is None:
                rec["notes"] = "missing_chain"
                rec["malformed"] = True
            else:
                rec["vh_len_pdb"] = chains[hid]["n"]
                rec["vl_len_pdb"] = chains[lid]["n"]
                rec["heavy_match"] = chains[hid]["seq"] == r["heavy"]
                rec["light_match"] = chains[lid]["seq"] == r["light"]
                if not rec["heavy_match"] or not rec["light_match"]:
                    # allow IMGT/insertion numbering but sequence must match
                    rec["notes"] = "sequence_mismatch"
                else:
                    rec["complete"] = True
                    esm_ok += 1
        except Exception as e:
            rec["malformed"] = True
            rec["notes"] = f"parse_error:{type(e).__name__}"
        rows.append(rec)

    cache_summary.append(
        {
            "structure_source": "ESMFold_native",
            "N": esm_ok,
            "H/L_chains": "A=VH,B=VL (mapped by length/seq)",
            "sequence_match": f"{esm_ok}/162 exact VH+VL",
            "complete": f"{esm_ok}/162",
            "participant_safe": "yes (Dev seq → ESMFold, label-independent)",
            "reusable": "yes",
            "notes": f"path={ESMFOLD_DIR}/{{id}}.pdb; also mirrored under gate_b2 hash names",
        }
    )

    # ABodyBuilder2
    abb_ok = 0
    for _, r in dev.iterrows():
        aid = r["id"]
        rec = {
            "antibody_id": aid,
            "structure_source": "ABodyBuilder2",
            "pdb_path": "",
            "exists": False,
            "heavy_match": False,
            "light_match": False,
            "vh_len_pdb": None,
            "vl_len_pdb": None,
            "chain_H": None,
            "chain_L": None,
            "n_chains": 0,
            "complete": False,
            "malformed": False,
            "notes": "",
        }
        if aid not in abb.index:
            rec["notes"] = "not_in_manifest"
            rows.append(rec)
            continue
        path = Path(abb.loc[aid, "pdb_path"])
        rec["pdb_path"] = str(path)
        rec["exists"] = path.exists()
        if not path.exists():
            rec["notes"] = "missing_pdb"
            rows.append(rec)
            continue
        try:
            chains = pdb_seq_and_coords(path)
            rec["n_chains"] = len(chains)
            hid, lid = map_hl(chains, len(r["heavy"]), len(r["light"]))
            rec["chain_H"], rec["chain_L"] = hid, lid
            if hid is None or lid is None:
                rec["notes"] = "missing_chain"
                rec["malformed"] = True
            else:
                rec["vh_len_pdb"] = chains[hid]["n"]
                rec["vl_len_pdb"] = chains[lid]["n"]
                # ABB uses H/L IMGT numbering; sequence string compare
                rec["heavy_match"] = chains[hid]["seq"] == r["heavy"]
                rec["light_match"] = chains[lid]["seq"] == r["light"]
                if not rec["heavy_match"] or not rec["light_match"]:
                    # sometimes ABB drops C-term; record length delta
                    dh = abs(chains[hid]["n"] - len(r["heavy"]))
                    dl = abs(chains[lid]["n"] - len(r["light"]))
                    rec["notes"] = f"seq_len_delta_H{dh}_L{dl}"
                    # treat near-complete if lengths close and prefix matches
                    if (
                        chains[hid]["seq"] == r["heavy"][: chains[hid]["n"]]
                        and chains[lid]["seq"] == r["light"][: chains[lid]["n"]]
                        and dh <= 2
                        and dl <= 2
                    ):
                        rec["complete"] = True
                        abb_ok += 1
                        rec["notes"] += ";prefix_ok"
                    else:
                        rec["complete"] = False
                else:
                    rec["complete"] = True
                    abb_ok += 1
        except Exception as e:
            rec["malformed"] = True
            rec["notes"] = f"parse_error:{type(e).__name__}"
        rows.append(rec)

    cache_summary.append(
        {
            "structure_source": "ABodyBuilder2",
            "N": abb_ok,
            "H/L_chains": "H/L (IMGT-numbered)",
            "sequence_match": f"{abb_ok}/162 usable (exact or prefix±2)",
            "complete": f"{abb_ok}/162",
            "participant_safe": "yes (Dev seq → ABB2, label-independent)",
            "reusable": "yes",
            "notes": "gate_b1/cache/structures/abodybuilder2 + abb2_manifest.csv",
        }
    )

    quality = pd.DataFrame(rows)
    summary = pd.DataFrame(cache_summary)
    return quality, summary


def compute_packing_table(dev: pd.DataFrame, quality: pd.DataFrame, source: str) -> pd.DataFrame:
    q = quality[quality.structure_source == ("ESMFold_native" if source == "ESMFold" else "ABodyBuilder2")].set_index(
        "antibody_id"
    )
    out_rows = []
    for _, r in dev.iterrows():
        aid = r["id"]
        qr = q.loc[aid]
        feat = {"antibody_id": aid}
        if not qr.exists or not Path(qr.pdb_path).exists():
            out_rows.append(feat)
            continue
        chains = pdb_seq_and_coords(Path(qr.pdb_path))
        hid, lid = qr.chain_H, qr.chain_L
        if hid not in chains or lid not in chains:
            out_rows.append(feat)
            continue
        h = packing_from_cas(chains[hid]["ca"])
        l = packing_from_cas(chains[lid]["ca"])
        fv_cas = chains[hid]["ca"] + chains[lid]["ca"]
        fv = packing_from_cas(fv_cas)
        for k, v in fv.items():
            feat[f"Fv_{k}"] = v
        for k, v in h.items():
            feat[f"VH_{k}"] = v
        for k, v in l.items():
            feat[f"VL_{k}"] = v
        feat["VH_VL_center_dist"] = center_dist(chains[hid]["ca"], chains[lid]["ca"])
        # cross-chain contacts
        ha = np.array([c for c in chains[hid]["ca"] if c is not None], float)
        la = np.array([c for c in chains[lid]["ca"] if c is not None], float)
        if len(ha) and len(la):
            d = np.linalg.norm(ha[:, None, :] - la[None, :, :], axis=-1)
            feat["interface_contact_count"] = int((d < CONTACT_A).sum())
            feat["interface_contact_density"] = float(feat["interface_contact_count"] / (len(ha) + len(la)))
        else:
            feat["interface_contact_count"] = 0
            feat["interface_contact_density"] = 0.0
        out_rows.append(feat)
    return pd.DataFrame(out_rows)


def classify_col(name: str) -> str:
    cl = name.lower()
    if name in ("mean_confidence", "mean_plddt"):
        return "GLOBAL"
    if "patch" in cl:
        return "PATCH"
    if any(x in cl for x in ["bsa", "interface", "iface", "vh_vl_center", "interface_contact"]):
        return "INTERFACE"
    if any(
        x in cl
        for x in [
            "sasa_hydrophobic",
            "sasa_aromatic",
            "sasa_positive",
            "sasa_negative",
            "sasa_polar",
            "rasa_w_",
            "exposed_pos",
            "exposed_neg",
            "hydrophobic_sasa_exposed",
        ]
    ):
        return "SURFACE_CHEM"
    if "rasa" in cl or "n_exposed" in cl or "frac_rasa" in cl:
        return "RASA"
    if "sasa" in cl:
        return "SASA"
    if any(x in cl for x in ["contact", "clash", "nn_dist", "compactness", "rg", "n_ca", "packing"]):
        return "PACKING"
    if "n_res" in cl or cl.endswith("_empty"):
        return "GLOBAL"
    return "GLOBAL"


def load_precomputed(dev: pd.DataFrame, prefix: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.set_index("antibody_id").loc[dev["id"]].reset_index()
    rename = {}
    for c in raw.columns:
        if c == "antibody_id":
            continue
        if c.startswith(prefix):
            rename[c] = c[len(prefix) :]
        else:
            rename[c] = c
    return raw.rename(columns=rename)


def build_feature_tables(dev, packing_esm, packing_abb):
    esm = load_precomputed(dev, "ESMFN_", G2_FEAT / "esmfold_native_sasa_rasa_patch.csv")
    abb = load_precomputed(dev, "ABB_", G2_FEAT / "abb_sasa_rasa_patch.csv")

    # merge packing
    esm = esm.merge(packing_esm, on="antibody_id", how="left")
    abb = abb.merge(packing_abb, on="antibody_id", how="left")

    # ratios for surface chem
    def add_ratios(df):
        out = df.copy()
        if "Fv_total_sasa" in out.columns and "Fv_sasa_hydrophobic" in out.columns:
            tot = out["Fv_total_sasa"].replace(0, np.nan)
            out["Fv_hydrophobic_sasa_ratio"] = out["Fv_sasa_hydrophobic"] / tot
            out["Fv_aromatic_sasa_ratio"] = out["Fv_sasa_aromatic"] / tot
            out["Fv_charge_balance"] = out.get("Fv_sasa_positive", 0) - out.get("Fv_sasa_negative", 0)
            if "Fv_exposed_pos" in out.columns:
                out["Fv_exposed_charge_proxy"] = out["Fv_exposed_pos"] - out["Fv_exposed_neg"]
        # buried fraction from rasa
        if "Fv_frac_rasa_gt_0_20" in out.columns:
            out["Fv_buried_frac"] = 1.0 - out["Fv_frac_rasa_gt_0_20"]
        return out

    esm = add_ratios(esm)
    abb = add_ratios(abb)

    # drop near-all-null interface frac cols from causing silent zeros — keep but impute later
    sources = {"ESMFold": esm, "ABodyBuilder2": abb}
    families = {}
    manifest_rows = []
    for src, df in sources.items():
        fam_cols = {f: [] for f in ["GLOBAL", "SASA", "RASA", "SURFACE_CHEM", "PATCH", "INTERFACE", "PACKING"]}
        for c in df.columns:
            if c == "antibody_id":
                continue
            fam = classify_col(c)
            if fam not in fam_cols:
                fam = "GLOBAL"
            fam_cols[fam].append(c)
            miss = int(df[c].isna().sum())
            level = "Fv"
            cl = c.lower()
            if cl.startswith("vh") or cl.startswith("h_"):
                level = "VH"
            elif cl.startswith("vl") or cl.startswith("l_"):
                level = "VL"
            if "hcdr3" in cl or "h_cdr3" in cl:
                level = "HCDR3"
            elif "cdr" in cl:
                level = "CDR"
            if "iface" in cl or "bsa" in cl or "interface" in cl:
                level = "interface"
            manifest_rows.append(
                {
                    "feature_name": c,
                    "family": f"STRUCT_{fam}",
                    "structure_source": src,
                    "level": level,
                    "definition": c,
                    "unit": "mixed",
                    "missing_count": miss,
                    "label_independent": True,
                    "notes": "from gate_b2 SASA cache + Stage3 packing" if "contact" in cl or "rg" in cl else "gate_b2 precomputed",
                }
            )
        # composites
        fam_cols["SURFACE_ALL"] = sorted(
            set(fam_cols["SASA"] + fam_cols["RASA"] + fam_cols["SURFACE_CHEM"] + fam_cols["PATCH"])
        )
        fam_cols["GEOMETRY_ALL"] = sorted(set(fam_cols["GLOBAL"] + fam_cols["INTERFACE"] + fam_cols["PACKING"]))
        fam_cols["ALL"] = sorted(set(c for c in df.columns if c != "antibody_id"))
        families[src] = {"df": df, "cols": fam_cols}

    manifest = pd.DataFrame(manifest_rows)
    return families, manifest


class RareCat:
    def __init__(self, min_count=3):
        self.min_count = min_count
        self.keep_ = {}
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
        self.keep_ = {}
        for c in X.columns:
            vc = X[c].astype(str).value_counts()
            self.keep_[c] = set(vc[vc >= self.min_count].index)
        Xt = self._raw(X)
        self.columns_ = list(Xt.columns)
        return self

    def _raw(self, X):
        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            s = X[c].astype(str)
            s = s.where(s.isin(self.keep_[c]), other="__OTHER__")
            out = pd.concat([out, pd.get_dummies(s, prefix=c)], axis=1)
        return out

    def transform(self, X):
        Xt = self._raw(X)
        for c in self.columns_:
            if c not in Xt.columns:
                Xt[c] = 0
        return Xt[self.columns_].astype(float)


def make_model(kind, params):
    if kind == "Ridge":
        return Ridge(alpha=params.get("alpha", 10.0), random_state=0)
    if kind == "ElasticNet":
        return ElasticNet(
            alpha=params.get("alpha", 0.05),
            l1_ratio=params.get("l1_ratio", 0.3),
            max_iter=8000,
            tol=1e-3,
            random_state=0,
        )
    if kind == "SVR":
        return SVR(
            kernel="rbf",
            C=params.get("C", 1.0),
            gamma=params.get("gamma", 0.01),
            epsilon=params.get("epsilon", 0.1),
        )
    raise ValueError(kind)


def fold_prepare(X_struct, X_plm, classical, tr, va, pca_dim):
    """Fit imputer/scaler/(PCA) inside training fold."""
    parts_tr, parts_va = [], []
    n_feat = 0

    if X_struct is not None:
        imp = SimpleImputer(strategy="median")
        Str = imp.fit_transform(X_struct[tr])
        Sva = imp.transform(X_struct[va])
        sc = StandardScaler()
        Str = sc.fit_transform(Str)
        Sva = sc.transform(Sva)
        parts_tr.append(Str)
        parts_va.append(Sva)
        n_feat += Str.shape[1]

    if X_plm is not None:
        scp = StandardScaler()
        Ptr = scp.fit_transform(X_plm[tr])
        Pva = scp.transform(X_plm[va])
        if pca_dim is not None and pca_dim > 0:
            n_comp = min(int(pca_dim), Ptr.shape[0] - 1, Ptr.shape[1])
            if n_comp >= 2:
                pca = PCA(n_components=n_comp, random_state=0)
                Ptr = pca.fit_transform(Ptr)
                Pva = pca.transform(Pva)
        parts_tr.append(Ptr)
        parts_va.append(Pva)
        n_feat += Ptr.shape[1]

    if classical is not None:
        num = classical["num"]
        cat = classical["cat"]
        imp = SimpleImputer(strategy="median")
        Ntr = imp.fit_transform(num.iloc[tr])
        Nva = imp.transform(num.iloc[va])
        sc2 = StandardScaler()
        Ntr = sc2.fit_transform(Ntr)
        Nva = sc2.transform(Nva)
        if cat is not None:
            enc = RareCat(3)
            Ctr = enc.fit(cat.iloc[tr]).transform(cat.iloc[tr]).to_numpy()
            Cva = enc.transform(cat.iloc[va]).to_numpy()
            Ntr = np.hstack([Ntr, Ctr])
            Nva = np.hstack([Nva, Cva])
        parts_tr.append(Ntr)
        parts_va.append(Nva)
        n_feat += Ntr.shape[1]

    Xtr = np.hstack(parts_tr)
    Xva = np.hstack(parts_va)
    return Xtr, Xva, n_feat


def cv_run(y, folds, X_struct, X_plm, classical, model_kind, params, pca_dim, save_oof=None, ids=None):
    n_folds = int(folds.max()) + 1
    oof = np.zeros(len(y))
    fold_maes = []
    n_feat = 0
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        Xtr, Xva, n_feat = fold_prepare(X_struct, X_plm, classical, tr, va, pca_dim)
        model = make_model(model_kind, params)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xva)
        oof[va] = pred
        fold_maes.append(mae(y[va], pred))
    fold_maes = np.asarray(fold_maes, float)
    if save_oof is not None:
        pd.DataFrame(
            {
                "id": ids,
                "fold": folds,
                "y_true": y,
                "y_pred": oof,
                "residual": y - oof,
                "experiment_id": Path(save_oof).stem,
            }
        ).to_csv(save_oof, index=False)
    return {
        "mae": float(fold_maes.mean()),
        "fold_sd": float(fold_maes.std(ddof=1)),
        "fold_maes": fold_maes.tolist(),
        "oof": oof,
        "n_features": int(n_feat),
        "pearson": float(pearsonr(y, oof)[0]) if np.std(oof) > 1e-12 else float("nan"),
        "spearman": float(spearmanr(y, oof)[0]) if np.std(oof) > 1e-12 else float("nan"),
        "pred_sd": float(np.std(oof)),
    }


def optuna_search(y, folds, X_struct, X_plm, classical, model_kind, pca_choices, n_trials=N_TRIALS):
    def objective(trial):
        pca_dim = trial.suggest_categorical("pca_dim", pca_choices)
        if model_kind == "Ridge":
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True)}
        elif model_kind == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }
        else:
            params = {
                "C": trial.suggest_float("C", 0.1, 50.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            }
        res = cv_run(y, folds, X_struct, X_plm, classical, model_kind, params, pca_dim)
        return res["mae"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params)
    pca_dim = bp.pop("pca_dim")
    return pca_dim, bp, float(study.best_value)


def hic_tail_mae(y, oof, folds, q=0.9):
    """Train-fold q90 rule approximated by overall train per fold; report mean of fold tail MAEs."""
    n_folds = int(folds.max()) + 1
    vals = []
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        thr = np.quantile(y[tr], q)
        m = va & (y >= thr)
        if m.sum() == 0:
            continue
        vals.append(mae(y[m], oof[m]))
    return float(np.mean(vals)) if vals else float("nan")


def plot_obs_pred(y, oof, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y, oof, s=18, alpha=0.7, c="#2F4B7C")
    lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    ax.plot(lims, lims, "--", color="#D45087", lw=1)
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_cache_audit_md(summary: pd.DataFrame, quality: pd.DataFrame):
    esm = quality[quality.structure_source == "ESMFold_native"]
    abb = quality[quality.structure_source == "ABodyBuilder2"]
    lines = [
        "# Stage 3 — Structure Cache Audit",
        "",
        "参加者側で利用可能な予測立体構造キャッシュの監査。organizer score / ranking は参照していない。",
        "",
        "## 概要表",
        "",
        "| structure source | N | H/L chains | sequence match | complete | participant-safe | reusable | notes |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['structure_source']} | {r['N']} | {r['H/L_chains']} | {r['sequence_match']} | "
            f"{r['complete']} | {r['participant_safe']} | {r['reusable']} | {r['notes']} |"
        )
    lines += [
        "",
        "## ESMFold_native",
        f"- path: `{ESMFOLD_DIR}/{{id}}.pdb`",
        f"- exact VH+VL match: {int((esm.heavy_match & esm.light_match).sum())}/162",
        f"- complete flag: {int(esm.complete.sum())}/162",
        f"- malformed: {int(esm.malformed.sum())}",
        "",
        "## ABodyBuilder2",
        f"- manifest: `{ABB_MAN}`",
        f"- complete (exact or prefix±2): {int(abb.complete.sum())}/162",
        f"- exact match: {(abb.heavy_match & abb.light_match).sum()}/162",
        f"- malformed: {int(abb.malformed.sum())}",
        "",
        "## 採用方針",
        "",
        "- **Primary structure source**: ESMFold_native（Dev 162/162、ADI-named PDB、配列完全一致）",
        "- **Secondary**: ABodyBuilder2（source ablation用）。同一特徴抽出ロジックの結果を比較する",
        "- B1 Gly-linker ESMFold は使用しない（非ネイティブ連結）",
        "",
        "label-independent な gate_b2 SASA/RASA/patch 特徴を再利用し、packing / Rg を Stage3 で追加計算した。",
        "",
    ]
    (OUT / "STRUCTURE_CACHE_AUDIT_JA.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for d in [OUT, CACHE, OOF, PLOTS, OUT / "artifacts", OUT / "scripts"]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== Load data ===", flush=True)
    dev = pd.read_csv(DEV_PATH)
    ann = pd.read_csv(ANN_PATH)
    folds_p = pd.read_csv(CV_P)["fold"].to_numpy()
    folds_s = pd.read_csv(CV_S)["fold"].to_numpy()
    assert list(pd.read_csv(CV_P)["id"]) == list(dev["id"])
    y_tm = dev["TmApp"].to_numpy(float)
    y_hic = dev["HIC"].to_numpy(float)
    ids = list(dev["id"])

    print("=== Structure quality audit ===", flush=True)
    quality, summary = audit_structures(dev)
    quality.to_csv(OUT / "stage3_structure_quality.csv", index=False)
    write_cache_audit_md(summary, quality)
    print(summary.to_string(index=False), flush=True)

    print("=== Packing features ===", flush=True)
    pack_path_e = CACHE / "packing_esmfold.csv"
    pack_path_a = CACHE / "packing_abb.csv"
    if pack_path_e.exists():
        packing_esm = pd.read_csv(pack_path_e)
    else:
        packing_esm = compute_packing_table(dev, quality, "ESMFold")
        packing_esm.to_csv(pack_path_e, index=False)
    if pack_path_a.exists():
        packing_abb = pd.read_csv(pack_path_a)
    else:
        packing_abb = compute_packing_table(dev, quality, "ABB")
        packing_abb.to_csv(pack_path_a, index=False)

    print("=== Feature tables ===", flush=True)
    families, manifest = build_feature_tables(dev, packing_esm, packing_abb)
    manifest.to_csv(OUT / "stage3_feature_manifest.csv", index=False)
    for src, blob in families.items():
        blob["df"].to_csv(CACHE / f"features_{src}.csv", index=False)

    # Classical features for overall fusion
    from run_stage1 import build_all_feature_tables, make_xy

    regions = pd.read_csv(REGIONS)
    tables = build_all_feature_tables(dev, ann, regions)
    classical = {}
    for fam in ["SEQ_BASIC", "SEQ_ALL"]:
        Xn, Xc = make_xy(tables, fam)
        classical[fam] = {"num": Xn, "cat": Xc}

    emb = np.load(EMB_NPZ, allow_pickle=True)
    assert list(emb["ids"]) == ids
    plm_tm = emb["ablang2__HL_paired"].astype(np.float32)
    plm_hic = emb["esm2__H"].astype(np.float32)

    registry = []
    struct_only_rows = []
    primary_rows = []
    shadow_rows = []
    fusion_rows = []
    source_cmp = []
    residual_rows = []

    primary_src = "ESMFold"
    fam_order = ["GLOBAL", "SASA", "RASA", "SURFACE_CHEM", "PATCH", "INTERFACE", "PACKING", "SURFACE_ALL", "GEOMETRY_ALL", "ALL"]

    def X_from(src, fam):
        df = families[src]["df"]
        cols = families[src]["cols"][fam]
        return df[cols].to_numpy(dtype=float)

    # ---------- Structure-only screen (Primary) ----------
    print("=== Structure-only screen ===", flush=True)
    best_struct = {"TmApp": None, "HIC": None}
    oof_store = {}

    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        for fam in fam_order:
            Xs = X_from(primary_src, fam)
            for kind, params in [
                ("Ridge", {"alpha": 10.0}),
                ("ElasticNet", {"alpha": 0.05, "l1_ratio": 0.3}),
            ]:
                eid = f"{target}__{primary_src}__STRUCT_{fam}__{kind}"
                res = cv_run(y, folds_p, Xs, None, None, kind, params, None)
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "structure_source": primary_src,
                    "feature_family": f"STRUCT_{fam}",
                    "model": kind,
                    "hyperparameters": json.dumps(params),
                    "n_features": res["n_features"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_plm_only": res["mae"] - INC[target]["plm_only"]["primary"],
                    "delta_vs_sequence_incumbent": res["mae"] - INC[target]["overall"]["primary"],
                    "status": "STRUCTURE_ONLY_FIXED",
                    "notes": "fixed HP screen",
                }
                registry.append(row)
                struct_only_rows.append(row)
                primary_rows.append(row)
                print(f"  {eid}: {res['mae']:.4f}", flush=True)

            # Optuna Ridge + ElasticNet for major families
            if fam in ["SASA", "RASA", "SURFACE_CHEM", "PATCH", "INTERFACE", "PACKING", "SURFACE_ALL", "GEOMETRY_ALL", "ALL"]:
                for kind in ["Ridge", "ElasticNet"]:
                    pca_dim, bp, _ = optuna_search(y, folds_p, Xs, None, None, kind, [None], n_trials=N_TRIALS)
                    eid = f"{target}__{primary_src}__STRUCT_{fam}__{kind}Opt"
                    res = cv_run(
                        y,
                        folds_p,
                        Xs,
                        None,
                        None,
                        kind,
                        bp,
                        None,
                        save_oof=OOF / f"{eid}.csv",
                        ids=ids,
                    )
                    oof_store[eid] = res["oof"]
                    row = {
                        "experiment_id": eid,
                        "target": target,
                        "structure_source": primary_src,
                        "feature_family": f"STRUCT_{fam}",
                        "model": f"{kind}Opt",
                        "hyperparameters": json.dumps(bp),
                        "n_features": res["n_features"],
                        "Primary_MAE": res["mae"],
                        "Primary_fold_SD": res["fold_sd"],
                        "Shadow_MAE": np.nan,
                        "delta_vs_plm_only": res["mae"] - INC[target]["plm_only"]["primary"],
                        "delta_vs_sequence_incumbent": res["mae"] - INC[target]["overall"]["primary"],
                        "status": "STRUCTURE_ONLY_OPTUNA",
                        "notes": f"trials={N_TRIALS}",
                    }
                    registry.append(row)
                    struct_only_rows.append(row)
                    primary_rows.append(row)
                    print(f"  {eid}: {res['mae']:.4f} {bp}", flush=True)
                    if best_struct[target] is None or res["mae"] < best_struct[target]["Primary_MAE"]:
                        best_struct[target] = row | {"oof": res["oof"], "params": bp, "kind": kind, "fam": fam}

        # SVR on top family
        top_fam = best_struct[target]["fam"]
        Xs = X_from(primary_src, top_fam)
        pca_dim, bp, _ = optuna_search(y, folds_p, Xs, None, None, "SVR", [None], n_trials=N_TRIALS)
        eid = f"{target}__{primary_src}__STRUCT_{top_fam}__SVROpt"
        res = cv_run(y, folds_p, Xs, None, None, "SVR", bp, None, save_oof=OOF / f"{eid}.csv", ids=ids)
        oof_store[eid] = res["oof"]
        row = {
            "experiment_id": eid,
            "target": target,
            "structure_source": primary_src,
            "feature_family": f"STRUCT_{top_fam}",
            "model": "SVROpt",
            "hyperparameters": json.dumps(bp),
            "n_features": res["n_features"],
            "Primary_MAE": res["mae"],
            "Primary_fold_SD": res["fold_sd"],
            "Shadow_MAE": np.nan,
            "delta_vs_plm_only": res["mae"] - INC[target]["plm_only"]["primary"],
            "delta_vs_sequence_incumbent": res["mae"] - INC[target]["overall"]["primary"],
            "status": "STRUCTURE_ONLY_OPTUNA",
            "notes": "SVR on best family",
        }
        registry.append(row)
        struct_only_rows.append(row)
        primary_rows.append(row)
        if res["mae"] < best_struct[target]["Primary_MAE"]:
            best_struct[target] = row | {"oof": res["oof"], "params": bp, "kind": "SVR", "fam": top_fam}

    # ---------- Source ablation ----------
    print("=== Structure source ablation ===", flush=True)
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        for fam in ["SASA", "SURFACE_CHEM", "INTERFACE", "SURFACE_ALL", "ALL"]:
            for src in ["ESMFold", "ABodyBuilder2"]:
                Xs = X_from(src, fam)
                pca_dim, bp, _ = optuna_search(y, folds_p, Xs, None, None, "Ridge", [None], n_trials=20)
                res = cv_run(y, folds_p, Xs, None, None, "Ridge", bp, None)
                source_cmp.append(
                    {
                        "target": target,
                        "feature_family": f"STRUCT_{fam}",
                        "structure_source": src,
                        "model": "RidgeOpt",
                        "Primary_MAE": res["mae"],
                        "Primary_fold_SD": res["fold_sd"],
                        "n_features": res["n_features"],
                        "hyperparameters": json.dumps(bp),
                    }
                )
                print(f"  src {target} {src} {fam}: {res['mae']:.4f}", flush=True)

    # ---------- Fusion ----------
    print("=== Fusion vs sequence incumbents ===", flush=True)
    # promising structure families for fusion
    fusion_fams = {
        "TmApp": ["PACKING", "INTERFACE", "GEOMETRY_ALL", "SURFACE_ALL", "ALL"],
        "HIC": ["SASA", "RASA", "SURFACE_CHEM", "PATCH", "SURFACE_ALL", "ALL"],
    }
    # also include best structure-only family
    for t in ["TmApp", "HIC"]:
        bf = best_struct[t]["fam"]
        if bf not in fusion_fams[t]:
            fusion_fams[t] = [bf] + fusion_fams[t]

    best_plm_struct = {"TmApp": None, "HIC": None}
    best_overall_struct = {"TmApp": None, "HIC": None}

    # PLM reference OOFs (recompute with frozen HPs)
    print("  recompute PLM/overall reference OOF", flush=True)
    ref_oof = {}
    # TmApp PLM-only Stage2b
    res = cv_run(
        y_tm,
        folds_p,
        None,
        plm_tm,
        None,
        "Ridge",
        {"alpha": 54.034535219662935},
        None,
        save_oof=OOF / "REF_TmApp_PLM_only.csv",
        ids=ids,
    )
    ref_oof["TmApp_plm"] = res["oof"]
    print(f"  REF TmApp PLM {res['mae']:.4f} (expect ~2.8634)", flush=True)

    # HIC PLM-only Stage2
    res = cv_run(
        y_hic,
        folds_p,
        None,
        plm_hic,
        None,
        "SVR",
        {"C": 0.7581105308624507, "gamma": 0.0007535488042221058, "epsilon": 0.006901875186793278},
        48,
        save_oof=OOF / "REF_HIC_PLM_only.csv",
        ids=ids,
    )
    ref_oof["HIC_plm"] = res["oof"]
    print(f"  REF HIC PLM {res['mae']:.4f} (expect ~0.4552)", flush=True)

    # TmApp overall
    res = cv_run(
        y_tm,
        folds_p,
        None,
        plm_tm,
        classical["SEQ_BASIC"],
        "SVR",
        {"C": 18.189956991772025, "gamma": 0.0002907403114744388, "epsilon": 0.3568072708595925},
        48,
        save_oof=OOF / "REF_TmApp_overall.csv",
        ids=ids,
    )
    ref_oof["TmApp_overall"] = res["oof"]
    print(f"  REF TmApp overall {res['mae']:.4f} (expect ~2.7756)", flush=True)

    # HIC overall
    res = cv_run(
        y_hic,
        folds_p,
        None,
        plm_hic,
        classical["SEQ_ALL"],
        "SVR",
        {"C": 2.7132375976290684, "gamma": 0.00021035972553881986, "epsilon": 0.0634962699970796},
        8,
        save_oof=OOF / "REF_HIC_overall.csv",
        ids=ids,
    )
    ref_oof["HIC_overall"] = res["oof"]
    print(f"  REF HIC overall {res['mae']:.4f} (expect ~0.4485)", flush=True)

    for target, y, plm, pca_choices in [
        ("TmApp", y_tm, plm_tm, [None, 32, 48, 64]),
        ("HIC", y_hic, plm_hic, [None, 8, 32, 48, 64]),
    ]:
        for fam in fusion_fams[target]:
            Xs = X_from(primary_src, fam)
            # PLM + structure
            for kind in ["Ridge", "SVR"]:
                pca_dim, bp, _ = optuna_search(y, folds_p, Xs, plm, None, kind, pca_choices, n_trials=N_TRIALS)
                eid = f"{target}__FUSION_PLM__{primary_src}__STRUCT_{fam}__{kind}Opt"
                res = cv_run(
                    y, folds_p, Xs, plm, None, kind, bp, pca_dim, save_oof=OOF / f"{eid}.csv", ids=ids
                )
                oof_store[eid] = res["oof"]
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "structure_source": primary_src,
                    "feature_family": f"STRUCT_{fam}",
                    "model": f"PLM+{kind}Opt",
                    "hyperparameters": json.dumps({"pca_dim": pca_dim, **bp}),
                    "n_features": res["n_features"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_plm_only": res["mae"] - INC[target]["plm_only"]["primary"],
                    "delta_vs_sequence_incumbent": res["mae"] - INC[target]["overall"]["primary"],
                    "status": "FUSION_PLM_STRUCT",
                    "notes": "PLM + structure",
                }
                registry.append(row)
                fusion_rows.append(row)
                primary_rows.append(row)
                print(f"  {eid}: {res['mae']:.4f}", flush=True)
                if best_plm_struct[target] is None or res["mae"] < best_plm_struct[target]["Primary_MAE"]:
                    best_plm_struct[target] = row | {
                        "oof": res["oof"],
                        "params": bp,
                        "kind": kind,
                        "fam": fam,
                        "pca_dim": pca_dim,
                        "Xs": Xs,
                        "plm": plm,
                        "classical": None,
                    }

            # overall + structure
            clas = classical["SEQ_BASIC"] if target == "TmApp" else classical["SEQ_ALL"]
            for kind in ["Ridge", "SVR"]:
                pca_dim, bp, _ = optuna_search(y, folds_p, Xs, plm, clas, kind, pca_choices, n_trials=N_TRIALS)
                eid = f"{target}__FUSION_OVERALL__{primary_src}__STRUCT_{fam}__{kind}Opt"
                res = cv_run(
                    y, folds_p, Xs, plm, clas, kind, bp, pca_dim, save_oof=OOF / f"{eid}.csv", ids=ids
                )
                oof_store[eid] = res["oof"]
                row = {
                    "experiment_id": eid,
                    "target": target,
                    "structure_source": primary_src,
                    "feature_family": f"STRUCT_{fam}",
                    "model": f"OVERALL+{kind}Opt",
                    "hyperparameters": json.dumps({"pca_dim": pca_dim, **bp}),
                    "n_features": res["n_features"],
                    "Primary_MAE": res["mae"],
                    "Primary_fold_SD": res["fold_sd"],
                    "Shadow_MAE": np.nan,
                    "delta_vs_plm_only": res["mae"] - INC[target]["plm_only"]["primary"],
                    "delta_vs_sequence_incumbent": res["mae"] - INC[target]["overall"]["primary"],
                    "status": "FUSION_OVERALL_STRUCT",
                    "notes": "overall sequence incumbent + structure",
                }
                registry.append(row)
                fusion_rows.append(row)
                primary_rows.append(row)
                print(f"  {eid}: {res['mae']:.4f}", flush=True)
                if best_overall_struct[target] is None or res["mae"] < best_overall_struct[target]["Primary_MAE"]:
                    best_overall_struct[target] = row | {
                        "oof": res["oof"],
                        "params": bp,
                        "kind": kind,
                        "fam": fam,
                        "pca_dim": pca_dim,
                        "Xs": Xs,
                        "plm": plm,
                        "classical": clas,
                    }

    # ---------- Shadow ----------
    print("=== Shadow audit ===", flush=True)

    def shadow_eval(target, y, rowmeta, label):
        Xs = X_from(primary_src, rowmeta["fam"])
        res = cv_run(
            y,
            folds_s,
            Xs,
            rowmeta.get("plm"),
            rowmeta.get("classical"),
            rowmeta["kind"],
            rowmeta["params"],
            rowmeta.get("pca_dim"),
        )
        plm_d = res["mae"] - INC[target]["plm_only"]["shadow"]
        ov_d = res["mae"] - INC[target]["overall"]["shadow"]
        prim_gain = rowmeta["Primary_MAE"] < INC[target]["overall"]["primary"] - 1e-6
        # status vs relevant baseline
        if label.startswith("struct"):
            base_p = INC[target]["plm_only"]["primary"]
            base_s = INC[target]["plm_only"]["shadow"]
            # structure-only rarely beats PLM; status relative to itself
            status = "STRUCT_SHADOW_EVAL"
        elif label.startswith("plm"):
            base_p = INC[target]["plm_only"]["primary"]
            base_s = INC[target]["plm_only"]["shadow"]
            gain_p = rowmeta["Primary_MAE"] < base_p - 1e-6
            gain_s = res["mae"] < base_s - 1e-6
            if gain_p and gain_s:
                status = "STRUCT_SHADOW_CONFIRMED"
            elif gain_p and not gain_s:
                status = "STRUCT_PRIMARY_ONLY_GAIN"
            elif abs(rowmeta["Primary_MAE"] - base_p) < 0.005 and abs(res["mae"] - base_s) < 0.005:
                status = "STRUCT_EQUIVALENT"
            else:
                status = "STRUCT_NO_GAIN"
        else:
            base_p = INC[target]["overall"]["primary"]
            base_s = INC[target]["overall"]["shadow"]
            gain_p = rowmeta["Primary_MAE"] < base_p - 1e-6
            gain_s = res["mae"] < base_s - 1e-6
            if gain_p and gain_s:
                status = "STRUCT_SHADOW_CONFIRMED"
            elif gain_p and not gain_s:
                status = "STRUCT_PRIMARY_ONLY_GAIN"
            elif abs(rowmeta["Primary_MAE"] - base_p) < 0.005 and abs(res["mae"] - base_s) < 0.005:
                status = "STRUCT_EQUIVALENT"
            else:
                status = "STRUCT_NO_GAIN"
        srow = {
            "experiment_id": rowmeta["experiment_id"],
            "target": target,
            "label": label,
            "feature_family": rowmeta["feature_family"],
            "model": rowmeta["model"],
            "Primary_MAE": rowmeta["Primary_MAE"],
            "Shadow_MAE": res["mae"],
            "Shadow_fold_SD": res["fold_sd"],
            "delta_vs_plm_only_shadow": plm_d,
            "delta_vs_overall_shadow": ov_d,
            "status": status,
        }
        shadow_rows.append(srow)
        # update registry
        for r in registry:
            if r["experiment_id"] == rowmeta["experiment_id"]:
                r["Shadow_MAE"] = res["mae"]
                r["status"] = status
        print(f"  SHADOW {rowmeta['experiment_id']}: {res['mae']:.4f} [{status}]", flush=True)
        return srow, res

    # top structure-only (up to 3 per target)
    so_df = pd.DataFrame(struct_only_rows)
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        sub = so_df[(so_df.target == target) & (so_df.model.str.contains("Opt"))].sort_values("Primary_MAE").head(3)
        for _, r in sub.iterrows():
            fam = r.feature_family.replace("STRUCT_", "")
            kind = r.model.replace("Opt", "")
            params = json.loads(r.hyperparameters)
            meta = {
                "experiment_id": r.experiment_id,
                "feature_family": r.feature_family,
                "model": r.model,
                "Primary_MAE": r.Primary_MAE,
                "fam": fam,
                "kind": kind,
                "params": params,
                "pca_dim": None,
                "plm": None,
                "classical": None,
            }
            shadow_eval(target, y, meta, "structure_only")

        # top fusion PLM and overall
        fu = pd.DataFrame(fusion_rows)
        for status_prefix, lab in [("FUSION_PLM_STRUCT", "plm_struct"), ("FUSION_OVERALL_STRUCT", "overall_struct")]:
            sub = fu[(fu.target == target) & (fu.status == status_prefix)].sort_values("Primary_MAE").head(3)
            for _, r in sub.iterrows():
                fam = r.feature_family.replace("STRUCT_", "")
                hp = json.loads(r.hyperparameters)
                pca_dim = hp.pop("pca_dim", None)
                kind = "SVR" if "SVR" in r.model else "Ridge"
                meta = {
                    "experiment_id": r.experiment_id,
                    "feature_family": r.feature_family,
                    "model": r.model,
                    "Primary_MAE": r.Primary_MAE,
                    "fam": fam,
                    "kind": kind,
                    "params": hp,
                    "pca_dim": pca_dim,
                    "plm": plm_tm if target == "TmApp" else plm_hic,
                    "classical": (classical["SEQ_BASIC"] if target == "TmApp" else classical["SEQ_ALL"])
                    if "OVERALL" in r.model
                    else None,
                }
                shadow_eval(target, y, meta, lab)

    # ---------- Residual analysis ----------
    print("=== Residual complementarity ===", flush=True)
    feat_esm = families["ESMFold"]["df"].set_index("antibody_id").loc[ids]
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        seq_res = y - ref_oof[f"{target}_overall"]
        plm_res = y - ref_oof[f"{target}_plm"]
        so = best_struct[target]
        so_pred = so["oof"]
        so_res = y - so_pred
        for name, vec in [
            ("seq_residual_vs_struct_pred", so_pred),
            ("seq_residual_vs_struct_residual", so_res),
            ("plm_residual_vs_struct_pred", so_pred),
        ]:
            if np.std(vec) < 1e-12:
                corr = float("nan")
            else:
                corr = float(pearsonr(seq_res if "seq" in name else plm_res, vec)[0])
            residual_rows.append(
                {
                    "target": target,
                    "analysis": name,
                    "feature": "best_structure_only_oof",
                    "pearson": corr,
                    "notes": so["experiment_id"],
                }
            )
        # feature vs residual
        cols_check = [
            c
            for c in [
                "Fv_sasa_hydrophobic",
                "Fv_sasa_aromatic",
                "Fv_total_sasa",
                "BSA",
                "Fv_contact_density",
                "Fv_rg",
                "Fv_buried_frac",
                "H_CDR3_mean_rasa",
                "total_hydrophobic_patch_sasa",
                "largest_hydrophobic_patch_sasa",
            ]
            if c in feat_esm.columns
        ]
        for c in cols_check:
            x = feat_esm[c].to_numpy(float)
            m = np.isfinite(x)
            if m.sum() < 20:
                continue
            residual_rows.append(
                {
                    "target": target,
                    "analysis": "seq_residual_vs_structure_feature",
                    "feature": c,
                    "pearson": float(pearsonr(seq_res[m], x[m])[0]),
                    "notes": "overall sequence residual",
                }
            )
            residual_rows.append(
                {
                    "target": target,
                    "analysis": "plm_residual_vs_structure_feature",
                    "feature": c,
                    "pearson": float(pearsonr(plm_res[m], x[m])[0]),
                    "notes": "PLM-only residual",
                }
            )

        # HIC high-tail diagnostic
        if target == "HIC":
            for label, oofv in [
                ("plm_only", ref_oof["HIC_plm"]),
                ("overall", ref_oof["HIC_overall"]),
                ("best_struct", so["oof"]),
                ("best_plm_struct", best_plm_struct["HIC"]["oof"]),
                ("best_overall_struct", best_overall_struct["HIC"]["oof"]),
            ]:
                residual_rows.append(
                    {
                        "target": "HIC",
                        "analysis": "high_tail_q90_mae",
                        "feature": label,
                        "pearson": hic_tail_mae(y, oofv, folds_p, 0.9),
                        "notes": f"pred_sd={np.std(oofv):.4f}; overall_mae={mae(y,oofv):.4f}",
                    }
                )

    # plots
    for target, y in [("TmApp", y_tm), ("HIC", y_hic)]:
        plot_obs_pred(y, best_struct[target]["oof"], f"{target} best structure-only", PLOTS / f"{target}_struct_only.png")
        plot_obs_pred(
            y, best_plm_struct[target]["oof"], f"{target} best PLM+struct", PLOTS / f"{target}_plm_struct.png"
        )
        plot_obs_pred(
            y,
            best_overall_struct[target]["oof"],
            f"{target} best overall+struct",
            PLOTS / f"{target}_overall_struct.png",
        )

    # save tables
    pd.DataFrame(registry).to_csv(OUT / "stage3_model_registry.csv", index=False)
    pd.DataFrame(struct_only_rows).to_csv(OUT / "stage3_structure_only_results.csv", index=False)
    pd.DataFrame(source_cmp).to_csv(OUT / "stage3_structure_source_comparison.csv", index=False)
    pd.DataFrame(primary_rows).to_csv(OUT / "stage3_primary_results.csv", index=False)
    pd.DataFrame(shadow_rows).to_csv(OUT / "stage3_shadow_results.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(OUT / "stage3_fusion_results.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "stage3_residual_analysis.csv", index=False)

    # shadow status lookup
    def shadow_status_for(eid):
        for r in shadow_rows:
            if r["experiment_id"] == eid:
                return r["status"], r["Shadow_MAE"]
        return "NOT_SHADOWED", None

    best_models = {}
    for target in ["TmApp", "HIC"]:
        bs = best_struct[target]
        bp = best_plm_struct[target]
        bo = best_overall_struct[target]
        st_s, sh_s = shadow_status_for(bs["experiment_id"])
        st_p, sh_p = shadow_status_for(bp["experiment_id"])
        st_o, sh_o = shadow_status_for(bo["experiment_id"])
        # overall update decision
        update = False
        if st_o == "STRUCT_SHADOW_CONFIRMED" and bo["Primary_MAE"] < INC[target]["overall"]["primary"]:
            update = True
        best_models[target] = {
            "sequence_reference_plm_only": INC[target]["plm_only"],
            "sequence_reference_overall": INC[target]["overall"],
            "best_structure_only": {
                "experiment_id": bs["experiment_id"],
                "family": bs["feature_family"],
                "model": bs["model"],
                "Primary_MAE": bs["Primary_MAE"],
                "Shadow_MAE": sh_s,
                "status": st_s,
            },
            "best_plm_structure": {
                "experiment_id": bp["experiment_id"],
                "family": bp["feature_family"],
                "model": bp["model"],
                "Primary_MAE": bp["Primary_MAE"],
                "Shadow_MAE": sh_p,
                "delta_vs_plm_only": bp["delta_vs_plm_only"],
                "status": st_p,
            },
            "best_overall_structure_fusion": {
                "experiment_id": bo["experiment_id"],
                "family": bo["feature_family"],
                "model": bo["model"],
                "Primary_MAE": bo["Primary_MAE"],
                "Shadow_MAE": sh_o,
                "delta_vs_sequence_incumbent": bo["delta_vs_sequence_incumbent"],
                "status": st_o,
            },
            "update_sequence_incumbent": update,
            "shadow_status": st_o if update else (st_p if st_p == "STRUCT_SHADOW_CONFIRMED" else st_o),
        }

    (OUT / "stage3_best_models.json").write_text(json.dumps(best_models, indent=2), encoding="utf-8")
    # dump meta for report writer
    (OUT / "artifacts" / "run_summary.json").write_text(
        json.dumps(
            {
                "best_models": best_models,
                "incumbents": INC,
                "primary_source": primary_src,
                "quality": {
                    "esm_complete": int(quality[quality.structure_source == "ESMFold_native"].complete.sum()),
                    "abb_complete": int(quality[quality.structure_source == "ABodyBuilder2"].complete.sum()),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== DONE modeling ===", flush=True)
    print(json.dumps(best_models, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
