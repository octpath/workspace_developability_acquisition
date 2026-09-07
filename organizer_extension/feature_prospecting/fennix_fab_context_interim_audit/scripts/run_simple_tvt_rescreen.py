#!/usr/bin/env python3
"""Competition-oriented Simple TVT rescreen of all frozen feature blocks.

FREE_ALPHA + BASE_FIXED_ALPHA. Does NOT touch FeNNix production worker.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
GAP = FP / "structure_gap_closure"
OUT = FP / "fennix_fab_context_interim_audit"
RES = OUT / "results"
OOF_DIR = ROOT / "virtual_participant/stage5_integration/oof"
PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
EMB = ROOT / "virtual_participant/stage2_plm/cache/stage2_embeddings.npz"
ARO = GAP / "cache/aromatic_features_esmfold.csv"
DEV = ROOT / "competition/data/distribution/dev.csv"
ANN = ROOT / "competition/data/distribution/dev_annotations.csv"
REGIONS = ROOT / "virtual_participant/stage1_features/cache/anarci_imgt_regions_dev.csv"
TMAPP_REF = "TmApp__META_performance__ridge_100.0"
HIC_REF = "HIC__SIMPLE_blend_seq_surf_adv"
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_CAP = 32
DIM_PCA_TRIGGER = 200
MIN_TR, MIN_VA, MIN_TE = 20, 5, 5
PREV_SIMPLE = OUT / "SIMPLE_TVT_CV_RESULTS.csv"

sys.path.insert(0, str(ROOT / "virtual_participant/stage1_features/scripts"))
from run_stage1 import build_all_feature_tables, make_xy  # noqa: E402


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def load_y(target):
    name = TMAPP_REF if target == "TmApp" else HIC_REF
    s = pd.read_csv(OOF_DIR / f"{name}.csv").set_index("id")["y_true"].astype(float)
    s.index = s.index.astype(str)
    return s


def numeric_X(df, drop_cols=None, col_filter=None, max_cols=None):
    drop_cols = set(drop_cols or []) | {
        "generator",
        "pdb_path",
        "pdb_sha256",
        "extraction_status",
        "status",
        "normalized_note",
        "shape_complementarity_status",
    }
    X = df.copy()
    if "id" in X.columns:
        X = X.set_index("id")
    elif "antibody_id" in X.columns:
        X = X.set_index("antibody_id")
    X.index = X.index.astype(str)
    for c in list(X.columns):
        if c in drop_cols:
            X = X.drop(columns=[c])
    if col_filter is not None:
        keep = [c for c in X.columns if col_filter(c)]
        X = X[keep]
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    # drop count-like
    cols = [
        c
        for c in X.columns
        if not str(c).endswith("_n")
        and not str(c).endswith("_n_sites")
        and str(c) not in ("n_sites", "VH_n", "VL_n", "nphys_used", "MPNN_n_res")
    ]
    X = X[cols]
    if max_cols is not None and X.shape[1] > max_cols:
        X = X.iloc[:, :max_cols]
    return X


def read_any(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_bases():
    dev = pd.read_csv(DEV)
    ids = [str(x) for x in dev["id"]]
    emb = np.load(EMB, allow_pickle=True)
    assert [str(x) for x in emb["ids"]] == ids
    ablang = pd.DataFrame(np.asarray(emb["ablang2__HL_paired"], float), index=ids)
    ablang.columns = [f"ablang2_{i}" for i in range(ablang.shape[1])]
    esm2h = pd.DataFrame(np.asarray(emb["esm2__H"], float), index=ids)
    esm2h.columns = [f"esm2h_{i}" for i in range(esm2h.shape[1])]
    tables = build_all_feature_tables(dev, pd.read_csv(ANN), pd.read_csv(REGIONS))
    seq_b, _ = make_xy(tables, "SEQ_BASIC")
    seq_a, _ = make_xy(tables, "SEQ_ALL")
    seq_b.index = ids
    seq_a.index = ids
    seq_b.columns = [f"seqB_{c}" for c in seq_b.columns]
    seq_a.columns = [f"seqA_{c}" for c in seq_a.columns]
    tmapp = pd.concat([ablang, seq_b], axis=1)
    hic = pd.concat([esm2h, seq_a], axis=1)
    aro = numeric_X(pd.read_csv(ARO))
    aro = aro.reindex(ids)
    aro.columns = [f"aro_{c}" for c in aro.columns]
    hic_aro = pd.concat([hic, aro], axis=1)
    return {
        "TmApp_BASE": tmapp,
        "HIC_BASE": hic,
        "HIC_BASE_ARO": hic_aro,
    }, {
        "TmApp_BASE": "ABLANG2_HL_PAIRED+SEQ_BASIC",
        "HIC_BASE": "ESM2_H+SEQ_ALL",
        "HIC_BASE_ARO": "ESM2_H+SEQ_ALL+AROMATIC_TOPO",
    }


def interim_usable():
    complete = pd.read_csv(OUT / "INTERIM_FENNIX_COMPLETE_SET.csv")
    complete["id"] = complete.id.astype(str)
    return set(complete.loc[complete.usable_interim.astype(bool) & complete.is_dev.astype(bool), "id"])


def fennix_v2_wide():
    df = pd.read_csv(FP / "foundation_stability_v2/FENNIX_V2_CURVATURE_FEATURES.csv")
    df = df[(df.generator == "esmfold") & (df.extraction_status == "SUCCESS")].copy()
    return numeric_X(df, drop_cols=["generator", "extraction_status"])


def build_family_catalog():
    """Return list of dict specs for evaluation."""
    usable = interim_usable()
    bio = pd.read_csv(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv")
    contact_cols = [c for c in bio.columns if "contact" in c.lower()]
    pair_cols = [c for c in bio.columns if "ca_rmsd" in c.lower()]
    flex_cols = [c for c in bio.columns if "rmsf" in c.lower()]
    shape_cols = [c for c in bio.columns if "_rg_" in c.lower() or c.endswith("_rg_mean") or c.endswith("_rg_sd")]

    pack = numeric_X(pd.read_csv(GAP / "results/TMAPP_PACKING_CAVITY_FEATURES.csv"))
    unsat = numeric_X(pd.read_csv(GAP / "results/TMAPP_BURIED_UNSAT_FEATURES.csv"))
    iface = numeric_X(pd.read_csv(GAP / "results/TMAPP_INTERFACE_FEATURES.csv"), drop_cols=["shape_complementarity_status"])
    core = pd.concat([pack, unsat], axis=1, join="inner")
    core = core.loc[:, ~core.columns.duplicated()]
    gap_all = pd.concat([core, iface], axis=1, join="inner")
    gap_all = gap_all.loc[:, ~gap_all.columns.duplicated()]
    surf = pd.read_csv(GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv")
    cont = numeric_X(surf[["id"] + [c for c in surf.columns if c.startswith("fv_esmfold__")]])
    aro = numeric_X(pd.read_csv(ARO))
    hic_surf_all = pd.concat([cont, aro], axis=1, join="inner")
    hic_surf_all = hic_surf_all.loc[:, ~hic_surf_all.columns.duplicated()]

    prev = {}
    if PREV_SIMPLE.exists():
        old = pd.read_csv(PREV_SIMPLE)
        for _, r in old.iterrows():
            prev[r.family] = f"PΔ={r.primary_delta:+.4f}/SΔ={r.shadow_delta:+.4f}/{r.verdict}"

    specs = []

    def add(target, family, source, path, X, base_key, scope, prev_eval, notes="", id_filter=None, label=""):
        already = family in prev or family.replace("INTERIM_", "") in prev or family.replace("GAP_", "") in prev
        # map interim names
        key_prev = family
        for k in prev:
            if k in family or family.endswith(k) or family == k:
                key_prev = k
                already = True
                break
        specs.append(
            {
                "target": target,
                "family": family,
                "source": source,
                "feature_path": path,
                "X": X,
                "base_key": base_key,
                "molecular_scope": scope,
                "previous_evaluation_type": prev_eval,
                "simple_tvt_already_run": bool(already and key_prev in prev),
                "simple_tvt_result_if_any": prev.get(key_prev, prev.get(family, "")),
                "available_N": int(X.shape[0]),
                "dimension": int(X.shape[1]),
                "notes": notes,
                "id_filter": id_filter,
                "label": label,
            }
        )

    # BioEmu
    add("TmApp", "BIOEMU_V12", "foundation_stability_v2", str(FP / "foundation_stability_v2/BIOEMU_V12_FEATURES.csv"),
        numeric_X(pd.read_csv(FP / "foundation_stability_v2/BIOEMU_V12_FEATURES.csv")), "TmApp_BASE", "VH+VL isolated",
        "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN prior")
    add("TmApp", "BIOEMU_NEW_CONTACT", "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
        numeric_X(bio[["id"] + contact_cols]), "TmApp_BASE", "VH+VL isolated", "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN")
    add("TmApp", "BIOEMU_NEW_PAIRWISE", "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
        numeric_X(bio[["id"] + pair_cols]), "TmApp_BASE", "VH+VL isolated", "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN")
    add("TmApp", "BIOEMU_NEW_FLEX", "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
        numeric_X(bio[["id"] + flex_cols]), "TmApp_BASE", "VH+VL isolated", "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN")
    add("TmApp", "BIOEMU_NEW_SHAPE", "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
        numeric_X(bio[["id"] + shape_cols]), "TmApp_BASE", "VH+VL isolated", "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN")
    add("TmApp", "BIOEMU_NEW_COMBINED", "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
        numeric_X(bio, drop_cols=["nphys_used", "VH_n", "VL_n"]), "TmApp_BASE", "VH+VL isolated", "standalone/residual/OOF", "SIMPLE_TVT_NOT_YET_RUN")
    # exploratory HIC BioEmu
    for fam, cols in [("BIOEMU_NEW_CONTACT", contact_cols), ("BIOEMU_NEW_PAIRWISE", pair_cols), ("BIOEMU_NEW_COMBINED", None)]:
        X = numeric_X(bio[["id"] + cols]) if cols else numeric_X(bio, drop_cols=["nphys_used", "VH_n", "VL_n"])
        add("HIC", fam, "bioemu_isolated_reassessment", str(FP / "bioemu_isolated_reassessment/BIOEMU_ISOLATED_REASSESS_FEATURES.csv"),
            X, "HIC_BASE_ARO", "VH+VL isolated", "exploratory", "EXPLORATORY_HIC_BIOEMU", label="EXPLORATORY_HIC_BIOEMU")

    # FeNNix v2 Fv
    add("TmApp", "FENNIX_V2_CURVATURE", "foundation_stability_v2", str(FP / "foundation_stability_v2/FENNIX_V2_CURVATURE_FEATURES.csv"),
        fennix_v2_wide(), "TmApp_BASE", "Fv", "standalone/residual/OOF", "esmfold SUCCESS only; SIMPLE_TVT_NOT_YET_RUN")

    # Gap closure
    add("TmApp", "PACKING_CAVITY", "structure_gap_closure", str(GAP / "results/TMAPP_PACKING_CAVITY_FEATURES.csv"), pack, "TmApp_BASE", "Fab", "simple_tvt+strict")
    add("TmApp", "BURIED_UNSAT", "structure_gap_closure", str(GAP / "results/TMAPP_BURIED_UNSAT_FEATURES.csv"), unsat, "TmApp_BASE", "Fab", "simple_tvt+strict")
    add("TmApp", "FAB_INTERFACE", "structure_gap_closure", str(GAP / "results/TMAPP_INTERFACE_FEATURES.csv"), iface, "TmApp_BASE", "Fab", "simple_tvt+strict")
    add("TmApp", "CORE_DEFECT", "structure_gap_closure", "derived:pack+unsat", core, "TmApp_BASE", "Fab", "simple_tvt+strict")
    add("TmApp", "GAP_ALL", "structure_gap_closure", "derived:core+iface", gap_all, "TmApp_BASE", "Fab", "simple_tvt+strict")
    add("HIC", "CONTINUOUS_SURFACE", "structure_gap_closure", str(GAP / "results/HIC_CONTINUOUS_SURFACE_FEATURES.csv"), cont, "HIC_BASE_ARO", "Fv", "simple_tvt")
    add("HIC", "HIC_SURFACE_ALL", "structure_gap_closure", "derived:cont+aro", hic_surf_all, "HIC_BASE", "Fv", "simple_tvt+strict", "BASE without ARO to avoid duplicate")
    add("HIC", "AROMATIC_TOPO", "AROMATIC-TOPO", str(FP / "AROMATIC-TOPO/features_esmfold.parquet"),
        numeric_X(read_any(FP / "AROMATIC-TOPO/features_esmfold.parquet"), drop_cols=["extraction_status"]), "HIC_BASE", "Fv", "standalone/residual", "BASE without ARO")

    # Interim FeNNix
    for name in ["DELTA_GEOM", "DELTA_ENV", "CONSTANT", "INTERFACE", "FULL_FAB_NORMALIZED", "COMBINED_PREDECLARED", "PREP_RELAX_SENSITIVITY"]:
        p = RES / f"INTERIM_{name}_FEATURES.csv"
        add("TmApp", f"INTERIM_{name}", "fennix_fab_context_interim_audit", str(p), numeric_X(pd.read_csv(p)),
            "TmApp_BASE", "Fab", "simple_tvt/strict provisional", "PROVISIONAL_SIMPLE_CV", id_filter=usable, label="PROVISIONAL_SIMPLE_CV")

    # Structure marathon
    marathon = [
        ("S1_STRUCTURE_GUIDED_POOLING", "structure_marathon/structure_guided_pooling/S1_FEATURES.csv", "BOTH", "Fv"),
        ("S2_GENERATOR_DISAGREEMENT", "structure_marathon/generator_disagreement/S2_DISAGREEMENT_FEATURES.csv", "BOTH", "Fv"),
        ("S3_SURFACE_PATCH_GRAPH", "structure_marathon/surface_patch_graph/S3_FEATURES.csv", "BOTH", "Fv"),
        ("S4_CONTACT_GRAPH", "structure_marathon/contact_graph/S4_FEATURES.csv", "BOTH", "Fv"),
        ("M1_PROTEINMPNN", "structure_marathon/proteinmpnn/M1_FEATURES.csv", "BOTH", "Fv"),
        ("M2_ESM_IF1", "structure_marathon/esm_if1/M2_ESMIF1_FEATURES.csv", "BOTH", "Fv"),
        ("M3_SAPROT", "structure_marathon/saprot/M3_FEATURES.csv", "BOTH", "Fv"),
    ]
    for fam, rel, tgt, scope in marathon:
        X = numeric_X(read_any(FP / rel), drop_cols=["status", "extraction_status"])
        for t in (["TmApp", "HIC"] if tgt == "BOTH" else [tgt]):
            bk = "TmApp_BASE" if t == "TmApp" else ("HIC_BASE_ARO" if fam not in ("S3_SURFACE_PATCH_GRAPH",) else "HIC_BASE_ARO")
            add(t, fam, "structure_marathon", str(FP / rel), X, bk, scope, "standalone/OOF scorecard",
                "PCA32 if dim>200; SIMPLE_TVT_NOT_YET_RUN")

    # Historical physics — ESMFold canonical
    hist = [
        ("ANM_SPECTRUM", "ANM-SPECTRUM/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("VHL_ANGLE", "VHL-ANGLE/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("PKA_SHIFT_CORRECTED", "PKA-SHIFT/TECHNICAL_CORRECTION/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("VOID_EXPLICIT", "VOID-EXPLICIT/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("POLAR_SAT", "POLAR-SAT/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("OPENMM_STRAIN", "OPENMM-STRAIN/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("INTERFACE_ENERGY", "INTERFACE-ENERGY/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("3DI_FROZEN", "3DI-FROZEN/features_esmfold.parquet", "TmApp", "Fv", "TmApp_BASE"),
        ("STATIC_SAP", "STATIC-SAP/features_esmfold.parquet", "HIC", "Fv", "HIC_BASE_ARO"),
        ("HYDRO_FIELD", "HYDRO-FIELD/features_esmfold.parquet", "HIC", "Fv", "HIC_BASE_ARO"),
        ("ELEC_HYDRO_COPATCH", "ELEC-HYDRO-COPATCH/features_esmfold.parquet", "HIC", "Fv", "HIC_BASE_ARO"),
        ("TITRATION_SHAPE", "TITRATION-SHAPE/features_esmfold.parquet", "BOTH", "Fv", None),
    ]
    for fam, rel, tgt, scope, bk in hist:
        path = FP / rel
        if not path.exists():
            continue
        X = numeric_X(read_any(path), drop_cols=["extraction_status", "status"])
        targets = ["TmApp", "HIC"] if tgt == "BOTH" else [tgt]
        for t in targets:
            base_key = bk if bk else ("TmApp_BASE" if t == "TmApp" else "HIC_BASE_ARO")
            add(t, fam, path.parent.name, str(path), X, base_key, scope, "standalone/residual", "SIMPLE_TVT_NOT_YET_RUN")

    return specs


def write_inventory(specs):
    rows = []
    for s in specs:
        rows.append(
            {
                "target": s["target"],
                "family": s["family"],
                "source": s["source"],
                "feature_path": s["feature_path"],
                "dimension": s["dimension"],
                "molecular_scope": s["molecular_scope"],
                "previous_evaluation_type": s["previous_evaluation_type"],
                "simple_tvt_already_run": s["simple_tvt_already_run"],
                "simple_tvt_result_if_any": s["simple_tvt_result_if_any"],
                "available_N": s["available_N"],
                "base_key": s["base_key"],
                "notes": s["notes"],
                "label": s["label"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "SIMPLE_TVT_FEATURE_INVENTORY.csv", index=False)
    return df


def rotation_splits(folds, ids):
    fmap = folds.set_index("id")["fold"].astype(int).to_dict()
    ids = [i for i in ids if i in fmap]
    for k in range(5):
        te = [i for i in ids if fmap[i] == k]
        va = [i for i in ids if fmap[i] == (k + 1) % 5]
        tr = [i for i in ids if fmap[i] not in (k, (k + 1) % 5)]
        yield k, tr, va, te


def impute_fit(X):
    return X.median(numeric_only=True).fillna(0.0)


def impute_apply(X, med):
    return X.fillna(med).fillna(0.0)


def select_alpha(Xtr, ytr, Xva, yva):
    best_a, best = 100.0, np.inf
    for a in ALPHAS:
        sc = StandardScaler()
        m = Ridge(alpha=a, random_state=0)
        m.fit(sc.fit_transform(Xtr), ytr)
        score = mae(yva, m.predict(sc.transform(Xva)))
        if score < best - 1e-15 or (abs(score - best) <= 1e-15 and a > best_a):
            best, best_a = score, a
    return best_a


def reduce_structure_block(X_struct_tr, X_struct_others):
    """PCA only the structure block when it is high-dimensional (not the BASE)."""
    if X_struct_tr.shape[1] <= DIM_PCA_TRIGGER:
        return X_struct_tr, X_struct_others, False
    n = min(PCA_CAP, X_struct_tr.shape[0] - 1, X_struct_tr.shape[1])
    if n < 2:
        return X_struct_tr, X_struct_others, False
    pca = PCA(n_components=n, random_state=0)
    Ztr = pca.fit_transform(X_struct_tr)
    cols = [f"spca_{i}" for i in range(n)]
    outs = [pd.DataFrame(pca.transform(X), index=X.index, columns=cols) for X in X_struct_others]
    return pd.DataFrame(Ztr, index=X_struct_tr.index, columns=cols), outs, True


def choose_alpha_on_train_concat(X_base, X_struct, y, tr, va, use_struct: bool):
    """Impute on TRAIN; optional PCA on structure only; pick alpha on VAL."""
    if use_struct:
        # impute structure and base separately on train
        med_b = impute_fit(X_base.loc[tr])
        med_s = impute_fit(X_struct.loc[tr])
        Btr = impute_apply(X_base.loc[tr], med_b)
        Bva = impute_apply(X_base.loc[va], med_b)
        Str = impute_apply(X_struct.loc[tr], med_s)
        Sva = impute_apply(X_struct.loc[va], med_s)
        Str2, [Sva2], _ = reduce_structure_block(Str, [Sva])
        Xtr = pd.concat([Btr, Str2], axis=1)
        Xva = pd.concat([Bva, Sva2], axis=1)
    else:
        med_b = impute_fit(X_base.loc[tr])
        Xtr = impute_apply(X_base.loc[tr], med_b)
        Xva = impute_apply(X_base.loc[va], med_b)
    return select_alpha(Xtr, y.loc[tr], Xva, y.loc[va])


def predict_with_alpha_concat(X_base, X_struct, y, tr, va, te, alpha, use_struct: bool):
    ids_tv = tr + va
    if use_struct:
        med_b = impute_fit(X_base.loc[ids_tv])
        med_s = impute_fit(X_struct.loc[ids_tv])
        Btv = impute_apply(X_base.loc[ids_tv], med_b)
        Bte = impute_apply(X_base.loc[te], med_b)
        Stv = impute_apply(X_struct.loc[ids_tv], med_s)
        Ste = impute_apply(X_struct.loc[te], med_s)
        Stv2, [Ste2], _ = reduce_structure_block(Stv, [Ste])
        Xtv = pd.concat([Btv, Stv2], axis=1)
        Xte = pd.concat([Bte, Ste2], axis=1)
    else:
        med_b = impute_fit(X_base.loc[ids_tv])
        Xtv = impute_apply(X_base.loc[ids_tv], med_b)
        Xte = impute_apply(X_base.loc[te], med_b)
    sc = StandardScaler()
    m = Ridge(alpha=alpha, random_state=0)
    m.fit(sc.fit_transform(Xtv), y.loc[ids_tv])
    return pd.Series(m.predict(sc.transform(Xte)), index=te)


def run_protocol(base, struct, y, folds, ids, mode):
    """mode: FREE_ALPHA | BASE_FIXED_ALPHA. PCA applies to structure block only."""
    common = sorted(set(base.index) & set(struct.index) & set(y.index) & set(folds.id.astype(str)) & set(ids))
    cov_ok = True
    coverage = []
    for k, tr, va, te in rotation_splits(folds, common):
        ok = len(tr) >= MIN_TR and len(va) >= MIN_VA and len(te) >= MIN_TE
        coverage.append((k, len(tr), len(va), len(te), ok))
        cov_ok = cov_ok and ok
    if not cov_ok or len(common) < 40:
        return None, coverage

    Xb = base.loc[common]
    Xs = struct.loc[common]
    # align / drop dup names in struct vs base
    overlap = [c for c in Xs.columns if c in Xb.columns]
    if overlap:
        Xs = Xs.drop(columns=overlap)
    yy = y.loc[common]
    base_oof = pd.Series(index=common, dtype=float)
    plus_oof = pd.Series(index=common, dtype=float)
    fold_deltas = []
    alphas_b, alphas_p = [], []

    for k, tr, va, te in rotation_splits(folds, common):
        a_base = choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=False)
        if mode == "FREE_ALPHA":
            a_plus = choose_alpha_on_train_concat(Xb, Xs, yy, tr, va, use_struct=True)
        else:
            a_plus = a_base
        pb = predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_base, use_struct=False)
        pp = predict_with_alpha_concat(Xb, Xs, yy, tr, va, te, a_plus, use_struct=True)
        base_oof.loc[te] = pb
        plus_oof.loc[te] = pp
        alphas_b.append(a_base)
        alphas_p.append(a_plus)
        fold_deltas.append(mae(yy.loc[te], pb) - mae(yy.loc[te], pp))

    return {
        "N": len(common),
        "base_dim": int(Xb.shape[1]),
        "struct_dim": int(Xs.shape[1]),
        "plus_dim": int(Xb.shape[1] + Xs.shape[1]),
        "base_mae": mae(yy, base_oof),
        "plus_mae": mae(yy, plus_oof),
        "delta": mae(yy, base_oof) - mae(yy, plus_oof),
        "fold_deltas": fold_deltas,
        "alphas_base": alphas_b,
        "alphas_plus": alphas_p,
        "coverage": coverage,
    }, coverage


def classify(dp, ds):
    if dp > 0.02 and ds > 0.02:
        return "DIRECT_STRONG_CANDIDATE"
    if dp > 0 and ds > 0:
        return "DIRECT_WEAK_CANDIDATE"
    if (dp > 0) != (ds > 0):
        return "DIRECT_MIXED"
    return "DIRECT_NO_IMPROVEMENT"


def scientific_verdict(dp, ds, fixed_dp, fixed_ds):
    if dp > 0 and ds > 0 and fixed_dp > 0 and fixed_ds > 0 and min(dp, ds) > 0.01:
        return "SCI_ROBUST_POSITIVE"
    if dp > 0 and ds > 0 and fixed_dp > 0 and fixed_ds > 0:
        return "SCI_WEAK_POSITIVE"
    if dp > 0 and ds > 0:
        return "SCI_ALPHA_SENSITIVE_POSITIVE"
    if (dp > 0) != (ds > 0):
        return "SCI_MIXED"
    return "SCI_NO_SIGNAL"


def competition_verdict(dp, ds, fixed_dp, fixed_ds):
    wc = min(dp, ds)
    wc_f = min(fixed_dp, fixed_ds)
    if wc > 0.02 and wc_f > 0:
        return "COMP_PRIORITY"
    if wc > 0 and (fixed_dp > 0 or fixed_ds > 0):
        return "COMP_TRY"
    if wc > 0:
        return "COMP_WATCH"
    return "COMP_DROP"


def main():
    print("Loading bases…", flush=True)
    bases, base_names = load_bases()
    print("Building catalog…", flush=True)
    specs = build_family_catalog()
    inv = write_inventory(specs)
    print(f"inventory rows={len(inv)} BioEmu_not_yet={(~inv.simple_tvt_already_run & inv.family.str.contains('BIOEMU')).sum()}", flush=True)

    primary = pd.read_csv(PRIMARY)
    shadow = pd.read_csv(SHADOW)
    primary["id"] = primary.id.astype(str)
    shadow["id"] = shadow.id.astype(str)
    y_tm, y_hic = load_y("TmApp"), load_y("HIC")

    rows = []
    for s in specs:
        target = s["target"]
        y = y_tm if target == "TmApp" else y_hic
        base = bases[s["base_key"]]
        X = s["X"]
        ids = list(base.index)
        if s["id_filter"] is not None:
            ids = [i for i in ids if i in s["id_filter"]]
        print(f"→ {target} {s['family']} dim={X.shape[1]} base={s['base_key']}", flush=True)
        by = {}
        for folds, tag in [(primary, "Primary"), (shadow, "Shadow")]:
            for mode in ("FREE_ALPHA", "BASE_FIXED_ALPHA"):
                out, cov = run_protocol(base, X, y, folds, ids, mode)
                key = (tag, mode)
                by[key] = out
                if out is None:
                    print(f"  SKIP {tag} {mode} cov={cov}", flush=True)
        if by.get(("Primary", "FREE_ALPHA")) is None or by.get(("Shadow", "FREE_ALPHA")) is None:
            continue
        pf, sf = by[("Primary", "FREE_ALPHA")], by[("Shadow", "FREE_ALPHA")]
        pfix, sfix = by[("Primary", "BASE_FIXED_ALPHA")], by[("Shadow", "BASE_FIXED_ALPHA")]
        if pfix is None or sfix is None:
            continue
        rows.append(
            {
                "target": target,
                "family": s["family"],
                "source": s["source"],
                "base_name": base_names[s["base_key"]],
                "base_key": s["base_key"],
                "N": pf["N"],
                "feature_dim": pf["struct_dim"],
                "base_dim": pf["base_dim"],
                "label": s["label"],
                "notes": s["notes"],
                # FREE
                "free_primary_base_mae": pf["base_mae"],
                "free_primary_combined_mae": pf["plus_mae"],
                "free_primary_delta": pf["delta"],
                "free_shadow_base_mae": sf["base_mae"],
                "free_shadow_combined_mae": sf["plus_mae"],
                "free_shadow_delta": sf["delta"],
                "free_mean_delta": 0.5 * (pf["delta"] + sf["delta"]),
                "free_worst_delta": min(pf["delta"], sf["delta"]),
                "free_verdict": classify(pf["delta"], sf["delta"]),
                "free_per_fold_primary": str(pf["fold_deltas"]),
                "free_per_fold_shadow": str(sf["fold_deltas"]),
                # FIXED
                "fixed_primary_base_mae": pfix["base_mae"],
                "fixed_primary_combined_mae": pfix["plus_mae"],
                "fixed_primary_delta": pfix["delta"],
                "fixed_shadow_base_mae": sfix["base_mae"],
                "fixed_shadow_combined_mae": sfix["plus_mae"],
                "fixed_shadow_delta": sfix["delta"],
                "fixed_mean_delta": 0.5 * (pfix["delta"] + sfix["delta"]),
                "fixed_worst_delta": min(pfix["delta"], sfix["delta"]),
                "fixed_verdict": classify(pfix["delta"], sfix["delta"]),
                "survives_fixed_alpha": bool(pfix["delta"] > 0 and sfix["delta"] > 0),
                "SCIENTIFIC_VERDICT": scientific_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
                "COMPETITION_VERDICT": competition_verdict(pf["delta"], sf["delta"], pfix["delta"], sfix["delta"]),
                "coverage_primary": str(pf["coverage"]),
                "coverage_shadow": str(sf["coverage"]),
            }
        )

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(["target", "free_worst_delta"], ascending=[True, False])
    all_df.to_csv(OUT / "SIMPLE_TVT_ALL_BLOCKS.csv", index=False)
    print("wrote ALL_BLOCKS", len(all_df), flush=True)

    # dump json for combination stage
    (OUT / "results" / "simple_tvt_rescreen_cache.json").write_text(
        json.dumps({"n": len(all_df), "families": all_df.family.tolist()}, indent=2)
    )
    return all_df


if __name__ == "__main__":
    main()
