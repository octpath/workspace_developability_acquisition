#!/usr/bin/env python3
"""Build participant-facing feature_extension v1 data artifacts from existing assets.

Uses only existing on-disk outputs. No heavy inference.
Run with <=4 cores: taskset -c 16-19 python tools/build_v1_data.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FE = Path(__file__).resolve().parents[1]
CROSSWALK = (
    ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv"
)
CV_PRIMARY = ROOT / "virtual_participant/stage0_cv/cv_primary.csv"
CV_SHADOW = ROOT / "virtual_participant/stage0_cv/cv_shadow.csv"
ESMFOLD_FV = ROOT / "esmfold_native"
ESMFOLD_FAB = (
    ROOT
    / "organizer_extension/feature_prospecting/fab_reconstruction/structures/esmfold_fab"
)
BIOEMU_FEAT = (
    ROOT
    / "organizer_extension/feature_prospecting/bioemu_isolated_reassessment"
    / "BIOEMU_ISOLATED_REASSESS_FEATURES.csv"
)
MARATHON = ROOT / "organizer_extension/feature_prospecting/structure_marathon"
GAP = ROOT / "organizer_extension/feature_prospecting/structure_gap_closure/results"
HIST = ROOT / "organizer_extension/feature_prospecting"

FORBIDDEN_COL_SUBSTR = (
    "tmap",
    "hic",
    "public",
    "private",
    "y_true",
    "y_pred",
    "residual",
    "incumbent",
    "oof",
    "leaderboard",
    "split",
    "pvalue",
    "p_value",
    "pearson",
    "spearman",
    "rmse",
    "mae",
    "r2_",
    "_r2",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_feature_frame(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"missing id column {id_col}; have {list(df.columns)[:20]}")
    out = df.copy()
    out = out.rename(columns={id_col: "id"})
    drop = []
    for c in out.columns:
        if c == "id":
            continue
        cl = c.lower()
        if any(s in cl for s in FORBIDDEN_COL_SUBSTR):
            drop.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            # keep only numeric feature cols + id
            drop.append(c)
    if drop:
        out = out.drop(columns=drop)
    # replace ±inf with NaN
    feat = out.drop(columns=["id"])
    feat = feat.replace([np.inf, -np.inf], np.nan)
    out = pd.concat([out[["id"]], feat], axis=1)
    out = out.drop_duplicates(subset=["id"], keep="first").sort_values("id").reset_index(drop=True)
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {
        "path": str(path.relative_to(FE)),
        "n_ids": int(df["id"].nunique()),
        "n_rows": len(df),
        "feature_dim": int(df.shape[1] - 1),
        "sha256": sha256_file(path),
        "columns": [c for c in df.columns if c != "id"],
    }


def build_folds() -> None:
    cw = pd.read_csv(CROSSWALK)
    # organizer training/dev IDs only (have labels for participants)
    split_col = None
    for cand in ("dataset_split", "split", "organizer_split", "role"):
        if cand in cw.columns:
            split_col = cand
            break
    if split_col is None:
        raise RuntimeError(f"Cannot identify DEV IDs from crosswalk cols={list(cw.columns)}")
    vals = cw[split_col].astype(str).str.upper()
    dev_mask = vals.isin(["DEV", "TRAIN", "TRAINING"])
    if not dev_mask.any():
        raise RuntimeError(f"No DEV IDs in column {split_col}")
    train_ids = set(cw.loc[dev_mask, "id"].astype(str))

    prim = pd.read_csv(CV_PRIMARY)
    shad = pd.read_csv(CV_SHADOW)
    prim["id"] = prim["id"].astype(str)
    shad["id"] = shad["id"].astype(str)
    # expected columns id, fold
    fold_p = prim.rename(columns={"fold": "fold_primary"})[["id", "fold_primary"]]
    fold_s = shad.rename(columns={"fold": "fold_shadow"})[["id", "fold_shadow"]]
    folds = fold_p.merge(fold_s, on="id", how="inner")
    folds = folds[folds["id"].isin(train_ids)].copy()
    # validate fold values 0-4
    for col in ("fold_primary", "fold_shadow"):
        folds[col] = folds[col].astype(int)
        bad = ~folds[col].isin(range(5))
        if bad.any():
            raise RuntimeError(f"invalid {col}")
    folds = folds.sort_values("id").reset_index(drop=True)
    out = FE / "folds.csv"
    folds.to_csv(out, index=False)
    print(f"folds.csv n={len(folds)} train_ids={len(train_ids)}")
    if len(folds) != len(train_ids):
        missing = train_ids - set(folds["id"])
        extra = set(folds["id"]) - train_ids
        print(f"  WARN fold/train mismatch missing={len(missing)} extra={len(extra)}")


def copy_pdbs(src: Path, dst: Path, ids: list[str]) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    missing = []
    for aid in ids:
        sp = src / f"{aid}.pdb"
        if not sp.is_file():
            missing.append(aid)
            continue
        shutil.copy2(sp, dst / f"{aid}.pdb")
        n += 1
    if missing:
        print(f"  missing PDBs from {src}: {len(missing)} e.g. {missing[:3]}")
    return n


def build_structures() -> dict:
    cw = pd.read_csv(CROSSWALK)
    ids = sorted(cw["id"].astype(str).unique())
    n_fv = copy_pdbs(ESMFOLD_FV, FE / "data/esmfold_fv", ids)
    n_fab = copy_pdbs(ESMFOLD_FAB, FE / "data/esmfold_fab", ids)
    # provenance for Fab constants
    pol = (
        ROOT
        / "organizer_extension/feature_prospecting/fab_reconstruction/sequences/CONSTANT_DOMAIN_POLICY.md"
    )
    fasta = (
        ROOT
        / "organizer_extension/feature_prospecting/fab_reconstruction/sequences/CONSTANT_DOMAIN_SEQUENCES.fasta"
    )
    dest = FE / "data/esmfold_fab"
    if pol.is_file():
        shutil.copy2(pol, dest / "CONSTANT_DOMAIN_POLICY.md")
    if fasta.is_file():
        shutil.copy2(fasta, dest / "CONSTANT_DOMAIN_SEQUENCES.fasta")
    return {"n_fv": n_fv, "n_fab": n_fab, "n_ids": len(ids)}


def build_bioemu() -> dict:
    """Package BioEmu isolated reassessment features.

    Logical organizer families (from eval_reassess.py) over column prefixes:
      NEW_CONTACT  — *contact*
      NEW_FLEX     — *rmsf*
      NEW_PAIRWISE — *ca_rmsd*
    plus Rg and related ensemble scalars present in the frozen table.
    QC columns nphys_used / VH_n / VL_n go to qc.csv only.
    """
    raw = pd.read_csv(BIOEMU_FEAT)
    raw["id"] = raw["id"].astype(str)

    qc = pd.DataFrame({"id": raw["id"]})
    for src, dst in [
        ("nphys_used", "physical_usable_frames"),
        ("VH_n", "vh_usable_frames"),
        ("VL_n", "vl_usable_frames"),
    ]:
        if src in raw.columns:
            qc[dst] = raw[src]
    qc["extraction_status"] = "OK"
    qc["qc_status"] = "OK"
    qc_path = FE / "data/bioemu_isolated/qc.csv"
    qc.to_csv(qc_path, index=False)

    drop_from_feat = {"nphys_used", "VH_n", "VL_n"}
    feat_raw = raw.drop(columns=[c for c in drop_from_feat if c in raw.columns])
    feat = clean_feature_frame(feat_raw, "id")
    meta = write_parquet(feat, FE / "data/bioemu_isolated/features.parquet")
    meta["qc_cols"] = list(qc.columns)
    meta["feature_columns"] = [c for c in feat.columns if c != "id"]
    return meta



def build_precomputed() -> list[dict]:
    specs = [
        (
            "proteinmpnn.parquet",
            MARATHON / "proteinmpnn/M1_FEATURES.csv",
            "ProteinMPNN",
            "Fv ESMFold",
            "TmApp/HIC",
        ),
        (
            "esm_if1.parquet",
            MARATHON / "esm_if1/M2_ESMIF1_FEATURES.csv",
            "ESM-IF1",
            "Fv ESMFold",
            "TmApp/HIC",
        ),
        (
            "saprot.parquet",
            MARATHON / "saprot/M3_FEATURES.csv",
            "SaProt",
            "Fv ESMFold",
            "TmApp/HIC",
        ),
        (
            "generator_disagreement.parquet",
            MARATHON / "generator_disagreement/S2_DISAGREEMENT_FEATURES.csv",
            "cross-generator disagreement",
            "Fv multi-generator",
            "TmApp/HIC",
        ),
        (
            "continuous_surface.parquet",
            GAP / "HIC_CONTINUOUS_SURFACE_FEATURES.csv",
            "FreeSASA continuous surface (Gap Closure)",
            "Fv/Fab ESMFold",
            "HIC",
        ),
        (
            "packing_cavity.parquet",
            GAP / "TMAPP_PACKING_CAVITY_FEATURES.csv",
            "packing/cavity (Gap Closure)",
            "Fab ESMFold",
            "TmApp",
        ),
        (
            "buried_unsatisfied.parquet",
            GAP / "TMAPP_BURIED_UNSAT_FEATURES.csv",
            "buried unsatisfied polar (Gap Closure)",
            "Fab ESMFold",
            "TmApp",
        ),
        (
            "fab_interface.parquet",
            GAP / "TMAPP_INTERFACE_FEATURES.csv",
            "Fab domain interface (Gap Closure)",
            "Fab ESMFold",
            "TmApp",
        ),
    ]
    # historical families
    hist_specs = [
        (
            "aromatic_topology.parquet",
            HIST / "AROMATIC-TOPO/features_esmfold.parquet",
            "AROMATIC-TOPO v1",
            "Fv ESMFold",
            "HIC",
        ),
        (
            "static_sap.parquet",
            HIST / "STATIC-SAP/features_esmfold.parquet",
            "STATIC-SAP",
            "Fv ESMFold",
            "HIC",
        ),
        (
            "hydro_field.parquet",
            HIST / "HYDRO-FIELD/features_esmfold.parquet",
            "HYDRO-FIELD",
            "Fv ESMFold",
            "HIC",
        ),
        (
            "titration_shape.parquet",
            HIST / "TITRATION-SHAPE/features_esmfold.parquet",
            "TITRATION_SHAPE",
            "Fv ESMFold",
            "HIC/TmApp",
        ),
    ]

    out_dir = FE / "data/precomputed_features"
    metas = []
    for name, src, model, scope, rec in specs + hist_specs:
        if not src.is_file():
            print(f"OMIT missing {src}")
            metas.append({"artifact": name, "status": "OMITTED_MISSING", "source": str(src)})
            continue
        if src.suffix == ".parquet":
            df = pd.read_parquet(src)
        else:
            df = pd.read_csv(src)
        # normalize id
        if "id" not in df.columns:
            for alt in ("antibody_id", "Ab_ID", "ID"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "id"})
                    break
        feat = clean_feature_frame(df)
        if feat.shape[1] <= 1:
            print(f"OMIT empty after clean {name}")
            metas.append({"artifact": name, "status": "OMITTED_EMPTY", "source": str(src)})
            continue
        m = write_parquet(feat, out_dir / name)
        m.update(
            {
                "artifact": name,
                "status": "INCLUDED",
                "generator_or_model": model,
                "molecular_scope": scope,
                "recommended_target": rec,
                "source": str(src.relative_to(ROOT)),
            }
        )
        metas.append(m)
        print(f"wrote {name} n={m['n_ids']} dim={m['feature_dim']}")
    return metas


def write_data_readmes(struct_meta: dict, bioemu_meta: dict, prec_metas: list[dict]) -> None:
    (FE / "data/README.md").write_text(
        """# data/

Participant-facing structure and precomputed feature assets for **feature_extension v1**.

| Subdirectory | Contents | Molecular scope |
|---|---|---|
| `esmfold_fv/` | Predicted Fv PDBs | Fv |
| `esmfold_fab/` | Reconstructed Fab PDBs | Fab (variable + surrogate constants) |
| `bioemu_isolated/` | Aggregated BioEmu features + QC | Isolated VH / VL monomers |
| `precomputed_features/` | Model / descriptor parquet tables | See per-file README |
| `optional/` | Reserved for large optional archives | — |

**Target labels are not included.** All assets here were generated independently of competition targets.
""",
        encoding="utf-8",
    )

    (FE / "data/esmfold_fv/README.md").write_text(
        f"""# esmfold_fv/

## What is this?
Predicted **Fv** structures from **ESMFold**, one PDB per antibody.

## Molecular scope
**Fv** (variable heavy + variable light only). Not Fab. Not IgG.

## Number of antibodies
{struct_meta.get("n_fv", "SEE_MANIFEST")} PDB files named `{{id}}.pdb`.

## Generation method
Organizer ESMFold native Fv generation used for structure-marathon / gap-closure inputs.
Model version: see RELEASE_NOTES (recorded as available from project provenance; otherwise `NOT_RECORDED`).

## Chain identities
Typically heavy = chain H (or A) and light = chain L (or B) depending on generator convention.
Inspect each PDB header/ATOM chain IDs before region-specific analysis.

## Sequence source
Competition antibody VH/VL sequences (target-blind structure generation).

## Confidence
ESMFold pLDDT is commonly stored in the PDB B-factor field when present. Treat as model confidence, not experimental B-factors.

## QC / limitations
- Predicted structures, not crystal structures.
- Fv-only: constant domains absent.
- No experimental refinement in this package.

## Target labels used during generation
**NO**
""",
        encoding="utf-8",
    )

    (FE / "data/esmfold_fab/README.md").write_text(
        f"""# esmfold_fab/

## What is this?
**Predicted / reconstructed Fab** structures built by attaching organizer **surrogate constant domains**
to competition variable sequences, then folding with ESMFold (organizer Fab reconstruction pipeline).

## Molecular scope
**Fab** (VH–CH1 + VL–CL). **Not** a full IgG. **Not** an experimentally solved Fab structure.

## Important scientific caveats
- Experimental **TmApp** assays were performed on **Fab** molecules.
- These files are **reconstructed** Fab models for feature extraction, not experimental coordinates.
- Variable sequences come from competition antibodies.
- Constant domains use **common surrogate** CH1 / Cκ / Cλ sequences (organizer POLICY B / UniProt-derived),
  not the exact historical allele/junction/papain terminus of each experimental Fab.
- Exact CH1–hinge / papain cleavage details are **not fully known** for the competition reagents.

See organizer provenance (copied conceptually into RELEASE_NOTES):
`fab_reconstruction/sequences/CONSTANT_DOMAIN_POLICY.md`.

## Number of antibodies
{struct_meta.get("n_fab", "SEE_MANIFEST")} PDB files named `{{id}}.pdb`.

## Target labels used during generation
**NO**

## Known issues
Any antibody-specific prep failures for later physics (e.g. OpenMM/FeNNix) are **out of scope** for this structure release.
Raw ESMFold Fab availability here is separate from later simulation readiness.
""",
        encoding="utf-8",
    )

    n_bio = bioemu_meta.get("n_ids", "SEE_MANIFEST")
    cols = bioemu_meta.get("feature_columns", [])
    col_preview = ", ".join(cols[:12]) + (" ..." if len(cols) > 12 else "")
    (FE / "data/bioemu_isolated/README.md").write_text(
        f"""# bioemu_isolated/

## What is this?
Aggregated **BioEmu** ensemble descriptors for **isolated VH and VL** monomeric chains
(final organizer physically QC-filtered reassessment features).

Files:
- `features.parquet` — frozen feature columns keyed by `id`
- `qc.csv` — target-independent QC metadata

## Molecular scope
**Isolated VH** and **Isolated VL** monomers sampled separately.
BioEmu in this release does **not** model the full Fab or IgG.

## Number of antibodies
{n_bio}

## Feature families (plain language)
Organizer evaluation groups columns from this table into logical families
(see BioEmu isolated reassessment `eval_reassess.py`):
- **NEW_CONTACT** — contact occupancy / persistence / entropy summaries (`*contact*`)
- **NEW_PAIRWISE** — CA-RMSD ensemble summaries (`*ca_rmsd*`)
- **NEW_FLEX** — RMSF / flexibility summaries (`*rmsf*`)
Additional Rg and related ensemble scalars are also included when present.

Exact column list (n={len(cols)}): starts with {col_preview}

Do **not** treat obsolete V12 BioEmu feature tables as equivalent; they are not the primary release.

## Sampling notes
- VH and VL ensembles were sampled **separately**.
- Features summarize **physically filtered** frames.
- Usable frame counts may differ by chain and antibody (see `qc.csv`).
- Extended **VL+CL** chain experiments are **not** part of this validated release.

## Target labels used during generation
**NO**

## Citation / model
Microsoft BioEmu (see RELEASE_NOTES licensing section).
""",
        encoding="utf-8",
    )

    lines = [
        "# precomputed_features/",
        "",
        "Expensive or frozen model-derived / descriptor feature tables.",
        "Each parquet has an `id` column and numeric feature columns only.",
        "",
        "**Target labels used: NO** for all blocks.",
        "",
        "## Feature dictionary",
        "",
        "| File | Model / family | Scope | Dim | Suggested target (heuristic) | Status |",
        "|---|---|---|---:|---|---|",
    ]
    for m in prec_metas:
        if m.get("status") != "INCLUDED":
            lines.append(
                f"| {m.get('artifact')} | — | — | — | — | {m.get('status')} |"
            )
            continue
        lines.append(
            f"| `{m['artifact']}` | {m.get('generator_or_model')} | {m.get('molecular_scope')} | "
            f"{m.get('feature_dim')} | {m.get('recommended_target')} | INCLUDED |"
        )
    lines += [
        "",
        "## Notes",
        "- Prefer these pooled/scalar tables over re-running ProteinMPNN / ESM-IF1 / SaProt.",
        "- Continuous surface descriptors are **molecular-surface** style from Gap Closure (FreeSASA LR),",
        "  not merely residue-adjacency graphs — see Gap Closure docs for definitions.",
        "- Packing / unsatisfied / interface blocks showed weak organizer CV increments but are scientifically valid for experimentation.",
        "",
        "## Target labels used during generation",
        "**NO**",
        "",
    ]
    (FE / "data/precomputed_features/README.md").write_text("\n".join(lines), encoding="utf-8")
    (FE / "data/optional/README.md").write_text(
        """# optional/

Reserved for optional large archives (e.g. raw BioEmu trajectories, high-dimensional embeddings).

**v1 core release does not require this directory.**
Raw BioEmu trajectories are omitted from v1 to keep the package lightweight
(`OMITTED_SIZE_AND_COMPLEXITY` — see RELEASE_NOTES).
""",
        encoding="utf-8",
    )


def build_manifest(struct_meta: dict, bioemu_meta: dict, prec_metas: list[dict]) -> None:
    rows = []

    def add(**kwargs):
        rows.append(kwargs)

    for scope, rel, n in [
        ("Fv", "data/esmfold_fv", struct_meta.get("n_fv")),
        ("Fab", "data/esmfold_fab", struct_meta.get("n_fab")),
    ]:
        add(
            artifact=Path(rel).name,
            relative_path=rel,
            artifact_type="structure_dir",
            molecular_scope=scope,
            format="pdb",
            n_ids=n,
            feature_dim="",
            generated_from="competition VH/VL sequences",
            generator_or_model="ESMFold",
            model_version="NOT_RECORDED",
            target_used="NO",
            recommended_target="both",
            qc_status="COPIED_FROM_AUTHORITATIVE",
            redistribution_status="SEE_RELEASE_NOTES",
            archive_name="",
            sha256="",
            notes="directory of PDBs",
        )

    add(
        artifact="bioemu_isolated_features",
        relative_path="data/bioemu_isolated/features.parquet",
        artifact_type="feature_table",
        molecular_scope="VH+VL_isolated",
        format="parquet",
        n_ids=bioemu_meta.get("n_ids"),
        feature_dim=bioemu_meta.get("feature_dim"),
        generated_from="BioEmu isolated reassessment",
        generator_or_model="BioEmu",
        model_version="NOT_RECORDED",
        target_used="NO",
        recommended_target="TmApp",
        qc_status="INCLUDED",
        redistribution_status="SEE_RELEASE_NOTES",
        archive_name="feature_extension_v1_bioemu_isolated.zip",
        sha256=bioemu_meta.get("sha256", ""),
        notes="NEW_* families",
    )

    for m in prec_metas:
        if m.get("status") != "INCLUDED":
            add(
                artifact=m.get("artifact", ""),
                relative_path=f"data/precomputed_features/{m.get('artifact', '')}",
                artifact_type="feature_table",
                molecular_scope="",
                format="parquet",
                n_ids="",
                feature_dim="",
                generated_from=m.get("source", ""),
                generator_or_model="",
                model_version="",
                target_used="NO",
                recommended_target="",
                qc_status=m.get("status", ""),
                redistribution_status="OMITTED",
                archive_name="",
                sha256="",
                notes=m.get("status", ""),
            )
            continue
        add(
            artifact=m["artifact"],
            relative_path=m["path"],
            artifact_type="feature_table",
            molecular_scope=m.get("molecular_scope", ""),
            format="parquet",
            n_ids=m.get("n_ids"),
            feature_dim=m.get("feature_dim"),
            generated_from=m.get("source", ""),
            generator_or_model=m.get("generator_or_model", ""),
            model_version="NOT_RECORDED",
            target_used="NO",
            recommended_target=m.get("recommended_target", ""),
            qc_status="INCLUDED",
            redistribution_status="SEE_RELEASE_NOTES",
            archive_name="feature_extension_v1_precomputed_features.zip",
            sha256=m.get("sha256", ""),
            notes="",
        )

    add(
        artifact="folds.csv",
        relative_path="folds.csv",
        artifact_type="cv_folds",
        molecular_scope="N/A",
        format="csv",
        n_ids=len(pd.read_csv(FE / "folds.csv")),
        feature_dim="",
        generated_from="virtual_participant/stage0_cv",
        generator_or_model="organizer Stage0 CV",
        model_version="frozen",
        target_used="NO",
        recommended_target="N/A",
        qc_status="INCLUDED",
        redistribution_status="OK",
        archive_name="feature_extension_v1_code.zip",
        sha256=sha256_file(FE / "folds.csv"),
        notes="DEV/train IDs only; fold_primary + fold_shadow",
    )

    pd.DataFrame(rows).to_csv(FE / "MANIFEST.csv", index=False)
    print(f"MANIFEST.csv rows={len(rows)}")


def main() -> None:
    print("=== build folds ===")
    build_folds()
    print("=== structures ===")
    struct_meta = build_structures()
    print(struct_meta)
    print("=== bioemu ===")
    bioemu_meta = build_bioemu()
    print({k: bioemu_meta[k] for k in ("n_ids", "feature_dim", "path") if k in bioemu_meta})
    print("=== precomputed ===")
    prec_metas = build_precomputed()
    write_data_readmes(struct_meta, bioemu_meta, prec_metas)
    build_manifest(struct_meta, bioemu_meta, prec_metas)
    meta_path = FE / "tools/_build_meta.json"
    meta_path.write_text(
        json.dumps(
            {"struct": struct_meta, "bioemu": bioemu_meta, "precomputed": prec_metas},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("DONE", meta_path)


if __name__ == "__main__":
    main()
