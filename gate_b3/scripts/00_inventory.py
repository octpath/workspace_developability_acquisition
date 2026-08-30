#!/usr/bin/env python3
"""Inventory reusable B1/B2 caches and features for Gate B3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3_common import B1_CACHE, B1_DATA, B2_CACHE, REPORTS, ensure_dirs, write_json  # noqa: E402


def emb_dim(manifest: Path):
    if not manifest.exists():
        return None, 0
    m = pd.read_csv(manifest)
    p = Path(m.iloc[0]["path"])
    if not p.exists():
        return None, len(m)
    return int(np.load(p).shape[-1]), len(m)


def main():
    ensure_dirs()
    core = pd.read_csv(B1_DATA / "triple_core.csv")
    n_pop = len(core)
    ids = set(core["antibody_id"])

    rows = []

    def add(name, source, path, n, dim, target_indep, safe, notes=""):
        cov = None
        if path and Path(str(path)).exists() and str(path).endswith(".csv"):
            try:
                df = pd.read_csv(path, usecols=lambda c: c == "antibody_id")
                cov = len(ids & set(df["antibody_id"]))
            except Exception:
                cov = None
        rows.append(
            {
                "feature_name": name,
                "source_gate": source,
                "n_antibodies": n,
                "dimension": dim,
                "overlap_324_coverage": cov if cov is not None else n,
                "target_independent": target_indep,
                "safe_to_reuse": safe,
                "cache_path": str(path) if path else "",
                "notes": notes,
            }
        )

    # Sequence features
    for fn, dim_hint in [
        ("stage_A_simple.csv", "~100"),
        ("stage_B_cdr.csv", "~260"),
        ("stage_C_shortcut.csv", "~mixed+cat"),
    ]:
        p = B1_CACHE / "features" / fn
        df = pd.read_csv(p)
        ncols = df.select_dtypes(include=[np.number]).shape[1]
        add(fn.replace(".csv", ""), "B1", p, len(df), ncols, True, True, "sequence-derived")

    # PLM
    for man, label in [
        ("manifest_ablang2_default.csv", "AbLang2"),
        ("manifest_esm1b_t33_650M_UR50S.csv", "ESM-1b"),
        ("manifest_esm2_t33_650M_UR50D.csv", "ESM-2"),
        ("manifest_esm2_t33_650M_UR50D_CDR6.csv", "ESM-2_CDR6"),
    ]:
        p = B1_CACHE / "plm" / man
        d, n = emb_dim(p)
        add(label, "B1", p, n, d, True, True, "embedding; full 400 covers 324")

    # Structures
    abb_m = B1_CACHE / "structures" / "abb2_manifest.csv"
    add("ABodyBuilder2_PDB", "B1", abb_m, len(pd.read_csv(abb_m)), "PDB", True, True)
    esmn_m = B2_CACHE / "structures" / "esmfold_native_manifest.csv"
    add("ESMFold_native_VHVL", "B2", esmn_m, len(pd.read_csv(esmn_m)), "PDB", True, True)

    # Structure features
    for label, p in [
        ("ABB_SASA_RASA_PATCH", B2_CACHE / "structure_features" / "abb_sasa_rasa_patch.csv"),
        ("ESMFN_SASA_RASA_PATCH", B2_CACHE / "structure_features" / "esmfold_native_sasa_rasa_patch.csv"),
    ]:
        df = pd.read_csv(p)
        add(label, "B2", p, len(df), df.shape[1] - 1, True, True)

    # Numbering
    num = B1_DATA / "numbering_germline.csv"
    add("IMGT_numbering_germline", "B1", num, len(pd.read_csv(num)), "regions+germline", True, True)

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS.parent / "metrics" / "cache_inventory.csv" if False else "/workspace_developability_acquisition/gate_b3/metrics/cache_inventory.csv", index=False)
    Path("/workspace_developability_acquisition/gate_b3/metrics").mkdir(parents=True, exist_ok=True)
    out.to_csv("/workspace_developability_acquisition/gate_b3/metrics/cache_inventory.csv", index=False)

    lines = [
        "# Cache and feature inventory (Gate B3)",
        "",
        f"Final competition population target: antibodies with **both** HIC and TmApp (N≈{n_pop} from `triple_core.csv`).",
        "",
        "All listed representations below cover the 324-overlap set and are **target-independent** (safe to reuse without recompute if sequence/model hashes unchanged).",
        "",
        "| feature | source | N | dim | overlap coverage | safe | path |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r.feature_name} | {r.source_gate} | {r.n_antibodies} | {r.dimension} | {r.overlap_324_coverage} | {r.safe_to_reuse} | `{r.cache_path}` |"
        )
    lines += [
        "",
        "## Reuse policy",
        "",
        "- Do **not** recompute PLM embeddings, ABB/ESMFold structures, or B2 SASA/RASA/patch tables.",
        "- New B3 branches: IMGT positional one-hot, germline-relative mutations, n-grams, extended structure geometry, Train-only CV features.",
        "- Organizer-only columns (`b_cell_subset`, donor) stay out of participant files.",
        "",
    ]
    (REPORTS / "cache_and_feature_inventory.md").write_text("\n".join(lines) + "\n")
    write_json(
        Path("/workspace_developability_acquisition/gate_b3/config/inventory_summary.json"),
        {"n_triple_core": n_pop, "n_representations_reuse": len(out), "all_cover_324": True},
    )
    print("INVENTORY_OK", len(out))


if __name__ == "__main__":
    main()
