"""Surface-patch helpers.

True continuous molecular-surface patches (Gap Closure FreeSASA Lee–Richards)
are distributed as precomputed tables under
``data/precomputed_features/continuous_surface.parquet``.

This module exposes a **residue-adjacency hydrophobic exposure summary** only,
and labels it clearly so it is not confused with continuous MS patches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import HYDROPHOBIC, compute_sasa, finite_sum, load_structure, residue_sasa_rows

CONTACT_A = 8.0


def extract(
    pdb_path: str | Path,
    heavy_sequence: str | None = None,
    light_sequence: str | None = None,
) -> dict[str, float]:
    """Residue-graph hydrophobic exposure summary (NOT continuous MS patches).

    Method: Bio.PDB Shrake–Rupley residue SASA → exposed hydrophobic residues
    (RASA ≥ 0.20) → connected components on CA–CA distance ≤ 8 Å.

    This does **not** implement Gap Closure continuous SAS-sampled surface patches
    (FreeSASA Lee–Richards + exterior sample points). Those live only in
    ``data/precomputed_features/continuous_surface.parquet`` and are
    **not numerically equivalent** to this extractor.
    """
    _ = (heavy_sequence, light_sequence)
    structure = load_structure(pdb_path)
    compute_sasa(structure)
    rows = residue_sasa_rows(structure)
    hydro = [r for r in rows if r["amino_acid"] in HYDROPHOBIC and r["is_exposed"]]
    total_hydro = finite_sum(r["sasa"] for r in hydro)

    # connected components on CA ≤ CONTACT_A
    n = len(hydro)
    comps = []
    if n:
        coords = np.vstack([r["ca"] for r in hydro])
        seen = set()
        for i in range(n):
            if i in seen:
                continue
            stack = [i]
            seen.add(i)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if v in seen:
                        continue
                    if not np.all(np.isfinite(coords[u])) or not np.all(np.isfinite(coords[v])):
                        continue
                    if float(np.linalg.norm(coords[u] - coords[v])) <= CONTACT_A:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)

    areas = [finite_sum(hydro[i]["sasa"] for i in c) for c in comps] if comps else []
    max_area = max(areas) if areas else 0.0

    return {
        "residue_graph_hydrophobic_sasa_total": total_hydro,
        "residue_graph_hydrophobic_patch_count": float(len(comps)),
        "residue_graph_largest_patch_sasa": float(max_area),
        "residue_graph_fragmentation": float(len(comps) / total_hydro) if total_hydro > 0 else float("nan"),
        "note_continuous_ms_not_computed": 1.0,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Residue-graph hydrophobic summary (not continuous MS patches)"
    )
    p.add_argument("--pdb", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    feats = extract(args.pdb)
    text = json.dumps(feats, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
