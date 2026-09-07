"""Aromatic exposure features aligned with organizer AROMATIC-TOPO definitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (
    AROMATIC,
    RASA_EXPOSED,
    compute_sasa,
    finite_sum,
    load_structure,
    residue_sasa_rows,
)

PATCH_CONTACT_A = 8.0
LOCAL_R = 10.0


def _patches(exposed: list[dict], radius: float = PATCH_CONTACT_A) -> list[list[int]]:
    n = len(exposed)
    if n == 0:
        return []
    coords = np.vstack([r["ca"] for r in exposed])
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not np.all(np.isfinite(coords[i])) or not np.all(np.isfinite(coords[j])):
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) <= radius:
                adj[i].append(j)
                adj[j].append(i)
    seen = set()
    comps = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def extract(
    pdb_path: str | Path,
    heavy_sequence: str | None = None,
    light_sequence: str | None = None,
    cdr_residue_keys: set[tuple[str, int]] | None = None,
) -> dict[str, float]:
    """Compute aromatic exposure / topology scalars from a PDB.

    Definitions follow organizer AROMATIC-TOPO v1 (Bio.PDB ShrakeRupley,
    RASA exposed ≥ 0.20, strongly exposed ≥ 0.50, patch contact 8 Å).

    CDR features require ``cdr_residue_keys`` as ``{(chain_id, resseq), ...}``.
    If omitted, CDR-specific fields are returned as NaN (not zero), except
    counts that are explicitly marked sequence-wide.

    Never returns -999 sentinels.
    """
    _ = (heavy_sequence, light_sequence)
    structure = load_structure(pdb_path)
    compute_sasa(structure)
    rows = residue_sasa_rows(structure)
    if not rows:
        raise ValueError(f"No standard residues in {pdb_path}")

    total_sasa = finite_sum(r["sasa"] for r in rows)
    arom = [r for r in rows if r["amino_acid"] in AROMATIC]
    exp = [r for r in arom if r["is_exposed"]]
    strong = [r for r in arom if r["is_strongly_exposed"]]
    arom_sasa = finite_sum(r["sasa"] for r in exp)

    if cdr_residue_keys is None:
        cdr_exp_count = float("nan")
        cdr_sasa = float("nan")
        cdr_frac = float("nan")
    else:
        cdr_exp = [r for r in exp if (r["chain"], r["resseq"]) in cdr_residue_keys]
        cdr_exp_count = float(len(cdr_exp))
        cdr_sasa = finite_sum(r["sasa"] for r in cdr_exp)
        cdr_frac = cdr_sasa / arom_sasa if arom_sasa > 0 else 0.0

    comps = _patches(exp)
    largest_n = max((len(c) for c in comps), default=0)
    largest_sasa = 0.0
    if comps:
        best = max(comps, key=len)
        largest_sasa = finite_sum(exp[i]["sasa"] for i in best)

    max_local = 0.0
    if exp:
        coords = np.vstack([r["ca"] for r in exp])
        for i, r in enumerate(exp):
            if not np.all(np.isfinite(coords[i])):
                continue
            local = 0.0
            for j, s in enumerate(exp):
                if not np.all(np.isfinite(coords[j])):
                    continue
                if float(np.linalg.norm(coords[i] - coords[j])) <= LOCAL_R:
                    local += float(s["sasa"]) if np.isfinite(s["sasa"]) else 0.0
            max_local = max(max_local, local)

    return {
        "exposed_TYR_count": float(sum(1 for r in exp if r["amino_acid"] == "Y")),
        "exposed_TRP_count": float(sum(1 for r in exp if r["amino_acid"] == "W")),
        "exposed_PHE_count": float(sum(1 for r in exp if r["amino_acid"] == "F")),
        "exposed_aromatic_total_count": float(len(exp)),
        "aromatic_exposed_SASA_total": arom_sasa,
        "aromatic_exposed_SASA_fraction": arom_sasa / total_sasa if total_sasa > 0 else 0.0,
        "strongly_exposed_aromatic_count": float(len(strong)),
        "strongly_exposed_aromatic_SASA": finite_sum(r["sasa"] for r in strong),
        "CDR_exposed_aromatic_count": cdr_exp_count,
        "CDR_aromatic_SASA": cdr_sasa,
        "CDR_aromatic_fraction": cdr_frac,
        "aromatic_patch_count": float(len(comps)),
        "largest_aromatic_patch_n_res": float(largest_n),
        "largest_aromatic_patch_exposed_SASA": float(largest_sasa),
        "max_local_aromatic_SASA": float(max_local),
        "sequence_aromatic_count": float(len(arom)),
        "rasa_exposed_threshold": float(RASA_EXPOSED),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract aromatic exposure features")
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
