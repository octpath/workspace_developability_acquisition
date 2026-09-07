"""PDB-derived SASA / RASA summaries (participant-facing)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (
    HYDROPHOBIC,
    POLAR,
    compute_sasa,
    finite_sum,
    load_structure,
    residue_sasa_rows,
)


def extract(pdb_path: str | Path, heavy_sequence: str | None = None, light_sequence: str | None = None) -> dict[str, float]:
    """Compute total and class-wise SASA features from a PDB.

    Parameters
    ----------
    pdb_path :
        Path to a PDB file (Fv or Fab).
    heavy_sequence, light_sequence :
        Optional; currently unused (reserved for future region mapping).
        Accepted so callers can share one API across extractors.

    Returns
    -------
    dict[str, float]
        Deterministic scalar features. Missing geometry yields NaN for
        affected fields; never returns sentinel values such as -999.

    Dependencies
    ------------
    biopython (Bio.PDB.SASA.ShrakeRupley)
    """
    _ = (heavy_sequence, light_sequence)
    structure = load_structure(pdb_path)
    compute_sasa(structure)
    rows = residue_sasa_rows(structure)
    if not rows:
        raise ValueError(f"No standard residues with SASA in {pdb_path}")

    total = finite_sum(r["sasa"] for r in rows)
    hydro = finite_sum(r["sasa"] for r in rows if r["amino_acid"] in HYDROPHOBIC)
    polar = finite_sum(r["sasa"] for r in rows if r["amino_acid"] in POLAR)
    rasas = np.asarray([r["rasa"] for r in rows if np.isfinite(r["rasa"])], dtype=float)

    return {
        "sasa_total": total,
        "sasa_hydrophobic": hydro,
        "sasa_polar": polar,
        "sasa_hydrophobic_fraction": hydro / total if total > 0 else float("nan"),
        "sasa_polar_fraction": polar / total if total > 0 else float("nan"),
        "rasa_mean": float(rasas.mean()) if rasas.size else float("nan"),
        "rasa_median": float(np.median(rasas)) if rasas.size else float("nan"),
        "n_residues": float(len(rows)),
        "n_exposed_rasa_ge_0_20": float(sum(1 for r in rows if r["is_exposed"])),
        "n_strongly_exposed_rasa_ge_0_50": float(sum(1 for r in rows if r["is_strongly_exposed"])),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract SASA features from a PDB")
    p.add_argument("--pdb", required=True)
    p.add_argument("--output", default=None, help="Optional JSON output path")
    args = p.parse_args(argv)
    feats = extract(args.pdb)
    text = json.dumps(feats, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
