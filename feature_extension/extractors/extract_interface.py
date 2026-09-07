"""Simple Fab domain-interface descriptors (BSA proxy; not classical Sc)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import compute_sasa, finite_sum, load_structure, residue_sasa_rows


def _chain_map(structure) -> dict[str, str]:
    """Map observed chain IDs to roles when possible.

    Common conventions in this project:
      Fv: H/L or A/B
      Fab: H (VH+CH1), L (VL+CL) — domain split by approximate residue index
            is NOT performed here; callers may pass explicit domain chain IDs.
    """
    chains = []
    for model in structure:
        for chain in model:
            chains.append(chain.id)
        break
    return {c: c for c in chains}


def extract(
    pdb_path: str | Path,
    heavy_sequence: str | None = None,
    light_sequence: str | None = None,
    chain_heavy: str | None = None,
    chain_light: str | None = None,
    contact_cutoff: float = 4.5,
) -> dict[str, float]:
    """Approximate VH–VL interface descriptors for a two-chain Fv/Fab PDB.

    Computes:
      - buried SASA proxy = SASA(H)+SASA(L) - SASA(complex) on interface-proximal residues
      - interface residue count (heavy-atom contacts ≤ contact_cutoff)
      - contact density = n_contacts / n_interface_residues

    This is **not** classical shape complementarity (Sc).
    Domain interfaces VH–CH1 / VL–CL / CH1–CL require four structural domains;
    for Fv-only PDBs only the heavy–light interface is reported.
    """
    _ = (heavy_sequence, light_sequence)
    structure = load_structure(pdb_path)
    chains = list(_chain_map(structure).keys())
    if chain_heavy is None or chain_light is None:
        if len(chains) < 2:
            raise ValueError(f"Need ≥2 chains for interface extraction; found {chains}")
        # prefer H/L naming
        if "H" in chains and "L" in chains:
            chain_heavy, chain_light = "H", "L"
        else:
            chain_heavy, chain_light = chains[0], chains[1]

    # Collect atoms by chain
    atoms = {chain_heavy: [], chain_light: []}
    for model in structure:
        for chain in model:
            if chain.id not in atoms:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    if atom.element == "H":
                        continue
                    atoms[chain.id].append((chain.id, res.id[1], atom.coord))

    if not atoms[chain_heavy] or not atoms[chain_light]:
        raise ValueError("Empty chain atom sets for interface")

    # Contact residues
    h_res = set()
    l_res = set()
    n_contacts = 0
    h_coords = np.asarray([a[2] for a in atoms[chain_heavy]], dtype=float)
    l_coords = np.asarray([a[2] for a in atoms[chain_light]], dtype=float)
    h_meta = [(a[0], a[1]) for a in atoms[chain_heavy]]
    l_meta = [(a[0], a[1]) for a in atoms[chain_light]]
    # block pairwise (modest size Fv/Fab)
    for i, hc in enumerate(h_coords):
        d = np.linalg.norm(l_coords - hc, axis=1)
        hits = np.where(d <= contact_cutoff)[0]
        if hits.size:
            h_res.add(h_meta[i])
            for j in hits:
                l_res.add(l_meta[j])
                n_contacts += 1

    n_iface = len(h_res) + len(l_res)
    compute_sasa(structure)
    rows = residue_sasa_rows(structure)
    iface_keys = h_res | l_res
    complex_iface_sasa = finite_sum(
        r["sasa"] for r in rows if (r["chain"], r["resseq"]) in iface_keys
    )

    return {
        "iface_heavy_light_n_res": float(n_iface),
        "iface_heavy_light_n_contacts": float(n_contacts),
        "iface_heavy_light_contact_density": float(n_contacts / n_iface) if n_iface else float("nan"),
        "iface_heavy_light_complex_sasa_proxy": float(complex_iface_sasa),
        "contact_cutoff_A": float(contact_cutoff),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract simple interface features")
    p.add_argument("--pdb", required=True)
    p.add_argument("--heavy-chain", default=None)
    p.add_argument("--light-chain", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    feats = extract(args.pdb, chain_heavy=args.heavy_chain, chain_light=args.light_chain)
    text = json.dumps(feats, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
