#!/usr/bin/env python3
"""Build STRUCTURE_INPUT_CROSSWALK.csv for competition N=324 (Gate 1.1).

No feature extraction / no model training — path + sequence audit only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting"
OUT_CSV = OUT / "STRUCTURE_INPUT_CROSSWALK.csv"
OUT_AUDIT = OUT / "STRUCTURE_INPUT_CROSSWALK_AUDIT.json"

DEV = ROOT / "competition/data/distribution/dev.csv"
TEST_FEAT = ROOT / "competition/data/distribution/test_features.csv"
SOL = ROOT / "competition/data/secret/solution.csv"
POP = ROOT / "gate_b3/frozen/organizer/final_population.csv"
ESMFOLD_CSV = ROOT / "esmfold_native/esmfold_native.csv"
ESMFOLD_ID_DIR = ROOT / "esmfold_native"
ESMFOLD_HASH_DIR = ROOT / "gate_b2/cache/structures/esmfold_native"
ABB2_DIR = ROOT / "gate_b1/cache/structures/abodybuilder2"

AA3 = {k.upper(): v for k, v in protein_letters_3to1.items()}
AA3["MSE"] = "M"
AA3["SEC"] = "U"
AA3["PYL"] = "O"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def seq_from_pdb(path: Path) -> tuple[str | None, str | None, str]:
    """Return (heavy_seq, light_seq, note). Prefer chains A/B; else first two polymer chains."""
    if not path.exists():
        return None, None, "pdb_missing"
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", str(path))
        model = next(structure.get_models())
        chains = list(model.get_chains())
        by_id = {c.id: c for c in chains}

        def chain_seq(chain) -> str:
            seq = []
            seen = set()
            for res in chain.get_residues():
                het, resseq, icode = res.id
                if het.strip() not in ("", "W"):
                    # skip hetero except standard
                    if het.strip() != "":
                        # allow MSE etc via AA3
                        pass
                key = (resseq, icode)
                if key in seen:
                    continue
                if "CA" not in res:
                    continue
                name = res.get_resname().upper()
                aa = AA3.get(name)
                if aa is None:
                    continue
                seen.add(key)
                seq.append(aa)
            return "".join(seq)

        if "A" in by_id and "B" in by_id:
            return chain_seq(by_id["A"]), chain_seq(by_id["B"]), "chains_A_B"
        # fallback: first two chains with CA
        polymer = []
        for c in chains:
            s = chain_seq(c)
            if len(s) >= 50:
                polymer.append((c.id, s))
        if len(polymer) >= 2:
            return polymer[0][1], polymer[1][1], f"chains_{polymer[0][0]}_{polymer[1][0]}"
        if len(polymer) == 1:
            return polymer[0][1], None, f"single_chain_{polymer[0][0]}"
        return None, None, "no_polymer_chains"
    except Exception as e:
        return None, None, f"parse_error:{type(e).__name__}"


def match_status(expected: str, observed: str | None) -> str:
    if observed is None:
        return "MISSING"
    if observed == expected:
        return "EXACT_MATCH"
    # allow if one is prefix of other (truncated termini) — still flag
    if expected in observed or observed in expected:
        return "PASS_EXPLAINED_SUBSTRING"
    # length-only mismatch diagnostics
    if abs(len(expected) - len(observed)) <= 2:
        # count mismatches
        n = min(len(expected), len(observed))
        mism = sum(1 for a, b in zip(expected[:n], observed[:n]) if a != b)
        if mism == 0:
            return "PASS_EXPLAINED_LENGTH_DELTA"
        if mism <= 3:
            return "PASS_EXPLAINED_NEAR_MATCH"
    return "MISMATCH"


def mapping_status(row: dict) -> str:
    if not row["esmfold_exists"] and not row["abb2_exists"]:
        return "FAIL_BOTH_MISSING"
    if not row["esmfold_exists"]:
        return "FAIL_ESMFOLD_MISSING"
    if not row["abb2_exists"]:
        return "FAIL_ABB2_MISSING"
    bad = {"MISMATCH", "MISSING"}
    ef = {row["esmfold_heavy_sequence_match"], row["esmfold_light_sequence_match"]}
    ab = {row["abb2_heavy_sequence_match"], row["abb2_light_sequence_match"]}
    if ef & bad or ab & bad:
        # explained pass still OK for canonical if exact on both for at least one generator?
        if "MISMATCH" in ef or "MISMATCH" in ab or "MISSING" in ef or "MISSING" in ab:
            # allow PASS_EXPLAINED* without FAIL if no MISMATCH/MISSING
            if ("MISMATCH" in ef | ab) or ("MISSING" in ef | ab):
                return "FAIL_SEQUENCE_MISMATCH"
    explained = {
        "PASS_EXPLAINED_SUBSTRING",
        "PASS_EXPLAINED_LENGTH_DELTA",
        "PASS_EXPLAINED_NEAR_MATCH",
    }
    if (ef | ab) & explained and not ((ef | ab) & {"MISMATCH", "MISSING"}):
        if ef <= {"EXACT_MATCH"} | explained and ab <= {"EXACT_MATCH"} | explained:
            return "PASS_EXPLAINED"
    if ef <= {"EXACT_MATCH"} and ab <= {"EXACT_MATCH"}:
        return "PASS"
    if (ef | ab) & {"MISMATCH", "MISSING"}:
        return "FAIL_SEQUENCE_MISMATCH"
    return "PASS_EXPLAINED"


def main() -> None:
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST_FEAT)
    sol = pd.read_csv(SOL)
    # sequences: prefer competition distribution files; fallback population
    pop = pd.read_csv(POP)
    seq_map = {}
    for df, hcol, lcol in [
        (dev, "heavy", "light"),
        (test, "heavy", "light"),
        (pop, "heavy", "light"),
    ]:
        for _, r in df.iterrows():
            seq_map.setdefault(r["id"], (r[hcol], r[lcol]))

    public_ids = set(sol.loc[sol["is_public"].astype(str).str.lower().isin(["true", "1"]), "id"])
    private_ids = set(sol.loc[sol["is_private"].astype(str).str.lower().isin(["true", "1"]), "id"])
    dev_ids = set(dev["id"])

    esm = pd.read_csv(ESMFOLD_CSV)
    esm_by_id = esm.set_index("antibody_id")

    # also ABB2 json index
    abb2_json_by_id = {}
    for jp in ABB2_DIR.glob("*.json"):
        try:
            meta = json.loads(jp.read_text())
            aid = meta.get("antibody_id")
            if aid:
                abb2_json_by_id[aid] = meta
        except Exception:
            pass

    rows = []
    for aid, (vh, vl) in sorted(seq_map.items()):
        if aid not in set(list(dev_ids) + list(sol["id"])):
            # keep competition only: union of dev + solution
            continue
        if aid in dev_ids:
            split = "DEV"
        elif aid in public_ids:
            split = "PUBLIC"
        elif aid in private_ids:
            split = "PRIVATE"
        else:
            split = "UNKNOWN"

        pair_hash = None
        if aid in esm_by_id.index:
            pair_hash = str(esm_by_id.loc[aid, "pair_hash"])
        elif aid in abb2_json_by_id:
            pair_hash = abb2_json_by_id[aid].get("pair_hash")

        id_path = ESMFOLD_ID_DIR / f"{aid}.pdb"
        hash_path = ESMFOLD_HASH_DIR / f"{pair_hash}.pdb" if pair_hash else None
        # canonical: prefer antibody-id path if exists, else hash path (inventory primary naming)
        if id_path.exists():
            canonical = id_path
            can_note = "antibody_id_path"
        elif hash_path is not None and hash_path.exists():
            canonical = hash_path
            can_note = "pair_hash_path_fallback"
        else:
            canonical = id_path
            can_note = "missing_both_esmfold_candidates"

        abb2_path = None
        if aid in abb2_json_by_id and abb2_json_by_id[aid].get("pdb_path"):
            abb2_path = Path(abb2_json_by_id[aid]["pdb_path"])
        elif pair_hash:
            abb2_path = ABB2_DIR / f"{pair_hash}.pdb"
        else:
            abb2_path = ABB2_DIR / f"{aid}.pdb"

        ef_h, ef_l, ef_note = seq_from_pdb(canonical if canonical.exists() else Path("/dev/null"))
        if not canonical.exists():
            ef_h, ef_l, ef_note = None, None, "pdb_missing"
        ab_h, ab_l, ab_note = seq_from_pdb(abb2_path) if abb2_path and abb2_path.exists() else (None, None, "pdb_missing")

        row = {
            "id": aid,
            "dataset_split": split,
            "expected_VH_sequence_hash": sha256_text(vh),
            "expected_VL_sequence_hash": sha256_text(vl),
            "esmfold_antibody_id_path": str(id_path),
            "esmfold_pair_hash_path": str(hash_path) if hash_path else "",
            "esmfold_canonical_path": str(canonical),
            "abodybuilder2_path": str(abb2_path) if abb2_path else "",
            "esmfold_exists": bool(canonical.exists()),
            "abb2_exists": bool(abb2_path.exists()) if abb2_path else False,
            "esmfold_heavy_sequence_match": match_status(vh, ef_h) if canonical.exists() else "MISSING",
            "esmfold_light_sequence_match": match_status(vl, ef_l) if canonical.exists() else "MISSING",
            "abb2_heavy_sequence_match": match_status(vh, ab_h) if abb2_path and abb2_path.exists() else "MISSING",
            "abb2_light_sequence_match": match_status(vl, ab_l) if abb2_path and abb2_path.exists() else "MISSING",
            "notes": f"esmfold_source={can_note};esmfold_parse={ef_note};abb2_parse={ab_note};pair_hash={pair_hash or ''}",
        }
        row["canonical_mapping_status"] = mapping_status(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    # ensure only competition 324
    assert len(df) == 324, f"expected 324, got {len(df)}"
    assert (df["dataset_split"] == "DEV").sum() == 162
    assert (df["dataset_split"] == "PUBLIC").sum() == 81
    assert (df["dataset_split"] == "PRIVATE").sum() == 81

    cols = [
        "id",
        "dataset_split",
        "expected_VH_sequence_hash",
        "expected_VL_sequence_hash",
        "esmfold_antibody_id_path",
        "esmfold_pair_hash_path",
        "esmfold_canonical_path",
        "abodybuilder2_path",
        "esmfold_exists",
        "abb2_exists",
        "esmfold_heavy_sequence_match",
        "esmfold_light_sequence_match",
        "abb2_heavy_sequence_match",
        "abb2_light_sequence_match",
        "canonical_mapping_status",
        "notes",
    ]
    df[cols].to_csv(OUT_CSV, index=False)

    audit = {
        "crosswalk_version": "v1",
        "n_rows": int(len(df)),
        "split_counts": df["dataset_split"].value_counts().to_dict(),
        "esmfold_exists": int(df["esmfold_exists"].sum()),
        "abb2_exists": int(df["abb2_exists"].sum()),
        "mapping_status_counts": df["canonical_mapping_status"].value_counts().to_dict(),
        "esmfold_heavy_match_counts": df["esmfold_heavy_sequence_match"].value_counts().to_dict(),
        "esmfold_light_match_counts": df["esmfold_light_sequence_match"].value_counts().to_dict(),
        "abb2_heavy_match_counts": df["abb2_heavy_sequence_match"].value_counts().to_dict(),
        "abb2_light_match_counts": df["abb2_light_sequence_match"].value_counts().to_dict(),
        "canonical_path_source": {
            "antibody_id_path": int(df["notes"].str.contains("esmfold_source=antibody_id_path").sum()),
            "pair_hash_path_fallback": int(df["notes"].str.contains("esmfold_source=pair_hash_path_fallback").sum()),
        },
        "output": str(OUT_CSV),
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
