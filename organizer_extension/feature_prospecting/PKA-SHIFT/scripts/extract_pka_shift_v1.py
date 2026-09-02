#!/usr/bin/env python3
"""Target-blind PKA-SHIFT_v1 extraction (PROPKA 3.5.1 only)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "PKA-SHIFT"
CROSSWALK = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
REF_PKA = {k: float(v) for k, v in SPEC["model_reference_pKa"].items()}
SIDE = set(SPEC["canonical_sidechain_residue_types"])
TERMINI = set(SPEC["termini_policy"]["exclude_from_canonical_features"])
RASA_EXPOSED = float(SPEC["region_definitions"]["buried_exposed"]["rasa_exposed_threshold"])
IFACE_CA = 5.0
PROBE = 1.4
PROPKA3 = ROOT / ".venv_stage4/bin/propka3"
N_WORKERS = 8

MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}

SUMMARY_RE = re.compile(
    r"^\s*(ASP|GLU|HIS|CYS|TYR|LYS|ARG|N\+|C-)\s+(\d+)\s+([A-Za-z0-9])\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_sequences() -> pd.DataFrame:
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def parse_summary(pka_text: str) -> list[dict]:
    lines = pka_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "SUMMARY OF THIS PREDICTION" in line:
            start = i + 1
            break
    if start is None:
        return []
    out = []
    for line in lines[start:]:
        if line.strip().startswith("---") and out:
            break
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        g, resseq, chain, pka, model = m.groups()
        out.append(
            {
                "group": g,
                "pdb_resseq": int(resseq),
                "pdb_chain": chain,
                "predicted_pKa": float(pka),
                "reference_pKa_from_file": float(model),
            }
        )
    return out


def run_propka(pdb_path: Path) -> tuple[str | None, str | None]:
    with tempfile.TemporaryDirectory(prefix="pka_") as td:
        td = Path(td)
        try:
            proc = subprocess.run(
                [str(PROPKA3), str(pdb_path)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as e:
            return None, f"propka_exception:{type(e).__name__}:{e}"
        pkas = list(td.glob("*.pka"))
        if not pkas:
            # sometimes written next to input
            alt = pdb_path.with_suffix(".pka")
            if alt.exists():
                text = alt.read_text(errors="ignore")
                try:
                    alt.unlink()
                except Exception:
                    pass
                return text, None
            return None, f"no_pka_file:rc={proc.returncode}:{proc.stderr[-300:]}"
        return pkas[0].read_text(errors="ignore"), None


def build_residue_table(pdb_path: Path, heavy: str, light: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    # SASA on full model
    sr = ShrakeRupley(probe_radius=PROBE, n_points=100)
    sr.compute(model, level="R")

    chain_seqs = {}
    for ch in model.get_chains():
        aas = []
        for res in ch.get_residues():
            if is_aa(res, standard=True):
                aas.append(aa1(res.get_resname()))
        chain_seqs[ch.id] = "".join(aas)

    # map PDB chain -> H/L by exact sequence match
    pdb_to_hl = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            pdb_to_hl[cid] = "H"
        elif seq == light:
            pdb_to_hl[cid] = "L"
    if set(pdb_to_hl.values()) != {"H", "L"}:
        # fallback: length-unique or first two
        return None, f"chain_map_fail:{pdb_to_hl}:{ {k:len(v) for k,v in chain_seqs.items()} }"

    rows = []
    for ch in model.get_chains():
        if ch.id not in pdb_to_hl:
            continue
        hl = pdb_to_hl[ch.id]
        seq_i = 0
        for res in ch.get_residues():
            if not is_aa(res, standard=True):
                continue
            a = aa1(res.get_resname())
            sasa = float(getattr(res, "sasa", np.nan))
            maxasa = MAX_ASA.get(a)
            rasa = sasa / maxasa if maxasa else np.nan
            ca = res["CA"].coord if "CA" in res else None
            rows.append(
                {
                    "pdb_chain": ch.id,
                    "chain": hl,
                    "sequence_index": seq_i,
                    "amino_acid": a,
                    "resname3": res.get_resname().strip().upper(),
                    "pdb_resseq": int(res.id[1]),
                    "pdb_icode": res.id[2],
                    "sasa": sasa,
                    "rasa": rasa,
                    "is_buried": bool(np.isfinite(rasa) and rasa < RASA_EXPOSED),
                    "ca": ca,
                }
            )
            seq_i += 1
    # verify sequence
    for hl, expected in [("H", heavy), ("L", light)]:
        got = "".join(r["amino_acid"] for r in rows if r["chain"] == hl)
        if got != expected:
            return None, f"seq_mismatch_{hl}:len={len(got)}/{len(expected)}"
    # interface
    h = [r for r in rows if r["chain"] == "H" and r["ca"] is not None]
    l = [r for r in rows if r["chain"] == "L" and r["ca"] is not None]
    iface = set()
    for i, ri in enumerate(h):
        for j, rj in enumerate(l):
            if np.linalg.norm(ri["ca"] - rj["ca"]) <= IFACE_CA:
                iface.add(("H", ri["sequence_index"]))
                iface.add(("L", rj["sequence_index"]))
    for r in rows:
        r["is_vh_vl_interface"] = (r["chain"], r["sequence_index"]) in iface
        del r["ca"]
    return rows, None


def aggregate_features(res_df: pd.DataFrame) -> dict:
    """res_df: sidechain-only titratable with delta_pKa."""
    d = res_df["delta_pKa"].to_numpy(float)
    ad = np.abs(d)
    out = {}

    def zempty(arr, fn):
        arr = np.asarray(arr, float)
        if len(arr) == 0:
            return 0.0
        return float(fn(arr))

    out["mean_abs_delta_pKa"] = zempty(ad, np.mean)
    out["rms_delta_pKa"] = float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0
    out["max_abs_delta_pKa"] = zempty(ad, np.max)
    out["mean_signed_delta_pKa"] = zempty(d, np.mean)
    out["frac_abs_delta_ge_1"] = float(np.mean(ad >= 1)) if len(ad) else 0.0
    out["frac_abs_delta_ge_2"] = float(np.mean(ad >= 2)) if len(ad) else 0.0
    out["max_positive_delta_pKa"] = zempty(d[d > 0] if np.any(d > 0) else [], np.max) if np.any(d > 0) else 0.0
    out["max_negative_magnitude_delta_pKa"] = (
        float(np.max(-d[d < 0])) if np.any(d < 0) else 0.0
    )

    acidic = res_df[res_df["residue_type"].isin(["ASP", "GLU"])]["delta_pKa"].to_numpy(float)
    basic = res_df[res_df["residue_type"].isin(["HIS", "LYS", "ARG"])]["delta_pKa"].to_numpy(float)
    out["acidic_mean_abs_delta"] = zempty(np.abs(acidic), np.mean)
    out["acidic_mean_signed_delta"] = zempty(acidic, np.mean)
    out["basic_mean_abs_delta"] = zempty(np.abs(basic), np.mean)
    out["basic_mean_signed_delta"] = zempty(basic, np.mean)

    cdr = res_df[res_df["is_cdr"]]["delta_pKa"].to_numpy(float)
    fw = res_df[res_df["is_framework"]]["delta_pKa"].to_numpy(float)
    out["cdr_mean_abs_delta"] = zempty(np.abs(cdr), np.mean)
    out["cdr_max_abs_delta"] = zempty(np.abs(cdr), np.max)
    out["cdr_frac_abs_ge_1"] = float(np.mean(np.abs(cdr) >= 1)) if len(cdr) else 0.0
    out["framework_mean_abs_delta"] = zempty(np.abs(fw), np.mean)
    out["cdr_minus_framework_mean_abs_delta"] = out["cdr_mean_abs_delta"] - out["framework_mean_abs_delta"]

    iface = res_df[res_df["is_vh_vl_interface"]]["delta_pKa"].to_numpy(float)
    non = res_df[~res_df["is_vh_vl_interface"]]["delta_pKa"].to_numpy(float)
    out["interface_mean_abs_delta"] = zempty(np.abs(iface), np.mean)
    out["interface_max_abs_delta"] = zempty(np.abs(iface), np.max)
    out["interface_frac_abs_ge_1"] = float(np.mean(np.abs(iface) >= 1)) if len(iface) else 0.0
    out["interface_minus_noninterface_mean_abs_delta"] = out["interface_mean_abs_delta"] - zempty(np.abs(non), np.mean)

    bur = res_df[res_df["is_buried"]]["delta_pKa"].to_numpy(float)
    exp = res_df[~res_df["is_buried"]]["delta_pKa"].to_numpy(float)
    out["buried_mean_abs_delta"] = zempty(np.abs(bur), np.mean)
    out["buried_max_abs_delta"] = zempty(np.abs(bur), np.max)
    out["buried_frac_abs_ge_1"] = float(np.mean(np.abs(bur) >= 1)) if len(bur) else 0.0
    out["buried_minus_exposed_mean_abs_delta"] = out["buried_mean_abs_delta"] - zempty(np.abs(exp), np.mean)
    return out


def process_one(args):
    aid, gen, pdb_path, heavy, light, cdr_rows = args
    pdb_path = Path(pdb_path)
    out = {
        "id": aid,
        "generator": gen,
        "pdb_path": str(pdb_path),
        "extraction_status": "FAIL",
    }
    if not pdb_path.exists():
        out["error"] = "missing_pdb"
        return out, [], {}
    rows, err = build_residue_table(pdb_path, heavy, light)
    if err:
        out["error"] = err
        return out, [], {}
    # attach CDR
    cdr_map = {(r["chain"], int(r["sequence_index"])): r for r in cdr_rows}
    for r in rows:
        key = (r["chain"], r["sequence_index"])
        c = cdr_map.get(key)
        if c is None:
            r["region"] = "UNMAPPED"
            r["is_cdr"] = False
            r["is_framework"] = False
            r["mapping_status"] = "CDR_INDEX_MISS"
        else:
            if c["amino_acid"] != r["amino_acid"]:
                r["region"] = c["region"]
                r["is_cdr"] = bool(c["is_cdr"])
                r["is_framework"] = bool(c["is_framework"])
                r["mapping_status"] = "AA_MISMATCH"
            else:
                r["region"] = c["region"]
                r["is_cdr"] = bool(c["is_cdr"])
                r["is_framework"] = bool(c["is_framework"])
                r["mapping_status"] = "OK"

    pka_text, perr = run_propka(pdb_path)
    if perr or not pka_text:
        out["error"] = perr or "empty_pka"
        return out, [], {}
    groups = parse_summary(pka_text)
    # index residues by (pdb_chain, pdb_resseq) — unique for standard Fv
    res_by_pdb = {}
    for r in rows:
        res_by_pdb[(r["pdb_chain"], r["pdb_resseq"])] = r

    residue_out = []
    n_termini = 0
    n_unmapped = 0
    for g in groups:
        key = (g["pdb_chain"], g["pdb_resseq"])
        if g["group"] in TERMINI:
            n_termini += 1
            # QC-only row optional: skip residue parquet for termini to keep canonical clean
            continue
        if g["group"] not in SIDE:
            continue
        r = res_by_pdb.get(key)
        if r is None:
            n_unmapped += 1
            continue
        # residue type from group name
        rt = g["group"]
        ref = REF_PKA[rt]
        # sanity: reference from file should match freeze
        pred = g["predicted_pKa"]
        delta = pred - ref
        residue_out.append(
            {
                "id": aid,
                "generator": gen,
                "chain": r["chain"],
                "sequence_index": r["sequence_index"],
                "residue_type": rt,
                "amino_acid": r["amino_acid"],
                "predicted_pKa": pred,
                "reference_pKa": ref,
                "delta_pKa": delta,
                "abs_delta_pKa": abs(delta),
                "region": r["region"],
                "is_cdr": r["is_cdr"],
                "is_framework": r["is_framework"],
                "is_vh_vl_interface": r["is_vh_vl_interface"],
                "rasa": r["rasa"],
                "is_buried": r["is_buried"],
                "mapping_status": r["mapping_status"],
                "pdb_chain": r["pdb_chain"],
                "pdb_resseq": r["pdb_resseq"],
            }
        )

    rdf = pd.DataFrame(residue_out)
    if len(rdf) == 0:
        out["error"] = "no_sidechain_pka_mapped"
        out["n_termini_excluded"] = n_termini
        out["n_unmapped_propka"] = n_unmapped
        return out, [], {}

    feat = aggregate_features(rdf)
    feat.update(
        {
            "id": aid,
            "generator": gen,
            "extraction_status": "SUCCESS",
            "pdb_path": str(pdb_path),
            "pdb_sha256": sha256_file(pdb_path),
            "n_titratable_sidechain": int(len(rdf)),
            "n_cdr_titratable": int(rdf["is_cdr"].sum()),
            "n_interface_titratable": int(rdf["is_vh_vl_interface"].sum()),
            "n_buried_titratable": int(rdf["is_buried"].sum()),
            "n_termini_excluded": n_termini,
            "n_unmapped_propka": n_unmapped,
            "sequence_length_HL": len(heavy) + len(light),
            "total_SASA": float(sum(r["sasa"] for r in rows if np.isfinite(r["sasa"]))),
            "n_aa_mismatch": int((rdf["mapping_status"] == "AA_MISMATCH").sum()),
            "n_cdr_index_miss": int((rdf["mapping_status"] == "CDR_INDEX_MISS").sum()),
        }
    )
    for aa in SIDE:
        feat[f"n_type_{aa}"] = int((rdf["residue_type"] == aa).sum())
    out.update(feat)
    out["extraction_status"] = "SUCCESS"
    out["error"] = ""
    return out, residue_out, feat


def main():
    FAMILY.mkdir(parents=True, exist_ok=True)
    cw = pd.read_csv(CROSSWALK)
    assert len(cw) == 324
    seqs = load_sequences()
    cdr = pd.read_csv(FAMILY / "cdr_sequence_index_imgt.csv")
    cdr_by_id = {aid: g.to_dict("records") for aid, g in cdr.groupby("id")}

    # SOFTWARE_ENV
    env = {
        "python": subprocess.check_output([str(ROOT / ".venv_stage4/bin/python"), "-V"], text=True).strip(),
        "propka_version": "3.5.1",
        "propka_cli": str(PROPKA3),
        "propka_package_path": str(ROOT / ".venv_stage4/lib/python3.12/site-packages/propka"),
        "license": "LGPL v2.1",
        "installation_source": ".venv_stage4 Round1 Stage4 env",
        "biopython_sasa": "Bio.PDB.SASA.ShrakeRupley",
    }
    (FAMILY / "SOFTWARE_ENV.json").write_text(json.dumps(env, indent=2) + "\n")

    all_qc = []
    all_man = []
    for gen, col in GEN_PATH.items():
        jobs = []
        for _, r in cw.iterrows():
            aid = r["id"]
            jobs.append(
                (
                    aid,
                    gen,
                    str(r[col]),
                    seqs.loc[aid, "heavy"],
                    seqs.loc[aid, "light"],
                    cdr_by_id[aid],
                )
            )
        feat_rows = []
        res_rows = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(process_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                out, residues, _ = fut.result()
                feat_rows.append(out)
                res_rows.extend(residues)
                if i % 50 == 0:
                    print(gen, i, flush=True)
        fdf = pd.DataFrame(feat_rows)
        rdf = pd.DataFrame(res_rows)
        fdf.to_parquet(FAMILY / f"features_{gen}.parquet", index=False)
        rdf.to_parquet(FAMILY / f"residue_pka_{gen}.parquet", index=False)
        ok = int((fdf["extraction_status"] == "SUCCESS").sum())
        print(f"{gen}: {ok}/{len(fdf)} residues={len(rdf)}", flush=True)
        all_qc.append(fdf)
        all_man.append(
            fdf[
                [
                    c
                    for c in [
                        "id",
                        "generator",
                        "extraction_status",
                        "pdb_path",
                        "pdb_sha256",
                        "error",
                        "n_titratable_sidechain",
                        "n_unmapped_propka",
                        "n_termini_excluded",
                        "n_aa_mismatch",
                    ]
                    if c in fdf.columns
                ]
            ]
        )

    qc = pd.concat(all_qc, ignore_index=True)
    qc.to_csv(FAMILY / "extraction_qc.csv", index=False)
    pd.concat(all_man, ignore_index=True).to_csv(FAMILY / "FEATURE_MANIFEST.csv", index=False)

    # hash freeze (before robustness is also hashed after)
    freeze = {
        "state": "PKA_SHIFT_V1_FEATURE_SPEC_FROZEN",
        "target_scoring_started_after_feature_freeze": False,
        "files": {},
    }
    for p in [
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "SOFTWARE_ENV.json",
        FAMILY / "cdr_sequence_index_imgt.csv",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
        FAMILY / "residue_pka_esmfold.parquet",
        FAMILY / "residue_pka_abodybuilder2.parquet",
        FAMILY / "residue_pka_boltz2.parquet",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {"sha256": sha256_file(p), "nbytes": p.stat().st_size}
    (FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print("wrote target-blind hashes (pre-robustness)")


if __name__ == "__main__":
    main()
