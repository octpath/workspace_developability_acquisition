#!/usr/bin/env python3
"""Target-blind TITRATION-SHAPE_v1: PROPKA HH independent-site Q(pH) curves."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "TITRATION-SHAPE"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
PROPKA3 = ROOT / ".venv_stage4/bin/propka3"
CACHE = FP / "_batch3_cache" / "titration"
CACHE.mkdir(parents=True, exist_ok=True)
# reuse PKA-SHIFT cache if present
PKA_CACHE = FP / "PKA-SHIFT" / "cache"
GENS = ["esmfold", "abodybuilder2", "boltz2"]
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}
PH = np.arange(4.0, 10.0 + 1e-9, 0.25)
ACIDS = {"ASP", "GLU", "CYS", "TYR"}
BASES = {"HIS", "LYS", "ARG"}
SUMMARY_RE = re.compile(
    r"^\s*(ASP|GLU|HIS|CYS|TYR|LYS|ARG|N\+|C-)\s+(\d+)\s+([A-Za-z0-9])\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)"
)
N_WORKERS = 8


def aa1(resname: str) -> str:
    try:
        return protein_letters_3to1[resname.capitalize()]
    except Exception:
        return "X"


def load_sequences():
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def load_cdr_map():
    cdr = pd.read_csv(FP / "cdr_sequence_index_imgt.csv")
    out = {}
    for aid, g in cdr.groupby("id"):
        s = set()
        for _, r in g.iterrows():
            flag = r["is_cdr"]
            if isinstance(flag, str):
                flag = flag.strip().lower() in ("1", "true", "t", "yes")
            if flag:
                s.add((str(r["chain"]), int(r["sequence_index"])))
        out[aid] = s
    return out


def parse_summary(text: str):
    rows = []
    for line in text.splitlines():
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "resname": m.group(1),
                "resseq": int(m.group(2)),
                "chain": m.group(3),
                "pka": float(m.group(4)),
                "model_pka": float(m.group(5)),
            }
        )
    return rows


def map_hl(pdb_path: Path, heavy: str, light: str):
    model = next(PDBParser(QUIET=True).get_structure("x", str(pdb_path)).get_models())
    chain_seqs = {}
    for ch in model.get_chains():
        chain_seqs[ch.id] = "".join(aa1(r.get_resname()) for r in ch.get_residues() if is_aa(r, standard=True))
    m = {}
    for cid, seq in chain_seqs.items():
        if seq == heavy:
            m[cid] = "H"
        elif seq == light:
            m[cid] = "L"
    if set(m.values()) != {"H", "L"}:
        return None, f"chain_map_fail:{m}"
    # resseq -> (hl, seq_i)
    key = {}
    for ch in model.get_chains():
        if ch.id not in m:
            continue
        hl = m[ch.id]
        seq_i = 0
        for res in ch.get_residues():
            if not is_aa(res, standard=True):
                continue
            key[(ch.id, res.id[1])] = (hl, seq_i)
            seq_i += 1
    return key, None


def site_charge(resname: str, pka: float, ph: float) -> float:
    if resname in ACIDS:
        return -1.0 / (1.0 + 10 ** (pka - ph))
    if resname in BASES:
        return 1.0 / (1.0 + 10 ** (ph - pka))
    return 0.0


def curve_features(sites, cdr_set, key_map):
    # sites: list of {resname,pka,hl,seq_i} sidechain only
    Q = np.zeros(len(PH))
    Q_cdr = np.zeros(len(PH))
    for i, ph in enumerate(PH):
        q = 0.0
        qc = 0.0
        for s in sites:
            c = site_charge(s["resname"], s["pka"], float(ph))
            q += c
            if (s["hl"], s["seq_i"]) in cdr_set:
                qc += c
        Q[i] = q
        Q_cdr[i] = qc
    # dQ/dpH central difference
    dQ = np.gradient(Q, PH)
    abs_dQ = np.abs(dQ)
    imax = int(np.argmax(abs_dQ))
    thr = 0.5 * abs_dQ[imax] if abs_dQ[imax] > 0 else np.inf
    # major switch regions
    mask = abs_dQ >= thr if np.isfinite(thr) else np.zeros_like(abs_dQ, dtype=bool)
    n_regions = 0
    width = 0.0
    if mask.any():
        in_seg = False
        seg_start = None
        spans = []
        for i, m in enumerate(mask):
            if m and not in_seg:
                in_seg = True
                seg_start = i
            elif not m and in_seg:
                spans.append((seg_start, i - 1))
                in_seg = False
        if in_seg:
            spans.append((seg_start, len(mask) - 1))
        n_regions = len(spans)
        width = max((PH[b] - PH[a] for a, b in spans), default=0.0)

    def Q_at(ph0):
        # nearest grid
        j = int(np.argmin(np.abs(PH - ph0)))
        return float(Q[j])

    def Qcdr_at(ph0):
        j = int(np.argmin(np.abs(PH - ph0)))
        return float(Q_cdr[j])

    mid = (PH >= 5.0) & (PH <= 8.0)
    return {
        "Q_pH5": Q_at(5.0),
        "Q_pH6": Q_at(6.0),
        "Q_pH6_5": Q_at(6.5),
        "Q_pH7": Q_at(7.0),
        "Q_pH7_4": Q_at(7.4),
        "Q_pH8": Q_at(8.0),
        "Q_range_4_10": float(Q.max() - Q.min()),
        "max_abs_dQ_dpH": float(abs_dQ[imax]),
        "pH_at_max_abs_dQ_dpH": float(PH[imax]),
        "mean_abs_dQ_dpH_5_8": float(abs_dQ[mid].mean()),
        "charge_transition_width": float(width),
        "number_of_major_switch_regions": int(n_regions),
        "Q_CDR_pH6_5": Qcdr_at(6.5),
        "Q_CDR_pH7_4": Qcdr_at(7.4),
        "Q_curve": Q.tolist(),
        "Q_cdr_curve": Q_cdr.tolist(),
        "pH_grid": PH.tolist(),
    }


def get_pka_text(aid, gen, pdb_path: Path) -> str:
    # prefer PKA-SHIFT cache
    cand = [
        PKA_CACHE / f"{aid}_{gen}" / f"{Path(pdb_path).stem}.pka",
        PKA_CACHE / f"{aid}_{gen}" / "out.pka",
        CACHE / f"{aid}_{gen}.pka",
    ]
    for c in cand:
        if c.exists():
            return c.read_text(errors="ignore")
    # search glob in pka cache dir
    d = PKA_CACHE / f"{aid}_{gen}"
    if d.exists():
        pkas = list(d.glob("*.pka"))
        if pkas:
            return pkas[0].read_text(errors="ignore")
    # run propka
    work = CACHE / f"{aid}_{gen}"
    work.mkdir(parents=True, exist_ok=True)
    out_pka = work / "out.pka"
    if not out_pka.exists():
        r = subprocess.run(
            [str(PROPKA3), str(pdb_path)],
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=180,
        )
        produced = list(work.glob("*.pka"))
        if not produced:
            # propka may write next to pdb
            produced = list(Path(pdb_path).parent.glob(Path(pdb_path).stem + ".pka"))
        if produced:
            out_pka.write_text(produced[0].read_text(errors="ignore"))
        elif r.returncode != 0:
            raise RuntimeError(r.stderr[-300:] or r.stdout[-300:])
    return out_pka.read_text(errors="ignore")


def run_one(args):
    aid, gen, pdb_path, heavy, light, cdr_set = args
    row = {"id": aid, "generator": gen, "extraction_status": "FAIL", "error": ""}
    try:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            row["error"] = "missing_pdb"
            return row, None
        key_map, err = map_hl(pdb_path, heavy, light)
        if err:
            row["error"] = err
            return row, None
        text = get_pka_text(aid, gen, pdb_path)
        parsed = parse_summary(text)
        sites = []
        n_excl = 0
        n_term = 0
        for p in parsed:
            if p["resname"] in ("N+", "C-"):
                n_term += 1
                continue
            if p["resname"] == "CYS" and p["pka"] >= 99.0:
                n_excl += 1
                continue
            if p["resname"] not in ACIDS | BASES:
                continue
            k = (p["chain"], p["resseq"])
            if k not in key_map:
                # try without chain rematch — skip
                continue
            hl, seq_i = key_map[k]
            sites.append({"resname": p["resname"], "pka": p["pka"], "hl": hl, "seq_i": seq_i})
        if len(sites) < 3:
            row["error"] = f"too_few_sites:{len(sites)}"
            return row, None
        feat = curve_features(sites, cdr_set, key_map)
        # QC termini curve
        term_sites = [
            {"resname": p["resname"], "pka": p["pka"], "hl": "H", "seq_i": -1}
            for p in parsed
            if p["resname"] in ("N+", "C-") or (p["resname"] in ACIDS | BASES and not (p["resname"] == "CYS" and p["pka"] >= 99))
        ]
        # simpler QC: Q_full at 7.4 including termini from parsed
        q_full = 0.0
        for p in parsed:
            if p["resname"] == "CYS" and p["pka"] >= 99:
                continue
            if p["resname"] in ACIDS | BASES:
                q_full += site_charge(p["resname"], p["pka"], 7.4)
            elif p["resname"] == "C-":
                q_full += site_charge("ASP", p["pka"], 7.4)  # approx as acid
            elif p["resname"] == "N+":
                q_full += site_charge("LYS", p["pka"], 7.4)
        curve = {
            "id": aid,
            "generator": gen,
            "pH": feat["pH_grid"],
            "Q_sidechain": feat["Q_curve"],
            "Q_CDR_sidechain": feat["Q_cdr_curve"],
        }
        out = {
            "id": aid,
            "generator": gen,
            "extraction_status": "SUCCESS",
            "error": "",
            **{k: feat[k] for k in SPEC["canonical_features"]},
            "Q_full_with_termini_pH7_4": float(q_full),
            "n_sidechain_sites": len(sites),
            "n_excluded_cys_sentinel": n_excl,
            "n_termini_sites_seen": n_term,
        }
        return out, curve
    except Exception as e:
        row["error"] = f"{type(e).__name__}:{e}"
        row["traceback"] = traceback.format_exc()[-400:]
        return row, None


def main():
    xw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    seqs = load_sequences()
    cdr = load_cdr_map()
    jobs = []
    for _, r in xw.iterrows():
        aid = r["id"]
        for gen in GENS:
            jobs.append((aid, gen, r[GEN_PATH[gen]], seqs.loc[aid, "heavy"], seqs.loc[aid, "light"], cdr[aid]))
    print("titration jobs", len(jobs), flush=True)
    rows = {g: [] for g in GENS}
    curves = {g: [] for g in GENS}
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            row, curve = fut.result()
            rows[row["generator"]].append(row)
            if curve:
                curves[row["generator"]].append(curve)
            done += 1
            if done % 100 == 0:
                print(f"done {done}/{len(jobs)}", flush=True)
    for gen in GENS:
        df = pd.DataFrame(rows[gen]).sort_values("id")
        df.to_parquet(FAMILY / f"features_{gen}.parquet", index=False)
        print(gen, (df.extraction_status == "SUCCESS").mean(), flush=True)
        if curves[gen]:
            # long format curve
            parts = []
            for c in curves[gen]:
                parts.append(
                    pd.DataFrame(
                        {
                            "id": c["id"],
                            "generator": gen,
                            "pH": c["pH"],
                            "Q_sidechain": c["Q_sidechain"],
                            "Q_CDR_sidechain": c["Q_CDR_sidechain"],
                        }
                    )
                )
            pd.concat(parts, ignore_index=True).to_parquet(FAMILY / f"titration_curves_{gen}.parquet", index=False)
    pd.concat([pd.read_parquet(FAMILY / f"features_{g}.parquet") for g in GENS]).to_csv(
        FAMILY / "FEATURE_MANIFEST.csv", index=False
    )
    pd.concat([pd.read_parquet(FAMILY / f"features_{g}.parquet") for g in GENS]).to_csv(
        FAMILY / "extraction_qc.csv", index=False
    )
    print("TITRATION done")


if __name__ == "__main__":
    main()
