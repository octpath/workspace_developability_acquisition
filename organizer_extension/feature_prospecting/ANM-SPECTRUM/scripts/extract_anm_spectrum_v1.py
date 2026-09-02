#!/usr/bin/env python3
"""Target-blind ANM-SPECTRUM_v1 extraction for 3 generators."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import prody as pr
from Bio.PDB import PDBParser

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "ANM-SPECTRUM"
CROSSWALK = FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv"
SPEC = json.loads((FAMILY / "FEATURE_SPEC.json").read_text())
CANON = SPEC["canonical_features"]
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}
CUTOFF = 15.0
GAMMA = 1.0
N_MODES = 20
CLASH_CA = 2.0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def structure_confidence(aid: str, gen: str, row: pd.Series, pdb_path: Path) -> float | None:
    if gen == "boltz2":
        v = row.get("boltz2_confidence_score")
        return float(v) if pd.notna(v) else None
    # ESMFold / ABB2: mean CA B-factor as available confidence proxy
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", str(pdb_path))
        bfs = []
        for atom in structure.get_atoms():
            if atom.get_name() == "CA":
                bfs.append(float(atom.get_bfactor()))
        return float(np.mean(bfs)) if bfs else None
    except Exception:
        return None


def severe_clash_ca(coords: np.ndarray, chain_ids: list[str], resseqs: list[int]) -> int:
    n = len(coords)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if chain_ids[i] == chain_ids[j] and abs(resseqs[i] - resseqs[j]) <= 1:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if d < CLASH_CA:
                count += 1
    return count


def ca_table(pdb_path: Path):
    atoms = pr.parsePDB(str(pdb_path))
    ca = atoms.select("calpha")
    if ca is None or len(ca) == 0:
        raise RuntimeError("no_calpha")
    coords = ca.getCoords().astype(float)
    ch = list(ca.getChids())
    res = [int(x) for x in ca.getResnums()]
    nan = int(np.isnan(coords).any(axis=1).sum())
    return ca, coords, ch, res, nan


def anm_features(ca) -> tuple[dict, dict]:
    anm = pr.ANM("anm")
    anm.buildHessian(ca, cutoff=CUTOFF, gamma=GAMMA)
    hess = anm.getHessian()
    # edge count from Kirchhoff-like: off-diagonal blocks nonzero for pairs
    # ProDy Kirchhoff for ANM: use getKirchhoff if available
    kir = anm.getKirchhoff()
    n = kir.shape[0]
    # undirected edges: upper triangle nonzero
    n_edges = int(np.count_nonzero(np.triu(kir, 1)))
    degrees = np.array([np.count_nonzero(kir[i]) - (1 if kir[i, i] != 0 else 0) for i in range(n)], dtype=float)
    # degree from off-diagonals
    degrees = np.count_nonzero(kir != 0, axis=1).astype(float)
    # diagonal is always set; subtract 1
    degrees = degrees - 1.0
    mean_degree = float(degrees.mean()) if n else np.nan
    max_edges = n * (n - 1) / 2.0
    density = float(n_edges / max_edges) if max_edges > 0 else np.nan

    anm.calcModes(n_modes=N_MODES)
    eigs = np.asarray(anm.getEigvals(), dtype=float)
    if len(eigs) < N_MODES:
        raise RuntimeError(f"insufficient_modes:{len(eigs)}")
    # ProDy calcModes(n_modes=20) returns 20 non-zero modes by default
    zeros_expected = 6
    # detect numerical zeros among computed spectrum: none expected in getEigvals
    detected_zero = int(np.sum(eigs < 1e-8))
    extra_zero = max(0, detected_zero)  # among returned modes

    lam = eigs[:N_MODES]
    soft5 = float(np.sum(1.0 / lam[:5]))
    soft10 = float(np.sum(1.0 / lam[:10]))
    soft20 = float(np.sum(1.0 / lam[:20]))
    k = np.arange(1, N_MODES + 1, dtype=float)
    slope = float(np.polyfit(np.log(k), np.log(lam), 1)[0])
    coll = [float(pr.calcCollectivity(anm[i])) for i in range(N_MODES)]

    feats = {
        "log_lambda_1": float(np.log(lam[0])),
        "log_lambda_2": float(np.log(lam[1])),
        "log_lambda_3": float(np.log(lam[2])),
        "log_lambda_5": float(np.log(lam[4])),
        "log_lambda_10": float(np.log(lam[9])),
        "log_lambda_20": float(np.log(lam[19])),
        "log_gmean_lambda_1_3": float(np.mean(np.log(lam[:3]))),
        "log_gmean_lambda_1_5": float(np.mean(np.log(lam[:5]))),
        "log_gmean_lambda_1_10": float(np.mean(np.log(lam[:10]))),
        "log_gmean_lambda_1_20": float(np.mean(np.log(lam[:20]))),
        "log_softness_1_5": float(np.log(soft5)),
        "log_softness_1_10": float(np.log(soft10)),
        "log_softness_1_20": float(np.log(soft20)),
        "softness_fraction_first5": float(soft5 / soft20),
        "spectral_slope_1_20": slope,
        "lambda2_over_lambda1": float(lam[1] / lam[0]),
        "lambda5_over_lambda1": float(lam[4] / lam[0]),
        "collectivity_mode1": coll[0],
        "collectivity_mean_modes1_5": float(np.mean(coll[:5])),
        "collectivity_mean_modes1_10": float(np.mean(coll[:10])),
    }
    qc = {
        "n_CA": int(n),
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "network_density": density,
        "expected_rigid_body_zero_modes": zeros_expected,
        "detected_zero_modes_in_returned_spectrum": detected_zero,
        "extra_zero_modes": extra_zero,
        "n_nonzero_modes_available": int(len(eigs)),
        "status_extra_zero_modes": "FLAG" if extra_zero > 0 else "OK",
    }
    return feats, qc


def msa_prov(aid: str) -> dict:
    meta_p = FP / "structure_sources/boltz2_fv_standard_v1/msa" / aid / "msa_meta.json"
    if not meta_p.exists():
        return {
            "paired_MSA_depth": "NOT_RECOVERABLE",
            "heavy_unpaired_MSA_depth": "NOT_RECOVERABLE",
            "light_unpaired_MSA_depth": "NOT_RECOVERABLE",
            "unpaired_fallback_used": "NOT_RECOVERABLE",
        }
    meta = json.loads(meta_p.read_text())
    mode = meta.get("pairing_mode", "")
    # CSV line counts approximate sequence rows (includes query)
    if mode == "unpaired_only":
        fallback = "true"
    elif mode:
        fallback = "false"
    else:
        fallback = "NOT_RECOVERABLE"
    return {
        "paired_MSA_depth": "NOT_RECOVERABLE_AS_EXPLICIT_PAIRED_COUNT",
        "heavy_unpaired_MSA_depth": str(meta.get("H_nlines", "NOT_RECOVERABLE")),
        "light_unpaired_MSA_depth": str(meta.get("L_nlines", "NOT_RECOVERABLE")),
        "unpaired_fallback_used": fallback,
        "msa_pairing_mode": mode or "NOT_RECOVERABLE",
        "msa_status": meta.get("status", "NOT_RECOVERABLE"),
    }


def process_one(aid: str, gen: str, path: Path, row: pd.Series) -> dict:
    out = {
        "id": aid,
        "generator": gen,
        "pdb_path": str(path),
        "pdb_sha256": sha256_file(path) if path.exists() else None,
        "extraction_status": "FAIL",
    }
    if not path.exists():
        out["error"] = "missing_pdb"
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ca, coords, ch, res, nan = ca_table(path)
            n_ca = len(ca)
            clash = severe_clash_ca(coords, ch, res)
            feats, qc = anm_features(ca)
            conf = structure_confidence(aid, gen, row, path)
            out.update(feats)
            out.update(qc)
            out.update(
                {
                    "n_residue": n_ca,  # CA nodes ≈ residues in Fv
                    "missing_CA": 0,  # only counted CA present; gaps not inferred
                    "nan_coords": nan,
                    "severe_clash_count": clash,
                    "severe_clash_per_100_residues": float(100.0 * clash / n_ca) if n_ca else np.nan,
                    "structure_confidence": conf,
                    "extraction_status": "SUCCESS",
                }
            )
            if gen == "boltz2":
                out.update(msa_prov(aid))
    except Exception as e:
        out["error"] = f"{type(e).__name__}:{e}"
        out["extraction_status"] = "FAIL"
    return out


def main() -> None:
    FAMILY.mkdir(parents=True, exist_ok=True)
    cw = pd.read_csv(CROSSWALK)
    assert len(cw) == 324
    all_qc = []
    man_rows = []
    pr.confProDy(verbosity="none")
    for gen, col in GEN_PATH.items():
        outp = FAMILY / f"features_{gen}.parquet"
        if outp.exists():
            df_existing = pd.read_parquet(outp)
            if int((df_existing.get("extraction_status") == "SUCCESS").sum()) == 324:
                print(f"skip existing {gen}: 324/324", flush=True)
                all_qc.extend(df_existing.to_dict("records"))
                for _, rec in df_existing.iterrows():
                    man_rows.append(
                        {
                            "id": rec["id"],
                            "generator": gen,
                            "extraction_status": rec["extraction_status"],
                            "pdb_path": rec.get("pdb_path"),
                            "pdb_sha256": rec.get("pdb_sha256"),
                            "n_CA": rec.get("n_CA"),
                            "n_edges": rec.get("n_edges"),
                            "error": rec.get("error"),
                        }
                    )
                continue
        rows = []
        for _, r in cw.iterrows():
            aid = r["id"]
            path = Path(str(r[col]))
            rec = process_one(aid, gen, path, r)
            rows.append(rec)
            all_qc.append(rec)
            man_rows.append(
                {
                    "id": aid,
                    "generator": gen,
                    "extraction_status": rec["extraction_status"],
                    "pdb_path": rec.get("pdb_path"),
                    "pdb_sha256": rec.get("pdb_sha256"),
                    "n_CA": rec.get("n_CA"),
                    "n_edges": rec.get("n_edges"),
                    "error": rec.get("error"),
                }
            )
            if len(rows) % 50 == 0:
                print(gen, len(rows), flush=True)
        df = pd.DataFrame(rows)
        # parquet-safe object columns
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda x: "" if x is None else str(x))
        df.to_parquet(outp, index=False)
        ok = int((df["extraction_status"] == "SUCCESS").sum())
        print(f"{gen}: {ok}/{len(df)} -> {outp}", flush=True)

    pd.DataFrame(all_qc).to_csv(FAMILY / "extraction_qc.csv", index=False)
    pd.DataFrame(man_rows).to_csv(FAMILY / "FEATURE_MANIFEST.csv", index=False)

    # freeze hashes of target-blind artifacts
    paths = [
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
    ]
    # robustness computed next; placeholder note
    freeze = {
        "state": "ANM_SPECTRUM_V1_FEATURE_SPEC_FROZEN",
        "target_scoring_started_after_feature_freeze": False,
        "files": {},
    }
    for p in paths:
        freeze["files"][str(p.relative_to(ROOT))] = {
            "sha256": sha256_file(p),
            "nbytes": p.stat().st_size,
        }
    (FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print("wrote TARGET_BLIND_ARTIFACT_HASHES.json (pre-robustness)")


if __name__ == "__main__":
    main()
