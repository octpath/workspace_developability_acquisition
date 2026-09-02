#!/usr/bin/env python3
"""Target-blind HYDRO-FIELD + ELEC-HYDRO-COPATCH extraction (shared surface)."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
import sys

sys.path.insert(0, str(FP))
from common.hydro_surface import build_surface_and_field, copatch_summaries, hydro_summaries  # noqa: E402
from common.structure_utils import GEN_PATH, load_cdr_map, load_sequences  # noqa: E402

PDB2PQR = ROOT / ".venv_stage4/bin/pdb2pqr30"
APBS_BIN = Path("/usr/bin/apbs")
PH = 6.5
IONIC = 0.15
N_WORKERS = 4
GENS = ["esmfold", "abodybuilder2", "boltz2"]
HYDRO_DIR = FP / "HYDRO-FIELD"
COPATCH_DIR = FP / "ELEC-HYDRO-COPATCH"
CACHE = FP / "_batch2_cache"
CACHE.mkdir(exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_apbs_in(pqr_path: Path, out_prefix: Path) -> Path:
    coords = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            parts = line.split()
            try:
                coords.append([float(parts[-5]), float(parts[-4]), float(parts[-3])])
            except Exception:
                pass
    coords = np.asarray(coords, float)
    cmin = coords.min(0) - 20
    cmax = coords.max(0) + 20
    center = 0.5 * (cmin + cmax)
    length = cmax - cmin
    inp = out_prefix.with_suffix(".in")
    text = f"""read
  mol pqr {pqr_path}
end
elec name pot
  mg-auto
  dime 97 97 97
  cglen {length[0]:.3f} {length[1]:.3f} {length[2]:.3f}
  fglen {max(length[0]-10,20):.3f} {max(length[1]-10,20):.3f} {max(length[2]-10,20):.3f}
  cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
  fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
  mol 1
  lpbe
  bcfl sdh
  pdie 2.0
  sdie 78.54
  srfm smol
  chgm spl2
  sdens 10.0
  srad 1.4
  swin 0.3
  temp 298.15
  ion charge 1 conc {IONIC} radius 2.0
  ion charge -1 conc {IONIC} radius 2.0
  calcenergy no
  calcforce no
  write pot dx {out_prefix}_pot
end
quit
"""
    inp.write_text(text)
    return inp


def load_dx(path: Path):
    lines = path.read_text().splitlines()
    dims = origin = None
    deltas = []
    data = []
    mode = None
    for line in lines:
        if line.startswith("object 1"):
            parts = line.split()
            dims = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
        elif line.startswith("origin"):
            origin = np.array(list(map(float, line.split()[1:4])))
        elif line.startswith("delta"):
            deltas.append(list(map(float, line.split()[1:4])))
        elif "data follows" in line:
            mode = "data"
            continue
        elif mode == "data":
            if (not line.strip()) or line.startswith(("attribute", "object", "end", "component")):
                if data:
                    break
                continue
            try:
                data.extend(map(float, line.split()))
            except ValueError:
                break
    if dims is None or origin is None or len(deltas) < 3:
        raise ValueError("bad dx")
    need = int(math.prod(dims))
    if len(data) < need:
        raise ValueError(f"dx_short:{len(data)}<{need}")
    grid = np.asarray(data[:need], float).reshape(dims, order="C")
    return origin, np.array(deltas), grid


def sample_potential(origin, dmat, grid, points):
    nx, ny, nz = grid.shape
    dx = dmat[0, 0] if abs(dmat[0, 0]) > 1e-9 else np.linalg.norm(dmat[0])
    dy = dmat[1, 1] if abs(dmat[1, 1]) > 1e-9 else np.linalg.norm(dmat[1])
    dz = dmat[2, 2] if abs(dmat[2, 2]) > 1e-9 else np.linalg.norm(dmat[2])
    vals = np.full(len(points), np.nan)
    for i, p in enumerate(points):
        ii = (p[0] - origin[0]) / dx
        jj = (p[1] - origin[1]) / dy
        kk = (p[2] - origin[2]) / dz
        i0, j0, k0 = int(np.floor(ii)), int(np.floor(jj)), int(np.floor(kk))
        if 0 <= i0 < nx - 1 and 0 <= j0 < ny - 1 and 0 <= k0 < nz - 1:
            vals[i] = grid[i0, j0, k0]
    return vals


def run_one(args):
    aid, gen, pdb_path, heavy, light, cdr_rows = args
    out = {"id": aid, "generator": gen, "extraction_status": "FAIL", "error": ""}
    try:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            out["error"] = "missing_pdb"
            return out, None, None, None
        surf, err = build_surface_and_field(pdb_path, heavy, light, cdr_rows)
        if err:
            out["error"] = err
            return out, None, None, None
        hydro = hydro_summaries(surf)
        hydro_row = {
            "id": aid,
            "generator": gen,
            "extraction_status": "SUCCESS",
            "error": "",
            **hydro,
            "copatch_status": "FAIL",
        }
        # APBS best-effort
        try:
            work = CACHE / "apbs" / f"{aid}_{gen}"
            work.mkdir(parents=True, exist_ok=True)
            pqr = work / "struct.pqr"
            if not pqr.exists():
                proc = subprocess.run(
                    [str(PDB2PQR), "--ff=AMBER", f"--with-ph={PH}", "--drop-water", str(pdb_path), str(pqr)],
                    capture_output=True, text=True, timeout=180,
                )
                if proc.returncode != 0 or not pqr.exists():
                    hydro_row["error"] = f"pdb2pqr:{proc.stderr[-200:]}"
                    return hydro_row, hydro, None, None
            prefix = work / "pot"
            dx_candidates = list(work.glob("*pot*.dx"))
            if not dx_candidates:
                inp = write_apbs_in(pqr, prefix)
                proc = subprocess.run([str(APBS_BIN), str(inp)], cwd=str(work), capture_output=True, text=True, timeout=300)
                dx_candidates = list(work.glob("*pot*.dx"))
                if not dx_candidates:
                    hydro_row["error"] = f"apbs:{(proc.stderr or proc.stdout)[-200:]}"
                    return hydro_row, hydro, None, None
            dx = dx_candidates[0]
            origin, dmat, grid = load_dx(dx)
            phi = sample_potential(origin, dmat, grid, surf["points"])
            ok = np.isfinite(phi)
            if ok.mean() < 0.5:
                hydro_row["error"] = "phi_sample_sparse"
                return hydro_row, hydro, None, None
            phi_f = np.where(ok, phi, 0.0)
            cop = copatch_summaries(surf, phi_f)
            hydro_row["copatch_status"] = "SUCCESS"
            hydro_row["phi_finite_frac"] = float(ok.mean())
            cop_row = {
                "id": aid, "generator": gen, "extraction_status": "SUCCESS", "error": "",
                **cop, "phi_finite_frac": float(ok.mean()),
            }
            field = {
                "id": [aid] * len(surf["H"]),
                "generator": [gen] * len(surf["H"]),
                "vertex_i": list(range(len(surf["H"]))),
                "x": surf["points"][:, 0],
                "y": surf["points"][:, 1],
                "z": surf["points"][:, 2],
                "H": surf["H"],
                "phi": phi_f,
                "area": surf["areas"],
                "is_cdr": surf["cdr"].astype(int),
            }
            return hydro_row, hydro, cop_row, field
        except Exception as e:
            hydro_row["error"] = f"copatch:{type(e).__name__}:{e}"
            return hydro_row, hydro, None, None
    except Exception as e:
        out["error"] = f"{type(e).__name__}:{e}"
        out["traceback"] = traceback.format_exc()[-500:]
        return out, None, None, None


def main():
    xw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    seqs = load_sequences()
    cdr = load_cdr_map()
    jobs = []
    for _, r in xw.iterrows():
        aid = r["id"]
        for gen in GENS:
            jobs.append(
                (
                    aid,
                    gen,
                    r[GEN_PATH[gen]],
                    seqs.loc[aid, "heavy"],
                    seqs.loc[aid, "light"],
                    cdr[aid],
                )
            )
    print(f"jobs={len(jobs)}", flush=True)
    hydro_rows = {g: [] for g in GENS}
    cop_rows = {g: [] for g in GENS}
    field_parts = {g: [] for g in GENS}
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            hydro_row, _, cop_row, field = fut.result()
            gen = hydro_row["generator"]
            hydro_rows[gen].append(hydro_row)
            if cop_row is not None:
                cop_rows[gen].append(cop_row)
            else:
                # placeholder FAIL for copatch alignment
                cop_rows[gen].append(
                    {
                        "id": hydro_row["id"],
                        "generator": gen,
                        "extraction_status": "FAIL",
                        "error": hydro_row.get("error", "copatch_fail"),
                    }
                )
            if field is not None and isinstance(field, dict) and "phi" in field:
                field_parts[gen].append(pd.DataFrame(field))
            done += 1
            if done % 50 == 0:
                print(f"done {done}/{len(jobs)}", flush=True)

    hydro_canon = json.loads((HYDRO_DIR / "FEATURE_SPEC.json").read_text())["canonical_features"]
    cop_canon = json.loads((COPATCH_DIR / "FEATURE_SPEC.json").read_text())["canonical_features"]
    for gen in GENS:
        hdf = pd.DataFrame(hydro_rows[gen]).sort_values("id")
        cdf = pd.DataFrame(cop_rows[gen]).sort_values("id")
        hdf.to_parquet(HYDRO_DIR / f"features_{gen}.parquet", index=False)
        cdf.to_parquet(COPATCH_DIR / f"features_{gen}.parquet", index=False)
        hdf.to_csv(HYDRO_DIR / "extraction_qc.csv" if gen == "esmfold" else HYDRO_DIR / f"extraction_qc_{gen}.csv", index=False)
        if field_parts[gen]:
            fdf = pd.concat(field_parts[gen], ignore_index=True)
            # keep compact: store only summary surface field sample? Spec asks surface_field_*.parquet
            fdf.to_parquet(HYDRO_DIR / f"surface_field_{gen}.parquet", index=False)
            fdf[["id", "generator", "vertex_i", "H", "phi", "area", "is_cdr"]].to_parquet(
                COPATCH_DIR / f"surface_copatch_{gen}.parquet", index=False
            )
        # manifests
        man_h = []
        for _, row in hdf.iterrows():
            man_h.append(
                {
                    "id": row["id"],
                    "generator": gen,
                    "status": row["extraction_status"],
                    **{c: row.get(c, np.nan) for c in hydro_canon},
                }
            )
        pd.DataFrame(man_h).to_csv(HYDRO_DIR / "FEATURE_MANIFEST.csv" if gen == GENS[-1] else HYDRO_DIR / f"FEATURE_MANIFEST_{gen}.csv", index=False)
    # combined manifests
    pd.concat([pd.read_parquet(HYDRO_DIR / f"features_{g}.parquet") for g in GENS]).to_csv(
        HYDRO_DIR / "FEATURE_MANIFEST.csv", index=False
    )
    pd.concat([pd.read_parquet(COPATCH_DIR / f"features_{g}.parquet") for g in GENS]).to_csv(
        COPATCH_DIR / "FEATURE_MANIFEST.csv", index=False
    )
    pd.concat([pd.read_parquet(HYDRO_DIR / f"features_{g}.parquet") for g in GENS]).to_csv(
        HYDRO_DIR / "extraction_qc.csv", index=False
    )
    pd.concat([pd.read_parquet(COPATCH_DIR / f"features_{g}.parquet") for g in GENS]).to_csv(
        COPATCH_DIR / "extraction_qc.csv", index=False
    )
    print("HYDRO/COPATCH extraction complete", flush=True)


if __name__ == "__main__":
    main()
