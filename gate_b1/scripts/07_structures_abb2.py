#!/usr/bin/env python3
"""Predict paired Fv structures with ABodyBuilder2; cache PDBs."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import CACHE, DATA, REPORTS, ensure_dirs, pair_hash, set_gpu0, write_json  # noqa: E402


def main():
    set_gpu0()
    ensure_dirs()
    out_dir = CACHE / "structures" / "abodybuilder2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "shehata_b1_full.csv")

    from ImmuneBuilder import ABodyBuilder2

    predictor = ABodyBuilder2(numbering_scheme="imgt")
    rows = []
    n_ok = n_fail = 0
    t_all = time.perf_counter()
    for i, r in df.iterrows():
        ph = pair_hash(r["heavy"], r["light"])
        pdb_path = out_dir / f"{ph}.pdb"
        meta_path = out_dir / f"{ph}.json"
        if pdb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            rows.append(meta)
            n_ok += 1
            continue
        t0 = time.perf_counter()
        try:
            ab = predictor.predict({"H": r["heavy"], "L": r["light"]})
            # save pdb
            if hasattr(ab, "save"):
                ab.save(str(pdb_path))
            else:
                # Biopython structure?
                from Bio.PDB import PDBIO

                io = PDBIO()
                io.set_structure(ab)
                io.save(str(pdb_path))
            pdb_bytes = pdb_path.read_bytes()
            conf = None
            for attr in ("error_estimates", "confidence", "mean_confidence", "get_confidence"):
                if hasattr(ab, attr):
                    val = getattr(ab, attr)
                    conf = val() if callable(val) else val
                    break
            meta = {
                "antibody_id": r["antibody_id"],
                "pair_hash": ph,
                "predictor": "ABodyBuilder2",
                "version": "ImmuneBuilder.ABodyBuilder2",
                "pdb_path": str(pdb_path),
                "structure_hash": hashlib.sha256(pdb_bytes).hexdigest()[:32],
                "runtime_s": time.perf_counter() - t0,
                "confidence": str(conf)[:500] if conf is not None else None,
                "success": True,
                "error": None,
            }
            n_ok += 1
        except Exception as e:
            meta = {
                "antibody_id": r["antibody_id"],
                "pair_hash": ph,
                "predictor": "ABodyBuilder2",
                "success": False,
                "error": str(e),
                "tb": traceback.format_exc()[-1000:],
                "runtime_s": time.perf_counter() - t0,
            }
            n_fail += 1
            print("FAIL", r["antibody_id"], e, flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        rows.append(meta)
        if (len(rows) % 25) == 0:
            print(f"PROGRESS {len(rows)}/{len(df)} ok={n_ok} fail={n_fail}", flush=True)

    pd.DataFrame(rows).to_csv(CACHE / "structures" / "abb2_manifest.csv", index=False)
    audit = {
        "n_total": len(df),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_rate": n_ok / len(df),
        "total_runtime_s": time.perf_counter() - t_all,
    }
    write_json(CACHE / "structures" / "abb2_audit.json", audit)
    (REPORTS / "structure_prediction_audit.md").write_text(
        "# Structure prediction audit\n\n## ABodyBuilder2\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in audit.items())
        + "\n"
    )
    print("ABB2_OK", audit)


if __name__ == "__main__":
    main()
