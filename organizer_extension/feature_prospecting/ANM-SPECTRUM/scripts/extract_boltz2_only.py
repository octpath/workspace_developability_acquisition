#!/usr/bin/env python3
"""Boltz2-only ANM extraction + rebuild manifests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import prody as pr

ROOT = Path("/workspace_developability_acquisition")
FAMILY = ROOT / "organizer_extension/feature_prospecting/ANM-SPECTRUM"
SCRIPT = FAMILY / "scripts/extract_anm_spectrum_v1.py"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_mod():
    spec = importlib.util.spec_from_file_location("ext", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = load_mod()
    pr.confProDy(verbosity="none")
    cw = pd.read_csv(mod.CROSSWALK)
    rows = []
    for i, r in cw.iterrows():
        rec = mod.process_one(r["id"], "boltz2", Path(str(r["boltz2_pdb_path"])), r)
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print("boltz2", i + 1, flush=True)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: "" if x is None else str(x))
    outp = FAMILY / "features_boltz2.parquet"
    df.to_parquet(outp, index=False)
    print("boltz2", int((df.extraction_status == "SUCCESS").sum()), "/", len(df), flush=True)

    all_rows = []
    man = []
    for gen in ["esmfold", "abodybuilder2", "boltz2"]:
        d = pd.read_parquet(FAMILY / f"features_{gen}.parquet")
        print(gen, int((d.extraction_status == "SUCCESS").sum()), "/", len(d), flush=True)
        all_rows.extend(d.to_dict("records"))
        for _, rec in d.iterrows():
            man.append(
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
    pd.DataFrame(all_rows).to_csv(FAMILY / "extraction_qc.csv", index=False)
    pd.DataFrame(man).to_csv(FAMILY / "FEATURE_MANIFEST.csv", index=False)
    freeze = {
        "state": "ANM_SPECTRUM_V1_FEATURE_SPEC_FROZEN",
        "target_scoring_started_after_feature_freeze": False,
        "files": {},
    }
    for p in [
        FAMILY / "FEATURE_SPEC.json",
        FAMILY / "FEATURE_MANIFEST.csv",
        FAMILY / "extraction_qc.csv",
        FAMILY / "features_esmfold.parquet",
        FAMILY / "features_abodybuilder2.parquet",
        FAMILY / "features_boltz2.parquet",
    ]:
        freeze["files"][str(p.relative_to(ROOT))] = {"sha256": sha(p), "nbytes": p.stat().st_size}
    (FAMILY / "TARGET_BLIND_ARTIFACT_HASHES.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print("wrote TARGET_BLIND_ARTIFACT_HASHES.json", flush=True)


if __name__ == "__main__":
    main()
