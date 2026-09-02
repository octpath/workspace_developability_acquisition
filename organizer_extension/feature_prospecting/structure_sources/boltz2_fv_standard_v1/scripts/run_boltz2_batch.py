#!/usr/bin/env python3
"""Resume-capable Boltz-2 runner for BOLTZ2_FV_STANDARD_v1 (N=324).

Does NOT use TmApp/HIC labels. Existing successful outputs are never overwritten
unless --force-failed-only retries a recorded FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace_developability_acquisition")
BASE = ROOT / "organizer_extension/feature_prospecting/structure_sources/boltz2_fv_standard_v1"
INP = BASE / "inputs"
NATIVE = BASE / "predictions_native"
MMCIF = BASE / "structures_mmcif"
PDB = BASE / "structures_pdb"
CONF = BASE / "confidence"
LOGS = BASE / "logs"
PROCESSED = BASE / "processed"
STATUS = BASE / "run_status.jsonl"
MANIFEST = BASE / "BOLTZ2_STRUCTURE_MANIFEST.csv"

BOLTZ = ROOT / ".venv_boltz/bin/boltz"
PYTHON = ROOT / ".venv_boltz/bin/python"
CACHE = Path(os.environ.get("BOLTZ_CACHE", str(ROOT / ".boltz_cache")))

# Frozen protocol (also in BOLTZ2_STRUCTURE_SPEC.json)
RECYCLING = 3
SAMPLING = 200
DIFFUSION = 1
STEP_SCALE = 1.5
MSA_URL = "https://api.colabfold.com"
MSA_PAIR = "greedy"
MAX_TRANSIENT_RETRY = 2


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_done() -> dict[str, dict]:
    done = {}
    if STATUS.exists():
        for line in STATUS.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[rec["id"]] = rec
    return done


def append_status(rec: dict) -> None:
    with STATUS.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def success_artifacts(aid: str) -> bool:
    cif = MMCIF / f"{aid}.cif"
    pdb = PDB / f"{aid}.pdb"
    conf = CONF / f"{aid}.json"
    return cif.exists() and pdb.exists() and conf.exists() and cif.stat().st_size > 0


def convert_mmcif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    # Keep chain IDs as produced (expect H/L)
    io = PDBIO()
    io.set_structure(structure)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(pdb_path))


def harvest(aid: str, pred_root: Path) -> dict:
    """Copy native outputs into canonical locations."""
    pred_dir = pred_root / "predictions" / aid
    if not pred_dir.exists():
        # boltz may nest under out_dir/boltz_results_*/
        cands = list(pred_root.rglob(f"{aid}_model_0.cif")) + list(pred_root.rglob(f"{aid}_model_0.mmcif"))
        if not cands:
            raise FileNotFoundError(f"no model cif for {aid} under {pred_root}")
        pred_dir = cands[0].parent

    cif_src = None
    for name in (f"{aid}_model_0.cif", f"{aid}_model_0.mmcif"):
        p = pred_dir / name
        if p.exists():
            cif_src = p
            break
    if cif_src is None:
        cifs = sorted(pred_dir.glob("*_model_0.cif")) + sorted(pred_dir.glob("*.cif"))
        if not cifs:
            raise FileNotFoundError(f"no cif in {pred_dir}")
        cif_src = cifs[0]

    conf_src = pred_dir / f"confidence_{aid}_model_0.json"
    if not conf_src.exists():
        confs = sorted(pred_dir.glob("confidence_*.json"))
        if not confs:
            raise FileNotFoundError(f"no confidence json in {pred_dir}")
        conf_src = confs[0]

    native_dir = NATIVE / aid
    native_dir.mkdir(parents=True, exist_ok=True)
    for p in pred_dir.iterdir():
        dest = native_dir / p.name
        if not dest.exists():
            if p.is_file():
                shutil.copy2(p, dest)
            elif p.is_dir():
                shutil.copytree(p, dest, dirs_exist_ok=True)

    MMCIF.mkdir(parents=True, exist_ok=True)
    CONF.mkdir(parents=True, exist_ok=True)
    cif_dst = MMCIF / f"{aid}.cif"
    conf_dst = CONF / f"{aid}.json"
    if not cif_dst.exists():
        shutil.copy2(cif_src, cif_dst)
    if not conf_dst.exists():
        shutil.copy2(conf_src, conf_dst)

    pdb_dst = PDB / f"{aid}.pdb"
    if not pdb_dst.exists():
        convert_mmcif_to_pdb(cif_dst, pdb_dst)

    # copy processed if present
    proc = pred_root / "processed"
    if proc.exists():
        dest = PROCESSED / aid
        if not dest.exists():
            shutil.copytree(proc, dest, dirs_exist_ok=True)

    conf = json.loads(conf_dst.read_text())
    return {
        "cif_sha256": sha256_file(cif_dst),
        "pdb_sha256": sha256_file(pdb_dst),
        "confidence_sha256": sha256_file(conf_dst),
        "confidence_score": conf.get("confidence_score"),
        "ptm": conf.get("ptm"),
        "iptm": conf.get("iptm"),
        "protein_iptm": conf.get("protein_iptm"),
        "complex_plddt": conf.get("complex_plddt"),
        "complex_iplddt": conf.get("complex_iplddt"),
    }


def run_one(aid: str, attempt: int) -> dict:
    yaml = INP / f"{aid}.yaml"
    assert yaml.exists(), yaml
    out_dir = LOGS / f"run_{aid}_a{attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{aid}_a{attempt}.log"
    # Prefer precomputed MSA CSV paths embedded in YAML (MSA-enabled protocol).
    # Do not call --use_msa_server when YAML already references msa/*.csv (resume-safe).
    yaml_text = yaml.read_text()
    has_precomputed_msa = "msa:" in yaml_text and "/msa/" in yaml_text
    cmd = [
        str(BOLTZ),
        "predict",
        str(yaml),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(CACHE),
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--recycling_steps",
        str(RECYCLING),
        "--sampling_steps",
        str(SAMPLING),
        "--diffusion_samples",
        str(DIFFUSION),
        "--step_scale",
        str(STEP_SCALE),
        "--output_format",
        "mmcif",
        "--write_full_pae",
        "--seed",
        "42",
        "--num_workers",
        "0",
        "--preprocessing-threads",
        "1",
    ]
    if has_precomputed_msa:
        # MSA already generated via ColabFold API (see precompute_msa.py)
        pass
    else:
        cmd.extend(
            [
                "--use_msa_server",
                "--msa_server_url",
                MSA_URL,
                "--msa_pairing_strategy",
                MSA_PAIR,
            ]
        )
    # potentials OFF by omitting --use_potentials
    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["BOLTZ_CACHE"] = str(CACHE)
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, text=True)
    runtime = time.time() - t0
    if proc.returncode != 0:
        return {
            "id": aid,
            "status": "FAIL_RUNTIME",
            "attempt": attempt,
            "runtime_s": runtime,
            "returncode": proc.returncode,
            "log": str(log_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    try:
        meta = harvest(aid, out_dir)
        return {
            "id": aid,
            "status": "SUCCESS",
            "attempt": attempt,
            "runtime_s": runtime,
            "log": str(log_path),
            "ts": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
    except Exception as e:
        return {
            "id": aid,
            "status": "FAIL_HARVEST",
            "attempt": attempt,
            "runtime_s": runtime,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
            "log": str(log_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        }


def is_transient(rec: dict) -> bool:
    if rec.get("status") == "FAIL_RUNTIME":
        return True
    err = (rec.get("error") or "") + Path(rec.get("log", "")).read_text()[-4000:] if rec.get("log") and Path(rec["log"]).exists() else ""
    keys = ["timeout", "Connection", "Temporary", "503", "502", "429", "CUDA out of memory", "OOM", "MSA"]
    return any(k.lower() in err.lower() for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    for d in (NATIVE, MMCIF, PDB, CONF, LOGS, PROCESSED, CACHE):
        d.mkdir(parents=True, exist_ok=True)

    ids = sorted(p.stem for p in INP.glob("*.yaml"))
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.limit:
        ids = ids[: args.limit]

    done = load_done()
    print(f"queue={len(ids)} already_status={len(done)} artifacts_ok={sum(1 for i in ids if success_artifacts(i))}")

    for aid in ids:
        if success_artifacts(aid):
            prev = done.get(aid)
            if not prev or prev.get("status") != "SUCCESS":
                # backfill status from artifacts
                conf = json.loads((CONF / f"{aid}.json").read_text())
                append_status(
                    {
                        "id": aid,
                        "status": "SUCCESS",
                        "attempt": 0,
                        "runtime_s": None,
                        "note": "existing_artifacts",
                        "confidence_score": conf.get("confidence_score"),
                        "ptm": conf.get("ptm"),
                        "iptm": conf.get("iptm"),
                        "complex_plddt": conf.get("complex_plddt"),
                        "cif_sha256": sha256_file(MMCIF / f"{aid}.cif"),
                        "pdb_sha256": sha256_file(PDB / f"{aid}.pdb"),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                )
            continue

        prev = done.get(aid)
        if prev and prev.get("status") == "SUCCESS":
            continue
        if prev and prev.get("status", "").startswith("FAIL") and not args.retry_failed:
            # allow automatic retry up to MAX if transient and attempts < max
            attempts = int(prev.get("attempt") or 0)
            if attempts >= MAX_TRANSIENT_RETRY or not is_transient(prev):
                continue

        start_attempt = int((prev or {}).get("attempt") or 0) + 1
        attempt = start_attempt
        while attempt <= MAX_TRANSIENT_RETRY + 1:
            print(f"[{datetime.now().isoformat()}] predicting {aid} attempt={attempt}", flush=True)
            rec = run_one(aid, attempt)
            append_status(rec)
            done[aid] = rec
            print(f"  -> {rec['status']} runtime={rec.get('runtime_s'):.1f}s" if rec.get("runtime_s") else f"  -> {rec['status']}", flush=True)
            if rec["status"] == "SUCCESS":
                break
            if not is_transient(rec) or attempt > MAX_TRANSIENT_RETRY:
                break
            attempt += 1

    n_ok = sum(1 for i in ids if success_artifacts(i))
    print(f"DONE success_artifacts={n_ok}/{len(ids)}")


if __name__ == "__main__":
    main()
