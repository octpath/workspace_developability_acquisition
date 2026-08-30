#!/usr/bin/env python3
"""Native ESMFold multimer via facebookresearch/esm: VH:VL → infer_pdb."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_common import (  # noqa: E402
    B1_DATA,
    CACHE,
    DATA,
    LOGS,
    REPORTS,
    ensure_dirs,
    pair_hash,
    set_gpu0,
    write_json,
)


def parse_chains(pdb_text: str) -> dict:
    chains = {}
    seqs = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        resname = line[17:20].strip()
        resseq = int(line[22:26])
        aa = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }.get(resname, "X")
        chains.setdefault(ch, []).append(resseq)
        seqs.setdefault(ch, [])
        # only append once per resseq
        if not seqs[ch] or chains[ch][-2] != resseq if len(chains[ch]) > 1 else True:
            # careful: we already appended resseq
            pass
    # rebuild sequences properly
    seqs = {}
    seen = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        resname = line[17:20].strip()
        resseq = int(line[22:26])
        key = (ch, resseq)
        if key in seen:
            continue
        seen[key] = True
        aa = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }.get(resname, "X")
        seqs.setdefault(ch, []).append(aa)
    return {ch: "".join(s) for ch, s in seqs.items()}


def com_distance(pdb_text: str, ch_a="A", ch_b="B") -> float | None:
    import numpy as np

    coords = {ch_a: [], ch_b: []}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        if ch not in coords:
            continue
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        coords[ch].append([x, y, z])
    if not coords[ch_a] or not coords[ch_b]:
        # try H/L
        return None
    a = np.array(coords[ch_a]).mean(0)
    b = np.array(coords[ch_b]).mean(0)
    return float(np.linalg.norm(a - b))


def main():
    set_gpu0()
    ensure_dirs()
    out_dir = CACHE / "structures" / "esmfold_native"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of HIC + TmApp sequences from B1 tables
    hic = pd.read_csv(B1_DATA / "hic_full.csv")
    tm = pd.read_csv(B1_DATA / "tmapp_full.csv")
    full = pd.read_csv(B1_DATA / "shehata_b1_full.csv")
    ids = sorted(set(hic["antibody_id"]) | set(tm["antibody_id"]))
    df = full[full["antibody_id"].isin(ids)].copy()
    df.to_csv(DATA / "hic_tmapp_union.csv", index=False)
    print(f"UNION n={len(df)} hic={len(hic)} tm={len(tm)}", flush=True)

    import esm

    t_load = time.perf_counter()
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    # chunk size for memory
    if hasattr(model, "set_chunk_size"):
        model.set_chunk_size(128)
    load_s = time.perf_counter() - t_load
    free, total = torch.cuda.mem_get_info(0)

    # Record model metadata
    meta_global = {
        "package": "facebookresearch/esm (fair-esm)",
        "esm_file": esm.__file__,
        "esm_version": getattr(esm, "__version__", None),
        "model": "esmfold_v1",
        "input_format": "f'{VH}:{VL}'",
        "api": "model.infer_pdb(sequence)",
        "chunk_size": 128,
        "load_s": load_s,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_mb": round(total / 1024**2, 1),
        "vram_free_after_load_mb": round(free / 1024**2, 1),
        "note": (
            "Official multimer input uses colon-separated chains. "
            "ESMFold supports multimer inputs through colon-separated chains and "
            "chain-aware inference machinery (internal linker/index-offset). "
            "Not an AlphaFold-Multimer equivalent."
        ),
    }
    # Probe defaults from source if possible
    import inspect

    try:
        meta_global["infer_signature"] = str(inspect.signature(model.infer))
        src = inspect.getsource(model.infer)
        meta_global["infer_source_excerpt"] = src[:4000]
    except Exception as e:
        meta_global["infer_source_error"] = str(e)
    write_json(CACHE / "structures" / "esmfold_native_model_meta.json", meta_global)

    # Validate on ~15 antibodies
    val_rows = []
    sample = df.head(15)
    print("VALIDATION on", len(sample), flush=True)
    with torch.no_grad():
        for _, r in sample.iterrows():
            seq = f"{r['heavy']}:{r['light']}"
            t0 = time.perf_counter()
            try:
                pdb = model.infer_pdb(seq)
                chains = parse_chains(pdb)
                chain_ids = sorted(chains.keys())
                # Expect A/B typically
                ok_len = False
                mapped = {}
                if len(chain_ids) >= 2:
                    # assign by length match
                    c0, c1 = chain_ids[0], chain_ids[1]
                    if len(chains[c0]) == len(r["heavy"]) and len(chains[c1]) == len(r["light"]):
                        mapped = {"VH": c0, "VL": c1}
                        ok_len = True
                    elif len(chains[c1]) == len(r["heavy"]) and len(chains[c0]) == len(r["light"]):
                        mapped = {"VH": c1, "VL": c0}
                        ok_len = True
                # no Gly25 as biological residue of wrong length
                linker_artifact = any("G" * 25 in s for s in chains.values())
                com = None
                if mapped:
                    com = com_distance(pdb, mapped["VH"], mapped["VL"])
                    if com is None:
                        # try raw A/B
                        com = com_distance(pdb, chain_ids[0], chain_ids[1])
                val_rows.append(
                    {
                        "antibody_id": r["antibody_id"],
                        "success": True,
                        "n_chains": len(chain_ids),
                        "chain_ids": chain_ids,
                        "chain_lens": {k: len(v) for k, v in chains.items()},
                        "vh_len": len(r["heavy"]),
                        "vl_len": len(r["light"]),
                        "length_match": ok_len,
                        "mapping": mapped,
                        "linker_as_bio_residue": linker_artifact,
                        "com_distance_A": com,
                        "runtime_s": time.perf_counter() - t0,
                        "pdb_chars": len(pdb),
                    }
                )
                print(
                    "VAL",
                    r["antibody_id"],
                    "chains",
                    chain_ids,
                    "ok_len",
                    ok_len,
                    "com",
                    com,
                    flush=True,
                )
            except Exception as e:
                val_rows.append(
                    {
                        "antibody_id": r["antibody_id"],
                        "success": False,
                        "error": str(e),
                        "tb": traceback.format_exc()[-800:],
                    }
                )
                print("VAL_FAIL", r["antibody_id"], e, flush=True)

    write_json(CACHE / "structures" / "esmfold_native_validation.json", val_rows)
    n_ok_val = sum(1 for v in val_rows if v.get("success") and v.get("length_match"))
    print(f"VALIDATION ok_len={n_ok_val}/{len(val_rows)}", flush=True)

    if n_ok_val < max(5, len(val_rows) // 2):
        print("VALIDATION_WEAK — still proceeding with full run but flagging", flush=True)

    # Full union
    rows = []
    n_ok = n_fail = 0
    t_all = time.perf_counter()
    with torch.no_grad():
        for i, (_, r) in enumerate(df.iterrows()):
            ph = pair_hash(r["heavy"], r["light"])
            pdb_path = out_dir / f"{ph}.pdb"
            meta_path = out_dir / f"{ph}.json"
            if pdb_path.exists() and meta_path.exists():
                rows.append(json.loads(meta_path.read_text()))
                n_ok += 1
                continue
            t0 = time.perf_counter()
            try:
                seq = f"{r['heavy']}:{r['light']}"
                # Single forward pass: infer → pdb string + pLDDT (avoid double infer_pdb+infer)
                out = model.infer(seq)
                pdb = model.output_to_pdb(out)[0]
                pdb_path.write_text(pdb)
                chains = parse_chains(pdb)
                chain_ids = sorted(chains.keys())
                mean_plddt = None
                if "plddt" in out:
                    plddt = out["plddt"]
                    if hasattr(plddt, "detach"):
                        plddt = plddt[0].detach().cpu().numpy()
                    mean_plddt = float(plddt.mean())
                meta = {
                    "antibody_id": r["antibody_id"],
                    "pair_hash": ph,
                    "predictor": "ESMFold_native",
                    "model": "esmfold_v1",
                    "input": "VH:VL",
                    "pdb_path": str(pdb_path),
                    "structure_hash": hashlib.sha256(pdb.encode()).hexdigest()[:32],
                    "runtime_s": time.perf_counter() - t0,
                    "success": True,
                    "n_chains": len(chain_ids),
                    "chain_ids": chain_ids,
                    "chain_lens": {k: len(v) for k, v in chains.items()},
                    "vh_len": len(r["heavy"]),
                    "vl_len": len(r["light"]),
                    "error": None,
                    "mean_plddt": mean_plddt,
                }
                n_ok += 1
            except Exception as e:
                meta = {
                    "antibody_id": r["antibody_id"],
                    "pair_hash": ph,
                    "predictor": "ESMFold_native",
                    "success": False,
                    "error": str(e),
                    "tb": traceback.format_exc()[-1200:],
                    "runtime_s": time.perf_counter() - t0,
                }
                n_fail += 1
                print("FAIL", r["antibody_id"], e, flush=True)
            meta_path.write_text(json.dumps(meta, indent=2))
            rows.append(meta)
            if (i + 1) % 10 == 0:
                print(f"PROGRESS {i+1}/{len(df)} ok={n_ok} fail={n_fail}", flush=True)
            if (i + 1) % 50 == 0:
                pd.DataFrame(rows).to_csv(CACHE / "structures" / "esmfold_native_manifest.csv", index=False)

    pd.DataFrame(rows).to_csv(CACHE / "structures" / "esmfold_native_manifest.csv", index=False)
    audit = {
        "n_total": len(df),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "success_rate": n_ok / max(len(df), 1),
        "total_runtime_s": time.perf_counter() - t_all,
        "validation_ok_len": n_ok_val,
        "model_meta": {k: meta_global[k] for k in meta_global if k != "infer_source_excerpt"},
    }
    write_json(CACHE / "structures" / "esmfold_native_audit.json", audit)

    md = [
        "# ESMFold native multimer audit",
        "",
        f"- Package: facebookresearch/esm (`{meta_global.get('esm_version')}`)",
        f"- API: `sequence = f'{{VH}}:{{VL}}'` → `model.infer_pdb(sequence)`",
        f"- Chunk size: 128",
        f"- Validation length-matched chains: {n_ok_val}/{len(val_rows)}",
        f"- Full union success: {n_ok}/{len(df)}",
        f"- Runtime: {audit['total_runtime_s']:.1f}s",
        "",
        meta_global["note"],
        "",
        "Internal linker/residue_index_offset: see `infer_source_excerpt` in model meta JSON.",
        "",
    ]
    (REPORTS / "esmfold_native_multimer_audit.md").write_text("\n".join(md) + "\n")
    print("ESMFOLD_NATIVE_OK", audit)


if __name__ == "__main__":
    main()
