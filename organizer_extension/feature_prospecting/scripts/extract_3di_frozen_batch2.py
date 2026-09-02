#!/usr/bin/env python3
"""Target-blind 3DI-FROZEN_v1: Foldseek structure→3Di → ProstT5 encoder → HL_CONCAT."""
from __future__ import annotations

import json
import subprocess
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

warnings.filterwarnings("ignore")

ROOT = Path("/workspace_developability_acquisition")
FP = ROOT / "organizer_extension/feature_prospecting"
FAMILY = FP / "3DI-FROZEN"
FOLDSEEK = ROOT / "tools/foldseek/bin/foldseek"
GEN_PATH = {
    "esmfold": "esmfold_canonical_path",
    "abodybuilder2": "abodybuilder2_path",
    "boltz2": "boltz2_pdb_path",
}
GENS = ["esmfold", "abodybuilder2", "boltz2"]
CACHE = FP / "_batch2_cache" / "3di"
CACHE.mkdir(parents=True, exist_ok=True)


def load_sequences():
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")[["id", "heavy", "light"]]
    te = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")[["id", "heavy", "light"]]
    return pd.concat([dev, te], ignore_index=True).drop_duplicates("id").set_index("id")


def foldseek_3di(pdb_path: Path, out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_prefix
    subprocess.run(
        [str(FOLDSEEK), "structureto3didescriptor", str(pdb_path), str(out_file), "--chain-name-mode", "1", "--threads", "2"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # lines: name\tAA\t3Di[\t...]
    mapping = {}
    text = Path(str(out_file)).read_text().splitlines()
    for line in text:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, aa, di = parts[0], parts[1], parts[2]
        # name ends with _A / _B or chain
        chain = name.rsplit("_", 1)[-1]
        mapping[chain] = {"aa": aa, "di": di}
    return mapping


def assign_hl(mapping: dict, heavy: str, light: str):
    # match by AA sequence
    out = {}
    for ch, d in mapping.items():
        if d["aa"] == heavy:
            out["H"] = d
        elif d["aa"] == light:
            out["L"] = d
    if "H" in out and "L" in out:
        return out, None
    # fallback length
    items = list(mapping.items())
    if len(items) >= 2:
        items = sorted(items, key=lambda x: -len(x[1]["aa"]))
        # assign closer length
        cand = {}
        used = set()
        for label, seq in [("H", heavy), ("L", light)]:
            best = None
            for ch, d in items:
                if ch in used:
                    continue
                score = abs(len(d["aa"]) - len(seq))
                if best is None or score < best[0]:
                    best = (score, ch, d)
            if best:
                cand[label] = best[2]
                used.add(best[1])
        if "H" in cand and "L" in cand:
            return cand, "length_fallback"
    return None, f"map_fail:{list(mapping)}"


def embed_3di(model, tok, device, di: str) -> np.ndarray:
    di = di.lower()
    s = "<fold2AA> " + " ".join(list(di))
    ids = tok([s], return_tensors="pt", padding=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**ids)
    # skip prefix token at position 0; take len(di) residues
    hid = out.last_hidden_state[0]
    # tokens: [prefix, r1, r2, ..., eos?]
    # Prefer mask-based: mean over non-special excluding first
    attn = ids["attention_mask"][0].bool()
    # drop first (prefix) and last if eos
    idx = torch.where(attn)[0]
    if len(idx) >= 2:
        # exclude first token
        use = idx[1:]
        # if last is eos-like, still include; mean over residue span matching len(di)
        use = use[: len(di)]
        emb = hid[use].float().mean(0).cpu().numpy()
    else:
        emb = hid.float().mean(0).cpu().numpy()
    return emb


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    tok = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()

    xw = pd.read_csv(FP / "STRUCTURE_INPUT_CROSSWALK_v2.csv")
    seqs = load_sequences()
    token_rows = {g: [] for g in GENS}
    emb_rows = {g: [] for g in GENS}
    feat_rows = {g: [] for g in GENS}
    fs_ver = subprocess.run([str(FOLDSEEK), "version"], capture_output=True, text=True).stdout.strip()

    for gi, gen in enumerate(GENS):
        print("===", gen, flush=True)
        for i, r in xw.iterrows():
            aid = r["id"]
            pdb = Path(r[GEN_PATH[gen]])
            row = {"id": aid, "generator": gen, "extraction_status": "FAIL", "error": "", "foldseek_version": fs_ver}
            try:
                if not pdb.exists():
                    row["error"] = "missing_pdb"
                    feat_rows[gen].append(row)
                    continue
                outp = CACHE / f"{aid}_{gen}"
                mapping = foldseek_3di(pdb, outp)
                hl, note = assign_hl(mapping, seqs.loc[aid, "heavy"], seqs.loc[aid, "light"])
                if hl is None:
                    row["error"] = note
                    feat_rows[gen].append(row)
                    continue
                H_di = hl["H"]["di"]
                L_di = hl["L"]["di"]
                H_aa = hl["H"]["aa"]
                L_aa = hl["L"]["aa"]
                token_rows[gen].append(
                    {
                        "id": aid,
                        "generator": gen,
                        "H_len_aa": len(H_aa),
                        "L_len_aa": len(L_aa),
                        "H_len_3di": len(H_di),
                        "L_len_3di": len(L_di),
                        "H_3di": H_di,
                        "L_3di": L_di,
                        "H_aa_match": int(H_aa == seqs.loc[aid, "heavy"]),
                        "L_aa_match": int(L_aa == seqs.loc[aid, "light"]),
                        "map_note": note or "exact",
                    }
                )
                h_emb = embed_3di(model, tok, device, H_di)
                l_emb = embed_3di(model, tok, device, L_di)
                hl_emb = np.concatenate([h_emb, l_emb])
                emb_rows[gen].append(
                    {
                        "id": aid,
                        "generator": gen,
                        "H_emb": h_emb,
                        "L_emb": l_emb,
                        "HL_emb": hl_emb,
                    }
                )
                # store HL_CONCAT dims as columns for parquet features
                feat = {
                    "id": aid,
                    "generator": gen,
                    "extraction_status": "SUCCESS",
                    "error": "",
                    "foldseek_version": fs_ver,
                    "H_len_3di": len(H_di),
                    "L_len_3di": len(L_di),
                }
                for d, v in enumerate(hl_emb):
                    feat[f"hl_{d}"] = float(v)
                feat_rows[gen].append(feat)
            except Exception as e:
                row["error"] = f"{type(e).__name__}:{e}"
                row["traceback"] = traceback.format_exc()[-300:]
                feat_rows[gen].append(row)
            if (len(feat_rows[gen])) % 20 == 0:
                print(f"  {gen} {len(feat_rows[gen])}/{len(xw)}", flush=True)

        pd.DataFrame(token_rows[gen]).to_parquet(FAMILY / f"tokens_{gen}.parquet", index=False)
        # embeddings as separate arrays
        er = emb_rows[gen]
        if er:
            edf = pd.DataFrame(
                {
                    "id": [e["id"] for e in er],
                    "generator": gen,
                    **{f"H_{i}": [float(e["H_emb"][i]) for e in er] for i in range(1024)},
                    **{f"L_{i}": [float(e["L_emb"][i]) for e in er] for i in range(1024)},
                }
            )
            edf.to_parquet(FAMILY / f"embeddings_{gen}.parquet", index=False)
        fdf = pd.DataFrame(feat_rows[gen]).sort_values("id")
        fdf.to_parquet(FAMILY / f"features_{gen}.parquet", index=False)
        print(gen, "success", (fdf.extraction_status == "SUCCESS").sum(), flush=True)

    pd.concat([pd.read_parquet(FAMILY / f"features_{g}.parquet") for g in GENS]).to_csv(
        FAMILY / "FEATURE_MANIFEST.csv", index=False
    )
    pd.concat([pd.read_parquet(FAMILY / f"features_{g}.parquet") for g in GENS]).to_csv(
        FAMILY / "extraction_qc.csv", index=False
    )
    meta = {
        "foldseek_version": fs_ver,
        "prostt5_model": "Rostlab/ProstT5",
        "hidden_size": 1024,
        "primary_representation": "HL_CONCAT",
        "device": str(device),
    }
    (FAMILY / "method_runtime_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("3DI done", meta)


if __name__ == "__main__":
    main()
