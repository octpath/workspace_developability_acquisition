#!/usr/bin/env python3
"""Build residue_level annotations + AbLingua/ESM-2 residue assets for the bundle.

Organizer-internal builder. Reuses validated TripleAA→residue mapping and
cdr_sequence_index_imgt.csv. Does not invent a new CDR definition.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parent
CDR_CSV = ROOT / "organizer_extension/feature_prospecting/cdr_sequence_index_imgt.csv"
ESM_CACHE = (
    ROOT
    / "organizer_extension/feature_prospecting/structure_marathon/cache/esm2_residue"
)
ABLINGUA_DIR = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
VENDOR = ABLINGUA_DIR / "vendor"
MODEL_ID = "IDEA-AI4S/AbLingua"
MODEL_REV = "4d1272df61a32805695f8789664e2716bb78bb60"
MAX_LEN = 256
BATCH = 4

IMGT_CDR = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}
IMGT_FR = {"FR1": (1, 26), "FR2": (39, 55), "FR3": (66, 104), "FR4": (118, 128)}


def region_from_imgt(region_raw: str, imgt_number) -> str:
    r = str(region_raw).strip()
    if r in ("CDR1", "CDR2", "CDR3"):
        return r
    if r in ("FR1", "FR2", "FR3", "FR4"):
        return r
    # historical CSV uses "framework"
    try:
        n = int(imgt_number) if pd.notna(imgt_number) and str(imgt_number).strip() != "" else None
    except (TypeError, ValueError):
        n = None
    if n is None:
        return "UNKNOWN"
    for name, (lo, hi) in IMGT_FR.items():
        if lo <= n <= hi:
            return name
    for name, (lo, hi) in IMGT_CDR.items():
        if lo <= n <= hi:
            return name
    return "UNKNOWN"


def imgt_position_str(number, icode) -> str:
    if pd.isna(number) or str(number).strip() == "":
        return "UNKNOWN"
    try:
        n = int(number)
    except (TypeError, ValueError):
        return "UNKNOWN"
    code = "" if pd.isna(icode) else str(icode).strip()
    if code == "" or code == "nan":
        return str(n)
    return f"{n}{code}"


def build_annotations(dev: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    seqs = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    seqs["id"] = seqs["id"].astype(str)
    if seqs["id"].duplicated().any():
        raise RuntimeError("duplicate ids in dev+test")
    cdr = pd.read_csv(CDR_CSV)
    cdr["id"] = cdr["id"].astype(str)
    rows = []
    n_unknown = 0
    abs_with_fallback = set()
    per_chain_ok = {"H": 0, "L": 0}
    per_chain_total = {"H": 0, "L": 0}

    for _, srow in seqs.iterrows():
        ab = str(srow["id"])
        for chain, seq in (("H", str(srow["heavy"])), ("L", str(srow["light"]))):
            per_chain_total[chain] += 1
            L = len(seq)
            sub = cdr[(cdr.id == ab) & (cdr.chain == chain)].sort_values("sequence_index")
            numbering_status = "OK"
            if len(sub) != L:
                numbering_status = "LENGTH_MISMATCH_FALLBACK"
                abs_with_fallback.add(ab)
                for i, aa in enumerate(seq):
                    rows.append(
                        {
                            "id": ab,
                            "chain": chain,
                            "seq_index": i,
                            "aa": aa,
                            "imgt_position": "UNKNOWN",
                            "imgt_number": np.nan,
                            "imgt_insertion": "",
                            "region": "UNKNOWN",
                            "is_cdr": False,
                            "cdr_number": 0,
                            "numbering_status": numbering_status,
                        }
                    )
                    n_unknown += 1
                continue

            reconstructed = "".join(sub["amino_acid"].astype(str).tolist())
            if reconstructed != seq:
                numbering_status = "SEQ_MISMATCH_FALLBACK"
                abs_with_fallback.add(ab)
                for i, aa in enumerate(seq):
                    rows.append(
                        {
                            "id": ab,
                            "chain": chain,
                            "seq_index": i,
                            "aa": aa,
                            "imgt_position": "UNKNOWN",
                            "imgt_number": np.nan,
                            "imgt_insertion": "",
                            "region": "UNKNOWN",
                            "is_cdr": False,
                            "cdr_number": 0,
                            "numbering_status": numbering_status,
                        }
                    )
                    n_unknown += 1
                continue

            per_chain_ok[chain] += 1
            for _, r in sub.iterrows():
                region = region_from_imgt(r["region"], r["imgt_number"])
                pos = imgt_position_str(r["imgt_number"], r.get("imgt_icode", ""))
                if region == "UNKNOWN" or pos == "UNKNOWN":
                    n_unknown += 1
                    abs_with_fallback.add(ab)
                    st = "PARTIAL_UNKNOWN"
                else:
                    st = "OK"
                cdr_number = {"CDR1": 1, "CDR2": 2, "CDR3": 3}.get(region, 0)
                insc = "" if pd.isna(r.get("imgt_icode", "")) else str(r.get("imgt_icode", "")).strip()
                if insc == "nan":
                    insc = ""
                rows.append(
                    {
                        "id": ab,
                        "chain": chain,
                        "seq_index": int(r["sequence_index"]),
                        "aa": str(r["amino_acid"]),
                        "imgt_position": pos,
                        "imgt_number": (
                            int(r["imgt_number"]) if pd.notna(r["imgt_number"]) else np.nan
                        ),
                        "imgt_insertion": insc,
                        "region": region,
                        "is_cdr": bool(cdr_number > 0),
                        "cdr_number": cdr_number,
                        "numbering_status": st,
                    }
                )

    ann = pd.DataFrame(rows)
    # reconstruction QC
    for _, srow in seqs.iterrows():
        ab = str(srow["id"])
        for chain, seq in (("H", str(srow["heavy"])), ("L", str(srow["light"]))):
            got = "".join(
                ann[(ann.id == ab) & (ann.chain == chain)]
                .sort_values("seq_index")["aa"]
                .tolist()
            )
            if got != seq:
                raise RuntimeError(f"reconstruction failed {ab} {chain}")

    qc = {
        "n_antibodies": int(seqs["id"].nunique()),
        "n_annotation_rows": int(len(ann)),
        "numbering_success_H": f"{per_chain_ok['H']}/{per_chain_total['H']}",
        "numbering_success_L": f"{per_chain_ok['L']}/{per_chain_total['L']}",
        "n_unknown_residues": int((ann.region == "UNKNOWN").sum()),
        "n_antibodies_with_any_fallback": int(len(abs_with_fallback)),
        "region_counts": {k: int(v) for k, v in ann.region.value_counts().to_dict().items()},
        "source_cdr_csv": str(CDR_CSV),
        "imgt_scheme": "IMGT via cdr_sequence_index_imgt.csv (validated AbLingua guided-pooling source)",
    }
    return ann, qc


def pack_padded(
    ids: list[str],
    arrays: list[np.ndarray],
    max_len: int,
    hidden: int,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(ids)
    emb = np.zeros((N, max_len, hidden), dtype=np.float16)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, a in enumerate(arrays):
        L = a.shape[0]
        if L > max_len:
            raise RuntimeError(f"seq longer than max_len: {ids[i]} {L}>{max_len}")
        if a.shape[1] != hidden:
            raise RuntimeError(f"hidden mismatch {ids[i]} {a.shape}")
        emb[i, :L] = a.astype(np.float16)
        mask[i, :L] = True
        if not np.isfinite(a.astype(np.float32)).all():
            raise RuntimeError(f"non-finite emb {ids[i]}")
        if np.allclose(a.astype(np.float32), 0):
            raise RuntimeError(f"all-zero emb {ids[i]}")
    return emb, mask


def build_esm2(ids: list[str], seqs: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = seqs.set_index("id")
    arrays = []
    max_len = 0
    for ab in ids:
        z = np.load(ESM_CACHE / f"{ab}.npz")
        h = z["H"].astype(np.float32)
        expect = len(str(seqs.loc[ab, "heavy"]))
        if h.shape[0] != expect:
            raise RuntimeError(f"ESM H len mismatch {ab}: {h.shape[0]} vs {expect}")
        arrays.append(h)
        max_len = max(max_len, h.shape[0])
    hidden = int(arrays[0].shape[1])
    # integrity vs pooled ESM2_H (float32 source, before float16 pack)
    pool = pd.read_parquet(BUNDLE / "data/esm2_heavy.parquet")
    pool = pool.set_index(pool["id"].astype(str))
    diffs = []
    for i, ab in enumerate(ids):
        m = arrays[i].astype(np.float64).mean(0)
        feat = pool.loc[ab].drop(labels=["id"], errors="ignore").to_numpy(dtype=float)
        diffs.append(float(np.max(np.abs(m - feat))))
    emb, mask = pack_padded(ids, arrays, max_len, hidden)
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=object))
    np.save(out_dir / "heavy_embeddings.npy", emb)
    np.save(out_dir / "heavy_mask.npy", mask)
    meta = {
        "model": "facebook/esm2_t33_650M_UR50D",
        "chain": "H",
        "n_ids": len(ids),
        "max_len": max_len,
        "hidden_dim": hidden,
        "dtype_on_disk": "float16",
        "dtype_in_memory": "float32",
        "source_cache": str(ESM_CACHE),
        "pool_vs_esm2_h_max_abs_diff_float32": float(max(diffs)),
        "pool_vs_esm2_h_mean_max_abs_diff_float32": float(np.mean(diffs)),
        "license_status": "REVIEW_MODEL_OUTPUT",
        "code_license_note": "fair-esm / ESM-2; see Meta ESM license",
        "model_weight_license_note": "facebook/esm2_t33_650M_UR50D weights",
        "derived_output_redistribution": "REVIEW_MODEL_OUTPUT — do not mark releasable without organizer approval",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def token_to_residue_spans(seq_len: int, gram: int = 3) -> list[list[int]]:
    """Validated BioTokenizer TripleAA overlap mapping (guided-pooling)."""
    spans: list[list[int]] = []
    for t in range(seq_len):
        res = []
        for p in range(t, t + gram):
            if 1 <= p <= seq_len:
                res.append(p - 1)
        spans.append(res)
    return spans


def residue_embeddings_from_tokens(
    hidden: np.ndarray,
    attn: np.ndarray,
    seq_len: int,
    spans: list[list[int]],
    input_ids: np.ndarray,
    special_ids: set[int],
) -> np.ndarray:
    H = hidden.shape[-1]
    out = np.zeros((seq_len, H), dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.float64)
    for t in range(seq_len):
        if attn[t] < 0.5:
            continue
        tid = int(input_ids[t])
        if tid in special_ids:
            continue
        for r in spans[t]:
            out[r] += hidden[t]
            counts[r] += 1.0
    if np.any(counts < 1):
        raise RuntimeError("residue with zero contributing tokens")
    return (out / counts[:, None]).astype(np.float32)


def extract_ablingua(ids: list[str], seqs: pd.DataFrame, out_dir: Path, device: str) -> dict:
    """Extract residue-level AbLingua embeddings using validated TripleAA mapping."""
    sys.path.insert(0, str(VENDOR))
    from AbLingua.collate import Simple_Collator  # noqa: E402
    from AbLingua.tokenizer import BioTokenizer  # noqa: E402
    from transformers import AutoModelForMaskedLM

    def make_collator(tok: BioTokenizer) -> Simple_Collator:
        import argparse as _argparse

        parser = _argparse.ArgumentParser()
        parser = Simple_Collator.add_args(parser)
        cargs = parser.parse_args([])
        cargs.max_len = MAX_LEN
        cargs.truncation = True
        cargs.truncation_mode = "cut"
        cargs.padding = True
        return Simple_Collator(tok, cargs)

    tok = BioTokenizer(vocab_path=str(VENDOR / "AbLingua/tokens.txt"))
    collator = make_collator(tok)
    special_ids = {
        int(tok.encode(n)) for n in ("[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]")
    }

    print(f"Loading AbLingua {MODEL_ID}@{MODEL_REV} on {device} ...", flush=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        cache_dir=str(ROOT / ".cache/huggingface"),
        output_hidden_states=True,
        return_dict=True,
    )
    model.eval().to(device)
    hidden_dim = int(model.config.hidden_size)

    seqs = seqs.set_index("id")
    h_arrays: list[np.ndarray] = []
    l_arrays: list[np.ndarray] = []
    t0 = time.time()

    def run_batch(batch_ids: list[str], chain: str) -> list[np.ndarray]:
        texts = []
        lengths = []
        for ab in batch_ids:
            seq = str(seqs.loc[ab, "heavy" if chain == "H" else "light"])
            texts.append(seq)
            lengths.append(len(seq))
        batch = collator(texts)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn)
        hs = out.hidden_states[-1].float().detach().cpu().numpy()
        attn_np = attn.detach().cpu().numpy()
        ids_np = input_ids.detach().cpu().numpy()
        outs = []
        for i, L in enumerate(lengths):
            spans = token_to_residue_spans(L, gram=3)
            res = residue_embeddings_from_tokens(
                hs[i], attn_np[i], L, spans, ids_np[i], special_ids
            )
            outs.append(res.astype(np.float32))
        return outs

    for chain, sink in (("H", h_arrays), ("L", l_arrays)):
        for start in range(0, len(ids), BATCH):
            batch_ids = ids[start : start + BATCH]
            sink.extend(run_batch(batch_ids, chain))
            print(f"  AbLingua {chain} {min(start + BATCH, len(ids))}/{len(ids)}", flush=True)

    max_h = max(a.shape[0] for a in h_arrays)
    max_l = max(a.shape[0] for a in l_arrays)
    emb_h, mask_h = pack_padded(ids, h_arrays, max_h, hidden_dim)
    emb_l, mask_l = pack_padded(ids, l_arrays, max_l, hidden_dim)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ids.npy", np.asarray(ids, dtype=object))
    np.save(out_dir / "heavy_embeddings.npy", emb_h)
    np.save(out_dir / "light_embeddings.npy", emb_l)
    np.save(out_dir / "heavy_mask.npy", mask_h)
    np.save(out_dir / "light_mask.npy", mask_l)

    meta = {
        "model_repo": MODEL_ID,
        "hf_revision": MODEL_REV,
        "representation_source": "outputs.hidden_states[-1]",
        "mapping": "TripleAA token_to_residue_spans + residue_embeddings_from_tokens (validated guided-pooling)",
        "n_ids": len(ids),
        "max_len_H": max_h,
        "max_len_L": max_l,
        "hidden_dim": hidden_dim,
        "dtype_on_disk": "float16",
        "dtype_in_memory": "float32",
        "extraction_wall_time_sec": float(time.time() - t0),
        "device": device,
        "license_status": "REVIEW_MODEL_OUTPUT",
        "code_license_note": "AbLingua vendor BioTokenizer + HF AutoModelForMaskedLM",
        "model_weight_license_note": "IDEA-AI4S/AbLingua weights — review redistribution",
        "derived_output_redistribution": "REVIEW_MODEL_OUTPUT — do not mark releasable without organizer approval",
        "local_regeneration": (
            "python scripts/build_residue_assets.py --skip-annotations --skip-esm2 "
            "--device cuda"
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--skip-esm2", action="store_true")
    ap.add_argument("--skip-ablingua", action="store_true")
    args = ap.parse_args()

    residue = BUNDLE / "residue_level"
    residue.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(BUNDLE / "dev.csv")
    test = pd.read_csv(BUNDLE / "test.csv")
    seqs = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]],
        ignore_index=True,
    )
    seqs["id"] = seqs["id"].astype(str)
    ids = seqs["id"].tolist()
    assert len(ids) == 324

    if not args.skip_annotations:
        ann, qc = build_annotations(dev, test)
        ann.to_parquet(residue / "annotations.parquet", index=False)
        (residue / "metadata.json").write_text(
            json.dumps(
                {
                    "n_ids": 324,
                    "annotation_rows": len(ann),
                    "columns": list(ann.columns),
                    "qc": qc,
                },
                indent=2,
            )
            + "\n"
        )
        (residue / "ANNOTATION_QC.md").write_text(
            "# Residue annotation QC\n\n"
            f"- Antibodies: **{qc['n_antibodies']}**\n"
            f"- Annotation rows: **{qc['n_annotation_rows']}**\n"
            f"- Numbering success H: **{qc['numbering_success_H']}**\n"
            f"- Numbering success L: **{qc['numbering_success_L']}**\n"
            f"- UNKNOWN residues: **{qc['n_unknown_residues']}**\n"
            f"- Antibodies with any fallback: **{qc['n_antibodies_with_any_fallback']}**\n"
            f"- Source: `{qc['source_cdr_csv']}`\n"
            f"- Scheme: {qc['imgt_scheme']}\n\n"
            "Region counts:\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in sorted(qc["region_counts"].items()))
            + "\n\nReconstruction of residues sorted by `seq_index` matches original "
            "heavy/light sequences for all 324 antibodies.\n"
        )
        print("annotations OK", qc)

    if not args.skip_esm2:
        meta = build_esm2(ids, seqs, residue / "esm2")
        print("esm2 OK", meta["pool_vs_esm2_h_max_abs_diff_float32"])

    if not args.skip_ablingua:
        device = args.device if torch.cuda.is_available() else "cpu"
        meta = extract_ablingua(ids, seqs, residue / "ablingua600m", device)
        print("ablingua OK", meta["hidden_dim"], meta["extraction_wall_time_sec"])

    # write asset audit stub (filled after)
    print("DONE")


if __name__ == "__main__":
    main()
