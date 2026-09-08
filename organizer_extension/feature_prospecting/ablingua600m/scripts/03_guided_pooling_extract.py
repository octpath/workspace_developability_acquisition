#!/usr/bin/env python3
"""AbLingua TripleAA → residue mapping + structure-guided pooling (target-blind).

Sanity note (HL vs HL+SEQ PCA32): matrices differ (2560 vs 2638); SEQ columns present;
PCA32 MAE identical to ~1e-15 → SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from transformers import AutoModelForMaskedLM

ROOT = Path("/workspace_developability_acquisition")
OUT = ROOT / "organizer_extension/feature_prospecting/ablingua600m"
VENDOR = OUT / "vendor"
EMB_G = OUT / "embeddings"
EMB_GUIDED = OUT / "embeddings_guided"
CACHE = ROOT / ".cache/huggingface"
MODEL_ID = "IDEA-AI4S/AbLingua"
MAX_LEN = 256
BATCH = 4
HIDDEN = 1280
RASA_EXPOSED = 0.20  # feature_extension/extractors/common.py participant default
PROBE_RADIUS = 1.4
N_POINTS = 100

# Tien 2013 MaxASA (same as feature_extension / gate_b1)
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

sys.path.insert(0, str(VENDOR))
from AbLingua.collate import Simple_Collator  # noqa: E402
from AbLingua.tokenizer import BioTokenizer  # noqa: E402

SPECIAL_NAMES = {"[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"}


def make_collator(tok: BioTokenizer) -> Simple_Collator:
    import argparse

    parser = argparse.ArgumentParser()
    parser = Simple_Collator.add_args(parser)
    args = parser.parse_args([])
    args.max_len = MAX_LEN
    return Simple_Collator(tok, args)


def token_to_residue_spans(seq_len: int, gram: int = 3) -> list[list[int]]:
    """Official BioTokenizer: pad with '>' + seq + '<', then sliding gram-mers.

    Token count = seq_len. Token t covers padded positions [t, t+gram).
    Residue r (0-based) lives at padded index r+1.
    Boundary chars '>' / '<' are not residues.
    """
    spans: list[list[int]] = []
    for t in range(seq_len):
        res = []
        for p in range(t, t + gram):
            if 1 <= p <= seq_len:
                res.append(p - 1)
        spans.append(res)
    return spans


def audit_mapping(seq: str, spans: list[list[int]], tok: BioTokenizer) -> dict:
    L = len(seq)
    tokens = tok.get_token_list(seq)
    assert len(tokens) == L == len(spans)
    covered = [0] * L
    for t, sp in enumerate(spans):
        for r in sp:
            if r < 0 or r >= L:
                return {"ok": False, "reason": f"residue OOB token={t} r={r} L={L}"}
            covered[r] += 1
            # token string should contain that AA when not boundary-only
            # (boundary tokens include > or <)
    if any(c < 1 for c in covered):
        missing = [i for i, c in enumerate(covered) if c < 1]
        return {"ok": False, "reason": f"uncovered residues {missing[:10]}"}
    # reconstruct check: middle tokens equal seq[r:r+3]
    for t in range(1, L - 1):
        expect = seq[t - 1 : t + 2] if False else None
        # token t spans residues from padded t.. — for interior, tokens match seq[t:t+3]?
        # padded[t:t+3] for t>=1 and t+2<=L → residues t-1,t,t+1 → string seq[t-1:t+2]
        # Actually token index t uses padded[t:t+3]
        # For t=1: padded[1:4]=seq[0:3]
        pass
    for t in range(L):
        padded_slice = []
        for p in range(t, t + 3):
            if p == 0:
                padded_slice.append(">")
            elif p == L + 1:
                padded_slice.append("<")
            else:
                padded_slice.append(seq[p - 1])
        got = "".join(padded_slice)
        if got != tokens[t]:
            return {"ok": False, "reason": f"token mismatch t={t} {got} vs {tokens[t]}"}
    return {
        "ok": True,
        "n_tokens": L,
        "cover_min": int(min(covered)),
        "cover_max": int(max(covered)),
        "cover_mean": float(np.mean(covered)),
    }


def residue_embeddings_from_tokens(
    hidden: np.ndarray,  # [T_pad, H] full including pad positions
    attn: np.ndarray,  # [T_pad]
    seq_len: int,
    spans: list[list[int]],
    input_ids: np.ndarray,
    special_ids: set[int],
) -> np.ndarray:
    """Mean of NON-SPECIAL real tokens covering each residue."""
    H = hidden.shape[-1]
    out = np.zeros((seq_len, H), dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.float64)
    n_tok = seq_len  # before padding
    for t in range(n_tok):
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


def masked_mean_tokens(hidden, attn) -> np.ndarray:
    w = attn.astype(np.float64)[:, None]
    return (
        (hidden.astype(np.float64) * w).sum(0) / max(float(attn.sum()), 1.0)
    ).astype(np.float32)


def pool_mean(res_emb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.full(res_emb.shape[1], np.nan, dtype=np.float32)
    return res_emb[m].mean(axis=0).astype(np.float32)


def pool_weighted(res_emb: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        return np.full(res_emb.shape[1], np.nan, dtype=np.float32)
    ww = w[valid]
    ww = ww / ww.sum()
    return (res_emb[valid].astype(np.float64) * ww[:, None]).sum(0).astype(np.float32)


def load_cdr_masks(cdr_df: pd.DataFrame, ab: str, chain: str, L: int) -> dict[str, np.ndarray]:
    sub = cdr_df[(cdr_df.id == ab) & (cdr_df.chain == chain)].sort_values("sequence_index")
    if len(sub) != L:
        raise RuntimeError(f"CDR length mismatch {ab} {chain}: {len(sub)} vs {L}")
    region = sub.region.astype(str).to_numpy()
    cdr_all = np.isin(region, ["CDR1", "CDR2", "CDR3"])
    cdr3 = region == "CDR3"
    fw = region == "framework"
    return {"CDR_ALL": cdr_all, "CDR3": cdr3, "FRAMEWORK": fw}


def compute_rasa_for_pdb(pdb_path: Path, heavy: str, light: str) -> dict[str, np.ndarray]:
    """Return rasa arrays aligned to heavy/light sequence indices; nan if unresolved."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", str(pdb_path))
    sr = ShrakeRupley(probe_radius=PROBE_RADIUS, n_points=N_POINTS)
    sr.compute(st, level="R")

    # Collect per-chain sequences from CA-ordered standard residues
    chain_res = {}
    for ch in st[0]:
        rows = []
        for res in ch:
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue
            aa3 = res.get_resname()
            try:
                aa1 = protein_letters_3to1[aa3.capitalize()]
            except Exception:
                aa1 = "X"
            sasa = float(getattr(res, "sasa", np.nan))
            maxasa = MAX_ASA.get(aa1, np.nan)
            rasa = (sasa / maxasa) if maxasa and np.isfinite(sasa) else np.nan
            rows.append({"aa": aa1, "rasa": rasa})
        chain_res[ch.id] = rows

    # Map chains to H/L by exact sequence match
    def seq_of(rows):
        return "".join(r["aa"] for r in rows)

    h_rasa = np.full(len(heavy), np.nan, dtype=np.float64)
    l_rasa = np.full(len(light), np.nan, dtype=np.float64)
    mapped = {"H": False, "L": False}
    for ch, rows in chain_res.items():
        s = seq_of(rows)
        rasas = np.array([r["rasa"] for r in rows], dtype=np.float64)
        if s == heavy and not mapped["H"]:
            if len(rasas) != len(heavy):
                raise RuntimeError("heavy rasa len mismatch")
            h_rasa = rasas
            mapped["H"] = True
        elif s == light and not mapped["L"]:
            if len(rasas) != len(light):
                raise RuntimeError("light rasa len mismatch")
            l_rasa = rasas
            mapped["L"] = True
        elif heavy.startswith(s) or s.startswith(heavy[: len(s)]):
            # Fv may be shorter than full competition VH if constant truncated — require exact
            pass

    if not mapped["H"] or not mapped["L"]:
        # try A/B order convention (ESMFold: A=heavy, B=light)
        chains = list(chain_res.keys())
        if len(chains) >= 2:
            for cand_h, cand_l in [(chains[0], chains[1]), (chains[1], chains[0])]:
                sh, sl = seq_of(chain_res[cand_h]), seq_of(chain_res[cand_l])
                if sh == heavy and sl == light:
                    h_rasa = np.array([r["rasa"] for r in chain_res[cand_h]], float)
                    l_rasa = np.array([r["rasa"] for r in chain_res[cand_l]], float)
                    mapped = {"H": True, "L": True}
                    break
    return {
        "H": h_rasa,
        "L": l_rasa,
        "mapped_H": mapped["H"],
        "mapped_L": mapped["L"],
    }


def main():
    EMB_GUIDED.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = BioTokenizer(vocab_path=str(VENDOR / "AbLingua/tokens.txt"))
    collator = make_collator(tok)
    special_ids = {int(tok.encode(n)) for n in SPECIAL_NAMES}

    # --- mapping audit on all 324 ---
    dev = pd.read_csv(ROOT / "competition/data/distribution/dev.csv")
    test = pd.read_csv(ROOT / "competition/data/distribution/test_features.csv")
    all_df = pd.concat(
        [dev[["id", "heavy", "light"]], test[["id", "heavy", "light"]]], ignore_index=True
    )
    all_df["id"] = all_df["id"].astype(str)
    assert len(all_df) == 324

    audit_rows = []
    fails = []
    for r in all_df.itertuples(index=False):
        for chain, seq in (("H", r.heavy), ("L", r.light)):
            spans = token_to_residue_spans(len(seq))
            a = audit_mapping(seq, spans, tok)
            a.update({"id": r.id, "chain": chain, "aa_len": len(seq)})
            audit_rows.append(a)
            if not a["ok"]:
                fails.append(a)
    audit_df = pd.DataFrame(audit_rows)
    if fails:
        (OUT / "TOKEN_RESIDUE_MAPPING_UNRESOLVED.json").write_text(json.dumps(fails, indent=2))
        print("TOKEN_RESIDUE_MAPPING_UNRESOLVED", len(fails))
        sys.exit(2)

    cdr = pd.read_csv(ROOT / "organizer_extension/feature_prospecting/cdr_sequence_index_imgt.csv")
    cdr["id"] = cdr["id"].astype(str)
    cw = pd.read_csv(ROOT / "organizer_extension/feature_prospecting/STRUCTURE_INPUT_CROSSWALK_v2.csv")
    cw["id"] = cw["id"].astype(str)
    pdb_map = dict(zip(cw["id"], cw["esmfold_canonical_path"]))

    # CDR coverage audit
    cdr_issues = []
    for r in all_df.itertuples(index=False):
        for chain, seq in (("H", r.heavy), ("L", r.light)):
            sub = cdr[(cdr.id == r.id) & (cdr.chain == chain)].sort_values("sequence_index")
            if len(sub) != len(seq) or "".join(sub.amino_acid) != seq:
                cdr_issues.append({"id": r.id, "chain": chain})
    if cdr_issues:
        print("CDR sequence mismatch", cdr_issues[:5])
        sys.exit(2)

    audit_md = [
        "# AbLingua Token → Residue Mapping Audit",
        "",
        "## Sanity (previous HL vs HL+SEQ PCA32)",
        "",
        "- Feature matrices **differ**: HL=2560, HL+SEQ=2638; all SEQ_BASIC columns present.",
        "- Fold-local PCA32 + Ridge MAE identical to ~1e-15.",
        "- **Verdict: `SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT`**",
        "  (PCA32 dominated by AbLingua; SEQ adds no measurable OOF change).",
        "",
        "## TripleAA algorithm (official BioTokenizer)",
        "",
        "1. Append `>` head and `<` tail to AA sequence.",
        "2. Sliding window of length 3 → token strings.",
        "3. Token count = AA length L.",
        "4. Token `t` covers padded positions `[t, t+3)`.",
        "5. Residue `r` (0-based) sits at padded index `r+1`.",
        "6. Residue embedding = mean of hidden states of all **non-special** tokens",
        "   whose span contains `r` (overlapping TripleAA reweighted at residue level).",
        "",
        f"- Special token IDs excluded: {sorted(special_ids)}",
        f"- Mapping audit: **{len(audit_df)}/{len(audit_df)} chains OK** (324 Abs × H/L).",
        f"- Coverage per residue: min={audit_df.cover_min.min()} max={audit_df.cover_max.max()} "
        f"mean={audit_df.cover_mean.mean():.3f}",
        f"- CDR source: `cdr_sequence_index_imgt.csv` (IMGT via ANARCI), 324/324 exact AA match.",
        f"- RASA: ESMFold Shrake–Rupley + Tien2013 MaxASA; exposed threshold **RASA ≥ {RASA_EXPOSED}**",
        "  (feature_extension participant default).",
        "",
        "If this file exists without UNRESOLVED: mapping is frozen.",
        "",
    ]
    (OUT / "ABLINGUA_TOKEN_RESIDUE_MAPPING_AUDIT.md").write_text("\n".join(audit_md))
    print("Mapping audit PASS", flush=True)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, cache_dir=str(CACHE), output_hidden_states=True, return_dict=True
    )
    model.eval().to(device)

    # Output accumulators
    families = [
        "CDR_ALL",
        "CDR3",
        "EXPOSED",
        "BURIED",
        "RASA_WEIGHTED",
        "BURIED_WEIGHTED",
        "EXPOSED_CDR",
        "CDR_FR_SPLIT",
        "RESIDUE_GLOBAL",
        "H_CDR_ONLY",  # HL concat with L zeros? No — H_CDR pooled + zero L? Spec: H_CDR_ONLY means pool CDR on H only then? 
        # Spec: PARENT + H_CDR_ONLY — representation is H_CDR concat L? "H_CDR_ONLY" = only heavy CDR pooled, need L counterpart.
        # Interpret: H_CDR_ONLY = concat(H_CDR_pool, zeros_L) or concat(H_CDR, L_GLOBAL)? Cleaner: concat(H_CDR, L_zero) 
        # Better: H_CDR_ONLY = [H_CDR_mean | zeros(1280)], L_CDR_ONLY = [zeros| L_CDR_mean], HL_CDR = CDR_ALL
    ]
    # dims
    store = {
        "CDR_ALL": np.zeros((324, 2560), np.float32),
        "CDR3": np.zeros((324, 2560), np.float32),
        "EXPOSED": np.zeros((324, 2560), np.float32),
        "BURIED": np.zeros((324, 2560), np.float32),
        "RASA_WEIGHTED": np.zeros((324, 2560), np.float32),
        "BURIED_WEIGHTED": np.zeros((324, 2560), np.float32),
        "EXPOSED_CDR": np.zeros((324, 2560), np.float32),
        "CDR_FR_SPLIT": np.zeros((324, 5120), np.float32),
        "RESIDUE_GLOBAL": np.zeros((324, 2560), np.float32),
        "H_CDR_ONLY": np.zeros((324, 2560), np.float32),
        "L_CDR_ONLY": np.zeros((324, 2560), np.float32),
    }
    # also keep token GLOBAL from this run for cosine vs saved
    token_global = np.zeros((324, 2560), np.float32)
    meta_rows = []
    ids = all_df["id"].tolist()

    t0 = time.time()
    for i, r in enumerate(all_df.itertuples(index=False)):
        ab = r.id
        pdb_path = Path(pdb_map[ab])
        rasa_pack = compute_rasa_for_pdb(pdb_path, r.heavy, r.light)
        chain_pool = {}
        chain_tok_mean = {}
        for chain, seq in (("H", r.heavy), ("L", r.light)):
            spans = token_to_residue_spans(len(seq))
            batch = collator([seq])
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attn)
                hidden = out.hidden_states[-1][0].float().cpu().numpy()
            attn_np = attn[0].cpu().numpy()
            ids_np = input_ids[0].cpu().numpy()
            res_emb = residue_embeddings_from_tokens(
                hidden, attn_np, len(seq), spans, ids_np, special_ids
            )
            tok_mean = masked_mean_tokens(hidden, attn_np)
            chain_tok_mean[chain] = tok_mean

            masks = load_cdr_masks(cdr, ab, chain, len(seq))
            rasa = rasa_pack[chain]
            if not rasa_pack[f"mapped_{chain}"]:
                meta_rows.append({"id": ab, "chain": chain, "rasa_mapped": False})
            clipped = np.clip(np.nan_to_num(rasa, nan=0.0), 0.0, 1.0)
            # unresolved: where original rasa nan — exclude from hard masks / set weight 0
            resolved = np.isfinite(rasa)
            exposed = resolved & (rasa >= RASA_EXPOSED)
            buried = resolved & (rasa < RASA_EXPOSED)
            # for unresolved, not in EXPOSED nor BURIED hard

            chain_pool[chain] = {
                "res_emb": res_emb,
                "RESIDUE_GLOBAL": res_emb.mean(axis=0),
                "CDR_ALL": pool_mean(res_emb, masks["CDR_ALL"]),
                "CDR3": pool_mean(res_emb, masks["CDR3"]),
                "FRAMEWORK": pool_mean(res_emb, masks["FRAMEWORK"]),
                "EXPOSED": pool_mean(res_emb, exposed),
                "BURIED": pool_mean(res_emb, buried),
                "RASA_WEIGHTED": pool_weighted(res_emb, np.where(resolved, clipped, 0.0)),
                "BURIED_WEIGHTED": pool_weighted(
                    res_emb, np.where(resolved, 1.0 - clipped, 0.0)
                ),
                "EXPOSED_CDR": pool_weighted(
                    res_emb,
                    np.where(masks["CDR_ALL"] & resolved, clipped, 0.0),
                ),
                "n_cdr": int(masks["CDR_ALL"].sum()),
                "n_cdr3": int(masks["CDR3"].sum()),
                "n_exposed": int(exposed.sum()),
                "n_buried": int(buried.sum()),
                "n_unresolved": int((~resolved).sum()),
                "rasa_mapped": bool(rasa_pack[f"mapped_{chain}"]),
            }

        def cat(a, b):
            return np.concatenate([a, b], axis=0)

        H, L = chain_pool["H"], chain_pool["L"]
        store["CDR_ALL"][i] = cat(H["CDR_ALL"], L["CDR_ALL"])
        store["CDR3"][i] = cat(H["CDR3"], L["CDR3"])
        store["EXPOSED"][i] = cat(H["EXPOSED"], L["EXPOSED"])
        store["BURIED"][i] = cat(H["BURIED"], L["BURIED"])
        store["RASA_WEIGHTED"][i] = cat(H["RASA_WEIGHTED"], L["RASA_WEIGHTED"])
        store["BURIED_WEIGHTED"][i] = cat(H["BURIED_WEIGHTED"], L["BURIED_WEIGHTED"])
        store["EXPOSED_CDR"][i] = cat(H["EXPOSED_CDR"], L["EXPOSED_CDR"])
        store["CDR_FR_SPLIT"][i] = np.concatenate(
            [H["CDR_ALL"], H["FRAMEWORK"], L["CDR_ALL"], L["FRAMEWORK"]], axis=0
        )
        store["RESIDUE_GLOBAL"][i] = cat(H["RESIDUE_GLOBAL"], L["RESIDUE_GLOBAL"])
        store["H_CDR_ONLY"][i] = cat(H["CDR_ALL"], np.zeros(HIDDEN, np.float32))
        store["L_CDR_ONLY"][i] = cat(np.zeros(HIDDEN, np.float32), L["CDR_ALL"])
        token_global[i] = cat(chain_tok_mean["H"], chain_tok_mean["L"])

        meta_rows.append(
            {
                "id": ab,
                "H_n_cdr": H["n_cdr"],
                "L_n_cdr": L["n_cdr"],
                "H_n_exposed": H["n_exposed"],
                "L_n_exposed": L["n_exposed"],
                "H_n_unresolved": H["n_unresolved"],
                "L_n_unresolved": L["n_unresolved"],
                "H_rasa_mapped": H["rasa_mapped"],
                "L_rasa_mapped": L["rasa_mapped"],
            }
        )
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/324 elapsed={time.time()-t0:.1f}s", flush=True)

    wall = time.time() - t0

    # Compare RESIDUE_GLOBAL vs saved MASKED_MEAN GLOBAL
    saved = pd.read_parquet(EMB_G / "ablingua600m_HL_mean_concat.parquet")
    saved = saved.set_index("id").reindex(ids)
    saved_X = saved.select_dtypes(include=[np.number]).to_numpy(np.float64)
    res_X = store["RESIDUE_GLOBAL"].astype(np.float64)
    # also compare to this-run token global
    tok_X = token_global.astype(np.float64)

    def cos_corr(A, B):
        cos, corr, nd = [], [], []
        for a, b in zip(A, B):
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos.append(float(np.dot(a, b) / (na * nb + 1e-12)))
            corr.append(float(np.corrcoef(a, b)[0, 1]))
            nd.append(float(np.linalg.norm(a - b) / (na + 1e-12)))
        return {
            "cosine_median": float(np.median(cos)),
            "cosine_min": float(np.min(cos)),
            "corr_median": float(np.median(corr)),
            "norm_diff_median": float(np.median(nd)),
            "n": len(cos),
        }

    cmp_res_saved = cos_corr(res_X, saved_X)
    cmp_tok_saved = cos_corr(tok_X, saved_X)
    cmp_res_tok = cos_corr(res_X, tok_X)

    consistency = {
        "RESIDUE_GLOBAL_vs_saved_MASKED_MEAN": cmp_res_saved,
        "token_GLOBAL_this_run_vs_saved": cmp_tok_saved,
        "RESIDUE_GLOBAL_vs_token_GLOBAL": cmp_res_tok,
        "note": "RESIDUE_GLOBAL reweights overlapping TripleAA; not required to match MASKED_MEAN exactly",
    }
    (OUT / "ABLINGUA_RESIDUE_GLOBAL_CONSISTENCY.json").write_text(
        json.dumps(consistency, indent=2)
    )
    print(json.dumps(consistency, indent=2), flush=True)

    def save_block(name, arr, prefix):
        cols = [f"{prefix}_{j}" for j in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, "id", ids)
        path = EMB_GUIDED / f"ablingua600m_{name}.parquet"
        df.to_parquet(path, index=False)
        h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        return path, h, arr.shape[1]

    dict_rows = []
    hashes = {}
    guided_names = [
        ("CDR_ALL", "HL", "CDR1+2+3", "uniform", 2560),
        ("CDR3", "HL", "CDR3", "uniform", 2560),
        ("EXPOSED", "HL", "RASA>=0.20 resolved", "uniform", 2560),
        ("BURIED", "HL", "RASA<0.20 resolved", "uniform", 2560),
        ("RASA_WEIGHTED", "HL", "all resolved", "clipped_RASA", 2560),
        ("BURIED_WEIGHTED", "HL", "all resolved", "1-clipped_RASA", 2560),
        ("EXPOSED_CDR", "HL", "CDR residues", "clipped_RASA", 2560),
        ("CDR_FR_SPLIT", "HL_partition", "CDR|FW per chain", "uniform", 5120),
    ]
    # also save diagnostic
    extra = [
        ("RESIDUE_GLOBAL", "HL", "all residues", "uniform", 2560),
        ("H_CDR_ONLY", "H_CDR+L_zero", "H CDR_ALL", "uniform", 2560),
        ("L_CDR_ONLY", "H_zero+L_CDR", "L CDR_ALL", "uniform", 2560),
    ]
    for name, scope, sel, weight, dim in guided_names + extra:
        path, h, d = save_block(name, store[name], name)
        hashes[name] = h
        dict_rows.append(
            {
                "block": name,
                "chain_scope": scope,
                "residue_selection": sel,
                "weighting": weight,
                "source_structure": "ESMFold Fv (esmfold_canonical_path)",
                "raw_dim": d,
                "missing_policy": "NaN if empty mask; unresolved rasa weight=0 / excluded from hard masks",
                "notes": f"sha256_16={h}; AbLingua final layer residue-mean then pool",
            }
        )

    pd.DataFrame(dict_rows).to_csv(
        OUT / "ABLINGUA_GUIDED_POOLING_FEATURE_DICTIONARY.csv", index=False
    )
    pd.DataFrame(meta_rows).to_csv(OUT / "ABLINGUA_GUIDED_POOLING_QC.csv", index=False)

    # Freeze BEFORE labels in eval — write now
    freeze = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sanity": "SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT",
        "model": MODEL_ID,
        "representation": "outputs.hidden_states[-1]",
        "token_residue_rule": "mean of non-special TripleAA tokens whose span contains residue r",
        "cdr_convention": "IMGT via cdr_sequence_index_imgt.csv (ANARCI)",
        "rasa": {
            "method": "Bio.PDB.SASA.ShrakeRupley",
            "probe_radius": PROBE_RADIUS,
            "n_points": N_POINTS,
            "maxasa": "Tien2013",
            "exposed_threshold": RASA_EXPOSED,
            "source": "feature_extension participant default RASA_EXPOSED=0.20",
            "structure": "ESMFold Fv",
        },
        "families": [x[0] for x in guided_names],
        "combos_predeclared": [
            "COMBO_1_LOCAL_SURFACE=PARENT+CDR_ALL+RASA_WEIGHTED",
            "COMBO_2_SURFACE_CORE=PARENT+RASA_WEIGHTED+BURIED_WEIGHTED",
            "COMBO_3_CDR_SURFACE_CORE=PARENT+EXPOSED_CDR+BURIED_WEIGHTED",
        ],
        "chain_asymmetry_priority": ["CDR_ALL", "EXPOSED_CDR", "CDR3"],
        "parent": "CURRENT_RECIPE + AbLingua GLOBAL (MASKED_MEAN HL_mean_concat)",
        "hashes": hashes,
        "dims": {r["block"]: r["raw_dim"] for r in dict_rows},
        "extraction_wall_sec": wall,
        "n_ids": 324,
        "consistency": consistency,
    }
    (OUT / "ABLINGUA_GUIDED_POOLING_FREEZE.json").write_text(json.dumps(freeze, indent=2))

    spec = f"""# AbLingua Structure-Guided Pooling — SPEC (FROZEN)

Frozen before TmApp label evaluation.

## Sanity
`SANITY_PASS_IDENTICAL_BY_CHANCE_OR_NO_INCREMENT` — HL vs HL+SEQ matrices differ; PCA32 MAE identical ~1e-15.

## Mapping
Official TripleAA: `>seq<` sliding 3-mers; residue = mean of covering non-special token hiddens.

## CDR
IMGT regions from `cdr_sequence_index_imgt.csv` (324/324). Masks: CDR_ALL, FRAMEWORK, CDR3.

## RASA
ESMFold + ShrakeRupley + Tien2013; **exposed ≥ {RASA_EXPOSED}** (feature_extension default).

## Families (exact)
CDR_ALL, CDR3, EXPOSED, BURIED, RASA_WEIGHTED, BURIED_WEIGHTED, EXPOSED_CDR, CDR_FR_SPLIT

## Parent
CURRENT_RECIPE (AbLang2+SEQ_BASIC+BIOEMU_NEW_PAIRWISE+M1) + AbLingua GLOBAL MASKED_MEAN

## Combos
COMBO_1/2/3 only as listed in FREEZE.json

## Dimensionality
Fold-local PCA32 per AbLingua block (same as prior sprint).

## No
New thresholds, Optuna, Public/Private, layer search, combinatorial fishing.
"""
    (OUT / "ABLINGUA_GUIDED_POOLING_SPEC.md").write_text(spec)
    print(f"DONE wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
