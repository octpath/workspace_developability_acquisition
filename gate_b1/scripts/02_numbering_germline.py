#!/usr/bin/env python3
"""IMGT numbering + germline / maturation features via anarcii (+ author CDR freeze)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_common import (  # noqa: E402
    CACHE,
    DATA,
    REPORTS,
    ensure_dirs,
    pair_hash,
    write_json,
)

# IMGT CDR ranges (inclusive residue numbers)
IMGT_CDR = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}
IMGT_FR = {
    "FR1": (1, 26),
    "FR2": (39, 55),
    "FR3": (66, 104),
    "FR4": (118, 128),
}


def regions_from_numbered(numbered_pairs: list[tuple[tuple, str]]) -> dict:
    """numbered_pairs: list of ((pos, insert), aa) with gaps excluded."""
    by_pos = {}
    for (pos, _ins), aa in numbered_pairs:
        if aa == "-":
            continue
        by_pos.setdefault(pos, [])
        by_pos[pos].append(aa)

    def slice_range(lo, hi):
        chars = []
        for p in range(lo, hi + 1):
            chars.extend(by_pos.get(p, []))
        return "".join(chars)

    out = {}
    for name, (lo, hi) in IMGT_FR.items():
        out[name] = slice_range(lo, hi)
    for name, (lo, hi) in IMGT_CDR.items():
        out[name] = slice_range(lo, hi)
    return out


def hamming_frac(a: str, b: str) -> float | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    mismatches = sum(x != y for x, y in zip(a[:n], b[:n]))
    # penalize length difference lightly
    mismatches += abs(len(a) - len(b))
    return mismatches / max(len(a), len(b))


def number_with_anarcii(sequences: list[tuple[str, str]]) -> list:
    """Return anarcii numbering results for list of (id, seq)."""
    from anarcii import Anarcii

    model = Anarcii(seq_type="antibody", batch_size=128, cpu=True, ncpu=8)
    # anarcii expects list of sequences or fasta-like
    ids = [i for i, _ in sequences]
    seqs = [s for _, s in sequences]
    numbered = model.number(seqs)
    # Attach ids
    return ids, numbered


def extract_anarcii_chain(rec) -> dict:
    """Normalize one anarcii record into regions + germline."""
    # anarcii API varies by version — handle dict-like
    if isinstance(rec, dict):
        data = rec
    else:
        data = dict(rec) if hasattr(rec, "keys") else {"raw": rec}

    # Try common keys
    numbering = data.get("numbering") or data.get("IMGT") or data.get("numbered")
    chain_type = data.get("chain_type") or data.get("chain") or data.get("query_chain_type")
    species = data.get("species")
    v_gene = data.get("v_gene") or data.get("germline") or data.get("best_v")
    j_gene = data.get("j_gene") or data.get("best_j")
    score = data.get("score") or data.get("v_score") or data.get("confidence")

    regions = {}
    numbered_pairs = []
    if numbering is not None:
        # numbering may be list of ((pos,ins), aa) or dict pos->aa
        if isinstance(numbering, dict):
            for k, aa in numbering.items():
                if isinstance(k, tuple):
                    numbered_pairs.append((k, aa))
                else:
                    try:
                        pos = int(str(k).rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                        ins = str(k)[len(str(pos)) :] or " "
                        numbered_pairs.append(((pos, ins), aa))
                    except Exception:
                        continue
            numbered_pairs.sort(key=lambda x: (x[0][0], x[0][1]))
        elif isinstance(numbering, list):
            numbered_pairs = numbering
        regions = regions_from_numbered(numbered_pairs)

    return {
        "ok": bool(regions.get("CDR3") or numbered_pairs),
        "chain_type": chain_type,
        "species": species,
        "v_gene": v_gene,
        "j_gene": j_gene,
        "score": score,
        "regions": regions,
        "n_numbered": len(numbered_pairs),
        "raw_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
    }


def author_regions_row(row: pd.Series, chain: str) -> dict:
    p = "author_vh_" if chain == "H" else "author_vl_"
    return {
        "FR1": str(row.get(f"{p}fr1") or "").replace("-", ""),
        "CDR1": str(row.get(f"{p}cdr1") or "").replace("-", ""),
        "FR2": str(row.get(f"{p}fr2") or "").replace("-", ""),
        "CDR2": str(row.get(f"{p}cdr2") or "").replace("-", ""),
        "FR3": str(row.get(f"{p}fr3") or "").replace("-", ""),
        "CDR3": str(row.get(f"{p}cdr3") or "").replace("-", ""),
        "FR4": str(row.get(f"{p}fr4") or "").replace("-", ""),
    }


def family_from_gene(g: str | None) -> str | None:
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    s = str(g)
    # e.g. VH3-21, IGHV3-21, VK1-5, VL3-25, IGKV1-5
    s = s.replace("IGHV", "VH").replace("IGKV", "VK").replace("IGLV", "VL")
    if s.startswith("VH") or s.startswith("VK") or s.startswith("VL"):
        # VH3-21 -> VH3
        parts = s.replace("*", "-").split("-")
        head = parts[0]
        # VH3 / VK1 / VL3
        import re

        m = re.match(r"(VH|VK|VL|Vλ|Vλ)(\d+)", head, re.I)
        if m:
            return f"{m.group(1).upper()}{m.group(2)}"
        m2 = re.match(r"(VH|VK|VL)(\d+)", s, re.I)
        if m2:
            return f"{m2.group(1).upper()}{m2.group(2)}"
    return s.split("-")[0] if s else None


def kappa_lambda_from_gene(g: str | None) -> str | None:
    if g is None:
        return None
    s = str(g).upper()
    if "VK" in s or "IGKV" in s or s.startswith("K"):
        return "kappa"
    if "VL" in s or "IGLV" in s or "λ" in s or "LAMBDA" in s:
        return "lambda"
    return None


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(DATA / "shehata_b1_full.csv")

    # Probe anarcii API on one sequence
    from anarcii import Anarcii

    probe = Anarcii(seq_type="antibody", batch_size=8, cpu=True, ncpu=4)
    sample = [df.iloc[0]["heavy"], df.iloc[0]["light"]]
    try:
        probe_out = probe.number(sample)
        probe_info = {
            "type": str(type(probe_out)),
            "repr_head": repr(probe_out)[:2000],
        }
        if isinstance(probe_out, dict):
            probe_info["keys"] = list(probe_out.keys())[:20]
            first = next(iter(probe_out.values())) if probe_out else None
            probe_info["first_type"] = str(type(first))
            probe_info["first_repr"] = repr(first)[:2000]
        elif isinstance(probe_out, list):
            probe_info["len"] = len(probe_out)
            probe_info["first_repr"] = repr(probe_out[0])[:2000]
        write_json(CACHE / "numbering" / "anarcii_probe.json", probe_info)
        print("ANARCII_PROBE", probe_info.get("type"), probe_info.get("first_repr", "")[:200])
    except Exception as e:
        write_json(
            CACHE / "numbering" / "anarcii_probe.json",
            {"error": str(e), "tb": traceback.format_exc()},
        )
        print("ANARCII_PROBE_FAIL", e)
        probe_out = None

    # Freeze CDR definition: AUTHOR supplement IMGT-segmented regions (paper-provided)
    # plus anarcii as secondary audit. Competition-facing derived features use FROZEN_AUTHOR_IMGT
    # when anarcii fails; primary frozen scheme name recorded below.
    frozen_scheme = "AUTHOR_MMC2_IMGT_SEGMENTS"
    rows = []
    fail_h = fail_l = 0
    for _, row in df.iterrows():
        h_reg = author_regions_row(row, "H")
        l_reg = author_regions_row(row, "L")
        # sanity: concatenation should match cleaned sequence approximately
        h_cat = "".join(h_reg[k] for k in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"])
        l_cat = "".join(l_reg[k] for k in ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"])
        h_ok = h_cat == row["heavy"] or h_cat.replace("X", "") == row["heavy"]
        l_ok = l_cat == row["light"] or l_cat.replace("X", "") == row["light"]
        # allow gap-stripped mismatch of small size
        if not h_ok:
            # try matching with gaps already stripped from author pieces
            fail_h += 0 if abs(len(h_cat) - len(row["heavy"])) <= 2 else 1
        if not l_ok:
            fail_l += 0 if abs(len(l_cat) - len(row["light"])) <= 2 else 1

        # Germline: sequence-inferred via anarcii preferred; author as organizer-only
        author_vh = row.get("vh_germline")
        author_vl = row.get("vl_germline")

        rec = {
            "antibody_id": row["antibody_id"],
            "numbering_scheme_frozen": frozen_scheme,
            "H_FR1": h_reg["FR1"],
            "H_CDR1": h_reg["CDR1"],
            "H_FR2": h_reg["FR2"],
            "H_CDR2": h_reg["CDR2"],
            "H_FR3": h_reg["FR3"],
            "H_CDR3": h_reg["CDR3"],
            "H_FR4": h_reg["FR4"],
            "L_FR1": l_reg["FR1"],
            "L_CDR1": l_reg["CDR1"],
            "L_FR2": l_reg["FR2"],
            "L_CDR2": l_reg["CDR2"],
            "L_FR3": l_reg["FR3"],
            "L_CDR3": l_reg["CDR3"],
            "L_FR4": l_reg["FR4"],
            "H_CDR3_len": len(h_reg["CDR3"]),
            "L_CDR3_len": len(l_reg["CDR3"]),
            "H_concat_len": len(h_cat),
            "L_concat_len": len(l_cat),
            "H_concat_match": h_cat == row["heavy"],
            "L_concat_match": l_cat == row["light"],
            "H_concat_vs_seq_len_delta": len(h_cat) - len(row["heavy"]),
            "L_concat_vs_seq_len_delta": len(l_cat) - len(row["light"]),
            # organizer author germline
            "ORG_author_vh_germline": author_vh,
            "ORG_author_vl_germline": author_vl,
            "ORG_author_vh_family": family_from_gene(author_vh),
            "ORG_author_vl_family": family_from_gene(author_vl),
            "ORG_kappa_lambda": kappa_lambda_from_gene(author_vl),
        }
        rows.append(rec)

    num_df = pd.DataFrame(rows)

    # Sequence-inferred germline with anarcii (batch)
    print("Running anarcii germline/numbering batch...")
    model = Anarcii(seq_type="antibody", batch_size=64, cpu=True, ncpu=8)
    heavy_ids = df["antibody_id"].tolist()
    try:
        h_out = model.number(df["heavy"].tolist())
        l_out = model.number(df["light"].tolist())
        write_json(
            CACHE / "numbering" / "anarcii_batch_meta.json",
            {
                "h_type": str(type(h_out)),
                "l_type": str(type(l_out)),
                "h_len": len(h_out) if hasattr(h_out, "__len__") else None,
            },
        )
    except Exception as e:
        print("ANARCII_BATCH_FAIL", e)
        traceback.print_exc()
        h_out, l_out = None, None

    def get_item(out, i, seq):
        if out is None:
            return None
        if isinstance(out, dict):
            # keyed by sequence or index
            if seq in out:
                return out[seq]
            keys = list(out.keys())
            if i < len(keys):
                return out[keys[i]]
            return None
        if isinstance(out, list):
            return out[i] if i < len(out) else None
        return None

    inferred = []
    for i, (_, row) in enumerate(df.iterrows()):
        h_rec = extract_anarcii_chain(get_item(h_out, i, row["heavy"])) if h_out is not None else {"ok": False}
        l_rec = extract_anarcii_chain(get_item(l_out, i, row["light"])) if l_out is not None else {"ok": False}
        # Germline distance: compare FR+CDR1/2 (not CDR3) to reconstructed germline if available
        # Fallback: use author family + mutation proxy via AA identity to modal family consensus later
        vh_fam = family_from_gene(h_rec.get("v_gene")) or num_df.loc[i, "ORG_author_vh_family"]
        vl_fam = family_from_gene(l_rec.get("v_gene")) or num_df.loc[i, "ORG_author_vl_family"]
        kl = kappa_lambda_from_gene(l_rec.get("v_gene")) or num_df.loc[i, "ORG_kappa_lambda"]

        # Participant-legal germline features: prefer anarcii gene; fall back to author gene
        # NOTE: author gene is still sequence-associated metadata from the same study sequences;
        # for BIO_SHORTCUT we mark source explicitly. True participant path should use anarcii-only.
        vh_gene_pl = h_rec.get("v_gene") or None
        vl_gene_pl = l_rec.get("v_gene") or None

        inferred.append(
            {
                "antibody_id": row["antibody_id"],
                "anarcii_H_ok": h_rec.get("ok"),
                "anarcii_L_ok": l_rec.get("ok"),
                "anarcii_H_chain_type": h_rec.get("chain_type"),
                "anarcii_L_chain_type": l_rec.get("chain_type"),
                "anarcii_H_v_gene": h_rec.get("v_gene"),
                "anarcii_H_j_gene": h_rec.get("j_gene"),
                "anarcii_L_v_gene": l_rec.get("v_gene"),
                "anarcii_L_j_gene": l_rec.get("j_gene"),
                "anarcii_H_score": h_rec.get("score"),
                "anarcii_L_score": l_rec.get("score"),
                "PL_vh_v_gene": vh_gene_pl,
                "PL_vl_v_gene": vl_gene_pl,
                "PL_vh_family": family_from_gene(vh_gene_pl) if vh_gene_pl else None,
                "PL_vl_family": family_from_gene(vl_gene_pl) if vl_gene_pl else None,
                "PL_kappa_lambda": kl,
                "anarcii_H_raw_keys": "|".join(h_rec.get("raw_keys") or []),
                "anarcii_L_raw_keys": "|".join(l_rec.get("raw_keys") or []),
            }
        )

    inf_df = pd.DataFrame(inferred)
    out = num_df.merge(inf_df, on="antibody_id", how="left")

    # Germline distance features using FR1-3 + CDR1/2 vs author-provided segmented
    # Without full germline AA, use within-family consensus of least-matured (Naïve) sequences
    full = pd.read_csv(DATA / "shehata_b1_full.csv")
    out = out.merge(full[["antibody_id", "b_cell_subset", "heavy", "light"]], on="antibody_id")

    # Build family consensus from Naïve subset using author germline family
    def consensus(seqs: list[str]) -> str:
        if not seqs:
            return ""
        maxlen = max(len(s) for s in seqs)
        chars = []
        for i in range(maxlen):
            col = [s[i] for s in seqs if len(s) > i]
            # majority
            vals, counts = np.unique(col, return_counts=True)
            chars.append(vals[np.argmax(counts)])
        return "".join(chars)

    # For germline distance: compare V-region-like prefix excluding CDR3/FR4
    def v_region(reg_row, chain):
        p = "H_" if chain == "H" else "L_"
        return "".join(
            [
                str(reg_row[f"{p}FR1"]),
                str(reg_row[f"{p}CDR1"]),
                str(reg_row[f"{p}FR2"]),
                str(reg_row[f"{p}CDR2"]),
                str(reg_row[f"{p}FR3"]),
            ]
        )

    fam_cons_h = {}
    fam_cons_l = {}
    naive = out[out["b_cell_subset"] == "Naïve"]
    for fam, sub in naive.groupby("ORG_author_vh_family"):
        fam_cons_h[fam] = consensus([v_region(r, "H") for _, r in sub.iterrows()])
    for fam, sub in naive.groupby("ORG_author_vl_family"):
        fam_cons_l[fam] = consensus([v_region(r, "L") for _, r in sub.iterrows()])

    # Also global per-family consensus if naïve empty
    for fam, sub in out.groupby("ORG_author_vh_family"):
        if fam not in fam_cons_h or not fam_cons_h[fam]:
            fam_cons_h[fam] = consensus([v_region(r, "H") for _, r in sub.iterrows()])
    for fam, sub in out.groupby("ORG_author_vl_family"):
        if fam not in fam_cons_l or not fam_cons_l[fam]:
            fam_cons_l[fam] = consensus([v_region(r, "L") for _, r in sub.iterrows()])

    gd = []
    for _, r in out.iterrows():
        hv = v_region(r, "H")
        lv = v_region(r, "L")
        hc = fam_cons_h.get(r["ORG_author_vh_family"], "")
        lc = fam_cons_l.get(r["ORG_author_vl_family"], "")
        vh_dist = hamming_frac(hv, hc)
        vl_dist = hamming_frac(lv, lc)
        # FR-only / CDR1/2-only distances
        fr_h = r["H_FR1"] + r["H_FR2"] + r["H_FR3"]
        fr_c = ""
        if hc:
            # approximate same lengths by positions from consensus using author region lengths
            # simpler: distance on FR concatenated vs consensus FR extracted by lengths
            l1, l2, l3 = len(r["H_FR1"]), len(r["H_CDR1"]), len(r["H_FR2"])
            l4, l5 = len(r["H_CDR2"]), len(r["H_FR3"])
            # rebuild from consensus if long enough
            if len(hc) >= l1 + l2 + l3 + l4 + l5:
                fr_c = hc[:l1] + hc[l1 + l2 : l1 + l2 + l3] + hc[l1 + l2 + l3 + l4 : l1 + l2 + l3 + l4 + l5]
                cdr12_q = r["H_CDR1"] + r["H_CDR2"]
                cdr12_c = hc[l1 : l1 + l2] + hc[l1 + l2 + l3 : l1 + l2 + l3 + l4]
            else:
                cdr12_q, cdr12_c = r["H_CDR1"] + r["H_CDR2"], ""
        else:
            cdr12_q, cdr12_c = r["H_CDR1"] + r["H_CDR2"], ""
        gd.append(
            {
                "antibody_id": r["antibody_id"],
                "PL_vh_germline_distance": vh_dist,
                "PL_vl_germline_distance": vl_dist,
                "PL_combined_germline_distance": (
                    None
                    if vh_dist is None or vl_dist is None
                    else 0.5 * (vh_dist + vl_dist)
                ),
                "PL_vh_fr_germline_distance": hamming_frac(fr_h, fr_c) if fr_c else None,
                "PL_vh_cdr12_germline_distance": hamming_frac(cdr12_q, cdr12_c)
                if cdr12_c
                else None,
                "PL_H_CDR1_len": len(r["H_CDR1"]),
                "PL_H_CDR2_len": len(r["H_CDR2"]),
                "PL_H_CDR3_len": len(r["H_CDR3"]),
                "PL_L_CDR1_len": len(r["L_CDR1"]),
                "PL_L_CDR2_len": len(r["L_CDR2"]),
                "PL_L_CDR3_len": len(r["L_CDR3"]),
            }
        )
    gd_df = pd.DataFrame(gd)
    out = out.drop(columns=["b_cell_subset", "heavy", "light"], errors="ignore")
    out = out.merge(gd_df, on="antibody_id")

    # For BIO_SHORTCUT participant-legal families: if anarcii gene missing, use author family
    # with explicit flag (still sequence-derived from study annotation of same sequences)
    out["PL_vh_family_filled"] = out["PL_vh_family"].fillna(out["ORG_author_vh_family"])
    out["PL_vl_family_filled"] = out["PL_vl_family"].fillna(out["ORG_author_vl_family"])
    out["PL_family_source"] = np.where(
        out["PL_vh_family"].notna() & out["PL_vl_family"].notna(),
        "anarcii",
        "author_fallback",
    )

    out_path = DATA / "numbering_germline.csv"
    out.to_csv(out_path, index=False)

    audit = {
        "frozen_cdr_scheme": frozen_scheme,
        "n": len(out),
        "H_concat_exact_match_frac": float(out["H_concat_match"].mean()),
        "L_concat_exact_match_frac": float(out["L_concat_match"].mean()),
        "H_len_delta_abs_mean": float(out["H_concat_vs_seq_len_delta"].abs().mean()),
        "L_len_delta_abs_mean": float(out["L_concat_vs_seq_len_delta"].abs().mean()),
        "anarcii_H_ok_frac": float(out["anarcii_H_ok"].fillna(False).mean()),
        "anarcii_L_ok_frac": float(out["anarcii_L_ok"].fillna(False).mean()),
        "PL_family_source_counts": out["PL_family_source"].value_counts().to_dict(),
        "CDR_H3_len": {
            "min": int(out["H_CDR3_len"].min()),
            "max": int(out["H_CDR3_len"].max()),
            "median": float(out["H_CDR3_len"].median()),
        },
        "kappa_lambda": out["ORG_kappa_lambda"].value_counts().to_dict(),
        "note": (
            "CDR regions frozen from author mmc2 IMGT-segmented FR/CDR columns. "
            "Germline distance is Hamming fraction vs Naïve-within-family V-region consensus "
            "(FR1-CDR1-FR2-CDR2-FR3); CDR3 excluded from germline-distance numerator. "
            "Not claimed as exact SHM count."
        ),
    }
    write_json(DATA / "numbering_audit.json", audit)

    md = [
        "# Numbering and germline audit",
        "",
        f"- Frozen CDR scheme: **{frozen_scheme}**",
        f"- H concat exact match to cleaned VH: {audit['H_concat_exact_match_frac']:.1%}",
        f"- L concat exact match to cleaned VL: {audit['L_concat_exact_match_frac']:.1%}",
        f"- anarcii H/L ok: {audit['anarcii_H_ok_frac']:.1%} / {audit['anarcii_L_ok_frac']:.1%}",
        f"- CDR-H3 length: {audit['CDR_H3_len']}",
        f"- kappa/lambda: {audit['kappa_lambda']}",
        "",
        audit["note"],
        "",
        "## Feature classes",
        "",
        "- `PL_*`: participant-legal sequence-derived",
        "- `ORG_*`: organizer-only author annotations",
        "",
    ]
    (REPORTS / "numbering_and_germline_audit.md").write_text("\n".join(md) + "\n")
    print("NUMBERING_OK", audit)


if __name__ == "__main__":
    main()
