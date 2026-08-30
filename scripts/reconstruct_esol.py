#!/usr/bin/env python3
"""Reconstruct eSOL sequence→solubility table from LSDB archive + UniProt E. coli proteome.

Preferred join keys (in order): B number (locus tag) → JW_ID → gene primary name.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
ESOL_CSV = ROOT / "raw" / "esol" / "extracted" / "esol.csv"
FASTA = ROOT / "raw" / "esol" / "uniprot_ecoli.fasta"
OLN = ROOT / "raw" / "esol" / "uniprot_gene_oln.tsv"
OUT = ROOT / "interim" / "esol_reconstructed.csv"
AUDIT = ROOT / "interim" / "esol_join_audit.csv"


def load_indexes():
    by_acc = {}
    for rec in SeqIO.parse(FASTA, "fasta"):
        acc = rec.id.split("|")[1] if "|" in rec.id else rec.id
        by_acc[acc] = str(rec.seq)

    oln = pd.read_csv(OLN, sep="\t")
    by_b, by_jw, by_gene = defaultdict(list), defaultdict(list), defaultdict(list)
    for _, r in oln.iterrows():
        acc = r["Entry"]
        if acc not in by_acc:
            continue
        ol = r.get("Gene Names (ordered locus)")
        if pd.notna(ol):
            for part in re.split(r"[;\s]+", str(ol).strip()):
                pl = part.lower()
                if re.match(r"^b\d+", pl):
                    by_b[pl].append(acc)
                elif re.match(r"^jw\d+", pl):
                    by_jw[pl].append(acc)
        primary = r.get("Gene Names (primary)")
        if pd.notna(primary):
            by_gene[str(primary).strip().lower()].append(acc)
        names = r.get("Gene Names")
        if pd.notna(names):
            for part in re.split(r"[;\s]+", str(names).strip()):
                if part and not re.match(r"^(b|jw)\d+", part.lower()):
                    by_gene[part.lower()].append(acc)
    return by_acc, by_b, by_jw, by_gene


def uniq(accs):
    seen = []
    for a in accs:
        if a not in seen:
            seen.append(a)
    return seen


def pick(by_acc, accs):
    accs = uniq(accs)
    if len(accs) == 1:
        return accs[0], by_acc[accs[0]], "MAPPED_UNIQUE", "EXACT_ACCESSION", ""
    if not accs:
        return "", "", "UNRESOLVED", "UNRESOLVED", ""
    seqs = {by_acc[a] for a in accs}
    if len(seqs) == 1:
        return (
            accs[0],
            by_acc[accs[0]],
            "MAPPED_MULTI_ID_SAME_SEQ",
            "EXACT_ACCESSION",
            ",".join(accs),
        )
    return ";".join(accs), "", "AMBIGUOUS", "AMBIGUOUS", ",".join(accs)


def resolve(row, by_acc, by_b, by_jw, by_gene):
    b = str(row["B number"]).strip().lower() if pd.notna(row["B number"]) else ""
    jw = str(row["JW_ID"]).strip().lower() if pd.notna(row["JW_ID"]) else ""
    gene = (
        str(row["Gene name K-12"]).strip().lower()
        if pd.notna(row["Gene name K-12"])
        else ""
    )
    if b and b in by_b:
        return pick(by_acc, by_b[b])
    if jw and jw in by_jw:
        return pick(by_acc, by_jw[jw])
    if gene and gene in by_gene:
        return pick(by_acc, by_gene[gene])
    syn = row.get("Synonyms of locus names K-12")
    if pd.notna(syn):
        for part in re.split(r"[;,\s]+", str(syn).strip()):
            pl = part.lower()
            if pl in by_b:
                return pick(by_acc, by_b[pl])
            if pl in by_jw:
                return pick(by_acc, by_jw[pl])
            if pl in by_gene:
                return pick(by_acc, by_gene[pl])
    return "", "", "UNRESOLVED", "UNRESOLVED", ""


def main() -> None:
    if not ESOL_CSV.exists():
        raise SystemExit(f"Missing {ESOL_CSV}")
    if not FASTA.exists() or not OLN.exists():
        raise SystemExit("Missing UniProt FASTA and/or gene_oln TSV")

    by_acc, by_b, by_jw, by_gene = load_indexes()
    esol = pd.read_csv(ESOL_CSV)
    labeled = esol[esol["Solubility (%)"].notna()].copy()
    rows = []
    for _, r in labeled.iterrows():
        acc, seq, status, conf, notes = resolve(r, by_acc, by_b, by_jw, by_gene)
        rows.append(
            {
                "id": r["JW_ID"],
                "original_identifier": r["JW_ID"],
                "b_number": r["B number"],
                "gene_name": r["Gene name K-12"],
                "sequence": seq,
                "solubility": r["Solubility (%)"],
                "yield_uM": r["Yield (uM)"],
                "yield_ug_ml": r["Yield (ug/ml)"],
                "sequence_mapping_status": status,
                "sequence_source": "UniProt_UP000000625_reviewed" if seq else "",
                "accession_original": r["JW_ID"],
                "accession_resolved": acc,
                "sequence_length": len(seq) if seq else None,
                "join_confidence": conf,
                "mapping_notes": notes,
                "cell_location": r["Cell location"],
                "label_source": "eSOL_LSDB_NBDC00440",
            }
        )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    audit = out.groupby("sequence_mapping_status").size().rename("count").reset_index()
    audit.to_csv(AUDIT, index=False)
    ok = int(out["sequence"].astype(bool).sum())
    print(audit.to_string(index=False))
    print(
        f"RECONSTRUCTION: {len(out)} labels; {ok} unambiguous sequences; "
        f"{(out.sequence_mapping_status=='AMBIGUOUS').sum()} ambiguous; "
        f"{(out.sequence_mapping_status=='UNRESOLVED').sum()} unresolved"
    )


if __name__ == "__main__":
    main()
