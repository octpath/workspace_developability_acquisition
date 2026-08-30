#!/usr/bin/env python3
"""Fetch UniProt sequences for a list of accessions (cached, rate-limited)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "uniprot"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_one(client: httpx.Client, acc: str) -> str | None:
    cached = CACHE / f"{acc}.fasta"
    if cached.exists():
        return cached.read_text()
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    for attempt in range(5):
        r = client.get(url)
        if r.status_code == 200:
            cached.write_text(r.text)
            return r.text
        if r.status_code == 404:
            return None
        time.sleep(2**attempt)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("accessions", nargs="+", help="UniProt accessions")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "interim" / "uniprot_sequences.fasta")
    args = ap.parse_args()
    client = httpx.Client(timeout=60.0, follow_redirects=True)
    chunks = []
    for acc in args.accessions:
        fa = fetch_one(client, acc)
        print(acc, "OK" if fa else "MISSING")
        if fa:
            chunks.append(fa.rstrip() + "\n")
        time.sleep(0.2)
    args.output.write_text("".join(chunks))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
