#!/usr/bin/env python3
"""Fetch polymer sequences from RCSB PDB/mmCIF for a list of PDB IDs (cached)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "rcsb"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_entry(client: httpx.Client, pdb_id: str) -> dict | None:
    pdb_id = pdb_id.upper()
    cached = CACHE / f"{pdb_id}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    for attempt in range(5):
        r = client.get(url)
        if r.status_code == 200:
            cached.write_text(r.text)
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2**attempt)
    return None


def fetch_polymer_entities(client: httpx.Client, pdb_id: str) -> list[dict]:
    pdb_id = pdb_id.upper()
    # GraphQL for polymer entity sequences
    query = """
    query ($id: String!) {
      entry(entry_id: $id) {
        polymer_entities {
          entity_poly {
            pdbx_seq_one_letter_code_can
            rcsb_entity_polymer_type
          }
          rcsb_polymer_entity_container_identifiers {
            auth_asym_ids
            reference_sequence_identifiers { database_accession database_name }
          }
          rcsb_polymer_entity { pdbx_description }
        }
      }
    }
    """
    cached = CACHE / f"{pdb_id}_polymers.json"
    if cached.exists():
        return json.loads(cached.read_text())
    r = client.post(
        "https://data.rcsb.org/graphql",
        json={"query": query, "variables": {"id": pdb_id}},
    )
    r.raise_for_status()
    entities = (((r.json() or {}).get("data") or {}).get("entry") or {}).get("polymer_entities") or []
    cached.write_text(json.dumps(entities, indent=2))
    return entities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_ids", nargs="+")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "interim" / "pdb_sequences.json")
    args = ap.parse_args()
    client = httpx.Client(timeout=60.0, follow_redirects=True)
    out = {}
    for pdb_id in args.pdb_ids:
        ents = fetch_polymer_entities(client, pdb_id)
        out[pdb_id.upper()] = ents
        print(pdb_id, "entities", len(ents))
        time.sleep(0.3)
    args.output.write_text(json.dumps(out, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
