#!/usr/bin/env bash
# Reassemble GitHub-split residue embedding .npy files from .part0 + .part1.
# Safe to re-run: skips outputs that already exist and match part sizes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

assemble_one() {
  local out="$1"
  local p0="${out}.part0"
  local p1="${out}.part1"

  if [[ ! -f "$p0" || ! -f "$p1" ]]; then
    echo "ERROR: missing parts for ${out}" >&2
    echo "  expected: ${p0}" >&2
    echo "  expected: ${p1}" >&2
    return 1
  fi

  local expected=$(( $(stat -c%s "$p0") + $(stat -c%s "$p1") ))
  if [[ -f "$out" ]]; then
    local actual
    actual=$(stat -c%s "$out")
    if [[ "$actual" -eq "$expected" ]]; then
      echo "OK (exists): ${out#"$ROOT"/}"
      return 0
    fi
    echo "WARN: size mismatch for ${out#"$ROOT"/} (${actual} vs ${expected}); rebuilding"
  fi

  local tmp="${out}.tmp.$$"
  cat "$p0" "$p1" > "$tmp"
  mv -f "$tmp" "$out"
  echo "assembled: ${out#"$ROOT"/} (${expected} bytes)"
}

assemble_hl_pack() {
  local sub="$1"
  if [[ -f "$ROOT/$sub/heavy_embeddings.npy.part0" ]]; then
    assemble_one "$ROOT/$sub/heavy_embeddings.npy"
    assemble_one "$ROOT/$sub/light_embeddings.npy"
  fi
}

main() {
  assemble_one "$ROOT/ablingua600m/heavy_embeddings.npy"
  assemble_one "$ROOT/ablingua600m/light_embeddings.npy"
  assemble_one "$ROOT/esm2/heavy_embeddings.npy"
  if [[ -f "$ROOT/esm2/light_embeddings.npy.part0" ]]; then
    assemble_one "$ROOT/esm2/light_embeddings.npy"
  fi
  assemble_hl_pack "ablang2"
  assemble_hl_pack "ablang1"
  assemble_hl_pack "esm1b"
  assemble_hl_pack "esmc600m"
  assemble_hl_pack "currab"
  assemble_hl_pack "ablang2_unpaired"
  assemble_hl_pack "currab_unpaired"
  echo "Done. Residue embeddings are ready under residue_level/."
}

main "$@"
