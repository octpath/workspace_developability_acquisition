"""ANARCI API shim using anarcii."""
from __future__ import annotations

scheme_short_to_long = {
    "imgt": "imgt",
    "chothia": "chothia",
    "kabat": "kabat",
    "aho": "aho",
    "martin": "martin",
}

_VALID = set("ACDEFGHIKLMNPQRSTVWY")
_model = None


def validate_sequence(sequence: str) -> None:
    bad = [c for c in sequence if c not in _VALID]
    if bad:
        raise ValueError(f"Invalid amino acids: {set(bad)}")


def _get_model():
    global _model
    if _model is None:
        from anarcii import Anarcii

        _model = Anarcii(seq_type="antibody", batch_size=8, cpu=True, ncpu=2)
    return _model


def anarci(sequences, scheme="imgt", output=False, allow=None, allowed_species=None):
    model = _get_model()
    seqs = [s for _, s in sequences]
    out = model.number(seqs)
    numbered = []
    alignment_details = []
    hit_tables = []
    keys = list(out.keys()) if isinstance(out, dict) else None
    for i, (sid, seq) in enumerate(sequences):
        if isinstance(out, dict):
            rec = out[keys[i]]
        else:
            rec = out[i]
        if not rec or rec.get("error"):
            numbered.append(None)
            alignment_details.append(None)
            hit_tables.append(None)
            continue
        chain = rec.get("chain_type")
        if allow is not None:
            allow_set = set(allow)
            ok = chain in allow_set or (chain == "K" and ("L" in allow_set or "K" in allow_set)) or (
                chain == "L" and ("L" in allow_set or "K" in allow_set)
            )
            if not ok:
                numbered.append(None)
                alignment_details.append(None)
                hit_tables.append(None)
                continue
        num = rec["numbering"]
        details = {"chain_type": chain, "scheme": scheme, "species": "human"}
        numbered.append([(num, details, None)])
        alignment_details.append([details])
        hit_tables.append(None)
    return numbered, alignment_details, hit_tables
