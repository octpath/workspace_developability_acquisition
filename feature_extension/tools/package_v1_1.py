#!/usr/bin/env python3
"""Package feature_extension v1.1 additive archives (does not delete v1 zips)."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

FE = Path(__file__).resolve().parents[1]
DIST = FE / "dist"
SKIP = {".git", "__pycache__", "dist", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip(path: Path) -> bool:
    rel = path.relative_to(FE)
    return any(p in SKIP for p in rel.parts) or path.suffix == ".pyc"


def write_zip(name: str, paths: list[Path]) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / name
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.is_file() and not should_skip(path):
                zf.write(path, str(Path("feature_extension") / path.relative_to(FE)))
    print(f"wrote {out.name} ({out.stat().st_size/1e6:.2f} MB)")
    return out


def main():
    archives = []
    # additive fennix-only
    fennix_files = [
        FE / "data/precomputed_features/fennix_fab_context.parquet",
        FE / "data/precomputed_features/FENNIX_FAB_CONTEXT_FEATURE_DICTIONARY.csv",
        FE / "data/precomputed_features/README.md",
        FE / "RELEASE_NOTES.md",
        FE / "RELEASE_AUDIT.md",
    ]
    archives.append(write_zip("feature_extension_v1.1_fennix_fab_context.zip", fennix_files))

    # full precomputed refresh
    prec = list((FE / "data/precomputed_features").rglob("*"))
    archives.append(write_zip("feature_extension_v1.1_precomputed_features.zip", prec))

    # code refresh
    code_paths = []
    for sub in ("extractors", "examples", "tests", "tools"):
        code_paths.extend((FE / sub).rglob("*"))
    for name in (
        "README.md",
        "RELEASE_NOTES.md",
        "RELEASE_AUDIT.md",
        "MANIFEST.csv",
        "BLOCK_COVERAGE.csv",
        "folds.csv",
        "requirements.txt",
        "__init__.py",
    ):
        code_paths.append(FE / name)
    archives.append(
        write_zip(
            "feature_extension_v1.1_code.zip",
            [p for p in code_paths if p.is_file() and not (p.parent.name == "tools" and p.name.startswith(("build_v1", "_build")))],
        )
    )

    # append to SHA256SUMS (keep v1 entries if present)
    sums_path = DIST / "SHA256SUMS.txt"
    existing = {}
    if sums_path.exists():
        for line in sums_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                existing[parts[-1]] = parts[0]
    for a in archives:
        existing[a.name] = sha256_file(a)
    # also refresh v1 hashes if files still present
    for p in DIST.glob("feature_extension_v1_*.zip"):
        existing[p.name] = sha256_file(p)
    lines = [f"{h}  {n}" for n, h in sorted(existing.items())]
    sums_path.write_text("\n".join(lines) + "\n")
    print("updated SHA256SUMS.txt")


if __name__ == "__main__":
    main()
