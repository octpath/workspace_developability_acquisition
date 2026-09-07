#!/usr/bin/env python3
"""Create participant download zips under feature_extension/dist/."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

FE = Path(__file__).resolve().parents[1]
DIST = FE / "dist"

SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "dist", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(FE)
    if any(p in SKIP_PARTS for p in rel.parts):
        return True
    if path.suffix in SKIP_SUFFIX:
        return True
    if path.name.startswith(".") and path.name not in {".gitkeep"}:
        return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_tree(zf: zipfile.ZipFile, src_root: Path, arc_prefix: str, predicate=None) -> int:
    n = 0
    for path in sorted(src_root.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        if predicate and not predicate(path):
            continue
        rel = path.relative_to(src_root) if src_root != FE else path.relative_to(FE)
        # when src_root is FE, rel is package-relative
        if src_root == FE:
            arcname = str(Path(arc_prefix) / path.relative_to(FE))
        else:
            arcname = str(Path(arc_prefix) / path.relative_to(src_root))
        zf.write(path, arcname)
        n += 1
    return n


def write_zip(name: str, callback) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / name
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        callback(zf)
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB)")
    return out


def main() -> None:
    archives = []

    def code_zip(zf):
        keep_dirs = {
            "extractors",
            "examples",
            "tests",
            "tools",
        }
        keep_files = {
            "README.md",
            "RELEASE_NOTES.md",
            "RELEASE_AUDIT.md",
            "MANIFEST.csv",
            "folds.csv",
            "requirements.txt",
            "__init__.py",
        }
        for path in FE.rglob("*"):
            if not path.is_file() or should_skip(path):
                continue
            rel = path.relative_to(FE)
            if rel.parts[0] == "data":
                continue
            if rel.parts[0] in keep_dirs or rel.name in keep_files or rel.parts[0] == "tools":
                if rel.parts[0] == "tools" and (
                    rel.name.startswith("_build") or rel.name == "build_v1_data.py"
                ):
                    continue
                zf.write(path, str(Path("feature_extension") / rel))

    archives.append(write_zip("feature_extension_v1_code.zip", code_zip))

    def struct_zip(subdir: str, zipname: str):
        def cb(zf):
            root = FE / "data" / subdir
            for path in root.rglob("*"):
                if path.is_file() and not should_skip(path):
                    zf.write(path, str(Path("feature_extension") / path.relative_to(FE)))
        archives.append(write_zip(zipname, cb))

    struct_zip("esmfold_fv", "feature_extension_v1_esmfold_fv.zip")
    struct_zip("esmfold_fab", "feature_extension_v1_esmfold_fab.zip")

    def bioemu(zf):
        root = FE / "data" / "bioemu_isolated"
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, str(Path("feature_extension") / path.relative_to(FE)))

    archives.append(write_zip("feature_extension_v1_bioemu_isolated.zip", bioemu))

    def prec(zf):
        root = FE / "data" / "precomputed_features"
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, str(Path("feature_extension") / path.relative_to(FE)))

    archives.append(write_zip("feature_extension_v1_precomputed_features.zip", prec))

    sums = DIST / "SHA256SUMS.txt"
    lines = []
    for a in archives:
        lines.append(f"{sha256_file(a)}  {a.name}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {sums}")


if __name__ == "__main__":
    main()
