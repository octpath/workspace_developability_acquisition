#!/usr/bin/env python3
"""Download Ginkgo GDPa1 (and GDPa3 if present) from Hugging Face.

Requires HF_TOKEN with access after accepting dataset terms:
  export HF_TOKEN=hf_...
  python scripts/download_ginkgo.py

Does not fabricate authentication. Without a token, prints gated metadata only.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "raw" / "ginkgo"
CACHE = ROOT / "cache" / "hf"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DATASETS = ["ginkgo-datapoints/GDPa1", "ginkgo-datapoints/GDPa3"]
ACCESS_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    h = {"User-Agent": "developability-acquisition/0.1"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_json(client: httpx.Client, url: str) -> tuple[int, object]:
    r = client.get(url)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def download_file(client: httpx.Client, repo: str, path: str, dest: Path) -> tuple[int, int]:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    r = client.get(url)
    if r.status_code == 200:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
    return r.status_code, len(r.content)


def main() -> int:
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"HF_TOKEN present: {token_present}")
    client = httpx.Client(timeout=120.0, follow_redirects=True, headers=headers())
    summary = []

    for repo in DATASETS:
        name = repo.split("/")[-1]
        print(f"\n=== {repo} ===")
        status, api = fetch_json(client, f"https://huggingface.co/api/datasets/{repo}")
        api_path = OUT / f"{name}_api.json"
        api_path.write_text(json.dumps(api, indent=2) if not isinstance(api, str) else api)
        print(f"API status={status}")
        if status != 200:
            summary.append({"repo": repo, "status": "ACCESS_BLOCKED_OR_MISSING", "http": status})
            continue

        gated = api.get("gated") if isinstance(api, dict) else None
        print(f"gated={gated}")
        t_status, tree = fetch_json(client, f"https://huggingface.co/api/datasets/{repo}/tree/main")
        (OUT / f"{name}_tree.json").write_text(
            json.dumps(tree, indent=2) if not isinstance(tree, str) else tree
        )
        if t_status != 200 or not isinstance(tree, list):
            summary.append({"repo": repo, "status": "TREE_BLOCKED", "http": t_status, "gated": gated})
            continue

        dest_dir = OUT / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        files_ok = []
        files_fail = []
        for item in tree:
            if item.get("type") != "file":
                continue
            path = item["path"]
            size = item.get("size") or 0
            # Skip large structure bundles unless explicitly needed later
            if path.startswith("structures/") and size > 5_000_000:
                print(f"SKIP large structure {path} ({size})")
                continue
            dest = dest_dir / path
            code, nbytes = download_file(client, repo, path, dest)
            print(f"  {path}: HTTP {code} bytes={nbytes}")
            if code == 200:
                files_ok.append(path)
            else:
                files_fail.append((path, code))

        if files_fail and not files_ok:
            st = "ACCESS_BLOCKED_ONLY_BY_AUTHENTICATION"
        elif files_ok and files_fail:
            st = "PARTIAL_DOWNLOAD"
        elif files_ok:
            st = "DOWNLOADED"
        else:
            st = "NO_FILES"
        summary.append(
            {
                "repo": repo,
                "status": st,
                "gated": gated,
                "files_ok": files_ok,
                "files_fail": files_fail,
                "access_date": ACCESS_DATE,
            }
        )

    out = OUT / "download_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\nWrote", out)
    if any(s["status"].startswith("ACCESS_BLOCKED") for s in summary):
        print(
            "STATUS: ACCESS_BLOCKED_ONLY_BY_AUTHENTICATION for gated repos without approved HF_TOKEN"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
