#!/usr/bin/env python3
"""Publish the latest dashboard artifacts for the shareable Streamlit app.

Run this after the data pipeline succeeds:

    python scripts/publish_streamlit_snapshot.py --commit --push

The script validates the small, app-facing processed files, writes a manifest,
and optionally commits/pushes them so Streamlit Cloud can redeploy from GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "dashboard_manifest.json"

REQUIRED_ARTIFACTS = [
    PROCESSED / "dashboard_data_500k.pkl",
    PROCESSED / "brand_posts_index.pkl",
]

OPTIONAL_ARTIFACTS = [
    PROCESSED / "forecast_data.pkl",
    PROCESSED / "cluster_brand_labels.csv",
    PROCESSED / "target_cluster_ids.txt",
    PROCESSED / "archive" / "dashboard_weekly_archive.pkl",
    PROCESSED / "archive" / "dashboard_weekly_archive.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def current_commit() -> str | None:
    result = git(["rev-parse", "--short", "HEAD"], check=False)
    return result.stdout.strip() or None


def artifact_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def build_manifest() -> dict[str, object]:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required dashboard artifacts: " + ", ".join(missing))

    artifacts = [artifact_record(p) for p in REQUIRED_ARTIFACTS]
    artifacts.extend(artifact_record(p) for p in OPTIONAL_ARTIFACTS if p.exists())

    return {
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": current_commit(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "streamlit_entrypoint": "dashboard_v2.py",
    }


def commit_and_push(push: bool) -> None:
    paths = [str(p.relative_to(ROOT)) for p in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS if p.exists()]
    paths.append(str(MANIFEST.relative_to(ROOT)))
    git(["add", *paths])

    status = git(["status", "--porcelain"]).stdout.strip()
    if not status:
        print("No dashboard artifact changes to commit.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    git(["commit", "-m", f"data: refresh streamlit dashboard snapshot ({stamp})"])
    if push:
        git(["push"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="Commit refreshed dashboard artifacts.")
    parser.add_argument("--push", action="store_true", help="Push after committing.")
    args = parser.parse_args()

    manifest = build_manifest()
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} with {manifest['artifact_count']} artifacts.")

    if args.commit or args.push:
        commit_and_push(push=args.push)


if __name__ == "__main__":
    main()
