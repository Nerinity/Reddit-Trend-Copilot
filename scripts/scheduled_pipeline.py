#!/usr/bin/env python3
"""Scheduled orchestration for the Trend Radar pipeline.

Commands:
  daily-scrape     Collect new raw signals only.
  weekly-publish   Collect latest raw signals, run incremental NLP/dashboard,
                   then publish lightweight Streamlit artifacts to GitHub.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "scraped_2026_large.csv"
STATE_DIR = ROOT / "data" / "state"
LOG_DIR = ROOT / "data" / "logs"
LOCK_FILE = STATE_DIR / "scheduled_pipeline.lock"
STATE_FILE = STATE_DIR / "scheduled_pipeline_state.json"

SCRAPE_SCRIPT = ROOT / "scripts" / "scrape_large_2026.py"
INCREMENTAL_SCRIPT = ROOT / "scripts" / "incremental_update.py"
DASHBOARD_SCRIPT = ROOT / "scripts" / "build_dashboard_500k.py"
FORECAST_SCRIPT = ROOT / "scripts" / "build_forecast.py"
PUBLISH_SCRIPT = ROOT / "scripts" / "publish_streamlit_snapshot.py"

DEFAULT_DAILY_SOURCES = "reddit_json,rss,gnews,hn"


def setup_logging(command: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{command}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    return log_path


log = logging.getLogger("scheduled_pipeline")


@contextmanager
def pipeline_lock():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = LOCK_FILE.open("x")
    except FileExistsError as exc:
        raise SystemExit(f"Pipeline is already running: {LOCK_FILE}") from exc
    try:
        fd.write(json.dumps({"pid": "local", "started_at_utc": datetime.now(timezone.utc).isoformat()}))
        fd.close()
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def run(cmd: list[str], label: str) -> None:
    log.info("▶ %s", label)
    log.info("  %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    log.info("✓ %s done in %.1f min", label, (time.time() - t0) / 60)


def count_raw_rows() -> int:
    if not RAW_CSV.exists():
        return 0
    try:
        return len(pd.read_csv(RAW_CSV, usecols=["mention_id"], low_memory=False))
    except Exception:
        return 0


def date_window(days_back: int) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    return start.isoformat(), today.isoformat()


def write_state(command: str, status: str, extra: dict | None = None) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "command": command,
        "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def daily_scrape(args: argparse.Namespace) -> None:
    start, end = date_window(args.days_back)
    current_rows = count_raw_rows()
    target = current_rows + args.daily_target

    run(
        [
            sys.executable,
            str(SCRAPE_SCRIPT),
            "--start",
            start,
            "--end",
            end,
            "--sources",
            args.sources,
            "--target",
            str(target),
            "--per-sub",
            str(args.per_sub),
            "--delay",
            str(args.delay),
        ],
        f"daily scrape ({start} → {end})",
    )

    write_state(
        "daily-scrape",
        "success",
        {
            "start": start,
            "end": end,
            "raw_rows_before": current_rows,
            "raw_rows_after": count_raw_rows(),
        },
    )


def weekly_publish(args: argparse.Namespace) -> None:
    if not args.skip_scrape:
        daily_scrape(args)

    run([sys.executable, str(INCREMENTAL_SCRIPT), "--skip-dashboard"], "incremental NLP update")
    run([sys.executable, str(DASHBOARD_SCRIPT)], "weekly dashboard build")

    if not args.skip_forecast:
        run([sys.executable, str(FORECAST_SCRIPT)], "forecast build")

    publish_cmd = [sys.executable, str(PUBLISH_SCRIPT), "--commit"]
    if args.push:
        publish_cmd.append("--push")
    run(publish_cmd, "publish Streamlit snapshot")

    write_state(
        "weekly-publish",
        "success",
        {
            "pushed_to_github": bool(args.push),
            "forecast_updated": not args.skip_forecast,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduled Trend Radar pipeline tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser("daily-scrape", help="Collect new raw data only.")
    daily.add_argument("--days-back", type=int, default=2, help="UTC lookback window for daily collection.")
    daily.add_argument("--daily-target", type=int, default=50_000, help="Additional raw rows to attempt per run.")
    daily.add_argument("--sources", default=DEFAULT_DAILY_SOURCES, help="Sources for scrape_large_2026.py.")
    daily.add_argument("--per-sub", type=int, default=80, help="Max records per subreddit for daily collection.")
    daily.add_argument("--delay", type=float, default=1.5, help="Request delay in seconds.")

    weekly = sub.add_parser("weekly-publish", help="Update NLP/dashboard and publish Streamlit artifacts.")
    weekly.add_argument("--days-back", type=int, default=3, help="UTC lookback window before weekly build.")
    weekly.add_argument("--daily-target", type=int, default=75_000, help="Additional raw rows to attempt before build.")
    weekly.add_argument("--sources", default=DEFAULT_DAILY_SOURCES, help="Sources for scrape_large_2026.py.")
    weekly.add_argument("--per-sub", type=int, default=120, help="Max records per subreddit before weekly build.")
    weekly.add_argument("--delay", type=float, default=1.5, help="Request delay in seconds.")
    weekly.add_argument("--skip-scrape", action="store_true", help="Only process existing raw data.")
    weekly.add_argument("--skip-forecast", action="store_true", help="Skip forecast rebuild.")
    weekly.add_argument("--push", action="store_true", help="Push published Streamlit artifacts to GitHub.")

    args = parser.parse_args()
    log_path = setup_logging(args.command)
    log.info("Log file: %s", log_path)

    with pipeline_lock():
        try:
            if args.command == "daily-scrape":
                daily_scrape(args)
            elif args.command == "weekly-publish":
                weekly_publish(args)
        except Exception:
            write_state(args.command, "failed")
            raise


if __name__ == "__main__":
    main()
