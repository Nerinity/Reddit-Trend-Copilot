#!/usr/bin/env python3
"""
Cluster-aware Reddit scraping: fetches posts relevant to each of the 171
target clusters defined in configs/taxonomy/target_clusters_171.csv.

Strategy
--------
For each target cluster:
  1. Build search queries from seed_keywords + cluster name + aliases.
  2. Hit Reddit public search API (no auth): /search.json?q={query}
  3. Optionally search within relevant subreddits from sources.json.
  4. Store posts with target_cluster_hint / target_cluster_id fields.
  5. Checkpoint progress by target_cluster_id.
  6. Deduplicate by mention_id and append to data/raw/scraped_2026_large.csv.

The cluster-aware scraper complements the semantic assignment in
assign_target_clusters.py: it actively pulls posts for under-covered clusters
rather than relying solely on what general-purpose scrapers happened to collect.

Usage
-----
    python scripts/scrape_by_target_clusters.py              # full run
    python scripts/scrape_by_target_clusters.py --dry-run    # show queries only
    python scripts/scrape_by_target_clusters.py --cluster C001 C005 C010
    python scripts/scrape_by_target_clusters.py --reset      # clear checkpoint
    python scripts/scrape_by_target_clusters.py --limit 25   # 25 posts/query (test)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("scrape_by_clusters")

_ROOT         = Path(__file__).parent.parent
TAXONOMY_CSV  = _ROOT / "configs" / "taxonomy" / "target_clusters_171.csv"
SOURCES_JSON  = _ROOT / "configs" / "sources.json"
RAW_CSV       = _ROOT / "data" / "raw" / "scraped_2026_large.csv"
CHECKPOINT    = _ROOT / "data" / "state" / "cluster_scrape_checkpoint.json"

REDDIT_BASE   = "https://www.reddit.com"
USER_AGENT    = "TrendCopilot/1.0 (+https://github.com/Nerinity/Reddit-Trend-Copilot)"
REQUEST_DELAY = 2.0     # seconds between API calls (respect rate limits)
MAX_PER_QUERY = 100     # Reddit JSON limit per page
MAX_PAGES     = 3       # pages per query (300 posts max per query)
MAX_QUERIES   = 4       # max queries per cluster
CUTOFF_TS     = 1735689600  # 2026-01-01 00:00 UTC

# Columns written to CSV (must match raw CSV schema + new hint fields)
CSV_FIELDNAMES = [
    "mention_id", "title", "text", "author", "community", "url",
    "engagement_score", "published_at", "source", "category",
    "target_cluster_hint", "target_cluster_id",
    "matched_query", "source_confidence", "taxonomy_level",
]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"done_clusters": [], "total_collected": 0}


def save_checkpoint(cp: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(cp, indent=2))


# ── Reddit public search API ──────────────────────────────────────────────────

def reddit_search(
    query: str,
    sort:  str        = "relevance",
    t:     str        = "year",
    after: str | None = None,
    limit: int        = MAX_PER_QUERY,
) -> tuple[list[dict], str | None]:
    """Call Reddit public /search.json. Returns (children, after_token)."""
    params: dict = {"q": query, "sort": sort, "t": t, "limit": limit, "type": "link"}
    if after:
        params["after"] = after
    try:
        resp = requests.get(
            f"{REDDIT_BASE}/search.json",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if resp.status_code == 429:
            log.warning("Rate-limited by Reddit, sleeping 90s …")
            time.sleep(90)
            return [], None
        if resp.status_code != 200:
            log.debug("HTTP %d for query %r", resp.status_code, query)
            return [], None
        data = resp.json().get("data", {})
        return data.get("children", []), data.get("after")
    except Exception as exc:
        log.debug("Search error for %r: %s", query, exc)
        return [], None


# ── Record builder ────────────────────────────────────────────────────────────

def child_to_record(
    child:        dict,
    cluster_id:   str,
    cluster_name: str,
    query:        str,
    parent_cat:   str,
) -> dict | None:
    """Convert a Reddit JSON child dict to a CSV row dict."""
    post = child.get("data", {})
    created = float(post.get("created_utc") or 0)
    if created < CUTOFF_TS:
        return None   # pre-2026 post

    title     = (post.get("title") or "").strip()
    if not title:
        return None

    text      = (post.get("selftext") or "").strip()
    if text in ("[deleted]", "[removed]"):
        text = ""

    permalink = post.get("permalink", "")
    mid       = str(post.get("id") or post.get("name") or "")
    subreddit = post.get("subreddit", "")
    score     = int(post.get("score") or 0) + int(post.get("num_comments") or 0) * 3
    pub       = (
        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        if created else ""
    )

    return {
        "mention_id":          mid,
        "title":               title,
        "text":                text[:400],
        "author":              post.get("author", ""),
        "community":           subreddit,
        "url":                 f"https://reddit.com{permalink}" if permalink else "",
        "engagement_score":    score,
        "published_at":        pub,
        "source":              "reddit_cluster_search",
        "category":            parent_cat,
        "target_cluster_hint": cluster_name,
        "target_cluster_id":   cluster_id,
        "matched_query":       query,
        "source_confidence":   0.8,
        "taxonomy_level":      "second_category",
    }


# ── Per-cluster scrape ────────────────────────────────────────────────────────

def _build_queries(row: pd.Series) -> list[str]:
    """Build search query list for one cluster (cluster name + top seed keywords)."""
    queries: list[str] = [str(row["target_cluster"])]

    # Add aliases (pipe-separated)
    if pd.notna(row.get("aliases")) and str(row["aliases"]).strip():
        for alias in str(row["aliases"]).split("|")[:2]:
            alias = alias.strip()
            if alias and alias not in queries:
                queries.append(alias)

    # Add seed keywords (pipe-separated), skip ones identical to cluster name
    if pd.notna(row.get("seed_keywords")) and str(row["seed_keywords"]).strip():
        cluster_lower = str(row["target_cluster"]).lower()
        for kw in str(row["seed_keywords"]).split("|"):
            kw = kw.strip()
            if kw and kw.lower() != cluster_lower and kw not in queries:
                queries.append(kw)
                if len(queries) >= MAX_QUERIES:
                    break

    return queries[:MAX_QUERIES]


def scrape_cluster(
    row:          pd.Series,
    existing_ids: set[str],
    dry_run:      bool = False,
    limit:        int  = MAX_PER_QUERY,
) -> list[dict]:
    """Scrape Reddit for one target cluster. Modifies existing_ids in-place."""
    cluster_id   = str(row["target_cluster_id"])
    cluster_name = str(row["target_cluster"])
    parent_cat   = str(row.get("parent_category") or cluster_name)
    queries      = _build_queries(row)

    if dry_run:
        log.info("[DRY-RUN] %s  %-40s  queries: %s",
                 cluster_id, cluster_name, queries)
        return []

    records: list[dict] = []
    for query in queries:
        after: str | None = None
        for _page in range(MAX_PAGES):
            children, after = reddit_search(query, limit=limit, after=after)
            if not children:
                break
            for child in children:
                rec = child_to_record(child, cluster_id, cluster_name, query, parent_cat)
                if rec and rec["mention_id"] and rec["mention_id"] not in existing_ids:
                    existing_ids.add(rec["mention_id"])
                    records.append(rec)
            time.sleep(REQUEST_DELAY)
            if not after:
                break

    log.info("  %s  %-40s  +%d posts (%d queries)",
             cluster_id, cluster_name, len(records), len(queries))
    return records


# ── CSV append ────────────────────────────────────────────────────────────────

def append_to_csv(records: list[dict]) -> None:
    if not records:
        return
    write_header = not RAW_CSV.exists()
    with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-aware Reddit scraping")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print queries without making any requests")
    parser.add_argument("--cluster",  nargs="+", metavar="ID",
                        help="Only run specific cluster IDs, e.g. C001 C005")
    parser.add_argument("--limit",    type=int, default=MAX_PER_QUERY,
                        help=f"Posts per API page (default {MAX_PER_QUERY})")
    parser.add_argument("--reset",    action="store_true",
                        help="Clear checkpoint and start from scratch")
    args = parser.parse_args()

    taxonomy = pd.read_csv(TAXONOMY_CSV)
    log.info("Taxonomy: %d target clusters", len(taxonomy))

    cp = load_checkpoint()
    if args.reset:
        cp = {"done_clusters": [], "total_collected": 0}
        log.info("Checkpoint cleared.")

    # Load existing mention IDs to avoid duplicates
    log.info("Loading existing mention IDs …")
    existing_ids: set[str] = set()
    if RAW_CSV.exists():
        existing_ids = set(
            pd.read_csv(RAW_CSV, usecols=["mention_id"], low_memory=False)
            ["mention_id"].dropna().astype(str)
        )
    log.info("Existing unique posts: %d", len(existing_ids))

    done_set   = set(cp.get("done_clusters", []))
    total_new  = 0

    for _, row in taxonomy.iterrows():
        cid = str(row["target_cluster_id"])
        if args.cluster and cid not in args.cluster:
            continue
        if cid in done_set and not args.cluster:
            log.debug("Skip %s (already in checkpoint)", cid)
            continue

        records = scrape_cluster(row, existing_ids, dry_run=args.dry_run, limit=args.limit)

        if not args.dry_run:
            append_to_csv(records)
            total_new += len(records)
            done_set.add(cid)
            cp["done_clusters"]   = list(done_set)
            cp["total_collected"] = cp.get("total_collected", 0) + len(records)
            save_checkpoint(cp)

    log.info("=" * 60)
    log.info("Scrape complete.  New posts collected: %d", total_new)
    log.info("Clusters done: %d / %d", len(done_set), len(taxonomy))


if __name__ == "__main__":
    main()
