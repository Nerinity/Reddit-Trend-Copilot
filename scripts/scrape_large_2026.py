#!/usr/bin/env python3
"""
Large-scale 2026 data collection script.

Targets 100k+ records from public forum/news sources for 2026.
Uses:
  - Reddit via pullpush.io (no auth needed) across 200+ subreddits
  - HackerNews Algolia API (no auth)
  - Google News RSS (no auth)
  - Reddit public JSON RSS feeds

Run:
  python scripts/scrape_large_2026.py
  python scripts/scrape_large_2026.py --target 150000 --start 2026-01-01 --end 2026-06-25
  python scripts/scrape_large_2026.py --sources reddit,hn,rss
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("scrape_large_2026")

# ── Constants ────────────────────────────────────────────────────────────────

OUT_FILE = _ROOT / "data" / "raw" / "scraped_2026_large.csv"
CHECKPOINT_FILE = _ROOT / "data" / "state" / "scrape_large_2026_checkpoint.json"
USER_AGENT = "TrendCopilotBot/2.0 (research project; contact via GitHub)"
DEFAULT_TARGET = 500_000
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
HN_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
REDDIT_JSON_BASE = "https://www.reddit.com"

# Load subreddit categories from sources.json
_SOURCES_JSON = _ROOT / "configs" / "sources.json"
try:
    with open(_SOURCES_JSON) as _f:
        _cfg = json.load(_f)
    SUBREDDIT_CATEGORIES: dict[str, list[str]] = _cfg["reddit"]["subreddit_categories"]
    log.info("Loaded %d subreddit groups (%d total subs) from sources.json",
             len(SUBREDDIT_CATEGORIES),
             sum(len(v) for v in SUBREDDIT_CATEGORIES.values()))
except Exception as _e:
    log.warning("Could not load sources.json (%s), using empty subreddit list", _e)
    SUBREDDIT_CATEGORIES = {}

HN_QUERIES = [
    "AI wearable", "consumer AI", "TikTok Shop", "creator economy", "social commerce",
    "ecommerce", "personalized shopping", "recommendation engine", "smart ring",
    "health tracking", "product hunt", "startup", "marketplace", "skincare",
    "beauty product", "wellness", "fitness tracker", "viral product", "Amazon",
    "live shopping", "affiliate marketing", "creator storefront",
]

RSS_QUERIES = [
    "viral TikTok product 2026", "TikTok Shop trend 2026", "Amazon viral products 2026",
    "beauty product trend 2026", "wellness product trend 2026", "creator commerce 2026",
    "social commerce trend 2026", "emerging consumer trend 2026",
    "Pinterest trends products 2026", "YouTube shopping trend 2026",
    "skincare trend 2026", "fashion trend 2026", "home decor trend 2026",
    "fitness product 2026", "pet product trend 2026",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stable_hash(*parts: object) -> str:
    joined = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:24]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_to_unix(value: str, end: bool = False) -> int:
    suffix = "T23:59:59+00:00" if end else "T00:00:00+00:00"
    return int(datetime.fromisoformat(value + suffix).timestamp())


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2))


def _load_existing_ids() -> set[str]:
    if OUT_FILE.exists():
        try:
            df = pd.read_csv(OUT_FILE, usecols=["mention_id"], low_memory=False)
            return set(df["mention_id"].dropna().astype(str))
        except Exception:
            pass
    return set()


def _append_records(records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_FILE.exists()
    df.to_csv(OUT_FILE, mode="a", header=write_header, index=False)


def _make_record(
    *,
    source: str,
    platform: str,
    sub_source: str,
    source_type: str,
    keyword: str,
    query: str,
    category: str,
    title: str,
    text: str,
    author: str,
    community: str,
    published_at: str,
    url: str,
    engagement_score: float = 0.0,
    metrics: dict | None = None,
) -> dict:
    full_text = f"{title} {text}".strip()
    mention_id = _stable_hash(source, platform, keyword, title, text, url)
    return {
        "mention_id": mention_id,
        "source": source,
        "platform": platform,
        "sub_source": sub_source,
        "source_type": source_type,
        "keyword": keyword,
        "query": query,
        "category": category,
        "title": title,
        "text": text,
        "full_text": full_text,
        "author": author,
        "community": community,
        "published_at": published_at,
        "collected_at": _now_utc(),
        "url": url,
        "engagement_score": engagement_score,
        "semantic_relevance_score": 0.0,
        "tiktok_relevance_score": 0.0,
        "business_context_label": "",
        "collector_priority_score": 0.0,
        "metrics_json": json.dumps(metrics or {}, ensure_ascii=False),
    }


# ── Reddit via arctic-shift (open Reddit archive, has 2026 data) ─────────────

def scrape_reddit_arctic(
    start_date: str,
    end_date: str,
    existing_ids: set[str],
    checkpoint: dict,
    target: int = DEFAULT_TARGET,
    per_subreddit: int = 1000,
    request_delay: float = 1.0,
) -> list[dict]:
    """
    Scrape Reddit posts via arctic-shift.photon-reddit.com (no auth, has 2026 data).
    Paginates by walking back `before` timestamp.
    """
    records: list[dict] = []
    headers = {"User-Agent": USER_AGENT}
    done_subs = set(checkpoint.get("done_arctic_subreddits", []))

    # Priority ordering — keys must match sources.json reddit.subreddit_categories
    subreddit_order: list[tuple[str, str]] = []
    priority_cats = [
        "creator_commerce", "deals_shopping_reviews",
        "beauty_skincare", "supplements_nutrition",
        "womens_fashion", "mens_fashion", "fashion_accessories",
    ]
    for cat in priority_cats:
        for sub in SUBREDDIT_CATEGORIES.get(cat, []):
            subreddit_order.append((cat, sub))
    for cat, subs in SUBREDDIT_CATEGORIES.items():
        if cat not in priority_cats:
            for sub in subs:
                subreddit_order.append((cat, sub))
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for item in subreddit_order:
        if item[1] not in seen:
            seen.add(item[1])
            deduped.append(item)
    subreddit_order = deduped

    cutoff_ts = _date_to_unix(start_date)

    for category, subreddit in subreddit_order:
        if len(existing_ids) >= target:
            break
        if subreddit in done_subs:
            continue

        sub_count = 0
        before = end_date
        pages = 0

        while sub_count < per_subreddit and pages < 20:
            params = {
                "subreddit": subreddit,
                "limit": 100,
                "after": start_date,
                "before": before,
                "sort": "desc",
            }
            try:
                resp = requests.get(
                    f"{ARCTIC_SHIFT_BASE}/posts/search",
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 429:
                    log.warning("arctic-shift rate limited on r/%s, waiting 30s", subreddit)
                    time.sleep(30)
                    continue
                if resp.status_code != 200:
                    log.debug("arctic-shift HTTP %s for r/%s", resp.status_code, subreddit)
                    break
                rows = resp.json().get("data", [])
            except Exception as exc:
                log.debug("arctic-shift failed r/%s: %s", subreddit, exc)
                break
            if not rows:
                break

            oldest_ts = None
            for row in rows:
                created = row.get("created_utc") or 0
                if created and created < cutoff_ts:
                    continue
                pub = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else ""
                title = row.get("title", "") or ""
                text = (row.get("selftext", "") or "").strip()
                if text in ("[deleted]", "[removed]"):
                    text = ""
                permalink = row.get("permalink", "")
                url_val = f"https://reddit.com{permalink}" if permalink else ""
                engagement = (row.get("score") or 0) + (row.get("num_comments") or 0) * 3
                rec = _make_record(
                    source="arctic_shift_reddit",
                    platform="reddit",
                    sub_source=f"r/{subreddit}",
                    source_type="submission",
                    keyword=f"r/{subreddit}",
                    query=f"r/{subreddit} 2026",
                    category=category,
                    title=title,
                    text=text,
                    author=row.get("author", "[deleted]"),
                    community=subreddit,
                    published_at=pub,
                    url=url_val,
                    engagement_score=float(engagement),
                    metrics={"score": row.get("score", 0), "num_comments": row.get("num_comments", 0)},
                )
                if rec["mention_id"] not in existing_ids:
                    records.append(rec)
                    existing_ids.add(rec["mention_id"])
                    sub_count += 1
                if oldest_ts is None or (created and created < oldest_ts):
                    oldest_ts = created

            pages += 1
            if len(rows) < 100 or oldest_ts is None:
                break
            # Move `before` to oldest post date for next page
            before = datetime.fromtimestamp(oldest_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            time.sleep(request_delay)

        done_subs.add(subreddit)
        checkpoint["done_arctic_subreddits"] = list(done_subs)
        if sub_count > 0:
            log.info("r/%s [%s]: +%d | total: %d", subreddit, category, sub_count, len(existing_ids))

        # Flush every 2000 records
        if len(records) % 2000 < sub_count + 1 and records:
            _append_records(records[-min(2000, len(records)):])
            _save_checkpoint(checkpoint)

    if records:
        _append_records(records)
    log.info("arctic-shift Reddit scrape done: %d new records", len(records))
    return records


# ── Reddit public JSON API (no auth, real-time 2026 data) ───────────────────

def scrape_reddit_json(
    existing_ids: set[str],
    checkpoint: dict,
    target: int = DEFAULT_TARGET,
    per_subreddit: int = 500,
    sorts: list[str] | None = None,
    request_delay: float = 1.5,
) -> list[dict]:
    """
    Scrape Reddit via the public JSON API (www.reddit.com/r/{sub}/{sort}.json).
    No authentication required. Returns live 2026 data.
    """
    if sorts is None:
        sorts = ["new", "hot", "top"]
    records: list[dict] = []
    headers = {"User-Agent": USER_AGENT}
    done_subs = set(checkpoint.get("done_json_subreddits", []))
    cutoff_ts = _date_to_unix("2026-01-01")

    # Flatten + prioritize subreddits — keys must match sources.json
    subreddit_order: list[tuple[str, str]] = []
    priority_cats = [
        "creator_commerce", "deals_shopping_reviews",
        "beauty_skincare", "supplements_nutrition",
        "womens_fashion", "mens_fashion", "fashion_accessories",
    ]
    for cat in priority_cats:
        for sub in SUBREDDIT_CATEGORIES.get(cat, []):
            subreddit_order.append((cat, sub))
    for cat, subs in SUBREDDIT_CATEGORIES.items():
        if cat not in priority_cats:
            for sub in subs:
                subreddit_order.append((cat, sub))
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for item in subreddit_order:
        if item[1] not in seen:
            seen.add(item[1])
            deduped.append(item)
    subreddit_order = deduped

    total_collected = len(checkpoint.get("json_total", 0) and [] or [])

    for category, subreddit in subreddit_order:
        if len(records) + len(existing_ids) >= target:
            break
        if subreddit in done_subs:
            continue

        sub_count = 0
        for sort in sorts:
            if sub_count >= per_subreddit:
                break
            # "top" needs timeframe parameter
            url = f"{REDDIT_JSON_BASE}/r/{subreddit}/{sort}.json?limit=100"
            if sort == "top":
                url += "&t=year"
            after_token: str | None = None
            pages = 0
            while sub_count < per_subreddit and pages < 10:
                page_url = url + (f"&after={after_token}" if after_token else "")
                try:
                    resp = requests.get(page_url, headers=headers, timeout=30)
                    if resp.status_code == 429:
                        log.warning("Reddit rate limited on r/%s, waiting 60s", subreddit)
                        time.sleep(60)
                        continue
                    if resp.status_code in (403, 404):
                        break  # Private/deleted subreddit
                    if resp.status_code != 200:
                        log.debug("Reddit JSON HTTP %s for r/%s/%s", resp.status_code, subreddit, sort)
                        break
                    data = resp.json()
                    listing = data.get("data", {})
                    children = listing.get("children", [])
                    after_token = listing.get("after")
                except Exception as exc:
                    log.debug("Reddit JSON failed r/%s/%s: %s", subreddit, sort, exc)
                    break
                if not children:
                    break

                for child in children:
                    post = child.get("data", {})
                    created = post.get("created_utc") or 0
                    # Only keep 2026 data
                    if created and created < cutoff_ts:
                        continue
                    pub = datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else ""
                    title = post.get("title", "") or ""
                    text = post.get("selftext", "") or ""
                    if text in ("[deleted]", "[removed]"):
                        text = ""
                    permalink = post.get("permalink", "")
                    url_val = f"https://reddit.com{permalink}" if permalink else ""
                    engagement = (post.get("score") or 0) + (post.get("num_comments") or 0) * 3
                    metrics = {
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0),
                    }
                    rec = _make_record(
                        source="reddit_json",
                        platform="reddit",
                        sub_source=f"r/{subreddit}",
                        source_type="submission",
                        keyword=f"r/{subreddit}",
                        query=f"r/{subreddit} {sort} 2026",
                        category=category,
                        title=title,
                        text=text,
                        author=post.get("author", "[deleted]"),
                        community=subreddit,
                        published_at=pub,
                        url=url_val,
                        engagement_score=float(engagement),
                        metrics=metrics,
                    )
                    if rec["mention_id"] not in existing_ids:
                        records.append(rec)
                        existing_ids.add(rec["mention_id"])
                        sub_count += 1

                pages += 1
                if not after_token or len(children) < 100:
                    break
                time.sleep(request_delay)

        done_subs.add(subreddit)
        checkpoint["done_json_subreddits"] = list(done_subs)
        current_total = len(existing_ids)
        if sub_count > 0:
            log.info("r/%s [%s]: +%d | total: %d", subreddit, category, sub_count, current_total)

        # Checkpoint every 2000 records
        if len(records) % 2000 < sub_count and records:
            _append_records(records[-sub_count:] if sub_count <= len(records) else records)
            _save_checkpoint(checkpoint)

    log.info("Reddit JSON scrape done: %d new records", len(records))
    return records


# ── Reddit public RSS (no pullpush fallback) ─────────────────────────────────

def scrape_reddit_rss(
    existing_ids: set[str],
    checkpoint: dict,
    per_subreddit: int = 100,
    request_delay: float = 2.0,
) -> list[dict]:
    """Scrape Reddit via public Atom RSS (no API auth, no pullpush)."""
    records: list[dict] = []
    done = set(checkpoint.get("done_rss_subreddits", []))
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    sorts = ["new", "hot", "top/.rss?t=day", "top/.rss?t=week", "top/.rss?t=month", "top/.rss?t=year", "rising"]
    cutoff = _date_to_unix("2026-01-01")

    all_subs: list[tuple[str, str]] = []
    for cat, subs in SUBREDDIT_CATEGORIES.items():
        for sub in subs:
            all_subs.append((cat, sub))

    for category, subreddit in all_subs:
        if subreddit in done:
            continue
        for sort in sorts:
            if "/.rss?" in sort:
                url = f"https://www.reddit.com/r/{subreddit}/{sort}&limit=100"
            else:
                url = f"https://www.reddit.com/r/{subreddit}/{sort}/.rss?limit=100"
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
                if resp.status_code == 429:
                    time.sleep(60)
                    continue
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.content)
                entries = root.findall("atom:entry", ns) or root.findall(".//item")
                for entry in entries[:per_subreddit]:
                    title_el = entry.find("atom:title", ns) or entry.find("title")
                    content_el = entry.find("atom:content", ns) or entry.find("description")
                    link_el = entry.find("atom:link", ns) or entry.find("link")
                    date_el = entry.find("atom:updated", ns) or entry.find("pubDate")
                    author_el = entry.find("atom:author/atom:name", ns)
                    title = (title_el.text or "") if title_el is not None else ""
                    text = (content_el.text or "") if content_el is not None else ""
                    pub_raw = (date_el.text or "") if date_el is not None else ""
                    url_val = (link_el.get("href") or link_el.text or "") if link_el is not None else ""
                    author_val = (author_el.text or "") if author_el is not None else ""
                    # Parse date, filter for 2026
                    try:
                        pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                        pub_ts = int(pub_dt.timestamp())
                        if pub_ts < cutoff:
                            continue
                        pub_iso = pub_dt.isoformat()
                    except Exception:
                        pub_iso = pub_raw

                    rec = _make_record(
                        source="reddit_rss",
                        platform="reddit",
                        sub_source=f"r/{subreddit}",
                        source_type="submission",
                        keyword=f"r/{subreddit}",
                        query=f"r/{subreddit} {sort}",
                        category=category,
                        title=title,
                        text=text,
                        author=author_val,
                        community=subreddit,
                        published_at=pub_iso,
                        url=url_val,
                    )
                    if rec["mention_id"] not in existing_ids:
                        records.append(rec)
                        existing_ids.add(rec["mention_id"])
                time.sleep(request_delay)
            except Exception as exc:
                log.debug("RSS failed r/%s %s: %s", subreddit, sort, exc)
        done.add(subreddit)
        checkpoint["done_rss_subreddits"] = list(done)
        sub_new = sum(1 for r in records if r.get("sub_source") == f"r/{subreddit}")
        if sub_new > 0:
            log.info("r/%s: +%d | running total: %d", subreddit, sub_new, len(existing_ids))
        # Flush to disk every 500 records
        if len(records) >= 500 and len(records) % 500 < 25:
            _append_records(records[-500:])
            _save_checkpoint(checkpoint)

    # Final flush
    if records:
        _append_records(records)
    log.info("Reddit RSS: %d records collected", len(records))
    return records


# ── HackerNews via Algolia ───────────────────────────────────────────────────

def scrape_hackernews(
    start_date: str,
    end_date: str,
    existing_ids: set[str],
    checkpoint: dict,
    per_query: int = 1000,
    request_delay: float = 1.0,
) -> list[dict]:
    records: list[dict] = []
    start_ts = _date_to_unix(start_date)
    end_ts = _date_to_unix(end_date, end=True)
    done = set(checkpoint.get("done_hn_queries", []))

    for query in HN_QUERIES:
        if query in done:
            continue
        page = 0
        query_count = 0
        while query_count < per_query:
            params = {
                "query": query,
                "tags": "story,comment",
                "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
                "hitsPerPage": 200,
                "page": page,
            }
            try:
                resp = requests.get(HN_SEARCH, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                data = resp.json()
                hits = data.get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    object_id = hit.get("objectID", "")
                    title = hit.get("title") or hit.get("story_title") or ""
                    text = hit.get("comment_text") or hit.get("story_text") or ""
                    pub = hit.get("created_at", "")
                    url_val = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
                    engagement = (hit.get("points") or 0) + (hit.get("num_comments") or 0) * 3
                    rec = _make_record(
                        source="hackernews_algolia",
                        platform="forum",
                        sub_source="hackernews",
                        source_type="comment" if "comment" in hit.get("_tags", []) else "story",
                        keyword=query,
                        query=query,
                        category="tech_startup_consumer",
                        title=title,
                        text=text,
                        author=hit.get("author", ""),
                        community="hackernews",
                        published_at=pub,
                        url=url_val,
                        engagement_score=float(engagement),
                        metrics={"points": hit.get("points", 0), "num_comments": hit.get("num_comments", 0)},
                    )
                    if rec["mention_id"] not in existing_ids:
                        records.append(rec)
                        existing_ids.add(rec["mention_id"])
                        query_count += 1
                if len(hits) < 200 or page >= data.get("nbPages", 1) - 1:
                    break
                page += 1
                time.sleep(request_delay)
            except Exception as exc:
                log.warning("HN failed for %s: %s", query, exc)
                break
        done.add(query)
        checkpoint["done_hn_queries"] = list(done)
        if query_count > 0:
            log.info("HN [%s]: +%d | running total: %d", query, query_count, len(existing_ids))

    if records:
        _append_records(records)
    log.info("HackerNews: %d records collected", len(records))
    return records


# ── Google News RSS ──────────────────────────────────────────────────────────

def scrape_google_news_rss(
    start_date: str,
    end_date: str,
    existing_ids: set[str],
    checkpoint: dict,
    per_query: int = 100,
    request_delay: float = 2.0,
) -> list[dict]:
    records: list[dict] = []
    done = set(checkpoint.get("done_rss_queries", []))

    for query in RSS_QUERIES:
        if query in done:
            continue
        search = f'({query}) after:{start_date} before:{end_date}'
        url = "https://news.google.com/rss/search?q=" + urllib.parse.quote_plus(search) + "&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code != 200:
                log.warning("Google News RSS HTTP %s for: %s", resp.status_code, query)
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:per_query]:
                title = item.findtext("title") or ""
                text = item.findtext("description") or ""
                pub = item.findtext("pubDate") or ""
                link = item.findtext("link") or ""
                source = item.findtext("source") or ""
                rec = _make_record(
                    source="google_news_rss",
                    platform="news",
                    sub_source=source,
                    source_type="article_snippet",
                    keyword=query,
                    query=query,
                    category="market_news",
                    title=title,
                    text=text,
                    author=source,
                    community=source,
                    published_at=pub,
                    url=link,
                )
                if rec["mention_id"] not in existing_ids:
                    records.append(rec)
                    existing_ids.add(rec["mention_id"])
            time.sleep(request_delay)
        except Exception as exc:
            log.warning("Google News RSS failed for %s: %s", query, exc)
        done.add(query)
        checkpoint["done_rss_queries"] = list(done)

    log.info("Google News RSS: %d records collected", len(records))
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale 2026 trend data scraper")
    parser.add_argument("--start", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-29", help="End date (YYYY-MM-DD)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target record count")
    parser.add_argument("--sources", default="arctic,reddit_json",
                        help="Comma-separated sources: arctic, reddit_json, rss, hn, gnews")
    parser.add_argument("--per-sub", type=int, default=500, help="Max records per subreddit")
    parser.add_argument("--delay", type=float, default=1.5, help="Request delay in seconds")
    parser.add_argument("--no-comments", action="store_true", help="(unused, kept for compat)")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    sources = set(s.strip() for s in args.sources.split(","))
    checkpoint = {} if args.reset else _load_checkpoint()
    existing_ids = _load_existing_ids()

    log.info("Starting large-scale 2026 scrape | target=%d | sources=%s | existing=%d",
             args.target, sources, len(existing_ids))

    all_records: list[dict] = []

    if "arctic" in sources:
        log.info("── Scraping Reddit via arctic-shift (open archive, 2026 data) ──")
        recs = scrape_reddit_arctic(
            start_date=args.start,
            end_date=args.end,
            existing_ids=existing_ids,
            checkpoint=checkpoint,
            target=args.target,
            per_subreddit=args.per_sub,
            request_delay=args.delay,
        )
        all_records.extend(recs)
        _save_checkpoint(checkpoint)
        log.info("arctic-shift done: %d new records (total: %d)", len(recs), len(existing_ids))

    if "reddit_json" in sources:
        log.info("── Scraping Reddit via public JSON API (live 2026 data) ──")
        recs = scrape_reddit_json(
            existing_ids=existing_ids,
            checkpoint=checkpoint,
            target=args.target,
            per_subreddit=args.per_sub,
            sorts=["new", "hot", "top"],
            request_delay=args.delay,
        )
        all_records.extend(recs)
        if recs:
            _append_records(recs)
        _save_checkpoint(checkpoint)
        log.info("Reddit JSON done: %d new records (total so far: %d)", len(recs), len(existing_ids))

    if "rss" in sources and len(existing_ids) < args.target:
        log.info("── Scraping Reddit public RSS ──")
        recs = scrape_reddit_rss(
            existing_ids=existing_ids,
            checkpoint=checkpoint,
            request_delay=max(args.delay, 2.0),
        )
        all_records.extend(recs)
        if recs:
            _append_records(recs)
        _save_checkpoint(checkpoint)
        log.info("Reddit RSS done: %d new records", len(recs))

    if "hn" in sources and len(existing_ids) < args.target:
        log.info("── Scraping HackerNews ──")
        recs = scrape_hackernews(
            start_date=args.start,
            end_date=args.end,
            existing_ids=existing_ids,
            checkpoint=checkpoint,
            per_query=1000,
            request_delay=args.delay,
        )
        all_records.extend(recs)
        if recs:
            _append_records(recs)
        _save_checkpoint(checkpoint)
        log.info("HackerNews done: %d new records", len(recs))

    if "gnews" in sources and len(existing_ids) < args.target:
        log.info("── Scraping Google News RSS ──")
        recs = scrape_google_news_rss(
            start_date=args.start,
            end_date=args.end,
            existing_ids=existing_ids,
            checkpoint=checkpoint,
            request_delay=max(args.delay, 2.0),
        )
        all_records.extend(recs)
        if recs:
            _append_records(recs)
        _save_checkpoint(checkpoint)
        log.info("Google News RSS done: %d new records", len(recs))

    total = len(existing_ids)
    log.info("=" * 60)
    log.info("SCRAPE COMPLETE: %d total unique records in %s", total, OUT_FILE)
    if total < args.target:
        log.warning("Target %d not reached (%d). Re-run to continue (checkpoint saved).", args.target, total)
    else:
        log.info("Target %d reached!", args.target)


if __name__ == "__main__":
    main()
