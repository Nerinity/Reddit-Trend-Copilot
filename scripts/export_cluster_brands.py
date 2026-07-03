#!/usr/bin/env python3
"""Export per-cluster brand mentions and trend scores to CSV.

Output columns
--------------
cluster_id        HDBSCAN cluster number
top_category      most common category label in this cluster
cluster_size_cur  posts in current biweekly window
brand             extracted brand name
cur_mentions      mentions in current window
prev_mentions     mentions in previous window
brand_spike       cur / prev (1.0 = flat)
avg_sentiment     mean VADER compound score for posts that mention this brand
communities       unique subreddits in which this brand appears (current window)
sentiment_score   (avg_sentiment + 1) / 2  → 0-1
norm_spike        brand_spike / 97th-pct spike within cluster → 0-1
comm_score        communities / max communities in cluster → 0-1
brand_trend_score 0.40*norm_spike + 0.35*sentiment_score + 0.25*comm_score
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ── Allow importing from scripts/ sibling ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from build_dashboard_500k import (
    load_whitelist,
    extract_brands_from_post,
    _norm,
    PRODUCT_TERMS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLUSTERED  = Path("data/processed/nlp_clustered_500k.parquet")
OUTPUT_CSV = Path("data/exports/cluster_brands.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Biweekly window: use latest 14 days as cur, prior 14 days as prev
MIN_CLUSTER_POSTS = 10   # skip tiny clusters with few cur-period posts
MIN_BRAND_MENTIONS = 2   # brand must appear >= N times in cluster to be kept


def main() -> None:
    log.info("Loading clustered parquet …")
    df = pd.read_parquet(CLUSTERED)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["ner_input"]    = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df["category"]     = df["category"].fillna("unknown")

    latest     = df["published_at"].max()
    cur_start  = latest - pd.Timedelta(days=14)
    prev_start = cur_start - pd.Timedelta(days=14)

    df_cur  = df[df["published_at"] >= cur_start].copy()
    df_prev = df[(df["published_at"] >= prev_start) & (df["published_at"] < cur_start)].copy()
    log.info("cur=%d  prev=%d", len(df_cur), len(df_prev))

    log.info("Loading whitelist …")
    whitelist = load_whitelist()

    # ── Per-cluster brand extraction ─────────────────────────────────────────
    valid_clusters = (
        df_cur[df_cur["cluster_id"] != -1]
        .groupby("cluster_id")
        .size()
        .loc[lambda s: s >= MIN_CLUSTER_POSTS]
        .index.tolist()
    )
    log.info("%d valid clusters (≥%d cur posts, excl. noise)", len(valid_clusters), MIN_CLUSTER_POSTS)

    records = []

    for cid in valid_clusters:
        cur_rows  = df_cur[df_cur["cluster_id"] == cid]
        prev_rows = df_prev[df_prev["cluster_id"] == cid]

        top_cat = cur_rows["category"].mode().iloc[0] if not cur_rows.empty else "unknown"
        n_cur   = len(cur_rows)

        # Brand extraction — current period
        brand_cur_cnt:   Counter          = Counter()
        brand_sentiments: dict[str, list] = defaultdict(list)
        brand_comms:      dict[str, set]  = defaultdict(set)

        for row in cur_rows.itertuples():
            matches = extract_brands_from_post(row.ner_input, whitelist)
            for m in matches:
                brand_cur_cnt[m.brand]          += 1
                brand_sentiments[m.brand].append(row.sentiment_compound)
                brand_comms[m.brand].add(row.community)

        # Brand extraction — previous period (mentions only)
        brand_prev_cnt: Counter = Counter()
        for row in prev_rows.itertuples():
            matches = extract_brands_from_post(row.ner_input, whitelist)
            for m in matches:
                brand_prev_cnt[m.brand] += 1

        # Filter and aggregate
        raw = []
        for brand, cc in brand_cur_cnt.items():
            if cc < MIN_BRAND_MENTIONS:
                continue
            pc       = brand_prev_cnt.get(brand, 0)
            sents    = brand_sentiments[brand]
            raw.append({
                "cluster_id":    cid,
                "top_category":  top_cat,
                "cluster_size_cur": n_cur,
                "brand":         brand,
                "cur_mentions":  cc,
                "prev_mentions": pc,
                "brand_spike":   round(cc / max(pc, 1), 3),
                "avg_sentiment": round(sum(sents) / len(sents), 3) if sents else 0.0,
                "communities":   len(brand_comms[brand]),
            })

        if not raw:
            continue

        bdf = pd.DataFrame(raw)

        # Normalise within cluster for trend score
        p97_spike = bdf["brand_spike"].quantile(0.97) or 1.0
        bdf["norm_spike"]     = (bdf["brand_spike"] / max(p97_spike, 1)).clip(upper=1.0).round(4)
        bdf["sentiment_score"] = ((bdf["avg_sentiment"] + 1) / 2).round(4)
        max_comm               = bdf["communities"].max() or 1
        bdf["comm_score"]      = (bdf["communities"] / max_comm).round(4)
        bdf["brand_trend_score"] = (
            0.40 * bdf["norm_spike"] +
            0.35 * bdf["sentiment_score"] +
            0.25 * bdf["comm_score"]
        ).round(4)

        records.append(bdf)

    if not records:
        log.warning("No records found.")
        return

    out = (
        pd.concat(records, ignore_index=True)
        .sort_values(["cluster_id", "brand_trend_score"], ascending=[True, False])
    )

    out.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved → %s  (%d rows, %d clusters)", OUTPUT_CSV, len(out), out["cluster_id"].nunique())
    print(f"\nPreview:\n{out.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
