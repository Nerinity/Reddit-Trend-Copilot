#!/usr/bin/env python3
"""
Build a coverage report for the 171 target clusters.

Reports per cluster:
  - mention count (how many posts have been assigned to this cluster)
  - community coverage (how many subreddits contain posts for this cluster)
  - average target_cluster_score (how confident are the assignments)
  - top 5 subreddits contributing to this cluster
  - coverage status: OK / thin / missing

Output:
    data/processed/target_cluster_coverage.csv

Usage:
    python scripts/build_cluster_coverage_report.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("cluster_coverage")

_ROOT         = Path(__file__).parent.parent
TAXONOMY_CSV  = _ROOT / "configs" / "taxonomy" / "target_clusters_171.csv"
F_CLUSTER_PKL = _ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
F_OUTPUT      = _ROOT / "data" / "processed" / "target_cluster_coverage.csv"

MIN_COVERAGE = 100   # clusters below this count are flagged as "thin"


def main() -> None:
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    log.info("Taxonomy: %d target clusters", len(taxonomy))

    # ── Load parquet ──────────────────────────────────────────────────────────
    log.info("Loading processed parquet …")
    import pyarrow.parquet as pq
    available = pq.read_schema(F_CLUSTER_PKL).names
    load_cols = [c for c in
                 ["mention_id", "community", "published_at",
                  "target_cluster", "target_cluster_id", "target_cluster_score"]
                 if c in available]
    df = pd.read_parquet(F_CLUSTER_PKL, columns=load_cols)
    log.info("Rows loaded: %d", len(df))

    if "target_cluster" not in df.columns:
        log.warning("target_cluster column not found. "
                    "Run scripts/assign_target_clusters.py first.")
        df["target_cluster"]       = "unassigned"
        df["target_cluster_id"]    = "C000"
        df["target_cluster_score"] = 0.0

    # Exclude unassigned rows from stats (they inflate counts without adding signal)
    assigned = df[df["target_cluster"] != "unassigned"].copy()
    log.info("Posts with assignment: %d / %d  (%.1f%%)",
             len(assigned), len(df), 100 * len(assigned) / len(df))

    # ── Per-cluster mention counts ────────────────────────────────────────────
    mention_cnt = (
        assigned
        .groupby("target_cluster_id", as_index=False)
        .agg(
            mentions        = ("mention_id",           "count"),
            communities     = ("community",            "nunique"),
            avg_score       = ("target_cluster_score", "mean"),
        )
    )

    # ── Top subreddits per cluster ────────────────────────────────────────────
    sub_counts = (
        assigned
        .groupby(["target_cluster_id", "community"])
        .size()
        .reset_index(name="sub_count")
    )
    top_subs = (
        sub_counts
        .sort_values(["target_cluster_id", "sub_count"], ascending=[True, False])
        .groupby("target_cluster_id")["community"]
        .apply(lambda x: ", ".join(x.head(5).tolist()))
        .reset_index(name="top_subreddits")
    )

    # ── Weekly momentum (posts in last 2 weeks) ───────────────────────────────
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"].astype(str), utc=True, errors="coerce")
        assigned = assigned.copy()
        assigned["published_at"] = pd.to_datetime(assigned["published_at"].astype(str), utc=True, errors="coerce")
        latest = df["published_at"].max()
        cutoff = latest - pd.Timedelta(days=14)
        recent = assigned[assigned["published_at"] >= cutoff]
        recent_cnt = (
            recent
            .groupby("target_cluster_id")
            .size()
            .reset_index(name="mentions_last_2w")
        )
    else:
        recent_cnt = pd.DataFrame(columns=["target_cluster_id", "mentions_last_2w"])

    # ── Assemble report ───────────────────────────────────────────────────────
    report = (
        taxonomy[["target_cluster_id", "target_cluster", "parent_category", "priority"]]
        .merge(mention_cnt,  on="target_cluster_id", how="left")
        .merge(top_subs,     on="target_cluster_id", how="left")
        .merge(recent_cnt,   on="target_cluster_id", how="left")
    )
    report["mentions"]          = report["mentions"].fillna(0).astype(int)
    report["communities"]       = report["communities"].fillna(0).astype(int)
    report["avg_score"]         = report["avg_score"].fillna(0.0).round(3)
    report["mentions_last_2w"]  = report["mentions_last_2w"].fillna(0).astype(int)
    report["top_subreddits"]    = report["top_subreddits"].fillna("")
    report["status"] = report["mentions"].apply(
        lambda m: "OK" if m >= MIN_COVERAGE else ("thin" if m > 0 else "missing")
    )
    report = report.sort_values("mentions", ascending=False).reset_index(drop=True)

    report.to_csv(F_OUTPUT, index=False)
    log.info("Coverage report saved → %s", F_OUTPUT)

    # ── Print summary ─────────────────────────────────────────────────────────
    ok      = (report["status"] == "OK").sum()
    thin    = (report["status"] == "thin").sum()
    missing = (report["status"] == "missing").sum()
    log.info("")
    log.info("=" * 60)
    log.info("COVERAGE SUMMARY  (threshold: %d mentions)", MIN_COVERAGE)
    log.info("  %-8s %3d clusters  (well covered)", "OK:", ok)
    log.info("  %-8s %3d clusters  (1 – %d mentions)", "Thin:", thin, MIN_COVERAGE - 1)
    log.info("  %-8s %3d clusters  (0 mentions — needs scraping)", "Missing:", missing)
    log.info("=" * 60)
    log.info("")

    under = report[report["status"] != "OK"].sort_values("mentions")
    if not under.empty:
        log.info("Under-covered clusters (run scrape_by_target_clusters.py for these):")
        for _, r in under.iterrows():
            log.info("  %s  %-42s  %4d mentions  [%s]",
                     r["target_cluster_id"], r["target_cluster"],
                     r["mentions"], r["status"])
    else:
        log.info("All 171 clusters have ≥ %d mentions. Coverage looks good!", MIN_COVERAGE)


if __name__ == "__main__":
    main()
