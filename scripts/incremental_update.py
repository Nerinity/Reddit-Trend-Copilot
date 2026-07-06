#!/usr/bin/env python3
"""
Incremental pipeline update — adds only new posts to existing processed data.

What it does (in order):
  1. Detect new mention_ids in raw CSV not yet in nlp_clustered_500k.parquet
  2. VADER sentiment on new rows only
  3. SentenceTransformer encode new rows only (384-dim)
  4. Assign cluster_id via nearest-neighbor (top-5 majority vote from saved embeddings)
  5. Append new rows to nlp_sentiment_500k.parquet + nlp_clustered_500k.parquet
  6. Append new embeddings to embeddings_500k.npy + embeddings_500k_ids.txt
  7. Re-run build_dashboard_500k.py
  8. Re-run build_forecast.py

Usage:
  python scripts/incremental_update.py
  python scripts/incremental_update.py --skip-dashboard   # only update parquets
  python scripts/incremental_update.py --dry-run          # count new rows, no writes
"""
from __future__ import annotations

import argparse
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("incremental_update")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

RAW_CSV        = _ROOT / "data" / "raw"       / "scraped_2026_large.csv"
F_SENTIMENT    = _ROOT / "data" / "processed" / "nlp_sentiment_500k.parquet"
F_CLUSTER      = _ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
F_EMB_NPY      = _ROOT / "data" / "processed" / "embeddings_500k.npy"
F_EMB_IDS      = _ROOT / "data" / "processed" / "embeddings_500k_ids.txt"
F_HDBSCAN_MDL  = _ROOT / "data" / "processed" / "hdbscan_model.pkl"

DASHBOARD_SCRIPT        = _ROOT / "scripts" / "build_dashboard_500k.py"
FORECAST_SCRIPT         = _ROOT / "scripts" / "build_forecast.py"
ASSIGN_CLUSTERS_SCRIPT  = _ROOT / "scripts" / "assign_target_clusters.py"

EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_BATCH   = 128
KNN_K         = 5     # neighbours to vote on cluster assignment


# ── Step 1: Find new rows ─────────────────────────────────────────────────────

def find_new_rows() -> pd.DataFrame:
    log.info("Reading existing parquet IDs …")
    existing_ids = set(
        pd.read_parquet(F_CLUSTER, columns=["mention_id"])["mention_id"].astype(str)
    )
    log.info("Existing parquet: %d rows", len(existing_ids))

    log.info("Reading raw CSV …")
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    raw["mention_id"] = raw["mention_id"].astype(str)
    new = raw[~raw["mention_id"].isin(existing_ids)].copy()
    log.info("New rows to process: %d", len(new))
    return new


# ── Step 2: VADER sentiment ───────────────────────────────────────────────────

def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    log.info("Running VADER on %d rows …", len(df))
    sia = SentimentIntensityAnalyzer()
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["ner_input"] = (df["title"] + ". " + df["text"].str[:400]).str.strip()

    texts = (df["title"] + " " + df["text"].str[:500]).tolist()
    comps, pos, neg, neu = [], [], [], []
    t0 = time.time()
    for i, txt in enumerate(texts):
        sc = sia.polarity_scores(str(txt))
        comps.append(sc["compound"])
        pos.append(sc["pos"])
        neg.append(sc["neg"])
        neu.append(sc["neu"])
        if (i + 1) % 20_000 == 0:
            log.info("  VADER %d/%d (%.0f/s)", i+1, len(texts), (i+1)/(time.time()-t0))

    df["sentiment_compound"] = comps
    df["sentiment_positive"]  = pos
    df["sentiment_negative"]  = neg
    df["sentiment_neutral"]   = neu
    df["sentiment_label"] = df["sentiment_compound"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    log.info("VADER done in %.1fs", time.time()-t0)
    return df


# ── Step 3: SentenceTransformer encode ───────────────────────────────────────

def encode_new(df: pd.DataFrame) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    log.info("Loading SentenceTransformer: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    texts = (df["title"] + " " + df["text"].str[:400]).fillna("").tolist()
    log.info("Encoding %d texts (batch_size=%d) …", len(texts), EMBED_BATCH)
    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    log.info("Encoding done in %.1fmin", (time.time()-t0)/60)
    return vecs


# ── Step 4: Nearest-neighbour cluster assignment ──────────────────────────────

def assign_clusters(new_vecs: np.ndarray) -> np.ndarray:
    """
    For each new embedding, find KNN_K nearest neighbours in the existing
    embedding bank and assign the majority cluster_id.
    Falls back to cluster -1 (noise) when majority is -1 or tie.
    """
    log.info("Loading existing embeddings (%s) …", F_EMB_NPY)
    old_vecs = np.load(F_EMB_NPY)               # (N_old, 384)

    log.info("Loading existing cluster labels …")
    old_ids   = F_EMB_IDS.read_text().strip().split("\n")
    label_map = dict(
        zip(
            pd.read_parquet(F_CLUSTER, columns=["mention_id"])["mention_id"].astype(str),
            pd.read_parquet(F_CLUSTER, columns=["cluster_id"])["cluster_id"],
        )
    )
    old_labels = np.array([label_map.get(mid, -1) for mid in old_ids], dtype=np.int32)

    log.info("KNN assignment: %d new × %d existing …", len(new_vecs), len(old_vecs))
    t0 = time.time()

    # Cosine similarity via dot product (embeddings are already L2-normalised)
    assigned = np.full(len(new_vecs), -1, dtype=np.int32)
    CHUNK = 500  # process in chunks to avoid OOM
    for start in range(0, len(new_vecs), CHUNK):
        chunk = new_vecs[start : start + CHUNK]          # (chunk, 384)
        sims  = chunk @ old_vecs.T                        # (chunk, N_old)
        topk_idx = np.argpartition(sims, -KNN_K, axis=1)[:, -KNN_K:]  # (chunk, K)
        for i, nbrs in enumerate(topk_idx):
            nbr_labels = old_labels[nbrs]
            # Majority vote, ignoring -1 noise
            valid = nbr_labels[nbr_labels >= 0]
            if len(valid) == 0:
                assigned[start + i] = -1
            else:
                counts = np.bincount(valid)
                assigned[start + i] = int(np.argmax(counts))

    log.info("KNN assignment done in %.1fs", time.time()-t0)
    noise_pct = 100 * (assigned == -1).mean()
    log.info("Noise (unassigned): %.1f%%", noise_pct)
    return assigned


# ── Step 5 & 6: Append to parquets + npy ─────────────────────────────────────

def append_to_parquets(df: pd.DataFrame, cluster_ids: np.ndarray, new_vecs: np.ndarray) -> None:
    df = df.copy()
    df["cluster_id"] = cluster_ids

    # Align columns to existing parquet schemas
    sent_cols   = pd.read_parquet(F_SENTIMENT,  columns=[]).columns.tolist()
    cluster_cols = pd.read_parquet(F_CLUSTER, columns=[]).columns.tolist()

    def _align(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for c in cols:
            if c not in frame.columns:
                frame[c] = None
        return frame[cols]

    log.info("Appending %d rows to sentiment parquet …", len(df))
    old_sent = pd.read_parquet(F_SENTIMENT)
    new_sent = pd.concat([old_sent, _align(df, sent_cols)], ignore_index=True)
    new_sent.to_parquet(F_SENTIMENT, index=False)

    log.info("Appending %d rows to cluster parquet …", len(df))
    old_clust = pd.read_parquet(F_CLUSTER)
    new_clust = pd.concat([old_clust, _align(df, cluster_cols)], ignore_index=True)
    new_clust.to_parquet(F_CLUSTER, index=False)

    log.info("Appending embeddings to .npy …")
    old_vecs = np.load(F_EMB_NPY)
    combined = np.vstack([old_vecs, new_vecs])
    np.save(F_EMB_NPY, combined)

    old_ids = F_EMB_IDS.read_text().strip().split("\n")
    new_ids = df["mention_id"].astype(str).tolist()
    F_EMB_IDS.write_text("\n".join(old_ids + new_ids))

    log.info("Parquets and embeddings updated.")
    log.info("  sentiment: %d rows", len(new_sent))
    log.info("  cluster:   %d rows", len(new_clust))
    log.info("  embeddings: %s", combined.shape)


# ── Step 7 & 8: Rebuild dashboard + forecast ─────────────────────────────────

def run_script(script: Path, label: str) -> None:
    log.info("Running %s …", label)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("%s failed (exit %d)", label, result.returncode)
        sys.exit(result.returncode)
    log.info("%s done in %.1fmin", label, (time.time()-t0)/60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental pipeline update")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip dashboard + forecast rebuild (just update parquets)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count new rows, no writes")
    args = parser.parse_args()

    t_total = time.time()

    # 1. Find new rows
    new_df = find_new_rows()
    if new_df.empty:
        log.info("No new rows — everything is up to date.")
        return

    if args.dry_run:
        log.info("DRY RUN — would process %d new rows. Exiting.", len(new_df))
        return

    # 2. VADER
    new_df = add_sentiment(new_df)

    # 3. Encode
    new_vecs = encode_new(new_df)

    # 4. Cluster assignment
    cluster_ids = assign_clusters(new_vecs)

    # 5 & 6. Append
    append_to_parquets(new_df, cluster_ids, new_vecs)

    # 7. Assign target clusters to all posts (incremental: only new embeddings change)
    run_script(ASSIGN_CLUSTERS_SCRIPT, "assign_target_clusters")

    if not args.skip_dashboard:
        # 8. Dashboard
        run_script(DASHBOARD_SCRIPT, "build_dashboard_500k")
        # 9. Forecast
        run_script(FORECAST_SCRIPT, "build_forecast")

    log.info("=" * 60)
    log.info("Incremental update complete in %.1fmin", (time.time()-t_total)/60)
    log.info("New rows added: %d", len(new_df))


if __name__ == "__main__":
    main()
