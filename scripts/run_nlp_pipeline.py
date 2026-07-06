#!/usr/bin/env python3
"""
NLP Pipeline for scraped_2026_large.csv
Stages (each checkpointed, resumable):
  1. Load + deduplicate
  2. VADER sentiment
  3. SentenceTransformers embeddings (all-MiniLM-L6-v2, 384-dim)
  4. UMAP 384 -> 15 dims
  5. HDBSCAN clustering
  6. Save final parquet + npy

Usage:
  python scripts/run_nlp_pipeline.py
  python scripts/run_nlp_pipeline.py --resume          # skip completed stages
  python scripts/run_nlp_pipeline.py --stage sentiment  # run single stage
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("nlp_pipeline")

_ROOT   = Path(__file__).parent.parent
RAW_CSV = _ROOT / "data" / "raw"  / "scraped_2026_large.csv"
OUT_DIR = _ROOT / "data" / "processed"
STATE   = _ROOT / "data" / "state"

OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)

# Stage output files
F_CLEAN       = OUT_DIR / "nlp_clean_500k.parquet"
F_SENTIMENT   = OUT_DIR / "nlp_sentiment_500k.parquet"
F_EMB_NPY     = OUT_DIR / "embeddings_500k.npy"
F_EMB_IDS     = OUT_DIR / "embeddings_500k_ids.txt"
F_UMAP_NPY    = OUT_DIR / "umap_500k.npy"
F_CLUSTER     = OUT_DIR / "nlp_clustered_500k.parquet"
F_HDBSCAN_MDL = OUT_DIR / "hdbscan_model.pkl"   # saved for incremental updates

EMB_CKPT    = STATE / "emb_checkpoint.npy"
EMB_CKPT_N  = STATE / "emb_checkpoint_n.txt"


# ── Stage 1: Load & deduplicate ──────────────────────────────────────────────

def stage_clean():
    log.info("=== STAGE 1: Load + Deduplicate ===")
    log.info("Reading %s …", RAW_CSV)
    df = pd.read_csv(RAW_CSV, low_memory=False)
    log.info("Raw rows: %d", len(df))

    df = df.drop_duplicates(subset=["mention_id"])
    log.info("After dedup: %d", len(df))

    # Build NER input text
    df["title"]    = df["title"].fillna("")
    df["text"]     = df["text"].fillna("")
    df["ner_input"] = (df["title"] + ". " + df["text"].str[:400]).str.strip()

    # Drop obviously empty
    df = df[df["ner_input"].str.len() >= 10].copy()
    log.info("After length filter: %d", len(df))

    df.to_parquet(F_CLEAN, index=False)
    log.info("Saved -> %s", F_CLEAN)
    return df


# ── Stage 2: VADER sentiment ─────────────────────────────────────────────────

def stage_sentiment(df: pd.DataFrame | None = None):
    log.info("=== STAGE 2: VADER Sentiment ===")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    if df is None:
        df = pd.read_parquet(F_CLEAN)
    log.info("Rows to score: %d", len(df))

    sia = SentimentIntensityAnalyzer()
    t0 = time.time()

    texts = (df["title"] + " " + df["text"].str[:500]).tolist()
    compounds, positives, negatives, neutrals = [], [], [], []

    for i, txt in enumerate(texts):
        sc = sia.polarity_scores(str(txt))
        compounds.append(sc["compound"])
        positives.append(sc["pos"])
        negatives.append(sc["neg"])
        neutrals.append(sc["neu"])
        if (i + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            log.info("  Sentiment: %d / %d (%.0f/s)", i+1, len(texts), (i+1)/elapsed)

    df["sentiment_compound"] = compounds
    df["sentiment_positive"]  = positives
    df["sentiment_negative"]  = negatives
    df["sentiment_neutral"]   = neutrals
    df["sentiment_label"] = df["sentiment_compound"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )

    log.info("Sentiment done in %.1fs", time.time() - t0)
    df.to_parquet(F_SENTIMENT, index=False)
    log.info("Saved -> %s", F_SENTIMENT)
    return df


# ── Stage 3: SentenceTransformers embeddings ─────────────────────────────────

def stage_embeddings(df: pd.DataFrame | None = None, batch_size: int = 128):
    log.info("=== STAGE 3: SentenceTransformers Embeddings (all-MiniLM-L6-v2) ===")
    from sentence_transformers import SentenceTransformer

    if df is None:
        df = pd.read_parquet(F_SENTIMENT)

    texts = (df["title"] + " " + df["text"].str[:400]).fillna("").tolist()
    n = len(texts)
    log.info("Total texts: %d, batch_size=%d", n, batch_size)

    # Resume from checkpoint
    start_i = 0
    emb_list = []
    if EMB_CKPT.exists() and EMB_CKPT_N.exists():
        start_i = int(EMB_CKPT_N.read_text().strip())
        emb_list = [np.load(EMB_CKPT)]
        log.info("Resuming from checkpoint at row %d", start_i)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    t0 = time.time()

    for i in range(start_i, n, batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        emb_list.append(vecs.astype(np.float32))

        if (i // batch_size) % 200 == 0:
            so_far = i + len(batch)
            elapsed = time.time() - t0
            rate = (so_far - start_i) / elapsed if elapsed > 0 else 0
            eta  = (n - so_far) / rate / 60 if rate > 0 else 0
            log.info("  Embedding: %d / %d  (%.0f/s, ETA %.0fmin)",
                     so_far, n, rate, eta)
            # Save checkpoint
            combined = np.vstack(emb_list)
            np.save(EMB_CKPT, combined)
            EMB_CKPT_N.write_text(str(so_far))

    embeddings = np.vstack(emb_list)
    if len(embeddings) > n:
        embeddings = embeddings[:n]

    log.info("Embedding shape: %s  (%.1fs total)", embeddings.shape, time.time() - t0)
    np.save(F_EMB_NPY, embeddings)
    F_EMB_IDS.write_text("\n".join(df["mention_id"].astype(str).tolist()))
    log.info("Saved -> %s", F_EMB_NPY)

    # Clean up checkpoint
    EMB_CKPT.unlink(missing_ok=True)
    EMB_CKPT_N.unlink(missing_ok=True)

    return embeddings


# ── Stage 4: UMAP 384 -> 15 dims ─────────────────────────────────────────────

def stage_umap(embeddings: np.ndarray | None = None):
    log.info("=== STAGE 4: UMAP 384 -> 15 dims ===")
    import umap

    if embeddings is None:
        embeddings = np.load(F_EMB_NPY)
    log.info("Input shape: %s", embeddings.shape)

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    emb_15 = reducer.fit_transform(embeddings).astype(np.float32)
    log.info("UMAP done: %s  (%.1fmin)", emb_15.shape, (time.time()-t0)/60)

    np.save(F_UMAP_NPY, emb_15)
    log.info("Saved -> %s", F_UMAP_NPY)
    return emb_15


# ── Stage 5: HDBSCAN clustering ──────────────────────────────────────────────

def stage_hdbscan(emb_15: np.ndarray | None = None, df: pd.DataFrame | None = None):
    log.info("=== STAGE 5: HDBSCAN Clustering ===")
    import hdbscan

    if emb_15 is None:
        emb_15 = np.load(F_UMAP_NPY)
    if df is None:
        df = pd.read_parquet(F_SENTIMENT)

    log.info("Clustering %d points …", len(emb_15))
    t0 = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
        prediction_data=True,   # required for approximate_predict on new points
    )
    labels = clusterer.fit_predict(emb_15)
    elapsed = time.time() - t0

    n_clusters = int((labels >= 0).sum())
    n_noise    = int((labels == -1).sum())
    n_unique   = len(set(labels)) - (1 if -1 in labels else 0)
    log.info("HDBSCAN done in %.1fmin", elapsed/60)
    log.info("Clusters: %d unique | Valid points: %d | Noise: %d (%.1f%%)",
             n_unique, n_clusters, n_noise, 100*n_noise/len(labels))

    # Save model for incremental updates
    import pickle
    with open(F_HDBSCAN_MDL, "wb") as _f:
        pickle.dump(clusterer, _f, protocol=4)
    log.info("Saved HDBSCAN model -> %s", F_HDBSCAN_MDL)

    df = df.copy()
    df["cluster_id"] = labels

    df.to_parquet(F_CLUSTER, index=False)
    log.info("Saved -> %s", F_CLUSTER)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose output files already exist")
    parser.add_argument("--stage", default="all",
                        choices=["all","clean","sentiment","embeddings","umap","hdbscan"],
                        help="Run a single stage only")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    def skip(f: Path, name: str) -> bool:
        if args.resume and f.exists():
            log.info("Skipping %s (output exists: %s)", name, f.name)
            return True
        return False

    run_all = args.stage == "all"

    # Stage 1
    df_clean = None
    if run_all or args.stage == "clean":
        if not skip(F_CLEAN, "clean"):
            df_clean = stage_clean()

    # Stage 2
    df_sent = None
    if run_all or args.stage == "sentiment":
        if not skip(F_SENTIMENT, "sentiment"):
            df_sent = stage_sentiment(df_clean)

    # Stage 3
    embeddings = None
    if run_all or args.stage == "embeddings":
        if not skip(F_EMB_NPY, "embeddings"):
            embeddings = stage_embeddings(df_sent, batch_size=args.batch_size)

    # Stage 4
    emb_15 = None
    if run_all or args.stage == "umap":
        if not skip(F_UMAP_NPY, "umap"):
            emb_15 = stage_umap(embeddings)

    # Stage 5
    if run_all or args.stage == "hdbscan":
        if not skip(F_CLUSTER, "hdbscan"):
            stage_hdbscan(emb_15)

    log.info("=== Pipeline complete ===")
    log.info("Final output: %s", F_CLUSTER)


if __name__ == "__main__":
    main()
