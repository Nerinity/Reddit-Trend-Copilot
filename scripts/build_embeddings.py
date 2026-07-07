#!/usr/bin/env python3
"""
Generate sentence-transformer dense embeddings for the NLP-ready dataset.
Uses all-MiniLM-L6-v2 (384-dim) on the cleaned token_str column.

Outputs:
  data/processed/embeddings_2026.npy          – (N, 384) float32 numpy array
  data/processed/embeddings_mention_ids.txt   – N mention_ids (same row order)
  data/processed/nlp_embedding_ready_2026.parquet – full dataset + embedding col

Usage:
  python scripts/build_embeddings.py
  python scripts/build_embeddings.py --model paraphrase-MiniLM-L3-v2  # faster
  python scripts/build_embeddings.py --batch-size 512 --resume
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("build_embeddings")

_ROOT = Path(__file__).parent.parent
INPUT_CSV  = _ROOT / "data" / "processed" / "nlp_sentiment_ready_2026.csv"
OUT_NPY    = _ROOT / "data" / "processed" / "embeddings_2026.npy"
OUT_IDS    = _ROOT / "data" / "processed" / "embeddings_mention_ids.txt"
OUT_PARQUET= _ROOT / "data" / "processed" / "nlp_embedding_ready_2026.parquet"
CKPT       = _ROOT / "data" / "state" / "embedding_checkpoint.npy"
CKPT_IDX   = _ROOT / "data" / "state" / "embedding_checkpoint_idx.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--text-col", default="token_str")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    log.info("Loading dataset…")
    df = pd.read_csv(args.input, low_memory=False)
    df = df.drop_duplicates(subset=["mention_id"]).reset_index(drop=True)
    total = len(df)
    log.info("Rows: %d", total)

    col = args.text_col if args.text_col in df.columns else "full_text"
    texts = df[col].fillna("").astype(str).tolist()
    ids   = df["mention_id"].astype(str).tolist()

    start_idx, existing = 0, []
    if args.resume and CKPT.exists() and CKPT_IDX.exists():
        existing = list(np.load(CKPT))
        start_idx = int(CKPT_IDX.read_text().strip())
        log.info("Resuming from row %d", start_idx)

    log.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)
    log.info("Embedding dim: %d", model.get_sentence_embedding_dimension())

    embeddings = existing[:]
    t0 = time.time()

    for i in range(start_idx, total, args.batch_size):
        batch = texts[i : i + args.batch_size]
        vecs  = model.encode(batch, batch_size=args.batch_size,
                             show_progress_bar=False, convert_to_numpy=True,
                             normalize_embeddings=True)
        embeddings.extend(vecs)
        done    = i + len(batch)
        elapsed = time.time() - t0
        rate    = (done - start_idx) / elapsed if elapsed else 0
        eta     = (total - done) / rate / 60 if rate else 0
        log.info("  %d/%d (%.0f%%) | %.0f/sec | ETA %.0f min",
                 done, total, done/total*100, rate, eta)

        if done % 10000 < args.batch_size:
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            np.save(CKPT, np.array(embeddings, dtype=np.float32))
            CKPT_IDX.write_text(str(done))
            log.info("  Checkpoint @ row %d", done)

    emb = np.array(embeddings, dtype=np.float32)
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, emb)
    OUT_IDS.write_text("\n".join(ids))
    log.info("Saved .npy %s  shape=%s", OUT_NPY, emb.shape)

    log.info("Building parquet with embedding column…")
    df["embedding"] = [v.tolist() for v in emb]
    df.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    log.info("Saved parquet %s  %.0f MB", OUT_PARQUET, OUT_PARQUET.stat().st_size/1e6)

    for f in (CKPT, CKPT_IDX):
        f.exists() and f.unlink()

    log.info("Done in %.1f min – %d embeddings", (time.time()-t0)/60, len(embeddings))

if __name__ == "__main__":
    main()
