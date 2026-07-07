#!/usr/bin/env python3
"""
Assign target_cluster_id / target_cluster / target_cluster_score to every post
using cosine similarity between SentenceTransformer post embeddings and
171 target cluster embeddings built from configs/taxonomy/target_clusters_171.csv.

How it works:
  1. Embed the 171 target clusters (target_cluster name + aliases + seed_keywords).
  2. Load existing post embeddings (embeddings_500k.npy).
  3. Compute cosine similarity (chunked) → best matching cluster per post.
  4. Write assignments to data/processed/target_cluster_assignments.parquet.
  5. Merge into nlp_clustered_500k.parquet (adds target_cluster_id, target_cluster,
     target_cluster_score columns).

The 171 target clusters replace unsupervised HDBSCAN as the primary dashboard
taxonomy.  HDBSCAN cluster_id is kept as a sub-trend discovery layer.

Usage:
    python scripts/assign_target_clusters.py
    python scripts/assign_target_clusters.py --sample 2000        # quick test
    python scripts/assign_target_clusters.py --recompute-clusters # re-embed clusters
    python scripts/assign_target_clusters.py --min-score 0.25     # stricter threshold
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
log = logging.getLogger("assign_target_clusters")

_ROOT            = Path(__file__).parent.parent
TAXONOMY_CSV     = _ROOT / "configs" / "taxonomy" / "target_clusters_171.csv"
F_EMB_NPY        = _ROOT / "data" / "processed" / "embeddings_500k.npy"
F_EMB_IDS        = _ROOT / "data" / "processed" / "embeddings_500k_ids.txt"
F_CLUSTER_PKL    = _ROOT / "data" / "processed" / "nlp_clustered_500k.parquet"
F_TC_EMBS        = _ROOT / "data" / "processed" / "target_cluster_embeddings.npy"
F_TC_IDS         = _ROOT / "data" / "processed" / "target_cluster_ids.txt"
F_ASSIGNMENTS    = _ROOT / "data" / "processed" / "target_cluster_assignments.parquet"

EMBED_MODEL      = "all-MiniLM-L6-v2"
DEFAULT_MIN_SCORE = 0.20    # minimum cosine sim to accept an assignment
CHUNK_SIZE        = 20_000  # rows processed per chunk (memory control)


# ── Cluster embedding ─────────────────────────────────────────────────────────

def _cluster_text(row: pd.Series) -> str:
    """Build a rich text representation of one target cluster for embedding."""
    parts = [str(row["target_cluster"])]
    if pd.notna(row.get("parent_category")):
        parts.append(str(row["parent_category"]))
    if pd.notna(row.get("aliases")) and str(row["aliases"]).strip():
        parts.append(str(row["aliases"]).replace("|", " "))
    if pd.notna(row.get("seed_keywords")) and str(row["seed_keywords"]).strip():
        # Take top 5 seed keywords to avoid over-weighting
        kws = [k.strip() for k in str(row["seed_keywords"]).split("|")[:5] if k.strip()]
        parts.append(" ".join(kws))
    return ". ".join(parts)


def build_cluster_embeddings(
    taxonomy: pd.DataFrame,
    recompute: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed the 171 target clusters.  Cached in data/processed/ to avoid re-running.
    Returns (embeddings shape (171,384), list of target_cluster_id strings).
    """
    if F_TC_EMBS.exists() and F_TC_IDS.exists() and not recompute:
        log.info("Loading cached cluster embeddings → %s", F_TC_EMBS)
        embs = np.load(F_TC_EMBS)
        ids  = F_TC_IDS.read_text().strip().split("\n")
        log.info("Cached: %d clusters, shape %s", len(ids), embs.shape)
        return embs, ids

    from sentence_transformers import SentenceTransformer
    log.info("Building cluster embeddings for %d target clusters …", len(taxonomy))
    model = SentenceTransformer(EMBED_MODEL)

    texts = [_cluster_text(row) for _, row in taxonomy.iterrows()]
    ids   = taxonomy["target_cluster_id"].astype(str).tolist()

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    log.info("Cluster embeddings built: shape %s in %.1fs", embs.shape, time.time() - t0)

    np.save(F_TC_EMBS, embs)
    F_TC_IDS.write_text("\n".join(ids))
    log.info("Saved → %s", F_TC_EMBS)
    return embs, ids


# ── Post-level assignment ─────────────────────────────────────────────────────

def assign_to_clusters(
    post_embs:    np.ndarray,
    cluster_embs: np.ndarray,
    cluster_ids:  list[str],
    id_to_name:   dict[str, str],
    min_score:    float,
) -> tuple[list[str], list[str], list[float]]:
    """
    Chunked cosine similarity → best cluster per post.

    Returns parallel lists: (assigned_ids, assigned_names, scores).
    Posts with best_score < min_score are marked 'C000' / 'unassigned'.
    """
    n = len(post_embs)
    assigned_ids:   list[str]   = []
    assigned_names: list[str]   = []
    assigned_scores: list[float] = []

    t0 = time.time()
    for start in range(0, n, CHUNK_SIZE):
        chunk = post_embs[start : start + CHUNK_SIZE]   # (C, 384)
        sims  = chunk @ cluster_embs.T                  # (C, 171)
        best_idx   = sims.argmax(axis=1)                # (C,)
        best_score = sims.max(axis=1)                   # (C,)

        for idx, score in zip(best_idx, best_score):
            if float(score) >= min_score:
                cid   = cluster_ids[int(idx)]
                label = id_to_name.get(cid, "unknown")
            else:
                cid   = "C000"
                label = "unassigned"
            assigned_ids.append(cid)
            assigned_names.append(label)
            assigned_scores.append(round(float(score), 4))

        done = min(start + CHUNK_SIZE, n)
        if done % 100_000 < CHUNK_SIZE or done == n:
            log.info("  %d / %d  (%.0fs)", done, n, time.time() - t0)

    return assigned_ids, assigned_names, assigned_scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Assign 171 target clusters to all posts")
    parser.add_argument("--sample", type=int, default=0,
                        help="Only assign this many posts (0 = all); for quick testing")
    parser.add_argument("--recompute-clusters", action="store_true",
                        help="Force re-embed target cluster texts (ignore cache)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum cosine similarity to accept assignment "
                             f"(default: {DEFAULT_MIN_SCORE})")
    args = parser.parse_args()

    taxonomy = pd.read_csv(TAXONOMY_CSV)
    log.info("Taxonomy loaded: %d target clusters", len(taxonomy))

    id_to_name = dict(zip(
        taxonomy["target_cluster_id"].astype(str),
        taxonomy["target_cluster"].astype(str),
    ))

    # Step 1: cluster embeddings
    cluster_embs, cluster_ids = build_cluster_embeddings(
        taxonomy, recompute=args.recompute_clusters
    )

    # Step 2: post embeddings
    log.info("Loading post embeddings from %s …", F_EMB_NPY)
    post_embs = np.load(F_EMB_NPY)
    emb_ids   = F_EMB_IDS.read_text().strip().split("\n")
    log.info("Post embeddings shape: %s  (%d IDs)", post_embs.shape, len(emb_ids))

    if args.sample > 0:
        log.info("Sampling first %d posts for testing", args.sample)
        post_embs = post_embs[:args.sample]
        emb_ids   = emb_ids[:args.sample]

    # Step 3: assign
    log.info("Assigning %d posts to %d clusters (min_score=%.2f) …",
             len(post_embs), len(cluster_ids), args.min_score)
    t0 = time.time()
    assigned_ids, assigned_names, assigned_scores = assign_to_clusters(
        post_embs, cluster_embs, cluster_ids, id_to_name, args.min_score
    )
    log.info("Assignment complete in %.1fmin", (time.time() - t0) / 60)

    # Step 4: build assignment dataframe
    result = pd.DataFrame({
        "mention_id":           emb_ids,
        "target_cluster_id":    assigned_ids,
        "target_cluster":       assigned_names,
        "target_cluster_score": assigned_scores,
    })

    # Step 4b: negative-keyword post-processing
    # For clusters that have negative_keywords defined, veto any assigned post
    # whose text matches one of those terms → force to unassigned.
    neg_map: dict[str, list[str]] = {}
    for _, row in taxonomy.iterrows():
        neg_raw = str(row.get("negative_keywords", "") or "")
        if neg_raw.strip():
            terms = [t.strip().lower() for t in neg_raw.split("|") if t.strip()]
            if terms:
                neg_map[str(row["target_cluster_id"])] = terms

    if neg_map:
        cids_with_neg = set(neg_map.keys())
        candidate_mask = result["target_cluster_id"].isin(cids_with_neg)
        n_candidates = candidate_mask.sum()
        log.info("Negative-keyword check: %d clusters with rules, %d candidate posts …",
                 len(neg_map), n_candidates)

        if n_candidates > 0:
            # Load post text only for candidates (cheap: single column from parquet)
            post_text_df = pd.read_parquet(
                F_CLUSTER_PKL, columns=["mention_id", "ner_input"]
            )
            post_text_df["mention_id"] = post_text_df["mention_id"].astype(str)
            post_text_df["_text_lower"] = post_text_df["ner_input"].fillna("").str.lower()

            candidates = result[candidate_mask].merge(
                post_text_df[["mention_id", "_text_lower"]], on="mention_id", how="left"
            )

            def _hits_neg(row_c: pd.Series) -> bool:
                terms = neg_map.get(row_c["target_cluster_id"], [])
                text  = row_c["_text_lower"] or ""
                return any(t in text for t in terms)

            veto_mask = candidates.apply(_hits_neg, axis=1)
            veto_ids  = set(candidates.loc[veto_mask, "mention_id"])
            log.info("Negative-keyword veto: %d posts overridden → unassigned", len(veto_ids))

            if veto_ids:
                veto_rows = result["mention_id"].isin(veto_ids)
                result.loc[veto_rows, "target_cluster_id"]    = "C000"
                result.loc[veto_rows, "target_cluster"]       = "unassigned"
                result.loc[veto_rows, "target_cluster_score"] = 0.0

    result.to_parquet(F_ASSIGNMENTS, index=False)
    log.info("Assignments saved → %s  (%d rows)", F_ASSIGNMENTS, len(result))

    # Stats
    n_assigned = (result["target_cluster"] != "unassigned").sum()
    log.info("Assigned: %d / %d  (%.1f%%)",
             n_assigned, len(result), 100 * n_assigned / len(result))
    log.info("Top 10 clusters:")
    for label, cnt in result["target_cluster"].value_counts().head(10).items():
        log.info("  %-45s %6d", label, cnt)

    if args.sample > 0:
        log.info("Sample run complete — skipping parquet update. "
                 "Re-run without --sample to update nlp_clustered_500k.parquet.")
        return

    # Step 5: merge into nlp_clustered_500k.parquet
    log.info("Merging into %s …", F_CLUSTER_PKL)
    df = pd.read_parquet(F_CLUSTER_PKL)
    df["mention_id"] = df["mention_id"].astype(str)

    # Drop stale columns
    for col in ["target_cluster_id", "target_cluster", "target_cluster_score"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    result["mention_id"] = result["mention_id"].astype(str)
    df = df.merge(result, on="mention_id", how="left")
    df["target_cluster"]       = df["target_cluster"].fillna("unassigned")
    df["target_cluster_id"]    = df["target_cluster_id"].fillna("C000")
    df["target_cluster_score"] = df["target_cluster_score"].fillna(0.0)

    df.to_parquet(F_CLUSTER_PKL, index=False)
    log.info("nlp_clustered_500k.parquet updated  (%d rows)", len(df))

    assigned_final = (df["target_cluster"] != "unassigned").sum()
    log.info("Posts with target_cluster: %d / %d  (%.1f%%)",
             assigned_final, len(df), 100 * assigned_final / len(df))


if __name__ == "__main__":
    main()
