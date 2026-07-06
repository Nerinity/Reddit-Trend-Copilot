# Prompt: Upgrade Reddit Trend Pipeline to Cover 171 Target Clusters

You are working in the `Nerinity/Reddit-Trend-Copilot` repository.

Goal: upgrade the Reddit data collection and NLP pipeline so the system can reliably cover the 171 target product/trend clusters defined in:

`configs/taxonomy/target_clusters_171.csv`

The current pipeline collects broad Reddit data from subreddit groups in `configs/sources.json`, then runs:

Reddit raw data -> VADER sentiment -> SentenceTransformer embeddings -> UMAP -> HDBSCAN

This currently produces many unsupervised clusters, but the product needs stable dashboard dimensions based on the 171 target clusters.

## Required Product Behavior

The dashboard should be able to show stable trend metrics for each of the 171 target clusters, such as:

- mention volume
- growth rate
- cross-subreddit coverage
- engagement
- sentiment
- representative posts
- emerging sub-trends inside the target cluster

Do not rely on HDBSCAN alone to create exactly 171 clusters. Treat the 171 rows in `target_clusters_171.csv` as the supervised target taxonomy.

## Data Collection Changes

Add a cluster-aware Reddit scraping layer.

Create or update code so scraping can run by target cluster:

1. Load `configs/taxonomy/target_clusters_171.csv`.
2. For each target cluster, generate/search using:
   - `target_cluster`
   - `aliases`
   - `seed_keywords`
3. Map each target cluster to relevant subreddit groups from `configs/sources.json`.
4. Collect Reddit posts from:
   - Arctic Shift archive
   - Reddit public JSON
   - Reddit RSS
5. Store raw mentions with these additional fields:
   - `target_cluster_hint`
   - `target_cluster_id`
   - `matched_query`
   - `matched_alias`
   - `source_confidence`
   - `taxonomy_level`

The output should remain compatible with the existing raw CSV schema in `data/raw/scraped_2026_large.csv`.

## Suggested New Files

Add:

`scripts/scrape_by_target_clusters.py`

Purpose:

- read the 171 target clusters
- run cluster-aware subreddit/query scraping
- preserve raw text
- dedupe by `mention_id`
- write to `data/raw/scraped_2026_large.csv`
- checkpoint progress by target cluster

Add:

`scripts/build_cluster_coverage_report.py`

Purpose:

- count raw and processed mentions per target cluster
- show under-covered clusters
- show top subreddits per cluster
- show top queries per cluster
- export `data/processed/target_cluster_coverage.csv`

## NLP Changes

Add a supervised target-cluster assignment stage before or after embeddings.

Suggested file:

`scripts/assign_target_clusters.py`

Logic:

1. Embed every post using the existing SentenceTransformer model.
2. Embed each target cluster using:
   - `target_cluster`
   - `aliases`
   - `seed_keywords`
   - `parent_category`
3. Compute cosine similarity between post embeddings and target cluster embeddings.
4. Assign:
   - `target_cluster_id`
   - `target_cluster`
   - `target_cluster_score`
5. If the post came from cluster-aware scraping, combine:
   - query/source hint
   - embedding similarity
   - subreddit source relevance
6. Keep HDBSCAN as a sub-trend discovery layer, not the main dashboard taxonomy.

Final processed rows should include both:

- `target_cluster`
- `cluster_id`

Where:

- `target_cluster` = stable 171 taxonomy label
- `cluster_id` = unsupervised sub-trend cluster

## Important Fix

In `scripts/scrape_large_2026.py`, the `priority_cats` list currently uses old category names that do not match `configs/sources.json`.

Fix this by replacing them with real keys from `sources.json`, such as:

- `creator_commerce`
- `deals_shopping_reviews`
- `beauty_skincare`
- `supplements_nutrition`
- `womens_fashion`
- `mens_fashion`
- `fashion_accessories`

## Validation

After implementation, run:

```bash
python scripts/scrape_by_target_clusters.py --dry-run
python scripts/build_cluster_coverage_report.py
python scripts/assign_target_clusters.py --sample 1000
```

Expected validation outputs:

- `configs/taxonomy/target_clusters_171.csv` loads with 171 rows.
- Coverage report includes all 171 target clusters.
- Under-covered clusters are listed clearly.
- Processed rows include `target_cluster_id`, `target_cluster`, and `target_cluster_score`.
- Existing dashboard pipeline still works.

## Design Principle

The 171 target clusters are not optional labels. They are the product taxonomy for the dashboard.

Use unsupervised clustering only to discover smaller sub-trends inside each target cluster.
