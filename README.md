# Reddit Product Trend Copilot

Reddit 社区讨论数据产品趋势雷达，覆盖 66 个品类、496k+ 帖子。  
Tracks product trends across 66 categories from 496k+ Reddit posts.

---

## 目录结构 / Structure

```
.
├── dashboard_v2.py                          # Streamlit 主看板（入口）
├── requirements.txt                         # Python 依赖
│
├── scripts/
│   ├── scrape_large_2026.py                 # Reddit 数据爬虫（arctic-shift API）
│   ├── run_nlp_pipeline.py                  # NLP 流水线：VADER → Embeddings → UMAP → HDBSCAN
│   ├── build_dashboard_500k.py              # 从 parquet 生成看板数据 PKL
│   └── export_dashboard_html.py             # 导出静态 HTML（用于文档截图）
│
├── data/
│   └── processed/
│       ├── dashboard_data_500k.pkl          # 预计算看板数据（4个双周窗口）
│       └── cluster_brand_labels.csv         # Cluster → 品牌映射表
│
└── configs/
    ├── pipeline.toml                        # 流水线参数
    ├── sources.json                         # 数据源配置
    └── taxonomy/
        └── product_taxonomy_clean.csv       # 66 个品类定义
```

> **大文件**（parquet / npy / raw csv）不包含在 repo 中，见 `.gitignore`。

---

## 数据流 / Data Flow

```
Reddit arctic-shift API
        ↓  scrape_large_2026.py
data/raw/scraped_2026_large.csv   (~500k posts)
        ↓  run_nlp_pipeline.py
nlp_clustered_500k.parquet        (VADER sentiment + UMAP + HDBSCAN clusters)
        ↓  build_dashboard_500k.py
dashboard_data_500k.pkl           (4x biweekly windows, brand extraction, trend scores)
        ↓
dashboard_v2.py  →  Streamlit app (port 8503)
```

---

## 本地运行 / Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard_v2.py --server.port 8503
```

---

## 分享版 Streamlit 自动更新 / Shareable App Refresh

稳定分享版 Streamlit app 建议绑定 GitHub 的稳定分支（例如 `main`），入口文件使用：

```text
dashboard_v2.py
```

每次本地或服务器 pipeline 跑完后，执行：

```bash
python scripts/build_dashboard_500k.py
python scripts/build_forecast.py          # 可选：如果预测数据也更新
python scripts/publish_streamlit_snapshot.py --commit --push
```

这会刷新并提交分享版看板需要的小数据包：

```text
data/processed/dashboard_data_500k.pkl
data/processed/brand_posts_index.pkl
data/processed/forecast_data.pkl
data/processed/dashboard_manifest.json
```

Streamlit Cloud 只要连接到同一个 GitHub 分支，就会在 GitHub 收到新 commit 后自动重启并读取最新数据。

不要把 raw data、parquet、embedding、模型大文件推到 GitHub；这些仍然由 `.gitignore` 排除。分享版只使用轻量 processed artifacts。

---

## 核心算法 / Key Algorithms

| 维度 | 方法 |
|------|------|
| 情感分析 | VADER (`vaderSentiment`) |
| 向量嵌入 | SentenceTransformers `all-MiniLM-L6-v2` (384 dim) |
| 降维 | UMAP 384→15 dims |
| 聚类 | HDBSCAN (1,519 clusters) |
| 趋势分 | Spike 25% + Cross-community 25% + Sentiment 25% + Engagement 25% |
| 品牌识别 | 103k 品牌白名单 + 英文词典过滤 + 跨品类频率过滤 |
