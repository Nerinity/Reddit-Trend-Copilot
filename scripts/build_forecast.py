#!/usr/bin/env python3
"""Forecast pipeline: XGBoost (rising probability) + Prophet (trend extrapolation).

Steps
-----
1. Aggregate raw posts → weekly category metrics (mentions, sentiment, communities).
2. Build lag-feature matrix (4-week window) for XGBoost.
3. Train XGBoost classifier to predict "will this category rise in the next 2-4 weeks?".
4. Fit one Prophet model per category → 4-week forecast.
5. Save everything to data/processed/forecast_data.pkl.
"""
from __future__ import annotations
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PARQUET = Path("data/processed/nlp_sentiment_500k.parquet")
OUTPUT  = Path("data/processed/forecast_data.pkl")
FORECAST_WEEKS = 4
MIN_WEEKS = 10  # minimum weekly observations to include a category


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: weekly aggregation
# ─────────────────────────────────────────────────────────────────────────────

def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["week"] = df["published_at"].dt.to_period("W").dt.start_time
    weekly = (
        df.groupby(["category", "week"])
        .agg(
            mentions=("mention_id", "count"),
            communities=("community", "nunique"),
            mean_sentiment=("sentiment_compound", "mean"),
            mean_engagement=("engagement_score", "mean"),
        )
        .reset_index()
        .sort_values(["category", "week"])
    )
    # Total posts per week across all categories (for market-share normalisation)
    total_per_week = weekly.groupby("week")["mentions"].sum().rename("total_mentions")
    total_comms    = weekly.groupby("week")["communities"].sum().rename("total_communities")
    weekly = weekly.merge(total_per_week, on="week").merge(total_comms, on="week")
    weekly["share"]      = weekly["mentions"]    / weekly["total_mentions"].clip(lower=1)
    weekly["comm_share"] = weekly["communities"] / weekly["total_communities"].clip(lower=1)
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_weekly_trend_score(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute a weekly version of the dashboard trend_score (4 equal components)."""
    weekly = weekly.copy()

    # spike: share-based w/w ratio (crawl-neutral)
    weekly["spike_ww"] = weekly.groupby("category")["share"].transform(
        lambda s: s / s.shift(1).clip(lower=1e-9)
    )
    # normalise spike within each week (rank → 0-1)
    weekly["spike_norm"] = weekly.groupby("week")["spike_ww"].transform(
        lambda s: (s.rank(pct=True))
    )
    # cross-community: fraction of max community count this week
    weekly["comm_norm"] = weekly.groupby("week")["communities"].transform(
        lambda s: s / max(s.max(), 1)
    )
    # sentiment score: (sentiment + 1) / 2  → 0-1
    weekly["sentiment_score"] = (weekly["mean_sentiment"] + 1) / 2
    # engagement momentum: w/w ratio (relative, not absolute)
    weekly["eng_mom"] = weekly.groupby("category")["mean_engagement"].transform(
        lambda s: (s / s.shift(1).clip(lower=1e-9)).clip(upper=4)
    )
    weekly["eng_mom_norm"] = weekly.groupby("week")["eng_mom"].transform(
        lambda s: s / max(s.quantile(0.97), 1e-9)
    )
    weekly["trend_score_w"] = (
        0.25 * weekly["spike_norm"] +
        0.25 * weekly["comm_norm"] +
        0.25 * weekly["sentiment_score"] +
        0.25 * weekly["eng_mom_norm"]
    ).clip(upper=1.0)

    return weekly


def engineer_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Features are all relative (shares, rates, normalised scores) — no raw counts."""
    weekly = add_weekly_trend_score(weekly)
    records = []

    for cat, grp in weekly.groupby("category"):
        grp = grp.sort_values("week").reset_index(drop=True)
        n = len(grp)
        if n < MIN_WEEKS:
            continue

        ts_roll_mean = grp["trend_score_w"].rolling(4, min_periods=2).mean()
        ts_roll_std  = grp["trend_score_w"].rolling(4, min_periods=2).std().clip(lower=1e-6)

        for i in range(4, n):
            cur  = grp.iloc[i]
            lags = grp.iloc[i - 4 : i]

            s0 = cur["share"]
            s1 = grp.iloc[i - 1]["share"]
            s4 = grp.iloc[i - 4]["share"]

            ts0 = cur["trend_score_w"]
            ts1 = grp.iloc[i - 1]["trend_score_w"]

            feat: dict = {
                "category":           cat,
                "week":               cur["week"],
                "week_num":           i,
                # ── trend_score components (all 0-1 normalised) ───────────────
                "trend_score":        ts0,
                "trend_score_lag1":   ts1,
                "trend_score_chg":    ts0 - ts1,
                "trend_score_z":      (ts0 - ts_roll_mean.iloc[i]) / ts_roll_std.iloc[i],
                "trend_score_mom":    float(
                    np.polyfit(range(4), lags["trend_score_w"].values, 1)[0]
                ),
                # ── individual components ─────────────────────────────────────
                "spike_norm":         cur["spike_norm"],
                "spike_norm_lag1":    grp.iloc[i - 1]["spike_norm"],
                "comm_norm":          cur["comm_norm"],
                "sentiment_score":    cur["sentiment_score"],
                "sentiment_chg":      cur["sentiment_score"] - grp.iloc[i - 1]["sentiment_score"],
                "eng_mom_norm":       cur["eng_mom_norm"],
                # ── share momentum (crawl-neutral) ────────────────────────────
                "share_chg_1w":       s0 - s1,
                "share_chg_4w":       s0 - s4,
                "comm_share":         cur["comm_share"],
                "comm_share_chg":     cur["comm_share"] - grp.iloc[i - 1]["comm_share"],
            }

            # Target: will trend_score rank (within all cats that week) improve?
            if i + 2 < n:
                future_ts = grp.iloc[i + 1 : i + 3]["trend_score_w"].mean()
                feat["rise_2w"] = int(future_ts > ts0)
            else:
                feat["rise_2w"] = np.nan

            if i + 4 < n:
                future_ts = grp.iloc[i + 1 : i + 5]["trend_score_w"].mean()
                feat["rise_4w"] = int(future_ts > ts0)
            else:
                feat["rise_4w"] = np.nan

            records.append(feat)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: XGBoost training + prediction
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "trend_score", "trend_score_lag1", "trend_score_chg",
    "trend_score_z", "trend_score_mom",
    "spike_norm", "spike_norm_lag1",
    "comm_norm", "sentiment_score", "sentiment_chg",
    "eng_mom_norm",
    "share_chg_1w", "share_chg_4w",
    "comm_share", "comm_share_chg",
    "week_num", "cat_encoded",
]


def train_xgb(feat_df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    feat_df = feat_df.copy()
    feat_df["cat_encoded"] = le.fit_transform(feat_df["category"])

    # For each category, the latest observation is the prediction row
    latest_idx = feat_df.groupby("category")["week_num"].idxmax()
    pred_mask  = feat_df.index.isin(latest_idx)

    results = []
    for target, weight in [("rise_2w", 0.55), ("rise_4w", 0.45)]:
        train_df = feat_df[~pred_mask & feat_df[target].notna()].copy()
        pred_df  = feat_df[pred_mask].copy()

        if train_df.empty:
            continue

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[target].astype(int)
        X_pred  = pred_df[FEATURE_COLS]

        # Class balance
        pos_rate = y_train.mean()
        scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_pred)[:, 1]
        tmp = pred_df[["category"]].copy()
        tmp[f"prob_{target}"] = proba
        tmp["_weight"] = weight
        results.append(tmp)

    if not results:
        return pd.DataFrame()

    r2 = results[0].rename(columns={"prob_rise_2w": "p2"})[["category", "p2"]]
    r4 = results[1].rename(columns={"prob_rise_4w": "p4"})[["category", "p4"]]
    merged = r2.merge(r4, on="category", how="outer").fillna(0.5)
    merged["rise_prob"] = (merged["p2"] * 0.55 + merged["p4"] * 0.45).round(3)
    merged = merged.sort_values("rise_prob", ascending=False).reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Prophet forecasts
# ─────────────────────────────────────────────────────────────────────────────

PROPHET_RECENT_WEEKS = 10  # fit only the most recent N weeks to avoid crawl ramp-up bias

def build_prophet_forecasts(weekly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from prophet import Prophet

    forecasts: dict[str, pd.DataFrame] = {}
    valid_cats = (
        weekly.groupby("category")["mentions"].count()
        .loc[lambda s: s >= MIN_WEEKS].index.tolist()
    )

    # Need spike_ww in weekly — add it if not already present
    if "spike_ww" not in weekly.columns:
        weekly = add_weekly_trend_score(weekly)

    for cat in valid_cats:
        cat_data = weekly[weekly["category"] == cat].sort_values("week").copy()

        # Use w/w spike ratio on share (crawl-neutral momentum signal)
        grp = cat_data[["week", "spike_ww"]].copy()
        grp.columns = ["ds", "y"]
        grp["ds"] = pd.to_datetime(grp["ds"]).dt.tz_localize(None)

        # Drop first row (NaN spike) and trim to recent weeks
        grp = grp.dropna(subset=["y"])
        grp = grp[grp["y"].between(0.01, 10)].tail(PROPHET_RECENT_WEEKS).reset_index(drop=True)
        if len(grp) < 6:
            continue

        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(grp)

        future   = model.make_future_dataframe(periods=FORECAST_WEEKS, freq="W")
        forecast = model.predict(future)

        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out["yhat"]       = out["yhat"].clip(lower=0).round(3)
        out["yhat_lower"] = out["yhat_lower"].clip(lower=0).round(3)
        out["yhat_upper"] = out["yhat_upper"].clip(lower=0).round(3)
        out["actual"]     = grp.set_index("ds").reindex(out["ds"])["y"].values
        forecasts[cat]    = out

    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading parquet …")
    df = pd.read_parquet(PARQUET)

    log.info("Building weekly series …")
    weekly = build_weekly(df)
    log.info("  %d category-weeks", len(weekly))

    log.info("Engineering features …")
    feat_df = engineer_features(weekly)
    log.info("  %d observations, %d categories", len(feat_df), feat_df["category"].nunique())

    log.info("Training XGBoost …")
    xgb_preds = train_xgb(feat_df)
    log.info("  predictions for %d categories", len(xgb_preds))

    log.info("Fitting Prophet models …")
    prophet_forecasts = build_prophet_forecasts(weekly)
    log.info("  forecasts for %d categories", len(prophet_forecasts))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(
            {
                "xgb_predictions":   xgb_preds,
                "prophet_forecasts": prophet_forecasts,
                "weekly":            weekly,
            },
            f,
        )
    log.info("Saved → %s", OUTPUT)


if __name__ == "__main__":
    main()
