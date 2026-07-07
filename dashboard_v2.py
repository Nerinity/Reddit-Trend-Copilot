#!/usr/bin/env python3
"""
Reddit Trend Intelligence Copilot — Dashboard v2
Workflow: Trend Command Center | Spike Radar | Cluster Workbench | Forecast Lab
"""
import pickle
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit Trend Intelligence Copilot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH        = Path(__file__).parent / "data" / "processed" / "dashboard_data_500k.pkl"
FORECAST_PATH    = Path(__file__).parent / "data" / "processed" / "forecast_data.pkl"
PARQUET_PATH     = Path(__file__).parent / "data" / "processed" / "nlp_clustered_500k.parquet"
BRAND_POSTS_PATH = Path(__file__).parent / "data" / "processed" / "brand_posts_index.pkl"
BG_PATH          = Path(__file__).parent / "assets" / "tiktok_neon_bg.png"
DEFAULT_WINDOW_LABEL = "06/20–06/27"

US_HOLIDAYS = {
    "2025-12-25": "🎄 Christmas",
    "2026-01-01": "🎆 New Year's Day",
    "2026-01-19": "MLK Day",
    "2026-02-14": "💝 Valentine's Day",
    "2026-02-16": "President's Day",
    "2026-03-08": "Women's Day",
    "2026-03-17": "St. Patrick's Day",
    "2026-04-05": "🐣 Easter",
    "2026-05-10": "💐 Mother's Day",
    "2026-05-25": "Memorial Day",
    "2026-06-19": "Juneteenth",
    "2026-06-21": "👨 Father's Day",
}

TIKTOK_EVENTS = [
    ("2025-12-25", "2025-12-31", "🛍 Holiday Sale"),
    ("2026-01-01", "2026-01-07", "🛒 New Year Sale"),
    ("2026-01-20", "2026-01-25", "📅 DFYD Jan"),
    ("2026-02-10", "2026-02-14", "💝 Valentine Sale"),
    ("2026-03-01", "2026-03-15", "🌸 Spring Sale"),
    ("2026-04-06", "2026-04-13", "📅 DFYD Apr"),
    ("2026-05-04", "2026-05-11", "💐 Mother's Day Sale"),
    ("2026-05-25", "2026-05-31", "☀️ Summer Kickoff"),
    ("2026-06-15", "2026-06-22", "☀️ Summer Sale"),
]

COLORS = {
    "rising":   "#00F2A9",
    "stable":   "#94A3B8",
    "declining":"#FF0050",
    "primary":  "#00F2EA",
}

DIR_ZH  = {"rising":"上升中","stable":"平稳","declining":"下降中"}
DIR_EN  = {"rising":"Rising","stable":"Stable","declining":"Declining"}
DIR_EMO = {"rising":"🟢","stable":"⚪","declining":"🔴"}

TREND_THRESHOLDS = {
    "min_current_mentions": 20,
    "min_prior_mentions": 20,
    "min_rise_abs_delta": 8,
    "min_rise_relative_change": 0.12,
    "min_decline_abs_delta": 30,
    "min_decline_relative_change": 0.25,
    "smoothing_mentions": 5,
}

def background_image_css() -> str:
    if not BG_PATH.exists():
        return "linear-gradient(135deg,#030711 0%,#090A16 50%,#120312 100%)"
    encoded = base64.b64encode(BG_PATH.read_bytes()).decode("utf-8")
    return f"url('data:image/png;base64,{encoded}')"

APP_BG = background_image_css()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;color:#F8FAFC;font-weight:650;}
.stApp{
  background-image:
    linear-gradient(180deg, rgba(1,4,12,.62), rgba(1,4,12,.82)),
    radial-gradient(circle at 18% 18%, rgba(0,242,234,.18), transparent 28%),
    radial-gradient(circle at 84% 22%, rgba(255,0,80,.18), transparent 31%),
    __APP_BG__;
  background-size:cover;
  background-position:center;
  background-attachment:fixed;
  color:#F8FAFC;
}
.stApp:before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(0,242,234,.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,0,80,.055) 1px, transparent 1px);
  background-size:34px 34px;
  mask-image:linear-gradient(to bottom,rgba(0,0,0,.38),rgba(0,0,0,.04));
}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1480px;position:relative;z-index:1;}
#MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"]{
  display:none!important;visibility:hidden!important;height:0!important;
}
div[data-testid="metric-container"]{
  background:linear-gradient(135deg,rgba(8,13,28,.74),rgba(13,17,35,.52));
  border:1px solid rgba(0,242,234,.22);
  border-radius:16px;padding:.9rem 1.1rem;
  box-shadow:0 18px 54px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.08);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);}
div[data-testid="metric-container"] label{font-size:.75rem;color:#FFFFFF;font-weight:800;}
div[data-testid="stMetricValue"]{color:#F8FAFC;}
div[data-testid="stMetricDelta"]{color:#D8F7F6;}
.hero{
  text-align:center;
  border:1px solid rgba(0,242,234,.24);
  border-radius:28px;padding:34px 30px 28px;margin:6px auto 18px;
  background:linear-gradient(135deg,rgba(4,10,24,.54),rgba(15,7,24,.36));
  box-shadow:0 24px 80px rgba(0,0,0,.38), 0 0 42px rgba(0,242,234,.10);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
}
.signal-badge{
  display:inline-flex;align-items:center;justify-content:center;
  border:1px solid rgba(255,0,80,.34);border-radius:999px;
  padding:5px 12px;margin-bottom:14px;
  color:#D8F7F6;background:rgba(255,0,80,.08);
  font-size:.76rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.hero-title{
  font-size:clamp(2.25rem,4.8vw,5.25rem);
  font-weight:800;color:#FFFFFF;letter-spacing:0;margin:0;line-height:.98;
  text-shadow:0 0 26px rgba(0,242,234,.25), 0 0 42px rgba(255,0,80,.16);
}
.hero-sub{font-size:1rem;color:#FFFFFF;margin:16px auto 0;line-height:1.55;max-width:760px;font-weight:700;}
.hero-kpis{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin-top:18px;}
.hero-kpi{
  background:rgba(8,13,28,.58);border:1px solid rgba(255,255,255,.12);
  border-radius:16px;padding:12px 14px;min-height:82px;
}
.hero-kpi .label{font-size:.72rem;color:#A8B5C9;font-weight:650;text-transform:uppercase;letter-spacing:.02em;}
.hero-kpi .value{font-size:1.55rem;color:#F8FAFC;font-weight:760;margin-top:4px;}
.trend-stat-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin:18px 0 10px;}
.trend-stat{
  min-height:104px;padding:16px 18px;border-radius:18px;
  background:linear-gradient(135deg,rgba(8,13,28,.74),rgba(13,17,35,.52));
  border:1px solid rgba(0,242,234,.18);
  box-shadow:0 18px 54px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.08);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
}
.trend-stat .label{font-size:.78rem;color:#FFFFFF;font-weight:800;line-height:1.25;}
.trend-stat .value{font-size:2.15rem;color:#F8FAFC;font-weight:800;margin-top:8px;line-height:1.05;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.trend-stat .value.period{font-size:1.18rem;letter-spacing:0;overflow:visible;white-space:normal;line-height:1.16;}
.trend-stat .delta{display:inline-block;margin-top:8px;font-size:.78rem;color:#D8F7F6;background:rgba(255,255,255,.06);border-radius:999px;padding:3px 8px;}
.period-range{display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px;}
.period-range div{
  background:rgba(0,0,0,.34);border:1px solid rgba(255,255,255,.10);
  border-radius:12px;padding:8px 10px;
}
.period-range span{display:block;color:#D8F7F6;font-size:.7rem;font-weight:850;margin-bottom:3px;}
.period-range strong{display:block;color:#FFFFFF;font-size:.95rem;font-weight:900;line-height:1.1;white-space:nowrap;}
@media(max-width:900px){.trend-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.trend-stat .value{font-size:1.8rem;}}
.chart-caption{
  font-size:.78rem;color:#FFFFFF;margin-bottom:.65rem;line-height:1.5;font-weight:750;
  padding:9px 12px;background:rgba(0,0,0,.68);border-left:3px solid #00F2EA;
  border-radius:12px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);}
.spike-card{
  background:linear-gradient(135deg,rgba(8,13,28,.72),rgba(18,11,28,.56));
  border:1px solid rgba(0,242,234,.20);border-radius:18px;
  padding:1rem 1.2rem;margin-bottom:.8rem;
  box-shadow:0 18px 50px rgba(0,0,0,.26);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);}
.nav-shell{
  max-width:760px;margin:0 auto 18px;padding:14px 16px;
  background:linear-gradient(135deg,rgba(6,10,24,.66),rgba(14,16,34,.46));
  border:1px solid rgba(255,0,80,.42);border-radius:18px;
  box-shadow:0 18px 54px rgba(0,0,0,.28),0 0 28px rgba(255,0,80,.16);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
}
.nav-title{font-size:.78rem;color:#FFFFFF;text-align:center;margin-bottom:8px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}
section[data-testid="stSidebar"],div[data-testid="stSelectbox"] div[data-baseweb="select"]>div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"]>div{
  background:rgba(8,13,28,.72)!important;
  border:1.5px solid rgba(255,0,80,.62)!important;
  border-radius:14px!important;
  color:#F8FAFC!important;
  box-shadow:0 0 0 1px rgba(255,0,80,.16),0 0 22px rgba(255,0,80,.16)!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div:hover,
div[data-testid="stMultiSelect"] div[data-baseweb="select"]>div:hover{
  border-color:#FF0050!important;
  box-shadow:0 0 0 1px rgba(255,0,80,.45),0 0 30px rgba(255,0,80,.26)!important;
}
div[data-baseweb="select"] span{color:#F8FAFC!important;}
.stSlider label,.stSelectbox label,.stMultiSelect label{color:#D8F7F6!important;}
div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li{
  color:#FFFFFF!important;font-weight:700;
}
.welcome-panel{
  max-width:980px;margin:22px auto 0;padding:24px 26px;border-radius:24px;
  background:rgba(0,0,0,.72);border:1px solid rgba(0,242,234,.24);
  box-shadow:0 24px 80px rgba(0,0,0,.42), 0 0 44px rgba(255,0,80,.10);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
}
.welcome-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:18px;}
.welcome-rule{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
  border-radius:16px;padding:14px 15px;min-height:132px;
}
.welcome-rule .rule-title{color:#FFFFFF;font-weight:850;font-size:.95rem;margin-bottom:8px;}
.welcome-rule .rule-body{color:#D8F7F6;font-size:.82rem;line-height:1.5;font-weight:700;}
div[data-testid="stButton"] button{
  width:100%;border:1px solid rgba(0,242,234,.42);border-radius:16px;
  background:linear-gradient(135deg,rgba(0,242,234,.18),rgba(255,0,80,.20)),#050711;
  color:#FFFFFF;font-weight:900;padding:1.05rem 1.35rem;min-height:56px;font-size:1rem;
  box-shadow:0 16px 42px rgba(0,0,0,.28),0 0 28px rgba(0,242,234,.14);
}
div[data-testid="stButton"] button:hover{
  border-color:#00F2EA;color:#FFFFFF;background:linear-gradient(135deg,rgba(0,242,234,.28),rgba(255,0,80,.26)),#050711;
}
.wip-banner{
  background:linear-gradient(135deg,rgba(255,0,80,.18),rgba(0,242,234,.10)),rgba(0,0,0,.72);
  border:1px solid rgba(255,0,80,.52);border-radius:18px;
  padding:16px 18px;margin-bottom:14px;
  box-shadow:0 18px 48px rgba(0,0,0,.28),0 0 26px rgba(255,0,80,.16);
}
.wip-pill{
  display:inline-flex;border:1px solid rgba(255,0,80,.70);border-radius:999px;
  padding:3px 10px;margin-bottom:8px;background:rgba(255,0,80,.16);
  color:#FFFFFF;font-size:.78rem;font-weight:900;letter-spacing:.08em;
}
.wip-banner .wip-title{font-size:1.08rem;font-weight:900;color:#FFFFFF;margin-bottom:4px;}
.wip-banner .wip-copy{font-size:.9rem;font-weight:750;color:#D8F7F6;line-height:1.5;}
@media(max-width:900px){.welcome-grid{grid-template-columns:1fr;}}
hr{border-color:rgba(255,255,255,.10)!important;}
h1,h2,h3,h4,h5,h6,p,span,div{letter-spacing:0;}
a{color:#00F2EA;}
</style>
""".replace("__APP_BG__", APP_BG), unsafe_allow_html=True)

if "signal_radar_entered" not in st.session_state:
    st.session_state.signal_radar_entered = False

if not st.session_state.signal_radar_entered:
    st.markdown("""
    <div class="hero">
      <div class="signal-badge">North America Consumer Trend Radar</div>
      <div class="hero-title">North America<br>Consumer Signal Radar</div>
      <div class="hero-sub">
        Enter a weekly signal room for Reddit consumer discussion, product demand, keyword evidence,
        and creator-commerce opportunity discovery.
      </div>
    </div>
    <div class="welcome-panel">
      <div style="font-size:1.35rem;font-weight:850;color:#FFFFFF;margin-bottom:8px">
        How to read this radar
      </div>
      <div style="color:#D8F7F6;font-weight:750;line-height:1.55">
        This dashboard reads recent Reddit discussions as market signals. It does not treat every mention
        as a product recommendation. The goal is to find where discussion is growing, what people are naming,
        and which clusters deserve deeper business review.
      </div>
      <div class="welcome-grid">
        <div class="welcome-rule">
          <div class="rule-title">1. Weekly signal window</div>
          <div class="rule-body">Each period compares the selected week with the prior week. The homepage shows active clusters, rising/stable/declining counts, and total discussion volume.</div>
        </div>
        <div class="welcome-rule">
          <div class="rule-title">2. Dynamic trend direction</div>
          <div class="rule-body">Rising/falling labels are recalculated on page load using lightweight thresholds, so threshold changes do not rebuild the 500K-post pipeline.</div>
        </div>
        <div class="welcome-rule">
          <div class="rule-title">3. Keyword evidence first</div>
          <div class="rule-body">Keywords include brands, product nouns, trend phrases, and user concern objects. Use hot posts to inspect the original context before making decisions.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    spacer_l, cta_col, spacer_r = st.columns([1, 1.45, 1])
    with cta_col:
        if st.button("Enter Signal Radar", type="primary"):
            st.session_state.signal_radar_entered = True
            st.rerun()
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_brand_posts_index() -> dict | None:
    if not BRAND_POSTS_PATH.exists():
        return None
    with open(BRAND_POSTS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_posts_df():
    if not PARQUET_PATH.exists():
        return None
    cols = ["category","title","ner_input","url","community",
            "engagement_score","sentiment_label","published_at"]
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df

def apply_dynamic_trend_direction(stats: pd.DataFrame) -> pd.DataFrame:
    """Re-label trend direction at dashboard time without rebuilding source data."""
    out = stats.copy()
    if "current_mentions" not in out.columns:
        return out

    cur = pd.to_numeric(out["current_mentions"], errors="coerce").fillna(0)
    if "previous_mentions" in out.columns:
        prev = pd.to_numeric(out["previous_mentions"], errors="coerce").fillna(0)
    elif "prev_mentions" in out.columns:
        prev = pd.to_numeric(out["prev_mentions"], errors="coerce").fillna(0)
    elif "mentions_delta" in out.columns:
        prev = cur - pd.to_numeric(out["mentions_delta"], errors="coerce").fillna(0)
        prev = prev.clip(lower=0)
    else:
        return out
    delta = cur - prev
    rel_change = delta / (prev + TREND_THRESHOLDS["smoothing_mentions"])

    rising = (
        (cur >= TREND_THRESHOLDS["min_current_mentions"])
        & (delta >= TREND_THRESHOLDS["min_rise_abs_delta"])
        & (rel_change >= TREND_THRESHOLDS["min_rise_relative_change"])
    )
    declining = (
        (prev >= TREND_THRESHOLDS["min_prior_mentions"])
        & (delta <= -TREND_THRESHOLDS["min_decline_abs_delta"])
        & (rel_change <= -TREND_THRESHOLDS["min_decline_relative_change"])
    )

    out["trend_direction"] = np.select(
        [rising, declining],
        ["rising", "declining"],
        default="stable",
    )
    out["dashboard_relative_change"] = rel_change
    out["dashboard_mentions_delta"] = delta
    return out

def display_period_start(label: str, win: dict) -> pd.Timestamp:
    return pd.Timestamp(win["cur_start"]) + pd.Timedelta(days=1)

def display_period_label(label: str) -> str:
    """Show non-overlapping inclusive dates for boundary-based weekly windows."""
    try:
        win = windows_data[label]["window"]
    except Exception:
        return label
    start = display_period_start(label, win)
    end = pd.Timestamp(win["cur_end"])
    return f"{start.strftime('%m/%d')}–{end.strftime('%m/%d')}"

all_data     = load_data()
win_labels   = all_data["window_labels"]
windows_data = all_data["windows"]
default_window_index = (
    win_labels.index(DEFAULT_WINDOW_LABEL)
    if DEFAULT_WINDOW_LABEL in win_labels
    else min(1, max(len(win_labels) - 1, 0))
)

# ── Signal Radar welcome header ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="signal-badge">North America Consumer Trend Radar</div>
  <div class="hero-title">North America<br>Consumer Signal Radar</div>
  <div class="hero-sub">
    A TikTok Shop style signal room for weekly Reddit consumer movement, emerging product demand,
    and keyword evidence from real discussions.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Entry controls (centered) ─────────────────────────────────────────────────
st.markdown('<div class="nav-shell"><div class="nav-title">Choose signal week and workspace</div>', unsafe_allow_html=True)
nav_l, nav_r = st.columns(2)
with nav_l:
    selected_window = st.selectbox(
        "时间周期 / Period",
        options=win_labels,
        index=default_window_index,
        format_func=display_period_label,
        label_visibility="collapsed",
    )
with nav_r:
    selected_view = st.selectbox(
        "进入工作台 / Enter Workspace",
        options=[
            "趋势总览 / Trend Command Center",
            "飙升排查 / Spike Radar",
            "类目工作台 / Cluster Workbench",
            "预测实验室 / Forecast Lab",
        ],
        index=0,
        label_visibility="collapsed",
    )
st.markdown('</div>', unsafe_allow_html=True)

# ── Load selected window data ─────────────────────────────────────────────────
wd             = windows_data[selected_window]
stats_all      = wd["stats"]
stats_all      = stats_all[stats_all["category"] != "Other"].copy()
stats_all      = apply_dynamic_trend_direction(stats_all)
weekly_by_cat  = wd["weekly"]
cat_brand_data = wd["cat_brand_data"]
win            = wd["window"]
selected_window_label = display_period_label(selected_window)

rising_total    = (stats_all["trend_direction"] == "rising").sum()
stable_total    = (stats_all["trend_direction"] == "stable").sum()
declining_total = (stats_all["trend_direction"] == "declining").sum()
total_mentions  = int(stats_all["current_mentions"].sum()) if "current_mentions" in stats_all else 0
period_start_label  = display_period_start(selected_window, win).strftime("%Y/%m/%d")
period_end_label    = win["cur_end"].strftime("%Y/%m/%d")
compare_start_label = (pd.Timestamp(win["prev_start"]) + pd.Timedelta(days=1)).strftime("%Y/%m/%d")
compare_end_label   = win["prev_end"].strftime("%Y/%m/%d")

# ── KPI summary row ───────────────────────────────────────────────────────────
cluster_total = max(len(stats_all), 1)
st.markdown(f"""
<div class="trend-stat-grid">
  <div class="trend-stat">
    <div class="label">当前周期 Current Week</div>
    <div class="value period">{selected_window_label}</div>
    <div class="period-range">
      <div><span>开始 Start</span><strong>{period_start_label}</strong></div>
      <div><span>结束 End</span><strong>{period_end_label}</strong></div>
    </div>
  </div>
  <div class="trend-stat">
    <div class="label">活跃类目 Clusters</div>
    <div class="value">{len(stats_all):,}</div>
    <div class="delta">weekly taxonomy coverage</div>
  </div>
  <div class="trend-stat">
    <div class="label">上升 Rising</div>
    <div class="value" style="color:{COLORS['rising']}">{rising_total:,}</div>
    <div class="delta">{rising_total/cluster_total:.0%} of clusters</div>
  </div>
  <div class="trend-stat">
    <div class="label">平稳 Stable</div>
    <div class="value" style="color:#F8FAFC">{stable_total:,}</div>
    <div class="delta">{stable_total/cluster_total:.0%} of clusters</div>
  </div>
  <div class="trend-stat">
    <div class="label">下降 Declining</div>
    <div class="value" style="color:{COLORS['declining']}">{declining_total:,}</div>
    <div class="delta">{declining_total/cluster_total:.0%} of clusters</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<div class='chart-caption' style='margin-top:6px'>"
    f"<b>本期讨论量</b> {total_mentions:,} 条帖子 · "
    f"当前周期：开始 {period_start_label} / 结束 {period_end_label} · "
    f"对比周期：开始 {compare_start_label} / 结束 {compare_end_label} · "
    "关键词包含品牌、产品词、趋势词和用户关心的具体对象<br>"
    f"<b>动态阈值</b> Rising = 本周≥{TREND_THRESHOLDS['min_current_mentions']} 且 环比≥+{TREND_THRESHOLDS['min_rise_abs_delta']} "
    f"且 相对增长≥{TREND_THRESHOLDS['min_rise_relative_change']:.0%}；"
    f"Declining = 上周≥{TREND_THRESHOLDS['min_prior_mentions']} 且 环比≤-{TREND_THRESHOLDS['min_decline_abs_delta']} "
    f"且 相对下降≤-{TREND_THRESHOLDS['min_decline_relative_change']:.0%}；其他为 Stable。</div>",
    unsafe_allow_html=True,
)

st.divider()

def cat_label(c: str) -> str:
    return c.replace("_", " ").title()

# ════════════════════════════════════════════════════════════════════════════
# VIEW 1 — 趋势总览 / Trend Command Center
# ════════════════════════════════════════════════════════════════════════════
if selected_view == "趋势总览 / Trend Command Center":

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        dir_filter = st.multiselect(
            "趋势方向 / Direction",
            options=["rising", "stable", "declining"],
            default=["rising", "stable", "declining"],
            format_func=lambda x: f"{DIR_EMO[x]} {DIR_ZH[x]} · {DIR_EN[x]}",
        )
    with fc2:
        top_n = st.slider("Top N", 5, 60, 25, 5)

    stats = stats_all[stats_all["trend_direction"].isin(dir_filter)].copy()
    if stats.empty:
        st.warning("当前筛选无匹配类目，请重置。/ No clusters match current filters.")
        st.stop()

    top_stats = stats.head(top_n).copy()
    top_stats["cat_label"]   = top_stats["category"].apply(cat_label)
    top_stats["spike_label"] = top_stats["spike_ratio"].apply(lambda x: f"{x:.1f}x")

    with st.container():
        st.markdown(
            '<div class="chart-caption">📌 <b>综合趋势分 Trend Score</b> = '
            '本周增长（25%）+ 跨社区扩散（25%）+ 用户好感度（25%）+ 互动参与（25%），四维等权。'
            '条形颜色代表趋势方向。<br>'
            'Equal-weight weekly score: Spike(25%) + Reach(25%) + Sentiment(25%) + Engagement(25%). '
            'Bar color = trend direction.</div>',
            unsafe_allow_html=True,
        )
        fig_bar = go.Figure()
        for d in ["rising", "stable", "declining"]:
            sub = top_stats[top_stats["trend_direction"] == d]
            if sub.empty:
                continue
            fig_bar.add_trace(go.Bar(
                x=sub["trend_score"], y=sub["cat_label"],
                orientation="h", name=f"{DIR_EMO[d]} {DIR_ZH[d]}",
                marker_color=COLORS[d],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "综合趋势分 Score: %{x:.2f}<br>"
                    "本周热度增长 Spike: %{customdata[0]:.1f}x<br>"
                    "本周帖子量 Posts: %{customdata[1]:,}<extra></extra>"
                ),
                customdata=sub[["spike_ratio", "current_mentions"]].values,
            ))
        fig_bar.update_layout(
            barmode="overlay",
            height=max(420, top_n * 22),
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10, color="#C9D7EA")),
            xaxis=dict(title="综合趋势分 / Trend Score", range=[0, 1.15],
                       showgrid=True, gridcolor="rgba(201,215,234,.12)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(color="#D8F7F6")),
            font=dict(color="#E5EEF9"),
            plot_bgcolor="rgba(5,8,22,.62)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=60, t=30, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.container():
        st.markdown(
            '<div class="chart-caption">📌 <b>热度 × 扩散象限图 Spike × Reach</b>：'
            '横轴 = 本周热度增长强度，纵轴 = 跨社区扩散广度，气泡大小 = 帖子量。右上角为核心机会区。<br>'
            'X = weekly spike intensity, Y = cross-community reach, bubble = post volume.</div>',
            unsafe_allow_html=True,
        )
        top_stats["bsize"] = (
            top_stats["current_mentions"] / top_stats["current_mentions"].max() * 55 + 8
        ).round()
        fig_bub = go.Figure()
        for x0,y0,x1,y1,col in [(.5,.5,1.05,1.05,"rgba(0,242,169,.12)"),
                                  (0,.5,.5,1.05,"rgba(148,163,184,.10)"),
                                  (.5,0,1.05,.5,"rgba(0,242,234,.10)"),
                                  (0,0,.5,.5,"rgba(255,255,255,.04)")]:
            fig_bub.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                              fillcolor=col, opacity=.55, line_width=0)
        for d in ["rising", "stable", "declining"]:
            sub = top_stats[top_stats["trend_direction"] == d]
            if sub.empty:
                continue
            fig_bub.add_trace(go.Scatter(
                x=sub["normalized_spike"], y=sub["cross_community"],
                mode="markers+text",
                name=f"{DIR_EMO[d]} {DIR_ZH[d]}",
                marker=dict(size=sub["bsize"], color=COLORS[d], opacity=.75,
                            line=dict(width=1, color="rgba(255,255,255,.70)")),
                text=sub["cat_label"].str[:12],
                textposition="top center", textfont=dict(size=8, color="#E5EEF9"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "本周热度 Spike: %{x:.2f}<br>"
                    "跨圈扩散 Reach: %{y:.2f}<br>"
                    "帖子量 Posts: %{customdata:,}<extra></extra>"
                ),
                customdata=sub["current_mentions"],
            ))
        for xp, yp, lab in [
            (.75, .97, "🔥 高热 / 高扩  High Heat · Broad Reach"),
            (.18, .97, "广扩散 / 低热  Broad · Low Heat"),
            (.75, .03, "局部爆点  Local Spike"),
            (.18, .03, "低优先  Low Priority"),
        ]:
            fig_bub.add_annotation(x=xp, y=yp, text=lab, showarrow=False,
                                   font=dict(size=8.5, color="#A8B5C9"))
        fig_bub.add_hline(y=.5, line_dash="dot", line_color="rgba(201,215,234,.35)", line_width=1)
        fig_bub.add_vline(x=.5, line_dash="dot", line_color="rgba(201,215,234,.35)", line_width=1)
        fig_bub.update_layout(
            xaxis=dict(title="热度增长强度 / Spike Intensity", range=[-.05, 1.1],
                       showgrid=True, gridcolor="rgba(201,215,234,.12)"),
            yaxis=dict(title="跨社区扩散度 / Cross-community Reach", range=[-.05, 1.1],
                       showgrid=True, gridcolor="rgba(201,215,234,.12)"),
            height=500, plot_bgcolor="rgba(5,8,22,.62)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(color="#D8F7F6")),
            font=dict(color="#E5EEF9"),
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_bub, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 2 — 飙升排查 / Spike Radar
# ════════════════════════════════════════════════════════════════════════════
elif selected_view == "飙升排查 / Spike Radar":

    st.markdown(
        '<div class="chart-caption">📌 <b>飙升排查 Spike Radar</b>：'
        '仅按本周热度增长倍数排序，找最近爆发最快的类目。'
        '卡片内 Bar 长度 = 该关键词在本类目的本周提及量。<br>'
        'Ranked by weekly spike ratio only. Bar length = keyword mention count in current week.</div>',
        unsafe_allow_html=True,
    )

    t2c1, t2c2 = st.columns(2)
    with t2c1:
        t2_n_cats = st.slider("展示类目数 / Show N clusters", 4, 20, 6, 2, key="t2_n_cats")
    with t2c2:
        t2_n_kw = st.slider("每张卡展示关键词数 / Keywords per card", 3, 30, 5, 1, key="t2_n_kw")
    st.markdown("")

    top_cats = (
        stats_all[stats_all["trend_direction"] == "rising"]
        .nlargest(t2_n_cats, "spike_ratio")
        .copy()
    )
    if top_cats.empty:
        st.info("本周暂无上升类目。/ No rising clusters this week.")
    else:
        for i in range(0, len(top_cats), 2):
            col_a, col_b = st.columns(2)
            for j, col in enumerate([col_a, col_b]):
                idx = i + j
                if idx >= len(top_cats):
                    break
                row    = top_cats.iloc[idx]
                cat    = row["category"]
                dcolor = COLORS[row["trend_direction"]]
                dlabel = f"{DIR_EMO[row['trend_direction']]} {DIR_ZH[row['trend_direction']]} · {DIR_EN[row['trend_direction']]}"
                delta_sign = "+" if row["mentions_delta"] >= 0 else ""
                bdf = cat_brand_data.get(cat, pd.DataFrame())

                with col:
                    st.markdown(f"""
                    <div class="spike-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                        <span style="font-size:.7rem;color:#A8B5C9;font-weight:600">#{idx+1}</span>
                        <span style="background:{dcolor}18;color:{dcolor};border-radius:6px;
                              padding:2px 8px;font-size:.7rem;font-weight:600">{dlabel}</span>
                      </div>
                      <div style="font-size:1rem;font-weight:700;color:#F8FAFC;margin-bottom:6px">
                        {cat_label(cat)}
                      </div>
                      <div style="display:flex;gap:18px;margin-bottom:4px">
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;color:{dcolor}">{row['spike_ratio']:.1f}x</div>
                          <div style="font-size:.7rem;color:#A8B5C9">本周热度 Spike</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;color:#F8FAFC">{int(row['current_mentions']):,}</div>
                          <div style="font-size:.7rem;color:#A8B5C9">本周帖子 Posts</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;
                               color={'#00F2A9' if row['mentions_delta']>=0 else '#FF0050'}">
                            {delta_sign}{int(row['mentions_delta']):,}
                          </div>
                          <div style="font-size:.7rem;color:#A8B5C9">环比增减 vs Prior Week</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if not bdf.empty:
                        bdf_show = bdf.head(t2_n_kw).copy()
                        bdf_show["kw_short"] = bdf_show["brand"].str[:20]
                        fig_b = go.Figure(go.Bar(
                            x=bdf_show["cur_mentions"],
                            y=bdf_show["kw_short"],
                            orientation="h",
                            marker_color=COLORS["primary"],
                            opacity=0.8,
                            text=bdf_show["cur_mentions"].apply(lambda v: f"{v:,}"),
                            textposition="outside",
                            textfont=dict(size=9),
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "该关键词本周提及 Current: %{x:,}<br>"
                                "上周提及 Prior Week: %{customdata:,}<extra></extra>"
                            ),
                            customdata=bdf_show["prev_mentions"],
                        ))
                        fig_b.update_layout(
                            height=260,
                            xaxis=dict(title="关键词提及量 / Keyword Mentions",
                                       showgrid=True, gridcolor="rgba(201,215,234,.12)"),
                            yaxis=dict(categoryorder="total ascending",
                                       tickfont=dict(size=9, color="#C9D7EA")),
                            font=dict(color="#E5EEF9"),
                            plot_bgcolor="rgba(5,8,22,.62)", paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=50, t=5, b=25),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                    else:
                        st.caption("暂无关键词数据 / No keyword data")

# ════════════════════════════════════════════════════════════════════════════
# VIEW 3 — 类目工作台 / Cluster Workbench
# ════════════════════════════════════════════════════════════════════════════
elif selected_view == "类目工作台 / Cluster Workbench":

    DIR_ARROW = {"rising": "🟢 ↑", "stable": "⚫ →", "declining": "🔴 ↓"}
    _dir_map  = dict(zip(stats_all["category"], stats_all["trend_direction"]))

    def cat_label_arrow(cat: str) -> str:
        arrow = DIR_ARROW.get(_dir_map.get(cat, "stable"), "⚫ →")
        return f"{cat_label(cat)}  {arrow}"

    all_cats_sorted = sorted(stats_all["category"].tolist(), key=lambda c: cat_label(c).lower())

    t3c1, t3c2 = st.columns([3, 1])
    with t3c1:
        selected = st.selectbox(
            "选择趋势类目 / Select Trend Cluster （可输入搜索）",
            options=all_cats_sorted,
            index=0,
            format_func=cat_label_arrow,
        )
    with t3c2:
        t3_n = st.slider("展示关键词数 / Show N keywords", 3, 30, 5, 1, key="t3_n")

    row = stats_all[stats_all["category"] == selected].iloc[0]

    st.markdown(f"### {cat_label(selected)}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("综合趋势分\nTrend Score", f"{row['trend_score']:.2f}",
              delta=f"{DIR_EMO[row['trend_direction']]} {DIR_ZH[row['trend_direction']]}",
              delta_color="normal" if row["trend_direction"] == "rising"
                          else ("off" if row["trend_direction"] == "stable" else "inverse"))
    c2.metric("本周热度增长\nWeekly Spike", f"{row['spike_ratio']:.1f}x",
              delta=f"{int(row['current_mentions']):,} 本周帖子/posts")
    c3.metric("用户好感度\nSentiment",
              f"{'+' if row['mean_sentiment'] >= 0 else ''}{row['mean_sentiment']:.2f}",
              delta="正面 Positive" if row["mean_sentiment"] >= 0.05
                    else ("中性 Neutral" if row["mean_sentiment"] >= -0.05 else "负面 Negative"),
              delta_color="normal" if row["mean_sentiment"] >= 0.05
                          else ("off" if row["mean_sentiment"] >= -0.05 else "inverse"))
    c4.metric("互动参与倍数\nEngagement", f"{row['eng_momentum']:.1f}x",
              delta="vs 上周 prior week", delta_color="off")
    c5.metric("活跃社区数\nActive Communities", f"{int(row['current_communities'])}",
              delta="subreddits", delta_color="off")

    st.markdown("")

    # Full keyword list (for hot posts filter) vs display list (capped at t3_n)
    bdf_all      = cat_brand_data.get(selected, pd.DataFrame())
    bdf_filtered = bdf_all.head(t3_n) if not bdf_all.empty else bdf_all

    b_col, w_col = st.columns([5, 4])

    with b_col:
        st.markdown(
            '<div class="chart-caption">📌 <b>关键词提及量 Keyword Mentions</b>：'
            '条形长度 = 本周提及帖子数；条形颜色 = 该关键词帖子平均好感度（连续色阶，红→黄→绿）。'
            '关键词包含品牌、产品词、趋势词和用户关心的具体对象。<br>'
            'Bar = current week posts. Color = avg sentiment (RdYlGn). '
            'Keywords include brands, product terms, trend terms, and user concern objects.</div>',
            unsafe_allow_html=True,
        )
        if not bdf_filtered.empty:
            bdf_show = bdf_filtered.copy()
            bdf_show["kw_short"] = bdf_show["brand"].str[:22]
            bdf_show["b_spike"]  = (bdf_show["cur_mentions"] / bdf_show["prev_mentions"].replace(0, 1)).round(2)

            fig_kw = go.Figure(go.Bar(
                x=bdf_show["cur_mentions"],
                y=bdf_show["kw_short"],
                orientation="h",
                marker=dict(
                    color=bdf_show["avg_sentiment"],
                    colorscale="RdYlGn",
                    cmin=-1, cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="好感度<br>Sentiment", side="right", font=dict(size=11)),
                        thickness=14, len=0.85,
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1<br>负面", "-0.5", "0<br>中性", "0.5", "+1<br>正面"],
                        tickfont=dict(size=9), outlinewidth=0,
                    ),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "本周提及 Current: %{x:,}<br>"
                    "上周提及 Prior: %{customdata[0]:,}<br>"
                    "好感度 Sentiment: %{customdata[1]:+.2f}<br>"
                    "增长倍数 Spike: %{customdata[2]:.1f}x<extra></extra>"
                ),
                customdata=bdf_show[["prev_mentions", "avg_sentiment", "b_spike"]].values,
            ))
            fig_kw.update_layout(
                height=max(260, t3_n * 46),
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10, color="#C9D7EA")),
                xaxis=dict(title="本周关键词提及量 / Weekly Keyword Mentions",
                           showgrid=True, gridcolor="rgba(201,215,234,.12)"),
                font=dict(color="#E5EEF9"),
                plot_bgcolor="rgba(5,8,22,.62)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=80, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("该类目暂无识别到的关键词数据。/ No keyword data identified for this cluster.")

    with w_col:
        st.markdown(
            '<div class="chart-caption">📌 <b>关键词数据矩阵 Keyword Matrix</b>：'
            '本周提及量、环比增减、平均好感度。<br>'
            'Current week mentions, week-over-week delta, and avg sentiment per keyword.</div>',
            unsafe_allow_html=True,
        )
        if not bdf_filtered.empty:
            bdf_matrix = bdf_filtered.copy()
            bdf_matrix["delta"] = (bdf_matrix["cur_mentions"] - bdf_matrix["prev_mentions"]).astype(int)

            def fmt_delta(d):
                color = "#22C55E" if d > 0 else ("#EF4444" if d < 0 else "#94A3B8")
                sign  = "+" if d > 0 else ""
                return f'<span style="color:{color};font-weight:600">{sign}{d:,}</span>'

            def fmt_sent(s):
                if s >= 0.05:    color, label = "#22C55E", f"+{s:.2f}"
                elif s <= -0.05: color, label = "#EF4444", f"{s:.2f}"
                else:            color, label = "#94A3B8", f"{s:.2f}"
                return f'<span style="color:{color};font-weight:600">{label}</span>'

            rows_html = ""
            for _, r in bdf_matrix.iterrows():
                rows_html += f"""
                <tr>
                  <td style="font-weight:500;max-width:110px;overflow:hidden;
                              text-overflow:ellipsis;white-space:nowrap">{r['brand']}</td>
                  <td style="text-align:right">{int(r['cur_mentions']):,}</td>
                  <td style="text-align:right">{int(r['prev_mentions']):,}</td>
                  <td style="text-align:right">{fmt_delta(r['delta'])}</td>
                  <td style="text-align:right">{fmt_sent(r['avg_sentiment'])}</td>
                </tr>"""

            st.markdown(f"""
            <style>
            .kw-matrix{{background:rgba(8,13,28,.56);border:1px solid rgba(0,242,234,.16);
              border-radius:14px;overflow:hidden;backdrop-filter:blur(16px);}}
            .kw-matrix table{{width:100%;border-collapse:collapse;font-size:.82rem;}}
            .kw-matrix th{{background:rgba(0,242,234,.08);padding:7px 10px;text-align:right;
              font-weight:600;color:#D8F7F6;font-size:.75rem;border-bottom:1px solid rgba(201,215,234,.14);}}
            .kw-matrix th:first-child{{text-align:left;}}
            .kw-matrix td{{padding:6px 10px;border-bottom:1px solid rgba(201,215,234,.10);color:#F8FAFC;}}
            .kw-matrix tr:last-child td{{border-bottom:none;}}
            .kw-matrix tr:hover td{{background:rgba(255,255,255,.05);}}
            </style>
            <div class="kw-matrix">
            <table>
              <thead><tr>
                <th style="text-align:left">关键词 Keyword</th>
                <th>本周 Cur</th>
                <th>上周 Prev</th>
                <th>环比 Delta</th>
                <th>好感度 Sent</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("暂无数据 / No data")

    # ── Hot Posts ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📄 近期热帖 / Recent Hot Posts**")

    pa_col, pb_col = st.columns([2, 1])
    with pa_col:
        # Use FULL keyword list (bdf_all), not the top-N display list (bdf_filtered)
        keyword_options = ["全部关键词 / All Keywords"] + (
            bdf_all["brand"].tolist() if not bdf_all.empty else []
        )
        keyword_filter = st.selectbox(
            "按关键词筛选热帖 / Filter by Keyword",
            options=keyword_options,
            key="t3_keyword_filter",
        )
    with pb_col:
        t3_post_n = st.slider("展示帖子数 / Show N posts", 5, 30, 10, 5, key="t3_post_n")

    brand_idx = load_brand_posts_index()

    if keyword_filter != "全部关键词 / All Keywords" and brand_idx is not None:
        records   = brand_idx.get(selected, {}).get(keyword_filter, [])
        cat_posts = pd.DataFrame(records) if records else pd.DataFrame()
        if not cat_posts.empty:
            cat_posts["published_at"] = pd.to_datetime(cat_posts["published_at"], errors="coerce")
    elif keyword_filter == "全部关键词 / All Keywords" and brand_idx is not None:
        all_records = []
        for posts in brand_idx.get(selected, {}).values():
            all_records.extend(posts)
        cat_posts = (
            pd.DataFrame(all_records).drop_duplicates(subset=["url"])
            if all_records else pd.DataFrame()
        )
        if not cat_posts.empty:
            cat_posts["published_at"] = pd.to_datetime(cat_posts["published_at"], errors="coerce")
    else:
        sample    = row["sample_posts"] if isinstance(row.get("sample_posts"), list) else []
        cat_posts = pd.DataFrame(sample) if sample else pd.DataFrame()
        if not cat_posts.empty:
            cat_posts["published_at"] = pd.NaT

    if not cat_posts.empty and "engagement_score" in cat_posts.columns:
        cat_posts = cat_posts.sort_values("engagement_score", ascending=False)
    cat_posts = cat_posts.head(t3_post_n)

    if cat_posts.empty:
        st.info("该筛选条件下暂无帖子。/ No posts found for this filter.")
    else:
        sent_color_map = {"positive": COLORS["rising"], "negative": COLORS["declining"], "neutral": COLORS["stable"]}
        sent_txt_map   = {"positive": "😊 正面", "negative": "😟 负面", "neutral": "😐 中性"}
        html_rows = ""
        for _, p in cat_posts.iterrows():
            title    = str(p.get("title", ""))[:120]
            comm     = p.get("community", "")
            score    = p.get("engagement_score", 0)
            label    = p.get("sentiment_label", "neutral")
            url      = p.get("url", "")
            date_str = p["published_at"].strftime("%m/%d") if pd.notna(p.get("published_at")) else ""
            sc       = sent_color_map.get(label, COLORS["stable"])
            st_txt   = sent_txt_map.get(label, "")
            link     = f'<a href="{url}" target="_blank" style="color:#F8FAFC;text-decoration:none">{title}</a>' if url else title
            html_rows += f"""
            <div style="padding:10px 0;border-bottom:1px solid rgba(201,215,234,.12)">
              <div style="font-size:.85rem;margin-bottom:3px;line-height:1.4">{link} <span style="font-size:.7rem;color:#00F2EA">↗</span></div>
              <span style="font-size:.72rem;color:#A8B5C9">r/{comm}</span>
              <span style="font-size:.72rem;color:#A8B5C9;margin-left:10px">📅 {date_str}</span>
              <span style="font-size:.72rem;color:#A8B5C9;margin-left:10px">互动 {int(score):,}</span>
              <span style="font-size:.72rem;color:{sc};margin-left:10px">{st_txt}</span>
            </div>"""
        st.markdown(f'<div style="max-height:520px;overflow-y:auto;background:rgba(8,13,28,.52);border:1px solid rgba(0,242,234,.14);border-radius:14px;padding:2px 12px;backdrop-filter:blur(16px)">{html_rows}</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 4 — 预测实验室 / Forecast Lab
# ════════════════════════════════════════════════════════════════════════════
elif selected_view == "预测实验室 / Forecast Lab":

    st.markdown(
        """
        <div class="wip-banner">
          <div class="wip-pill">WIP</div>
          <div class="wip-title">模型训练中 / Model Training In Progress</div>
          <div class="wip-copy">
            当前 tab 仅作为预测效果展示，不作为最终数据或正式业务判断依据。
            This page is a preview of the forecasting experience while the model is still being trained.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chart-caption">📌 <b>预测说明 Forecast</b>：'
        'XGBoost 模型基于近6个月的周度提及量、情绪、社区扩散等特征，预测各类目在未来2-4周的趋势。'
        'Prophet 时序模型展示历史走势 + 未来4周预测区间（灰色阴影）。<br>'
        'Left: XGBoost weekly rising probability for next 2–4 weeks. '
        'Right: Prophet trend + 4-week forecast (shaded = confidence interval).</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if not FORECAST_PATH.exists():
        st.warning("预测数据未生成。请先运行 `python scripts/build_forecast.py`。")
    else:
        @st.cache_data
        def load_forecast():
            with open(FORECAST_PATH, "rb") as f:
                return pickle.load(f)

        fdata       = load_forecast()
        xgb_preds   = fdata["xgb_predictions"]
        prophet_fcs = fdata["prophet_forecasts"]

        st.markdown("#### 预测结果 / Predictions")
        tab_2w, tab_4w = st.tabs(["未来2周 / Next 2 Weeks", "未来4周 / Next 4 Weeks"])

        def render_prediction_list(prob_col: str) -> None:
            rising_cats    = xgb_preds[xgb_preds[prob_col] >= 0.90]["category"].tolist()
            declining_cats = xgb_preds[xgb_preds[prob_col] <= 0.10]["category"].tolist()
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**🟢 预计上涨 / Likely Rising**")
                if rising_cats:
                    items = "".join(
                        f'<div style="padding:5px 10px;margin:3px 0;background:rgba(0,242,169,.10);'
                        f'border-left:3px solid {COLORS["rising"]};border-radius:4px;'
                        f'font-size:.85rem;color:#F8FAFC">{cat_label(c)}</div>'
                        for c in rising_cats
                    )
                    st.markdown(items, unsafe_allow_html=True)
                else:
                    st.caption("无高置信度上涨类目 / None above threshold")
            with rc2:
                st.markdown("**🔴 预计下行 / Likely Declining**")
                if declining_cats:
                    items = "".join(
                        f'<div style="padding:5px 10px;margin:3px 0;background:rgba(255,0,80,.10);'
                        f'border-left:3px solid {COLORS["declining"]};border-radius:4px;'
                        f'font-size:.85rem;color:#F8FAFC">{cat_label(c)}</div>'
                        for c in declining_cats
                    )
                    st.markdown(items, unsafe_allow_html=True)
                else:
                    st.caption("无高置信度下行类目 / None below threshold")

        with tab_2w:
            render_prediction_list("p2")
        with tab_4w:
            render_prediction_list("p4")

        st.markdown("---")
        st.markdown("#### Prophet 趋势预测 / Trend Forecast")
        prophet_cats = [c for c in xgb_preds["category"].tolist() if c in prophet_fcs]
        selected_fc  = st.selectbox(
            "选择类目 / Select Cluster",
            options=prophet_cats,
            format_func=cat_label,
            key="fc_cat",
        )

        if selected_fc and selected_fc in prophet_fcs:
            fc = prophet_fcs[selected_fc].copy()
            fc["ds"] = pd.to_datetime(fc["ds"])
            hist = fc[fc["actual"].notna()].copy()
            pred = fc.tail(4).copy()

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(37,99,235,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="预测区间",
            ))
            fig_p.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"],
                mode="lines",
                line=dict(color=COLORS["primary"], width=2, dash="dot"),
                name="Prophet 拟合/预测",
            ))
            fig_p.add_trace(go.Scatter(
                x=hist["ds"], y=hist["actual"],
                mode="lines+markers",
                line=dict(color="#F8FAFC", width=2),
                marker=dict(size=5, color="#F8FAFC"),
                name="实际提及量 Actual",
            ))
            if not pred.empty:
                fig_p.add_vrect(
                    x0=pred["ds"].iloc[0], x1=pred["ds"].iloc[-1],
                    fillcolor="rgba(34,197,94,0.07)",
                    layer="below", line_width=0,
                    annotation_text="预测区间", annotation_position="top left",
                    annotation_font_size=10, annotation_font_color="#22C55E",
                )

            x_min = fc["ds"].min()
            x_max = fc["ds"].max()
            for ts, te, tlabel in TIKTOK_EVENTS:
                ts_dt, te_dt = pd.Timestamp(ts), pd.Timestamp(te)
                if te_dt < x_min or ts_dt > x_max:
                    continue
                fig_p.add_vrect(
                    x0=ts_dt, x1=te_dt,
                    fillcolor="rgba(249,115,22,0.12)",
                    layer="below", line_width=0,
                    annotation_text=tlabel, annotation_position="bottom left",
                    annotation_font_size=8, annotation_font_color="#EA580C",
                )

            visible_holidays = [
                (pd.Timestamp(hdate), hlabel)
                for hdate, hlabel in US_HOLIDAYS.items()
                if x_min <= pd.Timestamp(hdate) <= x_max
            ]
            YSHIFTS = [0, -18, -36]
            for i, (hdt, hlabel) in enumerate(visible_holidays):
                fig_p.add_vline(
                    x=hdt.timestamp() * 1000,
                    line_dash="dot", line_color="#A78BFA", line_width=1.2,
                    annotation=dict(
                        text=hlabel, font=dict(size=8, color="#7C3AED"),
                        bgcolor="rgba(8,13,28,0.75)",
                        borderpad=2, yanchor="top",
                        yshift=YSHIFTS[i % 3],
                    ),
                    annotation_position="top left",
                )

            fig_p.add_hline(y=1.0, line_dash="dash", line_color="#94A3B8", line_width=1,
                            annotation_text="持平 flat",
                            annotation_font_size=9, annotation_font_color="#94A3B8")
            fig_p.update_layout(
                height=560,
                xaxis=dict(title="周 / Week", showgrid=True, gridcolor="rgba(201,215,234,.12)"),
                yaxis=dict(title="周环比增长倍数 / Weekly Spike (1.0 = flat)",
                           showgrid=True, gridcolor="rgba(201,215,234,.12)"),
                font=dict(color="#E5EEF9"),
                plot_bgcolor="rgba(5,8,22,.62)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=20, t=90, b=40),
                legend=dict(orientation="h", y=-0.10, font=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.markdown(
                '<div class="chart-caption">'
                '🟣 紫色虚线 = 美国节假日 &nbsp;|&nbsp; 🟠 橙色底纹 = TikTok Shop 站内活动 &nbsp;|&nbsp; '
                '🟢 绿色底纹 = Prophet 预测区间（未来4周）'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("该类目暂无 Prophet 预测数据。")
