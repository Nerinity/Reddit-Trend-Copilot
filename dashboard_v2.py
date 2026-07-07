#!/usr/bin/env python3
"""
Reddit Trend Intelligence Copilot — Dashboard v2
Workflow: Trend Command Center | Spike Radar | Cluster Workbench | Forecast Lab
"""
import pickle
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
    "rising":   "#22C55E",
    "stable":   "#94A3B8",
    "declining":"#EF4444",
    "primary":  "#2563EB",
}

DIR_ZH  = {"rising":"上升中","stable":"平稳","declining":"下降中"}
DIR_EN  = {"rising":"Rising","stable":"Stable","declining":"Declining"}
DIR_EMO = {"rising":"🟢","stable":"⚪","declining":"🔴"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{
  background:
    radial-gradient(circle at 18% 12%, rgba(255,0,102,.12), transparent 28%),
    radial-gradient(circle at 88% 4%, rgba(0,210,255,.14), transparent 32%),
    radial-gradient(circle at 72% 88%, rgba(132,92,255,.12), transparent 30%),
    linear-gradient(135deg,#fff7fb 0%,#f8fbff 45%,#f7fff9 100%);
}
.stApp:before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(15,23,42,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,23,42,.045) 1px, transparent 1px);
  background-size:32px 32px;
  mask-image:linear-gradient(to bottom,rgba(0,0,0,.55),rgba(0,0,0,.08));
}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1480px;position:relative;z-index:1;}
div[data-testid="metric-container"]{
  background:rgba(255,255,255,.64);border:1px solid rgba(255,255,255,.72);
  border-radius:16px;padding:.9rem 1.1rem;
  box-shadow:0 16px 42px rgba(15,23,42,.08);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);}
div[data-testid="metric-container"] label{font-size:.75rem;color:#6B7280;}
.hero{
  border:1px solid rgba(255,255,255,.70);
  border-radius:24px;padding:24px 26px;margin-bottom:18px;
  background:linear-gradient(135deg,rgba(255,255,255,.78),rgba(255,255,255,.48));
  box-shadow:0 24px 70px rgba(15,23,42,.11);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
}
.hero-title{font-size:2.05rem;font-weight:760;color:#111827;letter-spacing:0;margin:0;}
.hero-sub{font-size:.94rem;color:#475569;margin-top:6px;line-height:1.5;}
.hero-kpis{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin-top:18px;}
.hero-kpi{
  background:rgba(255,255,255,.58);border:1px solid rgba(255,255,255,.78);
  border-radius:16px;padding:12px 14px;min-height:82px;
}
.hero-kpi .label{font-size:.72rem;color:#64748B;font-weight:650;text-transform:uppercase;letter-spacing:.02em;}
.hero-kpi .value{font-size:1.55rem;color:#0F172A;font-weight:760;margin-top:4px;}
.chart-caption{
  font-size:.78rem;color:#475569;margin-bottom:.65rem;line-height:1.5;
  padding:9px 12px;background:rgba(255,255,255,.58);border-left:3px solid #2563EB;
  border-radius:12px;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);}
.spike-card{
  background:rgba(255,255,255,.62);border:1px solid rgba(255,255,255,.76);border-radius:18px;
  padding:1rem 1.2rem;margin-bottom:.8rem;
  box-shadow:0 18px 50px rgba(15,23,42,.10);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);}
section[data-testid="stSidebar"],div[data-testid="stSelectbox"] div[data-baseweb="select"]>div,
div[data-testid="stMultiSelect"] div[data-baseweb="select"]>div{
  background:rgba(255,255,255,.72)!important;
  border-color:rgba(255,255,255,.80)!important;
  border-radius:14px!important;
}
</style>
""", unsafe_allow_html=True)

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

all_data     = load_data()
win_labels   = all_data["window_labels"]
windows_data = all_data["windows"]

# ── Static hero header (no data needed) ───────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap">
    <div>
      <div class="hero-title">Reddit Trend Intelligence Copilot</div>
      <div class="hero-sub">
        Weekly consumer signal radar for product opportunities, trend movement, and keyword evidence.<br>
        每周消费趋势信号雷达，追踪类目升降、讨论热点、关键词证据和用户真实关注点。
      </div>
    </div>
    <div style="min-width:220px;text-align:right;color:#64748B;font-size:.82rem;line-height:1.5">
      <b style="color:#111827">Weekly Operating View</b><br>
      只保留近四周主看板数据 · 四周前数据自动归档
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Global controls row (always visible) ──────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 3, 1])
with ctrl1:
    selected_window = st.selectbox(
        "📅 时间周期 / Period",
        options=win_labels,
        index=0,
    )
with ctrl2:
    selected_view = st.selectbox(
        "🔭 工作台 / Workflow",
        options=[
            "趋势总览 / Trend Command Center",
            "飙升排查 / Spike Radar",
            "类目工作台 / Cluster Workbench",
            "预测实验室 / Forecast Lab",
        ],
        index=0,
    )
with ctrl3:
    dir_filter = st.multiselect(
        "趋势方向 / Direction",
        options=["rising", "stable", "declining"],
        default=["rising", "stable", "declining"],
        format_func=lambda x: f"{DIR_EMO[x]} {DIR_ZH[x]} · {DIR_EN[x]}",
    )
with ctrl4:
    top_n = st.slider("Top N", 5, 60, 25, 5)

# ── Load selected window data ─────────────────────────────────────────────────
wd             = windows_data[selected_window]
stats_all      = wd["stats"]
stats_all      = stats_all[stats_all["category"] != "Other"].copy()
weekly_by_cat  = wd["weekly"]
cat_brand_data = wd["cat_brand_data"]
win            = wd["window"]

rising_total    = (stats_all["trend_direction"] == "rising").sum()
stable_total    = (stats_all["trend_direction"] == "stable").sum()
declining_total = (stats_all["trend_direction"] == "declining").sum()
total_mentions  = int(stats_all["current_mentions"].sum()) if "current_mentions" in stats_all else 0
period_label    = f"{win['cur_start'].strftime('%Y/%m/%d')} → {win['cur_end'].strftime('%Y/%m/%d')}"
compare_label   = f"{win['prev_start'].strftime('%Y/%m/%d')} → {win['prev_end'].strftime('%Y/%m/%d')}"

# ── KPI summary row ───────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📅 分析区间", selected_window, delta=period_label, delta_color="off")
k2.metric("活跃类目 Clusters", f"{len(stats_all):,}")
k3.metric("🟢 上升 Rising", f"{rising_total:,}", delta=f"{rising_total/max(len(stats_all),1):.0%}")
k4.metric("⚪ 平稳 Stable", f"{stable_total:,}", delta=f"{stable_total/max(len(stats_all),1):.0%}", delta_color="off")
k5.metric("🔴 下降 Declining", f"{declining_total:,}", delta=f"{declining_total/max(len(stats_all),1):.0%}", delta_color="inverse")

st.markdown(
    f"<div class='chart-caption' style='margin-top:6px'>"
    f"<b>本期讨论量</b> {total_mentions:,} 条帖子 · 对比区间 {compare_label} · "
    "关键词包含品牌、产品词、趋势词和用户关心的具体对象</div>",
    unsafe_allow_html=True,
)

st.divider()

def cat_label(c: str) -> str:
    return c.replace("_", " ").title()

# ════════════════════════════════════════════════════════════════════════════
# VIEW 1 — 趋势总览 / Trend Command Center
# ════════════════════════════════════════════════════════════════════════════
if selected_view == "趋势总览 / Trend Command Center":

    stats = stats_all[stats_all["trend_direction"].isin(dir_filter)].copy()
    if stats.empty:
        st.warning("当前筛选无匹配类目，请重置。/ No clusters match current filters.")
        st.stop()

    top_stats = stats.head(top_n).copy()
    top_stats["cat_label"]   = top_stats["category"].apply(cat_label)
    top_stats["spike_label"] = top_stats["spike_ratio"].apply(lambda x: f"{x:.1f}x")

    g1, g2 = st.columns([5, 4])

    with g1:
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
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
            xaxis=dict(title="综合趋势分 / Trend Score", range=[0, 1.15]),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=60, t=30, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
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
        for x0,y0,x1,y1,col in [(.5,.5,1.05,1.05,"#F0FDF4"),
                                  (0,.5,.5,1.05,"#F8FAFC"),
                                  (.5,0,1.05,.5,"#EFF6FF"),
                                  (0,0,.5,.5,"#FAFAFA")]:
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
                            line=dict(width=1, color="white")),
                text=sub["cat_label"].str[:12],
                textposition="top center", textfont=dict(size=8),
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
                                   font=dict(size=8.5, color="#94A3B8"))
        fig_bub.add_hline(y=.5, line_dash="dot", line_color="#CBD5E1", line_width=1)
        fig_bub.add_vline(x=.5, line_dash="dot", line_color="#CBD5E1", line_width=1)
        fig_bub.update_layout(
            xaxis=dict(title="热度增长强度 / Spike Intensity", range=[-.05, 1.1]),
            yaxis=dict(title="跨社区扩散度 / Cross-community Reach", range=[-.05, 1.1]),
            height=500, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
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
                        <span style="font-size:.7rem;color:#6B7280;font-weight:600">#{idx+1}</span>
                        <span style="background:{dcolor}18;color:{dcolor};border-radius:6px;
                              padding:2px 8px;font-size:.7rem;font-weight:600">{dlabel}</span>
                      </div>
                      <div style="font-size:1rem;font-weight:700;color:#111827;margin-bottom:6px">
                        {cat_label(cat)}
                      </div>
                      <div style="display:flex;gap:18px;margin-bottom:4px">
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;color:{dcolor}">{row['spike_ratio']:.1f}x</div>
                          <div style="font-size:.7rem;color:#6B7280">本周热度 Spike</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;color:#111">{int(row['current_mentions']):,}</div>
                          <div style="font-size:.7rem;color:#6B7280">本周帖子 Posts</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;
                               color={'#22C55E' if row['mentions_delta']>=0 else '#EF4444'}">
                            {delta_sign}{int(row['mentions_delta']):,}
                          </div>
                          <div style="font-size:.7rem;color:#6B7280">环比增减 vs Prior Week</div>
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
                            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=9)),
                            xaxis=dict(title="关键词提及量 / Keyword Mentions",
                                       showgrid=True, gridcolor="#F3F4F6"),
                            plot_bgcolor="white", paper_bgcolor="white",
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
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
                xaxis=dict(title="本周关键词提及量 / Weekly Keyword Mentions",
                           showgrid=True, gridcolor="#F3F4F6"),
                plot_bgcolor="white", paper_bgcolor="white",
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
            .kw-matrix table{{width:100%;border-collapse:collapse;font-size:.82rem;}}
            .kw-matrix th{{background:#F8FAFC;padding:7px 10px;text-align:right;
              font-weight:600;color:#374151;font-size:.75rem;border-bottom:2px solid #E5E7EB;}}
            .kw-matrix th:first-child{{text-align:left;}}
            .kw-matrix td{{padding:6px 10px;border-bottom:1px solid #F3F4F6;color:#111;}}
            .kw-matrix tr:last-child td{{border-bottom:none;}}
            .kw-matrix tr:hover td{{background:#F9FAFB;}}
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
            link     = f'<a href="{url}" target="_blank" style="color:#111;text-decoration:none">{title}</a>' if url else title
            html_rows += f"""
            <div style="padding:8px 0;border-bottom:1px solid #F3F4F6">
              <div style="font-size:.85rem;margin-bottom:3px;line-height:1.4">{link} <span style="font-size:.7rem;color:#2563EB">↗</span></div>
              <span style="font-size:.72rem;color:#6B7280">r/{comm}</span>
              <span style="font-size:.72rem;color:#6B7280;margin-left:10px">📅 {date_str}</span>
              <span style="font-size:.72rem;color:#6B7280;margin-left:10px">互动 {int(score):,}</span>
              <span style="font-size:.72rem;color:{sc};margin-left:10px">{st_txt}</span>
            </div>"""
        st.markdown(f'<div style="max-height:520px;overflow-y:auto">{html_rows}</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VIEW 4 — 预测实验室 / Forecast Lab
# ════════════════════════════════════════════════════════════════════════════
elif selected_view == "预测实验室 / Forecast Lab":

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
                        f'<div style="padding:5px 10px;margin:3px 0;background:#F0FDF4;'
                        f'border-left:3px solid {COLORS["rising"]};border-radius:4px;'
                        f'font-size:.85rem;color:#111">{cat_label(c)}</div>'
                        for c in rising_cats
                    )
                    st.markdown(items, unsafe_allow_html=True)
                else:
                    st.caption("无高置信度上涨类目 / None above threshold")
            with rc2:
                st.markdown("**🔴 预计下行 / Likely Declining**")
                if declining_cats:
                    items = "".join(
                        f'<div style="padding:5px 10px;margin:3px 0;background:#FFF1F2;'
                        f'border-left:3px solid {COLORS["declining"]};border-radius:4px;'
                        f'font-size:.85rem;color:#111">{cat_label(c)}</div>'
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
                line=dict(color="#111827", width=2),
                marker=dict(size=5, color="#111827"),
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
                        bgcolor="rgba(255,255,255,0.75)",
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
                xaxis=dict(title="周 / Week", showgrid=True, gridcolor="#F3F4F6"),
                yaxis=dict(title="周环比增长倍数 / Weekly Spike (1.0 = flat)",
                           showgrid=True, gridcolor="#F3F4F6"),
                plot_bgcolor="white", paper_bgcolor="white",
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
