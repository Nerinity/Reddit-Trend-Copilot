#!/usr/bin/env python3
"""
Reddit Product Trend Copilot — Dashboard v2
Tab1 趋势品类目 / Tab2 飙升榜 Top 6 / Tab3 品类详情
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit Trend Copilot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).parent / "data" / "processed" / "dashboard_data_500k.pkl"

COLORS = {
    "rising":   "#22C55E",
    "stable":   "#94A3B8",
    "declining":"#EF4444",
    "primary":  "#2563EB",
    "bg":       "#F8FAFC",
}

DIR_ZH  = {"rising":"上升中","stable":"平稳","declining":"下降中"}
DIR_EN  = {"rising":"Rising","stable":"Stable","declining":"Declining"}
DIR_EMO = {"rising":"🟢","stable":"⚪","declining":"🔴"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.block-container{padding-top:1rem;padding-bottom:2rem;}
div[data-testid="metric-container"]{
  background:white;border:1px solid #E5E7EB;
  border-radius:10px;padding:.9rem 1.1rem;}
div[data-testid="metric-container"] label{font-size:.75rem;color:#6B7280;}
.chart-caption{
  font-size:.78rem;color:#6B7280;margin-bottom:.4rem;line-height:1.5;
  padding:6px 10px;background:#F8FAFC;border-left:3px solid #2563EB;
  border-radius:0 6px 6px 0;}
.spike-card{
  background:white;border:1px solid #E5E7EB;border-radius:12px;
  padding:1rem 1.2rem;margin-bottom:.8rem;
  box-shadow:0 1px 4px rgba(0,0,0,.06);}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)

all_data     = load_data()
win_labels   = all_data["window_labels"]
windows_data = all_data["windows"]

# ── Header ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3,2])
with hc1:
    st.markdown("## 📡 Reddit Product Trend Copilot")
    st.caption("Reddit 社区讨论产品趋势雷达  ·  Reddit Community Trend Radar")

with hc2:
    # Period selector
    selected_window = st.selectbox(
        "📅 当前分析周期 / Analysis Period",
        options=win_labels,
        index=0,
        help="选择要分析的双周时间段，将同时更新所有图表\nSelect the biweekly period to analyze — updates all charts"
    )

wd = windows_data[selected_window]
stats_all     = wd["stats"]
weekly_by_cat = wd["weekly"]
cat_brand_data= wd["cat_brand_data"]
win           = wd["window"]

# Show window dates below selector
with hc2:
    st.caption(
        f"当前 Current: {win['cur_start'].strftime('%Y/%m/%d')} → {win['cur_end'].strftime('%Y/%m/%d')}  "
        f"｜  对比 Compare: {win['prev_start'].strftime('%Y/%m/%d')} → {win['prev_end'].strftime('%Y/%m/%d')}"
    )

# ── Global filters ────────────────────────────────────────────────────────────
fc1, fc2 = st.columns([2,1])
with fc1:
    dir_filter = st.multiselect(
        "趋势方向 / Trend Direction",
        options=["rising","stable","declining"],
        default=["rising","stable","declining"],
        format_func=lambda x: f"{DIR_EMO[x]} {DIR_ZH[x]} · {DIR_EN[x]}",
    )
with fc2:
    top_n = st.slider("展示品类数 / Top N Categories", 10, 50, 25, 5)

stats = stats_all[stats_all["trend_direction"].isin(dir_filter)].copy()
if stats.empty:
    st.warning("当前筛选无匹配品类，请重置。/ No categories match current filters.")
    st.stop()

st.divider()

# ── Helpers ───────────────────────────────────────────────────────────────────
def cat_label(c: str) -> str:
    return c.replace("_"," ").title()

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — 趋势品类目
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 趋势品类目 / Trends",
    "🚀 飙升榜 / Spike Board",
    "🔍 品类详情 / Category Detail",
])

with tab1:
    # Metric cards
    r1, r2, r3, r4 = st.columns(4)
    rising_n    = (stats["trend_direction"]=="rising").sum()
    stable_n    = (stats["trend_direction"]=="stable").sum()
    declining_n = (stats["trend_direction"]=="declining").sum()
    r1.metric("活跃品类 / Active Categories", len(stats))
    r2.metric("🟢 上升 Rising",   rising_n,    delta=f"{rising_n/len(stats):.0%}")
    r3.metric("⚪ 平稳 Stable",   stable_n,    delta=f"{stable_n/len(stats):.0%}",    delta_color="off")
    r4.metric("🔴 下降 Declining",declining_n, delta=f"{declining_n/len(stats):.0%}", delta_color="inverse")

    st.markdown("")
    top_stats = stats.head(top_n).copy()
    top_stats["cat_label"] = top_stats["category"].apply(cat_label)
    top_stats["spike_label"] = top_stats["spike_ratio"].apply(lambda x: f"{x:.1f}x")

    g1, g2 = st.columns([5,4])

    with g1:
        st.markdown('<div class="chart-caption">📌 <b>综合趋势分 Trend Score</b> = 热度增长（25%）+ 跨社区扩散（25%）+ 用户好感度（25%）+ 互动参与（25%），四维等权。条形颜色代表趋势方向。<br>Equal-weight score: Spike(25%) + Reach(25%) + Sentiment(25%) + Engagement(25%). Bar color = trend direction.</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        for d in ["rising","stable","declining"]:
            sub = top_stats[top_stats["trend_direction"]==d]
            if sub.empty: continue
            fig_bar.add_trace(go.Bar(
                x=sub["trend_score"], y=sub["cat_label"],
                orientation="h", name=f"{DIR_EMO[d]} {DIR_ZH[d]}",
                marker_color=COLORS[d],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "综合趋势分 Score: %{x:.2f}<br>"
                    "热度增长 Spike: %{customdata[0]:.1f}x<br>"
                    "当前帖子量 Posts: %{customdata[1]:,}<extra></extra>"
                ),
                customdata=sub[["spike_ratio","current_mentions"]].values,
            ))
        fig_bar.update_layout(
            barmode="overlay",
            height=max(420, top_n*22),
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
            xaxis=dict(title="综合趋势分 / Trend Score", range=[0,1.15]),
            legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0),
            plot_bgcolor="white",paper_bgcolor="white",
            margin=dict(l=10,r=60,t=30,b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
        st.markdown('<div class="chart-caption">📌 <b>热度 × 扩散象限图 Spike × Reach</b>：横轴 = 近期热度增长强度，纵轴 = 跨社区扩散广度，气泡大小 = 帖子量。右上角为核心机会区。<br>X = spike intensity, Y = cross-community reach, bubble = post volume.</div>', unsafe_allow_html=True)
        top_stats["bsize"] = (top_stats["current_mentions"] / top_stats["current_mentions"].max() * 55 + 8).round()
        fig_bub = go.Figure()
        # Quadrant backgrounds
        for x0,y0,x1,y1,col in [(.5,.5,1.05,1.05,"#F0FDF4"),
                                  (0,.5,.5,1.05,"#F8FAFC"),
                                  (.5,0,1.05,.5,"#EFF6FF"),
                                  (0,0,.5,.5,"#FAFAFA")]:
            fig_bub.add_shape(type="rect",x0=x0,y0=y0,x1=x1,y1=y1,
                              fillcolor=col,opacity=.55,line_width=0)
        for d in ["rising","stable","declining"]:
            sub = top_stats[top_stats["trend_direction"]==d]
            if sub.empty: continue
            fig_bub.add_trace(go.Scatter(
                x=sub["normalized_spike"], y=sub["cross_community"],
                mode="markers+text",
                name=f"{DIR_EMO[d]} {DIR_ZH[d]}",
                marker=dict(size=sub["bsize"],color=COLORS[d],opacity=.75,
                            line=dict(width=1,color="white")),
                text=sub["cat_label"].str[:12],
                textposition="top center", textfont=dict(size=8),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "热度强度 Spike: %{x:.2f}<br>"
                    "跨圈扩散 Reach: %{y:.2f}<br>"
                    "帖子量 Posts: %{customdata:,}<extra></extra>"
                ),
                customdata=sub["current_mentions"],
            ))
        for xp,yp,lab in [(.75,.97,"🔥 高热 / 高扩  High Heat · Broad Reach"),
                           (.18,.97,"广扩散 / 低热  Broad · Low Heat"),
                           (.75,.03,"局部爆点  Local Spike"),
                           (.18,.03,"低优先  Low Priority")]:
            fig_bub.add_annotation(x=xp,y=yp,text=lab,showarrow=False,
                                   font=dict(size=8.5,color="#94A3B8"))
        fig_bub.add_hline(y=.5,line_dash="dot",line_color="#CBD5E1",line_width=1)
        fig_bub.add_vline(x=.5,line_dash="dot",line_color="#CBD5E1",line_width=1)
        fig_bub.update_layout(
            xaxis=dict(title="热度增长强度 / Spike Intensity",range=[-.05,1.1]),
            yaxis=dict(title="跨社区扩散度 / Cross-community Reach",range=[-.05,1.1]),
            height=500,plot_bgcolor="white",paper_bgcolor="white",
            legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0),
            margin=dict(l=10,r=10,t=30,b=30),
        )
        st.plotly_chart(fig_bub, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — 飙升榜 Top 6
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="chart-caption">📌 <b>飙升榜 Spike Board</b>：仅按近两周热度增长倍数排序，找最近爆发最快的品类。卡片内 Bar 长度 = 该品牌在本品类的近两周提及量。<br>Ranked by 2-week spike only (not overall trend score). Bar length = brand mention count in current period.</div>', unsafe_allow_html=True)

    t2c1, t2c2 = st.columns(2)
    with t2c1:
        t2_n_cats = st.slider("展示品类数 / Show N categories", 4, 20, 6, 2, key="t2_n_cats")
    with t2c2:
        t2_n_brands = st.slider("每图展示品牌数 / Brands per card", 3, 30, 5, 1, key="t2_n_brands")
    st.markdown("")

    top_cats = stats_all.nlargest(t2_n_cats, "normalized_spike").copy()

    for i in range(0, t2_n_cats, 2):
        col_a, col_b = st.columns(2)
        for j, col in enumerate([col_a, col_b]):
            idx = i + j
            if idx >= len(top_cats): break
            row = top_cats.iloc[idx]
            cat    = row["category"]
            dcolor = COLORS[row["trend_direction"]]
            dlabel = f"{DIR_EMO[row['trend_direction']]} {DIR_ZH[row['trend_direction']]} · {DIR_EN[row['trend_direction']]}"
            delta_sign = "+" if row["mentions_delta"]>=0 else ""

            # Brand bar for this category
            bdf = cat_brand_data.get(cat, pd.DataFrame())

            with col:
                with st.container():
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
                          <div style="font-size:.7rem;color:#6B7280">热度增长 Spike</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;color:#111">{int(row['current_mentions']):,}</div>
                          <div style="font-size:.7rem;color:#6B7280">近两周帖子 Posts</div>
                        </div>
                        <div>
                          <div style="font-size:1.35rem;font-weight:700;
                               color={'#22C55E' if row['mentions_delta']>=0 else '#EF4444'}">
                            {delta_sign}{int(row['mentions_delta']):,}
                          </div>
                          <div style="font-size:.7rem;color:#6B7280">环比增减 vs Prior</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Brand bar chart
                    if not bdf.empty:
                        bdf_show = bdf.head(t2_n_brands).copy()
                        bdf_show["brand_short"] = bdf_show["brand"].str[:20]
                        fig_b = go.Figure(go.Bar(
                            x=bdf_show["cur_mentions"],
                            y=bdf_show["brand_short"],
                            orientation="h",
                            marker_color=COLORS["primary"],
                            opacity=0.8,
                            text=bdf_show["cur_mentions"].apply(lambda v: f"{v:,}"),
                            textposition="outside",
                            textfont=dict(size=9),
                            hovertemplate="<b>%{y}</b><br>当前提及 Current: %{x:,}<br>"
                                          "上期提及 Prior: %{customdata:,}<extra></extra>",
                            customdata=bdf_show["prev_mentions"],
                        ))
                        fig_b.update_layout(
                            height=260,
                            yaxis=dict(categoryorder="total ascending",tickfont=dict(size=9)),
                            xaxis=dict(title="提及量 / Mentions",showgrid=True,gridcolor="#F3F4F6"),
                            plot_bgcolor="white",paper_bgcolor="white",
                            margin=dict(l=0,r=50,t=5,b=25),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                    else:
                        st.caption("暂无品牌数据 / No brand data")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — 品类详情
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    t3c1, t3c2 = st.columns([3, 1])
    with t3c1:
        selected = st.selectbox(
            "选择品类 / Select Category",
            options=stats_all["category"].tolist(),
            index=0,
            format_func=cat_label,
        )
    with t3c2:
        t3_n = st.slider("展示品牌数 / Show N brands", 3, 30, 5, 1, key="t3_n")

    row = stats_all[stats_all["category"]==selected].iloc[0]

    # ── 5 Metric cards ────────────────────────────────────────────────────
    st.markdown(f"### {cat_label(selected)}")
    c1,c2,c3,c4,c5 = st.columns(5)
    dcolor = COLORS[row["trend_direction"]]
    c1.metric("综合趋势分\nTrend Score", f"{row['trend_score']:.2f}",
              delta=f"{DIR_EMO[row['trend_direction']]} {DIR_ZH[row['trend_direction']]}",
              delta_color="normal" if row["trend_direction"]=="rising"
                          else("off" if row["trend_direction"]=="stable" else "inverse"))
    c2.metric("热度增长倍数\nSpike Ratio", f"{row['spike_ratio']:.1f}x",
              delta=f"{int(row['current_mentions']):,} 帖/posts")
    c3.metric("用户好感度\nSentiment",
              f"{'+' if row['mean_sentiment']>=0 else ''}{row['mean_sentiment']:.2f}",
              delta="正面 Positive" if row["mean_sentiment"]>=0.05
                    else("中性 Neutral" if row["mean_sentiment"]>=-0.05 else "负面 Negative"),
              delta_color="normal" if row["mean_sentiment"]>=0.05
                          else("off" if row["mean_sentiment"]>=-0.05 else "inverse"))
    c4.metric("互动参与倍数\nEngagement", f"{row['eng_momentum']:.1f}x",
              delta="vs 上期 prior period", delta_color="off")
    c5.metric("活跃社区数\nActive Communities", f"{int(row['current_communities'])}",
              delta="subreddits", delta_color="off")

    st.markdown("")

    # ── Brand breakdown (left) + Delta numbers (right) ───────────────────
    b_col, w_col = st.columns([5,4])

    bdf = cat_brand_data.get(selected, pd.DataFrame())

    with b_col:
        st.markdown('<div class="chart-caption">📌 <b>品牌提及量 Brand Mentions</b>：条形长度 = 近两周提及帖子数；条形颜色 = 该品牌帖子平均好感度（连续色阶，红→黄→绿，范围 −1 到 +1）。<br>Bar length = current-period posts. Color = avg sentiment per brand (continuous RdYlGn scale, −1 to +1).</div>', unsafe_allow_html=True)
        bdf_filtered = bdf.head(t3_n) if not bdf.empty else bdf
        if not bdf_filtered.empty:
            bdf_show = bdf_filtered.copy()
            bdf_show["brand_short"] = bdf_show["brand"].str[:22]
            bdf_show["b_spike"] = (bdf_show["cur_mentions"] / bdf_show["prev_mentions"].replace(0, 1)).round(2)

            fig_brands = go.Figure(go.Bar(
                x=bdf_show["cur_mentions"],
                y=bdf_show["brand_short"],
                orientation="h",
                marker=dict(
                    color=bdf_show["avg_sentiment"],
                    colorscale="RdYlGn",
                    cmin=-1,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="好感度<br>Sentiment", side="right", font=dict(size=11)),
                        thickness=14,
                        len=0.85,
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1<br>负面", "-0.5", "0<br>中性", "0.5", "+1<br>正面"],
                        tickfont=dict(size=9),
                        outlinewidth=0,
                    ),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "当前提及 Current: %{x:,}<br>"
                    "上期提及 Prior: %{customdata[0]:,}<br>"
                    "好感度 Sentiment: %{customdata[1]:+.2f}<br>"
                    "增长倍数 Spike: %{customdata[2]:.1f}x<extra></extra>"
                ),
                customdata=bdf_show[["prev_mentions", "avg_sentiment", "b_spike"]].values,
            ))
            fig_brands.update_layout(
                height=max(260, t3_n * 46),
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
                xaxis=dict(title="近两周提及量 / Current Mentions", showgrid=True, gridcolor="#F3F4F6"),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=80, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_brands, use_container_width=True)
        else:
            st.info("该品类暂无识别到的品牌数据。/ No brand data identified for this category.")

    with w_col:
        st.markdown('<div class="chart-caption">📌 <b>品牌数据矩阵 Brand Matrix</b>：本期提及量、环比增减、平均好感度。<br>Current mentions, period-over-period delta, and avg sentiment per brand.</div>', unsafe_allow_html=True)
        if not bdf_filtered.empty:
            bdf_matrix = bdf_filtered.copy()
            bdf_matrix["delta"] = (bdf_matrix["cur_mentions"] - bdf_matrix["prev_mentions"]).astype(int)

            def fmt_delta(d):
                color = "#22C55E" if d > 0 else ("#EF4444" if d < 0 else "#94A3B8")
                sign  = "+" if d > 0 else ""
                return f'<span style="color:{color};font-weight:600">{sign}{d:,}</span>'

            def fmt_sent(s):
                if s >= 0.05:   color, label = "#22C55E", f"+{s:.2f}"
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
            .brand-matrix table{{width:100%;border-collapse:collapse;font-size:.82rem;}}
            .brand-matrix th{{background:#F8FAFC;padding:7px 10px;text-align:right;
              font-weight:600;color:#374151;font-size:.75rem;border-bottom:2px solid #E5E7EB;}}
            .brand-matrix th:first-child{{text-align:left;}}
            .brand-matrix td{{padding:6px 10px;border-bottom:1px solid #F3F4F6;color:#111;}}
            .brand-matrix tr:last-child td{{border-bottom:none;}}
            .brand-matrix tr:hover td{{background:#F9FAFB;}}
            </style>
            <div class="brand-matrix">
            <table>
              <thead><tr>
                <th style="text-align:left">品牌 Brand</th>
                <th>本期 Cur</th>
                <th>上期 Prev</th>
                <th>环比 Delta</th>
                <th>好感度 Sent</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("暂无数据 / No data")

    # Sample posts below
    posts = row["sample_posts"] if isinstance(row.get("sample_posts"), list) else []
    if posts:
        st.markdown("**近期高互动帖子 / Top Posts**")
        st.markdown('<div class="chart-caption">近两周互动分最高的帖子，反映该品类讨论热点。/ Top posts by engagement score in current period.</div>', unsafe_allow_html=True)
        for p in posts[:4]:
            title = str(p.get("title", ""))[:90]
            comm  = p.get("community", "")
            score = p.get("engagement_score", 0)
            label = p.get("sentiment_label", "")
            sent_color = COLORS["rising"] if label=="positive" else (
                COLORS["declining"] if label=="negative" else COLORS["stable"])
            sent_txt = {"positive":"😊 正面","negative":"😟 负面","neutral":"😐 中性"}.get(label, "")
            st.markdown(f"""
            <div style="padding:6px 0;border-bottom:1px solid #F3F4F6">
              <div style="font-size:.84rem;color:#111;margin-bottom:2px">{title}</div>
              <span style="font-size:.72rem;color:#6B7280">r/{comm}</span>
              <span style="font-size:.72rem;color:#6B7280;margin-left:10px">互动 {int(score):,}</span>
              <span style="font-size:.72rem;color:{sent_color};margin-left:10px">{sent_txt}</span>
            </div>""", unsafe_allow_html=True)
