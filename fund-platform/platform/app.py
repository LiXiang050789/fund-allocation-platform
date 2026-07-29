"""
农行杯赛题一 · 指数基金智能配置决策平台
Streamlit 前端 — 4 页: 仪表盘 / 策略对比 / 配置推荐 / AI助手
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='指数基金智能配置平台', layout='wide', page_icon='📊')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 全局配色 =====
COLORS = {'等权':'#888888','风险平价':'#4CAF50','MVO':'#FF5722','动量':'#2196F3',
          '动态评分':'#E91E63','LGB融合':'#FF9800','PPO全局':'#9C27B0','沪深300':'#607D8B'}
HIGHLIGHT = {'动态评分','LGB融合'}

# ===== 数据缓存 =====
@st.cache_data
def load_data():
    score = pd.read_csv(f'{BASE}/data/clean/market_score_daily.csv', encoding='utf-8-sig', parse_dates=['date'])
    metrics = pd.read_csv(f'{BASE}/results/backtest/baseline_metrics.csv', encoding='utf-8-sig')
    nav = pd.read_csv(f'{BASE}/results/backtest/baseline_nav.csv', encoding='utf-8-sig', parse_dates=['date'])
    seg = pd.read_csv(f'{BASE}/results/backtest/segmented_backtest.csv', encoding='utf-8-sig')
    tiers = pd.read_csv(f'{BASE}/results/risk_tiers.csv', encoding='utf-8-sig')
    return score, metrics, nav, seg, tiers

score_df, metrics_df, nav_df, seg_df, tiers_df = load_data()
latest = score_df.iloc[-1]

# ===== 侧边栏 =====
with st.sidebar:
    st.title('🏦 农行杯赛题一')
    st.caption('指数基金智能配置决策平台')
    st.divider()
    st.metric('最新市场评分', f"{latest['total_score']:.0f}/100")
    st.metric('市场状态', latest['market_state'])
    st.divider()
    st.caption(f'数据日期: {latest["date"].date()}')
    st.caption('团队: 晓枫(C) / lx(A) / zt(B)')

# ===== 页1: 市场仪表盘 =====
tab1, tab2, tab3, tab4 = st.tabs(['📈 市场仪表盘', '📊 策略对比', '🎯 配置推荐', '🤖 AI助手'])

with tab1:
    st.header('市场环境仪表盘')

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric('估值维度', f'{latest["val_score"]:.0f}', '25分制')
    with col2: st.metric('宏观维度', f'{latest["macro_score"]:.0f}', '25分制')
    with col3: st.metric('情绪维度', f'{latest["sent_score"]:.0f}', '20分制')
    with col4: st.metric('趋势维度', f'{latest["trend_score"]:.0f}', '30分制')

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        # 雷达图
        import plotly.graph_objects as go
        fig_radar = go.Figure(go.Scatterpolar(
            r=[latest['val_score'], latest['macro_score'], latest['sent_score'], latest['trend_score']],
            theta=['估值(25)','宏观(25)','情绪(20)','趋势(30)'],
            fill='toself', fillcolor='rgba(233,30,99,0.3)',
            line=dict(color='#E91E63', width=2)
        ))
        fig_radar.update_layout(height=350, margin=dict(t=20,b=20), polar=dict(radialaxis=dict(range=[0,30])))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        # 历史评分走势
        chart_data = score_df.set_index('date')[['total_score']].tail(500)
        st.line_chart(chart_data, height=350)

    # 状态分布
    st.subheader('全期状态分布')
    dist = score_df['market_state'].value_counts()
    cols = st.columns(5)
    for i, state in enumerate(['上行','震荡偏强','震荡','下行','极寒']):
        cnt = dist.get(state, 0)
        pct = cnt / len(score_df) * 100
        cols[i].metric(state, f'{cnt}天', f'{pct:.1f}%')

# ===== 页2: 策略对比 =====
with tab2:
    st.header('8 套策略全期对比 (2015-2026)')

    # 高亮表格
    display_df = metrics_df.copy()
    for c in ['年化收益','最大回撤','夏普比率','卡玛比率']:
        display_df[c] = display_df[c].apply(lambda x: f'{x*100:.1f}%' if abs(x)<2 else f'{x:.3f}')

    def highlight_best(row):
        if row['策略'] in HIGHLIGHT:
            return ['background-color: rgba(233,30,99,0.08)'] * len(row)
        return [''] * len(row)

    st.dataframe(metrics_df.style.format({
        '年化收益':'{:.1%}','最大回撤':'{:.1%}','夏普比率':'{:.3f}','卡玛比率':'{:.3f}','月均换手率':'{:.1%}'
    }).apply(highlight_best, axis=1), use_container_width=True, height=320)

    # 净值图
    st.subheader('策略净值对比')
    st.image(f'{BASE}/results/charts/fig1_nav_comparison.png', use_container_width=True)

    # 分段回测
    st.subheader('分段回测')
    st.image(f'{BASE}/results/charts/fig2_segmented_backtest.png', use_container_width=True)

# ===== 页3: 配置推荐 =====
with tab3:
    st.header('智能配置推荐')

    risk = st.selectbox('选择风险档', ['保守型(C2) — 权益20%','平衡型(C3) — 权益50%','进取型(C4) — 权益70%'])
    tier_map = {'保守型(C2) — 权益20%':'保守型(C2)','平衡型(C3) — 权益50%':'平衡型(C3)','进取型(C4) — 权益70%':'进取型(C4)'}
    tier_name = tier_map[risk]

    tier_data = tiers_df[tiers_df['风险档'] == tier_name]

    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(tier_data.iterrows()):
        bg = '#E91E63' if row['策略'] == '动态评分' else '#FF9800'
        with [col1, col2][i]:
            st.markdown(f"""
            <div style='background:{bg}15; padding:15px; border-radius:10px; border-left:4px solid {bg}'>
                <h4>{row['策略']}</h4>
                <h2>{row['年化收益']*100:.1f}% <small style='font-size:14px;color:#888'>年化</small></h2>
                <p>回撤: {row['最大回撤']*100:.1f}% | 夏普: {row['夏普比率']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.caption('💡 平衡型(C3) 12.2%年化 + -24.7%回撤对大多数零售客户可接受。LGB融合在所有风险档全面优于纯动态评分。')

# ===== 页4: AI助手 =====
with tab4:
    st.header('AI 投顾助手')

    scenarios = {
        '😱 市场大跌怎么办': (
            '恐慌安抚',
            f'看到账户浮亏确实让人不安。但当前市场评分 **{latest["total_score"]:.0f} 分**，状态为 **{latest["market_state"]}**。\n\n'
            f'估值维度 **{latest["val_score"]:.0f}/25**，宏观维度 **{latest["macro_score"]:.0f}/25**，情绪维度 **{latest["sent_score"]:.0f}/20**。\n\n'
            '市场处于恐慌区间时，往往是长线布局的窗口。我们的动态评分系统建议当前维持防御性仓位，减少暴露，等待评分回升后再逐步加仓。\n\n'
            '— 来自您的智能投顾助手'
        ),
        '🔥 现在该加仓吗': (
            '过热止盈',
            f'当前市场评分 **{latest["total_score"]:.0f} 分（{latest["market_state"]}）**。\n\n'
            '在评分超过 60 时，市场处于偏强状态，建议适度参与但不追高。如果您的权益仓位已经接近中枢，不建议进一步加仓——\n'
            '市场环境评分系统在 2015 年股灾和 2018 年熊市中都验证了"过热时降低仓位"的价值。\n\n'
            '— 来自您的智能投顾助手'
        ),
        '📅 定投还有用吗': (
            '长期定投',
            '定投的核心优势在于"用时间分散买入成本"，不需要择时。\n\n'
            f'即使当前评分处于 **{latest["market_state"]}** 区间，定投仍然是长期积累的有效方式。\n\n'
            '我们的回测数据显示：在 2021 年后的长期震荡市中（1307 天），等权策略仅赚 10%，而动态评分配置实现 33% 收益——\n'
            '这说明"市场状态分类 → 差异化配置"加上定投纪律，在长期中显著跑赢被动持有。\n\n'
            '— 来自您的智能投顾助手'
        ),
    }

    col_scene, col_chat = st.columns([1, 2])
    with col_scene:
        for label in scenarios:
            if st.button(label, use_container_width=True):
                st.session_state['scene'] = label

    with col_chat:
        if 'scene' not in st.session_state:
            st.session_state['scene'] = '😱 市场大跌怎么办'
        title, content = scenarios[st.session_state['scene']]
        st.markdown(f'### {title}')
        st.info(content)

print('Streamlit 平台启动 — http://localhost:8501')
