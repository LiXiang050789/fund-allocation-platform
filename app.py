""" 农行杯赛题一 · 指数基金智能配置决策平台 (v3) Streamlit 前端 — 5 页 + 动态交互 + DeepSeek API """
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json, subprocess, requests

st.set_page_config(page_title='指数基金智能配置平台', layout='wide', page_icon='📊')

# ========== Windows本地调试硬编码路径，根据你的实际目录修改 ==========
# BASE = 当前app.py所在文件夹！！
BASE = r"E:/农行杯/2026/1/Streamlit平台"
BASE1 = r"E:/农行杯/2026"
REPO_ROOT = BASE

# PPO相关路径，全部在当前文件夹
PPO_MODEL_PATH = os.path.join(REPO_ROOT, "global_best.zip")
PPO_FEATURE_PATH = f'{BASE}/data/clean/train_feature_filtered_env.csv'
PPO_PRICE_PATH = f'{BASE}/data/clean/etf_price_clean.csv'
PPO_ENV_PATH = f'{BASE}/portfolio_env_global.py'
PPO_WEIGHT_PATH = f'{BASE}/data/results/ppo_weight.csv'
PPO_RUNTIME_PATH = f'{BASE}/ppo_infer_runtime.py'


# Windows：注释掉Linux conda路径，直接使用当前环境Python，竞赛演示优先
PPO_PY312_PATH = None

# ===== 全局配色 =====
COLORS = {'沪深300':'#607D8B','等权':'#888888','风险平价':'#4CAF50','MVO':'#FF5722','动量':'#2196F3',
          '动态评分':'#E91E63','LGB融合':'#FF9800','PPO全局':'#9C27B0'}
HIGHLIGHT = {'动态评分','LGB融合'}

ETF_CODES = ['hs300','zz500','kc50','consume','chip','gold','bond10']
ETF_CN = {
    'hs300': '沪深300',
    'zz500': '中证500',
    'kc50': '科创50',
    'consume': '消费',
    'chip': '芯片',
    'gold': '黄金',
    'bond10': '十年国债',
}

# ===== PPO 推理缓存 =====
RL_WINDOW = 40
REWARD_COEF = (3.0, 0.18, 1.3, 0.65)
PPO_EXPERIMENT_CFG = {
    'rebalance_monthly': False,
    'enable_bond_regime_reward': False,
    'enable_bond_loss_couple': True,
    'enable_ladder_dd': True,
    'enable_min_bond_hard_constraint': True,
    'enable_reward_norm': True,
    'enable_lgb_pred_obs': False,
}

# 安全图片加载工具
def safe_image(path, **kwargs):
    if os.path.exists(path):
        st.image(path,**kwargs)
    else:
        st.warning(f"图片缺失：{os.path.basename(path)}，请检查charts文件夹")

def _format_ppo_weights(weight_map):
    return '、'.join([f'{ETF_CN[k]} {weight_map[k]*100:.1f}%' for k in ETF_CODES])

def _load_cached_ppo_weight(as_of_date, reason=''):
    if not os.path.exists(PPO_WEIGHT_PATH):
        return {'ok': False, 'message': 'PPO 模型未部署，当前仅使用市场评分'}
    weight_df = pd.read_csv(PPO_WEIGHT_PATH, encoding='utf-8-sig', parse_dates=['date'])
    weight_df = weight_df.sort_values('date')
    weight_df = weight_df[weight_df['date'] <= pd.Timestamp(as_of_date)]
    if weight_df.empty:
        return {'ok': False, 'message': 'PPO 权重无有效日期，当前仅使用市场评分'}
    row = weight_df.iloc[-1]
    # 处理NaN空值
    weight_map = {}
    for code in ETF_CODES:
        val = row.get(code, 0.0)
        weight_map[code] = float(val) if pd.notna(val) else 0.0
    message = f'PPO 实时推理不可用，已使用最近一次离线权重：{reason}' if reason else ''
    return {
        'ok': True,
        'date': pd.Timestamp(row['date']),
        'weights': weight_map,
        'text': f'PPO 模型今日推荐配置：{_format_ppo_weights(weight_map)}',
        'source': 'cached_weight',
        'message': message,
    }

@st.cache_data(show_spinner=False)
def infer_latest_ppo_weight(as_of_date):
    if not os.path.exists(PPO_MODEL_PATH):
        return {'ok': False, 'message': 'PPO 模型未部署，当前仅使用市场评分'}

    try:
        proc = subprocess.run(
            [sys.executable,
             PPO_RUNTIME_PATH, str(pd.Timestamp(as_of_date).date())],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f'exit {proc.returncode}')[-200:]
            return _load_cached_ppo_weight(as_of_date, detail)

        lines = proc.stdout.strip().splitlines()
        candidate_lines = [line for line in lines if line.lstrip().startswith("{")]
        if not candidate_lines:
            detail = f"子进程无JSON输出，stdout:{proc.stdout[:300]}"
            return _load_cached_ppo_weight(as_of_date, detail)
        payload = json.loads(candidate_lines[-1])

        weight_map = {}
        for code in ETF_CODES:
            val = payload['weights'].get(code,0.0)
            weight_map[code] = float(val) if pd.notna(val) else 0.0

        return {
            'ok': True,
            'date': pd.Timestamp(payload['date']),
            'weights': weight_map,
            'text': f'PPO 模型今日推荐配置：{_format_ppo_weights(weight_map)}',
            'source': payload.get('source', 'live_inference'),
            'message': payload.get('message', ''),
        }
    except subprocess.TimeoutExpired:
        return _load_cached_ppo_weight(as_of_date, '实时推理超时')
    except Exception as e:
        return _load_cached_ppo_weight(as_of_date, str(e))

# ===== 数据缓存 =====
@st.cache_data
def load_data():
    score = pd.read_csv(f'{BASE}/data/clean/market_score_daily.csv', encoding='utf-8-sig', parse_dates=['date'])
    metrics = pd.read_csv(f'{BASE}/data/results/backtest/baseline_metrics.csv', encoding='utf-8-sig')
    nav = pd.read_csv(f'{BASE}/data/results/backtest/baseline_nav.csv', encoding='utf-8-sig', parse_dates=['date'])
    seg = pd.read_csv(f'{BASE}/data/results/backtest/segmented_backtest.csv', encoding='utf-8-sig')
    tiers = pd.read_csv(f'{BASE}/data/results/risk_tiers.csv', encoding='utf-8-sig')
    return score, metrics, nav, seg, tiers


# 路径校验打印
print("====== 平台路径校验 ======")
print(f"BASE: {BASE}")
print(f"PPO模型文件存在: {os.path.exists(PPO_MODEL_PATH)}")
print(f"PPO权重csv存在: {os.path.exists(PPO_WEIGHT_PATH)}")
print(f"PPO推理脚本存在: {os.path.exists(PPO_RUNTIME_PATH)}")
print(f"市场评分csv存在: {os.path.exists(f'{BASE}/data/clean/market_score_daily.csv')}")
print("==========================")

score_df, metrics_df, nav_df, seg_df, tiers_df = load_data()

# PPO指标手工补充(独立引擎)
ppo_row = pd.DataFrame([{'策略':'PPO全局','年化收益':0.269,'最大回撤':-0.211,'夏普比率':1.41,
                          '卡玛比率':1.28,'月均换手率':np.nan,'索提诺比率':np.nan,
                          '年化波动':np.nan,'胜率':np.nan,'95%VaR':np.nan,'年化收益/最大回撤':np.nan}])
metrics_all = pd.concat([metrics_df, ppo_row], ignore_index=True)

latest = score_df.iloc[-1]

# 初始化会话状态
if "scene" not in st.session_state:
    st.session_state["scene"] = '😱 市场大跌怎么办'
# ========== 数据采集页面会话锁，防止重复并发执行脚本 ==========
if "crawl_running" not in st.session_state:
    st.session_state["crawl_running"] = False
if "clean_running" not in st.session_state:
    st.session_state["clean_running"] = False
if "crawl_log" not in st.session_state:
    st.session_state["crawl_log"] = ""
if "clean_log" not in st.session_state:
    st.session_state["clean_log"] = ""

# ===== 侧边栏 =====
with st.sidebar:
    st.title('🏦 农行杯赛题一')
    st.caption('指数基金智能配置决策平台')
    st.divider()
    st.metric('最新市场评分', f"{latest['total_score']:.0f}/100")
    st.metric('市场状态', latest['market_state'])
    st.divider()
    st.caption(f'数据日期: {latest["date"].date()}')
    st.caption('团队: 三元智投队')
    st.caption('数据源: sf四维评分规则 + 北向NaN + 宏观滞后修正')
    with st.expander("ℹ️平台说明"):
        st.markdown("""
- 赛题一：指数基金智能配置决策平台
- 策略池：沪深300、等权、风险平价、MVO、动量、动态评分、LGB融合、PPO全局
- AI助手调用DeepSeek大模型，仅做投顾科普，**不构成投资建议**
""")

# ========= 全局提前调用PPO推理，tab3、tab4都可以访问ppo_info =========
ppo_info = infer_latest_ppo_weight(latest['date'])

# ===== 页1: 市场仪表盘 =====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['📈 市场仪表盘', '📊 策略对比','📥自动化数据采集', '🎯 配置推荐', '🛡️ 风险划分','🤖 AI助手' ])


with tab1:
    st.header('市场环境仪表盘')

    # 日期滑块
    date_min = score_df['date'].min().date()
    date_max = score_df['date'].max().date()
    selected_date = st.slider('选择历史日期', min_value=date_min, max_value=date_max,
                              value=date_max, format='YYYY-MM-DD')
    row = score_df[score_df['date'] == pd.Timestamp(selected_date)]
    if len(row) == 0:
        row = score_df.iloc[-1:]
    current = row.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric('估值维度', f'{current["val_score"]:.0f}', '25分制')
    with col2: st.metric('宏观维度', f'{current["macro_score"]:.0f}', '25分制')
    with col3: st.metric('情绪维度', f'{current["sent_score"]:.0f}', '20分制')
    with col4: st.metric('趋势维度', f'{current["trend_score"]:.0f}', '30分制')

    st.metric('总分', f'{current["total_score"]:.0f}/100', current['market_state'])

    col_left, col_right = st.columns([1, 1.5])
    with col_left:
        import plotly.graph_objects as go
        fig_radar = go.Figure(go.Scatterpolar(
            r=[current['val_score'], current['macro_score'], current['sent_score'], current['trend_score']],
            theta=['估值(25)','宏观(25)','情绪(20)','趋势(30)'],
            fill='toself', fillcolor='rgba(233,30,99,0.3)',
            line=dict(color='#E91E63', width=2)
        ))
        fig_radar.update_layout(height=350, margin=dict(t=20,b=20),
                                polar=dict(radialaxis=dict(range=[0,30])))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
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

    # 策略筛选多选框
    all_strategies = metrics_all['策略'].tolist()
    default_selected = ['沪深300','等权','动态评分','LGB融合','PPO全局']
    selected = st.multiselect('筛选策略', all_strategies, default=default_selected)
    filtered = metrics_all[metrics_all['策略'].isin(selected)]

    def highlight_best(row):
        if row['策略'] in HIGHLIGHT:
            return ['background-color: rgba(233,30,99,0.08)'] * len(row)
        return [''] * len(row)

    format_dict = {'年化收益':'{:.1%}','最大回撤':'{:.1%}','夏普比率':'{:.2f}',
                    '卡玛比率':'{:.2f}','月均换手率':'{:.1%}'}
    st.dataframe(filtered.style.format(format_dict, na_rep='—').apply(highlight_best, axis=1),
                  use_container_width=True, height=350)

    safe_image(f'{BASE}/data/results/charts/fig1_nav_comparison.png', use_container_width=True)
    safe_image(f'{BASE}/data/results/charts/fig2_segmented_backtest.png', use_container_width=True)

# ===== 页3: 配置推荐 =====
with tab3:
    st.header("📥 自动化数据采集流水线")
    st.markdown("""
    > 本平台具备完整可落地数据链路：**网络爬虫获取原始行情&宏观数据 → 原始数据落盘raw_data → 统一清洗加工 → 输出clean建模数据集，供给回测、PPO强化学习、市场评分模块**

    ⚠️ **重要风险提醒：执行爬虫会覆盖 raw_data 目录原始文件；如需保留历史原始数据，正式运行前请手动备份 raw_data 文件夹。**
    > 提示：爬虫会访问公开AKShare接口，受网络、接口限流影响，完整爬取耗时数分钟；不要连续重复点击；仅为演示用途。
    """)
    st.divider()


    # 工具函数：读取csv简单信息
    def get_file_info(filepath):
        fp = filepath  # ← 关键改动：直接用，不要 os.path.join(BASE1, filepath)
        if not os.path.exists(fp):
            return {"exist": False, "size_mb": 0, "rows": 0, "start": None, "end": None}
        size_mb = os.path.getsize(fp) / 1024 / 1024
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
            rows = len(df)
            dt_start = df["date"].min().date() if rows > 0 else None
            dt_end = df["date"].max().date() if rows > 0 else None
        except Exception:
            rows = -1
            dt_start = None
            dt_end = None
        return {"exist": True, "size_mb": round(size_mb, 2), "rows": rows, "start": dt_start, "end": dt_end}


    st.subheader("🔍 当前数据集状态检测")
    file_check_list = [
        {"name": "指数原始行情", "path": f"{BASE1}/raw_data/index/index_daily.csv"},
        {"name": "ETF原始行情", "path": f"{BASE1}/raw_data/etf/etf_daily.csv"},
        {"name": "北向资金原始", "path": f"{BASE1}/raw_data/macro/north_flow.csv"},
        # {"name":"PPI宏观原始", "path":f"{BASE1}/raw_data/macro/ppi.csv"},
        {"name": "清洗后ETF价格", "path": f"{BASE1}/1/clean/etf_price_clean.csv"},
        {"name": "清洗后市场评分", "path": f"{BASE1}/1/clean/market_score_clean.csv"},
        {"name": "PPO训练总特征表", "path": f"{BASE1}/1/clean/train_total_feature.csv"},
    ]

    check_df_rows = []
    for item in file_check_list:
        info = get_file_info(item["path"])
        status = "✅存在" if info["exist"] else "❌缺失"
        s_date = str(info["start"]) if info["start"] else "-"
        e_date = str(info["end"]) if info["end"] else "-"
        check_df_rows.append({
            "文件名": item["name"],
            "状态": status,
            "大小(MB)": info["size_mb"],
            "行数": info["rows"],
            "起始日期": s_date,
            "结束日期": e_date
        })
    st.dataframe(pd.DataFrame(check_df_rows), use_container_width=True)

    st.divider()
    st.subheader("⚙️ 执行流水线脚本（调用外部py文件）")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**1. 执行增量爬虫（最多允许爬取2个月区间）**")
        c1, c2 = st.columns(2)
        with c1:
            user_crawl_start = st.date_input("爬取开始日期", value=pd.Timestamp.now() - pd.Timedelta(days=60))
        with c2:
            user_crawl_end = st.date_input("爬取结束日期", value=pd.Timestamp.now())

        # 校验最大跨度：不能超过60天（2个月）
        delta_day = (user_crawl_end - user_crawl_start).days
        valid_range = True
        range_warn_text = ""
        if delta_day <= 0:
            valid_range = False
            range_warn_text = "❌结束日期必须大于开始日期"
        if delta_day > 60:
            valid_range = False
            range_warn_text = f"❌时间跨度{delta_day}天，禁止超过60天（2个月），缩小时间区间再执行"

        if not valid_range:
            st.error(range_warn_text)

        crawl_btn = st.button("🚀 启动数据爬虫", disabled=(st.session_state["crawl_running"] or (not valid_range)),
                              use_container_width=True)

        if crawl_btn:
            st.session_state["crawl_running"] = True
            st.session_state["crawl_log"] = ""
            # 转成YYYYMMDD字符串传给脚本
            start_str = user_crawl_start.strftime("%Y%m%d")
            end_str = user_crawl_end.strftime("%Y%m%d")
            with st.spinner(f"爬虫正在运行，区间 {start_str} ~ {end_str}，请勿关闭页面..."):
                try:
                    # 第一步运行爬取数据.py，传入两个日期参数
                    p1 = subprocess.run(
                        [sys.executable, os.path.join(BASE1, "爬取数据.py"), start_str, end_str],
                        cwd=BASE1, capture_output=True, text=True, timeout=600
                    )
                    # 第二步运行补充数据挖取.py，传入两个日期参数
                    p2 = subprocess.run(
                        [sys.executable, os.path.join(BASE1, "补充数据挖取.py"), start_str, end_str],
                        cwd=BASE1, capture_output=True, text=True, timeout=600
                    )
                    full_log = ("=====【爬取数据.py输出】=====\n" + p1.stdout + "\nstderr:\n" + p1.stderr
                                + "\n\n=====【补充数据挖取.py输出】=====\n" + p2.stdout + "\nstderr:\n" + p2.stderr)
                    st.session_state["crawl_log"] = full_log
                except subprocess.TimeoutExpired:
                    st.session_state["crawl_log"] = "【超时】爬虫运行超过10分钟被终止，请检查网络或者调大timeout"
                except Exception as e:
                    st.session_state["crawl_log"] = f"【异常】{str(e)}"
            st.session_state["crawl_running"] = False
            st.rerun()  # 刷新页面展示日志与文件状态

        if st.session_state["crawl_log"]:
            with st.expander("📋爬虫运行日志(点击展开)", expanded=False):
                st.text_area("日志输出", st.session_state["crawl_log"], height=250)
        st.divider()
        st.subheader("🔬 特征筛选")
        st.markdown("""
    > 完整流水线：`clean清洗数据集 → 特征筛选 → train_feature_filtered_env.csv`，作为LGB融合、PPO强化学习模型输入。

    **筛选规则：**
    1. 剔除高度共线性冗余字段；
    2. 剔除未来泄露字段（不能使用未来时刻数据做当前时刻观测）；
    3. 过滤缺失率过高的特征列；
    4. 保留量价技术因子、估值、利率、资金流、宏观滞后指标；
    5. 输出对齐交易日的建模样本。
    """)

        # 会话状态保存模拟筛选结果
        if "demo_filter_feature_df" not in st.session_state:
            st.session_state["demo_filter_feature_df"] = None

        run_filter_btn = st.button("🧪 模拟执行特征筛选（演示）", use_container_width=True)
        if run_filter_btn:
            with st.spinner("模拟特征筛选执行中……"):
                import time

                time.sleep(1.5)
                # --------模拟加载clean目录总特征表，取前80行做预览样本----------
                feature_sample_path = os.path.join(BASE1, "1/clean/train_total_feature.csv")
                if os.path.exists(feature_sample_path):
                    df_raw_sample = pd.read_csv(feature_sample_path, parse_dates=["date"], nrows=80,encoding='gbk')
                    # 模拟筛选：删掉一部分模拟的高共线性列
                    drop_sim_cols = ["mock_col1", "mock_col2"]
                    exist_drop = [c for c in drop_sim_cols if c in df_raw_sample.columns]
                    df_filtered_demo = df_raw_sample.drop(columns=exist_drop)
                    # 模拟：人工补齐字段（演示，不写回文件）
                    if "ppi_yoy" in df_filtered_demo.columns:
                        df_filtered_demo["ppi_yoy"] = df_filtered_demo["ppi_yoy"].fillna(
                            df_filtered_demo["ppi_yoy"].median())
                    st.session_state["demo_filter_feature_df"] = df_filtered_demo
                else:
                    st.warning("本地train_total_feature.csv不存在，生成空模拟样本")
                    # 构造假演示样本
                    demo_data = {
                        "date": pd.date_range(start="2024-01-02", periods=30, freq="B"),
                        "hs300_rsi14": np.random.uniform(20, 80, 30),
                        "kc50_pe_quantile": np.random.uniform(0, 1, 30),
                        "yield_10y": np.random.uniform(2.0, 3.5, 30),
                        "north_net": np.random.randn(30) * 20,
                        "ppi_yoy": np.random.uniform(-5, 8, 30),
                        "mock_manual_fill": [np.nan, 3.1, np.nan, 2.5] * 7 + [4.0, np.nan]
                    }
                    st.session_state["demo_filter_feature_df"] = pd.DataFrame(demo_data)

        if st.session_state["demo_filter_feature_df"] is not None:
            df_out = st.session_state["demo_filter_feature_df"]
            st.success(f"✅模拟筛选完成，预览样本：{len(df_out)}行，{len(df_out.columns)}列")
            st.caption("mock_manual_fill列为宏观字段，NaN代表待填充")
            st.dataframe(df_out.head(20), use_container_width=True)

            st.markdown("📌输出文件：`train_feature_filtered_env.csv`（PPO环境与LGB模型输入文件）")
            st.code(f"""
    # 真实项目执行伪代码（仅演示，页面不执行）
    df = pd.read_csv("data/clean/train_total_feature.csv",parse_dates=["date"])
    # 1. 人工读取外部Excel，补齐df["mock_manual_fill"]等缺失宏观字段
    manual_df = pd.read_excel("manual_macro_supplement.xlsx")
    df = pd.merge(df,manual_df,on="date",how="left")

    # 2. 特征过滤：删除共线性、未来泄露列
    drop_cols = get_high_correlation_drop_list(df)
    df = df.drop(columns=drop_cols)

    # 3. 过滤缺失过高列
    valid_cols = [c for c in df.columns if df[c].notna().sum() / len(df) > 0.6]
    df = df[valid_cols]

    # 4. 输出给模型
    df.to_csv("data/clean/train_feature_filtered_env.csv",index=False,encoding="utf-8-sig")
    """, language="python")


    st.divider()
    st.subheader("📋完整业务流程说明")
    st.markdown("""```
    [公开互联网数据源 AKShare]
                    ↓ 
    【爬虫脚本：爬取数据.py/ 补充数据挖取.py】
        raw_data/ 原始数据目录
        ├ index/      指数、黄金现货原始 K 线
        ├ etf/        ETF 行情原始 K 线
        ├ bond/       国债收益率原始数据
        └ macro/      PPI、北向资金、换手率宏观原始数据
                    ↓ 
    【清洗脚本：数据清洗.py】
        data/clean/ 清洗后数据集（供平台各模块调用）
        ├ etf_price_clean.csv        
        ├ market_score_daily.csv     → 市场仪表盘多维打分
        ├ train_total_feature.csv    → 总特征表
        ├ valuation_merged.csv       → 估值因子表
        ├ macro_merged.csv           → 宏观指标表
        └ … 其他衍生数据表
                    ↓
    【特征选取：特征筛选.py】
                    ↓
        train_feature_filtered.csv   → LGB策略、PPO 模型输入、回测对比
                    ↓
    Streamlit 平台：回测模块｜PPO 强化学习｜资产配置｜AI 理财助手｜风险分析
        """)
    st.info("💡竞赛演示价值：展示整套数据采集‑清洗‑建模的完整业务链路，证明系统具备真实落地能力。")



# ===== 页4: AI助手 =====
with tab4:
    st.header('智能配置推荐')
    risk = st.selectbox('选择风险档', ['保守型(C2) — 权益20%', '平衡型(C3) — 权益50%', '进取型(C4) — 权益70%'])
    tier_map = {'保守型(C2) — 权益20%': '保守型(C2)', '平衡型(C3) — 权益50%': '平衡型(C3)',
                '进取型(C4) — 权益70%': '进取型(C4)'}
    tier_name = tier_map[risk]
    tier_data = tiers_df[tiers_df['风险档'] == tier_name]
    captions = {
        '保守型(C2)': '💡 C2 适合退休/低风险投资者。LGB融合年化8.0%、回撤仅-12.7%，在保守档全面优于动态评分。',
        '平衡型(C3)': '💡 C3 是 sweet spot——LGB融合年化12.2%/-24.7%回撤，对大多数零售客户可接受。',
        '进取型(C4)': '💡 C4 年化15.0%但回撤-34.0%，超出零售客户通常承受范围（≤20%），谨慎推荐。',
    }
    col1, col2 = st.columns(2)
    for i, (_, row) in enumerate(tier_data.iterrows()):
        bg = '#E91E63' if row['策略'] == '动态评分' else '#FF9800'
        with [col1, col2][i]:
            st.markdown(f"""
                <div style='background:{bg}15; padding:15px; border-radius:10px; border-left:4px solid {bg}'>
                    <h4>{row['策略']}</h4>
                    <h2>{row['年化收益'] * 100:.1f}% <small style='font-size:14px;color:#888'>年化</small></h2>
                    <p>回撤: {row['最大回撤'] * 100:.1f}% | 夏普: {row['夏普比率']:.2f} | 净值: {row['最终净值']:.2f}×</p>
                </div>
                """, unsafe_allow_html=True)
    st.divider()
    st.subheader('推荐配置 — ETF 权重分布')
    import plotly.graph_objects as go

    # 优先使用PPO权重，失败回落风险档均分
    if ppo_info['ok']:
        weight_map = ppo_info['weights']
        etf_names = [ETF_CN[k] for k in ETF_CODES]
        etf_weights = [weight_map[k] for k in ETF_CODES]
        st.info(f"✅ 使用PPO全局模型输出权重（来源：{ppo_info['source']}，日期{ppo_info['date'].date()}）")
    else:
        equity_pct = {'保守型(C2)': 0.20, '平衡型(C3)': 0.50, '进取型(C4)': 0.70}[tier_name]
        etf_names = ['沪深300', '中证500', '科创50', '消费', '芯片', '黄金', '十年国债']
        etf_weights = [equity_pct / 5] * 5 + [(1 - equity_pct) / 2] * 2
        st.warning("⚠️ PPO模型不可用，展示风险档位均分参考权重")
    colors_bar = ['#E91E63'] * 5 + ['#FFC107', '#4CAF50']
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=etf_names,
        y=[w * 100 for w in etf_weights],
        marker_color=colors_bar,
        text=[f'{w * 100:.1f}%' for w in etf_weights],
        textposition='outside'
    ))
    y_max_val = max([w * 100 for w in etf_weights])
    y_max_val = max(y_max_val, 5)
    fig_bar.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title='权重(%)',
                          yaxis=dict(range=[0, y_max_val + 10]))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(captions.get(tier_name, ''))
    st.caption("⚠️配置推荐权重使用数据集最新日期，不受上方历史日期滑块控制；PPO原生输出不受风险档位下拉框约束。")

    # ==========新增：用户自选日期PPO推理交互模块==========
    st.divider()
    st.subheader("🧪 PPO历史日期回测推理（自定义日期）")
    st.caption("选择任意历史交易日，调用PPO脚本生成当日7类ETF资产权重，不会改变上方默认推荐结果")
    # 初始化session存储自定义ppo结果
    if "custom_ppo_result" not in st.session_state:
        st.session_state["custom_ppo_result"] = None
    col_date, col_btn = st.columns([3, 1])
    with col_date:
        user_choose_date = st.date_input(
            "选择要推理的日期",
            value=date_max,
            min_value=date_min,
            max_value=date_max
        )
    with col_btn:
        run_btn = st.button("🚀执行PPO推理", use_container_width=True)

    if run_btn:
        with st.spinner("正在调用PPO推理脚本，最多等待120s..."):
            custom_ppo = infer_latest_ppo_weight(as_of_date=user_choose_date)
            st.session_state["custom_ppo_result"] = custom_ppo

    # 如果会话里面存在自定义ppo结果就渲染
    if st.session_state["custom_ppo_result"] is not None:
        res = st.session_state["custom_ppo_result"]
        if res["ok"]:
            st.success(res["text"])
            st.caption(f"数据来源：{res['source']}，推理基准日期：{res['date'].date()}")
            # 构造表格展示7个资产权重
            weight_table = []
            for code in ETF_CODES:
                weight_table.append({
                    "资产名称": ETF_CN[code],
                    "代码标识": code,
                    "权重(%)": round(res["weights"][code] * 100, 2)
                })
            df_custom_weight = pd.DataFrame(weight_table)
            st.dataframe(df_custom_weight, use_container_width=True, hide_index=True)

            # 画该日期的bar图
            fig_custom = go.Figure()
            w_list = [res["weights"][k] for k in ETF_CODES]
            fig_custom.add_trace(go.Bar(
                x=[ETF_CN[k] for k in ETF_CODES],
                y=[x * 100 for x in w_list],
                text=[f'{x * 100:.2f}%' for x in w_list],
                textposition='outside'
            ))
            fig_custom.update_layout(height=320, yaxis_title="权重(%)", margin=dict(t=20, b=20))
            st.plotly_chart(fig_custom, use_container_width=True)
            if res["message"]:
                st.info(f"备注信息：{res['message']}")
        else:
            st.error(f"推理失败：{res['message']}")
    # ==========新增结束==========

# ===== 页5: 风险划分 =====
with tab5:
    st.header('投资者风险等级划分')

    st.subheader('C1-C5 五档风险等级')
    risk_table = pd.DataFrame([
        ['C1','保守型','仅限 R1（如货币基金）','≤20 分','超出回测范围'],
        ['C2','谨慎型','R1、R2（如债券基金）','21-40 分','年化 8.0%/回撤 -12.7%'],
        ['C3','稳健型','R1、R2、R3（如混合基金）','41-65 分','年化 12.2%/回撤 -24.7% ✨'],
        ['C4','积极型','R1-R4（如股票基金）','66-90 分','年化 15.0%/回撤 -34.0%'],
        ['C5','激进型','R1-R5（含期货期权）','≥91 分','超出回测范围'],
    ], columns=['等级','名称','可购买产品','问卷得分','回测覆盖'])
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader('问卷设计概要')
    st.markdown('''
    - **15 题多维测评**：年龄、学历、职业、收入来源、年收入、投资占比、债务情况、
      投资经验、投资品种、知识水平、投资倾向、亏损承受、亏损反应、投资目标、资金期限
    - **满分 148 分**，按总分映射到 C1-C5 五档
    - **有效期 12 个月**，财务状况重大变化需重新测评
    - **特殊认定**：第 12 题选"不能承受任何亏损"且总分<30 → 认定为最低风险类别
    ''')
    st.caption('数据源: xf《用户画像测评问卷+打分规则.docx》《风险等级划分规则.docx》')

    st.divider()
    st.subheader('三档策略匹配')
    st.dataframe(tiers_df.style.format({'年化收益':'{:.1%}','最大回撤':'{:.1%}','夏普比率':'{:.2f}'}),
                  use_container_width=True, hide_index=True)
    st.caption('C1/C5 超出回测覆盖范围，建议咨询客户经理后人工配置')

# ===== tab6：自动化数据采集流水线页面 =====
with tab6:
    st.header('AI 投顾助手')

    if ppo_info['ok']:
        source_label = '实时推理' if ppo_info.get('source') == 'live_inference' else '离线权重兜底'
        st.caption(f"{ppo_info['text']}（{source_label}，权重日期 {ppo_info['date'].date()}）")
        if ppo_info.get('message'):
            st.caption(ppo_info['message'])
    else:
        if '未部署' in ppo_info['message']:
            st.warning('PPO 模型未部署，当前仅使用市场评分')
        else:
            st.caption(ppo_info['message'])

    mode = st.radio('模式', ['🎯 预制场景', '💬 实时问答 (DeepSeek)'], horizontal=True)

    if mode == '🎯 预制场景':
        ppo_scene_text = (
            f'\n\n当前 {ppo_info["text"]}。'
            if ppo_info['ok']
            else '\n\n当前无 PPO 权重，仅基于市场评分回答。'
        )
        scenarios = {
            '😱 市场大跌怎么办': ('恐慌安抚',
                                 f'看到账户浮亏确实让人不安。当前市场评分 **{latest["total_score"]:.0f} 分（{latest["market_state"]}）**。\n\n'
                                 f'估值维度 **{latest["val_score"]:.0f}/25**，宏观维度 **{latest["macro_score"]:.0f}/25**。\n\n'
                                 '市场处于恐慌区间时，往往是长线布局的窗口。动态评分系统建议维持防御性仓位，等待评分回升后再逐步加仓。'
                                 + ppo_scene_text),
            '🔥 现在该加仓吗': ('过热止盈',
                               f'当前市场评分 **{latest["total_score"]:.0f} 分（{latest["market_state"]}）**。\n\n'
                               '评分超过 60 时市场偏强，建议适度参与但不追高。2015 股灾和 2018 熊市中验证了"过热时降低仓位"的价值。'
                               + ppo_scene_text),
            '📅 定投还有用吗': ('长期定投',
                               '定投的核心优势是"用时间分散买入成本"。\n\n'
                               '回测显示：2021 年后 1307 天震荡期，等权仅赚 10%，动态评分配置实现 33% 收益——\n'
                               '"市场状态分类 + 定投纪律"在长期中显著跑赢被动持有。'
                               + ppo_scene_text),
        }
        col_scene, col_chat = st.columns([1, 2])
        with col_scene:
            for label in scenarios:
                if st.button(label, use_container_width=True):
                    st.session_state['scene'] = label
        with col_chat:
            title, content = scenarios[st.session_state['scene']]
            st.markdown(f'### {title}')
            st.info(content)

    else:
        st.caption('基于 DeepSeek API，回答中嵌入当前市场数据')
        user_q = st.text_input('输入你的问题', placeholder='比如：现在适合加仓吗？')
        api_key_env = os.getenv("DEEPSEEK_API_KEY", "")
        api_key = st.text_input('DeepSeek API Key', type='password', placeholder='sk-...', value=api_key_env)

        if st.button('发送', use_container_width=True) and user_q and api_key:
            system_prompt = f"""你是一个智能投顾助手，为农行客户提供资产配置建议。 当前市场环境: 评分 {latest['total_score']:.0f}/100，状态 {latest['market_state']}。 估值{latest['val_score']:.0f}/25，宏观{latest['macro_score']:.0f}/25，情绪{latest['sent_score']:.0f}/20，趋势{latest['trend_score']:.0f}/30。 {ppo_info['text'] if ppo_info['ok'] else '无 PPO 权重，仅基于市场评分回答。'} 资产池: 沪深300/中证500/科创50/消费/芯片/黄金/十年国债 (7只ETF)。 策略: 动态评分配置(年化19.8%/-16.5%回撤)，LGB融合(年化12.4%/夏普0.66)。 回答要简洁专业，不生成具体投资建议。"""
            with st.spinner('思考中...'):
                try:
                    resp = requests.post(
                        'https://api.deepseek.com/v1/chat/completions',
                        headers={'Authorization': f'Bearer {api_key}',
                                 'Content-Type': 'application/json'},
                        json={'model': 'deepseek-chat', 'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_q}
                        ], 'max_tokens': 500, 'temperature': 0.7},
                        timeout=20
                    )
                    if resp.status_code == 200:
                        reply = resp.json()['choices'][0]['message']['content']
                        st.success(reply)
                    else:
                        st.error(f'API错误: {resp.status_code} — {resp.text[:200]}')
                except Exception as e:
                    st.error(f'请求失败: {e}')
                    st.info('💡 可切换到"预制场景"模式体验')




#python -m streamlit run "E:\农行杯\2026\1\Streamlit平台\app.py"
