"""
农行杯赛题一 · 5套基线回测 + 全套风险指标
纯 numpy/pandas 实现，零外部优化库依赖
"""

import pandas as pd
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f'{BASE}/data/clean'
OUT  = f'{BASE}/results/backtest'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 0. 参数
# ============================================================
INIT_CASH    = 1_000_000       # 起始资金
SLIPPAGE     = 0.0005          # 单边滑点 0.05%
FEE          = 0.001           # 双边成本 0.1%
MAX_WEIGHT   = 0.30            # 单 ETF 上限 30%
MIN_WEIGHT   = 0.0
RISK_FREE    = 0.025           # 无风险利率 2.5%
LOOKBACK_MOM = 12              # 动量回看月数
MVO_WINDOW   = 60              # MVO 协方差窗口 (交易日)
ETF_CODES    = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_ETFS  = ['hs300','zz500','kc50','consume','chip']
HEDGE_ETFS   = ['gold','bond10']

# ============================================================
# 1. 数据加载
# ============================================================
price_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
listed_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
score_df  = pd.read_csv(f'{DATA}/market_score_clean.csv', encoding='utf-8-sig', parse_dates=['date'])

# 价格矩阵
price_df = price_df.set_index('date')
prices = price_df[ETF_CODES].copy()  # 7 列收盘价

# 上市标志
listed_cols = [f'{e}_listed' for e in ETF_CODES]
avail = listed_df.set_index('date')[listed_cols].copy()
avail.columns = ETF_CODES

# 评分 → 状态序列
state_map = score_df.set_index('date')['market_state']

# 对齐日期
common_dates = prices.index.intersection(avail.index).intersection(state_map.index)
prices = prices.loc[common_dates]
avail  = avail.loc[common_dates]
state  = state_map.loc[common_dates]

returns = prices.pct_change().fillna(0)

print(f'[数据] 交易日: {len(common_dates)}, {common_dates[0].date()} ~ {common_dates[-1].date()}')
print(f'[数据] ETF: {prices.shape[1]}只, 评分状态类型: {sorted(state.dropna().unique())}')

# ============================================================
# 2. 工具函数
# ============================================================

def mask_available(arr, date_idx):
    """返回当日已上市ETF的布尔mask"""
    return avail.iloc[date_idx].values.astype(bool)

def solve_mvo(mu, cov, avail_mask):
    """
    无约束切点组合 (max Sharpe) → 约束修剪 (0 <= w <= 0.3, sum=1)
    mu, cov 只包含可用标的
    """
    n_assets = len(mu)
    if n_assets == 0:
        return np.array([])

    # 无约束切点: w = Σ⁻¹·(μ - rf) / 1ᵀ·Σ⁻¹·(μ - rf)
    excess = mu - RISK_FREE/252  # 日化无风险
    try:
        cov_inv = np.linalg.inv(cov)
        w_raw = cov_inv @ excess
        w_raw = w_raw / np.sum(w_raw)
    except np.linalg.LinAlgError:
        w_raw = np.ones(n_assets) / n_assets

    # 约束修剪
    w = np.clip(w_raw, MIN_WEIGHT, MAX_WEIGHT)
    s = w.sum()
    if s > 0:
        w = w / s

    # 二次裁剪
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    s = w.sum()
    if s > 0:
        w = w / s

    return w

def solve_risk_parity(cov, avail_mask):
    """波动率倒数加权 (传统风险平价近似) — 只含可用标的"""
    n = len(cov)
    if n == 0:
        return np.array([])
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w = 1.0 / vols
    w = w / w.sum()
    return np.clip(w, MIN_WEIGHT, MAX_WEIGHT)

def fill_full_weights(sub_w, avail_mask, sub_indices):
    """将子集权重填回全量 7 维向量"""
    full = np.zeros(7)
    for i, idx in enumerate(sub_indices):
        full[idx] = sub_w[i] if i < len(sub_w) else 0
    s = full.sum()
    if s > 0:
        full = full / s
    return full

# ============================================================
# 3. 五套策略 —— 返回月度权重矩阵 (index=date, columns=ETF)
# ============================================================

def get_monthly_rebalance_dates(dates):
    """每月第一个交易日"""
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

def strategy_equal_weight(dates):
    """策略1: 等权"""
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        n_avail = avail_mask.sum()
        if n_avail == 0:
            continue
        w = np.zeros(7)
        w[avail_mask] = 1.0 / n_avail
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_risk_parity(dates):
    """策略2: 风险平价 (波动率倒数加权)"""
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        if avail_mask.sum() == 0:
            continue
        # 用过去 MVO_WINDOW 日的收益率算协方差
        start = max(0, loc - MVO_WINDOW)
        ret_window = returns.iloc[start:loc+1]
        ret_window = ret_window.loc[:, avail_mask]
        cov = ret_window.cov().values
        sub_w = solve_risk_parity(cov, avail_mask)
        w = fill_full_weights(sub_w, avail_mask, np.where(avail_mask)[0])
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_mvo(dates):
    """策略3: MVO 最大化夏普比率"""
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        n_avail = avail_mask.sum()
        if n_avail < 2:
            if n_avail == 1:
                w = np.zeros(7)
                w[avail_mask] = 1.0
                weights[pd.Timestamp(d)] = w
            continue
        start = max(0, loc - MVO_WINDOW)
        ret_window = returns.iloc[start:loc+1]
        ret_window = ret_window.loc[:, avail_mask]
        mu  = ret_window.mean().values
        cov = ret_window.cov().values
        sub_w = solve_mvo(mu, cov, avail_mask)
        w = fill_full_weights(sub_w, avail_mask, np.where(avail_mask)[0])
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_momentum(dates):
    """策略4: 动量 — 近12月收益率排名，top3等权"""
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        n_avail = avail_mask.sum()
        if n_avail == 0:
            continue
        # 过去12个月的收益率
        lookback = min(LOOKBACK_MOM * 21, loc)  # ~252 交易日
        if lookback < 1:
            w = np.zeros(7)
            w[avail_mask] = 1.0 / n_avail
            weights[pd.Timestamp(d)] = w
            continue
        past = prices.iloc[loc-lookback:loc+1]
        if len(past) < 2:
            w = np.zeros(7)
            w[avail_mask] = 1.0 / n_avail
            weights[pd.Timestamp(d)] = w
            continue
        mom_ret = past.iloc[-1] / past.iloc[0] - 1
        mom_ret = mom_ret[avail_mask]  # 只排可用标的
        ranked = mom_ret.sort_values(ascending=False)
        top_n = min(3, len(ranked))
        top_etfs = ranked.index[:top_n]
        w = np.zeros(7)
        for etf in top_etfs:
            idx = ETF_CODES.index(etf)
            w[idx] = 1.0 / top_n
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_dynamic_score(dates):
    """
    策略5: 动态评分配置
    基于 market_state 确定权益仓位 → 权益内部 MVO → 非权益风险平价 → 合并
    """
    BASE_EQUITY = {
        '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
        '下行': 0.15, '极寒': 0.05
    }
    EQUITY_INDEXES = [ETF_CODES.index(e) for e in EQUITY_ETFS]
    HEDGE_INDEXES  = [ETF_CODES.index(e) for e in HEDGE_ETFS]

    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            loc = dates.get_loc(d)
        except KeyError:
            continue

        # 当日状态
        try:
            st = state.iloc[state.index.get_loc(d)]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st):
            st = '震荡'

        target_equity = BASE_EQUITY.get(st, 0.35)

        avail_mask = mask_available(None, loc)

        # 权益子集
        eq_avail = [i for i in EQUITY_INDEXES if avail_mask[i]]
        # 非权益子集
        hedge_avail = [i for i in HEDGE_INDEXES if avail_mask[i]]

        w = np.zeros(7)

        # 权益内 MVO
        if eq_avail and target_equity > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1]
            sub_mask = np.array([avail_mask[i] for i in eq_avail])
            sub_ret = ret_window.iloc[:, eq_avail].loc[:, sub_mask] if sub_mask.any() else ret_window.iloc[:, eq_avail]
            if sub_ret.shape[1] >= 2:
                mu  = sub_ret.mean().values
                cov = sub_ret.cov().values
                eq_w = solve_mvo(mu, cov, np.ones(len(eq_avail), dtype=bool))
            else:
                eq_w = np.ones(len(eq_avail)) / max(len(eq_avail), 1)
            for i, idx in enumerate(eq_avail):
                w[idx] = eq_w[i] * target_equity if i < len(eq_w) else 0

        # 非权益内 风险平价
        if hedge_avail and (1 - target_equity) > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1]
            sub_ret = ret_window.iloc[:, hedge_avail]
            if sub_ret.shape[1] >= 2:
                cov = sub_ret.cov().values
                h_w = solve_risk_parity(cov, np.ones(len(hedge_avail), dtype=bool))
            else:
                h_w = np.ones(len(hedge_avail))
            h_w = h_w / h_w.sum()
            for i, idx in enumerate(hedge_avail):
                w[idx] = h_w[i] * (1 - target_equity) if i < len(h_w) else 0

        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w[avail_mask] = 1.0 / avail_mask.sum()

        weights[pd.Timestamp(d)] = w

    return pd.DataFrame(weights, index=ETF_CODES).T

# ============================================================
# 4. 回测引擎 (前向填充月度权重 → 日度净值)
# ============================================================

def run_backtest(name, weight_df, dates, prices_df, returns_df):
    """
    weight_df: 月度调仓日权重 (index=date, columns=ETF)
    前向填充到日度，计算日收益、累计净值
    """
    daily_weights = pd.DataFrame(index=dates, columns=ETF_CODES, data=0.0)

    # 前向填充
    sorted_rebalance = weight_df.index.sort_values()
    for i, rd in enumerate(sorted_rebalance):
        if rd not in daily_weights.index:
            continue
        end = sorted_rebalance[i+1] if i+1 < len(sorted_rebalance) else dates[-1] + pd.Timedelta(days=1)
        mask = (daily_weights.index >= rd) & (daily_weights.index < end)
        daily_weights.loc[mask] = weight_df.loc[rd].values

    # 补开头 (第一个调仓日之前用等权)
    first_rebalance = sorted_rebalance[0]
    if first_rebalance in daily_weights.index:
        pre_mask = daily_weights.index < first_rebalance
        if pre_mask.any():
            daily_weights.loc[pre_mask] = weight_df.loc[first_rebalance].values

    # 对齐
    daily_weights = daily_weights.reindex(dates).ffill().fillna(0.0)
    # 归一化 (处理上市标志变化导致的小额偏差)
    row_sums = daily_weights.sum(axis=1)
    daily_weights = daily_weights.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    # 日组合收益 (扣除交易成本)
    # 换手率 = 新旧权重差的绝对值 / 2 * 双边费率
    daily_ret = (daily_weights * returns_df.loc[dates]).sum(axis=1)

    # 调仓日扣手续费
    for rd in sorted_rebalance:
        if rd not in daily_weights.index:
            continue
        loc = daily_weights.index.get_loc(rd)
        if loc > 0:
            prev_w = daily_weights.iloc[loc-1].values
            new_w  = daily_weights.iloc[loc].values
            turnover = np.abs(new_w - prev_w).sum() / 2
            cost = turnover * FEE
            daily_ret.iloc[loc] -= cost

    # 扣除滑点 (每日)
    # 这里简化为固定滑点折扣
    daily_ret = daily_ret - SLIPPAGE / 252  # 微小日滑点

    # 累计净值
    nav = (1 + daily_ret).cumprod() * INIT_CASH
    nav_returns = daily_ret.copy()

    return nav, nav_returns, daily_weights

# ============================================================
# 5. 风险指标计算
# ============================================================

def calc_metrics(name, daily_returns, weight_df, nav_series):
    """全套风险指标"""
    ret = daily_returns.dropna()
    if len(ret) == 0:
        return {}

    ann_ret   = ret.mean() * 252
    ann_vol   = ret.std() * np.sqrt(252)
    sharpe    = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    # 索提诺
    downside = ret[ret < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (ann_ret - RISK_FREE) / sortino_vol if sortino_vol > 0 else 0

    # 最大回撤
    cummax = nav_series.expanding().max()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()

    # 卡玛
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # 换手率 (月度)
    if weight_df is not None and len(weight_df) > 1:
        turnovers = weight_df.diff().abs().sum(axis=1) / 2
        # 排除第一个调仓日 (没有前一期的换手)
        monthly_turnover = turnovers.iloc[1:].mean() if len(turnovers) > 1 else 0
    else:
        monthly_turnover = 0

    # 权重集中度 HHI
    if weight_df is not None and len(weight_df) > 0:
        hhi = (weight_df ** 2).sum(axis=1).mean()
    else:
        hhi = 0

    # 胜率
    win_rate = (ret > 0).mean()

    # 95% VaR
    var_95 = ret.quantile(0.05)

    return {
        '策略': name,
        '年化收益': round(ann_ret, 4),
        '年化波动': round(ann_vol, 4),
        '夏普比率': round(sharpe, 4),
        '索提诺比率': round(sortino, 4),
        '最大回撤': round(max_dd, 4),
        '卡玛比率': round(calmar, 4),
        '月均换手率': round(monthly_turnover, 4),
        '权重集中度HHI': round(hhi, 4),
        '胜率': round(win_rate, 4),
        '95%VaR': round(var_95, 4),
        '年化收益/最大回撤': round(ann_ret/abs(max_dd) if max_dd!=0 else 0, 4),
    }

# ============================================================
# 6. 执行回测
# ============================================================
dates_arr = pd.DatetimeIndex(common_dates)
prices_mat = prices.loc[common_dates]
returns_mat = returns.loc[common_dates]

strategies = [
    ('等权',      strategy_equal_weight),
    ('风险平价',  strategy_risk_parity),
    ('MVO',       strategy_mvo),
    ('动量',      strategy_momentum),
    ('动态评分',  strategy_dynamic_score),
]

results = []
all_navs = {}
all_weights = {}

for name, strat_fn in strategies:
    print(f'[{name}] 计算月度权重...')
    w_df = strat_fn(common_dates)
    print(f'  调仓次数: {len(w_df)}, 日期: {w_df.index[0].date()} ~ {w_df.index[-1].date()}')

    nav, daily_ret, daily_w = run_backtest(name, w_df, dates_arr, prices_mat, returns_mat)
    all_navs[name] = nav
    all_weights[name] = daily_w

    m = calc_metrics(name, daily_ret, w_df, nav)
    results.append(m)
    print(f'  年化收益: {m["年化收益"]*100:.2f}%  夏普: {m["夏普比率"]:.3f}  最大回撤: {m["最大回撤"]*100:.2f}%  卡玛: {m["卡玛比率"]:.3f}')

# ============================================================
# 7. 保存结果
# ============================================================

metrics_df = pd.DataFrame(results)
print(f'\n{"="*80}')
print(metrics_df.to_string(index=False))
print(f'{"="*80}')

# 保存指标表
metrics_df.to_csv(f'{OUT}/baseline_metrics.csv', index=False, encoding='utf-8-sig')
print(f'\n✅ 指标表 → results/backtest/baseline_metrics.csv')

# 保存净值序列 (合并为一张宽表)
nav_df = pd.DataFrame(all_navs)
nav_df.index = common_dates
nav_df.to_csv(f'{OUT}/baseline_nav.csv', encoding='utf-8-sig')
print(f'✅ 净值序列 → results/backtest/baseline_nav.csv')

# 保存最后日期的权重 (供查看)
for name, w_df in all_weights.items():
    last_date = w_df.index[-1]
    last_w = w_df.loc[last_date]
    print(f'\n[{name}] 最终持仓 ({last_date.date()}):')
    for etf, w in last_w.items():
        print(f'  {etf}: {w*100:.1f}%')

print('\n===== 5套基线回测完成 =====')
