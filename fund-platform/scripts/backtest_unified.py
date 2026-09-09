"""
农行杯赛题一 · 5套基线回测 + PPO强化学习策略 全套统一回测
纯 numpy/pandas 实现，零外部优化库依赖
所有策略成本、指标口径完全统一，公平横向对比
更新：solve_mvo 修改为【最大夏普切点组合解析解】，复用全局RISK_FREE常量
修正：solve_mvo/solve_risk_parity 返回子集权重，策略函数内回填至7维
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "clean")
OUT  = os.path.join(BASE, "results", "backtest")
LGB_PRED_PATH = f'{BASE}/results/pred_return_lgb_daily.csv'
# PPO权重输出路径（需提前生成）
PPO_WEIGHT_PATH = os.path.join(BASE, "results", "ppo_weight.csv")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 0. 全局统一参数（所有策略共用，不可差异化）
# ============================================================
INIT_CASH    = 1_000_000       # 起始资金
SLIPPAGE     = 0.0005          # 单边滑点 0.05%
FEE          = 0.001           # 双边成本 0.1%
MAX_WEIGHT   = 0.30            # 单ETF权重上限
MIN_WEIGHT   = 0.0
RISK_FREE    = 0.025           # 年化无风险利率 2.5%
LOOKBACK_MOM = 12              # 动量回看月数
MVO_WINDOW   = 60              # MVO协方差窗口(交易日)
COV_WINDOW_EQ = 252            # 协方差一年窗口
ETF_CODES    = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_ETFS  = ['hs300','zz500','kc50','consume','chip']
HEDGE_ETFS   = ['gold','bond10']
CSI300_ETF   = 'hs300'
EQUITY_INDEXES = [ETF_CODES.index(e) for e in EQUITY_ETFS]
HEDGE_INDEXES  = [ETF_CODES.index(e) for e in HEDGE_ETFS]

NORTH_CUTOFF = pd.Timestamp('2024-08-19')
MACRO_LAG = 30

# ============================================================
# 1. 数据加载
# ============================================================
price_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
listed_df = pd.read_csv(f'{DATA}/etf_price_clean.csv', encoding='utf-8-sig', parse_dates=['date'])
score_df  = pd.read_csv(f'{DATA}/market_score_clean.csv', encoding='utf-8-sig', parse_dates=['date'])

price_df = price_df.set_index('date')
prices = price_df[ETF_CODES].copy()

listed_cols = [f'{e}_listed' for e in ETF_CODES]
avail = listed_df.set_index('date')[listed_cols].copy()
avail.columns = ETF_CODES

state_map = score_df.set_index('date')['market_state']

common_dates = prices.index.intersection(avail.index).intersection(state_map.index)
prices = prices.loc[common_dates]
avail  = avail.loc[common_dates]
state  = state_map.loc[common_dates]
returns = prices.pct_change().fillna(0)

print(f'[数据] 交易日: {len(common_dates)}, {common_dates[0].date()} ~ {common_dates[-1].date()}')
print(f'[数据] ETF: {prices.shape[1]}只, 评分状态类型: {sorted(state.dropna().unique())}')

# ============================================================
# 2. 通用工具函数（solve_mvo：最大夏普切点解析解；返回子集权重）
# ============================================================
def mask_available(arr, date_idx):
    """返回当日已上市ETF布尔mask"""
    return avail.iloc[date_idx].values.astype(bool)

def solve_mvo(mu, cov):
    """
    最大夏普切点组合（解析解），返回仅含有效资产的权重子集
    mu: 1D array,  cov: 2D array (均为有效资产维度)
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    # ==========新增：预期收益率收缩，向均值收缩，抑制权重剧烈跳动==========
    excess = mu - RISK_FREE / 252
    try:
        # 正则防止协方差矩阵奇异不可逆
        cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-6)
        w = cov_inv @ excess
        w = w / w.sum()
    except np.linalg.LinAlgError:
        # 求逆失败退化为等权
        w = np.ones(n) / n
    # 单资产权重裁剪 [0, MAX_WEIGHT]
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    s = w.sum()
    if s > 1e-12:
        w = w / s
    return w


def solve_risk_parity(cov):
    """波动率倒数加权，返回子集权重"""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w = 1.0 / vols
    return w / w.sum()

def get_monthly_rebalance_dates(dates):
    """获取每月第一个交易日"""
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

# ============================================================
# 3. 全部策略函数（权重均为7维完整向量）
# ============================================================
def strategy_csi300(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        w = np.zeros(7)
        idx = ETF_CODES.index(CSI300_ETF)
        if avail_mask[idx]:
            w[idx] = 1.0
        else:
            active = np.where(avail_mask)[0]
            if len(active) > 0:
                w[active] = 1.0 / len(active)
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_equal_weight(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        active = np.where(avail_mask)[0]
        w = np.zeros(7)
        if len(active) > 0:
            w[active] = 1.0 / len(active)
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_risk_parity(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        active = np.where(avail_mask)[0]
        n = len(active)
        if n == 0:
            continue
        if n == 1:
            w = np.zeros(7)
            w[active[0]] = 1.0
            weights[pd.Timestamp(d)] = w
            continue
        start = max(0, loc - MVO_WINDOW)
        ret_window = returns.iloc[start:loc+1].iloc[:, active]
        cov = ret_window.cov().values
        sub_w = solve_risk_parity(cov)
        w = np.zeros(7)
        w[active] = sub_w
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_mvo(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        active = np.where(avail_mask)[0]
        n = len(active)
        if n == 0:
            continue
        if n == 1:
            w = np.zeros(7)
            w[active[0]] = 1.0
            weights[pd.Timestamp(d)] = w
            continue
        start = max(0, loc - MVO_WINDOW)
        ret_window = returns.iloc[start:loc+1].iloc[:, active]
        mu = ret_window.mean().values
        cov = ret_window.cov().values
        sub_w = solve_mvo(mu, cov)
        w = np.zeros(7)
        w[active] = sub_w
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_momentum(dates):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        loc = dates.get_loc(d)
        avail_mask = mask_available(None, loc)
        active = np.where(avail_mask)[0]
        n = len(active)
        if n == 0:
            continue
        lookback = min(LOOKBACK_MOM * 21, loc)
        if lookback < 1:
            w = np.zeros(7)
            w[active] = 1.0 / n
            weights[pd.Timestamp(d)] = w
            continue
        past = prices.iloc[loc-lookback:loc+1]
        if len(past) < 2:
            w = np.zeros(7)
            w[active] = 1.0 / n
            weights[pd.Timestamp(d)] = w
            continue
        mom_ret = past.iloc[-1] / past.iloc[0] - 1
        mom_ret = mom_ret.iloc[active]
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
    BASE_EQUITY = {
        '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
        '下行': 0.15, '极寒': 0.05
    }
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            loc = dates.get_loc(d)
        except KeyError:
            continue
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st):
            st = '震荡'
        target_equity = BASE_EQUITY.get(st, 0.35)
        avail_mask = mask_available(None, loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)
        # 权益部分
        if eq_active and target_equity > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, eq_active]
            if ret_window.shape[1] >= 2:
                mu = ret_window.mean().values
                cov = ret_window.cov().values
                sub_w = solve_mvo(mu, cov)
            else:
                sub_w = np.ones(len(eq_active)) / len(eq_active)
            for j, idx in enumerate(eq_active):
                w[idx] = sub_w[j] * target_equity
        # 避险部分
        if hedge_active and (1 - target_equity) > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, hedge_active]
            if ret_window.shape[1] >= 2:
                cov = ret_window.cov().values
                sub_w = solve_risk_parity(cov)
            else:
                sub_w = np.ones(len(hedge_active)) / len(hedge_active)
            for j, idx in enumerate(hedge_active):
                w[idx] = sub_w[j] * (1 - target_equity)
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w[avail_mask] = 1.0 / avail_mask.sum()
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

# LGB预测数据
lgb_pred = None
if os.path.exists(LGB_PRED_PATH):
    lgb_pred = pd.read_csv(LGB_PRED_PATH, encoding='utf-8-sig', parse_dates=['date']).set_index('date')
    print(f'[LGB] 预测文件已加载: {len(lgb_pred)}天')

def strategy_lgb_fusion(dates):
    BASE_EQUITY = {
        '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
        '下行': 0.15, '极寒': 0.05
    }
    if lgb_pred is None:
        print('[LGB融合] ⚠️ 预测文件不存在，回退为纯动态评分')
        return strategy_dynamic_score(dates)
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            loc = dates.get_loc(d)
        except KeyError:
            continue
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st):
            st = '震荡'
        target_equity = BASE_EQUITY.get(st, 0.35)
        avail_mask = mask_available(None, loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)
        # 权益部分：LGB预测加权 softmax
        if eq_active and target_equity > 0:
            eq_etfs = [ETF_CODES[i] for i in eq_active]
            lgb_loc = lgb_pred.index.get_indexer([d], method='pad')[0]
            lgb_window = lgb_pred.index[max(0, lgb_loc-21):lgb_loc+1]
            eq_preds = np.zeros(len(eq_etfs))
            for j, e in enumerate(eq_etfs):
                if e in lgb_pred.columns:
                    vals = lgb_pred[e].loc[lgb_window.intersection(lgb_pred.index)].dropna()
                    eq_preds[j] = vals.mean() if len(vals) > 0 else 0
            temp = 0.5
            x = eq_preds / (np.std(eq_preds) + 1e-8) / temp
            exp_x = np.exp(x - np.max(x))
            sub_w = exp_x / exp_x.sum()
            for j, idx in enumerate(eq_active):
                w[idx] = sub_w[j] * target_equity
        # 避险部分：风险平价
        if hedge_active and (1 - target_equity) > 0:
            start = max(0, loc - MVO_WINDOW)
            ret_window = returns.iloc[start:loc+1].iloc[:, hedge_active]
            if ret_window.shape[1] >= 2:
                cov = ret_window.cov().values
                sub_w = solve_risk_parity(cov)
            else:
                sub_w = np.ones(len(hedge_active)) / len(hedge_active)
            for j, idx in enumerate(hedge_active):
                w[idx] = sub_w[j] * (1 - target_equity)
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w[avail_mask] = 1.0 / avail_mask.sum()
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

def strategy_ppo_monthly(dates):
    """PPO月度调仓（与基线对齐）"""
    if not os.path.exists(PPO_WEIGHT_PATH):
        raise FileNotFoundError(f"PPO权重文件不存在：{PPO_WEIGHT_PATH}")
    ppo_daily = pd.read_csv(PPO_WEIGHT_PATH, parse_dates=["date"])
    ppo_daily = ppo_daily.set_index("date")
    ppo_daily = ppo_daily.reindex(dates).ffill().fillna(0)
    ppo_daily = apply_listed_mask_to_weights(ppo_daily, dates)
    monthly_dates = get_monthly_rebalance_dates(dates)
    ppo_monthly = ppo_daily.loc[monthly_dates]
    return ppo_monthly

def strategy_ppo_daily(dates):
    """PPO原生日频调仓"""
    if not os.path.exists(PPO_WEIGHT_PATH):
        raise FileNotFoundError(f"PPO权重文件不存在：{PPO_WEIGHT_PATH}")
    ppo_daily = pd.read_csv(PPO_WEIGHT_PATH, parse_dates=["date"])
    ppo_daily = ppo_daily.set_index("date")
    ppo_daily = ppo_daily.reindex(dates).ffill().fillna(0)
    ppo_daily = apply_listed_mask_to_weights(ppo_daily, dates)
    return ppo_daily

def apply_listed_mask_to_weights(weight_df, dates):
    """按上市状态清零不可用ETF权重，并对可用权重重新归一化。"""
    masked = weight_df.reindex(dates).copy()
    available = avail.reindex(dates).astype(bool)
    masked = masked.where(available, 0.0)
    row_sums = masked.sum(axis=1)
    zero_rows = row_sums <= 1e-12
    masked = masked.div(row_sums.where(~zero_rows, 1.0), axis=0)
    if zero_rows.any():
        fallback = available.div(available.sum(axis=1).where(available.sum(axis=1) > 0, 1.0), axis=0)
        masked.loc[zero_rows] = fallback.loc[zero_rows]
    return masked[ETF_CODES]

# ============================================================
# 4. 回测引擎
# ============================================================
def run_backtest(name, weight_df, dates, prices_df, returns_df):
    daily_weights = pd.DataFrame(index=dates, columns=ETF_CODES, data=0.0)
    sorted_rebalance = weight_df.index.sort_values()
    for i, rd in enumerate(sorted_rebalance):
        if rd not in daily_weights.index:
            continue
        end = sorted_rebalance[i+1] if i+1 < len(sorted_rebalance) else dates[-1] + pd.Timedelta(days=1)
        mask = (daily_weights.index >= rd) & (daily_weights.index < end)
        daily_weights.loc[mask] = weight_df.loc[rd].values
    # 调仓日前数据用第一次调仓权重填充
    first_rebalance = sorted_rebalance[0]
    if first_rebalance in daily_weights.index:
        pre_mask = daily_weights.index < first_rebalance
        if pre_mask.any():
            daily_weights.loc[pre_mask] = weight_df.loc[first_rebalance].values
    daily_weights = daily_weights.reindex(dates).ffill().fillna(0.0)
    row_sums = daily_weights.sum(axis=1)
    daily_weights = daily_weights.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    daily_ret = (daily_weights * returns_df.loc[dates]).sum(axis=1)
    # 调仓日成本
    for rd in sorted_rebalance:
        if rd not in daily_weights.index:
            continue
        loc = daily_weights.index.get_loc(rd)
        if loc > 0:
            prev_w = daily_weights.iloc[loc-1].values
            new_w  = daily_weights.iloc[loc].values
            turnover = np.abs(new_w - prev_w).sum()
            cost = turnover * FEE
            daily_ret.iloc[loc] -= cost
    # 滑点每日摊销
    daily_ret = daily_ret - SLIPPAGE / 252
    # 管理费
    ANNUAL_FEE = np.array([0.002, 0.002, 0.002, 0.006, 0.006, 0.006, 0.002])
    daily_fee = (daily_weights * ANNUAL_FEE).sum(axis=1) / 252
    daily_ret = daily_ret - daily_fee

    nav = (1 + daily_ret).cumprod() * INIT_CASH
    return nav, daily_ret, daily_weights

# ============================================================
# 5. 风险指标计算
# ============================================================
def calc_metrics(name, daily_returns, weight_df, nav_series):
    ret = daily_returns.dropna()
    if len(ret) == 0:
        return {}
    ann_ret   = ret.mean() * 252
    ann_vol   = ret.std() * np.sqrt(252)
    sharpe    = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 1e-8 else 0
    downside = ret[ret < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (ann_ret - RISK_FREE) / sortino_vol
    cummax = nav_series.expanding().max()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    # 换手率
    monthly_turnover = 0
    if weight_df is not None and len(weight_df) > 1:
        turnovers = weight_df.diff().abs().sum(axis=1) / 2
        monthly_turnover = turnovers.iloc[1:].mean()
    hhi = 0
    if weight_df is not None and len(weight_df) > 0:
        hhi = (weight_df ** 2).sum(axis=1).mean()
    win_rate = (ret > 0).mean()
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
    }

# ============================================================
# 6. 批量执行
# ============================================================
dates_arr = pd.DatetimeIndex(common_dates)
prices_mat = prices.loc[common_dates]
returns_mat = returns.loc[common_dates]

# 月度策略（主实验）
strategies_monthly = [
    ('沪深300',   strategy_csi300),
    ('等权',      strategy_equal_weight),
    ('风险平价',  strategy_risk_parity),
    ('MVO',       strategy_mvo),
    ('动量',      strategy_momentum),
    ('动态评分',  strategy_dynamic_score),
    ('LGB融合',   strategy_lgb_fusion),
    ('PPO月度',   strategy_ppo_monthly),
]
# 日频PPO（附录对照）
strategies_daily = [('PPO原生日频', strategy_ppo_daily)]

results = []
all_navs = {}
all_weights = {}

print("========== 运行【月度调仓】主实验 ==========")
for name, strat_fn in strategies_monthly:
    print(f'\n[{name}] 正在计算权重...')
    w_df = strat_fn(common_dates)
    print(f'  调仓次数: {len(w_df)}')
    nav, daily_ret, daily_w = run_backtest(name, w_df, dates_arr, prices_mat, returns_mat)
    all_navs[name] = nav
    all_weights[name] = daily_w
    m = calc_metrics(name, daily_ret, w_df, nav)
    results.append(m)
    print(f'  年化收益: {m["年化收益"]*100:.2f}% | 夏普: {m["夏普比率"]:.3f} | 最大回撤: {m["最大回撤"]*100:.2f}%')

print("\n========== 运行【PPO日频】对照实验 ==========")
for name, strat_fn in strategies_daily:
    print(f'\n[{name}] 正在计算权重...')
    w_df = strat_fn(common_dates)
    print(f'  调仓天数: {len(w_df)}')
    nav, daily_ret, daily_w = run_backtest(name, w_df, dates_arr, prices_mat, returns_mat)
    all_navs[name] = nav
    all_weights[name] = daily_w
    m = calc_metrics(name, daily_ret, w_df, nav)
    results.append(m)
    print(f'  年化收益: {m["年化收益"]*100:.2f}% | 夏普: {m["夏普比率"]:.3f} | 最大回撤: {m["最大回撤"]*100:.2f}%')

# ============================================================
# 7. 输出结果
# ============================================================
metrics_df = pd.DataFrame(results)
print('\n' + '='*120)
print(metrics_df.to_string(index=False))
print('='*120)
metrics_df.to_csv(f'{OUT}/all_strategy_metrics.csv', index=False, encoding='utf-8-sig')
print(f'\n✅ 指标表保存至: {OUT}/all_strategy_metrics.csv')

nav_df = pd.DataFrame(all_navs)
nav_df.index = common_dates
nav_df.to_csv(f'{OUT}/all_strategy_nav.csv', encoding='utf-8-sig')
print(f'✅ 净值曲线保存至: {OUT}/all_strategy_nav.csv')

print("\n========== 期末持仓 ==========")
for name, w_df in all_weights.items():
    last_date = w_df.index[-1]
    last_w = w_df.loc[last_date]
    print(f'\n【{name}】 期末 {last_date.date()}')
    for etf, w in last_w.items():
        print(f'  {etf:8s}: {w*100:5.1f}%')

# ============================================================
# 8. 分阶段回测（可选）
# ============================================================
PERIODS = [
    ('2015牛市',     '2015-01-05', '2015-06-12'),
    ('2015股灾',     '2015-06-15', '2016-01-28'),
    ('2016-2017慢牛', '2016-01-29', '2018-01-26'),
    ('2018熊市',     '2018-01-29', '2019-01-04'),
    ('2019-2020牛市', '2019-01-07', '2021-02-18'),
    ('2021后震荡',    '2021-02-19', '2026-07-10'),
]
seg_results = []
nav_all = pd.DataFrame(all_navs)
nav_all.index = pd.DatetimeIndex(common_dates)
for pname, start, end in PERIODS:
    seg = nav_all.loc[start:end]
    if len(seg) < 20:
        continue
    row = {'阶段': pname, '交易日天数': len(seg)}
    for col in nav_all.columns:
        s = seg[col]
        cum = s.iloc[-1] / s.iloc[0] - 1
        days = max((s.index[-1] - s.index[0]).days, 1)
        ann = (1+cum)**(365/days) - 1
        peak = s.expanding().max()
        mdd = ((s - peak) / peak).min()
        row[f'{col}_累计收益'] = round(cum, 4)
        row[f'{col}_年化收益'] = round(ann, 4)
        row[f'{col}_最大回撤'] = round(mdd, 4)
    seg_results.append(row)
seg_df = pd.DataFrame(seg_results)
seg_df.to_csv(f'{OUT}/segment_all_strategy.csv', index=False, encoding='utf-8-sig')
print('\n✅ 分阶段回测表保存完成')
print('\n===== 全部回测执行完毕 =====')
