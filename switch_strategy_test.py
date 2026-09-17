"""
策略切换验证：
  版本1: 永远 PPO
  版本2: 永远动态评分
  版本3: market_state 为"下行/极寒" → 动态评分，否则 → PPO

所有策略共用统一回测框架（滑点、手续费、管理费一致）
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============ 路径 ============
BASE = r"E:/农行杯/2026/1/Streamlit平台"
DATA = r"E:\农行杯\2026\1\clean"
OUT  = r"E:\农行杯\2026\1\回测"
PPO_WEIGHT_PATH = r"E:\农行杯\2026\1\weight\ppo_weight1.csv"
os.makedirs(OUT, exist_ok=True)

# ============ 全局参数（与回测对比.py 完全一致）============
INIT_CASH    = 1_000_000
SLIPPAGE     = 0.0005
FEE          = 0.0005
MAX_WEIGHT   = 0.30
MIN_WEIGHT   = 0.0
RISK_FREE    = 0.025
MVO_WINDOW   = 60

ETF_CODES    = ['hs300','zz500','kc50','consume','chip','gold','bond10']
EQUITY_ETFS  = ['hs300','zz500','kc50','consume','chip']
HEDGE_ETFS   = ['gold','bond10']
EQUITY_INDEXES = [ETF_CODES.index(e) for e in EQUITY_ETFS]
HEDGE_INDEXES  = [ETF_CODES.index(e) for e in HEDGE_ETFS]

# ============ 数据加载 ============
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
print(f'[数据] 市场状态类型: {sorted(state.dropna().unique())}')

# ============ 通用工具 ============
def mask_available(date_idx):
    return avail.iloc[date_idx].values.astype(bool)

def solve_risk_parity(cov):
    n = cov.shape[0]
    if n == 0: return np.array([])
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols < 1e-10, 1e-10, vols)
    w = 1.0 / vols
    return w / w.sum()

def solve_mvo(mu, cov):
    n = len(mu)
    if n == 0: return np.array([])
    mu_shrink = 0.6 * mu + 0.4 * np.full_like(mu, mu.mean())
    excess = mu_shrink - RISK_FREE / 252
    try:
        cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-6)
        w = cov_inv @ excess
        w = w / w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    s = w.sum()
    if s > 1e-12: w = w / s
    return w

def get_monthly_rebalance_dates(dates):
    df = pd.DataFrame({'date': dates})
    df['ym'] = df['date'].dt.to_period('M')
    return df.groupby('ym')['date'].first().values

# ============ 动态评分权重 ============
BASE_EQUITY = {
    '上行': 0.55, '震荡偏强': 0.45, '震荡': 0.35,
    '下行': 0.15, '极寒': 0.05
}

def strategy_dynamic_score(dates):
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
        avail_mask = mask_available(loc)
        eq_active = [i for i in EQUITY_INDEXES if avail_mask[i]]
        hedge_active = [i for i in HEDGE_INDEXES if avail_mask[i]]
        w = np.zeros(7)
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

# ============ PPO 权重（日频）============
def load_ppo_weights(dates):
    ppo = pd.read_csv(PPO_WEIGHT_PATH, parse_dates=["date"]).set_index("date")
    ppo = ppo.clip(lower=MIN_WEIGHT, upper=MAX_WEIGHT)
    ppo = ppo.reindex(dates).ffill().fillna(0)
    row_sums = ppo.sum(axis=1)
    ppo = ppo.div(row_sums.where(row_sums > 0, 1.0), axis=0)
    return ppo

# ============ 切换策略 ============
def strategy_switch(dates, ppo_daily, score_monthly, switch_states=('下行', '极寒')):
    """
    每月调仓日决定：
      如果当前市场状态在 switch_states 中 → 用动态评分权重
      否则 → 用 PPO 当天的日频权重
    两次调仓之间保持不变（月度调仓）
    """
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    log = []
    for d in monthly:
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st):
            st = '震荡'

        if st in switch_states:
            # 用动态评分权重
            if pd.Timestamp(d) in score_monthly.index:
                w = score_monthly.loc[pd.Timestamp(d)].values
            else:
                # 找不到就用最近的
                avail_idx = score_monthly.index[score_monthly.index <= pd.Timestamp(d)]
                if len(avail_idx) > 0:
                    w = score_monthly.loc[avail_idx[-1]].values
                else:
                    w = np.zeros(7)
            source = "动态评分"
        else:
            # 用 PPO 当天的日频权重
            if pd.Timestamp(d) in ppo_daily.index:
                w = ppo_daily.loc[pd.Timestamp(d)].values
            else:
                avail_idx = ppo_daily.index[ppo_daily.index <= pd.Timestamp(d)]
                if len(avail_idx) > 0:
                    w = ppo_daily.loc[avail_idx[-1]].values
                else:
                    w = np.zeros(7)
            source = "PPO"

        # 归一化
        s = w.sum()
        if s > 0:
            w = w / s
        weights[pd.Timestamp(d)] = w
        log.append({'date': d, 'state': st, 'source': source})

    return pd.DataFrame(weights, index=ETF_CODES).T, pd.DataFrame(log)

# ============ 回测引擎 ============
def run_backtest(name, weight_df, dates, prices_df, returns_df):
    daily_weights = pd.DataFrame(index=dates, columns=ETF_CODES, data=0.0)
    sorted_rebalance = weight_df.index.sort_values()
    for i, rd in enumerate(sorted_rebalance):
        if rd not in daily_weights.index:
            continue
        end = sorted_rebalance[i+1] if i+1 < len(sorted_rebalance) else dates[-1] + pd.Timedelta(days=1)
        mask = (daily_weights.index >= rd) & (daily_weights.index < end)
        daily_weights.loc[mask] = weight_df.loc[rd].values
    first_rebalance = sorted_rebalance[0]
    if first_rebalance in daily_weights.index:
        pre_mask = daily_weights.index < first_rebalance
        if pre_mask.any():
            daily_weights.loc[pre_mask] = weight_df.loc[first_rebalance].values
    daily_weights = daily_weights.reindex(dates).ffill().fillna(0.0)
    row_sums = daily_weights.sum(axis=1)
    daily_weights = daily_weights.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    daily_ret = (daily_weights * returns_df.loc[dates]).sum(axis=1)
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
    daily_ret = daily_ret - SLIPPAGE / 252
    ANNUAL_FEE = np.array([0.002, 0.002, 0.002, 0.006, 0.006, 0.006, 0.002])
    daily_fee = (daily_weights * ANNUAL_FEE).sum(axis=1) / 252
    daily_ret = daily_ret - daily_fee
    nav = (1 + daily_ret).cumprod() * INIT_CASH
    return nav, daily_ret, daily_weights

def calc_metrics(name, daily_returns, weight_df, nav_series):
    ret = daily_returns.dropna()
    if len(ret) == 0: return {}
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
    monthly_turnover = 0
    if weight_df is not None and len(weight_df) > 1:
        turnovers = weight_df.diff().abs().sum(axis=1) / 2
        monthly_turnover = turnovers.iloc[1:].mean()
    hhi = (weight_df ** 2).sum(axis=1).mean() if weight_df is not None else 0
    win_rate = (ret > 0).mean()
    return {
        '策略': name,
        '年化收益': round(ann_ret, 4),
        '年化波动': round(ann_vol, 4),
        '夏普比率': round(sharpe, 4),
        '索提诺比率': round(sortino, 4),
        '最大回撤': round(max_dd, 4),
        '卡玛比率': round(calmar, 4),
        '月均换手率': round(monthly_turnover, 4),
        'HHI': round(hhi, 4),
        '胜率': round(win_rate, 4),
    }

# ============ 主流程 ============
if __name__ == "__main__":
    dates_arr = pd.DatetimeIndex(common_dates)
    prices_mat = prices.loc[common_dates]
    returns_mat = returns.loc[common_dates]

    # 加载 PPO 权重
    ppo_daily = load_ppo_weights(common_dates)
    print(f"[PPO] 权重文件加载: {len(ppo_daily)} 天")

    # 动态评分权重
    score_monthly = strategy_dynamic_score(common_dates)
    print(f"[动态评分] 权重生成: {len(score_monthly)} 个调仓日")

    # ========== 三个版本 ==========
    results = []
    all_navs = {}

    # 版本1：永远 PPO（月度调仓）
    print("\n[版本1] 永远 PPO")
    ppo_monthly = ppo_daily.loc[get_monthly_rebalance_dates(common_dates)]
    nav1, ret1, _ = run_backtest("PPO永久", ppo_monthly, dates_arr, prices_mat, returns_mat)
    results.append(calc_metrics("PPO永久", ret1, ppo_monthly, nav1))
    all_navs["PPO永久"] = nav1
    print(f"  夏普={results[-1]['夏普比率']:.4f}, 年化={results[-1]['年化收益']*100:.2f}%, MDD={results[-1]['最大回撤']*100:.2f}%")

    # 版本2：永远动态评分
    print("\n[版本2] 永远动态评分")
    nav2, ret2, _ = run_backtest("动态评分永久", score_monthly, dates_arr, prices_mat, returns_mat)
    results.append(calc_metrics("动态评分永久", ret2, score_monthly, nav2))
    all_navs["动态评分永久"] = nav2
    print(f"  夏普={results[-1]['夏普比率']:.4f}, 年化={results[-1]['年化收益']*100:.2f}%, MDD={results[-1]['最大回撤']*100:.2f}%")

    # 版本3：切换
    print("\n[版本3] market_state=下行/极寒 → 动态评分；否则 → PPO")
    switch_w, switch_log = strategy_switch(common_dates, ppo_daily, score_monthly)
    nav3, ret3, _ = run_backtest("切换策略", switch_w, dates_arr, prices_mat, returns_mat)
    results.append(calc_metrics("切换策略", ret3, switch_w, nav3))
    all_navs["切换策略"] = nav3
    print(f"  夏普={results[-1]['夏普比率']:.4f}, 年化={results[-1]['年化收益']*100:.2f}%, MDD={results[-1]['最大回撤']*100:.2f}%")

    # 统计切换频率
    print(f"\n  切换频率统计:")
    print(f"    PPO 使用: {(switch_log['source']=='PPO').sum()} 次")
    print(f"    动态评分使用: {(switch_log['source']=='动态评分').sum()} 次")
    print(f"    总调仓日: {len(switch_log)}")

    # 分状态统计
    print(f"\n  按状态统计:")
    for st in sorted(switch_log['state'].unique()):
        sub = switch_log[switch_log['state'] == st]
        print(f"    {st}: {len(sub)} 次，使用 {sub['source'].value_counts().to_dict()}")

    # ========== 输出 ==========
    print("\n" + "="*80)
    df_result = pd.DataFrame(results)
    print(df_result.to_string(index=False))
    print("="*80)
    df_result.to_csv(f"{OUT}/switch_strategy_metrics.csv", index=False, encoding='utf-8-sig')

    nav_df = pd.DataFrame(all_navs)
    nav_df.index = common_dates
    nav_df.to_csv(f"{OUT}/switch_strategy_nav.csv", encoding='utf-8-sig')

    # 保存切换日志
    switch_log.to_csv(f"{OUT}/switch_log.csv", index=False, encoding='utf-8-sig')

    print(f"\n✅ 结果保存至 {OUT}/switch_strategy_*.csv")