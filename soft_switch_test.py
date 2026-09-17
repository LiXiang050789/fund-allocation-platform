"""
软切换策略验证：
  根据市场状态分配 PPO 和动态评分的混合权重
  不再硬切换，而是按比例融合

对比：
  版本1: 永远 PPO（基准）
  版本2: 永远动态评分（基准）
  版本3: 硬切换（下行/极寒 → 动态评分，否则 → PPO）
  版本4: 软切换（按状态线性混合）
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============ 路径 ============
DATA = r"E:\农行杯\2026\1\clean"
OUT  = r"E:\农行杯\2026\1\回测"
PPO_WEIGHT_PATH = r"E:\农行杯\2026\1\weight\ppo_weight1.csv"
os.makedirs(OUT, exist_ok=True)

# ============ 全局参数 ============
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

# ============ 软切换配比（核心参数）============
# 可在此调整各状态下的 PPO / 动态评分 混合比例
STATE_MIX = {
    '上行':     {'ppo': 1.0, 'score': 0.0},   # 牛市全力进攻
    '震荡偏强': {'ppo': 0.8, 'score': 0.2},   # 偏强，略留防御
    '震荡':     {'ppo': 0.6, 'score': 0.4},   # 震荡，平衡
    '下行':     {'ppo': 0.2, 'score': 0.8},   # 下行，主防御
    '极寒':     {'ppo': 0.0, 'score': 1.0},   # 极寒，全防御
}
DEFAULT_MIX = {'ppo': 0.6, 'score': 0.4}      # 未知状态的默认配比

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
print(f'[数据] 市场状态: {sorted(state.dropna().unique())}')

# ============ 工具函数 ============
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

# ============ 动态评分 ============
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

def load_ppo_weights(dates):
    ppo = pd.read_csv(PPO_WEIGHT_PATH, parse_dates=["date"]).set_index("date")
    ppo = ppo.clip(lower=MIN_WEIGHT, upper=MAX_WEIGHT)
    ppo = ppo.reindex(dates).ffill().fillna(0)
    row_sums = ppo.sum(axis=1)
    ppo = ppo.div(row_sums.where(row_sums > 0, 1.0), axis=0)
    return ppo

# ============ 硬切换 ============
def strategy_hard_switch(dates, ppo_daily, score_monthly, switch_states=('下行', '极寒')):
    monthly = get_monthly_rebalance_dates(dates)
    weights = {}
    for d in monthly:
        try:
            st = state.loc[d]
        except (KeyError, IndexError):
            st = '震荡'
        if pd.isna(st):
            st = '震荡'
        if st in switch_states:
            key = pd.Timestamp(d)
            if key in score_monthly.index:
                w = score_monthly.loc[key].values
            else:
                cand = score_monthly.index[score_monthly.index <= key]
                w = score_monthly.loc[cand[-1]].values if len(cand) > 0 else np.zeros(7)
        else:
            key = pd.Timestamp(d)
            if key in ppo_daily.index:
                w = ppo_daily.loc[key].values
            else:
                cand = ppo_daily.index[ppo_daily.index <= key]
                w = ppo_daily.loc[cand[-1]].values if len(cand) > 0 else np.zeros(7)
        s = w.sum()
        if s > 0: w = w / s
        weights[pd.Timestamp(d)] = w
    return pd.DataFrame(weights, index=ETF_CODES).T

# ============ 软切换（核心）============
def strategy_soft_switch(dates, ppo_daily, score_monthly, state_mix=None):
    if state_mix is None:
        state_mix = STATE_MIX

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

        mix = state_mix.get(st, DEFAULT_MIX)
        w_ppo_ratio = mix['ppo']
        w_score_ratio = mix['score']

        key = pd.Timestamp(d)

        # 获取 PPO 权重
        if key in ppo_daily.index:
            w_ppo = ppo_daily.loc[key].values
        else:
            cand = ppo_daily.index[ppo_daily.index <= key]
            w_ppo = ppo_daily.loc[cand[-1]].values if len(cand) > 0 else np.zeros(7)

        # 获取动态评分权重
        if key in score_monthly.index:
            w_score = score_monthly.loc[key].values
        else:
            cand = score_monthly.index[score_monthly.index <= key]
            w_score = score_monthly.loc[cand[-1]].values if len(cand) > 0 else np.zeros(7)

        # 线性混合
        w = w_ppo_ratio * w_ppo + w_score_ratio * w_score

        # 归一化
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            avail_mask = mask_available(dates.get_loc(d))
            w = np.zeros(7)
            w[avail_mask] = 1.0 / avail_mask.sum()

        weights[pd.Timestamp(d)] = w
        log.append({
            'date': d, 'state': st,
            'ppo_ratio': w_ppo_ratio, 'score_ratio': w_score_ratio,
        })

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

    # 加载权重
    ppo_daily = load_ppo_weights(common_dates)
    score_monthly = strategy_dynamic_score(common_dates)
    ppo_monthly = ppo_daily.loc[get_monthly_rebalance_dates(common_dates)]

    print(f"[PPO] {len(ppo_daily)} 天")
    print(f"[动态评分] {len(score_monthly)} 调仓日")

    results = []
    all_navs = {}

    # 版本1：永远 PPO
    print("\n[版本1] 永远 PPO")
    nav1, ret1, _ = run_backtest("PPO永久", ppo_monthly, dates_arr, prices_mat, returns_mat)
    m1 = calc_metrics("PPO永久", ret1, ppo_monthly, nav1)
    results.append(m1); all_navs["PPO永久"] = nav1
    print(f"  Sharpe={m1['夏普比率']:.4f}, Ann={m1['年化收益']*100:.2f}%, MDD={m1['最大回撤']*100:.2f}%")

    # 版本2：永远动态评分
    print("\n[版本2] 永远动态评分")
    nav2, ret2, _ = run_backtest("动态评分永久", score_monthly, dates_arr, prices_mat, returns_mat)
    m2 = calc_metrics("动态评分永久", ret2, score_monthly, nav2)
    results.append(m2); all_navs["动态评分永久"] = nav2
    print(f"  Sharpe={m2['夏普比率']:.4f}, Ann={m2['年化收益']*100:.2f}%, MDD={m2['最大回撤']*100:.2f}%")

    # 版本3：硬切换
    print("\n[版本3] 硬切换（下行/极寒 → 动态评分）")
    hard_w = strategy_hard_switch(common_dates, ppo_daily, score_monthly)
    nav3, ret3, _ = run_backtest("硬切换", hard_w, dates_arr, prices_mat, returns_mat)
    m3 = calc_metrics("硬切换", ret3, hard_w, nav3)
    results.append(m3); all_navs["硬切换"] = nav3
    print(f"  Sharpe={m3['夏普比率']:.4f}, Ann={m3['年化收益']*100:.2f}%, MDD={m3['最大回撤']*100:.2f}%")

    # 版本4：软切换
    print("\n[版本4] 软切换（按状态线性混合）")
    soft_w, soft_log = strategy_soft_switch(common_dates, ppo_daily, score_monthly)
    nav4, ret4, _ = run_backtest("软切换", soft_w, dates_arr, prices_mat, returns_mat)
    m4 = calc_metrics("软切换", ret4, soft_w, nav4)
    results.append(m4); all_navs["软切换"] = nav4
    print(f"  Sharpe={m4['夏普比率']:.4f}, Ann={m4['年化收益']*100:.2f}%, MDD={m4['最大回撤']*100:.2f}%")

    # 按状态统计
    print(f"\n  软切换：按状态分布:")
    for st in sorted(soft_log['state'].unique()):
        sub = soft_log[soft_log['state'] == st]
        print(f"    {st}: {len(sub):3d} 次, PPO ratio={sub['ppo_ratio'].mean():.2f}, 评分 ratio={sub['score_ratio'].mean():.2f}")

    # 输出
    print("\n" + "="*100)
    df_result = pd.DataFrame(results)
    print(df_result.to_string(index=False))
    print("="*100)
    df_result.to_csv(f"{OUT}/soft_switch_metrics.csv", index=False, encoding='utf-8-sig')

    nav_df = pd.DataFrame(all_navs)
    nav_df.index = common_dates
    nav_df.to_csv(f"{OUT}/soft_switch_nav.csv", encoding='utf-8-sig')
    soft_log.to_csv(f"{OUT}/soft_switch_log.csv", index=False, encoding='utf-8-sig')

    print(f"\n✅ 结果保存至 {OUT}/soft_switch_*.csv")

    # 快速结论
    print("\n===== 结论 =====")
    best_sharpe = max(results, key=lambda x: x['夏普比率'])
    print(f"最高夏普: {best_sharpe['策略']} = {best_sharpe['夏普比率']:.4f}")
    best_calmar = max(results, key=lambda x: x['卡玛比率'])
    print(f"最高卡玛: {best_calmar['策略']} = {best_calmar['卡玛比率']:.4f}")
    print(f"\n对照基准（风险平价）: Sharpe=0.5456, MDD=-0.2162")